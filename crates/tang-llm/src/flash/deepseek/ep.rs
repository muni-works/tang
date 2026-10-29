//! Expert parallelism (M3): one [`GpuModel`] per GPU, one thread each, routed experts sharded
//! by id (expert `e` on rank `e % world`), everything else replicated. See
//! `docs/deepseek-ep.md`.
//!
//! Every rank runs the whole forward on the same tokens. In `moe` each rank runs only its own
//! experts into the [t][d] partial sum `y`, then [`GpuModel::moe_reduce`] sums `y` across the
//! ranks (NCCL all-reduce, or [`Backend::Host`]'s copy through host memory), and every rank adds
//! the shared expert to the identical sum. Rank 0's logits are the answer.
//!
//! A command reaches all ranks over channels (the ids are in host memory in one process, so no
//! device broadcast is needed); each rank's reply is collected, and rank 0's returned.

use super::engine::GpuModel;
use super::forward::{Draft, SpecSampler, SpecStep};
use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Barrier, Mutex};
use std::thread::JoinHandle;
use tang_moe::gpu::{DevBuf, Stream};

/// How ranks combine their partial MoE sums.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Read back, sum in rank order on the host, write back. Deterministic, needs no NCCL and
    /// works with several ranks on one device (the tests); slow.
    Host,
    /// NCCL all-reduce on each rank's stream. Needs distinct devices and libnccl.
    #[cfg(feature = "ep")]
    Nccl,
}

/// The rank that holds routed expert `e`.
pub fn owner(e: usize, world: usize) -> usize {
    e % world
}

/// [`Backend::Host`]: each rank posts its partial sum, then every rank sums the slots in rank
/// order, so all ranks get the same bits.
struct HostReduce {
    slots: Mutex<Vec<Vec<f32>>>,
    bar: Barrier,
}

impl HostReduce {
    fn reduce(&self, rank: usize, y: &DevBuf, n: usize, s: &Stream) -> Result<()> {
        super::obs::sync();
        super::obs::d2h(n * 4);
        super::obs::h2d(n * 4);
        s.sync()?;
        let v = y.read::<f32>(0, n)?;
        self.slots.lock().unwrap()[rank] = v;
        self.bar.wait();
        let sum = {
            let sl = self.slots.lock().unwrap();
            let mut acc = sl[0].clone();
            for r in &sl[1..] {
                for (a, b) in acc.iter_mut().zip(r) {
                    *a += *b;
                }
            }
            acc
        };
        // nobody overwrites a slot until everyone has summed
        self.bar.wait();
        // synchronous HtoD; the model's stream is a blocking one, so later kernels see it
        y.write(0, &sum)?;
        Ok(())
    }
}

enum Cmd {
    Forward(Vec<usize>, usize),
    Spec(usize, Vec<f32>, usize),
    /// A whole speculative step on every rank with an identically seeded sampler: the
    /// logits are the same on every rank (one NCCL reduction), so the drafts, the accept walk
    /// and the rewind agree without a broadcast.
    DecodeSpec(usize, Vec<f32>, usize, SpecSampler),
    Reset,
    Misses,
}

enum Reply {
    Forward(Vec<f32>, Vec<f32>),
    Spec(Option<Draft>),
    DecodeSpec(SpecStep, u64),
    Done,
    Misses(usize),
}

struct Rank {
    /// Commands with the group's sequence number: each rank's timeline records it as its
    /// steps' `parent`, so a step's per-rank records (and OTel spans) line up.
    tx: Sender<(u64, Cmd)>,
    rx: Receiver<Result<Reply>>,
    h: Option<JoinHandle<()>>,
}

pub struct EpGroup {
    ranks: Vec<Rank>,
    seq: std::cell::Cell<u64>,
    cfg: super::Config,
}

impl EpGroup {
    /// One rank per entry of `devices` (CUDA ordinals; [`Backend::Host`] allows repeats). Each
    /// rank opens the checkpoint on its own thread and device and uploads only its experts.
    pub fn open(dir: &Path, devices: &[usize], backend: Backend) -> Result<Self> {
        let world = devices.len();
        assert!(world > 0);
        let host = Arc::new(HostReduce {
            slots: Mutex::new(vec![Vec::new(); world]),
            bar: Barrier::new(world),
        });
        #[cfg(feature = "ep")]
        let nccl_id = match backend {
            Backend::Nccl => Some(tang_moe::nccl::unique_id().map_err(|e| anyhow!("{e}"))?),
            Backend::Host => None,
        };
        let mut ranks = Vec::new();
        for (rank, &dev) in devices.iter().enumerate() {
            let (tx, crx) = channel::<(u64, Cmd)>();
            let (rtx, rx) = channel::<Result<Reply>>();
            let dir: PathBuf = dir.to_path_buf();
            let host = host.clone();
            #[cfg(feature = "ep")]
            let nccl_id = nccl_id;
            let h = std::thread::Builder::new()
                .name(format!("ep-rank{rank}"))
                .spawn(move || {
                    let init = || -> Result<GpuModel> {
                        let mut m = GpuModel::open_on(&dir, dev)?;
                        m.obs.rank.set(rank);
                        m.ep_world = world;
                        m.owns_expert = Box::new(move |e| owner(e, world) == rank);
                        m.moe_reduce = match backend {
                            Backend::Host => {
                                Some(Box::new(move |y, n, s| host.reduce(rank, y, n, s)))
                            }
                            #[cfg(feature = "ep")]
                            Backend::Nccl => {
                                let comm = tang_moe::nccl::Comm::new(rank, world, nccl_id.unwrap())
                                    .map_err(|e| anyhow!("rank {rank}: {e}"))?;
                                // only enqueues: captured into the step's graph
                                m.reduce_graphable = true;
                                Some(Box::new(move |y, n, s| {
                                    comm.all_reduce_sum_f32(y, n, s).map_err(|e| anyhow!("{e}"))
                                }))
                            }
                        };
                        Ok(m)
                    };
                    let mut m = match init() {
                        Ok(m) => {
                            let _ = rtx.send(Ok(Reply::Done));
                            m
                        }
                        Err(e) => {
                            let _ = rtx.send(Err(e));
                            return;
                        }
                    };
                    for (seq, cmd) in crx {
                        m.obs.parent.set(seq);
                        let r = match cmd {
                            Cmd::Forward(ids, sp) => {
                                // only rank 0's output is read: the others skip the read-back
                                m.skip_readback = rank != 0;
                                let r = m.forward(&ids, sp, None);
                                m.skip_readback = false;
                                r.map(|(l, h)| Reply::Forward(l, h))
                            }
                            Cmd::Spec(tok, main, sp) => {
                                m.forward_spec(tok, &main, sp).map(Reply::Spec)
                            }
                            Cmd::DecodeSpec(tok, main, sp, mut pick) => m
                                .decode_spec(tok, &main, sp, &mut |_, _| {}, &mut pick)
                                .map(|st| Reply::DecodeSpec(st, pick.rng)),
                            Cmd::Reset => m.reset().map(|_| Reply::Done),
                            Cmd::Misses => Ok(Reply::Misses(m.expert_misses())),
                        };
                        if let Err(e) = &r {
                            // the peers are in (or heading into) this step's collectives and
                            // would wait forever: say what happened, then stop the process
                            eprintln!("tang-llm: deepseek ep rank {rank} failed: {e:#}");
                            #[cfg(feature = "ep")]
                            if matches!(backend, Backend::Nccl) {
                                eprintln!("tang-llm: exiting: the other ranks block in NCCL");
                                std::process::exit(3);
                            }
                        }
                        if rtx.send(r).is_err() {
                            return;
                        }
                    }
                })?;
            ranks.push(Rank { tx, rx, h: Some(h) });
        }
        let cfg = super::Config::load(&{
            let inf = dir.join("inference/config.json");
            if inf.exists() {
                inf
            } else {
                dir.join("config.json")
            }
        })?;
        let g = EpGroup {
            ranks,
            seq: std::cell::Cell::new(0),
            cfg,
        };
        g.collect()?;
        Ok(g)
    }

    pub fn config(&self) -> &super::Config {
        &self.cfg
    }

    pub fn world(&self) -> usize {
        self.ranks.len()
    }

    fn send(&self, mk: impl Fn() -> Cmd) -> Result<Vec<Reply>> {
        let seq = self.seq.get() + 1;
        self.seq.set(seq);
        for r in &self.ranks {
            r.tx.send((seq, mk()))
                .map_err(|_| anyhow!("ep rank exited"))?;
        }
        self.collect()
    }

    /// Every rank's reply to the last command; the first error wins.
    fn collect(&self) -> Result<Vec<Reply>> {
        let mut out = Vec::new();
        let mut err = None;
        for (i, r) in self.ranks.iter().enumerate() {
            match r.rx.recv() {
                Ok(Ok(v)) => out.push(v),
                Ok(Err(e)) => {
                    err.get_or_insert(e.context(format!("ep rank {i}")));
                }
                Err(_) => {
                    err.get_or_insert(anyhow!("ep rank {i} exited"));
                }
            }
        }
        err.map_or(Ok(out), Err)
    }

    /// [`GpuModel::forward`] on every rank; rank 0's logits and DSpark hiddens.
    pub fn forward(&self, ids: &[usize], sp: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        match self.send(|| Cmd::Forward(ids.to_vec(), sp))?.swap_remove(0) {
            Reply::Forward(l, h) => Ok((l, h)),
            _ => unreachable!(),
        }
    }

    /// [`GpuModel::forward_spec`] on every rank (its MoE is sharded too); rank 0's draft.
    pub fn forward_spec(&self, token: usize, main: &[f32], sp: usize) -> Result<Option<Draft>> {
        match self
            .send(|| Cmd::Spec(token, main.to_vec(), sp))?
            .swap_remove(0)
        {
            Reply::Spec(d) => Ok(d),
            _ => unreachable!(),
        }
    }

    /// [`GpuModel::decode_spec`] on every rank, each with a copy of `pick` (see
    /// [`Cmd::DecodeSpec`]); rank 0's step, and `pick`'s RNG set to rank 0's.
    pub fn decode_spec(
        &self,
        token: usize,
        main: &[f32],
        sp: usize,
        pick: &mut SpecSampler,
    ) -> Result<SpecStep> {
        let p = pick.clone();
        let (step, rng) = match self
            .send(|| Cmd::DecodeSpec(token, main.to_vec(), sp, p.clone()))?
            .swap_remove(0)
        {
            Reply::DecodeSpec(s, r) => (s, r),
            _ => unreachable!(),
        };
        pick.rng = rng;
        Ok(step)
    }

    pub fn reset(&self) -> Result<()> {
        self.send(|| Cmd::Reset).map(|_| ())
    }

    /// Routed-expert tensors each rank has uploaded.
    pub fn expert_misses(&self) -> Result<Vec<usize>> {
        Ok(self
            .send(|| Cmd::Misses)?
            .into_iter()
            .map(|r| match r {
                Reply::Misses(n) => n,
                _ => unreachable!(),
            })
            .collect())
    }
}

impl Drop for EpGroup {
    fn drop(&mut self) {
        for r in &mut self.ranks {
            // closing the channel ends the rank's loop
            let (tx, _) = channel::<(u64, Cmd)>();
            drop(std::mem::replace(&mut r.tx, tx));
        }
        for r in &mut self.ranks {
            if let Some(h) = r.h.take() {
                let _ = h.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::forward::tests::tiny;
    use super::*;
    use tang_moe::gpu::Gpu;

    fn kl(p: &[f32], q: &[f32]) -> f64 {
        let ls = |l: &[f32]| {
            let m = l.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
            let z = l.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln() + m;
            l.iter().map(|&x| x as f64 - z).collect::<Vec<f64>>()
        };
        let (p, q) = (ls(p), ls(q));
        p.iter().zip(&q).map(|(a, b)| a.exp() * (a - b)).sum()
    }

    fn argmax(v: &[f32]) -> usize {
        (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
    }

    /// The fixture's steps as (ids, start_pos), or None to skip.
    fn steps() -> Option<Vec<(Vec<usize>, usize)>> {
        Gpu::for_test()?;
        let meta = std::fs::read_to_string(format!("{}/ref/meta.json", tiny())).ok();
        let Some(meta) = meta else {
            eprintln!("skip: no {} (run scripts/deepseek_tiny.py)", tiny());
            return None;
        };
        let meta: serde_json::Value = serde_json::from_str(&meta).unwrap();
        Some(
            meta["steps"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| {
                    let ids = s["ids"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect();
                    (ids, s["start_pos"].as_u64().unwrap() as usize)
                })
                .collect(),
        )
    }

    fn dir() -> PathBuf {
        Path::new(&tiny()).join("hf")
    }

    /// world = 1: the EP path (threaded rank, filter, host reduce of one slot) is the
    /// single-GPU engine bit for bit, DSpark drafts included.
    #[test]
    fn ep_world1_is_exact() {
        let Some(steps) = steps() else { return };
        let mut a = GpuModel::open(&dir()).unwrap();
        let g = EpGroup::open(&dir(), &[0], Backend::Host).unwrap();
        for (ids, sp) in &steps {
            let (la, ma) = a.forward(ids, *sp, None).unwrap();
            let (lb, mb) = g.forward(ids, *sp).unwrap();
            assert_eq!(la, lb, "logits at {sp}");
            assert_eq!(ma, mb, "main hiddens at {sp}");
            let next = argmax(&la);
            let da = a.forward_spec(next, &ma, *sp).unwrap();
            let db = g.forward_spec(next, &mb, *sp).unwrap();
            assert_eq!(da.map(|d| d.logits), db.map(|d| d.logits), "draft at {sp}");
        }
    }

    /// Two ranks on ONE device with complementary experts (even / odd), partial sums added on
    /// the host: the unsharded engine's logits up to f32 summation order, and every expert
    /// tensor uploaded exactly once across the ranks. Exercises the sharding without NCCL.
    #[test]
    fn ep_two_ranks_one_device_match() {
        let Some(steps) = steps() else { return };
        let mut a = GpuModel::open(&dir()).unwrap();
        let g = EpGroup::open(&dir(), &[0, 0], Backend::Host).unwrap();
        let mut worst = 0f64;
        for (ids, sp) in &steps {
            let (la, ma) = a.forward(ids, *sp, None).unwrap();
            let (lb, mb) = g.forward(ids, *sp).unwrap();
            let k = kl(&la, &lb);
            worst = worst.max(k);
            eprintln!("sp {sp}: KL {k:.2e} top1 {}", argmax(&la) == argmax(&lb));
            assert!(k < 1e-4, "KL {k} at {sp}");
            assert_eq!(argmax(&la), argmax(&lb), "top-1 at {sp}");
            let next = argmax(&la);
            a.forward_spec(next, &ma, *sp).unwrap();
            g.forward_spec(next, &mb, *sp).unwrap();
        }
        let per = g.expert_misses().unwrap();
        eprintln!(
            "worst KL {worst:.2e}; expert tensors: unsharded {} vs ranks {per:?}",
            a.expert_misses()
        );
        assert_eq!(per.iter().sum::<usize>(), a.expert_misses());
        assert!(per.iter().all(|&n| n > 0 && n < a.expert_misses()));
    }

    /// The real thing: NCCL across `EP_WORLD` (default 2) distinct GPUs, against the
    /// unsharded engine on device 0. Needs ≥ 2 GPUs and libnccl; see docs/deepseek-ep.md.
    #[cfg(feature = "ep")]
    #[test]
    #[ignore]
    fn ep_nccl_matches_single_gpu() {
        let Some(steps) = steps() else { return };
        let world = std::env::var("EP_WORLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2usize);
        let devs: Vec<usize> = (0..world).collect();
        let g = EpGroup::open(&dir(), &devs, Backend::Nccl).unwrap();
        let mut a = GpuModel::open(&dir()).unwrap();
        for (ids, sp) in &steps {
            let (la, _) = a.forward(ids, *sp, None).unwrap();
            let (lb, _) = g.forward(ids, *sp).unwrap();
            let k = kl(&la, &lb);
            eprintln!("sp {sp}: KL {k:.2e}");
            assert!(k < 1e-4, "KL {k} at {sp}");
            assert_eq!(argmax(&la), argmax(&lb));
        }
    }
}
