//! The derived decode step split across GPUs (`tang_compile::tensor::shard`):
//! one thread and CUDA device per rank, each running its rank program cut
//! at the collectives into segments (one `Program` per segment), the
//! collectives on the host between them (read back, sum / concatenate in
//! rank order, write back: `ep.rs`'s `Backend::Host`, correct and slow —
//! NCCL on the device buffers is the next step). Compares rank 0's logits
//! and caches with the unsharded step on the interpreter.
//!
//! ```text
//! MGPU_MODEL=ds-tiny|coder-tiny|ds-real  MGPU_WORLD=8  ds_multi_gpu
//! ```
//!
//! `ds-real` reads `DS41_DIR` / `DS_REAL_LAYERS` like `ds_real_gpu` (run
//! `ir_ds_real::tests::real_layers_match_reference` first for the routed
//! expert tables). `MGPU_PLAN=eager` takes every split the rules allow
//! (default: the 8×H200 cost model). Exits 1 on a mismatch over 1e-3.
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use tang_bench::ir_shard::{
    case, coder_model, coder_wide, ds_model, ds_wide, native_bytes, rel, summary, Case,
};
use tang_bench::qwen_session::lower;
use tang_compile::tensor::cuda_gemv::DecodeOpts;
use tang_compile::tensor::shard::{
    analyze, assemble, input_shapes, plan_auto, rank_program, segments, CollKind, Comm, CostModel,
    Feed, HostComm,
};
use tang_compile::tensor::TensorId;
use tang_gpu::program::{Program, Runtime};

fn real_case() -> (Case, HashMap<TensorId, String>) {
    use tang_bench::ir_ds_prefill::{derived_step, prefill_graph, DsDims, SEQ};
    use tang_bench::ir_ds_real::*;
    use tang_compile::tensor::step::{seed_cache, StepInput};
    use tang_compile::tensor::Dim;
    let run =
        RealRun::from_env().unwrap_or_else(|| panic!("no {}/inference_config.json", ds_dir()));
    let t_pre = 3;
    let toks = run.rows_of(&prompt(t_pre + 1));
    let used = run
        .read_routed(toks.len())
        .expect("routed tables (run real_layers_match_reference)");
    let rows = used.iter().map(|u| u.len()).max().unwrap();
    // a world-divisible row count: pad the compact expert tables
    let world: usize = std::env::var("MGPU_WORLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let rows = rows.div_ceil(world) * world;
    let dm: DsDims = dims_from_config(&run.cfg, &run.layer_ids, run.vocab_ids.len(), rows, 8);
    let pg = prefill_graph(&dm);
    let mut w = run.dense_inputs(&pg.inputs, &dm).unwrap();
    for (l, u) in used.iter().enumerate() {
        run.expert_tables(l, u, rows, &dm, &mut w);
    }
    let names: Vec<String> = pg.inputs.iter().map(|s| s.name.clone()).collect();
    let mut base: Vec<Vec<f64>> = pg
        .inputs
        .iter()
        .map(|s| {
            if s.name == "tokens" {
                toks[..t_pre].to_vec()
            } else {
                w.remove(&s.name).unwrap()
            }
        })
        .collect();
    let st = derived_step(&pg, &dm, 1).unwrap();
    let nodes: Vec<TensorId> = st.caches.iter().map(|c| c.node).collect();
    let pre = pg.g.bind(&[(SEQ, t_pre)]).unwrap().eval(&base, &nodes);
    let caches: Vec<Vec<f64>> = nodes
        .iter()
        .zip(pre)
        .zip(&st.caches)
        .map(|((&n, v), c)| {
            let shape: Vec<usize> =
                pg.g.shape(n)
                    .iter()
                    .map(|d| if *d == Dim::Sym(SEQ) { t_pre } else { d.n() })
                    .collect();
            seed_cache(c, &shape, &v, t_pre, dm.ctx)
        })
        .collect();
    let mut wnames = HashMap::new();
    let ins: Vec<Vec<f64>> = st
        .inputs
        .iter()
        .enumerate()
        .map(|(k, &x)| match x {
            StepInput::Pos => vec![t_pre as f64],
            StepInput::New(_) => toks[t_pre..t_pre + 1].to_vec(),
            StepInput::Same(i) => {
                wnames.insert(
                    tang_bench::ir_model::input_node(&st.graph, k),
                    names[i].clone(),
                );
                std::mem::take(&mut base[i])
            }
            StepInput::Cache(c) => caches[c].clone(),
        })
        .collect();
    let mut outs = st.outputs.clone();
    outs.extend(st.caches.iter().map(|c| c.updated));
    let want = st.graph.eval(&ins, &outs);
    let weights: HashSet<TensorId> = wnames.keys().copied().collect();
    let caches = tang_bench::ir_shard::cache_pairs(&st);
    (
        Case {
            st,
            ins,
            outs,
            want,
            weights,
            caches,
        },
        wnames,
    )
}

fn use_nccl_env() -> bool {
    std::env::var("MGPU_COMM").as_deref() == Ok("nccl")
}

fn main() {
    let model = std::env::var("MGPU_MODEL").unwrap_or_else(|_| "ds-tiny".into());
    let world: usize = std::env::var("MGPU_WORLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let eager = std::env::var("MGPU_PLAN").as_deref() == Ok("eager");
    let t0 = Instant::now();
    let (c, names) = match model.as_str() {
        "ds-tiny" => (case(&ds_model(ds_wide()), 1, 5), HashMap::new()),
        "coder-tiny" => (case(&coder_model(coder_wide()), 1, 5), HashMap::new()),
        "ds-real" => real_case(),
        m => panic!("MGPU_MODEL {m}?"),
    };
    eprintln!(
        "{model}: step + interpreter reference in {:.1?}",
        t0.elapsed()
    );
    let elem = |t: TensorId| names.get(&t).map_or(1.0, |n| native_bytes(n));
    let cm = if eager {
        CostModel::eager()
    } else {
        CostModel::h200()
    };
    let plan = plan_auto(
        &c.st.graph,
        &c.outs,
        &c.weights,
        &c.caches,
        world,
        &elem,
        &cm,
    );
    let an = analyze(&c.st.graph, &c.outs, &plan);
    let s = summary(&plan, &an);
    eprintln!(
        "plan for {world} ranks: {} expert-parallel, {} tensor-parallel inputs; {} all-reduce + {} all-gather a step",
        s.experts, s.splits, s.all_reduce, s.all_gather
    );
    let comm = HostComm::new(world);
    // one rank per process (multi-host NCCL): MGPU_RANK, on MGPU_DEVICE (0)
    let only: Option<usize> = std::env::var("MGPU_RANK")
        .ok()
        .map(|v| v.parse().expect("MGPU_RANK"));
    let device: Option<usize> = std::env::var("MGPU_DEVICE")
        .ok()
        .map(|v| v.parse().expect("MGPU_DEVICE"));
    assert!(
        only.is_none() || use_nccl_env(),
        "MGPU_RANK needs MGPU_COMM=nccl"
    );
    let use_nccl = std::env::var("MGPU_COMM").as_deref() == Ok("nccl");
    #[cfg(feature = "nccl")]
    let nccl_id = use_nccl.then(|| nccl::shared_id(only));
    #[cfg(not(feature = "nccl"))]
    assert!(!use_nccl, "MGPU_COMM=nccl needs --features nccl");
    let shapes = input_shapes(&c.st.graph);
    let results: Vec<(Vec<Vec<f64>>, f64, f64)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..world)
            .filter(|r| only.is_none_or(|o| o == *r))
            .map(|r| {
                let (c, an, comm, shapes) = (&c, &an, comm.clone(), &shapes);
                sc.spawn(move || {
                    let ord = device.unwrap_or(if only.is_some() { 0 } else { r });
                    let rt =
                        Runtime::cuda_on(ord).unwrap_or_else(|| panic!("no CUDA device {ord}"));
                    let p = rank_program(&c.st.graph, &c.outs, an, &c.weights, r, comm.clone());
                    let ins = p.local_inputs(&c.ins, shapes);
                    let sg = segments(&p);
                    // one Program per segment, its fed values written each step
                    let progs: Vec<Option<(Program, Vec<TensorId>)>> = sg
                        .segments
                        .iter()
                        .map(|seg| {
                            if seg.outputs.is_empty() {
                                return None;
                            }
                            let (g, outs) = seg.graph.substitute_maps(&seg.outputs);
                            let (g, outs) = g.eliminate_common(&outs);
                            let dp = lower(&rt, &g, &outs, &DecodeOpts::default());
                            let init: Vec<Vec<f32>> = seg
                                .feeds
                                .iter()
                                .enumerate()
                                .map(|(k, f)| match f {
                                    Feed::Input(i) => ins[*i].iter().map(|&x| x as f32).collect(),
                                    Feed::Value(_) => vec![
                                        0.0;
                                        seg.graph
                                            .shape(tang_bench::ir_model::input_node(&seg.graph, k))
                                            .iter()
                                            .map(|d| d.n())
                                            .product()
                                    ],
                                })
                                .collect();
                            let refs: Vec<&[f32]> = init.iter().map(|v| v.as_slice()).collect();
                            let prog = Program::new(&rt, dp.program, &refs);
                            for o in &outs {
                                if !prog.has_buffer(*o) {
                                    eprintln!(
                                        "no buffer for {o:?}: {}",
                                        tang_compile::tensor::shard::describe(&g, *o)
                                    );
                                }
                            }
                            Some((prog, outs))
                        })
                        .collect();
                    let step = || -> Vec<Vec<f64>> {
                        let mut vals: HashMap<usize, Vec<f64>> = sg
                            .input_nodes
                            .iter()
                            .map(|&(i, k)| (i, ins[k].clone()))
                            .collect();
                        for (si, seg) in sg.segments.iter().enumerate() {
                            for b in &sg.before[si] {
                                let x = &vals[&b.src];
                                let y = match b.kind {
                                    CollKind::AllReduce => comm.all_reduce(r, x),
                                    CollKind::AllGather(_) => comm.all_gather(r, x, b.outer),
                                };
                                vals.insert(b.dst, y);
                            }
                            let Some((prog, outs)) = &progs[si] else {
                                continue;
                            };
                            for (k, f) in seg.feeds.iter().enumerate() {
                                if let Feed::Value(n) = f {
                                    if !prog.has_input(k) {
                                        continue;
                                    }
                                    let v: Vec<f32> = vals[n].iter().map(|&x| x as f32).collect();
                                    prog.write(&rt, prog.input(k), &v);
                                }
                            }
                            prog.run(&rt);
                            for (d, o) in seg.defines.iter().zip(outs) {
                                assert!(
                                    prog.has_buffer(*o),
                                    "segment {si} output {o:?} (node {d}) has no buffer"
                                );
                                vals.insert(
                                    *d,
                                    prog.read(&rt, *o).iter().map(|&x| x as f64).collect(),
                                );
                            }
                        }
                        sg.outputs.iter().map(|o| vals[o].clone()).collect()
                    };
                    #[cfg(feature = "nccl")]
                    if let Some(id) = nccl_id {
                        let mut dev = nccl::Rank::new(ord, r, world, id);
                        dev.stage_inputs(&sg, &ins);
                        let dstep = || dev.step(&rt, &sg, &progs, &ins);
                        let out = dstep();
                        dev.comm_ms.set(0.0);
                        let t = Instant::now();
                        let iters = 20;
                        for _ in 0..iters {
                            dstep();
                        }
                        let n = iters as f64;
                        return (
                            out,
                            t.elapsed().as_secs_f64() * 1e3 / n,
                            dev.comm_ms.get() / n,
                        );
                    }
                    let out = step();
                    let t = Instant::now();
                    let iters = 5;
                    for _ in 0..iters {
                        step();
                    }
                    (
                        out,
                        t.elapsed().as_secs_f64() * 1e3 / iters as f64,
                        f64::NAN,
                    )
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let worst = match only {
        // this rank alone: replicated outputs whole, split ones against its slice
        Some(r) => c
            .outs
            .iter()
            .enumerate()
            .map(|(o, t)| match an.out_local[o] {
                None => rel(&results[0].0[o], &c.want[o]),
                Some(d) => {
                    let full: Vec<usize> = c.st.graph.shape(*t).iter().map(|x| x.n()).collect();
                    let outer: usize = full[..d].iter().product();
                    let row = c.want[o].len() / outer / world;
                    let mine: Vec<f64> = (0..outer)
                        .flat_map(|i| {
                            c.want[o][(i * world + r) * row..(i * world + r + 1) * row].to_vec()
                        })
                        .collect();
                    rel(&results[0].0[o], &mine)
                }
            })
            .fold(0.0, f64::max),
        None => {
            let ranks: Vec<Vec<Vec<f64>>> = results.iter().map(|r| r.0.clone()).collect();
            let got = assemble(&c.st.graph, &c.outs, &an, &ranks);
            got.iter()
                .zip(&c.want)
                .map(|(a, b)| rel(a, b))
                .fold(0.0, f64::max)
        }
    };
    let ok = worst < 1e-3;
    println!(
        "{model} on {world} GPUs{}: vs unsharded interpreter rel err {worst:.2e}, {:.2} ms a step, {:.2} ms of it \
         in collectives ({} collectives, f32 weights): {}",
        only.map_or(String::new(), |r| format!(" (rank {r} of this process)")),
        results[0].1,
        results[0].2,
        if use_nccl { "NCCL device" } else { "host" },
        if ok { "PASS" } else { "FAIL" }
    );
    if !ok {
        std::process::exit(1);
    }
}

/// Collectives on the device: segment outputs stay in their programs'
/// buffers, a collective reads one in place into a staging buffer and NCCL
/// reduces / gathers it on the runtime's own stream, and fed inputs are
/// device-to-device copies. Nothing crosses to the host until the outputs.
#[cfg(feature = "nccl")]
mod nccl {
    use std::collections::HashMap;
    use std::mem::ManuallyDrop;

    use tang_compile::tensor::shard::{gather_copies, CollKind, Feed, Segmented};
    use tang_compile::tensor::TensorId;
    use tang_gpu::program::{Program, Runtime};
    use tang_moe::gpu::{copy_d2d_async, DevBuf, Gpu, Stream};
    use tang_moe::nccl::{Comm, UniqueId};

    pub struct Rank {
        comm: Comm,
        world: usize,
        // staging per collective (by its rank-program node): (send, recv, result)
        bufs: std::cell::RefCell<HashMap<usize, (DevBuf, DevBuf, DevBuf)>>,
        /// Graph inputs read straight by a collective or another segment, on the device.
        inputs: HashMap<usize, DevBuf>,
        /// Wall time in collectives (each bracketed by stream syncs).
        pub comm_ms: std::cell::Cell<f64>,
    }

    /// The NCCL id every rank joins with. One process: made here. One rank a
    /// process: rank 0 makes it and writes it (hex) to `MGPU_NCCL_ID_FILE`;
    /// other ranks wait for that file (copied over by the launcher).
    pub fn shared_id(only: Option<usize>) -> UniqueId {
        let Some(r) = only else {
            return tang_moe::nccl::unique_id().expect("libnccl");
        };
        let path = std::env::var("MGPU_NCCL_ID_FILE").expect("MGPU_NCCL_ID_FILE");
        if r == 0 {
            let id = tang_moe::nccl::unique_id().expect("libnccl");
            let hex: String =
                id.0.internal
                    .iter()
                    .map(|b| format!("{:02x}", *b as u8))
                    .collect();
            std::fs::write(&path, hex).expect("write NCCL id");
            return id;
        }
        let hex = loop {
            match std::fs::read_to_string(&path) {
                Ok(h) if h.len() >= 256 => break h,
                _ => std::thread::sleep(std::time::Duration::from_millis(200)),
            }
        };
        let mut id = tang_moe::nccl::unique_id().expect("libnccl");
        for (k, b) in id.0.internal.iter_mut().enumerate() {
            *b = u8::from_str_radix(&hex[2 * k..2 * k + 2], 16).unwrap() as _;
        }
        id
    }

    /// Where a rank-program node's value lives on the device.
    #[derive(Clone, Copy)]
    enum Loc {
        Prog(usize, TensorId),
        Buf(u64, usize),
    }

    impl Rank {
        /// Upload the graph inputs a collective or a fed value reads (once).
        pub fn stage_inputs(&mut self, sg: &Segmented, ins: &[Vec<f64>]) {
            let mut need: Vec<usize> = sg.before.iter().flatten().map(|b| b.src).collect();
            for seg in &sg.segments {
                need.extend(seg.feeds.iter().filter_map(|f| match f {
                    Feed::Value(n) => Some(*n),
                    _ => None,
                }));
            }
            for &(i, k) in &sg.input_nodes {
                if need.contains(&i) && !self.inputs.contains_key(&i) {
                    let v: Vec<f32> = ins[k].iter().map(|&x| x as f32).collect();
                    self.inputs.insert(i, DevBuf::from_slice(&v).unwrap());
                }
            }
        }

        pub fn new(ordinal: usize, rank: usize, world: usize, id: UniqueId) -> Self {
            Gpu::at(ordinal)
                .and_then(|g| g.bind())
                .expect("bind device");
            let comm = Comm::new(rank, world, id).expect("ncclCommInitRank");
            Rank {
                comm,
                world,
                bufs: Default::default(),
                inputs: Default::default(),
                comm_ms: Default::default(),
            }
        }

        pub fn step(
            &self,
            rt: &Runtime,
            sg: &Segmented,
            progs: &[Option<(Program, Vec<TensorId>)>],
            ins: &[Vec<f64>],
        ) -> Vec<Vec<f64>> {
            // the runtime's stream, not owned here
            let s = ManuallyDrop::new(Stream(rt.cu_stream()));
            let mut loc: HashMap<usize, Loc> = self
                .inputs
                .iter()
                .map(|(&i, b)| (i, Loc::Buf(b.ptr, b.len / 4)))
                .collect();
            let ptr = |l: Loc| -> (u64, usize) {
                match l {
                    Loc::Prog(si, t) => progs[si].as_ref().unwrap().0.cuda_buffer(rt, t),
                    Loc::Buf(p, n) => (p, n),
                }
            };
            let mut bufs = self.bufs.borrow_mut();
            for (si, seg) in sg.segments.iter().enumerate() {
                for b in &sg.before[si] {
                    rt.sync();
                    let tc = std::time::Instant::now();
                    let (src, n) = ptr(loc[&b.src]);
                    let full = match b.kind {
                        CollKind::AllReduce => n,
                        CollKind::AllGather(_) => n * self.world,
                    };
                    let (send, recv, res) = bufs.entry(b.dst).or_insert_with(|| {
                        (
                            DevBuf::zeroed(n * 4).unwrap(),
                            DevBuf::zeroed(full * 4).unwrap(),
                            DevBuf::zeroed(full * 4).unwrap(),
                        )
                    });
                    match b.kind {
                        CollKind::AllReduce => {
                            copy_d2d_async(res.ptr, src, n * 4, &s).unwrap();
                            self.comm.all_reduce_sum_f32(res, n, &s).unwrap();
                        }
                        CollKind::AllGather(_) => {
                            copy_d2d_async(send.ptr, src, n * 4, &s).unwrap();
                            self.comm.all_gather_f32(send, recv, n, &s).unwrap();
                            for (a, d, len) in gather_copies(self.world, b.outer, n) {
                                copy_d2d_async(
                                    res.ptr + 4 * d as u64,
                                    recv.ptr + 4 * a as u64,
                                    len * 4,
                                    &s,
                                )
                                .unwrap();
                            }
                        }
                    }
                    loc.insert(b.dst, Loc::Buf(res.ptr, full));
                    rt.sync();
                    self.comm_ms
                        .set(self.comm_ms.get() + tc.elapsed().as_secs_f64() * 1e3);
                }
                let Some((prog, outs)) = &progs[si] else {
                    continue;
                };
                for (k, f) in seg.feeds.iter().enumerate() {
                    if let Feed::Value(n) = f {
                        if !prog.has_input(k) {
                            continue;
                        }
                        let (dst, dn) = prog.cuda_buffer(rt, prog.input(k));
                        let (src, sn) = ptr(loc[n]);
                        assert_eq!(dn, sn, "fed value length");
                        copy_d2d_async(dst, src, dn * 4, &s).unwrap();
                    }
                }
                prog.run(rt);
                for (d, o) in seg.defines.iter().zip(outs) {
                    loc.insert(*d, Loc::Prog(si, *o));
                }
            }
            let _ = ins;
            sg.outputs
                .iter()
                .map(|o| match loc[o] {
                    Loc::Prog(si, t) => progs[si]
                        .as_ref()
                        .unwrap()
                        .0
                        .read(rt, t)
                        .iter()
                        .map(|&x| x as f64)
                        .collect(),
                    Loc::Buf(p, n) => {
                        rt.sync();
                        ManuallyDrop::new(DevBuf { ptr: p, len: n * 4 })
                            .read::<f32>(0, n)
                            .unwrap()
                            .iter()
                            .map(|&x| x as f64)
                            .collect()
                    }
                })
                .collect()
        }
    }
}
