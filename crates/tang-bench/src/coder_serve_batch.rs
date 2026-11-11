//! Qwen3-Coder-Next served from ONE batched derived step (`TANG_CODER_DERIVED_BATCH=1`,
//! `--batch` > 1): every sequence slot is a stream of [`batch_graph`]'s batch dim, so a
//! decode window over the ready slots is one launch (#232's batched step: ragged positions,
//! experts grouped across streams, batch int8 stages) instead of a step per slot.
//!
//! - **Caches**: the batched step owns them, `[B, …]` with the stream axis outermost, so
//!   slot `s`'s cache is one contiguous block (`s · W/B .. (s+1) · W/B` words).
//! - **Idle streams** step too (one launch does all `B`): their K/V write lands at their
//!   own next position, which nothing reads before the slot writes it again; their ring
//!   and state updates are not copied back (only the active streams' blocks are).
//! - **Sampling**: [`append_sampler_topk`] with per-stream params; when every active stream
//!   is greedy, a batched step without the sampler (sharing weights and caches) runs and
//!   the host takes each row's first maximum.
//! - **Prefill**: a single-stream step (no sampler) with its own caches: the slot's block is
//!   copied in, the prompt but its last token stepped, the block copied back; the last token
//!   goes through the batched step (so it is sampled like any decode token). #234's chunked
//!   step replaces the single-stream one when it lands.
//! - **Snapshots**: a slot's ring/state blocks (running) and its K/V rows (positional).
use std::cell::Cell;
use std::path::Path;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use tang_compile::tensor::int8_sched::BlockOpts;
use tang_compile::tensor::step::{
    batch_graph, derive_step, derive_step_with, Rows, StepGraph, StepInput, StepOptions, StepSpec,
};
use tang_compile::tensor::{Dim, TensorId};
use tang_gpu::program::{Program, Runtime, Shared};
use tang_llm::flash::engine::{Positional, RunningState, WinStats};
use tang_llm::flash::sampler::{RowSampling, Sampler};
use tang_llm::flash::serve::ServeEngine;
use tang_llm::gguf::{GgmlType, Gguf};

use crate::ir_coder_prefill::{prefill_graph, CoderInput, SEQ};
use crate::ir_coder_real::{bind, dims_from_gguf, lower_real_step, lower_real_step_with, RealOpts};
use crate::ir_sample::{append_sampler_topk, SEED, TEMPERATURE, TOP_K, TOP_P};
use crate::qwen_session::{runtime, slot_plan};

fn words_of_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks(4)
        .map(|c| {
            let mut a = [0u8; 4];
            a[..c.len()].copy_from_slice(c);
            f32::from_le_bytes(a)
        })
        .collect()
}

/// One lowered step program and where its inputs and outputs are.
struct Step {
    prog: Program,
    pos_in: usize,
    tok_in: usize,
    params_in: Option<usize>,
    /// `[logits, token?, updated caches…]`.
    outs: Vec<TensorId>,
    /// Outputs before the caches.
    nout: usize,
    /// Cache `c`'s step input.
    cache_in: Vec<usize>,
}

impl Step {
    fn cache(&self, c: usize) -> TensorId {
        self.prog.input(self.cache_in[c])
    }
    fn updated(&self, c: usize) -> TensorId {
        self.outs[self.nout + c]
    }
}

struct Slot {
    tokens: Vec<u32>,
    params: [f32; 4],
    allocated: bool,
}

/// What a cache holds.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// Rows by position: `row` words a position.
    Rows { row: usize },
    /// A ring or a carried state: replaced each step (copied back from its update).
    Running,
}

pub struct BatchedCoder {
    g: Gguf,
    rt: Runtime,
    b: usize,
    ctx: usize,
    vocab: usize,
    /// The batched step with the sampler.
    sampled: Step,
    /// The batched step without it (weights and caches bound to `sampled`'s).
    greedy: Step,
    /// The single-stream prefill step (no sampler; weights bound, own caches).
    solo: Step,
    /// Per cache: its kind and one stream's words.
    caches: Vec<(Kind, usize)>,
    slots: Vec<Slot>,
    /// The last step: whether it ran the greedy program, and which slot `logits()` reads.
    last_greedy: Cell<bool>,
    logits_slot: usize,
    last: WinStats,
    presence_noted: Cell<bool>,
}

fn find(st: &StepGraph, want: StepInput) -> Option<usize> {
    st.inputs.iter().position(|s| *s == want)
}

impl BatchedCoder {
    pub fn load(path: &Path, ctx: usize, nb: usize) -> Result<Self> {
        let t0 = Instant::now();
        let g = Gguf::open(path)?;
        let dims = dims_from_gguf(&g).map_err(anyhow::Error::msg)?;
        let mut pg = prefill_graph(&dims);
        let mut tys = bind(&pg, &g).map_err(anyhow::Error::msg)?;
        let s = append_sampler_topk(&mut pg.g, pg.logits);
        pg.inputs.push(CoderInput {
            name: "sample.params".into(),
            shape: vec![Dim::Sym(SEQ), Dim::Fixed(4)],
        });
        tys.push(None);
        let (tokens_k, params_k) = (pg.input("tokens"), pg.input("sample.params"));
        let spec = StepSpec {
            seq: SEQ,
            new: 1,
            capacity: ctx,
        };
        let (bg, bouts) =
            batch_graph(&pg.g, &[pg.logits, s.token], SEQ, "B").map_err(|e| anyhow::anyhow!("batch_graph: {}", e.0))?;
        let opts = StepOptions {
            rows: Rows::All,
            batch: Some("B"),
        };
        let st_s = derive_step_with(&bg, &bouts, &spec, &opts)
            .map_err(|e| anyhow::anyhow!("batched step: {}", e.0))?;
        let st_g = derive_step_with(&bg, &bouts[..1], &spec, &opts)
            .map_err(|e| anyhow::anyhow!("batched greedy step: {}", e.0))?;
        let st_1 = derive_step(&pg.g, &[pg.logits], &spec)
            .map_err(|e| anyhow::anyhow!("solo step: {}", e.0))?;
        let group = std::env::var("TANG_BATCH_GROUP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32);
        let ro = RealOpts {
            batch: Some(nb),
            group,
            block: BlockOpts::batched(),
        };
        let low_s = lower_real_step_with(&pg, &st_s, &tys, &ro);
        let low_g = lower_real_step_with(&pg, &st_g, &tys, &ro);
        let low_1 = lower_real_step(&pg, &st_1, &tys);
        eprintln!(
            "coder derived (batched): B={nb}: {} kernels sampled, {} greedy, {} solo prefill; built in {:.1} s",
            low_s.program.program.kernels.len(),
            low_g.program.program.kernels.len(),
            low_1.program.program.kernels.len(),
            t0.elapsed().as_secs_f64()
        );
        let rt = runtime();
        ensure!(matches!(rt, Runtime::Cuda(_)), "the batched derived engine needs CUDA");
        ensure!(
            st_s.caches.len() == st_1.caches.len() && st_g.caches.len() == st_1.caches.len(),
            "batched and solo steps cache different things"
        );

        // the sampled batched step owns the weights and the caches
        let plan = slot_plan(&low_s.program.program, &low_s.outs, &low_s.inplace);
        let empty: Vec<&[f32]> = vec![&[]; st_s.inputs.len()];
        let prog = Program::with_plan(&rt, low_s.program.program, &empty, &plan);
        let t1 = Instant::now();
        let mut bytes = 0usize;
        for (k, si) in st_s.inputs.iter().enumerate() {
            let StepInput::Same(i) = *si else { continue };
            let t = g.info(&pg.inputs[i].name)?;
            let words = match (low_s.storage[k], t.ty) {
                (Some(_), _) | (None, GgmlType::F32) => words_of_bytes(g.bytes(t)),
                (None, _) => g.dequantize(t)?,
            };
            bytes += words.len() * 4;
            prog.write(&rt, prog.input(k), &words);
        }
        eprintln!(
            "coder derived (batched): {:.1} GB of weights up in {:.1} s",
            bytes as f64 / 1e9,
            t1.elapsed().as_secs_f64()
        );
        let mk = |st: &StepGraph, prog: Program, outs: Vec<TensorId>, sampled: bool| -> Result<Step> {
            Ok(Step {
                prog,
                pos_in: find(st, StepInput::Pos).context("pos input")?,
                tok_in: find(st, StepInput::New(tokens_k)).context("token input")?,
                params_in: if sampled { find(st, StepInput::New(params_k)) } else { None },
                nout: if sampled { 2 } else { 1 },
                cache_in: (0..st.caches.len())
                    .map(|c| find(st, StepInput::Cache(c)).expect("cache input"))
                    .collect(),
                outs,
            })
        };
        let sampled = mk(&st_s, prog, low_s.outs, true)?;
        ensure!(sampled.params_in.is_some(), "no per-stream params input");

        // step-2 input -> sampled input: weights by GGUF input; caches by order (both
        // derived from the same batch graph)
        let bind_to = |st: &StepGraph, caches: bool| -> Vec<(usize, usize)> {
            st.inputs
                .iter()
                .enumerate()
                .filter_map(|(k, si)| match *si {
                    StepInput::Same(i) => find(&st_s, StepInput::Same(i)).map(|k1| (k, k1)),
                    StepInput::Cache(c) if caches => Some((k, sampled.cache_in[c])),
                    _ => None,
                })
                .collect()
        };
        let share = |map: &[(usize, usize)]| -> Vec<Shared<'_>> {
            map.iter()
                .map(|&(k, k1)| Shared {
                    input: k,
                    from: &sampled.prog,
                    tensor: sampled.prog.input(k1),
                })
                .collect()
        };
        let map_g = bind_to(&st_g, true);
        let plan_g = slot_plan(&low_g.program.program, &low_g.outs, &low_g.inplace);
        let prog_g = Program::with_shared(
            &rt,
            low_g.program.program,
            &vec![&[][..]; st_g.inputs.len()],
            &plan_g,
            &share(&map_g),
        );
        let map_1 = bind_to(&st_1, false);
        let plan_1 = slot_plan(&low_1.program.program, &low_1.outs, &low_1.inplace);
        let prog_1 = Program::with_shared(
            &rt,
            low_1.program.program,
            &vec![&[][..]; st_1.inputs.len()],
            &plan_1,
            &share(&map_1),
        );
        let mut greedy = mk(&st_g, prog_g, low_g.outs, false)?;
        let mut solo = mk(&st_1, prog_1, low_1.outs, false)?;

        // caches: the solo step's must be one stream's block of the batched one
        let mut caches = Vec::new();
        for (c, (cb, c1)) in st_s.caches.iter().zip(&st_1.caches).enumerate() {
            let wb = sampled.prog.words_of(sampled.cache(c));
            let w1 = solo.prog.words_of(solo.cache(c));
            ensure!(
                wb == nb * w1 && cb.state == c1.state && cb.ring == c1.ring,
                "cache {c}: batched {wb} words vs {nb} × {w1} solo (state {} / {}, ring {:?} / {:?})",
                cb.state,
                c1.state,
                cb.ring,
                c1.ring
            );
            let kind = if c1.state || c1.ring.is_some() {
                Kind::Running
            } else {
                ensure!(c1.axis == 0 && w1 % ctx == 0, "row cache {c}: axis {}", c1.axis);
                Kind::Rows { row: w1 / ctx }
            };
            caches.push((kind, w1));
        }
        let graphs = std::env::var("TANG_CODER_DERIVED_GRAPH").as_deref() != Ok("0");
        if graphs {
            for st in [&mut greedy, &mut solo] {
                st.prog.capture(&rt);
            }
        }
        let mut sampled = sampled;
        if graphs {
            sampled.prog.capture(&rt);
        }
        let mut me = BatchedCoder {
            g,
            rt,
            b: nb,
            ctx,
            vocab: dims.vocab,
            sampled,
            greedy,
            solo,
            caches,
            slots: (0..nb)
                .map(|_| Slot {
                    tokens: Vec::new(),
                    params: [0.0, 1.0, 0.0, 0.0],
                    allocated: false,
                })
                .collect(),
            last_greedy: Cell::new(true),
            logits_slot: 0,
            last: WinStats::default(),
            presence_noted: Cell::new(false),
        };
        for s in 0..nb {
            me.clear_slot(s);
        }
        let dev_s = me.sampled.prog.time(&me.rt, 10);
        let dev_g = me.greedy.prog.time(&me.rt, 10);
        let dev_1 = me.solo.prog.time(&me.rt, 10);
        let kv_mb: usize = me
            .caches
            .iter()
            .filter(|c| matches!(c.0, Kind::Rows { .. }))
            .map(|c| c.1 * 4)
            .sum::<usize>()
            >> 20;
        eprintln!(
            "coder derived (batched): B={nb} step {dev_s:.3} ms sampled / {dev_g:.3} ms greedy device, \
             solo prefill step {dev_1:.3} ms; ctx {ctx}, {kv_mb} MB of K/V a slot; ready in {:.1} s",
            t0.elapsed().as_secs_f64()
        );
        Ok(me)
    }

    /// Zero slot `s`'s running blocks and the K/V rows it wrote.
    fn clear_slot(&mut self, s: usize) {
        let n = self.slots[s].tokens.len();
        let p = &self.sampled.prog;
        for (c, &(kind, w)) in self.caches.iter().enumerate() {
            let t = self.sampled.cache(c);
            match kind {
                Kind::Running => p.write_words(&self.rt, t, s * w, &vec![0.0; w]),
                Kind::Rows { row } if n > 0 => {
                    p.write_words(&self.rt, t, s * w, &vec![0.0; n.min(self.ctx) * row])
                }
                Kind::Rows { .. } => {}
            }
        }
        self.slots[s].tokens.clear();
    }

    /// One batched step: `rows` (slot, token) advance; every other stream computes at its own
    /// next position and keeps its running state. Returns each row's next token.
    fn step(&mut self, rows: &[(usize, u32)]) -> Result<Vec<u32>> {
        let t0 = Instant::now();
        let mut pos = vec![0f32; self.b];
        let mut tok = vec![0f32; self.b];
        let mut params = vec![0f32; self.b * 4];
        for s in 0..self.b {
            pos[s] = self.slots[s].tokens.len().min(self.ctx - 1) as f32;
            params[4 * s + TOP_P] = 1.0;
        }
        let mut all_greedy = true;
        for &(s, t) in rows {
            ensure!(self.slots[s].tokens.len() < self.ctx, "slot {s} is past its context ({} tokens)", self.ctx);
            tok[s] = t as f32;
            params[4 * s..4 * s + 4].copy_from_slice(&self.slots[s].params);
            all_greedy &= self.slots[s].params[TEMPERATURE] <= 0.0;
        }
        let st = if all_greedy { &self.greedy } else { &self.sampled };
        let p = &st.prog;
        p.write(&self.rt, p.input(st.pos_in), &pos);
        p.write(&self.rt, p.input(st.tok_in), &tok);
        if let Some(k) = st.params_in {
            p.write(&self.rt, p.input(k), &params);
        }
        p.run(&self.rt);
        // the active streams' running blocks back into the caches
        for (c, &(kind, w)) in self.caches.iter().enumerate() {
            if kind != Kind::Running {
                continue;
            }
            for &(s, _) in rows {
                p.copy_words(&self.rt, st.cache(c), s * w, p, st.updated(c), s * w, w);
            }
        }
        let out: Vec<u32> = if all_greedy {
            let l = p.read(&self.rt, st.outs[0]);
            let v = self.vocab;
            rows.iter()
                .map(|&(s, _)| {
                    let r = &l[s * v..(s + 1) * v];
                    let mut a = 0;
                    for (i, &x) in r.iter().enumerate() {
                        if x > r[a] {
                            a = i;
                        }
                    }
                    a as u32
                })
                .collect()
        } else {
            let t = p.read(&self.rt, st.outs[1]);
            rows.iter().map(|&(s, _)| t[s] as u32).collect()
        };
        for &(s, t) in rows {
            self.slots[s].tokens.push(t);
        }
        self.last_greedy.set(all_greedy);
        self.last = WinStats {
            t: rows.len(),
            wall_ms: t0.elapsed().as_secs_f64() * 1e3,
            ..WinStats::default()
        };
        Ok(out)
    }

    /// Prefill `ids` into slot `s`: all but the last through the solo step on a copy of the
    /// slot's block, the last through the batched step.
    fn feed(&mut self, s: usize, ids: &[u32]) -> Result<u32> {
        ensure!(!ids.is_empty(), "empty prefill");
        let n0 = self.slots[s].tokens.len();
        ensure!(n0 + ids.len() <= self.ctx, "prompt past the context ({} tokens)", self.ctx);
        let head = &ids[..ids.len() - 1];
        if !head.is_empty() {
            let (bp, sp) = (&self.sampled.prog, &self.solo.prog);
            for (c, &(_, w)) in self.caches.iter().enumerate() {
                sp.copy_words(&self.rt, self.solo.cache(c), 0, bp, self.sampled.cache(c), s * w, w);
            }
            for (i, &id) in head.iter().enumerate() {
                sp.write(&self.rt, sp.input(self.solo.pos_in), &[(n0 + i) as f32]);
                sp.write(&self.rt, sp.input(self.solo.tok_in), &[id as f32]);
                sp.run(&self.rt);
                for (c, &(kind, _)) in self.caches.iter().enumerate() {
                    if kind == Kind::Running {
                        sp.copy(&self.rt, self.solo.cache(c), sp, self.solo.updated(c));
                    }
                }
            }
            for (c, &(_, w)) in self.caches.iter().enumerate() {
                bp.copy_words(&self.rt, self.sampled.cache(c), s * w, sp, self.solo.cache(c), 0, w);
            }
            self.slots[s].tokens.extend_from_slice(head);
        }
        let t = self.step(&[(s, ids[ids.len() - 1])])?[0];
        self.logits_slot = s;
        Ok(t)
    }

    fn verify_slot(&mut self, s: usize, cur: u32, drafts: &[u32]) -> Result<Vec<u32>> {
        let mut kept = vec![self.step(&[(s, cur)])?[0]];
        for &d in drafts {
            if kept[kept.len() - 1] != d {
                break;
            }
            kept.push(self.step(&[(s, d)])?[0]);
        }
        self.logits_slot = s;
        Ok(kept)
    }

    fn running(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.caches
            .iter()
            .enumerate()
            .filter(|(_, c)| c.0 == Kind::Running)
            .map(|(c, &(_, w))| (c, w))
    }

    fn rows(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.caches.iter().enumerate().filter_map(|(c, &(k, w))| match k {
            Kind::Rows { row } => Some((c, w, row)),
            Kind::Running => None,
        })
    }
}

impl ServeEngine for BatchedCoder {
    fn tokens(&self) -> &Vec<u32> {
        &self.slots[0].tokens
    }
    fn tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.slots[0].tokens
    }
    fn reset(&mut self) {
        self.clear_slot(0);
    }
    fn gguf(&self) -> &Gguf {
        &self.g
    }
    fn last(&self) -> WinStats {
        self.last
    }
    fn save_running(&mut self) -> Result<RunningState> {
        self.save_running_at(0)
    }
    fn restore_running(&mut self, s: &RunningState, prefix: &[u32]) -> Result<()> {
        self.restore_running_at(0, s, prefix)
    }
    fn save_positional(&mut self, from: usize, to: usize) -> Result<Positional> {
        self.save_positional_at(0, from, to)
    }
    fn restore_positional(&mut self, p: &Positional, from: usize) -> Result<()> {
        self.restore_positional_at(0, p, from)
    }
    fn prefill(&mut self, ids: &[u32], _chunk: usize) -> Result<u32> {
        self.feed(0, ids)
    }
    /// Slot 0, the caller having extended `tokens`: step `tokens[pos0..pos0 + t]`.
    fn window(&mut self, pos0: usize, t: usize) -> Result<Vec<u32>> {
        ensure!(pos0 + t <= self.slots[0].tokens.len(), "window past the sequence");
        // the caller extended `tokens` past this window (a span, window by window): step
        // these `t`, which re-appends them, and put the span's tail back
        let ids = self.slots[0].tokens[pos0..pos0 + t].to_vec();
        let tail = self.slots[0].tokens[pos0 + t..].to_vec();
        self.slots[0].tokens.truncate(pos0);
        let mut out = Vec::with_capacity(t);
        let mut res = Ok(());
        for id in ids {
            match self.step(&[(0, id)]) {
                Ok(v) => out.push(v[0]),
                Err(e) => {
                    res = Err(e);
                    break;
                }
            }
        }
        self.slots[0].tokens.extend_from_slice(&tail);
        res?;
        self.logits_slot = 0;
        Ok(out)
    }
    fn verify(&mut self, cur: u32, drafts: &[u32]) -> Result<Vec<u32>> {
        self.verify_slot(0, cur, drafts)
    }
    fn logits(&self, t: usize) -> Vec<f32> {
        let st = if self.last_greedy.get() { &self.greedy } else { &self.sampled };
        let l = st.prog.read(&self.rt, st.outs[0]);
        let v = self.vocab;
        let row = &l[self.logits_slot * v..(self.logits_slot + 1) * v];
        (0..t).flat_map(|_| row.iter().copied()).collect()
    }
    fn set_sampling(&mut self, temperature: f32, seed: u32) {
        self.slots[0].params = [temperature.max(0.0), 1.0, 0.0, (seed & 0xff_ffff) as f32];
    }
    fn sampler_mut(&mut self) -> Option<&mut Sampler> {
        None
    }
    fn sampler_begin(&mut self, _pos: usize) -> Result<()> {
        Ok(())
    }
    fn max_slots(&self) -> usize {
        self.b
    }
    fn fit_slots(&mut self) -> usize {
        self.b
    }
    fn slot_tokens(&self, slot: usize) -> &[u32] {
        &self.slots[slot].tokens
    }
    fn alloc_slot(&mut self) -> Result<usize> {
        let s = (0..self.b)
            .find(|&s| !self.slots[s].allocated)
            .context("every sequence slot is allocated")?;
        self.clear_slot(s);
        self.slots[s].allocated = true;
        Ok(s)
    }
    fn free_slot(&mut self, slot: usize) {
        self.clear_slot(slot);
        self.slots[slot].allocated = false;
    }
    fn prefill_slot(&mut self, slot: usize, ids: &[u32]) -> Result<u32> {
        self.feed(slot, ids)
    }
    fn decode_batch(&mut self, rows: &[(usize, u32)]) -> Result<Vec<u32>> {
        self.step(rows)
    }
    fn set_slot_sampling(&mut self, slot: usize, s: RowSampling) -> Result<()> {
        if s.presence != 0.0 && !self.presence_noted.replace(true) {
            eprintln!("coder derived: presence penalty not supported on the device sampler; ignored");
        }
        let mut p = [0.0f32; 4];
        p[TEMPERATURE] = s.temperature.max(0.0);
        p[TOP_P] = if s.top_p > 0.0 && s.top_p < 1.0 { s.top_p } else { 1.0 };
        p[TOP_K] = s.top_k as f32;
        p[SEED] = (s.seed & 0xff_ffff) as f32;
        self.slots[slot].params = p;
        Ok(())
    }
    fn save_running_at(&mut self, slot: usize) -> Result<RunningState> {
        let p = &self.sampled.prog;
        let mut parts: Vec<Vec<f32>> = self
            .running()
            .map(|(c, w)| p.read_words(&self.rt, self.sampled.cache(c), slot * w..(slot + 1) * w))
            .collect();
        ensure!(parts.len() % 2 == 0, "odd running-state caches");
        let gdn = parts.len() / 2;
        parts.push(Vec::new());
        parts.push(Vec::new());
        RunningState::from_parts(self.slots[slot].tokens.len(), gdn, parts)
    }
    fn restore_running_at(&mut self, slot: usize, s: &RunningState, prefix: &[u32]) -> Result<()> {
        ensure!(prefix.len() == s.pos, "prefix/state length mismatch");
        let parts = s.parts();
        let run: Vec<(usize, usize)> = self.running().collect();
        ensure!(parts.len() >= run.len(), "state has {} parts", parts.len());
        for (&(c, w), part) in run.iter().zip(&parts) {
            ensure!(part.len() == w, "state part size");
            self.sampled
                .prog
                .write_words(&self.rt, self.sampled.cache(c), slot * w, part);
        }
        self.slots[slot].tokens = prefix.to_vec();
        Ok(())
    }
    fn save_positional_at(&mut self, slot: usize, from: usize, to: usize) -> Result<Positional> {
        let p = &self.sampled.prog;
        let bufs = self
            .rows()
            .map(|(c, w, row)| {
                let o = slot * w;
                p.read_words(&self.rt, self.sampled.cache(c), o + from * row..o + to * row)
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect()
            })
            .collect();
        Ok(Positional::from_parts(from, to, bufs))
    }
    fn restore_positional_at(&mut self, slot: usize, pp: &Positional, from: usize) -> Result<()> {
        let rows: Vec<(usize, usize, usize)> = self.rows().collect();
        ensure!(pp.parts().len() == rows.len(), "positional parts");
        let skip = from.saturating_sub(pp.from);
        let start = pp.from.max(from);
        for (&(c, w, row), b) in rows.iter().zip(pp.parts()) {
            let words = words_of_bytes(b);
            self.sampled.prog.write_words(
                &self.rt,
                self.sampled.cache(c),
                slot * w + start * row,
                &words[skip * row..],
            );
        }
        Ok(())
    }
    fn verify_batch(&mut self, items: &[(usize, u32, Vec<u32>)]) -> Result<Vec<Vec<u32>>> {
        items
            .iter()
            .map(|(s, cur, d)| self.verify_slot(*s, *cur, d))
            .collect()
    }
}
