//! Qwen3-Coder-Next served from its derived step (`TANG_CODER_ENGINE=derived`): the
//! [`ServeEngine`] behind `tang-llm serve`'s OpenAI-compatible API, as an opt-in
//! alternative to the hand engine (`flash/coder.rs`).
//!
//! - **The step**: [`prefill_graph`] with [`append_sampler`] on its logits, [`derived_step`]
//!   for `T → T + 1` against `ctx`-row caches (outputs: logits, sampled token), lowered by
//!   [`lower_real_step`] (weights in their GGUF formats where the int8 GEMV reads them), one
//!   CUDA graph a slot.
//! - **Sequence slots**: one [`Program`] per slot. Slot 0 owns the weights; every other slot
//!   binds them with [`Program::with_shared`] and owns only its caches (K/V rows, GDN conv
//!   rings and states), its sampling parameters and its activations. A batch decode runs the
//!   ready slots' steps back to back on the stream and reads their tokens at the end.
//! - **Sampling on the device**: `ir_sample` (temperature, top-k, top-p, seeded by
//!   `(seed, position)`); greedy is a bit-exact first-maximum argmax of the logits. Logits
//!   are read back only when the server asks for them. No presence penalty (ignored, once
//!   noted).
//! - **Prefill** is one decode step a token (no chunked/wide prefill on this path yet).
//! - **Snapshots**: the running state is the GDN rings and states (read off the device),
//!   the positional state the K/V rows of a span — exact restores, no replay.
//! - **Drafts**: verify steps the chain token by token and stops at the first mismatch, so
//!   no rejected token ever reaches the caches (no rewind; one step a kept token).
use std::cell::Cell;
use std::path::Path;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use tang_compile::tensor::step::StepInput;
use tang_compile::tensor::TensorId;
use tang_gpu::program::{Program, Runtime, Shared};
use tang_llm::flash::engine::{Positional, RunningState, WinStats};
use tang_llm::flash::sampler::{RowSampling, Sampler};
use tang_llm::flash::serve::ServeEngine;
use tang_llm::gguf::{GgmlType, Gguf};

use crate::ir_coder_prefill::{prefill_graph, CoderInput};
use crate::ir_coder_real::{bind, dims_from_gguf, lower_real_step};
use crate::ir_sample::{append_sampler, SampleSpec, SEED, TEMPERATURE, TOP_K, TOP_P};
use crate::qwen_session::{runtime, slot_plan};

/// `tang-llm serve` with `TANG_CODER_ENGINE=derived` available: register before serving.
pub fn register() {
    tang_llm::flash::serve::register_derived_coder(load_boxed);
}

fn load_boxed(path: &Path, ctx: usize, slots: usize) -> Result<Box<dyn ServeEngine>> {
    Ok(Box::new(DerivedCoder::load(path, ctx, slots)?))
}

fn words_of_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks(4)
        .map(|c| {
            let mut a = [0u8; 4];
            a[..c.len()].copy_from_slice(c);
            f32::from_le_bytes(a)
        })
        .collect()
}

struct Slot {
    prog: Program,
    tokens: Vec<u32>,
    params: [f32; 4],
    /// Handed out by `alloc_slot` (running or parked) until `free_slot`.
    allocated: bool,
    /// The step without the sampler, sharing `prog`'s weights and caches: greedy slots
    /// step here (the sampler's reductions over the vocabulary cost ~3 ms a token).
    gprog: Option<Program>,
    /// The last step ran on `gprog` (its logits, a host argmax for the token).
    last_greedy: Cell<bool>,
}

/// The greedy step's own inputs and outputs (`[logits, updated caches…]`).
struct GreedyVar {
    pos_in: usize,
    tok_in: usize,
    outs: Vec<TensorId>,
    swaps: Vec<(usize, TensorId)>,
}

pub struct DerivedCoder {
    g: Gguf,
    rt: Runtime,
    slots: Vec<Slot>,
    /// `[logits, token, updated caches…]` in the lowered graph.
    outs: Vec<TensorId>,
    pos_in: usize,
    tok_in: usize,
    /// The sampler's params input (`None`: built without the sampler,
    /// `TANG_CODER_DERIVED_SAMPLER=0`; greedy argmax on the host).
    params_in: Option<usize>,
    /// (cache input, its update) for rings and states, copied back after each step.
    swaps: Vec<(usize, TensorId)>,
    /// Running state: ring and state cache inputs, in cache order.
    running: Vec<usize>,
    /// Positional state: row cache inputs and their words per row.
    rows: Vec<(usize, usize)>,
    zeros: Vec<f32>,
    vocab: usize,
    ctx: usize,
    last: WinStats,
    /// The slot whose logits [`ServeEngine::logits`] reads (the last one stepped alone).
    logits_slot: usize,
    presence_noted: Cell<bool>,
    greedy: Option<GreedyVar>,
}

impl DerivedCoder {
    pub fn load(path: &Path, ctx: usize, nslots: usize) -> Result<Self> {
        let t0 = Instant::now();
        let g = Gguf::open(path)?;
        let dims = dims_from_gguf(&g).map_err(anyhow::Error::msg)?;
        let mut pg = prefill_graph(&dims);
        let mut tys = bind(&pg, &g).map_err(anyhow::Error::msg)?;
        let sampled = std::env::var("TANG_CODER_DERIVED_SAMPLER").as_deref() != Ok("0");
        let mut step_outs = vec![pg.logits];
        if sampled {
            let s = append_sampler(&mut pg.g, pg.logits, SampleSpec::default());
            pg.inputs.push(CoderInput {
                name: "sample.params".into(),
                shape: vec![tang_compile::tensor::Dim::Fixed(4)],
            });
            tys.push(None);
            step_outs.push(s.token);
        }
        let nout = step_outs.len();
        let st = tang_compile::tensor::step::derive_step(
            &pg.g,
            &step_outs,
            &tang_compile::tensor::step::StepSpec {
                seq: crate::ir_coder_prefill::SEQ,
                new: 1,
                capacity: ctx,
            },
        )
        .map_err(|e| anyhow::anyhow!("derive_step: {}", e.0))?;
        let low = lower_real_step(&pg, &st, &tys);
        eprintln!(
            "coder derived: step of {} kernels ({} int8 GEMVs), {} caches, built in {:.1} s",
            low.program.program.kernels.len(),
            low.sched.gemvs,
            st.caches.len(),
            t0.elapsed().as_secs_f64()
        );
        let rt = runtime();
        ensure!(
            matches!(rt, Runtime::Cuda(_)),
            "TANG_CODER_ENGINE=derived needs CUDA (got {})",
            rt.name()
        );
        let plan = slot_plan(&low.program.program, &low.outs, &low.inplace);
        let empty: Vec<&[f32]> = vec![&[]; st.inputs.len()];
        let wprog = low.program.program;
        let storage = low.storage;
        let outs = low.outs;
        let params_k = sampled.then(|| pg.input("sample.params"));
        let input_k = |want: StepInput| st.inputs.iter().position(|s| *s == want);
        let params_in = match params_k {
            Some(k) => Some(input_k(StepInput::Same(k)).context("sampler params input")?),
            None => None,
        };
        let pos_in = input_k(StepInput::Pos).context("pos input")?;
        let tok_in = st
            .inputs
            .iter()
            .position(|s| matches!(s, StepInput::New(_)))
            .context("token input")?;

        // slot 0: weights streamed in from the mapping, one tensor at a time
        let first = Program::with_plan(&rt, wprog.clone(), &empty, &plan);
        let t1 = Instant::now();
        let mut bytes = 0usize;
        let mut weights = Vec::new();
        for (k, si) in st.inputs.iter().enumerate() {
            let StepInput::Same(i) = *si else { continue };
            if Some(i) == params_k {
                continue;
            }
            let t = g.info(&pg.inputs[i].name)?;
            let words = match (storage[k], t.ty) {
                (Some(_), _) | (None, GgmlType::F32) => words_of_bytes(g.bytes(t)),
                (None, _) => g.dequantize(t)?,
            };
            bytes += words.len() * 4;
            first.write(&rt, first.input(k), &words);
            weights.push(k);
        }
        eprintln!(
            "coder derived: {:.1} GB of weights up in {:.1} s",
            bytes as f64 / 1e9,
            t1.elapsed().as_secs_f64()
        );
        let mut progs = vec![first];
        for _ in 1..nslots.max(1) {
            let shared: Vec<Shared<'_>> = weights
                .iter()
                .map(|&k| Shared {
                    input: k,
                    from: &progs[0],
                    tensor: progs[0].input(k),
                })
                .collect();
            let p = Program::with_shared(&rt, wprog.clone(), &empty, &plan, &shared);
            drop(shared);
            progs.push(p);
        }
        // the greedy variant: the same graph's logits alone, its weights and caches bound to
        // each slot's sampled program (TANG_CODER_DERIVED_GREEDY_FAST=0: off)
        let fast = sampled && std::env::var("TANG_CODER_DERIVED_GREEDY_FAST").as_deref() != Ok("0");
        let mut gprogs: Vec<Option<Program>> = (0..progs.len()).map(|_| None).collect();
        let mut greedy_var = None;
        if fast {
            let st2 = tang_compile::tensor::step::derive_step(
                &pg.g,
                &[pg.logits],
                &tang_compile::tensor::step::StepSpec {
                    seq: crate::ir_coder_prefill::SEQ,
                    new: 1,
                    capacity: ctx,
                },
            )
            .map_err(|e| anyhow::anyhow!("derive_step (greedy): {}", e.0))?;
            let low2 = lower_real_step(&pg, &st2, &tys);
            let plan2 = slot_plan(&low2.program.program, &low2.outs, &low2.inplace);
            let empty2: Vec<&[f32]> = vec![&[]; st2.inputs.len()];
            // step-2 input -> step-1 input: same weights by GGUF input, same caches by node
            let mut map = Vec::new();
            for (k2, si) in st2.inputs.iter().enumerate() {
                let k1 = match *si {
                    StepInput::Same(i) if Some(i) != params_k => input_k(StepInput::Same(i)),
                    StepInput::Cache(c2) => st
                        .caches
                        .iter()
                        .position(|c| c.node == st2.caches[c2].node)
                        .and_then(|c1| input_k(StepInput::Cache(c1))),
                    _ => None,
                };
                if let Some(k1) = k1 {
                    map.push((k2, k1));
                }
            }
            for (s, p1) in progs.iter().enumerate() {
                let shared: Vec<Shared<'_>> = map
                    .iter()
                    .map(|&(k2, k1)| Shared {
                        input: k2,
                        from: p1,
                        tensor: p1.input(k1),
                    })
                    .collect();
                gprogs[s] = Some(Program::with_shared(
                    &rt,
                    low2.program.program.clone(),
                    &empty2,
                    &plan2,
                    &shared,
                ));
            }
            let in2 = |want: StepInput| st2.inputs.iter().position(|s| *s == want);
            let swaps2 = st2
                .caches
                .iter()
                .enumerate()
                .filter(|(_, c)| c.state || c.ring.is_some())
                .map(|(c, _)| (in2(StepInput::Cache(c)).unwrap(), low2.outs[1 + c]))
                .collect();
            greedy_var = Some(GreedyVar {
                pos_in: in2(StepInput::Pos).context("greedy pos input")?,
                tok_in: st2
                    .inputs
                    .iter()
                    .position(|s| matches!(s, StepInput::New(_)))
                    .context("greedy token input")?,
                outs: low2.outs,
                swaps: swaps2,
            });
        }
        let capture = std::env::var("TANG_CODER_DERIVED_GRAPH").as_deref() != Ok("0");
        if capture {
            for p in progs.iter_mut().chain(gprogs.iter_mut().flatten()) {
                p.capture(&rt);
            }
        }
        let cache_in = |c: usize| {
            st.inputs
                .iter()
                .position(|s| *s == StepInput::Cache(c))
                .expect("cache input")
        };
        let ncache = st.caches.len();
        let mut swaps = Vec::new();
        let mut running = Vec::new();
        let mut rows = Vec::new();
        for (c, cs) in st.caches.iter().enumerate() {
            let k = cache_in(c);
            if cs.state || cs.ring.is_some() {
                swaps.push((k, outs[nout + c]));
                running.push(k);
            } else {
                ensure!(cs.axis == 0, "row cache {c} on axis {}", cs.axis);
                let w = progs[0].words_of(progs[0].input(k));
                ensure!(w % ctx == 0, "row cache {c}: {w} words over {ctx} rows");
                rows.push((k, w / ctx));
            }
        }
        ensure!(running.len() % 2 == 0, "odd running-state caches");
        let zmax = running
            .iter()
            .map(|&k| progs[0].words_of(progs[0].input(k)))
            .max()
            .unwrap_or(0);
        let greedy = [0.0, 1.0, 0.0, 0.0];
        let slots: Vec<Slot> = progs
            .into_iter()
            .zip(gprogs)
            .map(|(prog, gprog)| Slot {
                prog,
                tokens: Vec::new(),
                params: greedy,
                allocated: false,
                gprog,
                last_greedy: Cell::new(false),
            })
            .collect();
        let mut me = DerivedCoder {
            g,
            rt,
            slots,
            outs,
            pos_in,
            tok_in,
            params_in,
            swaps,
            running,
            rows,
            zeros: vec![0.0; zmax],
            vocab: dims.vocab,
            ctx,
            last: WinStats::default(),
            logits_slot: 0,
            presence_noted: Cell::new(false),
            greedy: greedy_var,
        };
        for s in 0..me.slots.len() {
            me.write_params(s);
            me.clear_slot(s);
        }
        // where a token's time goes: the step graph on the device, the cache copies, and
        // the host side of a step (two input writes, one token read)
        {
            let p = &me.slots[0].prog;
            let dev = p.time(&me.rt, 20);
            let t = Instant::now();
            for _ in 0..20 {
                for &(k, upd) in &me.swaps {
                    p.copy(&me.rt, p.input(k), p, upd);
                }
            }
            let _ = me.token(0);
            let copies = t.elapsed().as_secs_f64() * 1e3 / 20.0;
            let t = Instant::now();
            for i in 0..20 {
                me.enqueue(0, 0, i)?;
                let _ = me.token(0);
            }
            let wall = t.elapsed().as_secs_f64() * 1e3 / 20.0;
            me.zero_rows(0, 0, 20);
            me.clear_slot(0);
            if let Some(gp) = &me.slots[0].gprog {
                eprintln!("coder derived: greedy step {:.3} ms device (no sampler)", gp.time(&me.rt, 20));
            }
            eprintln!(
                "coder derived: step {dev:.3} ms device, {} cache copies {copies:.3} ms, \
                 enqueue + token read {wall:.3} ms wall a token",
                me.swaps.len()
            );
        }
        let kv_mb: usize = me.rows.iter().map(|&(_, w)| w * ctx * 4).sum::<usize>() >> 20;
        eprintln!(
            "coder derived: {} slot(s), ctx {ctx}, {ncache} caches ({kv_mb} MB of K/V a slot), graphs {}, sampler {}, ready in {:.1} s",
            me.slots.len(),
            if capture { "on" } else { "off" },
            if me.params_in.is_some() { "on the device" } else { "off (host argmax)" },
            t0.elapsed().as_secs_f64()
        );
        Ok(me)
    }

    fn write_params(&self, s: usize) {
        let Some(k) = self.params_in else { return };
        let sl = &self.slots[s];
        sl.prog.write(&self.rt, sl.prog.input(k), &sl.params);
    }

    /// Zero a slot's running state (rings and states) and the K/V rows its sequence wrote:
    /// a reused slot must look like a fresh one (stale rows past the position changed
    /// greedy output on the H200).
    fn clear_slot(&mut self, s: usize) {
        let p = &self.slots[s].prog;
        for &k in &self.running {
            let t = p.input(k);
            p.write(&self.rt, t, &self.zeros[..p.words_of(t)]);
        }
        let n = self.slots[s].tokens.len();
        self.zero_rows(s, 0, n);
        self.slots[s].tokens.clear();
    }

    /// Zero rows `from..to` of slot `s`'s K/V caches.
    fn zero_rows(&self, s: usize, from: usize, to: usize) {
        if to <= from {
            return;
        }
        let p = &self.slots[s].prog;
        for &(k, w) in &self.rows {
            let z = vec![0.0f32; (to - from) * w];
            p.write_words(&self.rt, p.input(k), from * w, &z);
        }
    }

    /// Queue one step of slot `s`: token `tok` at position `pos`.
    fn enqueue(&self, s: usize, tok: u32, pos: usize) -> Result<()> {
        ensure!(pos < self.ctx, "slot {s} is past its context ({} tokens)", self.ctx);
        let sl = &self.slots[s];
        if let (Some(gp), Some(v)) = (&sl.gprog, &self.greedy) {
            if sl.params[TEMPERATURE] <= 0.0 {
                gp.write(&self.rt, gp.input(v.pos_in), &[pos as f32]);
                gp.write(&self.rt, gp.input(v.tok_in), &[tok as f32]);
                gp.run(&self.rt);
                for &(k, upd) in &v.swaps {
                    gp.copy(&self.rt, gp.input(k), gp, upd);
                }
                sl.last_greedy.set(true);
                return Ok(());
            }
        }
        sl.last_greedy.set(false);
        let p = &sl.prog;
        p.write(&self.rt, p.input(self.pos_in), &[pos as f32]);
        p.write(&self.rt, p.input(self.tok_in), &[tok as f32]);
        p.run(&self.rt);
        for &(k, upd) in &self.swaps {
            p.copy(&self.rt, p.input(k), p, upd);
        }
        Ok(())
    }

    /// Slot `s`'s sampled (or greedy) token from its last step.
    fn token(&self, s: usize) -> u32 {
        let l = self.last_logits(s);
        let sl = &self.slots[s];
        if self.params_in.is_some() && !sl.last_greedy.get() {
            return sl.prog.read(&self.rt, self.outs[1])[0] as u32;
        }
        // no sampler: the first maximum of the logits
        let l = &l[l.len() - self.vocab..];
        let mut a = 0;
        for (i, &x) in l.iter().enumerate() {
            if x > l[a] {
                a = i;
            }
        }
        a as u32
    }

    /// Step `ids` through slot `s` from its current end (appending them), returning the
    /// token after the last.
    fn feed(&mut self, s: usize, ids: &[u32]) -> Result<u32> {
        let t0 = Instant::now();
        for &id in ids {
            let pos = self.slots[s].tokens.len();
            self.enqueue(s, id, pos)?;
            self.slots[s].tokens.push(id);
        }
        let next = self.token(s);
        self.logits_slot = s;
        self.last = WinStats {
            t: ids.len(),
            wall_ms: t0.elapsed().as_secs_f64() * 1e3,
            ..WinStats::default()
        };
        Ok(next)
    }

    fn verify_slot(&mut self, s: usize, cur: u32, drafts: &[u32]) -> Result<Vec<u32>> {
        let mut kept = vec![self.feed(s, &[cur])?];
        for &d in drafts {
            if kept[kept.len() - 1] != d {
                break;
            }
            kept.push(self.feed(s, &[d])?);
        }
        Ok(kept)
    }

    /// Slot `s`'s last step's logits (whichever program ran it). Read lazily: only greedy
    /// tokens and `logits()` need them.
    fn last_logits(&self, s: usize) -> Vec<f32> {
        let sl = &self.slots[s];
        match (&sl.gprog, &self.greedy) {
            (Some(gp), Some(v)) if sl.last_greedy.get() => gp.read(&self.rt, v.outs[0]),
            _ if self.params_in.is_some() => Vec::new(),
            _ => sl.prog.read(&self.rt, self.outs[0]),
        }
    }

    fn gdn_layers(&self) -> usize {
        self.running.len() / 2
    }
}

impl ServeEngine for DerivedCoder {
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
    /// The caller has extended `tokens` through the window: step `tokens[pos0..pos0 + t]`.
    fn window(&mut self, pos0: usize, t: usize) -> Result<Vec<u32>> {
        let ids = self.slots[0].tokens[pos0..pos0 + t].to_vec();
        let t0 = Instant::now();
        let mut out = Vec::with_capacity(t);
        for (i, &id) in ids.iter().enumerate() {
            self.enqueue(0, id, pos0 + i)?;
            out.push(self.token(0));
        }
        self.logits_slot = 0;
        self.last = WinStats {
            t,
            wall_ms: t0.elapsed().as_secs_f64() * 1e3,
            ..WinStats::default()
        };
        Ok(out)
    }
    fn verify(&mut self, cur: u32, drafts: &[u32]) -> Result<Vec<u32>> {
        self.verify_slot(0, cur, drafts)
    }
    /// The last step's logits (one row: the step is one token wide), `t` rows asked for.
    fn logits(&self, t: usize) -> Vec<f32> {
        let sl = &self.slots[self.logits_slot];
        let l = match (&sl.gprog, &self.greedy) {
            (Some(gp), Some(g)) if sl.last_greedy.get() => gp.read(&self.rt, g.outs[0]),
            _ => sl.prog.read(&self.rt, self.outs[0]),
        };
        let mut v = Vec::with_capacity(t * self.vocab);
        for _ in 0..t {
            v.extend_from_slice(&l[l.len() - self.vocab..]);
        }
        v
    }
    fn set_sampling(&mut self, temperature: f32, seed: u32) {
        let mut p = [0.0, 1.0, 0.0, 0.0];
        p[TEMPERATURE] = temperature.max(0.0);
        p[TOP_P] = 1.0;
        p[TOP_K] = 0.0;
        p[SEED] = (seed & 0xff_ffff) as f32;
        self.slots[0].params = p;
        self.write_params(0);
    }
    fn sampler_mut(&mut self) -> Option<&mut Sampler> {
        None
    }
    fn sampler_begin(&mut self, _pos: usize) -> Result<()> {
        Ok(())
    }
    fn max_slots(&self) -> usize {
        self.slots.len()
    }
    fn fit_slots(&mut self) -> usize {
        self.slots.len()
    }
    fn slot_tokens(&self, slot: usize) -> &[u32] {
        &self.slots[slot].tokens
    }
    /// A slot no request holds (a just-allocated slot is still empty until its prefill, so
    /// emptiness can't mark it free); `Err` when all are held, and the batcher evicts a
    /// parked one with `free_slot`.
    fn alloc_slot(&mut self) -> Result<usize> {
        let s = (0..self.slots.len())
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
    /// Every row's step queued back to back, then their tokens read.
    fn decode_batch(&mut self, rows: &[(usize, u32)]) -> Result<Vec<u32>> {
        let t0 = Instant::now();
        for &(s, tok) in rows {
            let pos = self.slots[s].tokens.len();
            self.enqueue(s, tok, pos)?;
            self.slots[s].tokens.push(tok);
        }
        let out = rows.iter().map(|&(s, _)| self.token(s)).collect();
        self.last = WinStats {
            t: rows.len(),
            wall_ms: t0.elapsed().as_secs_f64() * 1e3,
            ..WinStats::default()
        };
        Ok(out)
    }
    fn set_slot_sampling(&mut self, slot: usize, s: RowSampling) -> Result<()> {
        ensure!(
            self.params_in.is_some() || !s.sampled(),
            "TANG_CODER_DERIVED_SAMPLER=0: greedy only (temperature {})",
            s.temperature
        );
        if s.presence != 0.0 && !self.presence_noted.replace(true) {
            eprintln!("coder derived: presence penalty not supported on the device sampler; ignored");
        }
        let mut p = [0.0f32; 4];
        p[TEMPERATURE] = s.temperature.max(0.0);
        p[TOP_P] = if s.top_p > 0.0 && s.top_p < 1.0 { s.top_p } else { 1.0 };
        p[TOP_K] = s.top_k as f32;
        p[SEED] = (s.seed & 0xff_ffff) as f32;
        self.slots[slot].params = p;
        self.write_params(slot);
        Ok(())
    }
    fn save_running_at(&mut self, slot: usize) -> Result<RunningState> {
        let p = &self.slots[slot].prog;
        let mut parts: Vec<Vec<f32>> = self
            .running
            .iter()
            .map(|&k| {
                let t = p.input(k);
                p.read_words(&self.rt, t, 0..p.words_of(t))
            })
            .collect();
        parts.push(Vec::new()); // no QSA rings
        parts.push(Vec::new()); // no PLE ring
        RunningState::from_parts(self.slots[slot].tokens.len(), self.gdn_layers(), parts)
    }
    fn restore_running_at(&mut self, slot: usize, s: &RunningState, prefix: &[u32]) -> Result<()> {
        ensure!(prefix.len() == s.pos, "prefix/state length mismatch");
        let parts = s.parts();
        ensure!(parts.len() >= self.running.len(), "state has {} parts", parts.len());
        let p = &self.slots[slot].prog;
        for (&k, part) in self.running.iter().zip(&parts) {
            let t = p.input(k);
            ensure!(part.len() == p.words_of(t), "state part size");
            p.write_words(&self.rt, t, 0, part);
        }
        let old = self.slots[slot].tokens.len();
        self.zero_rows(slot, prefix.len(), old);
        self.slots[slot].tokens = prefix.to_vec();
        Ok(())
    }
    fn save_positional_at(&mut self, slot: usize, from: usize, to: usize) -> Result<Positional> {
        let p = &self.slots[slot].prog;
        let bufs = self
            .rows
            .iter()
            .map(|&(k, w)| {
                p.read_words(&self.rt, p.input(k), from * w..to * w)
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect()
            })
            .collect();
        Ok(Positional::from_parts(from, to, bufs))
    }
    fn restore_positional_at(&mut self, slot: usize, pp: &Positional, from: usize) -> Result<()> {
        ensure!(pp.parts().len() == self.rows.len(), "positional parts");
        let p = &self.slots[slot].prog;
        let skip = from.saturating_sub(pp.from);
        for (&(k, w), b) in self.rows.iter().zip(pp.parts()) {
            let words = words_of_bytes(b);
            let start = pp.from.max(from);
            p.write_words(&self.rt, p.input(k), start * w, &words[skip * w..]);
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
