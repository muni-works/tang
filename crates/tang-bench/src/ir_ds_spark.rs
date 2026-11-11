//! DSpark, DeepSeek-V4.1-Flash's shipped drafter, as a graph over the main
//! model's positions, and draft → verify through derived steps.
//!
//! **The drafter** (`forward.rs` `forward_spec_with`, the checkpoint's
//! `n_mtp` MTP layers): after main position `t` with target hiddens
//! `mains[t]` (the mean over the `hc` streams entering each
//! `dspark_targets` layer, the main graph's `mains` output) and the token
//! that followed it, `next[t]`, a block of `dspark_block` slots at
//! positions `t + 1 …` — `next[t]`, then the noise token — runs through the
//! stages. A stage is a backbone block (mHC, attention, MoE with
//! `dspark_experts` top `dspark_active`) whose attention keys are the
//! stage's projection of `main_proj(mains)` over the window of main rows
//! `(t − window, t]` plus the block's own slots (all of them: the block is
//! not causal). The last stage's head logits plus a rank-`dspark_rank`
//! Markov bias on the previous slot's token give each slot's greedy draft,
//! in order. Written over `T` main rows, the graph's draft at row `t` is
//! the hand engine's draft after `t`; its derived step appends the main
//! rows a verify accepted and drafts from the last one ([`Rows::Last`]).
//! The block is `dspark_block` copies of one slot's ops sharing weights,
//! coupled only in attention.
//!
//! **Speculation.** The main model's derived step over a window `[last,
//! d₁ … d_{k−1}]` ([`Rows::All`]) gives every row's argmax and target
//! hiddens; the accepted prefix is prompt lookup's rule
//! ([`crate::prompt_lookup::Verifier`], rollback by rerun when a ring cache
//! advanced past the kept rows). The drafter's caches only ever see accepted
//! rows, so it never rolls back. The output is the greedy output.
use std::collections::HashMap;

use tang::Scalar;

use tang_compile::tensor::step::{
    derive_step_with, seed_cache, CacheSlot, Rows, StepGraph, StepInput, StepOptions, StepSpec,
};
use tang_compile::tensor::{Arg, Monoid, TensorGraph, TensorId};
use tang_expr::ExprId;

use crate::ir_ds_prefill::{
    arg, form, fx, ident, konst, lit, moe_graph, select, unit, DsDims, EngramMeta, PrefillGraph,
    PrefillInput, B, NEG, SEQ, T,
};
use crate::prompt_lookup::{argmax, Stats, Verifier};

pub struct SparkGraph {
    pub g: TensorGraph,
    pub inputs: Vec<PrefillInput>,
    /// `[T, block]`: the greedy drafts after row t (slot i drafts position t + 2 + i).
    pub ids: TensorId,
    /// `[T, block · vocab]`: each slot's logits, Markov bias included.
    pub logits: TensorId,
    /// `[T, block]`: the confidence head.
    pub conf: TensorId,
    pub named: Vec<(String, TensorId)>,
}

/// One stage's attention for every slot: queries and keys of slot i at
/// position `t + 1 + i`, the main rows' keys at `t`.
fn spark_attention(
    b: &mut B,
    p: &str,
    dm: &DsDims,
    main_x: TensorId,
    xs: &[TensorId],
) -> Vec<TensorId> {
    let (d, h, hd) = (dm.dim, dm.n_heads, dm.head_dim);
    let (ql, g, ol) = (dm.q_lora_rank, dm.o_groups, dm.o_lora_rank);
    let win = dm.window as f64;
    let fr = b.w(format!("{p}.freqs"), &[hd / 2]);
    let wq_a = b.w(format!("{p}.attn.wq_a.weight"), &[ql, d]);
    let q_norm = b.w(format!("{p}.attn.q_norm.weight"), &[ql]);
    let wq_b = b.w(format!("{p}.attn.wq_b.weight"), &[h * hd, ql]);
    let wkv = b.w(format!("{p}.attn.wkv.weight"), &[hd, d]);
    let kv_norm = b.w(format!("{p}.attn.kv_norm.weight"), &[hd]);
    let sink = b.w(format!("{p}.attn.attn_sink"), &[h]);
    let wo_a = b.w(format!("{p}.attn.wo_a.weight"), &[g * ol, h * hd / g]);
    let wo_b = b.w(format!("{p}.attn.wo_b.weight"), &[d, g * ol]);
    let mk0 = b.linear(main_x, wkv);
    let mk1 = b.rms(mk0, kv_norm);
    let mkv = b.rope(mk1, 1, hd, fr, 0, false);
    b.name(format!("{p}.main_kv"), mkv);
    let blk = xs.len();
    let mut qs = Vec::new();
    let mut kvs = Vec::new();
    for (i, &x) in xs.iter().enumerate() {
        let qr0 = b.linear(x, wq_a);
        let qr = b.rms(qr0, q_norm);
        let q0 = b.linear(qr, wq_b);
        qs.push(b.rope(q0, h, hd, fr, 1 + i as i64, false));
        let k0 = b.linear(x, wkv);
        let k1 = b.rms(k0, kv_norm);
        kvs.push(b.rope(k1, 1, hd, fr, 1 + i as i64, false));
    }
    let scale = (hd as f64).powf(-0.5);
    let in_win = move |t: ExprId, j: ExprId, _: &[ExprId]| j - t + lit(win - 0.5);
    let hsp = [T, fx(h)];
    let osp = [T, fx(h), fx(hd)];
    let qh = |q: TensorId, r: usize| {
        arg(
            q,
            vec![unit(r, 0), form(r, &[(1, hd as i64), (r - 1, 1)], 0)],
        )
    };
    let mut outs = Vec::new();
    for i in 0..blk {
        let q = qs[i];
        // scores vs the main rows [T, H, T]
        let sw =
            b.g.comp(
                &[T, fx(h), T],
                &[fx(hd)],
                vec![qh(q, 4), arg(mkv, vec![unit(4, 2), unit(4, 3)])],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1] * lit(scale)],
            )
            .unwrap();
        // scores vs every block slot [T, H]
        let sb: Vec<TensorId> = kvs
            .iter()
            .map(|&kv| {
                b.g.comp(
                    &hsp,
                    &[fx(hd)],
                    vec![qh(q, 3), arg(kv, vec![unit(3, 0), unit(3, 2)])],
                    Some(Monoid::sum()),
                    |v| vec![v[0] * v[1] * lit(scale)],
                )
                .unwrap()
            })
            .collect();
        let s3 = |src| arg(src, ident(3));
        let mw = b.attn_reduce(&hsp, 2, vec![s3(sw)], Monoid::max(), NEG, in_win, |v| v[0]);
        let mut margs: Vec<Arg> = vec![arg(mw, ident(2))];
        margs.extend(sb.iter().map(|&s| arg(s, ident(2))));
        margs.push(arg(sink, vec![unit(2, 1)]));
        let mm =
            b.g.comp(&hsp, &[], margs, None, |v| {
                vec![v.iter().skip(1).fold(v[0], |a, &x| a.max(x))]
            })
            .unwrap();
        let m3 = || arg(mm, vec![unit(3, 0), unit(3, 1)]);
        let den = b.attn_reduce(
            &hsp,
            2,
            vec![s3(sw), m3()],
            Monoid::sum(),
            0.0,
            in_win,
            |v| (v[0] - v[1]).exp(),
        );
        let num = b.attn_reduce(
            &osp,
            3,
            vec![
                arg(sw, vec![unit(4, 0), unit(4, 1), unit(4, 3)]),
                arg(mm, vec![unit(4, 0), unit(4, 1)]),
                arg(mkv, vec![unit(4, 3), unit(4, 2)]),
            ],
            Monoid::sum(),
            0.0,
            in_win,
            |v| (v[0] - v[1]).exp() * v[2],
        );
        // o[t, h, c]: the window part plus the slots, over the denominator
        let mut oargs = vec![
            arg(num, ident(3)),
            arg(den, vec![unit(3, 0), unit(3, 1)]),
            arg(mm, vec![unit(3, 0), unit(3, 1)]),
            arg(sink, vec![unit(3, 1)]),
        ];
        for (&s, &kv) in sb.iter().zip(&kvs) {
            oargs.push(arg(s, vec![unit(3, 0), unit(3, 1)]));
            oargs.push(arg(kv, vec![unit(3, 0), unit(3, 2)]));
        }
        let o =
            b.g.comp(&osp, &[], oargs, None, move |v| {
                let (m, sk) = (v[2], v[3]);
                let mut num = v[0];
                let mut den = v[1] + (sk - m).exp();
                for s in 0..blk {
                    let e = (v[4 + 2 * s] - m).exp();
                    num = num + e * v[5 + 2 * s];
                    den = den + e;
                }
                vec![num / den]
            })
            .unwrap();
        let o = b.g.reshape(o, &[T, fx(h * hd)]).unwrap();
        let o = b.rope(o, h, hd, fr, 1 + i as i64, true);
        let gd = h * hd / g;
        let mid =
            b.g.comp(
                &[T, fx(g), fx(ol)],
                &[fx(gd)],
                vec![
                    arg(o, vec![unit(4, 0), form(4, &[(1, gd as i64), (3, 1)], 0)]),
                    arg(
                        wo_a,
                        vec![form(4, &[(1, ol as i64), (2, 1)], 0), unit(4, 3)],
                    ),
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        let mid = b.g.reshape(mid, &[T, fx(g * ol)]).unwrap();
        outs.push(b.linear(mid, wo_b));
    }
    outs
}

/// Stack `[T]`-or-`[T, n]` per-slot nodes into `[T, slots · n]`.
fn stack(b: &mut B, parts: &[TensorId], n: usize) -> TensorId {
    let s = parts.len();
    let sp = [T, fx(s), fx(n)];
    let si = b.g.iota(&sp, 1);
    let mut args: Vec<Arg> = parts
        .iter()
        .map(|&p| {
            if b.g.shape(p).len() == 1 {
                arg(p, vec![unit(3, 0)])
            } else {
                arg(p, vec![unit(3, 0), unit(3, 2)])
            }
        })
        .collect();
    args.push(arg(si, ident(3)));
    let m =
        b.g.comp(&sp, &[], args, None, move |v| {
            let k = v[s];
            vec![(1..s).fold(v[0], |acc, i| select(k - lit(i as f64 - 0.5), v[i], acc))]
        })
        .unwrap();
    b.g.reshape(m, &[T, fx(s * n)]).unwrap()
}

/// The drafter over `T` main rows.
pub fn spark_graph(dm: &DsDims) -> SparkGraph {
    let (d, hc, blk, v) = (dm.dim, dm.hc, dm.dspark_block, dm.vocab);
    let nt = dm.dspark_targets.len();
    let mut b = B {
        g: TensorGraph::new(),
        inputs: Vec::new(),
        named: Vec::new(),
        eps: dm.eps,
        ws: Default::default(),
    };
    let mains = b.input("mains".into(), &[T, fx(nt * d)]);
    let next = b.input("next".into(), &[T]);
    let emb = b.w("embed.weight".into(), &[v, d]);
    let mp = b.w("mtp.0.main_proj.weight".into(), &[d, nt * d]);
    let mn = b.w("mtp.0.main_norm.weight".into(), &[d]);
    let mx0 = b.linear(mains, mp);
    let main_x = b.rms(mx0, mn);
    b.name("main_x".into(), main_x);
    // slot 0 embeds next[t], the others the noise token; every stream alike
    let e0 = b.g.gather(emb, 0, next, false).unwrap();
    let noise = dm.dspark_noise as i64;
    let ci = b.g.iota(&[T, fx(hc)], 1);
    let one_hot =
        b.g.comp(&[T, fx(hc)], &[], vec![arg(ci, ident(2))], None, |v| {
            vec![select(lit(0.5) - v[0], lit(1.0), lit(0.0))]
        })
        .unwrap();
    let mut hs: Vec<TensorId> = Vec::new();
    let mut pres: Vec<TensorId> = Vec::new();
    for i in 0..blk {
        let a = if i == 0 {
            arg(e0, vec![unit(3, 0), unit(3, 2)])
        } else {
            arg(emb, vec![konst(3, noise), unit(3, 2)])
        };
        let x3 =
            b.g.comp(&[T, fx(hc), fx(d)], &[], vec![a], None, |v| vec![v[0]])
                .unwrap();
        hs.push(b.g.reshape(x3, &[T, fx(hc * d)]).unwrap());
        pres.push(one_hot);
    }
    for s in 0..dm.n_mtp {
        let p = format!("mtp.{s}");
        let an = b.w(format!("{p}.attn_norm.weight"), &[d]);
        let mut amix = Vec::new();
        let mut xs = Vec::new();
        for i in 0..blk {
            let m = b.hc_mixes(&p, "attn", hs[i], dm);
            let hx = b.hc_pre(hs[i], pres[i], hc, d);
            xs.push(b.rms(hx, an));
            amix.push(m);
        }
        let att = spark_attention(&mut b, &p, dm, main_x, &xs);
        let fnw = b.w(format!("{p}.ffn_norm.weight"), &[d]);
        for i in 0..blk {
            let [apre, apost, acomb] = amix[i];
            let x = b.hc_post(att[i], hs[i], apost, acomb, hc, d);
            let [fpre, fpost, fcomb] = b.hc_mixes(&p, "ffn", x, dm);
            let hx = b.hc_pre(x, apre, hc, d);
            let xn = b.rms(hx, fnw);
            let y = moe_graph(
                &mut b,
                &p,
                xn,
                dm,
                dm.dspark_experts,
                dm.dspark_active,
                dm.expert_rows,
            );
            hs[i] = b.hc_post(y, x, fpost, fcomb, hc, d);
            pres[i] = fpre;
        }
    }
    let last = format!("mtp.{}", dm.n_mtp - 1);
    let nw = b.w(format!("{last}.norm.weight"), &[d]);
    let head = b.w("head.weight".into(), &[v, d]);
    let rk = dm.dspark_rank;
    let memb_w = b.w(format!("{last}.markov_head.embed.weight"), &[v, rk]);
    let mhead = b.w(format!("{last}.markov_head.head.weight"), &[v, rk]);
    let proj = b.w(format!("{last}.confidence_head.proj.weight"), &[1, d + rk]);
    let vi = b.g.iota(&[T, fx(v)], 1);
    let mut idx = next;
    let (mut ids, mut lgs, mut confs) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..blk {
        let x = b.hc_pre(hs[i], pres[i], hc, d);
        let xn = b.rms(x, nw);
        let lg = b.linear(xn, head);
        let memb = b.g.gather(memb_w, 0, idx, false).unwrap();
        let bias = b.linear(memb, mhead);
        let ll = b.g.map(&[lg, bias], |v| v[0] + v[1]).unwrap();
        let am =
            b.g.comp_tuple(
                &[T, fx(2)],
                &[fx(v)],
                vec![arg(ll, ident(2)), arg(vi, ident(2))],
                Some(Monoid::argmax()),
                |v| vec![v[1], v[0]],
            )
            .unwrap();
        idx =
            b.g.comp(
                &[T],
                &[],
                vec![arg(am, vec![unit(1, 0), konst(1, 0)])],
                None,
                |v| vec![v[0]],
            )
            .unwrap();
        // the confidence head on [x, memb]
        let cx =
            b.g.comp(
                &[T],
                &[fx(d)],
                vec![arg(x, ident(2)), arg(proj, vec![konst(2, 0), unit(2, 1)])],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        let cm =
            b.g.comp(
                &[T],
                &[fx(rk)],
                vec![
                    arg(memb, ident(2)),
                    arg(proj, vec![konst(2, 0), form(2, &[(1, 1)], d as i64)]),
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        confs.push(b.g.map(&[cx, cm], |v| v[0] + v[1]).unwrap());
        ids.push(idx);
        lgs.push(ll);
    }
    let ids = stack(&mut b, &ids, 1);
    let logits = stack(&mut b, &lgs, v);
    let conf = stack(&mut b, &confs, 1);
    SparkGraph {
        g: b.g,
        inputs: b.inputs,
        ids,
        logits,
        conf,
        named: b.named,
    }
}

/// The drafter's step appending `k` accepted main rows, outputs (ids,
/// logits, conf) for the last row only.
pub fn spark_step(sg: &SparkGraph, dm: &DsDims, k: usize) -> Result<StepGraph, String> {
    derive_step_with(
        &sg.g,
        &[sg.ids, sg.logits, sg.conf],
        &StepSpec {
            seq: SEQ,
            new: k,
            capacity: dm.ctx,
        },
        &StepOptions {
            rows: Rows::Last,
            batch: None,
        },
    )
    .map_err(|e| e.0)
}

/// The main model's verify window: every row's logits and target hiddens.
pub fn verify_step(pg: &PrefillGraph, dm: &DsDims, k: usize) -> Result<StepGraph, String> {
    derive_step_with(
        &pg.g,
        &[pg.logits, pg.mains.expect("a graph with DSpark targets")],
        &StepSpec {
            seq: SEQ,
            new: k,
            capacity: dm.ctx,
        },
        &StepOptions {
            rows: Rows::All,
            batch: None,
        },
    )
    .map_err(|e| e.0)
}

fn concrete(g: &TensorGraph, t: TensorId, rows: usize) -> Vec<usize> {
    g.shape(t)
        .iter()
        .map(|&d| if d == T { rows } else { d.n() })
        .collect()
}

type Key = (TensorId, bool);

fn key(c: &CacheSlot) -> Key {
    (c.node, c.state)
}

fn seed_all(
    g: &TensorGraph,
    slots: &[CacheSlot],
    vals: &[Vec<f64>],
    len: usize,
    cap: usize,
) -> HashMap<Key, Vec<f64>> {
    slots
        .iter()
        .zip(vals)
        .map(|(c, v)| {
            assert!(!c.state, "a state cache: seed it as derive_all::seed does");
            (
                key(c),
                seed_cache(c, &concrete(g, c.node, len), v, len, cap),
            )
        })
        .collect()
}

/// The main model's derived window in the interpreter (prompt lookup's
/// `CpuVerifier` with the target hiddens as a second output and the
/// Engram hashes as per-token inputs).
pub struct DsVerifier<'a> {
    pg: &'a PrefillGraph,
    dm: &'a DsDims,
    w: &'a HashMap<String, Vec<f64>>,
    meta: Option<&'a EngramMeta>,
    steps: HashMap<usize, StepGraph>,
    caches: HashMap<Key, Vec<f64>>,
    /// Every token fed so far.
    pub seq: Vec<usize>,
    #[allow(clippy::type_complexity)]
    pending: Option<(Vec<u32>, HashMap<Key, Vec<f64>>, Vec<f64>)>,
    /// The last window's target hiddens, `[k, n · dim]`.
    pub mains: Vec<f64>,
    pub reruns: usize,
}

impl<'a> DsVerifier<'a> {
    /// Per-token inputs of `toks` appended to `prefix`: tokens and hashes.
    fn row_inputs(&self, prefix: &[usize], toks: &[usize]) -> HashMap<usize, Vec<f64>> {
        let mut all = prefix.to_vec();
        all.extend_from_slice(toks);
        let hs = self.meta.map(|m| m.hashes(&all));
        let p0 = prefix.len();
        let mut out = HashMap::new();
        for (i, s) in self.pg.inputs.iter().enumerate() {
            if s.name == "tokens" {
                out.insert(i, toks.iter().map(|&t| t as f64).collect());
            } else if s.name.ends_with(".engram.hash") {
                let l: usize = s.name.split('.').nth(1).unwrap().parse().unwrap();
                let j = self.dm.engram_layers.iter().position(|&e| e == l).unwrap();
                let h = &hs.as_ref().expect("Engram hash constants")[j];
                out.insert(i, h[p0..].iter().flatten().map(|&v| v as f64).collect());
            }
        }
        out
    }

    /// Prefill `prompt`: the verifier after it, the greedy first token and
    /// the prompt's target hiddens.
    pub fn prefill(
        pg: &'a PrefillGraph,
        dm: &'a DsDims,
        w: &'a HashMap<String, Vec<f64>>,
        meta: Option<&'a EngramMeta>,
        prompt: &[usize],
    ) -> (Self, u32, Vec<f64>) {
        let mut v = DsVerifier {
            pg,
            dm,
            w,
            meta,
            steps: HashMap::new(),
            caches: HashMap::new(),
            seq: prompt.to_vec(),
            pending: None,
            mains: Vec::new(),
            reruns: 0,
        };
        let len = prompt.len();
        let slots = v.step(1).caches.clone();
        let rows = v.row_inputs(&[], prompt);
        let ins: Vec<Vec<f64>> = pg
            .inputs
            .iter()
            .enumerate()
            .map(|(i, s)| rows.get(&i).cloned().unwrap_or_else(|| w[&s.name].clone()))
            .collect();
        let mut ids = vec![pg.logits, pg.mains.unwrap()];
        ids.extend(slots.iter().map(|c| c.node));
        let g = pg.g.bind(&[(SEQ, len)]).unwrap();
        let mut vals = g.eval(&ins, &ids);
        v.caches = seed_all(&pg.g, &slots, &vals[2..], len, dm.ctx);
        let vocab = dm.vocab;
        let first = argmax(&vals[0][(len - 1) * vocab..len * vocab]);
        let mains = std::mem::take(&mut vals[1]);
        (v, first, mains)
    }

    fn step(&mut self, k: usize) -> &StepGraph {
        let (pg, dm) = (self.pg, self.dm);
        self.steps
            .entry(k)
            .or_insert_with(|| verify_step(pg, dm, k).expect("derives"))
    }

    #[allow(clippy::type_complexity)]
    fn run(
        &mut self,
        caches: &HashMap<Key, Vec<f64>>,
        toks: &[u32],
    ) -> (Vec<u32>, Vec<f64>, HashMap<Key, Vec<f64>>) {
        let k = toks.len();
        let ut: Vec<usize> = toks.iter().map(|&t| t as usize).collect();
        let rows = self.row_inputs(&self.seq, &ut);
        let pos = self.seq.len();
        self.step(k);
        let st = &self.steps[&k];
        let si: Vec<Vec<f64>> = st
            .inputs
            .iter()
            .map(|&x| match x {
                StepInput::Pos => vec![pos as f64],
                StepInput::New(i) => rows[&i].clone(),
                StepInput::Same(i) => self.w[&self.pg.inputs[i].name].clone(),
                StepInput::Cache(c) => caches[&key(&st.caches[c])].clone(),
            })
            .collect();
        let mut ids = st.outputs.clone();
        ids.extend(st.caches.iter().map(|c| c.updated));
        let mut got = st.graph.eval(&si, &ids);
        let logits = got.remove(0);
        let mains = got.remove(0);
        let am = logits.chunks(self.dm.vocab).map(argmax).collect();
        let upd = st.caches.iter().map(key).zip(got).collect();
        (am, mains, upd)
    }

    fn rows_only(&mut self) -> bool {
        self.step(1)
            .caches
            .iter()
            .all(|c| !c.state && c.ring.is_none())
    }
}

impl Verifier for DsVerifier<'_> {
    fn window(&mut self, tokens: &[u32]) -> Vec<u32> {
        let caches = std::mem::take(&mut self.caches);
        let (am, mains, upd) = self.run(&caches, tokens);
        self.caches = caches;
        self.mains = mains;
        self.pending = Some((tokens.to_vec(), upd, Vec::new()));
        am
    }

    fn accept(&mut self, m: usize) {
        let (toks, upd, _) = self.pending.take().expect("a window to accept");
        assert!((1..=toks.len()).contains(&m));
        if m == toks.len() || self.rows_only() {
            self.caches = upd;
        } else {
            let caches = std::mem::take(&mut self.caches);
            let (_, _, upd) = self.run(&caches, &toks[..m]);
            self.caches = upd;
            self.reruns += 1;
        }
        self.seq.extend(toks[..m].iter().map(|&t| t as usize));
        let w = self.mains.len() / toks.len();
        self.mains.truncate(m * w);
    }

    fn room(&self) -> usize {
        self.dm.ctx - self.seq.len()
    }
}

/// The DSpark drafter on derived steps, fed only accepted rows.
pub struct Drafter<'a> {
    sg: &'a SparkGraph,
    dm: &'a DsDims,
    w: &'a HashMap<String, Vec<f64>>,
    steps: HashMap<usize, StepGraph>,
    caches: HashMap<Key, Vec<f64>>,
    pub pos: usize,
    /// The drafts after the last row fed.
    pub draft: Vec<u32>,
}

impl<'a> Drafter<'a> {
    fn ins(&self, rows: &HashMap<&str, Vec<f64>>) -> Vec<Vec<f64>> {
        self.sg
            .inputs
            .iter()
            .map(|s| {
                rows.get(s.name.as_str())
                    .cloned()
                    .unwrap_or_else(|| self.w[&s.name].clone())
            })
            .collect()
    }

    /// Run the drafter graph over the prompt's `mains` rows, `next[t]` the
    /// token after row t.
    pub fn prefill(
        sg: &'a SparkGraph,
        dm: &'a DsDims,
        w: &'a HashMap<String, Vec<f64>>,
        mains: &[f64],
        next: &[u32],
    ) -> Self {
        let mut d = Drafter {
            sg,
            dm,
            w,
            steps: HashMap::new(),
            caches: HashMap::new(),
            pos: next.len(),
            draft: Vec::new(),
        };
        let len = next.len();
        let slots = d.step(1).caches.clone();
        let rows = HashMap::from([
            ("mains", mains.to_vec()),
            ("next", next.iter().map(|&t| t as f64).collect()),
        ]);
        let mut ids = vec![sg.ids];
        ids.extend(slots.iter().map(|c| c.node));
        let g = sg.g.bind(&[(SEQ, len)]).unwrap();
        let vals = g.eval(&d.ins(&rows), &ids);
        d.caches = seed_all(&sg.g, &slots, &vals[1..], len, dm.ctx);
        let b = dm.dspark_block;
        d.draft = vals[0][(len - 1) * b..len * b]
            .iter()
            .map(|&x| x as u32)
            .collect();
        d
    }

    fn step(&mut self, k: usize) -> &StepGraph {
        let (sg, dm) = (self.sg, self.dm);
        self.steps
            .entry(k)
            .or_insert_with(|| spark_step(sg, dm, k).expect("derives"))
    }

    /// Append accepted rows (`mains` `[m, n · dim]`, `next` `[m]`).
    pub fn advance(&mut self, mains: &[f64], next: &[u32]) {
        let k = next.len();
        let rows = HashMap::from([
            ("mains", mains.to_vec()),
            ("next", next.iter().map(|&t| t as f64).collect()),
        ]);
        let ins = self.ins(&rows);
        let pos = self.pos;
        self.step(k);
        let st = &self.steps[&k];
        let si: Vec<Vec<f64>> = st
            .inputs
            .iter()
            .map(|&x| match x {
                StepInput::Pos => vec![pos as f64],
                StepInput::New(i) | StepInput::Same(i) => ins[i].clone(),
                StepInput::Cache(c) => self.caches[&key(&st.caches[c])].clone(),
            })
            .collect();
        let mut ids = st.outputs.clone();
        ids.extend(st.caches.iter().map(|c| c.updated));
        let mut got = st.graph.eval(&si, &ids);
        let draft: Vec<u32> = got[0].iter().map(|&x| x as u32).collect();
        let upd: HashMap<Key, Vec<f64>> = st.caches.iter().map(key).zip(got.drain(3..)).collect();
        self.caches = upd;
        self.draft = draft;
        self.pos += k;
    }
}

/// Speculative decoding with DSpark: `n` tokens after `first` (the
/// prefill's greedy token), windows of at most `k` (≤ block + 1) tokens.
/// `over` replaces the drafts (tests: drafts wrong on purpose); the drafter
/// is fed the accepted rows either way.
pub fn generate_spark(
    v: &mut DsVerifier,
    d: &mut Drafter,
    first: u32,
    n: usize,
    k: usize,
    mut over: Option<&mut dyn FnMut(&[u32], usize) -> Vec<u32>>,
) -> (Vec<u32>, Stats) {
    let mut out = vec![first];
    let mut st = Stats::default();
    let mut last = first;
    while out.len() < n && v.room() > 0 {
        let want = (n - out.len()).min(v.room()).min(k);
        let mut toks = vec![last];
        let drafts = match over.as_mut() {
            Some(f) => f(&out, want - 1),
            None => d.draft.clone(),
        };
        toks.extend(drafts.into_iter().take(want - 1));
        let am = v.window(&toks);
        let mut m = 1;
        while m < toks.len() && am[m - 1] == toks[m] {
            m += 1;
        }
        v.accept(m);
        st.windows += 1;
        st.rows += toks.len();
        st.drafted += toks.len() - 1;
        st.accepted += m - 1;
        st.emitted += m;
        // the kept rows' hiddens; the token after kept row r is am[r]
        let mains = v.mains.clone();
        d.advance(&mains, &am[..m]);
        out.extend_from_slice(&am[..m]);
        last = am[m - 1];
    }
    out.truncate(n);
    (out, st)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_ds_prefill::prefill_graph;
    use crate::ir_ds_prefill::tests::{
        mm, rel, rope, setup, tiny_engram, topk_stable, Bugs, Ref, Weights,
    };

    /// forward.rs `dspark_attention` for the block after main row `t`
    /// (`x` `[blk, d]`, `mkv` the stage's main keys `[T, hd]`), in f64.
    fn spark_attn(r: &Ref, p: &str, x: &[f64], t: usize, mkv: &[f64]) -> Vec<f64> {
        let dm = r.dm;
        let (h, hd, rd, d, win) = (dm.n_heads, dm.head_dim, dm.rope_head_dim, dm.dim, dm.window);
        let b = x.len() / d;
        let pos0 = t + 1;
        let fr = r.w(&format!("{p}.freqs")).to_vec();
        let mut qr = r.linear(x, b, &format!("{p}.attn.wq_a.weight"), dm.q_lora_rank);
        r.rms(&mut qr, &format!("{p}.attn.q_norm.weight"));
        let mut q = r.linear(&qr, b, &format!("{p}.attn.wq_b.weight"), h * hd);
        rope(&mut q, hd, rd, h, &fr, |i| (pos0 + i) as f64, false);
        let mut kv = r.linear(x, b, &format!("{p}.attn.wkv.weight"), hd);
        r.rms(&mut kv, &format!("{p}.attn.kv_norm.weight"));
        rope(&mut kv, hd, rd, 1, &fr, |i| (pos0 + i) as f64, false);
        let lo = (t + 1).saturating_sub(win);
        let mut keys: Vec<f64> = mkv[lo * hd..(t + 1) * hd].to_vec();
        keys.extend_from_slice(&kv);
        let nk = keys.len() / hd;
        let sink = r.w(&format!("{p}.attn.attn_sink"));
        let scale = (hd as f64).powf(-0.5);
        let mut o = vec![0.0; b * h * hd];
        for bi in 0..b {
            for hi in 0..h {
                let qv = &q[(bi * h + hi) * hd..(bi * h + hi + 1) * hd];
                let s: Vec<f64> = (0..nk)
                    .map(|j| {
                        qv.iter()
                            .zip(&keys[j * hd..(j + 1) * hd])
                            .map(|(a, c)| a * c)
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let mx = s.iter().fold(sink[hi], |m, &x| m.max(x));
                let pr: Vec<f64> = s.iter().map(|&x| (x - mx).exp()).collect();
                let den = pr.iter().sum::<f64>() + (sink[hi] - mx).exp();
                let out = &mut o[(bi * h + hi) * hd..(bi * h + hi + 1) * hd];
                for (j, &pj) in pr.iter().enumerate() {
                    for (ov, &kc) in out.iter_mut().zip(&keys[j * hd..(j + 1) * hd]) {
                        *ov += pj * kc;
                    }
                }
                out.iter_mut().for_each(|v| *v /= den);
            }
        }
        rope(&mut o, hd, rd, h, &fr, |i| (pos0 + i) as f64, true);
        let (g, ol) = (dm.o_groups, dm.o_lora_rank);
        let gd = h * hd / g;
        let wo_a = r.w(&format!("{p}.attn.wo_a.weight"));
        let mut mid = vec![0.0; b * g * ol];
        for bi in 0..b {
            for gi in 0..g {
                let ov = &o[(bi * g + gi) * gd..(bi * g + gi + 1) * gd];
                for ri in 0..ol {
                    let wr = &wo_a[(gi * ol + ri) * gd..(gi * ol + ri + 1) * gd];
                    mid[(bi * g + gi) * ol + ri] = ov.iter().zip(wr).map(|(a, c)| a * c).sum();
                }
            }
        }
        r.linear(&mid, b, &format!("{p}.attn.wo_b.weight"), d)
    }

    /// forward.rs `forward_spec_with` (greedy) after every main row:
    /// `(ids [T][blk], logits [T][blk · V], conf [T][blk])`.
    #[allow(clippy::type_complexity)]
    pub(crate) fn spark_ref(
        r: &Ref,
        mains: &[f64],
        next: &[u32],
    ) -> (Vec<Vec<u32>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let dm = r.dm;
        let (hc, d, blk, v, rk) = (dm.hc, dm.dim, dm.dspark_block, dm.vocab, dm.dspark_rank);
        let (hd, rd) = (dm.head_dim, dm.rope_head_dim);
        let t_all = next.len();
        let mut main_x = r.linear(mains, t_all, "mtp.0.main_proj.weight", d);
        r.rms(&mut main_x, "mtp.0.main_norm.weight");
        let mkvs: Vec<Vec<f64>> = (0..dm.n_mtp)
            .map(|s| {
                let p = format!("mtp.{s}");
                let fr = r.w(&format!("{p}.freqs")).to_vec();
                let mut k = r.linear(&main_x, t_all, &format!("{p}.attn.wkv.weight"), hd);
                r.rms(&mut k, &format!("{p}.attn.kv_norm.weight"));
                rope(&mut k, hd, rd, 1, &fr, |j| j as f64, false);
                k
            })
            .collect();
        let e = r.w("embed.weight");
        let last = format!("mtp.{}", dm.n_mtp - 1);
        let (mut ids_all, mut lg_all, mut cf_all) = (Vec::new(), Vec::new(), Vec::new());
        for t in 0..t_all {
            let mut ids = vec![dm.dspark_noise; blk];
            ids[0] = next[t] as usize;
            let mut h = Vec::with_capacity(blk * hc * d);
            for &id in &ids {
                for _ in 0..hc {
                    h.extend_from_slice(&e[id * d..(id + 1) * d]);
                }
            }
            let mut pre: Vec<Vec<f64>> = (0..blk)
                .map(|_| (0..hc).map(|i| (i == 0) as u8 as f64).collect())
                .collect();
            for (s, mkv) in mkvs.iter().enumerate() {
                let p = format!("mtp.{s}");
                let am = r.hc_mixes(&p, "attn", &h);
                let mut x = r.hc_pre(&h, &pre);
                r.rms(&mut x, &format!("{p}.attn_norm.weight"));
                let a = spark_attn(r, &p, &x, t, mkv);
                h = r.hc_post(&a, &h, &am);
                let fm = r.hc_mixes(&p, "ffn", &h);
                let a_pre: Vec<Vec<f64>> = am.iter().map(|m| m.0.clone()).collect();
                let mut x = r.hc_pre(&h, &a_pre);
                r.rms(&mut x, &format!("{p}.ffn_norm.weight"));
                let y = r.moe_at(&p, 0, &x, dm.dspark_experts, dm.dspark_active);
                h = r.hc_post(&y, &h, &fm);
                pre = fm.into_iter().map(|m| m.0).collect();
            }
            let x = r.hc_pre(&h, &pre);
            let mut xn = x.clone();
            r.rms(&mut xn, &format!("{last}.norm.weight"));
            let mut logits = r.linear(&xn, blk, "head.weight", v);
            let memb_w = r.w(&format!("{last}.markov_head.embed.weight"));
            let mhead = r.w(&format!("{last}.markov_head.head.weight"));
            let proj = r.w(&format!("{last}.confidence_head.proj.weight"));
            let mut out = vec![next[t] as usize];
            let mut conf = Vec::new();
            for i in 0..blk {
                let memb = &memb_w[out[i] * rk..(out[i] + 1) * rk];
                let bias = mm(memb, 1, mhead, v, rk);
                let li = &mut logits[i * v..(i + 1) * v];
                for (a, c) in li.iter_mut().zip(&bias) {
                    *a += c;
                }
                out.push(topk_stable(li, 1)[0]);
                let feat: Vec<f64> = x[i * d..(i + 1) * d].iter().chain(memb).copied().collect();
                conf.push(mm(&feat, 1, proj, 1, d + rk)[0]);
            }
            ids_all.push(out[1..].iter().map(|&i| i as u32).collect());
            lg_all.push(logits);
            cf_all.push(conf);
        }
        (ids_all, lg_all, cf_all)
    }

    fn fixture() -> (DsDims, PrefillGraph, SparkGraph, Weights, Vec<f64>) {
        let dm = DsDims {
            ctx: 32,
            ..DsDims::tiny_full()
        };
        let pg = prefill_graph(&dm);
        let sg = spark_graph(&dm);
        let (mut w, all) = setup(&dm, &pg, dm.ctx);
        // the drafter's own weights (names not in the main graph)
        let sw = crate::ir_ds_prefill::tests::weights_of(&sg.inputs, &dm);
        for (k, v) in sw {
            w.entry(k).or_insert(v);
        }
        (dm, pg, sg, w, all)
    }

    /// Main-model hiddens of the run's tokens (from the reference) and the
    /// next-token stream, the drafter's inputs.
    fn main_rows(dm: &DsDims, w: &Weights, all: &[f64], t: usize) -> (Vec<f64>, Vec<u32>) {
        let (_, mains) = Ref {
            dm,
            w,
            bugs: Bugs::default(),
            load: None,
        }
        .forward_full(&all[..t]);
        let next = all[1..=t].iter().map(|&x| x as u32).collect();
        (mains, next)
    }

    fn spark_inputs(sg: &SparkGraph, w: &Weights, mains: &[f64], next: &[u32]) -> Vec<Vec<f64>> {
        sg.inputs
            .iter()
            .map(|s| match s.name.as_str() {
                "mains" => mains.to_vec(),
                "next" => next.iter().map(|&x| x as f64).collect(),
                n => w[n].clone(),
            })
            .collect()
    }

    /// The main graph's DSpark target hiddens against the reference's.
    #[test]
    fn mains_match_reference() {
        let (dm, pg, _, w, all) = fixture();
        let t = 9;
        let g = pg.g.bind(&[(SEQ, t)]).unwrap();
        let ins = crate::ir_ds_prefill::tests::inputs(&pg, &w, &all[..t]);
        let got = g.eval(&ins, &[pg.mains.unwrap()]);
        let (want, _) = main_rows(&dm, &w, &all, t);
        let e = rel(&got[0], &want);
        eprintln!("mains T={t}: max rel err {e:.2e}");
        assert!(e < 1e-12, "{e:e}");
    }

    /// The drafter graph, every row, against the reference: drafts equal,
    /// logits and confidences to f64 rounding.
    #[test]
    fn spark_prefill_matches_reference() {
        let (dm, _, sg, w, all) = fixture();
        let r = Ref {
            dm: &dm,
            w: &w,
            bugs: Bugs::default(),
            load: None,
        };
        for t in [1, 3, 6, 9] {
            let (mains, next) = main_rows(&dm, &w, &all, t);
            let g = sg.g.bind(&[(SEQ, t)]).unwrap();
            let got = g.eval(
                &spark_inputs(&sg, &w, &mains, &next),
                &[sg.ids, sg.logits, sg.conf],
            );
            let (ids, lg, cf) = spark_ref(&r, &mains, &next);
            let gids: Vec<u32> = got[0].iter().map(|&x| x as u32).collect();
            assert_eq!(gids, ids.concat(), "T={t}: drafts");
            let el = rel(&got[1], &lg.concat());
            let ec = rel(&got[2], &cf.concat());
            eprintln!(
                "dspark T={t}: drafts {:?}; logits err {el:.2e}, conf err {ec:.2e}",
                ids.last().unwrap()
            );
            assert!(el < 1e-10 && ec < 1e-10, "{el:e} {ec:e}");
        }
    }

    /// The drafter's derived steps (k accepted rows, drafts for the last)
    /// against its graph over the grown sequence.
    #[test]
    fn spark_step_matches_prefill() {
        let (dm, _, sg, w, all) = fixture();
        for ks in [&[1usize][..], &[3, 1, 2]] {
            let p0 = 4;
            let total = p0 + ks.iter().sum::<usize>();
            let (mains, next) = main_rows(&dm, &w, &all, total);
            let nd = mains.len() / total;
            let mut d = Drafter::prefill(&sg, &dm, &w, &mains[..p0 * nd], &next[..p0]);
            let mut p = p0;
            for &k in ks {
                d.advance(&mains[p * nd..(p + k) * nd], &next[p..p + k]);
                p += k;
                let g = sg.g.bind(&[(SEQ, p)]).unwrap();
                let want = g.eval(
                    &spark_inputs(&sg, &w, &mains[..p * nd], &next[..p]),
                    &[sg.ids],
                );
                let b = dm.dspark_block;
                let wl: Vec<u32> = want[0][(p - 1) * b..p * b]
                    .iter()
                    .map(|&x| x as u32)
                    .collect();
                assert_eq!(d.draft, wl, "{ks:?} at {p}");
            }
            eprintln!("dspark steps {ks:?} from {p0}: drafts match the drafter graph");
        }
    }

    /// Draft → verify through derived windows equals greedy decoding
    /// (k = 1 windows) equals the prefill's argmax over the sequence, with
    /// DSpark's own drafts and with drafts corrupted on purpose (so windows
    /// stop at every offset and the ring caches roll back).
    #[test]
    fn spark_speculation_is_greedy() {
        let (dm, pg, sg, w, all) = fixture();
        let meta = tiny_engram(&dm);
        let prompt: Vec<usize> = all[..5].iter().map(|&x| x as usize).collect();
        let n = 9;
        // greedy a little past n: the token after the last fed one
        let start = || {
            let (v, first, mains) = DsVerifier::prefill(&pg, &dm, &w, Some(&meta), &prompt);
            let mut next: Vec<u32> = prompt[1..].iter().map(|&t| t as u32).collect();
            next.push(first);
            (v, first, mains, next)
        };
        let (mut g, first, mains, next) = start();
        let mut d = Drafter::prefill(&sg, &dm, &w, &mains, &next);
        let long = n + dm.dspark_block + 1;
        let (truth, gs) = generate_spark(&mut g, &mut d, first, long, 1, None);
        assert_eq!(gs.windows, long - 1);
        let greedy = truth[..n].to_vec();
        // the greedy chain vs one prefill over the whole sequence
        let mut seq = prompt.clone();
        seq.extend(truth[..long - 1].iter().map(|&t| t as usize));
        let seqf: Vec<f64> = seq.iter().map(|&t| t as f64).collect();
        let mut wh = w.clone();
        crate::ir_ds_prefill::tests::with_hashes(&dm, &meta, &mut wh, &seqf);
        let gr = pg.g.bind(&[(SEQ, seq.len())]).unwrap();
        let l = gr.eval(
            &crate::ir_ds_prefill::tests::inputs(&pg, &wh, &seqf),
            &[pg.logits],
        );
        let want: Vec<u32> = l[0].chunks(dm.vocab).map(argmax).collect();
        assert_eq!(&want[prompt.len() - 1..], &truth[..], "steps vs prefill");
        eprintln!("greedy {greedy:?}");
        for k in [2, dm.dspark_block + 1] {
            let (mut v, f, mains, next) = start();
            let mut d = Drafter::prefill(&sg, &dm, &w, &mains, &next);
            let (spec, st) = generate_spark(&mut v, &mut d, f, n, k, None);
            assert_eq!(spec, greedy, "dspark k={k}");
            eprintln!(
                "dspark k={k}: {:.2} tokens/window, acceptance {:.0}% ({}/{}), {} reruns",
                st.tokens_per_window(),
                100.0 * st.acceptance(),
                st.accepted,
                st.drafted,
                v.reruns
            );
            // the truth with every third draft corrupted
            let (mut v, f, mains, next) = start();
            let mut d = Drafter::prefill(&sg, &dm, &w, &mains, &next);
            let mut i = 0u32;
            let vocab = dm.vocab as u32;
            let mut over = |out: &[u32], mx: usize| -> Vec<u32> {
                let at = out.len();
                (0..mx)
                    .filter_map(|j| truth.get(at + j).copied())
                    .map(|t| {
                        i += 1;
                        if i % 3 == 0 {
                            (t + 1) % vocab
                        } else {
                            t
                        }
                    })
                    .collect()
            };
            let (bad, bs) = generate_spark(&mut v, &mut d, f, n, k, Some(&mut over));
            assert_eq!(bad, greedy, "corrupted drafts k={k}");
            assert!(bs.accepted > 0 && bs.accepted < bs.drafted);
            // the drafter, fed m > 1 rows at a time, still agrees with its graph
            let seq: Vec<usize> = v.seq.clone();
            let t = seq.len();
            let sf: Vec<f64> = seq.iter().map(|&x| x as f64).collect();
            let mut wh = w.clone();
            crate::ir_ds_prefill::tests::with_hashes(&dm, &meta, &mut wh, &sf);
            let gm = pg.g.bind(&[(SEQ, t)]).unwrap();
            let mrows = gm.eval(
                &crate::ir_ds_prefill::tests::inputs(&pg, &wh, &sf),
                &[pg.mains.unwrap()],
            );
            let mut nx: Vec<u32> = seq[1..].iter().map(|&x| x as u32).collect();
            nx.push(truth[t - prompt.len()]);
            let gs = sg.g.bind(&[(SEQ, t)]).unwrap();
            let dr = gs.eval(&spark_inputs(&sg, &w, &mrows[0], &nx), &[sg.ids]);
            let b = dm.dspark_block;
            let wl: Vec<u32> = dr[0][(t - 1) * b..t * b]
                .iter()
                .map(|&x| x as u32)
                .collect();
            assert_eq!(d.draft, wl, "drafter after corrupted windows k={k}");
            eprintln!(
                "corrupted drafts k={k}: {:.2} tokens/window, acceptance {:.0}%, {} reruns; drafter state matches its graph",
                bs.tokens_per_window(),
                100.0 * bs.acceptance(),
                v.reruns
            );
        }
    }
}
