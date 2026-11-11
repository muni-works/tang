//! Ring caches: O(window) state instead of O(context) for the two idioms
//! that only look a bounded way back.
//!
//! - A reduction over `T` whose every contribution is masked to a window
//!   `t − W < j ≤ t` (sliding-window attention, DeepSeek's compressor
//!   group) reads its reduce coordinate in [`Mode::Win`]`(W − 1)`.
//! - A guarded read along `T` at `t + δ`, `δ ∈ [−c, 0]` over fixed
//!   coordinates (a causal conv with zero history, a pool of the last
//!   few keys) reads its source in `Win(c)`, the view shifted by `c`.
//!
//! **Layout: a shifted buffer, not a modular ring.** The cache input holds
//! the last `S` old positions in order, row `s` at position `p0 − S + s`
//! (NaN before position 0). The step builds the extended rows
//! `E = old ++ new` (`S + k` rows, row `s` at `p0 − S + s`, zero before
//! position 0: a conv's zero history) and writes back `E[k..k + S]`.
//! Every read is affine with a constant offset — no `mod capacity`, which
//! `tang_expr` cannot yet say — and `k` may change between steps since the
//! persistent state is `S` rows whatever `k` is. The price is an `O(S)`
//! copy per step (small next to the reads it saves), and the write-back
//! cannot be in place: it reads row `s + k` of the rows it replaces.
//!
//! An iota read in `Win` mode is the position, or `1e9` before position 0:
//! an unwritten row is then in every query's future, and the causal mask
//! the derivation already demands turns it into the monoid identity.

use tang::Scalar;
use tang_expr::{ExprGraph, ExprId, Node};

use super::{
    err, Arg, CacheSlot, Comp, Deriver, Dim, Mode, Op, ShapeError, StepInput, TensorId, View,
};

/// A row position past every query: masked by the causal mask.
const FUTURE: f64 = 1e9;

fn sel(c: ExprId, a: ExprId, b: ExprId) -> ExprId {
    tang_expr::with_graph(|eg| eg.select(c, a, b))
}

fn lit(v: f64) -> ExprId {
    ExprId::from_f64(v)
}

impl CacheSlot {
    /// Rows of this cache's sequence axis at `capacity`.
    pub fn rows(&self, capacity: usize) -> usize {
        self.ring.unwrap_or(capacity)
    }

    /// The position cache row `row` holds once the sequence is `len` long
    /// (`None`: not yet written).
    pub fn position(&self, row: usize, len: usize) -> Option<usize> {
        let p = match self.ring {
            Some(s) => (len + row).checked_sub(s)?,
            None => row,
        };
        (p < len).then_some(p)
    }
}

/// The cache buffer of a row slot (not a carried state) after a prefill of
/// `len` positions, from its node's prefill value `v` of shape `shape`
/// (the sequence axis `len` long): unwritten rows NaN.
pub fn seed_cache(
    slot: &CacheSlot,
    shape: &[usize],
    v: &[f64],
    len: usize,
    capacity: usize,
) -> Vec<f64> {
    assert!(!slot.state, "a carried state is not rows");
    let ax = slot.axis;
    assert_eq!(shape[ax], len);
    let outer: usize = shape[..ax].iter().product();
    let inner: usize = shape[ax + 1..].iter().product();
    let rows = slot.rows(capacity);
    let mut buf = vec![f64::NAN; outer * rows * inner];
    for o in 0..outer {
        for r in 0..rows {
            if let Some(p) = slot.position(r, len) {
                let (src, dst) = ((o * len + p) * inner, (o * rows + r) * inner);
                buf[dst..dst + inner].copy_from_slice(&v[src..src + inner]);
            }
        }
    }
    buf
}

impl Deriver<'_> {
    /// Ring node `n` (one `T`-axis, `axis`) in `Win(s)`: the extended rows
    /// `E`; the slot's input is the last `s` old rows, its update `E[k..]`.
    pub(super) fn ring(&mut self, n: usize, axis: usize, s: usize) -> Result<TensorId, ShapeError> {
        if let Some(&c) = self.ring_of.get(&(n, s)) {
            // E is rebuilt from the slot's input (memoized by `get`)
            let _ = c;
        }
        let new = self.get(n, self.modes_with(n, Mode::New))?;
        let k = self.spec.new;
        let mut bshape = self.g.nodes[n].shape.clone();
        bshape[axis] = Dim::Fixed(s);
        let mut eshape = bshape.clone();
        eshape[axis] = Dim::Fixed(s + k);
        let slot = self.caches.len();
        let input = self.sg.input(&bshape);
        self.inputs.push(StepInput::Cache(slot));
        let r = eshape.len();
        let mut nv = View::identity(r, r);
        nv.forms[axis].offset = -(s as i64);
        let iota = self.sg.iota(&eshape, axis);
        let args = vec![
            Arg {
                src: input,
                view: View::identity(r, r).guarded(),
            },
            Arg {
                src: new,
                view: nv.guarded(),
            },
            Arg {
                src: iota,
                view: View::identity(r, r),
            },
            self.pos_arg(&eshape)?,
        ];
        let sf = s as f64;
        let ext = self.sg.comp(&eshape, &[], args, None, move |v| {
            let (old, new, row, p0) = (v[0], v[1], v[2], v[3]);
            let pos = p0 - lit(sf) + row;
            let half = lit(0.5);
            vec![sel(
                pos + half,
                sel(row - lit(sf) + half, new, old),
                lit(0.0),
            )]
        })?;
        let mut ev = View::identity(r, r);
        ev.forms[axis].offset = k as i64;
        let updated = self
            .sg
            .comp(&bshape, &[], vec![Arg { src: ext, view: ev }], None, |v| {
                vec![v[0]]
            })?;
        self.caches.push(CacheSlot {
            node: TensorId(n as u32),
            axis,
            input,
            updated,
            state: false,
            ring: Some(s),
        });
        self.ring_of.insert((n, s), slot);
        Ok(ext)
    }

    /// An iota along a `Win(s)` axis: the row's position, [`FUTURE`] before
    /// position 0.
    pub(super) fn win_iota(
        &mut self,
        it: TensorId,
        shape: &[Dim],
        s: usize,
    ) -> Result<TensorId, ShapeError> {
        if self.opts.batch.is_some() {
            return err("step: a windowed position read with batched sessions is not derived yet");
        }
        let r = shape.len();
        let args = vec![
            Arg {
                src: it,
                view: View::identity(r, r),
            },
            self.pos_arg(shape)?,
        ];
        let sf = s as f64;
        self.sg.comp(shape, &[], args, None, move |v| {
            let pos = v[0] + v[1] - lit(sf);
            vec![sel(pos + lit(0.5), pos, lit(FUTURE))]
        })
    }

    /// Switch each sequence reduce coordinate of `c` whose every
    /// contribution is masked to a window behind a new (or last) point
    /// coordinate to `Win(W − 1)`.
    pub(super) fn windowed(&self, c: &Comp, coord: &mut [Option<Mode>], pts: usize) {
        for j in pts..coord.len() {
            if coord[j] != Some(Mode::Full) {
                continue;
            }
            let w = (0..pts)
                .filter(|&i| matches!(coord[i], Some(Mode::New | Mode::Last)))
                .filter_map(|i| self.window(c, i, j))
                .min();
            if let Some(w) = w {
                coord[j] = Some(Mode::Win(w.saturating_sub(1).max(1)));
            }
        }
    }

    /// `W` when every value `c`'s body contributes is the monoid identity
    /// unless `j > i − W` (point coord `i`, reduce coord `j`): a chain of
    /// `select(cond, ·, identity)` holding one such affine lower bound.
    fn window(&self, c: &Comp, i: usize, j: usize) -> Option<usize> {
        let m = c.monoid.as_ref()?;
        let args: Vec<(TensorId, View)> = c.args.iter().map(|a| (a.src, a.view.clone())).collect();
        let mut x = 0.0f64;
        for (k, &out) in c.body_outs.iter().enumerate() {
            x = x.max(self.lower(&c.body, &args, c.rank(), out, m.identity[k], i, j)?);
        }
        (x > 0.0 && x < 1e6).then(|| x.ceil() as usize)
    }

    /// The tightest `x` with `e` the identity unless `j − i > −x`.
    #[allow(clippy::too_many_arguments)]
    fn lower(
        &self,
        body: &ExprGraph,
        args: &[(TensorId, View)],
        rank: usize,
        e: ExprId,
        id: f64,
        i: usize,
        j: usize,
    ) -> Option<f64> {
        let nodes = body.nodes_slice();
        match nodes[e.index() as usize] {
            Node::Select(cond, t, f) => {
                let l = nodes[f.index() as usize].as_f64()?;
                if !(l == id || (id <= -1e30 && l <= -1e30)) {
                    return None;
                }
                // cond = a·(j − i) + b over i, j only, a > 0, open at j = i
                let here = self.affine(body, args, rank, cond).and_then(|(coefs, b)| {
                    let only = coefs
                        .iter()
                        .enumerate()
                        .all(|(d, &k)| k == 0.0 || d == i || d == j);
                    (only && coefs[j] > 0.0 && coefs[i] == -coefs[j] && b > 0.0)
                        .then(|| b / coefs[j])
                });
                let inner = self.lower(body, args, rank, t, id, i, j);
                match (here, inner) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                }
            }
            Node::Var(v) => {
                let (src, view) = args.get(v as usize)?;
                let Op::Comp(m) = &self.g.nodes[src.n()].op else {
                    return None;
                };
                if !m.red.is_empty() || m.tuple != 0 || view.is_gather() || m.has_gather() {
                    return None;
                }
                let inner: Vec<(TensorId, View)> = m
                    .args
                    .iter()
                    .map(|a| (a.src, a.view.compose(view, rank)))
                    .collect();
                self.lower(&m.body, &inner, rank, m.body_outs[0], id, i, j)
            }
            _ => None,
        }
    }

    /// A guarded read of a one-`T` source at `t + δ`, `δ ∈ [−c, 0]` (`t` a
    /// new or last coordinate with coefficient 1, `δ` an offset plus fixed
    /// coordinates): the source in `Win(c)` and the view shifted onto it.
    /// The guard stays (harmless: the shifted read is in bounds along `T`,
    /// and other axes keep their padding).
    pub(super) fn shifted(
        &self,
        c: &Comp,
        a: &Arg,
        coord: &[Option<Mode>],
    ) -> Option<(Vec<Option<Mode>>, View)> {
        if a.view.is_gather() {
            return None;
        }
        let shape = &self.g.nodes[a.src.n()].shape;
        let tax: Vec<usize> = (0..shape.len())
            .filter(|&d| self.is_seq(shape[d]))
            .collect();
        let [sa] = tax[..] else {
            return None;
        };
        let on_seq = |d: usize, k: i64| k != 0 && coord[d].is_some();
        for (ax, f) in a.view.forms.iter().enumerate() {
            if ax != sa && f.coefs.iter().enumerate().any(|(d, &k)| on_seq(d, k)) {
                return None;
            }
        }
        let f = &a.view.forms[sa];
        let seq: Vec<usize> = (0..f.coefs.len())
            .filter(|&d| on_seq(d, f.coefs[d]))
            .collect();
        let [d] = seq[..] else {
            return None;
        };
        if f.coefs[d] != 1 {
            return None;
        }
        // a plain read along T (the guard pads a fixed axis: partial rope)
        if f.offset == 0 && f.coefs.iter().filter(|&&k| k != 0).count() == 1 {
            let mut smodes = vec![None; shape.len()];
            smodes[sa] = coord[d];
            return Some((smodes, a.view.clone()));
        }
        let last = match coord[d] {
            Some(Mode::New) => 0,
            Some(Mode::Last) => self.spec.new as i64 - 1,
            _ => return None,
        };
        let space = c.space();
        let (mut lo, mut hi) = (f.offset, f.offset);
        for (e, &k) in f.coefs.iter().enumerate() {
            if e == d || k == 0 {
                continue;
            }
            let Dim::Fixed(n) = space[e] else {
                return None;
            };
            let span = k * (n as i64 - 1);
            if k > 0 {
                hi += span;
            } else {
                lo += span;
            }
        }
        if hi > 0 {
            return None;
        }
        let mut smodes = vec![None; shape.len()];
        let mut view = a.view.clone();
        if lo == 0 {
            smodes[sa] = coord[d];
            return Some((smodes, view));
        }
        let s = (-lo) as usize;
        view.forms[sa].offset += s as i64 + last;
        smodes[sa] = Some(Mode::Win(s));
        Some((smodes, view))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::super::super::{Form, Monoid, TensorGraph};
    use super::super::{derive_step, StepGraph, StepInput, StepSpec};
    use super::*;

    const T: Dim = Dim::Sym("T");
    const CAP: usize = 32;

    fn rand(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * 0.8
            })
            .collect()
    }

    fn fx(n: usize) -> Dim {
        Dim::Fixed(n)
    }

    fn arg(src: TensorId, forms: Vec<Form>, guard: bool) -> Arg {
        Arg {
            src,
            view: View {
                forms,
                guard,
                gather: None,
            },
        }
    }

    fn u(r: usize, d: usize) -> Form {
        Form::unit(r, d)
    }

    fn id(r: usize) -> Vec<Form> {
        (0..r).map(|d| u(r, d)).collect()
    }

    fn linear(g: &mut TensorGraph, x: TensorId, i: usize, o: usize) -> TensorId {
        let w = g.input(&[fx(o), fx(i)]);
        g.comp(
            &[T, fx(o)],
            &[fx(i)],
            vec![
                arg(x, vec![u(3, 0), u(3, 2)], false),
                arg(w, vec![u(3, 1), u(3, 2)], false),
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap()
    }

    /// Masked reduce over `j` (last space dim) of `[T, …, T]`-indexed args:
    /// `f` where `i − w < j ≤ i`, identity elsewhere (`w = 0`: causal only).
    fn win_reduce(
        g: &mut TensorGraph,
        out: &[Dim],
        args: Vec<Arg>,
        m: Monoid,
        idv: f64,
        w: usize,
        f: impl Fn(&[ExprId]) -> ExprId,
    ) -> TensorId {
        let mut sp = out.to_vec();
        sp.push(T);
        let r = sp.len();
        let (ti, ji) = (g.iota(&sp, 0), g.iota(&sp, r - 1));
        let na = args.len();
        let mut args = args;
        args.push(arg(ti, id(r), false));
        args.push(arg(ji, id(r), false));
        let wf = w as f64;
        g.comp(out, &[T], args, Some(m), move |v| {
            let (t, j) = (v[na], v[na + 1]);
            let inner = if w == 0 {
                f(v)
            } else {
                sel(j - t + lit(wf - 0.5), f(v), lit(idv))
            };
            vec![sel(t - j + lit(0.5), inner, lit(idv))]
        })
        .unwrap()
    }

    /// One window-attention layer (`w = 0`: plain causal) over `x [T, d]`.
    fn window_attn(g: &mut TensorGraph, x: TensorId, d: usize, w: usize) -> TensorId {
        let (q, k, v) = (linear(g, x, d, d), linear(g, x, d, d), linear(g, x, d, d));
        let s = g
            .comp(
                &[T, T],
                &[fx(d)],
                vec![
                    arg(q, vec![u(3, 0), u(3, 2)], false),
                    arg(k, vec![u(3, 1), u(3, 2)], false),
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        let s2 = |r: usize| arg(s, vec![u(r, 0), u(r, r - 1)], false);
        let mx = win_reduce(
            g,
            &[T],
            vec![s2(2)],
            Monoid::max(),
            f64::NEG_INFINITY,
            w,
            |v| v[0],
        );
        let den = win_reduce(
            g,
            &[T],
            vec![s2(2), arg(mx, vec![u(2, 0)], false)],
            Monoid::sum(),
            0.0,
            w,
            |v| (v[0] - v[1]).exp(),
        );
        let num = win_reduce(
            g,
            &[T, fx(d)],
            vec![
                s2(3),
                arg(mx, vec![u(3, 0)], false),
                arg(v, vec![u(3, 2), u(3, 1)], false),
            ],
            Monoid::sum(),
            0.0,
            w,
            |v| (v[0] - v[1]).exp() * v[2],
        );
        g.comp(
            &[T, fx(d)],
            &[],
            vec![
                arg(num, id(2), false),
                arg(den, vec![u(2, 0)], false),
                arg(x, id(2), false),
            ],
            None,
            |v| vec![v[0] / v[1] + v[2]],
        )
        .unwrap()
    }

    /// Depthwise causal conv, zero history: `y[t, c] = Σ_i w[c, i]·x[t − (kc−1) + i, c]`.
    fn conv(g: &mut TensorGraph, x: TensorId, d: usize, kc: usize) -> TensorId {
        let w = g.input(&[fx(d), fx(kc)]);
        let mut tf = Form::zero(3);
        tf.coefs[0] = 1;
        tf.coefs[2] = 1;
        tf.offset = 1 - kc as i64;
        let y = g
            .comp(
                &[T, fx(d)],
                &[fx(kc)],
                vec![
                    arg(x, vec![tf, u(3, 1)], true),
                    arg(w, vec![u(3, 1), u(3, 2)], false),
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        g.map(&[y], |v| v[0] / (lit(1.0) + (-v[0]).exp())).unwrap()
    }

    fn concrete(shape: &[Dim], t: usize) -> Vec<usize> {
        shape
            .iter()
            .map(|&d| if d == T { t } else { d.n() })
            .collect()
    }

    fn rows(buf: &[f64], shape: &[usize], ax: usize, r0: usize, r1: usize) -> Vec<f64> {
        let outer: usize = shape[..ax].iter().product();
        let inner: usize = shape[ax + 1..].iter().product();
        let mut out = Vec::new();
        for o in 0..outer {
            for r in r0..r1 {
                let b = (o * shape[ax] + r) * inner;
                out.extend_from_slice(&buf[b..b + inner]);
            }
        }
        out
    }

    fn max_err(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let e = (x - y).abs() / (1.0 + y.abs());
                if e.is_finite() {
                    e
                } else {
                    f64::INFINITY
                }
            })
            .fold(0.0, f64::max)
    }

    fn step(g: &TensorGraph, out: TensorId, k: usize) -> StepGraph {
        derive_step(
            g,
            &[out],
            &StepSpec {
                seq: "T",
                new: k,
                capacity: CAP,
            },
        )
        .unwrap()
    }

    #[derive(Clone, Copy, Default)]
    struct Skew {
        pos: usize,
        /// seed rings one position stale
        stale: bool,
    }

    /// Prefill `p0`, then chained derived steps `ks`, each against a
    /// prefill of the grown sequence: new output rows and every cache's
    /// written rows. Unwritten cache rows hold NaN.
    fn check(g: &TensorGraph, out: TensorId, p0: usize, ks: &[usize], skew: Skew) -> f64 {
        let tmax = p0 + ks.iter().sum::<usize>();
        let mut ins: Vec<(usize, Vec<Dim>)> = g
            .nodes()
            .iter()
            .filter_map(|nd| match nd.op {
                Op::Input(i) => Some((i, nd.shape.clone())),
                _ => None,
            })
            .collect();
        ins.sort_by_key(|x| x.0);
        let data: Vec<(Vec<Dim>, Vec<f64>)> = ins
            .into_iter()
            .map(|(i, s)| {
                let n = concrete(&s, tmax).iter().product();
                (s, rand(n, 7 + i as u64))
            })
            .collect();
        let pre = |t: usize| -> Vec<Vec<f64>> {
            data.iter()
                .map(|(s, v)| match s.iter().position(|&d| d == T) {
                    Some(a) => rows(v, &concrete(s, tmax), a, 0, t),
                    None => v.clone(),
                })
                .collect()
        };
        let key = |c: &CacheSlot| (c.node, c.ring);
        let mut caches: HashMap<(TensorId, Option<usize>), Vec<f64>> = HashMap::new();
        let mut pos = p0;
        let mut worst = 0.0f64;
        for &k in ks {
            let st = step(g, out, k);
            if caches.is_empty() {
                let nodes: Vec<TensorId> = st.caches.iter().map(|c| c.node).collect();
                let at = if skew.stale { pos - 1 } else { pos };
                let vals = g.bind(&[("T", at)]).unwrap().eval(&pre(at), &nodes);
                for (c, v) in st.caches.iter().zip(vals) {
                    let mut b = seed_cache(c, &concrete(g.shape(c.node), at), &v, at, CAP);
                    if skew.stale && c.ring.is_none() {
                        let cur = g.bind(&[("T", pos)]).unwrap().eval(&pre(pos), &[c.node]);
                        b = seed_cache(c, &concrete(g.shape(c.node), pos), &cur[0], pos, CAP);
                    }
                    caches.insert(key(c), b);
                }
            }
            let inputs: Vec<Vec<f64>> = st
                .inputs
                .iter()
                .map(|si| match *si {
                    StepInput::Pos => vec![(pos + skew.pos) as f64],
                    StepInput::Same(i) => data[i].1.clone(),
                    StepInput::New(i) => {
                        let (s, v) = &data[i];
                        let a = s.iter().position(|&d| d == T).unwrap();
                        rows(v, &concrete(s, tmax), a, pos, pos + k)
                    }
                    StepInput::Cache(c) => caches[&key(&st.caches[c])].clone(),
                })
                .collect();
            let mut ids = st.outputs.clone();
            ids.extend(st.caches.iter().map(|c| c.updated));
            let got = st.graph.eval(&inputs, &ids);
            let grown = pos + k;
            let mut rids = vec![out];
            rids.extend(st.caches.iter().map(|c| c.node));
            let want = g.bind(&[("T", grown)]).unwrap().eval(&pre(grown), &rids);
            let os = concrete(g.shape(out), grown);
            worst = worst.max(max_err(&got[0], &rows(&want[0], &os, 0, pos, grown)));
            for (ci, c) in st.caches.iter().enumerate() {
                let ws = concrete(g.shape(c.node), grown);
                let exp = seed_cache(c, &ws, &want[1 + ci], grown, CAP);
                // compare the written rows; unwritten ones stay NaN
                let gv = &got[1 + ci];
                let e = gv
                    .iter()
                    .zip(&exp)
                    .map(|(x, y)| match (x.is_nan(), y.is_nan()) {
                        (true, true) => 0.0,
                        (false, false) => (x - y).abs() / (1.0 + y.abs()),
                        // a ring may zero rows before position 0
                        _ if y.is_nan() => 0.0,
                        _ => f64::INFINITY,
                    })
                    .fold(0.0, f64::max);
                worst = worst.max(e);
                caches.insert(key(c), gv.clone());
            }
            pos = grown;
        }
        worst
    }

    fn ring_sizes(g: &TensorGraph, out: TensorId) -> Vec<(Option<usize>, Vec<Dim>)> {
        let st = step(g, out, 1);
        st.caches
            .iter()
            .map(|c| (c.ring, st.graph.shape(c.input).to_vec()))
            .collect()
    }

    const CHAINS: [&[usize]; 4] = [&[1], &[4], &[1, 4, 1, 1, 2, 3, 1, 4], &[2, 2, 2, 2, 2, 2]];

    fn window_model(w: usize) -> (TensorGraph, TensorId) {
        let mut g = TensorGraph::new();
        let x = g.input(&[T, fx(4)]);
        let h = window_attn(&mut g, x, 4, w);
        let o = window_attn(&mut g, h, 4, w);
        (g, o)
    }

    #[test]
    fn window_attention_keeps_a_ring() {
        let (g, o) = window_model(3);
        let sizes = ring_sizes(&g, o);
        eprintln!("window attention W=3, 2 layers: caches {sizes:?}");
        // K and V per layer, W − 1 = 2 rows each, nothing at capacity
        assert_eq!(sizes.len(), 4);
        assert!(sizes.iter().all(|(r, s)| *r == Some(2) && s[0] == fx(2)));
        for p0 in [1, 2, 5] {
            for ks in CHAINS {
                let e = check(&g, o, p0, ks, Skew::default());
                eprintln!("window W=3 from {p0}, steps {ks:?}: {e:.2e}");
                assert!(e < 1e-12, "{p0} {ks:?}: {e:e}");
            }
        }
    }

    #[test]
    fn causal_conv_keeps_a_ring() {
        let mut g = TensorGraph::new();
        let x = g.input(&[T, fx(4)]);
        let h = linear(&mut g, x, 4, 4);
        let y = conv(&mut g, h, 4, 4);
        let z = conv(&mut g, y, 4, 3);
        let o = window_attn(&mut g, z, 4, 0);
        let sizes = ring_sizes(&g, o);
        eprintln!("conv(4) → conv(3) → causal attention: caches {sizes:?}");
        let rings: Vec<usize> = sizes.iter().filter_map(|x| x.0).collect();
        assert_eq!(rings, [3, 2]);
        for p0 in [1, 2, 6] {
            for ks in CHAINS {
                let e = check(&g, o, p0, ks, Skew::default());
                eprintln!("conv from {p0}, steps {ks:?}: {e:.2e}");
                assert!(e < 1e-12, "{p0} {ks:?}: {e:e}");
            }
        }
    }

    #[test]
    fn ring_negative_controls() {
        let (g, o) = window_model(3);
        let mut c = TensorGraph::new();
        let x = c.input(&[T, fx(4)]);
        let y = conv(&mut c, x, 4, 4);
        let co = linear(&mut c, y, 4, 4);
        for (name, g, o) in [("window", &g, o), ("conv", &c, co)] {
            let pos = check(
                g,
                o,
                5,
                &[1, 4],
                Skew {
                    pos: 1,
                    ..Skew::default()
                },
            );
            let stale = check(
                g,
                o,
                5,
                &[1, 4],
                Skew {
                    stale: true,
                    ..Skew::default()
                },
            );
            eprintln!("{name}: position off by one {pos:.2e}, ring one row stale {stale:.2e}");
            // these toys have no position dependence: a skewed p0 is harmless
            // once the ring is full; a stale ring is not
            let _ = pos;
            assert!(stale > 1e-3);
        }
        // a window mask against a fixed coordinate is not a window: full cache
        let (w, _) = window_model(0);
        let full = ring_sizes(&w, TensorId(w.nodes().len() as u32 - 1));
        assert!(full.iter().all(|(r, s)| r.is_none() && s[0] == fx(CAP)));
        // a read ahead along T stays refused
        let mut f = TensorGraph::new();
        let x = f.input(&[T, fx(2)]);
        let mut tf = Form::unit(2, 0);
        tf.offset = 1;
        let ahead = f
            .comp(
                &[T, fx(2)],
                &[],
                vec![arg(x, vec![tf, u(2, 1)], true)],
                None,
                |v| vec![v[0]],
            )
            .unwrap();
        let r = derive_step(
            &f,
            &[ahead],
            &StepSpec {
                seq: "T",
                new: 1,
                capacity: CAP,
            },
        );
        assert!(r.is_err());
    }

    /// coder-next's GDN conv alone: `qkv = x·W`, 4 taps, zero history, SiLU.
    #[test]
    fn gdn_conv_alone_keeps_a_ring() {
        let mut g = TensorGraph::new();
        let x = g.input(&[T, fx(4)]);
        let qkv = linear(&mut g, x, 4, 12);
        let o = conv(&mut g, qkv, 12, 4);
        let sizes = ring_sizes(&g, o);
        eprintln!("gdn conv alone: caches {sizes:?}");
        assert_eq!(sizes.len(), 1);
        assert_eq!(sizes[0].0, Some(3));
        for p0 in [1, 3, 7] {
            for ks in CHAINS {
                let e = check(&g, o, p0, ks, Skew::default());
                assert!(e < 1e-12, "{p0} {ks:?}: {e:e}");
            }
        }
    }

    /// NeoX partial rope (`rot` of `hd` dims) with the partner read through a
    /// guarded shift on the head axis: a plain read along `T`.
    fn partial_rope(g: &mut TensorGraph, q: TensorId, hd: usize, rot: usize) -> TensorId {
        let half = rot / 2;
        let ti = g.iota(&[T, fx(hd)], 0);
        let ci = g.iota(&[T, fx(hd)], 1);
        let shift = |o: i64| {
            let mut f = Form::unit(2, 1);
            f.offset = o;
            arg(q, vec![u(2, 0), f], true)
        };
        let (h, r) = (half as f64, rot as f64);
        g.comp(
            &[T, fx(hd)],
            &[],
            vec![
                arg(q, id(2), false),
                shift(half as i64),
                shift(-(half as i64)),
                arg(ti, id(2), false),
                arg(ci, id(2), false),
            ],
            None,
            move |v| {
                let (x, up, dn, t, c) = (v[0], v[1], v[2], v[3], v[4]);
                let lower = lit(h - 0.5) - c; // c < half
                let pair = sel(lower, c, c - lit(h));
                let th = t * (lit(-0.7) * pair).exp();
                let rotd = sel(
                    lower,
                    x * th.cos() - up * th.sin(),
                    x * th.cos() + dn * th.sin(),
                );
                vec![sel(lit(r - 0.5) - c, rotd, x)]
            },
        )
        .unwrap()
    }

    #[test]
    fn partial_rope_guard_on_a_fixed_axis_derives() {
        let mut g = TensorGraph::new();
        let x = g.input(&[T, fx(8)]);
        let q0 = linear(&mut g, x, 8, 8);
        let q = partial_rope(&mut g, q0, 8, 4);
        let k0 = linear(&mut g, x, 8, 8);
        let k = partial_rope(&mut g, k0, 8, 4);
        let s = g
            .comp(
                &[T, T],
                &[fx(8)],
                vec![
                    arg(q, vec![u(3, 0), u(3, 2)], false),
                    arg(k, vec![u(3, 1), u(3, 2)], false),
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        let o = win_reduce(
            &mut g,
            &[T, fx(8)],
            vec![
                arg(s, vec![u(3, 0), u(3, 2)], false),
                arg(x, vec![u(3, 2), u(3, 1)], false),
            ],
            Monoid::sum(),
            0.0,
            0,
            |v| v[0].exp() * v[1],
        );
        let sizes = ring_sizes(&g, o);
        eprintln!("partial rope + causal attention: caches {sizes:?}");
        assert!(sizes.iter().all(|(r, _)| r.is_none()));
        for ks in CHAINS {
            let e = check(&g, o, 3, ks, Skew::default());
            assert!(e < 1e-12, "{ks:?}: {e:e}");
        }
    }
}
