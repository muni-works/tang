//! Reverse-mode autodiff over comprehensions.
//!
//! The gradient of a comprehension is a comprehension. For
//! `y[i] = Σ_k body(x₀[v₀(i,k)], …)` and an arg `a = x_j` read through a
//! pick view `v_j`, the contribution to `∂L/∂a[c]` sums `∂L/∂y[i] ·
//! ∂body/∂x_j` over every `(i,k)` with `v_j(i,k) = c`. Inverting a pick
//! view is renaming coordinates: the iteration coords `v_j` picks become
//! `a`'s out coords, the rest become reduce coords. Every other arg's view
//! is pushed through the same renaming, and the scalar body is
//! differentiated by `tang_expr`'s symbolic `diff`.
//!
//! Supported: `Sum`/`Max` reductions and maps, width 1, pick views for
//! the differentiated arg (any affine view for the others). `Max` routes
//! the gradient to every element equal to the max. Not yet: tuple
//! monoids, non-injective views such as diagonals.
//! Differentiate the *algorithm* graph, then optimize forward and backward
//! together.

use std::collections::{HashMap, HashSet};

use tang_expr::{ExprGraph, ExprId, Node};

use super::{
    err, replay, Arg, Comp, Dim, Form, Monoid, MonoidKind, Op, Segment, ShapeError, TensorGraph,
    TensorId, View,
};

/// Options for [`TensorGraph::backward_with`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackwardOptions {
    /// Keep the gradient terms of stabilizing maxes
    /// ([`stabilizer_maxes`](TensorGraph::stabilizer_maxes)) instead of
    /// dropping them. The terms sum to zero in real arithmetic, so the
    /// default (off) is exact there and cheaper; in floating point,
    /// keeping them reproduces the gradient the explicit `[s == m]`
    /// routing gives, which matters when the softmax saturates (one-hot
    /// rows, true dWq/dWk ≈ 0): dropping them leaves the rounding noise
    /// of `dP − D` (as FlashAttention-2 does) instead of the cancellation.
    pub exact_stabilizers: bool,
}

impl BackwardOptions {
    /// The default, with `TANG_EXACT_STABILIZERS=1` turning
    /// [`exact_stabilizers`](Self::exact_stabilizers) on.
    pub fn from_env() -> Self {
        Self {
            exact_stabilizers: std::env::var("TANG_EXACT_STABILIZERS").is_ok_and(|v| v == "1"),
        }
    }
}

impl TensorGraph {
    /// Gradients of `sum(y)` with respect to each of `wrt`, appended to
    /// this graph. A `wrt` that `y` does not depend on gets a zero map.
    /// Options from the environment ([`BackwardOptions::from_env`]).
    pub fn backward(&mut self, y: TensorId, wrt: &[TensorId]) -> Result<Vec<TensorId>, ShapeError> {
        self.backward_with(y, wrt, BackwardOptions::from_env())
    }

    /// [`backward`](Self::backward) with explicit options.
    pub fn backward_with(
        &mut self,
        y: TensorId,
        wrt: &[TensorId],
        opts: BackwardOptions,
    ) -> Result<Vec<TensorId>, ShapeError> {
        let live = self.live(&[y]);
        let mut grads: HashMap<usize, TensorId> = HashMap::new();
        let ones = {
            let shape = self.shape(y).to_vec();
            self.comp(&shape, &[], vec![], None, |_| vec![ExprId::ONE])?
        };
        grads.insert(y.n(), ones);
        // Stabilizing maxes: `y` provably does not depend on them, so
        // their gradient is exactly zero and is not built at all (unless
        // `opts` asks for the terms anyway).
        let stable = if opts.exact_stabilizers {
            HashSet::new()
        } else {
            self.stabilizer_maxes(y)
        };

        // Reverse order over the original nodes; everything appended by
        // this pass sits after `y` and is never a dependency of it.
        for j in (0..=y.n()).rev() {
            if !live[j] {
                continue;
            }
            let Some(&dy) = grads.get(&j) else { continue };
            if self.nodes[j].detach {
                continue;
            }
            if let Op::Reshape(src) = self.nodes[j].op {
                let shape = self.shape(src).to_vec();
                let g = self.reshape(dy, &shape)?;
                let acc = match grads.get(&src.n()) {
                    Some(&prev) => self.map(&[prev, g], |v| v[0] + v[1])?,
                    None => g,
                };
                grads.insert(src.n(), acc);
                continue;
            }
            if let Op::Scan(sc) = &self.nodes[j].op {
                let inputs = sc.inputs.clone();
                let gs = self.scan_backward(j, dy)?;
                for (src, g) in inputs.iter().zip(gs) {
                    let Some(g) = g else { continue };
                    let acc = match grads.get(&src.n()) {
                        Some(&prev) => self.map(&[prev, g], |v| v[0] + v[1])?,
                        None => g,
                    };
                    grads.insert(src.n(), acc);
                }
                continue;
            }
            if let Op::Operator(o) = &self.nodes[j].op {
                let (op, inputs) = (o.op.clone(), o.inputs.clone());
                let gs = op.backward(self, &inputs, TensorId(j as u32), dy)?;
                if gs.len() != inputs.len() {
                    return err(format!(
                        "operator `{}` returned {} cotangents for {} inputs",
                        op.name(),
                        gs.len(),
                        inputs.len()
                    ));
                }
                for (src, g) in inputs.iter().zip(gs) {
                    let Some(g) = g else { continue };
                    let acc = match grads.get(&src.n()) {
                        Some(&prev) => self.map(&[prev, g], |v| v[0] + v[1])?,
                        None => g,
                    };
                    grads.insert(src.n(), acc);
                }
                continue;
            }
            let Some(c) = self.comp_at(j) else { continue };
            let c = c.clone();
            let generic = c.monoid.as_ref().is_some_and(|m| match m.kind() {
                MonoidKind::OnlineSoftmax => false,
                MonoidKind::Sum | MonoidKind::Max => c.tuple == 1,
                MonoidKind::Other => true,
            });
            let tuple_map = c.monoid.is_none() && c.tuple == 1;
            if generic || tuple_map {
                let gs = if generic {
                    self.monoid_backward(TensorId(j as u32), &c, dy)?
                } else {
                    self.tuple_map_backward(&c, dy)?
                };
                for (a, g) in gs.into_iter().enumerate() {
                    if stable.contains(&c.args[a].src.n()) {
                        continue;
                    }
                    let src = c.args[a].src.n();
                    let acc = match grads.get(&src) {
                        Some(&prev) => self.map(&[prev, g], |v| v[0] + v[1])?,
                        None => g,
                    };
                    grads.insert(src, acc);
                }
                continue;
            }
            for a in 0..c.args.len() {
                if stable.contains(&c.args[a].src.n()) {
                    continue;
                }
                let g = self.arg_grad(TensorId(j as u32), &c, a, dy)?;
                let src = c.args[a].src.n();
                let acc = match grads.get(&src) {
                    Some(&prev) => self.map(&[prev, g], |v| v[0] + v[1])?,
                    None => g,
                };
                grads.insert(src, acc);
            }
        }

        wrt.iter()
            .map(|t| match grads.get(&t.n()) {
                Some(&g) => Ok(g),
                None => {
                    let shape = self.shape(*t).to_vec();
                    self.comp(&shape, &[], vec![], None, |_| vec![ExprId::ZERO])
                }
            })
            .collect()
    }

    /// Max reductions `m` that `y` does not depend on: every use of `m`
    /// is a shift `exp(x − m)` whose results reach `y` only through
    /// expressions homogeneous of degree 0 in `t = exp(−m)`, the
    /// normalized ratios of a softmax. The argument is exact, not
    /// numerical: each node downstream of `m` is classified as `t^k·h`
    /// (`Hom(k)`, `h` independent of `m`) or `a + κ·m` (`Lin(κ)`), with
    /// the row of `m` its factor comes from tracked as a view, and the
    /// rules only use identities (`t^a·t^b = t^(a+b)`, `Σ_j t·h_j =
    /// t·Σ_j h_j` when `t`'s row does not depend on `j`, `2^(a+κm) =
    /// 2^a·t^(−κ ln 2)`). A node of degree 0 is then a function of the
    /// other inputs alone, so `∂y/∂m ≡ 0` and every `[s == m]` term of
    /// `m`'s gradient sums to zero identically. Any use the rules do not
    /// cover disqualifies `m`.
    pub fn stabilizer_maxes(&self, y: TensorId) -> HashSet<usize> {
        let live = &self.live(&[y]);
        let mut found = HashSet::new();
        for m in 0..self.nodes.len() {
            if !live[m] || m == y.n() {
                continue;
            }
            let Some(mc) = self.comp_at(m) else { continue };
            let is_max = mc.monoid.as_ref().map(Monoid::kind) == Some(MonoidKind::Max);
            if !is_max || mc.tuple != 0 || mc.width() != 1 {
                continue;
            }
            if self.shift_invariant(m, y, live) {
                found.insert(m);
            }
        }
        found
    }

    fn shift_invariant(&self, m: usize, y: TensorId, live: &[bool]) -> bool {
        let m_rank = self.nodes[m].shape.len();
        // node -> (degree, factor view: node coords -> m coords)
        let mut dep: HashMap<usize, (Deg, View)> = HashMap::new();
        dep.insert(m, (Deg::Lin(1.0), View::identity(m_rank, m_rank)));
        let mut used = false;
        for i in m + 1..self.nodes.len() {
            if !live[i] || !self.deps(i).iter().any(|d| dep.contains_key(d)) {
                continue;
            }
            used = true;
            let Some(c) = self.comp_at(i) else {
                return false; // a reshape of an m-dependent value
            };
            if c.tuple != 0 || c.width() != 1 {
                return false;
            }
            let rank = c.rank();
            let mut factor: Option<View> = None;
            let mut arg_deg = Vec::with_capacity(c.args.len());
            for a in &c.args {
                match dep.get(&a.src.n()) {
                    None => arg_deg.push(Deg::Hom(0.0)),
                    Some((d, f)) => {
                        if (a.view.guard && matches!(d, Deg::Lin(_))) || a.view.is_gather() {
                            return false;
                        }
                        let fv = f.compose(&a.view, rank);
                        match &factor {
                            None => factor = Some(fv),
                            Some(prev) if prev.forms == fv.forms => {}
                            Some(_) => return false,
                        }
                        arg_deg.push(*d);
                    }
                }
            }
            let Some(d) = body_degree(&c.body, c.body_outs[0], &arg_deg) else {
                return false;
            };
            if d.is_clean() {
                continue; // independent of m from here on
            }
            if i == y.n() {
                return false;
            }
            let factor = factor.expect("a dependent body reads a dependent arg");
            let n_pts = c.points().len();
            if !c.red.is_empty() {
                let sum = c.monoid.as_ref().map(Monoid::kind) == Some(MonoidKind::Sum);
                let ok = sum
                    && matches!(d, Deg::Hom(_))
                    && factor
                        .forms
                        .iter()
                        .all(|f| f.coefs[n_pts..].iter().all(|&x| x == 0));
                if !ok {
                    return false;
                }
            }
            let forms = factor
                .forms
                .iter()
                .map(|f| Form {
                    coefs: f.coefs[..n_pts].to_vec(),
                    offset: f.offset,
                })
                .collect();
            dep.insert(
                i,
                (
                    d,
                    View {
                        guard: false,
                        forms,
                        gather: None,
                    },
                ),
            );
        }
        used
    }

    /// Backward of a reduction with an arbitrary (associative) tuple
    /// monoid — the monoids [`discover`](super::discover) emits, top-k,
    /// argmax. For `y = e₀ ⊕ … ⊕ e_{K−1}` along the one reduce dim,
    /// `∂y/∂e_k` is the Jacobian of `P_k ⊕ e ⊕ S_k` in `e`, with `P_k`,
    /// `S_k` the exclusive prefix and suffix folds: two scans of the
    /// same monoid, then an elementwise vjp of the combine (derived by
    /// `tang-expr`), chained through the body by the ordinary
    /// [`arg_grad`](Self::arg_grad) of the map `Σⱼ ēⱼ·bodyⱼ`. Exact for
    /// any associative combine; no structure assumed.
    fn monoid_backward(
        &mut self,
        self_id: TensorId,
        c: &Comp,
        dy: TensorId,
    ) -> Result<Vec<TensorId>, ShapeError> {
        use super::discover::inline;
        let m = c.monoid.clone().expect("a reduction");
        if let Some(sv) = m.saved.clone() {
            if c.tuple == 1 && c.red.len() == 1 && !c.has_gather() && body_is(c, &sv.element) {
                return self.saved_backward(self_id, c, dy, &sv);
            }
        }
        let w = m.width;
        if c.red.len() != 1 || c.has_gather() {
            return err("backward: generic monoids need one reduce dim and affine views");
        }
        let pts = c.points().to_vec();
        let np = pts.len();
        let mut space = pts.clone();
        space.push(c.red[0]);
        let r = np + 1;
        let na = c.args.len();
        // elements, one tensor per slot, over the comp's own space
        let mut elems = vec![];
        for j in 0..w {
            let (body, out) = (c.body.clone(), c.body_outs[j]);
            elems.push(self.comp(&space, &[], c.args.clone(), None, move |v| {
                inline(&body, &[out], v)
            })?);
        }
        let pre = self.scan(&elems, np, m.clone(), true, false)?;
        let suf = self.scan(&elems, np, m.clone(), true, true)?;
        let slot = |j: usize, wide: bool| {
            let mut forms: Vec<Form> = (0..r).map(|d| Form::unit(r, d)).collect();
            if wide {
                forms.push(Form::constant(r, j as i64));
            }
            View {
                forms,
                guard: false,
                gather: None,
            }
        };
        let dyv = |j: usize| {
            let mut forms: Vec<Form> = (0..np).map(|d| Form::unit(r, d)).collect();
            if c.tuple == 1 {
                forms.push(Form::constant(r, j as i64));
            }
            View {
                forms,
                guard: false,
                gather: None,
            }
        };
        // ē = dyᵀ ∂((pre ⊕ e) ⊕ suf)/∂e: vars pre 0..w, e w..2w, suf 2w..3w, seed 3w..4w
        let (vg, vouts) = {
            let (mut g, outs) = tang_expr::trace(|| {
                let v: Vec<ExprId> = (0..3 * w).map(|i| ExprId::var(i as u16)).collect();
                let mut a = v[..2 * w].to_vec();
                let pe = inline(&m.combine, &m.outs, &a);
                a = pe;
                a.extend_from_slice(&v[2 * w..]);
                inline(&m.combine, &m.outs, &a)
            });
            let seeds: Vec<ExprId> = (0..w).map(|i| g.var((3 * w + i) as u16)).collect();
            let gr = g.vjp(&outs, &seeds, 3 * w);
            (g, gr[w..2 * w].to_vec())
        };
        let mut ebar = vec![];
        for jj in 0..w {
            let mut args = c.args.clone();
            for j in 0..w {
                args.push(Arg {
                    src: pre,
                    view: slot(j, w > 1),
                });
            }
            for j in 0..w {
                args.push(Arg {
                    src: suf,
                    view: slot(j, w > 1),
                });
            }
            for j in 0..w {
                args.push(Arg {
                    src: dy,
                    view: dyv(j),
                });
            }
            let (body, bouts) = (c.body.clone(), c.body_outs.clone());
            let (vg, out) = (vg.clone(), vouts[jj]);
            ebar.push(self.comp(&space, &[], args, None, move |v| {
                let e = inline(&body, &bouts, &v[..na]);
                let mut u = v[na..na + w].to_vec();
                u.extend(e);
                u.extend_from_slice(&v[na + w..na + 2 * w]);
                u.extend_from_slice(&v[na + 2 * w..na + 3 * w]);
                inline(&vg, &[out], &u)
            })?);
        }
        // the map Σⱼ ēⱼ·bodyⱼ(args), differentiated in each original arg
        let mut args = c.args.clone();
        for &e in &ebar {
            args.push(Arg {
                src: e,
                view: slot(0, false),
            });
        }
        let (body, bouts) = (c.body.clone(), c.body_outs.clone());
        let (cb, cbo) = tang_expr::trace(|| {
            let v: Vec<ExprId> = (0..na + w).map(|i| ExprId::var(i as u16)).collect();
            let b = inline(&body, &bouts, &v[..na]);
            let mut acc = b[0] * v[na];
            for j in 1..w {
                acc += b[j] * v[na + j];
            }
            vec![acc]
        });
        let synth = Comp {
            out: space.clone(),
            tuple: 0,
            red: vec![],
            args,
            body: cb,
            body_outs: cbo,
            monoid: None,
        };
        let ones = self.comp(&space, &[], vec![], None, |_| vec![ExprId::ONE])?;
        (0..na)
            .map(|a| self.arg_grad(ones, &synth, a, ones))
            .collect()
    }

    /// Elementwise backward of a discovered online lift from its saved
    /// final state (see [`SavedBwd::element_grad`](super::discover::SavedBwd::element_grad)):
    /// one pass reading the elements, the state and its cotangent — the
    /// FlashAttention-backward shape, derived for every lift.
    fn saved_backward(
        &mut self,
        self_id: TensorId,
        c: &Comp,
        dy: TensorId,
        sv: &super::discover::SavedBwd,
    ) -> Result<Vec<TensorId>, ShapeError> {
        use super::discover::inline;
        let w = sv.w1 + sv.r;
        let na = c.args.len();
        let pts = c.points().to_vec();
        let np = pts.len();
        let mut space = pts.clone();
        space.push(c.red[0]);
        let r = np + 1;
        let row = |src: TensorId, j: usize| {
            let mut forms: Vec<Form> = (0..np).map(|d| Form::unit(r, d)).collect();
            forms.push(Form::constant(r, j as i64));
            Arg {
                src,
                view: View {
                    forms,
                    guard: false,
                    gather: None,
                },
            }
        };
        let (eg, eo) = sv.element_grad();
        let mut xbar = vec![];
        for a in 0..na {
            let mut args = c.args.clone();
            args.extend((0..w).map(|j| row(self_id, j)));
            args.extend((0..w).map(|j| row(dy, j)));
            let (eg, out) = (eg.clone(), eo[a]);
            xbar.push(self.comp(&space, &[], args, None, move |v| inline(&eg, &[out], v))?);
        }
        let mut args = c.args.clone();
        for &x in &xbar {
            args.push(Arg {
                src: x,
                view: View::identity(r, r),
            });
        }
        let (cb, cbo) = tang_expr::trace(|| {
            let v: Vec<ExprId> = (0..2 * na).map(|i| ExprId::var(i as u16)).collect();
            let mut acc = v[0] * v[na];
            for a in 1..na {
                acc += v[a] * v[na + a];
            }
            vec![acc]
        });
        let synth = Comp {
            out: space.clone(),
            tuple: 0,
            red: vec![],
            args,
            body: cb,
            body_outs: cbo,
            monoid: None,
        };
        let ones = self.comp(&space, &[], vec![], None, |_| vec![ExprId::ONE])?;
        (0..na)
            .map(|a| self.arg_grad(ones, &synth, a, ones))
            .collect()
    }

    /// Backward of a tuple-valued map (`comp_tuple` without a monoid):
    /// the scalar map `Σⱼ dyⱼ·bodyⱼ(args)` with `dy`'s slots as extra
    /// args, differentiated by [`arg_grad`](Self::arg_grad).
    fn tuple_map_backward(&mut self, c: &Comp, dy: TensorId) -> Result<Vec<TensorId>, ShapeError> {
        use super::discover::inline;
        let w = c.width();
        let pts = c.points().to_vec();
        let r = pts.len();
        let na = c.args.len();
        let mut args = c.args.clone();
        for j in 0..w {
            let mut forms: Vec<Form> = (0..r).map(|d| Form::unit(r, d)).collect();
            forms.push(Form::constant(r, j as i64));
            args.push(Arg {
                src: dy,
                view: View {
                    forms,
                    guard: false,
                    gather: None,
                },
            });
        }
        let (body, bouts) = (c.body.clone(), c.body_outs.clone());
        let (cb, cbo) = tang_expr::trace(|| {
            let v: Vec<ExprId> = (0..na + w).map(|i| ExprId::var(i as u16)).collect();
            let b = inline(&body, &bouts, &v[..na]);
            let mut acc = b[0] * v[na];
            for j in 1..w {
                acc += b[j] * v[na + j];
            }
            vec![acc]
        });
        let synth = Comp {
            out: pts.clone(),
            tuple: 0,
            red: c.red.clone(),
            args,
            body: cb,
            body_outs: cbo,
            monoid: None,
        };
        let ones = self.comp(&pts, &[], vec![], None, |_| vec![ExprId::ONE])?;
        (0..na)
            .map(|a| self.arg_grad(ones, &synth, a, ones))
            .collect()
    }

    /// `∂L/∂(arg a of c)` given `dy = ∂L/∂c`.
    fn arg_grad(
        &mut self,
        self_id: TensorId,
        c: &Comp,
        a: usize,
        dy: TensorId,
    ) -> Result<TensorId, ShapeError> {
        let kind = c.monoid.as_ref().map(Monoid::kind);
        // Online softmax: slots (m, acc₁…accₖ) with accᵢ = Σ_j gᵢ·exp(s_j − m).
        // Differentiated with the forward's m and acc read back, the
        // argmax gated like Max, so the flash-fused graph has a backward.
        let online = c.tuple == 1 && kind == Some(MonoidKind::OnlineSoftmax);
        if (c.tuple != 0 || c.width() != 1) && !online {
            return err("backward: tuple-valued comps are not differentiable yet");
        }
        match kind {
            None
            | Some(MonoidKind::Sum)
            | Some(MonoidKind::Max)
            | Some(MonoidKind::OnlineSoftmax) => {}
            Some(k) => {
                return err(format!(
                    "backward: {k:?} reductions are not differentiable yet"
                ))
            }
        }
        let old_rank = c.rank();
        let space = c.space();
        let arg = &c.args[a];
        let a_shape = self.shape(arg.src).to_vec();

        // A dim of `a` that reads one iteration coord outright is inverted
        // by renaming that coord. Any other form — broadcast, offset,
        // stride, diagonal, sum of coords — becomes a *scatter dim*: its
        // coordinate stays an out dim and the body is gated by
        // `[coord == form(...)]`, with the form's coords reduced over.
        let mut rename: Vec<Option<usize>> = vec![None; old_rank];
        let mut scatter: Vec<usize> = Vec::new();
        // A gathered dim is always a scatter dim: its coordinate is data,
        // so the gate compares against `form + coef · index`, the index
        // read through the renamed index arg (a scatter-add written as a
        // dense gated sum — see docs/tensor-ir.md for what this costs).
        let gathered = |d: usize| arg.view.gather.as_ref().is_some_and(|g| g.coefs[d] != 0);
        for (d, f) in arg.view.forms.iter().enumerate() {
            let zero = f.coefs.iter().all(|&x| x == 0) && f.offset == 0;
            if gathered(d) {
                scatter.push(d);
                continue;
            }
            match f.as_unit() {
                Some(coord) if rename[coord].is_none() => rename[coord] = Some(d),
                // size-1 dim read at 0: nothing to gate
                _ if zero && a_shape[d] == Dim::Fixed(1) => {}
                _ => scatter.push(d),
            }
        }
        // A guarded read's `t + 3k − 9` is inverted by *eliminating* an
        // out coord with coefficient ±1: `t = τ − 3k + 9`. The coord may
        // then leave its range, so every view that reads it is guarded —
        // dy among them, which zeroes the out-of-range terms. Contributions
        // the forward dropped as padding never reach `a`.
        let n_points = c.points().len();
        let mut elim: Vec<Option<usize>> = vec![None; old_rank];
        if arg.view.guard {
            scatter.retain(|&d| {
                if gathered(d) {
                    return true;
                }
                let f = &arg.view.forms[d];
                let pick = (0..n_points).find(|&x| {
                    f.coefs[x].abs() == 1
                        && rename[x].is_none()
                        && elim[x].is_none()
                        && arg
                            .view
                            .forms
                            .iter()
                            .enumerate()
                            .all(|(e, g)| e == d || g.coefs[x] == 0)
                        && arg.view.forms[d]
                            .coefs
                            .iter()
                            .enumerate()
                            .all(|(y, &cy)| y == x || cy == 0 || elim[y].is_none())
                });
                match pick {
                    Some(x) => {
                        elim[x] = Some(d);
                        false
                    }
                    None => true,
                }
            });
        }
        let free: Vec<usize> = (0..old_rank)
            .filter(|&x| rename[x].is_none() && elim[x].is_none())
            .collect();
        for (i, &x) in free.iter().enumerate() {
            rename[x] = Some(a_shape.len() + i);
        }
        let new_rank = a_shape.len() + free.len();
        let red: Vec<Dim> = free.iter().map(|&x| space[x]).collect();
        // old coord -> affine form over the new space
        let mut subst: Vec<Form> = (0..old_rank)
            .map(|x| rename[x].map_or_else(|| Form::zero(new_rank), |n| Form::unit(new_rank, n)))
            .collect();
        for x in 0..old_rank {
            let Some(d) = elim[x] else { continue };
            let f = &arg.view.forms[d];
            let sgn = f.coefs[x]; // ±1
            let mut e = Form::unit(new_rank, d);
            e.offset -= f.offset;
            for (y, &cy) in f.coefs.iter().enumerate() {
                if y != x && cy != 0 {
                    for (o, i) in e.coefs.iter_mut().zip(&subst[y].coefs) {
                        *o -= cy * i;
                    }
                    e.offset -= cy * subst[y].offset;
                }
            }
            for v in e.coefs.iter_mut() {
                *v *= sgn;
            }
            e.offset *= sgn;
            subst[x] = e;
        }
        // A guarded arg's coords may also be renamed onto a wider range
        // (a source longer than the space reads), so guard every read.
        let guard_all = arg.view.guard;
        let subst_view = View {
            guard: false,
            forms: subst.clone(),
            gather: None,
        };
        let base = if online { 2 * c.width() } else { 1 };
        let rename_view = |v: &View| {
            let mut r = v.compose(&subst_view, new_rank);
            r.guard = v.guard || guard_all;
            // the index arg moves with every other arg, `base` slots on
            if let Some(g) = &mut r.gather {
                g.arg += base;
            }
            r
        };
        // dy is indexed by the old out coords (plus a slot for tuple comps)
        let w = c.width();
        let dy_view = |slot: Option<i64>| View {
            guard: guard_all,
            gather: None,
            forms: (0..c.points().len())
                .map(|x| subst[x].clone())
                .chain(slot.map(|s| Form::constant(new_rank, s)))
                .collect(),
        };
        let mut args: Vec<Arg> = Vec::new();
        if online {
            for s in 0..w {
                args.push(Arg {
                    src: dy,
                    view: dy_view(Some(s as i64)),
                });
            }
            for s in 0..w {
                args.push(Arg {
                    src: self_id,
                    view: dy_view(Some(s as i64)),
                });
            }
        } else {
            args.push(Arg {
                src: dy,
                view: dy_view(None),
            });
        }
        debug_assert_eq!(base, args.len());
        args.extend(c.args.iter().map(|x| Arg {
            src: x.src,
            view: rename_view(&x.view),
        }));

        let mut src = c.body.clone();
        let mut body = ExprGraph::new();
        let vars: Vec<ExprId> = (0..base + c.args.len())
            .map(|i| body.var(i as u16))
            .collect();
        let arg_vars = &vars[base..];
        let mut out;
        if online {
            // s, ∂s/∂a, e = exp(s − m), [s == m]
            let ds = src.diff(c.body_outs[0], a as u16);
            let ds = src.simplify(ds);
            let r = replay(&src, &[c.body_outs[0], ds], arg_vars, &mut body);
            let (s, ds) = (r[0], r[1]);
            let m = vars[w];
            let nm = body.neg(m);
            let diff = body.add(s, nm);
            let l2e = body.lit(std::f64::consts::LOG2_E);
            let scaled = body.mul(diff, l2e);
            let e = body.exp2(scaled);
            let ns = body.neg(s);
            let below = body.add(m, ns); // m - s
            let inner = body.select(below, ExprId::ZERO, ExprId::ONE);
            let hit = body.select(diff, ExprId::ZERO, inner); // s == m
            let hit_ds = body.mul(hit, ds);
            // slot 0 (m) behaves like Max
            out = body.mul(vars[0], hit_ds);
            for i in 1..w {
                let dg = src.diff(c.body_outs[i], a as u16);
                let dg = src.simplify(dg);
                let r = replay(&src, &[c.body_outs[i], dg], arg_vars, &mut body);
                let (g, dg) = (r[0], r[1]);
                let g_ds = body.mul(g, ds);
                let inner = body.add(dg, g_ds);
                let direct = body.mul(e, inner);
                let acc = vars[w + i];
                let via_m = body.mul(acc, hit_ds);
                let nvm = body.neg(via_m);
                let d_acc = body.add(direct, nvm);
                let term = body.mul(vars[i], d_acc);
                out = body.add(out, term);
            }
        } else {
            // body' = dy · ∂body/∂x_a
            let d = src.diff(c.body_outs[0], a as u16);
            let d = src.simplify(d);
            let dexpr = replay(&src, &[d], arg_vars, &mut body)[0];
            out = body.mul(vars[0], dexpr);
        }

        // scatter dims: gate by [iota_d == renamed form]
        if !scatter.is_empty() {
            let mut full: Vec<Dim> = a_shape.clone();
            full.extend(red.iter().copied());
            let mut coord_var = |k: usize, body: &mut ExprGraph, args: &mut Vec<Arg>| {
                let io = self.iota(&full, k);
                args.push(Arg {
                    src: io,
                    view: View::identity(new_rank, new_rank),
                });
                body.var((args.len() - 1) as u16)
            };
            for &d in &scatter {
                let f = rename_view(&View {
                    guard: false,
                    forms: vec![arg.view.forms[d].clone()],
                    gather: None,
                })
                .forms
                .remove(0);
                let mut target = body.lit(f.offset as f64);
                if let Some(g) = arg.view.gather.as_ref().filter(|g| g.coefs[d] != 0) {
                    let cl = body.lit(g.coefs[d] as f64);
                    let t = body.mul(cl, vars[base + g.arg]);
                    target = body.add(target, t);
                }
                for (k, &coef) in f.coefs.iter().enumerate() {
                    if coef != 0 {
                        let v = coord_var(k, &mut body, &mut args);
                        let cl = body.lit(coef as f64);
                        let t = body.mul(cl, v);
                        target = body.add(target, t);
                    }
                }
                let here = coord_var(d, &mut body, &mut args);
                let (nt, nh) = (body.neg(target), body.neg(here));
                let above = body.add(here, nt); // here - target
                let below = body.add(target, nh); // target - here
                let inner = body.select(below, ExprId::ZERO, ExprId::ONE);
                let hit = body.select(above, ExprId::ZERO, inner);
                out = body.mul(out, hit);
            }
        }
        if kind == Some(MonoidKind::Max) {
            // Gate to the argmax: read the forward max like dy and keep
            // the term only where body == max.
            let m = body.var(args.len() as u16);
            args.push(Arg {
                src: self_id,
                view: args[0].view.clone(),
            });
            let val = replay(&src, &[c.body_outs[0]], arg_vars, &mut body)[0];
            let (nm, nv) = (body.neg(m), body.neg(val));
            let below = body.add(val, nm); // val - m
            let above = body.add(m, nv); // m - val
            let inner = body.select(above, ExprId::ZERO, ExprId::ONE);
            let hit = body.select(below, ExprId::ZERO, inner);
            out = body.mul(out, hit);
        }

        self.push_comp(Comp {
            out: a_shape,
            tuple: 0,
            red,
            args,
            body,
            body_outs: vec![out],
            monoid: (!free.is_empty()).then(Monoid::sum),
        })
        .inspect(|id| {
            // a dim gathered as `a[key[x]]`, key sorted and x reduced here:
            // only x's run with key == the dim's coord contributes
            let seg = arg.view.gather.as_ref().and_then(|g| {
                let ia = &c.args[g.arg];
                let key = ia.src;
                if !self.nodes[key.n()].sorted || ia.view.forms.len() != 1 || ia.view.is_gather() {
                    return None;
                }
                let x = ia.view.forms[0].as_unit()?;
                let red = free.iter().position(|&f| f == x)?;
                let d = scatter.iter().copied().find(|&d| {
                    let f = &arg.view.forms[d];
                    g.coefs[d] == 1 && f.offset == 0 && f.coefs.iter().all(|&v| v == 0)
                })?;
                Some(Segment { red, key, point: d })
            });
            self.nodes[id.n()].segment = seg;
        })
    }
}

/// How a scalar depends on a stabilizing max `m`, with `t = exp(−m)`:
/// `Hom(k)` is `t^k·h`, `Lin(κ)` is `a + κ·m` (`h`, `a` independent of
/// `m`). `Hom(0)` and `Lin(0)` are both independent of `m`.
#[derive(Clone, Copy, Debug)]
enum Deg {
    Hom(f64),
    Lin(f64),
}

impl Deg {
    fn is_clean(self) -> bool {
        match self {
            Deg::Hom(k) | Deg::Lin(k) => k.abs() < 1e-9,
        }
    }

    /// The coefficient of `m` if this is affine in `m`.
    fn lin(self) -> Option<f64> {
        match self {
            Deg::Lin(k) => Some(k),
            d if d.is_clean() => Some(0.0),
            _ => None,
        }
    }
}

/// Whether `c`'s body is `element(args)` (probed at a few points).
fn body_is(c: &Comp, element: &super::discover::Fun) -> bool {
    let na = c.args.len();
    if element.outs.len() != c.body_outs.len() {
        return false;
    }
    (0..3).all(|t| {
        let v: Vec<f64> = (0..na)
            .map(|i| 0.31 + ((i * 5 + t * 3) % 7) as f64 * 0.23)
            .collect();
        let a = c.body.eval_many::<f64>(&c.body_outs, &v);
        let b = element.call(&v);
        a.iter()
            .zip(&b)
            .all(|(x, y)| (x - y).abs() <= 1e-12 * (1.0 + y.abs()) || (x.is_nan() && y.is_nan()))
    })
}

fn same(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// Degree of `out` given the degrees of the body's vars, or `None` when
/// the rules cannot establish one.
fn body_degree(g: &ExprGraph, out: ExprId, vars: &[Deg]) -> Option<Deg> {
    let nodes = g.nodes_slice();
    let mut memo: Vec<Option<Option<Deg>>> = vec![None; nodes.len()];
    go_degree(nodes, out, vars, &mut memo)
}

fn go_degree(
    nodes: &[Node],
    id: ExprId,
    vars: &[Deg],
    memo: &mut Vec<Option<Option<Deg>>>,
) -> Option<Deg> {
    let k = id.index() as usize;
    if let Some(r) = memo[k] {
        return r;
    }
    let lit = |e: ExprId| nodes[e.index() as usize].as_f64();
    let clean = Deg::Hom(0.0);
    let mut go = |e: ExprId| go_degree(nodes, e, vars, memo);
    let r = (|| match nodes[k] {
        Node::Var(v) => Some(vars.get(v as usize).copied().unwrap_or(clean)),
        Node::Lit(_) => Some(clean),
        Node::Neg(a) => match go(a)? {
            Deg::Lin(x) => Some(Deg::Lin(-x)),
            h => Some(h),
        },
        Node::Add(a, b) => {
            let (da, db) = (go(a)?, go(b)?);
            match (da, db) {
                _ if lit(a) == Some(0.0) => Some(db),
                _ if lit(b) == Some(0.0) => Some(da),
                (Deg::Hom(x), Deg::Hom(y)) if same(x, y) => Some(da),
                _ => Some(Deg::Lin(da.lin()? + db.lin()?)),
            }
        }
        Node::Mul(a, b) => {
            let (da, db) = (go(a)?, go(b)?);
            match (da, db) {
                (Deg::Hom(x), Deg::Hom(y)) => Some(Deg::Hom(x + y)),
                (Deg::Lin(x), _) => Some(Deg::Lin(x * lit(b)?)),
                (_, Deg::Lin(y)) => Some(Deg::Lin(y * lit(a)?)),
            }
        }
        Node::Recip(a) => match go(a)? {
            Deg::Hom(x) => Some(Deg::Hom(-x)),
            d => d.is_clean().then_some(clean),
        },
        Node::Sqrt(a) => match go(a)? {
            Deg::Hom(x) => Some(Deg::Hom(x / 2.0)),
            d => d.is_clean().then_some(clean),
        },
        // 2^(a + κm) = 2^a · e^(κ ln2 · m) = 2^a · t^(−κ ln 2)
        Node::Exp2(a) => match go(a)? {
            Deg::Lin(x) => Some(Deg::Hom(-x * std::f64::consts::LN_2)),
            d => d.is_clean().then_some(clean),
        },
        Node::Sin(a) | Node::Log2(a) => go(a)?.is_clean().then_some(clean),
        Node::Atan2(a, b) => (go(a)?.is_clean() && go(b)?.is_clean()).then_some(clean),
        Node::Select(c, a, b) => {
            if !go(c)?.is_clean() {
                return None;
            }
            let (da, db) = (go(a)?, go(b)?);
            match (da, db) {
                // t^k·0 = 0: a zero arm takes the other's degree
                (_, Deg::Hom(_)) if lit(a) == Some(0.0) => Some(db),
                (Deg::Hom(_), _) if lit(b) == Some(0.0) => Some(da),
                (Deg::Hom(x), Deg::Hom(y)) if same(x, y) => Some(da),
                (Deg::Lin(x), Deg::Lin(y)) if same(x, y) => Some(da),
                _ => (da.is_clean() && db.is_clean()).then_some(clean),
            }
        }
    })()
    .map(|d| if d.is_clean() { clean } else { d });
    memo[k] = Some(r);
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use tang::Scalar;

    /// Finite-difference check of `∂sum(y)/∂x` for input `x_idx`.
    fn check_grad(g: &TensorGraph, y: TensorId, inputs: &[Vec<f64>], x_idx: usize, grad: TensorId) {
        let got = g.eval(inputs, &[grad]).remove(0);
        let h = 1e-6;
        for i in 0..inputs[x_idx].len() {
            let mut plus = inputs.to_vec();
            plus[x_idx][i] += h;
            let mut minus = inputs.to_vec();
            minus[x_idx][i] -= h;
            let f = |inp: &[Vec<f64>]| g.eval(inp, &[y]).remove(0).iter().sum::<f64>();
            let fd = (f(&plus) - f(&minus)) / (2.0 * h);
            assert!(
                (got[i] - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
                "input {x_idx}[{i}]: got {} vs fd {fd}",
                got[i]
            );
        }
    }

    fn ramp(n: usize, scale: f64, off: f64) -> Vec<f64> {
        (0..n)
            .map(|i| ((i * 7919) % 97) as f64 * scale + off)
            .collect()
    }

    #[test]
    fn matmul_grads() {
        let (m, k, n) = (3, 4, 2);
        let mut g = TensorGraph::new();
        let a = g.input(&[m, k]);
        let b = g.input(&[k, n]);
        let c = g.matmul(a, b).unwrap();
        let grads = g.backward(c, &[a, b]).unwrap();
        let inputs = [ramp(m * k, 0.1, -0.5), ramp(k * n, 0.2, 0.1)];
        check_grad(&g, c, &inputs, 0, grads[0]);
        check_grad(&g, c, &inputs, 1, grads[1]);
    }

    #[test]
    fn rmsnorm_grads_through_reduce_and_broadcast() {
        let (rows, dim) = (2, 5);
        let mut g = TensorGraph::new();
        let x = g.input(&[rows, dim]);
        let w = g.input(&[1, dim]);
        let sq = g.map(&[x], |v| v[0] * v[0]).unwrap();
        let sum = g.reduce(sq, 1, Monoid::sum()).unwrap();
        let inv = g
            .map(&[sum], |v| {
                ExprId::from_f64(1.0)
                    / (v[0] / ExprId::from_f64(dim as f64) + ExprId::from_f64(1e-6)).sqrt()
            })
            .unwrap();
        let inv = g.broadcast(inv, &[rows, dim]).unwrap();
        let wb = g.broadcast(w, &[rows, dim]).unwrap();
        let y = g.map(&[x, inv, wb], |v| v[0] * v[1] * v[2]).unwrap();
        let grads = g.backward(y, &[x, w]).unwrap();
        let inputs = [ramp(rows * dim, 0.3, -1.0), ramp(dim, 0.1, 0.5)];
        check_grad(&g, y, &inputs, 0, grads[0]);
        check_grad(&g, y, &inputs, 1, grads[1]);
    }

    #[test]
    fn transpose_and_shared_input() {
        // y = x · xᵀ : both args are the same input through different views
        let n = 3;
        let mut g = TensorGraph::new();
        let x = g.input(&[n, n]);
        let xt = g.transpose(x, &[1, 0]).unwrap();
        let y = g.matmul(x, xt).unwrap();
        let grads = g.backward(y, &[x]).unwrap();
        let inputs = [ramp(n * n, 0.2, -0.3)];
        check_grad(&g, y, &inputs, 0, grads[0]);
    }

    #[test]
    fn backward_graph_optimizes_like_any_other() {
        let (m, k, n) = (4, 3, 5);
        let mut g = TensorGraph::new();
        let a = g.input(&[m, k]);
        let b = g.input(&[k, n]);
        let c = g.matmul(a, b).unwrap();
        let act = g.map(&[c], |v| v[0] * v[0]).unwrap();
        let grads = g.backward(act, &[a, b]).unwrap();
        let inputs = [ramp(m * k, 0.1, -0.5), ramp(k * n, 0.2, 0.1)];
        let want: Vec<Vec<f64>> = g.eval(&inputs, &grads);
        let (fg, outs) = g.substitute_maps(&grads);
        let got = fg.eval(&inputs, &outs);
        for (w, g) in want.iter().zip(&got) {
            for (x, y) in w.iter().zip(g) {
                assert!((x - y).abs() < 1e-9, "{x} vs {y}");
            }
        }
        // ones · 2c reads two buffers, so cost-aware substitution keeps it
        // materialized: forward matmul, the dy map, two gradient reductions.
        assert_eq!(fg.count_comps(&outs), 4);
        let (fg, outs) = g.substitute_maps_with(&grads, false);
        assert_eq!(fg.count_comps(&outs), 3);
    }

    #[test]
    fn grads_through_reshape_and_strided_views() {
        // y[a, b] = sum_k r[a, 2k+1] * w[k, b] with r = reshape(x, [3, 4])
        let mut g = TensorGraph::new();
        let x = g.input(&[2, 6]);
        let w = g.input(&[2, 5]);
        let r = g.reshape(x, &[3, 4]).unwrap();
        let odd = g
            .comp(
                &[3, 2],
                &[],
                vec![Arg {
                    src: r,
                    view: View {
                        gather: None,
                        guard: false,
                        forms: vec![
                            Form::unit(2, 0),
                            Form {
                                coefs: vec![0, 2],
                                offset: 1,
                            },
                        ],
                    },
                }],
                None,
                |v| vec![v[0]],
            )
            .unwrap();
        let y = g.matmul(odd, w).unwrap();
        let grads = g.backward(y, &[x, w]).unwrap();
        let inputs = [ramp(12, 0.3, -1.0), ramp(10, 0.2, 0.1)];
        check_grad(&g, y, &inputs, 0, grads[0]);
        check_grad(&g, y, &inputs, 1, grads[1]);
    }

    #[test]
    fn max_routes_gradient_to_argmax() {
        let mut g = TensorGraph::new();
        let x = g.input(&[2, 3]);
        let m = g.reduce(x, 1, Monoid::max()).unwrap();
        let grads = g.backward(m, &[x]).unwrap();
        let got = g
            .eval(&[vec![1.0, 5.0, 2.0, 7.0, 0.0, 3.0]], &grads)
            .remove(0);
        assert_eq!(got, vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn attention_grads() {
        let (n, d) = (4, 3);
        let mut g = TensorGraph::new();
        let q = g.input(&[n, d]);
        let k = g.input(&[n, d]);
        let v = g.input(&[n, d]);
        let kt = g.transpose(k, &[1, 0]).unwrap();
        let s = g.matmul(q, kt).unwrap();
        let m = g.reduce(s, 1, Monoid::max()).unwrap();
        let mb = g.broadcast(m, &[n, n]).unwrap();
        let p = g.map(&[s, mb], |x| (x[0] - x[1]).exp()).unwrap();
        let l = g.reduce(p, 1, Monoid::sum()).unwrap();
        let o = g.matmul(p, v).unwrap();
        let lb = g.broadcast(l, &[n, d]).unwrap();
        let y = g.map(&[o, lb], |x| x[0] / x[1]).unwrap();
        let grads = g.backward(y, &[q, k, v]).unwrap();
        let inputs = [
            ramp(n * d, 0.3, -0.5),
            ramp(n * d, 0.2, 0.1),
            ramp(n * d, 0.4, -1.0),
        ];
        for (i, &gr) in grads.iter().enumerate() {
            check_grad(&g, y, &inputs, i, gr);
        }
    }

    /// Gradients through the flash-fused graph equal the naive graph's.
    #[test]
    fn flash_attention_grads_match_naive() {
        let (n, d) = (6, 4);
        let build = || {
            let mut g = TensorGraph::new();
            let q = g.input(&[n, d]);
            let k = g.input(&[n, d]);
            let v = g.input(&[n, d]);
            let kt = g.transpose(k, &[1, 0]).unwrap();
            let s = g.matmul(q, kt).unwrap();
            let m = g.reduce(s, 1, Monoid::max()).unwrap();
            let mb = g.broadcast(m, &[n, n]).unwrap();
            let p = g.map(&[s, mb], |x| (x[0] - x[1]).exp()).unwrap();
            let l = g.reduce(p, 1, Monoid::sum()).unwrap();
            let o = g.matmul(p, v).unwrap();
            let lb = g.broadcast(l, &[n, d]).unwrap();
            let y = g.map(&[o, lb], |x| x[0] / x[1]).unwrap();
            (g, q, k, v, y)
        };
        let inputs = [
            ramp(n * d, 0.3, -0.5),
            ramp(n * d, 0.2, 0.1),
            ramp(n * d, 0.4, -1.0),
        ];
        let (mut naive, q, k, v, y) = build();
        let ng = naive.backward(y, &[q, k, v]).unwrap();
        let want = naive.eval(&inputs, &ng);

        let (g, q, k, v, y) = build();
        let (g, outs) = g.substitute_maps(&[y]);
        let (mut g, outs) = g.stabilize_softmax(&outs);
        assert!(g
            .reduction_kinds(&outs)
            .contains(&MonoidKind::OnlineSoftmax));
        // inputs keep their ids through the passes (they come first)
        let fg = g.backward(outs[0], &[q, k, v]).unwrap();
        let got = g.eval(&inputs, &fg);
        for (w, gg) in want.iter().zip(&got) {
            for (x, y) in w.iter().zip(gg) {
                assert!((x - y).abs() <= 1e-9 * (1.0 + y.abs()), "{x} vs {y}");
            }
        }
    }

    /// The attention max is a stabilizer: its gradient is dropped, and
    /// the gradients still match finite differences. An unnormalized
    /// `Σ exp(s − m)` or the max itself as output keeps it.
    #[test]
    fn stabilizing_max_has_no_gradient() {
        let (n, d) = (4, 3);
        let mut g = TensorGraph::new();
        let q = g.input(&[n, d]);
        let k = g.input(&[n, d]);
        let v = g.input(&[n, d]);
        let kt = g.transpose(k, &[1, 0]).unwrap();
        let s = g.matmul(q, kt).unwrap();
        let m = g.reduce(s, 1, Monoid::max()).unwrap();
        let mb = g.broadcast(m, &[n, n]).unwrap();
        let p = g.map(&[s, mb], |x| (x[0] - x[1]).exp()).unwrap();
        let l = g.reduce(p, 1, Monoid::sum()).unwrap();
        let o = g.matmul(p, v).unwrap();
        let lb = g.broadcast(l, &[n, d]).unwrap();
        let y = g.map(&[o, lb], |x| x[0] / x[1]).unwrap();
        let sq = g.map(&[y], |x| x[0] * x[0]).unwrap();
        assert_eq!(g.stabilizer_maxes(sq).len(), 1);
        assert!(g.stabilizer_maxes(l).is_empty());
        assert!(g.stabilizer_maxes(m).is_empty());
        let before = g.nodes().len();
        let grads = g.backward(sq, &[q, k, v]).unwrap();
        // no argmax gate: nothing in the backward reads m's gradient
        assert!(!g.nodes()[before..].iter().any(|t| matches!(&t.op,
            Op::Comp(c) if c.args.iter().any(|a| a.src == m) && c.red.len() == 1 && c.out == g.shape(s))));
        let inputs = [
            ramp(n * d, 0.3, -0.5),
            ramp(n * d, 0.2, 0.1),
            ramp(n * d, 0.4, -1.0),
        ];
        for (i, &gr) in grads.iter().enumerate() {
            check_grad(&g, sq, &inputs, i, gr);
        }
    }

    /// `Σ_k f(i)·g(i, k)` becomes `f(i)·Σ_k g(i, k)`, with the same values.
    #[test]
    fn invariant_factors_leave_sums() {
        let mut g = TensorGraph::new();
        let a = g.input(&[3]);
        let b = g.input(&[3, 4]);
        let ab = g
            .comp(
                &[3],
                &[4],
                vec![
                    Arg {
                        src: a,
                        view: View::pick(2, &[Some(0)]),
                    },
                    Arg {
                        src: b,
                        view: View::identity(2, 2),
                    },
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1] * v[1]],
            )
            .unwrap();
        let inputs = [ramp(3, 0.3, -0.5), ramp(12, 0.2, 0.1)];
        let want = g.eval(&inputs, &[ab]);
        let (h, outs) = g.hoist_invariant_factors(&[ab]);
        assert!(h.comp_at(outs[0].index()).unwrap().monoid.is_none());
        let got = h.eval(&inputs, &outs);
        for (x, y) in want[0].iter().zip(&got[0]) {
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
    }

    #[test]
    fn online_softmax_backward_is_supported() {
        let mut g = TensorGraph::new();
        let x = g.input(&[2, 3]);
        let m = g.reduce(x, 1, Monoid::max()).unwrap();
        let mb = g.broadcast(m, &[2, 3]).unwrap();
        let p = g.map(&[x, mb], |v| (v[0] - v[1]).exp()).unwrap();
        let l = g.reduce(p, 1, Monoid::sum()).unwrap();
        let y = g.map(&[l], |v| v[0] * v[0]).unwrap();
        let (g, outs) = g.substitute_maps(&[y]);
        let (mut g, outs) = g.stabilize_softmax(&outs);
        assert!(g
            .reduction_kinds(&outs)
            .contains(&MonoidKind::OnlineSoftmax));
        assert!(g.backward(outs[0], &[x]).is_ok());
    }

    #[test]
    fn grads_through_guarded_conv() {
        // y[o,t] = Σ_{c,k} w[o,c,k]·x[c, t + 2k − 2], zero padded
        let (ci, co, t, k) = (2, 3, 6, 3);
        let mut g = TensorGraph::new();
        let x = g.input(&[ci, t]);
        let w = g.input(&[co, ci, k]);
        let xv = View {
            gather: None,
            forms: vec![
                Form::unit(4, 2),
                Form {
                    coefs: vec![0, 1, 0, 2],
                    offset: -2,
                },
            ],
            guard: true,
        };
        let y = g
            .comp(
                &[co, t],
                &[ci, k],
                vec![
                    Arg { src: x, view: xv },
                    Arg {
                        src: w,
                        view: View::pick(4, &[Some(0), Some(2), Some(3)]),
                    },
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        let act = g.map(&[y], |v| v[0] * v[0]).unwrap();
        let grads = g.backward(act, &[x, w]).unwrap();
        let inputs = [ramp(ci * t, 0.3, -1.0), ramp(co * ci * k, 0.2, -0.4)];
        check_grad(&g, act, &inputs, 0, grads[0]);
        check_grad(&g, act, &inputs, 1, grads[1]);
        // dx is a gather by renaming, not a scatter over all of t: no iota
        assert!(g.nodes()[grads[0].index()..]
            .iter()
            .all(|n| !matches!(n.op, Op::Iota(_))));
    }

    #[test]
    fn grads_through_guarded_polyphase_upsample() {
        // out[o, q, r] = Σ_{j,c} x[c, q − j] · w[o, c, r + s·j]  (conv_transpose)
        let (ci, co, t, s, taps) = (2, 2, 4, 3, 2);
        let mut g = TensorGraph::new();
        let x = g.input(&[ci, t]);
        let w = g.input(&[co, ci, s * taps]);
        // space: (o, q, r, j, c)
        let xv = View {
            gather: None,
            forms: vec![
                Form::unit(5, 4),
                Form {
                    coefs: vec![0, 1, 0, -1, 0],
                    offset: 0,
                },
            ],
            guard: true,
        };
        let wv = View {
            gather: None,
            forms: vec![
                Form::unit(5, 0),
                Form::unit(5, 4),
                Form {
                    coefs: vec![0, 0, 1, s as i64, 0],
                    offset: 0,
                },
            ],
            guard: false,
        };
        let y = g
            .comp(
                &[co, t, s],
                &[taps, ci],
                vec![Arg { src: x, view: xv }, Arg { src: w, view: wv }],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        let act = g.map(&[y], |v| v[0] * v[0]).unwrap();
        let grads = g.backward(act, &[x, w]).unwrap();
        let inputs = [ramp(ci * t, 0.3, -1.0), ramp(co * ci * s * taps, 0.2, -0.4)];
        check_grad(&g, act, &inputs, 0, grads[0]);
        check_grad(&g, act, &inputs, 1, grads[1]);
        // and the optimized backward agrees
        let want = g.eval(&inputs, &grads);
        let (fg, outs) = g.substitute_maps(&grads);
        let got = fg.eval(&inputs, &outs);
        for (a, b) in want.iter().zip(&got) {
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn guarded_crop_backward_stays_in_range() {
        // y[t] = x[t − 2] for t in 0..4 from an x of length 7: the renamed
        // coord τ ranges over 7 but dy only has 4
        let mut g = TensorGraph::new();
        let x = g.input(&[7]);
        let v = View {
            gather: None,
            forms: vec![Form {
                coefs: vec![1],
                offset: -2,
            }],
            guard: true,
        };
        let y = g
            .comp(&[4], &[], vec![Arg { src: x, view: v }], None, |v| {
                vec![v[0] * v[0]]
            })
            .unwrap();
        let grads = g.backward(y, &[x]).unwrap();
        let inputs = [ramp(7, 0.3, -1.0)];
        check_grad(&g, y, &inputs, 0, grads[0]);
    }

    /// A softmax attention whose rows saturate (one-hot), `sum(y²)`, and
    /// the f64 gradients w.r.t. q, k, v under `backward_fn`.
    fn saturated_attention_grads(
        backward_fn: impl Fn(&mut TensorGraph, TensorId, &[TensorId]) -> Vec<TensorId>,
    ) -> Vec<Vec<f64>> {
        let (n, d) = (6, 4);
        let mut g = TensorGraph::new();
        let q = g.input(&[n, d]);
        let k = g.input(&[n, d]);
        let v = g.input(&[n, d]);
        let kt = g.transpose(k, &[1, 0]).unwrap();
        let s = g.matmul(q, kt).unwrap();
        let m = g.reduce(s, 1, Monoid::max()).unwrap();
        let mb = g.broadcast(m, &[n, n]).unwrap();
        let p = g.map(&[s, mb], |x| (x[0] - x[1]).exp()).unwrap();
        let l = g.reduce(p, 1, Monoid::sum()).unwrap();
        let o = g.matmul(p, v).unwrap();
        let lb = g.broadcast(l, &[n, d]).unwrap();
        let y = g.map(&[o, lb], |x| x[0] / x[1]).unwrap();
        let sq = g.map(&[y], |x| x[0] * x[0]).unwrap();
        let grads = backward_fn(&mut g, sq, &[q, k, v]);
        // logits spread over hundreds: every row's softmax is one-hot
        let inputs = [
            ramp(n * d, 0.9, -40.0),
            ramp(n * d, 0.7, 5.0),
            ramp(n * d, 0.4, -1.0),
        ];
        g.eval(&inputs, &grads)
    }

    fn bits_hash(vs: &[Vec<f64>]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for x in vs.iter().flatten() {
            for b in x.to_bits().to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
            }
        }
        h
    }

    /// Saturated softmax: the f64 gradients of q and k are ~1e-63 (zero
    /// up to the one-hot tails). `exact_stabilizers` rebuilds the
    /// max-gradient terms and gives, bit for bit, what `backward` gave
    /// before stabilizing maxes were dropped (hash recorded on #52's
    /// parent, d1122a2^1). The default is the fast path, unchanged.
    #[test]
    fn exact_stabilizers_restore_the_saturated_gradient() {
        const BEFORE_DROPPING_MAX_GRADIENTS: u64 = 0x8500_844f_f35f_70f8;
        let exact = saturated_attention_grads(|g, y, w| {
            let opts = BackwardOptions {
                exact_stabilizers: true,
            };
            g.backward_with(y, w, opts).unwrap()
        });
        assert_eq!(bits_hash(&exact), BEFORE_DROPPING_MAX_GRADIENTS);
        let fast = saturated_attention_grads(|g, y, w| {
            g.backward_with(y, w, BackwardOptions::default()).unwrap()
        });
        assert_ne!(bits_hash(&fast), BEFORE_DROPPING_MAX_GRADIENTS);
        assert!(!BackwardOptions::default().exact_stabilizers);
        if std::env::var_os("TANG_EXACT_STABILIZERS").is_none() {
            let plain = saturated_attention_grads(|g, y, w| g.backward(y, w).unwrap());
            assert_eq!(bits_hash(&plain), bits_hash(&fast), "default changed");
        }
        eprintln!("fast {:016x} dq {:?}", bits_hash(&fast), &fast[0][..8]);
        // v's gradient does not involve the max either way
        assert_eq!(exact[2], fast[2]);
        for (a, b) in exact.iter().zip(&fast).take(2) {
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() < 1e-9, "{x} vs {y}");
            }
        }
    }

    // ---- implicit differentiation (the KL projection of ribt's glottis) ----

    fn sh(off: i64) -> View {
        let mut f = Form::unit(1, 0);
        f.offset = off;
        View {
            forms: vec![f],
            guard: true,
            gather: None,
        }
    }

    /// `y[m] = Σ_k W[k,m]·x[k]`
    fn to_bands(g: &mut TensorGraph, w: TensorId, x: TensorId, m: usize, k: usize) -> TensorId {
        g.comp(
            &[m],
            &[k],
            vec![
                Arg {
                    src: x,
                    view: View::pick(2, &[Some(1)]),
                },
                Arg {
                    src: w,
                    view: View::pick(2, &[Some(1), Some(0)]),
                },
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap()
    }

    /// `y[k] = Σ_m W[k,m]·z[m]`
    fn to_bins(g: &mut TensorGraph, w: TensorId, z: TensorId, m: usize, k: usize) -> TensorId {
        g.comp(
            &[k],
            &[m],
            vec![
                Arg {
                    src: z,
                    view: View::pick(2, &[Some(1)]),
                },
                Arg {
                    src: w,
                    view: View::pick(2, &[Some(0), Some(1)]),
                },
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap()
    }

    /// Symmetric tridiagonal solve over `[M]` by parallel cyclic reduction
    /// (`up[m]` couples m and m+1), as ribt's `glottis::pcr`.
    fn pcr(
        g: &mut TensorGraph,
        valid: TensorId,
        up: TensorId,
        d: TensorId,
        r: TensorId,
        m: usize,
    ) -> TensorId {
        let id = || View::identity(1, 1);
        let mut lo = g
            .comp(
                &[m],
                &[],
                vec![Arg {
                    src: up,
                    view: sh(-1),
                }],
                None,
                |v| vec![v[0]],
            )
            .unwrap();
        let (mut d, mut up, mut r) = (d, up, r);
        let mut s = 1i64;
        while (s as usize) < m {
            let q = [lo, d, up, r];
            let args = || {
                let mut a: Vec<Arg> = q.iter().map(|&t| Arg { src: t, view: id() }).collect();
                for o in [-s, s] {
                    for &t in &q {
                        a.push(Arg {
                            src: t,
                            view: sh(o),
                        });
                    }
                    a.push(Arg {
                        src: valid,
                        view: sh(o),
                    });
                }
                a
            };
            // v: 0 lo,1 d,2 up,3 r | 4 lo⁻,5 d⁻,6 up⁻,7 r⁻,8 valid⁻ | 9 lo⁺,10 d⁺,11 up⁺,12 r⁺,13 valid⁺
            let k = |v: &[ExprId]| {
                (
                    v[0] / (v[5] + ExprId::ONE - v[8]),
                    v[2] / (v[10] + ExprId::ONE - v[13]),
                )
            };
            let nlo = g
                .comp(&[m], &[], args(), None, |v| vec![-(k(v).0 * v[4])])
                .unwrap();
            let nd = g
                .comp(&[m], &[], args(), None, |v| {
                    let (k1, k2) = k(v);
                    vec![v[1] - k1 * v[6] - k2 * v[9]]
                })
                .unwrap();
            let nup = g
                .comp(&[m], &[], args(), None, |v| vec![-(k(v).1 * v[11])])
                .unwrap();
            let nr = g
                .comp(&[m], &[], args(), None, |v| {
                    let (k1, k2) = k(v);
                    vec![v[3] - k1 * v[7] - k2 * v[12]]
                })
                .unwrap();
            (lo, d, up, r) = (nlo, nd, nup, nr);
            s *= 2;
        }
        g.map(&[r, d], |v| v[0] / v[1]).unwrap()
    }

    /// The KL projection `s = x ⊙ exp(F λ)` with `Fᵀ s = exp(mel)`, by
    /// `newton` Newton steps on the convex dual (Hessian `Fᵀ diag(s) F`,
    /// tridiagonal), as ribt's `glottis::project_keep`: steps clamped to
    /// ±4, exp arguments capped at 80. `implicit`: detach the iteration
    /// and take one more step with a detached Hessian
    /// ([`TensorGraph::detach`]). Returns `s` and the last step `Δλ`.
    /// Inputs: x [K], mel [M], F, F², F·F₊ [K, M], valid [M].
    fn project(
        g: &mut TensorGraph,
        m: usize,
        k: usize,
        newton: usize,
        implicit: bool,
    ) -> (TensorId, TensorId) {
        let x = g.input(&[k]);
        let mel = g.input(&[m]);
        let fb = g.input(&[k, m]);
        let fb2 = g.input(&[k, m]);
        let fbn = g.input(&[k, m]);
        let valid = g.input(&[m]);
        let e = g.map(&[mel], |v| v[0].exp()).unwrap();
        let fx = to_bands(g, fb, x, m, k);
        let lit = ExprId::from_f64;
        let mut lam = g
            .map(&[mel, fx], |v| v[0] - (v[1] + lit(1e-30)).ln())
            .unwrap();
        let s_of = |g: &mut TensorGraph, lam: TensorId| {
            let fl = to_bins(g, fb, lam, m, k);
            g.map(&[x, fl], |v| {
                let z = ExprId::select(v[1] - lit(80.0), lit(80.0), v[1]);
                v[0] * z.exp()
            })
            .unwrap()
        };
        let step = |g: &mut TensorGraph, lam: TensorId, detach_h: bool| {
            let s = s_of(g, lam);
            let fs = to_bands(g, fb, s, m, k);
            let mut d = to_bands(g, fb2, s, m, k);
            let mut up = to_bands(g, fbn, s, m, k);
            if detach_h {
                d = g.detach(d).unwrap();
                up = g.detach(up).unwrap();
            }
            let r = g.map(&[e, fs], |v| v[0] - v[1]).unwrap();
            let dl = pcr(g, valid, up, d, r, m);
            let next = g
                .map(&[lam, dl], |v| {
                    let hi = ExprId::select(v[1] - lit(4.0), lit(4.0), v[1]);
                    v[0] + ExprId::select(lit(-4.0) - v[1], lit(-4.0), hi)
                })
                .unwrap();
            (next, dl)
        };
        let mut dl = lam;
        for _ in 0..newton {
            (lam, dl) = step(g, lam, false);
        }
        if implicit {
            let fixed = g.detach(lam).unwrap();
            (lam, dl) = step(g, fixed, true);
        }
        (s_of(g, lam), dl)
    }

    #[test]
    fn implicit_gradient_of_a_newton_solve() {
        let (m, k, newton) = (8usize, 33usize, 8usize);
        // overlapping triangles: band c peaks at bin 4c + 2, reaches ±4 bins
        let mut fb = vec![0.0; k * m];
        for kk in 0..k {
            for c in 0..m {
                let w = 1.0 - (kk as f64 - (4 * c + 2) as f64).abs() / 4.0;
                fb[kk * m + c] = w.max(0.0);
            }
        }
        let fb2: Vec<f64> = fb.iter().map(|w| w * w).collect();
        let fbn: Vec<f64> = (0..k * m)
            .map(|i| {
                if i % m + 1 < m {
                    fb[i] * fb[i + 1]
                } else {
                    0.0
                }
            })
            .collect();
        let x: Vec<f64> = ramp(k, 0.02, 0.4);
        let mel: Vec<f64> = (0..m).map(|c| 0.3 * ((c * 5) % 7) as f64 - 0.5).collect();
        let ins = vec![x, mel, fb.clone(), fb2, fbn, vec![1.0; m]];

        let build = |implicit: bool, newton: usize| {
            let mut g = TensorGraph::new();
            let (s, dl) = project(&mut g, m, k, newton, implicit);
            // a fixed non-uniform readout, so the gradient is not trivial
            let wts = g.iota(&[k], 0);
            let y = g
                .map(&[s, wts], |v| v[0] * (v[1] * ExprId::from_f64(0.37)).sin())
                .unwrap();
            let (xi, mi) = (TensorId(0), TensorId(1));
            let before = g.nodes().len();
            let grads = g.backward(y, &[xi, mi]).unwrap();
            let added = g.live(&grads).iter().skip(before).filter(|l| **l).count();
            (g, y, s, dl, grads, added)
        };

        // the projection holds: Fᵀ s = exp(mel)
        let (g, y, s, dl, gi, n_imp) = build(true, newton);
        let mut ev = g.eval(&ins, &[s, dl]);
        let (sv, dlv) = (ev.remove(0), ev.remove(0));
        // converged: the last step is far inside the ±4 clamp, so the
        // clamp is the identity there and the implicit step is exact
        let step = dlv.iter().fold(0.0f64, |a, d| a.max(d.abs()));
        assert!(step < 1e-9, "last Newton step {step}");
        for c in 0..m {
            let b: f64 = (0..k).map(|kk| fb[kk * m + c] * sv[kk]).sum();
            assert!((b - ins[1][c].exp()).abs() < 1e-10, "band {c}: {b}");
        }
        // implicit gradient = finite differences of the solve
        check_grad(&g, y, &ins, 0, gi[0]);
        check_grad(&g, y, &ins, 1, gi[1]);
        // ... = the gradient unrolled through every Newton step
        let (gu, _, _, _, gun, n_unr) = build(false, newton);
        let (a, b) = (g.eval(&ins, &gi), gu.eval(&ins, &gun));
        for (p, q) in a.iter().zip(&b) {
            for (u, v) in p.iter().zip(q) {
                assert!((u - v).abs() <= 1e-8 * (1.0 + v.abs()), "{u} vs {v}");
            }
        }
        // and its backward does not grow with the iteration count
        let (.., n_imp2) = build(true, 2 * newton);
        let (.., n_unr2) = build(false, 2 * newton);
        println!("backward nodes: implicit {n_imp} / {n_imp2}, unrolled {n_unr} / {n_unr2}");
        assert_eq!(n_imp, n_imp2);
        assert!(n_unr2 > n_unr && n_unr > 3 * n_imp);
    }

    // ---- detach survives rewrites ----

    /// `y = Σ (detach(2x)·x + c·(2x))`, `c` a fixed weight: `dy/dx =
    /// 2x + 2c`. The detached node and its undetached twin `t` (the same
    /// identity map of `a`) are what a careless pass would merge, splice
    /// or drop.
    fn detach_graph(n: usize) -> (TensorGraph, TensorId) {
        let mut g = TensorGraph::new();
        let x = g.input(&[n]);
        let a = g.map(&[x], |v| v[0] * ExprId::from_f64(2.0)).unwrap();
        let d = g.detach(a).unwrap();
        let t = g.map(&[a], |v| v[0]).unwrap();
        let c = g.iota(&[n], 0);
        let e = g
            .map(&[d, x, c, t], |v| v[0] * v[1] + (v[2] + ExprId::ONE) * v[3])
            .unwrap();
        let y = g.reduce(e, 0, Monoid::sum()).unwrap();
        (g, y)
    }

    fn input_node(g: &TensorGraph, k: usize) -> TensorId {
        let i = g
            .nodes()
            .iter()
            .position(|nd| matches!(nd.op, Op::Input(j) if j == k))
            .unwrap();
        TensorId(i as u32)
    }

    fn assert_detached_grad(g: &mut TensorGraph, y: TensorId, pass: &str) {
        assert!(
            g.nodes().iter().any(|nd| nd.detach),
            "{pass}: detach node dropped"
        );
        let x = input_node(g, 0);
        let gx = g.backward(y, &[x]).unwrap()[0];
        let n = g.shape(x)[0].fixed().unwrap();
        let xs = ramp(n, 0.1, -0.3);
        let got = g.eval(std::slice::from_ref(&xs), &[gx]).remove(0);
        for i in 0..n {
            let want = 2.0 * xs[i] + 2.0 * (i as f64 + 1.0);
            assert!(
                (got[i] - want).abs() < 1e-12,
                "{pass}[{i}]: {} vs {want}",
                got[i]
            );
        }
    }

    #[test]
    fn detach_survives_rewrite_passes() {
        type Pass = fn(&TensorGraph, &[TensorId]) -> (TensorGraph, Vec<TensorId>);
        let passes: [(&str, Pass); 7] = [
            ("substitute_maps", |g, o| g.substitute_maps(o)),
            ("substitute_maps_all", |g, o| {
                g.substitute_maps_with(o, false)
            }),
            ("eliminate_common", |g, o| g.eliminate_common(o)),
            ("eliminate_dead", |g, o| g.eliminate_dead(o)),
            ("merge_siblings", |g, o| g.merge_siblings(o)),
            ("relayout", |g, o| g.relayout(o)),
            ("stabilize_softmax", |g, o| g.stabilize_softmax(o)),
        ];
        let (g0, y0) = detach_graph(6);
        assert_detached_grad(&mut g0.clone(), y0, "none");
        for (name, pass) in passes {
            let (mut g, o) = pass(&g0, &[y0]);
            assert_detached_grad(&mut g, o[0], name);
        }
        // all of them in a row, then the scheduling passes (in place)
        let (mut g, mut o) = (g0.clone(), vec![y0]);
        for (_, pass) in passes {
            (g, o) = pass(&g, &o);
        }
        g.schedule_auto(&o);
        g.schedule_inline(&o);
        assert_detached_grad(&mut g, o[0], "all");
    }

    /// A detached term inside a softmax: `stabilize_softmax` reads
    /// reduction bodies through their maps, and must stop at the detach.
    #[test]
    fn detach_survives_stabilize_softmax() {
        let n = 6;
        let mut g0 = TensorGraph::new();
        let x = g0.input(&[n]);
        let d = g0.detach(x).unwrap();
        let c = g0.iota(&[n], 0);
        let s = g0.map(&[x, d, c], |v| v[0] + v[1] * v[2]).unwrap();
        let m = g0.reduce(s, 0, Monoid::max()).unwrap();
        let p = g0
            .comp(
                &[n],
                &[],
                vec![
                    Arg {
                        src: s,
                        view: View::identity(1, 1),
                    },
                    Arg {
                        src: m,
                        view: View::pick(1, &[None]),
                    },
                ],
                None,
                |v| vec![(v[0] - v[1]).exp()],
            )
            .unwrap();
        let w = g0.map(&[p, c], |v| v[0] * v[1]).unwrap();
        let num = g0.reduce(w, 0, Monoid::sum()).unwrap();
        let den = g0.reduce(p, 0, Monoid::sum()).unwrap();
        let y = g0.map(&[num, den], |v| v[0] / v[1]).unwrap();
        let xs = ramp(n, 0.1, -0.2);
        let want = {
            let mut g = g0.clone();
            let gx = g.backward(y, &[x]).unwrap()[0];
            g.eval(std::slice::from_ref(&xs), &[gx]).remove(0)
        };
        let (g1, o) = g0.substitute_maps(&[y]);
        let (mut g2, o) = g1.stabilize_softmax(&o);
        assert!(g2.nodes().iter().any(|nd| nd.detach));
        // the rewrite fired (max and sums fused into one reduction)
        assert!(g2.nodes().len() < g1.nodes().len());
        let x2 = input_node(&g2, 0);
        let gx = g2.backward(o[0], &[x2]).unwrap()[0];
        let got = g2.eval(&[xs], &[gx]).remove(0);
        for (u, v) in got.iter().zip(&want) {
            assert!((u - v).abs() < 1e-12, "{u} vs {v}");
        }
    }

    /// After `backward`, the detach node is an identity map that schedules
    /// inline: lowered, it adds no buffer and no kernel.
    #[test]
    fn detach_costs_nothing_lowered() {
        let lower = |detach: bool| {
            let mut g = TensorGraph::new();
            let x = g.input(&[64]);
            let a = g.map(&[x], |v| v[0].exp()).unwrap();
            let d = if detach { g.detach(a).unwrap() } else { a };
            let e = g.map(&[d, x], |v| v[0] * v[1]).unwrap();
            let y = g.reduce(e, 0, Monoid::sum()).unwrap();
            let gx = g.backward(y, &[x]).unwrap()[0];
            let (mut g, o) = g.substitute_maps(&[y, gx]);
            let (mut g, o) = {
                let (h, o) = g.eliminate_common(&o);
                g = h;
                (g, o)
            };
            g.schedule_auto(&o);
            let p = g.to_wgsl(&o);
            (p.kernels.len(), p.buffers.len())
        };
        let (k1, b1) = lower(true);
        let (k0, b0) = lower(false);
        assert_eq!(
            (k1, b1),
            (k0, b0),
            "detach: {k1} kernels / {b1} buffers vs {k0} / {b0}"
        );
    }

    /// vjp-of-vjp: a gradient graph that reads a detached node treats it
    /// as a constant again. `y = Σ x²·detach(x)`: `∂y/∂x = 2x·d`, and the
    /// derivative of `Σ w·∂y/∂x` is `2·d·w`, not the `6x·w` of `Σ x³`.
    #[test]
    fn detach_second_order_is_stop_gradient() {
        let n = 5;
        let mut g = TensorGraph::new();
        let x = g.input(&[n]);
        let d = g.detach(x).unwrap();
        let e = g.map(&[x, d], |v| v[0] * v[0] * v[1]).unwrap();
        let y = g.reduce(e, 0, Monoid::sum()).unwrap();
        let gx = g.backward(y, &[x]).unwrap()[0];
        let w = g.iota(&[n], 0);
        let z = g.map(&[gx, w], |v| v[0] * v[1]).unwrap();
        let zs = g.reduce(z, 0, Monoid::sum()).unwrap();
        let hx = g.backward(zs, &[x]).unwrap()[0];
        let xs = ramp(n, 0.2, 0.1);
        let mut ev = g.eval(std::slice::from_ref(&xs), &[gx, hx]);
        let (gv, hv) = (ev.remove(0), ev.remove(0));
        for i in 0..n {
            assert!((gv[i] - 2.0 * xs[i] * xs[i]).abs() < 1e-12);
            assert!(
                (hv[i] - 2.0 * xs[i] * i as f64).abs() < 1e-12,
                "{} vs {}",
                hv[i],
                2.0 * xs[i] * i as f64
            );
        }
    }
}
