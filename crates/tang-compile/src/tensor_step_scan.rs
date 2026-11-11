//! Carried states of [`derive_step`](super::derive_step): the cache of a
//! scan along the sequence is its state at `p0 − 1`, and the cache of a
//! reduction over the sequence into no sequence axis is its running
//! monoid state.
//!
//! - **Monoid scan** (`ScanKind::Monoid`): the new rows' scan, each row
//!   combined with the cached state; the new state is the last row.
//! - **Affine scan** (`ScanKind::Affine`, real, dense): `S_t = A_t·S_{t−1} +
//!   U_t` from the cached `S_{p0−1}`, by folding it into the first new row
//!   (`U'_0 = A_0·S + U_0`) and scanning the new rows from zero. Any `A`:
//!   linear attention, Mamba-2, the (gated) delta rule, a per-channel gate
//!   written as an explicit `A` (GLM's KDA). The step's scan has no
//!   [`DeltaForm`](super::super::DeltaForm) (`U'_0` is not `β·k·vᵀ`).
//! - **Running reduction**: `total = cached ⊕ reduce(new rows)` (see
//!   `Deriver::comp`).
//!
//! Exclusive, reverse, log/max-plus and banded scans are refused.

use tang::Scalar;
use tang_expr::ExprId;

use super::super::{
    Arg, Comp, Dim, Form, Monoid, Scan, ScanKind, Semiring, ShapeError, TensorId, View,
};
use super::{err, CacheSlot, Deriver, Mode, Modes, StepInput};

impl Deriver<'_> {
    /// `out[p…] = m.combine(acc[…], el[p…])` over `shape` (a trailing tuple
    /// dim when the monoid is wider than 1): `acc_picks` maps `acc`'s dims
    /// (without the tuple dim) to point coordinates.
    pub(super) fn combine(
        &mut self,
        m: &Monoid,
        shape: &[Dim],
        acc: TensorId,
        acc_picks: &[Option<usize>],
        el: TensorId,
    ) -> Result<TensorId, ShapeError> {
        let w = m.width;
        let tuple = usize::from(w > 1);
        let pts = shape.len() - tuple;
        let read = |src: TensorId, picks: &[Option<usize>], slot: usize| {
            let mut v = View::pick(pts, picks);
            if w > 1 {
                v.forms.push(Form::constant(pts, slot as i64));
            }
            Arg { src, view: v }
        };
        let el_picks: Vec<Option<usize>> = (0..pts).map(Some).collect();
        let mut args: Vec<Arg> = (0..w).map(|s| read(acc, acc_picks, s)).collect();
        args.extend((0..w).map(|s| read(el, &el_picks, s)));
        self.sg.push_comp(Comp {
            out: shape.to_vec(),
            tuple,
            red: Vec::new(),
            args,
            body: m.combine.clone(),
            body_outs: m.outs.clone(),
            monoid: None,
        })
    }

    pub(super) fn state_slot(&mut self, n: usize, axis: usize, input: TensorId, updated: TensorId) {
        let slot = self.caches.len();
        self.caches.push(CacheSlot {
            node: TensorId(n as u32),
            axis,
            input,
            updated,
            state: true,
        });
        self.cache_of.insert(n, slot);
    }

    /// A scan along the sequence: the scan of the new rows, started from
    /// the cached state at `p0 − 1`; the updated state is its last row.
    pub(super) fn scan(
        &mut self,
        n: usize,
        sc: &Scan,
        modes: &Modes,
    ) -> Result<TensorId, ShapeError> {
        let shape = self.g.nodes[n].shape.clone();
        let tax = self.t_axes(n);
        if tax.is_empty() {
            // no sequence: the same scan over the same (sequence-free) inputs
            let ins = sc
                .inputs
                .iter()
                .map(|t| self.get(t.n(), vec![None; self.g.nodes[t.n()].shape.len()]))
                .collect::<Result<Vec<_>, _>>()?;
            return match &sc.kind {
                ScanKind::Monoid(m) => {
                    self.sg
                        .scan(&ins, sc.dim, m.clone(), sc.exclusive, sc.reverse)
                }
                ScanKind::Affine {
                    form: None,
                    ring: Semiring::Real,
                    bands: None,
                } => self
                    .sg
                    .linear_scan(ins[0], ins[1], sc.dim, sc.exclusive, sc.reverse),
                _ => err("step: this scan kind is not derived yet"),
            };
        }
        if tax != [sc.dim] {
            return err(format!(
                "step: scan node {n} runs along dim {} but its sequence axes are {tax:?}",
                sc.dim
            ));
        }
        if sc.exclusive || sc.reverse {
            return err(format!(
                "step: scan node {n} is exclusive or reversed (a reverse scan reads the future)"
            ));
        }
        match modes[sc.dim] {
            Some(Mode::New) => {}
            Some(Mode::Last) => {
                let new = self.get(n, self.modes_with(n, Mode::New))?;
                return self.last_row(new, sc.dim, true);
            }
            _ => {
                return err(format!(
                    "step: scan node {n} is read at old rows (cache the state, not the rows)"
                ))
            }
        }
        let dim = sc.dim;
        let k = self.spec.new;
        let step_shape = self.step_shape(&shape, modes);
        // state: the output's shape without the scan dim
        let mut st_shape = step_shape.clone();
        st_shape.remove(dim);
        let new_in = |d: &mut Self, t: TensorId| {
            let m = d.modes_with(t.n(), Mode::New);
            d.get(t.n(), m)
        };
        match &sc.kind {
            ScanKind::Monoid(m) => {
                let ins = sc
                    .inputs
                    .iter()
                    .map(|&t| new_in(self, t))
                    .collect::<Result<Vec<_>, _>>()?;
                let part = self.sg.scan(&ins, dim, m.clone(), false, false)?;
                let input = self.sg.input(&st_shape);
                self.inputs.push(StepInput::Cache(self.caches.len()));
                let pts = step_shape.len() - usize::from(m.width > 1);
                let acc_picks: Vec<Option<usize>> =
                    (0..pts).filter(|&d| d != dim).map(Some).collect();
                let out = self.combine(m, &step_shape, input, &acc_picks, part)?;
                let updated = self.last_row(out, dim, false)?;
                self.state_slot(n, dim, input, updated);
                Ok(out)
            }
            ScanKind::Affine {
                ring: Semiring::Real,
                bands: None,
                ..
            } => {
                // S_t = A_t·S_{t−1} + U_t from S_{p0−1}: fold the cached
                // state into the first new row, U'_0 = A_0·S + U_0
                let a = new_in(self, sc.inputs[0])?;
                let u = new_in(self, sc.inputs[1])?;
                let input = self.sg.input(&st_shape);
                self.inputs.push(StepInput::Cache(self.caches.len()));
                let r = step_shape.len(); // [P…, n, m]
                let nn = step_shape[r - 2];
                // AS[P…, i, c] = Σ_j A[P…, i, j]·S[P…\dim, j, c]: space (P…, i, c, j)
                let mut a_picks: Vec<Option<usize>> = (0..r - 1).map(Some).collect();
                a_picks.push(Some(r));
                let mut s_picks: Vec<Option<usize>> =
                    (0..r - 2).filter(|&d| d != dim).map(Some).collect();
                s_picks.extend([Some(r), Some(r - 1)]);
                let as_ = self.sg.comp(
                    &step_shape,
                    &[nn],
                    vec![
                        Arg {
                            src: a,
                            view: View::pick(r + 1, &a_picks),
                        },
                        Arg {
                            src: input,
                            view: View::pick(r + 1, &s_picks),
                        },
                    ],
                    Some(Monoid::sum()),
                    |v| vec![v[0] * v[1]],
                )?;
                let t = self.sg.iota(&step_shape, dim);
                let u0 = self.sg.comp(
                    &step_shape,
                    &[],
                    vec![
                        Arg {
                            src: as_,
                            view: View::identity(r, r),
                        },
                        Arg {
                            src: u,
                            view: View::identity(r, r),
                        },
                        Arg {
                            src: t,
                            view: View::identity(r, r),
                        },
                    ],
                    None,
                    |v| {
                        let first = ExprId::from_f64(0.5) - v[2]; // t' == 0
                        let folded = v[0] + v[1];
                        vec![tang_expr::with_graph(|eg| eg.select(first, folded, v[1]))]
                    },
                )?;
                // (the chunked schedule's DeltaForm no longer describes U')
                let out = self.sg.linear_scan(a, u0, dim, false, false)?;
                let updated = self.last_row(out, dim, false)?;
                self.state_slot(n, dim, input, updated);
                let _ = k;
                Ok(out)
            }
            _ => err(format!(
                "step: scan node {n} is a log/max-plus or banded scan (not derived yet)"
            )),
        }
    }
}
