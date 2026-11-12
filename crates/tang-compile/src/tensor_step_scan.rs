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
//!   written as an explicit `A` (GLM's KDA). With a
//!   [`DeltaForm`](super::super::DeltaForm) the fold skips the dense
//!   `O(n²·m)` product: `A·S = α·(S − β·k·(kᵀS))`, `O(n·m)`. The step's
//!   own scan (k > 1) has no `DeltaForm` (`U'_0` is not `β·k·vᵀ`).
//! - **Running reduction**: `total = cached ⊕ reduce(new rows)` (see
//!   `Deriver::comp`).
//!
//! Exclusive, reverse, log/max-plus and banded scans are refused.

use tang::Scalar;
use tang_expr::ExprId;

use super::super::{
    Arg, Comp, DeltaForm, Dim, Form, Monoid, Scan, ScanKind, Semiring, ShapeError, TensorId, View,
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
            ring: None,
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
                // [P…, n, m]
                let r = step_shape.len();
                // AS[P…, i, c] = Σ_j A[P…, i, j]·S[P…\dim, j, c]: space (P…,
                // i, c, j), O(n³) a step; a delta-rule A = α·(I − β·k·kᵀ)
                // factors to O(n²) (see `delta_as`)
                let factored = match &sc.kind {
                    ScanKind::Affine { form: Some(f), .. } => {
                        Some(self.delta_as(sc, f, input, &step_shape, dim)?)
                    }
                    _ => None,
                };
                let as_ = match factored {
                    Some(t) => t,
                    None => self.dense_as(a, input, &step_shape, dim)?,
                };
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
                // one new row: its state is U'_0 itself (the scan of one
                // element from zero), so no scan kernel
                let out = match &sc.kind {
                    _ if k == 1 => u0,
                    // the form, with the cached state as the start, rides
                    // along for the fused CUDA lowering (`ScanKind::Delta`)
                    ScanKind::Affine { form: Some(f), .. } => {
                        let mut ins = vec![a, u0];
                        let mut slot = |d: &mut Self, t: TensorId| -> Result<usize, ShapeError> {
                            ins.push(new_in(d, t)?);
                            Ok(ins.len() - 1)
                        };
                        let fk = slot(self, sc.inputs[f.k])?;
                        let fv = slot(self, sc.inputs[f.v])?;
                        let fa = f.alpha.map(|i| slot(self, sc.inputs[i])).transpose()?;
                        let fb = f.beta.map(|i| slot(self, sc.inputs[i])).transpose()?;
                        ins.push(input);
                        let init = Some(ins.len() - 1);
                        let form = DeltaForm {
                            alpha: fa,
                            beta: fb,
                            k: fk,
                            v: fv,
                            delta: f.delta,
                            init,
                        };
                        self.sg.affine_scan(
                            ins,
                            dim,
                            false,
                            false,
                            Some(form),
                            Semiring::Real,
                            None,
                        )?
                    }
                    _ => self.sg.linear_scan(a, u0, dim, false, false)?,
                };
                let updated = self.last_row(out, dim, false)?;
                self.state_slot(n, dim, input, updated);
                Ok(out)
            }
            _ => err(format!(
                "step: scan node {n} is a log/max-plus or banded scan (not derived yet)"
            )),
        }
    }

    /// `AS[P…, i, c] = Σ_j A[P…, i, j]·S[P…\dim, j, c]` for a dense `A`.
    fn dense_as(
        &mut self,
        a: TensorId,
        input: TensorId,
        step_shape: &[Dim],
        dim: usize,
    ) -> Result<TensorId, ShapeError> {
        let r = step_shape.len();
        let nn = step_shape[r - 2];
        let mut a_picks: Vec<Option<usize>> = (0..r - 1).map(Some).collect();
        a_picks.push(Some(r));
        let mut s_picks: Vec<Option<usize>> = (0..r - 2).filter(|&d| d != dim).map(Some).collect();
        s_picks.extend([Some(r), Some(r - 1)]);
        self.sg.comp(
            step_shape,
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
        )
    }

    /// `AS` of a delta-rule scan, `A = α·(I − β·k·kᵀ)` (the [`DeltaForm`]'s
    /// gates and keys), without forming `A`: `AS[P…, i, c] = α·(S[i, c] −
    /// β·kᵢ·(kᵀS)[c])`, one `O(n·m)` reduction `(kᵀS)[P…, c] = Σ_j k_j·S[j,
    /// c]` and a map, where the dense product costs `O(n²·m)`.
    fn delta_as(
        &mut self,
        sc: &Scan,
        f: &DeltaForm,
        input: TensorId,
        step_shape: &[Dim],
        dim: usize,
    ) -> Result<TensorId, ShapeError> {
        let r = step_shape.len();
        let p = r - 2;
        let new_in = |d: &mut Self, t: TensorId| {
            let m = d.modes_with(t.n(), Mode::New);
            d.get(t.n(), m)
        };
        let k = new_in(self, sc.inputs[f.k])?;
        let alpha = f.alpha.map(|i| new_in(self, sc.inputs[i])).transpose()?;
        let beta = f.beta.map(|i| new_in(self, sc.inputs[i])).transpose()?;
        let pts: Vec<Option<usize>> = (0..p).map(Some).collect();
        let with = |extra: &[Option<usize>]| {
            let mut v = pts.clone();
            v.extend_from_slice(extra);
            v
        };
        // the state has no `dim`
        let s_pts: Vec<Option<usize>> = (0..p).filter(|&d| d != dim).map(Some).collect();
        let s_with = |extra: &[Option<usize>]| {
            let mut v = s_pts.clone();
            v.extend_from_slice(extra);
            v
        };
        let mut args = vec![Arg {
            src: input,
            view: View::pick(r, &s_with(&[Some(p), Some(p + 1)])),
        }];
        if f.delta {
            // (kᵀS)[P…, c] = Σ_j k[P…, j]·S[j, c]: space (P…, c, j)
            let mut ks_shape = step_shape[..p].to_vec();
            ks_shape.push(step_shape[p + 1]);
            let ks = self.sg.comp(
                &ks_shape,
                &[step_shape[p]],
                vec![
                    Arg {
                        src: k,
                        view: View::pick(p + 2, &with(&[Some(p + 1)])),
                    },
                    Arg {
                        src: input,
                        view: View::pick(p + 2, &s_with(&[Some(p + 1), Some(p)])),
                    },
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )?;
            args.push(Arg {
                src: k,
                view: View::pick(r, &with(&[Some(p)])),
            });
            args.push(Arg {
                src: ks,
                view: View::pick(r, &with(&[Some(p + 1)])),
            });
        }
        let ia = alpha.map(|t| {
            args.push(Arg {
                src: t,
                view: View::pick(r, &pts),
            });
            args.len() - 1
        });
        let ib = beta.map(|t| {
            args.push(Arg {
                src: t,
                view: View::pick(r, &pts),
            });
            args.len() - 1
        });
        let delta = f.delta;
        self.sg.comp(step_shape, &[], args, None, |x| {
            let mut v = x[0];
            if delta {
                let kk = x[1] * x[2];
                v -= ib.map_or(kk, |b| x[b] * kk);
            }
            vec![ia.map_or(v, |a| x[a] * v)]
        })
    }
}
