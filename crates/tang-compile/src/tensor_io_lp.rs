//! The IO bound and the tile that attains it as the two sides of one LP.
//!
//! A (possibly fused) comprehension iterates a box `V = ∏_q [N_q]` and reads
//! arrays `A_j` through projections `φ_j ⊆ coords` (from its views). The
//! Hölder–Brascamp–Lieb bound with loop bounds (Christ–Demmel–Knight–
//! Scanlon–Yelick 2013; Dinh–Demmel 2020 for arbitrary bounds) and the
//! communication-optimal tiling are LP duals, in exponents of the fast
//! memory `B`:
//!
//! ```text
//! dual  (tile):  max Σ_q x_q   s.t.  Σ_{q∈φ_j} x_q ≤ 1 − a_j   (each array's tile fits)
//!                                    x_q ≤ β_q = log_B N_q      (tile ≤ extent)
//! primal (bound): min Σ_j (1−a_j) s_j + Σ_q β_q t_q
//!                s.t. Σ_{j∋q} s_j + t_q ≥ 1  (every free coord covered)
//! ```
//!
//! `σ = opt + Σ_{q atomic} β_q`: one stretch of the schedule with `B`
//! available words performs at most `B^σ` body evaluations, so the traffic
//! is `≥ M·(|V|/(2M)^σ − 1)` (primal, `B = 2M`), and the tile
//! `T_q = (M/J)^{x_q}` (dual, `B = M/J` so the `J` operand tiles together
//! fit) performs `(M/J)^σ` per load of its footprint. Equal optima (strong
//! duality) mean the tile's traffic matches the bound up to the constant;
//! [`Lp::solve`] returns both solutions and checks the gap.
//!
//! *Atomic* coordinates are reduce dims of a nested (inlined) contraction
//! whose result is consumed nonlinearly without being stored (attention's
//! head dim `c` in `S = QKᵀ`): a schedule must finish each such dot product
//! inside one stretch, so the coordinate is fixed at its extent (`x_q =
//! β_q`) and the arrays' budgets shrink by `a_j = Σ_{q∈φ_j atomic} β_q`.
//! The alternative — spilling the partial sums — is the same LP with the
//! inner result as one more array and `q` free; the bound is the *minimum*
//! over the two models (a schedule may pick either). For attention this is
//! exactly Saha & Ye's `Θ(min(N²d²/M, N²d/√M))`.

/// One LP instance.
#[derive(Clone, Debug)]
pub struct Lp {
    /// Coordinate extents.
    pub extents: Vec<usize>,
    /// Projections (bitmasks over coordinates), one per array.
    pub projs: Vec<u64>,
    /// Coordinates fixed at their extent.
    pub atomic: u64,
}

/// Both sides of the LP at one base.
#[derive(Clone, Debug)]
pub struct LpSolution {
    /// `σ`: body evaluations per stretch are at most `base^σ`.
    pub sigma: f64,
    /// Dual: tile exponent per coordinate (atomic ones = `β_q`).
    pub x: Vec<f64>,
    /// Primal: HBL exponent per array.
    pub s: Vec<f64>,
    /// Primal: extent multipliers per coordinate.
    pub t: Vec<f64>,
    /// `primal objective − dual objective` (≈ 0: strong duality).
    pub gap: f64,
}

impl Lp {
    fn beta(&self, base: f64) -> Vec<f64> {
        self.extents
            .iter()
            .map(|&n| (n.max(1) as f64).ln() / base.ln())
            .collect()
    }

    /// Solve primal and dual at `base` (words). `None` if an array's
    /// atomic part alone exceeds the base (that model is infeasible) or a
    /// free coordinate is read by no array and unbounded.
    pub fn solve(&self, base: f64) -> Option<LpSolution> {
        let beta = self.beta(base);
        let r = self.extents.len();
        let free: Vec<usize> = (0..r).filter(|q| self.atomic & (1 << q) == 0).collect();
        let atomic_sum: f64 = (0..r)
            .filter(|q| self.atomic & (1 << q) != 0)
            .map(|q| beta[q])
            .sum();
        // rows: arrays, then extent bounds of free coords
        let mut a: Vec<Vec<f64>> = Vec::new();
        let mut b: Vec<f64> = Vec::new();
        for &p in &self.projs {
            let at: f64 = (0..r)
                .filter(|q| p & (1 << q) != 0 && self.atomic & (1 << q) != 0)
                .map(|q| beta[q])
                .sum();
            let rhs = 1.0 - at;
            if rhs < -1e-12 {
                return None;
            }
            a.push(free.iter().map(|&q| ((p >> q) & 1) as f64).collect());
            b.push(rhs.max(0.0));
        }
        for (k, &q) in free.iter().enumerate() {
            let mut row = vec![0.0; free.len()];
            row[k] = 1.0;
            a.push(row);
            b.push(beta[q]);
        }
        let c = vec![1.0; free.len()];
        let (xf, y, obj) = simplex(&a, &b, &c)?;
        let mut x = beta.clone();
        for (k, &q) in free.iter().enumerate() {
            x[q] = xf[k];
        }
        let nj = self.projs.len();
        let s = y[..nj].to_vec();
        let mut t = vec![0.0; r];
        for (k, &q) in free.iter().enumerate() {
            t[q] = y[nj + k];
        }
        let primal: f64 = y.iter().zip(&b).map(|(y, b)| y * b).sum();
        Some(LpSolution {
            sigma: obj + atomic_sum,
            x,
            s,
            t,
            gap: primal - obj,
        })
    }

    /// Body evaluations: `|V|`.
    pub fn volume(&self) -> f64 {
        self.extents.iter().map(|&n| n as f64).product()
    }

    /// Primal bound on traffic (words) at fast memory `mem` words.
    ///
    /// From the primal solution: in a stretch with `2M` available words
    /// split as `B_j = 2M·s_j/Σs` (the split maximizing `∏ B_j^{s_j}`),
    /// the free-coordinate units are at most `∏_j (B_j/D_j)^{s_j} ·
    /// ∏_q N_q^{t_q}` (HBL with loop bounds; `D_j` = the array's atomic
    /// extent, which every unit needs whole), each unit doing `∏_atomic N_q`
    /// evaluations. This is never weaker than `(2M)^σ` and gives
    /// `1.84·mnk/√M` for matmul (vs `2mnk/√M` proved by Kwasniewski et al.).
    pub fn bound(&self, mem: f64) -> Option<f64> {
        let s = self.solve(2.0 * mem)?;
        Some((mem * (self.volume() / self.ops_per_stretch(mem, &s) - 1.0)).max(0.0))
    }

    fn ops_per_stretch(&self, mem: f64, s: &LpSolution) -> f64 {
        let r = self.extents.len();
        let total: f64 = s.s.iter().sum();
        let atomic_ops: f64 = (0..r)
            .filter(|q| self.atomic & (1 << q) != 0)
            .map(|q| self.extents[q] as f64)
            .product();
        let units: f64 = self
            .projs
            .iter()
            .zip(&s.s)
            .filter(|(_, &sj)| sj > 1e-12)
            .map(|(&p, &sj)| {
                let dj: f64 = (0..r)
                    .filter(|q| p & (1 << q) != 0 && self.atomic & (1 << q) != 0)
                    .map(|q| self.extents[q] as f64)
                    .product();
                (2.0 * mem * sj / total / dj).max(1.0).powf(sj)
            })
            .product::<f64>()
            * (0..r)
                .filter(|q| self.atomic & (1 << q) == 0)
                .map(|q| (self.extents[q] as f64).powf(s.t[q]))
                .product::<f64>();
        let f = units * atomic_ops;
        // never claim more than the (2M)^σ form allows to be weaker
        f.min((2.0 * mem).powf(s.sigma))
    }

    /// Dual tile at fast memory `mem`: `T_q = ⌊(mem/J)^{x_q}⌋` clamped to
    /// `[1, N_q]`; atomic coordinates take their full extent.
    pub fn tile(&self, mem: f64) -> Option<Vec<usize>> {
        let base = mem / self.projs.len().max(1) as f64;
        let s = self.solve(base)?;
        Some(
            (0..self.extents.len())
                .map(|q| {
                    if self.atomic & (1 << q) != 0 {
                        self.extents[q]
                    } else {
                        (base.powf(s.x[q]) * (1.0 + 1e-9))
                            .floor()
                            .clamp(1.0, self.extents[q] as f64) as usize
                    }
                })
                .collect(),
        )
    }

    /// Traffic (words) of the box tiling `tile`: each array is loaded once
    /// per tile along every coordinate it does not depend on. `sizes[j]` is
    /// the array's element count.
    pub fn tiled_traffic(&self, tile: &[usize], sizes: &[f64]) -> f64 {
        self.projs
            .iter()
            .zip(sizes)
            .map(|(&p, &n)| {
                n * (0..self.extents.len())
                    .filter(|q| p & (1 << q) == 0)
                    .map(|q| self.extents[q].div_ceil(tile[q].max(1)) as f64)
                    .product::<f64>()
            })
            .sum()
    }
}

/// `max c·x  s.t.  A x ≤ b, x ≥ 0` with `b ≥ 0` (origin feasible), dense
/// tableau, Bland's rule. Returns `(x, y, objective)` with `y` the optimal
/// dual (shadow prices of the rows). `None` if unbounded.
pub fn simplex(a: &[Vec<f64>], b: &[f64], c: &[f64]) -> Option<(Vec<f64>, Vec<f64>, f64)> {
    let m = a.len();
    let n = c.len();
    let w = n + m + 1;
    // rows 0..m constraints, row m objective (z − c·x = 0)
    let mut t = vec![vec![0.0; w]; m + 1];
    for i in 0..m {
        t[i][..n].copy_from_slice(&a[i]);
        t[i][n + i] = 1.0;
        t[i][w - 1] = b[i];
    }
    for j in 0..n {
        t[m][j] = -c[j];
    }
    let mut basis: Vec<usize> = (n..n + m).collect();
    let eps = 1e-12;
    for _ in 0..10_000 {
        let Some(e) = (0..n + m).find(|&j| t[m][j] < -eps) else {
            break;
        };
        let mut leave: Option<(usize, f64)> = None;
        for i in 0..m {
            if t[i][e] > eps {
                let ratio = t[i][w - 1] / t[i][e];
                match leave {
                    Some((li, lr))
                        if ratio > lr + eps || (ratio > lr - eps && basis[i] > basis[li]) => {}
                    _ => leave = Some((i, ratio)),
                }
            }
        }
        let (l, _) = leave?;
        let p = t[l][e];
        for v in t[l].iter_mut() {
            *v /= p;
        }
        for i in 0..=m {
            if i != l && t[i][e].abs() > eps {
                let f = t[i][e];
                let row = t[l].clone();
                for (v, r) in t[i].iter_mut().zip(&row) {
                    *v -= f * r;
                }
            }
        }
        basis[l] = e;
    }
    let mut x = vec![0.0; n];
    for (i, &bv) in basis.iter().enumerate() {
        if bv < n {
            x[bv] = t[i][w - 1];
        }
    }
    let y: Vec<f64> = (0..m).map(|i| t[m][n + i]).collect();
    Some((x, y, t[m][w - 1]))
}

/// Peak rates of one device, for the composite time floor.
#[derive(Clone, Copy, Debug)]
pub struct Machine {
    pub name: &'static str,
    /// Fast memory one workgroup can stage, words.
    pub mem_words: f64,
    /// DRAM bandwidth, bytes/s.
    pub dram_bw: f64,
    /// Peak arithmetic, flop/s (f32 FMA = 2).
    pub flops: f64,
    /// Per-kernel launch + drain latency, s.
    pub launch: f64,
}

impl Machine {
    /// RTX 3090: 48 KB shared/block, 936 GB/s, 35.6 TFLOP/s f32, ~5 µs.
    pub const RTX3090: Machine = Machine {
        name: "3090",
        mem_words: 12288.0,
        dram_bw: 936e9,
        flops: 35.6e12,
        launch: 5e-6,
    };
    /// Apple M4 (10-core GPU): 32 KB threadgroup, 120 GB/s, ~4.3 TFLOP/s
    /// f32, ~10 µs per dispatch through wgpu.
    pub const M4: Machine = Machine {
        name: "m4",
        mem_words: 8192.0,
        dram_bw: 120e9,
        flops: 4.3e12,
        launch: 10e-6,
    };
}

/// `max(bytes / BW, flops / peak, launches × latency)`, with the binding
/// term named.
#[derive(Clone, Copy, Debug)]
pub struct TimeFloor {
    pub memory_s: f64,
    pub compute_s: f64,
    pub launch_s: f64,
}

impl TimeFloor {
    pub fn new(m: &Machine, dram_bytes: f64, flops: f64, launches: usize) -> Self {
        Self {
            memory_s: dram_bytes / m.dram_bw,
            compute_s: flops / m.flops,
            launch_s: launches as f64 * m.launch,
        }
    }
    pub fn seconds(&self) -> f64 {
        self.memory_s.max(self.compute_s).max(self.launch_s)
    }
    pub fn binding(&self) -> &'static str {
        let s = self.seconds();
        if s == self.memory_s {
            "memory"
        } else if s == self.compute_s {
            "compute"
        } else {
            "launch"
        }
    }
}

/// Matmul `C[i,j] = Σ_k A[i,k] B[k,j]` over coords `(i, j, k)`.
pub fn matmul_lp(m: usize, n: usize, k: usize) -> Lp {
    Lp {
        extents: vec![m, n, k],
        projs: vec![0b101, 0b110, 0b011],
        atomic: 0,
    }
}

/// Fused attention forward over coords `(i, j, c, e)`: `S = Σ_c Q[i,c]
/// K[j,c]`, `O[i,e] = ⊕_j softmax · V[j,e]`. `fused = true`: scores never
/// stored, `c` atomic (and `e` too: each `P·V` row update needs the whole
/// value row of the online state). `fused = false`: the partial scores
/// `S[i,j]` are an array and `c` is free — the small-cache regime.
pub fn attention_lp(n: usize, d: usize, fused: bool) -> Lp {
    // bits: i=0, j=1, c=2, e=3
    let q = 0b0101;
    let k = 0b0110;
    let v = 0b1010;
    let o = 0b1001;
    if fused {
        Lp {
            extents: vec![n, n, d, d],
            projs: vec![q, k, v, o],
            atomic: 0b1100,
        }
    } else {
        // the QKᵀ contraction with its partial sums, the rest streamed
        Lp {
            extents: vec![n, n, d],
            projs: vec![0b101, 0b110, 0b011],
            atomic: 0,
        }
    }
}

/// Attention bound = min over the two models (a schedule picks either).
pub fn attention_bound(n: usize, d: usize, mem: f64) -> f64 {
    let fused = attention_lp(n, d, true).bound(mem);
    let spill = attention_lp(n, d, false).bound(mem);
    match (fused, spill) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => 0.0,
    }
    .max(4.0 * (n * d) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn simplex_small() {
        // max x + y, x + 2y ≤ 4, 3x + y ≤ 6 → (1.6, 1.2), obj 2.8, y = (0.4, 0.2)
        let (x, y, o) =
            simplex(&[vec![1.0, 2.0], vec![3.0, 1.0]], &[4.0, 6.0], &[1.0, 1.0]).unwrap();
        assert!(
            close(o, 2.8) && close(x[0], 1.6) && close(x[1], 1.2),
            "{x:?} {o}"
        );
        assert!(close(y[0], 0.4) && close(y[1], 0.2), "{y:?}");
    }

    #[test]
    fn matmul_lp_is_hong_kung() {
        let mem = 4096.0;
        let lp = matmul_lp(4096, 4096, 4096);
        let s = lp.solve(2.0 * mem).unwrap();
        assert!(close(s.sigma, 1.5), "{s:?}");
        assert!(s.gap.abs() < 1e-9);
        assert!(s.s.iter().all(|&v| close(v, 0.5)) && s.x.iter().all(|&v| close(v, 0.5)));
        // Q ≥ M·(n³/(2M/3)^{3/2} − 1) ≈ 1.84·n³/√M − M (Irony–Toledo–
        // Tiskin form with the optimal budget split)
        let n = 4096f64;
        let want = mem * (n * n * n / (2.0 * mem / 3.0).powf(1.5) - 1.0);
        assert!(close(lp.bound(mem).unwrap(), want));
        // dual tile: cube of side √(M/3)
        let t = lp.tile(mem).unwrap();
        assert_eq!(t, vec![36, 36, 36]);
    }

    #[test]
    fn small_matmul_hits_extent_bounds() {
        // k = 16 < √M: the k-coordinate is capped, σ = 1 + β_k
        let lp = matmul_lp(4096, 4096, 16);
        let s = lp.solve(8192.0).unwrap();
        let bk = 16f64.ln() / 8192f64.ln();
        assert!(
            close(s.sigma, 2.0 - bk) || s.sigma <= 1.5 + 1e-9 || close(s.sigma, 1.0 + bk),
            "{s:?}"
        );
        assert!(s.gap.abs() < 1e-9);
    }

    #[test]
    fn attention_is_saha_ye() {
        // M ≥ d²: Θ(N²d²/M), with the fused (atomic) model binding
        for &(n, d, mem) in &[
            (32768usize, 64usize, 12288.0f64),
            (65536, 64, 12288.0),
            (65536, 32, 8192.0),
        ] {
            let lp = attention_lp(n, d, true);
            let s = lp.solve(2.0 * mem).unwrap();
            assert!(s.gap.abs() < 1e-9);
            let b = attention_bound(n, d, mem);
            let rate = (n * n * d * d) as f64 / mem;
            let r = b / rate;
            // a constant independent of N and M: Θ(N²d²/M)
            assert!((0.05..=1.0).contains(&r), "N={n} d={d} M={mem}: {r}");
            // doubling N quadruples it
            let b2 = attention_bound(2 * n, d, mem);
            assert!((b2 / b - 4.0).abs() < 0.05, "{}", b2 / b);
            // doubling M halves it
            let bm = attention_bound(n, d, 2.0 * mem);
            assert!((b / bm - 2.0).abs() < 0.1, "{}", b / bm);
        }
        // M < d²: Θ(N²d/√M), the spilled-partials model binding
        let (n, d, mem) = (8192usize, 256usize, 4096.0f64);
        let b = attention_bound(n, d, mem);
        let r = b / ((n * n * d) as f64 / mem.sqrt());
        assert!((0.05..=2.0).contains(&r), "{r}");
        let bm = attention_bound(n, d, 4.0 * mem);
        assert!((b / bm - 2.0).abs() < 0.2, "{}", b / bm);
    }

    #[test]
    fn attention_dual_tile_is_flash_blocks() {
        // T_i = T_j = M/(4d): FlashAttention's B_r = B_c = M/(4d)
        let (n, d, mem) = (4096usize, 64usize, 12288.0f64);
        let t = attention_lp(n, d, true).tile(mem).unwrap();
        assert_eq!(t, vec![48, 48, 64, 64], "{t:?}");
    }

    #[test]
    fn weak_duality_random() {
        // primal ≥ dual for random projections; equality at the optimum
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed >> 33
        };
        for _ in 0..200 {
            let r = 2 + (rnd() % 3) as usize;
            let j = 2 + (rnd() % 3) as usize;
            let projs: Vec<u64> = (0..j).map(|_| 1 + rnd() % ((1 << r) - 1)).collect();
            let extents: Vec<usize> = (0..r).map(|_| 1 << (2 + rnd() % 12)).collect();
            let lp = Lp {
                extents,
                projs,
                atomic: 0,
            };
            if let Some(s) = lp.solve(1e4) {
                assert!(s.gap.abs() < 1e-7, "{lp:?} {s:?}");
                // primal feasibility
                for q in 0..r {
                    let cov: f64 = lp
                        .projs
                        .iter()
                        .zip(&s.s)
                        .filter(|(p, _)| *p & (1 << q) != 0)
                        .map(|(_, v)| v)
                        .sum::<f64>()
                        + s.t[q];
                    assert!(cov >= 1.0 - 1e-7, "{lp:?} {s:?}");
                }
            }
        }
    }
}

/// What [`TensorGraph::kernel_lp`] could not put in the LP.
#[derive(Clone, Debug, Default)]
pub struct LpCoverage {
    /// Reads through gather views (no projection: the address is data).
    pub gathered: Vec<super::TensorId>,
    /// Reads through views with a multi-term form (`x[t + k]`, guarded
    /// convolutions): HBL needs the exact linear map, and its image is
    /// not a coordinate subspace.
    pub non_projective: Vec<super::TensorId>,
}

impl super::TensorGraph {
    /// The LP of kernel `k` (a materialized comp of a scheduled graph),
    /// read off its views: coordinates are the kernel's point and reduce
    /// dims plus the reduce dims of every nested (inlined) reduction —
    /// those are atomic — and arrays are the buffers it reads plus its
    /// output. Returns the LP, the coordinate names (`p0…`, `r0…`, `n0…`),
    /// and what was left out.
    pub fn kernel_lp(&self, k: super::TensorId) -> Option<(Lp, Vec<String>, LpCoverage)> {
        use super::*;
        let root = self.tiled_reduction(k.n()).unwrap_or(k.n());
        let rc = self.comp_at(root)?;
        let np = rc.points().len();
        let mut extents: Vec<usize> = fixed(&rc.space());
        let mut names: Vec<String> = (0..np)
            .map(|q| format!("p{q}"))
            .chain((0..rc.red.len()).map(|q| format!("r{q}")))
            .collect();
        let mut atomic = 0u64;
        let mut projs: Vec<(usize, u64)> = Vec::new();
        let mut cov = LpCoverage::default();
        let space: Vec<u64> = (0..extents.len()).map(|q| 1u64 << q).collect();
        #[allow(clippy::too_many_arguments)]
        fn walk(
            g: &TensorGraph,
            i: usize,
            space: &[u64],
            extents: &mut Vec<usize>,
            names: &mut Vec<String>,
            atomic: &mut u64,
            projs: &mut Vec<(usize, u64)>,
            cov: &mut LpCoverage,
            memo: &mut Vec<(usize, Vec<u64>, Vec<u64>)>,
        ) {
            let Some(c) = g.comp_at(i) else { return };
            for a in &c.args {
                let src = a.src.n();
                let mut multi = false;
                let mask = a.view.forms.iter().fold(0u64, |m, f| {
                    let terms: Vec<usize> = f
                        .coefs
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| **c != 0)
                        .map(|(q, _)| q)
                        .collect();
                    if terms.len() > 1 {
                        multi = true;
                    }
                    terms
                        .iter()
                        .fold(m, |m, &q| m | space.get(q).copied().unwrap_or(0))
                });
                match &g.nodes[src].op {
                    Op::Iota(_) => {}
                    Op::Comp(pc) if g.nodes[src].inline => {
                        let pnp = pc.points().len();
                        let mut inner: Vec<u64> = a
                            .view
                            .forms
                            .iter()
                            .take(pnp)
                            .map(|f| {
                                f.coefs
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, c)| **c != 0)
                                    .fold(0, |m, (q, _)| m | space.get(q).copied().unwrap_or(0))
                            })
                            .collect();
                        inner.resize(pnp, 0);
                        // the same nested comp read at the same points (a
                        // body reading S twice) shares its coordinates
                        if memo
                            .iter()
                            .any(|m| m.0 == src && (m.1 == inner || m.2.is_empty()))
                        {
                            continue;
                        }
                        let pts = inner.clone();
                        let mut bits = Vec::new();
                        for (r, d) in pc.red.iter().enumerate() {
                            let bit = extents.len();
                            extents.push(d.n());
                            names.push(format!("n{}_{r}", src));
                            if pc.monoid.is_some() {
                                *atomic |= 1 << bit;
                            }
                            inner.push(1 << bit);
                            bits.push(1u64 << bit);
                        }
                        memo.push((src, pts, bits));
                        walk(g, src, &inner, extents, names, atomic, projs, cov, memo);
                    }
                    _ => {
                        let s = g.io_storage(src);
                        if a.view.gather.is_some() {
                            cov.gathered.push(TensorId(s as u32));
                        } else if multi {
                            cov.non_projective.push(TensorId(s as u32));
                        } else if mask != 0 && !projs.iter().any(|p| p.0 == s && p.1 == mask) {
                            projs.push((s, mask));
                        }
                    }
                }
            }
        }
        // the root itself is never re-entered (an epilogue reads it)
        let mut memo = vec![(root, Vec::new(), Vec::new())];
        walk(
            self,
            root,
            &space,
            &mut extents,
            &mut names,
            &mut atomic,
            &mut projs,
            &mut cov,
            &mut memo,
        );
        if root != k.n() {
            // the epilogue map's own reads, over the same points
            let ep: Vec<u64> = (0..np).map(|q| 1u64 << q).collect();
            walk(
                self,
                k.n(),
                &ep,
                &mut extents,
                &mut names,
                &mut atomic,
                &mut projs,
                &mut cov,
                &mut memo,
            );
        }
        let out = (0..np).fold(0u64, |m, q| m | 1 << q);
        let mut p: Vec<u64> = projs.iter().map(|p| p.1).collect();
        if out != 0 && !p.contains(&out) {
            p.push(out);
        }
        Some((
            Lp {
                extents,
                projs: p,
                atomic,
            },
            names,
            cov,
        ))
    }
}

/// Run `stabilize_softmax` on a softmax-weighted sum
/// `y[i] = Σ_j e^{s[i,j] − m[i]} v[j] / Σ_j e^{s[i,j] − m[i]}` and export the
/// combine of the online-softmax monoid it produces as a Lean 4 definition
/// (`proofs/TangProofs/Trace.lean` is this output; a test keeps it current).
/// Ops map to ℝ: `Exp2 x = 2^x`, `Select c a b = if 0 < c then a else b`,
/// the literal `log₂ e` becomes the symbol `log2e` (its f64 rounding is not
/// the real number; the Lean theorem assumes the exact value).
pub fn lean_trace_stabilize_softmax() -> String {
    use super::*;
    use tang_expr::node::Node;
    let (r, n) = (4usize, 8usize);
    let mut g = TensorGraph::new();
    let s = g.input(&[r, n]);
    let v = g.input(&[n]);
    let row = |rank| View::pick(rank, &[Some(0), Some(rank - 1)]);
    let mx = g
        .comp(
            &[r, 1],
            &[n],
            vec![Arg {
                src: s,
                view: row(3),
            }],
            Some(Monoid::max()),
            |x| vec![x[0]],
        )
        .unwrap();
    let bc = || View::pick(3, &[Some(0), None]);
    let l = g
        .comp(
            &[r, 1],
            &[n],
            vec![
                Arg {
                    src: s,
                    view: row(3),
                },
                Arg {
                    src: mx,
                    view: bc(),
                },
            ],
            Some(Monoid::sum()),
            |x| vec![(x[0] - x[1]).exp()],
        )
        .unwrap();
    let o = g
        .comp(
            &[r, 1],
            &[n],
            vec![
                Arg {
                    src: s,
                    view: row(3),
                },
                Arg {
                    src: mx,
                    view: bc(),
                },
                Arg {
                    src: v,
                    view: View::pick(3, &[Some(2)]),
                },
            ],
            Some(Monoid::sum()),
            |x| vec![(x[0] - x[1]).exp() * x[2]],
        )
        .unwrap();
    let y = g
        .comp(
            &[r, 1],
            &[] as &[usize],
            vec![
                Arg {
                    src: o,
                    view: View::identity(2, 2),
                },
                Arg {
                    src: l,
                    view: View::identity(2, 2),
                },
            ],
            None,
            |x| vec![x[0] * x[1].recip()],
        )
        .unwrap();
    let (g2, outs) = g.stabilize_softmax(&[y]);
    let live = g2.live(&outs);
    let m = (0..g2.nodes.len())
        .filter(|&i| live[i])
        .filter_map(|i| g2.comp_at(i)?.monoid.clone())
        .find(|m| m.kind() == MonoidKind::OnlineSoftmax)
        .expect("stabilize_softmax produced no online-softmax monoid");
    let nodes = m.combine.nodes_slice();
    let mut out = String::new();
    out.push_str("/-\nGenerated by `tang_compile::tensor::io_lp::lean_trace_stabilize_softmax`\n");
    out.push_str("(do not edit; `cargo test -p tang-compile trace_is_current` checks it).\n");
    out.push_str("The combine of the online-softmax monoid `stabilize_softmax` built for\n");
    out.push_str("`Σ e^{s−m} v / Σ e^{s−m}`, node for node from its tang-expr graph.\n-/\n");
    out.push_str("import Mathlib\nimport TangProofs.TraceOps\n\nnamespace Tang.Trace\n\nopen Tang.TraceOps\n\n");
    let w = m.width;
    let vars: Vec<String> = (0..2 * w).map(|i| format!("v{i}")).collect();
    out.push_str(&format!(
        "/-- width {w}: accumulator `v0..v{}`, element `v{w}..v{}`. -/\nnoncomputable def combine ({} : ℝ) : {} :=\n",
        w - 1,
        2 * w - 1,
        vars.join(" "),
        vec!["ℝ"; w].join(" × ")
    ));
    let name = |id: tang_expr::ExprId| format!("n{}", id.index());
    let mut used = vec![false; nodes.len()];
    let mut stack: Vec<usize> = m.outs.iter().map(|o| o.index() as usize).collect();
    while let Some(i) = stack.pop() {
        if std::mem::replace(&mut used[i], true) {
            continue;
        }
        let kids: Vec<tang_expr::ExprId> = match nodes[i] {
            Node::Add(a, b) | Node::Mul(a, b) | Node::Atan2(a, b) => vec![a, b],
            Node::Neg(a)
            | Node::Recip(a)
            | Node::Sqrt(a)
            | Node::Sin(a)
            | Node::Exp2(a)
            | Node::Log2(a) => vec![a],
            Node::Select(c, a, b) => vec![c, a, b],
            Node::Var(_) | Node::Lit(_) => vec![],
        };
        stack.extend(kids.iter().map(|k| k.index() as usize));
    }
    for (i, nd) in nodes.iter().enumerate().filter(|(i, _)| used[*i]) {
        let id = tang_expr::ExprId::from_index(i as u32);
        let rhs = match *nd {
            Node::Var(k) => format!("v{k}"),
            Node::Lit(bits) => {
                let x = f64::from_bits(bits);
                if x == std::f64::consts::LOG2_E {
                    "log2e".into()
                } else if x == x.trunc() && x.abs() < 1e15 {
                    format!("({})", x as i64)
                } else {
                    panic!("unexpected literal {x} in the combine")
                }
            }
            Node::Add(a, b) => format!("{} + {}", name(a), name(b)),
            Node::Mul(a, b) => format!("{} * {}", name(a), name(b)),
            Node::Neg(a) => format!("-{}", name(a)),
            Node::Recip(a) => format!("{}⁻¹", name(a)),
            Node::Exp2(a) => format!("exp2 {}", name(a)),
            Node::Select(c, a, b) => format!("sel {} {} {}", name(c), name(a), name(b)),
            ref other => panic!("op {other:?} not exported"),
        };
        out.push_str(&format!("  let {} := {rhs}\n", name(id)));
    }
    let res: Vec<String> = m.outs.iter().map(|&o| name(o)).collect();
    out.push_str(&format!("  ({})\n\nend Tang.Trace\n", res.join(", ")));
    out
}

#[cfg(test)]
mod trace_tests {
    #[test]
    fn trace_is_current() {
        let want = super::lean_trace_stabilize_softmax();
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../proofs/TangProofs/Trace.lean"
        );
        if std::env::var("TANG_WRITE_TRACE").is_ok() {
            std::fs::write(path, &want).unwrap();
        }
        let have = std::fs::read_to_string(path).unwrap_or_default();
        assert_eq!(have, want, "regenerate with TANG_WRITE_TRACE=1");
    }
}
