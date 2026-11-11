//! IO lower bounds and the traffic of the schedule tang picks.
//!
//! Two numbers per graph, both in *words* (one word = one element, so
//! multiply by the storage width for bytes):
//!
//! * a **lower bound** on slow↔fast-memory traffic over *every* schedule of
//!   the comprehension graph (red–blue pebbling, Hong & Kung 1981), as a
//!   function of the fast-memory size `M` in words;
//! * the **staged traffic** of the schedule tang actually picked: every
//!   kernel reads each buffer once per workgroup tile that needs it and
//!   writes its output once. Tiles' operand reuse is what the schedule
//!   stages in workgroup memory; reuse through hardware caches across
//!   workgroups is *not* credited, so this is an upper estimate of DRAM
//!   traffic (and an exact count of the loads the kernels issue).
//!
//! The generic bound ([`TensorGraph::io_lower_bound`]) is the larger of
//!
//! * the *compulsory* traffic — every graph input is read once and every
//!   output written once (a red–blue pebbling must place a blue pebble on
//!   each output and start from blue pebbles on each input), and
//! * a Hölder–Brascamp–Lieb bound per reduction (Christ, Demmel, Knight,
//!   Scanlon, Yelick 2013; the matmul case is Irony, Toledo, Tiskin 2004):
//!   in any stretch of the schedule with `M` loads/stores at most `2M`
//!   words of each array are available, and by the discrete HBL
//!   inequality the number of body evaluations whose operands are all
//!   available is at most `∏ⱼ xⱼ^{sⱼ}` with `Σ xⱼ ≤ 2M`, for any `s`
//!   covering every loop index (`Σ_{j ∋ i} sⱼ ≥ 1`). So the schedule needs
//!   `≥ M · (⌊|V| / F(2M)⌋)` traffic. Only arrays that are graph inputs
//!   (and the reduction's output, if it is a graph output) enter: an
//!   intermediate can be recomputed in fast memory without traffic.
//!
//! Closed forms with tighter constants for the kernels the paper tables
//! ([`bounds`]) cite their sources.
use super::*;

/// Lower bounds from the literature, in words, `m` = fast memory in words.
pub mod bounds {
    /// Kwasniewski et al., "Red-blue pebbling revisited" (SC'19), Thm. 1:
    /// classical `m×k · k×n` matmul needs `≥ 2mnk/√M + mn` I/Os
    /// (stated for `M < min(mn, mk, nk)`; below that the compulsory
    /// `mk + kn + mn` dominates anyway).
    pub fn matmul(m: f64, n: f64, k: f64, mem: f64) -> f64 {
        let c = m * n + m * k + k * n;
        (2.0 * m * n * k / mem.sqrt() + m * n).max(c)
    }

    /// Attention forward `softmax(QKᵀ)V`, `n` queries/keys, head dim `d`.
    /// Saha & Ye, "The I/O complexity of attention" (ICML'24): Θ(n²d²/M)
    /// for `M ≥ d²` (FlashAttention's rate, which they show optimal) and
    /// Θ(n²d/√M) for `M < d²`. Their counting lemma bounds the entries of
    /// `QKᵀ` one stretch of `M` I/Os can produce by `2M²/d²`, which gives
    /// the explicit `n²d²/(2M) − M` used here (constant ours, from that
    /// lemma; the paper states only Θ). `causal` halves the score count.
    pub fn attention(n: f64, d: f64, mem: f64, causal: bool) -> f64 {
        let scores = if causal { n * (n + 1.0) / 2.0 } else { n * n };
        let compulsory = 4.0 * n * d;
        let reuse = if mem >= d * d {
            scores * d * d / (2.0 * mem) - mem
        } else {
            // M < d²: each score is a d-term dot product; HBL over (i, j, c)
            // with Q, K and the scores' partial sums (s = ½, ½, ½) allows
            // (2M/3)^{3/2} multiply-adds per stretch
            mem * (scores * d / (2.0 * mem / 3.0).powf(1.5) - 1.0)
        };
        compulsory.max(reuse)
    }

    /// Attention backward (dQ, dK, dV from Q, K, V, O, dO, lse). Every score
    /// must be produced again (recomputed or read), so the forward's
    /// counting applies with one more operand per score (`dP = dO·Vᵀ`
    /// shares the stretch): `n²d²/(3M) − M`. Compulsory: read 5 `n×d`
    /// tensors + `n` statistics, write 3.
    pub fn attention_bwd(n: f64, d: f64, mem: f64, causal: bool) -> f64 {
        let scores = if causal { n * (n + 1.0) / 2.0 } else { n * n };
        let compulsory = 8.0 * n * d + n;
        compulsory.max(scores * d * d / (3.0 * mem) - mem)
    }

    /// Row softmax / RMSNorm / LayerNorm over `rows × cols`. No output of
    /// a row can be written before its whole row is read (the normalizer
    /// depends on all of it), and each output needs its input again: of
    /// the `cols` inputs at most `M` stay resident, so `cols − M` are read
    /// twice. Per row: `2·cols + max(0, cols − M)`.
    pub fn row_normalize(rows: f64, cols: f64, mem: f64) -> f64 {
        rows * (2.0 * cols + (cols - mem).max(0.0))
    }

    /// MoE expert FFN over `tokens` routed `top_k` ways to `experts`
    /// experts with `d → f` weights, assuming balanced routing: every
    /// active expert's weight is read once (compulsory) and each expert's
    /// grouped matmul obeys [`matmul`] with `tokens·top_k/experts` rows.
    pub fn moe_matmul(tokens: f64, top_k: f64, experts: f64, d: f64, f: f64, mem: f64) -> f64 {
        let rows = tokens * top_k / experts;
        experts * matmul(rows, f, d, mem)
    }

    /// Chunked linear-attention / gated-delta scan: `t` steps, state
    /// `dk × dv` per head. If the state fits in fast memory the
    /// compulsory read of q, k, v (and gates) plus the output is tight;
    /// otherwise the state must be streamed once per chunk of `c` steps.
    pub fn chunked_scan(t: f64, dk: f64, dv: f64, c: f64, mem: f64) -> f64 {
        let compulsory = t * (2.0 * dk + 2.0 * dv + 1.0);
        let state = dk * dv;
        if state <= mem {
            compulsory
        } else {
            compulsory + (t / c) * (state - mem)
        }
    }
}

/// One buffer read by a kernel (through inlined producers).
#[derive(Clone, Copy, Debug)]
struct Leaf {
    storage: usize,
    /// Kernel point dims the read depends on.
    mask: u64,
    /// Gathered read: `(extent of the gathered source dim, point dims the
    /// index depends on)`.
    gather: Option<(usize, u64)>,
}

/// Traffic of the best *known* schedules (upper bounds on the optimum), for
/// reading the gap between a lower bound and what is achievable.
pub mod achievable {
    /// FlashAttention-2 forward with square blocks `B = ⌊M/(4d)⌋`
    /// (Dao et al. 2022, Alg. 1 block sizes): Q, O, lse once; K and V once
    /// per query block, only up to the diagonal when causal.
    pub fn flash_fwd(n: f64, d: f64, mem: f64, causal: bool) -> f64 {
        let b = (mem / (4.0 * d)).floor().max(1.0);
        let nb = (n / b).ceil();
        let pairs = if causal {
            nb * (nb + 1.0) / 2.0
        } else {
            nb * nb
        };
        2.0 * n * d + n + 2.0 * d * b * pairs
    }

    /// FlashAttention-2 backward: K, V read and dK, dV written once; per
    /// (key block, query block) pair read Q, dO, lse, D and read+write dQ.
    pub fn flash_bwd(n: f64, d: f64, mem: f64, causal: bool) -> f64 {
        let b = (mem / (4.0 * d)).floor().max(1.0);
        let nb = (n / b).ceil();
        let pairs = if causal {
            nb * (nb + 1.0) / 2.0
        } else {
            nb * nb
        };
        4.0 * n * d + pairs * (4.0 * b * d + 2.0 * b)
    }
}

/// Traffic of one kernel under its schedule.
#[derive(Clone, Debug)]
pub struct KernelIo {
    /// The kernel's root node.
    pub node: TensorId,
    /// Points per workgroup along each point dim.
    pub tile: Vec<usize>,
    /// `(buffer, words read)` per distinct buffer.
    pub reads: Vec<(TensorId, f64)>,
    /// Words written.
    pub write: f64,
}

impl KernelIo {
    pub fn words(&self) -> f64 {
        self.reads.iter().map(|r| r.1).sum::<f64>() + self.write
    }
}

/// [`TensorGraph::io_report`].
#[derive(Clone, Debug)]
pub struct IoReport {
    pub kernels: Vec<KernelIo>,
    /// Inputs read once + outputs written once.
    pub compulsory: f64,
    /// Largest per-reduction HBL bound at the given `M`.
    pub hbl: f64,
}

impl IoReport {
    /// Staged traffic of the schedule, words.
    pub fn achieved(&self) -> f64 {
        self.kernels.iter().map(KernelIo::words).sum()
    }
    /// `max(compulsory, hbl)`.
    pub fn bound(&self) -> f64 {
        self.compulsory.max(self.hbl)
    }
}

impl TensorGraph {
    /// Follow reshapes to the node that owns the bytes.
    pub(crate) fn io_storage(&self, mut t: usize) -> usize {
        while let Op::Reshape(s) = &self.nodes[t].op {
            t = s.n();
        }
        t
    }

    /// Leaf reads of comp `i` (through inlined producers): `(storage,
    /// mask)` with `mask` the kernel point dims the read depends on, given
    /// `space[q]` = mask of coordinate `q` of `i`'s iteration space.
    fn io_leaves(&self, i: usize, space: &[u64], out: &mut Vec<Leaf>) {
        let Some(c) = self.comp_at(i) else { return };
        let form_mask = |f: &Form| {
            f.coefs
                .iter()
                .enumerate()
                .filter(|(_, c)| **c != 0)
                .fold(0u64, |m, (q, _)| m | space.get(q).copied().unwrap_or(0))
        };
        let arg_masks: Vec<Vec<u64>> = c
            .args
            .iter()
            .map(|a| a.view.forms.iter().map(form_mask).collect())
            .collect();
        for (ai, a) in c.args.iter().enumerate() {
            let mut forms = arg_masks[ai].clone();
            let src = a.src.n();
            let mut gather = None;
            if let Some(g) = &a.view.gather {
                let idx = arg_masks[g.arg].iter().fold(0, |m, x| m | x);
                let shape = fixed(&self.nodes[self.io_storage(src)].shape);
                for (d, (f, &k)) in forms.iter_mut().zip(&g.coefs).enumerate() {
                    if k != 0 {
                        *f |= idx;
                        gather = Some((shape.get(d).copied().unwrap_or(1), idx));
                    }
                }
            }
            match &self.nodes[src].op {
                Op::Iota(_) => {}
                Op::Comp(pc) if self.nodes[src].inline => {
                    // inner space: its point dims read through this view,
                    // its reduce dims are fresh (swept inside the tile)
                    let np = pc.points().len();
                    let inner: Vec<u64> = (0..np + pc.red.len())
                        .map(|q| if q < np { forms[q] } else { 0 })
                        .collect();
                    self.io_leaves(src, &inner, out);
                }
                _ => {
                    let m = forms.iter().fold(0, |m, x| m | x);
                    out.push(Leaf {
                        storage: self.io_storage(src),
                        mask: m,
                        gather,
                    });
                }
            }
        }
    }

    /// Points per workgroup along each point dim of kernel `i`.
    fn io_tile(&self, i: usize) -> Vec<usize> {
        let node = &self.nodes[i];
        let Some(c) = self
            .tiled_reduction(i)
            .and_then(|r| self.comp_at(r))
            .or_else(|| self.comp_at(i))
        else {
            return Vec::new();
        };
        let pts = fixed(c.points());
        let mut t = vec![1; pts.len()];
        if node.tile.0 == 0 && node.split == 0 {
            // one thread per point, 256 consecutive (row-major) points per
            // workgroup: a read constant across them is one fetch
            let mut left = 256usize;
            for d in (0..pts.len()).rev() {
                if left <= 1 {
                    break;
                }
                t[d] = pts[d].min(left);
                left /= t[d].max(1);
            }
        } else if node.tile.0 > 0 {
            let (lane, row) = tile_axes(&pts);
            if let Some(x) = lane {
                t[x] = (node.tile.0 * node.reg.0.max(1)) as usize;
            }
            if let Some(y) = row {
                t[y] = (node.tile.1.max(1) * node.reg.1.max(1)) as usize;
            }
        }
        t
    }

    /// Staged traffic of the scheduled graph's kernels, plus the generic
    /// lower bound at fast-memory size `mem` (words).
    pub fn io_report(&self, outputs: &[TensorId], mem: f64) -> IoReport {
        let live = self.live(outputs);
        let mut kernels = Vec::new();
        for i in (0..self.nodes.len()).filter(|&i| live[i] && !self.nodes[i].inline) {
            match &self.nodes[i].op {
                Op::Comp(c) => {
                    // the kernel iterates its tiled reduction's points when
                    // it is an epilogue map over one
                    let root = self.tiled_reduction(i).unwrap_or(i);
                    let rc = self.comp_at(root).unwrap();
                    let pts = fixed(rc.points());
                    let tile = self.io_tile(i);
                    let tile = if tile.len() == pts.len() {
                        tile
                    } else {
                        vec![1; pts.len()]
                    };
                    let ntiles: Vec<f64> = pts
                        .iter()
                        .zip(&tile)
                        .map(|(&n, &t)| n.div_ceil(t.max(1)) as f64)
                        .collect();
                    let np = pts.len();
                    let space: Vec<u64> = (0..np + rc.red.len())
                        .map(|q| if q < np { 1u64 << q } else { 0 })
                        .collect();
                    let mut leaves = Vec::new();
                    if root == i {
                        self.io_leaves(i, &space, &mut leaves);
                    } else {
                        // epilogue map: its own reads are per point of the
                        // same space (identity-shaped), plus the reduction's
                        let ep_space: Vec<u64> = (0..c.points().len())
                            .map(|q| if q < np { 1u64 << q } else { 0 })
                            .collect();
                        self.io_leaves(i, &ep_space, &mut leaves);
                    }
                    let mut reads: Vec<(TensorId, f64)> = Vec::new();
                    for Leaf {
                        storage: s,
                        mask: m,
                        gather,
                    } in leaves
                    {
                        let size = numel(&self.nodes[s].shape) as f64;
                        let words = match gather {
                            None => {
                                size * (0..np)
                                    .filter(|p| m & (1 << p) == 0)
                                    .map(|p| ntiles[p])
                                    .product::<f64>()
                            }
                            Some((extent, idx)) => {
                                // per workgroup: one slice per distinct index
                                // value among its points (at most `extent`)
                                let per_tile: f64 = (0..np)
                                    .filter(|p| idx & (1 << p) != 0)
                                    .map(|p| tile[p] as f64)
                                    .product();
                                let slice = size / extent as f64;
                                let own = m & !idx;
                                slice
                                    * per_tile.min(extent as f64)
                                    * (0..np)
                                        .filter(|p| idx & (1 << p) != 0)
                                        .map(|p| ntiles[p])
                                        .product::<f64>()
                                    * (0..np)
                                        .filter(|p| own & (1 << p) == 0 && idx & (1 << p) == 0)
                                        .map(|p| ntiles[p])
                                        .product::<f64>()
                            }
                        };
                        match reads.iter_mut().find(|r| r.0.n() == s) {
                            Some(r) => r.1 = r.1.max(words),
                            None => reads.push((TensorId(s as u32), words)),
                        }
                    }
                    kernels.push(KernelIo {
                        node: TensorId(i as u32),
                        tile,
                        reads,
                        write: numel(&c.out) as f64,
                    });
                }
                Op::Scan(_) | Op::Operator(_) => {
                    // streamed once: inputs read, output written
                    let reads = self
                        .deps(i)
                        .into_iter()
                        .map(|d| {
                            let s = self.io_storage(d);
                            (TensorId(s as u32), numel(&self.nodes[s].shape) as f64)
                        })
                        .collect();
                    kernels.push(KernelIo {
                        node: TensorId(i as u32),
                        tile: Vec::new(),
                        reads,
                        write: numel(&self.nodes[i].shape) as f64,
                    });
                }
                _ => {}
            }
        }
        let compulsory = self.io_compulsory(outputs);
        let hbl = self.io_hbl(outputs, mem);
        IoReport {
            kernels,
            compulsory,
            hbl,
        }
    }

    /// Every live graph input read once, every output written once.
    pub fn io_compulsory(&self, outputs: &[TensorId]) -> f64 {
        let live = self.live(outputs);
        // an input read only through gathers: at most one slice per index
        // point can be touched (only the touched ones are compulsory)
        let mut cap = vec![f64::INFINITY; self.nodes.len()];
        let mut plain = vec![false; self.nodes.len()];
        for c in (0..self.nodes.len())
            .filter(|&i| live[i])
            .filter_map(|i| self.comp_at(i))
        {
            for a in &c.args {
                let s = self.io_storage(a.src.n());
                match &a.view.gather {
                    None => plain[s] = true,
                    Some(g) => {
                        let shape = fixed(&self.nodes[s].shape);
                        let d = g.coefs.iter().position(|k| *k != 0).unwrap_or(0);
                        let ix = &c.args[g.arg];
                        let idx_pts = numel(&self.nodes[self.io_storage(ix.src.n())].shape) as f64;
                        let slice = numel(&self.nodes[s].shape) as f64 / shape[d].max(1) as f64;
                        let c = if cap[s].is_finite() { cap[s] } else { 0.0 };
                        cap[s] = c + slice * idx_pts;
                    }
                }
            }
        }
        let ins: f64 = (0..self.nodes.len())
            .filter(|&i| live[i] && matches!(self.nodes[i].op, Op::Input(_)))
            .map(|i| {
                let n = numel(&self.nodes[i].shape) as f64;
                if plain[i] {
                    n
                } else {
                    n.min(cap[i])
                }
            })
            .sum();
        let mut outs: Vec<usize> = outputs.iter().map(|o| self.io_storage(o.n())).collect();
        outs.sort_unstable();
        outs.dedup();
        ins + outs
            .iter()
            .map(|&o| numel(&self.nodes[o].shape) as f64)
            .sum::<f64>()
    }

    /// The generic lower bound: max over live reductions of the HBL bound
    /// (module docs), at fast-memory size `mem` words.
    pub fn io_hbl(&self, outputs: &[TensorId], mem: f64) -> f64 {
        let live = self.live(outputs);
        let is_out = |i: usize| outputs.iter().any(|o| self.io_storage(o.n()) == i);
        let mut best = 0.0f64;
        for i in (0..self.nodes.len()).filter(|&i| live[i]) {
            let Some(c) = self.comp_at(i) else { continue };
            if c.monoid.is_none() || c.red.is_empty() {
                continue;
            }
            let space = fixed(&c.space());
            let np = c.points().len();
            // projections: coordinate sets read by input args through
            // pick views (each source dim one coordinate or broadcast)
            let mut projs: Vec<u64> = Vec::new();
            for a in &c.args {
                let src = self.io_storage(a.src.n());
                if !matches!(self.nodes[src].op, Op::Input(_)) || a.view.gather.is_some() {
                    continue;
                }
                let mut m = 0u64;
                let mut ok = true;
                for f in &a.view.forms {
                    match f.coefs.iter().filter(|c| **c != 0).count() {
                        0 => {}
                        1 => {
                            let q = f.coefs.iter().position(|c| *c != 0).unwrap();
                            m |= 1 << q;
                        }
                        _ => ok = false,
                    }
                }
                if ok && m != 0 && !projs.contains(&m) {
                    projs.push(m);
                }
            }
            if is_out(i) {
                let m = (0..np).fold(0u64, |m, q| m | 1 << q);
                if m != 0 && !projs.contains(&m) {
                    projs.push(m);
                }
            }
            let ops: f64 = space.iter().map(|&n| n as f64).product();
            if let Some(f) = hbl_ops_per_stretch(&projs, space.len(), 2.0 * mem) {
                best = best.max(mem * (ops / f).floor());
            }
        }
        best
    }
}

/// Max body evaluations one stretch with `budget` available words can
/// perform: `min_s ∏ (budget·sⱼ/σ)^{sⱼ}` over HBL exponents `s` (grid of
/// twelfths, each feasible point a valid bound; the min the best). `None`
/// if some coordinate is read by no projection (no bound).
fn hbl_ops_per_stretch(projs: &[u64], rank: usize, budget: f64) -> Option<f64> {
    if projs.is_empty() || projs.len() > 5 {
        return None;
    }
    let steps = 12usize;
    let k = projs.len();
    let mut best: Option<f64> = None;
    let mut s = vec![0usize; k];
    loop {
        let feasible = (0..rank).all(|q| {
            projs
                .iter()
                .zip(&s)
                .filter(|(p, _)| *p & (1 << q) != 0)
                .map(|(_, &x)| x)
                .sum::<usize>()
                >= steps
        });
        if feasible {
            let sigma: f64 = s.iter().map(|&x| x as f64 / steps as f64).sum();
            // maximize ∏ x_j^{s_j} subject to Σ x_j ≤ budget: x_j ∝ s_j
            let f: f64 = s
                .iter()
                .filter(|&&x| x > 0)
                .map(|&x| {
                    let sj = x as f64 / steps as f64;
                    (budget * sj / sigma).powf(sj)
                })
                .product();
            best = Some(best.map_or(f, |b: f64| b.min(f)));
        }
        // next grid point
        let mut j = 0;
        loop {
            if j == k {
                return best;
            }
            s[j] += 1;
            if s[j] <= steps {
                break;
            }
            s[j] = 0;
            j += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hbl_matmul_exponents() {
        // (i, j, k): A = {i,k}, B = {k,j}, C = {i,j}; σ = 3/2
        let f = hbl_ops_per_stretch(&[0b101, 0b110, 0b011], 3, 3.0).unwrap();
        // x = 1 each, product of 1^(1/2) = 1
        assert!((f - 1.0).abs() < 1e-9);
        let f = hbl_ops_per_stretch(&[0b101, 0b110, 0b011], 3, 300.0).unwrap();
        assert!((f - 1000.0).abs() < 1e-6, "{f}");
    }

    #[test]
    fn matmul_bound_below_cosma_and_staged_traffic_above_it() {
        let (m, n, k) = (512, 512, 512);
        let mut g = TensorGraph::new();
        let a = g.input(&[m, k]);
        let b = g.input(&[k, n]);
        let c = g.matmul(a, b).unwrap();
        g.schedule_auto(&[c]);
        let mem = 4096.0;
        let r = g.io_report(&[c], mem);
        let cosma = bounds::matmul(m as f64, n as f64, k as f64, mem);
        // generic HBL is a valid but weaker bound than COSMA's
        assert!(
            r.hbl > 0.5 * cosma && r.hbl <= cosma,
            "{} vs {cosma}",
            r.hbl
        );
        assert!(r.achieved() >= cosma, "{} < {cosma}", r.achieved());
        assert_eq!(r.kernels.len(), 1);
    }

    #[test]
    fn map_traffic_is_compulsory() {
        let mut g = TensorGraph::new();
        let x = g.input(&[64, 64]);
        let y = g.map(&[x], |v| v[0] * v[0]).unwrap();
        g.schedule_auto(&[y]);
        let r = g.io_report(&[y], 1024.0);
        assert_eq!(r.achieved(), r.compulsory);
    }
}
