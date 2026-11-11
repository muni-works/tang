//! Small reverse-mode tensor tape. Dense contractions use tang-compute; scalar nonlinear
//! Jacobians use tang-ad. Values are host resident so frozen experts need no optimizer state.
use std::sync::Arc;
use tang_compute::ComputeDevice;

pub type Id = usize;
#[derive(Clone)]
enum Op {
    Leaf,
    Unary(Id, Vec<f32>),
    Add(Id, Id),
    Mul(Id, Id),
    Gather(Id, Vec<usize>),
    Join(Vec<Id>),
    Linear {
        x: Id,
        w: Id,
        m: usize,
        k: usize,
        n: usize,
    },
    Norm {
        x: Id,
        w: Id,
        dim: usize,
        inv: Vec<f32>,
    },
    Softmax {
        x: Id,
        dim: usize,
    },
    Attention {
        q: Id,
        k: Id,
        v: Id,
        nh: usize,
        nkv: usize,
        hd: usize,
        visible: Vec<Vec<usize>>,
        probs: Vec<Vec<f32>>,
    },
}
struct Node {
    value: Arc<Vec<f32>>,
    op: Op,
    track: bool,
}
/// Device copies of frozen leaf values (head, embedding rows, dequantized experts), keyed by
/// the `Arc`'s address and kept alive with it, so they upload once per run, not per op.
pub struct DeviceCache<D: ComputeDevice> {
    map: std::cell::RefCell<std::collections::HashMap<usize, (Arc<Vec<f32>>, Arc<D::Buffer>)>>,
}
impl<D: ComputeDevice> Default for DeviceCache<D> {
    fn default() -> Self {
        Self {
            map: Default::default(),
        }
    }
}
pub struct Tape<'a, D: ComputeDevice> {
    pub dev: &'a D,
    nodes: Vec<Node>,
    cache: Option<&'a DeviceCache<D>>,
    /// Trainable weights' device copies for this tape only (they change every step).
    local: std::cell::RefCell<std::collections::HashMap<usize, Arc<D::Buffer>>>,
}
impl<'a, D: ComputeDevice> Tape<'a, D> {
    pub fn new(dev: &'a D) -> Self {
        Self {
            dev,
            nodes: Vec::new(),
            cache: None,
            local: Default::default(),
        }
    }
    pub fn with_cache(dev: &'a D, cache: &'a DeviceCache<D>) -> Self {
        Self {
            dev,
            nodes: Vec::new(),
            cache: Some(cache),
            local: Default::default(),
        }
    }
    /// The device copy of node `i`: cached for untracked leaves, else a fresh upload.
    fn device(&self, i: Id) -> Arc<D::Buffer> {
        let node = &self.nodes[i];
        // Only values shared with an owner outside the tape (Weights); a tape constant's
        // fresh `Arc` would leak an entry per window.
        let shared = Arc::strong_count(&node.value) > 1;
        let key = Arc::as_ptr(&node.value) as usize;
        if shared && node.track && matches!(node.op, Op::Leaf) {
            let mut local = self.local.borrow_mut();
            return local
                .entry(key)
                .or_insert_with(|| Arc::new(self.dev.upload_f32(&node.value)))
                .clone();
        }
        let Some(cache) = self
            .cache
            .filter(|_| shared && !node.track && matches!(node.op, Op::Leaf))
        else {
            return Arc::new(self.dev.upload_f32(&node.value));
        };
        if let Some((_, b)) = cache.map.borrow().get(&key) {
            return b.clone();
        }
        let b = Arc::new(self.dev.upload_f32(&node.value));
        cache
            .map
            .borrow_mut()
            .insert(key, (node.value.clone(), b.clone()));
        b
    }
    fn push(&mut self, value: Vec<f32>, op: Op, parents: &[Id]) -> Id {
        let track = parents.iter().any(|&i| self.nodes[i].track);
        let i = self.nodes.len();
        self.nodes.push(Node {
            value: Arc::new(value),
            op,
            track,
        });
        i
    }
    pub fn leaf(&mut self, value: Arc<Vec<f32>>, track: bool) -> Id {
        let i = self.nodes.len();
        self.nodes.push(Node {
            value,
            op: Op::Leaf,
            track,
        });
        i
    }
    pub fn constant(&mut self, v: Vec<f32>) -> Id {
        self.leaf(Arc::new(v), false)
    }
    pub fn data(&self, i: Id) -> &[f32] {
        &self.nodes[i].value
    }
    pub fn gather(&mut self, a: Id, idx: Vec<usize>) -> Id {
        use rayon::prelude::*;
        let src = self.data(a);
        let v = idx
            .par_iter()
            .with_min_len(1 << 16)
            .map(|&j| src[j])
            .collect();
        self.push(v, Op::Gather(a, idx), &[a])
    }
    pub fn join(&mut self, a: &[Id]) -> Id {
        let v = a
            .iter()
            .flat_map(|&i| self.data(i).iter().copied())
            .collect();
        self.push(v, Op::Join(a.to_vec()), a)
    }
    pub fn scale(&mut self, a: Id, s: f32) -> Id {
        let v = self.data(a).iter().map(|v| v * s).collect();
        self.push(v, Op::Unary(a, vec![s; self.data(a).len()]), &[a])
    }
    pub fn add(&mut self, a: Id, b: Id) -> Id {
        assert_eq!(self.data(a).len(), self.data(b).len());
        let v = self
            .data(a)
            .iter()
            .zip(self.data(b))
            .map(|(a, b)| a + b)
            .collect();
        self.push(v, Op::Add(a, b), &[a, b])
    }
    pub fn mul(&mut self, a: Id, b: Id) -> Id {
        assert_eq!(self.data(a).len(), self.data(b).len());
        let v = self
            .data(a)
            .iter()
            .zip(self.data(b))
            .map(|(a, b)| a * b)
            .collect();
        self.push(v, Op::Mul(a, b), &[a, b])
    }
    pub fn nonlinear(&mut self, a: Id, silu: bool) -> Id {
        let v = self
            .data(a)
            .iter()
            .map(|&x| {
                let p = 1.0 / (1.0 + (-x).exp());
                if silu {
                    x * p
                } else {
                    p
                }
            })
            .collect();
        use rayon::prelude::*;
        let parts: Vec<Vec<f32>> = self
            .data(a)
            .par_chunks(1024)
            .map(|chunk| {
                let x: Vec<f64> = chunk.iter().map(|&v| v as f64).collect();
                let g = tang_ad::grad(
                    |vars| {
                        let terms: Vec<_> = vars
                            .iter()
                            .map(|x| {
                                // Equivalent sigmoid with a bounded intermediate at extreme logits.
                                let p = (x * 0.5).tanh() * 0.5 + 0.5;
                                if silu {
                                    x * &p
                                } else {
                                    p
                                }
                            })
                            .collect();
                        terms.into_iter().reduce(|a, b| a + b).unwrap()
                    },
                    &x,
                );
                g.as_slice().iter().map(|&v| v as f32).collect()
            })
            .collect();
        let jac: Vec<f32> = parts.concat();
        self.push(v, Op::Unary(a, jac), &[a])
    }
    pub fn linear(&mut self, x: Id, w: Id, m: usize, k: usize, n: usize) -> Id {
        assert_eq!(self.data(x).len(), m * k);
        assert_eq!(self.data(w).len(), n * k);
        let a = self.dev.upload_f32(self.data(x));
        let b = self.device(w);
        let c = self.dev.matmul_b_transposed(&a, &b, m, k, n);
        self.push(self.dev.download(&c), Op::Linear { x, w, m, k, n }, &[x, w])
    }
    pub fn norm(&mut self, x: Id, w: Id, dim: usize, eps: f32) -> Id {
        let gamma = self.data(w);
        assert_eq!(self.data(x).len() % dim, 0);
        assert_eq!(gamma.len() % dim, 0);
        use rayon::prelude::*;
        let mut out = vec![0.0; self.data(x).len()];
        let inv: Vec<f32> = out
            .par_chunks_mut(dim)
            .zip(self.data(x).par_chunks(dim))
            .enumerate()
            .map(|(r, (o, row))| {
                let z = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / dim as f32 + eps).sqrt();
                for (i, &v) in row.iter().enumerate() {
                    o[i] = v * z * gamma[(r * dim + i) % gamma.len()];
                }
                z
            })
            .collect();
        self.push(out, Op::Norm { x, w, dim, inv }, &[x, w])
    }
    pub fn softmax(&mut self, x: Id, dim: usize) -> Id {
        let mut out = Vec::with_capacity(self.data(x).len());
        for row in self.data(x).chunks(dim) {
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let p: Vec<_> = row.iter().map(|v| (v - mx).exp()).collect();
            let sum = p.iter().sum::<f32>();
            out.extend(p.iter().map(|v| v / sum));
        }
        self.push(out, Op::Softmax { x, dim }, &[x])
    }
    pub fn rope(
        &mut self,
        x: Id,
        heads: usize,
        hd: usize,
        rot: usize,
        base: f32,
        pos: &[usize],
    ) -> Id {
        // A sparse linear map implemented as gather/mul/add, including untouched dimensions.
        let mut idx1 = Vec::new();
        let mut idx2 = Vec::new();
        let mut c1 = Vec::new();
        let mut c2 = Vec::new();
        for (t, &p) in pos.iter().enumerate() {
            for h in 0..heads {
                for j in 0..hd {
                    let off = (t * heads + h) * hd;
                    if j < rot {
                        let f = j % (rot / 2);
                        let theta = p as f32 * base.powf(-2.0 * f as f32 / rot as f32);
                        let (s, c) = theta.sin_cos();
                        idx1.push(off + j);
                        c1.push(c);
                        idx2.push(
                            off + if j < rot / 2 {
                                j + rot / 2
                            } else {
                                j - rot / 2
                            },
                        );
                        c2.push(if j < rot / 2 { -s } else { s });
                    } else {
                        idx1.push(off + j);
                        idx2.push(off + j);
                        c1.push(1.0);
                        c2.push(0.0);
                    }
                }
            }
        }
        // Keep construction deliberately explicit: transpose of this mapping is backward RoPE.
        let a = self.gather(x, idx1);
        let b = self.gather(x, idx2);
        let ca = self.constant(c1);
        let cb = self.constant(c2);
        let a = self.mul(a, ca);
        let b = self.mul(b, cb);
        self.add(a, b)
    }
    pub fn attention(
        &mut self,
        q: Id,
        k: Id,
        v: Id,
        nh: usize,
        nkv: usize,
        hd: usize,
        visible: Vec<Vec<usize>>,
    ) -> Id {
        use rayon::prelude::*;
        let scale = 1.0 / (hd as f32).sqrt();
        let (qd, kd, vd) = (self.data(q), self.data(k), self.data(v));
        // One (query row, head) per task; rows see up to the whole teacher prefix.
        let cells: Vec<(Vec<f32>, Vec<f32>)> = (0..visible.len() * nh)
            .into_par_iter()
            .map(|i| {
                let (t, h) = (i / nh, i % nh);
                let rows = &visible[t];
                let qh = &qd[i * hd..(i + 1) * hd];
                let g = h / (nh / nkv);
                let scores: Vec<_> = rows
                    .iter()
                    .map(|&r| {
                        qh.iter()
                            .zip(&kd[(r * nkv + g) * hd..(r * nkv + g + 1) * hd])
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                            * scale
                    })
                    .collect();
                let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut p: Vec<_> = scores.iter().map(|v| (v - mx).exp()).collect();
                let z = p.iter().sum::<f32>();
                for x in &mut p {
                    *x /= z;
                }
                let mut o = vec![0.0; hd];
                for (j, &r) in rows.iter().enumerate() {
                    let vr = &vd[(r * nkv + g) * hd..(r * nkv + g + 1) * hd];
                    for d in 0..hd {
                        o[d] += p[j] * vr[d];
                    }
                }
                (o, p)
            })
            .collect();
        let mut out = Vec::with_capacity(visible.len() * nh * hd);
        let mut probs = Vec::with_capacity(cells.len());
        for (o, p) in cells {
            out.extend(o);
            probs.push(p);
        }
        self.push(
            out,
            Op::Attention {
                q,
                k,
                v,
                nh,
                nkv,
                hd,
                visible,
                probs,
            },
            &[q, k, v],
        )
    }
    /// Return gradients for leaves; multiple seeds implement the weighted recursive objective.
    pub fn backward(&self, seeds: &[(Id, Vec<f32>)]) -> Vec<Option<Vec<f32>>> {
        let mut grads: Vec<Option<Vec<f32>>> = vec![None; self.nodes.len()];
        let put = |grads: &mut Vec<Option<Vec<f32>>>, id: Id, g: Vec<f32>| {
            if !self.nodes[id].track {
                return;
            }
            let dst = grads[id].get_or_insert_with(|| vec![0.0; self.data(id).len()]);
            assert_eq!(dst.len(), g.len());
            for (d, v) in dst.iter_mut().zip(g) {
                *d += v;
            }
        };
        for (id, g) in seeds {
            put(&mut grads, *id, g.clone());
        }
        for i in (0..self.nodes.len()).rev() {
            if matches!(self.nodes[i].op, Op::Leaf) {
                continue;
            }
            let Some(g) = grads[i].take() else {
                continue;
            };
            match &self.nodes[i].op {
                Op::Leaf => {}
                Op::Unary(a, j) => put(
                    &mut grads,
                    *a,
                    g.iter().zip(j).map(|(g, j)| g * j).collect(),
                ),
                Op::Add(a, b) => {
                    put(&mut grads, *a, g.clone());
                    put(&mut grads, *b, g);
                }
                Op::Mul(a, b) => {
                    put(
                        &mut grads,
                        *a,
                        g.iter().zip(self.data(*b)).map(|(g, v)| g * v).collect(),
                    );
                    put(
                        &mut grads,
                        *b,
                        g.iter().zip(self.data(*a)).map(|(g, v)| g * v).collect(),
                    );
                }
                Op::Gather(a, idx) => {
                    let mut ga = vec![0.0; self.data(*a).len()];
                    for (&j, v) in idx.iter().zip(g) {
                        ga[j] += v;
                    }
                    put(&mut grads, *a, ga);
                }
                Op::Join(a) => {
                    let mut start = 0;
                    for &a in a {
                        let end = start + self.data(a).len();
                        put(&mut grads, a, g[start..end].to_vec());
                        start = end;
                    }
                }
                Op::Linear { x, w, m, k, n } => {
                    let gb = self.dev.upload_f32(&g);
                    if self.nodes[*x].track {
                        let wb = self.device(*w);
                        let gx = self.dev.matmul(&gb, &wb, *m, *n, *k);
                        put(&mut grads, *x, self.dev.download(&gx));
                    }
                    if self.nodes[*w].track {
                        let xb = self.dev.upload_f32(self.data(*x));
                        let gw = self.dev.matmul_a_transposed(&gb, &xb, *n, *m, *k);
                        put(&mut grads, *w, self.dev.download(&gw));
                    }
                }
                Op::Norm { x, w, dim, inv } => {
                    use rayon::prelude::*;
                    let (xd, wd, dim) = (self.data(*x), self.data(*w), *dim);
                    let wl = wd.len();
                    let mut gx = vec![0.0; xd.len()];
                    // gx rows in parallel; gw (per-feature sums) folded per thread.
                    let gw = gx
                        .par_chunks_mut(dim)
                        .zip(xd.par_chunks(dim))
                        .enumerate()
                        .fold(
                            || vec![0.0f32; wl],
                            |mut gw, (r, (gxr, row))| {
                                let z = inv[r];
                                let start = r * dim;
                                let dot = (0..dim)
                                    .map(|j| g[start + j] * wd[(start + j) % wl] * row[j])
                                    .sum::<f32>();
                                for j in 0..dim {
                                    let wi = (start + j) % wl;
                                    gxr[j] = z * g[start + j] * wd[wi]
                                        - row[j] * z * z * z * dot / dim as f32;
                                    gw[wi] += g[start + j] * row[j] * z;
                                }
                                gw
                            },
                        )
                        .reduce(
                            || vec![0.0f32; wl],
                            |mut a, b| {
                                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                                a
                            },
                        );
                    put(&mut grads, *x, gx);
                    put(&mut grads, *w, gw);
                }
                Op::Softmax { x, dim } => {
                    let mut gx = vec![0.0; g.len()];
                    for (r, p) in self.data(i).chunks(*dim).enumerate() {
                        let start = r * dim;
                        let dot = (0..*dim).map(|j| g[start + j] * p[j]).sum::<f32>();
                        for j in 0..*dim {
                            gx[start + j] = p[j] * (g[start + j] - dot);
                        }
                    }
                    put(&mut grads, *x, gx);
                }
                Op::Attention {
                    q,
                    k,
                    v,
                    nh,
                    nkv,
                    hd,
                    visible,
                    probs,
                } => {
                    use rayon::prelude::*;
                    let scale = 1.0 / (*hd as f32).sqrt();
                    let (qd, kd, vd) = (self.data(*q), self.data(*k), self.data(*v));
                    let (nh, nkv, hd) = (*nh, *nkv, *hd);
                    let zero = || {
                        (
                            vec![0.0f32; qd.len()],
                            vec![0.0f32; kd.len()],
                            vec![0.0f32; vd.len()],
                        )
                    };
                    // Per-thread accumulators over (query row, head) tasks, summed at the end.
                    let tasks = visible.len() * nh;
                    let (gq, gk, gv) = (0..tasks)
                        .into_par_iter()
                        // Few, large splits: each accumulator spans every K/V row.
                        .with_min_len(tasks.div_ceil(rayon::current_num_threads()).max(1))
                        .fold(zero, |(mut gq, mut gk, mut gv), i| {
                            let (t, h) = (i / nh, i % nh);
                            let rows = &visible[t];
                            let group = h / (nh / nkv);
                            let qo = i * hd;
                            let p = &probs[i];
                            let dp: Vec<_> = rows
                                .iter()
                                .map(|&r| {
                                    let ko = (r * nkv + group) * hd;
                                    (0..hd).map(|d| g[qo + d] * vd[ko + d]).sum::<f32>()
                                })
                                .collect();
                            let dot = p.iter().zip(&dp).map(|(a, b)| a * b).sum::<f32>();
                            for (j, &r) in rows.iter().enumerate() {
                                let ko = (r * nkv + group) * hd;
                                let ds = p[j] * (dp[j] - dot) * scale;
                                for d in 0..hd {
                                    gq[qo + d] += ds * kd[ko + d];
                                    gk[ko + d] += ds * qd[qo + d];
                                    gv[ko + d] += p[j] * g[qo + d];
                                }
                            }
                            (gq, gk, gv)
                        })
                        .reduce(zero, |mut a, b| {
                            for (x, y) in [(&mut a.0, &b.0), (&mut a.1, &b.1), (&mut a.2, &b.2)] {
                                x.iter_mut().zip(y).for_each(|(x, y)| *x += y);
                            }
                            a
                        });
                    put(&mut grads, *q, gq);
                    put(&mut grads, *k, gk);
                    put(&mut grads, *v, gv);
                }
            }
        }
        grads
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use tang_compute::CpuDevice;
    fn check<F: Fn(&mut Tape<CpuDevice>, Id) -> Id>(x: &[f32], f: F, tol: f32) {
        let dev = CpuDevice::new();
        let mut t = Tape::new(&dev);
        let input = t.leaf(Arc::new(x.to_vec()), true);
        let y = f(&mut t, input);
        let g = t.backward(&[(y, vec![1.0; t.data(y).len()])])[input]
            .take()
            .unwrap();
        for j in 0..x.len() {
            let mut losses = Vec::new();
            for sign in [-1.0, 1.0] {
                let mut v = x.to_vec();
                v[j] += sign * 0.002;
                let mut t = Tape::new(&dev);
                let input = t.leaf(Arc::new(v), true);
                let y = f(&mut t, input);
                losses.push(t.data(y).iter().sum::<f32>());
            }
            let fd = (losses[1] - losses[0]) / 0.004;
            assert!(
                (g[j] - fd).abs() < tol,
                "index {j}: AD {} finite-diff {fd}",
                g[j]
            );
        }
    }
    #[test]
    fn nonlinear_norm_linear_and_recursive_gradients() {
        check(
            &[0.3, -0.2, 0.5, 0.7],
            |t, x| {
                let w = t.constant(vec![0.1, 0.3, -0.5, 0.2]);
                let y = t.linear(x, w, 2, 2, 2);
                let gamma = t.constant(vec![1.1, 0.9]);
                let y = t.norm(y, gamma, 2, 1e-3);
                let y = t.nonlinear(y, true);
                let z = t.linear(y, w, 2, 2, 2);
                t.add(y, z)
            },
            0.004,
        );
    }
    #[test]
    fn gamma_repeats_by_stream() {
        check(
            &[0.3, 0.9, 1.1, 0.8],
            |t, gamma| {
                let x = t.constant(vec![0.1, 0.3, -0.5, 0.2, 0.7, 0.4, 0.8, -0.1]);
                t.norm(x, gamma, 2, 1e-3)
            },
            0.002,
        );
    }
    #[test]
    fn linear_weight_gradients() {
        check(
            &[0.3, -0.2, 0.5, 0.7],
            |t, w| {
                let x = t.constant(vec![0.2, -0.7, 0.8, 0.3, 0.1, -0.2]);
                t.linear(x, w, 3, 2, 2)
            },
            0.002,
        );
    }
    #[test]
    fn repeated_gathers_sum_gradients() {
        check(
            &[0.2, 0.3],
            |t, x| {
                let y = t.gather(x, vec![0, 1, 0, 0]);
                let z = t.softmax(y, 2);
                t.mul(y, z)
            },
            0.002,
        );
    }
    #[test]
    fn rope_gradient_and_reference() {
        check(
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            |t, x| t.rope(x, 1, 6, 4, 1000.0, &[7]),
            0.001,
        );
        let dev = CpuDevice::new();
        let mut t = Tape::new(&dev);
        let a = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let x = t.constant(a.clone());
        let y = t.rope(x, 1, 6, 4, 1000.0, &[7]);
        let mut expected = a;
        crate::flash::reference::rope_neox(&mut expected, 7, 4, 1000.0);
        for (a, b) in t.data(y).iter().zip(expected) {
            assert!((a - b).abs() < 1e-6);
        }
    }
    #[test]
    fn attention_qkv_and_chain_visibility() {
        for which in 0..3 {
            check(
                &[0.1, 0.4, -0.3, 0.2, 0.3, -0.5],
                |t, x| {
                    let other = t.constant(vec![0.2, -0.1, 0.1, 0.3, -0.2, 0.4]);
                    let (q, k, v) = match which {
                        0 => (x, other, other),
                        1 => (other, x, other),
                        _ => (other, other, x),
                    };
                    t.attention(q, k, v, 1, 1, 2, vec![vec![0], vec![0, 1], vec![0, 2]])
                },
                0.001,
            );
        }
    }
    #[test]
    fn grouped_attention_gradients() {
        check(
            &[0.1, 0.4, -0.3, 0.2, 0.3, -0.5, 0.2, 0.4],
            |t, q| {
                let k = t.constant(vec![0.2, -0.1, 0.1, 0.3]);
                let v = t.constant(vec![0.3, -0.7, 0.2, 0.5]);
                t.attention(q, k, v, 2, 1, 2, vec![vec![0], vec![0, 1]])
            },
            0.001,
        );
    }
}
