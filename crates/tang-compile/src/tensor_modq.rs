//! Integer tensors mod q, with a secrecy label on every tensor.
//!
//! A sibling of [`TensorGraph`](super::TensorGraph) for lattice
//! cryptography: same affine [`View`]s, but values are `u32` residues mod a
//! prime `q < 2^31` and every node is `Public` or `Secret`.
//!
//! The shape of the IR is the constant-time argument. Views are fixed affine
//! maps and shapes are static, so a map's control flow and addresses cannot
//! depend on any value; its scalar ops ([`Body`]) are branch-free (Montgomery
//! multiplication, mask-based conditional subtraction and select). Only three
//! ops look at data, and [`ModGraph::check`] polices them:
//!
//! - [`ModGraph::gather`] reads a table at a runtime index — the index must be
//!   `Public`.
//! - [`ModGraph::filter`] keeps the values below a bound (rejection sampling)
//!   — its input must be `Public`.
//! - [`ModGraph::declassify`] turns a `Secret` into a `Public`, naming the
//!   proof that makes it safe.
//!
//! and a tensor the caller declares `Public` must not carry a secret.
//!
//! ```
//! use tang_compile::tensor::modq::{Label, ModGraph};
//!
//! let mut g = ModGraph::new(17);
//! let s = g.input(&[4], Label::Secret);
//! let t = g.map(&[4], &[(s, ModGraph::id(1))], |b| {
//!     let x = b.arg(0);
//!     b.add(x, x)
//! });
//! assert_eq!(g.eval(&[vec![1, 9, 16, 0]], &[t]), vec![vec![2, 1, 15, 0]]);
//! assert!(g.check(&[(t, Label::Public)]).is_err());   // a secret, published
//! assert!(g.check(&[(t, Label::Secret)]).is_ok());
//! ```

use super::{Form, View};

#[path = "tensor_modq_c.rs"]
pub mod c;
#[path = "tensor_modq_gpu.rs"]
pub mod gpu;

/// Who may learn a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Label {
    Public,
    Secret,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModId(u32);

/// One scalar op inside a map body. Every op runs in constant time.
#[derive(Clone, Copy, Debug)]
enum BOp {
    Arg(usize),
    Lit(u32),
    /// Coordinate along a point dim (public by construction).
    Coord(usize),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    /// `a` where `c == 1`, `b` where `c == 0`, by mask.
    Sel(usize, usize, usize),
    /// round(x·2^d / q) mod 2^d: keep the top d bits of a residue.
    Compress(usize, u32),
    /// round(y·q / 2^d): back to a residue.
    Decompress(usize, u32),
    /// round(x / 2^d), ties down: the high part when the low d bits are dropped.
    Power2Round(usize, u32),
    // Word ops on raw u32, for hashing and bit extraction. Shift amounts are
    // taken mod 32; a shift is constant-time on every target we lower to.
    Xor(usize, usize),
    And(usize, usize),
    Or(usize, usize),
    Not(usize),
    Shl(usize, usize),
    Shr(usize, usize),
    /// 1 if equal, else 0, by mask.
    Eq(usize, usize),
    /// 1 if a < b, else 0, for a, b < 2^31.
    Lt(usize, usize),
    /// floor(a·b / 2^32).
    MulHi(usize, usize),
    /// HighBits / LowBits of FIPS 204's Decompose for α = 2γ₂: (r₁, r₀ mod q).
    HighBits(usize, u32),
    LowBits(usize, u32),
}

/// A branch-free scalar program over residues mod q.
pub struct Body {
    ops: Vec<BOp>,
}

/// A value inside a [`Body`].
#[derive(Clone, Copy, Debug)]
pub struct BVal(usize);

impl Body {
    fn push(&mut self, op: BOp) -> BVal {
        self.ops.push(op);
        BVal(self.ops.len() - 1)
    }
    pub fn arg(&mut self, i: usize) -> BVal {
        self.push(BOp::Arg(i))
    }
    /// A raw word. Mod-q ops expect residues, so a literal fed to one must be < q.
    pub fn lit(&mut self, v: u32) -> BVal {
        self.push(BOp::Lit(v))
    }
    pub fn coord(&mut self, d: usize) -> BVal {
        self.push(BOp::Coord(d))
    }
    pub fn add(&mut self, a: BVal, b: BVal) -> BVal {
        self.push(BOp::Add(a.0, b.0))
    }
    pub fn sub(&mut self, a: BVal, b: BVal) -> BVal {
        self.push(BOp::Sub(a.0, b.0))
    }
    pub fn mul(&mut self, a: BVal, b: BVal) -> BVal {
        self.push(BOp::Mul(a.0, b.0))
    }
    pub fn sel(&mut self, c: BVal, a: BVal, b: BVal) -> BVal {
        self.push(BOp::Sel(c.0, a.0, b.0))
    }
    pub fn compress(&mut self, x: BVal, d: u32) -> BVal {
        self.push(BOp::Compress(x.0, d))
    }
    pub fn decompress(&mut self, y: BVal, d: u32) -> BVal {
        self.push(BOp::Decompress(y.0, d))
    }
    pub fn power2round(&mut self, x: BVal, d: u32) -> BVal {
        self.push(BOp::Power2Round(x.0, d))
    }
    pub fn xor(&mut self, a: BVal, b: BVal) -> BVal {
        self.push(BOp::Xor(a.0, b.0))
    }
    pub fn and(&mut self, a: BVal, b: BVal) -> BVal {
        self.push(BOp::And(a.0, b.0))
    }
    pub fn or(&mut self, a: BVal, b: BVal) -> BVal {
        self.push(BOp::Or(a.0, b.0))
    }
    pub fn not(&mut self, a: BVal) -> BVal {
        self.push(BOp::Not(a.0))
    }
    pub fn shl(&mut self, a: BVal, by: BVal) -> BVal {
        self.push(BOp::Shl(a.0, by.0))
    }
    pub fn shr(&mut self, a: BVal, by: BVal) -> BVal {
        self.push(BOp::Shr(a.0, by.0))
    }
    pub fn eq(&mut self, a: BVal, b: BVal) -> BVal {
        self.push(BOp::Eq(a.0, b.0))
    }
    pub fn lt(&mut self, a: BVal, b: BVal) -> BVal {
        self.push(BOp::Lt(a.0, b.0))
    }
    pub fn mulhi(&mut self, a: BVal, b: BVal) -> BVal {
        self.push(BOp::MulHi(a.0, b.0))
    }
    /// r₁ with r ≡ r₁·2γ₂ + r₀, −γ₂ < r₀ ≤ γ₂ (and r₁ = 0 at the q − 1 edge).
    pub fn high_bits(&mut self, r: BVal, gamma2: u32) -> BVal {
        self.push(BOp::HighBits(r.0, gamma2))
    }
    /// r₀ as a residue.
    pub fn low_bits(&mut self, r: BVal, gamma2: u32) -> BVal {
        self.push(BOp::LowBits(r.0, gamma2))
    }
}

enum MOp {
    /// Input index, and whether it holds residues (reduced mod q) or raw words.
    Input(usize, bool),
    Const(Vec<u32>),
    Reshape(ModId),
    Map {
        args: Vec<(ModId, View)>,
        body: Body,
        /// Which body op is the result.
        out: usize,
    },
    Gather {
        table: ModId,
        index: ModId,
    },
    Filter {
        src: ModId,
        bound: u32,
    },
    Declassify {
        src: ModId,
        proof: &'static str,
    },
}

struct MNode {
    op: MOp,
    shape: Vec<usize>,
    /// Declared label of an input; derived for everything else.
    label: Label,
}

/// Why [`ModGraph::check`] refused a graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TaintError {
    /// A secret decides which table entry is read: a cache channel.
    SecretIndex(ModId),
    /// A secret decides how many values survive: a timing channel.
    SecretFilter(ModId),
    /// A tensor declared public carries a secret.
    SecretOutput(ModId),
}

/// Constant-time arithmetic mod q, Montgomery with R = 2^32.
#[derive(Clone, Copy)]
struct Field {
    q: u32,
    /// −q⁻¹ mod 2^32
    qneg_inv: u32,
    /// R² mod q, to leave the Montgomery domain after one product.
    r2: u32,
    /// floor(2^64 / q), for division by q without a divide instruction.
    barrett: u64,
}

impl Field {
    fn new(q: u32) -> Self {
        assert!(q % 2 == 1 && q < 1 << 31, "q must be odd and below 2^31");
        let mut inv: u32 = 1;
        for _ in 0..5 {
            inv = inv.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(inv)));
        }
        let r2 = ((1u128 << 64) % q as u128) as u32;
        Self {
            q,
            qneg_inv: inv.wrapping_neg(),
            r2,
            barrett: ((1u128 << 64) / q as u128) as u64,
        }
    }
    /// x − q if x ≥ q, for x < 2q, by mask.
    fn csub(&self, x: u32) -> u32 {
        let y = x.wrapping_sub(self.q);
        let mask = 0u32.wrapping_sub(y >> 31);
        y.wrapping_add(self.q & mask)
    }
    fn add(&self, a: u32, b: u32) -> u32 {
        self.csub(a + b)
    }
    fn sub(&self, a: u32, b: u32) -> u32 {
        self.csub(a + self.q - b)
    }
    /// a·b·R⁻¹ mod q.
    fn mont(&self, a: u32, b: u32) -> u32 {
        let t = a as u64 * b as u64;
        let m = (t as u32).wrapping_mul(self.qneg_inv);
        self.csub(((t + m as u64 * self.q as u64) >> 32) as u32)
    }
    fn mul(&self, a: u32, b: u32) -> u32 {
        self.mont(self.mont(a, b), self.r2)
    }
    /// floor(v / q) for v < 2^56: the Barrett estimate is low by at most one,
    /// fixed by mask.
    fn div_q(&self, v: u64) -> u64 {
        let est = ((v as u128 * self.barrett as u128) >> 64) as u64;
        let r = v - est * self.q as u64;
        let ge = (((self.q as u64).wrapping_sub(1).wrapping_sub(r)) >> 63) & 1;
        est + ge
    }
    fn compress(&self, x: u32, d: u32) -> u32 {
        let v = ((x as u64) << d) + (self.q as u64 >> 1);
        (self.div_q(v) as u32) & ((1u32 << d) - 1)
    }
    fn decompress(&self, y: u32, d: u32) -> u32 {
        ((y as u64 * self.q as u64 + (1u64 << d >> 1)) >> d) as u32
    }
    /// FIPS 204 Decompose(r) for α = 2γ₂, branch-free: (r₁, r₀ mod q).
    fn decompose(&self, r: u32, gamma2: u32) -> (u32, u32) {
        let alpha = 2 * gamma2 as u64;
        let m = (1u64 << 40) / alpha;
        let mut r1 = (r as u64 * m) >> 40;
        let rem = r as u64 - r1 * alpha;
        let ge = ((alpha - 1).wrapping_sub(rem) >> 63) & 1; // rem ≥ α
        r1 += ge;
        let rp = r as u64 - r1 * alpha; // in [0, α)
        let up = ((gamma2 as u64).wrapping_sub(rp) >> 63) & 1; // rp > γ₂
        r1 += up;
        let r0 = (rp as i64) - (up as i64) * alpha as i64; // in (−γ₂, γ₂]
                                                           // the edge r − r₀ = q − 1: r₁ = 0, r₀ − 1
        let d = (r1 * alpha) as u32 ^ (self.q - 1);
        let edge = ((d | d.wrapping_neg()) >> 31) ^ 1;
        let emask = 0u32.wrapping_sub(edge);
        let r1 = (r1 as u32) & !emask;
        let r0 = r0 - edge as i64;
        (r1, self.csub((r0 + self.q as i64) as u32))
    }
    /// round(x / 2^d), ties down. d is a constant of the op (public); d = 0
    /// is the identity.
    fn power2round(&self, x: u32, d: u32) -> u32 {
        if d == 0 {
            return x;
        }
        (x + (1 << (d - 1)) - 1) >> d
    }
    fn sel(&self, c: u32, a: u32, b: u32) -> u32 {
        let mask = 0u32.wrapping_sub(c & 1);
        (a & mask) | (b & !mask)
    }
}

/// A graph of integer tensors mod q.
pub struct ModGraph {
    f: Field,
    nodes: Vec<MNode>,
    inputs: usize,
}

impl ModGraph {
    pub fn new(q: u32) -> Self {
        Self {
            f: Field::new(q),
            nodes: Vec::new(),
            inputs: 0,
        }
    }

    /// Every node's element count, in graph order.
    pub fn nodes_sizes(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .map(|n| n.shape.iter().product())
            .collect()
    }

    pub fn q(&self) -> u32 {
        self.f.q
    }

    pub fn shape(&self, t: ModId) -> &[usize] {
        &self.nodes[t.0 as usize].shape
    }

    fn push(&mut self, op: MOp, shape: Vec<usize>, label: Label) -> ModId {
        self.nodes.push(MNode { op, shape, label });
        ModId(self.nodes.len() as u32 - 1)
    }

    /// The identity view of a rank-`r` source.
    pub fn id(r: usize) -> View {
        View::identity(r, r)
    }

    /// A view whose source coordinate `s` is `offsets[s] + Σ coefs[s][d]·i_d`.
    pub fn view(coefs: &[&[i64]], offsets: &[i64]) -> View {
        View {
            forms: coefs
                .iter()
                .zip(offsets)
                .map(|(c, &o)| Form {
                    coefs: c.to_vec(),
                    offset: o,
                })
                .collect(),
            guard: false,
            gather: None,
        }
    }

    /// A residue input. Values must be canonical (< q): no backend reduces them.
    pub fn input(&mut self, shape: &[usize], label: Label) -> ModId {
        self.inputs += 1;
        self.push(MOp::Input(self.inputs - 1, true), shape.to_vec(), label)
    }

    /// An input of raw 32-bit words (hash states, random bytes), not reduced.
    pub fn input_words(&mut self, shape: &[usize], label: Label) -> ModId {
        self.inputs += 1;
        self.push(MOp::Input(self.inputs - 1, false), shape.to_vec(), label)
    }

    /// A public table of raw words, not reduced.
    pub fn constant_words(&mut self, values: &[u32]) -> ModId {
        self.push(
            MOp::Const(values.to_vec()),
            vec![values.len()],
            Label::Public,
        )
    }

    /// A public table, reduced mod q.
    pub fn constant(&mut self, values: &[u32]) -> ModId {
        let q = self.f.q;
        let v = values.iter().map(|x| x % q).collect();
        self.push(MOp::Const(v), vec![values.len()], Label::Public)
    }

    /// The same values row-major under a new shape. Free.
    pub fn reshape(&mut self, src: ModId, shape: &[usize]) -> ModId {
        let n: usize = shape.iter().product();
        assert_eq!(
            n,
            self.shape(src).iter().product::<usize>(),
            "reshape changes size"
        );
        self.push(MOp::Reshape(src), shape.to_vec(), Label::Public)
    }

    /// `out[i] = body(args[0][view_0(i)], …)` over every point of `shape`.
    pub fn map(
        &mut self,
        shape: &[usize],
        args: &[(ModId, View)],
        f: impl FnOnce(&mut Body) -> BVal,
    ) -> ModId {
        for (src, v) in args {
            assert_eq!(
                v.forms.len(),
                self.shape(*src).len(),
                "view rank ≠ source rank"
            );
            assert!(
                v.forms.iter().all(|f| f.coefs.len() == shape.len()),
                "view arity ≠ map rank"
            );
        }
        let mut body = Body { ops: Vec::new() };
        let out = f(&mut body).0;
        let op = MOp::Map {
            args: args.to_vec(),
            body,
            out,
        };
        self.push(op, shape.to_vec(), Label::Public)
    }

    /// `out[i] = table[index[i]]`. Data-dependent address: index must be public.
    pub fn gather(&mut self, table: ModId, index: ModId) -> ModId {
        let shape = self.shape(index).to_vec();
        self.push(MOp::Gather { table, index }, shape, Label::Public)
    }

    /// The values of `src` below `bound`, in order, padded with zeros to the
    /// same length. How many survive depends on data: src must be public.
    pub fn filter(&mut self, src: ModId, bound: u32) -> ModId {
        let shape = self.shape(src).to_vec();
        self.push(MOp::Filter { src, bound }, shape, Label::Public)
    }

    /// Publish a secret, naming the proof that it reveals nothing.
    pub fn declassify(&mut self, src: ModId, proof: &'static str) -> ModId {
        let shape = self.shape(src).to_vec();
        self.push(MOp::Declassify { src, proof }, shape, Label::Public)
    }

    /// Every node's label, inputs as declared, the rest joined from their
    /// sources.
    pub fn labels(&self) -> Vec<Label> {
        let mut l: Vec<Label> = Vec::with_capacity(self.nodes.len());
        for n in &self.nodes {
            let label = match &n.op {
                MOp::Input(..) => n.label,
                MOp::Const(_) | MOp::Declassify { .. } => Label::Public,
                MOp::Reshape(s) | MOp::Filter { src: s, .. } => l[s.0 as usize],
                MOp::Map { args, .. } => args
                    .iter()
                    .map(|(s, _)| l[s.0 as usize])
                    .max()
                    .unwrap_or(Label::Public),
                MOp::Gather { table, index } => l[table.0 as usize].max(l[index.0 as usize]),
            };
            l.push(label);
        }
        l
    }

    /// Refuse the graph if a secret steers an address or a count, or reaches
    /// an output declared public. Returns the proofs every declassify relies on.
    pub fn check(&self, outputs: &[(ModId, Label)]) -> Result<Vec<&'static str>, TaintError> {
        let l = self.labels();
        let secret = |t: &ModId| l[t.0 as usize] == Label::Secret;
        let mut proofs = Vec::new();
        for (i, n) in self.nodes.iter().enumerate() {
            let id = ModId(i as u32);
            match &n.op {
                MOp::Gather { index, .. } if secret(index) => {
                    return Err(TaintError::SecretIndex(id))
                }
                MOp::Filter { src, .. } if secret(src) => return Err(TaintError::SecretFilter(id)),
                MOp::Declassify { proof, .. } => proofs.push(*proof),
                _ => {}
            }
        }
        for (t, want) in outputs {
            if *want == Label::Public && secret(t) {
                return Err(TaintError::SecretOutput(*t));
            }
        }
        Ok(proofs)
    }

    /// Run the graph. Residue inputs must be canonical (< q), as in every backend.
    pub fn eval(&self, inputs: &[Vec<u32>], outputs: &[ModId]) -> Vec<Vec<u32>> {
        let f = self.f;
        let mut vals: Vec<Vec<u32>> = Vec::with_capacity(self.nodes.len());
        for n in &self.nodes {
            let size: usize = n.shape.iter().product();
            let v = match &n.op {
                MOp::Input(i, reduce) => {
                    assert_eq!(inputs[*i].len(), size, "input {i} has the wrong length");
                    // residues must arrive canonical: the C and GPU lowerings use
                    // them as given (reducing a secret with `%` would divide)
                    if *reduce {
                        assert!(
                            inputs[*i].iter().all(|&x| x < f.q),
                            "input {i} holds a value ≥ q; residue inputs must be canonical"
                        );
                    }
                    inputs[*i].clone()
                }
                MOp::Const(c) => c.clone(),
                MOp::Reshape(s) | MOp::Declassify { src: s, .. } => vals[s.0 as usize].clone(),
                MOp::Gather { table, index } => {
                    let t = &vals[table.0 as usize];
                    vals[index.0 as usize]
                        .iter()
                        .map(|&i| t[i as usize % t.len()])
                        .collect()
                }
                MOp::Filter { src, bound } => {
                    let mut kept: Vec<u32> = vals[src.0 as usize]
                        .iter()
                        .copied()
                        .filter(|x| x < bound)
                        .collect();
                    kept.resize(size, 0);
                    kept
                }
                MOp::Map {
                    args,
                    body,
                    out: res,
                } => {
                    let mut out = Vec::with_capacity(size);
                    let mut reg = vec![0u32; body.ops.len()];
                    let mut idx = vec![0usize; n.shape.len()];
                    for p in 0..size {
                        if p > 0 {
                            // odometer step to the next row-major point
                            for d in (0..idx.len()).rev() {
                                idx[d] += 1;
                                if idx[d] < n.shape[d] {
                                    break;
                                }
                                idx[d] = 0;
                            }
                        }
                        for (k, op) in body.ops.iter().enumerate() {
                            reg[k] = match *op {
                                BOp::Arg(a) => {
                                    let (src, view) = &args[a];
                                    let shape = self.shape(*src);
                                    let mut at = 0usize;
                                    let mut inside = true;
                                    for (form, &len) in view.forms.iter().zip(shape) {
                                        let c = form.eval(&idx);
                                        inside &= (0..len as i64).contains(&c);
                                        at = at * len + c.clamp(0, len as i64 - 1) as usize;
                                    }
                                    assert!(
                                        inside || view.guard,
                                        "unguarded view reads off the edge"
                                    );
                                    if inside {
                                        vals[src.0 as usize][at]
                                    } else {
                                        0 // a guarded read off the edge
                                    }
                                }
                                BOp::Lit(x) => x,
                                BOp::Coord(d) => idx[d] as u32,
                                BOp::Add(a, b) => f.add(reg[a], reg[b]),
                                BOp::Sub(a, b) => f.sub(reg[a], reg[b]),
                                BOp::Mul(a, b) => f.mul(reg[a], reg[b]),
                                BOp::Sel(c, a, b) => f.sel(reg[c], reg[a], reg[b]),
                                BOp::Compress(a, d) => f.compress(reg[a], d),
                                BOp::Decompress(a, d) => f.decompress(reg[a], d),
                                BOp::Power2Round(a, d) => f.power2round(reg[a], d),
                                BOp::Xor(a, b) => reg[a] ^ reg[b],
                                BOp::And(a, b) => reg[a] & reg[b],
                                BOp::Or(a, b) => reg[a] | reg[b],
                                BOp::Not(a) => !reg[a],
                                BOp::Shl(a, b) => reg[a] << (reg[b] & 31),
                                BOp::Shr(a, b) => reg[a] >> (reg[b] & 31),
                                BOp::Eq(a, b) => {
                                    let d = reg[a] ^ reg[b];
                                    ((d | d.wrapping_neg()) >> 31) ^ 1
                                }
                                BOp::Lt(a, b) => reg[a].wrapping_sub(reg[b]) >> 31,
                                BOp::MulHi(a, b) => ((reg[a] as u64 * reg[b] as u64) >> 32) as u32,
                                BOp::HighBits(a, g2) => f.decompose(reg[a], g2).0,
                                BOp::LowBits(a, g2) => f.decompose(reg[a], g2).1,
                            };
                        }
                        out.push(reg[*res]);
                    }
                    out
                }
            };
            vals.push(v);
        }
        outputs.iter().map(|t| vals[t.0 as usize].clone()).collect()
    }
}

/// The negacyclic NTT over `Z_q[x]/(x^n + 1)` as a graph of log2(n)
/// Cooley–Tukey stages, output in bit-reversed order (the Dilithium order).
/// `zeta` is a primitive 2n-th root of unity. Pointwise products of two
/// transforms are the transform of the negacyclic product.
pub fn ntt(g: &mut ModGraph, x: ModId, zeta: u32) -> ModId {
    let n = g.shape(x)[0];
    let zetas = g.constant(&twiddles(g.q(), zeta, n));
    let mut cur = x;
    let mut len = n / 2;
    while len >= 1 {
        let blocks = n / (2 * len);
        let src = g.reshape(cur, &[blocks, 2, len]);
        let lo = ModGraph::view(&[&[1, 0, 0], &[0, 0, 0], &[0, 0, 1]], &[0, 0, 0]);
        let hi = ModGraph::view(&[&[1, 0, 0], &[0, 0, 0], &[0, 0, 1]], &[0, 1, 0]);
        let z = ModGraph::view(&[&[1, 0, 0]], &[blocks as i64]);
        let out = g.map(
            &[blocks, 2, len],
            &[(src, lo), (src, hi), (zetas, z)],
            |b| {
                let (x0, x1, z) = (b.arg(0), b.arg(1), b.arg(2));
                let t = b.mul(z, x1);
                let (top, bot) = (b.add(x0, t), b.sub(x0, t));
                let h = b.coord(1);
                b.sel(h, bot, top)
            },
        );
        cur = g.reshape(out, &[n]);
        len /= 2;
    }
    cur
}

/// The inverse of [`ntt`]: Gentleman–Sande stages, then a scale by n⁻¹.
pub fn intt(g: &mut ModGraph, x: ModId, zeta: u32) -> ModId {
    let n = g.shape(x)[0];
    let q = g.q();
    let neg: Vec<u32> = twiddles(q, zeta, n).iter().map(|z| (q - z) % q).collect();
    let zetas = g.constant(&neg);
    let mut cur = x;
    let mut len = 1;
    while len < n {
        let blocks = n / (2 * len);
        let src = g.reshape(cur, &[blocks, 2, len]);
        let lo = ModGraph::view(&[&[1, 0, 0], &[0, 0, 0], &[0, 0, 1]], &[0, 0, 0]);
        let hi = ModGraph::view(&[&[1, 0, 0], &[0, 0, 0], &[0, 0, 1]], &[0, 1, 0]);
        let z = ModGraph::view(&[&[-1, 0, 0]], &[2 * blocks as i64 - 1]);
        let out = g.map(
            &[blocks, 2, len],
            &[(src, lo), (src, hi), (zetas, z)],
            |b| {
                let (x0, x1, z) = (b.arg(0), b.arg(1), b.arg(2));
                let top = b.add(x0, x1);
                let d = b.sub(x0, x1);
                let bot = b.mul(z, d);
                let h = b.coord(1);
                b.sel(h, bot, top)
            },
        );
        cur = g.reshape(out, &[n]);
        len *= 2;
    }
    let ninv = pow(n as u32, q - 2, q);
    g.map(&[n], &[(cur, ModGraph::id(1))], |b| {
        let (x, s) = (b.arg(0), b.lit(ninv));
        b.mul(x, s)
    })
}

fn pow(b: u32, mut e: u32, q: u32) -> u32 {
    let mut r = 1u64;
    let mut b64 = b as u64 % q as u64;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b64 % q as u64;
        }
        b64 = b64 * b64 % q as u64;
        e >>= 1;
    }
    r as u32
}

/// `zetas[k] = zeta^brv(k)` for k < n, brv over log2(n) bits. Public constants.
fn twiddles(q: u32, zeta: u32, n: usize) -> Vec<u32> {
    let bits = n.trailing_zeros();
    (0..n as u32)
        .map(|k| pow(zeta, k.reverse_bits() >> (32 - bits), q))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q: u32 = 8380417;
    const ZETA: u32 = 1753;
    const N: usize = 256;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as u32) % Q
    }

    fn schoolbook(a: &[u32], b: &[u32]) -> Vec<u32> {
        let mut c = vec![0i128; N];
        for i in 0..N {
            for j in 0..N {
                let p = a[i] as i128 * b[j] as i128;
                if i + j < N {
                    c[i + j] += p
                } else {
                    c[i + j - N] -= p
                }
            }
        }
        c.iter().map(|x| x.rem_euclid(Q as i128) as u32).collect()
    }

    #[test]
    fn field_matches_naive() {
        let f = Field::new(Q);
        let mut s = 7;
        for _ in 0..10_000 {
            let (a, b) = (lcg(&mut s), lcg(&mut s));
            assert_eq!(f.add(a, b) as u64, (a as u64 + b as u64) % Q as u64);
            assert_eq!(
                f.sub(a, b) as u64,
                (a as u64 + Q as u64 - b as u64) % Q as u64
            );
            assert_eq!(f.mul(a, b) as u64, a as u64 * b as u64 % Q as u64);
        }
        assert_eq!(f.mul(Q - 1, Q - 1), 1);
    }

    #[test]
    fn compress_matches_rounding() {
        let f = Field::new(Q);
        let mut s = 3;
        for d in [1, 5, 10, 11] {
            for _ in 0..5_000 {
                let x = lcg(&mut s);
                let want =
                    ((((x as u128) << d) + Q as u128 / 2) / Q as u128) as u32 & ((1 << d) - 1);
                assert_eq!(f.compress(x, d), want);
                let y = x & ((1 << d) - 1);
                let back = ((y as u128 * Q as u128 + (1 << d >> 1)) >> d) as u32;
                assert_eq!(f.decompress(y, d), back);
            }
            for x in [0, 1, Q / 2, Q / 2 + 1, Q - 1] {
                let want =
                    ((((x as u128) << d) + Q as u128 / 2) / Q as u128) as u32 & ((1 << d) - 1);
                assert_eq!(f.compress(x, d), want, "x={x} d={d}");
            }
        }
        // d = 0 drops nothing
        for x in [0, 1, Q - 1] {
            assert_eq!(f.power2round(x, 0), x);
        }
        // dropping 13 bits leaves an error in (−2^12, 2^12]
        for x in [0, 4095, 4096, 4097, Q - 1] {
            let e = x as i64 - ((f.power2round(x, 13) as i64) << 13);
            assert!(-(1 << 12) < e && e <= 1 << 12, "x={x} e={e}");
        }
    }

    #[test]
    fn decompose_matches_fips204() {
        // the reference: r₀ = r mod± α; if r − r₀ = q − 1 then (0, r₀ − 1)
        let f = Field::new(Q);
        let g2 = (Q - 1) / 32;
        let alpha = 2 * g2 as i64;
        let mut s = 11;
        let mut xs: Vec<u32> = (0..20_000).map(|_| lcg(&mut s)).collect();
        xs.extend([0, 1, g2, g2 + 1, Q - 1, Q - 2, Q - 1 - g2, Q - g2]);
        for r in xs {
            let mut r0 = r as i64 % alpha;
            if r0 > alpha / 2 {
                r0 -= alpha;
            }
            let (r1, r0) = if r as i64 - r0 == Q as i64 - 1 {
                (0, r0 - 1)
            } else {
                ((r as i64 - r0) / alpha, r0)
            };
            let got = f.decompose(r, g2);
            assert_eq!(got, (r1 as u32, r0.rem_euclid(Q as i64) as u32), "r = {r}");
        }
    }

    #[test]
    fn zeta_is_a_primitive_512th_root() {
        assert_eq!(pow(ZETA, 256, Q), Q - 1);
        assert_eq!(pow(ZETA, 512, Q), 1);
    }

    #[test]
    fn ntt_round_trips() {
        let mut g = ModGraph::new(Q);
        let a = g.input(&[N], Label::Secret);
        let fa = ntt(&mut g, a, ZETA);
        let back = intt(&mut g, fa, ZETA);
        let mut s = 1;
        let x: Vec<u32> = (0..N).map(|_| lcg(&mut s)).collect();
        assert_eq!(g.eval(std::slice::from_ref(&x), &[back])[0], x);
    }

    #[test]
    fn ntt_multiplies_negacyclically() {
        let mut g = ModGraph::new(Q);
        let (a, b) = (g.input(&[N], Label::Secret), g.input(&[N], Label::Public));
        let (fa, fb) = (ntt(&mut g, a, ZETA), ntt(&mut g, b, ZETA));
        let prod = g.map(&[N], &[(fa, ModGraph::id(1)), (fb, ModGraph::id(1))], |b| {
            let (x, y) = (b.arg(0), b.arg(1));
            b.mul(x, y)
        });
        let c = intt(&mut g, prod, ZETA);
        let mut s = 42;
        let x: Vec<u32> = (0..N).map(|_| lcg(&mut s)).collect();
        let y: Vec<u32> = (0..N).map(|_| lcg(&mut s)).collect();
        assert_eq!(g.eval(&[x.clone(), y.clone()], &[c])[0], schoolbook(&x, &y));
        // the product of a secret is secret, and the NTT itself needed no declassify
        assert_eq!(g.check(&[(c, Label::Secret)]), Ok(vec![]));
        assert_eq!(
            g.check(&[(c, Label::Public)]),
            Err(TaintError::SecretOutput(c))
        );
    }

    #[test]
    fn x_to_the_n_is_minus_one() {
        // x^255 · x = x^256 = −1 in the ring
        let mut a = vec![0; N];
        let mut b = vec![0; N];
        a[255] = 1;
        b[1] = 1;
        let mut want = vec![0; N];
        want[0] = Q - 1;
        assert_eq!(schoolbook(&a, &b), want);
        let mut g = ModGraph::new(Q);
        let (x, y) = (g.input(&[N], Label::Public), g.input(&[N], Label::Public));
        let (fx, fy) = (ntt(&mut g, x, ZETA), ntt(&mut g, y, ZETA));
        let p = g.map(&[N], &[(fx, ModGraph::id(1)), (fy, ModGraph::id(1))], |b| {
            let (u, v) = (b.arg(0), b.arg(1));
            b.mul(u, v)
        });
        let c = intt(&mut g, p, ZETA);
        assert_eq!(g.eval(&[a, b], &[c])[0], want);
    }

    #[test]
    fn secret_index_is_refused() {
        let mut g = ModGraph::new(Q);
        let table = g.constant(&[10, 20, 30, 40]);
        let s = g.input(&[2], Label::Secret);
        let r = g.gather(table, s);
        assert_eq!(
            g.check(&[(r, Label::Secret)]),
            Err(TaintError::SecretIndex(r))
        );

        let mut g = ModGraph::new(Q);
        let table = g.constant(&[10, 20, 30, 40]);
        let p = g.input(&[2], Label::Public);
        let r = g.gather(table, p);
        assert_eq!(g.check(&[(r, Label::Public)]), Ok(vec![]));
        assert_eq!(g.eval(&[vec![3, 1]], &[r])[0], vec![40, 20]);
    }

    #[test]
    fn rejection_sampling_is_public_only() {
        // expanding the public matrix A: reject candidates ≥ q (here from a seed)
        let mut g = ModGraph::new(Q);
        let seed = g.input(&[4], Label::Public);
        let a = g.filter(seed, 100);
        assert_eq!(g.check(&[(a, Label::Public)]), Ok(vec![]));
        assert_eq!(g.eval(&[vec![5, 200, 7, 99]], &[a])[0], vec![5, 7, 99, 0]);

        let mut g = ModGraph::new(Q);
        let s = g.input(&[4], Label::Secret);
        let bad = g.filter(s, 100);
        assert_eq!(
            g.check(&[(bad, Label::Secret)]),
            Err(TaintError::SecretFilter(bad))
        );
    }

    #[test]
    fn declassify_names_its_proof() {
        // the signature's attempt count: secret-derived, provably independent of the key
        let mut g = ModGraph::new(Q);
        let s = g.input(&[1], Label::Secret);
        let attempts = g.declassify(s, "fswa-abort-count-independent");
        let next = g.filter(attempts, 5);
        assert_eq!(
            g.check(&[(next, Label::Public)]),
            Ok(vec!["fswa-abort-count-independent"])
        );
    }
}
