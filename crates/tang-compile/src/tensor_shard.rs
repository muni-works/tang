//! Multi-GPU decode: the split of a derived step graph across `N` ranks,
//! read off the graph.
//!
//! Every node gets a **placement** over the ranks:
//!
//! - `R`: replicated, every rank holds the whole tensor;
//! - `S(d)`: split along dim `d` into `N` contiguous equal chunks;
//! - `P`: partial, the tensor is the sum of the ranks' local tensors;
//! - `M(mask)`: masked partial, each point is held by exactly one rank
//!   (whose `idx[forms(point)]` falls in its expert range) and is 0 on the
//!   others: an expert-parallel value. Summed over ranks it is the tensor,
//!   so it converts to `R` with the same all-reduce as `P`.
//!
//! Weights enter with a placement from a [`ShardPlan`]: `Experts` (dim 0
//! split, read only through a gather on that dim: expert parallel),
//! `Split(d)` (tensor parallel), or replicated. [`analyze`] propagates
//! placements through the comprehensions in one forward sweep:
//!
//! - a comp whose `S` args all read their split dim through one comp
//!   coordinate `j` (`s·x_j + rest`, `rest` inside one stride, the chunks
//!   aligned) is computed on a slice of `j`: output `S(j)` when `j` is a
//!   point, `P` when it is a summed reduce coordinate (any body: a sum
//!   splits over a partition of its range);
//! - a comp reading `M` / expert args that agree on the mask, whose body
//!   vanishes when they are 0, is `M` (mask over points) or `P` (the mask
//!   moves with a summed coordinate: the sum over picks);
//! - a comp linear in its `P` args (others `R`, sum or no reduction) is `P`;
//! - anything else reads its args replicated: an **all-gather** (`S`) or
//!   **all-reduce** (`P`, `M`) node is inserted before it. Outputs are
//!   always made `R`.
//!
//! So the rules are never wrong, only more or less communication: the
//! fallback is always a collective. [`plan_auto`] picks the tensor-parallel
//! splits greedily against a [`CostModel`] (weight bytes a rank reads vs.
//! collective latency and volume); expert parallelism is chosen for every
//! input that is only ever gathered on dim 0 by an index computed from a
//! top-k (a router).
//!
//! [`rank_program`] then writes rank `r`'s program: local shapes, view
//! offsets shifted by the rank's chunk origin, expert gathers re-based and
//! guarded (an expert another rank owns reads as 0), collectives as
//! [`Operator`] nodes, and every replicated weight cut to the box this
//! rank's program actually reads — which is how a weight read after a
//! column-split producer (Megatron's row-parallel half) is stored as its
//! slice without being planned. [`HostComm`] runs the collectives between
//! threads, so `N` ranks simulate in-process on the reference interpreter.
use super::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Barrier, Mutex};

/// Where a weight input lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightPlace {
    Replicated,
    /// Tensor parallel: dim `d` in `N` contiguous chunks.
    Split(usize),
    /// Expert parallel: dim 0 in `N` contiguous blocks of experts.
    Experts,
}

#[derive(Clone, Debug, Default)]
pub struct ShardPlan {
    pub world: usize,
    /// Input node → placement (absent: replicated).
    pub inputs: HashMap<TensorId, WeightPlace>,
    /// Caches: `(input node, index into outputs of its updated value)`. A
    /// cache whose update comes out split is kept split per rank (its
    /// input takes the same split): the GDN state and KV split by head.
    pub caches: Vec<(TensorId, usize)>,
}

/// A masked partial's owner test: rank `r` holds point `p` iff
/// `idx[forms(p)]` (an expert id or row) is in `r`'s block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask {
    pub idx: TensorId,
    pub forms: Vec<Form>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Place {
    R,
    S(usize),
    P,
    M(Mask),
    /// An expert-parallel weight input (only read through gathers).
    Ep,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CollKind {
    /// Sum over ranks (a `P` or `M` tensor made `R`).
    AllReduce,
    /// Concatenate the ranks' chunks along `dim` (an `S(dim)` tensor made `R`).
    AllGather(usize),
}

/// One collective of the step: on `src` (a node of the analyzed graph).
#[derive(Clone, Debug)]
pub struct Collective {
    pub kind: CollKind,
    pub src: TensorId,
    /// Elements of the replicated result.
    pub numel: usize,
}

impl Collective {
    /// Bytes each rank sends in a ring implementation (`elem` bytes per
    /// element): all-reduce `2(N−1)/N`, all-gather `(N−1)/N` of the result.
    pub fn ring_bytes(&self, world: usize, elem: f64) -> f64 {
        let n = world as f64;
        let f = match self.kind {
            CollKind::AllReduce => 2.0 * (n - 1.0) / n,
            CollKind::AllGather(_) => (n - 1.0) / n,
        };
        f * self.numel as f64 * elem
    }
}

/// The result of [`analyze`].
#[derive(Clone, Debug)]
pub struct Analysis {
    pub world: usize,
    /// Output placement per node (live nodes; dead ones `R`).
    pub place: Vec<Place>,
    /// Per comp: the comp coordinate computed on a slice.
    pub split: Vec<Option<usize>>,
    /// Per node, per dep position: the collective its source goes through.
    pub conv: Vec<Vec<Option<CollKind>>>,
    /// The step outputs' collectives (an output not `R`).
    pub out_conv: Vec<Option<CollKind>>,
    /// Outputs left split across ranks (sharded caches): the dim.
    pub out_local: Vec<Option<usize>>,
    pub collectives: Vec<Collective>,
}

impl Analysis {
    pub fn count(&self, kind: fn(&CollKind) -> bool) -> usize {
        self.collectives.iter().filter(|c| kind(&c.kind)).count()
    }
}

fn dep_srcs(g: &TensorGraph, i: usize) -> Vec<TensorId> {
    match &g.nodes[i].op {
        Op::Comp(c) => c.args.iter().map(|a| a.src).collect(),
        Op::Reshape(s) => vec![*s],
        Op::Operator(o) => o.inputs.clone(),
        Op::Scan(s) => s.inputs.clone(),
        Op::Input(_) | Op::Iota(_) => vec![],
    }
}

fn to_r(p: &Place) -> Option<CollKind> {
    match p {
        Place::R => None,
        Place::S(d) => Some(CollKind::AllGather(*d)),
        Place::P | Place::M(_) => Some(CollKind::AllReduce),
        Place::Ep => panic!("an expert-parallel weight read other than by its gather"),
    }
}

/// The comp coordinate `j` through which `view` reads source dim `d`
/// (extent `n`) so that splitting `j` into `world` chunks reads exactly the
/// source's chunks: `forms[d] = s·x_j + rest` with `0 ≤ rest < s` over the
/// space and `ext_j · s = n`.
fn split_coord(view: &View, d: usize, n: usize, space: &[usize], world: usize) -> Option<usize> {
    if view.gather.as_ref().is_some_and(|g| g.coefs[d] != 0) {
        return None;
    }
    let f = &view.forms[d];
    (0..space.len()).find(|&j| {
        let s = f.coefs[j];
        if s <= 0 || space[j] % world != 0 || space[j] as i64 * s != n as i64 {
            return false;
        }
        let mut rest = f.clone();
        rest.coefs[j] = 0;
        let (lo, hi) = rest.range(space);
        lo >= 0 && hi < s
    })
}

/// An abstract value of a body expression: a known constant, a value
/// that depends only on the "other" args, homogeneous linear in the
/// marked args, or anything else.
#[derive(Clone, Copy, Debug, PartialEq)]
enum A {
    K(f64),
    Free,
    Lin,
    Non,
}

/// Evaluate the body abstractly, arg `a` starting as `var(a)`, folding
/// constants (a select on a constant condition takes its branch). Sound:
/// `K(0)` means the body is 0 for every value of the `Free` args, `Lin`
/// that it is linear and homogeneous in the `Lin` args.
fn abstract_body(c: &Comp, var: &dyn Fn(usize) -> A) -> Vec<A> {
    use A::*;
    let nodes = c.body.nodes_slice();
    let n = c
        .body_outs
        .iter()
        .map(|o| o.index() as usize + 1)
        .max()
        .unwrap_or(0);
    let mut v: Vec<A> = Vec::with_capacity(n);
    let zero = |x: A| x == K(0.0);
    let lin0 = |x: A| x == Lin || zero(x);
    for node in &nodes[..n] {
        let get = |e: &ExprId| v[e.index() as usize];
        let unary = |x: A, f: fn(f64) -> f64| match x {
            K(a) => K(f(a)),
            Free => Free,
            _ => Non,
        };
        let r = match node {
            Node::Var(i) => var(*i as usize),
            Node::Lit(bits) => K(f64::from_bits(*bits)),
            Node::Add(a, b) => match (get(a), get(b)) {
                (K(x), K(y)) => K(x + y),
                (x, y) if zero(x) => y,
                (x, y) if zero(y) => x,
                (Lin, Lin) => Lin,
                (Free | K(_), Free | K(_)) => Free,
                _ => Non,
            },
            Node::Mul(a, b) => match (get(a), get(b)) {
                (K(x), K(y)) => K(x * y),
                (x, _) | (_, x) if zero(x) => K(0.0),
                (Lin, Free | K(_)) | (Free | K(_), Lin) => Lin,
                (Free | K(_), Free | K(_)) => Free,
                _ => Non,
            },
            Node::Neg(a) => match get(a) {
                K(x) => K(-x),
                o => o,
            },
            Node::Recip(a) => unary(get(a), |x| 1.0 / x),
            Node::Sqrt(a) => unary(get(a), f64::sqrt),
            Node::Sin(a) => unary(get(a), f64::sin),
            Node::Floor(a) => unary(get(a), f64::floor),
            Node::Exp2(a) => unary(get(a), f64::exp2),
            Node::Log2(a) => unary(get(a), f64::log2),
            Node::Atan2(a, b) => match (get(a), get(b)) {
                (K(y), K(x)) => K(y.atan2(x)),
                (Free | K(_), Free | K(_)) => Free,
                _ => Non,
            },
            Node::Select(cnd, a, b) => match (get(cnd), get(a), get(b)) {
                (K(x), t, f) => {
                    if x > 0.0 {
                        t
                    } else {
                        f
                    }
                }
                (_, t, f) if zero(t) && zero(f) => K(0.0),
                (Free, t, f) if lin0(t) && lin0(f) => Lin,
                (Free, Free | K(_), Free | K(_)) => Free,
                _ => Non,
            },
        };
        v.push(r);
    }
    c.body_outs.iter().map(|o| v[o.index() as usize]).collect()
}

/// The body is 0 whenever the args in `zero` are 0, whatever the others.
fn vanishes(c: &Comp, zero: &[usize]) -> bool {
    abstract_body(c, &|a| {
        if zero.contains(&a) {
            A::K(0.0)
        } else {
            A::Free
        }
    })
    .iter()
    .all(|x| *x == A::K(0.0))
}

/// The body is linear and homogeneous in the args `lin` jointly.
fn linear_in(c: &Comp, lin: &[usize]) -> bool {
    abstract_body(c, &|a| if lin.contains(&a) { A::Lin } else { A::Free })
        .iter()
        .all(|x| matches!(x, A::Lin) || *x == A::K(0.0))
}

/// Propagate placements through `g` (topologically ordered, as built).
type Sweep = (Vec<Place>, Vec<Option<usize>>, Vec<Vec<Option<CollKind>>>);

fn sweep(
    g: &TensorGraph,
    live: &[bool],
    plan: &ShardPlan,
    cache_place: &HashMap<TensorId, Place>,
) -> Sweep {
    let n = g.nodes.len();
    let world = plan.world;
    let mut place = vec![Place::R; n];
    let mut split = vec![None; n];
    let mut conv: Vec<Vec<Option<CollKind>>> =
        (0..n).map(|i| vec![None; dep_srcs(g, i).len()]).collect();
    for i in (0..n).filter(|&i| live[i]) {
        let shape = fixed(&g.nodes[i].shape);
        match &g.nodes[i].op {
            Op::Input(_) => {
                let t = TensorId(i as u32);
                place[i] = match plan.inputs.get(&t) {
                    Some(WeightPlace::Split(d)) => Place::S(*d),
                    Some(WeightPlace::Experts) => Place::Ep,
                    _ => cache_place.get(&t).cloned().unwrap_or(Place::R),
                }
            }
            Op::Iota(_) => {}
            Op::Reshape(src) => {
                let sp = place[src.n()].clone();
                let src_shape = fixed(&g.nodes[src.n()].shape);
                place[i] = match sp {
                    Place::R => Place::R,
                    Place::P => Place::P,
                    Place::S(d) => match reshape_split(&src_shape, d, &shape, world) {
                        Some(d2) => Place::S(d2),
                        None => {
                            conv[i][0] = to_r(&sp);
                            Place::R
                        }
                    },
                    other => {
                        conv[i][0] = to_r(&other);
                        Place::R
                    }
                }
            }
            Op::Scan(sc) => {
                place[i] = scan_rule(g, sc, &place, &mut conv[i], &shape);
            }
            Op::Operator(_) => {
                for (k, s) in dep_srcs(g, i).iter().enumerate() {
                    conv[i][k] = to_r(&place[s.n()]);
                }
            }
            Op::Comp(c) => {
                let (p, j) = comp_rule(g, c, &place, &mut conv[i], world);
                place[i] = p;
                split[i] = j;
            }
        }
    }
    (place, split, conv)
}

/// A scan is independent along its leading (non-matrix, non-scan) dims:
/// inputs all split on the same such dim give an output split on it.
fn scan_rule(
    g: &TensorGraph,
    sc: &Scan,
    place: &[Place],
    conv: &mut [Option<CollKind>],
    out: &[usize],
) -> Place {
    let lead = match &sc.kind {
        ScanKind::Monoid(m) => out.len() - usize::from(m.width > 1),
        ScanKind::Affine { .. } => out.len() - 2,
    };
    let ps: Vec<Place> = sc.inputs.iter().map(|t| place[t.n()].clone()).collect();
    if let Some(Place::S(d)) = ps.first() {
        let d = *d;
        let ok = d < lead
            && d != sc.dim
            && ps
                .iter()
                .zip(&sc.inputs)
                .all(|(p, t)| *p == Place::S(d) && g.nodes[t.n()].shape.len() > d);
        if ok {
            return Place::S(d);
        }
    }
    for (k, p) in ps.iter().enumerate() {
        conv[k] = to_r(p);
    }
    Place::R
}

/// Propagate placements through `g` (topologically ordered, as built).
pub fn analyze(g: &TensorGraph, outputs: &[TensorId], plan: &ShardPlan) -> Analysis {
    let n = g.nodes.len();
    let world = plan.world;
    let live = g.live(outputs);
    // caches follow their updates: shard a cache whose update comes out
    // split, drop it again if the split doesn't hold (each tried once)
    let mut cache_place: HashMap<TensorId, Place> = HashMap::new();
    let mut tried: HashSet<TensorId> = HashSet::new();
    let (place, split, conv) = loop {
        let (place, split, conv) = sweep(g, &live, plan, &cache_place);
        let mut changed = false;
        for &(inp, oi) in &plan.caches {
            let po = &place[outputs[oi].n()];
            let cur = cache_place.get(&inp).cloned().unwrap_or(Place::R);
            match po {
                Place::S(_) if cur == Place::R && tried.insert(inp) => {
                    cache_place.insert(inp, po.clone());
                    changed = true;
                }
                _ if cur != Place::R && *po != cur => {
                    cache_place.remove(&inp);
                    changed = true;
                }
                _ => {}
            }
        }
        if !changed {
            break (place, split, conv);
        }
    };
    let mut out_local: Vec<Option<usize>> = vec![None; outputs.len()];
    for &(inp, oi) in &plan.caches {
        if let (Some(Place::S(d)), Place::S(e)) = (cache_place.get(&inp), &place[outputs[oi].n()]) {
            if d == e {
                out_local[oi] = Some(*d);
            }
        }
    }
    let out_conv: Vec<Option<CollKind>> = outputs
        .iter()
        .zip(&out_local)
        .map(|(o, l)| {
            if l.is_some() {
                None
            } else {
                to_r(&place[o.n()])
            }
        })
        .collect();
    // one collective per (source, kind)
    let mut seen = HashSet::new();
    let mut collectives = Vec::new();
    let mut note = |src: TensorId, k: CollKind| {
        if seen.insert((src, k)) {
            collectives.push(Collective {
                kind: k,
                src,
                numel: numel(&g.nodes[src.n()].shape),
            });
        }
    };
    for i in (0..n).filter(|&i| live[i]) {
        for (k, s) in dep_srcs(g, i).iter().enumerate() {
            if let Some(kind) = conv[i][k] {
                note(*s, kind);
            }
        }
    }
    for (o, k) in outputs.iter().zip(&out_conv) {
        if let Some(k) = k {
            note(*o, *k);
        }
    }
    Analysis {
        world,
        place,
        split,
        conv,
        out_conv,
        out_local,
        collectives,
    }
}

/// The dim of `dst` an `S(d)` split of `src` lands on through a row-major
/// reshape: the same prefix volume and the same chunk volume.
fn reshape_split(src: &[usize], d: usize, dst: &[usize], world: usize) -> Option<usize> {
    let pre: usize = src[..d].iter().product();
    let chunk: usize = src[d..].iter().product::<usize>() / world;
    (0..dst.len()).find(|&e| {
        dst[..e].iter().product::<usize>() == pre
            && dst[e] % world == 0
            && dst[e..].iter().product::<usize>() / world == chunk
    })
}

fn comp_rule(
    g: &TensorGraph,
    c: &Comp,
    place: &[Place],
    conv: &mut [Option<CollKind>],
    world: usize,
) -> (Place, Option<usize>) {
    let space: Vec<usize> = c.space().iter().map(|d| d.n()).collect();
    let np = c.points().len();
    let pl = |a: usize, conv: &[Option<CollKind>]| -> Place {
        if conv[a].is_some() {
            Place::R
        } else {
            place[c.args[a].src.n()].clone()
        }
    };
    let index_args = c.index_args();
    // index reads see values, not sums: they must be replicated
    for &a in &index_args {
        conv[a] = to_r(&place[c.args[a].src.n()]);
    }
    let has_ep = (0..c.args.len()).any(|a| pl(a, conv) == Place::Ep);
    // 1. the split coordinate
    let mut j: Option<usize> = None;
    for a in 0..c.args.len() {
        if let Place::S(d) = pl(a, conv) {
            let n = g.nodes[c.args[a].src.n()].shape[d].n();
            match split_coord(&c.args[a].view, d, n, &space, world) {
                Some(jj) if !has_ep && j.is_none_or(|j| j == jj) => j = Some(jj),
                _ => conv[a] = Some(CollKind::AllGather(d)),
            }
        }
    }
    let sum = c.monoid.as_ref().map(|m| m.kind() == MonoidKind::Sum);
    if let Some(jj) = j {
        if jj >= np && sum != Some(true) {
            // a split max / top-k / softmax: read whole
            for a in 0..c.args.len() {
                if let Place::S(d) = pl(a, conv) {
                    conv[a] = Some(CollKind::AllGather(d));
                }
            }
            j = None;
        }
    }
    if let Some(jj) = j {
        for a in 0..c.args.len() {
            if matches!(pl(a, conv), Place::P | Place::M(_)) {
                conv[a] = Some(CollKind::AllReduce);
            }
        }
        return (if jj < np { Place::S(jj) } else { Place::P }, Some(jj));
    }
    // 2. masks (expert-parallel values)
    let rank = space.len();
    let mut masks: Vec<(usize, Mask)> = Vec::new();
    for a in 0..c.args.len() {
        match pl(a, conv) {
            Place::Ep => {
                let gth = c.args[a]
                    .view
                    .gather
                    .as_ref()
                    .expect("expert weight read without a gather");
                assert!(gth.coefs[0] != 0, "expert weight gathered off dim 0");
                let ia = &c.args[gth.arg];
                masks.push((
                    a,
                    Mask {
                        idx: ia.src,
                        forms: ia.view.forms.clone(),
                    },
                ));
            }
            Place::M(m) => {
                let v = &c.args[a].view;
                if v.is_gather() {
                    conv[a] = Some(CollKind::AllReduce);
                    continue;
                }
                let forms = m.forms.iter().map(|f| f.compose(&v.forms, rank)).collect();
                masks.push((a, Mask { idx: m.idx, forms }));
            }
            _ => {}
        }
    }
    if !masks.is_empty() {
        let ep_mask = masks
            .iter()
            .find(|(a, _)| pl(*a, conv) == Place::Ep)
            .map(|(_, m)| m.clone());
        let want = ep_mask.unwrap_or_else(|| masks[0].1.clone());
        let mut keep: Vec<usize> = Vec::new();
        for (a, m) in &masks {
            if *m == want {
                keep.push(*a);
            } else {
                assert!(
                    pl(*a, conv) != Place::Ep,
                    "expert weights of one comp under different masks"
                );
                conv[*a] = Some(CollKind::AllReduce);
            }
        }
        let ok = sum != Some(false) && vanishes(c, &keep);
        if !ok {
            assert!(
                !has_ep,
                "an expert-parallel comp whose body does not vanish off its experts"
            );
            for a in keep {
                conv[a] = Some(CollKind::AllReduce);
            }
        } else {
            // P args alongside a mask: read them whole
            for a in 0..c.args.len() {
                if pl(a, conv) == Place::P {
                    conv[a] = Some(CollKind::AllReduce);
                }
            }
            let over_red = want
                .forms
                .iter()
                .any(|f| f.coefs[np..].iter().any(|&x| x != 0));
            if over_red {
                return (Place::P, None);
            }
            let forms = want
                .forms
                .iter()
                .map(|f| Form {
                    coefs: f.coefs[..np].to_vec(),
                    offset: f.offset,
                })
                .collect();
            return (
                Place::M(Mask {
                    idx: want.idx,
                    forms,
                }),
                None,
            );
        }
    }
    // 3. partials
    let ps: Vec<usize> = (0..c.args.len())
        .filter(|&a| pl(a, conv) == Place::P)
        .collect();
    if !ps.is_empty() {
        if sum != Some(false) && linear_in(c, &ps) {
            return (Place::P, None);
        }
        for a in ps {
            conv[a] = Some(CollKind::AllReduce);
        }
    }
    (Place::R, None)
}

// ---------------------------------------------------------------------------
// Per-rank programs
// ---------------------------------------------------------------------------

/// The collectives of one rank's program, run against the other ranks.
pub trait Comm: Send + Sync + std::panic::RefUnwindSafe {
    fn all_reduce(&self, rank: usize, x: &[f64]) -> Vec<f64>;
    /// `x` is the rank's chunk `[outer, chunk·inner]`; the result is
    /// `[outer, world·chunk·inner]`, chunks in rank order.
    fn all_gather(&self, rank: usize, x: &[f64], outer: usize) -> Vec<f64>;
}

/// Ranks as threads of one process: a barrier and a shared slot table.
/// Sums in rank order, so every rank gets the same bits.
pub struct HostComm {
    world: usize,
    barrier: Barrier,
    slots: Mutex<Vec<Vec<f64>>>,
}

impl HostComm {
    pub fn new(world: usize) -> Arc<Self> {
        Arc::new(Self {
            world,
            barrier: Barrier::new(world),
            slots: Mutex::new(vec![Vec::new(); world]),
        })
    }

    fn exchange(&self, rank: usize, x: &[f64]) -> Vec<Vec<f64>> {
        self.slots.lock().unwrap()[rank] = x.to_vec();
        self.barrier.wait();
        let all = self.slots.lock().unwrap().clone();
        self.barrier.wait();
        all
    }
}

impl std::panic::RefUnwindSafe for HostComm {}

impl Comm for HostComm {
    fn all_reduce(&self, rank: usize, x: &[f64]) -> Vec<f64> {
        let all = self.exchange(rank, x);
        let mut out = all[0].clone();
        for p in &all[1..] {
            for (o, v) in out.iter_mut().zip(p) {
                *o += v;
            }
        }
        out
    }

    fn all_gather(&self, rank: usize, x: &[f64], outer: usize) -> Vec<f64> {
        let all = self.exchange(rank, x);
        let row = x.len() / outer;
        let mut out = Vec::with_capacity(x.len() * self.world);
        for o in 0..outer {
            for p in &all {
                out.extend_from_slice(&p[o * row..(o + 1) * row]);
            }
        }
        out
    }
}

struct CollOp {
    kind: CollKind,
    rank: usize,
    outer: usize,
    comm: Arc<dyn Comm>,
}

impl Operator for CollOp {
    fn name(&self) -> &str {
        match self.kind {
            CollKind::AllReduce => "all_reduce",
            CollKind::AllGather(_) => "all_gather",
        }
    }

    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        match self.kind {
            CollKind::AllReduce => self.comm.all_reduce(self.rank, inputs[0]),
            CollKind::AllGather(_) => self.comm.all_gather(self.rank, inputs[0], self.outer),
        }
    }

    fn backward(
        &self,
        _g: &mut TensorGraph,
        _inputs: &[TensorId],
        _output: TensorId,
        _cot: TensorId,
    ) -> Result<Vec<Option<TensorId>>, ShapeError> {
        err("collectives have no adjoint here")
    }
}

/// Rank `r`'s program and how to cut its inputs from the full tensors.
pub struct RankProgram {
    pub graph: TensorGraph,
    pub outputs: Vec<TensorId>,
    /// Per graph input index `k`: the box `[lo, hi)` per dim of the full
    /// input this rank stores (full extent for replicated activations).
    pub boxes: Vec<Vec<(usize, usize)>>,
}

impl RankProgram {
    /// Cut full row-major `inputs` to this rank's boxes.
    pub fn local_inputs(&self, full: &[Vec<f64>], shapes: &[Vec<usize>]) -> Vec<Vec<f64>> {
        full.iter()
            .zip(shapes)
            .zip(&self.boxes)
            .map(|((x, s), b)| cut(x, s, b))
            .collect()
    }

    /// Elements this rank stores of each input.
    pub fn stored(&self, k: usize) -> usize {
        self.boxes[k].iter().map(|(a, b)| b - a).product()
    }
}

pub fn cut(x: &[f64], shape: &[usize], bx: &[(usize, usize)]) -> Vec<f64> {
    if bx.iter().zip(shape).all(|(&(a, b), &n)| a == 0 && b == n) {
        return x.to_vec();
    }
    let ext: Vec<usize> = bx.iter().map(|(a, b)| b - a).collect();
    let total: usize = ext.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..total {
        let mut off = 0;
        for d in 0..shape.len() {
            off = off * shape[d] + bx[d].0 + idx[d];
        }
        out.push(x[off]);
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < ext[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

/// Rank `rank`'s program of `g` under `an`. `weights`: the inputs that may
/// be stored cut to the box the program reads.
pub fn rank_program(
    g: &TensorGraph,
    outputs: &[TensorId],
    an: &Analysis,
    weights: &HashSet<TensorId>,
    rank: usize,
    comm: Arc<dyn Comm>,
) -> RankProgram {
    let w = an.world;
    let live = g.live(outputs);
    let mut ng = TensorGraph {
        nodes: Vec::new(),
        n_inputs: g.n_inputs,
    };
    let mut remap: Vec<Option<TensorId>> = vec![None; g.nodes.len()];
    let mut colls: HashMap<(usize, CollKind), TensorId> = HashMap::new();
    let mut boxes: Vec<Vec<(usize, usize)>> = vec![Vec::new(); g.n_inputs];
    // a collective on the (local) source `s` of global node `src`
    let mut collective = |ng: &mut TensorGraph,
                          src: usize,
                          kind: CollKind,
                          remap: &[Option<TensorId>]|
     -> TensorId {
        if let Some(&t) = colls.get(&(src, kind)) {
            return t;
        }
        let full = fixed(&g.nodes[src].shape);
        let outer = match kind {
            CollKind::AllGather(d) => full[..d].iter().product(),
            CollKind::AllReduce => 1,
        };
        let op = Arc::new(CollOp {
            kind,
            rank,
            outer,
            comm: comm.clone(),
        });
        let t = ng.operator(op, &[remap[src].unwrap()], &full);
        colls.insert((src, kind), t);
        t
    };
    let local_shape = |shape: &[usize], p: &Place| -> Vec<usize> {
        let mut s = shape.to_vec();
        match p {
            Place::S(d) => s[*d] /= w,
            Place::Ep => s[0] /= w,
            _ => {}
        }
        s
    };
    for i in (0..g.nodes.len()).filter(|&i| live[i]) {
        let mut node = g.nodes[i].clone();
        let full = fixed(&node.shape);
        let p = &an.place[i];
        match &mut node.op {
            Op::Input(k) => {
                boxes[*k] = full
                    .iter()
                    .enumerate()
                    .map(|(d, &n)| match p {
                        Place::S(sd) if *sd == d => (rank * n / w, (rank + 1) * n / w),
                        Place::Ep if d == 0 => (rank * n / w, (rank + 1) * n / w),
                        _ => (0, n),
                    })
                    .collect();
                node.shape = local_shape(&full, p).into_iter().map(Dim::Fixed).collect();
            }
            Op::Iota(_) => {}
            Op::Reshape(src) => {
                let s = src.n();
                *src = match an.conv[i][0] {
                    Some(k) => collective(&mut ng, s, k, &remap),
                    None => remap[s].unwrap(),
                };
                node.shape = local_shape(&full, p).into_iter().map(Dim::Fixed).collect();
            }
            Op::Operator(o) => {
                for (k, t) in o.inputs.iter_mut().enumerate() {
                    let s = t.n();
                    *t = match an.conv[i][k] {
                        Some(kind) => collective(&mut ng, s, kind, &remap),
                        None => remap[s].unwrap(),
                    };
                }
            }
            Op::Scan(sc) => {
                for (k, t) in sc.inputs.iter_mut().enumerate() {
                    let s = t.n();
                    *t = match an.conv[i][k] {
                        Some(kind) => collective(&mut ng, s, kind, &remap),
                        None => remap[s].unwrap(),
                    };
                }
                // split by a leading (independent) dim: each rank scans its slice
                node.shape = local_shape(&full, p).into_iter().map(Dim::Fixed).collect();
            }
            Op::Comp(c) => {
                let space: Vec<usize> = c.space().iter().map(|d| d.n()).collect();
                let np = c.points().len();
                let j = an.split[i];
                for (k, a) in c.args.iter_mut().enumerate() {
                    let s = a.src.n();
                    let src_place = match an.conv[i][k] {
                        Some(kind) => {
                            a.src = collective(&mut ng, s, kind, &remap);
                            Place::R
                        }
                        None => {
                            a.src = remap[s].unwrap();
                            an.place[s].clone()
                        }
                    };
                    if let Some(j) = j {
                        let lo = (rank * space[j] / w) as i64;
                        for (d, f) in a.view.forms.iter_mut().enumerate() {
                            if src_place == Place::S(d) {
                                continue;
                            }
                            f.offset += f.coefs[j] * lo;
                        }
                    }
                    if src_place == Place::Ep {
                        let e = g.nodes[s].shape[0].n();
                        a.view.forms[0].offset -= (rank * e / w) as i64;
                        a.view.guard = true;
                    }
                }
                if let Some(j) = j {
                    if j < np {
                        c.out[j] = Dim::Fixed(space[j] / w);
                    } else {
                        c.red[j - np] = Dim::Fixed(space[j] / w);
                    }
                }
                node.shape = c.out.clone();
            }
        }
        ng.nodes.push(node);
        remap[i] = Some(TensorId(ng.nodes.len() as u32 - 1));
    }
    let outs: Vec<TensorId> = outputs
        .iter()
        .zip(&an.out_conv)
        .map(|(o, k)| match k {
            Some(kind) => collective(&mut ng, o.n(), *kind, &remap),
            None => remap[o.n()].unwrap(),
        })
        .collect();
    // cut replicated weights to what this rank reads
    let wl: HashSet<usize> = weights
        .iter()
        .filter_map(|t| remap[t.n()])
        .map(|t| t.n())
        .collect();
    // readers of each weight: (node, comp arg) or (node, None) for a non-comp reader
    let mut readers: HashMap<usize, Vec<(usize, Option<usize>)>> = HashMap::new();
    for i in 0..ng.nodes.len() {
        match &ng.nodes[i].op {
            Op::Comp(c) => {
                for (ai, a) in c.args.iter().enumerate() {
                    if wl.contains(&a.src.n()) {
                        readers.entry(a.src.n()).or_default().push((i, Some(ai)));
                    }
                }
            }
            _ => {
                for s in dep_srcs(&ng, i) {
                    if wl.contains(&s.n()) {
                        readers.entry(s.n()).or_default().push((i, None));
                    }
                }
            }
        }
    }
    for (t, rs) in readers {
        let Op::Input(k) = ng.nodes[t].op else {
            continue;
        };
        if rs.iter().any(|(_, a)| a.is_none()) {
            continue;
        }
        let shape = fixed(&ng.nodes[t].shape);
        let mut lo: Vec<i64> = vec![i64::MAX; shape.len()];
        let mut hi: Vec<i64> = vec![i64::MIN; shape.len()];
        for &(i, ai) in &rs {
            let Op::Comp(c) = &ng.nodes[i].op else {
                unreachable!()
            };
            let space: Vec<usize> = c.space().iter().map(|d| d.n()).collect();
            let a = &c.args[ai.unwrap()];
            for (d, f) in a.view.forms.iter().enumerate() {
                let gathered = a.view.gather.as_ref().is_some_and(|gg| gg.coefs[d] != 0);
                let (l, h) = if gathered {
                    (0, shape[d] as i64 - 1)
                } else {
                    f.range(&space)
                };
                let (l, h) = if a.view.guard {
                    (l.max(0), h.min(shape[d] as i64 - 1))
                } else {
                    (l, h)
                };
                lo[d] = lo[d].min(l);
                hi[d] = hi[d].max(h);
            }
        }
        if (0..shape.len()).all(|d| lo[d] == 0 && hi[d] == shape[d] as i64 - 1) {
            continue;
        }
        for &(i, ai) in &rs {
            let Op::Comp(c) = &mut ng.nodes[i].op else {
                unreachable!()
            };
            for (d, f) in c.args[ai.unwrap()].view.forms.iter_mut().enumerate() {
                f.offset -= lo[d];
            }
        }
        ng.nodes[t].shape = (0..shape.len())
            .map(|d| Dim::Fixed((hi[d] - lo[d] + 1) as usize))
            .collect();
        for d in 0..shape.len() {
            let base = boxes[k][d].0;
            boxes[k][d] = (base + lo[d] as usize, base + hi[d] as usize + 1);
        }
    }
    RankProgram {
        graph: ng,
        outputs: outs,
        boxes,
    }
}

/// Run every rank's program on its threads (inputs full, cut per rank);
/// returns rank 0's outputs (all ranks' outputs are replicated).
pub fn simulate(
    g: &TensorGraph,
    outputs: &[TensorId],
    an: &Analysis,
    weights: &HashSet<TensorId>,
    inputs: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let comm = HostComm::new(an.world);
    let shapes: Vec<Vec<usize>> = input_shapes(g);
    let progs: Vec<RankProgram> = (0..an.world)
        .map(|r| rank_program(g, outputs, an, weights, r, comm.clone()))
        .collect();
    let res: Vec<Vec<Vec<f64>>> = std::thread::scope(|s| {
        let hs: Vec<_> = progs
            .iter()
            .map(|p| {
                let ins = p.local_inputs(inputs, &shapes);
                s.spawn(move || p.graph.eval(&ins, &p.outputs))
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for r in 1..res.len() {
        let same = res[r]
            .iter()
            .zip(&res[0])
            .zip(&an.out_local)
            .all(|((a, b), l)| {
                l.is_some()
                    || (a.len() == b.len()
                        && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()))
            });
        assert!(same, "rank {r} disagrees with rank 0");
    }
    assemble(g, outputs, an, &res)
}

/// The step's outputs from every rank's: replicated outputs from rank 0,
/// outputs left split ([`Analysis::out_local`]) concatenated along their dim.
pub fn assemble(
    g: &TensorGraph,
    outputs: &[TensorId],
    an: &Analysis,
    ranks: &[Vec<Vec<f64>>],
) -> Vec<Vec<f64>> {
    (0..outputs.len())
        .map(|o| match an.out_local[o] {
            None => ranks[0][o].clone(),
            Some(d) => {
                let full = fixed(&g.nodes[outputs[o].n()].shape);
                let outer: usize = full[..d].iter().product();
                let row = ranks[0][o].len() / outer;
                let mut v = Vec::with_capacity(row * outer * an.world);
                for i in 0..outer {
                    for r in ranks {
                        v.extend_from_slice(&r[o][i * row..(i + 1) * row]);
                    }
                }
                v
            }
        })
        .collect()
}

/// Full shapes of the graph inputs by input index.
pub fn input_shapes(g: &TensorGraph) -> Vec<Vec<usize>> {
    let mut v = vec![Vec::new(); g.n_inputs];
    for n in &g.nodes {
        if let Op::Input(k) = n.op {
            v[k] = fixed(&n.shape);
        }
    }
    v
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

/// What a decode step costs on one rank: weight bytes over HBM bandwidth
/// plus per-collective latency and ring volume over the link.
#[derive(Clone, Copy, Debug)]
pub struct CostModel {
    pub hbm: f64,
    pub link: f64,
    pub latency: f64,
    /// Bytes per activation element on the wire.
    pub act_bytes: f64,
}

impl CostModel {
    /// 8× H200 SXM: 4.8 TB/s HBM, NVLink ~450 GB/s a direction, 12 µs per
    /// small collective (`scripts/ds-roofline.py`).
    pub fn h200() -> Self {
        Self {
            hbm: 4.8e12,
            link: 450e9,
            latency: 12e-6,
            act_bytes: 4.0,
        }
    }

    /// Collectives free: split everything the rules can (a stress plan).
    pub fn eager() -> Self {
        Self {
            hbm: 1.0,
            link: f64::INFINITY,
            latency: 0.0,
            act_bytes: 0.0,
        }
    }
}

/// Per-rank step estimate of `an`'s rank-0 program.
pub struct StepCost {
    pub weight_bytes: f64,
    pub coll_bytes: f64,
    pub n_coll: usize,
    pub seconds: f64,
}

pub fn step_cost(
    g: &TensorGraph,
    outputs: &[TensorId],
    an: &Analysis,
    weights: &HashSet<TensorId>,
    elem: &dyn Fn(TensorId) -> f64,
    cm: &CostModel,
) -> StepCost {
    // a dummy comm: the program is only measured
    let p = rank_program(g, outputs, an, weights, 0, HostComm::new(an.world));
    // a weight read through a gather (an embedding row) costs rows, not the table
    let gathered: HashSet<usize> = g
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            Op::Comp(c) => Some(c),
            _ => None,
        })
        .flat_map(|c| {
            c.args
                .iter()
                .filter(|a| {
                    a.view
                        .gather
                        .as_ref()
                        .is_some_and(|gg| gg.coefs.iter().any(|&x| x != 0))
                })
                .map(|a| a.src.n())
        })
        .collect();
    let mut wb = 0.0;
    for (i, n) in g.nodes.iter().enumerate() {
        if gathered.contains(&i) {
            continue;
        }
        if let Op::Input(k) = n.op {
            let t = TensorId(i as u32);
            if weights.contains(&t) && an.place[i] != Place::Ep && !p.boxes[k].is_empty() {
                wb += p.stored(k) as f64 * elem(t);
            }
        }
    }
    let cb: f64 = an
        .collectives
        .iter()
        .map(|c| c.ring_bytes(an.world, cm.act_bytes))
        .sum();
    let n_coll = an.collectives.len();
    StepCost {
        weight_bytes: wb,
        coll_bytes: cb,
        n_coll,
        seconds: wb / cm.hbm + cb / cm.link + n_coll as f64 * cm.latency,
    }
}

/// Inputs only ever read through a gather on dim 0 whose index comes from
/// a top-k reduction (directly or through one gathered remap): routed
/// experts.
pub fn expert_inputs(g: &TensorGraph, outputs: &[TensorId], world: usize) -> HashSet<TensorId> {
    let live = g.live(outputs);
    let from_topk = |t: TensorId| -> bool {
        let is_topk = |t: TensorId| matches!(&g.nodes[t.n()].op, Op::Comp(c) if c.monoid.as_ref().is_some_and(|m| m.width >= 2 && m.width % 2 == 0 && m.matches(&Monoid::topk(m.width / 2))));
        is_topk(t)
            || matches!(&g.nodes[t.n()].op, Op::Comp(c) if c.monoid.is_none() && c.args.iter().any(|a| is_topk(a.src)))
    };
    let mut ok: HashMap<usize, bool> = HashMap::new();
    for i in (0..g.nodes.len()).filter(|&i| live[i]) {
        match &g.nodes[i].op {
            Op::Comp(c) => {
                for a in &c.args {
                    if !matches!(g.nodes[a.src.n()].op, Op::Input(_)) {
                        continue;
                    }
                    let good = a.view.gather.as_ref().is_some_and(|gg| {
                        gg.coefs[0] != 0
                            && gg.coefs[1..].iter().all(|&x| x == 0)
                            && a.view.forms[0].coefs.iter().all(|&x| x == 0)
                            && from_topk(c.args[gg.arg].src)
                    }) && c.args.iter().all(|b| {
                        b.view
                            .gather
                            .as_ref()
                            .is_none_or(|gg| c.args[gg.arg].src != a.src)
                    });
                    let e = ok.entry(a.src.n()).or_insert(true);
                    *e &= good;
                }
            }
            _ => {
                for s in dep_srcs(g, i) {
                    ok.insert(s.n(), false);
                }
            }
        }
    }
    ok.into_iter()
        .filter(|&(t, good)| {
            good && g.nodes[t].shape[0].n() % world == 0 && g.nodes[t].shape[0].n() >= world
        })
        .map(|(t, _)| TensorId(t as u32))
        .collect()
}

/// The plan: experts by [`expert_inputs`], then a greedy pass over the
/// other weights (largest first) keeping each `Split(d)` that lowers
/// [`step_cost`].
pub fn plan_auto(
    g: &TensorGraph,
    outputs: &[TensorId],
    weights: &HashSet<TensorId>,
    caches: &[(TensorId, usize)],
    world: usize,
    elem: &dyn Fn(TensorId) -> f64,
    cm: &CostModel,
) -> ShardPlan {
    let mut plan = ShardPlan {
        world,
        inputs: HashMap::new(),
        caches: caches.to_vec(),
    };
    if world == 1 {
        return plan;
    }
    for t in expert_inputs(g, outputs, world) {
        if weights.contains(&t) {
            plan.inputs.insert(t, WeightPlace::Experts);
        }
    }
    let cost = |plan: &ShardPlan| {
        let an = analyze(g, outputs, plan);
        step_cost(g, outputs, &an, weights, elem, cm).seconds
    };
    let mut best = cost(&plan);
    let mut cands: Vec<TensorId> = weights
        .iter()
        .copied()
        .filter(|t| !plan.inputs.contains_key(t) && g.nodes[t.n()].shape.len() >= 2)
        // too small to matter: saving all of it is under 5% of one collective
        .filter(|t| numel(&g.nodes[t.n()].shape) as f64 * elem(*t) / cm.hbm >= 0.05 * cm.latency)
        .collect();
    cands.sort_by_key(|t| (std::cmp::Reverse(numel(&g.nodes[t.n()].shape)), t.n()));
    let mut failed = Vec::new();
    for t in cands {
        let shape = fixed(&g.nodes[t.n()].shape);
        let mut pick = None;
        for d in 0..shape.len() {
            if shape[d] % world != 0 || shape[d] < world {
                continue;
            }
            plan.inputs.insert(t, WeightPlace::Split(d));
            let c = cost(&plan);
            if c < best * (1.0 - 1e-9) {
                best = c;
                pick = Some(d);
            }
        }
        match pick {
            Some(d) => plan.inputs.insert(t, WeightPlace::Split(d)),
            None => plan.inputs.remove(&t),
        };
        if pick.is_none() {
            failed.push(t);
        }
    }
    // Joint moves: a block split by head (GDN: in-projections, conv, decay
    // and output projection) only pays once every weight in it is split; one
    // alone just adds gathers. For each split that lost on its own, keep it
    // tentatively and greedily co-split the weights built next to it (input
    // order follows the builder, so neighbours are the same layer, 1-D ones
    // included); adopt the group if the step got cheaper.
    let mut near: Vec<TensorId> = weights
        .iter()
        .copied()
        .filter(|t| !plan.inputs.contains_key(t))
        .collect();
    near.sort_by_key(|t| t.n());
    // repeated layers repeat their neighbourhoods: search each shape pattern
    // once, then replay its answer (one cost check) at the other layers
    type Sig = (Vec<usize>, Vec<Vec<usize>>);
    let mut memo: HashMap<Sig, Option<Vec<(usize, usize)>>> = HashMap::new();
    for t in failed {
        if plan.inputs.contains_key(&t) {
            continue;
        }
        let at = near.iter().position(|u| *u == t).unwrap();
        let hood: Vec<TensorId> = near
            [at.saturating_sub(JOINT_REACH)..(at + JOINT_REACH + 1).min(near.len())]
            .iter()
            .copied()
            .filter(|u| *u != t && !plan.inputs.contains_key(u))
            .collect();
        let shape = fixed(&g.nodes[t.n()].shape);
        let sig: Sig = (
            shape.clone(),
            hood.iter().map(|u| fixed(&g.nodes[u.n()].shape)).collect(),
        );
        if let Some(known) = memo.get(&sig) {
            // (hood index + 1, dim); 0 is t itself
            if let Some(rel) = known {
                let mut trial = plan.clone();
                for &(k, e) in rel {
                    let u = if k == 0 { t } else { hood[k - 1] };
                    trial.inputs.insert(u, WeightPlace::Split(e));
                }
                let c = cost(&trial);
                if c < best * (1.0 - 1e-9) {
                    best = c;
                    plan = trial;
                }
            }
            continue;
        }
        let mut adopt: Option<(f64, Vec<(TensorId, usize)>)> = None;
        for d in 0..shape.len() {
            if shape[d] % world != 0 || shape[d] < world {
                continue;
            }
            let mut trial = plan.clone();
            trial.inputs.insert(t, WeightPlace::Split(d));
            let mut group = vec![(t, d)];
            let mut cur = cost(&trial);
            for &u in &hood {
                let us = fixed(&g.nodes[u.n()].shape);
                let mut upick = None;
                for e in 0..us.len() {
                    if us[e] % world != 0 || us[e] < world {
                        continue;
                    }
                    trial.inputs.insert(u, WeightPlace::Split(e));
                    let c = cost(&trial);
                    if c < cur * (1.0 - 1e-9) {
                        cur = c;
                        upick = Some(e);
                    }
                }
                match upick {
                    Some(e) => {
                        trial.inputs.insert(u, WeightPlace::Split(e));
                        group.push((u, e));
                    }
                    None => {
                        trial.inputs.remove(&u);
                    }
                }
            }
            let bar = adopt.as_ref().map_or(best, |a| a.0);
            if cur < bar * (1.0 - 1e-9) {
                adopt = Some((cur, group));
            }
        }
        let rel = adopt.as_ref().map(|(_, group)| {
            group
                .iter()
                .map(|&(u, e)| {
                    (
                        if u == t {
                            0
                        } else {
                            1 + hood.iter().position(|h| *h == u).unwrap()
                        },
                        e,
                    )
                })
                .collect()
        });
        memo.insert(sig, rel);
        if let Some((c, group)) = adopt {
            best = c;
            for (u, e) in group {
                plan.inputs.insert(u, WeightPlace::Split(e));
            }
        }
    }
    plan
}

/// How many weights on each side (in input order) a joint move tries.
const JOINT_REACH: usize = 12;

/// One line about node `t`: its op, args and views (debugging plans).
pub fn describe(g: &TensorGraph, t: TensorId) -> String {
    match &g.nodes[t.n()].op {
        Op::Input(k) => format!("input {k}"),
        Op::Iota(d) => format!("iota {d}"),
        Op::Reshape(s) => format!("reshape of {}", s.n()),
        Op::Operator(o) => format!("operator {} of {:?}", o.op.name(), o.inputs),
        Op::Scan(s) => format!("scan dim {} of {:?}", s.dim, s.inputs),
        Op::Comp(c) => {
            let args: Vec<String> = c
                .args
                .iter()
                .map(|a| {
                    let fs: Vec<String> = a
                        .view
                        .forms
                        .iter()
                        .map(|f| format!("{:?}+{}", f.coefs, f.offset))
                        .collect();
                    format!(
                        "{}[{}]{}{}",
                        a.src.n(),
                        fs.join(","),
                        if a.view.guard { "g" } else { "" },
                        a.view.gather.as_ref().map_or(String::new(), |gg| format!(
                            " gather@{}{:?}",
                            gg.arg, gg.coefs
                        ))
                    )
                })
                .collect();
            let nodes = c.body.nodes_slice();
            let n = c
                .body_outs
                .iter()
                .map(|o| o.index() as usize + 1)
                .max()
                .unwrap_or(0);
            let body: Vec<String> = (0..n).map(|i| format!("{i}:{:?}", nodes[i])).collect();
            format!(
                "comp out {:?} red {:?} {:?} args {} body [{}] outs {:?}",
                c.out,
                c.red,
                c.monoid.as_ref().map(|m| m.kind()),
                args.join("; "),
                body.join(" "),
                c.body_outs.iter().map(|o| o.index()).collect::<Vec<_>>()
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Segments: a rank program cut at its collectives, for backends that do not
// lower operator nodes (every device backend today)
// ---------------------------------------------------------------------------

/// Where a segment input comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feed {
    /// Rank program input `k`.
    Input(usize),
    /// The value of rank-program node `n` (an earlier segment's output or
    /// a collective's result).
    Value(usize),
}

/// One collective between segments, on rank-program nodes.
#[derive(Clone, Copy, Debug)]
pub struct Boundary {
    pub kind: CollKind,
    /// The node it reads and the node it defines.
    pub src: usize,
    pub dst: usize,
    /// For an all-gather: the volume before the gathered dim.
    pub outer: usize,
}

/// A collective-free piece of a rank program.
pub struct Segment {
    pub graph: TensorGraph,
    pub feeds: Vec<Feed>,
    /// Segment outputs and the rank-program node each one is.
    pub outputs: Vec<TensorId>,
    pub defines: Vec<usize>,
}

/// A rank program as `[collectives before 0] segment 0 [collectives before
/// 1] segment 1 …`: `before[s]` run once segments `< s` are done.
pub struct Segmented {
    pub segments: Vec<Segment>,
    pub before: Vec<Vec<Boundary>>,
    /// The rank program's outputs as rank-program nodes.
    pub outputs: Vec<usize>,
    /// Input nodes and their input index (a collective may read one).
    pub input_nodes: Vec<(usize, usize)>,
}

/// Cut `p` at its collective nodes. A node's level is the number of
/// collectives on its longest input path; segment `s` holds the level-`s`
/// nodes, and reads anything earlier as a fed input.
pub fn segments(p: &RankProgram) -> Segmented {
    let g = &p.graph;
    let n = g.nodes.len();
    let is_coll = |i: usize| matches!(&g.nodes[i].op, Op::Operator(o) if o.op.name() == "all_reduce" || o.op.name() == "all_gather");
    let mut level = vec![0usize; n];
    for i in 0..n {
        let deps = g.deps(i);
        let l = deps.iter().map(|&d| level[d]).max().unwrap_or(0);
        level[i] = if is_coll(i) { l + 1 } else { l };
    }
    let n_seg = (0..n).map(|i| level[i]).max().unwrap_or(0) + 1;
    let mut before: Vec<Vec<Boundary>> = vec![Vec::new(); n_seg];
    let mut used_later = vec![false; n];
    for i in 0..n {
        for d in g.deps(i) {
            if is_coll(i) || level[d] < level[i] {
                used_later[d] = true;
            }
        }
    }
    for o in &p.outputs {
        used_later[o.n()] = true;
    }
    let mut segs = Vec::with_capacity(n_seg);
    for s in 0..n_seg {
        let mut ng = TensorGraph {
            nodes: Vec::new(),
            n_inputs: 0,
        };
        let mut feeds = Vec::new();
        let mut map: HashMap<usize, TensorId> = HashMap::new();
        let fed = |ng: &mut TensorGraph,
                   feeds: &mut Vec<Feed>,
                   map: &mut HashMap<usize, TensorId>,
                   d: usize|
         -> TensorId {
            if let Some(&t) = map.get(&d) {
                return t;
            }
            let feed = match g.nodes[d].op {
                Op::Input(k) => Feed::Input(k),
                _ => Feed::Value(d),
            };
            let t = ng.input(&g.nodes[d].shape.clone());
            feeds.push(feed);
            map.insert(d, t);
            t
        };
        let (mut outputs, mut defines) = (Vec::new(), Vec::new());
        for i in (0..n).filter(|&i| level[i] == s) {
            if is_coll(i) {
                let Op::Operator(o) = &g.nodes[i].op else {
                    unreachable!()
                };
                let src = o.inputs[0].n();
                let full = fixed(&g.nodes[i].shape);
                let (kind, outer) = if o.op.name() == "all_reduce" {
                    (CollKind::AllReduce, 1)
                } else {
                    let local = fixed(&g.nodes[src].shape);
                    let d = (0..full.len()).find(|&d| full[d] != local[d]).unwrap_or(0);
                    (CollKind::AllGather(d), full[..d].iter().product())
                };
                before[s].push(Boundary {
                    kind,
                    src,
                    dst: i,
                    outer,
                });
                continue;
            }
            if matches!(g.nodes[i].op, Op::Input(_)) {
                // inputs are fed where they are read
                continue;
            }
            let mut node = g.nodes[i].clone();
            let mut remap = |t: &mut TensorId| {
                let d = t.n();
                *t = match map.get(&d) {
                    Some(&m) if level[d] == s && !matches!(g.nodes[d].op, Op::Input(_)) => m,
                    Some(&m) if matches!(g.nodes[d].op, Op::Iota(_)) => m,
                    // an iota is rebuilt where it is read (it has no buffer to pass)
                    _ if matches!(g.nodes[d].op, Op::Iota(_)) => {
                        ng.nodes.push(g.nodes[d].clone());
                        let m = TensorId(ng.nodes.len() as u32 - 1);
                        map.insert(d, m);
                        m
                    }
                    _ => fed(&mut ng, &mut feeds, &mut map, d),
                };
            };
            match &mut node.op {
                Op::Comp(c) => c.args.iter_mut().for_each(|a| remap(&mut a.src)),
                Op::Reshape(src) => remap(src),
                Op::Operator(o) => o.inputs.iter_mut().for_each(&mut remap),
                Op::Scan(sc) => sc.inputs.iter_mut().for_each(&mut remap),
                Op::Input(_) | Op::Iota(_) => {}
            }
            ng.nodes.push(node);
            let t = TensorId(ng.nodes.len() as u32 - 1);
            map.insert(i, t);
            if used_later[i] && !matches!(g.nodes[i].op, Op::Iota(_)) {
                outputs.push(t);
                defines.push(i);
            }
        }
        segs.push(Segment {
            graph: ng,
            feeds,
            outputs,
            defines,
        });
    }
    Segmented {
        segments: segs,
        before,
        outputs: p.outputs.iter().map(|t| t.n()).collect(),
        input_nodes: (0..n)
            .filter_map(|i| match g.nodes[i].op {
                Op::Input(k) => Some((i, k)),
                _ => None,
            })
            .collect(),
    }
}

/// Run a segmented rank program on the interpreter, collectives through
/// `comm` (what a device runner does with a `Program` per segment).
pub fn run_segmented(
    sg: &Segmented,
    rank: usize,
    comm: &dyn Comm,
    inputs: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let mut vals: HashMap<usize, Vec<f64>> = sg
        .input_nodes
        .iter()
        .map(|&(i, k)| (i, inputs[k].clone()))
        .collect();
    for (s, seg) in sg.segments.iter().enumerate() {
        for b in &sg.before[s] {
            let x = &vals[&b.src];
            let y = match b.kind {
                CollKind::AllReduce => comm.all_reduce(rank, x),
                CollKind::AllGather(_) => comm.all_gather(rank, x, b.outer),
            };
            vals.insert(b.dst, y);
        }
        let ins: Vec<Vec<f64>> = seg
            .feeds
            .iter()
            .map(|f| match f {
                Feed::Input(k) => inputs[*k].clone(),
                Feed::Value(n) => vals[n].clone(),
            })
            .collect();
        let outs = seg.graph.eval(&ins, &seg.outputs);
        for (d, v) in seg.defines.iter().zip(outs) {
            vals.insert(*d, v);
        }
    }
    sg.outputs.iter().map(|o| vals[o].clone()).collect()
}

/// [`simulate`] through [`segments`] / [`run_segmented`].
pub fn simulate_segmented(
    g: &TensorGraph,
    outputs: &[TensorId],
    an: &Analysis,
    weights: &HashSet<TensorId>,
    inputs: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let comm = HostComm::new(an.world);
    let shapes = input_shapes(g);
    let progs: Vec<RankProgram> = (0..an.world)
        .map(|r| rank_program(g, outputs, an, weights, r, comm.clone()))
        .collect();
    let res: Vec<Vec<Vec<f64>>> = std::thread::scope(|s| {
        let hs: Vec<_> = progs
            .iter()
            .enumerate()
            .map(|(r, p)| {
                let ins = p.local_inputs(inputs, &shapes);
                let comm = comm.clone();
                s.spawn(move || run_segmented(&segments(p), r, &*comm, &ins))
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assemble(g, outputs, an, &res)
}

/// An all-gather done by a collective that concatenates rank-major (NCCL:
/// rank `r`'s whole chunk at `r · len`) laid out as [`Comm::all_gather`]
/// wants it (`[outer, world · inner]`): the copies `(src, dst, len)` in
/// elements from the rank-major buffer to the result. One copy when
/// `outer == 1`.
pub fn gather_copies(world: usize, outer: usize, local: usize) -> Vec<(usize, usize, usize)> {
    let row = local / outer;
    if outer == 1 {
        return vec![(0, 0, local * world)];
    }
    (0..outer)
        .flat_map(|o| (0..world).map(move |r| (r * local + o * row, (o * world + r) * row, row)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_copies_match_host_all_gather() {
        for (world, outer, row) in [(2usize, 1usize, 3usize), (4, 3, 2), (8, 5, 1)] {
            let local = outer * row;
            let chunks: Vec<Vec<f64>> = (0..world)
                .map(|r| (0..local).map(|i| (r * 1000 + i) as f64).collect())
                .collect();
            let comm = HostComm::new(world);
            let want: Vec<Vec<f64>> = std::thread::scope(|s| {
                let hs: Vec<_> = (0..world)
                    .map(|r| {
                        let (c, x) = (comm.clone(), chunks[r].clone());
                        s.spawn(move || c.all_gather(r, &x, outer))
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let rank_major: Vec<f64> = chunks.concat();
            let mut got = vec![f64::NAN; local * world];
            for (src, dst, len) in gather_copies(world, outer, local) {
                got[dst..dst + len].copy_from_slice(&rank_major[src..src + len]);
            }
            assert_eq!(got, want[0], "world {world} outer {outer}");
        }
    }
}
