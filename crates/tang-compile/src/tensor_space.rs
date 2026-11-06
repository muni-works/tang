//! The schedule search space, declared rather than listed.
//!
//! A [`Knob`] is one schedule field: the values it may take on a given
//! node, how to set one, and how to read the node's current value back.
//! A [`Space`] is a set of knobs plus constraints between them; its
//! candidates are the constrained cartesian product. The built-in knobs
//! are the thread mapping (`split`, `tile`, `reg`) and Metal fragments;
//! new schedule fields (pipeline `stages`, an `mma` shape, split-K, …)
//! plug in with [`register_knob`] (process-wide, picked up by
//! [`Space::standard`]) or [`Space::with`] (one search), without touching
//! the tuner.
//!
//! [`features`] turns a (node, candidate) pair into numbers for a cost
//! model; [`CostModel`] is a small gradient-boosted regression-tree
//! ensemble over them, trained on logged autotune measurements and used
//! to decide which candidates are worth timing.

use super::*;
use std::sync::{Arc, Mutex};

/// One knob value: a few integers (a tile is three, a flag is one).
pub type Value = Vec<u32>;

type ValuesFn = dyn Fn(&TensorGraph, TensorId) -> Vec<Value> + Send + Sync;
type ApplyFn = dyn Fn(&mut TensorGraph, TensorId, &[u32]) + Send + Sync;
type ReadFn = dyn Fn(&TensorGraph, TensorId) -> Value + Send + Sync;
type ConstraintFn = dyn Fn(&Space, &Candidate) -> bool + Send + Sync;
/// A constraint between knob values: `false` drops the candidate.
pub type Constraint = Box<ConstraintFn>;

/// One schedule field the tuner may set.
#[derive(Clone)]
pub struct Knob {
    pub name: &'static str,
    values: Arc<ValuesFn>,
    apply: Arc<ApplyFn>,
    read: Arc<ReadFn>,
}

impl Knob {
    /// `values(g, t)`: the settings worth trying on `t` (the first is the
    /// neutral one, "off"); `apply` sets one; `read` returns `t`'s current
    /// setting (so a schedule found by rule is a candidate too).
    pub fn new(
        name: &'static str,
        values: impl Fn(&TensorGraph, TensorId) -> Vec<Value> + Send + Sync + 'static,
        apply: impl Fn(&mut TensorGraph, TensorId, &[u32]) + Send + Sync + 'static,
        read: impl Fn(&TensorGraph, TensorId) -> Value + Send + Sync + 'static,
    ) -> Self {
        Self {
            name,
            values: Arc::new(values),
            apply: Arc::new(apply),
            read: Arc::new(read),
        }
    }
}

/// One point of a [`Space`]: a value per knob, in knob order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Candidate(pub Vec<Value>);

impl Candidate {
    /// `name=v,v;name=v` — stable text for logs and caches.
    pub fn encode(&self, space: &Space) -> String {
        space
            .knobs
            .iter()
            .zip(&self.0)
            .map(|(k, v)| {
                let v: Vec<String> = v.iter().map(|x| x.to_string()).collect();
                format!("{}={}", k.name, v.join(","))
            })
            .collect::<Vec<_>>()
            .join(";")
    }

    /// Inverse of [`encode`](Self::encode); knobs missing from `text`
    /// (registered since it was written) take their neutral value.
    pub fn decode(space: &Space, g: &TensorGraph, t: TensorId, text: &str) -> Option<Self> {
        let mut vals: Vec<Option<Value>> = vec![None; space.knobs.len()];
        for part in text.split(';').filter(|p| !p.is_empty()) {
            let (name, v) = part.split_once('=')?;
            let k = space.knobs.iter().position(|k| k.name == name)?;
            vals[k] = Some(if v.is_empty() {
                Vec::new()
            } else {
                v.split(',')
                    .map(|x| x.parse().ok())
                    .collect::<Option<_>>()?
            });
        }
        Some(Candidate(
            vals.into_iter()
                .zip(&space.knobs)
                .map(|(v, k)| v.unwrap_or_else(|| (k.values)(g, t).swap_remove(0)))
                .collect(),
        ))
    }

    /// The value of knob `name`, if the space has it.
    pub fn get<'a>(&'a self, space: &Space, name: &str) -> Option<&'a [u32]> {
        let i = space.knobs.iter().position(|k| k.name == name)?;
        Some(&self.0[i])
    }
}

/// Knobs plus the constraints between them.
#[derive(Clone)]
pub struct Space {
    pub knobs: Vec<Knob>,
    constraints: Vec<Arc<ConstraintFn>>,
}

static REGISTRY: Mutex<Vec<(Knob, Vec<Arc<ConstraintFn>>)>> = Mutex::new(Vec::new());

/// Add a knob (and constraints involving it) to every [`Space::standard`]
/// built from now on, process-wide: how a new schedule field joins the
/// search. Registering a name twice replaces the earlier knob.
pub fn register_knob(
    knob: Knob,
    constraints: Vec<Constraint>,
) {
    let mut r = REGISTRY.lock().unwrap();
    r.retain(|(k, _)| k.name != knob.name);
    r.push((knob, constraints.into_iter().map(Arc::from).collect()));
}

impl Space {
    pub fn empty() -> Self {
        Self {
            knobs: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// The built-in knobs plus everything [`register_knob`]ed.
    pub fn standard() -> Self {
        let mut s = Self::builtin();
        for (k, cs) in REGISTRY.lock().unwrap().iter() {
            s.knobs.retain(|x| x.name != k.name);
            s.knobs.push(k.clone());
            s.constraints.extend(cs.iter().cloned());
        }
        s
    }

    /// This space with one more knob (replacing one of the same name).
    pub fn with(
        mut self,
        knob: Knob,
        constraint: Option<Constraint>,
    ) -> Self {
        self.knobs.retain(|x| x.name != knob.name);
        self.knobs.push(knob);
        self.constraints.extend(constraint.map(Arc::from));
        self
    }

    /// Keep only candidates for which `f` holds.
    pub fn constrain(
        mut self,
        f: impl Fn(&Space, &Candidate) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.constraints.push(Arc::new(f));
        self
    }

    /// `split`, `tile`, `reg` and `fragment`.
    pub fn builtin() -> Self {
        let split = Knob::new(
            "split",
            |g, t| {
                let mut v = vec![vec![0]];
                if let Some(c) = g.tiled_reduction(t.n()).and_then(|r| g.comp_at(r)) {
                    // a tree-combine of a wide tuple through workgroup
                    // memory is never the answer (59 ms for attention's
                    // 66-wide state on a 3090)
                    if c.width() <= 8 && numel(&c.red) >= 16 {
                        v.extend([vec![64], vec![128], vec![256]]);
                    }
                }
                v
            },
            |g, t, v| g.nodes[t.n()].split = v[0],
            |g, t| vec![g.nodes[t.n()].split],
        );
        let tile = Knob::new(
            "tile",
            |g, t| {
                let mut v = vec![vec![0, 0, 0]];
                let Some(c) = g.tiled_reduction(t.n()).and_then(|r| g.comp_at(r)) else {
                    return v;
                };
                if !tileable_red(&c.red) {
                    return v;
                }
                let pts = fixed(c.points());
                let (ax, ay) = tile_axes(&pts);
                let lanes = ax.map_or(1, |d| pts[d]);
                let rows = ay.is_some();
                let mut push = |bx: usize, by: usize, chunk: u32| {
                    let s = vec![bx as u32, by as u32, chunk];
                    if !v.contains(&s) {
                        v.push(s);
                    }
                };
                for chunk in [16, 32] {
                    if (16..=128).contains(&lanes) && rows {
                        // rows × lanes, the lane axis whole (attention's
                        // value columns)
                        for wg in [64, 128, 256] {
                            if wg >= lanes && wg % lanes == 0 {
                                push(lanes, wg / lanes, chunk);
                            }
                        }
                    }
                    if lanes >= 64 && !rows {
                        push(lanes.min(256), 1, chunk);
                    }
                    if lanes >= 16 && rows {
                        // 2-D shared-memory tiles (matmul-shaped comps)
                        for (bx, by) in [
                            (8, 8),
                            (16, 8),
                            (8, 16),
                            (16, 16),
                            (32, 8),
                            (32, 4),
                            (64, 4),
                        ] {
                            if bx <= lanes {
                                push(bx, by, chunk);
                            }
                        }
                    }
                }
                v
            },
            |g, t, v| g.nodes[t.n()].tile = (v[0], v[1], v[2]),
            |g, t| {
                let (a, b, c) = g.nodes[t.n()].tile;
                vec![a, b, c]
            },
        );
        let reg = Knob::new(
            "reg",
            |g, t| {
                let mut v = vec![vec![1, 1]];
                let tiled = g
                    .tiled_reduction(t.n())
                    .and_then(|r| g.comp_at(r))
                    .is_some_and(|c| tileable_red(&c.red));
                if tiled {
                    v.extend([
                        vec![2, 2],
                        vec![4, 4],
                        vec![2, 4],
                        vec![4, 2],
                        vec![1, 4],
                        vec![4, 1],
                    ]);
                }
                v
            },
            |g, t, v| g.nodes[t.n()].reg = (v[0], v[1]),
            |g, t| {
                let (a, b) = g.nodes[t.n()].reg;
                vec![a, b]
            },
        );
        let fragment = Knob::new(
            "fragment",
            |g, t| {
                let mut v = vec![vec![0]];
                if g.fragment_plan(t).is_some() {
                    v.push(vec![1]);
                }
                v
            },
            |g, t, v| g.set_fragment(t, v[0] != 0),
            |g, t| vec![g.nodes[t.n()].fragment as u32],
        );
        Self {
            knobs: vec![split, tile, reg, fragment],
            constraints: vec![
                // one thread mapping at a time
                Arc::new(|s: &Space, c: &Candidate| {
                    let split = c.get(s, "split").map_or(0, |v| v[0]);
                    let tile = c.get(s, "tile").map_or(0, |v| v[0]);
                    split == 0 || tile == 0
                }),
                // register tiles live inside a 2-D shared-memory tile
                Arc::new(|s: &Space, c: &Candidate| {
                    let reg = c.get(s, "reg").unwrap_or(&[1, 1]);
                    if reg == [1, 1] {
                        return true;
                    }
                    let t = c.get(s, "tile").unwrap_or(&[0, 0, 0]);
                    t[0] > 0 && t[1] > 1 && t[0] * t[1] <= 256
                }),
            ],
        }
    }

    /// Every candidate for `t`, the node's current schedule first, then
    /// the cartesian product of the knobs' values in a fixed order
    /// (constraints applied, duplicates dropped).
    pub fn candidates(&self, g: &TensorGraph, t: TensorId) -> Vec<Candidate> {
        let per: Vec<Vec<Value>> = self.knobs.iter().map(|k| (k.values)(g, t)).collect();
        let mut out = vec![self.current(g, t)];
        let mut idx = vec![0usize; per.len()];
        if per.iter().any(|v| v.is_empty()) {
            return out;
        }
        loop {
            let c = Candidate(idx.iter().zip(&per).map(|(&i, v)| v[i].clone()).collect());
            if self.constraints.iter().all(|f| f(self, &c)) && !out.contains(&c) {
                out.push(c);
            }
            // odometer, last knob fastest
            let mut k = per.len();
            loop {
                if k == 0 {
                    return out;
                }
                k -= 1;
                idx[k] += 1;
                if idx[k] < per[k].len() {
                    break;
                }
                idx[k] = 0;
            }
        }
    }

    /// `t`'s schedule as it stands.
    pub fn current(&self, g: &TensorGraph, t: TensorId) -> Candidate {
        Candidate(self.knobs.iter().map(|k| (k.read)(g, t)).collect())
    }

    /// Set every knob of `t` to `c`'s value.
    pub fn apply(&self, g: &mut TensorGraph, t: TensorId, c: &Candidate) {
        for (k, v) in self.knobs.iter().zip(&c.0) {
            (k.apply)(g, t, v);
        }
    }

    /// `t` with every knob at its neutral value: the schedule-independent
    /// form a cache key is computed from.
    pub fn neutral(&self, g: &TensorGraph, t: TensorId) -> Candidate {
        Candidate(
            self.knobs
                .iter()
                .map(|k| (k.values)(g, t).swap_remove(0))
                .collect(),
        )
    }

    /// Names of [`features`]' entries for this space.
    pub fn feature_names(&self) -> Vec<String> {
        let mut n: Vec<String> = BASE_FEATURES.iter().map(|s| s.to_string()).collect();
        for k in &self.knobs {
            for i in 0..KNOB_WIDTH {
                n.push(format!("{}{i}", k.name));
            }
        }
        n.extend(DERIVED_FEATURES.iter().map(|s| s.to_string()));
        n
    }
}

const BASE_FEATURES: [&str; 10] = [
    "log_points",
    "log_red",
    "width",
    "args",
    "reduce",
    "lanes",
    "rows",
    "log_lanes_n",
    "log_rows_n",
    "log_bytes_in",
];
const DERIVED_FEATURES: [&str; 4] = ["log_wg", "log_groups", "log_per_thread", "stage_bytes"];
/// Entries per knob in the feature vector (values past this are dropped,
/// short values padded with 0): `log2(1 + v)` of each.
const KNOB_WIDTH: usize = 3;

fn lg(x: f64) -> f32 {
    (1.0 + x).log2() as f32
}

/// Numbers describing kernel `t` under candidate `c`, for a cost model:
/// the reduction's shape, then every knob's values, then derived launch
/// geometry. Same length and order as [`Space::feature_names`].
pub fn features(space: &Space, g: &TensorGraph, t: TensorId, c: &Candidate) -> Vec<f32> {
    let comp = g
        .tiled_reduction(t.n())
        .and_then(|r| g.comp_at(r))
        .or_else(|| g.comp_at(t.n()));
    let mut f = Vec::new();
    let (pts, lanes_n, rows_n) = match comp {
        Some(c) => {
            let pts = fixed(c.points());
            let (ax, ay) = tile_axes(&pts);
            f.push(lg(pts.iter().product::<usize>() as f64));
            f.push(lg(numel(&c.red) as f64));
            f.push(c.width() as f32);
            f.push(c.args.len() as f32);
            f.push(c.monoid.is_some() as u8 as f32);
            f.push(ax.is_some() as u8 as f32);
            f.push(ay.is_some() as u8 as f32);
            let l = ax.map_or(1, |d| pts[d]);
            let r = ay.map_or(1, |d| pts[d]);
            f.push(lg(l as f64));
            f.push(lg(r as f64));
            let bytes: usize = c
                .args
                .iter()
                .map(|a| numel(g.shape(a.src)) * g.precision(a.src).storage.bytes())
                .sum();
            f.push(lg(bytes as f64));
            (pts.iter().product::<usize>(), l, r)
        }
        None => {
            f.extend([0.0; BASE_FEATURES.len()]);
            (numel(g.shape(t)), 1, 1)
        }
    };
    for v in &c.0 {
        for i in 0..KNOB_WIDTH {
            f.push(v.get(i).map_or(0.0, |&x| lg(x as f64)));
        }
    }
    let split = c.get(space, "split").map_or(0, |v| v[0]) as usize;
    let tile = c
        .get(space, "tile")
        .map_or([0, 0, 0], |v| [v[0], v[1], v[2]]);
    let reg = c.get(space, "reg").map_or([1, 1], |v| [v[0], v[1]]);
    let (wg, per_thread) = if tile[0] > 0 {
        let wg = (tile[0] * tile[1]) as usize;
        (wg, (reg[0] * reg[1]) as usize)
    } else if split > 0 {
        (split, 1)
    } else {
        (64, 1)
    };
    let threads = if split > 0 {
        pts * split
    } else {
        pts.div_ceil(per_thread)
    };
    f.push(lg(wg as f64));
    f.push(lg(threads.div_ceil(wg) as f64));
    f.push(lg(per_thread as f64));
    let stage = if tile[0] > 0 {
        let _ = (lanes_n, rows_n);
        (tile[0] * reg[0] + tile[1] * reg[1]) as usize * tile[2] as usize * 4
    } else {
        0
    };
    f.push(lg(stage as f64));
    f
}

/// One logged measurement: features of a candidate and its time relative
/// to the same kernel's baseline schedule (`ln(t / t_base)`).
#[derive(Clone, Debug)]
pub struct Sample {
    pub features: Vec<f32>,
    pub target: f32,
    /// Which problem it came from (for leave-one-out evaluation).
    pub group: u64,
}

#[derive(Clone, Debug)]
enum Tree {
    Leaf(f32),
    Split {
        feature: usize,
        threshold: f32,
        left: Box<Tree>,
        right: Box<Tree>,
    },
}

impl Tree {
    fn eval(&self, x: &[f32]) -> f32 {
        match self {
            Tree::Leaf(v) => *v,
            Tree::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if x.get(*feature).copied().unwrap_or(0.0) <= *threshold {
                    left.eval(x)
                } else {
                    right.eval(x)
                }
            }
        }
    }
}

/// Gradient-boosted regression trees (squared loss) predicting
/// `ln(t / t_base)` from [`features`]. Small on purpose: hundreds of
/// samples, depth-4 trees, no dependencies. Only the ranking it induces
/// among one kernel's candidates is used.
#[derive(Clone, Debug)]
pub struct CostModel {
    trees: Vec<Tree>,
    base: f32,
    rate: f32,
    pub n_features: usize,
    pub n_samples: usize,
}

impl CostModel {
    pub fn train(samples: &[Sample]) -> Option<Self> {
        Self::train_with(samples, 120, 4, 0.15)
    }

    pub fn train_with(samples: &[Sample], rounds: usize, depth: usize, rate: f32) -> Option<Self> {
        let n_features = samples.first()?.features.len();
        let samples: Vec<&Sample> = samples
            .iter()
            .filter(|s| s.features.len() == n_features)
            .collect();
        if samples.len() < 8 {
            return None;
        }
        let base = samples.iter().map(|s| s.target).sum::<f32>() / samples.len() as f32;
        let mut pred = vec![base; samples.len()];
        let mut trees = Vec::with_capacity(rounds);
        let all: Vec<usize> = (0..samples.len()).collect();
        for _ in 0..rounds {
            let resid: Vec<f32> = samples
                .iter()
                .zip(&pred)
                .map(|(s, p)| s.target - p)
                .collect();
            let tree = grow(&samples, &resid, &all, depth, n_features);
            for (i, s) in samples.iter().enumerate() {
                pred[i] += rate * tree.eval(&s.features);
            }
            trees.push(tree);
        }
        Some(Self {
            trees,
            base,
            rate,
            n_features,
            n_samples: samples.len(),
        })
    }

    pub fn predict(&self, x: &[f32]) -> f32 {
        self.base + self.rate * self.trees.iter().map(|t| t.eval(x)).sum::<f32>()
    }
}

fn grow(samples: &[&Sample], resid: &[f32], idx: &[usize], depth: usize, nf: usize) -> Tree {
    const MIN_LEAF: usize = 3;
    let mean = idx.iter().map(|&i| resid[i]).sum::<f32>() / idx.len().max(1) as f32;
    if depth == 0 || idx.len() < 2 * MIN_LEAF {
        return Tree::Leaf(mean);
    }
    let total: f32 = idx.iter().map(|&i| resid[i]).sum();
    let n = idx.len() as f32;
    let mut best: Option<(f32, usize, f32)> = None;
    let mut order = idx.to_vec();
    for f in 0..nf {
        order.sort_by(|&a, &b| samples[a].features[f].total_cmp(&samples[b].features[f]));
        let mut left = 0.0f32;
        for k in 0..order.len() - 1 {
            left += resid[order[k]];
            let (xa, xb) = (
                samples[order[k]].features[f],
                samples[order[k + 1]].features[f],
            );
            let nl = (k + 1) as f32;
            if xa == xb || k + 1 < MIN_LEAF || order.len() - k - 1 < MIN_LEAF {
                continue;
            }
            let right = total - left;
            // variance reduction ∝ left²/nl + right²/nr
            let gain = left * left / nl + right * right / (n - nl);
            if best.is_none_or(|b| gain > b.0) {
                best = Some((gain, f, 0.5 * (xa + xb)));
            }
        }
    }
    let Some((gain, feature, threshold)) = best else {
        return Tree::Leaf(mean);
    };
    if gain <= total * total / n + 1e-9 {
        return Tree::Leaf(mean);
    }
    let (l, r): (Vec<usize>, Vec<usize>) = idx
        .iter()
        .partition(|&&i| samples[i].features[feature] <= threshold);
    Tree::Split {
        feature,
        threshold,
        left: Box::new(grow(samples, resid, &l, depth - 1, nf)),
        right: Box::new(grow(samples, resid, &r, depth - 1, nf)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Monoid;

    fn matmul(n: usize) -> (TensorGraph, TensorId) {
        let mut g = TensorGraph::new();
        let a = g.input(&[n, n]);
        let b = g.input(&[n, n]);
        let c = g.matmul(a, b).unwrap();
        (g, c)
    }

    #[test]
    fn space_covers_the_fixed_candidates() {
        let (g, c) = matmul(256);
        let s = Space::builtin();
        let cands = s.candidates(&g, c);
        let old = g.schedule_candidates(c);
        for (split, tile, reg) in old {
            let want = Candidate(vec![
                vec![split],
                vec![tile.0, tile.1, tile.2],
                vec![reg.0, reg.1],
                vec![0],
            ]);
            assert!(cands.contains(&want), "missing {want:?}");
        }
        assert!(cands.len() > 20, "{}", cands.len());
        // constraints hold
        for c in &cands {
            assert!(c.0[0][0] == 0 || c.0[1][0] == 0);
        }
    }

    #[test]
    fn registered_knob_joins_the_space() {
        let (g, c) = matmul(64);
        let n0 = Space::standard().candidates(&g, c).len();
        register_knob(
            Knob::new(
                "test_stages",
                |_, _| vec![vec![1], vec![2]],
                |_, _, _| {},
                |_, _| vec![1],
            ),
            vec![],
        );
        let s = Space::standard();
        assert_eq!(s.candidates(&g, c).len(), 2 * n0);
        let enc = s.candidates(&g, c)[3].encode(&s);
        assert!(enc.contains("test_stages="));
        let back = Candidate::decode(&s, &g, c, &enc).unwrap();
        assert_eq!(back, s.candidates(&g, c)[3]);
        assert_eq!(features(&s, &g, c, &back).len(), s.feature_names().len());
        REGISTRY
            .lock()
            .unwrap()
            .retain(|(k, _)| k.name != "test_stages");
    }

    #[test]
    fn cost_model_learns_a_ranking() {
        // target = -x0 + 0.5 x1² on a grid: the trees must rank it
        let mut s = Vec::new();
        for i in 0..20 {
            for j in 0..10 {
                let (x0, x1) = (i as f32 / 4.0, j as f32 / 3.0);
                s.push(Sample {
                    features: vec![x0, x1, 0.0],
                    target: -x0 + 0.5 * x1 * x1,
                    group: 0,
                });
            }
        }
        let m = CostModel::train(&s).unwrap();
        assert!(m.predict(&[4.0, 0.0, 0.0]) < m.predict(&[1.0, 0.0, 0.0]));
        assert!(m.predict(&[2.0, 3.0, 0.0]) > m.predict(&[2.0, 0.5, 0.0]));
        let _ = Monoid::sum();
    }
}
