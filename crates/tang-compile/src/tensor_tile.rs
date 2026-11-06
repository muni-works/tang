//! Tiled WGSL schedule: a `bx × by` block of points per workgroup, one
//! point per thread, with shared reads staged chunk by chunk in workgroup
//! memory.
//!
//! Unlike the function-per-comp emitter in `tensor_wgsl`, this one fully
//! inlines the closure into a single loop nest and tracks every buffer
//! address as an affine form over the loop variables. A read that depends
//! on the reduce variable but on at most one of the tiled point
//! coordinates is shared along the other tile axis, so it is loaded
//! cooperatively once per chunk. Attention stages its k/v tiles, matmul
//! stages both its `a` and `b` tiles. Nothing here knows about either.
//!
//! The tile's x / y axes are the last two point dims of at least 16
//! (`tile_axes`); every other point dim is uniform across the workgroup.
//!
//! Multi-dim reductions (a conv's `(c, k)`) chunk their largest reduce dim
//! as `r0` and unroll the small ones (`u{i}`) inside each chunk step;
//! guarded reads are zero-padded when staged. A map root is an epilogue:
//! the reduction it applies to is tiled and the map runs at write-out.

use std::collections::BTreeMap;
use std::fmt::Write;

use tang_expr::{ExprGraph, ExprId, Node};

use super::wgsl::{identity, lit, load, quant, store, WgslKernel};
use super::Format;
use super::{fixed, Comp, Form, Op, TensorGraph, TensorId};

/// Affine expression over named loop variables.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct Sym {
    terms: BTreeMap<String, i64>,
    offset: i64,
}

impl Sym {
    fn var(name: &str) -> Self {
        let mut s = Sym::default();
        s.terms.insert(name.to_string(), 1);
        s
    }

    fn scaled(&self, c: i64) -> Self {
        Sym {
            terms: self.terms.iter().map(|(k, v)| (k.clone(), v * c)).collect(),
            offset: self.offset * c,
        }
    }

    fn add(&mut self, other: &Sym) {
        for (k, v) in &other.terms {
            *self.terms.entry(k.clone()).or_insert(0) += v;
        }
        self.terms.retain(|_, v| *v != 0);
        self.offset += other.offset;
    }

    fn wgsl(&self) -> String {
        let mut parts = vec![self.offset.to_string()];
        for (k, v) in &self.terms {
            parts.push(format!("{v} * i32({k})"));
        }
        format!("u32({})", parts.join(" + "))
    }
}

/// Reads of one buffer sharing every coefficient and differing only in
/// offset are one staged region.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RegionKey {
    buffer: usize,
    /// coefficient on the x tile coord (`p{k-1}`), if any
    px: i64,
    /// coefficient on the y tile coord (`p{k-2}`), if any
    py: i64,
    /// coefficient on the reduce var `r0`
    red: i64,
    /// (inner loop var, coefficient)
    inner: Vec<(String, i64)>,
    /// (uniform var, coefficient): leading point dims
    uniform: Vec<(String, i64)>,
    /// zero-padding guard applied at load time: the source coords and
    /// shape of a guarded view (exact offsets, so no span merging)
    guard: Option<(Vec<Sym>, Vec<usize>)>,
}

struct Region {
    key: RegionKey,
    lo: i64,
    hi: i64,
    /// extents of the inner vars, in `key.inner` order
    extents: Vec<usize>,
}

impl Region {
    fn span(&self) -> usize {
        (self.hi - self.lo + 1) as usize
    }

    /// Staging layout for a `px_n × py_n` tile and `chunk` reduce steps.
    ///
    /// Every var of the region (lanes, `t = r0 − chunk`, inner vars) is
    /// either a grid axis — one staged copy per value — or folded into a
    /// window over the address line. A conv's `x[c, t + d·k − p]` folds
    /// the lane `t` and tap `k` into one `tile_t + d·(K − 1)` window
    /// instead of `K` overlapping copies. A var is folded when that
    /// shrinks the region: a window grown by `|coef|·(ext − 1)` beats
    /// `ext` copies of it.
    fn layout(
        &self,
        names: (&str, Option<&str>),
        px_n: usize,
        py_n: usize,
        chunk: usize,
    ) -> Layout {
        let mut vars: Vec<LVar> = vec![LVar {
            kind: LKind::T,
            name: "r0".into(),
            coef: self.key.red,
            ext: chunk,
        }];
        for ((name, coef), &ext) in self.key.inner.iter().zip(&self.extents) {
            vars.push(LVar {
                kind: LKind::Inner,
                name: name.clone(),
                coef: *coef,
                ext,
            });
        }
        if self.key.py != 0 {
            vars.push(LVar {
                kind: LKind::Y,
                name: names.1.unwrap().into(),
                coef: self.key.py,
                ext: py_n,
            });
        }
        if self.key.px != 0 {
            vars.push(LVar {
                kind: LKind::X,
                name: names.0.into(),
                coef: self.key.px,
                ext: px_n,
            });
        }
        // Greedy folds. A neutral fold (the lane axis alone: a 64-wide
        // window is 64 copies of one) can enable the next one (the taps),
        // so try with neutral folds allowed and keep that only if it ends
        // smaller.
        let fold = |neutral: bool| {
            let mut size = self.span() as i64;
            let mut win = vec![false; vars.len()];
            loop {
                let best = (0..vars.len())
                    .filter(|&i| !win[i] && vars[i].ext > 1)
                    .map(|i| {
                        let grown = size + vars[i].coef.abs() * (vars[i].ext as i64 - 1);
                        (i, grown as f64 / (size as f64 * vars[i].ext as f64))
                    })
                    .filter(|&(_, ratio)| ratio < 1.0 || (neutral && ratio <= 1.0))
                    .min_by(|a, b| a.1.total_cmp(&b.1));
                let Some((i, _)) = best else { break };
                win[i] = true;
                size += vars[i].coef.abs() * (vars[i].ext as i64 - 1);
            }
            // A guarded window must recover its guarded coordinate from
            // the window position: one source coord may depend on it.
            let mut guard_coord = None;
            if let Some((coords, shape)) = &self.key.guard {
                let on_win: Vec<usize> = (0..coords.len())
                    .filter(|&c| {
                        vars.iter()
                            .zip(&win)
                            .any(|(v, &w)| w && coords[c].terms.contains_key(&v.name))
                    })
                    .collect();
                match on_win.as_slice() {
                    [] => {}
                    [c] => {
                        guard_coord = Some((*c, shape[c + 1..].iter().product::<usize>() as i64))
                    }
                    _ => {
                        win.iter_mut().for_each(|w| *w = false);
                        size = self.span() as i64;
                    }
                }
            }
            let grid: usize = vars
                .iter()
                .zip(&win)
                .filter(|(_, &w)| !w)
                .map(|(v, _)| v.ext)
                .product();
            (grid * size as usize, size, win, guard_coord)
        };
        let strict = fold(false);
        let loose = fold(true);
        let (_, size, win, guard_coord) = if loose.0 < strict.0 { loose } else { strict };
        let wmin: i64 = vars
            .iter()
            .zip(&win)
            .filter(|(_, &w)| w)
            .map(|(v, _)| (v.coef * (v.ext as i64 - 1)).min(0))
            .sum();
        let (mut grid, mut window) = (Vec::new(), Vec::new());
        for (v, w) in vars.into_iter().zip(win) {
            if w {
                window.push(v)
            } else {
                grid.push(v)
            }
        }
        Layout {
            grid,
            window,
            wsize: size as usize,
            wmin,
            guard_coord,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LKind {
    T,
    Inner,
    Y,
    X,
}

#[derive(Clone, Debug)]
struct LVar {
    kind: LKind,
    /// the var's name in address forms (`r0`, `u1`, `p1`, …)
    name: String,
    coef: i64,
    ext: usize,
}

/// How a region sits in workgroup memory: grid axes in order (`t`, inner
/// vars, then lanes innermost so consecutive lanes hit consecutive words),
/// with the window between the inner vars and the lanes.
struct Layout {
    grid: Vec<LVar>,
    window: Vec<LVar>,
    /// window length in elements (the offset span when nothing is folded)
    wsize: usize,
    /// smallest `Σ coef·var` over the window vars (≤ 0)
    wmin: i64,
    /// `(coord, row-major stride)` of the guarded source coord that
    /// depends on window vars
    guard_coord: Option<(usize, i64)>,
}

impl Layout {
    fn total(&self) -> usize {
        self.grid.iter().map(|v| v.ext).product::<usize>() * self.wsize
    }
    /// grid axes outside the window, then inside (lanes)
    fn split(&self) -> (Vec<&LVar>, Vec<&LVar>) {
        let lane = |v: &&LVar| matches!(v.kind, LKind::X | LKind::Y);
        (
            self.grid.iter().filter(|v| !lane(v)).collect(),
            self.grid.iter().filter(lane).collect(),
        )
    }
}

/// A per-thread, reduce-invariant read: loaded once before the reduce
/// loop into a private array indexed by the inner loop vars.
struct Hoist {
    buffer: usize,
    /// (var, coefficient) for every var in the address (point coords and
    /// inner vars; no `r0`), sorted
    coefs: Vec<(String, i64)>,
    lo: i64,
    hi: i64,
    /// (inner var, extent) in `coefs` order, inner vars only
    inner: Vec<(String, usize)>,
}

impl Hoist {
    fn span(&self) -> usize {
        (self.hi - self.lo + 1) as usize
    }
    fn len(&self) -> usize {
        self.inner.iter().map(|(_, e)| *e).product::<usize>() * self.span()
    }
}

/// An inlined comp whose point coordinates depend on the reduce var but
/// not on the lane (x) axis: computed cooperatively once per chunk into
/// workgroup memory, indexed by `(ly, t)`.
struct CompRegion {
    node: usize,
    /// point coords with `r0` and `py` as symbols
    coords: Vec<Sym>,
    width: usize,
    uses_py: bool,
}

enum Val {
    Expr(String),
    Arr(String),
}

impl Val {
    fn slot(&self, slot: &Sym) -> String {
        match self {
            Val::Expr(e) => e.clone(),
            Val::Arr(a) => format!("{a}[{}]", slot.wgsl()),
        }
    }
}

/// A read of a staged region inside the reduce loop body: `local` and
/// `off` fix the element along the reduce/inner/offset axes, the point
/// axis is the thread's register tile. The body refers to it by a token
/// (`svec{n}_`); the r0 loop loads it once per thread, as a `vec4` when
/// the register tile makes the thread's points contiguous.
struct Site {
    region: usize,
    local: String,
    off: usize,
}

struct Ctx<'a> {
    g: &'a TensorGraph,
    reads: &'a [usize],
    px: String,
    py: Option<String>,
    bx: usize,
    by: usize,
    chunk: usize,
    sites: Vec<Site>,
    /// extents of inner loop vars
    extents: BTreeMap<String, usize>,
    regions: Vec<Region>,
    hoists: Vec<Hoist>,
    comp_regions: Vec<CompRegion>,
    /// inside a cooperative load loop: every read goes straight to its buffer
    in_load: bool,
    /// first pass collects regions, second emits against them
    collect: bool,
    /// node whose value is already in a named array (an epilogue's
    /// reduction, read back from its accumulator)
    subst: Option<(usize, String)>,
    /// pending guard of the arg being read (see `RegionKey::guard`)
    guard: Option<(Vec<Sym>, Vec<usize>)>,
    /// the last read consumed `guard`: its value is already zero-padded
    guard_done: bool,
    counter: usize,
    /// cooperative load of a staged reduction: `(node, lanes)` — its
    /// reduce loop starts at `lane_` and strides by `lanes`
    load_lanes: Option<(usize, usize)>,
}

impl Ctx<'_> {
    fn fresh(&mut self) -> usize {
        self.counter += 1;
        self.counter
    }

    /// Read of buffer-backed node `j` at an address.
    fn buffer_read(&mut self, j: usize, addr: Sym) -> String {
        if !addr.terms.contains_key("r0") || self.in_load {
            // only staged regions apply a guard at load time
            self.guard = None;
        }
        let fmt = self.g.nodes[j].precision.storage;
        if self.in_load {
            return load(fmt, &format!("b{j}"), &addr.wgsl());
        }
        let px = addr.terms.get(&self.px).copied().unwrap_or(0);
        let py = self
            .py
            .as_ref()
            .and_then(|n| addr.terms.get(n))
            .copied()
            .unwrap_or(0);
        let shared = addr.terms.contains_key("r0") && (px == 0 || py == 0);
        if !shared {
            self.guard = None;
            let has_inner = addr.terms.keys().any(|k| !k.starts_with('p') && k != "r0");
            if !addr.terms.contains_key("r0") && has_inner {
                return self.hoisted_read(j, addr);
            }
            return load(fmt, &format!("b{j}"), &addr.wgsl());
        }
        let is_tiled = |k: &str| k == self.px || self.py.as_deref() == Some(k);
        let inner: Vec<(String, i64)> = addr
            .terms
            .iter()
            .filter(|(k, _)| k.as_str() != "r0" && !is_tiled(k) && !k.starts_with('p'))
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        let uniform: Vec<(String, i64)> = addr
            .terms
            .iter()
            .filter(|(k, _)| k.starts_with('p') && !is_tiled(k))
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        let key = RegionKey {
            buffer: j,
            px,
            py,
            red: addr.terms["r0"],
            inner,
            uniform,
            guard: self.guard.take(),
        };
        if key.guard.is_some() {
            self.guard_done = true;
        }
        let idx = match self.regions.iter().position(|r| r.key == key) {
            Some(i) => {
                let r = &mut self.regions[i];
                r.lo = r.lo.min(addr.offset);
                r.hi = r.hi.max(addr.offset);
                i
            }
            None => {
                let extents = key.inner.iter().map(|(k, _)| self.extents[k]).collect();
                self.regions.push(Region {
                    key,
                    lo: addr.offset,
                    hi: addr.offset,
                    extents,
                });
                self.regions.len() - 1
            }
        };
        if self.collect {
            return String::new();
        }
        let r = &self.regions[idx];
        let lay = r.layout((&self.px, self.py.as_deref()), self.bx, self.by, self.chunk);
        if r.key.inner.is_empty() && lay.window.is_empty() {
            // a plain point-minor region: the r0 loop loads the thread's
            // points once (as vec4s when they are contiguous), see `Site`
            self.sites.push(Site {
                region: idx,
                local: "(r0 - chunk)".into(),
                off: (addr.offset - r.lo) as usize,
            });
            return format!("svec{}_", self.sites.len() - 1);
        }
        let local = |v: &LVar| match v.kind {
            LKind::T => "(r0 - chunk)".to_string(),
            LKind::Inner => v.name.clone(),
            LKind::Y => "ly_".to_string(),
            LKind::X => "lx_".to_string(),
        };
        let mut wpos = format!("{}", addr.offset - r.lo - lay.wmin);
        for v in &lay.window {
            wpos.push_str(&format!(" + {} * i32({})", v.coef, local(v)));
        }
        let (outer, lanes) = lay.split();
        let mut index = String::from("0u");
        for v in &outer {
            index = format!("({index}) * {}u + {}", v.ext, local(v));
        }
        index = format!("({index}) * {}u + u32({wpos})", lay.wsize);
        for v in &lanes {
            index = format!("({index}) * {}u + {}", v.ext, local(v));
        }
        format!("stage_{idx}[{index}]")
    }

    /// A read that does not depend on the reduce var: served from a
    /// private array filled once per thread.
    fn hoisted_read(&mut self, j: usize, addr: Sym) -> String {
        let coefs: Vec<(String, i64)> = addr.terms.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let idx = match self
            .hoists
            .iter()
            .position(|h| h.buffer == j && h.coefs == coefs)
        {
            Some(i) => {
                let h = &mut self.hoists[i];
                h.lo = h.lo.min(addr.offset);
                h.hi = h.hi.max(addr.offset);
                i
            }
            None => {
                let inner = coefs
                    .iter()
                    .filter(|(k, _)| !k.starts_with('p'))
                    .map(|(k, _)| (k.clone(), self.extents[k]))
                    .collect();
                self.hoists.push(Hoist {
                    buffer: j,
                    coefs,
                    lo: addr.offset,
                    hi: addr.offset,
                    inner,
                });
                self.hoists.len() - 1
            }
        };
        if self.collect {
            return String::new();
        }
        let h = &self.hoists[idx];
        let mut flat = String::from("0u");
        for (name, ext) in &h.inner {
            flat = format!("({flat}) * {ext}u + {name}");
        }
        format!(
            "hoist_{idx}[({flat}) * {}u + {}u]",
            h.span(),
            addr.offset - h.lo
        )
    }

    /// Value of node `j` at full-shape coordinates `coords`.
    fn expand(&mut self, out: &mut String, j: usize, coords: &[Sym]) -> Val {
        if let Some((n, name)) = &self.subst {
            if *n == j {
                return Val::Arr(name.clone());
            }
        }
        let node = &self.g.nodes[j];
        match &node.op {
            Op::Input(_) => {
                let addr = flat(coords, &fixed(&node.shape));
                Val::Expr(self.buffer_read(j, addr))
            }
            Op::Iota(d) => Val::Expr(format!("f32({})", coords[*d].wgsl())),
            Op::Reshape(_) => {
                // flat index into the storage node's buffer; affine
                let storage = self.g.storage(TensorId(j as u32)).n();
                let addr = flat(coords, &fixed(&node.shape));
                Val::Expr(self.buffer_read(storage, addr))
            }
            Op::Comp(c) if self.reads.contains(&j) => {
                let k = c.points().len();
                let mut addr = flat(&coords[..k], &fixed(c.points())).scaled(c.width() as i64);
                addr.add(&flat(&coords[k..], &fixed(&c.out[k..])));
                Val::Expr(self.buffer_read(j, addr))
            }
            Op::Comp(c)
                if !self.in_load && self.is_workgroup_shared(&coords[..c.points().len()]) =>
            {
                let k = c.points().len();
                let w = c.width();
                let pts: Vec<Sym> = coords[..k].to_vec();
                let uses_py = self
                    .py
                    .as_ref()
                    .is_some_and(|py| pts.iter().any(|c| c.terms.contains_key(py)));
                let idx = match self
                    .comp_regions
                    .iter()
                    .position(|r| r.node == j && r.coords == pts)
                {
                    Some(i) => i,
                    None => {
                        self.comp_regions.push(CompRegion {
                            node: j,
                            coords: pts,
                            width: w,
                            uses_py,
                        });
                        self.comp_regions.len() - 1
                    }
                };
                if self.collect {
                    return Val::Expr(String::new());
                }
                let slot = flat(&coords[k..], &fixed(&c.out[k..]));
                let row = if uses_py { "ly_" } else { "0u" };
                Val::Expr(format!(
                    "cstage_{idx}[(({row}) * {}u + (r0 - chunk)) * {w}u + {}]",
                    self.chunk,
                    slot.wgsl()
                ))
            }
            Op::Comp(c) => {
                let k = c.points().len();
                let n = self.fresh();
                let w = c.width();
                let red_vars: Vec<String> = (0..c.red.len()).map(|i| format!("q{n}_{i}")).collect();
                for (v, ext) in red_vars.iter().zip(fixed(&c.red)) {
                    self.extents.insert(v.clone(), ext);
                }
                let mut space: Vec<Sym> = coords[..k].to_vec();
                space.extend(red_vars.iter().map(|v| Sym::var(v)));

                let mut inner = String::new();
                let arg_vals = self.arg_values(&mut inner, c, &space);
                let body =
                    emit_scalar(&mut inner, &c.body, &c.body_outs, &format!("t{n}_"), &|v| {
                        arg_vals[v as usize].clone()
                    });
                let Some(m) = &c.monoid else {
                    out.push_str(&inner);
                    if w == 1 {
                        return Val::Expr(body[0].clone());
                    }
                    writeln!(out, "var v{n} = array<f32, {w}>({});", body.join(", ")).unwrap();
                    return Val::Arr(format!("v{n}"));
                };
                writeln!(out, "var acc{n} = array<f32, {w}>({});", identity(m)).unwrap();
                let lanes = match self.load_lanes {
                    Some((node, l)) if node == j && c.red.len() == 1 => l,
                    _ => 1,
                };
                self.load_lanes = None;
                for (v, ext) in red_vars.iter().zip(fixed(&c.red)) {
                    if lanes > 1 {
                        writeln!(
                            out,
                            "for (var {v}: u32 = lane_; {v} < {ext}u; {v} = {v} + {lanes}u) {{"
                        )
                        .unwrap();
                        continue;
                    }
                    writeln!(
                        out,
                        "for (var {v}: u32 = 0u; {v} < {ext}u; {v} = {v} + 1u) {{"
                    )
                    .unwrap();
                }
                out.push_str(&inner);
                let combined = emit_scalar(out, &m.combine, &m.outs, &format!("c{n}_"), &|v| {
                    if (v as usize) < w {
                        format!("acc{n}[{v}]")
                    } else {
                        body[v as usize - w].clone()
                    }
                });
                let accum = self.g.nodes[j].precision.accum;
                let combined: Vec<String> = combined.iter().map(|e| quant(accum, e)).collect();
                writeln!(out, "acc{n} = array<f32, {w}>({});", combined.join(", ")).unwrap();
                for _ in &c.red {
                    writeln!(out, "}}").unwrap();
                }
                Val::Arr(format!("acc{n}"))
            }
        }
    }

    /// Expressions for each arg of `c` evaluated at iteration coords `space`.
    fn arg_values(&mut self, out: &mut String, c: &Comp, space: &[Sym]) -> Vec<String> {
        c.args
            .iter()
            .map(|a| {
                let src_shape = fixed(&self.g.nodes[a.src.n()].shape);
                let cs: Vec<Sym> = a
                    .view
                    .forms
                    .iter()
                    .map(|f| {
                        let mut s = Sym {
                            offset: f.offset,
                            ..Default::default()
                        };
                        for (d, &coef) in f.coefs.iter().enumerate() {
                            if coef != 0 {
                                s.add(&space[d].scaled(coef));
                            }
                        }
                        s
                    })
                    .collect();
                let pk = src_shape.len() - self.tuple_rank(a.src.n());
                let slot = flat(&cs[pk..], &src_shape[pk..]);
                let direct = match &self.g.nodes[a.src.n()].op {
                    Op::Input(_) | Op::Reshape(_) => true,
                    Op::Comp(_) => self.reads.contains(&a.src.n()),
                    Op::Iota(_) => false,
                };
                self.guard = (a.view.guard && direct && pk == src_shape.len())
                    .then(|| (cs.clone(), src_shape.clone()));
                self.guard_done = false;
                let v = self.expand(out, a.src.n(), &cs);
                self.guard = None;
                let v = v.slot(&slot);
                if !a.view.guard || self.collect || std::mem::take(&mut self.guard_done) {
                    return v;
                }
                // zero padding: the (possibly wrapped) read is discarded
                let cond = cs
                    .iter()
                    .zip(&src_shape)
                    .map(|(c, n)| format!("({} < {n}u)", c.wgsl()))
                    .collect::<Vec<_>>()
                    .join(" && ");
                format!("select(0.0, {v}, {cond})")
            })
            .collect()
    }

    /// Point coords that vary with the reduce var but not the lane axis.
    fn is_workgroup_shared(&self, pts: &[Sym]) -> bool {
        let on_r0 = pts.iter().any(|c| c.terms.contains_key("r0"));
        let on_px = pts.iter().any(|c| c.terms.contains_key(&self.px));
        let inner = pts
            .iter()
            .any(|c| c.terms.keys().any(|k| k != "r0" && !k.starts_with('p')));
        on_r0 && !on_px && !inner && self.bx > 1
    }

    fn tuple_rank(&self, j: usize) -> usize {
        match &self.g.nodes[j].op {
            Op::Comp(c) => c.tuple,
            Op::Input(_) | Op::Iota(_) | Op::Reshape(_) => 0,
        }
    }
}

/// The largest non-unit point dim other than `dx` (ties: the innermost).
pub(super) fn largest_other(pts: &[usize], dx: usize) -> Option<usize> {
    (0..pts.len())
        .filter(|&d| d != dx && pts[d] > 1)
        .max_by_key(|&d| pts[d])
}

/// An affine form over symbolic coords.
fn sym_of(f: &Form, coords: &[Sym]) -> Sym {
    let mut s = Sym {
        offset: f.offset,
        ..Default::default()
    };
    for (d, &coef) in f.coefs.iter().enumerate() {
        if coef != 0 {
            s.add(&coords[d].scaled(coef));
        }
    }
    s
}

/// Staged f32 words per workgroup: 16 KiB, WebGPU's guaranteed
/// `maxComputeWorkgroupStorageSize` and within Metal's 32 KiB.
pub(crate) const MAX_STAGE_WORDS: usize = 4096;

/// Row-major flat index of symbolic coords over `shape`.
fn flat(coords: &[Sym], shape: &[usize]) -> Sym {
    let mut acc = Sym::default();
    for (c, &n) in coords.iter().zip(shape) {
        acc = acc.scaled(n as i64);
        acc.add(c);
    }
    acc
}

/// Emit `let {prefix}{i} = …;` for the live nodes of `g` and return the
/// expressions for `outs`. `var_expr` names input var `i`.
pub(super) fn emit_scalar(
    out: &mut String,
    g: &ExprGraph,
    outs: &[ExprId],
    prefix: &str,
    var_expr: &dyn Fn(u16) -> String,
) -> Vec<String> {
    let mut live: Vec<usize> = g.live_set(outs).into_iter().collect();
    live.sort_unstable();
    let nodes = g.nodes_slice();
    let e = |id: &ExprId| match nodes[id.index() as usize] {
        Node::Var(v) => var_expr(v),
        _ => format!("{prefix}{}", id.index()),
    };
    for i in live {
        let rhs = match &nodes[i] {
            Node::Var(_) => continue,
            Node::Lit(b) => lit(f64::from_bits(*b)),
            Node::Add(a, b) => format!("{} + {}", e(a), e(b)),
            Node::Mul(a, b) => format!("{} * {}", e(a), e(b)),
            Node::Neg(a) => format!("-{}", e(a)),
            Node::Recip(a) => format!("1.0 / {}", e(a)),
            Node::Sqrt(a) => format!("sqrt({})", e(a)),
            Node::Sin(a) => format!("sin({})", e(a)),
            Node::Atan2(a, b) => format!("atan2({}, {})", e(a), e(b)),
            Node::Exp2(a) => format!("exp2({})", e(a)),
            Node::Log2(a) => format!("log2({})", e(a)),
            Node::Select(c, a, b) => format!("select({}, {}, {} > 0.0)", e(b), e(a), e(c)),
        };
        writeln!(out, "let {prefix}{i} = {rhs};").unwrap();
    }
    outs.iter().map(e).collect()
}

impl TensorGraph {
    /// Element count of buffer `j` as a kernel expression.
    fn buffer_len_expr(&self, j: usize) -> String {
        if super::wgsl::dialect() != super::wgsl::Dialect::Wgsl {
            // MSL and CUDA have no runtime array length; the size is static
            return format!("{}u", super::numel(&self.nodes[j].shape));
        }
        let per_word = match self.nodes[j].precision.storage {
            Format::F16 => " * 2u",
            _ => "",
        };
        format!("arrayLength(&b{j}){per_word}")
    }

    /// Emit the tiled kernel for `root`, continuing `s` (which already
    /// holds the bindings and per-node functions).
    /// `None` when the staging cannot fit [`MAX_STAGE_WORDS`] even at a
    /// one-step chunk; the caller then emits the untiled kernel.
    pub(super) fn tile_kernel(
        &self,
        root: usize,
        mut s: String,
        reads: Vec<usize>,
    ) -> Option<WgslKernel> {
        let c = self.comp_at(root).unwrap();
        // the reduction this kernel chunks: root, or the one inlined into
        // root when root is its epilogue map
        let red_node = self.tiled_reduction(root).expect("tile needs a reduction");
        let rc = self.comp_at(red_node).unwrap();
        let epilogue = red_node != root;
        let (bx, by, chunk) = self.nodes[root].tile;
        let (bx, by, chunk) = (bx as usize, by as usize, chunk as usize);
        let (rx, ry) = self.nodes[root].reg;
        let (rx, ry) = (rx.max(1) as usize, ry.max(1) as usize);
        // points per workgroup along x / y; threads own rx × ry of them
        let (px_n, py_n) = (bx * rx, by * ry);
        let w = rc.width();
        let w_out = c.width();
        let pts_v = fixed(c.points());
        let pts = pts_v.as_slice();
        let k = pts.len();
        // Multi-dim reductions: the largest reduce dim is the chunked
        // `r0`; the others (`u{i}`, e.g. a conv's taps) are inner loops
        // inside each chunk step, staged alongside it like any inner var.
        let red = fixed(&rc.red);
        let rsel = (0..red.len())
            .rev()
            .max_by_key(|&i| red[i])
            .expect("tile needs a reduce dim");
        let red_name = |i: usize| {
            if i == rsel {
                "r0".to_string()
            } else {
                format!("u{i}")
            }
        };
        let inner_red: Vec<(String, usize)> = (0..red.len())
            .filter(|&i| i != rsel)
            .map(|i| (red_name(i), red[i]))
            .collect();
        let r_total = red[rsel];
        let m = rc.monoid.as_ref().expect("tile needs a reduction");
        let threads = bx * by;

        // Tile the last two non-unit point dims; everything else is
        // uniform across the workgroup.
        let (ax, ay) = super::tile_axes(pts);
        let dx = ax.unwrap_or(k - 1);
        let dy = if by > 1 { ay } else { None };
        let others: Vec<usize> = (0..k).filter(|&d| d != dx && Some(d) != dy).collect();
        let nx = pts[dx];
        let ny = dy.map_or(1, |d| pts[d]);
        let tiles_x = nx.div_ceil(px_n);
        let tiles_y = ny.div_ceil(py_n);
        let n_tiles = tiles_x * tiles_y * others.iter().map(|&d| pts[d]).product::<usize>();

        let mut ctx = Ctx {
            g: self,
            reads: &reads,
            px: format!("p{dx}"),
            py: dy.map(|d| format!("p{d}")),
            bx: px_n,
            by: py_n,
            chunk,
            sites: Vec::new(),
            extents: BTreeMap::new(),
            regions: Vec::new(),
            hoists: Vec::new(),
            comp_regions: Vec::new(),
            in_load: false,
            collect: true,
            subst: None,
            guard: None,
            guard_done: false,
            counter: 0,
            load_lanes: None,
        };
        ctx.extents.insert("r0".into(), r_total);
        for (name, n) in &inner_red {
            ctx.extents.insert(name.clone(), *n);
        }
        let pts_syms: Vec<Sym> = (0..k).map(|d| Sym::var(&format!("p{d}"))).collect();
        // the reduction's point coords: root's, or the epilogue's view of
        // them; then its reduce coords (`r0` and the unrolled `u{i}`)
        let space: Vec<Sym> = if epilogue {
            self.epilogue_read(c, red_node)
                .unwrap()
                .iter()
                .map(|f| sym_of(f, &pts_syms))
                .collect()
        } else {
            pts_syms.clone()
        }
        .into_iter()
        .chain((0..red.len()).map(|i| Sym::var(&red_name(i))))
        .collect();
        let run = |ctx: &mut Ctx, body: &mut String| {
            let arg_vals = ctx.arg_values(body, rc, &space);
            let vals = emit_scalar(body, &rc.body, &rc.body_outs, "t0_", &|v| {
                arg_vals[v as usize].clone()
            });
            emit_scalar(body, &m.combine, &m.outs, "c0_", &|v| {
                if (v as usize) < w {
                    format!("acc[{v}]")
                } else {
                    vals[v as usize - w].clone()
                }
            })
        };
        // the epilogue: root's body with the reduction's value bound to
        // the accumulator
        let run_epi = |ctx: &mut Ctx, body: &mut String| -> Vec<String> {
            if !epilogue {
                return (0..w).map(|s| format!("acc[{s}]")).collect();
            }
            ctx.subst = Some((red_node, "acc".into()));
            let vals = ctx.arg_values(body, c, &pts_syms);
            ctx.subst = None;
            emit_scalar(body, &c.body, &c.body_outs, "e0_", &|v| {
                vals[v as usize].clone()
            })
        };
        // Expand twice: first to discover staged regions, then for real.
        let mut body = String::new();
        let mut epi = String::new();
        run(&mut ctx, &mut body);
        // Keep workgroup memory within 16 KiB (the WebGPU default): halve
        // the chunk until the staged regions fit. Inner reduce dims
        // multiply every region; at 28 KiB a direct conv ran one
        // workgroup per core on an M4 and took 5.0 ms, at 14 KiB 2.4 ms.
        let staged = |ctx: &Ctx, chunk: usize| {
            let regions: usize = ctx
                .regions
                .iter()
                .map(|r| {
                    r.layout((&ctx.px, ctx.py.as_deref()), px_n, py_n, chunk)
                        .total()
                })
                .sum();
            let comps: usize = ctx
                .comp_regions
                .iter()
                .map(|r| (if r.uses_py { py_n } else { 1 }) * chunk * r.width)
                .sum();
            regions + comps
        };
        let mut chunk = chunk;
        while chunk > 1 && staged(&ctx, chunk) > MAX_STAGE_WORDS {
            chunk -= 1;
        }
        // Still over at chunk 1 (wide inner reduce dims, e.g. Kokoro's
        // iSTFT wanted 328 KiB): no valid tiled schedule on any target.
        if staged(&ctx, chunk) > MAX_STAGE_WORDS {
            return None;
        }
        ctx.chunk = chunk;
        run_epi(&mut ctx, &mut epi);
        ctx.collect = false;
        ctx.counter = 0;
        body.clear();
        epi.clear();
        let combined = run(&mut ctx, &mut body);
        let epi_vals = run_epi(&mut ctx, &mut epi);

        let layouts: Vec<Layout> = ctx
            .regions
            .iter()
            .map(|r| r.layout((&ctx.px, ctx.py.as_deref()), px_n, py_n, chunk))
            .collect();
        // (point axis: 'x' | 'y' | '-', vector width) of a site region (no
        // inner vars, no window): a thread's rx (ry) consecutive points
        // along the tile axis are contiguous in the point-minor stage, so
        // a register tile that is a multiple of 4 wide reads them as vec4s
        let axis_of = |r: &Region| -> (char, usize) {
            let v4 = r.key.inner.is_empty() && r.key.guard.is_none();
            if r.key.px != 0 && r.key.py == 0 {
                (
                    'x',
                    if v4 && px_n % 4 == 0 && rx % 4 == 0 {
                        4
                    } else {
                        1
                    },
                )
            } else if r.key.py != 0 && r.key.px == 0 {
                (
                    'y',
                    if v4 && py_n % 4 == 0 && ry % 4 == 0 {
                        4
                    } else {
                        1
                    },
                )
            } else {
                ('-', 1)
            }
        };
        // vec4 stages only where a site reads them; windowed / inner-var
        // regions stay scalar
        let vec_w: Vec<usize> = ctx
            .regions
            .iter()
            .zip(&layouts)
            .map(|(r, lay)| {
                if r.key.inner.is_empty() && lay.window.is_empty() {
                    axis_of(r).1
                } else {
                    1
                }
            })
            .collect();
        for (i, lay) in layouts.iter().enumerate() {
            if vec_w[i] == 4 {
                writeln!(
                    s,
                    "var<workgroup> stage_{i}: array<vec4<f32>, {}>;",
                    lay.total() / 4
                )
                .unwrap();
            } else {
                writeln!(s, "var<workgroup> stage_{i}: array<f32, {}>;", lay.total()).unwrap();
            }
        }
        // Cooperative loads for shared comps are expanded in the load
        // phase with r0 := chunk + t and py := tile_y*by + ly.
        // (rows, load code, lanes per staged value, fold of the lanes)
        let mut comp_loads: Vec<(usize, String, usize, String)> = Vec::new();
        for i in 0..ctx.comp_regions.len() {
            let r = &ctx.comp_regions[i];
            let rows = if r.uses_py { py_n } else { 1 };
            writeln!(
                s,
                "var<workgroup> cstage_{i}: array<f32, {}>;",
                rows * chunk * r.width
            )
            .unwrap();
            let mut sub = |c: &Sym| {
                let mut out = Sym {
                    offset: c.offset,
                    ..Default::default()
                };
                for (k, v) in &c.terms {
                    if k == "r0" {
                        out.add(&Sym::var("chunk").scaled(*v));
                        out.add(&Sym::var("t").scaled(*v));
                    } else if Some(k) == ctx.py.as_ref() {
                        out.add(&Sym::var("gy").scaled(*v));
                    } else {
                        out.add(&Sym::var(k).scaled(*v));
                    }
                }
                out
            };
            let coords: Vec<Sym> = r.coords.iter().map(&mut sub).collect();
            let node = r.node;
            let width = r.width;
            // A staged value that is itself a reduction (q·k in attention)
            // (a serial loop per thread, few of them busy): split its reduce
            // loop over `lanes` threads each and fold their partials with
            // its own monoid, so the load phase uses the whole workgroup.
            let total = rows * chunk;
            let red_comp = ctx
                .g
                .comp_at(node)
                .filter(|c| c.monoid.is_some() && c.red.len() == 1);
            let lanes = red_comp.map_or(1, |c| {
                let red = c.red[0].n();
                let mut l = 1;
                // up to a warp per value (coalesced reads of its operand
                // rows), threads looping when values × lanes exceed them
                while l < load_lanes() && l * 4 <= red && total * l * 2 * width <= 4096 {
                    l *= 2;
                }
                l
            });
            let mut code = String::new();
            ctx.in_load = true;
            ctx.load_lanes = (lanes > 1).then_some((node, lanes));
            let saved = ctx.counter;
            ctx.counter = 1000 * (i + 1);
            let v = ctx.expand(&mut code, node, &coords);
            ctx.counter = saved;
            ctx.in_load = false;
            ctx.load_lanes = None;
            let mut fold = String::new();
            if lanes > 1 {
                writeln!(
                    s,
                    "var<workgroup> cscratch_{i}: array<f32, {}>;",
                    total * lanes * width
                )
                .unwrap();
                for slot in 0..width {
                    let sl = Sym {
                        offset: slot as i64,
                        ..Default::default()
                    };
                    writeln!(
                        code,
                        "cscratch_{i}[idx * {width}u + {slot}u] = {};",
                        v.slot(&sl)
                    )
                    .unwrap();
                }
                let m = red_comp.unwrap().monoid.as_ref().unwrap();
                let part = |l: &str, slot: usize| {
                    format!("cscratch_{i}[(idx * {lanes}u + {l}) * {width}u + {slot}u]")
                };
                let first: Vec<String> = (0..width).map(|sl| part("0u", sl)).collect();
                writeln!(fold, "var f = array<f32, {width}>({});", first.join(", ")).unwrap();
                writeln!(fold, "for (var l: u32 = 1u; l < {lanes}u; l = l + 1u) {{").unwrap();
                let combined = emit_scalar(&mut fold, &m.combine, &m.outs, "f_", &|v| {
                    if (v as usize) < width {
                        format!("f[{v}]")
                    } else {
                        part("l", v as usize - width)
                    }
                });
                writeln!(fold, "f = array<f32, {width}>({});", combined.join(", ")).unwrap();
                writeln!(fold, "}}").unwrap();
                for slot in 0..width {
                    writeln!(
                        fold,
                        "cstage_{i}[((ly) * {chunk}u + t) * {width}u + {slot}u] = f[{slot}];"
                    )
                    .unwrap();
                }
            } else {
                for slot in 0..width {
                    let sl = Sym {
                        offset: slot as i64,
                        ..Default::default()
                    };
                    writeln!(
                        code,
                        "cstage_{i}[((ly) * {chunk}u + t) * {width}u + {slot}u] = {};",
                        v.slot(&sl)
                    )
                    .unwrap();
                }
            }
            comp_loads.push((rows, code, lanes, fold));
        }

        writeln!(s, "@compute @workgroup_size({threads})").unwrap();
        writeln!(
            s,
            "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{"
        )
        .unwrap();
        writeln!(s, "    let tid = lid.x;").unwrap();
        writeln!(s, "    let tx = tid % {bx}u;").unwrap();
        writeln!(s, "    let ty = tid / {bx}u;").unwrap();
        writeln!(s, "    let tile = wid.x + wid.y * 65535u;").unwrap();
        writeln!(s, "    if (tile >= {n_tiles}u) {{ return; }}").unwrap();
        writeln!(s, "    let tile_x = tile % {tiles_x}u;").unwrap();
        writeln!(s, "    let tile_y = (tile / {tiles_x}u) % {tiles_y}u;").unwrap();
        writeln!(s, "    var lead = tile / {}u;", tiles_x * tiles_y).unwrap();
        for &d in others.iter().rev() {
            writeln!(s, "    let p{d} = lead % {}u;", pts[d]).unwrap();
            writeln!(s, "    lead = lead / {}u;", pts[d]).unwrap();
        }
        // one copy of the per-point state per register-tile element
        let copies: Vec<(usize, usize)> =
            (0..ry).flat_map(|j| (0..rx).map(move |i| (i, j))).collect();
        let tag = |i: usize, j: usize| format!("_{i}_{j}");
        for &(i, j) in &copies {
            let t = tag(i, j);
            writeln!(
                s,
                "    let x_raw{t} = tile_x * {px_n}u + tx * {rx}u + {i}u;"
            )
            .unwrap();
            writeln!(
                s,
                "    let y_raw{t} = tile_y * {py_n}u + ty * {ry}u + {j}u;"
            )
            .unwrap();
            writeln!(
                s,
                "    let is_live{t} = x_raw{t} < {nx}u && y_raw{t} < {ny}u;"
            )
            .unwrap();
        }
        // per-copy preamble: point coords, hoisted reads, accumulators
        let mut hoist_fill = String::new();
        for (i, h) in ctx.hoists.iter().enumerate() {
            let s = &mut hoist_fill;
            writeln!(s, "    {{").unwrap();
            let mut flat = String::from("0u");
            for (name, ext) in &h.inner {
                writeln!(
                    s,
                    "    for (var {name}: u32 = 0u; {name} < {ext}u; {name} = {name} + 1u) {{"
                )
                .unwrap();
                flat = format!("({flat}) * {ext}u + {name}");
            }
            let span = h.span();
            let addr = h
                .coefs
                .iter()
                .map(|(k, c)| format!(" + {c} * i32({k})"))
                .collect::<String>();
            writeln!(s, "    for (var o: u32 = 0u; o < {span}u; o = o + 1u) {{").unwrap();
            writeln!(
                s,
                "        hoist_{i}[({flat}) * {span}u + o] = {};",
                load(
                    self.nodes[h.buffer].precision.storage,
                    &format!("b{}", h.buffer),
                    &format!("u32({}{addr} + i32(o))", h.lo)
                )
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
            for _ in &h.inner {
                writeln!(s, "    }}").unwrap();
            }
            writeln!(s, "    }}").unwrap();
        }
        let hoist_lens: Vec<usize> = ctx.hoists.iter().map(|h| h.len()).collect();
        for &(i, j) in &copies {
            let t = tag(i, j);
            for (hi, len) in hoist_lens.iter().enumerate() {
                writeln!(s, "    var hoist{t}_{hi}: array<f32, {len}>;").unwrap();
            }
            writeln!(s, "    var acc{t} = array<f32, {w}>({});", identity(m)).unwrap();
            writeln!(s, "    {{").unwrap();
            writeln!(s, "    let p{dx} = min(x_raw{t}, {}u);", nx - 1).unwrap();
            if let Some(dy) = dy {
                writeln!(s, "    let p{dy} = min(y_raw{t}, {}u);", ny - 1).unwrap();
            }
            s.push_str(&hoist_fill.replace("hoist_", &format!("hoist{t}_")));
            writeln!(s, "    }}").unwrap();
        }
        writeln!(
            s,
            "    for (var chunk: u32 = 0u; chunk < {r_total}u; chunk = chunk + {chunk}u) {{"
        )
        .unwrap();
        for (i, (r, lay)) in ctx.regions.iter().zip(&layouts).enumerate() {
            let v = vec_w[i];
            let total = lay.total() / v;
            writeln!(
                s,
                "        for (var idx: u32 = tid; idx < {total}u; idx = idx + {threads}u) {{"
            )
            .unwrap();
            // The stage is point-minor (grid vars, window, lanes innermost)
            // so the reduce loop's reads are conflict-free. The load loop
            // walks the *buffer* in address order instead — the var with
            // the smallest address stride innermost — so a warp's loads
            // coalesce; each element is then written to its stage slot. A
            // vec4 stage is filled a whole vector at a time (`v`
            // consecutive points along the tile axis), since component
            // writes from different threads would race.
            let (outer, lanes) = lay.split();
            let lane_name = |v: &LVar| if v.kind == LKind::X { "lx" } else { "ly" };
            // (name, extent, address coefficient), in stage order
            let mut dims: Vec<(String, usize, i64)> = Vec::new();
            for o in &outer {
                let name = if o.kind == LKind::T {
                    "t".into()
                } else {
                    o.name.clone()
                };
                dims.push((name, o.ext, o.coef));
            }
            dims.push(("w".into(), lay.wsize, 1));
            for l in &lanes {
                dims.push((lane_name(l).into(), l.ext / v, l.coef));
            }
            // ties: the innermost stage dim first
            let mut load_order: Vec<usize> = (0..dims.len()).rev().collect();
            load_order.sort_by_key(|&d| dims[d].2.unsigned_abs());
            writeln!(s, "            var rem = idx;").unwrap();
            for &d in &load_order {
                let (name, ext, _) = &dims[d];
                writeln!(s, "            let {name} = rem % {ext}u;").unwrap();
                writeln!(s, "            rem = rem / {ext}u;").unwrap();
            }
            let mut sidx = String::from("0u");
            for (name, ext, _) in &dims {
                sidx = format!("({sidx}) * {ext}u + {name}");
            }
            // global value of a grid var (lanes of component `c`), and of a
            // window var's base
            let global = |v: &LVar, c: &str| match v.kind {
                LKind::T => "chunk + t".to_string(),
                LKind::Inner => v.name.clone(),
                LKind::Y => format!("gy{c}"),
                LKind::X => format!("gx{c}"),
            };
            let base = |v: &LVar| match v.kind {
                LKind::T => "chunk".to_string(),
                LKind::Inner => "0u".to_string(),
                LKind::Y => format!("tile_y * {py_n}u"),
                LKind::X => format!("tile_x * {px_n}u"),
            };
            for c in 0..v {
                for l in &lanes {
                    let (g, tile, ext, ln) = match l.kind {
                        LKind::X => ("gx", "tile_x", px_n, "lx"),
                        _ => ("gy", "tile_y", py_n, "ly"),
                    };
                    let sfx = if v == 1 { String::new() } else { c.to_string() };
                    if v == 1 {
                        writeln!(s, "            let {g} = {tile} * {ext}u + {ln};").unwrap();
                    } else {
                        writeln!(
                            s,
                            "            let {g}{sfx} = {tile} * {ext}u + {ln} * {v}u + {c}u;"
                        )
                        .unwrap();
                    }
                }
            }
            // address and liveness of component `c`
            let component = |c: &str| -> (String, String) {
                let mut live = String::from("true");
                for l in &lanes {
                    let n = if l.kind == LKind::X { nx } else { ny };
                    live.push_str(&format!(" && {} < {n}u", global(l, c)));
                }
                if outer.iter().any(|o| o.kind == LKind::T) {
                    live.push_str(&format!(" && chunk + t < {r_total}u"));
                }
                let mut addr = format!("{} + i32(w)", r.lo + lay.wmin);
                for g in &lay.grid {
                    addr.push_str(&format!(" + {} * i32({})", g.coef, global(g, c)));
                }
                for wv in &lay.window {
                    addr.push_str(&format!(" + {} * i32({})", wv.coef, base(wv)));
                }
                for (name, coef) in &r.key.uniform {
                    addr.push_str(&format!(" + {coef} * i32({name})"));
                }
                (addr, live)
            };
            if v == 1 {
                let (addr, mut live) = component("");
                writeln!(s, "            let a = {addr};").unwrap();
                if !lay.window.is_empty() {
                    // a window spans lanes / steps past the edge: keep the
                    // read inside the buffer (those slots are never used)
                    live.push_str(&format!(
                        " && u32(a) < {}",
                        self.buffer_len_expr(r.key.buffer)
                    ));
                }
                writeln!(s, "            if ({live}) {{").unwrap();
                let mut val = load(
                    self.nodes[r.key.buffer].precision.storage,
                    &format!("b{}", r.key.buffer),
                    "u32(a)",
                );
                if let Some((coords, shape)) = &r.key.guard {
                    let cond = coords
                        .iter()
                        .zip(shape)
                        .enumerate()
                        .map(|(ci, (c, n))| {
                            let mut parts = vec![c.offset.to_string()];
                            for (k, v) in &c.terms {
                                let var =
                                    lay.grid.iter().chain(&lay.window).find(|lv| &lv.name == k);
                                let windowed = lay.window.iter().any(|lv| &lv.name == k);
                                let e = match var {
                                    Some(lv) if windowed => base(lv),
                                    Some(lv) => global(lv, ""),
                                    None => k.clone(),
                                };
                                parts.push(format!("{v} * i32({e})"));
                            }
                            if let Some((gc, stride)) = lay.guard_coord {
                                if gc == ci {
                                    parts.push(format!("({} + i32(w)) / {stride}", lay.wmin));
                                }
                            }
                            format!("(u32({}) < {n}u)", parts.join(" + "))
                        })
                        .collect::<Vec<_>>()
                        .join(" && ");
                    val = format!("select(0.0, {val}, {cond})");
                }
                writeln!(s, "                stage_{i}[{sidx}] = {val};").unwrap();
                writeln!(s, "            }}").unwrap();
            } else {
                let mut elems = Vec::with_capacity(v);
                for c in 0..v {
                    let (addr, live) = component(&c.to_string());
                    writeln!(s, "            var e{c} = 0.0;").unwrap();
                    let e = load(
                        self.nodes[r.key.buffer].precision.storage,
                        &format!("b{}", r.key.buffer),
                        &format!("u32({addr})"),
                    );
                    writeln!(s, "            if ({live}) {{ e{c} = {e}; }}").unwrap();
                    elems.push(format!("e{c}"));
                }
                writeln!(
                    s,
                    "            stage_{i}[{sidx}] = vec{v}<f32>({});",
                    elems.join(", ")
                )
                .unwrap();
            }
            writeln!(s, "        }}").unwrap();
        }
        for (i, (rows, code, lanes, fold)) in comp_loads.iter().enumerate() {
            let total = rows * chunk;
            let lanes = *lanes;
            if lanes > 1 {
                // partials: thread idx is lane idx % lanes of value idx / lanes
                writeln!(
                    s,
                    "        for (var idx: u32 = tid; idx < {}u; idx = idx + {threads}u) {{",
                    total * lanes
                )
                .unwrap();
                writeln!(s, "            let lane_ = idx % {lanes}u;").unwrap();
                writeln!(s, "            let t = (idx / {lanes}u) % {chunk}u;").unwrap();
                writeln!(s, "            let ly = idx / {lanes}u / {chunk}u;").unwrap();
                writeln!(
                    s,
                    "            let gy = min(tile_y * {py_n}u + ly, {}u);",
                    ny - 1
                )
                .unwrap();
                writeln!(s, "            if (chunk + t < {r_total}u) {{").unwrap();
                for line in code.lines() {
                    writeln!(s, "                {line}").unwrap();
                }
                writeln!(s, "            }}").unwrap();
                writeln!(s, "        }}").unwrap();
                writeln!(s, "        workgroupBarrier();").unwrap();
                writeln!(
                    s,
                    "        for (var idx: u32 = tid; idx < {total}u; idx = idx + {threads}u) {{"
                )
                .unwrap();
                writeln!(s, "            let t = idx % {chunk}u;").unwrap();
                writeln!(s, "            let ly = idx / {chunk}u;").unwrap();
                writeln!(s, "            if (chunk + t < {r_total}u) {{").unwrap();
                for line in fold.lines() {
                    writeln!(s, "                {line}").unwrap();
                }
                writeln!(s, "            }}").unwrap();
                writeln!(s, "        }}").unwrap();
                let _ = i;
                continue;
            }
            writeln!(
                s,
                "        for (var idx: u32 = tid; idx < {total}u; idx = idx + {threads}u) {{"
            )
            .unwrap();
            writeln!(s, "            let t = idx % {chunk}u;").unwrap();
            writeln!(s, "            let ly = idx / {chunk}u;").unwrap();
            writeln!(
                s,
                "            let gy = min(tile_y * {py_n}u + ly, {}u);",
                ny - 1
            )
            .unwrap();
            writeln!(s, "            if (chunk + t < {r_total}u) {{").unwrap();
            for line in code.lines() {
                writeln!(s, "                {line}").unwrap();
            }
            writeln!(s, "            }}").unwrap();
            writeln!(s, "        }}").unwrap();
            let _ = i;
        }
        writeln!(s, "        workgroupBarrier();").unwrap();
        writeln!(
            s,
            "        for (var r0: u32 = chunk; r0 < min(chunk + {chunk}u, {r_total}u); r0 = r0 + 1u) {{"
        )
        .unwrap();
        // staged reads of site regions, once per thread per r0 (outside
        // the inner reduce dims: a site's address does not depend on
        // them): the thread's points along the tile axis, in groups of
        // the vector width
        for (n, site) in ctx.sites.iter().enumerate() {
            let r = &ctx.regions[site.region];
            let (axis, _) = axis_of(r);
            let v = vec_w[site.region];
            let points: usize = layouts[site.region]
                .split()
                .1
                .iter()
                .map(|l| l.ext)
                .product();
            let base = format!(
                "(({}) * {}u + {}u) * {points}u",
                site.local,
                r.span(),
                site.off
            );
            let (reg, first) = match axis {
                'x' => (rx, format!("tx * {rx}u")),
                'y' => (ry, format!("ty * {ry}u")),
                _ => (1, "0u".to_string()),
            };
            for g in 0..reg / v {
                if v == 4 {
                    writeln!(
                        s,
                        "            let svec{n}_g{g} = stage_{}[({base}) / 4u + ({first}) / 4u + {g}u];",
                        site.region
                    )
                    .unwrap();
                } else {
                    writeln!(
                        s,
                        "            let svec{n}_g{g} = stage_{}[{base} + {first} + {g}u];",
                        site.region
                    )
                    .unwrap();
                }
            }
        }
        let site_names: Vec<(String, char, usize)> = ctx
            .sites
            .iter()
            .enumerate()
            .map(|(n, site)| {
                let (axis, _) = axis_of(&ctx.regions[site.region]);
                (format!("svec{n}_"), axis, vec_w[site.region])
            })
            .collect();
        // inner reduce dims loop outside the register-tile copies, so every
        // copy shares the staged reads of one (r0, u…) step
        let outer = std::mem::take(&mut s);
        for &(i, j) in &copies {
            let t = tag(i, j);
            writeln!(s, "            {{").unwrap();
            writeln!(s, "            let p{dx} = min(x_raw{t}, {}u);", nx - 1).unwrap();
            if let Some(dy) = dy {
                writeln!(s, "            let p{dy} = min(y_raw{t}, {}u);", ny - 1).unwrap();
            }
            writeln!(s, "            let lx_ = tx * {rx}u + {i}u;").unwrap();
            writeln!(s, "            let ly_ = ty * {ry}u + {j}u;").unwrap();
            let rename = |text: &str| {
                let mut out = text
                    .replace("acc[", &format!("acc{t}["))
                    .replace("hoist_", &format!("hoist{t}_"));
                for (name, axis, v) in &site_names {
                    let k = match axis {
                        'x' => i,
                        'y' => j,
                        _ => 0,
                    };
                    let val = if *v == 4 {
                        format!("{name}g{}[{}]", k / 4, k % 4)
                    } else {
                        format!("{name}g{k}")
                    };
                    out = out.replace(name, &val);
                }
                out
            };
            for line in rename(&body).lines() {
                writeln!(s, "            {line}").unwrap();
            }
            let accum = self.nodes[red_node].precision.accum;
            let combined_t: Vec<String> =
                combined.iter().map(|e| quant(accum, &rename(e))).collect();
            writeln!(
                s,
                "            acc{t} = array<f32, {w}>({});",
                combined_t.join(", ")
            )
            .unwrap();
            writeln!(s, "            }}").unwrap();
        }
        let step = std::mem::replace(&mut s, outer);
        let n_inner: usize = inner_red.iter().map(|(_, n)| n).product();
        if n_inner <= 16 {
            // small inner extents (conv taps): unrolled, each step a
            // constant the compiler folds into the staged addresses
            for flat in 0..n_inner {
                writeln!(s, "        {{").unwrap();
                let mut rem = flat;
                for (name, n) in inner_red.iter().rev() {
                    writeln!(s, "        let {name}: u32 = {}u;", rem % n).unwrap();
                    rem /= n;
                }
                s.push_str(&step);
                writeln!(s, "        }}").unwrap();
            }
        } else {
            for (name, n) in &inner_red {
                writeln!(
                    s,
                    "        for (var {name}: u32 = 0u; {name} < {n}u; {name} = {name} + 1u) {{"
                )
                .unwrap();
            }
            s.push_str(&step);
            for _ in &inner_red {
                writeln!(s, "        }}").unwrap();
            }
        }
        writeln!(s, "        }}").unwrap();
        writeln!(s, "        workgroupBarrier();").unwrap();
        writeln!(s, "    }}").unwrap();
        for &(i, j) in &copies {
            let t = tag(i, j);
            writeln!(s, "    if (is_live{t}) {{").unwrap();
            let point_flat = {
                let mut acc = String::from("0u");
                for (d, n) in pts.iter().enumerate() {
                    let coord = if d == dx {
                        format!("x_raw{t}")
                    } else if Some(d) == dy {
                        format!("y_raw{t}")
                    } else {
                        format!("p{d}")
                    };
                    acc = format!("({acc}) * {n}u + {coord}");
                }
                acc
            };
            writeln!(s, "        let g = {point_flat};").unwrap();
            writeln!(s, "        let p{dx} = x_raw{t};").unwrap();
            if let Some(dy) = dy {
                writeln!(s, "        let p{dy} = y_raw{t};").unwrap();
            }
            writeln!(s, "        let lx_ = tx * {rx}u + {i}u;").unwrap();
            writeln!(s, "        let ly_ = ty * {ry}u + {j}u;").unwrap();
            let rename = |text: &str| {
                text.replace("acc[", &format!("acc{t}["))
                    .replace("hoist_", &format!("hoist{t}_"))
            };
            for line in rename(&epi).lines() {
                writeln!(s, "        {line}").unwrap();
            }
            for (slot, val) in epi_vals.iter().enumerate() {
                let st = store(
                    self.nodes[root].precision.storage,
                    &format!("g * {w_out}u + {slot}u"),
                    &rename(val),
                );
                writeln!(s, "        {st}").unwrap();
            }
            writeln!(s, "    }}").unwrap();
        }
        writeln!(s, "}}").unwrap();

        Some(WgslKernel {
            source: s,
            node: TensorId(root as u32),
            reads: reads.into_iter().map(|j| TensorId(j as u32)).collect(),
            points: pts.iter().product::<usize>() as u32,
            count: (n_tiles * threads) as u32,
            workgroup_size: threads as u32,
        })
    }
}

/// Lanes per staged reduction value in a cooperative load (`TANG_LOAD_LANES`
/// overrides; 1 is the one-thread-per-value load). 4, 8 and 16 were within
/// 5% on decode attention on a 3090; 32 (a warp per value) lost 30% to the
/// serial fold of the partials.
fn load_lanes() -> usize {
    static L: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *L.get_or_init(|| {
        std::env::var("TANG_LOAD_LANES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8)
    })
}
