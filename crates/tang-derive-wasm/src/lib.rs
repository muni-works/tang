//! Derive attention kernels in the browser.
//!
//! The reader types an attention variant as two expressions, FlexAttention-style:
//!
//! - `score`: the score after modification, in `s` (the scaled q·k), `i` (query row),
//!   `j` (key column), `n` and `d`. `s` is plain attention; `30*tanh(s/30)` is a soft cap;
//!   `s - 0.25*(i-j)` is ALiBi.
//! - `mask`: which keys a query sees, as comparisons joined by `&&` / `||`. `j <= i` is
//!   causal; `j <= i && i - j < 64` is a causal sliding window. Empty means no mask.
//!
//! [`derive`] builds the naive graph `softmax(score)·v` (and, with `backward`, the gradients
//! of `Σ O·dO` with respect to q, k and v), runs tang's real pass pipeline over it, and
//! returns both programs as JSON: WGSL sources, bindings and launch sizes, ready for a
//! WebGPU runner. [`reference`] evaluates the naive graph in f64 for the correctness check.
//! Nothing here knows what attention is beyond building the naive graph; flash attention
//! comes out of `stabilize_softmax`, `substitute_maps` and `schedule_auto` like everywhere
//! else in tang.

use std::fmt::Write as _;

use tang_compile::tensor::wgsl::WgslProgram;
use tang_compile::tensor::{Monoid, TensorGraph, TensorId};
use tang_expr::{with_graph, ExprId};
use wasm_bindgen::prelude::*;

// ---- expressions -------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Ast {
    Num(f64),
    Var(char),
    Neg(Box<Ast>),
    Bin(char, Box<Ast>, Box<Ast>),
    Call(String, Vec<Ast>),
    /// comparison, as a value that is > 0 exactly when it holds (integer coordinates)
    Cmp(&'static str, Box<Ast>, Box<Ast>),
    And(Box<Ast>, Box<Ast>),
    Or(Box<Ast>, Box<Ast>),
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Parser {
            s: s.as_bytes(),
            i: 0,
        }
    }
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.s.get(self.i).copied()
    }
    fn eat(&mut self, t: &str) -> bool {
        self.ws();
        if self.s[self.i..].starts_with(t.as_bytes()) {
            self.i += t.len();
            true
        } else {
            false
        }
    }
    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("{what} at column {}", self.i + 1))
    }
    fn parse(mut self) -> Result<Ast, String> {
        let e = self.or()?;
        if self.peek().is_some() {
            return self.err("unexpected input");
        }
        Ok(e)
    }
    fn or(&mut self) -> Result<Ast, String> {
        let mut a = self.and()?;
        while self.eat("||") {
            a = Ast::Or(Box::new(a), Box::new(self.and()?));
        }
        Ok(a)
    }
    fn and(&mut self) -> Result<Ast, String> {
        let mut a = self.cmp()?;
        while self.eat("&&") {
            a = Ast::And(Box::new(a), Box::new(self.cmp()?));
        }
        Ok(a)
    }
    fn cmp(&mut self) -> Result<Ast, String> {
        let a = self.sum()?;
        for op in ["<=", ">=", "==", "!=", "<", ">"] {
            if self.eat(op) {
                let b = self.sum()?;
                return Ok(Ast::Cmp(op, Box::new(a), Box::new(b)));
            }
        }
        Ok(a)
    }
    fn sum(&mut self) -> Result<Ast, String> {
        let mut a = self.prod()?;
        loop {
            match self.peek() {
                Some(c @ (b'+' | b'-')) => {
                    self.i += 1;
                    a = Ast::Bin(c as char, Box::new(a), Box::new(self.prod()?));
                }
                _ => return Ok(a),
            }
        }
    }
    fn prod(&mut self) -> Result<Ast, String> {
        let mut a = self.unary()?;
        loop {
            match self.peek() {
                Some(c @ (b'*' | b'/')) => {
                    self.i += 1;
                    a = Ast::Bin(c as char, Box::new(a), Box::new(self.unary()?));
                }
                _ => return Ok(a),
            }
        }
    }
    fn unary(&mut self) -> Result<Ast, String> {
        if self.eat("-") {
            return Ok(Ast::Neg(Box::new(self.unary()?)));
        }
        let base = self.atom()?;
        if self.eat("^") {
            let e = self.unary()?;
            return Ok(Ast::Call("pow".into(), vec![base, e]));
        }
        Ok(base)
    }
    fn atom(&mut self) -> Result<Ast, String> {
        match self.peek() {
            Some(b'(') => {
                self.i += 1;
                let e = self.or()?;
                if !self.eat(")") {
                    return self.err("expected )");
                }
                Ok(e)
            }
            Some(c) if c.is_ascii_digit() || c == b'.' => {
                let st = self.i;
                while self.i < self.s.len()
                    && (self.s[self.i].is_ascii_digit()
                        || self.s[self.i] == b'.'
                        || self.s[self.i] == b'e')
                {
                    self.i += 1;
                }
                let t = std::str::from_utf8(&self.s[st..self.i]).unwrap();
                t.parse().map(Ast::Num).or_else(|_| self.err("bad number"))
            }
            Some(c) if c.is_ascii_alphabetic() => {
                let st = self.i;
                while self.i < self.s.len()
                    && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'_')
                {
                    self.i += 1;
                }
                let name = std::str::from_utf8(&self.s[st..self.i])
                    .unwrap()
                    .to_string();
                if self.eat("(") {
                    let mut args = vec![self.or()?];
                    while self.eat(",") {
                        args.push(self.or()?);
                    }
                    if !self.eat(")") {
                        return self.err("expected )");
                    }
                    return Ok(Ast::Call(name, args));
                }
                match name.as_str() {
                    "s" | "i" | "j" | "n" | "d" => Ok(Ast::Var(name.chars().next().unwrap())),
                    "pi" => Ok(Ast::Num(std::f64::consts::PI)),
                    _ => self.err(&format!("unknown name `{name}` (use s, i, j, n, d)")),
                }
            }
            _ => self.err("expected a number, a name or ("),
        }
    }
}

fn uses(a: &Ast, v: char) -> bool {
    match a {
        Ast::Num(_) => false,
        Ast::Var(c) => *c == v,
        Ast::Neg(x) => uses(x, v),
        Ast::Bin(_, x, y) | Ast::Cmp(_, x, y) | Ast::And(x, y) | Ast::Or(x, y) => {
            uses(x, v) || uses(y, v)
        }
        Ast::Call(_, xs) => xs.iter().any(|x| uses(x, v)),
    }
}

const LOG2E: f64 = std::f64::consts::LOG2_E;
const LN2: f64 = std::f64::consts::LN_2;

fn lit(v: f64) -> ExprId {
    with_graph(|g| g.lit(v))
}
fn exp(x: ExprId) -> ExprId {
    with_graph(|g| g.exp2(x)) // caller scales
}
fn e_exp(x: ExprId) -> ExprId {
    exp(x * lit(LOG2E))
}
fn sel(c: ExprId, a: ExprId, b: ExprId) -> ExprId {
    with_graph(|g| g.select(c, a, b))
}

/// Lower an expression to a scalar graph. `env` gives s, i, j, n, d.
fn lower(a: &Ast, env: &dyn Fn(char) -> ExprId) -> Result<ExprId, String> {
    Ok(match a {
        Ast::Num(v) => lit(*v),
        Ast::Var(c) => env(*c),
        Ast::Neg(x) => -lower(x, env)?,
        Ast::Bin(op, x, y) => {
            let (x, y) = (lower(x, env)?, lower(y, env)?);
            match op {
                '+' => x + y,
                '-' => x - y,
                '*' => x * y,
                _ => x / y,
            }
        }
        // integer coordinates: a comparison becomes a value > 0 exactly when it holds
        Ast::Cmp(op, x, y) => {
            let (x, y) = (lower(x, env)?, lower(y, env)?);
            let h = lit(0.5);
            match *op {
                "<=" => y - x + h,
                ">=" => x - y + h,
                "<" => y - x - h,
                ">" => x - y - h,
                "==" => h - (x - y) * (x - y),
                _ => (x - y) * (x - y) - h,
            }
        }
        Ast::And(x, y) => sel(lower(x, env)?, lower(y, env)?, lit(-1.0)),
        Ast::Or(x, y) => sel(lower(x, env)?, lit(1.0), lower(y, env)?),
        Ast::Call(f, xs) => {
            let mut v = Vec::new();
            for x in xs {
                v.push(lower(x, env)?);
            }
            let arity = |k: usize| {
                if v.len() == k {
                    Ok(())
                } else {
                    Err(format!("{f} takes {k} argument(s)"))
                }
            };
            match f.as_str() {
                "exp" => {
                    arity(1)?;
                    e_exp(v[0])
                }
                "log" => {
                    arity(1)?;
                    with_graph(|g| g.log2(v[0])) * lit(LN2)
                }
                "sqrt" => {
                    arity(1)?;
                    with_graph(|g| g.sqrt(v[0]))
                }
                "abs" => {
                    arity(1)?;
                    with_graph(|g| g.sqrt(v[0] * v[0]))
                }
                "sin" => {
                    arity(1)?;
                    with_graph(|g| g.sin(v[0]))
                }
                "cos" => {
                    arity(1)?;
                    with_graph(|g| g.sin(v[0] + lit(std::f64::consts::FRAC_PI_2)))
                }
                "tanh" => {
                    arity(1)?;
                    // 1 − 2/(e^{2x} + 1), safe at both ends
                    lit(1.0) - lit(2.0) / (e_exp(v[0] * lit(2.0)) + lit(1.0))
                }
                "sigmoid" => {
                    arity(1)?;
                    lit(1.0) / (lit(1.0) + e_exp(-v[0]))
                }
                "min" => {
                    arity(2)?;
                    sel(v[1] - v[0], v[0], v[1])
                }
                "max" => {
                    arity(2)?;
                    sel(v[0] - v[1], v[0], v[1])
                }
                "pow" => {
                    arity(2)?;
                    exp(with_graph(|g| g.log2(v[0])) * v[1])
                }
                _ => return Err(format!("unknown function `{f}`")),
            }
        }
    })
}

// ---- the graph ---------------------------------------------------------------------------

pub struct Variant {
    score: Ast,
    mask: Option<Ast>,
}

pub fn parse_variant(score: &str, mask: &str) -> Result<Variant, String> {
    let score = Parser::new(score)
        .parse()
        .map_err(|e| format!("score: {e}"))?;
    let mask = if mask.trim().is_empty() {
        None
    } else {
        let m = Parser::new(mask)
            .parse()
            .map_err(|e| format!("mask: {e}"))?;
        if uses(&m, 's') {
            return Err("mask: the mask may use i, j, n and d, not s".into());
        }
        Some(m)
    };
    Ok(Variant { score, mask })
}

/// The naive graph: `O = softmax(score(q·kᵀ/√d, i, j) masked)·v`. Inputs q, k `[n,d]`, v
/// `[n,d]` and, for the backward, dO `[n,d]`. Returns the outputs to compute.
pub fn naive_graph(
    v: &Variant,
    n: usize,
    d: usize,
    backward: bool,
) -> Result<(TensorGraph, Vec<TensorId>), String> {
    let e = |r: Result<TensorId, tang_compile::tensor::ShapeError>| r.map_err(|e| format!("{e:?}"));
    let mut g = TensorGraph::new();
    let q = g.input(&[n, d]);
    let k = g.input(&[n, d]);
    let vv = g.input(&[n, d]);
    let scale = 1.0 / (d as f64).sqrt();
    let kt = e(g.transpose(k, &[1, 0]))?;
    let raw = e(g.matmul(q, kt))?;
    let ii = g.iota(&[n, n], 0);
    let jj = g.iota(&[n, n], 1);
    let (nf, df) = (n as f64, d as f64);
    // lower once outside to report errors; the closures below cannot return them
    {
        let env = |c: char| match c {
            'n' => lit(nf),
            'd' => lit(df),
            _ => ExprId::ZERO,
        };
        let _ = tang_expr::trace(|| lower(&v.score, &env).map(|_| ()));
        if let Some(m) = &v.mask {
            tang_expr::trace(|| lower(m, &env).map(|_| ())).1?;
        }
        tang_expr::trace(|| lower(&v.score, &env).map(|_| ())).1?;
    }
    let score_ast = v.score.clone();
    let mut fail = None;
    let s = e(g.map(&[raw, ii, jj], |x| {
        let env = |c: char| match c {
            's' => x[0] * lit(scale),
            'i' => x[1],
            'j' => x[2],
            'n' => lit(nf),
            _ => lit(df),
        };
        lower(&score_ast, &env).unwrap_or_else(|er| {
            fail = Some(er);
            ExprId::ZERO
        })
    }))?;
    if let Some(er) = fail {
        return Err(er);
    }
    let s = match &v.mask {
        None => s,
        Some(m) => {
            let m = m.clone();
            e(g.map(&[s, ii, jj], |x| {
                let env = |c: char| match c {
                    'i' => x[1],
                    'j' => x[2],
                    'n' => lit(nf),
                    _ => lit(df),
                };
                let keep = lower(&m, &env).unwrap_or(ExprId::ONE);
                // a masked score is a large negative additive bias, as tang's causal mask
                x[0] + sel(keep, ExprId::ZERO, lit(-1e30))
            }))?
        }
    };
    let mx = e(g.reduce(s, 1, Monoid::max()))?;
    let mb = e(g.broadcast(mx, &[n, n]))?;
    let p = e(g.map(&[s, mb], |x| e_exp(x[0] - x[1])))?;
    let l = e(g.reduce(p, 1, Monoid::sum()))?;
    let o = e(g.matmul(p, vv))?;
    let lb = e(g.broadcast(l, &[n, d]))?;
    let y = e(g.map(&[o, lb], |x| x[0] / x[1]))?;
    if !backward {
        return Ok((g, vec![y]));
    }
    let dout = g.input(&[n, d]);
    let prod = e(g.map(&[y, dout], |x| x[0] * x[1]))?;
    let r1 = e(g.reduce(prod, 1, Monoid::sum()))?;
    let loss = e(g.reduce(r1, 0, Monoid::sum()))?;
    let grads = g
        .backward(loss, &[q, k, vv])
        .map_err(|e| format!("{e:?}"))?;
    Ok((g, grads))
}

/// tang's pass pipeline, as the training examples run it.
pub fn derive_graph(g: &TensorGraph, outs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
    let (g, outs) = g.stabilize_softmax(outs);
    let (g, outs) = g.substitute_maps(&outs);
    let (g, outs) = g.eliminate_common(&outs);
    let (g, outs) = g.merge_siblings(&outs);
    let (g, outs) = g.relayout(&outs);
    let (g, outs) = g.merge_reduce_dims(&outs);
    let (mut g, outs) = g.split_reduce_dims(&outs);
    g.schedule_auto(&outs);
    (g, outs)
}

/// The naive graph run as written: every comprehension materialized, one thread per point.
pub fn naive_program(g: &TensorGraph, outs: &[TensorId]) -> WgslProgram {
    g.to_wgsl(outs)
}

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 16);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o
}

fn program_json(p: &WgslProgram, outs: &[TensorId], n: usize) -> String {
    let nn = n * n;
    let mut s = String::from("{\"kernels\":[");
    for (x, k) in p.kernels.iter().enumerate() {
        if x > 0 {
            s.push(',');
        }
        let reads: Vec<String> = k
            .reads
            .iter()
            .map(|t| p.storage(*t).index().to_string())
            .collect();
        let _ = write!(
            s,
            "{{\"source\":\"{}\",\"node\":{},\"reads\":[{}],\"points\":{},\"count\":{},\"workgroup_size\":{}}}",
            esc(&k.source),
            p.storage(k.node).index(),
            reads.join(","),
            k.points,
            k.count,
            k.workgroup_size
        );
    }
    s.push_str("],\"buffers\":[");
    for (x, b) in p.buffers.iter().enumerate() {
        if x > 0 {
            s.push(',');
        }
        let input = b.input.map_or("null".into(), |i| i.to_string());
        let _ = write!(
            s,
            "{{\"node\":{},\"len\":{},\"input\":{}}}",
            b.node.index(),
            b.len,
            input
        );
    }
    let outs: Vec<String> = outs
        .iter()
        .map(|t| p.storage(*t).index().to_string())
        .collect();
    let big = p
        .buffers
        .iter()
        .filter(|b| b.input.is_none() && b.len >= nn)
        .count();
    let _ = write!(
        s,
        "],\"outputs\":[{}],\"nxn_buffers\":{}}}",
        outs.join(","),
        big
    );
    s
}

/// Derive both programs. Returns JSON: `{naive, derived}` programs, or `{error}`.
#[wasm_bindgen]
pub fn derive(score: &str, mask: &str, n: usize, d: usize, backward: bool) -> String {
    let run = || -> Result<String, String> {
        if !(8..=4096).contains(&n) || !(1..=256).contains(&d) {
            return Err("n must be 8–4096 and d 1–256".into());
        }
        let v = parse_variant(score, mask)?;
        let (g, outs) = naive_graph(&v, n, d, backward)?;
        let naive = naive_program(&g, &outs);
        let (dg, douts) = derive_graph(&g, &outs);
        let derived = dg.to_wgsl(&douts);
        // other thread mappings for each derived kernel, one kernel changed at a time (what
        // tang's per-kernel autotuner tries); the lab times them all on the reader's GPU
        let mut cands = Vec::new();
        for k in &derived.kernels {
            for (split, tile, reg) in dg.schedule_candidates(k.node) {
                let mut v = dg.clone();
                v.set_schedule(k.node, split, tile, reg);
                let name = if split > 0 {
                    format!("split · {split} threads")
                } else if tile.0 > 0 {
                    let r = if reg.0 * reg.1 > 1 {
                        format!(" · {}×{} regs", reg.0, reg.1)
                    } else {
                        String::new()
                    };
                    format!("tile · {}×{}{r}", tile.0, tile.1)
                } else {
                    "one thread per point".into()
                };
                let name = if derived.kernels.len() > 1 {
                    format!("kernel {} · {name}", k.node.index())
                } else {
                    name
                };
                cands.push(format!(
                    "{{\"name\":\"{}\",\"program\":{}}}",
                    esc(&name),
                    program_json(&v.to_wgsl(&douts), &douts, n)
                ));
                if cands.len() >= 12 {
                    break;
                }
            }
        }
        Ok(format!(
            "{{\"naive\":{},\"derived\":{},\"candidates\":[{}]}}",
            program_json(&naive, &outs, n),
            program_json(&derived, &douts, n),
            cands.join(",")
        ))
    };
    match std::panic::catch_unwind(run) {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => format!("{{\"error\":\"{}\"}}", esc(&e)),
        Err(_) => "{\"error\":\"the compiler panicked on this variant\"}".into(),
    }
}

/// f64 reference for the check: the naive graph through tang's interpreter. `inputs` is the
/// concatenation of q, k, v (and dO for the backward), each `n·d` values. Returns the
/// outputs concatenated (O, or dq ‖ dk ‖ dv).
#[wasm_bindgen]
pub fn reference(
    score: &str,
    mask: &str,
    n: usize,
    d: usize,
    backward: bool,
    inputs: &[f32],
) -> Vec<f32> {
    let Ok(v) = parse_variant(score, mask) else {
        return vec![];
    };
    let Ok((g, outs)) = naive_graph(&v, n, d, backward) else {
        return vec![];
    };
    let k = if backward { 4 } else { 3 };
    if inputs.len() != k * n * d {
        return vec![];
    }
    let ins: Vec<Vec<f64>> = inputs
        .chunks(n * d)
        .map(|c| c.iter().map(|&x| x as f64).collect())
        .collect();
    g.eval(&ins, &outs)
        .into_iter()
        .flatten()
        .map(|x| x as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_derive() {
        for (score, mask) in [
            ("s", "j <= i"),
            ("30*tanh(s/30)", "j <= i"),
            ("s - 0.25*(i-j)", "j <= i && i - j < 16"),
            ("s", ""),
        ] {
            for bwd in [false, true] {
                let j = derive(score, mask, 64, 16, bwd);
                assert!(!j.contains("\"error\""), "{score} / {mask} / {bwd}: {j}");
                let derived = &j[j.find("\"derived\"").unwrap()..j.find("\"candidates\"").unwrap()];
                // neither pass writes n×n: the backward nests S and dP in its tiles
                assert!(
                    derived.contains("\"nxn_buffers\":0"),
                    "{score} / {mask} / {bwd}: derived writes n×n"
                );
                let nx = |p: &str| {
                    p.rsplit("\"nxn_buffers\":")
                        .next()
                        .unwrap()
                        .trim_end_matches(['}'])
                        .parse::<usize>()
                        .unwrap()
                };
                let naive = &j[..j.find("\"derived\"").unwrap()];
                eprintln!(
                    "{score:>16} | {mask:<22} | bwd {bwd:<5} | n×n naive {} derived {}",
                    nx(naive.trim_end_matches(',')),
                    nx(derived.trim_end_matches(','))
                );
            }
        }
    }

    #[test]
    fn reports_errors() {
        assert!(derive("s +", "", 64, 16, false).contains("error"));
        assert!(derive("q", "", 64, 16, false).contains("unknown name"));
        assert!(derive("s", "s > 0", 64, 16, false).contains("not s"));
    }

    #[test]
    fn reference_matches_shape() {
        let (n, d) = (16, 4);
        let x: Vec<f32> = (0..3 * n * d)
            .map(|i| ((i * 37 % 17) as f32 - 8.0) / 8.0)
            .collect();
        assert_eq!(reference("s", "j <= i", n, d, false, &x).len(), n * d);
        let x: Vec<f32> = (0..4 * n * d)
            .map(|i| ((i * 37 % 17) as f32 - 8.0) / 8.0)
            .collect();
        assert_eq!(reference("s", "j <= i", n, d, true, &x).len(), 3 * n * d);
    }
}
