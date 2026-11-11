//! Jacobi (fixed-point) parallel decoding on Qwen3 over the MLX 4-bit
//! checkpoint, on the CPU: how many tokens each k-row window finalizes.
//!
//! ```text
//! jacobi_cpu <model dir> <prompt dir> [max_new] [threads]
//! ```
//!
//! Every prompt is decoded greedily (the reference), then by each method:
//!
//! - `jacobi`: one window of k rows, the root (the last finalized token)
//!   and k-1 guesses; each iteration replaces the guesses with the model's
//!   greedy outputs and keeps the prefix that agreed.
//! - `jacobi+pool`: the k rows split into G branches of (k-1)/G guesses:
//!   the Jacobi carry plus G-1 continuations from an n-gram pool (n-grams of
//!   the prompt, the finalized text and every Jacobi trajectory, keyed by
//!   their first n-1 tokens, chained to fill a branch).
//! - `pld`: prompt lookup: one branch, the continuation of the longest
//!   (≤ 4) n-gram match in the prompt and finalized text, k-1 tokens.
//!
//! Every iteration is one pass over the weights (k rows), so tokens per
//! iteration is tokens per weight read. Outputs must equal the greedy
//! decode token for token; the forward computes every row with the same
//! arithmetic in the same order whatever k is, so a row on a correct prefix
//! reproduces the greedy logits bit for bit.
//!
//! The forward is a plain f32 reference (the 4-bit dequantization per
//! group, rows independent), threaded over output rows and heads.
use std::collections::HashMap;
use std::time::Instant;

use tang_bench::qwen_session::{St, dims_for};

const GROUP: usize = 64;

struct Q4 {
    q: *const u32,
    s: Vec<f32>,
    b: Vec<f32>,
    rows: usize,
    cols: usize,
}
unsafe impl Sync for Q4 {}
unsafe impl Send for Q4 {}

impl Q4 {
    fn load(st: &St, stem: &str) -> Self {
        let w = format!("{stem}.weight");
        let shape = &st
            .index
            .get(&w)
            .unwrap_or_else(|| &st.index[&w.replace("lm_head", "model.embed_tokens")])
            .1;
        let (rows, cols) = (shape[0], shape[1] * 8);
        Q4 {
            q: st.raw(&w).as_ptr() as *const u32,
            s: st.bf16(&format!("{stem}.scales")),
            b: st.bf16(&format!("{stem}.biases")),
            rows,
            cols,
        }
    }

    /// One row dequantized (the embedding lookup).
    fn row(&self, r: usize) -> Vec<f32> {
        let g = self.cols / GROUP;
        let mut out = vec![0.0; self.cols];
        for k in 0..self.cols {
            let w = unsafe { *self.q.add(r * self.cols / 8 + k / 8) };
            let n = ((w >> (4 * (k % 8))) & 15) as f32;
            out[k] = n * self.s[r * g + k / GROUP] + self.b[r * g + k / GROUP];
        }
        out
    }
}

/// `x` is `[m][cols]`; returns `[m][rows]`. Each output is the same
/// sequence of operations for any m.
fn matmul(w: &Q4, x: &[Vec<f32>], threads: usize) -> Vec<Vec<f32>> {
    let m = x.len();
    let groups = w.cols / GROUP;
    // per-input group sums (the bias term)
    let xs: Vec<Vec<f32>> = x
        .iter()
        .map(|v| v.chunks(GROUP).map(|c| c.iter().sum()).collect())
        .collect();
    let chunk = w.rows.div_ceil(threads);
    let parts: Vec<Vec<f32>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let (x, xs) = (&x, &xs);
                sc.spawn(move || {
                    let r0 = (t * chunk).min(w.rows);
                    let r1 = ((t + 1) * chunk).min(w.rows);
                    let mut out = vec![0.0f32; (r1 - r0) * m];
                    let mut nib = [0.0f32; GROUP];
                    for r in r0..r1 {
                        let mut acc = [0.0f32; 32];
                        for g in 0..groups {
                            let base = unsafe { w.q.add(r * w.cols / 8 + g * GROUP / 8) };
                            for j in 0..GROUP / 8 {
                                let word = unsafe { *base.add(j) };
                                for e in 0..8 {
                                    nib[j * 8 + e] = ((word >> (4 * e)) & 15) as f32;
                                }
                            }
                            let s = w.s[r * groups + g];
                            let b = w.b[r * groups + g];
                            // independent accumulators per input; each
                            // input's sum order is the same for any m
                            let mut l = [[0.0f32; 8]; 32];
                            for c in 0..GROUP / 8 {
                                let wv = &nib[c * 8..c * 8 + 8];
                                for i in 0..m {
                                    let xv = &x[i][g * GROUP + c * 8..g * GROUP + c * 8 + 8];
                                    for e in 0..8 {
                                        l[i][e] += wv[e] * xv[e];
                                    }
                                }
                            }
                            for i in 0..m {
                                let l = &l[i];
                                let d = ((l[0] + l[1]) + (l[2] + l[3]))
                                    + ((l[4] + l[5]) + (l[6] + l[7]));
                                acc[i] += s * d + b * xs[i][g];
                            }
                        }
                        out[(r - r0) * m..(r - r0 + 1) * m].copy_from_slice(&acc[..m]);
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut y = vec![vec![0.0f32; w.rows]; m];
    let mut r = 0;
    for p in parts {
        for row in p.chunks(m) {
            for i in 0..m {
                y[i][r] = row[i];
            }
            r += 1;
        }
    }
    y
}

fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ss + eps).sqrt();
    x.iter().zip(w).map(|(a, b)| a * r * b).collect()
}

struct Layer {
    ln1: Vec<f32>,
    ln2: Vec<f32>,
    qn: Vec<f32>,
    kn: Vec<f32>,
    q: Q4,
    k: Q4,
    v: Q4,
    o: Q4,
    gate: Q4,
    up: Q4,
    down: Q4,
}

struct Model {
    embed: Q4,
    head: Q4,
    norm: Vec<f32>,
    layers: Vec<Layer>,
    heads: usize,
    kvh: usize,
    hd: usize,
    eps: f32,
    inv_freq: Vec<f32>,
    threads: usize,
}

/// One window row: its token, position and the window rows it attends to
/// (its ancestors and itself, in position order).
#[derive(Clone)]
struct Row {
    tok: u32,
    pos: usize,
    vis: Vec<usize>,
}

#[derive(Clone)]
struct Cache {
    k: Vec<Vec<f32>>, // [layer][pos * kvh * hd]
    v: Vec<Vec<f32>>,
    len: usize,
}

impl Model {
    fn rope(&self, x: &mut [f32], pos: usize) {
        let h = self.hd / 2;
        for i in 0..h {
            let a = pos as f32 * self.inv_freq[i];
            let (s, c) = a.sin_cos();
            let (x0, x1) = (x[i], x[i + h]);
            x[i] = x0 * c - x1 * s;
            x[i + h] = x1 * c + x0 * s;
        }
    }

    /// Runs the window over the cache; returns each row's argmax and the
    /// rows' per-layer K/V (to commit the accepted ones).
    fn forward(
        &self,
        cache: &Cache,
        rows: &[Row],
    ) -> (Vec<u32>, Vec<Vec<Vec<f32>>>, Vec<Vec<Vec<f32>>>) {
        let m = rows.len();
        let kd = self.kvh * self.hd;
        let mut x: Vec<Vec<f32>> = rows
            .iter()
            .map(|r| self.embed.row(r.tok as usize))
            .collect();
        let mut ks = Vec::new();
        let mut vs = Vec::new();
        for (li, l) in self.layers.iter().enumerate() {
            let h: Vec<Vec<f32>> = x.iter().map(|v| rms(v, &l.ln1, self.eps)).collect();
            let mut q = matmul(&l.q, &h, self.threads);
            let mut k = matmul(&l.k, &h, self.threads);
            let v = matmul(&l.v, &h, self.threads);
            for (i, r) in rows.iter().enumerate() {
                for hh in 0..self.heads {
                    let s = &mut q[i][hh * self.hd..(hh + 1) * self.hd];
                    let n = rms(s, &l.qn, self.eps);
                    s.copy_from_slice(&n);
                    self.rope(s, r.pos);
                }
                for hh in 0..self.kvh {
                    let s = &mut k[i][hh * self.hd..(hh + 1) * self.hd];
                    let n = rms(s, &l.kn, self.eps);
                    s.copy_from_slice(&n);
                    self.rope(s, r.pos);
                }
            }
            // attention, threaded over (row, head)
            let ck = &cache.k[li];
            let cv = &cache.v[li];
            let clen = cache.len;
            let scale = 1.0 / (self.hd as f32).sqrt();
            let jobs: Vec<(usize, usize)> = (0..m)
                .flat_map(|i| (0..self.heads).map(move |h| (i, h)))
                .collect();
            let per = jobs.len().div_ceil(self.threads);
            let (qr, kr, vr) = (&q, &k, &v);
            let outs: Vec<Vec<(usize, usize, Vec<f32>)>> = std::thread::scope(|sc| {
                let hs: Vec<_> = jobs
                    .chunks(per)
                    .map(|js| {
                        sc.spawn(move || {
                            js.iter()
                                .map(|&(i, hh)| {
                                    let g = hh / (self.heads / self.kvh);
                                    let qv = &qr[i][hh * self.hd..(hh + 1) * self.hd];
                                    let mut sc_ = Vec::with_capacity(clen + rows[i].vis.len());
                                    for p in 0..clen {
                                        let kv =
                                            &ck[p * kd + g * self.hd..p * kd + (g + 1) * self.hd];
                                        sc_.push(dot(qv, kv) * scale);
                                    }
                                    for &j in &rows[i].vis {
                                        let kv = &kr[j][g * self.hd..(g + 1) * self.hd];
                                        sc_.push(dot(qv, kv) * scale);
                                    }
                                    let mx = sc_.iter().cloned().fold(f32::MIN, f32::max);
                                    let mut sum = 0.0;
                                    for s in sc_.iter_mut() {
                                        *s = (*s - mx).exp();
                                        sum += *s;
                                    }
                                    let mut o = vec![0.0f32; self.hd];
                                    for p in 0..clen {
                                        let vv =
                                            &cv[p * kd + g * self.hd..p * kd + (g + 1) * self.hd];
                                        let w = sc_[p] / sum;
                                        for d in 0..self.hd {
                                            o[d] += w * vv[d];
                                        }
                                    }
                                    for (t, &j) in rows[i].vis.iter().enumerate() {
                                        let vv = &vr[j][g * self.hd..(g + 1) * self.hd];
                                        let w = sc_[clen + t] / sum;
                                        for d in 0..self.hd {
                                            o[d] += w * vv[d];
                                        }
                                    }
                                    (i, hh, o)
                                })
                                .collect()
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let mut att = vec![vec![0.0f32; self.heads * self.hd]; m];
            for part in outs {
                for (i, hh, o) in part {
                    att[i][hh * self.hd..(hh + 1) * self.hd].copy_from_slice(&o);
                }
            }
            let o = matmul(&l.o, &att, self.threads);
            for i in 0..m {
                for d in 0..x[i].len() {
                    x[i][d] += o[i][d];
                }
            }
            let h: Vec<Vec<f32>> = x.iter().map(|v| rms(v, &l.ln2, self.eps)).collect();
            let g = matmul(&l.gate, &h, self.threads);
            let u = matmul(&l.up, &h, self.threads);
            let a: Vec<Vec<f32>> = g
                .iter()
                .zip(&u)
                .map(|(g, u)| {
                    g.iter()
                        .zip(u)
                        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                        .collect()
                })
                .collect();
            let dn = matmul(&l.down, &a, self.threads);
            for i in 0..m {
                for d in 0..x[i].len() {
                    x[i][d] += dn[i][d];
                }
            }
            ks.push(k);
            vs.push(v);
        }
        let h: Vec<Vec<f32>> = x.iter().map(|v| rms(v, &self.norm, self.eps)).collect();
        let logits = matmul(&self.head, &h, self.threads);
        let am = logits
            .iter()
            .map(|l| {
                let mut b = 0;
                for (i, &v) in l.iter().enumerate() {
                    if v > l[b] {
                        b = i;
                    }
                }
                b as u32
            })
            .collect();
        (am, ks, vs)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut l = [0.0f32; 8];
    for c in 0..a.len() / 8 {
        for e in 0..8 {
            l[e] += a[c * 8 + e] * b[c * 8 + e];
        }
    }
    ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]))
}

impl Cache {
    fn new(layers: usize, cap: usize, kd: usize) -> Self {
        Cache {
            k: (0..layers).map(|_| Vec::with_capacity(cap * kd)).collect(),
            v: (0..layers).map(|_| Vec::with_capacity(cap * kd)).collect(),
            len: 0,
        }
    }
    fn commit(&mut self, ks: &[Vec<Vec<f32>>], vs: &[Vec<Vec<f32>>], rows: &[usize]) {
        for l in 0..ks.len() {
            for &r in rows {
                self.k[l].extend_from_slice(&ks[l][r]);
                self.v[l].extend_from_slice(&vs[l][r]);
            }
        }
        self.len += rows.len();
    }
}

#[derive(Clone, Copy, Debug)]
enum Method {
    Greedy,
    Jacobi { k: usize },
    Pool { k: usize, n: usize, g: usize },
}

impl Method {
    fn name(&self) -> String {
        match self {
            Method::Greedy => "greedy".into(),
            Method::Jacobi { k } => format!("jacobi k={k}"),
            Method::Pool { k, n, g } => format!("jacobi+pool k={k} n={n} G={g}"),
        }
    }
}

/// n-gram pool: (n-1)-gram key -> continuation tokens, most recent first,
/// at most `cap` per key.
struct Pool {
    n: usize,
    cap: usize,
    map: HashMap<Vec<u32>, Vec<u32>>,
}

impl Pool {
    fn add(&mut self, key: &[u32], next: u32) {
        let e = self.map.entry(key.to_vec()).or_default();
        if let Some(i) = e.iter().position(|&t| t == next) {
            e.remove(i);
        }
        e.insert(0, next);
        e.truncate(self.cap);
    }
    fn add_seq(&mut self, s: &[u32]) {
        if s.len() < self.n {
            return;
        }
        for w in s.windows(self.n) {
            self.add(&w[..self.n - 1], w[self.n - 1]);
        }
    }
    /// Up to `g` distinct continuations of `ctx`, each chained to `len`.
    fn candidates(&self, ctx: &[u32], g: usize, len: usize) -> Vec<Vec<u32>> {
        let key = &ctx[ctx.len() - (self.n - 1)..];
        let Some(firsts) = self.map.get(key) else {
            return vec![];
        };
        firsts
            .iter()
            .take(g)
            .map(|&f| {
                let mut c = vec![f];
                let mut tail: Vec<u32> = key.to_vec();
                tail.push(f);
                while c.len() < len {
                    let k = &tail[tail.len() - (self.n - 1)..];
                    match self.map.get(k) {
                        Some(v) => {
                            c.push(v[0]);
                            tail.push(v[0]);
                        }
                        None => break,
                    }
                }
                c
            })
            .collect()
    }
}

struct Stats {
    iters: usize,
    rows: usize,
    toks: usize,
    secs: f64,
}

/// Decode `max_new` tokens after `prompt` by `method`; returns the tokens.
/// The prompt but its last token in the cache, 32 rows at a time (causal).
fn prefill(m: &Model, prompt: &[u32], cap: usize) -> Cache {
    let mut cache = Cache::new(m.layers.len(), cap, m.kvh * m.hd);
    let mut i0 = 0;
    while i0 + 1 < prompt.len() {
        let i1 = (i0 + 32).min(prompt.len() - 1);
        let pre: Vec<Row> = (i0..i1)
            .map(|i| Row {
                tok: prompt[i],
                pos: i,
                vis: (0..=i - i0).collect(),
            })
            .collect();
        let (_, ks, vs) = m.forward(&cache, &pre);
        cache.commit(&ks, &vs, &(0..pre.len()).collect::<Vec<_>>());
        i0 = i1;
    }
    cache
}

fn decode(
    m: &Model,
    pre: &Cache,
    prompt: &[u32],
    max_new: usize,
    method: Method,
    eos: &[u32],
) -> (Vec<u32>, Stats) {
    let mut cache = pre.clone();
    let mut root = *prompt.last().unwrap();
    let mut out: Vec<u32> = Vec::new();
    let mut st = Stats {
        iters: 0,
        rows: 0,
        toks: 0,
        secs: 0.0,
    };
    let (k, n, g) = match method {
        Method::Greedy => (1, 0, 0),
        Method::Jacobi { k } => (k, 0, 1),
        Method::Pool { k, n, g } => (k, n, g),
    };
    let mut pool = Pool {
        n: n.max(2),
        cap: 16,
        map: HashMap::new(),
    };
    if n > 0 {
        pool.add_seq(prompt);
    }
    // Jacobi carry: guesses for the positions after root
    let mut carry: Vec<u32> = Vec::new();
    let t1 = Instant::now();
    while out.len() < max_new {
        let p = cache.len;
        // branches of guesses: pool candidates (G-1, each <= (k-1)/G),
        // then the Jacobi carry filling the remaining rows (first branch)
        let mut branches: Vec<Vec<u32>> = Vec::new();
        if k > 1 {
            if n > 0 {
                let mut ctx: Vec<u32> = prompt.to_vec();
                ctx.extend_from_slice(&out);
                branches = pool.candidates(&ctx, g - 1, (k - 1) / g);
            }
            let room = (k - 1) - branches.iter().map(|b| b.len()).sum::<usize>();
            let mut j = carry.clone();
            if j.is_empty() {
                j.push(root);
            }
            while j.len() < room {
                j.push(*j.last().unwrap());
            }
            j.truncate(room);
            branches.insert(0, j);
        }
        // rows: root then each branch
        let mut rows = vec![Row {
            tok: root,
            pos: p,
            vis: vec![0],
        }];
        let mut spans = Vec::new();
        for b in &branches {
            let start = rows.len();
            for (d, &t) in b.iter().enumerate() {
                let mut vis = vec![0];
                vis.extend(start..start + d + 1);
                rows.push(Row {
                    tok: t,
                    pos: p + 1 + d,
                    vis,
                });
            }
            spans.push(start);
        }
        let (am, ks, vs) = m.forward(&cache, &rows);
        st.iters += 1;
        st.rows += rows.len();
        // best branch: guesses accepted while they equal the previous row's output
        let mut best = (0usize, usize::MAX);
        for (bi, b) in branches.iter().enumerate() {
            let s = spans[bi];
            let mut acc = 0;
            let mut prev = am[0];
            for d in 0..b.len() {
                if b[d] != prev {
                    break;
                }
                acc += 1;
                prev = am[s + d];
            }
            if acc > best.0 {
                best = (acc, bi);
            }
        }
        let mut keep = vec![0usize];
        let mut new = vec![am[0]];
        if best.1 != usize::MAX {
            let s = spans[best.1];
            for d in 0..best.0 {
                keep.push(s + d);
                new.push(am[s + d]);
            }
        }
        cache.commit(&ks, &vs, &keep);
        // trajectory n-grams: each branch row's inputs followed by its output
        if n > 0 {
            for (bi, b) in branches.iter().enumerate() {
                let s = spans[bi];
                let mut seq = vec![root];
                seq.extend_from_slice(b);
                for d in 0..b.len() {
                    // inputs up to row s+d are seq[..=d+1]; its output am[s+d]
                    if d + 2 >= pool.n - 1 {
                        let key = &seq[d + 2 - (pool.n - 1)..d + 2];
                        pool.add(key, am[s + d]);
                    }
                }
            }
        }
        // Jacobi carry: the carry branch's outputs past the accepted count
        // (row d predicts position p+2+d; the new root sits at p+1+acc)
        carry.clear();
        if !branches.is_empty() {
            let acc = if best.1 == usize::MAX { 0 } else { best.0 };
            for d in acc..branches[0].len() {
                carry.push(am[spans[0] + d]);
            }
        }
        let before = out.len();
        let mut stop = false;
        for &t in &new {
            out.push(t);
            if eos.contains(&t) || out.len() >= max_new {
                stop = true;
                break;
            }
        }
        if n > 0 {
            let mut ctx: Vec<u32> = prompt.to_vec();
            ctx.extend_from_slice(&out);
            let from = (prompt.len() + before + 1).saturating_sub(pool.n);
            pool.add_seq(&ctx[from..]);
        }
        if stop {
            break;
        }
        root = *out.last().unwrap();
    }
    st.toks = out.len();
    st.secs = t1.elapsed().as_secs_f64();
    (out, st)
}

/// Prompt lookup, simulated from the greedy tokens (its acceptance depends
/// only on them): per iteration, the continuation (≤ k-1) of the longest
/// n-gram (n = 4..1) ending the text that occurs earlier in prompt+text.
fn pld(prompt: &[u32], greedy: &[u32], k: usize) -> (usize, usize) {
    let mut done = 0;
    let (mut iters, mut rows) = (0, 0);
    while done < greedy.len() {
        let mut text: Vec<u32> = prompt.to_vec();
        text.extend_from_slice(&greedy[..done]);
        let mut cand: Vec<u32> = vec![];
        'n: for n in (1..=4).rev() {
            if text.len() <= n {
                continue;
            }
            let key = &text[text.len() - n..];
            for i in (0..text.len() - n).rev() {
                if &text[i..i + n] == key {
                    let end = (i + n + k - 1).min(text.len());
                    cand = text[i + n..end].to_vec();
                    if !cand.is_empty() {
                        break 'n;
                    }
                }
            }
        }
        // the root row finalizes greedy[done]; each guess equal to the
        // next greedy token finalizes one more
        let mut acc = 1;
        for (d, &t) in cand.iter().enumerate() {
            if done + d < greedy.len() && greedy[done + d] == t {
                acc += 1;
            } else {
                break;
            }
        }
        // a guess c_d equals greedy[done + d]: row d's input is the token at
        // position done+d; accepted iff the previous rows agreed
        iters += 1;
        rows += 1 + cand.len();
        done += acc.min(greedy.len() - done);
    }
    (iters, rows)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let dir = &a[0];
    let pdir = &a[1];
    let max_new: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(64);
    let threads: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(12);
    let methods_env = std::env::var("JACOBI_METHODS").unwrap_or_default();
    let d = dims_for(dir, 4096);
    let st = St::open(&format!("{dir}/model.safetensors"));
    let tok = tokenizers::Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    let bf = |n: &str| st.bf16(n);
    let layers = (0..d.layers)
        .map(|i| {
            let p = format!("model.layers.{i}");
            Layer {
                ln1: bf(&format!("{p}.input_layernorm.weight")),
                ln2: bf(&format!("{p}.post_attention_layernorm.weight")),
                qn: bf(&format!("{p}.self_attn.q_norm.weight")),
                kn: bf(&format!("{p}.self_attn.k_norm.weight")),
                q: Q4::load(&st, &format!("{p}.self_attn.q_proj")),
                k: Q4::load(&st, &format!("{p}.self_attn.k_proj")),
                v: Q4::load(&st, &format!("{p}.self_attn.v_proj")),
                o: Q4::load(&st, &format!("{p}.self_attn.o_proj")),
                gate: Q4::load(&st, &format!("{p}.mlp.gate_proj")),
                up: Q4::load(&st, &format!("{p}.mlp.up_proj")),
                down: Q4::load(&st, &format!("{p}.mlp.down_proj")),
            }
        })
        .collect();
    let hd = d.head_dim;
    let model = Model {
        embed: Q4::load(&st, "model.embed_tokens"),
        head: Q4::load(&st, "lm_head"),
        norm: bf("model.norm.weight"),
        layers,
        heads: d.heads,
        kvh: d.kv_heads,
        hd,
        eps: d.eps as f32,
        inv_freq: (0..hd / 2)
            .map(|i| 1.0 / (d.theta as f32).powf(2.0 * i as f32 / hd as f32))
            .collect(),
        threads,
    };
    let eos = [151645u32, 151643];
    let mut files: Vec<_> = std::fs::read_dir(pdir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "txt"))
        .collect();
    files.sort();
    let mut methods = vec![];
    for k in [4usize, 8, 16] {
        methods.push(Method::Jacobi { k });
        for n in [2usize, 3, 4] {
            methods.push(Method::Pool { k, n, g: 2 });
        }
        if k >= 8 {
            methods.push(Method::Pool { k, n: 3, g: 4 });
        }
    }
    if !methods_env.is_empty() {
        methods.retain(|m| methods_env.split(";").any(|f| m.name() == f.trim()));
    }
    println!("# prompt\tmethod\ttokens\titers\trows\tsecs\texact");
    for f in &files {
        let name = f.file_stem().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(f).unwrap();
        let prompt: Vec<u32> = tok.encode(text.as_str(), false).unwrap().get_ids().to_vec();
        let tp = Instant::now();
        let pre = prefill(&model, &prompt, prompt.len() + max_new + 32);
        eprintln!(
            "{name}: prefill {} tokens {:.1}s",
            prompt.len(),
            tp.elapsed().as_secs_f64()
        );
        let (greedy, gs) = decode(&model, &pre, &prompt, max_new, Method::Greedy, &eos);
        println!(
            "{name}\tgreedy\t{}\t{}\t{}\t{:.1}\t1\tprompt_tokens={}",
            gs.toks,
            gs.iters,
            gs.rows,
            gs.secs,
            prompt.len()
        );
        eprintln!("{name}: {:?}", tok.decode(&greedy, false).unwrap());
        for k in [4usize, 8, 16] {
            let (it, rows) = pld(&prompt, &greedy, k);
            println!("{name}\tpld k={k}\t{}\t{it}\t{rows}\t0\t1", greedy.len());
        }
        for &me in &methods {
            let (o, s) = decode(&model, &pre, &prompt, max_new, me, &eos);
            let exact = (o == greedy) as u8;
            println!(
                "{name}\t{}\t{}\t{}\t{}\t{:.1}\t{exact}",
                me.name(),
                s.toks,
                s.iters,
                s.rows,
                s.secs
            );
            if exact == 0 {
                eprintln!("MISMATCH {name} {}", me.name());
            }
        }
    }
}
