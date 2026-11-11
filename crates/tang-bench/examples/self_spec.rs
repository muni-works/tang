//! Self-speculation acceptance on real weights, CPU only: how many tokens a
//! drafter derived from the model's own graph (layers skipped, early exit,
//! harder quantization; zero training) gets accepted per verify window
//! under greedy verification.
//!
//! A plain CPU Qwen3 forward over the MLX 4-bit checkpoint (dequantized in
//! the dot products, never materialized), rayon over matrix rows. For each
//! prompt: the full model's greedy continuation; then, for every drafter
//! and every position `t` of that continuation, the drafter drafts from the
//! verified prefix `x[..=t]` until its first miss (cap 8) — `match[t]`.
//! A speculative loop with window `k` is then simulated from `match`
//! exactly: a window at `t` accepts `min(match[t], k)` drafts plus the
//! verify's own token and the next window starts after them.
//!
//! KV: `shared` drafters attend over the full model's KV for the verified
//! prefix (the verify pass wrote it; the drafter's layers read theirs),
//! `own` drafters over a cache their own forward wrote over the prefix.
//!
//! ```text
//! QWEN_DIR=.../Qwen3-8B-4bit RAYON_NUM_THREADS=12 \
//!   cargo run --release -p tang-bench --example self_spec -- [gen] [prompts]
//! ```
//! Writes one JSON line per (prompt, drafter) to stdout; a summary to stderr.
use rayon::prelude::*;
use std::time::Instant;
use tang_bench::qwen_session::St;

const G: usize = 64;
const KMAX: usize = 8;

struct QMat {
    rows: usize,
    cols: usize,
    bits: u32,
    wpr: usize,
    q: Vec<u32>,
    s: Vec<f32>,
    b: Vec<f32>,
}

fn bf16(raw: &[u8]) -> Vec<f32> {
    raw.chunks(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

impl QMat {
    fn load(st: &St, stem: &str) -> Self {
        let w = format!("{stem}.weight");
        let shape = &st
            .index
            .get(&w)
            .unwrap_or_else(|| &st.index[&tang_bench::qwen_session::tied(&w)])
            .1;
        let (rows, cols) = (shape[0], shape[1] * 8);
        let q = st
            .raw(&w)
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Self {
            rows,
            cols,
            bits: 4,
            wpr: cols / 8,
            q,
            s: bf16(st.raw(&format!("{stem}.scales"))),
            b: bf16(st.raw(&format!("{stem}.biases"))),
        }
    }

    /// Weight bytes a matrix-vector product reads (codes + bf16 scale/bias).
    fn bytes(&self) -> f64 {
        let n = (self.rows * self.cols) as f64;
        n * self.bits as f64 / 8.0 + n / G as f64 * 4.0
    }

    fn code(&self, r: usize, k: usize) -> u32 {
        let per = (32 / self.bits) as usize;
        let w = self.q[r * self.wpr + k / per];
        (w >> ((k % per) as u32 * self.bits)) & ((1 << self.bits) - 1)
    }

    fn row(&self, r: usize, out: &mut [f32]) {
        let gpr = self.cols / G;
        for k in 0..self.cols {
            let g = r * gpr + k / G;
            out[k] = self.code(r, k) as f32 * self.s[g] + self.b[g];
        }
    }

    /// Round-to-nearest affine requantization of the dequantized weights
    /// to `bits` per group of 64 (min/max), as a drafter's copy.
    fn requant(&self, bits: u32) -> Self {
        let per = (32 / bits) as usize;
        let gpr = self.cols / G;
        let wpr = gpr * G.div_ceil(per);
        let levels = ((1u32 << bits) - 1) as f32;
        let rows: Vec<(Vec<u32>, Vec<f32>, Vec<f32>)> = (0..self.rows)
            .into_par_iter()
            .map(|r| {
                let mut v = vec![0.0; self.cols];
                self.row(r, &mut v);
                let (mut q, mut s, mut b) = (vec![0u32; wpr], vec![0.0; gpr], vec![0.0; gpr]);
                for g in 0..gpr {
                    let x = &v[g * G..(g + 1) * G];
                    let lo = x.iter().cloned().fold(f32::INFINITY, f32::min);
                    let hi = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sc = if hi > lo { (hi - lo) / levels } else { 1.0 };
                    s[g] = sc;
                    b[g] = lo;
                    for (i, &xv) in x.iter().enumerate() {
                        let c = (((xv - lo) / sc).round() as u32).min(levels as u32);
                        q[g * G.div_ceil(per) + i / per] |= c << ((i % per) as u32 * bits);
                    }
                }
                (q, s, b)
            })
            .collect();
        let mut out = Self {
            rows: self.rows,
            cols: self.cols,
            bits,
            wpr,
            q: Vec::with_capacity(self.rows * wpr),
            s: Vec::with_capacity(self.rows * gpr),
            b: Vec::with_capacity(self.rows * gpr),
        };
        for (q, s, b) in rows {
            out.q.extend(q);
            out.s.extend(s);
            out.b.extend(b);
        }
        out
    }

    /// `x` is `[t, cols]`; returns `[t, rows]`.
    fn matmul(&self, x: &[f32], t: usize) -> Vec<f32> {
        match self.bits {
            4 => self.mm::<4>(x, t),
            3 => self.mm::<3>(x, t),
            2 => self.mm::<2>(x, t),
            b => panic!("bits {b}"),
        }
    }

    fn mm<const B: u32>(&self, x: &[f32], t: usize) -> Vec<f32> {
        let (cols, gpr) = (self.cols, self.cols / G);
        let per = (32 / B) as usize;
        let mask = (1u32 << B) - 1;
        // per-token group sums: the bias term of every group's dot
        let gsum: Vec<f32> = (0..t * gpr)
            .map(|i| x[(i / gpr) * cols + (i % gpr) * G..][..G].iter().sum())
            .collect();
        let rb = 16;
        let mut rt = vec![0.0f32; self.rows * t];
        rt.par_chunks_mut(rb * t)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let mut n = [0.0f32; 80];
                for (ri, acc) in chunk.chunks_mut(t).enumerate() {
                    let r = ci * rb + ri;
                    let qrow = &self.q[r * self.wpr..(r + 1) * self.wpr];
                    for g in 0..gpr {
                        let wq = &qrow[g * G.div_ceil(per)..];
                        for (w, nc) in wq[..G.div_ceil(per)].iter().zip(n.chunks_mut(per)) {
                            for (j, nv) in nc.iter_mut().enumerate() {
                                *nv = ((*w >> (j as u32 * B)) & mask) as f32;
                            }
                        }
                        let (s, b) = (self.s[r * gpr + g], self.b[r * gpr + g]);
                        for ti in 0..t {
                            let xs = &x[ti * cols + g * G..][..G];
                            let mut l = [0.0f32; 16];
                            for (nc, xc) in n[..G].chunks_exact(16).zip(xs.chunks_exact(16)) {
                                for j in 0..16 {
                                    l[j] += nc[j] * xc[j];
                                }
                            }
                            let d: f32 = l.iter().sum();
                            acc[ti] += s * d + b * gsum[ti * gpr + g];
                        }
                    }
                }
            });
        let mut out = vec![0.0f32; t * self.rows];
        for r in 0..self.rows {
            for ti in 0..t {
                out[ti * self.rows + r] = rt[r * t + ti];
            }
        }
        out
    }
}

struct Layer {
    q: QMat,
    k: QMat,
    v: QMat,
    o: QMat,
    gate: QMat,
    up: QMat,
    down: QMat,
}

impl Layer {
    fn bytes(&self) -> f64 {
        [
            &self.q, &self.k, &self.v, &self.o, &self.gate, &self.up, &self.down,
        ]
        .iter()
        .map(|m| m.bytes())
        .sum()
    }
    fn requant(&self, bits: u32) -> Self {
        Self {
            q: self.q.requant(bits),
            k: self.k.requant(bits),
            v: self.v.requant(bits),
            o: self.o.requant(bits),
            gate: self.gate.requant(bits),
            up: self.up.requant(bits),
            down: self.down.requant(bits),
        }
    }
}

struct Norms {
    input: Vec<f32>,
    post: Vec<f32>,
    qn: Vec<f32>,
    kn: Vec<f32>,
}

struct Model {
    d: usize,
    heads: usize,
    kvh: usize,
    hd: usize,
    eps: f32,
    theta: f32,
    embed: QMat,
    head: QMat,
    norm: Vec<f32>,
    norms: Vec<Norms>,
    layers: Vec<Layer>,
}

/// Per layer `[pos][kv_heads * head_dim]` keys and values.
#[derive(Clone)]
struct Kv {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl Kv {
    fn new(n: usize) -> Self {
        Self {
            k: vec![Vec::new(); n],
            v: vec![Vec::new(); n],
        }
    }
    fn clear(&mut self) {
        self.k.iter_mut().for_each(Vec::clear);
        self.v.iter_mut().for_each(Vec::clear);
    }
}

fn rms(x: &mut [f32], w: &[f32], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    for (v, w) in x.iter_mut().zip(w) {
        *v *= r * w;
    }
}

/// Which layers a forward runs, and whose weights (`alt`: a requantized
/// copy indexed like `Model::layers`).
struct Plan<'a> {
    layers: &'a [usize],
    alt: Option<&'a [Layer]>,
}

impl Model {
    fn rope(&self, x: &mut [f32], pos: usize) {
        let h = self.hd / 2;
        for i in 0..h {
            let f = (pos as f32) * self.theta.powf(-((2 * i) as f32) / self.hd as f32);
            let (s, c) = f.sin_cos();
            let (a, b) = (x[i], x[i + h]);
            x[i] = a * c - b * s;
            x[i + h] = a * s + b * c;
        }
    }

    /// Feeds `tokens` at positions `plen + scratch.len()..`; attention reads
    /// `prefix[..plen]` then `scratch`. Returns the last token's logits.
    fn forward(
        &self,
        plan: &Plan,
        tokens: &[u32],
        prefix: Option<(&Kv, usize)>,
        scratch: &mut Kv,
    ) -> Vec<f32> {
        let (d, t, hd) = (self.d, tokens.len(), self.hd);
        let kvd = self.kvh * hd;
        let mut x = vec![0.0f32; t * d];
        for (i, &tok) in tokens.iter().enumerate() {
            self.embed.row(tok as usize, &mut x[i * d..(i + 1) * d]);
        }
        let plen = prefix.map_or(0, |p| p.1);
        let scale = 1.0 / (hd as f32).sqrt();
        for &li in plan.layers {
            let w = plan.alt.map_or(&self.layers[li], |a| &a[li]);
            let nw = &self.norms[li];
            let s0 = scratch.k[li].len() / kvd;
            let mut h = x.clone();
            h.chunks_mut(d).for_each(|r| rms(r, &nw.input, self.eps));
            let mut q = w.q.matmul(&h, t);
            let mut k = w.k.matmul(&h, t);
            let v = w.v.matmul(&h, t);
            for i in 0..t {
                let pos = plen + s0 + i;
                for hq in q[i * d..(i + 1) * d].chunks_mut(hd) {
                    rms(hq, &nw.qn, self.eps);
                    self.rope(hq, pos);
                }
                for hk in k[i * kvd..(i + 1) * kvd].chunks_mut(hd) {
                    rms(hk, &nw.kn, self.eps);
                    self.rope(hk, pos);
                }
            }
            scratch.k[li].extend_from_slice(&k);
            scratch.v[li].extend_from_slice(&v);
            let (sk, sv) = (&scratch.k[li], &scratch.v[li]);
            let (pk, pv): (&[f32], &[f32]) = match prefix {
                Some((p, n)) => (&p.k[li][..n * kvd], &p.v[li][..n * kvd]),
                None => (&[], &[]),
            };
            let group = self.heads / self.kvh;
            let mut att = vec![0.0f32; t * d];
            att.par_chunks_mut(hd).enumerate().for_each(|(ih, out)| {
                let (i, hh) = (ih / self.heads, ih % self.heads);
                let qv = &q[i * d + hh * hd..][..hd];
                let off = (hh / group) * hd;
                let n = plen + s0 + i + 1;
                let key = |j: usize| -> (&[f32], &[f32]) {
                    if j < plen {
                        (&pk[j * kvd + off..][..hd], &pv[j * kvd + off..][..hd])
                    } else {
                        let j = j - plen;
                        (&sk[j * kvd + off..][..hd], &sv[j * kvd + off..][..hd])
                    }
                };
                let mut sc: Vec<f32> = (0..n)
                    .map(|j| key(j).0.iter().zip(qv).map(|(a, b)| a * b).sum::<f32>() * scale)
                    .collect();
                let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut z = 0.0;
                sc.iter_mut().for_each(|s| {
                    *s = (*s - m).exp();
                    z += *s;
                });
                for (j, p) in sc.iter().enumerate() {
                    let vv = key(j).1;
                    for e in 0..hd {
                        out[e] += p / z * vv[e];
                    }
                }
            });
            let o = w.o.matmul(&att, t);
            x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);
            let mut h = x.clone();
            h.chunks_mut(d).for_each(|r| rms(r, &nw.post, self.eps));
            let gt = w.gate.matmul(&h, t);
            let up = w.up.matmul(&h, t);
            let act: Vec<f32> = gt
                .iter()
                .zip(&up)
                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let dn = w.down.matmul(&act, t);
            x.iter_mut().zip(&dn).for_each(|(a, b)| *a += b);
        }
        let mut last = x[(t - 1) * d..].to_vec();
        rms(&mut last, &self.norm, self.eps);
        self.head.matmul(&last, 1)
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut b = 0;
    for i in 1..v.len() {
        if v[i] > v[b] {
            b = i;
        }
    }
    b as u32
}

const PROMPTS: [&str; 10] = [
    "Write a Rust function that parses a semantic version string like \"1.2.3-beta\" into a struct, returning a Result.",
    "Implement an LRU cache in Python with O(1) get and put, using a dict and a doubly linked list.",
    "Write a TypeScript React hook useDebounce(value, delay) and show how to use it in a search box.",
    "Explain and implement binary search on a sorted array in C, handling the overflow in the midpoint.",
    "Write a SQL query that returns the top 3 customers by total order amount per country, using window functions.",
    "Write a bash script that finds all files larger than 100MB under a directory and prints them sorted by size.",
    "Implement a thread-safe bounded queue in Go with Push and Pop that block when full or empty.",
    "Write a Python function that merges overlapping intervals, with a few unit tests using pytest.",
    "Refactor this JavaScript to async/await: fetch(url).then(r => r.json()).then(d => console.log(d)).catch(e => console.error(e));",
    "Write a Rust iterator adapter that yields consecutive pairs (windows of 2) from any iterator, generic over the item type.",
];

struct Drafter {
    name: String,
    layers: Vec<usize>,
    bits: u32,
    own_kv: bool,
}

fn main() {
    let dir = std::env::var("QWEN_DIR").expect("QWEN_DIR");
    let args: Vec<String> = std::env::args().collect();
    let gen: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(192);
    let nprompts: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(10);
    let stride: usize = std::env::var("STRIDE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let only: Option<String> = std::env::var("DRAFTERS").ok();
    let t0 = Instant::now();
    let st = St::open(&format!("{dir}/model.safetensors"));
    let tok = tokenizers::Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json")).unwrap())
            .unwrap();
    let cu = |k: &str| cfg[k].as_u64().unwrap() as usize;
    let nl = cu("num_hidden_layers");
    let lay = |i: usize, p: &str| format!("model.layers.{i}.{p}");
    let model = Model {
        d: cu("hidden_size"),
        heads: cu("num_attention_heads"),
        kvh: cu("num_key_value_heads"),
        hd: cu("head_dim"),
        eps: 1e-6,
        theta: 1e6,
        embed: QMat::load(&st, "model.embed_tokens"),
        head: QMat::load(&st, "lm_head"),
        norm: bf16(st.raw("model.norm.weight")),
        norms: (0..nl)
            .map(|i| Norms {
                input: bf16(st.raw(&lay(i, "input_layernorm.weight"))),
                post: bf16(st.raw(&lay(i, "post_attention_layernorm.weight"))),
                qn: bf16(st.raw(&lay(i, "self_attn.q_norm.weight"))),
                kn: bf16(st.raw(&lay(i, "self_attn.k_norm.weight"))),
            })
            .collect(),
        layers: (0..nl)
            .map(|i| Layer {
                q: QMat::load(&st, &lay(i, "self_attn.q_proj")),
                k: QMat::load(&st, &lay(i, "self_attn.k_proj")),
                v: QMat::load(&st, &lay(i, "self_attn.v_proj")),
                o: QMat::load(&st, &lay(i, "self_attn.o_proj")),
                gate: QMat::load(&st, &lay(i, "mlp.gate_proj")),
                up: QMat::load(&st, &lay(i, "mlp.up_proj")),
                down: QMat::load(&st, &lay(i, "mlp.down_proj")),
            })
            .collect(),
    };
    drop(st);
    if std::env::var("BENCH").is_ok() {
        let x = vec![0.01f32; 4096];
        for _ in 0..3 {
            let tb = Instant::now();
            for l in &model.layers {
                std::hint::black_box(l.gate.matmul(&x, 1));
            }
            eprintln!("36 gate matmuls: {:.3}s", tb.elapsed().as_secs_f64());
        }
        return;
    }
    let all: Vec<usize> = (0..nl).collect();
    let r = |a: usize, b: usize| (a..b).collect::<Vec<_>>();
    let mut drafters = vec![
        Drafter {
            name: "full-sanity".into(),
            layers: all.clone(),
            bits: 4,
            own_kv: false,
        },
        Drafter {
            name: format!("exit-L{}", nl / 2),
            layers: r(0, nl / 2),
            bits: 4,
            own_kv: false,
        },
        Drafter {
            name: format!("exit-L{}", nl * 2 / 3),
            layers: r(0, nl * 2 / 3),
            bits: 4,
            own_kv: false,
        },
        Drafter {
            name: format!("exit-L{}", nl * 5 / 6),
            layers: r(0, nl * 5 / 6),
            bits: 4,
            own_kv: false,
        },
        Drafter {
            name: format!("exit-L{}", nl - 3),
            layers: r(0, nl - 3),
            bits: 4,
            own_kv: false,
        },
        Drafter {
            name: "skip-odd".into(),
            layers: all.iter().cloned().filter(|i| i % 2 == 0).collect(),
            bits: 4,
            own_kv: false,
        },
        Drafter {
            // keep the first and last 4, every other layer between
            name: "skip-odd-mid".into(),
            layers: all
                .iter()
                .cloned()
                .filter(|&i| i < 4 || i >= nl - 4 || i % 2 == 0)
                .collect(),
            bits: 4,
            own_kv: false,
        },
        Drafter {
            name: "skip-every4th".into(),
            layers: all.iter().cloned().filter(|i| i % 4 != 3).collect(),
            bits: 4,
            own_kv: false,
        },
        Drafter {
            name: "q3-all".into(),
            layers: all.clone(),
            bits: 3,
            own_kv: false,
        },
        Drafter {
            name: "q2-all".into(),
            layers: all.clone(),
            bits: 2,
            own_kv: false,
        },
        Drafter {
            name: "skip-odd-mid-own".into(),
            layers: all
                .iter()
                .cloned()
                .filter(|&i| i < 4 || i >= nl - 4 || i % 2 == 0)
                .collect(),
            bits: 4,
            own_kv: true,
        },
        Drafter {
            name: "skip-every4th-own".into(),
            layers: all.iter().cloned().filter(|i| i % 4 != 3).collect(),
            bits: 4,
            own_kv: true,
        },
    ];
    if let Some(o) = &only {
        let keep: Vec<&str> = o.split(',').collect();
        drafters.retain(|d| keep.contains(&d.name.as_str()));
    }
    let mut alts: std::collections::HashMap<u32, Vec<Layer>> = Default::default();
    for d in &drafters {
        if d.bits != 4 && !alts.contains_key(&d.bits) {
            alts.insert(
                d.bits,
                model.layers.iter().map(|l| l.requant(d.bits)).collect(),
            );
        }
    }
    let layer_bytes: Vec<f64> = model.layers.iter().map(Layer::bytes).collect();
    let full_bytes: f64 = layer_bytes.iter().sum::<f64>() + model.head.bytes();
    eprintln!(
        "loaded in {:.0}s; full weight read {:.2} GB",
        t0.elapsed().as_secs_f64(),
        full_bytes / 1e9
    );
    for d in &drafters {
        let lb: f64 = d
            .layers
            .iter()
            .map(|&i| alts.get(&d.bits).map_or(layer_bytes[i], |a| a[i].bytes()))
            .sum();
        let cost = (lb + model.head.bytes()) / full_bytes;
        eprintln!(
            "drafter {} layers={} bits={} cost={cost:.3}",
            d.name,
            d.layers.len(),
            d.bits
        );
        println!(
            "{{\"drafter\":\"{}\",\"layers\":{},\"bits\":{},\"own_kv\":{},\"cost\":{cost:.4}}}",
            d.name,
            d.layers.len(),
            d.bits,
            d.own_kv
        );
    }

    let eos = [151645u32, 151643];
    for (pi, prompt) in PROMPTS.iter().take(nprompts).enumerate() {
        let text = format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let ptoks = tok.encode(text, false).unwrap().get_ids().to_vec();
        let p = ptoks.len();
        let full = Plan {
            layers: &all,
            alt: None,
        };
        let mut cache = Kv::new(nl);
        let tg = Instant::now();
        let mut x = ptoks.clone();
        let mut next = argmax(&model.forward(&full, &ptoks, None, &mut cache));
        while x.len() < p + gen {
            x.push(next);
            if eos.contains(&next) || x.len() == p + gen {
                break;
            }
            next = argmax(&model.forward(&full, &[next], None, &mut cache));
        }
        let n = x.len();
        eprintln!(
            "prompt {pi}: {p} + {} tokens in {:.0}s ({:.2} tok/s decode)\n---\n{}\n---",
            n - p,
            tg.elapsed().as_secs_f64(),
            (n - p) as f64 / tg.elapsed().as_secs_f64(),
            tok.decode(&x[p..], false).unwrap()
        );
        // cache holds positions 0..n-1 (the last token was never fed)
        for d in &drafters {
            let td = Instant::now();
            let plan = Plan {
                layers: &d.layers,
                alt: alts.get(&d.bits).map(|v| v.as_slice()),
            };
            let own;
            let pre = if d.own_kv {
                let mut c = Kv::new(nl);
                model.forward(&plan, &x[..n - 1], None, &mut c);
                own = c;
                &own
            } else {
                &cache
            };
            // windows start at t = p..n-1: x[..=t] verified, drafts x[t+1..]
            let ts: Vec<usize> = if d.name == "full-sanity" {
                (p..n - 1).take(if pi == 0 { 12 } else { 0 }).collect()
            } else {
                (p..n - 1).step_by(stride).collect()
            };
            let mut scratch = Kv::new(nl);
            let mut matches = Vec::with_capacity(ts.len());
            for &t in &ts {
                scratch.clear();
                let cap = KMAX.min(n - 1 - t);
                let mut cur = x[t];
                let mut m = 0;
                while m < cap {
                    let dtok = argmax(&model.forward(&plan, &[cur], Some((pre, t)), &mut scratch));
                    if dtok != x[t + 1 + m] {
                        break;
                    }
                    m += 1;
                    cur = dtok;
                }
                matches.push(m);
            }
            let mean = matches.iter().sum::<usize>() as f64 / matches.len().max(1) as f64;
            eprintln!(
                "  {} : {} windows, mean match {mean:.2} ({:.0}s)",
                d.name,
                matches.len(),
                td.elapsed().as_secs_f64()
            );
            println!(
                "{{\"prompt\":{pi},\"drafter\":\"{}\",\"p\":{p},\"n\":{n},\"match\":{:?}}}",
                d.name, matches
            );
        }
    }
}
