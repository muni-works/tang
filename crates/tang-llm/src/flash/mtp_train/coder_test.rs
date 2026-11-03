//! The Qwen3-Next (Coder) MTP cell on a tiny synthetic trunk + sidecar: the tape forward against
//! a plain loop reference of the engine's chain, gradients reach the trainable block, and the
//! same-size export reloads (engine contract and trainer) to the same forward.
use super::data::{Batch, Prefix};
use super::model;
use super::tape::Tape;
use super::weights::{encode, Kind, Weights};
use crate::flash::coder_mtp::{check_sidecar, sidecar_tensors};
use crate::flash::reference::{rope_neox, Arch, Hparams};
use crate::gguf::{GgmlType, Gguf};
use std::path::Path;
use std::sync::Arc;
use tang_compute::CpuDevice;

const N: usize = 32;
const V: usize = 40;

fn hp() -> Hparams {
    Hparams {
        arch: Arch::Qwen3Next,
        n_embd: N,
        n_layer: 1,
        n_vocab: V,
        eps: 1e-6,
        hc: 1,
        hc_lr: 0,
        n_head: 2,
        n_head_kv: 1,
        head_dim: 32,
        n_rot: 16,
        rope_base: 10000.0,
        rope_sections: [0; 4],
        idx_heads: 0,
        idx_dim: 0,
        idx_top_k: 0,
        kpool: 0,
        ssm_d_conv: 4,
        ssm_d_state: 128,
        ssm_k_heads: 1,
        ssm_v_heads: 1,
        n_expert: 4,
        n_expert_used: 2,
        n_ff_exp: 32,
        expert_weights_scale: 1.0,
        is_recurrent: vec![false],
        ple: None,
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32, bias: f32) -> Vec<f32> {
        (0..n).map(|_| bias + scale * self.next()).collect()
    }
}

/// A GGUF v3 file of `(name, dims, type, bytes)` tensors, each 32-byte aligned.
fn write_gguf(path: &Path, tensors: &[(String, Vec<u64>, GgmlType, Vec<u8>)]) {
    let mut b = Vec::new();
    let s = |b: &mut Vec<u8>, x: &str| {
        b.extend((x.len() as u64).to_le_bytes());
        b.extend(x.as_bytes());
    };
    b.extend(b"GGUF");
    b.extend(3u32.to_le_bytes());
    b.extend((tensors.len() as u64).to_le_bytes());
    b.extend(1u64.to_le_bytes());
    s(&mut b, "general.architecture");
    b.extend(8u32.to_le_bytes());
    s(&mut b, "qwen3next-mtp");
    let mut data = Vec::new();
    for (name, dims, ty, bytes) in tensors {
        s(&mut b, name);
        b.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend(d.to_le_bytes());
        }
        let id = match ty {
            GgmlType::F32 => 0u32,
            GgmlType::Q8_0 => 8,
            _ => unreachable!(),
        };
        b.extend(id.to_le_bytes());
        while data.len() % 32 != 0 {
            data.push(0);
        }
        b.extend((data.len() as u64).to_le_bytes());
        data.extend(bytes);
    }
    while b.len() % 32 != 0 {
        b.push(0);
    }
    b.extend(data);
    std::fs::write(path, b).unwrap();
}

fn build(dir: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let hp = hp();
    let mut rng = Rng(7);
    let f32b = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let main = dir.join("trunk.gguf");
    write_gguf(
        &main,
        &[
            (
                "token_embd.weight".into(),
                vec![N as u64, V as u64],
                GgmlType::F32,
                f32b(&rng.vec(N * V, 1.0, 0.0)),
            ),
            (
                "output.weight".into(),
                vec![N as u64, V as u64],
                GgmlType::F32,
                f32b(&rng.vec(N * V, 0.5, 0.0)),
            ),
        ],
    );
    let mut ts = Vec::new();
    for (name, rows, k, ne) in sidecar_tensors(&hp) {
        let numel = rows * k * ne;
        if rows == 1 {
            // Norms ~1 (as stored: 1 + w); the shared-expert gate row as is. F32, like the script.
            let bias = if name.contains("norm") { 1.0 } else { 0.0 };
            ts.push((
                name.to_string(),
                vec![k as u64],
                GgmlType::F32,
                f32b(&rng.vec(numel, 0.3, bias)),
            ));
            continue;
        }
        let v = rng.vec(numel, 0.4 / (k as f32).sqrt() * 3.0, 0.0);
        let dims = if ne > 1 {
            vec![k as u64, rows as u64, ne as u64]
        } else {
            vec![k as u64, rows as u64]
        };
        ts.push((
            name.to_string(),
            dims,
            GgmlType::Q8_0,
            encode(GgmlType::Q8_0, &v).unwrap(),
        ));
    }
    let mtp = dir.join("mtp.gguf");
    write_gguf(&mtp, &ts);
    (main, mtp)
}

// ---- a plain reference of the engine's MTP chain
struct Ref<'a> {
    g: &'a Gguf,
    main: &'a Gguf,
    hp: Hparams,
}
impl Ref<'_> {
    fn t(&self, name: &str) -> Vec<f32> {
        self.g.tensor_f32(&format!("mtp.{name}.weight")).unwrap()
    }
    fn mv(w: &[f32], x: &[f32]) -> Vec<f32> {
        w.chunks(x.len())
            .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }
    fn rms(&self, x: &[f32], w: &[f32]) -> Vec<f32> {
        let z = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + self.hp.eps).sqrt();
        x.iter().zip(w).map(|(a, b)| a * z * b).collect()
    }
    fn sig(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    /// `r = fc([enorm(emb) | hnorm(h)])`, `x = attn_norm(r)`.
    fn input(&self, h: &[f32], tok: u32) -> (Vec<f32>, Vec<f32>) {
        let e = self
            .main
            .rows(
                self.main.info("token_embd.weight").unwrap(),
                tok as usize,
                1,
            )
            .unwrap();
        let mut cat = self.rms(&e, &self.t("enorm"));
        cat.extend(self.rms(h, &self.t("hnorm")));
        let r = Self::mv(&self.t("fc"), &cat);
        let x = self.rms(&r, &self.t("attn_norm"));
        (r, x)
    }
    fn kv(&self, x: &[f32], pos: usize) -> (Vec<f32>, Vec<f32>) {
        let hd = self.hp.head_dim;
        let mut k = Self::mv(&self.t("attn_k"), x);
        let kn = self.t("attn_k_norm");
        for h in k.chunks_mut(hd) {
            let n = self.rms(h, &kn);
            h.copy_from_slice(&n);
            rope_neox(h, pos, self.hp.n_rot, self.hp.rope_base);
        }
        (k, Self::mv(&self.t("attn_v"), x))
    }
    /// The layer after its input: returns (normed output, k, v, logits).
    fn layer(
        &self,
        r: Vec<f32>,
        x: &[f32],
        pos: usize,
        cache: &[(Vec<f32>, Vec<f32>)],
    ) -> (Vec<f32>, (Vec<f32>, Vec<f32>), Vec<f32>) {
        let hp = &self.hp;
        let (nh, nkv, hd) = (hp.n_head, hp.n_head_kv, hp.head_dim);
        let qf = Self::mv(&self.t("attn_q"), x);
        let kv = self.kv(x, pos);
        let mut keys: Vec<_> = cache.to_vec();
        keys.push(kv.clone());
        let qn = self.t("attn_q_norm");
        let mut att = vec![0.0; nh * hd];
        for h in 0..nh {
            let mut q = self.rms(&qf[h * 2 * hd..h * 2 * hd + hd], &qn);
            rope_neox(&mut q, pos, hp.n_rot, hp.rope_base);
            let g = h / (nh / nkv);
            let s: Vec<f32> = keys
                .iter()
                .map(|(k, _)| {
                    q.iter()
                        .zip(&k[g * hd..(g + 1) * hd])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        / (hd as f32).sqrt()
                })
                .collect();
            let m = s.iter().copied().fold(f32::MIN, f32::max);
            let z: f32 = s.iter().map(|v| (v - m).exp()).sum();
            for (j, (_, v)) in keys.iter().enumerate() {
                let p = (s[j] - m).exp() / z;
                for d in 0..hd {
                    att[h * hd + d] += p * v[g * hd + d];
                }
            }
            for d in 0..hd {
                att[h * hd + d] *= Self::sig(qf[h * 2 * hd + hd + d]);
            }
        }
        let o = Self::mv(&self.t("attn_output"), &att);
        let r: Vec<f32> = r.iter().zip(&o).map(|(a, b)| a + b).collect();
        let x = self.rms(&r, &self.t("post_attention_norm"));
        // Qwen MoE: softmax over all experts, top-k renormalised; sigmoid-gated shared expert.
        let lg = Self::mv(&self.t("ffn_gate_inp"), &x);
        let mut idx: Vec<usize> = (0..hp.n_expert).collect();
        idx.sort_by(|&a, &b| lg[b].total_cmp(&lg[a]).then(a.cmp(&b)));
        idx.truncate(hp.n_expert_used);
        let m = lg.iter().copied().fold(f32::MIN, f32::max);
        let p: Vec<f32> = idx.iter().map(|&e| (lg[e] - m).exp()).collect();
        let ps: f32 = p.iter().sum();
        let ffn = |gate: &[f32], up: &[f32], down: &[f32]| -> Vec<f32> {
            let gv = Self::mv(gate, &x);
            let uv = Self::mv(up, &x);
            let hh: Vec<f32> = gv
                .iter()
                .zip(&uv)
                .map(|(g, u)| g * Self::sig(*g) * u)
                .collect();
            Self::mv(down, &hh)
        };
        let per = hp.n_ff_exp * N;
        let (ge, ue, de) = (
            self.t("ffn_gate_exps"),
            self.t("ffn_up_exps"),
            self.t("ffn_down_exps"),
        );
        let mut y = vec![0.0; N];
        for (j, &e) in idx.iter().enumerate() {
            let ye = ffn(
                &ge[e * per..(e + 1) * per],
                &ue[e * per..(e + 1) * per],
                &de[e * per..(e + 1) * per],
            );
            for d in 0..N {
                y[d] += p[j] / ps * ye[d];
            }
        }
        let sh = ffn(
            &self.t("ffn_gate_shexp"),
            &self.t("ffn_up_shexp"),
            &self.t("ffn_down_shexp"),
        );
        let sg = Self::sig(Self::mv(&self.t("ffn_gate_inp_shexp"), &x)[0]);
        let r: Vec<f32> = (0..N).map(|d| r[d] + y[d] + sg * sh[d]).collect();
        let out = self.rms(&r, &self.t("norm"));
        let logits = Self::mv(&self.main.tensor_f32("output.weight").unwrap(), &out);
        (out, kv, logits)
    }
    /// Per depth, per row: (normed output, logits), as the engine computes them: a teacher pass
    /// over prefix + window rows (each sees teacher cells at positions up to its own), then
    /// from each row a chain on teacher-forced tokens that sees its own earlier chain cells.
    fn chain(&self, b: &Batch, depths: usize) -> Vec<Vec<(Vec<f32>, Vec<f32>)>> {
        let mut teacher = Vec::new();
        if let Some(p) = &b.prefix {
            for (j, &pos) in p.pos.iter().enumerate() {
                let (_, x) = self.input(&p.h[j * N..(j + 1) * N], p.next_tokens[j]);
                teacher.push(self.kv(&x, pos + 1));
            }
        }
        let base = teacher.len();
        let rows = b.pos.len();
        let mut out = vec![Vec::new(); depths];
        let mut first = Vec::new();
        for i in 0..rows {
            let (r, x) = self.input(&b.h[i * N..(i + 1) * N], b.tokens[i + 1]);
            let (o, kv, lg) = self.layer(r, &x, b.pos[i] + 1, &teacher);
            teacher.push(kv);
            first.push((o, lg));
        }
        for i in 0..rows - 1 {
            let mut cache: Vec<_> = teacher[..base + i + 1].to_vec();
            let mut h = first[i].0.clone();
            out[0].push(first[i].clone());
            for d in 2..=depths {
                if i >= rows - d {
                    break;
                }
                let (r, x) = self.input(&h, b.tokens[i + d]);
                let (o, kv, lg) = self.layer(r, &x, b.pos[i] + d, &cache);
                cache.push(kv);
                h = o.clone();
                out[d - 1].push((o, lg));
            }
        }
        out
    }
}

fn batch(rng: &mut Rng, prefix: bool) -> Batch {
    let rows = 6;
    let p0 = 9;
    let tokens: Vec<u32> = (0..=rows).map(|i| (i * 7 + 3) as u32 % V as u32).collect();
    Batch {
        h: rng.vec(rows * N, 2.0, 0.0),
        tokens,
        pos: (p0..p0 + rows).collect(),
        prefix: prefix.then(|| Prefix {
            h: rng.vec(p0 * N, 2.0, 0.0),
            next_tokens: (0..p0).map(|i| (i * 5 + 1) as u32 % V as u32).collect(),
            pos: (0..p0).collect(),
        }),
    }
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let e: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
    let s: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    (e / s.max(1e-30)).sqrt()
}

fn forward(w: &mut Weights, b: &Batch, depths: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    let dev = CpuDevice::new();
    let mut t = Tape::new(&dev);
    let leaves = w.leafs(&mut t);
    let out = model::forward_full(&mut t, w, &leaves, b, &hp(), depths).unwrap();
    out.iter()
        .map(|o| (t.data(o.r).to_vec(), t.data(o.logits).to_vec()))
        .collect()
}

#[test]
fn coder_cell_matches_engine_chain_and_exports_same_size() {
    let dir = std::env::temp_dir().join(format!("tang-coder-mtp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (main, mtp) = build(&dir);
    let g = Gguf::open(&mtp).unwrap();
    check_sidecar(&g, &hp()).unwrap();
    let mut w = Weights::open(&main, &mtp, 1e-3, 0).unwrap();
    assert_eq!(w.kind, Kind::Coder);
    assert!(!w.expert_q4);
    assert!(w.params.keys().all(|n| !n.contains("_exps")));
    assert_eq!(w.params.len(), 17);
    let trunk = Gguf::open(&main).unwrap();
    let reference = Ref {
        g: &g,
        main: &trunk,
        hp: hp(),
    };
    let mut rng = Rng(99);
    for (prefix, depths) in [(false, 3), (true, 3), (true, 4)] {
        let b = batch(&mut rng, prefix);
        let got = forward(&mut w, &b, depths);
        let want = reference.chain(&b, depths);
        for d in 0..depths {
            let (r, lg): (Vec<f32>, Vec<f32>) = (
                want[d].iter().flat_map(|x| x.0.clone()).collect(),
                want[d].iter().flat_map(|x| x.1.clone()).collect(),
            );
            assert_eq!(got[d].1.len(), lg.len(), "depth {} rows", d + 1);
            let (er, el) = (rel(&got[d].0, &r), rel(&got[d].1, &lg));
            assert!(
                er < 1e-4 && el < 1e-4,
                "prefix {prefix} depth {}: {er} {el}",
                d + 1
            );
        }
    }

    // Gradients reach the dense block, router and shared expert.
    let b = batch(&mut rng, true);
    let dev = CpuDevice::new();
    let mut t = Tape::new(&dev);
    let leaves = w.leafs(&mut t);
    let lg = model::forward(&mut t, &mut w, &leaves, &b, &hp(), 3).unwrap();
    let (loss, seeds, _, totals) = model::objective(&t, &w, &lg, &b, 0.8, 0).unwrap();
    assert_eq!((loss.len(), totals), (3, vec![5, 4, 3]));
    let grads = t.backward(&seeds);
    for n in w.params.keys() {
        let g = grads[leaves[n]]
            .as_ref()
            .unwrap_or_else(|| panic!("no gradient for {n}"));
        assert!(g.iter().any(|&v| v != 0.0), "zero gradient for {n}");
    }
    drop(t);

    // Pilot-2 recipe: target logits from `output · h`, soft term and VAT weights. The soft
    // gradient is exactly p - q: with the head's own logits as the target, it vanishes.
    let mut t = Tape::new(&dev);
    let leaves = w.leafs(&mut t);
    let lg = model::forward(&mut t, &mut w, &leaves, &b, &hp(), 3).unwrap();
    let teacher = model::teacher_logits(&mut t, &w, &leaves, &b, &hp()).unwrap();
    let recipe = model::Recipe {
        beta: 0.6,
        soft: 1.0,
        vat: true,
    };
    let o = model::objective_with(&t, &w, &lg, Some(teacher), &b, recipe, 0).unwrap();
    assert_eq!(o.totals, vec![5, 4, 3]);
    assert!(o.kl.iter().all(|v| v.is_finite() && *v >= -1e-9));
    for d in 0..3 {
        assert!(o.accepted[d] <= o.hits[d]);
        if d > 0 {
            assert!(o.accepted[d] <= o.accepted[d - 1]);
        }
    }
    let grads = t.backward(&o.seeds);
    for n in w.params.keys() {
        assert!(grads[leaves[n]]
            .as_ref()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }
    let plain = model::objective_with(&t, &w, &lg, None, &b, model::Recipe::plain(0.6), 0).unwrap();
    let selfsoft = model::Recipe {
        beta: 0.6,
        soft: 1.0,
        vat: false,
    };
    // Depth 1 reads target row r for its row r: the head's own logits give KL 0.
    let own = t.data(lg[0]).to_vec();
    let own = t.constant(own);
    let o1 = model::objective_with(&t, &w, &lg[..1], Some(own), &b, selfsoft, 0).unwrap();
    assert!(o1.kl[0].abs() < 1e-6, "self-KL {}", o1.kl[0]);
    assert_eq!(o1.seeds[0].1.len(), plain.seeds[0].1.len());
    drop(t);

    // An untouched export is the original file, byte for byte.
    let same = dir.join("same.gguf");
    w.export(&same).unwrap();
    assert_eq!(std::fs::read(&same).unwrap(), std::fs::read(&mtp).unwrap());

    // A changed export: same size, the engine's contract holds, and the reloaded layer's
    // forward equals the in-memory one at the exported (requantized) values.
    let mut i = 0usize;
    for (n, p) in w.params.iter_mut() {
        for x in p.data.data_mut() {
            i += 1;
            *x += 0.01 * ((i % 7) as f32 - 3.0);
        }
        let ty = g.info(&format!("mtp.{n}")).unwrap().ty;
        let mut q = vec![0.0; p.data.numel()];
        let bytes = encode(ty, p.data.data()).unwrap();
        crate::gguf::dequantize(ty, &bytes, &mut q).unwrap();
        w.dense.get_mut(n).unwrap().value = Arc::new(q);
    }
    let changed = dir.join("changed.gguf");
    w.export(&changed).unwrap();
    assert_eq!(
        std::fs::metadata(&changed).unwrap().len(),
        std::fs::metadata(&mtp).unwrap().len()
    );
    check_sidecar(&Gguf::open(&changed).unwrap(), &hp()).unwrap();
    let mut reloaded = Weights::open(&main, &changed, 1e-3, 0).unwrap();
    let b = batch(&mut rng, true);
    let (x, y) = (forward(&mut w, &b, 3), forward(&mut reloaded, &b, 3));
    for d in 0..3 {
        assert_eq!(x[d].1, y[d].1, "depth {} logits after reload", d + 1);
    }
    std::fs::remove_dir_all(&dir).ok();
}
