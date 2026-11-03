use super::{
    data::Batch,
    tape::{Id, Tape},
    weights::{Kind, Weights},
};
use crate::flash::reference::Hparams;
use anyhow::{ensure, Result};
use std::collections::BTreeMap;
use tang_compute::ComputeDevice;

pub struct CellOut {
    pub r: Id,
    pub k: Id,
    pub v: Id,
    pub logits: Id,
}
fn linear<D: ComputeDevice>(
    t: &mut Tape<D>,
    w: &Weights,
    ids: &BTreeMap<String, Id>,
    x: Id,
    name: &str,
    rows: usize,
) -> Id {
    let mat = &w.dense[name];
    t.linear(x, ids[name], rows, mat.cols, mat.rows)
}
fn hc_read<D: ComputeDevice>(
    t: &mut Tape<D>,
    w: &Weights,
    ids: &BTreeMap<String, Id>,
    r: Id,
    rows: usize,
    pre: &str,
    hp: &Hparams,
    inject: bool,
) -> (Id, Option<Id>) {
    let n = hp.n_embd;
    let hc = hp.hc;
    let xn = t.norm(r, ids[&format!("{pre}norm.weight")], n, hp.eps);
    let lo = linear(t, w, ids, xn, &format!("{pre}down.weight"), rows);
    let lo = t.scale(lo, 1.0 / hc as f32);
    let lo = t.nonlinear(lo, true);
    let gate = linear(t, w, ids, lo, &format!("{pre}up.weight"), rows);
    let gate = t.nonlinear(gate, false);
    let mixed = t.mul(xn, gate);
    let mut streams = Vec::new();
    for c in 0..hc {
        streams.push(
            t.gather(
                mixed,
                (0..rows)
                    .flat_map(|r| (0..n).map(move |j| (r * hc + c) * n + j))
                    .collect(),
            ),
        );
    }
    let mut x = streams[0];
    for &s in &streams[1..] {
        x = t.add(x, s);
    }
    let x = t.scale(x, 1.0 / hc as f32);
    let inj = inject.then(|| linear(t, w, ids, xn, &format!("{pre}inject.weight"), rows));
    (x, inj)
}
fn hc_write<D: ComputeDevice>(
    t: &mut Tape<D>,
    r: Id,
    y: Id,
    inj: Id,
    rows: usize,
    hp: &Hparams,
) -> Id {
    let g = t.scale(inj, 1.0 / hp.hc as f32);
    let g = t.nonlinear(g, false);
    let g = t.scale(g, 2.0);
    let g = t.gather(
        g,
        (0..rows * hp.hc)
            .flat_map(|i| std::iter::repeat_n(i, hp.n_embd))
            .collect(),
    );
    let y = t.gather(
        y,
        (0..rows)
            .flat_map(|r| {
                (0..hp.hc).flat_map(move |_| (0..hp.n_embd).map(move |j| r * hp.n_embd + j))
            })
            .collect(),
    );
    let y = t.mul(y, g);
    t.add(r, y)
}
fn moe<D: ComputeDevice>(
    t: &mut Tape<D>,
    w: &mut Weights,
    ids: &BTreeMap<String, Id>,
    x: Id,
    rows: usize,
    hp: &Hparams,
) -> Result<Id> {
    let (n, ff, k, ne) = (hp.n_embd, hp.n_ff_exp, hp.n_expert_used, hp.n_expert);
    let logits = linear(t, w, ids, x, "ffn_gate_inp.weight", rows);
    let mut selected = Vec::new();
    let mut groups: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for r in 0..rows {
        let lg = &t.data(logits)[r * ne..(r + 1) * ne];
        let mut idx: Vec<_> = (0..ne).collect();
        idx.sort_by(|&a, &b| lg[b].total_cmp(&lg[a]).then(a.cmp(&b)));
        idx.truncate(k);
        for (rank, &e) in idx.iter().enumerate() {
            selected.push(r * ne + e);
            groups.entry(e).or_default().push((r, rank));
        }
    }
    let route = t.gather(logits, selected);
    let route = t.softmax(route, k);
    let route = t.scale(
        route,
        if hp.expert_weights_scale == 0.0 {
            1.0
        } else {
            hp.expert_weights_scale
        },
    );
    let mut y = t.constant(vec![0.0; rows * n]);
    for (e, pairs) in groups {
        let xe = t.gather(
            x,
            pairs
                .iter()
                .flat_map(|&(r, _)| (0..n).map(move |j| r * n + j))
                .collect(),
        );
        let gate = t.leaf(w.expert("ffn_gate_exps.weight", e)?, false);
        let up = t.leaf(w.expert("ffn_up_exps.weight", e)?, false);
        let down = t.leaf(w.expert("ffn_down_exps.weight", e)?, false);
        let gv = t.linear(xe, gate, pairs.len(), n, ff);
        let gv = t.nonlinear(gv, true);
        let uv = t.linear(xe, up, pairs.len(), n, ff);
        let h = t.mul(gv, uv);
        let ye = t.linear(h, down, pairs.len(), ff, n);
        let rw = t.gather(
            route,
            pairs
                .iter()
                .flat_map(|&(r, rank)| std::iter::repeat_n(r * k + rank, n))
                .collect(),
        );
        let ye = t.mul(ye, rw);
        let zero = t.constant(vec![0.0; n]);
        let padded = t.join(&[ye, zero]);
        let mut inverse = vec![pairs.len(); rows];
        for (j, &(r, _)) in pairs.iter().enumerate() {
            inverse[r] = j;
        }
        let ye = t.gather(
            padded,
            inverse
                .into_iter()
                .flat_map(|j| (0..n).map(move |d| j * n + d))
                .collect(),
        );
        y = t.add(y, ye);
    }
    let gate = linear(t, w, ids, x, "ffn_gate_shexp.weight", rows);
    let gate = t.nonlinear(gate, true);
    let up = linear(t, w, ids, x, "ffn_up_shexp.weight", rows);
    let h = t.mul(gate, up);
    let sh = linear(t, w, ids, h, "ffn_down_shexp.weight", rows);
    let sg = linear(t, w, ids, x, "ffn_gate_inp_shexp.weight", rows);
    let sg = t.nonlinear(sg, false);
    let sg = t.gather(
        sg,
        (0..rows).flat_map(|r| std::iter::repeat_n(r, n)).collect(),
    );
    let sh = t.mul(sh, sg);
    Ok(t.add(y, sh))
}
fn prepare<D: ComputeDevice>(
    t: &mut Tape<D>,
    w: &Weights,
    ids: &BTreeMap<String, Id>,
    h: Id,
    tokens: &[u32],
    hp: &Hparams,
) -> Result<(Id, Id, Option<Id>)> {
    if w.kind == Kind::Coder {
        return prepare_coder(t, w, ids, h, tokens, hp);
    }
    let (n, hc, rows) = (hp.n_embd, hp.hc, tokens.len());
    let ti = w.main.info("token_embd.weight")?;
    let mut emb = Vec::with_capacity(rows * n);
    for &tok in tokens {
        emb.extend(w.main.rows(ti, tok as usize, 1)?);
    }
    let e = t.constant(emb);
    let e = t.norm(e, ids["nextn.enorm.weight"], n, hp.eps);
    let hn = t.norm(h, ids["nextn.hnorm.weight"], n, hp.eps);
    let packed = t.join(&[e, hn]);
    let mut cat = Vec::with_capacity(rows * hc * 2 * n);
    for r in 0..rows {
        for c in 0..hc {
            cat.extend((0..n).map(|j| r * n + j));
            cat.extend((0..n).map(|j| rows * n + (r * hc + c) * n + j));
        }
    }
    let cat = t.gather(packed, cat);
    let r = linear(t, w, ids, cat, "nextn.eh_proj.weight", rows * hc);
    let (x, inj) = hc_read(t, w, ids, r, rows, "hc_attn_", hp, true);
    Ok((r, x, inj))
}
fn embed(w: &Weights, tokens: &[u32]) -> Result<Vec<f32>> {
    let ti = w.main.info("token_embd.weight")?;
    let mut emb = Vec::with_capacity(tokens.len() * ti.row_len());
    for &tok in tokens {
        emb.extend(w.main.rows(ti, tok as usize, 1)?);
    }
    Ok(emb)
}
/// Qwen3-Next: `r = fc([rmsnorm(emb)·enorm | rmsnorm(h)·hnorm])` (`cx_mtp_in` + `mtp_fc`),
/// `x = rmsnorm(r)·attn_norm`.
fn prepare_coder<D: ComputeDevice>(
    t: &mut Tape<D>,
    w: &Weights,
    ids: &BTreeMap<String, Id>,
    h: Id,
    tokens: &[u32],
    hp: &Hparams,
) -> Result<(Id, Id, Option<Id>)> {
    let (n, rows) = (hp.n_embd, tokens.len());
    let e = t.constant(embed(w, tokens)?);
    let e = t.norm(e, ids["enorm.weight"], n, hp.eps);
    let hn = t.norm(h, ids["hnorm.weight"], n, hp.eps);
    let packed = t.join(&[e, hn]);
    let cat = (0..rows)
        .flat_map(|r| {
            (0..n)
                .map(move |j| r * n + j)
                .chain((0..n).map(move |j| (rows + r) * n + j))
        })
        .collect();
    let cat = t.gather(packed, cat);
    let r = linear(t, w, ids, cat, "fc.weight", rows);
    let x = t.norm(r, ids["attn_norm.weight"], n, hp.eps);
    Ok((r, x, None))
}
fn teacher_kv<D: ComputeDevice>(
    t: &mut Tape<D>,
    w: &Weights,
    ids: &BTreeMap<String, Id>,
    x: Id,
    pos: &[usize],
    hp: &Hparams,
) -> (Id, Id) {
    let (rows, hd, nkv) = (pos.len(), hp.head_dim, hp.n_head_kv);
    let k = linear(t, w, ids, x, "attn_k.weight", rows);
    let k = t.norm(k, ids["attn_k_norm.weight"], hd, hp.eps);
    let k = t.rope(k, nkv, hd, hp.n_rot, hp.rope_base, pos);
    let v = linear(t, w, ids, x, "attn_v.weight", rows);
    (k, v)
}
fn visibility(prefix: usize, counts: &[usize], rows: usize) -> Vec<Vec<usize>> {
    (0..rows)
        .map(|i| {
            let mut mask: Vec<_> = (0..prefix + i + 1).collect();
            if !counts.is_empty() {
                let mut offset = prefix + counts[0];
                for &count in &counts[1..] {
                    mask.push(offset + i);
                    offset += count;
                }
                mask.push(offset + i);
            }
            mask
        })
        .collect()
}
/// Gated attention of one recursive depth: q|gate per head, q/k rmsnorm, partial NeoX RoPE,
/// `sigmoid(gate) · attn → attn_output`. Returns (output, k, v).
#[allow(clippy::too_many_arguments)]
fn attend<D: ComputeDevice>(
    t: &mut Tape<D>,
    w: &Weights,
    ids: &BTreeMap<String, Id>,
    x: Id,
    pos: &[usize],
    previous: &[(Id, Id)],
    visible: Vec<Vec<usize>>,
    hp: &Hparams,
) -> (Id, Id, Id) {
    let rows = pos.len();
    let qf = linear(t, w, ids, x, "attn_q.weight", rows);
    let nh = hp.n_head;
    let nkv = hp.n_head_kv;
    let hd = hp.head_dim;
    let q = t.gather(
        qf,
        (0..rows * nh)
            .flat_map(|i| (0..hd).map(move |j| i * 2 * hd + j))
            .collect(),
    );
    let gate = t.gather(
        qf,
        (0..rows * nh)
            .flat_map(|i| (0..hd).map(move |j| i * 2 * hd + hd + j))
            .collect(),
    );
    let q = t.norm(q, ids["attn_q_norm.weight"], hd, hp.eps);
    let q = t.rope(q, nh, hd, hp.n_rot, hp.rope_base, pos);
    let (k, v) = teacher_kv(t, w, ids, x, pos, hp);
    let ks: Vec<_> = previous
        .iter()
        .map(|p| p.0)
        .chain(std::iter::once(k))
        .collect();
    let vs: Vec<_> = previous
        .iter()
        .map(|p| p.1)
        .chain(std::iter::once(v))
        .collect();
    let allk = t.join(&ks);
    let allv = t.join(&vs);
    let a = t.attention(q, allk, allv, nh, nkv, hd, visible);
    let gate = t.nonlinear(gate, false);
    let a = t.mul(a, gate);
    let a = linear(t, w, ids, a, "attn_output.weight", rows);
    (a, k, v)
}
#[allow(clippy::too_many_arguments)]
fn cell<D: ComputeDevice>(
    t: &mut Tape<D>,
    w: &mut Weights,
    ids: &BTreeMap<String, Id>,
    h: Id,
    tokens: &[u32],
    pos: &[usize],
    previous: &[(Id, Id)],
    visible: Vec<Vec<usize>>,
    hp: &Hparams,
) -> Result<CellOut> {
    let rows = tokens.len();
    let (r, x, inj) = prepare(t, w, ids, h, tokens, hp)?;
    let (a, k, v) = attend(t, w, ids, x, pos, previous, visible, hp);
    if w.kind == Kind::Coder {
        // Plain pre-norm residual; the next depth (and the head) take `rmsnorm(r)·norm`, the
        // rows the engine's chain gathers from `s.x`.
        let n = hp.n_embd;
        let r = t.add(r, a);
        let x = t.norm(r, ids["post_attention_norm.weight"], n, hp.eps);
        let y = moe(t, w, ids, x, rows, hp)?;
        let r = t.add(r, y);
        let out = t.norm(r, ids["norm.weight"], n, hp.eps);
        let logits = linear(t, w, ids, out, "output.weight", rows);
        return Ok(CellOut {
            r: out,
            k,
            v,
            logits,
        });
    }
    let r = hc_write(t, r, a, inj.unwrap(), rows, hp);
    let (x, inj) = hc_read(t, w, ids, r, rows, "hc_ffn_", hp, true);
    let y = moe(t, w, ids, x, rows, hp)?;
    let r = hc_write(t, r, y, inj.unwrap(), rows, hp);
    let (x, _) = hc_read(t, w, ids, r, rows, "nextn.hc_head_", hp, false);
    let logits = linear(t, w, ids, x, "output.weight", rows);
    Ok(CellOut { r, k, v, logits })
}
/// Depth k uses MTP residuals from k-1 and teacher token i+k. Attention sees teacher cells
/// through i and only its own preceding recursive cells, exactly the inference chain mask.
pub fn forward_full<D: ComputeDevice>(
    t: &mut Tape<D>,
    w: &mut Weights,
    ids: &BTreeMap<String, Id>,
    b: &Batch,
    hp: &Hparams,
    depths: usize,
) -> Result<Vec<CellOut>> {
    let len = b.pos.len();
    ensure!(
        depths >= 1 && len > depths,
        "{depths}-step training needs at least {} residual rows",
        depths + 1
    );
    let off = w.kind.pos_offset();
    let width = hp.n_embd * hp.hc;
    let mut residual = t.constant(b.h.clone());
    let mut prev = Vec::new();
    let prefix = if let Some(p) = &b.prefix {
        let h = t.constant(p.h.clone());
        let (_, x, _) = prepare(t, w, ids, h, &p.next_tokens, hp)?;
        let pos: Vec<_> = p.pos.iter().map(|&q| q + off).collect();
        prev.push(teacher_kv(t, w, ids, x, &pos, hp));
        p.pos.len()
    } else {
        0
    };
    let mut logits = Vec::new();
    let mut counts = Vec::new();
    for depth in 1..=depths {
        let rows = len - depth;
        let h = t.gather(residual, (0..rows * width).collect());
        let tokens = &b.tokens[depth..depth + rows];
        let pos: Vec<_> = b.pos[..rows].iter().map(|&p| p + depth - 1 + off).collect();
        let visible = visibility(prefix, &counts, rows);
        let out = cell(t, w, ids, h, tokens, &pos, &prev, visible, hp)?;
        residual = out.r;
        prev.push((out.k, out.v));
        logits.push(out);
        counts.push(rows);
    }
    Ok(logits)
}
/// Loss recipe on top of the recursive teacher-forced chain.
#[derive(Clone, Copy, Debug)]
pub struct Recipe {
    /// Per-depth decay `β^(d-1)` (FastMTP: 0.6).
    pub beta: f64,
    /// Weight of the soft term: CE against the target's own next-token distribution
    /// `softmax(output · h)` (forward KL up to a constant, Soft-MTP / VAT ℓ^soft). 0 = off.
    pub soft: f64,
    /// Verification-aware weights (VAT): per window, weight 1 at every depth up to and
    /// including the first simulated rejection (argmax ≠ teacher token), then the decay
    /// restarts: `β^(d-k*)` after it. Off: plain `β^(d-1)`.
    pub vat: bool,
}
impl Recipe {
    pub fn plain(beta: f64) -> Self {
        Recipe {
            beta,
            soft: 0.0,
            vat: false,
        }
    }
}
pub struct Objective {
    /// Hard CE per depth (teacher token), comparable across recipes.
    pub ce: Vec<f64>,
    /// KL(target ‖ head) per depth when target logits are given.
    pub kl: Vec<f64>,
    pub seeds: Vec<(Id, Vec<f32>)>,
    pub hits: Vec<usize>,
    pub totals: Vec<usize>,
    /// Windows whose argmax matched the teacher at every depth 1..=d (simulated acceptance).
    pub accepted: Vec<usize>,
    /// Target logits whose argmax equals the dumped teacher token (sanity of `output · h`).
    pub teacher_agree: usize,
}
/// Target logits for the rows the chain is labelled with: `output · h` over residual rows
/// 1..len (depth d, row r is labelled by token r+d+2, the target's argmax at row r+d+1).
pub fn teacher_logits<D: ComputeDevice>(
    t: &mut Tape<D>,
    w: &Weights,
    ids: &BTreeMap<String, Id>,
    b: &Batch,
    hp: &Hparams,
) -> Result<Id> {
    ensure!(
        w.kind == Kind::Coder && hp.hc == 1,
        "target logits need the Coder sidecar (h is the trunk's normed final row)"
    );
    let n = hp.n_embd;
    let rows = b.pos.len() - 1;
    let h = t.constant(b.h[n..(rows + 1) * n].to_vec());
    Ok(linear(t, w, ids, h, "output.weight", rows))
}
pub fn objective<D: ComputeDevice>(
    t: &Tape<D>,
    w: &Weights,
    logits: &[Id],
    b: &Batch,
    beta: f64,
    burn: usize,
) -> Result<(Vec<f64>, Vec<(Id, Vec<f32>)>, Vec<usize>, Vec<usize>)> {
    let o = objective_with(t, w, logits, None, b, Recipe::plain(beta), burn)?;
    Ok((o.ce, o.seeds, o.hits, o.totals))
}
fn softmax(row: &[f32]) -> (f64, f64) {
    let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z = row.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>();
    (mx, z)
}
fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
        .unwrap()
        .0
}
pub fn objective_with<D: ComputeDevice>(
    t: &Tape<D>,
    w: &Weights,
    logits: &[Id],
    teacher: Option<Id>,
    b: &Batch,
    recipe: Recipe,
    burn: usize,
) -> Result<Objective> {
    let depths = logits.len();
    let beta = recipe.beta;
    let vocab = w.vocab.len();
    ensure!(
        recipe.soft == 0.0 || teacher.is_some(),
        "soft loss needs target logits"
    );
    let rows0 = t.data(logits[0]).len() / vocab;
    ensure!(burn < rows0 - (depths - 1), "burn-in leaves no loss rows");
    let mut labels = vec![vec![0usize; rows0]; depths];
    let mut best = vec![vec![0usize; rows0]; depths];
    for (d, &id) in logits.iter().enumerate() {
        let rows = rows0 - d;
        for r in burn..rows {
            let label = b.tokens[r + d + 2];
            labels[d][r] = w.vocab.binary_search(&label).map_err(|_| {
                anyhow::anyhow!("label {label} excluded from vocabulary; use --vocab 0")
            })?;
            best[d][r] = argmax(&t.data(id)[r * vocab..(r + 1) * vocab]);
        }
    }
    // Per window (row r), per depth: the loss weight before normalisation.
    let mut weights = vec![vec![0.0f64; rows0]; depths];
    for r in burn..rows0 {
        let mut rejected: Option<usize> = None;
        for d in 0..depths {
            if r >= rows0 - d {
                break;
            }
            weights[d][r] = match (recipe.vat, rejected) {
                (true, None) => 1.0,
                (true, Some(k)) => beta.powi((d - k) as i32),
                (false, _) => beta.powi(d as i32),
            };
            if rejected.is_none() && best[d][r] != labels[d][r] {
                rejected = Some(d);
            }
        }
    }
    let total_weight: f64 = weights.iter().flatten().sum();
    let mut o = Objective {
        ce: vec![0.0; depths],
        kl: vec![0.0; depths],
        seeds: Vec::new(),
        hits: vec![0; depths],
        totals: vec![0; depths],
        accepted: vec![0; depths],
        teacher_agree: 0,
    };
    let tl = teacher.map(|id| t.data(id));
    if let Some(tl) = tl {
        for r in 0..rows0 {
            let row = &tl[r * vocab..(r + 1) * vocab];
            if r + 2 < b.tokens.len() {
                let lab = w.vocab.binary_search(&b.tokens[r + 2]).ok();
                o.teacher_agree += usize::from(lab == Some(argmax(row)));
            }
        }
    }
    for (d, &id) in logits.iter().enumerate() {
        let rows = rows0 - d;
        let mut g = vec![0.0; rows * vocab];
        let mut q = vec![0.0f64; vocab];
        for r in burn..rows {
            let target = labels[d][r];
            let row = &t.data(id)[r * vocab..(r + 1) * vocab];
            let (mx, z) = softmax(row);
            let lse = mx + z.ln();
            o.ce[d] += lse - row[target] as f64;
            o.hits[d] += usize::from(best[d][r] == target);
            o.totals[d] += 1;
            o.accepted[d] += usize::from((0..=d).all(|e| best[e][r] == labels[e][r]));
            if let Some(tl) = tl {
                // Teacher row r+d+1 of the residual; the target tensor starts at row 1.
                let trow = &tl[(r + d) * vocab..(r + d + 1) * vocab];
                let (tm, tz) = softmax(trow);
                let mut kl = 0.0;
                for j in 0..vocab {
                    q[j] = (trow[j] as f64 - tm).exp() / tz;
                    if q[j] > 0.0 {
                        kl += q[j] * (q[j].ln() - (row[j] as f64 - lse));
                    }
                }
                o.kl[d] += kl;
            }
            let weight = weights[d][r] / total_weight;
            for j in 0..vocab {
                let p = (row[j] as f64 - mx).exp() / z;
                let mut gj = p - f64::from(j == target);
                if tl.is_some() && recipe.soft != 0.0 {
                    gj += recipe.soft * (p - q[j]);
                }
                g[r * vocab + j] = (gj * weight) as f32;
            }
        }
        o.ce[d] /= o.totals[d] as f64;
        o.kl[d] /= o.totals[d] as f64;
        o.seeds.push((id, g));
    }
    Ok(o)
}

pub fn forward<D: ComputeDevice>(
    t: &mut Tape<D>,
    w: &mut Weights,
    ids: &BTreeMap<String, Id>,
    b: &Batch,
    hp: &Hparams,
    depths: usize,
) -> Result<Vec<Id>> {
    Ok(forward_full(t, w, ids, b, hp, depths)?
        .iter()
        .map(|o| o.logits)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::visibility;
    #[test]
    fn prefix_mask_excludes_future_teacher_and_other_recursive_chains() {
        assert_eq!(
            visibility(3, &[], 2),
            vec![vec![0, 1, 2, 3], vec![0, 1, 2, 3, 4]]
        );
        assert_eq!(
            visibility(3, &[5], 2),
            vec![vec![0, 1, 2, 3, 8], vec![0, 1, 2, 3, 4, 9]]
        );
        assert_eq!(
            visibility(3, &[5, 4], 2),
            vec![vec![0, 1, 2, 3, 8, 12], vec![0, 1, 2, 3, 4, 9, 13]]
        );
        assert_eq!(
            visibility(0, &[5, 4], 2),
            vec![vec![0, 5, 9], vec![0, 1, 6, 10]]
        );
    }
}
