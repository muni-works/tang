//! The Coder engine's MTP sidecar contract (`scripts/qwen3next_mtp_sidecar.py`), checked on the
//! host: every `mtp.*` tensor the engine loads (`Coder::load`, `load_layer(.., "mtp.", ..)`),
//! with the shape the trunk's hyperparameters require. The engine checks a sidecar with it
//! before uploading; `flash-mtp-train` checks its exports with it.
use super::reference::Hparams;
use crate::gguf::Gguf;
use anyhow::{ensure, Context, Result};
use std::path::Path;

/// `(name, rows, row length)` of every tensor the engine reads from a sidecar; experts are
/// `n_expert` stacked matrices of that shape. Rows 1 means a vector.
pub fn sidecar_tensors(hp: &Hparams) -> Vec<(&'static str, usize, usize, usize)> {
    let (n, hd, nh, nkv, ff, ne) = (
        hp.n_embd,
        hp.head_dim,
        hp.n_head,
        hp.n_head_kv,
        hp.n_ff_exp,
        hp.n_expert,
    );
    vec![
        ("mtp.fc.weight", n, 2 * n, 1),
        ("mtp.enorm.weight", 1, n, 1),
        ("mtp.hnorm.weight", 1, n, 1),
        ("mtp.norm.weight", 1, n, 1),
        ("mtp.attn_norm.weight", 1, n, 1),
        ("mtp.post_attention_norm.weight", 1, n, 1),
        ("mtp.attn_q.weight", 2 * nh * hd, n, 1),
        ("mtp.attn_k.weight", nkv * hd, n, 1),
        ("mtp.attn_v.weight", nkv * hd, n, 1),
        ("mtp.attn_output.weight", n, nh * hd, 1),
        ("mtp.attn_q_norm.weight", 1, hd, 1),
        ("mtp.attn_k_norm.weight", 1, hd, 1),
        ("mtp.ffn_gate_inp.weight", ne, n, 1),
        ("mtp.ffn_gate_inp_shexp.weight", 1, n, 1),
        ("mtp.ffn_gate_shexp.weight", ff, n, 1),
        ("mtp.ffn_up_shexp.weight", ff, n, 1),
        ("mtp.ffn_down_shexp.weight", n, ff, 1),
        ("mtp.ffn_gate_exps.weight", ff, n, ne),
        ("mtp.ffn_up_exps.weight", ff, n, ne),
        ("mtp.ffn_down_exps.weight", n, ff, ne),
    ]
}

/// Check that `g` holds a loadable MTP layer for a trunk with hyperparameters `hp`.
pub fn check_sidecar(g: &Gguf, hp: &Hparams) -> Result<()> {
    for (name, rows, k, ne) in sidecar_tensors(hp) {
        let t = g.info(name).context("MTP sidecar")?;
        ensure!(
            t.ty.can_dequantize(),
            "{name}: type {} unreadable",
            t.ty.name()
        );
        ensure!(
            t.row_len() == k && t.n_elements() as usize == rows * k * ne,
            "{name}: dims {:?}, want {ne} x {rows} x {k}",
            t.dims
        );
        ensure!(
            ne == 1 || t.dims.len() == 3,
            "{name}: not an expert stack ({:?})",
            t.dims
        );
    }
    Ok(())
}

/// `flash-coder-dump`'s MTP training rows, in `flash-mtp-train`'s format. For every row the
/// MTP layer takes (position `p`, its token, the token that followed, and `m.mh`: the final
/// hidden row after the output norm, which `cx_mtp_in` reads): positions `<= from` (the
/// prompt) go to the prefix sidecar (`--prefix-data`), positions `>= from` to the data, until
/// a row whose own token is a stop token. With each data row, the engine's own drafts
/// (`draft.u32`: position, depth, token, probability bits): depth 1 for every row, deeper
/// ones along each window's chain, for `flash-mtp-train --check-forward`.
pub struct TrainDump {
    data: [std::io::BufWriter<std::fs::File>; 3],
    prefix: [std::io::BufWriter<std::fs::File>; 2],
    from: usize,
    stop: Vec<u32>,
    closed: bool,
    next_data: usize,
    next_prefix: usize,
    /// Data rows written.
    pub rows: usize,
}

impl TrainDump {
    pub fn create(data: &Path, prefix: &Path, from: usize, stop: &[u32]) -> Result<Self> {
        let f = |d: &Path, n: &str| -> Result<std::io::BufWriter<std::fs::File>> {
            std::fs::create_dir_all(d)?;
            Ok(std::io::BufWriter::new(
                std::fs::OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(d.join(n))
                    .with_context(|| format!("{}", d.join(n).display()))?,
            ))
        };
        Ok(Self {
            data: [
                f(data, "h.f16")?,
                f(data, "ids.u32")?,
                f(data, "draft.u32")?,
            ],
            prefix: [f(prefix, "h.f16")?, f(prefix, "ids.u32")?],
            from,
            stop: stop.to_vec(),
            closed: false,
            next_data: from,
            next_prefix: 0,
            rows: 0,
        })
    }

    /// Rows at `pos0..`, tokens `toks`, next tokens `next`, hidden rows `mh` (`n` f32 each).
    /// Returns which rows went to the data.
    pub fn rows(
        &mut self,
        pos0: usize,
        toks: &[u32],
        next: &[u32],
        mh: &[f32],
        n: usize,
    ) -> Result<Vec<bool>> {
        use std::io::Write;
        let mut keep = Vec::with_capacity(next.len());
        for (i, (&tok, &nx)) in toks.iter().zip(next).enumerate() {
            let p = pos0 + i;
            let h: Vec<u8> = mh[i * n..(i + 1) * n]
                .iter()
                .flat_map(|&x| crate::gguf::f32_to_f16(x).to_le_bytes())
                .collect();
            let rec: Vec<u8> = [p as u32, tok, nx]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            if p <= self.from {
                ensure!(
                    p == self.next_prefix,
                    "dump: prompt row {p}, want {}",
                    self.next_prefix
                );
                self.next_prefix += 1;
                self.prefix[0].write_all(&h)?;
                self.prefix[1].write_all(&rec)?;
            }
            if p > self.from && self.stop.contains(&tok) {
                self.closed = true;
            }
            let k = p >= self.from && !self.closed;
            if k {
                ensure!(
                    p == self.next_data,
                    "dump: row {p}, want {}",
                    self.next_data
                );
                self.next_data += 1;
                self.data[0].write_all(&h)?;
                self.data[1].write_all(&rec)?;
                self.rows += 1;
            }
            keep.push(k);
        }
        Ok(keep)
    }

    pub fn draft(&mut self, pos: usize, depth: usize, d: (u32, f32)) -> Result<()> {
        use std::io::Write;
        for v in [pos as u32, depth as u32, d.0, d.1.to_bits()] {
            self.data[2].write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        use std::io::Write;
        for w in self.data.iter_mut().chain(self.prefix.iter_mut()) {
            w.flush()?;
            w.get_ref().sync_all()?;
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "mtp-train"))]
mod tests {
    use super::*;
    use crate::flash::mtp_train::{read_drafts, Sequence};

    #[test]
    fn dump_rows_load_as_training_data_with_prefix() {
        let root = std::env::temp_dir().join(format!("tang-coder-dump-{}", std::process::id()));
        let (data, prefix) = (root.join("data/p001"), root.join("prefix/p001"));
        let n = 2;
        // Prompt 0..5 (from = 4), generated 5.., stop token 99 at position 9.
        let toks: Vec<u32> = vec![10, 11, 12, 13, 14, 20, 21, 22, 23, 99, 30];
        let mut d = TrainDump::create(&data, &prefix, 4, &[99]).unwrap();
        let mut feed = |d: &mut TrainDump, p0: usize, len: usize| {
            let mh: Vec<f32> = (0..len * n).map(|i| (p0 * n + i) as f32).collect();
            let keep = d
                .rows(p0, &toks[p0..p0 + len], &toks[p0 + 1..p0 + len + 1], &mh, n)
                .unwrap();
            for (i, &k) in keep.iter().enumerate() {
                if k {
                    d.draft(p0 + i, 1, (toks[p0 + i + 2], 0.5)).unwrap();
                }
            }
            if *keep.last().unwrap() {
                d.draft(p0 + len - 1, 2, (7, 0.25)).unwrap();
            }
            keep
        };
        // Prefill windows of 3 + 2, then verify windows keeping 2, 1, 2 rows.
        assert_eq!(feed(&mut d, 0, 3), [false; 3]);
        assert_eq!(feed(&mut d, 3, 2), [false, true]);
        assert_eq!(feed(&mut d, 5, 2), [true, true]);
        assert_eq!(feed(&mut d, 7, 1), [true]);
        // The row whose token is the stop token, and anything after it, stays out.
        assert_eq!(feed(&mut d, 8, 2), [true, false]);
        assert!(
            d.rows(3, &[13], &[14], &[0.0; 2], n).is_err(),
            "gaps are refused"
        );
        assert_eq!(d.rows, 5);
        d.finish().unwrap();
        for dir in [&data, &prefix] {
            std::fs::write(dir.join("done"), b"").unwrap();
        }
        let mut s = Sequence::open(&data, n).unwrap();
        assert_eq!(s.ids.len(), 5);
        assert_eq!(s.ids[0], [4, 14, 20]);
        assert_eq!(s.ids[4], [8, 23, 99]);
        s.attach_prefix(&root.join("prefix")).unwrap();
        let b = s.batch(1, 4).unwrap();
        let p = b.prefix.unwrap();
        assert_eq!(p.pos, (0..5).collect::<Vec<_>>());
        assert_eq!(p.next_tokens, [11, 12, 13, 14, 20]);
        assert_eq!(b.tokens, [20, 21, 22, 23, 99]);
        assert_eq!(b.h[..2], [10.0, 11.0]);
        let drafts = read_drafts(&data).unwrap();
        assert_eq!(drafts[&(4, 1)], (21, 0.5));
        assert_eq!(drafts[&(7, 2)], (7, 0.25));
        assert_eq!(drafts.len(), 5 + 3);
        std::fs::remove_dir_all(&root).ok();
    }
}
