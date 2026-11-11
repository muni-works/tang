//! Per-layer dump hooks for the GLM-5.3-Flash engine (W0's R3 bisector).
//!
//! W0 owns this file; `engine.rs` (W1) calls three one-liners, each a no-op without
//! `TANG_DUMP_DIR`:
//!
//! - [`DumpSink::open`] once after `Glm5Engine::open` (env read here, not in engine.rs);
//! - [`DumpSink::layer`] after each block's second `mhc_post` (the `ffn` half), feeding
//!   that token row's post-FFN streams `[hc][n_embd]`;
//! - [`DumpSink::finish`] after the head, with the final collapsed hidden row `[n_embd]`
//!   and the logits row `[n_vocab]`.
//!
//! Output: one `.f32` per entry plus `index.json` in the same layout as the reference
//! dump (see `forward.rs`). Only the LAST position of each window is dumped — a bisect
//! run decodes token-by-token, so windows are length 1 and "last" is the whole window.
use anyhow::{Context, Result};
use std::io::Write as _;
use std::path::PathBuf;

/// Collects per-layer dumps when `TANG_DUMP_DIR` is set; no-op otherwise.
pub struct DumpSink {
    dir: PathBuf,
    entries: Vec<DumpEntry>,
    pos: usize,
}

struct DumpEntry {
    name: String,
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl DumpSink {
    /// Open from `$TANG_DUMP_DIR`; `None` (and everything is a no-op) when unset.
    pub fn open() -> Option<Self> {
        let dir = std::env::var("TANG_DUMP_DIR").ok().map(PathBuf::from)?;
        Some(Self {
            dir,
            entries: Vec::new(),
            pos: 0,
        })
    }

    pub fn enabled(&self) -> bool {
        !self.dir.as_os_str().is_empty()
    }

    /// Record one generated position's per-layer post-FFN streams.
    ///
    /// `l` is the layer index, `pos` the absolute token position, `streams` the hc×n_embd
    /// rows after the block's FFN mHC mix.
    pub fn layer(&mut self, l: usize, pos: usize, streams: &[f32]) {
        if !self.enabled() {
            return;
        }
        self.entries.push(DumpEntry {
            name: format!("L{l:02}.post_ffn"),
            shape: vec![streams.len()],
            data: streams.to_vec(),
        });
        self.pos = pos;
    }

    /// Record the post-head hidden row and the logits row, then write everything.
    pub fn finish(
        &mut self,
        tokens: &[u32],
        hidden: &[f32],
        logits: &[f32],
        _n_vocab: usize,
    ) -> Result<()> {
        if !self.enabled() {
            return Ok(());
        }
        self.entries.push(DumpEntry {
            name: "final_x".to_string(),
            shape: vec![hidden.len()],
            data: hidden.to_vec(),
        });
        self.entries.push(DumpEntry {
            name: "logits".to_string(),
            shape: vec![logits.len()],
            data: logits.to_vec(),
        });
        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("creating dump dir {}", self.dir.display()))?;
        let mut idx = Vec::with_capacity(self.entries.len());
        for e in &self.entries {
            let file = format!("{}.f32", e.name);
            let mut b = Vec::with_capacity(e.data.len() * 4);
            for v in &e.data {
                b.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(self.dir.join(&file), b)
                .with_context(|| format!("writing {}", self.dir.join(&file).display()))?;
            idx.push(serde_json::json!({
                "name": e.name, "file": file, "dtype": "f32", "shape": e.shape,
            }));
        }
        let index = serde_json::json!({
            "model": "glm5-next-engine",
            "pos": self.pos,
            "tokens": tokens,
            "layout": "little-endian raw f32 arrays; per-layer rows are the post-FFN mHC streams [hc*n_embd]",
            "tensors": idx,
        });
        let mut f = std::fs::File::create(self.dir.join("index.json"))?;
        writeln!(f, "{}", serde_json::to_string_pretty(&index)?)?;
        Ok(())
    }
}
