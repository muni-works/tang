//! A safetensors reader that knows DeepSeek's dtypes.
//!
//! The `safetensors` crate this workspace pins (0.4) rejects a header holding `F8_E8M0`, and
//! every DeepSeek-V4.1 shard has one, so this parses the header itself: an 8-byte little-endian
//! length, a JSON map of `name -> {dtype, shape, data_offsets}`, then the bytes. Files are
//! mapped, so opening the 48 real shards costs headers only.

use anyhow::{bail, ensure, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::quant::{e4m3_to_f32, e8m0_to_f32, fp4_to_f32};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32,
    BF16,
    F8E4M3,
    F8E8M0,
    I8,
    U8,
}

impl Dtype {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "F32" => Self::F32,
            "BF16" => Self::BF16,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E8M0" => Self::F8E8M0,
            "I8" => Self::I8,
            "U8" => Self::U8,
            d => bail!("unsupported dtype {d}"),
        })
    }

    fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::BF16 => 2,
            _ => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Info {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    file: usize,
    start: usize,
    end: usize,
}

pub struct St {
    files: Vec<PathBuf>,
    maps: Vec<Mmap>,
    index: HashMap<String, Info>,
    /// Tensors copied out of the maps into anonymous memory (see [`St::pin`]).
    pinned: HashMap<String, Arc<[u8]>>,
}

/// Pinned tensors, shared by every `St` of the process over the same file (the expert-parallel
/// ranks each open the checkpoint): keyed by (file, tensor).
static PINNED: Mutex<Option<HashMap<(PathBuf, String), Arc<[u8]>>>> = Mutex::new(None);

impl St {
    /// Every `*.safetensors` in `dir`.
    pub fn open(dir: &Path) -> Result<Self> {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        files.sort();
        ensure!(
            !files.is_empty(),
            "no .safetensors files in {}",
            dir.display()
        );
        let mut maps = Vec::new();
        let mut index = HashMap::new();
        for (i, f) in files.iter().enumerate() {
            let file = std::fs::File::open(f)?;
            // Safety: checkpoint files aren't modified while we run.
            let map = unsafe { Mmap::map(&file)? };
            ensure!(map.len() >= 8, "{}: too short", f.display());
            let n = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
            let header: serde_json::Map<String, serde_json::Value> =
                serde_json::from_slice(&map[8..8 + n])
                    .with_context(|| format!("header of {}", f.display()))?;
            for (name, v) in header {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = Dtype::parse(v["dtype"].as_str().context("dtype")?)?;
                let shape: Vec<usize> = v["shape"]
                    .as_array()
                    .context("shape")?
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect();
                let offs = v["data_offsets"].as_array().context("data_offsets")?;
                let (start, end) = (
                    8 + n + offs[0].as_u64().unwrap() as usize,
                    8 + n + offs[1].as_u64().unwrap() as usize,
                );
                ensure!(
                    end - start == shape.iter().product::<usize>() * dtype.size(),
                    "{name}: {} bytes for shape {shape:?} {dtype:?}",
                    end - start
                );
                index.insert(
                    name,
                    Info {
                        dtype,
                        shape,
                        file: i,
                        start,
                        end,
                    },
                );
            }
            maps.push(map);
        }
        Ok(St {
            files,
            maps,
            index,
            pinned: HashMap::new(),
        })
    }

    /// Every tensor name, unordered.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(|k| k.as_str())
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn info(&self, name: &str) -> Result<&Info> {
        self.index
            .get(name)
            .with_context(|| format!("missing tensor {name}"))
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        if let Some(p) = self.pinned.get(name) {
            return Ok(p);
        }
        let i = self.info(name)?;
        Ok(&self.maps[i.file][i.start..i.end])
    }

    /// Copy a tensor out of its map into anonymous memory, so later reads never go back to the
    /// file. For tables read a few random rows at a time (Engram's, 100 GB each): on a network
    /// mount a cold row is a round trip (~2 ms on RunPod's MooseFS), and the page cache that
    /// `prefault` fills is evicted again within the hour there. One copy per (file, tensor) per
    /// process, shared by every `St` over the same checkpoint. The copy reads in parallel.
    pub fn pin(&mut self, name: &str) -> Result<()> {
        if self.pinned.contains_key(name) {
            return Ok(());
        }
        let i = self.info(name)?.clone();
        let key = (self.files[i.file].clone(), name.to_string());
        let mut g = PINNED.lock().unwrap();
        let cache = g.get_or_insert_with(HashMap::new);
        let p = match cache.get(&key) {
            Some(p) => p.clone(),
            None => {
                let src = &self.maps[i.file][i.start..i.end];
                let mut v: Vec<u8> = Vec::with_capacity(src.len());
                // Safety: every byte is written below before the Vec is read.
                #[allow(clippy::uninit_vec)]
                unsafe {
                    v.set_len(src.len())
                };
                // 2 MB pages for the copy (THP is often `madvise`-only): Engram's per-step gather
                // reads a few hundred random rows of these 100 GB tables, a TLB miss each on 4 KB
                // pages
                #[cfg(target_os = "linux")]
                unsafe {
                    let (a, n) = (v.as_mut_ptr() as usize, v.len());
                    let lo = a.next_multiple_of(2 << 20);
                    let hi = (a + n) & !((2 << 20) - 1);
                    if hi > lo {
                        libc::madvise(lo as *mut libc::c_void, hi - lo, libc::MADV_HUGEPAGE);
                    }
                }
                let threads = std::thread::available_parallelism().map_or(16, |n| n.get().min(64));
                let chunk = src
                    .len()
                    .div_ceil(threads)
                    .next_multiple_of(1 << 20)
                    .max(1 << 20);
                std::thread::scope(|sc| {
                    for (dst, s) in v.chunks_mut(chunk).zip(src.chunks(chunk)) {
                        sc.spawn(move || dst.copy_from_slice(s));
                    }
                });
                let p: Arc<[u8]> = v.into();
                cache.insert(key, p.clone());
                p
            }
        };
        self.pinned.insert(name.to_string(), p);
        Ok(())
    }

    /// Fault every page of a tensor into the page cache, in parallel. For tables read a few
    /// random rows at a time from a memory map (Engram's, 100 GB each): on a network mount a
    /// cold row costs a round trip (~2 ms on RunPod's MooseFS), so a step that touches 300 of
    /// them stalls for the better part of a second until the rows it needs happen to be warm.
    pub fn prefault(&self, name: &str) -> Result<()> {
        let b = self.bytes(name)?;
        const PAGE: usize = 4096;
        let threads = std::thread::available_parallelism().map_or(16, |n| n.get().min(64));
        let chunk = b.len().div_ceil(threads).next_multiple_of(PAGE);
        std::thread::scope(|sc| {
            for part in b.chunks(chunk.max(PAGE)) {
                sc.spawn(move || {
                    let mut acc = 0u8;
                    for page in part.chunks(PAGE) {
                        acc ^= std::hint::black_box(page[0]);
                    }
                    std::hint::black_box(acc);
                });
            }
        });
        Ok(())
    }

    /// Any tensor as f32 with its shape. Quantized weights (`F8_E4M3`, or `I8`-packed fp4) are
    /// dequantized with their `.scale` sibling: per 32 × 32 block, or per row × 32 when the
    /// scale has one row per weight row.
    pub fn f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let info = self.info(name)?.clone();
        let b = self.bytes(name)?;
        let raw: Vec<f32> = match info.dtype {
            Dtype::F32 => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            Dtype::BF16 => b
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            Dtype::F8E4M3 => b.iter().map(|&v| e4m3_to_f32(v)).collect(),
            Dtype::F8E8M0 => b.iter().map(|&v| e8m0_to_f32(v)).collect(),
            Dtype::I8 | Dtype::U8 => b
                .iter()
                .flat_map(|&v| [fp4_to_f32(v & 0xF), fp4_to_f32(v >> 4)])
                .collect(),
        };
        let mut shape = info.shape.clone();
        if matches!(info.dtype, Dtype::I8 | Dtype::U8) {
            *shape.last_mut().unwrap() *= 2;
        }
        if !matches!(info.dtype, Dtype::F8E4M3 | Dtype::I8 | Dtype::U8) {
            return Ok((raw, shape));
        }
        let sname = format!(
            "{}scale",
            name.strip_suffix("weight")
                .context("quantized non-weight")?
        );
        let (s, ss) = self.f32(&sname)?;
        let (rows, cols) = (shape[0], shape[1]);
        let mut out = raw;
        if ss == [rows, cols / 32] {
            for r in 0..rows {
                for c in 0..cols {
                    out[r * cols + c] *= s[r * ss[1] + c / 32];
                }
            }
        } else {
            ensure!(
                ss == [rows.div_ceil(32), cols.div_ceil(32)],
                "{sname}: scale {ss:?} for weight {shape:?}"
            );
            for r in 0..rows {
                for c in 0..cols {
                    out[r * cols + c] *= s[(r / 32) * ss[1] + c / 32];
                }
            }
        }
        Ok((out, shape))
    }
}
