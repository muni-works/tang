//! A model's identity across nodes (frog's `docs/node-protocol.md`, section 6): `weights_hash`
//! and `quant`, as `/node` reports them for loaded and on-disk models.
//!
//! `weights_hash` = lowercase hex SHA-256 over `config.json`'s bytes, then for each weight file
//! (`*.safetensors`, `*.gguf`) sorted by name: the name, `0x00`, its size (u64 little-endian)
//! and its first 1 MiB. Cheap, stable across machines and paths, different across quants.

use crate::model::Dtype;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// How much of each weight file goes into the hash.
pub const HEAD: u64 = 1 << 20;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Ident {
    pub weights_hash: Option<String>,
    pub quant: Option<String>,
}

/// A checkpoint's config file (if any) and weight files, sorted by file name. `path` is a
/// checkpoint directory or a GGUF file (its shards are found next to it).
pub fn files(path: &Path) -> (Option<PathBuf>, Vec<PathBuf>) {
    let is_gguf = |p: &Path| p.extension().is_some_and(|x| x == "gguf");
    let (dir, mut weights) = if path.is_file() && is_gguf(path) {
        let shards =
            crate::gguf::discover_shards(path).unwrap_or_else(|_| vec![path.to_path_buf()]);
        (path.parent().map(Path::to_path_buf), shards)
    } else {
        let w = std::fs::read_dir(path)
            .map(|rd| {
                rd.filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| is_gguf(p) || p.extension().is_some_and(|x| x == "safetensors"))
                    .collect()
            })
            .unwrap_or_default();
        (Some(path.to_path_buf()), w)
    };
    weights.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    let config = dir.map(|d| d.join("config.json")).filter(|p| p.is_file());
    (config, weights)
}

/// The hash as section 6 defines it; `None` without any weight file.
pub fn weights_hash(path: &Path) -> std::io::Result<Option<String>> {
    let (config, weights) = files(path);
    if weights.is_empty() {
        return Ok(None);
    }
    let mut h = Sha256::new();
    if let Some(c) = config {
        h.update(std::fs::read(c)?);
    }
    let mut buf = Vec::with_capacity(HEAD as usize);
    for w in &weights {
        let name = w
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        // Follows the HF cache's symlinks to the blob.
        let f = std::fs::File::open(w)?;
        let size = f.metadata()?.len();
        h.update(name.as_bytes());
        h.update([0u8]);
        h.update(size.to_le_bytes());
        buf.clear();
        f.take(HEAD).read_to_end(&mut buf)?;
        h.update(&buf);
    }
    Ok(Some(crate::auth::hex(&h.finalize())))
}

/// How the checkpoint's weights are stored: `mlx-4bit`/`mlx-8bit` for MLX-quantized ones,
/// `gguf-<type>` (the type holding most matrix bytes) for GGUF, else `f32` or `bf16` from the
/// config's dtype (float16 counts as `bf16`: half precision, the closest listed).
pub fn stored_quant(path: &Path) -> Option<String> {
    let (config, weights) = files(path);
    if let Some(g) = weights
        .iter()
        .find(|p| p.extension().is_some_and(|x| x == "gguf"))
    {
        let g = crate::gguf::Gguf::open(g).ok()?;
        let mut by: HashMap<String, u64> = HashMap::new();
        for t in g.tensors.iter().filter(|t| t.dims.len() >= 2) {
            let bytes = t.ty.bytes_for(t.n_elements() as usize).unwrap_or(0) as u64;
            *by.entry(t.ty.name().to_lowercase()).or_default() += bytes;
        }
        let top = by
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))?;
        return Some(format!("gguf-{}", top.0));
    }
    let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(config?).ok()?).ok()?;
    if let Some(bits) = raw["quantization"]["bits"].as_u64() {
        return Some(format!("mlx-{bits}bit"));
    }
    Some(
        match raw["torch_dtype"].as_str().or(raw["dtype"].as_str()) {
            Some("float32") => "f32",
            _ => "bf16",
        }
        .into(),
    )
}

/// Identity of the checkpoint at `path`. `loaded`: the dtype the loader keeps it as, for a
/// loaded model (pre-quantized checkpoints and GGUF keep their stored format).
pub fn identify(path: &Path, loaded: Option<Dtype>) -> Ident {
    let stored = stored_quant(path);
    let quant = match (loaded, stored.as_deref()) {
        (Some(d), Some("bf16" | "f32")) | (Some(d), None) => Some(
            match d {
                Dtype::F32 => "f32",
                Dtype::Bf16 => "bf16",
                Dtype::Q4 => "q4",
            }
            .to_string(),
        ),
        _ => stored,
    };
    Ident {
        weights_hash: cached_hash(path),
        quant,
    }
}

/// The hash, recomputed only when the files' names, sizes or modification times change.
fn cached_hash(path: &Path) -> Option<String> {
    static CACHE: Mutex<Option<HashMap<PathBuf, (String, Option<String>)>>> = Mutex::new(None);
    let (config, weights) = files(path);
    let stamp: String = config
        .iter()
        .chain(&weights)
        .map(|p| {
            let m = std::fs::metadata(p).ok();
            format!(
                "{}:{}:{:?};",
                p.display(),
                m.as_ref().map_or(0, |m| m.len()),
                m.and_then(|m| m.modified().ok())
            )
        })
        .collect();
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let cache = cache.get_or_insert_with(HashMap::new);
    if let Some((s, h)) = cache.get(path) {
        if *s == stamp {
            return h.clone();
        }
    }
    let h = weights_hash(path).ok().flatten();
    cache.insert(path.to_path_buf(), (stamp, h.clone()));
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(config: &str, shards: &[(&str, usize, u8)]) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "tang-model-id-{}",
            crate::auth::hex(&crate::auth::random_bytes::<8>())
        ));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("config.json"), config).unwrap();
        for (name, len, fill) in shards {
            std::fs::write(d.join(name), vec![*fill; *len]).unwrap();
        }
        std::fs::write(d.join("tokenizer.json"), "{}").unwrap();
        d
    }

    #[test]
    fn hash_follows_the_spec_and_is_stable() {
        let a = checkpoint(
            r#"{"torch_dtype":"bfloat16"}"#,
            &[
                ("model-00002.safetensors", 10, 2),
                ("model-00001.safetensors", (HEAD + 5) as usize, 1),
            ],
        );
        // Same bytes under another path: same hash.
        let b = checkpoint(
            r#"{"torch_dtype":"bfloat16"}"#,
            &[
                ("model-00001.safetensors", (HEAD + 5) as usize, 1),
                ("model-00002.safetensors", 10, 2),
            ],
        );
        let ha = weights_hash(&a).unwrap().unwrap();
        assert_eq!(Some(ha.clone()), weights_hash(&b).unwrap());
        assert_eq!(identify(&a, None).weights_hash, Some(ha.clone()));
        // By hand.
        let mut h = Sha256::new();
        h.update(br#"{"torch_dtype":"bfloat16"}"#);
        h.update(b"model-00001.safetensors\0");
        h.update((HEAD + 5).to_le_bytes());
        h.update(vec![1u8; HEAD as usize]);
        h.update(b"model-00002.safetensors\0");
        h.update(10u64.to_le_bytes());
        h.update([2u8; 10]);
        assert_eq!(ha, crate::auth::hex(&h.finalize()));
        // Bytes past the first MiB don't count (that's the point: no full read).
        let c = checkpoint(r#"{"torch_dtype":"bfloat16"}"#, &[]);
        let mut big = vec![1u8; (HEAD + 5) as usize];
        big[HEAD as usize + 1] = 9;
        std::fs::write(c.join("model-00001.safetensors"), big).unwrap();
        std::fs::write(c.join("model-00002.safetensors"), [2u8; 10]).unwrap();
        assert_eq!(weights_hash(&c).unwrap(), Some(ha));
        for d in [a, b, c] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    #[test]
    fn hash_and_quant_tell_quants_apart() {
        let bf16 = checkpoint(
            r#"{"torch_dtype":"bfloat16"}"#,
            &[("model.safetensors", 64, 1)],
        );
        let q4 = checkpoint(
            r#"{"torch_dtype":"bfloat16","quantization":{"group_size":64,"bits":4}}"#,
            &[("model.safetensors", 64, 1)],
        );
        let q8 = checkpoint(
            r#"{"torch_dtype":"bfloat16","quantization":{"group_size":64,"bits":8}}"#,
            &[("model.safetensors", 64, 1)],
        );
        let f32 = checkpoint(
            r#"{"torch_dtype":"float32"}"#,
            &[("model.safetensors", 64, 1)],
        );
        let (a, b, c) = (
            identify(&bf16, None),
            identify(&q4, None),
            identify(&q8, None),
        );
        assert_ne!(a.weights_hash, b.weights_hash);
        assert_ne!(b.weights_hash, c.weights_hash);
        assert_eq!(a.quant.as_deref(), Some("bf16"));
        assert_eq!(b.quant.as_deref(), Some("mlx-4bit"));
        assert_eq!(c.quant.as_deref(), Some("mlx-8bit"));
        assert_eq!(identify(&f32, None).quant.as_deref(), Some("f32"));
        // Loaded: what the loader keeps; a pre-quantized checkpoint stays as it is.
        assert_eq!(
            identify(&bf16, Some(Dtype::Q4)).quant.as_deref(),
            Some("q4")
        );
        assert_eq!(
            identify(&f32, Some(Dtype::Bf16)).quant.as_deref(),
            Some("bf16")
        );
        assert_eq!(
            identify(&q4, Some(Dtype::Bf16)).quant.as_deref(),
            Some("mlx-4bit")
        );
        // A changed file changes the cached hash.
        let before = identify(&bf16, None).weights_hash;
        std::fs::write(bf16.join("model.safetensors"), [3u8; 65]).unwrap();
        assert_ne!(identify(&bf16, None).weights_hash, before);
        for d in [bf16, q4, q8, f32] {
            let _ = std::fs::remove_dir_all(d);
        }
    }
}
