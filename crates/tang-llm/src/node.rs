//! What `GET /node` reports about this machine: a stable id, the hardware and its free memory,
//! the models on disk and what loading one would take, and rates measured over real requests.
//! See `docs/node.md`.

use crate::model::Dtype;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Version of `/node`'s JSON shape. Bumped when a field changes meaning or goes away (new
/// fields don't bump it).
pub const SCHEMA: u32 = 2;

/// Where tang keeps its state: `~/.cache/tang` (`TANG_CACHE_DIR` overrides).
pub fn cache_dir() -> Option<PathBuf> {
    std::env::var_os("TANG_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/tang")))
}

/// This machine's node id: made once (128 random bits, hex) and kept in `<cache>/node-id`,
/// so it survives restarts and model swaps. Without a cache directory it lasts the process.
pub fn node_id() -> String {
    // Concurrent listeners in one process must not truncate the same temporary file.
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let path = cache_dir().map(|d| d.join("node-id"));
    if let Some(id) = path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| s.len() == 32 && s.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        return id;
    }
    let id = format!("{:016x}{:016x}", random_u64(), random_u64());
    if let Some(p) = path {
        let _ = p.parent().map(std::fs::create_dir_all);
        // Written whole, then renamed: two servers starting at once agree on one id.
        let tmp = p.with_extension(format!("tmp{}-{:016x}", std::process::id(), random_u64()));
        if std::fs::write(&tmp, &id).is_ok() && std::fs::rename(&tmp, &p).is_ok() {
            if let Ok(s) = std::fs::read_to_string(&p) {
                let s = s.trim();
                if s.len() == 32 && s.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return s.to_string();
                }
            }
        }
    }
    id
}

fn random_u64() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    // RandomState is seeded from the OS RNG; time and pid for good measure.
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|t| t.as_nanos())
            .unwrap_or(0),
    );
    h.write_u32(std::process::id());
    h.finish()
}

/// The device models run on and its memory, as of now.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Hardware {
    /// `metal`, `cuda` or `cpu`.
    pub kind: &'static str,
    pub name: String,
    /// The GPU shares the system's RAM (Metal, CPU).
    pub unified: bool,
    /// VRAM, or RAM for unified memory.
    pub total_bytes: u64,
    /// What a model load may use without taking memory from anything else: free VRAM (every
    /// process's use counted) on CUDA; on unified memory, RAM the system has available (free
    /// and inactive pages), capped on Metal by what's left of the GPU's working set.
    pub free_bytes: u64,
}

impl Hardware {
    pub fn json(&self) -> Value {
        json!({
            "kind": self.kind,
            "name": self.name,
            "unified_memory": self.unified,
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
        })
    }
}

/// Measures the hardware (called per `/node` and per load, from any thread).
pub type Probe = Arc<dyn Fn() -> Hardware + Send + Sync>;

/// The system's RAM: total, and available to a new allocation without swapping anything out.
pub fn host_memory() -> Option<(u64, u64)> {
    host::memory()
}

#[cfg(target_os = "macos")]
mod host {
    // Mach's VM statistics, as `vm_stat` reads them.
    #[repr(C)]
    #[derive(Default)]
    struct VmStatistics64 {
        free_count: u32,
        active_count: u32,
        inactive_count: u32,
        wire_count: u32,
        zero_fill_count: u64,
        reactivations: u64,
        pageins: u64,
        pageouts: u64,
        faults: u64,
        cow_faults: u64,
        lookups: u64,
        hits: u64,
        purges: u64,
        purgeable_count: u32,
        speculative_count: u32,
        decompressions: u64,
        compressions: u64,
        swapins: u64,
        swapouts: u64,
        compressor_page_count: u32,
        throttled_count: u32,
        external_page_count: u32,
        internal_page_count: u32,
        total_uncompressed_pages_in_compressor: u64,
    }

    const HOST_VM_INFO64: i32 = 4;

    extern "C" {
        fn mach_host_self() -> u32;
        fn host_statistics64(host: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
        fn host_page_size(host: u32, size: *mut usize) -> i32;
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            old: *mut std::ffi::c_void,
            oldlen: *mut usize,
            new: *mut std::ffi::c_void,
            newlen: usize,
        ) -> i32;
    }

    pub fn memory() -> Option<(u64, u64)> {
        let mut total = 0u64;
        let mut len = std::mem::size_of::<u64>();
        let name = c"hw.memsize";
        // SAFETY: hw.memsize is a u64; `len` says how much room `total` has.
        let r = unsafe {
            sysctlbyname(
                name.as_ptr(),
                (&mut total as *mut u64).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if r != 0 {
            return None;
        }
        let mut vm = VmStatistics64::default();
        let mut count = (std::mem::size_of::<VmStatistics64>() / 4) as u32;
        let mut page = 0usize;
        // SAFETY: `count` is the struct's size in 32-bit words, as the call expects.
        let ok = unsafe {
            let host = mach_host_self();
            host_page_size(host, &mut page) == 0
                && host_statistics64(
                    host,
                    HOST_VM_INFO64,
                    (&mut vm as *mut VmStatistics64).cast(),
                    &mut count,
                ) == 0
        };
        if !ok {
            return None;
        }
        // Free (speculative pages are counted in free_count), plus inactive pages, which the
        // system reclaims before it would swap. Purgeable pages are left out: they're also on
        // the active and inactive lists, and only their owner says when they may go.
        let pages = vm.free_count as u64 + vm.inactive_count as u64;
        Some((total, (pages * page as u64).min(total)))
    }
}

#[cfg(not(target_os = "macos"))]
mod host {
    pub fn memory() -> Option<(u64, u64)> {
        let info = std::fs::read_to_string("/proc/meminfo").ok()?;
        let field = |name: &str| -> Option<u64> {
            let kb: u64 = info
                .lines()
                .find_map(|l| l.strip_prefix(name))?
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse()
                .ok()?;
            Some(kb * 1024)
        };
        Some((field("MemTotal:")?, field("MemAvailable:")?))
    }
}

/// A model checkpoint on disk.
#[derive(Debug, Clone, PartialEq)]
pub struct OnDisk {
    /// What `/models/load` takes: the Hugging Face repo id (or the directory, for one given
    /// as a path).
    pub id: String,
    pub path: PathBuf,
    /// The checkpoint's weight files.
    pub size_bytes: u64,
}

/// Models in the Hugging Face cache that tang can load (a snapshot with a config, tokenizer,
/// chat template config and safetensors), largest first.
pub fn models_on_disk() -> Vec<OnDisk> {
    let hub = crate::hf_hub_dir();
    let Ok(rd) = std::fs::read_dir(&hub) else {
        return Vec::new();
    };
    let mut out: Vec<OnDisk> = rd
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            let repo = name.strip_prefix("models--")?.replacen("--", "/", 1);
            let path = crate::resolve_model(&repo).ok()?;
            loadable(&path).then(|| OnDisk {
                size_bytes: weight_bytes(&path),
                id: repo,
                path,
            })
        })
        .collect();
    out.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes).then(a.id.cmp(&b.id)));
    out
}

fn loadable(dir: &Path) -> bool {
    ["config.json", "tokenizer.json", "tokenizer_config.json"]
        .iter()
        .all(|f| dir.join(f).exists())
        && weight_bytes(dir) > 0
}

/// Bytes of a checkpoint's `.safetensors` files (following the HF cache's symlinks).
pub fn weight_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
                .filter_map(|e| std::fs::metadata(e.path()).ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// Device memory a model needs, estimated before loading it: its weights as `dtype` keeps them,
/// and headroom for a working KV cache and scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Need {
    pub weights_bytes: u64,
    pub headroom_bytes: u64,
}

impl Need {
    pub fn total(&self) -> u64 {
        self.weights_bytes + self.headroom_bytes
    }
}

/// KV positions a fresh load must have room for (on top of its weights): a typical turn.
pub const HEADROOM_POSITIONS: u64 = 8192;
/// Scratch for activations, staged weight chunks and the like (CUDA's prefill stages up to
/// 128 MB of dequantized weights; cuBLAS wants a workspace).
pub const HEADROOM_SCRATCH: u64 = 1 << 30;

/// What loading the checkpoint in `dir` as `dtype` takes. Pre-quantized (MLX 4-bit)
/// checkpoints load as they are; others are converted from their stored precision.
pub fn need(dir: &Path, dtype: Dtype) -> anyhow::Result<Need> {
    let cfg = crate::Config::from_json(&std::fs::read(dir.join("config.json"))?)?;
    let raw: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let file = weight_bytes(dir);
    let weights = if cfg.quantization.is_some() {
        file
    } else {
        let stored = match raw["torch_dtype"].as_str().or(raw["dtype"].as_str()) {
            Some("float32") => 4.0,
            _ => 2.0,
        };
        let kept = match dtype {
            Dtype::F32 => 4.0,
            Dtype::Bf16 => 2.0,
            // 4 bits, plus a bf16 scale and bias per 64.
            Dtype::Q4 => 0.5 + 4.0 / 64.0,
        };
        (file as f64 / stored * kept) as u64
    };
    let kv_row = 2 * cfg.num_hidden_layers as u64 * cfg.kv_dim() as u64 * 2;
    Ok(Need {
        weights_bytes: weights,
        headroom_bytes: kv_row * HEADROOM_POSITIONS + HEADROOM_SCRATCH,
    })
}

/// An exponential moving average of a rate measured over real requests; `None` until the
/// first sample.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Ema {
    pub value: Option<f64>,
    pub samples: u64,
}

/// Weight of the newest sample.
pub const EMA_ALPHA: f64 = 0.25;

impl Ema {
    pub fn observe(&mut self, x: f64) {
        if !x.is_finite() || x <= 0.0 {
            return;
        }
        self.value = Some(match self.value {
            None => x,
            Some(v) => v + EMA_ALPHA * (x - v),
        });
        self.samples += 1;
    }
}

/// Fewest positions a prefill must have computed to count toward the prefill rate (shorter
/// ones are mostly fixed overhead).
pub const MIN_PREFILL_SAMPLE: usize = 64;
/// Fewest tokens a completion must have generated to count toward the decode rate.
pub const MIN_DECODE_SAMPLE: usize = 16;

/// Tokens per second a loaded model has done, prefilling and decoding.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Rates {
    pub prefill: Ema,
    pub decode: Ema,
}

impl Rates {
    /// A prefill of `tokens` positions that took `secs`.
    pub fn prefilled(&mut self, tokens: usize, secs: f64) {
        if tokens >= MIN_PREFILL_SAMPLE && secs > 0.0 {
            self.prefill.observe(tokens as f64 / secs);
        }
    }

    /// A completion of `tokens` tokens at `tok_s` (speculative decoding included).
    pub fn decoded(&mut self, tokens: usize, tok_s: f64) {
        if tokens >= MIN_DECODE_SAMPLE {
            self.decode.observe(tok_s);
        }
    }

    pub fn json(&self) -> Value {
        json!({
            "prefill_tok_s": self.prefill.value,
            "prefill_samples": self.prefill.samples,
            "decode_tok_s": self.decode.value,
            "decode_samples": self.decode.samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_are_null_until_measured_then_move_toward_new_samples() {
        let mut r = Rates::default();
        assert_eq!(r.json()["prefill_tok_s"], Value::Null);
        // Too short to count.
        r.prefilled(10, 0.001);
        r.decoded(3, 50.0);
        assert_eq!(r.prefill.samples + r.decode.samples, 0);
        r.prefilled(1000, 1.0);
        assert_eq!(r.prefill.value, Some(1000.0));
        r.prefilled(2000, 1.0);
        assert_eq!(r.prefill.value, Some(1000.0 + EMA_ALPHA * 1000.0));
        r.decoded(100, 40.0);
        assert_eq!(r.json()["decode_tok_s"], 40.0);
        assert_eq!(r.json()["prefill_samples"], 2);
    }

    #[test]
    fn node_id_is_made_once() {
        let dir = std::env::temp_dir().join(format!("tang-node-id-{}", random_u64()));
        std::env::set_var("TANG_CACHE_DIR", &dir);
        let a = node_id();
        let b = node_id();
        let threads: Vec<_> = (0..8).map(|_| std::thread::spawn(node_id)).collect();
        for thread in threads {
            assert_eq!(thread.join().unwrap(), a);
        }
        std::env::remove_var("TANG_CACHE_DIR");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(a, b);
        assert_eq!(a.len(), 32);
    }

    #[test]
    fn host_memory_is_sane() {
        let (total, free) = host_memory().expect("host memory");
        assert!(total > 1 << 30, "{total}");
        assert!(free > 0 && free <= total, "{free} of {total}");
    }
}
