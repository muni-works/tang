//! Measured schedule choices: the knobs the CUDA decode lowering used to read
//! from the environment (`TANG_GEMV_WARPS`, `TANG_GEMV_UPL`, `TANG_WIN_TILE`, …)
//! go through [`knob`], which answers from a tuning cache first.
//!
//! The cache is keyed by GPU name and *kernel signature*: a hash of what the
//! kernel computes (root shape and storage, reduce dims, the int8 GEMV schedule,
//! each read's shape and storage), so the 48 layers of a model share one entry
//! per distinct kernel and two models with the same projection shapes share
//! theirs. Step-level choices (multi-output kernels, staged maps, split rows)
//! live under the signature `step:<name>`.
//!
//! Files: `tune/<gpu>.tsv` in the repo (committed for known GPUs) or
//! `TANG_TUNE_CACHE=<file>`; `TANG_TUNE=0` ignores every cache. A line is
//! `sig<TAB>KNOB=V,KNOB=V<TAB>ms<TAB>default_ms<TAB>note`; `#` starts a comment.
//! An environment variable always wins over the cache, so a hand experiment
//! still sees exactly the value it set.
//!
//! While lowering, [`set_kernel`] names the kernel being generated; every
//! [`knob`] read is recorded against it ([`take_report`]) so the tuner knows
//! which knobs a kernel actually consults and sweeps only those.
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};

/// One kernel's choices: knob → value.
pub type Choice = BTreeMap<String, String>;

#[derive(Default)]
struct Cache {
    gpu: String,
    by_sig: HashMap<String, Choice>,
}

static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();

thread_local! {
    static CURRENT: RefCell<Option<String>> = const { RefCell::new(None) };
    /// (kernel index, signature, knobs consulted) of the lowering in flight.
    static REPORT: RefCell<Vec<(usize, String, Vec<String>)>> = const { RefCell::new(Vec::new()) };
    /// Forced choices (the tuner's sweep): beat the cache, lose to the env.
    static FORCE: RefCell<HashMap<String, Choice>> = RefCell::new(HashMap::new());
    /// One choice forced on every kernel and step (a sweep point).
    static FORCE_ALL: RefCell<Choice> = const { RefCell::new(BTreeMap::new()) };
}

fn cache() -> &'static Mutex<Cache> {
    CACHE.get_or_init(|| Mutex::new(Cache::default()))
}

/// A GPU name as a file stem: `NVIDIA H200` → `nvidia-h200`.
pub fn gpu_key(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    while s.contains("--") {
        s = s.replace("--", "-");
    }
    s.trim_matches('-').to_string()
}

/// Parse a cache file's text.
pub fn parse(text: &str) -> HashMap<String, Choice> {
    let mut m = HashMap::new();
    for l in text.lines() {
        let l = l.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let mut f = l.split('\t');
        let (Some(sig), Some(kv)) = (f.next(), f.next()) else {
            continue;
        };
        let c: Choice = kv
            .split(',')
            .filter_map(|p| p.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        m.insert(sig.to_string(), c);
    }
    m
}

/// The committed cache for `gpu`: `<repo>/tune/<gpu_key>.tsv`.
pub fn repo_file(gpu: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tune")
        .join(format!("{}.tsv", gpu_key(gpu)))
}

/// Load the cache for `gpu` (call once the device is known, before lowering).
/// `TANG_TUNE_CACHE` names a file to use instead; `TANG_TUNE=0` loads nothing.
/// Returns the number of entries loaded.
pub fn load_for_gpu(gpu: &str) -> usize {
    let mut c = cache().lock().unwrap();
    c.gpu = gpu.to_string();
    c.by_sig.clear();
    if std::env::var("TANG_TUNE").as_deref() == Ok("0") {
        return 0;
    }
    let path = std::env::var("TANG_TUNE_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| repo_file(gpu));
    if let Ok(t) = std::fs::read_to_string(&path) {
        c.by_sig = parse(&t);
    }
    c.by_sig.len()
}

/// Replace the loaded entries (tests, the tuner's verification pass).
pub fn set_entries(m: HashMap<String, Choice>) {
    cache().lock().unwrap().by_sig = m;
}

/// Force choices for signatures (the tuner's sweep). Cleared by `force(empty)`.
pub fn force(m: HashMap<String, Choice>) {
    FORCE.with(|f| *f.borrow_mut() = m);
}

/// Force `c` on every kernel and step knob (below the environment).
pub fn force_all(c: Choice) {
    FORCE_ALL.with(|f| *f.borrow_mut() = c);
}

/// Name the kernel being lowered (`None` after the loop).
pub fn set_kernel(index: usize, sig: Option<String>) {
    if let Some(s) = &sig {
        REPORT.with(|r| r.borrow_mut().push((index, s.clone(), Vec::new())));
    }
    CURRENT.with(|c| *c.borrow_mut() = sig);
}

/// The (kernel index, signature, knobs consulted) of every kernel lowered on
/// this thread since the last call.
pub fn take_report() -> Vec<(usize, String, Vec<String>)> {
    REPORT.with(|r| std::mem::take(&mut *r.borrow_mut()))
}

fn lookup(sig: &str, name: &str) -> Option<String> {
    if let Some(v) = FORCE_ALL.with(|f| f.borrow().get(name).cloned()) {
        return Some(v);
    }
    if let Some(v) = FORCE.with(|f| f.borrow().get(sig).and_then(|c| c.get(name).cloned())) {
        return Some(v);
    }
    let c = cache().lock().unwrap();
    c.by_sig.get(sig).and_then(|c| c.get(name).cloned())
}

/// A schedule knob of the kernel being lowered: the environment, else the
/// forced sweep value, else the tuning cache, else `None` (the caller's default).
pub fn knob(name: &str) -> Option<String> {
    if let Ok(v) = std::env::var(name) {
        note(name);
        return Some(v);
    }
    let sig = CURRENT.with(|c| c.borrow().clone());
    note(name);
    sig.and_then(|s| lookup(&s, name))
}

/// [`knob`] parsed.
pub fn knob_parse<T: std::str::FromStr>(name: &str) -> Option<T> {
    knob(name).and_then(|v| v.parse().ok())
}

/// A step-level knob (`step:<step>`): the environment, else forced, else cached.
pub fn step_knob(step: &str, name: &str) -> Option<String> {
    if let Ok(v) = std::env::var(name) {
        return Some(v);
    }
    lookup(&format!("step:{step}"), name)
}

fn note(name: &str) {
    REPORT.with(|r| {
        if CURRENT.with(|c| c.borrow().is_some()) {
            if let Some(last) = r.borrow_mut().last_mut() {
                if !last.2.iter().any(|k| k == name) {
                    last.2.push(name.to_string());
                }
            }
        }
    });
}

/// FNV-1a, hex: stable across runs and builds (the cache is committed).
pub fn hash_hex(s: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_beats_default_env_beats_cache() {
        let mut m = HashMap::new();
        m.insert(
            "abc".to_string(),
            [("TANG_TEST_TUNE_KNOB".to_string(), "4".to_string())].into(),
        );
        set_entries(m);
        set_kernel(0, Some("abc".into()));
        assert_eq!(knob("TANG_TEST_TUNE_KNOB").as_deref(), Some("4"));
        set_kernel(1, Some("other".into()));
        assert_eq!(knob("TANG_TEST_TUNE_KNOB"), None);
        set_kernel(0, None);
        let r = take_report();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].2, vec!["TANG_TEST_TUNE_KNOB".to_string()]);
        set_entries(HashMap::new());
    }

    #[test]
    fn parse_and_keys() {
        let m = parse("# c\nab\tTANG_GEMV_WARPS=4,TANG_GEMV_UPL=1\t0.01\t0.02\tx\n");
        assert_eq!(m["ab"]["TANG_GEMV_WARPS"], "4");
        assert_eq!(gpu_key("NVIDIA H200"), "nvidia-h200");
        assert_eq!(
            gpu_key("NVIDIA RTX PRO 6000 Blackwell"),
            "nvidia-rtx-pro-6000-blackwell"
        );
    }
}
