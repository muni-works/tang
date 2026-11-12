//! Downloading a model onto this node (`POST /models/pull`), into the Hugging Face cache's
//! layout so [`crate::resolve_model`] and [`crate::node::models_on_disk`] find it:
//!
//! ```text
//! <hub>/models--<org>--<name>/blobs/<sha256>            each file, once verified
//! <hub>/models--<org>--<name>/blobs/<sha256>.part       a file being downloaded (resumable)
//! <hub>/models--<org>--<name>/snapshots/<rev>/<path>    symlink to ../../blobs/<sha256>
//! <hub>/models--<org>--<name>/refs/main                 <rev>, written last
//! ```
//!
//! The caller (frog, from its catalog) names every file with its URL, size and sha256; the
//! node downloads them one after another, resuming `.part` files with Range requests, and
//! checks each against its sha256 before it counts. One pull at a time. See `docs/node.md`.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// One file of a pull.
#[derive(Debug, Clone, Deserialize)]
pub struct File {
    /// Where it goes in the snapshot (`model.safetensors`, `tokenizer.json`, ...).
    pub path: String,
    pub url: String,
    pub sha256: String,
    pub size: u64,
}

/// `POST /models/pull`'s body.
#[derive(Debug, Clone, Deserialize)]
pub struct Spec {
    /// The Hugging Face repo id it's cached under (`org/name`).
    pub model: String,
    /// The snapshot's revision (the HF commit sha the files came from). Without one, a name
    /// made from the files' hashes.
    #[serde(default)]
    pub revision: Option<String>,
    pub files: Vec<File>,
}

impl Spec {
    /// Refuse anything that could write outside the model's cache directory.
    pub fn check(&self) -> Result<(), String> {
        let ok_part = |s: &str| {
            !s.is_empty()
                && s != "."
                && s != ".."
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        };
        match self.model.split_once('/') {
            Some((org, name)) if ok_part(org) && ok_part(name) => {}
            _ => {
                return Err(format!(
                    "model must be a repo id like org/name, not {:?}",
                    self.model
                ))
            }
        }
        if let Some(rev) = &self.revision {
            if !ok_part(rev) {
                return Err(format!("bad revision {rev:?}"));
            }
        }
        if self.files.is_empty() {
            return Err("files is empty".into());
        }
        let mut seen = std::collections::HashSet::new();
        for f in &self.files {
            if f.path.starts_with('/') || !f.path.split('/').all(ok_part) {
                return Err(format!("bad file path {:?}", f.path));
            }
            if !seen.insert(&f.path) {
                return Err(format!("{} is listed twice", f.path));
            }
            if f.sha256.len() != 64 || !f.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(format!("{}: sha256 must be 64 hex digits", f.path));
            }
            if !(f.url.starts_with("https://") || f.url.starts_with("http://")) {
                return Err(format!("{}: url must be http(s)", f.path));
            }
        }
        Ok(())
    }

    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    /// Bytes of the checkpoint's weights (what [`crate::node::need_from`] scales).
    pub fn weight_bytes(&self) -> u64 {
        self.files
            .iter()
            .filter(|f| f.path.ends_with(".safetensors"))
            .map(|f| f.size)
            .sum()
    }

    pub fn revision(&self) -> String {
        self.revision.clone().unwrap_or_else(|| {
            let mut h = Sha256::new();
            for f in &self.files {
                h.update(f.path.as_bytes());
                h.update(f.sha256.to_ascii_lowercase().as_bytes());
            }
            hex(&h.finalize())[..40].to_string()
        })
    }
}

/// The model's directory in the hub cache.
pub fn repo_dir(model: &str) -> PathBuf {
    crate::hf_hub_dir().join(format!("models--{}", model.replace('/', "--")))
}

/// Bytes already downloaded towards a pull (verified blobs, plus `.part` files): what it
/// doesn't need to fetch again.
fn have_bytes(dir: &Path, spec: &Spec) -> u64 {
    spec.files
        .iter()
        .map(|f| {
            let blob = dir.join("blobs").join(f.sha256.to_ascii_lowercase());
            match std::fs::metadata(&blob) {
                Ok(m) if m.len() == f.size => f.size,
                _ => std::fs::metadata(blob.with_extension("part"))
                    .map(|m| m.len().min(f.size))
                    .unwrap_or(0),
            }
        })
        .sum()
}

/// Free bytes on the filesystem holding `path` (or its nearest existing parent).
pub fn free_disk(path: &Path) -> Option<u64> {
    let mut p = path;
    while !p.exists() {
        p = p.parent()?;
    }
    let c = std::ffi::CString::new(p.as_os_str().as_encoded_bytes()).ok()?;
    // SAFETY: `c` is a NUL-terminated path; statvfs fills `st`.
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return None;
    }
    #[allow(clippy::unnecessary_cast)]
    Some(st.f_bavail as u64 * st.f_frsize as u64)
}

/// What's being pulled, for `/node`.
pub struct Progress {
    pub model: String,
    pub total_bytes: u64,
    pub done: Arc<AtomicU64>,
    cancel: Arc<AtomicBool>,
}

/// How the last pull ended.
#[derive(Clone)]
pub struct Ended {
    pub model: String,
    pub ok: bool,
    pub cancelled: bool,
    pub error: Option<String>,
}

/// The node's pull slot.
#[derive(Default)]
pub struct Puller {
    state: Mutex<(Option<Progress>, Option<Ended>)>,
}

/// Why a pull wasn't started.
pub enum Refused {
    Bad(String),
    Busy(String),
    TooBig(String),
}

impl Puller {
    fn lock(&self) -> std::sync::MutexGuard<'_, (Option<Progress>, Option<Ended>)> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `pulling` and `last_pull` for `/node`.
    pub fn json(&self) -> (serde_json::Value, serde_json::Value) {
        let st = self.lock();
        let pulling = st.0.as_ref().map(|p| {
            serde_json::json!({
                "model": p.model,
                "done_bytes": p.done.load(Ordering::Relaxed),
                "total_bytes": p.total_bytes,
            })
        });
        let last = st.1.as_ref().map(|e| {
            serde_json::json!({
                "model": e.model, "ok": e.ok, "cancelled": e.cancelled, "error": e.error,
            })
        });
        (pulling.into(), last.into())
    }

    /// The model being pulled.
    pub fn pulling(&self) -> Option<String> {
        self.lock().0.as_ref().map(|p| p.model.clone())
    }

    /// Ask the running pull to stop (its `.part` files stay, so pulling again resumes).
    pub fn cancel(&self) -> Option<String> {
        let st = self.lock();
        let p = st.0.as_ref()?;
        p.cancel.store(true, Ordering::Relaxed);
        Some(p.model.clone())
    }

    /// Check a pull fits and start it on the runtime; `fits` gets the bytes of `config.json`
    /// (when the pull has one) and the weight bytes, and says why it won't fit.
    pub async fn start(
        self: &Arc<Self>,
        spec: Spec,
        fits: impl FnOnce(Option<&[u8]>, u64) -> Result<(), String>,
    ) -> Result<u64, Refused> {
        spec.check().map_err(Refused::Bad)?;
        // Claim the slot first so two pulls racing can't both pass the checks.
        let done = Arc::new(AtomicU64::new(0));
        let cancel = Arc::new(AtomicBool::new(false));
        {
            let mut st = self.lock();
            if let Some(p) = &st.0 {
                return Err(Refused::Busy(format!("already pulling {}", p.model)));
            }
            st.0 = Some(Progress {
                model: spec.model.clone(),
                total_bytes: spec.total_bytes(),
                done: done.clone(),
                cancel: cancel.clone(),
            });
        }
        let release = |me: &Self| me.lock().0 = None;
        let client = match reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                release(self);
                return Err(Refused::Bad(e.to_string()));
            }
        };
        let dir = repo_dir(&spec.model);
        // The config is small: fetch it now, so the memory check runs before anything big.
        let config = match spec.files.iter().find(|f| f.path == "config.json") {
            Some(f) => match fetch_small(&client, f).await {
                Ok(b) => Some(b),
                Err(e) => {
                    release(self);
                    return Err(Refused::Bad(format!("config.json: {e}")));
                }
            },
            None => None,
        };
        if let Err(e) = fits(config.as_deref(), spec.weight_bytes()) {
            release(self);
            return Err(Refused::TooBig(e));
        }
        let have = have_bytes(&dir, &spec);
        let want = spec.total_bytes().saturating_sub(have);
        if let Some(free) = free_disk(&crate::hf_hub_dir()) {
            if want > free {
                release(self);
                return Err(Refused::TooBig(format!(
                    "{} needs {want} more bytes on disk; {free} free",
                    spec.model
                )));
            }
        }
        done.store(have, Ordering::Relaxed);
        let me = self.clone();
        let total = spec.total_bytes();
        tokio::spawn(async move {
            let model = spec.model.clone();
            let result = run(&client, &dir, &spec, &done, &cancel).await;
            let cancelled = cancel.load(Ordering::Relaxed);
            match &result {
                Ok(()) => eprintln!("tang-llm: pulled {model}"),
                Err(e) => eprintln!("tang-llm: pull {model}: {e:#}"),
            }
            let mut st = me.lock();
            st.0 = None;
            st.1 = Some(Ended {
                model,
                ok: result.is_ok(),
                cancelled: cancelled && result.is_err(),
                error: result.err().map(|e| format!("{e:#}")),
            });
        });
        Ok(total)
    }
}

async fn fetch_small(client: &reqwest::Client, f: &File) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        f.size <= 16 << 20,
        "{} bytes is too big for a config",
        f.size
    );
    let body = client
        .get(&f.url)
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    anyhow::ensure!(
        body.len() as u64 == f.size && hex(&Sha256::digest(&body)) == f.sha256.to_ascii_lowercase(),
        "doesn't match its size and sha256"
    );
    Ok(body.to_vec())
}

/// Download every file, link the snapshot, then write `refs/main`.
async fn run(
    client: &reqwest::Client,
    dir: &Path,
    spec: &Spec,
    done: &AtomicU64,
    cancel: &AtomicBool,
) -> anyhow::Result<()> {
    let blobs = dir.join("blobs");
    tokio::fs::create_dir_all(&blobs).await?;
    for f in &spec.files {
        let sha = f.sha256.to_ascii_lowercase();
        let blob = blobs.join(&sha);
        if tokio::fs::metadata(&blob)
            .await
            .is_ok_and(|m| m.len() == f.size)
        {
            continue;
        }
        fetch(client, f, &blob, done, cancel).await?;
    }
    let rev = spec.revision();
    let snap = dir.join("snapshots").join(&rev);
    for f in &spec.files {
        let link = snap.join(&f.path);
        let parent = link.parent().unwrap_or(&snap);
        tokio::fs::create_dir_all(parent).await?;
        let depth = f.path.matches('/').count();
        let target = PathBuf::from("../".repeat(2 + depth))
            .join("blobs")
            .join(f.sha256.to_ascii_lowercase());
        let _ = tokio::fs::remove_file(&link).await;
        link_or_copy(&target, &link)?;
    }
    let refs = dir.join("refs");
    tokio::fs::create_dir_all(&refs).await?;
    let tmp = refs.join(".main.tmp");
    tokio::fs::write(&tmp, &rev).await?;
    tokio::fs::rename(&tmp, refs.join("main")).await?;
    Ok(())
}

#[cfg(unix)]
fn link_or_copy(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

#[cfg(not(unix))]
fn link_or_copy(target: &Path, link: &Path) -> std::io::Result<()> {
    std::fs::copy(link.parent().unwrap().join(target), link).map(|_| ())
}

/// One file into `<blob>.part`, resuming what's there, then checked and renamed to `blob`.
async fn fetch(
    client: &reqwest::Client,
    f: &File,
    blob: &Path,
    done: &AtomicU64,
    cancel: &AtomicBool,
) -> anyhow::Result<()> {
    use tokio::io::AsyncWriteExt;
    let part = blob.with_extension("part");
    let mut have = tokio::fs::metadata(&part)
        .await
        .map(|m| m.len())
        .unwrap_or(0);
    if have > f.size {
        tokio::fs::remove_file(&part).await?;
        done.fetch_sub(f.size, Ordering::Relaxed);
        have = 0;
    }
    // Hash what's already there (off the runtime: it can be gigabytes).
    let mut hasher = if have > 0 {
        let p = part.clone();
        tokio::task::spawn_blocking(move || -> std::io::Result<Sha256> {
            let mut h = Sha256::new();
            std::io::copy(&mut std::fs::File::open(p)?, &mut h)?;
            Ok(h)
        })
        .await??
    } else {
        Sha256::new()
    };
    if have < f.size {
        let mut req = client.get(&f.url);
        if have > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let mut resp = req.send().await?.error_for_status()?;
        let mut out = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&part)
            .await?;
        if have > 0 && resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            // The server sent the whole file: start over.
            out.set_len(0).await?;
            done.fetch_sub(have, Ordering::Relaxed);
            have = 0;
            hasher = Sha256::new();
        }
        while let Some(chunk) = resp.chunk().await? {
            if cancel.load(Ordering::Relaxed) {
                out.flush().await?;
                anyhow::bail!("cancelled");
            }
            anyhow::ensure!(
                have + chunk.len() as u64 <= f.size,
                "{}: the server sent more than {} bytes",
                f.path,
                f.size
            );
            out.write_all(&chunk).await?;
            hasher.update(&chunk);
            have += chunk.len() as u64;
            done.fetch_add(chunk.len() as u64, Ordering::Relaxed);
        }
        out.flush().await?;
        out.sync_all().await?;
    }
    anyhow::ensure!(have == f.size, "{}: got {have} of {} bytes", f.path, f.size);
    let got = hex(&hasher.finalize());
    if got != f.sha256.to_ascii_lowercase() {
        let _ = tokio::fs::remove_file(&part).await;
        done.fetch_sub(have, Ordering::Relaxed);
        anyhow::bail!("{}: sha256 is {got}, expected {}", f.path, f.sha256);
    }
    tokio::fs::rename(&part, blob).await?;
    Ok(())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Remove a model from the hub cache (`POST /models/delete`); false if it wasn't there.
pub fn delete(model: &str) -> std::io::Result<bool> {
    let dir = repo_dir(model);
    if !dir.exists() {
        return Ok(false);
    }
    std::fs::remove_dir_all(dir)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str) -> File {
        File {
            path: path.into(),
            url: "https://weights.example/x".into(),
            sha256: "a".repeat(64),
            size: 1,
        }
    }

    fn spec(model: &str, files: Vec<File>) -> Spec {
        Spec {
            model: model.into(),
            revision: None,
            files,
        }
    }

    #[test]
    fn refuses_paths_that_escape_the_cache() {
        assert!(spec(
            "Qwen/Qwen3-0.6B",
            vec![file("config.json"), file("sub/x.json")]
        )
        .check()
        .is_ok());
        for m in ["Qwen", "../x", "a/b/c", "a/..", "a/b c", "/a"] {
            assert!(spec(m, vec![file("x")]).check().is_err(), "{m}");
        }
        for p in ["../x", "/etc/passwd", "a/../../x", "a//b", ""] {
            assert!(spec("a/b", vec![file(p)]).check().is_err(), "{p}");
        }
        let mut f = file("x");
        f.url = "file:///etc/passwd".into();
        assert!(spec("a/b", vec![f]).check().is_err());
        let mut f = file("x");
        f.sha256 = "zz".into();
        assert!(spec("a/b", vec![f]).check().is_err());
        assert!(spec("a/b", vec![file("x"), file("x")]).check().is_err());
        assert!(spec("a/b", vec![]).check().is_err());
    }

    #[test]
    fn revision_is_stable_without_one() {
        let a = spec("a/b", vec![file("x")]);
        assert_eq!(a.revision(), a.revision());
        assert_eq!(a.revision().len(), 40);
        let mut b = spec("a/b", vec![file("x")]);
        b.revision = Some("abc123".into());
        assert_eq!(b.revision(), "abc123");
    }
}
