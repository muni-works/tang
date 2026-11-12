//! Content-addressed prefix-state store for `flash serve` (the disk tier under the host-RAM
//! prefix states).
//!
//! A saved prefix state is the recurrent (GDN/conv) running state after `n` tokens plus the
//! positional (K/V) rows `0..n`. The rows are split into the segments the host copies already
//! have (`Snap::pos`: each state only copies the rows past the longest saved state it extends),
//! and each segment is one file named by what it holds:
//!
//! - `kv/{H(seed, ids[..to])}-{from}-{to}.kv`: rows `from..to` of a prefix that hashes to `H`.
//!   Two states that share a prefix share its segment files: a tools block saved by forty
//!   sessions is on disk once.
//! - `states/{n}-{H(seed, ids[..n])}.state`: the running state, the `n` token ids (checked
//!   against the prompt on load) and the list of segments.
//!
//! `H` is FNV-1a over the token ids, started from `seed` (0 for a state any tenant may use, the
//! tenant's hash otherwise: see `serve::Share`), so it streams: the prompt's prefix hashes at every
//! saved length come out of one pass, and [`KvStore::find`] is a hash-map probe per distinct
//! saved length instead of a directory scan.
//!
//! Two tiers: `hot` (local NVMe, `TANG_FLASH_STATE_DIR`, `TANG_FLASH_DISK_GB` budget, default 8)
//! and an optional `cold` (`TANG_KV_COLD_DIR`, say the `/models` network volume,
//! `TANG_KV_COLD_GB`, default 64). Over budget, the least recently used files move hot -> cold,
//! and cold ones are deleted. Files are written on a thread from the host copies (the serve loop
//! only pays the device copies it already made), complete or not at all (write + rename). An
//! index entry whose files vanished (budget, TTL, another pod) is a miss, not an error.

use super::engine::{Positional, RunningState};
use anyhow::{ensure, Context, Result};
use std::collections::{BTreeSet, HashMap};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const STATE_MAGIC: &[u8; 8] = b"TANGRS03";
const KV_MAGIC: &[u8; 8] = b"TANGKV01";

/// FNV-1a over token ids from a seed; `seed == 0` is the plain FNV basis.
pub fn seeded_hash(seed: u64, ids: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ seed;
    for &i in ids {
        h = step(h, i);
    }
    h
}

fn step(mut h: u64, id: u32) -> u64 {
    for b in id.to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// A state read back: tokens, running state, segments covering `0..n`, and what the read cost.
pub struct Loaded {
    pub tokens: Vec<u32>,
    pub state: RunningState,
    pub pos: Vec<Positional>,
    pub bytes: usize,
    pub secs: f64,
    pub cold: bool,
}

pub struct KvStore {
    hot: PathBuf,
    cold: Option<PathBuf>,
    hot_budget: u64,
    cold_budget: u64,
    /// Prefix hash -> length, for every state file seen (both tiers).
    index: HashMap<u64, usize>,
    lens: BTreeSet<usize>,
}

impl KvStore {
    /// Open (creating) the tiers and index what is already there: a restarted pod restores the
    /// prefixes it saw before.
    pub fn open(hot: PathBuf, cold: Option<PathBuf>, hot_budget: u64, cold_budget: u64) -> Self {
        let mut s = KvStore {
            hot,
            cold,
            hot_budget,
            cold_budget,
            index: HashMap::new(),
            lens: BTreeSet::new(),
        };
        for dir in s.tiers() {
            let _ = std::fs::create_dir_all(dir.join("states"));
            let _ = std::fs::create_dir_all(dir.join("kv"));
        }
        let found: Vec<(u64, usize)> = s
            .tiers()
            .iter()
            .flat_map(|d| std::fs::read_dir(d.join("states")).into_iter().flatten())
            .flatten()
            .filter_map(|e| parse_state_name(&e.file_name().to_string_lossy()))
            .collect();
        for (h, n) in found {
            s.note(h, n);
        }
        s
    }

    /// From the environment: `hot` (the state dir), `TANG_KV_COLD_DIR`, the budgets.
    /// `TANG_KV_STORE=off` keeps nothing on disk.
    pub fn from_env(hot: &Path) -> Option<Self> {
        if std::env::var("TANG_KV_STORE").is_ok_and(|v| v == "off") {
            return None;
        }
        let gb = |k: &str, d: f64| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .map_or(d, |g| g)
                * 1e9
        };
        let cold = std::env::var("TANG_KV_COLD_DIR")
            .ok()
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);
        Some(Self::open(
            hot.to_path_buf(),
            cold,
            gb("TANG_FLASH_DISK_GB", 8.0) as u64,
            gb("TANG_KV_COLD_GB", 64.0) as u64,
        ))
    }

    fn tiers(&self) -> Vec<PathBuf> {
        std::iter::once(self.hot.clone())
            .chain(self.cold.clone())
            .collect()
    }

    fn note(&mut self, h: u64, n: usize) {
        self.index.insert(h, n);
        self.lens.insert(n);
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Is the state after `ids` (under `seed`) indexed?
    pub fn has(&self, seed: u64, ids: &[u32]) -> bool {
        self.index.get(&seeded_hash(seed, ids)) == Some(&ids.len())
    }

    /// The longest indexed state that `ids[..n]`, `n <= limit`, matches under one of `seeds`
    /// (by hash; [`load`](Self::load) checks the ids). Returns (n, seed).
    pub fn find(&self, ids: &[u32], limit: usize, seeds: &[u64]) -> Option<(usize, u64)> {
        let limit = limit.min(ids.len());
        let mut best: Option<(usize, u64)> = None;
        for &seed in seeds {
            let mut h = seeded_hash(seed, &[]);
            let mut at = 0;
            for &n in self.lens.range(1..=limit) {
                for &id in &ids[at..n] {
                    h = step(h, id);
                }
                at = n;
                if self.index.get(&h) == Some(&n) && best.map_or(true, |b| n > b.0) {
                    best = Some((n, seed));
                }
            }
        }
        best
    }

    /// Write the state after `ids` (under `seed`): `pos` covers rows `0..ids.len()` in order.
    /// Indexed now; the files are written on a thread (segments first, then the state file).
    pub fn save(
        &mut self,
        seed: u64,
        ids: &[u32],
        state: Arc<RunningState>,
        pos: Vec<Arc<Positional>>,
    ) {
        let h = seeded_hash(seed, ids);
        if self.index.get(&h) == Some(&ids.len()) {
            for d in self.tiers() {
                crate::retention::touch(&d.join("states").join(state_name(ids.len(), h)));
            }
            return;
        }
        self.note(h, ids.len());
        let ids = ids.to_vec();
        let (hot, cold) = (self.hot.clone(), self.cold.clone());
        let (hb, cb) = (self.hot_budget, self.cold_budget);
        std::thread::spawn(move || {
            if let Err(e) = write_state(&hot, cold.as_deref(), seed, &ids, &state, &pos) {
                eprintln!("tang-llm: saving prefix state at {}: {e:#}", ids.len());
            }
            enforce(&hot, cold.as_deref(), hb, cb);
        });
    }

    /// Read the state after `ids[..n]` under `seed`. Drops the index entry when its files are
    /// gone or don't match (a hash collision, another model's state), and returns `None`.
    pub fn load(
        &mut self,
        ids: &[u32],
        n: usize,
        seed: u64,
        shape: &[usize],
        gdn_layers: usize,
    ) -> Option<Loaded> {
        let h = seeded_hash(seed, &ids[..n]);
        let t = Instant::now();
        let mut cold = false;
        let mut read = || -> Result<(Vec<u32>, RunningState, Vec<Positional>, usize)> {
            let name = state_name(n, h);
            let (path, is_cold) = self
                .tiers()
                .into_iter()
                .enumerate()
                .map(|(i, d)| (d.join("states").join(&name), i > 0))
                .find(|(p, _)| p.exists())
                .context("state file gone")?;
            cold = is_cold;
            let b = read_file(&path)?;
            let mut r = Reader { b: &b, at: 0 };
            ensure!(r.take(8)? == STATE_MAGIC, "not a state file");
            ensure!(r.word()? == n, "length");
            let tokens: Vec<u32> = r
                .take(4 * n)?
                .chunks(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            ensure!(tokens == ids[..n], "tokens differ");
            ensure!(r.word()? == shape.len(), "parts");
            let mut parts = Vec::with_capacity(shape.len());
            for &want in shape {
                ensure!(r.word()? == want, "part size");
                parts.push(
                    r.take(4 * want)?
                        .chunks(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect(),
                );
            }
            let nseg = r.word()?;
            let mut pos = Vec::with_capacity(nseg);
            let mut bytes = b.len();
            let mut at = 0;
            for _ in 0..nseg {
                let (from, to) = (r.word()?, r.word()?);
                ensure!(from == at && to > from && to <= n, "segments");
                at = to;
                let kh = seeded_hash(seed, &ids[..to]);
                let kname = kv_name(kh, from, to);
                let kp = self
                    .tiers()
                    .into_iter()
                    .map(|d| d.join("kv").join(&kname))
                    .find(|p| p.exists())
                    .context("segment file gone")?;
                let kb = read_file(&kp)?;
                bytes += kb.len();
                crate::retention::touch(&kp);
                let mut k = Reader { b: &kb, at: 0 };
                ensure!(k.take(8)? == KV_MAGIC, "not a segment file");
                ensure!(k.word()? == from && k.word()? == to, "segment range");
                let nbuf = k.word()?;
                let mut bufs = Vec::with_capacity(nbuf);
                for _ in 0..nbuf {
                    let len = k.word()?;
                    bufs.push(k.take(len)?.to_vec());
                }
                pos.push(Positional::from_parts(from, to, bufs));
            }
            ensure!(at == n, "segments end at {at}, not {n}");
            crate::retention::touch(&path);
            Ok((
                tokens,
                RunningState::from_parts(n, gdn_layers, parts)?,
                pos,
                bytes,
            ))
        };
        match read() {
            Ok((tokens, state, pos, bytes)) => Some(Loaded {
                tokens,
                state,
                pos,
                bytes,
                secs: t.elapsed().as_secs_f64(),
                cold,
            }),
            Err(e) => {
                eprintln!("tang-llm: prefix state at {n} not read from disk: {e:#}");
                self.index.remove(&h);
                None
            }
        }
    }

    /// Delete files unused for the TTL (both tiers) and drop their index entries.
    pub fn expire(&mut self, ttl: Option<std::time::Duration>) {
        if ttl.is_none() {
            return;
        }
        for d in self.tiers() {
            for sub in ["states", "kv"] {
                for ent in std::fs::read_dir(d.join(sub))
                    .into_iter()
                    .flatten()
                    .flatten()
                {
                    let p = ent.path();
                    if crate::retention::stale_file(&p, ttl) {
                        if let Some((h, _)) = parse_state_name(&ent.file_name().to_string_lossy()) {
                            self.index.remove(&h);
                        }
                        let _ = std::fs::remove_file(&p);
                    }
                }
            }
        }
    }
}

fn state_name(n: usize, h: u64) -> String {
    format!("{n}-{h:016x}.state")
}

fn kv_name(h: u64, from: usize, to: usize) -> String {
    format!("{h:016x}-{from}-{to}.kv")
}

/// `{n}-{hash}.state` -> (hash, n).
fn parse_state_name(name: &str) -> Option<(u64, usize)> {
    let (n, h) = name.strip_suffix(".state")?.split_once('-')?;
    Some((u64::from_str_radix(h, 16).ok()?, n.parse().ok()?))
}

fn read_file(p: &Path) -> Result<Vec<u8>> {
    let mut b = Vec::new();
    std::fs::File::open(p)?.read_to_end(&mut b)?;
    Ok(b)
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self.b.get(self.at..self.at + n).context("file is short")?;
        self.at += n;
        Ok(s)
    }

    fn word(&mut self) -> Result<usize> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?) as usize)
    }
}

fn f32_bytes(p: &[f32]) -> &[u8] {
    // SAFETY: any byte view of a valid `[f32]` is valid `u8`s (little-endian hosts only; see
    // the cfg check at the call site).
    unsafe { std::slice::from_raw_parts(p.as_ptr().cast::<u8>(), std::mem::size_of_val(p)) }
}

/// A temporary file extension no other writer (thread, process, pod on the shared volume) uses.
fn tmp_ext() -> String {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    format!("tmp{}-{n}-{t}", std::process::id())
}

/// Write `bytes` to `path` complete or not at all.
fn put(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension(tmp_ext());
    std::fs::File::create(&tmp)
        .and_then(|mut f| f.write_all(bytes))
        .and_then(|_| std::fs::rename(&tmp, path))
        .with_context(|| format!("writing {}", path.display()))
}

fn write_state(
    hot: &Path,
    cold: Option<&Path>,
    seed: u64,
    ids: &[u32],
    state: &RunningState,
    pos: &[Arc<Positional>],
) -> Result<()> {
    ensure!(cfg!(target_endian = "little"), "big-endian host");
    let n = ids.len();
    let mut segs = Vec::with_capacity(pos.len());
    let mut at = 0;
    for p in pos {
        ensure!(p.from == at && p.to > p.from, "segments not contiguous");
        at = p.to;
        let name = kv_name(seeded_hash(seed, &ids[..p.to]), p.from, p.to);
        segs.push((p.from, p.to));
        // Content-addressed: already written (by this state's base, or another tenant-free
        // state with the same prefix), in either tier.
        let exists = [Some(hot), cold]
            .into_iter()
            .flatten()
            .any(|d| d.join("kv").join(&name).exists());
        if exists {
            continue;
        }
        let mut b = Vec::with_capacity(p.bytes() + 64);
        b.extend_from_slice(KV_MAGIC);
        for w in [p.from, p.to, p.parts().len()] {
            b.extend_from_slice(&(w as u64).to_le_bytes());
        }
        for part in p.parts() {
            b.extend_from_slice(&(part.len() as u64).to_le_bytes());
            b.extend_from_slice(part);
        }
        put(&hot.join("kv").join(&name), &b)?;
    }
    ensure!(at == n, "segments end at {at}, not {n}");
    let mut b = Vec::with_capacity(state.bytes() + 4 * n + 64);
    b.extend_from_slice(STATE_MAGIC);
    b.extend_from_slice(&(n as u64).to_le_bytes());
    for &i in ids {
        b.extend_from_slice(&i.to_le_bytes());
    }
    let parts = state.parts();
    b.extend_from_slice(&(parts.len() as u64).to_le_bytes());
    for p in parts {
        b.extend_from_slice(&(p.len() as u64).to_le_bytes());
        b.extend_from_slice(f32_bytes(p));
    }
    b.extend_from_slice(&(segs.len() as u64).to_le_bytes());
    for (f, t) in segs {
        b.extend_from_slice(&(f as u64).to_le_bytes());
        b.extend_from_slice(&(t as u64).to_le_bytes());
    }
    put(
        &hot.join("states")
            .join(state_name(n, seeded_hash(seed, ids))),
        &b,
    )
}

/// Keep each tier under its budget, least recently used first: hot files move to cold (when
/// there is one), cold files are deleted.
fn enforce(hot: &Path, cold: Option<&Path>, hot_budget: u64, cold_budget: u64) {
    let files = |d: &Path| {
        let mut v: Vec<(std::time::SystemTime, u64, PathBuf)> = ["states", "kv"]
            .iter()
            .flat_map(|s| std::fs::read_dir(d.join(s)).into_iter().flatten())
            .flatten()
            .filter(|e| {
                let p = e.path();
                p.extension().is_some_and(|x| x == "state" || x == "kv")
            })
            .filter_map(|e| {
                let m = e.metadata().ok()?;
                Some((m.modified().ok()?, m.len(), e.path()))
            })
            .collect();
        v.sort();
        v
    };
    let shrink = |d: &Path, budget: u64, to: Option<&Path>| {
        let fs = files(d);
        let mut total: u64 = fs.iter().map(|f| f.1).sum();
        for (_, len, p) in fs {
            if total <= budget {
                break;
            }
            if let (Some(to), Some(sub), Some(name)) =
                (to, p.parent().and_then(|x| x.file_name()), p.file_name())
            {
                let dst = to.join(sub).join(name);
                if std::fs::rename(&p, &dst).is_err() {
                    let tmp = dst.with_extension(tmp_ext());
                    let _ = std::fs::copy(&p, &tmp).and_then(|_| std::fs::rename(&tmp, &dst));
                }
            }
            let _ = std::fs::remove_file(&p);
            total = total.saturating_sub(len);
        }
    };
    shrink(hot, hot_budget, cold);
    if let Some(c) = cold {
        shrink(c, cold_budget, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(n: usize, x: f32) -> Arc<RunningState> {
        // One GDN layer (two parts), no ring buffers, a PLE ring.
        Arc::new(
            RunningState::from_parts(n, 1, vec![vec![x; 6], vec![x + 1.0; 3], vec![x; 2]]).unwrap(),
        )
    }

    fn seg(from: usize, to: usize) -> Arc<Positional> {
        let rows = |k: u8| (from..to).map(|i| i as u8 ^ k).collect::<Vec<u8>>();
        Arc::new(Positional::from_parts(from, to, vec![rows(0), rows(0x5a)]))
    }

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("tang-kvstore-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn wait_for(p: &Path) {
        for _ in 0..500 {
            if std::fs::read_dir(p).map_or(0, |r| r.count()) > 0 {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    fn files(d: &Path) -> usize {
        std::fs::read_dir(d).map_or(0, |r| {
            r.flatten()
                .filter(|e| !e.file_name().to_string_lossy().contains(".tmp"))
                .count()
        })
    }

    #[test]
    fn round_trip_shared_segments_and_restart() {
        let hot = dir("rt");
        let shape = [6, 3, 2];
        let ids: Vec<u32> = (0..40).collect();
        let mut s = KvStore::open(hot.clone(), None, u64::MAX, u64::MAX);
        let base = seg(0, 16);
        s.save(0, &ids[..16], state(16, 1.0), vec![base.clone()]);
        // Extends the first: shares its segment file.
        s.save(0, &ids[..30], state(30, 2.0), vec![base, seg(16, 30)]);
        for _ in 0..500 {
            if files(&hot.join("states")) == 2 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(
            files(&hot.join("kv")),
            2,
            "the shared segment is stored once"
        );
        // A prompt that runs past both finds the longer; a shorter limit the shorter.
        let mut prompt = ids.clone();
        prompt.push(999);
        assert_eq!(s.find(&prompt, 39, &[0]).map(|f| f.0), Some(30));
        assert_eq!(s.find(&prompt, 20, &[0]).map(|f| f.0), Some(16));
        // Other tenants' seeds don't see untagged... and vice versa.
        assert_eq!(s.find(&prompt, 39, &[7]), None);
        // A restart indexes what's on disk.
        let mut s = KvStore::open(hot.clone(), None, u64::MAX, u64::MAX);
        let l = s.load(&prompt, 30, 0, &shape, 1).expect("loads");
        assert_eq!(l.tokens, ids[..30]);
        assert_eq!(l.pos.len(), 2);
        assert_eq!(l.pos[1].parts()[1], seg(16, 30).parts()[1]);
        assert_eq!(l.state.parts()[1], &[3.0; 3][..]);
        // A different prompt with a colliding length doesn't load.
        let mut other = prompt.clone();
        other[3] = 77;
        assert_eq!(s.find(&other, 39, &[0]), None);
        let _ = std::fs::remove_dir_all(&hot);
    }

    #[test]
    fn budget_moves_to_cold_then_restores() {
        let (hot, cold) = (dir("hot"), dir("cold"));
        let ids: Vec<u32> = (100..164).collect();
        let mut s = KvStore::open(hot.clone(), Some(cold.clone()), 1, u64::MAX);
        s.save(5, &ids[..64], state(64, 4.0), vec![seg(0, 64)]);
        wait_for(&cold.join("states"));
        for _ in 0..500 {
            if files(&hot.join("states")) + files(&hot.join("kv")) == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let mut prompt = ids.clone();
        prompt.push(1);
        assert_eq!(s.find(&prompt, 64, &[0, 5]), Some((64, 5)));
        let l = s.load(&prompt, 64, 5, &[6, 3, 2], 1).expect("from cold");
        assert!(l.cold);
        let _ = std::fs::remove_dir_all(&hot);
        let _ = std::fs::remove_dir_all(&cold);
    }
}
