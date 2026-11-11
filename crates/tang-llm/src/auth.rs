//! Who may use a node: the paired clients' keys, the node's own identity, and the gate in front
//! of every route but `/health` and `/pair/*`. See frog's `docs/node-protocol.md`, sections 1–2.
//!
//! - Client keys live in `<tang cache>/node-keys.json` (0600) as SHA-256 hashes only, one entry
//!   per paired client. `TANG_API_KEY` / `--api-key-file` count as one more key.
//! - Beyond loopback every gated route needs `Authorization: Bearer <key>`; with no key at all
//!   the node answers `401 {"error": "not paired"}` until a client pairs. On loopback a static
//!   key gates as it always did, and without one nothing is asked (paired keys are accepted but
//!   not required there, so a local frog keeps working after this machine has paired others).
//! - The node's identity is an Ed25519 key pair in `<tang cache>/node-identity` (0600); clients
//!   pin its fingerprint.

use axum::{
    extract::{Request as HttpRequest, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// One paired client, as `node-keys.json` keeps it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyEntry {
    pub client_id: String,
    pub client_name: String,
    /// Lowercase hex SHA-256 of the client's key.
    pub key_sha256: String,
    /// Unix seconds.
    pub paired_at: u64,
}

/// `<tang cache>/node-keys.json`.
pub fn keys_path() -> Option<PathBuf> {
    crate::node::cache_dir().map(|d| d.join("node-keys.json"))
}

/// `<tang cache>/node-identity`.
pub fn identity_path() -> Option<PathBuf> {
    crate::node::cache_dir().map(|d| d.join("node-identity"))
}

pub fn sha256_hex(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn random_bytes<const N: usize>() -> [u8; N] {
    let mut b = [0u8; N];
    getrandom::getrandom(&mut b).expect("the OS random number generator");
    b
}

/// Read the paired clients (an absent or unreadable file is no clients).
pub fn read_keys(path: &Path) -> Vec<KeyEntry> {
    std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

/// Write `data` to `path` with mode 0600: whole, then renamed over the old file.
pub fn write_private(path: &Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension(format!(
        "tmp{}-{}",
        std::process::id(),
        hex(&random_bytes::<8>())
    ));
    {
        use std::io::Write;
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(&tmp)?;
        f.write_all(data)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

pub fn write_keys(path: &Path, keys: &[KeyEntry]) -> std::io::Result<()> {
    write_private(
        path,
        &serde_json::to_vec_pretty(keys).expect("keys serialize"),
    )
}

/// Drop `client_id`'s entry (or every entry, for `all`); returns how many went.
pub fn remove_keys(path: &Path, client_id: &str) -> std::io::Result<usize> {
    let keys = read_keys(path);
    let kept: Vec<KeyEntry> = keys
        .iter()
        .filter(|k| client_id != "all" && k.client_id != client_id)
        .cloned()
        .collect();
    let gone = keys.len() - kept.len();
    if gone > 0 {
        write_keys(path, &kept)?;
    }
    Ok(gone)
}

/// The node's Ed25519 key pair.
pub struct Identity {
    pub signing: ed25519_dalek::SigningKey,
}

impl Identity {
    /// Load `<tang cache>/node-identity`, making it on first run. Without a cache directory the
    /// identity lasts the process.
    pub fn load_or_create(path: Option<&Path>) -> anyhow::Result<Identity> {
        if let Some(p) = path {
            if let Ok(b) = std::fs::read(p) {
                let secret: [u8; 32] = b
                    .as_slice()
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("{} is not a 32-byte key", p.display()))?;
                return Ok(Identity {
                    signing: ed25519_dalek::SigningKey::from_bytes(&secret),
                });
            }
        }
        let secret = random_bytes::<32>();
        if let Some(p) = path {
            // Two servers starting at once: the first rename wins and both read it back.
            write_private(p, &secret)?;
            return Identity::load_or_create(Some(p));
        }
        Ok(Identity {
            signing: ed25519_dalek::SigningKey::from_bytes(&secret),
        })
    }

    pub fn public_key(&self) -> [u8; 32] {
        self.signing.verifying_key().to_bytes()
    }

    /// First 16 bytes of SHA-256 of the public key, lowercase hex.
    pub fn fingerprint(&self) -> String {
        fingerprint(&self.public_key())
    }
}

pub fn fingerprint(public_key: &[u8; 32]) -> String {
    hex(&Sha256::digest(public_key)[..16])
}

/// The fingerprint in groups of four, as users see it.
pub fn grouped(fp: &str) -> String {
    fp.as_bytes()
        .chunks(4)
        .map(|c| std::str::from_utf8(c).unwrap_or(""))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Who a request authenticated as, put on gated requests.
#[derive(Debug, Clone, PartialEq)]
pub enum Caller {
    /// No auth asked (loopback without a static key).
    Open,
    /// The static key (`TANG_API_KEY` / `--api-key-file`).
    Static,
    /// A paired client.
    Client(String),
}

/// Why a key was turned away: the `401` body's `error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Denied {
    MissingKey,
    BadKey,
    NotPaired,
}

impl Denied {
    pub fn reason(self) -> &'static str {
        match self {
            Denied::MissingKey => "missing key",
            Denied::BadKey => "bad key",
            Denied::NotPaired => "not paired",
        }
    }
}

pub fn unauthorized(reason: &str) -> Response {
    (StatusCode::UNAUTHORIZED, Json(json!({ "error": reason }))).into_response()
}

/// The node's keys and how strict to be with them.
pub struct Auth {
    /// The server listens beyond loopback.
    pub beyond: bool,
    pub static_key: Option<String>,
    /// `node-keys.json`; `None` keeps paired clients in memory only.
    pub path: Option<PathBuf>,
    /// The paired clients, and the file's modification time they were read at (so
    /// `tang-llm unpair` takes effect on a running server).
    clients: Mutex<(Option<SystemTime>, Vec<KeyEntry>)>,
}

impl Auth {
    pub fn new(beyond: bool, static_key: Option<String>, path: Option<PathBuf>) -> Auth {
        let a = Auth {
            beyond,
            static_key: static_key.filter(|k| !k.is_empty()),
            path,
            clients: Mutex::new((None, Vec::new())),
        };
        a.refresh();
        a
    }

    fn mtime(&self) -> Option<SystemTime> {
        self.path
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok())
            .and_then(|m| m.modified().ok())
    }

    /// Re-read the file if it changed since the last read.
    fn refresh(&self) {
        let Some(path) = &self.path else { return };
        let now = self.mtime();
        let mut c = self.clients.lock().unwrap_or_else(|e| e.into_inner());
        if c.0 != now || now.is_none() {
            *c = (now, read_keys(path));
        }
    }

    pub fn clients(&self) -> Vec<KeyEntry> {
        self.refresh();
        self.clients
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .1
            .clone()
    }

    /// Requests must present a key.
    pub fn required(&self) -> bool {
        self.beyond || self.static_key.is_some()
    }

    /// Check a request's bearer key (`None`: no `Authorization: Bearer` header).
    pub fn check(&self, given: Option<&str>) -> Result<Caller, Denied> {
        let clients = self.clients();
        let matched = given.and_then(|g| self.matches(g, &clients));
        if !self.required() {
            return Ok(matched.unwrap_or(Caller::Open));
        }
        if self.static_key.is_none() && clients.is_empty() {
            return Err(Denied::NotPaired);
        }
        match (given, matched) {
            (None, _) => Err(Denied::MissingKey),
            (Some(_), None) => Err(Denied::BadKey),
            (Some(_), Some(c)) => Ok(c),
        }
    }

    fn matches(&self, given: &str, clients: &[KeyEntry]) -> Option<Caller> {
        if self
            .static_key
            .as_ref()
            .is_some_and(|k| same(given.as_bytes(), k.as_bytes()))
        {
            return Some(Caller::Static);
        }
        let h = sha256_hex(given.as_bytes());
        // Every entry compared, in constant time each.
        let mut found = None;
        for k in clients {
            if same(h.as_bytes(), k.key_sha256.as_bytes()) {
                found = Some(Caller::Client(k.client_id.clone()));
            }
        }
        found
    }

    /// Store a client's key (replacing any earlier key under the same `client_id`).
    pub fn add_client(&self, entry: KeyEntry) -> std::io::Result<()> {
        let mut c = self.clients.lock().unwrap_or_else(|e| e.into_inner());
        let mut keys = match &self.path {
            Some(p) => read_keys(p),
            None => c.1.clone(),
        };
        keys.retain(|k| k.client_id != entry.client_id);
        keys.push(entry);
        if let Some(p) = &self.path {
            write_keys(p, &keys)?;
        }
        *c = (self.mtime(), keys);
        Ok(())
    }

    /// Drop a client; whether it was there.
    pub fn remove_client(&self, client_id: &str) -> std::io::Result<bool> {
        let mut c = self.clients.lock().unwrap_or_else(|e| e.into_inner());
        let mut keys = match &self.path {
            Some(p) => read_keys(p),
            None => c.1.clone(),
        };
        let before = keys.len();
        keys.retain(|k| k.client_id != client_id);
        let gone = keys.len() != before;
        if gone {
            if let Some(p) = &self.path {
                write_keys(p, &keys)?;
            }
        }
        *c = (self.mtime(), keys);
        Ok(gone)
    }
}

/// The `Authorization: Bearer` key of a request, if any.
pub fn bearer(headers: &axum::http::HeaderMap) -> Option<&str> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim)
}

/// The gate in front of every route but `/health` and `/pair/*`.
pub async fn gate(State(auth): State<Arc<Auth>>, mut req: HttpRequest, next: Next) -> Response {
    match auth.check(bearer(req.headers())) {
        Ok(caller) => {
            req.extensions_mut().insert(caller);
            next.run(req).await
        }
        Err(d) => unauthorized(d.reason()),
    }
}

pub fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("tang-auth-{name}-{}", hex(&random_bytes::<8>())))
    }

    fn entry(id: &str, key: &str) -> KeyEntry {
        KeyEntry {
            client_id: id.into(),
            client_name: "mac".into(),
            key_sha256: sha256_hex(key.as_bytes()),
            paired_at: 1,
        }
    }

    #[test]
    fn the_matrix() {
        use Denied::*;
        let k = Some("tnk_a");
        // loopback, nothing configured: open
        let a = Auth::new(false, None, None);
        assert_eq!(a.check(None), Ok(Caller::Open));
        assert_eq!(a.check(Some("junk")), Ok(Caller::Open));
        // loopback with a static key: as before
        let a = Auth::new(false, Some("s3".into()), None);
        assert_eq!(a.check(None), Err(MissingKey));
        assert_eq!(a.check(Some("no")), Err(BadKey));
        assert_eq!(a.check(Some("s3")), Ok(Caller::Static));
        // loopback with only paired clients: open, the client still recognized
        let a = Auth::new(false, None, None);
        a.add_client(entry("c1", "tnk_a")).unwrap();
        assert_eq!(a.check(None), Ok(Caller::Open));
        assert_eq!(a.check(k), Ok(Caller::Client("c1".into())));
        // beyond loopback, no keys: not paired, whatever is sent
        let a = Auth::new(true, None, None);
        assert_eq!(a.check(None), Err(NotPaired));
        assert_eq!(a.check(k), Err(NotPaired));
        // beyond loopback, paired
        a.add_client(entry("c1", "tnk_a")).unwrap();
        assert_eq!(a.check(None), Err(MissingKey));
        assert_eq!(a.check(Some("tnk_b")), Err(BadKey));
        assert_eq!(a.check(k), Ok(Caller::Client("c1".into())));
        assert!(a.remove_client("c1").unwrap());
        assert_eq!(a.check(k), Err(NotPaired));
        // beyond loopback, static key only, and both
        let a = Auth::new(true, Some("s3".into()), None);
        assert_eq!(a.check(None), Err(MissingKey));
        assert_eq!(a.check(Some("s3")), Ok(Caller::Static));
        a.add_client(entry("c1", "tnk_a")).unwrap();
        assert_eq!(a.check(k), Ok(Caller::Client("c1".into())));
        assert_eq!(a.check(Some("s3")), Ok(Caller::Static));
    }

    #[test]
    fn keys_file_is_private_hashed_and_watched() {
        let p = tmp("keys").join("node-keys.json");
        let a = Auth::new(true, None, Some(p.clone()));
        a.add_client(entry("c1", "tnk_secret")).unwrap();
        a.add_client(entry("c2", "tnk_other")).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(!text.contains("tnk_secret"));
        assert!(text.contains(&sha256_hex(b"tnk_secret")));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&p).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        // Another process (`tang-llm unpair`) edits the file: the running gate follows.
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(remove_keys(&p, "c1").unwrap(), 1);
        assert_eq!(a.check(Some("tnk_secret")), Err(Denied::BadKey));
        assert!(a.check(Some("tnk_other")).is_ok());
        assert_eq!(remove_keys(&p, "all").unwrap(), 1);
        assert_eq!(a.check(Some("tnk_other")), Err(Denied::NotPaired));
        let _ = std::fs::remove_dir_all(p.parent().unwrap());
    }

    #[test]
    fn identity_is_kept_and_fingerprinted() {
        let p = tmp("id").join("node-identity");
        let a = Identity::load_or_create(Some(&p)).unwrap();
        let b = Identity::load_or_create(Some(&p)).unwrap();
        assert_eq!(a.public_key(), b.public_key());
        let fp = a.fingerprint();
        assert_eq!(fp.len(), 32);
        assert_eq!(fp, hex(&Sha256::digest(a.public_key())[..16]));
        assert_eq!(grouped("3f2a91c0aaaa").as_str(), "3f2a 91c0 aaaa");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&p).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        let _ = std::fs::remove_dir_all(p.parent().unwrap());
    }
}
