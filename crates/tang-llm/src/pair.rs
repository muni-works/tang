//! Pairing a client with a 6-digit code (frog's `docs/node-protocol.md`, section 4).
//!
//! The node opens a 120 s window (`tang-llm pair`, or `POST /pair/open` from loopback) and shows
//! a code. The client runs SPAKE2 (Ed25519 group, `A = "frog-client"`, `B = "tang-node"`,
//! password = the code) through `/pair/start`; both sides derive confirmation and wrapping keys
//! with HKDF-SHA256, the client proves the code with `/pair/finish`, and the node answers with
//! its own confirmation and the new API key sealed with AES-256-GCM. Five wrong finishes close
//! the window; one exchange runs at a time.
// handlers return a ready axum `Response` as their error (large, by design)
#![allow(clippy::result_large_err)]

use crate::auth::{random_bytes, sha256_hex, unauthorized, Auth, Identity, KeyEntry};
use aes_gcm::aead::{Aead, KeyInit, Payload};
use axum::{
    extract::{ConnectInfo, Path as UrlPath, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use hmac::Mac;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use spake2::{Ed25519Group, Identity as SpakeId, Password, Spake2};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const WINDOW: Duration = Duration::from_secs(120);
pub const MAX_FAILURES: u32 = 5;
/// An exchange the client walked away from stops blocking others after this long.
pub const EXCHANGE_TIMEOUT: Duration = Duration::from_secs(30);
pub const ID_CLIENT: &[u8] = b"frog-client";
pub const ID_NODE: &[u8] = b"tang-node";

type HmacSha256 = hmac::Hmac<Sha256>;

/// The keys both sides derive from SPAKE2's output.
pub struct Keys {
    pub confirm_client: [u8; 32],
    pub confirm_node: [u8; 32],
    pub wrap: [u8; 32],
}

pub fn derive(k: &[u8]) -> Keys {
    let hk = hkdf::Hkdf::<Sha256>::new(None, k);
    let get = |info: &[u8]| {
        let mut out = [0u8; 32];
        hk.expand(info, &mut out)
            .expect("32 bytes is a valid HKDF length");
        out
    };
    Keys {
        confirm_client: get(b"frog-pair confirm client"),
        confirm_node: get(b"frog-pair confirm node"),
        wrap: get(b"frog-pair wrap"),
    }
}

/// `SHA-256(client_id || 0x00 || client msg || node msg || identity || node_id)`.
pub fn transcript(
    client_id: &str,
    client_msg: &[u8],
    node_msg: &[u8],
    identity: &[u8],
    node_id: &str,
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(client_id.as_bytes());
    h.update([0u8]);
    h.update(client_msg);
    h.update(node_msg);
    h.update(identity);
    h.update(node_id.as_bytes());
    h.finalize().into()
}

pub fn hmac(key: &[u8; 32], msg: &[u8]) -> [u8; 32] {
    let mut m = <HmacSha256 as Mac>::new_from_slice(key).expect("any key length");
    m.update(msg);
    m.finalize().into_bytes().into()
}

/// `nonce(12) || AES-256-GCM(key, nonce, plaintext, aad)`.
pub fn seal(key: &[u8; 32], plaintext: &[u8], aad: &[u8]) -> Vec<u8> {
    let nonce = random_bytes::<12>();
    let c = aes_gcm::Aes256Gcm::new(key.into())
        .encrypt(
            (&nonce).into(),
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .expect("AES-GCM sealing");
    [nonce.as_slice(), &c].concat()
}

pub fn open(key: &[u8; 32], boxed: &[u8], aad: &[u8]) -> Option<Vec<u8>> {
    if boxed.len() < 12 {
        return None;
    }
    let (nonce, c) = boxed.split_at(12);
    aes_gcm::Aes256Gcm::new(key.into())
        .decrypt(nonce.into(), Payload { msg: c, aad })
        .ok()
}

/// A fresh API key: `tnk_` + 32 random bytes, base64url without padding.
pub fn new_api_key() -> String {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    format!("tnk_{}", URL_SAFE_NO_PAD.encode(random_bytes::<32>()))
}

/// A uniformly random 6-digit code.
pub fn new_code() -> String {
    loop {
        let x = u32::from_le_bytes(random_bytes::<4>());
        // Reject the tail that would bias the remainder.
        if x < u32::MAX - u32::MAX % 1_000_000 {
            return format!("{:06}", x % 1_000_000);
        }
    }
}

struct Exchange {
    pair_id: String,
    client_id: String,
    client_name: String,
    keys: Keys,
    transcript: [u8; 32],
    started: Instant,
}

struct Window {
    code: String,
    expires: Instant,
    failures: u32,
    exchange: Option<Exchange>,
}

/// The node's pairing state, shared by the routes and the Bonjour advert.
pub struct Pairing {
    pub auth: Arc<Auth>,
    pub identity: Arc<Identity>,
    pub node_id: String,
    pub name: String,
    window: Mutex<Option<Window>>,
}

pub enum StartError {
    NoWindow,
    Busy,
}

pub enum FinishError {
    WrongCode,
    Expired,
}

impl Pairing {
    pub fn new(auth: Arc<Auth>, identity: Arc<Identity>, node_id: String, name: String) -> Pairing {
        Pairing {
            auth,
            identity,
            node_id,
            name,
            window: Mutex::new(None),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Window>> {
        self.window.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Open (or replace) the window with a fresh code; the code.
    pub fn open_window(&self) -> String {
        self.open_with(new_code(), WINDOW)
    }

    /// Open a window with a given code and length (tests).
    pub fn open_with(&self, code: String, len: Duration) -> String {
        *self.lock() = Some(Window {
            code: code.clone(),
            expires: Instant::now() + len,
            failures: 0,
            exchange: None,
        });
        code
    }

    /// A window is open (for Bonjour's `pair`).
    pub fn is_open(&self) -> bool {
        let mut w = self.lock();
        if w.as_ref().is_some_and(|w| Instant::now() >= w.expires) {
            *w = None;
        }
        w.is_some()
    }

    /// `/pair/start`: the node's SPAKE2 message and the exchange's id.
    pub fn start(
        &self,
        client_id: &str,
        client_name: &str,
        client_msg: &[u8],
    ) -> Result<(String, Vec<u8>), StartError> {
        let mut guard = self.lock();
        let now = Instant::now();
        let w = match guard.as_mut() {
            Some(w) if now < w.expires => w,
            _ => {
                *guard = None;
                return Err(StartError::NoWindow);
            }
        };
        if w.exchange
            .as_ref()
            .is_some_and(|e| now.duration_since(e.started) < EXCHANGE_TIMEOUT)
        {
            return Err(StartError::Busy);
        }
        let (state, node_msg) = Spake2::<Ed25519Group>::start_b(
            &Password::new(w.code.as_bytes()),
            &SpakeId::new(ID_CLIENT),
            &SpakeId::new(ID_NODE),
        );
        // A malformed client message can't be told from a wrong code here; it fails at finish
        // with a key the client can't match. Use a random key so it can't succeed.
        let k = state
            .finish(client_msg)
            .unwrap_or_else(|_| random_bytes::<32>().to_vec());
        let identity = self.identity.public_key();
        let t = transcript(client_id, client_msg, &node_msg, &identity, &self.node_id);
        let pair_id = crate::auth::hex(&random_bytes::<16>());
        w.exchange = Some(Exchange {
            pair_id: pair_id.clone(),
            client_id: client_id.to_string(),
            client_name: client_name.to_string(),
            keys: derive(&k),
            transcript: t,
            started: now,
        });
        Ok((pair_id, node_msg))
    }

    /// `/pair/finish`: on the right code, the node's confirmation and the sealed key.
    pub fn finish(
        &self,
        pair_id: &str,
        confirm: &[u8],
    ) -> Result<([u8; 32], Vec<u8>), FinishError> {
        let mut guard = self.lock();
        let now = Instant::now();
        let w = match guard.as_mut() {
            Some(w) if now < w.expires => w,
            _ => {
                *guard = None;
                return Err(FinishError::Expired);
            }
        };
        let Some(ex) = w.exchange.take_if(|e| e.pair_id == pair_id) else {
            return Err(FinishError::Expired);
        };
        let want = hmac(&ex.keys.confirm_client, &ex.transcript);
        if !crate::auth::same(&want, confirm) {
            w.failures += 1;
            if w.failures >= MAX_FAILURES {
                *guard = None;
            }
            return Err(FinishError::WrongCode);
        }
        let key = new_api_key();
        let entry = KeyEntry {
            client_id: ex.client_id.clone(),
            client_name: ex.client_name.clone(),
            key_sha256: sha256_hex(key.as_bytes()),
            paired_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
        };
        // One pairing per window.
        *guard = None;
        drop(guard);
        if let Err(e) = self.auth.add_client(entry) {
            eprintln!("tang-llm: storing the paired key: {e}");
            return Err(FinishError::Expired);
        }
        eprintln!("tang-llm: paired {} ({})", ex.client_name, ex.client_id);
        let node_confirm = hmac(&ex.keys.confirm_node, &ex.transcript);
        let key_box = seal(&ex.keys.wrap, key.as_bytes(), &ex.transcript);
        Ok((node_confirm, key_box))
    }

    /// What `/pair/open` and `tang-llm pair` show.
    pub fn open_json(&self) -> Value {
        let code = self.open_window();
        eprintln!("tang-llm: pairing window open for {}s", WINDOW.as_secs());
        json!({
            "code": code,
            "expires_in": WINDOW.as_secs(),
            "fingerprint": self.identity.fingerprint(),
        })
    }
}

fn err(status: StatusCode, msg: &str) -> Response {
    (status, Json(json!({ "error": msg }))).into_response()
}

fn field<'a>(body: &'a Value, k: &str) -> Result<&'a str, Response> {
    body[k]
        .as_str()
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, &format!("{k} must be a string")))
}

fn b64_field(body: &Value, k: &str) -> Result<Vec<u8>, Response> {
    B64.decode(field(body, k)?)
        .map_err(|_| err(StatusCode::BAD_REQUEST, &format!("{k} must be base64")))
}

async fn open_route(
    State(p): State<Arc<Pairing>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
) -> Response {
    if !peer.ip().is_loopback() {
        return err(
            StatusCode::FORBIDDEN,
            "pairing opens only from this machine",
        );
    }
    Json(p.open_json()).into_response()
}

async fn start_route(State(p): State<Arc<Pairing>>, Json(body): Json<Value>) -> Response {
    let r = (|| {
        let client_id = field(&body, "client_id")?;
        let client_name = body["client_name"].as_str().unwrap_or("");
        let msg = b64_field(&body, "spake")?;
        if client_id.is_empty() {
            return Err(err(StatusCode::BAD_REQUEST, "client_id must not be empty"));
        }
        Ok((client_id.to_string(), client_name.to_string(), msg))
    })();
    let (client_id, client_name, msg) = match r {
        Ok(x) => x,
        Err(r) => return r,
    };
    match p.start(&client_id, &client_name, &msg) {
        Ok((pair_id, node_msg)) => Json(json!({
            "pair_id": pair_id,
            "spake": B64.encode(node_msg),
            "node_id": p.node_id,
            "identity": B64.encode(p.identity.public_key()),
            "name": p.name,
        }))
        .into_response(),
        Err(StartError::NoWindow) => err(StatusCode::FORBIDDEN, "no pairing window"),
        Err(StartError::Busy) => err(StatusCode::CONFLICT, "pairing busy"),
    }
}

async fn finish_route(State(p): State<Arc<Pairing>>, Json(body): Json<Value>) -> Response {
    let pair_id = match field(&body, "pair_id") {
        Ok(x) => x.to_string(),
        Err(r) => return r,
    };
    let confirm = match b64_field(&body, "confirm") {
        Ok(x) => x,
        Err(r) => return r,
    };
    match p.finish(&pair_id, &confirm) {
        Ok((c, key_box)) => Json(json!({
            "confirm": B64.encode(c),
            "key_box": B64.encode(key_box),
        }))
        .into_response(),
        Err(FinishError::WrongCode) => err(StatusCode::UNAUTHORIZED, "wrong code"),
        Err(FinishError::Expired) => err(StatusCode::GONE, "pairing expired"),
    }
}

/// `DELETE /pair/<client_id>` with that client's own key.
async fn unpair_route(
    State(p): State<Arc<Pairing>>,
    UrlPath(client_id): UrlPath<String>,
    headers: HeaderMap,
) -> Response {
    let Some(given) = crate::auth::bearer(&headers) else {
        return unauthorized("missing key");
    };
    let h = sha256_hex(given.as_bytes());
    let owns = p.auth.clients().iter().any(|k| {
        k.client_id == client_id && crate::auth::same(h.as_bytes(), k.key_sha256.as_bytes())
    });
    if !owns {
        return unauthorized("bad key");
    }
    match p.auth.remove_client(&client_id) {
        Ok(_) => {
            eprintln!("tang-llm: unpaired {client_id}");
            StatusCode::NO_CONTENT.into_response()
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// `/pair/*`, outside the gate. Serve with `into_make_service_with_connect_info::<SocketAddr>`
/// (`/pair/open` checks the peer is loopback).
pub fn router(p: Arc<Pairing>) -> Router {
    Router::new()
        .route("/pair/open", post(open_route))
        .route("/pair/start", post(start_route))
        .route("/pair/finish", post(finish_route))
        .route("/pair/{client_id}", delete(unpair_route))
        .with_state(p)
}

/// The client's side of pairing (tests, and anyone writing a client in Rust).
pub mod client {
    use super::*;

    /// After `/pair/start`'s request is built: the client's SPAKE2 message and state.
    pub struct Started {
        pub msg: Vec<u8>,
        state: Spake2<Ed25519Group>,
    }

    pub fn start(code: &str) -> Started {
        let (state, msg) = Spake2::<Ed25519Group>::start_a(
            &Password::new(code.as_bytes()),
            &SpakeId::new(ID_CLIENT),
            &SpakeId::new(ID_NODE),
        );
        Started { msg, state }
    }

    /// What the client holds between `/pair/start`'s answer and `/pair/finish`'s.
    pub struct Pending {
        pub keys: Keys,
        pub transcript: [u8; 32],
    }

    impl Started {
        /// With the node's answer: the confirmation to send.
        pub fn answer(
            self,
            client_id: &str,
            node_msg: &[u8],
            identity: &[u8],
            node_id: &str,
        ) -> anyhow::Result<(Pending, [u8; 32])> {
            let k = self
                .state
                .finish(node_msg)
                .map_err(|e| anyhow::anyhow!("spake2: {e:?}"))?;
            let t = transcript(client_id, &self.msg, node_msg, identity, node_id);
            let keys = derive(&k);
            let confirm = hmac(&keys.confirm_client, &t);
            Ok((
                Pending {
                    keys,
                    transcript: t,
                },
                confirm,
            ))
        }
    }

    impl Pending {
        /// Check the node's confirmation, then unseal the key.
        pub fn key(&self, node_confirm: &[u8], key_box: &[u8]) -> anyhow::Result<String> {
            let want = hmac(&self.keys.confirm_node, &self.transcript);
            anyhow::ensure!(
                crate::auth::same(&want, node_confirm),
                "node confirmation mismatch"
            );
            let k = open(&self.keys.wrap, key_box, &self.transcript)
                .ok_or_else(|| anyhow::anyhow!("key_box does not open"))?;
            Ok(String::from_utf8(k)?)
        }
    }
}

/// Digits five rows tall, for `tang-llm pair`.
fn big(code: &str) -> String {
    const F: [[&str; 5]; 10] = [
        ["###", "# #", "# #", "# #", "###"],
        ["  #", "  #", "  #", "  #", "  #"],
        ["###", "  #", "###", "#  ", "###"],
        ["###", "  #", "###", "  #", "###"],
        ["# #", "# #", "###", "  #", "  #"],
        ["###", "#  ", "###", "  #", "###"],
        ["###", "#  ", "###", "# #", "###"],
        ["###", "  #", "  #", "  #", "  #"],
        ["###", "# #", "###", "# #", "###"],
        ["###", "# #", "###", "  #", "###"],
    ];
    (0..5)
        .map(|row| {
            let mut line = String::from("    ");
            for (i, c) in code.bytes().enumerate() {
                if i == 3 {
                    line.push_str("   ");
                }
                let d = (c - b'0') as usize;
                line.push_str(
                    &F[d.min(9)][row]
                        .replace('#', "\u{2588}\u{2588}")
                        .replace(' ', "  "),
                );
                line.push_str("  ");
            }
            line.trim_end().to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `tang-llm pair [--port P]`: open a window on the server running on this machine and show
/// its code and fingerprint.
pub fn cli_pair(args: &[String]) -> anyhow::Result<()> {
    use anyhow::Context;
    let mut port = 8911u16;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--port" => port = it.next().context("--port P")?.parse()?,
            other => anyhow::bail!("usage: tang-llm pair [--port P] (not {other})"),
        }
    }
    let url = format!("http://127.0.0.1:{port}/pair/open");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let (status, text) = rt.block_on(async {
        let r = reqwest::Client::new()
            .post(&url)
            .send()
            .await
            .with_context(|| {
                format!("no tang-llm server at {url} (start one with `tang-llm serve`)")
            })?;
        anyhow::Ok((r.status(), r.text().await?))
    })?;
    let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    anyhow::ensure!(status.is_success(), "{url}: {status} {text}");
    let code = v["code"].as_str().context("no code in the answer")?;
    let fp = v["fingerprint"].as_str().unwrap_or("");
    println!(
        "\nPair with this code (valid {}s):\n",
        v["expires_in"].as_u64().unwrap_or(120)
    );
    println!("{}\n", big(code));
    println!("    {} {}\n", &code[..3], &code[3..]);
    println!("Node fingerprint: {}", crate::auth::grouped(fp));
    Ok(())
}

/// `tang-llm unpair <client_id | all>`: drop paired clients (a running server notices).
pub fn cli_unpair(args: &[String]) -> anyhow::Result<()> {
    use anyhow::Context;
    let usage = "usage: tang-llm unpair <client_id | all>";
    let who = args.first().context(usage)?;
    let path =
        crate::auth::keys_path().context("no cache directory (set HOME or TANG_CACHE_DIR)")?;
    if who == "list" || who == "--list" {
        for k in crate::auth::read_keys(&path) {
            println!("{}\t{}\t{}", k.client_id, k.client_name, k.paired_at);
        }
        return Ok(());
    }
    let n = crate::auth::remove_keys(&path, who)?;
    anyhow::ensure!(n > 0 || who == "all", "no paired client {who}");
    println!("unpaired {n} client{}", if n == 1 { "" } else { "s" });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_six_digits() {
        for _ in 0..100 {
            let c = new_code();
            assert_eq!(c.len(), 6);
            assert!(c.bytes().all(|b| b.is_ascii_digit()));
        }
        let k = new_api_key();
        assert!(k.starts_with("tnk_") && k.len() == 4 + 43, "{k}");
    }

    #[test]
    fn seal_opens_only_with_the_same_aad() {
        let key = [7u8; 32];
        let b = seal(&key, b"tnk_x", b"aad");
        assert_eq!(open(&key, &b, b"aad").as_deref(), Some(&b"tnk_x"[..]));
        assert!(open(&key, &b, b"other").is_none());
    }

    #[test]
    fn hkdf_infos_give_distinct_keys() {
        let k = derive(b"spake output");
        assert_ne!(k.confirm_client, k.confirm_node);
        assert_ne!(k.confirm_node, k.wrap);
    }
}
