//! `POST /models/pull`, `DELETE /models/pull` and `POST /models/delete` through the real server,
//! against a small local file server (Range requests, slow files): files land in the Hugging
//! Face cache layout, `.part` files resume, sha256 mismatches fail, pulls that don't fit are
//! refused up front, and `/node` reports `pulling` and `last_pull`.
//!
//! Its own test binary: it points `HF_HUB_CACHE` at a temporary directory for the process.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tang_compute::CpuDevice;
use tang_llm::node::Hardware;
use tang_llm::server::{serve, Options};
use tang_llm::Engine;

const KEY: &str = "test-key";

/// The hub cache every test in this binary shares (set before any server starts).
fn hub() -> &'static PathBuf {
    static HUB: OnceLock<PathBuf> = OnceLock::new();
    HUB.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("tang-pull-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // SAFETY: set once, before this binary starts any server thread that reads it.
        unsafe { std::env::set_var("HF_HUB_CACHE", &dir) };
        dir
    })
}

fn start() -> String {
    hub();
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let addr = format!("127.0.0.1:{port}");
    let opts = Options {
        addr: addr.clone(),
        key: Some(KEY.into()),
        model: None,
        dtype: tang_llm::Dtype::Bf16,
        hardware: Arc::new(|| Hardware {
            kind: "cpu",
            name: "test".into(),
            unified: true,
            total_bytes: 8 << 30,
            free_bytes: 1 << 20,
        }),
    };
    std::thread::spawn(move || {
        serve(opts, |spec: &str| -> anyhow::Result<Engine<CpuDevice>> {
            anyhow::bail!("no loads in this test ({spec})")
        })
    });
    for _ in 0..200 {
        if TcpStream::connect(&addr).is_ok() {
            return addr;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("server didn't start");
}

fn call(addr: &str, method: &str, path: &str, body: &str) -> (u16, Value) {
    let mut s = TcpStream::connect(addr).unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\nAuthorization: Bearer {KEY}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    let status = out[9..12].parse().unwrap();
    let (head, rest) = out.split_once("\r\n\r\n").unwrap();
    let body = if head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        rest.split_once("\r\n")
            .map_or("", |x| x.1)
            .trim_end_matches("\r\n0\r\n\r\n")
    } else {
        rest
    };
    (status, serde_json::from_str(body).unwrap_or(Value::Null))
}

/// A file server: `/<name>` serves its bytes (honouring `Range: bytes=N-`), `slow` ones a
/// little at a time. It records each request's Range header.
struct Files {
    addr: String,
    ranges: Arc<Mutex<Vec<(String, Option<String>)>>>,
}

fn files(content: HashMap<String, (Vec<u8>, bool)>) -> Files {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap().to_string();
    let ranges = Arc::new(Mutex::new(Vec::new()));
    let seen = ranges.clone();
    let content = Arc::new(content);
    std::thread::spawn(move || {
        for s in l.incoming() {
            let Ok(mut s) = s else { continue };
            let content = content.clone();
            let seen = seen.clone();
            std::thread::spawn(move || {
                let mut r = BufReader::new(s.try_clone().unwrap());
                let mut line = String::new();
                r.read_line(&mut line).unwrap();
                let path = line
                    .split(' ')
                    .nth(1)
                    .unwrap_or("/")
                    .trim_start_matches('/');
                let path = path.to_string();
                let mut range = None;
                loop {
                    let mut h = String::new();
                    if r.read_line(&mut h).unwrap_or(0) == 0 || h == "\r\n" {
                        break;
                    }
                    if let Some(v) = h.to_ascii_lowercase().strip_prefix("range:") {
                        range = Some(v.trim().to_string());
                    }
                }
                seen.lock().unwrap().push((path.clone(), range.clone()));
                let Some((bytes, slow)) = content.get(&path) else {
                    let _ = write!(s, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
                    return;
                };
                let from: usize = range
                    .as_deref()
                    .and_then(|r| r.strip_prefix("bytes="))
                    .and_then(|r| r.trim_end_matches('-').parse().ok())
                    .unwrap_or(0);
                let body = &bytes[from.min(bytes.len())..];
                let status = if range.is_some() {
                    "206 Partial Content"
                } else {
                    "200 OK"
                };
                if write!(
                    s,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .is_err()
                {
                    return;
                }
                for chunk in body.chunks(if *slow { 1024 } else { 1 << 20 }) {
                    if s.write_all(chunk).is_err() {
                        return;
                    }
                    if *slow {
                        std::thread::sleep(Duration::from_millis(20));
                    }
                }
            });
        }
    });
    Files { addr, ranges }
}

fn sha(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

/// A tiny MLX-4-bit-style Qwen3 config (so the memory check is just the file's bytes).
fn config() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model_type": "qwen3",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": true,
        "quantization": { "group_size": 64, "bits": 4 },
    }))
    .unwrap()
}

/// A checkpoint's files: name -> bytes.
fn checkpoint(weights: usize) -> Vec<(String, Vec<u8>)> {
    vec![
        ("config.json".into(), config()),
        ("tokenizer.json".into(), b"{}".to_vec()),
        ("tokenizer_config.json".into(), b"{}".to_vec()),
        (
            "model.safetensors".into(),
            (0..weights).map(|i| (i * 7 % 251) as u8).collect(),
        ),
    ]
}

/// The pull body for `files` served at `srv`, under `prefix/`.
fn body(model: &str, srv: &Files, prefix: &str, files: &[(String, Vec<u8>)]) -> String {
    json!({
        "model": model,
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "files": files.iter().map(|(p, b)| json!({
            "path": p, "url": format!("http://{}/{prefix}{p}", srv.addr), "sha256": sha(b), "size": b.len(),
        })).collect::<Vec<_>>(),
    })
    .to_string()
}

fn serve_files(prefix: &str, files: &[(String, Vec<u8>)], slow: bool) -> Files {
    self::files(
        files
            .iter()
            .map(|(p, b)| (format!("{prefix}{p}"), (b.clone(), slow)))
            .collect(),
    )
}

/// Poll `/node` until nothing is pulling; `models` from the last read.
fn wait_pulled(addr: &str) -> Value {
    for _ in 0..1000 {
        let (_, v) = call(addr, "GET", "/node", "");
        if v["models"]["pulling"].is_null() {
            return v["models"].clone();
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("pull didn't finish");
}

fn repo(model: &str) -> PathBuf {
    hub().join(format!("models--{}", model.replace('/', "--")))
}

#[test]
fn pulls_into_the_hub_cache_and_deletes() {
    let addr = start();
    let ck = checkpoint(300_000);
    let srv = serve_files("a/", &ck, false);
    let model = "test-org/pull-ok";
    let (st, v) = call(&addr, "POST", "/models/pull", &body(model, &srv, "a/", &ck));
    assert_eq!(st, 202, "{v}");
    assert_eq!(v["pulling"], model);
    let models = wait_pulled(&addr);
    assert_eq!(models["last_pull"]["model"], model, "{models}");
    assert_eq!(models["last_pull"]["ok"], true, "{models}");
    let dir = repo(model);
    let rev = std::fs::read_to_string(dir.join("refs/main")).unwrap();
    assert_eq!(rev, "0123456789abcdef0123456789abcdef01234567");
    for (p, b) in &ck {
        let link = dir.join("snapshots").join(&rev).join(p);
        assert!(std::fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(&std::fs::read(&link).unwrap(), b, "{p}");
    }
    assert!(!std::fs::read_dir(dir.join("blobs")).unwrap().any(|e| e
        .unwrap()
        .path()
        .extension()
        .is_some_and(|x| x == "part")));
    // It's on disk, as `/models/load` would find it.
    let on_disk = models["on_disk"].as_array().unwrap();
    let m = on_disk.iter().find(|m| m["id"] == model).expect("on_disk");
    assert_eq!(m["size_bytes"], 300_000);
    // Pulling it again downloads nothing.
    let before = srv.ranges.lock().unwrap().len();
    let (st, _) = call(&addr, "POST", "/models/pull", &body(model, &srv, "a/", &ck));
    assert_eq!(st, 202);
    assert_eq!(wait_pulled(&addr)["last_pull"]["ok"], true);
    // (config.json is fetched for the fit check.)
    assert!(srv.ranges.lock().unwrap().len() <= before + 1);

    let del = json!({ "model": model }).to_string();
    let (st, v) = call(&addr, "POST", "/models/delete", &del);
    assert_eq!((st, &v["deleted"]), (200, &json!(model)), "{v}");
    assert!(!dir.exists());
    assert_eq!(call(&addr, "POST", "/models/delete", &del).0, 404);
    assert_eq!(
        call(&addr, "POST", "/models/delete", r#"{"model": "../etc"}"#).0,
        400
    );
}

#[test]
fn resumes_a_part_file_with_a_range_request() {
    let addr = start();
    let ck = checkpoint(200_000);
    let srv = serve_files("r/", &ck, false);
    let model = "test-org/pull-resume";
    let weights = &ck[3].1;
    let blobs = repo(model).join("blobs");
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::write(
        blobs.join(format!("{}.part", sha(weights))),
        &weights[..123_456],
    )
    .unwrap();
    let (st, v) = call(&addr, "POST", "/models/pull", &body(model, &srv, "r/", &ck));
    assert_eq!(st, 202, "{v}");
    let models = wait_pulled(&addr);
    assert_eq!(models["last_pull"]["ok"], true, "{models}");
    let ranges = srv.ranges.lock().unwrap().clone();
    let asked: Vec<_> = ranges
        .iter()
        .filter(|(p, _)| p == "r/model.safetensors")
        .collect();
    assert_eq!(asked.len(), 1, "{ranges:?}");
    assert_eq!(asked[0].1.as_deref(), Some("bytes=123456-"));
    assert_eq!(&std::fs::read(blobs.join(sha(weights))).unwrap(), weights);
}

#[test]
fn a_wrong_sha256_fails_the_pull() {
    let addr = start();
    let ck = checkpoint(10_000);
    let srv = serve_files("s/", &ck, false);
    let model = "test-org/pull-badsha";
    let mut b: Value = serde_json::from_str(&body(model, &srv, "s/", &ck)).unwrap();
    b["files"][3]["sha256"] = json!("0".repeat(64));
    let (st, v) = call(&addr, "POST", "/models/pull", &b.to_string());
    assert_eq!(st, 202, "{v}");
    let models = wait_pulled(&addr);
    let last = &models["last_pull"];
    assert_eq!(last["ok"], false, "{models}");
    assert!(last["error"].as_str().unwrap().contains("sha256"), "{last}");
    assert!(!repo(model).join("refs/main").exists());
    assert!(!models["on_disk"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m["id"] == model));
}

#[test]
fn refuses_what_wont_fit_or_is_malformed() {
    let addr = start();
    let ck = checkpoint(1000);
    let srv = serve_files("f/", &ck, false);
    // Memory: the weights' bytes (MLX 4-bit) beyond the node's 8 GiB.
    let mut b: Value = serde_json::from_str(&body("test-org/too-big", &srv, "f/", &ck)).unwrap();
    b["files"][3]["size"] = json!(9u64 << 30);
    let (st, v) = call(&addr, "POST", "/models/pull", &b.to_string());
    assert_eq!(st, 507, "{v}");
    assert!(
        v["error"]["message"].as_str().unwrap().contains("memory"),
        "{v}"
    );
    // Disk: no config (so no memory check), a file bigger than any disk.
    let huge = json!({
        "model": "test-org/no-room",
        "files": [{ "path": "big.bin", "url": format!("http://{}/f/x", srv.addr), "sha256": "a".repeat(64), "size": 1u64 << 60 }],
    });
    let (st, v) = call(&addr, "POST", "/models/pull", &huge.to_string());
    assert_eq!(st, 507, "{v}");
    assert!(
        v["error"]["message"].as_str().unwrap().contains("disk"),
        "{v}"
    );
    for bad in [
        r#"{"model": "x"}"#.to_string(),
        json!({"model": "../x", "files": [{"path": "a", "url": "http://x/a", "sha256": "a".repeat(64), "size": 1}]}).to_string(),
        json!({"model": "a/b", "files": [{"path": "../../a", "url": "http://x/a", "sha256": "a".repeat(64), "size": 1}]}).to_string(),
    ] {
        assert_eq!(call(&addr, "POST", "/models/pull", &bad).0, 400, "{bad}");
    }
    // Nothing left running.
    let (_, v) = call(&addr, "GET", "/node", "");
    assert!(v["models"]["pulling"].is_null(), "{v}");
    assert_eq!(call(&addr, "DELETE", "/models/pull", "").0, 404);
}

#[test]
fn one_pull_at_a_time_and_cancel_keeps_the_part_file() {
    let addr = start();
    // 1024 bytes every 20 ms: ~10 s for the whole file, so it's still going when we look.
    let ck = checkpoint(500_000);
    let srv = serve_files("c/", &ck, true);
    let model = "test-org/pull-cancel";
    let (st, v) = call(&addr, "POST", "/models/pull", &body(model, &srv, "c/", &ck));
    assert_eq!(st, 202, "{v}");
    let total = ck.iter().map(|(_, b)| b.len() as u64).sum::<u64>();
    // Progress shows up in /node.
    let mut seen = 0;
    for _ in 0..500 {
        let (_, v) = call(&addr, "GET", "/node", "");
        let p = &v["models"]["pulling"];
        assert_eq!(p["model"], model, "{v}");
        assert_eq!(p["total_bytes"], total);
        seen = p["done_bytes"].as_u64().unwrap();
        if seen > 50_000 {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(seen > 50_000 && seen < total, "{seen}");
    let other = serve_files("o/", &checkpoint(10), false);
    let (st, v) = call(
        &addr,
        "POST",
        "/models/pull",
        &body("test-org/other", &other, "o/", &checkpoint(10)),
    );
    assert_eq!(st, 409, "{v}");
    let del = json!({ "model": model }).to_string();
    assert_eq!(call(&addr, "POST", "/models/delete", &del).0, 409);
    let (st, v) = call(&addr, "DELETE", "/models/pull", "");
    assert_eq!((st, &v["cancelled"]), (200, &json!(model)), "{v}");
    let models = wait_pulled(&addr);
    assert_eq!(models["last_pull"]["cancelled"], true, "{models}");
    assert_eq!(models["last_pull"]["ok"], false);
    let part = repo(model)
        .join("blobs")
        .join(format!("{}.part", sha(&ck[3].1)));
    let kept = std::fs::metadata(&part).unwrap().len();
    assert!(kept > 0 && kept < 500_000, "{kept}");
    assert!(!repo(model).join("refs/main").exists());
}
