//! The node API's shape, through the real server, without a GPU or a model: `/node` behind the
//! API key, `/models/load` refusing what isn't on disk or doesn't fit, `/models/unload`, and
//! requests turned away while no model is loaded.

use serde_json::Value;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;
use tang_compute::CpuDevice;
use tang_llm::node::Hardware;
use tang_llm::server::{serve, Options};
use tang_llm::Engine;

const KEY: &str = "test-key";

/// Start a server with no model on a free port; its address.
fn start() -> String {
    start_with_images(false)
}

fn start_with_images(images: bool) -> String {
    let port = std::net::TcpListener::bind("127.0.0.1:0")
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
            // Too little for any real model.
            free_bytes: 1 << 20,
        }),
    };
    std::thread::spawn(move || {
        let loader = |spec: &str| -> anyhow::Result<Engine<CpuDevice>> {
            anyhow::bail!("no loads in this test ({spec})")
        };
        if images {
            tang_llm::server::serve_with_images(
                opts,
                loader,
                Some((
                    std::env::temp_dir().join(format!("tang-image-missing-{port}")),
                    || Ok(CpuDevice::new()),
                )),
            )
        } else {
            serve(opts, loader)
        }
    });
    for _ in 0..200 {
        if TcpStream::connect(&addr).is_ok() {
            return addr;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("server didn't start");
}

/// One HTTP/1.1 request; the status and the body as JSON (null if it isn't).
fn call(addr: &str, method: &str, path: &str, key: Option<&str>, body: &str) -> (u16, Value) {
    let mut s = TcpStream::connect(addr).unwrap();
    let auth = key.map_or(String::new(), |k| format!("Authorization: Bearer {k}\r\n"));
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n{auth}\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    let status = out[9..12].parse().unwrap();
    let (head, rest) = out.split_once("\r\n\r\n").unwrap();
    // Chunked bodies: the JSON is between the first size line and the terminator.
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

#[test]
fn node_reports_its_shape() {
    let addr = start();
    let (status, _) = call(&addr, "GET", "/node", None, "");
    assert_eq!(status, 401, "/node wants the key");
    let (status, v) = call(&addr, "GET", "/node", Some(KEY), "");
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["schema"], tang_llm::node::SCHEMA);
    let id = v["node_id"].as_str().unwrap();
    assert!(
        id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit()),
        "{id}"
    );
    assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
    let hw = &v["hardware"];
    assert_eq!(hw["kind"], "cpu");
    assert_eq!(hw["name"], "test");
    assert_eq!(hw["unified_memory"], true);
    assert_eq!(hw["total_bytes"], 8u64 << 30);
    assert_eq!(hw["free_bytes"], 1 << 20);
    assert_eq!(v["models"]["loaded"], serde_json::json!([]));
    assert_eq!(v["models"]["loading"], Value::Null);
    for m in v["models"]["on_disk"].as_array().unwrap() {
        assert!(m["id"].is_string() && m["path"].is_string(), "{m}");
        assert!(m["size_bytes"].as_u64().unwrap() > 0, "{m}");
        assert_eq!(m["loaded"], false);
    }
    assert_eq!(v["queue"]["running"], serde_json::json!([]));
    assert_eq!(v["queue"]["waiting"], serde_json::json!([]));
    // The id is the machine's, not the process's.
    let (_, again) = call(&addr, "GET", "/node", Some(KEY), "");
    assert_eq!(again["node_id"], v["node_id"]);
}

#[test]
fn image_worker_shares_authentication_and_reports_failed_loads_without_poisoning() {
    let addr = start_with_images(true);
    let request = r#"{"model":"z-image-turbo","prompt":"frog","size":"64x64","steps":8}"#;
    assert_eq!(call(&addr, "GET", "/health", None, "").0, 200);
    assert_eq!(
        call(&addr, "POST", "/v1/images/generations", None, request).0,
        401
    );
    for _ in 0..2 {
        let (status, value) = call(&addr, "POST", "/v1/images/generations", Some(KEY), request);
        assert_eq!(status, 503, "{value}");
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("loading image model"));
    }
    let (status, value) = call(&addr, "GET", "/v1/models", Some(KEY), "");
    assert_eq!(status, 200);
    assert_eq!(value["data"][0]["type"], "image");
    assert_eq!(value["data"][0]["resident"], false);
    let (status, value) = call(&addr, "GET", "/node", Some(KEY), "");
    assert_eq!(status, 200);
    assert_eq!(value["image_model"]["id"], "z-image-turbo");
    assert_eq!(value["image_model"]["resident"], false);
    let invalid = request.replace("64x64", "65x64");
    assert_eq!(
        call(&addr, "POST", "/v1/images/generations", Some(KEY), &invalid).0,
        400
    );
}

#[test]
fn loads_and_requests_without_a_model() {
    let addr = start();
    let (status, v) = call(&addr, "POST", "/models/unload", Some(KEY), "");
    assert_eq!((status, &v["unloaded"]), (200, &Value::Null), "{v}");
    let (status, v) = call(
        &addr,
        "POST",
        "/models/load",
        Some(KEY),
        r#"{"model": "nobody/no-such-model"}"#,
    );
    assert_eq!(status, 404, "{v}");
    let (status, _) = call(&addr, "POST", "/models/load", Some(KEY), r#"{"x": 1}"#);
    assert_eq!(status, 400);
    // Any model that's on disk needs more than the 1 MiB this node says is free.
    if let Some(m) = tang_llm::node::models_on_disk().first() {
        let body = format!(r#"{{"model": "{}"}}"#, m.id);
        let (status, v) = call(&addr, "POST", "/models/load", Some(KEY), &body);
        assert_eq!(status, 507, "{v}");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("free"), "{msg}");
    }
    let chat = r#"{"messages": [{"role": "user", "content": "hi"}]}"#;
    let (status, _) = call(&addr, "POST", "/v1/chat/completions", Some(KEY), chat);
    assert_eq!(status, 503);
    let (status, v) = call(&addr, "GET", "/v1/models", Some(KEY), "");
    assert_eq!((status, &v["data"]), (200, &serde_json::json!([])));
}
