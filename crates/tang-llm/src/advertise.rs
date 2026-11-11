//! Bonjour: a node listening beyond loopback says so on the local network as `_frog-node._tcp`,
//! so frog on a Mac nearby finds it without anyone typing an address.
//!
//! The TXT record carries just enough to draw the node before pairing (`/node` itself is
//! behind the API key when there is one): `v` (TXT shape), `id` (the node id), `kind`
//! (`metal`, `cuda` or `cpu`), `gpu` (the device name), `mem` (total VRAM or unified memory,
//! bytes), `unified` (`1` when the GPU shares RAM), `key` (`1` when requests need a key).
//!
//! `TANG_NO_BONJOUR=1` turns it off. Loopback-only servers never advertise: nothing else could
//! reach them.

use crate::node::Hardware;
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};

pub const SERVICE: &str = "_frog-node._tcp.local.";
/// Version of the TXT record's shape.
pub const TXT_VERSION: u32 = 1;

/// Keeps the advertisement up while held; dropping it withdraws it.
pub struct Advert {
    daemon: mdns_sd::ServiceDaemon,
    fullname: String,
}

impl Drop for Advert {
    fn drop(&mut self) {
        if let Ok(rx) = self.daemon.unregister(&self.fullname) {
            let _ = rx.recv_timeout(std::time::Duration::from_millis(500));
        }
        let _ = self.daemon.shutdown();
    }
}

/// Whether a server bound to `addr` is reachable from other machines.
pub fn reachable(addr: &str) -> bool {
    let Ok(mut addrs) = addr.to_socket_addrs() else {
        return false;
    };
    addrs.any(|a: SocketAddr| match a.ip() {
        IpAddr::V4(v4) => !v4.is_loopback(),
        IpAddr::V6(v6) => !v6.is_loopback(),
    })
}

/// The TXT properties for this node.
pub fn txt(id: &str, hw: &Hardware, key: bool) -> Vec<(String, String)> {
    vec![
        ("v".into(), TXT_VERSION.to_string()),
        ("id".into(), id.to_string()),
        ("kind".into(), hw.kind.to_string()),
        // TXT strings cap at 255 bytes each; device names are far shorter, but be sure.
        ("gpu".into(), hw.name.chars().take(64).collect()),
        ("mem".into(), hw.total_bytes.to_string()),
        ("unified".into(), if hw.unified { "1" } else { "0" }.into()),
        ("key".into(), if key { "1" } else { "0" }.into()),
    ]
}

/// The machine's short host name (`mew`, `cams-mbp`), for the instance and host names.
pub fn host_name() -> String {
    let mut buf = [0u8; 256];
    // SAFETY: gethostname writes at most buf.len() bytes into buf.
    let ok = unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) } == 0;
    let raw = if ok {
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        String::from_utf8_lossy(&buf[..end]).into_owned()
    } else {
        String::new()
    };
    let short = raw.split('.').next().unwrap_or("").trim().to_string();
    if short.is_empty() {
        "tang-node".into()
    } else {
        short
    }
}

/// Advertise the server at `addr` (if it's reachable from elsewhere and not turned off).
/// Failures are reported once and never stop the server.
pub fn start(addr: &str, id: &str, hw: &Hardware, key: bool) -> Option<Advert> {
    if std::env::var_os("TANG_NO_BONJOUR").is_some_and(|v| v != "0") || !reachable(addr) {
        return None;
    }
    let port = addr.to_socket_addrs().ok()?.next()?.port();
    let host = host_name();
    let props = txt(id, hw, key);
    let result = (|| -> anyhow::Result<Advert> {
        let daemon = mdns_sd::ServiceDaemon::new()?;
        let info = mdns_sd::ServiceInfo::new(
            SERVICE,
            &host,
            &format!("{host}.local."),
            "",
            port,
            &props[..],
        )?
        .enable_addr_auto();
        let fullname = info.get_fullname().to_string();
        daemon.register(info)?;
        Ok(Advert { daemon, fullname })
    })();
    match result {
        Ok(a) => {
            eprintln!("tang-llm: advertising {host} as {SERVICE} on port {port}");
            Some(a)
        }
        Err(e) => {
            eprintln!("tang-llm: not advertising on Bonjour: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_is_not_reachable() {
        assert!(!reachable("127.0.0.1:8911"));
        assert!(!reachable("[::1]:8911"));
        assert!(reachable("0.0.0.0:8911"));
        assert!(reachable("192.168.2.10:8911"));
    }

    #[test]
    fn txt_carries_the_card() {
        let hw = Hardware {
            kind: "cuda",
            name: "NVIDIA GeForce RTX 3090".into(),
            unified: false,
            total_bytes: 24 << 30,
            free_bytes: 1,
        };
        let t = txt("ab", &hw, true);
        let get = |k: &str| t.iter().find(|(a, _)| a == k).map(|(_, b)| b.as_str());
        assert_eq!(get("v"), Some("1"));
        assert_eq!(get("kind"), Some("cuda"));
        assert_eq!(get("mem"), Some("25769803776"));
        assert_eq!(get("unified"), Some("0"));
        assert_eq!(get("key"), Some("1"));
        assert!(t
            .iter()
            .map(|(k, v)| k.len() + v.len() + 1)
            .all(|n| n <= 255));
    }
}
