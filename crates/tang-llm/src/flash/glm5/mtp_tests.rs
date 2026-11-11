//! MTP draft-and-verify on the tiny fixture: the output must equal plain greedy decode for
//! any drafter (real MTP, junk, oracle), and the oracle must be accepted every time.

use super::engine::Glm5Engine;
use super::mtp::{generate, Drafter, Mtp, MtpDrafter};
use anyhow::Result;
use std::path::Path;

const TINY: &str = "/private/tmp/glm5-tiny/glm5-tiny-f32.gguf";

fn tiny() -> Option<String> {
    let p = std::env::var("TANG_GLM5_TINY").unwrap_or_else(|_| TINY.to_string());
    if Path::new(&p).exists() {
        Some(p)
    } else {
        eprintln!("SKIP: no tiny fixture at {p}");
        None
    }
}

const PROMPT: [u32; 6] = [3, 17, 42, 7, 99, 5];

fn plain(p: &str, n: usize) -> Vec<u32> {
    let mut e = Glm5Engine::open(Path::new(p)).unwrap();
    let mut cur = e.prefill(&PROMPT, 4).unwrap();
    let mut g = vec![cur];
    while g.len() < n {
        cur = e.step(cur).unwrap();
        g.push(cur);
    }
    g
}

/// Proposes fixed junk.
struct Junk(u32);
impl Drafter for Junk {
    fn observe(&mut self, _: &Glm5Engine, _: &[f32], _: &[u32]) -> Result<()> {
        Ok(())
    }
    fn draft(&mut self, _: &Glm5Engine, k: usize) -> Result<Vec<u32>> {
        self.0 = self.0.wrapping_mul(31).wrapping_add(7) % 200;
        Ok(vec![self.0; k])
    }
}

/// Knows the answer.
struct Oracle {
    want: Vec<u32>,
    done: usize,
}
impl Drafter for Oracle {
    fn observe(&mut self, _: &Glm5Engine, _: &[f32], next: &[u32]) -> Result<()> {
        if self.done == 0 {
            self.done = 1; // the prefill's token
        } else {
            self.done += next.len();
        }
        Ok(())
    }
    fn draft(&mut self, _: &Glm5Engine, k: usize) -> Result<Vec<u32>> {
        Ok(self.want[self.done..].iter().take(k).copied().collect())
    }
}

#[test]
fn mtp_any_drafter_matches_plain_greedy() {
    let _env = crate::flash::glm5::env_read();
    let Some(p) = tiny() else { return };
    let n = 12;
    let want = plain(&p, n);
    for k in [1, 2, 3] {
        let mut e = Glm5Engine::open(Path::new(&p)).unwrap();
        let mut dr = MtpDrafter::new(Mtp::load(&e).unwrap());
        let (g, st) = generate(&mut e, &mut dr, &PROMPT, n, 4, k).unwrap();
        eprint!("k={k} mtp\n{}", st.report());
        assert_eq!(g, want, "mtp k={k}");

        let mut e = Glm5Engine::open(Path::new(&p)).unwrap();
        let (g, st) = generate(&mut e, &mut Junk(1), &PROMPT, n, 4, k).unwrap();
        eprint!("k={k} junk\n{}", st.report());
        assert_eq!(g, want, "junk k={k}");

        let mut e = Glm5Engine::open(Path::new(&p)).unwrap();
        let mut o = Oracle {
            want: want.clone(),
            done: 0,
        };
        let (g, st) = generate(&mut e, &mut o, &PROMPT, n, 4, k).unwrap();
        eprint!("k={k} oracle\n{}", st.report());
        assert_eq!(g, want, "oracle k={k}");
        assert_eq!(st.rerun, 0, "oracle drafts must all be accepted");
        assert_eq!(st.verify, (n - 1).div_ceil(k + 1), "oracle k={k}: passes");
    }
}

/// The MTP cache rewinds exactly: drafting (chain cells + truncate) leaves the cache as if it
/// never ran, so the same committed cells give the same draft twice.
#[test]
fn mtp_draft_is_side_effect_free() {
    let _env = crate::flash::glm5::env_read();
    let Some(p) = tiny() else { return };
    let mut e = Glm5Engine::open(Path::new(&p)).unwrap();
    let lg = e.window(&PROMPT).unwrap();
    let nv = e.m.hp.n_vocab;
    let cur = super::engine::argmax(&lg[(PROMPT.len() - 1) * nv..]);
    let mut nexts = PROMPT[1..].to_vec();
    nexts.push(cur);
    let mut dr = MtpDrafter::new(Mtp::load(&e).unwrap());
    dr.observe(&e, &e.last_h, &nexts).unwrap();
    let len = dr.mtp.len();
    let a = dr.draft(&e, 3).unwrap();
    assert_eq!(dr.mtp.len(), len);
    let b = dr.draft(&e, 3).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.len(), 3);
}

/// The device MTP block computes what the host reference computes: a GPU engine decodes
/// 64 greedy tokens one window at a time; after each, both blocks run the same
/// teacher-forced cell (h, next). The cell outputs `hn` must agree to bf16-latent noise
/// (rel l2 < 2e-2) and the draft ids must be identical wherever the host's top-2 logit
/// margin is clear of that noise (> 0.05; the tiny random-weight head is near-flat, so
/// near-ties flip on any reordering — the real-weight gate is the pod's debug log).
#[cfg(feature = "cuda")]
#[test]
fn mtp_device_draft_matches_host() {
    let _env = crate::flash::glm5::env_read();
    let Some(p) = tiny() else { return };
    let mut e = Glm5Engine::open(Path::new(&p)).unwrap();
    if !e.use_gpu().unwrap() {
        assert!(std::env::var("TANG_REQUIRE_CUDA").is_err(), "no GPU");
        return;
    }
    let mut dm = Mtp::load(&e).unwrap();
    assert!(dm.on_device(), "device MTP block not built");
    let mut hm = Mtp::load(&e).unwrap().host_only();
    let nv = e.m.hp.n_vocab;
    let lg = e.window(&PROMPT).unwrap();
    let mut cur = super::engine::argmax(&lg[(PROMPT.len() - 1) * nv..]);
    let mut nexts = PROMPT[1..].to_vec();
    nexts.push(cur);
    let mut h = e.last_h.clone();
    let (mut clear, mut same, mut worst) = (0, 0, 0f64);
    for i in 0..64 {
        let (a, b) = (
            dm.cells(&e, &h, &nexts).unwrap(),
            hm.cells(&e, &h, &nexts).unwrap(),
        );
        assert_eq!(dm.len(), hm.len());
        let num: f64 =
            a.h.iter()
                .zip(&b.h)
                .map(|(x, y)| ((x - y) as f64).powi(2))
                .sum();
        let den: f64 = b.h.iter().map(|y| (*y as f64).powi(2)).sum();
        let rel = (num / den.max(1e-30)).sqrt();
        worst = worst.max(rel);
        let n = b.h.len() / nexts.len();
        let lgh = hm.host_logits(&b.h[b.h.len() - n..]);
        let top = super::engine::argmax(&lgh) as usize;
        let second = lgh
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != top)
            .map(|(_, v)| *v)
            .fold(f32::MIN, f32::max);
        let (da, db) = (*a.top.last().unwrap(), *b.top.last().unwrap());
        if lgh[top] - second > 0.05 {
            clear += 1;
            assert_eq!(
                da,
                db,
                "token {i}: clear margin {} but device {da} != host {db}",
                lgh[top] - second
            );
        }
        same += usize::from(da == db);
        let lg = e.window(&[cur]).unwrap();
        cur = super::engine::argmax(&lg[..nv]);
        h = e.last_h.clone();
        nexts = vec![cur];
    }
    eprintln!("mtp device vs host: {same}/64 drafts identical, {clear} with a clear margin (all equal), worst hn rel {worst:.2e}");
    assert!(worst < 2e-2, "device hn rel {worst}");
}

/// The device draft cell (`DraftCell`, one graph per cell) against the host cell on the
/// tiny fixture: teacher-forced observe + 1 draft per step through `MtpDrafter`; ids must
/// agree wherever the host's top-2 margin is clear of reordering noise, and the device
/// cell must actually have replayed (not the host fallback).
#[cfg(feature = "cuda")]
#[test]
fn mtp_device_cell_matches_host() {
    let _env = crate::flash::glm5::env_write();
    let Some(p) = tiny() else { return };
    std::env::set_var("TANG_GLM5_MOE", "fast");
    let mut e = Glm5Engine::open(Path::new(&p)).unwrap();
    if !e.use_gpu().unwrap() {
        std::env::remove_var("TANG_GLM5_MOE");
        assert!(std::env::var("TANG_REQUIRE_CUDA").is_err(), "no GPU");
        return;
    }
    let mut dd = MtpDrafter::new(Mtp::load(&e).unwrap());
    let mut hm = Mtp::load(&e).unwrap().host_only();
    let nv = e.m.hp.n_vocab;
    let lg = e.window(&PROMPT).unwrap();
    let mut cur = super::engine::argmax(&lg[(PROMPT.len() - 1) * nv..]);
    let mut nexts = PROMPT[1..].to_vec();
    nexts.push(cur);
    let mut h = e.last_h.clone();
    let (mut clear, mut same) = (0, 0);
    for i in 0..48 {
        dd.observe(&e, &h, &nexts).unwrap();
        let da = dd.draft(&e, 1).unwrap()[0];
        let b = hm.cells(&e, &h, &nexts).unwrap();
        let n = b.h.len() / nexts.len();
        let lgh = hm.host_logits(&b.h[b.h.len() - n..]);
        let top = super::engine::argmax(&lgh) as usize;
        let second = lgh
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != top)
            .map(|(_, v)| *v)
            .fold(f32::MIN, f32::max);
        let db = *b.top.last().unwrap();
        if lgh[top] - second > 0.05 {
            clear += 1;
            assert_eq!(
                da,
                db,
                "step {i}: clear margin {} but device cell {da} != host {db}",
                lgh[top] - second
            );
        }
        same += usize::from(da == db);
        let lg = e.window(&[cur]).unwrap();
        cur = super::engine::argmax(&lg[..nv]);
        h = e.last_h.clone();
        nexts = vec![cur];
    }
    let replays = dd.mtp.cell_replays();
    std::env::remove_var("TANG_GLM5_MOE");
    eprintln!("mtp device cell vs host: {same}/48 identical, {clear} clear-margin (all equal), {replays} cell replays");
    assert!(
        replays > 0,
        "the device cell never ran (host fallback only)"
    );
}
