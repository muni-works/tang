//! The hand path on [`super::tiny`]'s model: it opens, runs deterministically, the loop matters,
//! a prompt past the window runs, incremental decoding matches a full run, and bad geometry is
//! rejected.

use super::forward::{InklingRef, Session};
use super::tiny::*;
use std::path::PathBuf;

#[test]
fn tiny_model_opens_and_runs_deterministically() {
    let dir = tmp("run");
    let p = write(&dir).unwrap();
    let m = InklingRef::open(&p).unwrap();
    assert_eq!(m.hp.n_layer, N_LAYER);
    assert_eq!(m.hp.n_slot(), N_LAYER + (N_PASSES - 1) * (LOOP_END - LOOP_START));
    assert_eq!(m.hp.layer_phys(N_LAYER), LOOP_START);
    assert_eq!(m.hp.layer_phys(N_LAYER + (LOOP_END - LOOP_START) - 1), LOOP_END - 1);

    let ids = [1u32, 5, 9, 2];
    let a = m.forward(&mut Session::new(&m), &ids, false).unwrap();
    assert_eq!(a.len(), N_VOCAB_UNPADDED);
    assert!(a.iter().all(|v| v.is_finite()), "non-finite logits");
    // Every logit is inside the soft cap.
    assert!(a.iter().all(|v| v.abs() <= 30.0));
    // Same input, same output.
    let b = m.forward(&mut Session::new(&m), &ids, false).unwrap();
    assert_eq!(a, b);
    // All positions when asked.
    let all = m.forward(&mut Session::new(&m), &ids, true).unwrap();
    assert_eq!(all.len(), ids.len() * N_VOCAB_UNPADDED);
    assert_eq!(&all[(ids.len() - 1) * N_VOCAB_UNPADDED..], &a[..]);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn incremental_decoding_matches_a_full_run() {
    // The session keeps the KV and convolution state, so stepping one token at a time must land
    // on the same logits as running the whole prefix in one call.
    let dir = tmp("session");
    let p = write(&dir).unwrap();
    let m = InklingRef::open(&p).unwrap();
    let ids = [3u32, 7, 1, 12, 5];
    let full = m.forward(&mut Session::new(&m), &ids, false).unwrap();

    let mut s = Session::new(&m);
    let first = m.forward(&mut s, &ids[..1], false).unwrap();
    assert!(first.iter().all(|v| v.is_finite()));
    let mut last = first;
    for &id in &ids[1..] {
        last = m.forward(&mut s, &[id], false).unwrap();
    }
    assert_eq!(s.pos, ids.len());
    let d = last
        .iter()
        .zip(&full)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(d < 1e-3, "incremental vs full: {d}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_loop_changes_the_answer() {
    // With one pass the core is visited once, not three times: the output must differ, or the
    // loop is not wired up.
    let dir = tmp("loop");
    let p = write(&dir).unwrap();
    let mut m = InklingRef::open(&p).unwrap();
    let ids = [3u32, 7, 1];
    let three = m.forward(&mut Session::new(&m), &ids, false).unwrap();
    m.hp.n_passes = 1;
    let one = m.forward(&mut Session::new(&m), &ids, false).unwrap();
    assert_ne!(three, one, "the extra passes had no effect");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn past_the_window_a_local_layer_forgets() {
    // A prompt longer than the sliding window must still run, and the last token's logits must
    // not depend on tokens the local layers can no longer see... which the global layers do see,
    // so check the weaker property: a longer prompt runs and stays finite.
    let dir = tmp("window");
    let p = write(&dir).unwrap();
    let m = InklingRef::open(&p).unwrap();
    let ids: Vec<u32> = (0..(N_SWA as u32 + 5)).map(|i| i % N_VOCAB as u32).collect();
    let a = m.forward(&mut Session::new(&m), &ids, false).unwrap();
    assert!(a.iter().all(|v| v.is_finite()));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn bad_geometry_is_rejected() {
    let dir = tmp("bad");
    let p = write(&dir).unwrap();
    let g = crate::gguf::Gguf::open(&p).unwrap();
    let mut hp = super::Hparams::from_gguf(&g).unwrap();
    hp.check().unwrap();
    // A loop core that does not sit on block boundaries cannot preserve the residual bank.
    hp.loop_start = 1;
    assert!(hp.check().is_err());
    hp.loop_start = LOOP_START;
    hp.n_passes = 1;
    assert!(hp.check().is_err());
    hp.n_passes = N_PASSES;
    hp.n_expert_used = N_EXPERT + 1;
    assert!(hp.check().is_err());
    let _ = std::fs::remove_dir_all(dir);
}
