//! Host-only tests of the harness: the GLM-5.3 KDA k/v byte-offset bug replayed on a
//! simulated device buffer, plus the other anomaly classes.
use crate::*;

/// Distinct, non-constant values per (section, token, row).
fn val(sec: usize, t: usize, r: usize) -> f32 {
    ((sec * 7919 + t * 104_729 + r * 31) as f32 * 0.001).sin() + 0.01 * sec as f32
}

/// The simulated `k5_gemv` launches into a NaN-poisoned stacked `[t][3·di]` buffer, with
/// the k/v destination bases given in ELEMENTS (`[0, k_base, v_base]`).
fn qkv_device(t: usize, di: usize, k_base: usize, v_base: usize) -> Vec<f32> {
    let stride = 3 * di;
    let mut buf = vec![f32::NAN; t * stride];
    for (sec, base) in [(0, 0), (1, k_base), (2, v_base)] {
        for ti in 0..t {
            for r in 0..di {
                let at = base + ti * stride + r;
                if at < buf.len() {
                    buf[at] = val(sec, ti, r);
                }
            }
        }
    }
    buf
}

fn qkv_spec(di: usize) -> Spec {
    let inner = Layout::new().dim("tok", 0).dim("h", di / 16).dim("d", 16);
    Spec::new("kda")
        .stacked(
            "qkv",
            &[
                ("qkv.q", di, "gemv attn_q"),
                ("qkv.k", di, "gemv attn_k"),
                ("qkv.v", di, "gemv attn_v"),
            ],
            inner.clone(),
            &[],
        )
        .tap(
            Tap::new("o", inner)
                .after(&["qkv.q", "qkv.k", "qkv.v"])
                .what("core"),
        )
}

fn run(gpu_qkv: Vec<f32>, t: usize, di: usize) -> Report {
    let mut gpu = Taps::new();
    let mut cpu = Taps::new();
    gpu.put("qkv", gpu_qkv);
    cpu.put("qkv", qkv_device(t, di, di, 2 * di));
    // the core reads poisoned/wrong inputs: its output is wrong too, but not a candidate
    gpu.put("o", vec![f32::NAN; t * di]);
    cpu.put("o", (0..t * di).map(|i| (i as f32).cos()).collect());
    let mut v = Verifier::new("kda", Config::default());
    v.verbose = false;
    v.check(0, "", &qkv_spec(di), &gpu, &cpu)
}

#[test]
fn kv_byte_offset_bug_names_the_k_writer_not_q() {
    let (t, di) = (5, 64);
    // the bug: offsets di / 2·di FLOATS added to a BYTE address = di/4, di/2 elements
    let r = run(qkv_device(t, di, di / 4, di / 2), t, di);
    let v = r.verdict.expect("must fail");
    // pipeline order alone would say qkv.q (first failing, no inputs); the values in q's
    // block are k's, so k's writer is the suspect
    assert_eq!(v.suspect, "qkv.k", "{}", v.message());
    match &v.cause {
        Cause::Misplaced {
            victim,
            actual_base,
            expected_base,
            hint,
            ..
        } => {
            assert_eq!(*victim, "qkv.q");
            assert_eq!(*actual_base, Some(16));
            assert_eq!(*expected_base, Some(64));
            assert!(hint.contains("BYTE address"), "{hint}");
        }
        c => panic!("cause {c:?}"),
    }
    let ev = v.evidence.join("\n");
    assert!(
        ev.contains("`qkv.v`"),
        "v's write into k is traced too:\n{ev}"
    );
    // NaN counts match the offset arithmetic: k's tail [di/2+.., di) per token never
    // written, all of v
    let row = |n: &str| r.rows.iter().find(|x| x.name == n).unwrap().stat.clone();
    assert_eq!(row("qkv.v").extra_nonfinite(), t * di);
    assert_eq!(row("qkv.k").extra_nonfinite(), t * di / 2);
    assert!(row("qkv.q").extra_nonfinite() == 0 && row("qkv.q").bad(1e-3));
}

#[test]
fn correct_offsets_pass() {
    let (t, di) = (5, 64);
    let mut gpu = Taps::new();
    let mut cpu = Taps::new();
    gpu.put("qkv", qkv_device(t, di, di, 2 * di));
    cpu.put("qkv", qkv_device(t, di, di, 2 * di));
    let mut v = Verifier::new("kda", Config::default());
    let r = v.check(0, "t5", &qkv_spec(di), &gpu, &cpu);
    assert!(r.ok());
    assert_eq!(r.rows.len(), 3, "o absent on both sides: skipped");
}

#[test]
fn unwritten_tail_is_unwritten() {
    let (t, di) = (4, 32);
    let cpu = qkv_device(t, di, di, 2 * di);
    let mut gpu = cpu.clone();
    for ti in 0..t {
        // v's kernel stops at half its rows
        for r in di / 2..di {
            gpu[ti * 3 * di + 2 * di + r] = f32::NAN;
        }
    }
    let (mut g, mut c) = (Taps::new(), Taps::new());
    g.put("qkv", gpu);
    c.put("qkv", cpu);
    let mut v = Verifier::new("kda", Config::default());
    let ver = v.check(0, "", &qkv_spec(di), &g, &c).verdict.unwrap();
    assert_eq!(ver.suspect, "qkv.v");
    assert_eq!(
        ver.cause,
        Cause::Unwritten {
            count: t * di / 2,
            n: t * di
        }
    );
}

#[test]
fn wrong_values_fall_back_to_pipeline_order() {
    let (t, di) = (3, 32);
    let cpu = qkv_device(t, di, di, 2 * di);
    let gpu: Vec<f32> = cpu
        .iter()
        .enumerate()
        .map(|(i, x)| if i % 96 >= 64 { x * 1.5 + 0.3 } else { *x })
        .collect();
    let (mut g, mut c) = (Taps::new(), Taps::new());
    g.put("qkv", gpu);
    c.put("qkv", cpu);
    let mut v = Verifier::new("kda", Config::default());
    let ver = v.check(0, "", &qkv_spec(di), &g, &c).verdict.unwrap();
    assert_eq!((ver.suspect, &ver.cause), ("qkv.v", &Cause::Wrong));
}

#[test]
fn zeroed_output_is_zeroed() {
    let lay = Layout::new().dim("i", 0);
    let spec = Spec::new("k")
        .tap(Tap::new("a", lay.clone()).what("ka"))
        .tap(Tap::new("b", lay).what("kb"));
    let (mut g, mut c) = (Taps::new(), Taps::new());
    let a: Vec<f32> = (0..256).map(|i| (i as f32 * 0.37).sin()).collect();
    g.put("a", a.clone());
    c.put("a", a.clone());
    g.put("b", vec![0.0; 256]);
    c.put("b", a.iter().map(|x| x + 2.0).collect());
    let mut v = Verifier::new("k", Config::default());
    let ver = v.check(0, "", &spec, &g, &c).verdict.unwrap();
    assert_eq!(ver.suspect, "b");
    assert!(matches!(ver.cause, Cause::Zeroed { .. }), "{:?}", ver.cause);
}

#[test]
fn row_mixup_is_traced_within_a_tensor() {
    // a stride bug: token 1's row written where token 2's belongs
    let lay = Layout::new().dim("tok", 0).dim("c", 64);
    let spec = Spec::new("k").tap(Tap::new("y", lay).what("ky"));
    let cpu: Vec<f32> = (0..4 * 64)
        .map(|i| ((i * 13) as f32 * 0.01).sin())
        .collect();
    let mut gpu = cpu.clone();
    gpu.copy_within(64..128, 128);
    let (mut g, mut c) = (Taps::new(), Taps::new());
    g.put("y", gpu);
    c.put("y", cpu);
    let mut v = Verifier::new("k", Config::default());
    let ver = v.check(0, "", &spec, &g, &c).verdict.unwrap();
    match &ver.cause {
        Cause::Misplaced {
            victim,
            at,
            from,
            hint,
            ..
        } => {
            assert_eq!(
                (*victim, at.as_str(), from.as_str()),
                ("y", "tok2 c0", "tok1 c0")
            );
            assert!(hint.contains("shifted by 64"), "{hint}");
        }
        c => panic!("{c:?}"),
    }
}

#[test]
fn dumps_and_counts() {
    let dir = std::env::temp_dir().join(format!("tang-verify-test-{}", std::process::id()));
    let cfg = Config {
        dir: Some(dir.clone()),
        ..Config::default()
    };
    let mut v = Verifier::new("k", cfg).with_unit("l");
    let spec = Spec::new("k").tap(Tap::new("y", Layout::flat()));
    let (mut g, mut c) = (Taps::new(), Taps::new());
    g.put("y", vec![1.0, 2.0]);
    c.put("y", vec![1.0, 2.0]);
    assert!(v.check(3, "", &spec, &g, &c).ok());
    g.put("y", vec![1.0, f32::NAN]);
    assert!(!v.check(3, "", &spec, &g, &c).ok());
    assert_eq!((v.checked(), v.failed(), v.next_window(3)), (2, 1, 2));
    assert!(dir.join("w0_l3_y.gpu.f32").exists() && dir.join("w1_l3_y.cpu.f32").exists());
    assert!(v.summary()[0].starts_with("SUMMARY: 2 windows checked, 1 failed"));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn cases_grid_reports_failures() {
    let cases = Cases::new("toy")
        .shapes(&[4usize, 9])
        .dtypes(&["good", "bad"]);
    let fails = cases.check(|&n, &dt, p| {
        let cpu: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        let mut gpu = vec![f32::NAN; n];
        let upto = if dt == "bad" { n - 1 } else { n };
        gpu[..upto].copy_from_slice(&cpu[..upto]);
        p.put("y", gpu);
        let mut c = Taps::new();
        c.put("y", cpu);
        Ok((
            Spec::new("toy").tap(Tap::new("y", Layout::flat()).what("toy kernel")),
            c,
        ))
    });
    assert_eq!(fails.len(), 2, "{fails:?}");
    assert!(
        fails[0].contains("dtype \"bad\"") && fails[0].contains("unwritten"),
        "{fails:?}"
    );
}
