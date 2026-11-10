//! Log and max-plus scans (dense and banded), forward and backward, on the
//! wgpu device (Metal on macOS) against the f64 interpreter. Skipped
//! without a device; the error report is `tang_verify::stat`.

use tang_compile::tensor::{Semiring, TensorGraph, TensorId};
use tang_gpu::program::{Program, Runtime};

fn rand_like(n: usize, seed: u64, scale: f64) -> Vec<f64> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2.0 * scale
        })
        .collect()
}

#[test]
fn semiring_scans_match_the_interpreter() {
    let Ok(rt) = Runtime::select() else {
        eprintln!("skipped: no GPU runtime");
        return;
    };
    eprintln!("device: {}", rt.name());
    let (b, tn, n) = (4, 64, 16);
    for bands in [None, Some(vec![-2i64, -1, 0])] {
        let k = bands.as_ref().map_or(n, |o| o.len());
        let a = rand_like(b * tn * n * k, 1, 2.0);
        let u = rand_like(b * tn * n, 2, 2.0);
        for ring in [Semiring::Log, Semiring::MaxPlus] {
            for rev in [false, true] {
                let mut g = TensorGraph::new();
                let ai = g.input(&[b, tn, n, k]);
                let ui = g.input(&[b, tn, n, 1]);
                let s = match &bands {
                    None => g.semiring_scan(ai, ui, 1, false, rev, ring),
                    Some(o) => g.semiring_scan_banded(ai, ui, o, 1, rev, ring),
                }
                .unwrap();
                let gr = g.backward(s, &[ai, ui]).unwrap();
                let outs: Vec<TensorId> = vec![s, gr[0], gr[1]];
                let want = g.eval(&[a.clone(), u.clone()], &outs);
                let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
                let (a32, u32_) = (f32s(&a), f32s(&u));
                let p = Program::compile(&rt, &g, &outs, &[&a32, &u32_]);
                p.run(&rt);
                for (name, (t, w)) in ["S", "dA", "dU"].iter().zip(outs.iter().zip(&want)) {
                    let got = p.read(&rt, *t);
                    let st = tang_verify::stat(&got, &f32s(w), 1e-3);
                    eprintln!(
                        "{ring:?} {} rev {rev} {name}: rel_l2 {:.2e} max_abs {:.2e} nonfinite {}",
                        if bands.is_some() { "banded" } else { "dense" },
                        st.rel_l2,
                        st.max_abs,
                        st.nonfinite_gpu
                    );
                    assert_eq!(st.nonfinite_gpu, 0);
                    assert!(st.norm_cpu > 0.0, "{name} is all zeros");
                    assert!(st.rel_l2 < 1e-4, "{ring:?} {name}: rel_l2 {}", st.rel_l2);
                }
            }
        }
    }
}
