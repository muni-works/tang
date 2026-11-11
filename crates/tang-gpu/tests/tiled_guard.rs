//! A tiled contraction that recomputes a nested, zero-padded strided read
//! (a stride-4 transposed conv's backward under a contraction), on the
//! wgpu device (Metal on macOS) against the f64 interpreter. Skipped
//! without a device.
//!
//! `y[o, b, x] = Σ_r w[o, r] · Σ_q a[b, x + q − 2, r − 4q + 2] · h[same]`:
//! the flat address `4x + r − 6` has no `q`, but the padding depends on it.

use tang_compile::tensor::{Arg, Form, Monoid, TensorGraph, View};
use tang_gpu::program::{Program, Runtime};

#[test]
fn nested_strided_guard_matches_the_interpreter() {
    let Ok(rt) = Runtime::select() else {
        eprintln!("skipped: no GPU runtime");
        return;
    };
    eprintln!("device: {}", rt.name());
    let mut g = TensorGraph::new();
    let a = g.input(&[2, 16, 4]);
    let h = g.input(&[16, 4]);
    let w = g.input(&[9, 16]);
    // space: (b, x, r, q)
    let taps = |lead: Vec<Form>| View {
        gather: None,
        forms: lead
            .into_iter()
            .chain([
                Form {
                    coefs: vec![0, 1, 0, 1],
                    offset: -2,
                },
                Form {
                    coefs: vec![0, 0, 1, -4],
                    offset: 2,
                },
            ])
            .collect(),
        guard: true,
    };
    let inner = g
        .comp(
            &[2, 16, 16],
            &[5],
            vec![
                Arg {
                    src: a,
                    view: taps(vec![Form::unit(4, 0)]),
                },
                Arg {
                    src: h,
                    view: taps(vec![]),
                },
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    // space: (o, b, x, r)
    let y = g
        .comp(
            &[9, 2, 16],
            &[16],
            vec![
                Arg {
                    src: inner,
                    view: View::pick(4, &[Some(1), Some(2), Some(3)]),
                },
                Arg {
                    src: w,
                    view: View::pick(4, &[Some(0), Some(3)]),
                },
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    g.set_inline(inner, true);
    g.set_schedule(y, 0, (16, 16, 32), (1, 1));

    let ramp = |n: usize, scale: f64, off: f64| -> Vec<f64> {
        (0..n)
            .map(|i| ((i * 7919) % 97) as f64 * scale + off)
            .collect()
    };
    let ins = [
        ramp(2 * 16 * 4, 0.02, -1.0),
        ramp(16 * 4, 0.03, -1.4),
        ramp(9 * 16, 0.01, -0.5),
    ];
    let want = &g.eval(&ins, &[y])[0];
    let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
    let ins32: Vec<Vec<f32>> = ins.iter().map(|v| f32s(v)).collect();
    let refs: Vec<&[f32]> = ins32.iter().map(|v| v.as_slice()).collect();
    let p = Program::compile(&rt, &g, &[y], &refs);
    p.run(&rt);
    let st = tang_verify::stat(&p.read(&rt, y), &f32s(want), 1e-3);
    eprintln!("rel_l2 {:.2e} max_abs {:.2e}", st.rel_l2, st.max_abs);
    assert_eq!(st.nonfinite_gpu, 0);
    assert!(st.norm_cpu > 0.0, "y is all zeros");
    assert!(st.rel_l2 < 1e-4, "rel_l2 {}", st.rel_l2);
}
