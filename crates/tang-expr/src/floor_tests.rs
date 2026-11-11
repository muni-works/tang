//! `Floor` and `imod` through every evaluator and emitter.

use crate::codegen::Dialect;
use crate::graph::ExprGraph;
use crate::node::{ExprId, Node};

const RS: [i64; 9] = [1, 2, 3, 4, 5, 7, 32, 100, 128];

/// `[floor(x / r), imod(x, r), floor(x)]` of var 0 (an integer) and var 1 (r).
fn graph() -> (ExprGraph, Vec<ExprId>) {
    let mut g = ExprGraph::new();
    let (x, r) = (g.var(0), g.var(1));
    let half = g.lit(0.5);
    let xh = g.add(x, half);
    let rr = g.recip(r);
    let q = g.mul(xh, rr);
    let div = g.floor(q);
    let m = g.imod(x, r);
    let third = g.lit(1.0 / 3.0);
    let xt = g.mul(x, third);
    let fl = g.floor(xt);
    (g, vec![div, m, fl])
}

fn want(x: i64, r: i64) -> [f64; 3] {
    [
        x.div_euclid(r) as f64,
        x.rem_euclid(r) as f64,
        (x as f64 / 3.0).floor(),
    ]
}

fn cases() -> impl Iterator<Item = (i64, i64)> {
    RS.into_iter()
        .flat_map(|r| (-300..3000).map(move |x| (x, r)))
}

#[test]
fn interpreters() {
    let (g, outs) = graph();
    let many = g.compile_many(&outs);
    let mut buf = vec![0.0; 3];
    for (x, r) in cases() {
        let inp = [x as f64, r as f64];
        let w = want(x, r);
        for (k, &o) in outs.iter().enumerate() {
            assert_eq!(g.eval::<f64>(o, &inp), w[k], "eval {k} x={x} r={r}");
            assert_eq!((g.compile(o))(&inp), w[k], "compile {k} x={x} r={r}");
        }
        assert_eq!(g.eval_many::<f64>(&outs, &inp), w.to_vec());
        many(&inp, &mut buf);
        assert_eq!(buf, w.to_vec(), "compile_many x={x} r={r}");
    }
}

#[test]
fn imod_with_literal_r_in_f32() {
    // f32 kernels: the half offset keeps floor's argument off integers
    for r in RS {
        let mut g = ExprGraph::new();
        let x = g.var(0);
        let rl = g.lit(r as f64);
        let m = g.imod(x, rl);
        for xi in -300..20000i64 {
            let v = g.eval::<f32>(m, &[xi as f32]);
            assert_eq!(v as i64, xi.rem_euclid(r), "x={xi} r={r}");
        }
    }
}

#[test]
fn folds_and_has_zero_derivative() {
    let mut g = ExprGraph::new();
    let l = g.lit(-2.5);
    let fl = g.floor(l);
    assert_eq!(g.node(fl).as_f64(), Some(-3.0));
    let x = g.var(0);
    let xx = g.mul(x, x);
    let f = g.floor(xx);
    assert!(matches!(g.node(f), Node::Floor(_)));
    let d = g.diff(f, 0);
    let d = g.simplify(d);
    assert_eq!(g.node(d).as_f64(), Some(0.0));
    // d/dx (x + floor(x)) = 1
    let fx = g.floor(x);
    let s = g.add(x, fx);
    let ds = g.diff(s, 0);
    assert_eq!(g.eval::<f64>(ds, &[1.7]), 1.0);
    let v = g.vjp(&[s], &[ExprId::ONE], 1);
    assert_eq!(g.eval::<f64>(v[0], &[1.7]), 1.0);
    assert_eq!(g.fmt_expr(fx), "floor(x0)");
}

#[test]
fn emitters_name_floor() {
    let (g, outs) = graph();
    for d in [Dialect::Wgsl, Dialect::Msl, Dialect::Cuda, Dialect::C] {
        let k = g.to_kernel(&outs, 2, d);
        assert!(k.source.contains("floor("), "{d:?}\n{}", k.source);
    }
    assert!(g.to_wgsl(&outs, 2).source.contains("floor("));
    // FLOOR_CUDA_DUMP: the CUDA kernel, for an NVRTC compile check
    if let Ok(dir) = std::env::var("FLOOR_CUDA_DUMP") {
        let k = g.to_kernel(&outs, 2, Dialect::Cuda);
        std::fs::write(format!("{dir}/expr.cu"), k.source).unwrap();
    }
}

fn validate_wgsl(src: &str) {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("{}\n{src}", e.emit_to_string(src)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{e:?}\n{src}"));
}

#[test]
fn wgsl_validates() {
    let (g, outs) = graph();
    validate_wgsl(&g.to_kernel(&outs, 2, Dialect::Wgsl).source);
    validate_wgsl(&g.to_wgsl(&outs, 2).source);
}

/// Build the C kernels with the system `cc` and run them over every case
/// (skipped when there is no `cc`).
#[test]
fn c_kernels_run() {
    use std::fmt::Write;
    let (g, outs) = graph();
    let dir = std::env::temp_dir().join(format!("tang-floor-c-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cs: Vec<(i64, i64)> = cases().collect();
    for (tag, real, src) in [
        ("f64", "double", g.to_c_f64(&outs, 2).source),
        ("f32", "float", g.to_kernel(&outs, 2, Dialect::C).source),
    ] {
        let mut main = src.clone();
        writeln!(main, "#include <stdio.h>\nint main(void) {{").unwrap();
        writeln!(
            main,
            "  static {real} in[{}], out[{}];",
            2 * cs.len(),
            3 * cs.len()
        )
        .unwrap();
        writeln!(main, "  int n = 0; double x, r;").unwrap();
        writeln!(
            main,
            "  while (scanf(\"%lf %lf\", &x, &r) == 2) {{ in[2*n] = x; in[2*n+1] = r; n++; }}"
        )
        .unwrap();
        writeln!(main, "  k0(in, out, n);").unwrap();
        writeln!(main, "  for (int i = 0; i < 3*n; i++) printf(\"%.17g\\n\", (double)out[i]);\n  return 0;\n}}").unwrap();
        let c = dir.join(format!("{tag}.c"));
        let exe = dir.join(tag);
        std::fs::write(&c, main).unwrap();
        let Ok(st) = std::process::Command::new("cc")
            .args(["-O2", "-o"])
            .arg(&exe)
            .arg(&c)
            .arg("-lm")
            .status()
        else {
            eprintln!("no cc: skipped");
            return;
        };
        assert!(st.success(), "cc failed on {tag}");
        let input: String = cs.iter().map(|(x, r)| format!("{x} {r}\n")).collect();
        let mut child = std::process::Command::new(&exe)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        use std::io::Write as _;
        child
            .stdin
            .take()
            .unwrap()
            .write_all(input.as_bytes())
            .unwrap();
        let out = child.wait_with_output().unwrap();
        let got: Vec<f64> = String::from_utf8(out.stdout)
            .unwrap()
            .lines()
            .map(|l| l.parse().unwrap())
            .collect();
        for (i, &(x, r)) in cs.iter().enumerate() {
            let w = want(x, r);
            for k in 0..3 {
                // f32 floor(x / 3) may round x/3 across an integer above 2^22 / 3;
                // the cases stay well below, so every output is exact
                assert_eq!(got[3 * i + k], w[k], "{tag} out {k} x={x} r={r}");
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
