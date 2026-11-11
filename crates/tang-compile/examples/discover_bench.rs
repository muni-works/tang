//! Writes C for the two-pass and the discovered one-pass softmax-weighted
//! variance (`Var_p[v]`, `p = softmax(s)` per row) over `s, v [rows, n]`.
//! Usage: `cargo run --release --example discover_bench -- OUT_DIR rows n`;
//! then build `bench/discover/harness.c` against each unit.
use tang_compile::tensor::discover::online_graphs;
use tang_compile::tensor::discover::online_lift;
use tang_compile::tensor::discover_catalog::online_catalog;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (dir, rows, n): (&str, usize, usize) =
        (&a[1], a[2].parse().unwrap(), a[3].parse().unwrap());
    let case = online_catalog()
        .into_iter()
        .find(|c| c.spec.name.starts_with("softmax-weighted variance"))
        .unwrap();
    let lift = online_lift(&case.spec).unwrap_or_else(|e| panic!("{e:?}"));
    let (g2, two, g1, one) = online_graphs(&case.spec, &lift, rows, n);
    std::fs::write(format!("{dir}/two_pass.c"), g2.to_c(&[two]).source).unwrap();
    std::fs::write(format!("{dir}/one_pass.c"), g1.to_c(&[one]).source).unwrap();
    println!(
        "monoid width {} (R1 3 + basis {})",
        lift.monoid.width, lift.dim
    );
}
