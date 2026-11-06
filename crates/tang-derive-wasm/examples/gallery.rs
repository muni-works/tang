//! The PDF's derivation gallery: each preset of the post's live lab, derived at n = 1024,
//! d = 64, with kernel and n×n-buffer counts for the graph as written and as derived.
//! `cargo run -p tang-derive-wasm --example gallery > gallery.csv`
use tang_derive_wasm::{derive_graph, naive_graph, naive_program, parse_variant};

fn main() {
    let (n, d) = (1024, 64);
    let presets = [
        ("Causal", "s", "j <= i"),
        ("Sliding window", "s", "j <= i && i - j < 128"),
        ("ALiBi", "s - 0.0625*(i - j)", "j <= i"),
        ("Soft cap", "30*tanh(s/30)", "j <= i"),
        ("Prefix LM", "s", "j <= i || j < 64"),
        ("Bidirectional", "s", ""),
    ];
    println!("variant,score,mask,pass,naive_kernels,naive_nxn,derived_kernels,derived_nxn");
    for (name, score, mask) in presets {
        for bwd in [false, true] {
            let v = parse_variant(score, mask).unwrap();
            let (g, outs) = naive_graph(&v, n, d, bwd).unwrap();
            let a = naive_program(&g, &outs);
            let (dg, douts) = derive_graph(&g, &outs);
            let b = dg.to_wgsl(&douts);
            let nxn = |p: &tang_compile::tensor::wgsl::WgslProgram| {
                p.buffers
                    .iter()
                    .filter(|x| x.input.is_none() && x.len >= n * n)
                    .count()
            };
            println!(
                "{name},{score},{mask},{},{},{},{},{}",
                if bwd { "fwd+bwd" } else { "fwd" },
                a.kernels.len(),
                nxn(&a),
                b.kernels.len(),
                nxn(&b)
            );
        }
    }
}
