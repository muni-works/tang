use tang_compile::tensor::{Op, TensorGraph, TensorId};
use tang_derive_wasm::{derive_graph, naive_graph, parse_variant};

fn dump(g: &TensorGraph, outs: &[TensorId]) {
    for (i, t) in g.nodes().iter().enumerate() {
        let desc = match &t.op {
            Op::Input(k) => format!("input {k}"),
            Op::Iota(d) => format!("iota {d}"),
            Op::Reshape(s) => format!("reshape {}", s.index()),
            Op::Comp(c) => {
                let args: Vec<String> = c
                    .args
                    .iter()
                    .map(|a| {
                        let f: Vec<String> = a.view.forms.iter().map(|f| format!("{:?}+{}", f.coefs, f.offset)).collect();
                        format!("{}[{}]", a.src.index(), f.join(","))
                    })
                    .collect();
                format!(
                    "comp out={:?} tuple={} red={:?} monoid={:?} w={} args={}  body={:?}",
                    c.out.iter().map(|d| d.n()).collect::<Vec<_>>(),
                    c.tuple,
                    c.red.iter().map(|d| d.n()).collect::<Vec<_>>(),
                    c.monoid.as_ref().map(|m| m.kind()),
                    c.body_outs.len(),
                    args.join(" "),
                    c.body.nodes_slice().len()
                )
            }
        };
        println!(
            "{i:3} inl={} tile={:?} reg={:?} split={} {desc}",
            t.inline as u8, t.tile, t.reg, t.split
        );
    }
    println!("outs {:?}", outs.iter().map(|o| o.index()).collect::<Vec<_>>());
}

fn main() {
    let n: usize = std::env::var("N").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let mask = std::env::var("MASK").unwrap_or("j <= i".into());
    let v = parse_variant("s", &mask).unwrap();
    let (g, outs) = naive_graph(&v, n, 64, true).unwrap();
    if std::env::var("PRE").is_ok() {
        let (a, o) = g.stabilize_softmax(&outs);
        let (a, o) = a.substitute_maps(&o);
        let (a, o) = a.eliminate_common(&o);
        dump(&a, &o);
        return;
    }
    let (dg, douts) = derive_graph(&g, &outs);
    let mut g2 = dg.clone();
    g2.schedule_nested_contractions_scalar(&douts);
    dump(&g2, &douts);
    let p = dg.to_wgsl(&douts);
    for k in &p.kernels {
        println!("kernel node {} reads {:?} points {} count {} wg {}", k.node.index(), k.reads.iter().map(|r| r.index()).collect::<Vec<_>>(), k.points, k.count, k.workgroup_size);
    }
    if std::env::var("SRC").is_ok() {
        for k in &p.kernels { println!("// ===== {}\n{}", k.node.index(), k.source); }
    }
}
