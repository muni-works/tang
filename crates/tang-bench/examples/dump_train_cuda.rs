//! Write every kernel of the training step's `to_cuda` program (native
//! lowerings on) to `DIR/k{i}_n{node}.cu`, and print each kernel's comp
//! structure: the shapes a flash-backward lowering has to recognize.
//! `cargo run --release -p tang-bench --example dump_train_cuda [dir] [seq]`
use tang_bench::ir_block::{train_loss_graph, BlockDims};
use tang_compile::tensor::Op;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/tang_train_cuda".into());
    let seq = std::env::args().nth(2).map_or(128, |s| s.parse().unwrap());
    std::fs::create_dir_all(&dir).unwrap();
    let dims = BlockDims {
        seq,
        kv_len: seq,
        rope: true,
        bias: true,
        d_model: 576,
        n_heads: 9,
        n_kv_heads: 3,
        head_dim: 64,
        ff_dim: 1536,
    };
    let (tg, outs, _) = train_loss_graph(&dims);
    let (tg, outs) = tg.stabilize_softmax(&outs);
    let (tg, outs) = tg.substitute_maps(&outs);
    let (tg, outs) = tg.eliminate_common(&outs);
    let (tg, outs) = tg.merge_siblings(&outs);
    let (tg, outs) = tg.relayout(&outs);
    let (tg, outs) = tg.merge_reduce_dims(&outs);
    let (mut tg, outs) = tg.split_reduce_dims(&outs);
    tg.schedule_auto(&outs);
    let native = tg.to_cuda_with(&outs, true);
    let plain = tg.to_cuda_with(&outs, false);
    let dump = |g: &tang_compile::tensor::TensorGraph, n: usize, indent: &str| {
        let Some(c) = g.comp_at(n) else { return };
        for (a_i, a) in c.args.iter().enumerate() {
            let src = &g.nodes()[a.src.index()];
            let what = match &src.op {
                Op::Input(j) => format!("input {j}"),
                Op::Iota(d) => format!("iota {d}"),
                Op::Reshape(s) => format!("reshape of {}", s.index()),
                Op::Operator(_) => "operator".to_string(),
                Op::Scan(_) => "scan".to_string(),
                Op::Comp(pc) => format!(
                    "comp(red {:?}, monoid {:?}, inline {})",
                    pc.red.iter().map(|d| d.n()).collect::<Vec<_>>(),
                    pc.monoid.as_ref().map(|m| m.kind()),
                    src.inline
                ),
            };
            let forms: Vec<String> = a
                .view
                .forms
                .iter()
                .map(|f| format!("{:?}+{}", f.coefs, f.offset))
                .collect();
            println!(
                "{indent}arg {a_i}: n{} {what} shape {:?} [{}]",
                a.src.index(),
                src.shape.iter().map(|d| d.n()).collect::<Vec<_>>(),
                forms.join("; ")
            );
        }
        println!(
            "{indent}body {:?} outs {:?}",
            c.body.nodes_slice(),
            c.body_outs
        );
    };
    for (i, (k, kp)) in native.kernels.iter().zip(&plain.kernels).enumerate() {
        let n = k.node.index();
        std::fs::write(format!("{dir}/k{i:02}_n{n}.cu"), &k.source).unwrap();
        std::fs::write(format!("{dir}/k{i:02}_n{n}_plain.cu"), &kp.source).unwrap();
        let node = &tg.nodes()[n];
        let c = tg.comp_at(n);
        println!(
            "k{i:02} n{n} {} shape {:?} red {:?} monoid {:?} tile {:?} split {} reads {:?}",
            if k.source != kp.source {
                "native"
            } else {
                "plain "
            },
            node.shape.iter().map(|d| d.n()).collect::<Vec<_>>(),
            c.map(|c| c.red.iter().map(|d| d.n()).collect::<Vec<_>>()),
            c.and_then(|c| c.monoid.as_ref().map(|m| m.kind())),
            node.tile,
            node.split,
            k.reads.iter().map(|t| t.index()).collect::<Vec<_>>(),
        );
        if std::env::var("TANG_VERBOSE").is_err() {
            continue;
        }
        dump(&tg, n, "    ");
        if let Some(c) = tg.comp_at(n) {
            for a in &c.args {
                let s = a.src.index();
                if tg.nodes()[s].inline && tg.comp_at(s).is_some() {
                    println!("    -- inline n{s}:");
                    dump(&tg, s, "        ");
                    if let Some(c2) = tg.comp_at(s) {
                        for a2 in &c2.args {
                            let s2 = a2.src.index();
                            if tg.nodes()[s2].inline && tg.comp_at(s2).is_some() {
                                println!("        -- inline n{s2}:");
                                dump(&tg, s2, "            ");
                            }
                        }
                    }
                }
            }
        }
    }
}
