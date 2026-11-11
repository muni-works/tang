//! Print the structure of the block's fused attention comp (after the
//! passes): args, views, body, monoid — the shape a tensor-core
//! attention lowering has to recognize.
use tang_bench::ir_block::{compiled_block, BlockDims};
use tang_compile::tensor::{MonoidKind, Op};

fn main() {
    let dims = BlockDims {
        seq: 128,
        kv_len: 128,
        rope: true,
        bias: true,
        d_model: 576,
        n_heads: 9,
        n_kv_heads: 3,
        head_dim: 64,
        ff_dim: 1536,
    };
    let (g, _y) = compiled_block(&dims);
    for (i, node) in g.nodes().iter().enumerate() {
        let Some(c) = g.comp_at(i) else { continue };
        let kind = c.monoid.as_ref().map(|m| m.kind());
        if kind != Some(MonoidKind::OnlineSoftmax)
            && !c.args.iter().any(|a| {
                g.comp_at(a.src.index()).is_some_and(|p| {
                    p.monoid.as_ref().map(|m| m.kind()) == Some(MonoidKind::OnlineSoftmax)
                })
            })
        {
            continue;
        }
        println!(
            "node {i}: shape {:?} inline {} tile {:?} red {:?} tuple {} monoid {:?}",
            node.shape, node.inline, node.tile, c.red, c.tuple, kind
        );
        for (k, a) in c.args.iter().enumerate() {
            let src = &g.nodes()[a.src.index()];
            let what = match &src.op {
                Op::Input(j) => format!("input {j}"),
                Op::Iota(d) => format!("iota {d}"),
                Op::Reshape(s) => format!("reshape of {}", s.index()),
                Op::Operator(_) => "operator".to_string(),
                Op::Scan(_) => "scan".to_string(),
                Op::Comp(pc) => format!(
                    "comp(red {:?}, inline {}, {} args)",
                    pc.red,
                    src.inline,
                    pc.args.len()
                ),
            };
            let forms: Vec<String> = a
                .view
                .forms
                .iter()
                .map(|f| format!("{:?}+{}", f.coefs, f.offset))
                .collect();
            println!(
                "  arg {k}: node {} {what} shape {:?} view [{}]",
                a.src.index(),
                src.shape,
                forms.join(" ; ")
            );
            if let Op::Comp(pc) = &src.op {
                if src.inline {
                    for (kk, aa) in pc.args.iter().enumerate() {
                        let s2 = &g.nodes()[aa.src.index()];
                        let forms: Vec<String> = aa
                            .view
                            .forms
                            .iter()
                            .map(|f| format!("{:?}+{}", f.coefs, f.offset))
                            .collect();
                        println!(
                            "      sub-arg {kk}: node {} shape {:?} view [{}]",
                            aa.src.index(),
                            s2.shape,
                            forms.join(" ; ")
                        );
                    }
                    println!(
                        "      sub-body: {:?} outs {:?}",
                        pc.body.nodes_slice(),
                        pc.body_outs
                    );
                }
            }
        }
        println!("  body: {:?}", c.body.nodes_slice());
        println!("  outs: {:?}", c.body_outs);
    }
}
