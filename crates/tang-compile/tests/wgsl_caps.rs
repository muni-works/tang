//! The WGSL caps lowerings (subgroup split combine, native f16) emit
//! what they claim and pass naga validation with the matching
//! capabilities; without caps the output is the portable kernel.
use tang_compile::tensor::{Format, Monoid, Precision, TensorGraph, WgslCaps};

fn graph(m: Monoid, f16: bool) -> (TensorGraph, tang_compile::tensor::TensorId) {
    let mut g = TensorGraph::new();
    let x = g.input(&[8, 4096]);
    let y = g.reduce(x, 1, m).unwrap();
    if f16 {
        g.set_precision(
            x,
            Precision {
                storage: Format::F16,
                accum: Format::F32,
            },
        );
        g.set_precision(
            y,
            Precision {
                storage: Format::F16,
                accum: Format::F32,
            },
        );
    }
    g.set_schedule(y, 256, (0, 0, 0), (1, 1));
    g.set_subgroup(y, true);
    (g, y)
}

fn validate(src: &str) {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("{}\n{src}", e.emit_to_string(src)));
    let caps = naga::valid::Capabilities::all();
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
        .subgroup_stages(naga::valid::ShaderStages::all())
        .subgroup_operations(naga::valid::SubgroupOperationSet::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{e:?}\n{src}"));
}

#[test]
fn portable_without_caps() {
    let (g, y) = graph(Monoid::sum(), true);
    let src = &g.to_wgsl(&[y]).kernels[0].source;
    assert!(!src.contains("subgroup") && !src.contains("enable f16"));
    assert!(src.contains("atomic<u32>"));
    validate(src);
}

#[test]
fn caps_emit_subgroups_and_f16() {
    let caps = WgslCaps {
        subgroups: true,
        f16: true,
    };
    for (m, op) in [
        (Monoid::sum(), "subgroupAdd"),
        (Monoid::max(), "subgroupMax"),
        (Monoid::argmax(), "subgroupShuffleXor"),
    ] {
        let tuple = m.width > 1;
        let (g, y) = if tuple {
            // argmax state over (index, value) pairs
            let mut g = TensorGraph::new();
            let x = g.input(&[8, 4096]);
            let y = g
                .comp_tuple(
                    &[8, 2],
                    &[4096],
                    vec![tang_compile::tensor::Arg {
                        src: x,
                        view: tang_compile::tensor::View::identity(2, 2),
                    }],
                    Some(m),
                    |v| vec![v[0], v[0]],
                )
                .unwrap();
            g.set_schedule(y, 256, (0, 0, 0), (1, 1));
            g.set_subgroup(y, true);
            (g, y)
        } else {
            graph(m, true)
        };
        let src = &g.to_wgsl_caps(&[y], caps).kernels[0].source;
        assert!(src.contains(op), "{op}\n{src}");
        if !tuple {
            assert!(
                src.starts_with("enable f16;") && src.contains("array<f16>"),
                "{src}"
            );
            assert!(!src.contains("atomic"), "{src}");
        }
        validate(src);
    }
}

#[test]
fn small_splits_keep_the_tree() {
    let (mut g, y) = graph(Monoid::sum(), false);
    g.set_schedule(y, 64, (0, 0, 0), (1, 1));
    let src = &g
        .to_wgsl_caps(
            &[y],
            WgslCaps {
                subgroups: true,
                f16: false,
            },
        )
        .kernels[0]
        .source;
    assert!(!src.contains("subgroup"), "{src}");
}
