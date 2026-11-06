//! Placement, host-mediated: the flash-fused attention reduction is split
//! across two GPU devices (two wgpu devices on this machine), each computes
//! its partial (m, l, acc) state over half the keys, and the online-softmax
//! combine — written as a comprehension — folds them. Same answer as one
//! device.
use tang_bench::ir_gpu::{flash_attention, naive_attention, Executable};
use tang_compile::tensor::{Arg, Form, Monoid, Op, TensorGraph, TensorId, View};
use tang_gpu::{GpuDevice, KernelCache};

fn main() {
    let (n, d) = (256usize, 64usize);
    let mut rng = tang_bench::Rng::new(9);
    let mut rand = || {
        (0..n * d)
            .map(|_| rng.f32() * 2.0 - 1.0)
            .collect::<Vec<f32>>()
    };
    let (q, k, v) = (rand(), rand(), rand());
    let inputs: Vec<&[f32]> = vec![&q, &k, &v];

    // reference: one device, the whole graph
    let dev0 = GpuDevice::new_sync().expect("gpu");
    let dev1 = GpuDevice::new_sync().expect("second gpu device");
    let mut c0 = KernelCache::new();
    let mut c1 = KernelCache::new();
    let (g, y) = flash_attention(n, d, d);
    let exe = Executable::new(&dev0, g.to_wgsl(&[y]), &inputs);
    exe.run(&dev0, &mut c0);
    c0.flush(&dev0);
    let want = exe.read(&dev0, y);

    // shard the fused reduction over keys: partial p on device p
    let (g, y_naive) = naive_attention(n, d, d);
    let (g, _outs) = {
        let (g, o) = g.stabilize_softmax(&[y_naive]);
        g.substitute_maps(&o)
    };
    let fused = (0..g.nodes().len())
        .find(|&i| matches!(&g.nodes()[i].op, Op::Comp(c) if c.tuple == 1))
        .map(TensorId::from_index)
        .unwrap();
    let (sg, partials, _) = g.shard_reduction(fused, 2).unwrap();
    let mut shard_results = Vec::new();
    for (p, (dev, cache)) in [(&dev0, &mut c0), (&dev1, &mut c1)].into_iter().enumerate() {
        let mut pg = sg.clone();
        pg.schedule_auto(&[partials[p]]);
        let exe = Executable::new(dev, pg.to_wgsl(&[partials[p]]), &inputs);
        exe.run(dev, cache);
        cache.flush(dev);
        shard_results.push(exe.read(dev, partials[p]));
        println!(
            "device {p} ({}): partial state {} floats",
            dev.name,
            shard_results[p].len()
        );
    }

    // the all-reduce: ship partials to device 0, fold with the monoid,
    // finish y = acc / l
    let shape = sg.shape(partials[0]).to_vec();
    let width = shape.last().unwrap().n();
    let mut cg = TensorGraph::new();
    let p0 = cg.input(&shape);
    let p1 = cg.input(&shape);
    let folded = cg
        .fold_partials(&[p0, p1], &Monoid::online_softmax(width - 1))
        .unwrap();
    let rank = shape.len() - 1; // [n, 1, dv]
    let pts: Vec<usize> = shape[..rank].iter().map(|x| x.n()).collect();
    let slot = |s: i64| View {
        gather: None,
        guard: false,
        forms: (0..rank)
            .map(|dd| Form::unit(rank, dd))
            .chain([Form::constant(rank, s)])
            .collect(),
    };
    let y2 = cg
        .comp(
            &pts,
            &[],
            vec![
                Arg {
                    src: folded,
                    view: slot(2),
                },
                Arg {
                    src: folded,
                    view: slot(1),
                },
            ],
            None,
            |x| vec![x[0] / x[1]],
        )
        .unwrap();
    let combine = Executable::new(
        &dev0,
        cg.to_wgsl(&[y2]),
        &[&shard_results[0], &shard_results[1]],
    );
    combine.run(&dev0, &mut c0);
    c0.flush(&dev0);
    let got = combine.read(&dev0, y2);
    let err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "sharded across 2 devices vs one device: max err {err} over {} outputs",
        got.len()
    );
    assert!(err < 1e-4);
}
