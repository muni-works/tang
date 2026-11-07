//! GPU timing of the discovered one-pass softmax-weighted variance vs the
//! two-pass definition (`docs/discovery.md`). Both graphs come from
//! `online_graphs`; both get `schedule_auto`.
use std::time::Instant;
use tang_bench::ir_gpu::{program_for, Executable};
use tang_compile::tensor::discover::{online_graphs, online_lift};
use tang_compile::tensor::discover_catalog::online_catalog;
use tang_gpu::{GpuDevice, KernelCache};

fn main() {
    let device = GpuDevice::new_sync().expect("gpu");
    let mut cache = KernelCache::new();
    let case = online_catalog()
        .into_iter()
        .find(|c| c.spec.name.starts_with("softmax-weighted variance"))
        .unwrap();
    let lift = online_lift(&case.spec).unwrap_or_else(|e| panic!("{e:?}"));
    println!("{}", device.name);
    for (rows, n) in [(4096usize, 4096usize), (1024, 16384), (16384, 1024)] {
        let (mut g2, two, mut g1, one) = online_graphs(&case.spec, &lift, rows, n);
        g2.schedule_auto(&[two]);
        g1.schedule_auto(&[one]);
        let mut rng = tang_bench::Rng::new(3);
        let s: Vec<f32> = (0..rows * n).map(|_| 6.0 * (rng.f32() - 0.5)).collect();
        let v: Vec<f32> = (0..rows * n).map(|_| 4.0 + rng.f32()).collect();
        let mut res = vec![];
        for (g, o) in [(&g2, two), (&g1, one)] {
            let exe = Executable::new(&device, program_for(&device, g, &[o]), &[&s, &v]);
            for _ in 0..3 {
                exe.run(&device, &mut cache);
            }
            cache.flush(&device);
            let _ = exe.read(&device, o);
            let t = Instant::now();
            for _ in 0..30 {
                exe.run(&device, &mut cache);
            }
            cache.flush(&device);
            let out = exe.read(&device, o);
            res.push((t.elapsed().as_secs_f64() * 1e3 / 30.0, out));
        }
        let err = res[0]
            .1
            .iter()
            .zip(&res[1].1)
            .map(|(a, b)| ((a - b) / a.abs().max(1e-6)).abs())
            .fold(0f32, f32::max);
        println!(
            "rows {rows} n {n}: two-pass {:.3} ms, one-pass {:.3} ms ({:.2}x), max rel diff {err:.1e}",
            res[0].0,
            res[1].0,
            res[0].0 / res[1].0
        );
    }
}
