//! The batched int8 contractions of `batch_decode::int8_case` at coder-next
//! shapes on the GPU, per schedule, against the f64 references.
//!
//! `int8_batch_gpu [B…]`
use tang_bench::batch_decode::{int8_case, Int8Dims};
use tang_bench::qwen_session::runtime;
use tang_compile::tensor::int8_sched::BlockOpts;
use tang_gpu::program::Program;

fn main() {
    let bs: Vec<usize> = std::env::args()
        .skip(1)
        .map(|v| v.parse().unwrap())
        .collect();
    let rt = runtime();
    for &nb in if bs.is_empty() { &[4usize, 16][..] } else { &bs[..] } {
        let cases = [
            ("decode GEMV", BlockOpts::default(), 0),
            ("pos-tiled", BlockOpts { pos_tile: 8, mma_min: 0, ksplit: 0 }, 0),
            ("grouped GEMV", BlockOpts::default(), 2),
            ("dense MMA", BlockOpts { pos_tile: 1, mma_min: 2, ksplit: 2 }, 0),
            ("grouped MMA", BlockOpts { pos_tile: 1, mma_min: 2, ksplit: 2 }, 2),
        ];
        for (name, bo, group) in cases {
            for (shape, d) in [("gate", Int8Dims::coder(nb)), ("down", Int8Dims::coder_down(nb))] {
            let c = int8_case(&d, &bo, group);
            let (ey, ez, stages) = {
                let ins: Vec<&[f32]> = c.inputs.iter().map(|v| v.as_slice()).collect();
                let stages = c.stages.clone();
                let outs = c.outs.clone();
                let want = c.want.clone();
                let prog = Program::new(&rt, c.program.program, &ins);
                prog.run(&rt);
                let got: Vec<Vec<f32>> = outs.iter().map(|&t| prog.read(&rt, t)).collect();
                let f = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
                (
                    tang_bench::derive_all::rel(&f(&got[0]), &want[0]),
                    tang_bench::derive_all::rel(&f(&got[1]), &want[1]),
                    stages,
                )
            };
            println!(
                "B={nb} {shape} {name:13} stages {:?}: routed rel err {ey:.2e}, dense rel err {ez:.2e}",
                stages
            );
            }
        }
    }
}
