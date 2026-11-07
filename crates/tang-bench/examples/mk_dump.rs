//! Offline: the Qwen3-8B decode step's megakernel source (no GPU needed).
//! `mk_dump <ctx> <out.cu> [threads]`
use tang_bench::ir_qwen::*;
use tang_bench::qwen_session::slot_plan;
use tang_compile::tensor::cuda_mega::{megakernel, MegaOpts, MegaStage};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ctx: usize = a.get(1).map_or(4352, |v| v.parse().unwrap());
    let threads: u32 = a.get(3).map_or(256, |v| v.parse().unwrap());
    let dims = QwenDims::qwen3_8b(ctx);
    let dg = step_graph(&dims, 1, ctx);
    let (g, outs) = compiled_decode(&dg, true);
    let opts = decode_opts(&dg, &outs);
    let p = g.to_cuda_decode(&outs, &opts).program;
    if let Ok(dir) = std::env::var("MK_KERNELS") {
        std::fs::create_dir_all(&dir).unwrap();
        for (i, k) in p.kernels.iter().enumerate() {
            std::fs::write(format!("{dir}/k{i:03}.cu"), &k.source).unwrap();
        }
    }
    let plan = slot_plan(&p, &outs, &opts.inplace);
    let slot = |t| {
        let s = p.storage(t);
        plan.slot[p.buffers.iter().position(|b| b.node == s).unwrap()]
    };
    let stages: Vec<MegaStage> = plan
        .order
        .iter()
        .map(|&i| {
            let k = &p.kernels[i];
            MegaStage {
                source: &k.source,
                threads: k.workgroup_size,
                blocks: k.count.div_ceil(k.workgroup_size),
                count: k.points,
                args: k
                    .reads
                    .iter()
                    .chain(std::iter::once(&k.node))
                    .map(|&t| slot(t))
                    .collect(),
            }
        })
        .collect();
    let dup: u32 = a.get(4).map_or(0, |v| v.parse().unwrap());
    let slot_bytes = plan.slot_len.iter().map(|n| 4 * n).collect();
    let prefetch: usize = a.get(5).map_or(0, |v| v.parse().unwrap());
    let m = megakernel(
        &stages,
        &MegaOpts {
            threads,
            dup,
            min_blocks: 0,
            prefetch: prefetch << 10,
            slot_bytes,
            trace: false,
        },
    );
    eprintln!(
        "{} kernels, {} phases, smem {}",
        stages.len(),
        m.phases.len(),
        m.smem
    );
    let sizes: Vec<usize> = m.phases.iter().map(|p| p.len()).collect();
    eprintln!("phase sizes: {sizes:?}");
    std::fs::write(a.get(2).map_or("/tmp/mega.cu", |s| s.as_str()), &m.source).unwrap();
}
