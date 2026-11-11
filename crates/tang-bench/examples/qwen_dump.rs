//! Dump a Qwen-shaped step program (no device needed):
//!
//! - `qwen_dump <dir> [seq] [--msl]`: a small model's kernels into `<dir>`
//!   (CUDA, or Metal with `--msl`);
//! - `qwen_dump --plan <seq>`: Qwen3-8B's step over `seq` tokens, its
//!   kernels by kind and the memory plan.
fn main() {
    use tang_bench::ir_qwen::*;
    let mut args = std::env::args().skip(1);
    let first = args.next().expect("output dir or --plan");
    let rest: Vec<String> = args.collect();
    let msl = rest.iter().any(|a| a == "--msl");
    let seq: usize = rest
        .iter()
        .find(|a| !a.starts_with("--"))
        .map_or(1, |s| s.parse().unwrap());
    if first == "--plan" {
        let dims = QwenDims::qwen3_8b(4352);
        let kctx = if seq == 1 {
            4352
        } else {
            seq.next_multiple_of(64)
        };
        let dg = step_graph(&dims, seq, kctx);
        let (g, outs) = compiled_decode(&dg, !msl);
        let opts = decode_opts(&dg, &outs);
        let mut program = if msl {
            g.to_msl_decode(&outs, &opts).program
        } else {
            g.to_cuda_decode(&outs, &opts).program
        };
        lazy_order(&mut program);
        let input_t = |k: usize| {
            program
                .buffers
                .iter()
                .find(|b| b.input == Some(k))
                .unwrap()
                .node
        };
        let donate: Vec<_> = opts.inplace.iter().map(|&(o, k)| (o, input_t(k))).collect();
        let plan = g.plan_memory(&program, &outs, &donate);
        let mut kinds: Vec<(String, usize)> = Vec::new();
        for k in &program.kernels {
            let kind = k.source.lines().next().unwrap_or("").to_string();
            match kinds.iter_mut().find(|x| x.0 == kind) {
                Some(x) => x.1 += 1,
                None => kinds.push((kind, 1)),
            }
        }
        for (k, n) in kinds {
            println!("{n:5}  {k}");
        }
        let (slot, slot_words) = plan_slots(&program, &outs, &opts.inplace);
        if std::env::var_os("QWEN_DUMP_SLOTS").is_some() {
            for (s, &w) in slot_words.iter().enumerate() {
                let users: Vec<_> = (0..slot.len())
                    .filter(|&b| slot[b] == s)
                    .map(|b| {
                        let n = program.buffers[b].node;
                        (program.buffers[b].input, g.nodes()[n.index()].shape.clone())
                    })
                    .collect();
                if w > 2_500_000 && users[0].0.is_none() {
                    println!(
                        "slot {s}: {:.1} MB, {} users, first {:?}",
                        w as f64 * 4e-6,
                        users.len(),
                        users[0].1
                    );
                }
            }
            // who reads the first users of the last few slots
            for b in (0..slot.len()).filter(|&b| slot[b] + 3 >= slot_words.len()) {
                let node = program.buffers[b].node;
                let readers: Vec<usize> = program
                    .kernels
                    .iter()
                    .enumerate()
                    .filter(|(_, k)| k.reads.iter().any(|&t| program.storage(t) == node))
                    .map(|(i, _)| i)
                    .collect();
                let writer = program.kernels.iter().position(|k| k.node == node);
                println!(
                    "buffer {b} {:?}: written by {writer:?}, read by {readers:?}",
                    g.nodes()[node.index()].shape
                );
            }
        }
        println!(
            "plan_slots: {:.0} MB in {} slots",
            slot_words.iter().sum::<usize>() as f64 * 4e-6,
            slot_words.len()
        );
        let s = plan.stats;
        println!(
            "persistent {:.0} MB, intermediates planned {:.0} MB (live peak {:.0} MB, naive {:.0} MB), {} in place",
            s.persistent_bytes as f64 / 1e6,
            s.planned_bytes as f64 / 1e6,
            s.live_peak_bytes as f64 / 1e6,
            s.naive_bytes as f64 / 1e6,
            plan.inplace.len()
        );
        return;
    }
    let dims = QwenDims {
        d_model: 128,
        heads: 4,
        kv_heads: 2,
        head_dim: 32,
        ff: 192,
        vocab: 256,
        layers: 1,
        ctx: 16.max(seq),
        eps: 1e-6,
        theta: 1e4,
        kv_f16: true,
        vocab_chunks: 1,
        prefill_q4: false,
        g3: false,
        hadamard: 0,
    };
    let dg = step_graph(&dims, seq, dims.ctx);
    let (g, outs) = compiled_decode(&dg, !msl);
    let opts = decode_opts(&dg, &outs);
    let p = if msl {
        g.to_msl_decode(&outs, &opts)
    } else {
        g.to_cuda_decode(&outs, &opts)
    };
    for (i, (k, n)) in p.program.kernels.iter().zip(&p.native).enumerate() {
        let node = &g.nodes()[k.node.index()];
        println!(
            "k{i:02} native={n} shape={:?} reads={} {}",
            node.shape,
            k.reads.len(),
            k.source.lines().next().unwrap_or("")
        );
        let ext = if k.source.starts_with("#include") {
            "metal"
        } else if msl {
            "wgsl"
        } else {
            "cu"
        };
        std::fs::write(format!("{first}/k{i:02}.{ext}"), &k.source).unwrap();
    }
}
