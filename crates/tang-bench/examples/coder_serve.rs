//! `tang-llm serve` with the derived-step Qwen3-Coder-Next engine available:
//!
//! ```text
//! TANG_CODER_ENGINE=derived coder_serve <coder-next.gguf> [tang-llm serve options: --port, --ctx, --batch, ...]
//! ```
//!
//! Without `TANG_CODER_ENGINE=derived` it serves exactly as `tang-llm serve` (the hand engine).
//! `coder_serve --selftest <gguf> [ctx]`: hand and derived engines in one process on the same
//! prompts (greedy, 64 tokens): token equality, the hand logits' top-2 gap where they part, and
//! every derived slot on the same prompt (slot determinism).
fn main() -> anyhow::Result<()> {
    tang_bench::coder_serve::register();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--selftest") {
        return selftest(&args[1..]);
    }
    let hardware: tang_llm::node::Probe = std::sync::Arc::new(|| {
        let (total, free) = tang_llm::node::host_memory().unwrap_or((0, 0));
        tang_llm::node::Hardware {
            kind: "cuda",
            name: String::new(),
            unified: false,
            total_bytes: total,
            free_bytes: free,
        }
    });
    tang_llm::flash::serve::serve_cli(&args, hardware)
}

fn top2(l: &[f32]) -> (usize, f32, usize, f32) {
    let (mut a, mut b) = (0usize, usize::MAX);
    for i in 1..l.len() {
        if l[i] > l[a] {
            b = a;
            a = i;
        } else if b == usize::MAX || l[i] > l[b] {
            b = i;
        }
    }
    (a, l[a], b, l[b])
}

fn greedy(e: &mut dyn tang_llm::flash::serve::ServeEngine, slot: Option<usize>, ids: &[u32], n: usize)
    -> anyhow::Result<(Vec<u32>, Vec<Vec<f32>>)> {
    let mut out = Vec::new();
    let mut lg = Vec::new();
    let mut cur = match slot {
        Some(s) => e.prefill_slot(s, ids)?,
        None => e.prefill(ids, 8)?,
    };
    for _ in 0..n {
        let v = e.logits(1);
        lg.push(v);
        out.push(cur);
        cur = match slot {
            Some(s) => e.decode_batch(&[(s, cur)])?[0],
            None => {
                let k = e.verify(cur, &[])?;
                k[0]
            }
        };
    }
    Ok((out, lg))
}

fn selftest(args: &[String]) -> anyhow::Result<()> {
    use tang_llm::flash::serve::ServeEngine;
    let path = std::path::PathBuf::from(&args[0]);
    let ctx: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(4096);
    let n = 64;
    let tok = tang_llm::flash::tokenize::FlashTokenizer::from_gguf(&tang_llm::gguf::Gguf::open(&path)?)?;
    let prompts = [
        "Write a Python function that returns the n-th Fibonacci number iteratively.",
        "Explain what a Rust lifetime is in two sentences.",
        "Write a bash one-liner that counts lines in all .rs files under the current directory.",
        "What does the HTTP status code 429 mean, and how should a client react?",
    ];
    let ids: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| tok.encode(&format!("<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n")))
        .collect::<anyhow::Result<_>>()?;
    // hand: one sequence, reset between prompts
    let no_hand = args.iter().any(|a| a == "--no-hand");
    let mut h = Vec::new();
    if !no_hand {
    let mut hand = tang_llm::flash::coder::Coder::load(
        &path,
        tang_llm::flash::coder::CoderOpts { max_ctx: ctx, mtp: None },
    )?;
    hand.use_graphs = true;
    for p in &ids {
        ServeEngine::reset(&mut hand);
        h.push(greedy(&mut hand, None, p, n)?);
    }
    drop(hand);
    }
    let mut d = tang_bench::coder_serve::DerivedCoder::load(&path, ctx, 4)?;
    let mut all_ok = true;
    for (i, p) in ids.iter().enumerate() {
        let mut per_slot = Vec::new();
        for s in 0..4 {
            d.free_slot(s);
            per_slot.push(greedy(&mut d, Some(s), p, n)?);
        }
        d.free_slot(0);
        let again = greedy(&mut d, Some(0), p, n)?;
        let slots_same = per_slot.iter().all(|x| x.0 == per_slot[0].0) && again.0 == per_slot[0].0;
        if no_hand {
            let l0 = &per_slot[0].1[0];
            let (a, la, b, lb) = top2(l0);
            println!(
                "prompt {i}: slots deterministic {slots_same}; first step top2 {a}:{la:.6} {b}:{lb:.6}; tokens {:?}",
                per_slot[0].0
            );
            all_ok &= slots_same;
            continue;
        }
        let (ht, hl) = &h[i];
        let (dt, dl) = &per_slot[0];
        let k = ht.iter().zip(dt).take_while(|(a, b)| a == b).count();
        let note = if k < n {
            let (a, la, b, lb) = top2(&hl[k]);
            let (c, lc, e2, le) = top2(&dl[k]);
            format!(
                "part at {k}: hand top2 {a}:{la:.4} {b}:{lb:.4} (gap {:.4}), derived {c}:{lc:.4} {e2}:{le:.4}, derived logit of hand's pick {:.4}",
                la - lb, dl[k][a]
            )
        } else {
            "equal".into()
        };
        let maxdiff = hl[0].iter().zip(&dl[0]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("prompt {i}: {} tokens, hand==derived {}/{n} ({note}); first-step max |dlogit| {maxdiff:.4}; slots deterministic {slots_same}", p.len(), k);
        all_ok &= slots_same;
    }
    // the server's C=4 shape: four prefills, then decode_batch over all four rows at once
    let mut solo = Vec::new();
    for p in &ids {
        d.free_slot(0);
        solo.push(greedy(&mut d, Some(0), p, n)?.0);
    }
    let mut cur = Vec::new();
    for (s, p) in ids.iter().enumerate() {
        d.free_slot(s);
        cur.push(d.prefill_slot(s, p)?);
    }
    let mut outs: Vec<Vec<u32>> = vec![Vec::new(); ids.len()];
    for _ in 0..n {
        let rows: Vec<(usize, u32)> = cur.iter().enumerate().map(|(s, &t)| (s, t)).collect();
        for (s, &t) in cur.iter().enumerate() {
            outs[s].push(t);
        }
        cur = d.decode_batch(&rows)?;
    }
    let batched_same: Vec<bool> = outs.iter().zip(&solo).map(|(a, b)| a == b).collect();
    println!("selftest: slots deterministic {all_ok}; batched (4 rows a decode_batch) == solo {batched_same:?}");
    Ok(())
}
