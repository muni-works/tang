//! Offline three-step recursive MTP fine-tuning on `flash-generate --dump-mtp-train` dumps.
//! Dense block weights (including HC, router and shared expert) train with tang-train Adam;
//! routed experts, main embedding and main head stay frozen. No target model occupies VRAM.
//! Tensor contractions use tang-compute, nonlinear Jacobians use tang-ad. The export preserves
//! every GGUF directory entry and untouched byte; Q8/BF16/F32 updates keep their original type.
mod data;
pub(crate) use data::Sequence;
mod model;
mod tape;
mod weights;
#[cfg(test)]
mod coder_test;
use crate::flash::reference::Hparams;
use anyhow::{bail, ensure, Context, Result};
use std::{
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use tang_compute::ComputeDevice;
use weights::Weights;

struct Args {
    main: PathBuf,
    mtp: PathBuf,
    data: PathBuf,
    out: PathBuf,
    steps: usize,
    seq: usize,
    burn: usize,
    lr: f64,
    beta: f64,
    clip: f64,
    vocab: usize,
    seed: u64,
    cpu: bool,
    eval_every: usize,
    resume: Option<PathBuf>,
    check_forward: bool,
    check_context: bool,
    prefix_data: Option<PathBuf>,
    eval_windows: usize,
    /// Recursive depths trained (and evaluated); the engine drafts `TANG_CODER_MTP_K`.
    depth: usize,
}
fn parse(args: &[String]) -> Result<Args> {
    let usage="flash-mtp-train <main.gguf> <mtp.gguf> --data DIR --out DIR [--steps 100 --seq 32 --burn 8 --lr 1e-5 --beta .8 --clip 1 --vocab 0 --seed 42 --eval-every 20 --eval-windows 1 --depth 3 --prefix-data DIR --resume DIR --cpu --check-context]";
    let mut it = args.iter();
    let main = it.next().context(usage)?.into();
    let mtp = it.next().context(usage)?.into();
    let mut a = Args {
        main,
        mtp,
        data: PathBuf::new(),
        out: PathBuf::new(),
        steps: 100,
        seq: 32,
        burn: 8,
        lr: 1e-5,
        beta: 0.8,
        clip: 1.0,
        vocab: 0,
        seed: 42,
        cpu: false,
        eval_every: 20,
        resume: None,
        check_forward: false,
        check_context: false,
        prefix_data: None,
        eval_windows: 1,
        depth: 3,
    };
    while let Some(k) = it.next() {
        if k == "--cpu" {
            a.cpu = true;
            continue;
        }
        if k == "--check-context" {
            a.check_context = true;
            continue;
        }
        if k == "--check-forward" {
            a.check_forward = true;
            continue;
        }
        let v = it.next().with_context(|| format!("{k} needs a value"))?;
        match k.as_str() {
            "--data" => a.data = v.into(),
            "--prefix-data" => a.prefix_data = Some(v.into()),
            "--eval-windows" => a.eval_windows = v.parse()?,
            "--depth" => a.depth = v.parse()?,
            "--out" => a.out = v.into(),
            "--steps" => a.steps = v.parse()?,
            "--seq" => a.seq = v.parse()?,
            "--burn" => a.burn = v.parse()?,
            "--lr" => a.lr = v.parse()?,
            "--beta" => a.beta = v.parse()?,
            "--clip" => a.clip = v.parse()?,
            "--vocab" => a.vocab = v.parse()?,
            "--seed" => a.seed = v.parse()?,
            "--eval-every" => a.eval_every = v.parse()?,
            "--resume" => a.resume = Some(v.into()),
            _ => bail!("unknown option {k}: {usage}"),
        }
    }
    ensure!(
        !a.data.as_os_str().is_empty() && !a.out.as_os_str().is_empty(),
        "{usage}"
    );
    ensure!((1..=8).contains(&a.depth), "--depth must be 1..=8");
    ensure!(
        a.seq > a.depth && a.burn < a.seq - a.depth,
        "invalid sequence/burn length"
    );
    ensure!(
        a.eval_windows > 0
            && a.eval_every > 0
            && a.lr > 0.0
            && a.lr.is_finite()
            && a.beta > 0.0
            && a.beta <= 1.0
            && a.clip > 0.0
            && a.clip.is_finite(),
        "invalid training hyperparameters"
    );
    Ok(a)
}
pub fn cli(args: &[String]) -> Result<()> {
    let a = parse(args)?;
    if a.cpu {
        return run(&tang_compute::CpuDevice::new(), a);
    }
    #[cfg(feature = "cuda")]
    {
        let dev = tang_compute::CudaComputeDevice::new().map_err(|e| anyhow::anyhow!("{e}"))?;
        dev.set_tf32(false);
        run(&dev, a)
    }
    #[cfg(not(feature = "cuda"))]
    {
        bail!("build with cuda,mtp-train or pass --cpu")
    }
}
fn random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn run<D: ComputeDevice>(dev: &D, a: Args) -> Result<()> {
    ensure!(
        !a.out.exists(),
        "output directory already exists: {}",
        a.out.display()
    );
    let mut w = Weights::open(&a.main, &a.mtp, a.lr, a.vocab)?;
    let hp = match w.kind {
        weights::Kind::Coder => Hparams::from_gguf(&w.main)?,
        weights::Kind::DeepSeek => Hparams::from_gguf(&w.mtp)?,
    };
    let mut seqs = data::discover(&a.data, hp.hc * hp.n_embd)?;
    if let Some(root) = &a.prefix_data {
        for s in &mut seqs {
            s.attach_prefix(root)?;
        }
    }
    let eligible: Vec<_> = seqs
        .iter()
        .enumerate()
        .filter(|(_, s)| s.ids.len() >= a.seq)
        .map(|(i, _)| i)
        .collect();
    let valid: Vec<_> = eligible
        .iter()
        .copied()
        .filter(|&i| prompt_number(&seqs[i].path).is_multiple_of(10))
        .collect();
    let train: Vec<_> = eligible
        .iter()
        .copied()
        .filter(|&i| !prompt_number(&seqs[i].path).is_multiple_of(10))
        .collect();
    ensure!(
        !train.is_empty() && !valid.is_empty(),
        "need completed training and held-out prompts (pNNN % 10 == 0 held out)"
    );
    if a.check_context {
        return check_context(dev, &mut w, &seqs[valid[0]], &hp, &a);
    }
    if a.check_forward {
        if w.kind == weights::Kind::Coder {
            return check_forward_coder(dev, &mut w, &seqs, &valid, &hp, &a);
        }
        return check_forward(dev, &mut w, &seqs[valid[0]], &hp, &a);
    }
    let mut rng = a.seed;
    let mut offset = 0;
    if let Some(p) = &a.resume {
        let restored = restore(p, &mut w)?;
        offset = restored.0;
        rng = restored.1;
    }
    std::fs::create_dir_all(&a.out)?;
    let manifest = serde_json::json!({"main":a.main,"mtp":a.mtp,"data":a.data,"prefix_data":a.prefix_data,"eval_windows":a.eval_windows,"eval_every":a.eval_every,"depth":a.depth,"kind":format!("{:?}",w.kind),"context":if a.prefix_data.is_some(){"full-teacher-prefix"}else{"short-window"},"steps":a.steps,"seq":a.seq,"burn":a.burn,"lr":a.lr,"beta":a.beta,"clip":a.clip,"vocab":a.vocab,"seed":a.seed,"resume":a.resume,"start_step":offset,"train":train.iter().map(|&i|&seqs[i].path).collect::<Vec<_>>(),"valid":valid.iter().map(|&i|&seqs[i].path).collect::<Vec<_>>(),"trainable":w.params.keys().collect::<Vec<_>>()});
    std::fs::write(
        a.out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    eprintln!("MTP train: {} train / {} held-out prompts; {} parameters; frozen {} routed experts, main embedding/head; {}-row windows, {} burn-in, {} depths; CPU={} start={offset}",train.len(),valid.len(),w.params.values().map(|p|p.data.numel()).sum::<usize>(),if w.expert_q4{"Q4"}else{"Q8"},a.seq,a.burn,a.depth,a.cpu);
    let mut log = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(a.out.join("metrics.jsonl"))?;
    evaluate(dev, &mut w, &seqs, &valid, &hp, &a, offset, &mut log)?;
    for local in 1..=a.steps {
        let step = offset + local;
        let s = &seqs[train[random(&mut rng) as usize % train.len()]];
        let start = random(&mut rng) as usize % (s.ids.len() - a.seq + 1);
        let b = s.batch(start, a.seq)?;
        let now = Instant::now();
        let mut t = tape::Tape::new(dev);
        let leaves = w.leafs(&mut t);
        let logits = model::forward(&mut t, &mut w, &leaves, &b, &hp, a.depth)?;
        let (loss, seeds, hits, totals) = model::objective(&t, &w, &logits, &b, a.beta, a.burn)?;
        ensure!(loss.iter().all(|v| v.is_finite()), "nonfinite loss");
        let grads = t.backward(&seeds);
        drop(t);
        let norm = w.update(&leaves, grads, a.clip)?;
        dev.pool_clear();
        let metric = serde_json::json!({"kind":"train","step":step,"prompt":s.path,"start":start,"prefix_rows":b.prefix.as_ref().map_or(0,|p|p.pos.len()),"ce":loss,"teacher_hits":hits,"teacher_total":totals,"grad_norm":norm,"seconds":now.elapsed().as_secs_f64()});
        writeln!(log, "{metric}")?;
        log.flush()?;
        eprintln!(
            "step {step}: CE {}, grad {norm:.3}, {:.2}s",
            loss.iter()
                .map(|l| format!("{l:.4}"))
                .collect::<Vec<_>>()
                .join("/"),
            now.elapsed().as_secs_f64()
        );
        if local % a.eval_every == 0 || local == a.steps {
            evaluate(dev, &mut w, &seqs, &valid, &hp, &a, step, &mut log)?;
            let dir = a.out.join(format!("step-{step:06}"));
            std::fs::create_dir(&dir)?;
            checkpoint(&dir, &w, step, rng)?;
            w.export(&dir.join("mtp.gguf"))?;
        }
    }
    Ok(())
}
fn prompt_number(p: &Path) -> usize {
    p.file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_prefix('p'))
        .and_then(|n| n.parse().ok())
        .unwrap_or(usize::MAX)
}
fn evaluate<D: ComputeDevice>(
    dev: &D,
    w: &mut Weights,
    seqs: &[data::Sequence],
    valid: &[usize],
    hp: &Hparams,
    a: &Args,
    step: usize,
    log: &mut FileLike,
) -> Result<()> {
    let mut loss = vec![0.0; a.depth];
    let mut hits = vec![0; a.depth];
    let mut total = vec![0; a.depth];
    // Fixed slices per prompt, never random windows from training prompts. Report teacher-token
    // recursive accuracy as a diagnostic; actual greedy draft acceptance is measured by the engine.
    for &i in valid {
        let s = &seqs[i];
        for window in 0..a.eval_windows {
            let start = if a.eval_windows == 1 {
                (s.ids.len() - a.seq) / 2
            } else {
                (s.ids.len() - a.seq) * window / (a.eval_windows - 1)
            };
            let b = s.batch(start, a.seq)?;
            let mut t = tape::Tape::new(dev);
            let leaves = w.leafs(&mut t);
            let lg = model::forward(&mut t, w, &leaves, &b, hp, a.depth)?;
            let (l, _, h, n) = model::objective(&t, w, &lg, &b, a.beta, a.burn)?;
            for d in 0..a.depth {
                loss[d] += l[d] * n[d] as f64;
                hits[d] += h[d];
                total[d] += n[d];
            }
            writeln!(
                log,
                "{}",
                serde_json::json!({"kind":"valid_window","step":step,"prompt":s.path,"start":start,"prefix_rows":b.prefix.as_ref().map_or(0,|p|p.pos.len()),"ce":l,"teacher_hits":h,"teacher_total":n})
            )?;
            drop(t);
            dev.pool_clear();
        }
    }
    for d in 0..a.depth {
        loss[d] /= total[d] as f64;
    }
    let metric = serde_json::json!({"kind":"valid","step":step,"ce":loss,"teacher_hits":hits,"teacher_total":total});
    writeln!(log, "{metric}")?;
    log.flush()?;
    eprintln!("valid {step}: {metric}");
    Ok(())
}
type FileLike = std::fs::File;
fn checkpoint(dir: &Path, w: &Weights, step: usize, rng: u64) -> Result<()> {
    let (mut f, mut meta) = (
        BufWriter::with_capacity(1 << 20, std::fs::File::create(dir.join("state.bin"))?),
        Vec::new(),
    );
    for (n, p) in &w.params {
        meta.push((n, p.data.numel()));
        for v in p.data.data() {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    let (m, v, t) = w.optimizer.state_vecs();
    for vectors in [m, v] {
        for values in vectors {
            for x in values {
                f.write_all(&x.to_le_bytes())?;
            }
        }
    }
    f.flush()?;
    f.get_ref().sync_all()?;
    std::fs::write(
        dir.join("state.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"version":1,"step":step,"rng":rng,"adam_step":t,"params":meta}),
        )?,
    )?;
    Ok(())
}
fn restore(dir: &Path, w: &mut Weights) -> Result<(usize, u64)> {
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("state.json"))?)?;
    ensure!(meta["version"] == 1, "checkpoint version");
    let shape: Vec<(String, usize)> = serde_json::from_value(meta["params"].clone())?;
    ensure!(
        shape
            == w.params
                .iter()
                .map(|(n, p)| (n.clone(), p.data.numel()))
                .collect::<Vec<_>>(),
        "checkpoint parameter geometry"
    );
    let mut f = BufReader::with_capacity(1 << 20, std::fs::File::open(dir.join("state.bin"))?);
    for (n, p) in &mut w.params {
        for x in p.data.data_mut() {
            let mut b = [0; 4];
            f.read_exact(&mut b)?;
            *x = f32::from_le_bytes(b);
        }
        ensure!(
            p.data.data().iter().all(|v| v.is_finite()),
            "nonfinite restored weight"
        );
        w.dense.get_mut(n).unwrap().value = std::sync::Arc::new(p.data.data().to_vec());
    }
    let mut states = Vec::new();
    for _ in 0..2 {
        let mut s = Vec::new();
        for (_, n) in &shape {
            let mut v = Vec::with_capacity(*n);
            for _ in 0..*n {
                let mut b = [0; 8];
                f.read_exact(&mut b)?;
                v.push(f64::from_le_bytes(b));
            }
            ensure!(v.iter().all(|v| v.is_finite()), "nonfinite Adam state");
            s.push(v);
        }
        states.push(s);
    }
    ensure!(f.read(&mut [0u8; 1])? == 0, "trailing checkpoint bytes");
    let v = states.pop().unwrap();
    let m = states.pop().unwrap();
    w.optimizer.load_state_vecs(
        m,
        v,
        meta["adam_step"].as_u64().context("Adam step")? as usize,
    );
    Ok((
        meta["step"].as_u64().context("step")? as usize,
        meta["rng"].as_u64().context("rng")?,
    ))
}

fn check_forward<D: ComputeDevice>(
    dev: &D,
    w: &mut Weights,
    s: &data::Sequence,
    hp: &Hparams,
    a: &Args,
) -> Result<()> {
    ensure!(a.vocab == 0, "oracle check needs full vocabulary");
    w.expert_q4 = false;
    let b = s.batch(0, 4)?;
    let mut t = tape::Tape::new(dev);
    let leaves = w.leafs(&mut t);
    let out = model::forward_full(&mut t, w, &leaves, &b, hp, 3)?;
    let main = crate::flash::reference::FlashRef::open(&a.main)?;
    let mut mtp = crate::flash::mtp::Mtp::open(&a.mtp)?;
    mtp.main_head = true;
    let oracle = crate::flash::mtp::training_oracle(&main, &mtp, &b.h, &b.tokens, &b.pos)?;
    for (d, (o, (r, top))) in out.iter().zip(oracle).enumerate() {
        let got = t.data(o.r);
        let max = got
            .iter()
            .zip(&r)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let rms = (got
            .iter()
            .zip(&r)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            / r.len() as f64)
            .sqrt();
        let scale = (r.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / r.len() as f64).sqrt();
        eprintln!(
            "oracle depth {}: residual max_abs {max:.6}, relative RMS {:.6}",
            d + 1,
            rms / scale.max(1e-9)
        );
        ensure!(
            rms / scale.max(1e-9) < 0.002,
            "recursive residual parity failed"
        );
        for (row, expected) in t.data(o.logits).chunks(w.vocab.len()).zip(top) {
            let j = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
                .unwrap()
                .0;
            ensure!(
                w.vocab[j] == expected.0,
                "oracle top token differs at depth {}",
                d + 1
            );
        }
    }
    eprintln!("recursive CPU oracle parity: PASS");
    Ok(())
}

/// Splitting a teacher prefix must preserve recursive outputs and parameter gradients.
fn check_context<D: ComputeDevice>(
    dev: &D,
    w: &mut Weights,
    s: &data::Sequence,
    hp: &Hparams,
    a: &Args,
) -> Result<()> {
    ensure!(
        a.prefix_data.is_some(),
        "--check-context requires --prefix-data"
    );
    let full = s.batch(0, 12)?;
    let split = s.batch(4, 8)?;
    let mut t = tape::Tape::new(dev);
    let leaves = w.leafs(&mut t);
    let out = model::forward_full(&mut t, w, &leaves, &split, hp, a.depth)?;
    let logits: Vec<_> = out.iter().map(|o| o.logits).collect();
    let (_, seeds, _, _) = model::objective(&t, w, &logits, &split, a.beta, 0)?;
    let expected: Vec<_> = out
        .iter()
        .map(|o| (t.data(o.r).to_vec(), t.data(o.logits).to_vec()))
        .collect();
    let grads = t.backward(&seeds);
    let split_grads: std::collections::BTreeMap<_, _> = leaves
        .iter()
        .map(|(name, &id)| {
            (
                name.clone(),
                grads[id]
                    .clone()
                    .unwrap_or_else(|| vec![0.0; t.data(id).len()]),
            )
        })
        .collect();
    drop(t);
    dev.pool_clear();
    let mut t = tape::Tape::new(dev);
    let leaves = w.leafs(&mut t);
    let out = model::forward_full(&mut t, w, &leaves, &full, hp, a.depth)?;
    let mut full_seeds = Vec::new();
    for (d, o) in out.iter().enumerate() {
        for (label, id, width, expected) in [
            ("residual", o.r, hp.hc * hp.n_embd, &expected[d].0),
            ("logits", o.logits, w.vocab.len(), &expected[d].1),
        ] {
            let values = &t.data(id)[4 * width..4 * width + expected.len()];
            let rms = (values
                .iter()
                .zip(expected)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum::<f64>()
                / expected.len() as f64)
                .sqrt();
            let scale = (expected.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>()
                / expected.len() as f64)
                .sqrt();
            eprintln!(
                "context depth {} {label}: relative RMS {}",
                d + 1,
                rms / scale.max(1e-9)
            );
            ensure!(
                rms / scale.max(1e-9) < 0.002,
                "split context output differs"
            );
        }
        let mut g = vec![0.0; t.data(o.logits).len()];
        g[4 * w.vocab.len()..4 * w.vocab.len() + seeds[d].1.len()].copy_from_slice(&seeds[d].1);
        full_seeds.push((o.logits, g));
    }
    let grads = t.backward(&full_seeds);
    for (name, &id) in &leaves {
        let expected = &split_grads[name];
        let got = grads[id].as_deref().unwrap_or(&[]);
        if got.is_empty() {
            ensure!(
                expected.iter().all(|&g| g == 0.0),
                "missing prefix gradient {name}"
            );
            continue;
        }
        let error = got
            .iter()
            .zip(expected)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = expected
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>()
            .sqrt();
        ensure!(
            error / scale.max(1e-8) < 0.02,
            "prefix gradient differs for {name}: {}",
            error / scale.max(1e-8)
        );
    }
    eprintln!("full-prefix split outputs and gradients: PASS");
    Ok(())
}

/// The engine's own MTP drafts dumped with the residuals (`draft.u32`, `flash-coder-dump`):
/// `(position of h, depth, token, probability bits)` per record. Depth 1 for every row; deeper
/// only at the chain's row (each window's last kept row).
pub(crate) fn read_drafts(
    dir: &Path,
) -> Result<std::collections::HashMap<(usize, usize), (u32, f32)>> {
    let bytes = std::fs::read(dir.join("draft.u32"))
        .with_context(|| format!("{}: no draft.u32 (dump with flash-coder-dump)", dir.display()))?;
    ensure!(bytes.len() % 16 == 0, "{}: partial draft record", dir.display());
    Ok(bytes
        .chunks_exact(16)
        .map(|b| {
            let u = |i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
            ((u(0) as usize, u(1) as usize), (u(2), f32::from_bits(u(3))))
        })
        .collect())
}

/// Forward oracle for the Coder MTP: the trainer cell on dumped rows against the engine's own
/// draft argmax (and its probability) for the same rows. Deeper depths are compared only where
/// the engine's earlier drafts equal the teacher tokens the trainer feeds. With `--prefix-data`
/// the cell sees every earlier teacher K/V, as the engine does; without it, early rows of a
/// window miss context and agreement drops.
fn check_forward_coder<D: ComputeDevice>(
    dev: &D,
    w: &mut Weights,
    seqs: &[data::Sequence],
    valid: &[usize],
    hp: &Hparams,
    a: &Args,
) -> Result<()> {
    if a.prefix_data.is_none() {
        eprintln!("warning: no --prefix-data; windows miss earlier MTP K/V and agreement is approximate");
    }
    let mut agree = vec![0usize; a.depth];
    let mut total = vec![0usize; a.depth];
    let mut dprob = vec![0f64; a.depth];
    for &i in valid.iter().take(4) {
        let s = &seqs[i];
        let drafts = read_drafts(&s.path)?;
        let mut start = 0;
        for _ in 0..4 {
            if start + a.seq > s.ids.len() {
                break;
            }
            let b = s.batch(start, a.seq)?;
            let mut t = tape::Tape::new(dev);
            let leaves = w.leafs(&mut t);
            let out = model::forward_full(&mut t, w, &leaves, &b, hp, a.depth)?;
            let vocab = w.vocab.len();
            for (d0, o) in out.iter().enumerate() {
                let d = d0 + 1;
                for (r, row) in t.data(o.logits).chunks(vocab).enumerate() {
                    let p = b.pos[r];
                    let Some(&(want, prob)) = drafts.get(&(p, d)) else {
                        continue;
                    };
                    // The engine chained on its own drafts; compare where they were the teacher's.
                    if (1..d).any(|j| drafts.get(&(p, j)).map(|x| x.0) != Some(b.tokens[r + j + 1])) {
                        continue;
                    }
                    let (j, mx) = row
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
                        .map(|(j, &m)| (j, m))
                        .unwrap();
                    let z: f64 = row.iter().map(|&v| f64::from(v - mx).exp()).sum();
                    total[d0] += 1;
                    agree[d0] += usize::from(w.vocab[j] == want);
                    dprob[d0] += (1.0 / z - f64::from(prob)).abs();
                }
            }
            drop(t);
            dev.pool_clear();
            start += a.seq;
        }
    }
    let mut ok = true;
    for d in 0..a.depth {
        let rate = agree[d] as f64 / total[d].max(1) as f64;
        eprintln!(
            "engine oracle depth {}: argmax {}/{} ({:.4}), mean |Δp| {:.4}",
            d + 1,
            agree[d],
            total[d],
            rate,
            dprob[d] / total[d].max(1) as f64
        );
        // int8 activations, bf16 K/V and f16 dumps move near-ties; a port error moves most rows.
        ok &= total[d] == 0 || rate >= if d == 0 { 0.97 } else { 0.9 };
    }
    ensure!(total[0] > 0, "no comparable rows");
    ensure!(ok, "engine MTP oracle parity failed");
    eprintln!("Coder MTP engine oracle parity: PASS");
    Ok(())
}
