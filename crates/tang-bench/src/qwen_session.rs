//! Qwen3 decode on a GPU through the tensor IR, as a library: the step
//! programs (`Step`: decode, or a prefill of `seq` tokens), the MLX 4-bit
//! checkpoint reader (`St`) and the device-input conversion, moved from
//! `examples/qwen_decode.rs`; and [`Session`], a chat-turn API over them
//! that keeps the KV cache across turns (prefix reuse: the system prompt is
//! prefilled once).
use std::collections::HashMap;
use std::time::Instant;

use crate::ir_qwen::*;
use tang_compile::tensor::cuda_gemv::DecodeProgram;
use tang_compile::tensor::wgsl::WgslProgram;
use tang_compile::tensor::{TensorGraph, TensorId};
use tang_gpu::program::{BackendKind, BufferPlan, Program, Runtime, Shared};

/// CUDA when built with it and a device is present, else wgpu (Metal:
/// the MSL decode kernels).
pub fn runtime() -> Runtime {
    // TANG_BACKEND=opencl: the OpenCL decode kernels (the K1's Adreno)
    #[cfg(feature = "opencl")]
    if std::env::var("TANG_BACKEND").as_deref() == Ok("opencl") {
        return Runtime::opencl().expect("no OpenCL GPU");
    }
    #[cfg(feature = "cuda")]
    if let Some(rt) = Runtime::cuda_capturable() {
        return rt;
    }
    Runtime::wgpu().expect("no GPU")
}

pub fn lower(
    rt: &Runtime,
    g: &TensorGraph,
    outs: &[TensorId],
    opts: &tang_compile::tensor::cuda_gemv::DecodeOpts,
) -> DecodeProgram {
    match rt {
        // Vulkan (the K1's Adreno, ...): the decode kernels as GLSL,
        // compiled to SPIR-V by tang-gpu (feature `glsl`)
        Runtime::Wgpu { device, .. } if !device.msl_passthrough && device.spirv_passthrough => {
            g.to_glsl_decode(outs, opts)
        }
        _ => match rt.kind() {
            BackendKind::Cuda => g.to_cuda_decode(outs, opts),
            BackendKind::Wgpu => g.to_msl_decode(outs, opts),
            BackendKind::OpenCl => g.to_opencl_decode(outs, opts),
        },
    }
}

/// OpenCL prefill over the packed 4-bit weights (dequantized in the
/// register-block kernel) instead of materialized f32 copies per pass;
/// `TANG_CL_PREFILL_Q4=0` restores the f32 maps.
pub fn prefill_q4(rt: &Runtime) -> bool {
    rt.kind() == BackendKind::OpenCl
        && std::env::var("TANG_CL_PREFILL_Q4").as_deref() != Ok("0")
        && std::env::var("TANG_CL_MM").as_deref() != Ok("off")
}

pub fn env(k: &str) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// The program's buffers on slots (`ir_qwen::plan_slots`): in-place
/// outputs share their donor's, intermediates are reused by liveness.
/// The program's buffers on slots (`ir_qwen::plan_slots`): in-place
/// outputs share their donor's, intermediates are reused by liveness.
pub fn slot_plan(
    program: &WgslProgram,
    outputs: &[TensorId],
    inplace: &[(TensorId, usize)],
) -> BufferPlan {
    let (slot, slot_len) = plan_slots(program, outputs, inplace);
    BufferPlan {
        slot,
        slot_len,
        order: (0..program.kernels.len()).collect(),
    }
}

/// One step program: decode (`seq == 1`) or a prefill of `seq` tokens.
/// One step program: decode (`seq == 1`) or a prefill of `seq` tokens.
pub struct Step {
    pub dg: DecodeGraph,
    pub prog: Program,
    /// `dg.outputs` in the compiled graph, then `dg.keep`; with
    /// `QWEN_KEEP_ALL`, every kernel's result after those (from `named`).
    pub outs: Vec<TensorId>,
    /// How many of `outs` are the step's own outputs.
    pub named: usize,
    /// The compiled graph `outs` index (for checking kernels one by one).
    pub g: TensorGraph,
    pub native: Vec<bool>,
    /// Bytes this program allocated (shared buffers count for their owner).
    pub bytes: usize,
}

impl Step {
    /// Lower and allocate the step; inputs from `inputs` (device words,
    /// per graph input), or — every input named like one of `base`'s
    /// — bound to `base`'s buffers.
    pub fn new(
        rt: &Runtime,
        dims: &QwenDims,
        seq: usize,
        kctx: usize,
        inputs: &dyn Fn(&DecodeGraph) -> Vec<Vec<f32>>,
        base: Option<&Step>,
    ) -> Self {
        let t = Instant::now();
        let dims = &QwenDims {
            prefill_q4: dims.prefill_q4 || prefill_q4(rt),
            ..dims.clone()
        };
        let dg = if derived_mode() {
            // QWEN_DERIVED=1: the step derived from the one prefill graph
            // (`ir_qwen::prefill`) instead of the hand-written `step_graph`;
            // its position is the `pos` input, its tokens `tokens`
            let pg = prefill::prefill_graph(dims);
            // QWEN_DERIVED_ROWS=last: a window computes only its last
            // row's logits (as the hand prefill step does)
            let rows = match std::env::var("QWEN_DERIVED_ROWS").as_deref() {
                Ok("last") => tang_compile::tensor::step::Rows::Last,
                _ => tang_compile::tensor::step::Rows::All,
            };
            let step = prefill::derived_step_rows(&pg, dims, seq, rows);
            prefill::as_decode_graph(&pg, &step, dims)
        } else {
            step_graph(dims, seq, kctx)
        };
        let (g, mut outs) = compiled_decode(&dg, rt.kind() == BackendKind::Cuda);
        let named = outs.len();
        let options = |outs: &[TensorId]| {
            let mut opts = decode_opts(&dg, outs);
            opts.rows = env("TANG_DECODE_ROWS");
            opts.warps = env("TANG_DECODE_WARPS");
            // CUDA decode: 8 GEMV rows per warp, 2 warps per block,
            // amortize each lane's activation loads over more weight words
            // (10.0 -> 6.7 ms per Qwen3-8B step on the 3090 with the
            // materialized norms and split-K of `compiled_decode`;
            // BENCHMARKS.md)
            if (seq == 1 || dg.gemv_window()) && rt.kind() == BackendKind::Cuda {
                if opts.rows == 0 {
                    opts.rows = 8;
                }
                if opts.warps == 0 {
                    opts.warps = 2;
                }
            }
            opts.unroll = env("TANG_DECODE_UNROLL");
            opts.mma_shape = env("TANG_MMA_SHAPE");
            opts
        };
        let mut opts = options(&outs);
        let mut dp = lower(rt, &g, &outs, &opts);
        // QWEN_KEEP_ALL=POS: every kernel's result is an output too, so a
        // test can compare each against the interpreter (debugging aid;
        // it changes fusion where a result had been inlined)
        if std::env::var_os("QWEN_KEEP_ALL").is_some() {
            for k in &dp.program.kernels {
                if !outs.contains(&k.node) {
                    outs.push(k.node);
                }
            }
            opts = options(&outs);
            dp = lower(rt, &g, &outs, &opts);
        }
        let native = dp.native.clone();
        let mut program = dp.program;
        if seq > 1 {
            lazy_order(&mut program);
        }
        if let Ok(dir) = std::env::var("QWEN_DUMP") {
            for (i, k) in program.kernels.iter().enumerate() {
                let ext = if k.source.starts_with("#include") {
                    "metal"
                } else if rt.kind() == BackendKind::Cuda {
                    "cu"
                } else if rt.kind() == BackendKind::OpenCl {
                    "cl"
                } else {
                    "wgsl"
                };
                std::fs::write(format!("{dir}/s{seq}_k{i:03}.{ext}"), &k.source).unwrap();
            }
        }
        let plan = slot_plan(&program, &outs, &opts.inplace);
        let lowered = t.elapsed().as_secs_f64();
        let shared: Vec<Shared> = match base {
            Some(b) => dg
                .inputs
                .iter()
                .enumerate()
                .filter_map(|(k, s)| {
                    // (same name and shape: a derived window's `tokens` is
                    // its own)
                    let j =
                        b.dg.inputs
                            .iter()
                            .position(|bs| bs.name == s.name && bs.shape == s.shape)?;
                    Some(Shared {
                        input: k,
                        from: &b.prog,
                        tensor: b.prog.input(j),
                    })
                })
                .collect(),
            None => Vec::new(),
        };
        // own inputs left empty (a prefill's chunk) start zeroed
        let words = inputs(&dg);
        let refs: Vec<&[f32]> = words.iter().map(|v| v.as_slice()).collect();
        let owned: usize = {
            let mut borrowed = vec![false; plan.slot_len.len()];
            for sh in &shared {
                let i = program
                    .buffers
                    .iter()
                    .position(|b| b.input == Some(sh.input))
                    .unwrap();
                borrowed[plan.slot[i]] = true;
            }
            plan.slot_len
                .iter()
                .zip(&borrowed)
                .filter(|(_, b)| !**b)
                .map(|(n, _)| 4 * n)
                .sum()
        };
        let t = Instant::now();
        let mut prog = Program::with_shared(rt, program, &refs, &plan, &shared);
        drop(words);
        // TANG_MEGA=1: the decode step as one persistent kernel
        // (TANG_MEGA_THREADS per block, TANG_MEGA_PER_SM blocks per SM,
        // stages of at most TANG_MEGA_DUP blocks replicated per block,
        // TANG_MEGA_MINB min blocks per SM, TANG_MEGA_MAX_BLOCKS: only kernels
        // of at most that many blocks fuse (runs between the big ones), TANG_MEGA_PREFETCH KiB of weights
        // prefetched into L2 at each barrier)
        #[cfg(feature = "cuda")]
        if seq == 1 && rt.kind() == BackendKind::Cuda && env("TANG_MEGA") == 1 {
            let tm = Instant::now();
            let (launches, phases, grid) = prog.megakernel(
                rt,
                &tang_compile::tensor::cuda_mega::MegaOpts {
                    threads: env("TANG_MEGA_THREADS") as u32,
                    dup: env("TANG_MEGA_DUP") as u32,
                    min_blocks: env("TANG_MEGA_MINB") as u32,
                    prefetch: env("TANG_MEGA_PREFETCH") << 10,
                    slot_bytes: Vec::new(),
                    trace: env("TANG_MEGA_TRACE") == 1,
                },
                env("TANG_MEGA_PER_SM") as u32,
                env("TANG_MEGA_MAX_BLOCKS") as u32,
            );
            eprintln!(
                "megakernel: {} kernels as {launches} launches, {phases} phases, grid {grid}, built in {:.1}s",
                prog.program.kernels.len(),
                tm.elapsed().as_secs_f64()
            );
        }
        prog.capture(rt);
        eprintln!(
            "step seq {seq} (window {kctx}): {} kernels ({} native), lowered in {lowered:.2}s, \
             compiled + bound in {:.2}s, {:.0} MB allocated ({} inputs shared)",
            prog.program.kernels.len(),
            native.iter().filter(|n| **n).count(),
            t.elapsed().as_secs_f64(),
            owned as f64 / 1e6,
            shared.len()
        );
        Self {
            dg,
            prog,
            outs,
            named,
            g,
            native,
            bytes: owned,
        }
    }

    pub fn tensor(&self, name: &str) -> TensorId {
        self.prog.input(self.dg.input(name))
    }

    /// `(next token, its position)`.
    pub fn state(&self, rt: &Runtime) -> (usize, usize) {
        let s = self.prog.read(rt, self.outs[1]);
        (s[0] as usize, s[1] as usize)
    }

    pub fn set_state(&self, rt: &Runtime, tok: usize, pos: usize) {
        self.prog
            .write(rt, self.tensor("state"), &[tok as f32, pos as f32]);
    }

    /// The full logits: block 0, then the other vocabulary blocks.
    pub fn logits(&self, rt: &Runtime) -> Vec<f32> {
        let mut l = self.prog.read(rt, self.outs[0]);
        let first = self.dg.outputs.len() + self.dg.keep.len();
        for i in 0..self.dg.logit_chunks.len() {
            l.extend(self.prog.read(rt, self.outs[first + i]));
        }
        l
    }

    /// A step derived from the prefill graph (`QWEN_DERIVED=1`): no
    /// `state`; tokens and position are inputs, the next token is the
    /// host's argmax.
    pub fn is_derived(&self) -> bool {
        self.dg.inputs.iter().any(|s| s.name == "pos")
    }

    /// The step's tokens (`seq` of them) at positions `p0..`, before a
    /// run: `state` (decode) or `chunk` (prefill) for the hand step,
    /// `tokens` and `pos` for a derived one.
    pub fn feed(&self, rt: &Runtime, tokens: &[usize], p0: usize) {
        assert_eq!(tokens.len(), self.dg.seq);
        if self.is_derived() {
            let t: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
            self.prog.write(rt, self.tensor("tokens"), &t);
            self.prog.write(rt, self.tensor("pos"), &[p0 as f32]);
        } else if self.dg.seq == 1 {
            self.set_state(rt, tokens[0], p0);
        } else {
            self.prog
                .write(rt, self.tensor("chunk"), &chunk_input(tokens, p0));
        }
    }

    /// The greedy next token after a run: `state`'s, or (derived) the
    /// argmax of the last row's logits.
    pub fn next_token(&self, rt: &Runtime) -> usize {
        if !self.is_derived() {
            return self.state(rt).0;
        }
        let l = self.logits(rt);
        // all new rows, or (QWEN_DERIVED_ROWS=last) the last one
        let rows = self.dg.g.shape(self.dg.outputs[0])[0].n();
        let vocab = l.len() / rows;
        argmax(&l[(rows - 1) * vocab..])
    }

    /// Prefill: `tokens` at positions `p0..`, then `state` holds the next
    /// token and its position.
    pub fn prefill(&self, rt: &Runtime, tokens: &[usize], p0: usize) {
        assert_eq!(tokens.len(), self.dg.seq);
        self.prog
            .write(rt, self.tensor("chunk"), &chunk_input(tokens, p0));
        self.prog.run(rt);
    }
}

/// `QWEN_DERIVED=1`: [`Step`]s are derived from the one prefill graph.
pub fn derived_mode() -> bool {
    std::env::var("QWEN_DERIVED").as_deref() == Ok("1")
}

pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// `QWEN_LOAD_DUTY` (0..1, unset = 1): the fraction of wall time the
/// checkpoint conversion may keep a core busy. A robot without
/// cooling trips its thermal limit while loading at full speed.
pub fn load_duty() -> f64 {
    std::env::var("QWEN_LOAD_DUTY")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map_or(1.0, |d| d.clamp(0.01, 1.0))
}

/// Sleep so the work since `start` was `load_duty()` of the time.
/// Sleep so the work since `start` was `load_duty()` of the time.
pub fn pace(start: Instant) {
    let d = load_duty();
    if d < 1.0 {
        std::thread::sleep(start.elapsed().mul_f64(1.0 / d - 1.0));
    }
}

pub struct St {
    // mapped, not read: on the 10 GB robot a read copy plus the converted
    // words overflow RAM into zram, whose compression heats every core
    pub bytes: memmap2::Mmap,
    pub base: usize,
    pub index: HashMap<String, (String, Vec<usize>, usize, usize)>,
}

impl St {
    pub fn open(path: &str) -> Self {
        let f = std::fs::File::open(path).expect("open safetensors");
        let bytes = unsafe { memmap2::Mmap::map(&f) }.expect("map safetensors");
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
        let mut index = HashMap::new();
        for (k, v) in header.as_object().unwrap() {
            if k == "__metadata__" {
                continue;
            }
            let dtype = v["dtype"].as_str().unwrap().to_string();
            let shape = v["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let o = v["data_offsets"].as_array().unwrap();
            let (a, b) = (
                o[0].as_u64().unwrap() as usize,
                o[1].as_u64().unwrap() as usize,
            );
            index.insert(k.clone(), (dtype, shape, a, b));
        }
        Self {
            bytes,
            base: 8 + n,
            index,
        }
    }

    pub fn raw(&self, name: &str) -> &[u8] {
        let (_, _, a, b) = self
            .index
            .get(name)
            .or_else(|| self.index.get(&tied(name)))
            .unwrap_or_else(|| panic!("missing tensor {name}"));
        &self.bytes[self.base + a..self.base + b]
    }

    /// 32-bit words of a tensor's bytes (u32 or packed bf16 pairs).
    pub fn words(&self, name: &str) -> Vec<f32> {
        self.raw(name)
            .chunks(4)
            .map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap())))
            .collect()
    }

    pub fn bf16(&self, name: &str) -> Vec<f32> {
        self.raw(name)
            .chunks(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()
    }

    /// A 4-bit matrix `[rows, cols]` dequantized exactly (f32).
    pub fn dequant(&self, stem: &str) -> (Vec<f32>, usize, usize) {
        let q = self.raw(&format!("{stem}.weight"));
        let s = self.bf16(&format!("{stem}.scales"));
        let b = self.bf16(&format!("{stem}.biases"));
        let w = format!("{stem}.weight");
        let (_, shape, _, _) = self.index.get(&w).unwrap_or_else(|| &self.index[&tied(&w)]);
        let (rows, cols) = (shape[0], shape[1] * 8);
        let groups = cols / GROUP;
        let mut w = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for k in 0..cols {
                let wi = r * (cols / 8) + k / 8;
                let word = u32::from_le_bytes(q[4 * wi..4 * wi + 4].try_into().unwrap());
                let n = ((word >> (4 * (k % 8))) & 15) as f32;
                let g = r * groups + k / GROUP;
                w[r * cols + k] = n * s[g] + b[g];
            }
        }
        (w, rows, cols)
    }
}

/// Tied embeddings (`tie_word_embeddings`, e.g. Qwen3-0.6B/1.7B/4B): no
/// `lm_head.*` tensors, the LM head is the embedding.
/// Tied embeddings (`tie_word_embeddings`, e.g. Qwen3-0.6B/1.7B/4B): no
/// `lm_head.*` tensors, the LM head is the embedding.
pub fn tied(name: &str) -> String {
    match name.strip_prefix("lm_head.") {
        Some(rest) => format!("model.embed_tokens.{rest}"),
        None => name.to_string(),
    }
}

/// The model's dims from `DIR/config.json` (any Qwen3 size), with a cache
/// of `ctx` positions; Qwen3-8B's when there is no config.
/// The model's dims from `DIR/config.json` (any Qwen3 size), with a cache
/// of `ctx` positions; Qwen3-8B's when there is no config.
pub fn dims_for(dir: &str, ctx: usize) -> QwenDims {
    let mut d = QwenDims::qwen3_8b(ctx);
    let Ok(text) = std::fs::read_to_string(format!("{dir}/config.json")) else {
        return d;
    };
    let c: serde_json::Value = serde_json::from_str(&text).expect("config.json");
    let u = |k: &str, v: &mut usize| {
        if let Some(x) = c[k].as_u64() {
            *v = x as usize;
        }
    };
    u("hidden_size", &mut d.d_model);
    u("num_attention_heads", &mut d.heads);
    u("num_key_value_heads", &mut d.kv_heads);
    u("head_dim", &mut d.head_dim);
    u("intermediate_size", &mut d.ff);
    u("vocab_size", &mut d.vocab);
    u("num_hidden_layers", &mut d.layers);
    if let Some(x) = c["rms_norm_eps"].as_f64() {
        d.eps = x;
    }
    if let Some(x) = c["rope_theta"].as_f64() {
        d.theta = x;
    }
    // bench/lowbit's `export`: GPTQ 3-bit decoder blocks
    d.g3 = c["quantization"]["tang_bits"].as_u64() == Some(3);
    d.hadamard = c["quantization"]["tang_hadamard"].as_u64().unwrap_or(0) as usize;
    d
}

/// Device inputs for a step graph from an MLX checkpoint, converted as the
/// iterator is drawn (empty for state, which starts zeroed).
/// Device inputs for a step graph from an MLX checkpoint, converted as the
/// iterator is drawn (empty for state, which starts zeroed).
/// The normalized Sylvester Hadamard matrix `[n, n]` (f32), as
/// bench/lowbit's `hadamard`: `H[i][j] = (-1)^popcount(i & j) / √n`.
pub fn hadamard_matrix(n: usize) -> Vec<f32> {
    let s = 1.0 / (n as f32).sqrt();
    (0..n * n)
        .map(|e| {
            if (e / n & e % n).count_ones() % 2 == 0 {
                s
            } else {
                -s
            }
        })
        .collect()
}

pub fn load_inputs<'a>(
    st: &'a St,
    dims: &'a QwenDims,
    dg: &'a DecodeGraph,
) -> impl Iterator<Item = Vec<f32>> + 'a {
    let proj = |p: &str| -> &'static [&'static str] {
        match p {
            "qkv" => &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            "o" => &["self_attn.o_proj"],
            "gate" => &["mlp.gate_proj"],
            "up" => &["mlp.up_proj"],
            "down" => &["mlp.down_proj"],
            _ => unreachable!(),
        }
    };
    let suffix = |s: &str| match s {
        "q" => "weight",
        "s" => "scales",
        "b" => "biases",
        _ => unreachable!(),
    };
    let one = move |s: &InputSpec| -> Vec<f32> {
        let n: usize = s.shape.iter().product();
        let name = &s.name;
        if name == "state"
            || name == "chunk"
            || name == "pos"
            || name == "tokens"
            || name.ends_with("cache")
        {
            return Vec::new();
        }
        if name == "rope" {
            return rope_table(dims);
        }
        if name == "hadamard" {
            return hadamard_matrix(dims.hadamard);
        }
        if name == "model.norm" {
            return st.bf16("model.norm.weight");
        }
        // the embedding and LM head, whole (`embed.q`) or as row block
        // `c` of `dims.vocab_chunks` (`embed.2.q`): rows are contiguous
        // in the packed words
        let block = |base: &str, rest: &str| -> Vec<f32> {
            let (c, part) = match rest.split_once('.') {
                Some((c, part)) => (c.parse::<usize>().ok(), part),
                None => (None, rest),
            };
            let w = st.words(&format!("{base}.{}", suffix(part)));
            match c {
                None => w,
                Some(c) => {
                    let per = w.len() / dims.vocab_chunks;
                    w[c * per..(c + 1) * per].to_vec()
                }
            }
        };
        if let Some(rest) = name.strip_prefix("lm_head.") {
            return block("lm_head", rest);
        }
        if let Some(rest) = name.strip_prefix("embed.") {
            return block("model.embed_tokens", rest);
        }
        // model.layers.{l}.{what}
        let rest = name.strip_prefix("model.layers.").unwrap();
        let (l, what) = rest.split_once('.').unwrap();
        let pre = format!("model.layers.{l}.");
        match what {
            "ln1" => st.bf16(&format!("{pre}input_layernorm.weight")),
            "ln2" => st.bf16(&format!("{pre}post_attention_layernorm.weight")),
            "q_norm" => st.bf16(&format!("{pre}self_attn.q_norm.weight")),
            "k_norm" => st.bf16(&format!("{pre}self_attn.k_norm.weight")),
            _ => {
                let (p, part) = what.split_once('.').unwrap();
                if part == "had" {
                    // one rotation per input: q/k/v (gate/up) share signs
                    return st.words(&format!("{pre}{}.had", proj(p)[0]));
                }
                // row-major [rows, …]: concatenating rows is
                // concatenating the word arrays
                let mut v = Vec::new();
                for t in proj(p) {
                    v.extend(st.words(&format!("{pre}{t}.{}", suffix(part))));
                }
                let want = s.packing.unwrap().words(n);
                assert_eq!(v.len(), want, "{name}");
                v
            }
        }
    };
    // QWEN_LOAD_DUTY: hold the conversion to a CPU duty cycle (a hot robot)
    dg.inputs.iter().map(move |s| {
        let t = Instant::now();
        let v = one(s);
        pace(t);
        v
    })
}

/// Prefill chunk sizes a [`Session`] builds programs for (largest first);
/// a remainder below the smallest goes through decode steps.
/// `QWEN_PREFILL_CHUNKS=64,8` overrides. Default `64,8`; on OpenCL
/// `64,32,16,8,4,2`: a voice turn's ~20 new tokens run as 16 + 4 (two
/// prefill runs) instead of 8 + 8 and four decode steps (each a full pass
/// over the weights at decode speed, ~55 ms on the K1), and a tail never
/// needs more than one decode step. Each extra size is one more program
/// per window (its own intermediates: ~0.55 MB per token for Qwen3-1.7B).
fn chunk_sizes(rt: &Runtime) -> Vec<usize> {
    std::env::var("QWEN_PREFILL_CHUNKS")
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|x| x.parse().ok())
                .collect::<Vec<usize>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| {
            if rt.kind() == BackendKind::OpenCl {
                vec![64, 32, 16, 8, 4, 2]
            } else {
                vec![64, 8]
            }
        })
}

/// The attention window of a prefill reaching position `end`: a power of
/// two from 512 (so a handful of programs serve every turn), at most `ctx`.
fn window(end: usize, ctx: usize) -> usize {
    end.next_power_of_two().max(512).min(ctx)
}

/// How a [`Session`] picks the next token.
#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    /// 0: greedy (the device argmax, no logits readback).
    pub temperature: f32,
    pub top_p: f32,
    /// > 1 discourages tokens already in the context (HF convention).
    pub repetition_penalty: f32,
    /// OpenAI's: subtracted from a token's logit once if it is in the reply
    /// so far, and per occurrence.
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: 0,
        }
    }
}

/// A loaded model and its KV cache, for chat turns. The cache holds the
/// tokens of the last prompt and reply ([`Session::cached`]); a new prompt
/// sharing a prefix with them (the system prompt, the conversation so far)
/// prefills only the rest.
pub struct Session {
    pub rt: Runtime,
    pub dims: QwenDims,
    pub dec: Step,
    prefill: HashMap<(usize, usize), Step>,
    pub tokenizer: tokenizers::Tokenizer,
    /// Tokens whose keys and values are in the cache, position order.
    cached: Vec<u32>,
    /// The token the next decode step consumes, and its position.
    next: (u32, usize),
    /// Where the current reply starts in `cached` (presence/frequency penalties).
    reply_start: usize,
    /// The prefill program that ran last (`None`: the decode step).
    last: Option<(usize, usize)>,
    /// Tokens the next pick may not be ([`ban_next`](Self::ban_next)).
    ban: Vec<u32>,
    rng: u64,
    /// Parent of this session's spans (feature `otel`): [`set_trace_parent`](Self::set_trace_parent).
    #[cfg(feature = "otel")]
    trace_parent: Option<tang_otel::Context>,
}

impl Session {
    /// Load `dir` (an MLX 4-bit Qwen3 checkpoint with `config.json` and
    /// `tokenizer.json`) on the default runtime ([`runtime`]: honours
    /// `TANG_BACKEND`), with a KV cache of `ctx` positions.
    pub fn open(dir: &str, ctx: usize) -> Result<Self, String> {
        let tokenizer = tokenizers::Tokenizer::from_file(format!("{dir}/tokenizer.json"))
            .map_err(|e| format!("tokenizer: {e}"))?;
        let mut dims = dims_for(dir, ctx);
        dims.kv_f16 = std::env::var("TANG_KV_F32").is_err();
        let rt = runtime();
        for (k, v) in apply_tuning(&rt, &dims) {
            eprintln!("tang: tuned {k}={v}");
        }
        dims.vocab_chunks = match env("QWEN_VOCAB_CHUNKS") {
            0 => match &rt {
                Runtime::Wgpu { device, .. } if !device.msl_passthrough => 4,
                _ if rt.kind() == BackendKind::OpenCl => {
                    (dims.vocab * dims.d_model / 2).div_ceil(255 << 20).max(1)
                }
                _ => 1,
            },
            n => n,
        };
        let st = St::open(&format!("{dir}/model.safetensors"));
        let dec = Step::new(
            &rt,
            &dims,
            1,
            ctx,
            &|dg| vec![Vec::new(); dg.inputs.len()],
            None,
        );
        for (k, v) in load_inputs(&st, &dims, &dec.dg).enumerate() {
            if !v.is_empty() {
                dec.prog.write(&rt, dec.prog.input(k), &v);
            }
        }
        Ok(Self {
            rt,
            dims,
            dec,
            prefill: HashMap::new(),
            tokenizer,
            cached: Vec::new(),
            next: (0, 0),
            last: None,
            ban: Vec::new(),
            reply_start: 0,
            rng: 0x9e37_79b9_7f4a_7c15,
            #[cfg(feature = "otel")]
            trace_parent: None,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, false).unwrap_or_default()
    }

    /// The single id of a special token such as `<|im_end|>`.
    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.tokenizer.token_to_id(s)
    }

    /// Forbid `toks` for the next token picked (by [`prompt`](Self::prompt)
    /// or [`step`](Self::step)) only, e.g. to block a reply from copying a
    /// span of an earlier turn. Forces the logits readback for that pick.
    pub fn ban_next(&mut self, toks: &[u32]) {
        self.ban.clear();
        self.ban.extend_from_slice(toks);
    }

    /// Put this session's spans (prefill, decode steps, and the OpenCL runs
    /// and builds under them) in the caller's trace: a W3C `traceparent`
    /// (`00-<trace id>-<span id>-01`), e.g. broca's per-turn span. `None`:
    /// spans start their own traces again. Without the `otel` feature this
    /// does nothing. Returns whether the parent was accepted.
    pub fn set_trace_parent(&mut self, traceparent: Option<&str>) -> bool {
        #[cfg(feature = "otel")]
        {
            self.trace_parent = traceparent.and_then(tang_otel::parent_from_traceparent);
            self.trace_parent.is_some() || traceparent.is_none()
        }
        #[cfg(not(feature = "otel"))]
        {
            let _ = traceparent;
            false
        }
    }

    /// [`set_trace_parent`](Self::set_trace_parent) from a trace id (32 hex)
    /// and parent span id (16 hex).
    pub fn set_trace_ids(&mut self, trace_id: &str, span_id: &str) -> bool {
        self.set_trace_parent(Some(&format!("00-{trace_id}-{span_id}-01")))
    }

    /// Forget the cache (the next prompt prefills from position 0).
    pub fn reset(&mut self) {
        self.cached.clear();
    }

    pub fn cached(&self) -> &[u32] {
        &self.cached
    }

    /// Reply tokens that fit after a prompt of `n` tokens.
    pub fn room_for(&self, n: usize) -> usize {
        self.dims.ctx.saturating_sub(n + 1)
    }

    /// Positions still free in the cache.
    pub fn room(&self) -> usize {
        self.dims.ctx.saturating_sub(self.next.1)
    }

    /// Build the prefill programs a prompt of up to `end` positions uses
    /// (compile time off the first turn).
    pub fn warm(&mut self, end: usize) {
        for seq in chunk_sizes(&self.rt) {
            let mut w = 512;
            while w <= window(end, self.dims.ctx) {
                self.step_for(seq, w);
                w *= 2;
            }
        }
    }

    fn step_for(&mut self, seq: usize, kctx: usize) -> &Step {
        if !self.prefill.contains_key(&(seq, kctx)) {
            let s = Step::new(
                &self.rt,
                &self.dims,
                seq,
                kctx,
                &|dg| vec![Vec::new(); dg.inputs.len()],
                Some(&self.dec),
            );
            self.prefill.insert((seq, kctx), s);
        }
        &self.prefill[&(seq, kctx)]
    }

    /// Make the cache hold `tokens` (a whole prompt, from position 0) and
    /// pick the first reply token. Reuses the longest common prefix with
    /// what is cached; returns `(first token, positions reused)`.
    pub fn prompt(&mut self, tokens: &[u32], sampling: &Sampling) -> (u32, usize) {
        assert!(!tokens.is_empty(), "empty prompt");
        assert!(tokens.len() < self.dims.ctx, "prompt longer than the cache");
        let mut lcp = self
            .cached
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count();
        // the last prompt token is always run: its logits pick the reply
        lcp = lcp.min(tokens.len() - 1);
        let mut p = lcp;
        let sizes = chunk_sizes(&self.rt);
        #[cfg(feature = "otel")]
        let tel = tel::Op::start("qwen.prefill", &self.trace_parent);
        #[cfg(feature = "otel")]
        let _attached = tel.as_ref().map(|t| t.cx.clone().attach());
        #[cfg(feature = "otel")]
        let mut chunks: Vec<usize> = Vec::new();
        while p < tokens.len() {
            let left = tokens.len() - p;
            #[cfg(feature = "otel")]
            let c0 = std::time::SystemTime::now();
            match sizes.iter().copied().find(|&s| s <= left) {
                Some(seq) => {
                    let kctx = window(p + seq, self.dims.ctx);
                    let ids: Vec<usize> = tokens[p..p + seq].iter().map(|&t| t as usize).collect();
                    let rt = &self.rt as *const Runtime;
                    let step = self.step_for(seq, kctx);
                    // SAFETY: `rt` outlives this call; step_for only borrows `prefill`
                    step.prefill(unsafe { &*rt }, &ids, p);
                    self.last = Some((seq, kctx));
                    #[cfg(feature = "otel")]
                    if let Some(t) = &tel {
                        t.chunk(c0, seq, kctx, p);
                        chunks.push(seq);
                    }
                    p += seq;
                }
                None => {
                    self.dec.set_state(&self.rt, tokens[p] as usize, p);
                    self.dec.prog.run(&self.rt);
                    self.last = None;
                    #[cfg(feature = "otel")]
                    if let Some(t) = &tel {
                        t.chunk(c0, 1, 0, p);
                        chunks.push(1);
                    }
                    p += 1;
                }
            }
        }
        self.cached = tokens.to_vec();
        self.reply_start = tokens.len();
        let first = self.choose(sampling);
        #[cfg(feature = "otel")]
        if let Some(t) = tel {
            t.end_prefill(tokens.len(), lcp, &chunks);
        }
        (first, lcp)
    }

    /// The reply token after `tok` (the one [`prompt`](Self::prompt) or the
    /// last `step` returned): runs one decode step.
    pub fn step(&mut self, sampling: &Sampling) -> u32 {
        let (tok, pos) = self.next;
        #[cfg(feature = "otel")]
        let tel = tel::Op::start_decode(&self.trace_parent);
        #[cfg(feature = "otel")]
        let _attached = tel.as_ref().map(|t| t.cx.clone().attach());
        self.dec.set_state(&self.rt, tok as usize, pos);
        self.dec.prog.run(&self.rt);
        self.last = None;
        self.cached.push(tok);
        let next = self.choose(sampling);
        #[cfg(feature = "otel")]
        if let Some(t) = tel {
            t.end_decode(pos);
        }
        next
    }

    /// The next token from the state the last run left (greedy: the device
    /// argmax) or by sampling its logits; records it as `next`.
    fn choose(&mut self, s: &Sampling) -> u32 {
        let (argmax_tok, pos) = self.dec.state(&self.rt);
        let plain =
            s.repetition_penalty == 1.0 && s.presence_penalty == 0.0 && s.frequency_penalty == 0.0;
        let tok = if s.temperature <= 0.0 && plain && self.ban.is_empty() {
            argmax_tok as u32
        } else {
            let mut l = self.last_logits();
            for &t in &self.ban {
                if let Some(x) = l.get_mut(t as usize) {
                    *x = f32::NEG_INFINITY;
                }
            }
            self.sample(&mut l, s)
        };
        self.ban.clear();
        self.next = (tok, pos);
        tok
    }

    /// Logits of the last position run, from the program that ran it (the
    /// state is shared by every step program, the logits are each one's own).
    pub fn last_logits(&self) -> Vec<f32> {
        match self.last {
            Some(k) => self.prefill[&k].logits(&self.rt),
            None => self.dec.logits(&self.rt),
        }
    }

    fn sample(&mut self, l: &mut [f32], s: &Sampling) -> u32 {
        if s.repetition_penalty != 1.0 {
            let mut seen = std::collections::HashSet::new();
            for &t in &self.cached {
                if seen.insert(t) {
                    let x = &mut l[t as usize];
                    *x = if *x > 0.0 {
                        *x / s.repetition_penalty
                    } else {
                        *x * s.repetition_penalty
                    };
                }
            }
        }
        if s.presence_penalty != 0.0 || s.frequency_penalty != 0.0 {
            let mut count: HashMap<u32, f32> = HashMap::new();
            for &t in &self.cached[self.reply_start.min(self.cached.len())..] {
                *count.entry(t).or_default() += 1.0;
            }
            for (t, n) in count {
                l[t as usize] -= s.presence_penalty + s.frequency_penalty * n;
            }
        }
        if s.temperature <= 0.0 {
            return argmax(l) as u32;
        }
        if self.rng == 0x9e37_79b9_7f4a_7c15 && s.seed != 0 {
            self.rng ^= s.seed;
        }
        let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut idx: Vec<(f32, u32)> = l
            .iter()
            .enumerate()
            .map(|(i, &x)| (((x - m) / s.temperature).exp(), i as u32))
            .collect();
        idx.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let total: f32 = idx.iter().map(|x| x.0).sum();
        let mut keep = 0.0;
        let mut n = 0;
        for x in &idx {
            keep += x.0;
            n += 1;
            if keep >= s.top_p * total {
                break;
            }
        }
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let u = (self.rng.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32;
        let mut acc = 0.0;
        for x in &idx[..n] {
            acc += x.0 / keep;
            if u < acc {
                return x.1;
            }
        }
        idx[n - 1].1
    }
}

// ---------------------------------------------------------------------------
// tuning: per device and model shape, measured on the device (`tune`), read at
// startup (`apply_tuning`)

/// The tuning file: `$TANG_TUNE`, else `~/.config/tang/tune.json`. A JSON
/// object keyed by [`tune_key`], each value the env knobs that won:
/// `{"QUALCOMM Adreno(TM) 740|qwen3 d2048 ff6144 l28": {"TANG_DECODE_WARPS": "4", …}}`.
pub fn tune_path() -> std::path::PathBuf {
    if let Some(p) = std::env::var_os("TANG_TUNE") {
        return p.into();
    }
    let home = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_default();
    home.join(".config/tang/tune.json")
}

/// Device name and model shape: what a tuning result is valid for.
pub fn tune_key(rt: &Runtime, dims: &QwenDims) -> String {
    format!(
        "{}|qwen3 d{} ff{} l{} h{} kv{}",
        rt.name(),
        dims.d_model,
        dims.ff,
        dims.layers,
        dims.heads,
        dims.kv_heads
    )
}

/// The knobs the tuner sets (decode GEMV shape, OpenCL prefill register block).
pub const TUNE_KNOBS: [&str; 7] = [
    "TANG_DECODE_WARPS",
    "TANG_DECODE_ROWS",
    "TANG_CL_MM",
    "TANG_CL_MM_LANES",
    "TANG_CL_MM_VEC",
    "TANG_CL_MM_KIND",
    "TANG_CL_STAGE",
];

/// Set this device and model's tuned knobs from [`tune_path`], except any
/// already set in the environment (an explicit env var wins). Returns what
/// it applied.
pub fn apply_tuning(rt: &Runtime, dims: &QwenDims) -> Vec<(String, String)> {
    let Ok(text) = std::fs::read_to_string(tune_path()) else {
        return Vec::new();
    };
    let Ok(all) = serde_json::from_str::<serde_json::Value>(&text) else {
        return Vec::new();
    };
    let mut applied = Vec::new();
    if let Some(m) = all[tune_key(rt, dims)].as_object() {
        for (k, v) in m {
            let Some(v) = v.as_str() else { continue };
            if TUNE_KNOBS.contains(&k.as_str()) && std::env::var_os(k).is_none() {
                std::env::set_var(k, v);
                applied.push((k.clone(), v.to_string()));
            }
        }
    }
    applied
}

/// Hottest GPU temperature in °C, if the board exposes one (kgsl on the K1).
fn gpu_temp() -> Option<f64> {
    let t: f64 = std::fs::read_to_string("/sys/class/kgsl/kgsl-3d0/temp")
        .ok()?
        .trim()
        .parse()
        .ok()?;
    Some(if t > 1000.0 { t / 1000.0 } else { t })
}

fn load1() -> Option<f64> {
    std::fs::read_to_string("/proc/loadavg")
        .ok()?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// Prefill matmul candidates of [`tune`]: `(TANG_CL_MM_KIND, config,
/// lanes)`. The first is the default (no knobs set), measured as the
/// baseline; staged before register-block.
const PREFILL_CANDIDATES: &[(&str, &str, &str)] = &[
    ("", "stage 128x8x2x128", ""),
    ("stage", "128x4x2x256", ""),
    ("stage", "128x4x4x128", ""),
    ("stage", "64x4x2x256", ""),
    ("stage", "256x8x2x64", ""),
    ("stage", "128x8x1x128", ""),
    ("stage", "64x8x4x128", ""),
    ("stage", "256x16x2x64", ""),
    ("reg", "4x4x64", "16"),
    ("reg", "4x4x128", "16"),
    ("reg", "4x4x128", "32"),
    ("reg", "4x4x64", "8"),
    ("reg", "4x8x64", "16"),
    ("reg", "8x4x64", "16"),
    ("reg", "2x8x128", "16"),
    ("reg", "0x4x64", "16"),
    ("reg", "4x4x64", "1"),
];

/// The prefill winner among `results` (`(ms, candidate index)`): the
/// fastest, and only if strictly faster than the measured default
/// `baseline`; `None` keeps the default (also with no baseline or no
/// results).
fn pick_prefill(baseline: Option<f64>, results: &[(f64, usize)]) -> Option<(f64, usize)> {
    let b = baseline.filter(|b| b.is_finite())?;
    results
        .iter()
        .copied()
        .filter(|(ms, _)| ms.is_finite() && *ms < b)
        .min_by(|x, y| x.0.total_cmp(&y.0))
}

/// Wait (bounded) for the GPU to cool below `cool` °C and the 1-minute load
/// to fall below `load`; false when `deadline` passed first.
fn settle(hot: f64, cool: f64, load: f64, deadline: Instant) -> bool {
    let mut waited = false;
    loop {
        let t = gpu_temp().unwrap_or(0.0);
        let l = load1().unwrap_or(0.0);
        let too_hot = t >= hot || (waited && t > cool);
        if !too_hot && l < load {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        if !waited {
            eprintln!("tune: waiting (gpu {t:.1} C, load {l:.2})");
        }
        waited = true;
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
}

/// Measure, on this device, the decode GEMV shape (`TANG_DECODE_WARPS` ×
/// `TANG_DECODE_ROWS`) and, on OpenCL, the prefill register block
/// (`TANG_CL_MM`) for the model in `dir`, within `budget` seconds; write the
/// winners into [`tune_path`] under [`tune_key`]. Candidates share the first
/// program's weights and caches (one upload). Pauses at `TANG_TUNE_HOT` °C
/// (default 82) until 75, and while the load is above `TANG_TUNE_LOAD`
/// (default 12). Returns the winners.
pub fn tune(dir: &str, budget: f64) -> Vec<(String, String)> {
    let t0 = Instant::now();
    let deadline = t0 + std::time::Duration::from_secs_f64(budget);
    let fenv = |k: &str, d: f64| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let (hot, load) = (fenv("TANG_TUNE_HOT", 82.0), fenv("TANG_TUNE_LOAD", 12.0));
    for k in TUNE_KNOBS {
        std::env::remove_var(k);
    }
    let ctx = 1024;
    let mut dims = dims_for(dir, ctx);
    dims.kv_f16 = true;
    let rt = runtime();
    if rt.kind() == BackendKind::OpenCl {
        dims.vocab_chunks = (dims.vocab * dims.d_model / 2).div_ceil(255 << 20).max(1);
    }
    let key = tune_key(&rt, &dims);
    eprintln!("tune: {key}, budget {budget:.0} s");
    #[cfg(feature = "otel")]
    let tel = tel::Op::start("qwen.tune", &None);
    #[cfg(feature = "otel")]
    let _attached = tel.as_ref().map(|t| t.cx.clone().attach());
    #[cfg(feature = "otel")]
    let mut candidates = 0i64;
    let st = St::open(&format!("{dir}/model.safetensors"));
    let base = Step::new(
        &rt,
        &dims,
        1,
        ctx,
        &|dg| vec![Vec::new(); dg.inputs.len()],
        None,
    );
    for (k, v) in load_inputs(&st, &dims, &base.dg).enumerate() {
        if !v.is_empty() {
            base.prog.write(&rt, base.prog.input(k), &v);
        }
    }
    let time_step = |s: &Step, pos: usize| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            s.set_state(&rt, 1, pos);
            best = best.min(s.prog.time(&rt, 10));
        }
        best
    };
    let mut won = Vec::new();
    // prefill: (measured default ms, winning candidate ms if one beat it)
    let mut prefill_ms: Option<(f64, Option<f64>)> = None;
    // decode: warps x rows
    let mut best: Option<(f64, usize, usize)> = None;
    for (w, r) in [(4, 2), (4, 4), (8, 4), (4, 8), (2, 4), (8, 2), (8, 8)] {
        if !settle(hot, 75.0, load, deadline) {
            eprintln!("tune: budget spent");
            break;
        }
        std::env::set_var("TANG_DECODE_WARPS", w.to_string());
        std::env::set_var("TANG_DECODE_ROWS", r.to_string());
        #[cfg(feature = "otel")]
        let c0 = std::time::SystemTime::now();
        let s = Step::new(
            &rt,
            &dims,
            1,
            ctx,
            &|dg| vec![Vec::new(); dg.inputs.len()],
            Some(&base),
        );
        let ms = time_step(&s, 64);
        eprintln!(
            "tune: decode warps {w} rows {r}: {ms:.2} ms (gpu {:.1} C)",
            gpu_temp().unwrap_or(0.0)
        );
        #[cfg(feature = "otel")]
        if let Some(t) = &tel {
            candidates += 1;
            t.candidate(c0, "decode", &format!("warps={w} rows={r}"), ms, gpu_temp());
        }
        if best.is_none_or(|b| ms < b.0) {
            best = Some((ms, w, r));
        }
    }
    std::env::remove_var("TANG_DECODE_WARPS");
    std::env::remove_var("TANG_DECODE_ROWS");
    if let Some((_, w, r)) = best {
        won.push(("TANG_DECODE_WARPS".to_string(), w.to_string()));
        won.push(("TANG_DECODE_ROWS".to_string(), r.to_string()));
    }
    // prefill matmul kernel (OpenCL): the register block or the staged one
    if rt.kind() == BackendKind::OpenCl {
        // a voice turn's size (QWEN_TUNE_SEQ overrides)
        let seq = match env("QWEN_TUNE_SEQ") {
            0 => 20,
            n => n,
        };
        let toks: Vec<usize> = (0..seq).map(|i| 1000 + i).collect();
        // register block: (RMxRNxWG, lanes along k), local partials
        // RM*RN*WG floats within the Adreno's 32 KB. staged:
        // WGxLANESxRNxKC, activations (<= 32/RN rows x KC) and partials in
        // local memory, <= 32 accumulators per item. The default (no knobs:
        // stage 128x8x2x128) is measured first as the baseline; staged
        // candidates go before the register ones, which lost on the K1.
        let measure = |kind: &str, cfg: &str, lanes: &str| -> f64 {
            if !kind.is_empty() {
                std::env::set_var("TANG_CL_MM_KIND", kind);
            }
            if kind == "stage" {
                std::env::set_var("TANG_CL_STAGE", cfg);
            } else if kind == "reg" {
                std::env::set_var("TANG_CL_MM", cfg);
                std::env::set_var("TANG_CL_MM_LANES", lanes);
            }
            let s = Step::new(
                &rt,
                &dims,
                seq,
                512,
                &|dg| vec![Vec::new(); dg.inputs.len()],
                Some(&base),
            );
            s.prefill(&rt, &toks, 0);
            let ms = s.prog.time(&rt, 1);
            for k in [
                "TANG_CL_MM_KIND",
                "TANG_CL_STAGE",
                "TANG_CL_MM",
                "TANG_CL_MM_LANES",
            ] {
                std::env::remove_var(k);
            }
            ms
        };
        let mut results: Vec<(f64, usize)> = Vec::new();
        let mut baseline = None;
        // wall time of the slowest candidate so far (build + run): a
        // candidate starts only if one more of those fits the budget
        let mut cost = std::time::Duration::ZERO;
        for (i, &(kind, cfg, lanes)) in PREFILL_CANDIDATES.iter().enumerate() {
            if i > 0 && Instant::now() + cost >= deadline {
                eprintln!("tune: budget spent");
                break;
            }
            if !settle(hot, 75.0, load, deadline) && i > 0 {
                eprintln!("tune: budget spent");
                break;
            }
            #[cfg(feature = "otel")]
            let c0 = std::time::SystemTime::now();
            let w0 = Instant::now();
            let ms = measure(kind, cfg, lanes);
            cost = cost.max(w0.elapsed());
            let name = if i == 0 {
                format!("default ({cfg})")
            } else {
                format!("{kind} {cfg} lanes {lanes}")
            };
            eprintln!(
                "tune: prefill {seq} tokens, {name}: {ms:.1} ms ({:.0} tok/s, gpu {:.1} C)",
                seq as f64 * 1e3 / ms,
                gpu_temp().unwrap_or(0.0)
            );
            #[cfg(feature = "otel")]
            if let Some(t) = &tel {
                candidates += 1;
                t.candidate(c0, "prefill", &name, ms, gpu_temp());
            }
            if i == 0 {
                baseline = Some(ms);
            } else {
                results.push((ms, i));
            }
        }
        let pick = pick_prefill(baseline, &results);
        if let Some(b) = baseline {
            prefill_ms = Some((b, pick.map(|(ms, _)| ms)));
        }
        match pick {
            Some((ms, i)) => {
                let (kind, cfg, lanes) = PREFILL_CANDIDATES[i];
                eprintln!(
                    "tune: prefill {kind} {cfg} {ms:.1} ms beats the default {:.1} ms",
                    baseline.unwrap_or(f64::NAN)
                );
                won.push(("TANG_CL_MM_KIND".to_string(), kind.to_string()));
                if kind == "stage" {
                    won.push(("TANG_CL_STAGE".to_string(), cfg.to_string()));
                } else {
                    won.push(("TANG_CL_MM".to_string(), cfg.to_string()));
                    won.push(("TANG_CL_MM_LANES".to_string(), lanes.to_string()));
                }
            }
            None => eprintln!(
                "tune: prefill keeps the default ({:.1} ms)",
                baseline.unwrap_or(f64::NAN)
            ),
        }
    }
    // merge into the tuning file
    let path = tune_path();
    let mut all: serde_json::Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    let mut entry: serde_json::Map<String, serde_json::Value> = won
        .iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
        .collect();
    // numbers, not knobs: apply_tuning skips non-strings
    if let Some((b, w)) = prefill_ms {
        entry.insert("prefill_baseline_ms".into(), serde_json::json!(b));
        if let Some(w) = w {
            entry.insert("prefill_best_ms".into(), serde_json::json!(w));
        }
    }
    all[key.as_str()] = serde_json::Value::Object(entry);
    if let Some(d) = path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    std::fs::write(&path, serde_json::to_string_pretty(&all).unwrap()).expect("write tuning file");
    eprintln!(
        "tune: wrote {} in {:.0} s: {won:?}",
        path.display(),
        t0.elapsed().as_secs_f64()
    );
    #[cfg(feature = "otel")]
    if let Some(t) = tel {
        t.end_tune(&key, budget, candidates, &won);
        drop(_attached);
        tang_otel::flush();
    }
    won
}

/// Prompt text of [`prefill_bench`]: fixed, so runs on different devices
/// see the same token ids.
const BENCH_TEXT: &str = "The robot rolled into the kitchen, paused by the window, \
and asked whether anyone wanted tea. Nobody answered, so it checked the kettle, \
counted the cups on the shelf, and wrote a short note about the weather: cloudy \
in the morning, clear by noon, a light wind from the west. ";

/// One prefill variant against the f32-map baseline ([`prefill_bench`]).
#[derive(Clone, Debug)]
pub struct PrefillCheck {
    pub variant: &'static str,
    pub seq: usize,
    /// Largest |logit − baseline logit| over the vocabulary (last token).
    pub max_abs: f32,
    pub max_logit: f32,
    pub top1_agree: bool,
}

/// Prefill and decode timings for the model in `dir` on the default
/// runtime ([`runtime`]: `TANG_BACKEND`), tuned knobs applied: for each of
/// `sizes`, one prefill program of exactly that many tokens of a fixed
/// prompt from position 0 (device time, best of 5 runs); then `gen` greedy
/// decode steps after the largest, end to end (run + state readback).
/// With `check` (OpenCL), each size's last-token logits also run through
/// the f32-map prefill (`TANG_CL_PREFILL_Q4=0`, the baseline), the
/// packed-weight one and the packed-weight fp16 one
/// (`TANG_CL_PREFILL_F16=1`), returned as max |Δlogit| and top-1 agreement
/// against the baseline. Prints as it goes.
pub fn prefill_bench(dir: &str, sizes: &[usize], gen: usize, check: bool) -> Vec<PrefillCheck> {
    // QWEN_BENCH_TURN=0: skip the 20-token turn (two more programs)
    let turn_on = std::env::var("QWEN_BENCH_TURN").as_deref() != Ok("0");
    // QWEN_BENCH_PROFILE=1: per-kernel device time of each size's prefill
    let profile = std::env::var("QWEN_BENCH_PROFILE").is_ok_and(|v| v == "1");
    let t_start = Instant::now();
    let phase = |what: &str| {
        println!("phase {:7.2} s: {what}", t_start.elapsed().as_secs_f64());
    };
    let tokenizer =
        tokenizers::Tokenizer::from_file(format!("{dir}/tokenizer.json")).expect("tokenizer.json");
    let max = sizes.iter().copied().max().unwrap_or(1);
    let mut ids: Vec<usize> = Vec::new();
    while ids.len() < max {
        let e = tokenizer
            .encode(BENCH_TEXT.repeat(8), false)
            .expect("encode");
        ids.extend(e.get_ids().iter().map(|&t| t as usize));
    }
    ids.truncate(max);
    phase("tokenizer + prompt");
    let ctx = (max + gen + 1).next_power_of_two().max(1024);
    let mut dims = dims_for(dir, ctx);
    dims.kv_f16 = std::env::var("TANG_KV_F32").is_err();
    let rt = runtime();
    phase("runtime");
    for (k, v) in apply_tuning(&rt, &dims) {
        println!("tang: tuned {k}={v}");
    }
    if rt.kind() == BackendKind::OpenCl {
        dims.vocab_chunks = (dims.vocab * dims.d_model / 2).div_ceil(255 << 20).max(1);
    }
    println!(
        "bench: {} | d{} ff{} l{} | prefill_q4 {} fp16 {}",
        rt.name(),
        dims.d_model,
        dims.ff,
        dims.layers,
        prefill_q4(&rt),
        tang_compile::tensor::opencl_gemv::prefill_f16()
    );
    let st = St::open(&format!("{dir}/model.safetensors"));
    phase("checkpoint mapped");
    let dec = Step::new(
        &rt,
        &dims,
        1,
        ctx,
        &|dg| vec![Vec::new(); dg.inputs.len()],
        None,
    );
    phase("decode program built + buffers allocated");
    for (k, v) in load_inputs(&st, &dims, &dec.dg).enumerate() {
        if !v.is_empty() {
            dec.prog.write(&rt, dec.prog.input(k), &v);
        }
    }
    rt.sync();
    phase("weights converted + uploaded");
    let build = |seq: usize| {
        Step::new(
            &rt,
            &dims,
            seq,
            window(seq, ctx),
            &|dg| vec![Vec::new(); dg.inputs.len()],
            Some(&dec),
        )
    };
    let mut sorted = sizes.to_vec();
    sorted.sort_unstable();
    let mut last = None;
    for &seq in &sorted {
        let t = Instant::now();
        let s = build(seq);
        let built = t.elapsed().as_secs_f64();
        let t = Instant::now();
        s.prefill(&rt, &ids[..seq], 0);
        let _ = s.state(&rt);
        let first = t.elapsed().as_secs_f64();
        let ms = run_ms(&rt, &s.prog, 2);
        println!(
            "prefill {seq:4} tokens: built in {built:.1} s, first run {:.0} ms wall",
            first * 1e3
        );
        println!(
            "prefill {seq:4} tokens: {ms:8.1} ms device ({:6.0} tok/s), {:.0} MB own buffers",
            seq as f64 * 1e3 / ms,
            s.bytes as f64 / 1e6
        );
        phase(&format!("prefill {seq} timed"));
        if profile {
            kernel_table(&rt, &s, 25);
            phase(&format!("prefill {seq} profiled"));
        }
        last = Some(s);
    }
    // a voice turn: 20 new tokens after the largest prompt, split as a
    // Session splits them (QWEN_PREFILL_CHUNKS), device time summed
    if turn_on && max + 20 < ctx {
        let (mut p, mut ms, mut parts) = (max, 0.0, Vec::new());
        let ladder = chunk_sizes(&rt);
        let turn: Vec<usize> = (0..max + 20).map(|i| ids[i % max]).collect();
        while p < max + 20 {
            let left = max + 20 - p;
            match ladder.iter().copied().find(|&c| c <= left) {
                Some(seq) => {
                    let s = Step::new(
                        &rt,
                        &dims,
                        seq,
                        window(p + seq, ctx),
                        &|dg| vec![Vec::new(); dg.inputs.len()],
                        Some(&dec),
                    );
                    s.prefill(&rt, &turn[p..p + seq], p);
                    ms += run_ms(&rt, &s.prog, 2);
                    parts.push(seq.to_string());
                    p += seq;
                }
                None => {
                    dec.set_state(&rt, turn[p], p);
                    ms += run_ms(&rt, &dec.prog, 2);
                    parts.push("1 (decode)".into());
                    p += 1;
                }
            }
        }
        println!(
            "turn: 20 new tokens after {max} as {}: {ms:.1} ms device",
            parts.join(" + ")
        );
    }
    // decode after a prefill of the largest size
    if gen > 0 {
        let s = last.take().unwrap_or_else(|| build(max));
        s.prefill(&rt, &ids[..max], 0);
        let _ = s.state(&rt);
        drop(s);
        let t = Instant::now();
        let mut n = 0;
        for _ in 0..gen {
            dec.prog.run(&rt);
            let (_, p) = dec.state(&rt);
            n += 1;
            if p + 1 >= ctx {
                break;
            }
        }
        let secs = t.elapsed().as_secs_f64();
        println!(
            "decode {n} tokens after {max}: {:.1} ms/token ({:.1} tok/s end to end)",
            secs * 1e3 / n as f64,
            n as f64 / secs
        );
    }
    let mut out = Vec::new();
    if !check || rt.kind() != BackendKind::OpenCl {
        return out;
    }
    let saved: Vec<(&str, Option<String>)> = [
        "TANG_CL_PREFILL_Q4",
        "TANG_CL_PREFILL_F16",
        "TANG_CL_MM_VEC",
    ]
    .into_iter()
    .map(|k| (k, std::env::var(k).ok()))
    .collect();
    let variant = |q4: &str, f16: &str, vec: &str, seq: usize| -> Vec<f32> {
        std::env::set_var("TANG_CL_PREFILL_Q4", q4);
        std::env::set_var("TANG_CL_PREFILL_F16", f16);
        std::env::set_var("TANG_CL_MM_VEC", vec);
        let s = build(seq);
        s.prefill(&rt, &ids[..seq], 0);
        s.logits(&rt)
    };
    for &seq in &sorted {
        let base = variant("0", "0", "1", seq);
        let top = argmax(&base);
        let max_logit = base.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        for (name, q4, f16, vec) in [
            ("q4", "1", "0", "1"),
            ("q4 uint4", "1", "0", "4"),
            ("q4+fp16", "1", "1", "1"),
        ] {
            let l = variant(q4, f16, vec, seq);
            let max_abs = l
                .iter()
                .zip(&base)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let c = PrefillCheck {
                variant: name,
                seq,
                max_abs,
                max_logit,
                top1_agree: argmax(&l) == top,
            };
            println!(
                "check {name:8} {seq:4} tokens vs f32 maps: max |dlogit| {max_abs:.2e} (|logit| <= {max_logit:.1}), top-1 {}",
                if c.top1_agree { "agrees" } else { "DIFFERS" }
            );
            out.push(c);
        }
    }
    for (k, v) in saved {
        match v {
            Some(v) => std::env::set_var(k, v),
            None => std::env::remove_var(k),
        }
    }
    out
}

/// Best of `reps` runs of `prog`, each synced, in ms of wall time (the
/// bench's timing: `Program::time` makes 28 runs, ~22 s for one 0.8 s
/// prefill on the K1).
fn run_ms(rt: &Runtime, prog: &Program, reps: usize) -> f64 {
    (0..reps.max(1))
        .map(|_| {
            let t = Instant::now();
            prog.run(rt);
            rt.sync();
            t.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min)
}

/// Device time per kernel of one run of `step`, summed by kernel kind (its
/// first source line), the `top` largest: the per-kernel table of
/// [`prefill_bench`] (`QWEN_BENCH_PROFILE=1`).
pub fn kernel_table(rt: &Runtime, step: &Step, top: usize) {
    let p = &step.prog;
    let mut groups: Vec<(String, f64, usize)> = Vec::new();
    let mut total = 0.0;
    for (i, k) in p.program.kernels.iter().enumerate() {
        let ms = p.time_kernel(rt, i, 1);
        total += ms;
        let head: String = k
            .source
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(110)
            .collect();
        let name = format!("{head} [{} items]", k.count);
        match groups.iter_mut().find(|g| g.0 == name) {
            Some(g) => {
                g.1 += ms;
                g.2 += 1;
            }
            None => groups.push((name, ms, 1)),
        }
    }
    groups.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!(
        "kernels: {} launches, {total:.1} ms summed (each alone)",
        p.program.kernels.len()
    );
    for (name, ms, n) in groups.iter().take(top) {
        println!("  {ms:8.2} ms {:5.1}%  x{n:<4} {name}", 100.0 * ms / total);
    }
}

/// OpenTelemetry for [`Session`] and [`tune`] (feature `otel`).
#[cfg(feature = "otel")]
mod tel {
    use std::sync::atomic::AtomicU64;
    use std::time::SystemTime;
    use tang_otel::{Context, KeyValue, TraceContextExt as _, Tracer as _};

    /// A span in flight, current in `cx`.
    pub struct Op {
        pub cx: Context,
        start: SystemTime,
    }

    impl Op {
        pub fn start(name: &'static str, parent: &Option<Context>) -> Option<Self> {
            let t = tang_otel::tracer()?;
            let parent = parent.clone().unwrap_or_else(Context::current);
            let span = t.span_builder(name).start_with_context(t, &parent);
            Some(Op {
                cx: parent.with_span(span),
                start: SystemTime::now(),
            })
        }

        /// A decode step's span: every `TANG_OTEL_DECODE_SAMPLE`th (default 1).
        pub fn start_decode(parent: &Option<Context>) -> Option<Self> {
            static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            static N: AtomicU64 = AtomicU64::new(0);
            let every = *EVERY.get_or_init(|| {
                if tang_otel::enabled() {
                    tang_otel::env_u64("TANG_OTEL_DECODE_SAMPLE", 1)
                } else {
                    0
                }
            });
            if !tang_otel::every(&N, every) {
                return None;
            }
            Self::start("qwen.decode.step", parent)
        }

        fn ms(&self) -> f64 {
            SystemTime::now()
                .duration_since(self.start)
                .unwrap_or_default()
                .as_secs_f64()
                * 1e3
        }

        /// One prefill chunk (host time: enqueue, plus whatever the device made it wait).
        pub fn chunk(&self, c0: SystemTime, seq: usize, kctx: usize, pos: usize) {
            let now = SystemTime::now();
            tang_otel::span_at(
                "qwen.prefill.chunk",
                &self.cx,
                c0,
                now,
                vec![
                    KeyValue::new("tang.qwen.chunk_tokens", seq as i64),
                    KeyValue::new("tang.qwen.kctx", kctx as i64),
                    KeyValue::new("tang.qwen.pos", pos as i64),
                    KeyValue::new(
                        "tang.qwen.host_ms",
                        now.duration_since(c0).unwrap_or_default().as_secs_f64() * 1e3,
                    ),
                ],
            );
        }

        pub fn end_prefill(self, prompt: usize, reused: usize, chunks: &[usize]) {
            let ms = self.ms();
            let new = prompt - reused;
            let s = self.cx.span();
            s.set_attribute(KeyValue::new("tang.qwen.prompt_tokens", prompt as i64));
            s.set_attribute(KeyValue::new("tang.qwen.reused_tokens", reused as i64));
            s.set_attribute(KeyValue::new("tang.qwen.new_tokens", new as i64));
            s.set_attribute(KeyValue::new("tang.qwen.chunks", chunks.len() as i64));
            let sizes: Vec<String> = chunks.iter().map(|c| c.to_string()).collect();
            s.set_attribute(KeyValue::new("tang.qwen.chunk_sizes", sizes.join(",")));
            s.set_attribute(KeyValue::new("tang.qwen.ms", ms));
            if ms > 0.0 {
                s.set_attribute(KeyValue::new(
                    "tang.qwen.tokens_per_second",
                    new as f64 * 1e3 / ms,
                ));
            }
            s.end();
        }

        pub fn end_decode(self, pos: usize) {
            let ms = self.ms();
            let s = self.cx.span();
            s.set_attribute(KeyValue::new("tang.qwen.pos", pos as i64));
            s.set_attribute(KeyValue::new("tang.qwen.tokens", 1i64));
            s.set_attribute(KeyValue::new("tang.qwen.ms", ms));
            s.end();
        }

        pub fn candidate(
            &self,
            c0: SystemTime,
            kind: &str,
            config: &str,
            ms: f64,
            temp: Option<f64>,
        ) {
            let mut a = vec![
                KeyValue::new("tang.tune.kind", kind.to_string()),
                KeyValue::new("tang.tune.config", config.to_string()),
                KeyValue::new("tang.tune.ms", ms),
            ];
            if let Some(t) = temp {
                a.push(KeyValue::new("tang.tune.gpu_temp_c", t));
            }
            tang_otel::span_at("qwen.tune.candidate", &self.cx, c0, SystemTime::now(), a);
        }

        pub fn end_tune(self, key: &str, budget: f64, candidates: i64, won: &[(String, String)]) {
            let s = self.cx.span();
            s.set_attribute(KeyValue::new("tang.tune.key", key.to_string()));
            s.set_attribute(KeyValue::new("tang.tune.budget_s", budget));
            s.set_attribute(KeyValue::new("tang.tune.candidates", candidates));
            let chosen: Vec<String> = won.iter().map(|(k, v)| format!("{k}={v}")).collect();
            s.set_attribute(KeyValue::new("tang.tune.chosen", chosen.join(" ")));
            s.end();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::pick_prefill;

    #[test]
    fn slower_than_baseline_never_wins() {
        // the K1 case: reg at 568 ms vs the staged default at 374 ms
        assert_eq!(pick_prefill(Some(374.0), &[(568.0, 9), (400.0, 2)]), None);
        assert_eq!(pick_prefill(Some(374.0), &[(374.0, 3)]), None);
        assert_eq!(
            pick_prefill(Some(374.0), &[(568.0, 9), (350.0, 2), (360.0, 4)]),
            Some((350.0, 2))
        );
    }

    #[test]
    fn empty_or_timed_out_sweep_keeps_default() {
        assert_eq!(pick_prefill(Some(374.0), &[]), None);
        assert_eq!(pick_prefill(None, &[(100.0, 1)]), None);
        assert_eq!(pick_prefill(Some(f64::NAN), &[(100.0, 1)]), None);
    }
}
