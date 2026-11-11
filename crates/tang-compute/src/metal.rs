//! Metal compute backend for Apple Silicon.

use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use metal::objc::rc::autoreleasepool;
use metal::*;

use crate::device::{ComputeBuffer, ComputeDevice};
use crate::kernels::{adamw_msl, attention_msl, backward_msl, llm_msl, matmul_msl, reduce_msl};
use tang_expr::codegen::Dialect;
use tang_expr::node::ExprId;
use tang_expr::trace;

/// Metal buffer wrapping a `metal::Buffer`.
pub struct MetalBuffer {
    buffer: metal::Buffer,
    /// Element count.
    len: usize,
    /// Element format. `Q4`, and `Bf16` from `upload_bf16`, are read-only weights for
    /// `linear`/`embedding`; `Bf16` from `alloc_bf16` is a KV cache.
    kind: Kind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    F32,
    Bf16,
    /// 4-bit affine groups in MLX's layout: `w = scale * q + bias` per `group` consecutive
    /// weights of a row. The buffer holds the packed nibbles (`[n, k/8]` u32, low nibble first),
    /// then bf16 scales, then bf16 biases (each `[n, k/group]`). `len` is `n * k`.
    Q4 {
        group: u32,
    },
}

impl MetalBuffer {
    /// Byte offsets of the scales and biases in a `Q4` buffer.
    fn q4_offsets(&self, group: usize) -> (u64, u64) {
        let packed = self.len / 2;
        let scales = self.len / group * 2;
        (packed as u64, (packed + scales) as u64)
    }
}

impl ComputeBuffer for MetalBuffer {
    fn len(&self) -> usize {
        self.len
    }

    fn to_vec(&self) -> Vec<f32> {
        if let Kind::Q4 { group } = self.kind {
            let group = group as usize;
            let base = self.buffer.contents() as *const u8;
            let (so, bo) = self.q4_offsets(group);
            let words = unsafe { std::slice::from_raw_parts(base as *const u32, self.len / 8) };
            let bf = |off: u64, i: usize| unsafe {
                f32::from_bits((*(base.add(off as usize) as *const u16).add(i) as u32) << 16)
            };
            return (0..self.len)
                .map(|i| {
                    let q = (words[i / 8] >> (4 * (i % 8))) & 0xf;
                    bf(so, i / group) * q as f32 + bf(bo, i / group)
                })
                .collect();
        }
        if self.kind == Kind::Bf16 {
            let ptr = self.buffer.contents() as *const u16;
            let bits = unsafe { std::slice::from_raw_parts(ptr, self.len) };
            return bits
                .iter()
                .map(|&b| f32::from_bits((b as u32) << 16))
                .collect();
        }
        let ptr = self.buffer.contents() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.len) };
        slice.to_vec()
    }
}

/// Metal compute device.
pub struct MetalDevice {
    device: metal::Device,
    queue: CommandQueue,
    pipeline_cache: RefCell<HashMap<u64, ComputePipelineState>>,
    /// Active command buffer — accumulates work, committed on `sync()`.
    active_cb: RefCell<Option<CommandBuffer>>,
    /// Open compute encoder on `active_cb`, reused across dispatches (dispatches in a serial
    /// encoder run in order); closed before blits and on `sync()`.
    active_enc: RefCell<Option<ComputeCommandEncoder>>,
    /// Command buffers committed by `flush()` and not yet waited on.
    in_flight: RefCell<Vec<CommandBuffer>>,
}

impl MetalDevice {
    /// Create a new Metal device using the system default GPU.
    pub fn new() -> Option<Self> {
        let device = metal::Device::system_default()?;
        let queue = device.new_command_queue();
        Some(MetalDevice {
            device,
            queue,
            pipeline_cache: RefCell::new(HashMap::new()),
            active_cb: RefCell::new(None),
            active_enc: RefCell::new(None),
            in_flight: RefCell::new(Vec::new()),
        })
    }

    /// Get or compile a pipeline from MSL source.
    fn get_pipeline(&self, source: &str, fn_name: &str) -> ComputePipelineState {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        fn_name.hash(&mut hasher);
        let hash = hasher.finish();

        if let Some(pipeline) = self.pipeline_cache.borrow().get(&hash) {
            return pipeline.clone();
        }

        let options = CompileOptions::new();
        if matches!(
            fn_name,
            "rms_norm_modulation" | "group_norm_affine" | "channel_modulation" | "tkv_decode"
        ) {
            options.set_fast_math_enabled(false);
        }
        let library = self
            .device
            .new_library_with_source(source, &options)
            .expect("Failed to compile MSL");
        let func = library
            .get_function(fn_name, None)
            .expect("Failed to get function");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .expect("Failed to create pipeline");

        self.pipeline_cache
            .borrow_mut()
            .insert(hash, pipeline.clone());
        pipeline
    }

    /// Create a Metal buffer from f32 data.
    fn make_buffer(&self, data: &[f32]) -> metal::Buffer {
        let len = std::mem::size_of_val(data);
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            len as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create a Metal buffer from u32 data.
    fn make_buffer_u32(&self, data: &[u32]) -> metal::Buffer {
        let len = std::mem::size_of_val(data);
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            len as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Allocate an empty Metal buffer.
    fn make_buffer_empty(&self, byte_len: usize) -> metal::Buffer {
        self.device
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared)
    }

    /// Get or create the active command buffer for batching dispatches.
    fn ensure_cb(&self) {
        let mut cb_ref = self.active_cb.borrow_mut();
        if cb_ref.is_none() {
            let cb = autoreleasepool(|| self.queue.new_command_buffer().to_owned());
            *cb_ref = Some(cb);
        }
    }

    /// Encode into the open compute encoder, opening one if needed.
    fn with_encoder(&self, f: impl FnOnce(&ComputeCommandEncoderRef)) {
        self.ensure_cb();
        autoreleasepool(|| {
            let mut enc_ref = self.active_enc.borrow_mut();
            if enc_ref.is_none() {
                let cb_ref = self.active_cb.borrow();
                let cmd = cb_ref.as_ref().unwrap();
                *enc_ref = Some(cmd.new_compute_command_encoder().to_owned());
            }
            f(enc_ref.as_ref().unwrap());
        });
    }

    fn modulated_rms(
        &self,
        data: &MetalBuffer,
        weight: &MetalBuffer,
        modulation: &MetalBuffer,
        residual: Option<&MetalBuffer>,
        offset: usize,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> MetalBuffer {
        assert!(dim > 0 && data.len() == rows * dim && weight.len() == dim);
        assert!(modulation.len() >= offset + dim);
        assert!(residual.is_none_or(|r| r.len() == data.len()));
        for buffer in [data, weight, modulation, residual.unwrap_or(data)] {
            assert_eq!(buffer.kind, Kind::F32);
        }
        let pipeline = self.get_pipeline(reduce_msl::MODULATED_RMS_MSL, "rms_norm_modulation");
        let output = self.make_buffer_empty(rows * dim * 4);
        let params = self.make_buffer_u32(&[
            rows as u32,
            dim as u32,
            offset as u32,
            residual.is_some() as u32,
            eps.to_bits(),
        ]);
        self.dispatch_groups(
            &pipeline,
            &[
                &data.buffer,
                &weight.buffer,
                &modulation.buffer,
                &residual.unwrap_or(data).buffer,
                &output,
                &params,
            ],
            (rows, 1),
            dim.min(256).next_power_of_two(),
        );
        MetalBuffer {
            buffer: output,
            len: rows * dim,
            kind: Kind::F32,
        }
    }

    /// Close the open compute encoder (before a blit, or before committing).
    fn end_compute(&self) {
        if let Some(enc) = self.active_enc.borrow_mut().take() {
            enc.end_encoding();
        }
    }

    /// Dispatch a compute pipeline with given buffers.
    fn dispatch(&self, pipeline: &ComputePipelineState, buffers: &[&metal::Buffer], threads: u64) {
        let tg_size = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), 256);
        let tg_count = threads.div_ceil(tg_size);

        self.with_encoder(|enc| {
            enc.set_compute_pipeline_state(pipeline);
            for (i, buf) in buffers.iter().enumerate() {
                enc.set_buffer(i as u64, Some(buf), 0);
            }
            enc.dispatch_thread_groups(MTLSize::new(tg_count, 1, 1), MTLSize::new(tg_size, 1, 1));
        });
    }

    /// Dispatch with 2D grid.
    fn dispatch_2d(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[&metal::Buffer],
        grid: (u64, u64),
        tg_size: (u64, u64),
    ) {
        let tg_count_x = grid.0.div_ceil(tg_size.0);
        let tg_count_y = grid.1.div_ceil(tg_size.1);

        self.with_encoder(|enc| {
            enc.set_compute_pipeline_state(pipeline);
            for (i, buf) in buffers.iter().enumerate() {
                enc.set_buffer(i as u64, Some(buf), 0);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(tg_count_x, tg_count_y, 1),
                MTLSize::new(tg_size.0, tg_size.1, 1),
            );
        });
    }
}

impl MetalDevice {
    /// Dispatch `groups` threadgroups of `threads` threads each.
    fn dispatch_groups(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[&metal::Buffer],
        groups: (usize, usize),
        threads: usize,
    ) {
        self.with_encoder(|enc| {
            enc.set_compute_pipeline_state(pipeline);
            for (i, buf) in buffers.iter().enumerate() {
                enc.set_buffer(i as u64, Some(buf), 0);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(groups.0 as u64, groups.1 as u64, 1),
                MTLSize::new(threads as u64, 1, 1),
            );
        });
    }

    fn copy_f32(
        &self,
        src: &metal::Buffer,
        src_off: usize,
        dst: &metal::Buffer,
        dst_off: usize,
        n: usize,
    ) {
        self.copy_with("copy_f32", src, src_off, dst, dst_off, n);
    }

    /// `n` elements with one of `COPY_MSL`'s kernels (same format, or converting).
    fn copy_with(
        &self,
        kernel: &str,
        src: &metal::Buffer,
        src_off: usize,
        dst: &metal::Buffer,
        dst_off: usize,
        n: usize,
    ) {
        if n == 0 {
            return;
        }
        let pipeline = self.get_pipeline(llm_msl::COPY_MSL, kernel);
        let params = self.make_buffer_u32(&[src_off as u32, dst_off as u32, n as u32]);
        self.dispatch(&pipeline, &[src, dst, &params], n as u64);
    }

    /// A bf16 KV cache widened to f32 (for the fallback kernels that only read f32).
    fn widen(&self, buf: &MetalBuffer) -> MetalBuffer {
        let mut out = self.alloc(buf.len);
        self.write_into(&mut out, 0, buf);
        out
    }

    /// The KV-reading kernel source `src` for caches `k` and `v` (see `llm_msl::kv_source`).
    fn kv_src(src: &'static str, k: &MetalBuffer, v: &MetalBuffer) -> &'static str {
        assert_eq!(k.kind, v.kind, "K and V caches differ in format");
        assert!(
            matches!(k.kind, Kind::F32 | Kind::Bf16),
            "a KV cache is f32 or bf16"
        );
        llm_msl::kv_source(src, k.kind == Kind::Bf16)
    }

    /// Tiled causal attention for prefill (`llm_msl::FLASH_PREFILL_MSL`). Query rows are
    /// padded to a multiple of 32 and the padding is dropped from the output.
    #[allow(clippy::too_many_arguments)]
    fn flash_prefill(
        &self,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        cache_start: usize,
        q_len: usize,
        n_heads: usize,
        n_kv: usize,
        d: usize,
        bidir: bool,
        pages: Option<Paging>,
    ) -> MetalBuffer {
        let q_pad = q_len.next_multiple_of(32);
        let width = n_heads * d;
        let padded;
        let qb = if q_pad == q_len {
            q
        } else {
            let mut p = self.alloc(q_pad * width);
            self.write_into(&mut p, 0, q);
            padded = p;
            &padded
        };
        let out = self.make_buffer_empty(q_pad * width * 4);
        let params = self.make_buffer_u32(&[
            cache_start as u32,
            q_len as u32,
            n_heads as u32,
            n_kv as u32,
            d as u32,
            bidir as u32,
            pages.map_or(0, |p| p.1),
        ]);
        let pipeline = self.get_pipeline(
            Self::kv_src(llm_msl::FLASH_PREFILL_MSL, k, v),
            "attn_prefill",
        );
        let table = pages.map_or(&params, |p| p.0);
        self.with_encoder(|enc| {
            enc.set_compute_pipeline_state(&pipeline);
            for (i, b) in [&qb.buffer, &k.buffer, &v.buffer, &out, &params, table]
                .iter()
                .enumerate()
            {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            enc.dispatch_thread_groups(
                MTLSize::new((q_pad / 32) as u64, n_heads as u64, 1),
                MTLSize::new(128, 1, 1),
            );
        });
        let full = MetalBuffer {
            buffer: out,
            len: q_pad * width,
            kind: Kind::F32,
        };
        if q_pad == q_len {
            full
        } else {
            self.slice_buffer(&full, 0, q_len * width)
        }
    }

    /// `x · wᵀ` for 2..=32 rows over 4-bit or bf16 weights with `qmm_small_*` /
    /// `gemm_small_bf16_*`: the weights are read once for all the rows, and K is split when the
    /// columns alone give too few threadgroups.
    fn gemm_small(
        &self,
        x: &MetalBuffer,
        w: &MetalBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> MetalBuffer {
        let out = self.make_buffer_empty(m * n * 4);
        let (group, (so, bo)) = match w.kind {
            Kind::Q4 { group } => (group, w.q4_offsets(group as usize)),
            _ => (32, (0, 0)),
        };
        let name = match (w.kind == Kind::Bf16, m) {
            (false, ..=8) => "qmm_small_8",
            (false, ..=16) => "qmm_small_16",
            (false, _) => "qmm_small_32",
            (true, ..=8) => "gemm_small_bf16_8",
            (true, ..=16) => "gemm_small_bf16_16",
            (true, _) => "gemm_small_bf16_32",
        };
        let pipeline = self.get_pipeline(llm_msl::Q4_MSL, name);
        // Each split keeps at least 8 steps of 32.
        let (col_groups, steps) = (n.div_ceil(128), k / 32);
        let want = SMALL_GEMM_GROUPS.div_ceil(col_groups);
        let per = steps.div_ceil(want.clamp(1, (steps / 8).max(1)));
        let splits = steps.div_ceil(per);
        let params = self.make_buffer_u32(&[m as u32, k as u32, n as u32, group, per as u32]);
        let partial = (splits > 1).then(|| self.make_buffer_empty(splits * m * n * 4));
        self.with_encoder(|enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&x.buffer), 0);
            enc.set_buffer(1, Some(&w.buffer), 0);
            enc.set_buffer(2, Some(&w.buffer), so);
            enc.set_buffer(3, Some(&w.buffer), bo);
            enc.set_buffer(4, Some(partial.as_ref().unwrap_or(&out)), 0);
            enc.set_buffer(5, Some(&params), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(col_groups as u64, splits as u64, 1),
                MTLSize::new(256, 1, 1),
            );
        });
        if let Some(p) = partial {
            let sum = self.get_pipeline(llm_msl::Q4_MSL, "sum_splits");
            let sp = self.make_buffer_u32(&[(m * n) as u32, splits as u32]);
            self.dispatch(&sum, &[&p, &out, &sp], (m * n) as u64);
        }
        MetalBuffer {
            buffer: out,
            len: m * n,
            kind: Kind::F32,
        }
    }

    /// Split-KV attention for a few queries (`llm_msl::FLASH_MULTI_MSL`, then
    /// `attn_combine`): threadgroups over (KV head, key split, group of (query, head) rows),
    /// each reading its split's keys once for all the rows.
    #[allow(clippy::too_many_arguments)]
    fn flash_multi(
        &self,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        cache_start: usize,
        q_len: usize,
        n_heads: usize,
        n_kv: usize,
        d: usize,
        window: usize,
        bidir: bool,
        pages: Option<Paging>,
    ) -> MetalBuffer {
        let gqa_rows = q_len * (n_heads / n_kv);
        let (name, rows, tile, threads) = match (d <= 128, gqa_rows) {
            (false, _) => ("attn_multi_d256", 8, 16, 128),
            (true, ..=4) => ("attn_multi_r4", 4, 32, 128),
            (true, ..=8) => ("attn_multi_r8", 8, 32, 256),
            (true, ..=16) => ("attn_multi_r16", 16, 32, 256),
            _ => ("attn_multi_r32", 32, 16, 256),
        };
        let longest = cache_start + q_len;
        let base = if window > 0 && !bidir {
            (cache_start + 1).saturating_sub(window)
        } else {
            0
        };
        let span = longest - base;
        let groups = gqa_rows.div_ceil(rows);
        let want = MULTI_ATTN_GROUPS.div_ceil(n_kv * groups).max(1);
        let split_len = span.div_ceil(want).max(2 * tile).next_multiple_of(tile);
        let n_splits = span.div_ceil(split_len);
        let params = self.make_buffer_u32(&[
            cache_start as u32,
            q_len as u32,
            n_heads as u32,
            n_kv as u32,
            d as u32,
            n_splits as u32,
            split_len as u32,
            window as u32,
            base as u32,
            bidir as u32,
            pages.map_or(0, |p| p.1),
        ]);
        let table = pages.map_or(&params, |p| p.0);
        let partial = self.make_buffer_empty(q_len * n_heads * n_splits * (d + 2) * 4);
        let out = self.make_buffer_empty(q_len * n_heads * d * 4);
        let p1 = self.get_pipeline(Self::kv_src(llm_msl::FLASH_MULTI_MSL, k, v), name);
        let p2 = self.get_pipeline(
            llm_msl::kv_source(llm_msl::FLASH_DECODE_MSL, false),
            "attn_combine",
        );
        self.with_encoder(|enc| {
            enc.set_compute_pipeline_state(&p1);
            for (i, b) in [&q.buffer, &k.buffer, &v.buffer, &partial, &params, table]
                .iter()
                .enumerate()
            {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(n_kv as u64, n_splits as u64, groups as u64),
                MTLSize::new(threads, 1, 1),
            );
            enc.set_compute_pipeline_state(&p2);
            for (i, b) in [&partial, &out, &params].iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(n_heads as u64, q_len as u64, 1),
                MTLSize::new(32, 1, 1),
            );
        });
        MetalBuffer {
            buffer: out,
            len: q_len * n_heads * d,
            kind: Kind::F32,
        }
    }

    /// Split-KV attention (`llm_msl::FLASH_DECODE_MSL`). Decode splits the keys across up to 32
    /// threadgroups per head; prefill already has a threadgroup per (query, head).
    #[allow(clippy::too_many_arguments)]
    fn flash_attention(
        &self,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        cache_start: usize,
        q_len: usize,
        n_heads: usize,
        n_kv: usize,
        d: usize,
        window: usize,
        bidir: bool,
        pages: Option<Paging>,
    ) -> MetalBuffer {
        let longest = cache_start + q_len;
        // Decode with a sliding window only needs the last `window` keys.
        let base = if q_len == 1 && window > 0 {
            longest.saturating_sub(window)
        } else {
            0
        };
        let span = longest - base;
        // Short contexts: per-key simdgroup loop (lower fixed cost). Longer: lane-per-key.
        let lane_keys = q_len == 1 && span >= 512;
        let n_splits = match q_len {
            1 if lane_keys => span.div_ceil(256).clamp(1, 64),
            1 => span.div_ceil(256).clamp(1, 32),
            _ => 1,
        };
        let split_len = if q_len == 1 {
            span.div_ceil(n_splits)
        } else {
            longest
        };
        let params = self.make_buffer_u32(&[
            cache_start as u32,
            q_len as u32,
            n_heads as u32,
            n_kv as u32,
            d as u32,
            n_splits as u32,
            split_len as u32,
            window as u32,
            base as u32,
            bidir as u32,
            pages.map_or(0, |p| p.1),
        ]);
        let table = pages.map_or(&params, |p| p.0);
        let partial = self.make_buffer_empty(q_len * n_heads * n_splits * (d + 2) * 4);
        let out = self.make_buffer_empty(q_len * n_heads * d * 4);
        let p1 = self.get_pipeline(
            Self::kv_src(llm_msl::FLASH_DECODE_MSL, k, v),
            if lane_keys {
                "attn_decode"
            } else {
                "attn_partial"
            },
        );
        let grid1 = MTLSize::new(n_heads as u64, n_splits as u64, q_len as u64);
        let p2 = self.get_pipeline(
            llm_msl::kv_source(llm_msl::FLASH_DECODE_MSL, false),
            "attn_combine",
        );
        self.with_encoder(|enc| {
            enc.set_compute_pipeline_state(&p1);
            for (i, b) in [&q.buffer, &k.buffer, &v.buffer, &partial, &params, table]
                .iter()
                .enumerate()
            {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            enc.dispatch_thread_groups(grid1, MTLSize::new(256, 1, 1));
            enc.set_compute_pipeline_state(&p2);
            for (i, b) in [&partial, &out, &params].iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(n_heads as u64, q_len as u64, 1),
                MTLSize::new(32, 1, 1),
            );
        });
        MetalBuffer {
            buffer: out,
            len: q_len * n_heads * d,
            kind: Kind::F32,
        }
    }

    /// The fused attention prologue (`ATTN_PREP_MSL`), into contiguous or paged caches.
    #[allow(clippy::too_many_arguments)]
    fn prep(
        &self,
        qkv: &MetalBuffer,
        q_norm: Option<&MetalBuffer>,
        k_norm: Option<&MetalBuffer>,
        cos: &MetalBuffer,
        sin: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        seq: usize,
        (nh, nkv, hd): (usize, usize, usize),
        pos: usize,
        eps: f32,
        pages: Option<Paging>,
    ) -> MetalBuffer {
        let pipeline = self.get_pipeline(
            Self::kv_src(llm_msl::ATTN_PREP_MSL, k_cache, v_cache),
            "attention_prep",
        );
        let q = self.make_buffer_empty(seq * nh * hd * 4);
        let params = self.make_buffer_u32(&[
            seq as u32,
            nh as u32,
            nkv as u32,
            hd as u32,
            pos as u32,
            eps.to_bits(),
            q_norm.is_some() as u32,
            k_norm.is_some() as u32,
            pages.map_or(0, |p| p.1),
        ]);
        let table = pages.map_or(&params, |p| p.0);
        let qn = q_norm.unwrap_or(qkv);
        let kn = k_norm.unwrap_or(qkv);
        self.with_encoder(|enc| {
            enc.set_compute_pipeline_state(&pipeline);
            let bufs = [
                &qkv.buffer,
                &qn.buffer,
                &kn.buffer,
                &cos.buffer,
                &sin.buffer,
                &q,
                &k_cache.buffer,
                &v_cache.buffer,
                &params,
                table,
            ];
            for (i, b) in bufs.iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            enc.dispatch_thread_groups(
                MTLSize::new((nh + 2 * nkv) as u64, seq as u64, 1),
                MTLSize::new(32, 1, 1),
            );
        });
        MetalBuffer {
            buffer: q,
            len: seq * nh * hd,
            kind: Kind::F32,
        }
    }

    /// Elementwise `f(a[i], b[i])` with a native kernel from `llm_msl::ELEMENTWISE_MSL`.
    fn binary(&self, a: &MetalBuffer, b: &MetalBuffer, numel: usize, kernel: &str) -> MetalBuffer {
        let pipeline = self.get_pipeline(llm_msl::ELEMENTWISE_MSL, kernel);
        let out = self.make_buffer_empty(numel * 4);
        let params = self.make_buffer_u32(&[numel as u32]);
        self.dispatch(
            &pipeline,
            &[&a.buffer, &b.buffer, &out, &params],
            numel as u64,
        );
        MetalBuffer {
            buffer: out,
            len: numel,
            kind: Kind::F32,
        }
    }
}

/// A paged cache's block table (u32 on device) and `log2` of its block size, for the attention
/// kernels (`None`: contiguous).
type Paging<'a> = (&'a metal::Buffer, u32);

/// Forwards of up to this many queries use split-KV attention (`flash_multi`) instead of the
/// tiled prefill kernel, which has too few threadgroups to stream a long cache quickly.
const MULTI_ATTN_ROWS: usize = 32;

/// Batches of 2 up to this many rows run `qmm_small_*` (weights read once for all the rows)
/// instead of the tiled matmul.
const SMALL_GEMM_ROWS: usize = 32;

/// Threadgroups `qmm_small_*` aims for (K is split until there are about this many).
const SMALL_GEMM_GROUPS: usize = 160;

/// Threadgroups `flash_multi` aims for (keys are split until there are about this many).
const MULTI_ATTN_GROUPS: usize = 256;

fn multi_attn(k: &MetalBuffer, q_len: usize, n_heads: usize, n_kv: usize, d: usize) -> bool {
    (1..=MULTI_ATTN_ROWS).contains(&q_len)
        && d.is_multiple_of(4)
        && d <= 256
        && n_heads.is_multiple_of(n_kv)
        && (k.kind != Kind::Bf16 || d.is_multiple_of(8))
}

impl ComputeDevice for MetalDevice {
    fn bf16_storage_bytes(&self) -> usize {
        2
    }
    type Buffer = MetalBuffer;

    fn dialect(&self) -> Dialect {
        Dialect::Msl
    }

    /// The most memory the GPU should use (Metal's recommended working set: unified memory,
    /// so a share of RAM).
    fn total_memory_bytes(&self) -> usize {
        self.device.recommended_max_working_set_size() as usize
    }

    /// The working set left: [`total_memory_bytes`](Self::total_memory_bytes) less what this
    /// process has allocated on the device. Other processes' use of RAM isn't counted.
    fn free_memory_bytes(&self) -> usize {
        self.total_memory_bytes()
            .saturating_sub(self.device.current_allocated_size() as usize)
    }

    fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    fn peak_flops_f32(&self) -> Option<f64> {
        let name = self.device.name().to_lowercase();
        // Apple Silicon GPU peak FP32 TFLOPS (from Apple specs)
        let tflops = if name.contains("m4 max") {
            // M4 Max: 40-core ~18.4 TFLOPS, 36-core ~14.7 TFLOPS
            if name.contains("40") {
                18.4
            } else {
                14.7
            }
        } else if name.contains("m4 pro") {
            8.7
        } else if name.contains("m4") {
            4.6
        } else if name.contains("m3 max") {
            14.2
        } else if name.contains("m3 pro") {
            7.0
        } else if name.contains("m3") {
            3.6
        } else if name.contains("m2 max") {
            13.6
        } else if name.contains("m2 ultra") {
            27.2
        } else if name.contains("m2 pro") {
            6.8
        } else if name.contains("m2") {
            3.6
        } else if name.contains("m1 max") {
            10.4
        } else if name.contains("m1 ultra") {
            21.0
        } else if name.contains("m1 pro") {
            5.2
        } else if name.contains("m1") {
            2.6
        } else {
            return None;
        };
        Some(tflops * 1e12)
    }

    fn upload(&self, data: &[f32]) -> MetalBuffer {
        MetalBuffer {
            buffer: self.make_buffer(data),
            len: data.len(),
            kind: Kind::F32,
        }
    }

    fn upload_bf16(&self, bits: &[u16]) -> MetalBuffer {
        let buffer = self.device.new_buffer_with_data(
            bits.as_ptr() as *const _,
            (bits.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        MetalBuffer {
            buffer,
            len: bits.len(),
            kind: Kind::Bf16,
        }
    }

    fn upload_q4(
        &self,
        packed: &[u32],
        scales: &[u16],
        biases: &[u16],
        group: usize,
    ) -> MetalBuffer {
        let mut bytes: Vec<u8> = Vec::with_capacity(packed.len() * 4 + scales.len() * 4);
        bytes.extend(packed.iter().flat_map(|w| w.to_le_bytes()));
        bytes.extend(scales.iter().flat_map(|w| w.to_le_bytes()));
        bytes.extend(biases.iter().flat_map(|w| w.to_le_bytes()));
        let buffer = self.device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        MetalBuffer {
            buffer,
            len: packed.len() * 8,
            kind: Kind::Q4 {
                group: group as u32,
            },
        }
    }

    fn tkv_decode(
        &self,
        stream: &MetalBuffer,
        lanes: &MetalBuffer,
        chans: &MetalBuffer,
        prior: &MetalBuffer,
        scratch: &MetalBuffer,
        out: &MetalBuffer,
        rope: Option<(&MetalBuffer, &MetalBuffer)>,
        n_lanes: usize,
    ) {
        assert!(matches!(out.kind, Kind::F32 | Kind::Bf16));
        let pipe = self.get_pipeline(
            llm_msl::kv_source(crate::kernels::kvdec_msl::KVDEC_MSL, out.kind == Kind::Bf16),
            "tkv_decode",
        );
        let nl = self.make_buffer_u32(&[n_lanes as u32]);
        let (cs, sn) = rope.map_or((&prior.buffer, &prior.buffer), |(c, s)| {
            (&c.buffer, &s.buffer)
        });
        self.with_encoder(|enc| {
            enc.set_compute_pipeline_state(&pipe);
            for (i, b) in [
                &stream.buffer,
                &lanes.buffer,
                &chans.buffer,
                &prior.buffer,
                &scratch.buffer,
                &out.buffer,
                cs,
                sn,
                &nl,
            ]
            .iter()
            .enumerate()
            {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(n_lanes.div_ceil(32) as u64, 1, 1),
                MTLSize::new(32, 1, 1),
            );
        });
    }

    fn kivi_dequant(
        &self,
        codes: &MetalBuffer,
        sz: &MetalBuffer,
        out: &MetalBuffer,
        pages: Option<(&MetalBuffer, u32)>,
        p: [u32; 6],
    ) {
        let pipe = self.get_pipeline(
            llm_msl::kv_source(crate::kernels::kvdec_msl::KVDEC_MSL, out.kind == Kind::Bf16),
            "kivi_dequant",
        );
        let prm =
            self.make_buffer_u32(&[p[0], p[1], p[2], p[3], p[4], p[5], pages.map_or(0, |x| x.1)]);
        let t = pages.map_or(&prm, |x| &x.0.buffer);
        let n = (p[2] * p[3] * p[4]) as u64;
        self.with_encoder(|enc| {
            enc.set_compute_pipeline_state(&pipe);
            for (i, b) in [&codes.buffer, &sz.buffer, &out.buffer, &prm, t]
                .iter()
                .enumerate()
            {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(256), 1, 1),
                MTLSize::new(256, 1, 1),
            );
        });
    }

    fn upload_u32(&self, data: &[u32]) -> MetalBuffer {
        MetalBuffer {
            buffer: self.make_buffer_u32(data),
            len: data.len(),
            kind: Kind::F32,
        }
    }

    fn alloc(&self, len: usize) -> MetalBuffer {
        MetalBuffer {
            buffer: self.make_buffer_empty(len * 4),
            len,
            kind: Kind::F32,
        }
    }

    fn bf16_storage(&self) -> bool {
        true
    }

    fn alloc_bf16(&self, len: usize) -> MetalBuffer {
        MetalBuffer {
            buffer: self.make_buffer_empty(len * 2),
            len,
            kind: Kind::Bf16,
        }
    }

    fn download_bf16(&self, buf: &MetalBuffer) -> Vec<u16> {
        self.sync();
        if buf.kind != Kind::Bf16 {
            return buf.to_vec().into_iter().map(crate::f32_to_bf16).collect();
        }
        let ptr = buf.buffer.contents() as *const u16;
        unsafe { std::slice::from_raw_parts(ptr, buf.len) }.to_vec()
    }

    fn download(&self, buf: &MetalBuffer) -> Vec<f32> {
        self.sync();
        buf.to_vec()
    }

    fn elementwise(
        &self,
        inputs: &[&MetalBuffer],
        numel: usize,
        f: &dyn Fn(&[ExprId]) -> ExprId,
    ) -> MetalBuffer {
        let n_inputs = inputs.len();

        // Trace closure → MSL kernel
        let (graph, output) = trace(|| {
            let vars: Vec<ExprId> = (0..n_inputs as u16).map(ExprId::var).collect();
            f(&vars)
        });
        let kernel = graph.to_kernel(&[output], n_inputs, Dialect::Msl);
        let pipeline = self.get_pipeline(&kernel.source, kernel.entry_point);

        // Sync before reading input buffers that may have pending GPU writes
        self.sync();

        // Pre-download all inputs once, then interleave
        let input_vecs: Vec<Vec<f32>> = inputs.iter().map(|inp| inp.to_vec()).collect();
        let mut interleaved = vec![0.0f32; numel * n_inputs];
        for i in 0..numel {
            for (j, vec) in input_vecs.iter().enumerate() {
                interleaved[i * n_inputs + j] = vec[i];
            }
        }
        let input_buf = self.make_buffer(&interleaved);
        let output_buf = self.make_buffer_empty(numel * 4);
        let count_buf = self.make_buffer_u32(&[numel as u32]);

        self.dispatch(
            &pipeline,
            &[&input_buf, &output_buf, &count_buf],
            numel as u64,
        );

        MetalBuffer {
            buffer: output_buf,
            len: numel,
            kind: Kind::F32,
        }
    }

    fn matmul(
        &self,
        a: &MetalBuffer,
        b: &MetalBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> MetalBuffer {
        let output_buf = self.make_buffer_empty(m * n * 4);
        let params = self.make_buffer_u32(&[m as u32, k as u32, n as u32]);

        if m.is_multiple_of(32) && n.is_multiple_of(32) && k.is_multiple_of(8) && m >= 32 && n >= 32
        {
            // Simdgroup matmul for aligned dimensions (8x8 tile loads require alignment)
            let pipeline = self.get_pipeline(matmul_msl::MATMUL_MSL, "matmul");
            let tg_x = (n / 32) as u64;
            let tg_y = (m / 32) as u64;
            self.with_encoder(|enc| {
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(&a.buffer), 0);
                enc.set_buffer(1, Some(&b.buffer), 0);
                enc.set_buffer(2, Some(&output_buf), 0);
                enc.set_buffer(3, Some(&params), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(tg_x, tg_y, 1),
                    MTLSize::new(128, 1, 1), // 4 simdgroups × 32 threads
                );
            });
        } else {
            // Naive fallback for unaligned dimensions
            let pipeline = self.get_pipeline(matmul_msl::MATMUL_NAIVE_MSL, "matmul_naive");
            self.dispatch_2d(
                &pipeline,
                &[&a.buffer, &b.buffer, &output_buf, &params],
                (n as u64, m as u64),
                (16, 16),
            );
        }

        MetalBuffer {
            buffer: output_buf,
            len: m * n,
            kind: Kind::F32,
        }
    }

    fn matmul_b_transposed(
        &self,
        a: &MetalBuffer, // [m, k]
        b: &MetalBuffer, // [n, k] row-major (logically transposed)
        m: usize,
        k: usize,
        n: usize,
    ) -> MetalBuffer {
        if m.is_multiple_of(32) && n.is_multiple_of(32) && k.is_multiple_of(8) && m >= 32 && n >= 32
        {
            let pipeline = self.get_pipeline(matmul_msl::MATMUL_BT_MSL, "matmul_bt");
            let output_buf = self.make_buffer_empty(m * n * 4);
            let params = self.make_buffer_u32(&[m as u32, k as u32, n as u32]);
            let tg_x = (n / 32) as u64;
            let tg_y = (m / 32) as u64;
            self.with_encoder(|enc| {
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(&a.buffer), 0);
                enc.set_buffer(1, Some(&b.buffer), 0);
                enc.set_buffer(2, Some(&output_buf), 0);
                enc.set_buffer(3, Some(&params), 0);
                enc.dispatch_thread_groups(MTLSize::new(tg_x, tg_y, 1), MTLSize::new(128, 1, 1));
            });
            MetalBuffer {
                buffer: output_buf,
                len: m * n,
                kind: Kind::F32,
            }
        } else {
            let b_t = self.transpose_2d(b, n, k);
            self.matmul(a, &b_t, m, k, n)
        }
    }

    fn matmul_a_transposed(
        &self,
        a: &MetalBuffer, // [k, m] row-major
        b: &MetalBuffer, // [k, n] row-major
        m: usize,
        k: usize,
        n: usize,
    ) -> MetalBuffer {
        if m.is_multiple_of(32) && n.is_multiple_of(32) && k.is_multiple_of(8) && m >= 32 && n >= 32
        {
            let pipeline = self.get_pipeline(matmul_msl::MATMUL_AT_MSL, "matmul_at");
            let output_buf = self.make_buffer_empty(m * n * 4);
            let params = self.make_buffer_u32(&[m as u32, k as u32, n as u32]);
            let tg_x = (n / 32) as u64;
            let tg_y = (m / 32) as u64;
            self.with_encoder(|enc| {
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(&a.buffer), 0);
                enc.set_buffer(1, Some(&b.buffer), 0);
                enc.set_buffer(2, Some(&output_buf), 0);
                enc.set_buffer(3, Some(&params), 0);
                enc.dispatch_thread_groups(MTLSize::new(tg_x, tg_y, 1), MTLSize::new(128, 1, 1));
            });
            MetalBuffer {
                buffer: output_buf,
                len: m * n,
                kind: Kind::F32,
            }
        } else {
            let a_t = self.transpose_2d(a, k, m);
            self.matmul(&a_t, b, m, k, n)
        }
    }

    fn matmul_accumulate(
        &self,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &mut MetalBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        if m.is_multiple_of(32) && n.is_multiple_of(32) && k.is_multiple_of(8) && m >= 32 && n >= 32
        {
            let pipeline = self.get_pipeline(matmul_msl::MATMUL_ACC_MSL, "matmul_acc");
            let params = self.make_buffer_u32(&[m as u32, k as u32, n as u32]);
            let tg_x = (n / 32) as u64;
            let tg_y = (m / 32) as u64;
            self.with_encoder(|enc| {
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(&a.buffer), 0);
                enc.set_buffer(1, Some(&b.buffer), 0);
                enc.set_buffer(2, Some(&c.buffer), 0);
                enc.set_buffer(3, Some(&params), 0);
                enc.dispatch_thread_groups(MTLSize::new(tg_x, tg_y, 1), MTLSize::new(128, 1, 1));
            });
        } else {
            let tmp = self.matmul(a, b, m, k, n);
            self.add_assign(c, &tmp);
        }
    }

    fn matmul_accumulate_a_transposed(
        &self,
        a: &MetalBuffer, // [k, m] row-major
        b: &MetalBuffer, // [k, n] row-major
        c: &mut MetalBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        if m.is_multiple_of(32) && n.is_multiple_of(32) && k.is_multiple_of(8) && m >= 32 && n >= 32
        {
            let pipeline = self.get_pipeline(matmul_msl::MATMUL_ACC_AT_MSL, "matmul_acc_at");
            let params = self.make_buffer_u32(&[m as u32, k as u32, n as u32]);
            let tg_x = (n / 32) as u64;
            let tg_y = (m / 32) as u64;
            self.with_encoder(|enc| {
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(&a.buffer), 0);
                enc.set_buffer(1, Some(&b.buffer), 0);
                enc.set_buffer(2, Some(&c.buffer), 0);
                enc.set_buffer(3, Some(&params), 0);
                enc.dispatch_thread_groups(MTLSize::new(tg_x, tg_y, 1), MTLSize::new(128, 1, 1));
            });
        } else {
            let a_t = self.transpose_2d(a, k, m);
            self.matmul_accumulate(&a_t, b, c, m, k, n);
        }
    }

    fn softmax(&self, data: &MetalBuffer, n_rows: usize, row_len: usize) -> MetalBuffer {
        let pipeline = self.get_pipeline(reduce_msl::SOFTMAX_MSL, "softmax");
        let output_buf = self.make_buffer_empty(data.len * 4);
        let params = self.make_buffer_u32(&[n_rows as u32, row_len as u32]);
        let tg_size = std::cmp::min(row_len as u64, 256).next_power_of_two();

        self.with_encoder(|enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&data.buffer), 0);
            enc.set_buffer(1, Some(&output_buf), 0);
            enc.set_buffer(2, Some(&params), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(n_rows as u64, 1, 1),
                MTLSize::new(tg_size, 1, 1),
            );
        });

        MetalBuffer {
            buffer: output_buf,
            len: data.len,
            kind: Kind::F32,
        }
    }

    fn rms_norm(
        &self,
        data: &MetalBuffer,
        weight: &MetalBuffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
    ) -> MetalBuffer {
        let pipeline = self.get_pipeline(reduce_msl::RMS_NORM_MSL, "rms_norm");
        let output_buf = self.make_buffer_empty(data.len * 4);
        let params = self.make_buffer_u32(&[n_groups as u32, dim as u32, eps.to_bits()]);
        let tg_size = std::cmp::min(dim as u64, 256).next_power_of_two();

        self.with_encoder(|enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&data.buffer), 0);
            enc.set_buffer(1, Some(&weight.buffer), 0);
            enc.set_buffer(2, Some(&output_buf), 0);
            enc.set_buffer(3, Some(&params), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(n_groups as u64, 1, 1),
                MTLSize::new(tg_size, 1, 1),
            );
        });

        MetalBuffer {
            buffer: output_buf,
            len: data.len,
            kind: Kind::F32,
        }
    }

    fn embedding(
        &self,
        weight: &MetalBuffer,
        ids: &MetalBuffer,
        seq_len: usize,
        dim: usize,
    ) -> MetalBuffer {
        if let Kind::Q4 { group } = weight.kind {
            let pipeline = self.get_pipeline(llm_msl::Q4_MSL, "embedding_q4");
            let out = self.make_buffer_empty(seq_len * dim * 4);
            let params = self.make_buffer_u32(&[seq_len as u32, dim as u32, group]);
            let (so, bo) = weight.q4_offsets(group as usize);
            let total = (seq_len * dim) as u64;
            self.with_encoder(|enc| {
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(&weight.buffer), 0);
                enc.set_buffer(1, Some(&weight.buffer), so);
                enc.set_buffer(2, Some(&weight.buffer), bo);
                enc.set_buffer(3, Some(&ids.buffer), 0);
                enc.set_buffer(4, Some(&out), 0);
                enc.set_buffer(5, Some(&params), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(total.div_ceil(256), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            });
            return MetalBuffer {
                buffer: out,
                len: seq_len * dim,
                kind: Kind::F32,
            };
        }
        if weight.kind == Kind::Bf16 {
            let pipeline = self.get_pipeline(llm_msl::BF16_MSL, "embedding_bf16");
            let out = self.make_buffer_empty(seq_len * dim * 4);
            let params = self.make_buffer_u32(&[seq_len as u32, dim as u32]);
            self.dispatch(
                &pipeline,
                &[&weight.buffer, &ids.buffer, &out, &params],
                (seq_len * dim) as u64,
            );
            return MetalBuffer {
                buffer: out,
                len: seq_len * dim,
                kind: Kind::F32,
            };
        }
        // Simple MSL embedding kernel inline
        let src = r#"
#include <metal_stdlib>
using namespace metal;

kernel void embedding(
    device const float* weight [[buffer(0)]],
    device const uint* ids [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint seq_len = params[0];
    uint dim = params[1];
    if (gid >= seq_len) return;

    uint id = ids[gid];
    for (uint d = 0; d < dim; d++) {
        output[gid * dim + d] = weight[id * dim + d];
    }
}
"#;
        let pipeline = self.get_pipeline(src, "embedding");
        let output_buf = self.make_buffer_empty(seq_len * dim * 4);
        let params = self.make_buffer_u32(&[seq_len as u32, dim as u32]);

        self.dispatch(
            &pipeline,
            &[&weight.buffer, &ids.buffer, &output_buf, &params],
            seq_len as u64,
        );

        MetalBuffer {
            buffer: output_buf,
            len: seq_len * dim,
            kind: Kind::F32,
        }
    }

    fn reduce_sum(&self, data: &MetalBuffer, shape: &[usize], axis: usize) -> MetalBuffer {
        // CPU fallback for reduce_sum — complex axis handling
        self.sync();
        let cpu_data = data.to_vec();
        let cpu_dev = crate::CpuDevice::new();
        let cpu_buf = cpu_dev.upload(&cpu_data);
        let result = cpu_dev.reduce_sum(&cpu_buf, shape, axis);
        let out = cpu_dev.download(&result);
        self.upload(&out)
    }

    fn causal_attention(
        &self,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> MetalBuffer {
        let pipeline = self.get_pipeline(attention_msl::CAUSAL_ATTENTION_MSL, "causal_attention");
        let total_dim = n_heads * head_dim;
        let output_buf = self.make_buffer_empty(seq_len * total_dim * 4);
        let params = self.make_buffer_u32(&[
            seq_len as u32,
            n_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
        ]);
        let tg_size = std::cmp::min(head_dim as u64, 256).next_power_of_two();

        self.with_encoder(|enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&q.buffer), 0);
            enc.set_buffer(1, Some(&k.buffer), 0);
            enc.set_buffer(2, Some(&v.buffer), 0);
            enc.set_buffer(3, Some(&output_buf), 0);
            enc.set_buffer(4, Some(&params), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(seq_len as u64, n_heads as u64, 1),
                MTLSize::new(tg_size, 1, 1),
            );
        });

        MetalBuffer {
            buffer: output_buf,
            len: seq_len * total_dim,
            kind: Kind::F32,
        }
    }

    fn kv_attention(
        &self,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        cache_start: usize,
        q_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> MetalBuffer {
        if multi_attn(k_cache, q_len, n_heads, n_kv_heads, head_dim) {
            return self.flash_multi(
                q,
                k_cache,
                v_cache,
                cache_start,
                q_len,
                n_heads,
                n_kv_heads,
                head_dim,
                0,
                false,
                None,
            );
        }
        // Tiled prefill needs K/V readable to a 32-row boundary past the end.
        let kv_rows_needed = (cache_start + q_len).next_multiple_of(32);
        if q_len > 1
            && head_dim.is_multiple_of(8)
            && head_dim <= 128
            && n_heads.is_multiple_of(n_kv_heads)
            && k_cache.len >= kv_rows_needed * n_kv_heads * head_dim
            && v_cache.len >= kv_rows_needed * n_kv_heads * head_dim
        {
            return self.flash_prefill(
                q,
                k_cache,
                v_cache,
                cache_start,
                q_len,
                n_heads,
                n_kv_heads,
                head_dim,
                false,
                None,
            );
        }
        if head_dim.is_multiple_of(32) && head_dim <= 256 && n_heads.is_multiple_of(n_kv_heads) {
            return self.flash_attention(
                q,
                k_cache,
                v_cache,
                cache_start,
                q_len,
                n_heads,
                n_kv_heads,
                head_dim,
                0,
                false,
                None,
            );
        }
        if k_cache.kind == Kind::Bf16 {
            let (k, v) = (self.widen(k_cache), self.widen(v_cache));
            return self.kv_attention(q, &k, &v, cache_start, q_len, n_heads, n_kv_heads, head_dim);
        }
        let total_dim = n_heads * head_dim;
        let tg_size = std::cmp::min(head_dim as u64, 256).next_power_of_two();

        if q_len == 1 {
            let pipeline = self.get_pipeline(attention_msl::KV_ATTENTION_MSL, "kv_attention");
            let total_len = cache_start + 1;
            let output_buf = self.make_buffer_empty(total_dim * 4);
            let params = self.make_buffer_u32(&[
                total_len as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
            ]);

            self.with_encoder(|enc| {
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(&q.buffer), 0);
                enc.set_buffer(1, Some(&k_cache.buffer), 0);
                enc.set_buffer(2, Some(&v_cache.buffer), 0);
                enc.set_buffer(3, Some(&output_buf), 0);
                enc.set_buffer(4, Some(&params), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(n_heads as u64, 1, 1),
                    MTLSize::new(tg_size, 1, 1),
                );
            });

            MetalBuffer {
                buffer: output_buf,
                len: total_dim,
                kind: Kind::F32,
            }
        } else {
            let pipeline = self.get_pipeline(
                attention_msl::KV_ATTENTION_PREFILL_MSL,
                "kv_attention_prefill",
            );
            let output_buf = self.make_buffer_empty(q_len * total_dim * 4);
            let params = self.make_buffer_u32(&[
                cache_start as u32,
                q_len as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
            ]);

            self.with_encoder(|enc| {
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(&q.buffer), 0);
                enc.set_buffer(1, Some(&k_cache.buffer), 0);
                enc.set_buffer(2, Some(&v_cache.buffer), 0);
                enc.set_buffer(3, Some(&output_buf), 0);
                enc.set_buffer(4, Some(&params), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(q_len as u64, n_heads as u64, 1),
                    MTLSize::new(tg_size, 1, 1),
                );
            });

            MetalBuffer {
                buffer: output_buf,
                len: q_len * total_dim,
                kind: Kind::F32,
            }
        }
    }

    fn transpose_2d(&self, buf: &MetalBuffer, rows: usize, cols: usize) -> MetalBuffer {
        let pipeline = self.get_pipeline(backward_msl::TRANSPOSE_2D_MSL, "transpose_2d");
        let output_buf = self.make_buffer_empty(rows * cols * 4);
        let params = self.make_buffer_u32(&[rows as u32, cols as u32]);

        self.dispatch_2d(
            &pipeline,
            &[&buf.buffer, &output_buf, &params],
            (cols as u64, rows as u64),
            (16, 16),
        );

        MetalBuffer {
            buffer: output_buf,
            len: rows * cols,
            kind: Kind::F32,
        }
    }

    fn softmax_backward(
        &self,
        softmax_out: &MetalBuffer,
        grad_output: &MetalBuffer,
        n_rows: usize,
        row_len: usize,
    ) -> MetalBuffer {
        let pipeline = self.get_pipeline(backward_msl::SOFTMAX_BACKWARD_MSL, "softmax_backward");
        let output_buf = self.make_buffer_empty(n_rows * row_len * 4);
        let params = self.make_buffer_u32(&[n_rows as u32, row_len as u32]);
        let tg_size = std::cmp::min(row_len as u64, 256).next_power_of_two();

        self.with_encoder(|enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&softmax_out.buffer), 0);
            enc.set_buffer(1, Some(&grad_output.buffer), 0);
            enc.set_buffer(2, Some(&output_buf), 0);
            enc.set_buffer(3, Some(&params), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(n_rows as u64, 1, 1),
                MTLSize::new(tg_size, 1, 1),
            );
        });

        MetalBuffer {
            buffer: output_buf,
            len: n_rows * row_len,
            kind: Kind::F32,
        }
    }

    fn rms_norm_backward(
        &self,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        grad_output: &MetalBuffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
    ) -> (MetalBuffer, MetalBuffer) {
        let pipeline = self.get_pipeline(backward_msl::RMS_NORM_BACKWARD_MSL, "rms_norm_backward");
        let grad_input_raw = self.make_buffer_empty(n_groups * dim * 4);
        let grad_weight_raw = self.make_buffer(&vec![0.0f32; dim]);
        let params = self.make_buffer_u32(&[n_groups as u32, dim as u32, eps.to_bits()]);
        let tg_size = std::cmp::min(dim as u64, 256).next_power_of_two();

        self.with_encoder(|enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&input.buffer), 0);
            enc.set_buffer(1, Some(&weight.buffer), 0);
            enc.set_buffer(2, Some(&grad_output.buffer), 0);
            enc.set_buffer(3, Some(&grad_input_raw), 0);
            enc.set_buffer(4, Some(&grad_weight_raw), 0);
            enc.set_buffer(5, Some(&params), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(n_groups as u64, 1, 1),
                MTLSize::new(tg_size, 1, 1),
            );
        });

        (
            MetalBuffer {
                buffer: grad_input_raw,
                len: n_groups * dim,
                kind: Kind::F32,
            },
            MetalBuffer {
                buffer: grad_weight_raw,
                len: dim,
                kind: Kind::F32,
            },
        )
    }

    fn embedding_backward(
        &self,
        grad_output: &MetalBuffer,
        ids: &MetalBuffer,
        vocab_size: usize,
        seq_len: usize,
        dim: usize,
    ) -> MetalBuffer {
        let pipeline =
            self.get_pipeline(backward_msl::EMBEDDING_BACKWARD_MSL, "embedding_backward");
        // Zero-init for atomic accumulation
        let grad_weight_raw = self.make_buffer(&vec![0.0f32; vocab_size * dim]);
        let params = self.make_buffer_u32(&[vocab_size as u32, seq_len as u32, dim as u32]);

        self.dispatch(
            &pipeline,
            &[&grad_output.buffer, &ids.buffer, &grad_weight_raw, &params],
            seq_len as u64,
        );

        MetalBuffer {
            buffer: grad_weight_raw,
            len: vocab_size * dim,
            kind: Kind::F32,
        }
    }

    fn causal_attention_backward(
        &self,
        grad_output: &MetalBuffer,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> (MetalBuffer, MetalBuffer, MetalBuffer) {
        assert!(
            seq_len <= 2048,
            "causal_attention_backward: seq_len {} exceeds MAX_SEQ 2048",
            seq_len
        );
        let pipeline = self.get_pipeline(
            backward_msl::CAUSAL_ATTENTION_BACKWARD_MSL,
            "causal_attention_backward",
        );
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let grad_q_raw = self.make_buffer_empty(seq_len * total_dim * 4);
        let grad_k_raw = self.make_buffer(&vec![0.0f32; seq_len * kv_dim]);
        let grad_v_raw = self.make_buffer(&vec![0.0f32; seq_len * kv_dim]);
        let params = self.make_buffer_u32(&[
            seq_len as u32,
            n_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
        ]);
        let tg_size = std::cmp::min(head_dim as u64, 256).next_power_of_two();

        self.with_encoder(|enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&grad_output.buffer), 0);
            enc.set_buffer(1, Some(&q.buffer), 0);
            enc.set_buffer(2, Some(&k.buffer), 0);
            enc.set_buffer(3, Some(&v.buffer), 0);
            enc.set_buffer(4, Some(&grad_q_raw), 0);
            enc.set_buffer(5, Some(&grad_k_raw), 0);
            enc.set_buffer(6, Some(&grad_v_raw), 0);
            enc.set_buffer(7, Some(&params), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(seq_len as u64, n_heads as u64, 1),
                MTLSize::new(tg_size, 1, 1),
            );
        });

        (
            MetalBuffer {
                buffer: grad_q_raw,
                len: seq_len * total_dim,
                kind: Kind::F32,
            },
            MetalBuffer {
                buffer: grad_k_raw,
                len: seq_len * kv_dim,
                kind: Kind::F32,
            },
            MetalBuffer {
                buffer: grad_v_raw,
                len: seq_len * kv_dim,
                kind: Kind::F32,
            },
        )
    }

    fn cross_entropy_forward_backward(
        &self,
        logits: &MetalBuffer,
        targets: &MetalBuffer,
        n_positions: usize,
        vocab_size: usize,
        pad_id: u32,
    ) -> (f32, MetalBuffer) {
        // Pre-count non-padded positions on CPU (targets are small)
        self.sync();
        let target_data = targets.to_vec();
        let count = target_data.iter().filter(|t| t.to_bits() != pad_id).count() as u32;

        let pipeline = self.get_pipeline(backward_msl::CROSS_ENTROPY_MSL, "cross_entropy_fwd_bwd");
        let grad_buf = self.make_buffer_empty(n_positions * vocab_size * 4);
        let loss_raw = self.make_buffer(&[0.0f32]);
        let params = self.make_buffer_u32(&[n_positions as u32, vocab_size as u32, pad_id, count]);
        let tg_size = std::cmp::min(vocab_size as u64, 256).next_power_of_two();

        self.with_encoder(|enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&logits.buffer), 0);
            enc.set_buffer(1, Some(&targets.buffer), 0);
            enc.set_buffer(2, Some(&grad_buf), 0);
            enc.set_buffer(3, Some(&loss_raw), 0);
            enc.set_buffer(4, Some(&params), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(n_positions as u64, 1, 1),
                MTLSize::new(tg_size, 1, 1),
            );
        });

        self.sync();
        let loss_ptr = loss_raw.contents() as *const f32;
        let loss = unsafe { *loss_ptr };
        (
            loss,
            MetalBuffer {
                buffer: grad_buf,
                len: n_positions * vocab_size,
                kind: Kind::F32,
            },
        )
    }

    fn flush(&self) {
        self.end_compute();
        if let Some(cb) = self.active_cb.borrow_mut().take() {
            cb.commit();
            // Keep it alive until it's done; sync() waits on the newest buffer, and command
            // buffers on one queue run in order.
            self.in_flight.borrow_mut().push(cb);
        }
    }

    fn attention_prep(
        &self,
        qkv: &MetalBuffer,
        q_norm: Option<&MetalBuffer>,
        k_norm: Option<&MetalBuffer>,
        cos: &MetalBuffer,
        sin: &MetalBuffer,
        k_cache: &mut MetalBuffer,
        v_cache: &mut MetalBuffer,
        seq: usize,
        (nh, nkv, hd): (usize, usize, usize),
        pos: usize,
        eps: f32,
    ) -> MetalBuffer {
        if hd % 64 != 0 || hd > 256 {
            return crate::device::attention_prep_default(
                self,
                qkv,
                q_norm,
                k_norm,
                cos,
                sin,
                k_cache,
                v_cache,
                seq,
                (nh, nkv, hd),
                pos,
                eps,
            );
        }
        self.prep(
            qkv,
            q_norm,
            k_norm,
            cos,
            sin,
            k_cache,
            v_cache,
            seq,
            (nh, nkv, hd),
            pos,
            eps,
            None,
        )
    }

    fn attention_prep_paged(
        &self,
        qkv: &MetalBuffer,
        q_norm: Option<&MetalBuffer>,
        k_norm: Option<&MetalBuffer>,
        cos: &MetalBuffer,
        sin: &MetalBuffer,
        k_pool: &mut MetalBuffer,
        v_pool: &mut MetalBuffer,
        pages: &crate::Pages<MetalBuffer>,
        seq: usize,
        (nh, nkv, hd): (usize, usize, usize),
        pos: usize,
        eps: f32,
    ) -> MetalBuffer {
        if hd % 64 != 0 || hd > 256 {
            return crate::device::attention_prep_paged_default(
                self,
                qkv,
                q_norm,
                k_norm,
                cos,
                sin,
                k_pool,
                v_pool,
                pages,
                seq,
                (nh, nkv, hd),
                pos,
                eps,
            );
        }
        self.prep(
            qkv,
            q_norm,
            k_norm,
            cos,
            sin,
            k_pool,
            v_pool,
            seq,
            (nh, nkv, hd),
            pos,
            eps,
            Some((&pages.table.buffer, pages.shift())),
        )
    }

    fn kv_attention_paged(
        &self,
        q: &MetalBuffer,
        k_pool: &MetalBuffer,
        v_pool: &MetalBuffer,
        pages: &crate::Pages<MetalBuffer>,
        cache_start: usize,
        q_len: usize,
        (nh, nkv, hd): (usize, usize, usize),
        window: usize,
        causal: bool,
    ) -> MetalBuffer {
        let pg = Some((&pages.table.buffer, pages.shift()));
        let (k, v) = (k_pool, v_pool);
        let plain = causal && (window == 0 || cache_start + q_len <= window);
        let window = if plain { 0 } else { window };
        if hd > 256
            || nh % nkv != 0
            || (plain && hd % 32 != 0 && !multi_attn(k, q_len, nh, nkv, hd))
        {
            return crate::device::kv_attention_paged_default(
                self,
                q,
                k,
                v,
                pages,
                cache_start,
                q_len,
                (nh, nkv, hd),
                window,
                causal,
            );
        }
        if multi_attn(k, q_len, nh, nkv, hd) {
            return self.flash_multi(
                q,
                k,
                v,
                cache_start,
                q_len,
                nh,
                nkv,
                hd,
                window,
                !causal,
                pg,
            );
        }
        if plain && q_len > 1 && hd % 8 == 0 && hd <= 128 {
            return self.flash_prefill(q, k, v, cache_start, q_len, nh, nkv, hd, false, pg);
        }
        self.flash_attention(
            q,
            k,
            v,
            cache_start,
            q_len,
            nh,
            nkv,
            hd,
            window,
            !causal,
            pg,
        )
    }

    fn kv_attention_window(
        &self,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        cache_start: usize,
        q_len: usize,
        (n_heads, n_kv_heads, head_dim): (usize, usize, usize),
        window: usize,
        causal: bool,
    ) -> MetalBuffer {
        if causal && (window == 0 || cache_start + q_len <= window) {
            return self.kv_attention(
                q,
                k_cache,
                v_cache,
                cache_start,
                q_len,
                n_heads,
                n_kv_heads,
                head_dim,
            );
        }
        assert!(
            head_dim <= 256 && n_heads % n_kv_heads == 0,
            "windowed or bidirectional attention needs head_dim at most 256"
        );
        if multi_attn(k_cache, q_len, n_heads, n_kv_heads, head_dim) {
            return self.flash_multi(
                q,
                k_cache,
                v_cache,
                cache_start,
                q_len,
                n_heads,
                n_kv_heads,
                head_dim,
                window,
                !causal,
                None,
            );
        }
        self.flash_attention(
            q,
            k_cache,
            v_cache,
            cache_start,
            q_len,
            n_heads,
            n_kv_heads,
            head_dim,
            window,
            !causal,
            None,
        )
    }

    fn attention_full(
        &self,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        n: usize,
        nh: usize,
        hd: usize,
    ) -> MetalBuffer {
        // The tiled simdgroup-matrix kernel when the shape allows (vision towers: D 64-128).
        let rows = n.next_multiple_of(32) * nh * hd;
        if hd.is_multiple_of(8) && hd <= 128 && k.len >= rows && v.len >= rows {
            return self.flash_prefill(q, k, v, 0, n, nh, nh, hd, true, None);
        }
        self.flash_attention(q, k, v, 0, n, nh, nh, hd, 0, true, None)
    }

    fn modulate_channels(
        &self,
        data: &MetalBuffer,
        scale: &MetalBuffer,
        offset: usize,
        rows: usize,
        dim: usize,
    ) -> MetalBuffer {
        assert!(dim > 0 && data.len() == rows * dim && scale.len() >= offset + dim);
        assert_eq!(data.kind, Kind::F32);
        assert_eq!(scale.kind, Kind::F32);
        let pipeline = self.get_pipeline(reduce_msl::CHANNEL_MODULATION_MSL, "channel_modulation");
        let out = self.make_buffer_empty(rows * dim * 4);
        let params = self.make_buffer_u32(&[(rows * dim) as u32, dim as u32, offset as u32]);
        self.dispatch(
            &pipeline,
            &[&data.buffer, &scale.buffer, &out, &params],
            (rows * dim) as u64,
        );
        MetalBuffer {
            buffer: out,
            len: rows * dim,
            kind: Kind::F32,
        }
    }

    fn rms_norm_scale(
        &self,
        data: &MetalBuffer,
        weight: &MetalBuffer,
        scale: &MetalBuffer,
        offset: usize,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> MetalBuffer {
        self.modulated_rms(data, weight, scale, None, offset, rows, dim, eps)
    }

    fn rms_norm_gate_residual(
        &self,
        data: &MetalBuffer,
        weight: &MetalBuffer,
        gate: &MetalBuffer,
        residual: &MetalBuffer,
        offset: usize,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> MetalBuffer {
        self.modulated_rms(data, weight, gate, Some(residual), offset, rows, dim, eps)
    }

    fn layer_norm(
        &self,
        x: &MetalBuffer,
        w: &MetalBuffer,
        b: &MetalBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> MetalBuffer {
        let pipeline = self.get_pipeline(llm_msl::FUSED_MSL, "layer_norm");
        let out = self.make_buffer_empty(rows * dim * 4);
        let eps = self.make_buffer(&[eps]);
        let params = self.make_buffer_u32(&[rows as u32, dim as u32]);
        self.dispatch_groups(
            &pipeline,
            &[&x.buffer, &w.buffer, &b.buffer, &out, &eps, &params],
            (rows, 1),
            256,
        );
        MetalBuffer {
            buffer: out,
            len: rows * dim,
            kind: Kind::F32,
        }
    }

    fn conv2d_nchw(
        &self,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        bias: &MetalBuffer,
        input: usize,
        output: usize,
        h: usize,
        w: usize,
        kernel: usize,
    ) -> MetalBuffer {
        assert!(
            input > 0 && output > 0 && h > 0 && w > 0 && kernel > 0 && !kernel.is_multiple_of(2)
        );
        assert_eq!(x.len(), input * h * w);
        assert_eq!(weight.len(), output * input * kernel * kernel);
        assert_eq!(bias.len(), output);
        for b in [x, weight, bias] {
            assert_eq!(b.kind, Kind::F32);
        }
        let pipeline = self.get_pipeline(matmul_msl::CONV2D_MSL, "conv2d_nchw");
        let out = self.make_buffer_empty(output * h * w * 4);
        let params = self.make_buffer_u32(&[
            input as u32,
            output as u32,
            h as u32,
            w as u32,
            kernel as u32,
        ]);
        self.dispatch_groups(
            &pipeline,
            &[&x.buffer, &weight.buffer, &bias.buffer, &out, &params],
            (output.div_ceil(32), (h * w).div_ceil(32)),
            128,
        );
        MetalBuffer {
            buffer: out,
            len: output * h * w,
            kind: Kind::F32,
        }
    }

    fn group_norm_affine(
        &self,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        bias: &MetalBuffer,
        channels: usize,
        spatial: usize,
        groups: usize,
        eps: f32,
    ) -> MetalBuffer {
        assert!(groups > 0 && channels > 0 && spatial > 0 && channels.is_multiple_of(groups));
        assert_eq!(x.len(), channels * spatial);
        assert_eq!(weight.len(), channels);
        assert_eq!(bias.len(), channels);
        for b in [x, weight, bias] {
            assert_eq!(b.kind, Kind::F32);
        }
        let pipeline = self.get_pipeline(reduce_msl::VAE_MSL, "group_norm_affine");
        let out = self.make_buffer_empty(x.len() * 4);
        let params = self.make_buffer_u32(&[
            channels as u32,
            spatial as u32,
            groups as u32,
            eps.to_bits(),
        ]);
        self.dispatch_groups(
            &pipeline,
            &[&x.buffer, &weight.buffer, &bias.buffer, &out, &params],
            (groups, 1),
            256,
        );
        MetalBuffer {
            buffer: out,
            len: x.len(),
            kind: Kind::F32,
        }
    }

    fn upsample_nearest_2x(
        &self,
        x: &MetalBuffer,
        channels: usize,
        h: usize,
        w: usize,
    ) -> MetalBuffer {
        assert!(channels > 0 && h > 0 && w > 0);
        assert_eq!(x.len(), channels * h * w);
        assert_eq!(x.kind, Kind::F32);
        let pipeline = self.get_pipeline(reduce_msl::VAE_MSL, "upsample_nearest_2x");
        let out = self.make_buffer_empty(x.len() * 16);
        let params = self.make_buffer_u32(&[channels as u32, h as u32, w as u32]);
        self.dispatch(&pipeline, &[&x.buffer, &out, &params], (x.len() * 4) as u64);
        MetalBuffer {
            buffer: out,
            len: x.len() * 4,
            kind: Kind::F32,
        }
    }

    fn gelu_tanh(&self, x: &MetalBuffer, n: usize) -> MetalBuffer {
        let pipeline = self.get_pipeline(llm_msl::FUSED_MSL, "gelu_tanh");
        let out = self.make_buffer_empty(n * 4);
        let params = self.make_buffer_u32(&[n as u32]);
        self.dispatch(&pipeline, &[&x.buffer, &out, &params], n as u64);
        MetalBuffer {
            buffer: out,
            len: n,
            kind: Kind::F32,
        }
    }

    fn gelu_erf(&self, x: &MetalBuffer, n: usize) -> MetalBuffer {
        let pipeline = self.get_pipeline(llm_msl::FUSED_MSL, "gelu_erf");
        let out = self.make_buffer_empty(n * 4);
        let params = self.make_buffer_u32(&[n as u32]);
        self.dispatch(&pipeline, &[&x.buffer, &out, &params], n as u64);
        MetalBuffer {
            buffer: out,
            len: n,
            kind: Kind::F32,
        }
    }

    fn geglu_split(&self, gu: &MetalBuffer, rows: usize, ff: usize) -> MetalBuffer {
        let pipeline = self.get_pipeline(llm_msl::FUSED_MSL, "geglu_split");
        let out = self.make_buffer_empty(rows * ff * 4);
        let params = self.make_buffer_u32(&[rows as u32, ff as u32]);
        self.dispatch(&pipeline, &[&gu.buffer, &out, &params], (rows * ff) as u64);
        MetalBuffer {
            buffer: out,
            len: rows * ff,
            kind: Kind::F32,
        }
    }

    fn swiglu_split(&self, gu: &MetalBuffer, rows: usize, ff: usize) -> MetalBuffer {
        let pipeline = self.get_pipeline(llm_msl::FUSED_MSL, "swiglu_split");
        let out = self.make_buffer_empty(rows * ff * 4);
        let params = self.make_buffer_u32(&[rows as u32, ff as u32]);
        self.dispatch(&pipeline, &[&gu.buffer, &out, &params], (rows * ff) as u64);
        MetalBuffer {
            buffer: out,
            len: rows * ff,
            kind: Kind::F32,
        }
    }

    fn sync(&self) {
        self.end_compute();
        if let Some(cb) = self.active_cb.borrow_mut().take() {
            cb.commit();
            cb.wait_until_completed();
        }
        for cb in self.in_flight.borrow_mut().drain(..) {
            cb.wait_until_completed();
        }
    }

    fn copy_buffer(&self, src: &MetalBuffer) -> MetalBuffer {
        let dst = self.make_buffer_empty(src.len * 4);
        self.end_compute();
        self.ensure_cb();
        autoreleasepool(|| {
            let cb_ref = self.active_cb.borrow();
            let cmd = cb_ref.as_ref().unwrap();
            let blit = cmd.new_blit_command_encoder();
            blit.copy_from_buffer(&src.buffer, 0, &dst, 0, (src.len * 4) as u64);
            blit.end_encoding();
        });
        MetalBuffer {
            buffer: dst,
            len: src.len,
            kind: Kind::F32,
        }
    }

    fn bias_add(
        &self,
        matrix: &MetalBuffer,
        bias: &MetalBuffer,
        numel: usize,
        dim: usize,
    ) -> MetalBuffer {
        let src = r#"
#include <metal_stdlib>
using namespace metal;

kernel void bias_add(
    device const float* matrix [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint numel = params[0];
    uint dim = params[1];
    if (gid >= numel) return;
    output[gid] = matrix[gid] + bias[gid % dim];
}
"#;
        let pipeline = self.get_pipeline(src, "bias_add");
        let output_buf = self.make_buffer_empty(numel * 4);
        let params = self.make_buffer_u32(&[numel as u32, dim as u32]);

        self.dispatch(
            &pipeline,
            &[&matrix.buffer, &bias.buffer, &output_buf, &params],
            numel as u64,
        );

        MetalBuffer {
            buffer: output_buf,
            len: numel,
            kind: Kind::F32,
        }
    }

    fn add_assign(&self, dst: &mut MetalBuffer, src: &MetalBuffer) {
        let pipeline = self.get_pipeline(adamw_msl::ADD_ASSIGN_MSL, "add_assign");
        let params = self.make_buffer_u32(&[dst.len as u32]);
        self.dispatch(
            &pipeline,
            &[&dst.buffer, &src.buffer, &params],
            dst.len as u64,
        );
    }

    fn zero_buffer(&self, buf: &mut MetalBuffer) {
        self.end_compute();
        self.ensure_cb();
        autoreleasepool(|| {
            let cb_ref = self.active_cb.borrow();
            let cmd = cb_ref.as_ref().unwrap();
            let blit = cmd.new_blit_command_encoder();
            blit.fill_buffer(&buf.buffer, metal::NSRange::new(0, (buf.len * 4) as u64), 0);
            blit.end_encoding();
        });
    }

    fn reduce_sum_sq_accumulate(&self, src: &MetalBuffer, acc: &mut MetalBuffer) {
        let pipeline = self.get_pipeline(adamw_msl::REDUCE_SUM_SQ_MSL, "reduce_sum_sq");
        let params = self.make_buffer_u32(&[src.len as u32]);
        // Each thread handles 4 elements
        let threads = src.len.div_ceil(4) as u64;
        self.dispatch(&pipeline, &[&src.buffer, &acc.buffer, &params], threads);
    }

    fn scale_buffer(&self, buf: &mut MetalBuffer, scale: f32) {
        let pipeline = self.get_pipeline(adamw_msl::SCALE_BUFFER_MSL, "scale_buffer");
        let scale_buf = self.make_buffer(&[scale]);
        let params = self.make_buffer_u32(&[buf.len as u32]);
        self.dispatch(
            &pipeline,
            &[&buf.buffer, &scale_buf, &params],
            buf.len as u64,
        );
    }

    fn linear(
        &self,
        x: &MetalBuffer,
        w: &MetalBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> MetalBuffer {
        if let Kind::Q4 { group } = w.kind {
            let out = self.make_buffer_empty(m * n * 4);
            // TANG_Q4_SMALL=old: the previous small-row paths (for comparison).
            let new_small = std::env::var("TANG_Q4_SMALL").as_deref() != Ok("old");
            if new_small
                && (4..=8).contains(&m)
                && k.is_multiple_of(128)
                && group.is_multiple_of(64)
            {
                // A few rows (a verify forward): the matrix units, weights streamed once.
                let pipeline = self.get_pipeline(llm_msl::Q4_MSL, "qmv_mma");
                let (so, bo) = w.q4_offsets(group as usize);
                let params = self.make_buffer_u32(&[m as u32, k as u32, n as u32, group]);
                self.with_encoder(|enc| {
                    enc.set_compute_pipeline_state(&pipeline);
                    enc.set_buffer(0, Some(&x.buffer), 0);
                    enc.set_buffer(1, Some(&w.buffer), 0);
                    enc.set_buffer(2, Some(&w.buffer), so);
                    enc.set_buffer(3, Some(&w.buffer), bo);
                    enc.set_buffer(4, Some(&out), 0);
                    enc.set_buffer(5, Some(&params), 0);
                    enc.dispatch_thread_groups(
                        MTLSize::new(n.div_ceil(32) as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                });
                return MetalBuffer {
                    buffer: out,
                    len: m * n,
                    kind: Kind::F32,
                };
            }
            if new_small && (2..=3).contains(&m) && k.is_multiple_of(16) {
                // Two or three rows: each weight unpacked once and applied to every row.
                let pipeline = self.get_pipeline(llm_msl::Q4_MSL, &format!("gemv_q4_m{m}"));
                let (so, bo) = w.q4_offsets(group as usize);
                let params = self.make_buffer_u32(&[m as u32, k as u32, n as u32, group]);
                self.with_encoder(|enc| {
                    enc.set_compute_pipeline_state(&pipeline);
                    enc.set_buffer(0, Some(&x.buffer), 0);
                    enc.set_buffer(1, Some(&w.buffer), 0);
                    enc.set_buffer(2, Some(&w.buffer), so);
                    enc.set_buffer(3, Some(&w.buffer), bo);
                    enc.set_buffer(4, Some(&out), 0);
                    enc.set_buffer(5, Some(&params), 0);
                    enc.dispatch_thread_groups(
                        MTLSize::new(n.div_ceil(8) as u64, 1, 1),
                        MTLSize::new(128, 1, 1),
                    );
                });
                return MetalBuffer {
                    buffer: out,
                    len: m * n,
                    kind: Kind::F32,
                };
            }
            if (2..=4).contains(&m) && k.is_multiple_of(16) {
                // A few rows: the GEMV that unpacks each weight once for all of them.
                let pipeline = self.get_pipeline(llm_msl::Q4_MSL, "gemv_q4_rows");
                let (so, bo) = w.q4_offsets(group as usize);
                let params = self.make_buffer_u32(&[m as u32, k as u32, n as u32, group]);
                self.with_encoder(|enc| {
                    enc.set_compute_pipeline_state(&pipeline);
                    enc.set_buffer(0, Some(&x.buffer), 0);
                    enc.set_buffer(1, Some(&w.buffer), 0);
                    enc.set_buffer(2, Some(&w.buffer), so);
                    enc.set_buffer(3, Some(&w.buffer), bo);
                    enc.set_buffer(4, Some(&out), 0);
                    enc.set_buffer(5, Some(&params), 0);
                    enc.dispatch_thread_groups(
                        MTLSize::new(n.div_ceil(8) as u64, 1, 1),
                        MTLSize::new(64, 1, 1),
                    );
                });
                return MetalBuffer {
                    buffer: out,
                    len: m * n,
                    kind: Kind::F32,
                };
            }
            if (2..=SMALL_GEMM_ROWS).contains(&m) && k.is_multiple_of(32) {
                return self.gemm_small(x, w, m, k, n);
            }
            let params = self.make_buffer_u32(&[m as u32, k as u32, n as u32, group]);
            let (so, bo) = w.q4_offsets(group as usize);
            let (name, groups, threads) = if m <= 8 {
                ("gemv_q4", (n.div_ceil(8), 1), 64)
            } else {
                ("matmul_bt_q4", (n.div_ceil(32), m.div_ceil(32)), 128)
            };
            let pipeline = self.get_pipeline(llm_msl::Q4_MSL, name);
            self.with_encoder(|enc| {
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(&x.buffer), 0);
                enc.set_buffer(1, Some(&w.buffer), 0);
                enc.set_buffer(2, Some(&w.buffer), so);
                enc.set_buffer(3, Some(&w.buffer), bo);
                enc.set_buffer(4, Some(&out), 0);
                enc.set_buffer(5, Some(&params), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(groups.0 as u64, groups.1 as u64, 1),
                    MTLSize::new(threads, 1, 1),
                );
            });
            return MetalBuffer {
                buffer: out,
                len: m * n,
                kind: Kind::F32,
            };
        }
        if w.kind == Kind::Bf16 && (5..=SMALL_GEMM_ROWS).contains(&m) && k.is_multiple_of(32) {
            return self.gemm_small(x, w, m, k, n);
        }
        if w.kind == Kind::Bf16 {
            let out = self.make_buffer_empty(m * n * 4);
            let params = self.make_buffer_u32(&[m as u32, k as u32, n as u32]);
            let bufs = [&x.buffer, &w.buffer, &out, &params];
            if m <= 8 {
                let pipeline = self.get_pipeline(llm_msl::BF16_MSL, "gemv_bt_bf16");
                self.dispatch_groups(&pipeline, &bufs, (n.div_ceil(8), 1), 256);
            } else {
                let pipeline = self.get_pipeline(llm_msl::BF16_MSL, "matmul_bt_bf16");
                self.dispatch_groups(&pipeline, &bufs, (n.div_ceil(32), m.div_ceil(32)), 128);
            }
            return MetalBuffer {
                buffer: out,
                len: m * n,
                kind: Kind::F32,
            };
        }
        if m <= 8 {
            let pipeline = self.get_pipeline(llm_msl::GEMV_BT_MSL, "gemv_bt");
            let out = self.make_buffer_empty(m * n * 4);
            let params = self.make_buffer_u32(&[m as u32, k as u32, n as u32]);
            self.with_encoder(|enc| {
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(&x.buffer), 0);
                enc.set_buffer(1, Some(&w.buffer), 0);
                enc.set_buffer(2, Some(&out), 0);
                enc.set_buffer(3, Some(&params), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(n.div_ceil(8) as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            });
            return MetalBuffer {
                buffer: out,
                len: m * n,
                kind: Kind::F32,
            };
        }
        if !n.is_multiple_of(32) || !k.is_multiple_of(8) {
            return self.matmul_b_transposed(x, w, m, k, n);
        }
        // The simdgroup kernel works in 32-row tiles: pad the rows, then drop the padding.
        let mp = m.next_multiple_of(32);
        if mp == m {
            return self.matmul_b_transposed(x, w, m, k, n);
        }
        let mut padded = self.alloc(mp * k);
        self.write_into(&mut padded, 0, x);
        let y = self.matmul_b_transposed(&padded, w, mp, k, n);
        self.slice_buffer(&y, 0, m * n)
    }

    fn add_tensors_buf(&self, a: &MetalBuffer, b: &MetalBuffer, numel: usize) -> MetalBuffer {
        self.binary(a, b, numel, "add_f32")
    }

    fn silu_buf(&self, input: &MetalBuffer, numel: usize) -> MetalBuffer {
        let pipeline = self.get_pipeline(llm_msl::ELEMENTWISE_MSL, "silu_f32");
        let out = self.make_buffer_empty(numel * 4);
        let params = self.make_buffer_u32(&[numel as u32]);
        self.dispatch(&pipeline, &[&input.buffer, &out, &params], numel as u64);
        MetalBuffer {
            buffer: out,
            len: numel,
            kind: Kind::F32,
        }
    }

    fn swiglu_fused_buf(&self, gate: &MetalBuffer, up: &MetalBuffer, numel: usize) -> MetalBuffer {
        self.binary(gate, up, numel, "swiglu_f32")
    }

    fn rope_half_cached(
        &self,
        input: &MetalBuffer,
        cos_buf: &MetalBuffer,
        sin_buf: &MetalBuffer,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
    ) -> MetalBuffer {
        let pipeline = self.get_pipeline(llm_msl::ROPE_HALF_MSL, "rope_half");
        let out = self.make_buffer_empty(input.len * 4);
        let params = self.make_buffer_u32(&[
            seq_len as u32,
            n_heads as u32,
            head_dim as u32,
            start_pos as u32,
        ]);
        self.dispatch(
            &pipeline,
            &[
                &input.buffer,
                &cos_buf.buffer,
                &sin_buf.buffer,
                &out,
                &params,
            ],
            (seq_len * n_heads * head_dim / 2) as u64,
        );
        MetalBuffer {
            buffer: out,
            len: input.len,
            kind: Kind::F32,
        }
    }

    fn slice_buffer(&self, buf: &MetalBuffer, offset: usize, len: usize) -> MetalBuffer {
        if buf.kind == Kind::Bf16 {
            let dst = self.make_buffer_empty(len * 2);
            self.copy_with("copy_u16", &buf.buffer, offset, &dst, 0, len);
            return MetalBuffer {
                buffer: dst,
                len,
                kind: Kind::Bf16,
            };
        }
        let dst = self.make_buffer_empty(len * 4);
        self.copy_f32(&buf.buffer, offset, &dst, 0, len);
        MetalBuffer {
            buffer: dst,
            len,
            kind: Kind::F32,
        }
    }

    fn write_into(&self, dst: &mut MetalBuffer, offset: usize, src: &MetalBuffer) {
        assert!(offset + src.len <= dst.len, "write_into out of bounds");
        let kernel = match (dst.kind, src.kind) {
            (Kind::F32, Kind::F32) => "copy_f32",
            (Kind::Bf16, Kind::Bf16) => "copy_u16",
            (Kind::Bf16, Kind::F32) => "f32_to_bf16",
            (Kind::F32, Kind::Bf16) => "bf16_to_f32",
            (d, s) => panic!("write_into: can't copy {s:?} into {d:?}"),
        };
        self.copy_with(kernel, &src.buffer, 0, &dst.buffer, offset, src.len);
    }

    fn adamw_step(
        &self,
        param: &mut MetalBuffer,
        grad: &MetalBuffer,
        m: &mut MetalBuffer,
        v: &mut MetalBuffer,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step_t: usize,
    ) {
        let pipeline = self.get_pipeline(adamw_msl::ADAMW_STEP_MSL, "adamw_step");
        let beta1_pow = beta1.powi(step_t as i32);
        let beta2_pow = beta2.powi(step_t as i32);
        let hparams =
            self.make_buffer(&[lr, beta1, beta2, eps, weight_decay, beta1_pow, beta2_pow]);
        let count = self.make_buffer_u32(&[param.len as u32]);
        self.dispatch(
            &pipeline,
            &[
                &param.buffer,
                &grad.buffer,
                &m.buffer,
                &v.buffer,
                &hparams,
                &count,
            ],
            param.len as u64,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuDevice;
    use crate::device::ComputeDevice;

    fn get_metal_device() -> MetalDevice {
        MetalDevice::new().expect("Metal device not available")
    }

    #[test]
    fn metal_upload_download() {
        let dev = get_metal_device();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let buf = dev.upload(&data);
        assert_eq!(dev.download(&buf), data);
    }

    #[test]
    fn metal_elementwise_add() {
        let dev = get_metal_device();
        let a = dev.upload(&[1.0, 2.0, 3.0]);
        let b = dev.upload(&[4.0, 5.0, 6.0]);
        let c = dev.elementwise(&[&a, &b], 3, &|vars| vars[0] + vars[1]);
        let out = dev.download(&c);
        assert!((out[0] - 5.0).abs() < 1e-5);
        assert!((out[1] - 7.0).abs() < 1e-5);
        assert!((out[2] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn metal_matmul_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2

        let a_m = metal.upload(&a_data);
        let b_m = metal.upload(&b_data);
        let c_m = metal.matmul(&a_m, &b_m, 2, 3, 2);
        let metal_out = metal.download(&c_m);

        let a_c = cpu.upload(&a_data);
        let b_c = cpu.upload(&b_data);
        let c_c = cpu.matmul(&a_c, &b_c, 2, 3, 2);
        let cpu_out = cpu.download(&c_c);

        for i in 0..4 {
            assert!(
                (metal_out[i] - cpu_out[i]).abs() < 1e-3,
                "matmul mismatch at {i}: metal={} cpu={}",
                metal_out[i],
                cpu_out[i]
            );
        }
    }

    #[test]
    fn metal_softmax_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let m_buf = metal.upload(&data);
        let m_out = metal.download(&metal.softmax(&m_buf, 2, 3));

        let c_buf = cpu.upload(&data);
        let c_out = cpu.download(&cpu.softmax(&c_buf, 2, 3));

        for i in 0..6 {
            assert!(
                (m_out[i] - c_out[i]).abs() < 1e-5,
                "softmax mismatch at {i}: metal={} cpu={}",
                m_out[i],
                c_out[i]
            );
        }
    }

    #[test]
    fn metal_rms_norm_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0];

        let m_data = metal.upload(&data);
        let m_weight = metal.upload(&weight);
        let m_out = metal.download(&metal.rms_norm(&m_data, &m_weight, 2, 2, 1e-5));

        let c_data = cpu.upload(&data);
        let c_weight = cpu.upload(&weight);
        let c_out = cpu.download(&cpu.rms_norm(&c_data, &c_weight, 2, 2, 1e-5));

        for i in 0..4 {
            assert!(
                (m_out[i] - c_out[i]).abs() < 1e-4,
                "rms_norm mismatch at {i}: metal={} cpu={}",
                m_out[i],
                c_out[i]
            );
        }
    }

    #[test]
    fn metal_embedding() {
        let metal = get_metal_device();
        let weight = metal.upload(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let ids = metal.upload_u32(&[2, 0, 1]);
        let result = metal.embedding(&weight, &ids, 3, 2);
        let out = metal.download(&result);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.6).abs() < 1e-6);
        assert!((out[2] - 0.1).abs() < 1e-6);
        assert!((out[3] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn metal_kv_attention_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let q_data = vec![1.0, 0.0];
        let k_data = vec![1.0, 0.0, 0.0, 1.0];
        let v_data = vec![1.0, 2.0, 3.0, 4.0];

        let m_q = metal.upload(&q_data);
        let m_k = metal.upload(&k_data);
        let m_v = metal.upload(&v_data);
        let m_out = metal.download(&metal.kv_attention(&m_q, &m_k, &m_v, 1, 1, 1, 1, 2));

        let c_q = cpu.upload(&q_data);
        let c_k = cpu.upload(&k_data);
        let c_v = cpu.upload(&v_data);
        let c_out = cpu.download(&cpu.kv_attention(&c_q, &c_k, &c_v, 1, 1, 1, 1, 2));

        for i in 0..2 {
            assert!(
                (m_out[i] - c_out[i]).abs() < 1e-3,
                "kv_attention mismatch at {i}: metal={} cpu={}",
                m_out[i],
                c_out[i]
            );
        }
    }

    #[test]
    fn metal_transpose_2d_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3

        let m_buf = metal.upload(&data);
        let m_out = metal.download(&metal.transpose_2d(&m_buf, 2, 3));

        let c_buf = cpu.upload(&data);
        let c_out = cpu.download(&cpu.transpose_2d(&c_buf, 2, 3));

        assert_eq!(m_out.len(), c_out.len());
        for i in 0..m_out.len() {
            assert!(
                (m_out[i] - c_out[i]).abs() < 1e-5,
                "transpose mismatch at {i}: metal={} cpu={}",
                m_out[i],
                c_out[i]
            );
        }
    }

    #[test]
    fn metal_softmax_backward_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let sm_data = vec![0.2, 0.3, 0.5, 0.1, 0.6, 0.3];
        let grad_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        let m_sm = metal.upload(&sm_data);
        let m_grad = metal.upload(&grad_data);
        let m_out = metal.download(&metal.softmax_backward(&m_sm, &m_grad, 2, 3));

        let c_sm = cpu.upload(&sm_data);
        let c_grad = cpu.upload(&grad_data);
        let c_out = cpu.download(&cpu.softmax_backward(&c_sm, &c_grad, 2, 3));

        for i in 0..m_out.len() {
            assert!(
                (m_out[i] - c_out[i]).abs() < 1e-4,
                "softmax_backward mismatch at {i}: metal={} cpu={}",
                m_out[i],
                c_out[i]
            );
        }
    }

    #[test]
    fn metal_embedding_backward_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let grad_data = vec![1.0, 2.0, 3.0, 0.1, 0.2, 0.3];
        let ids: Vec<u32> = vec![1, 3];

        let m_grad = metal.upload(&grad_data);
        let m_ids = metal.upload_u32(&ids);
        let m_out = metal.download(&metal.embedding_backward(&m_grad, &m_ids, 4, 2, 3));

        let c_grad = cpu.upload(&grad_data);
        let c_ids = cpu.upload_u32(&ids);
        let c_out = cpu.download(&cpu.embedding_backward(&c_grad, &c_ids, 4, 2, 3));

        for i in 0..m_out.len() {
            assert!(
                (m_out[i] - c_out[i]).abs() < 1e-4,
                "embedding_backward mismatch at {i}: metal={} cpu={}",
                m_out[i],
                c_out[i]
            );
        }
    }

    #[test]
    fn metal_cross_entropy_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let targets: Vec<u32> = vec![2, 0];

        let m_logits = metal.upload(&logits);
        let m_targets = metal.upload_u32(&targets);
        let (m_loss, m_grad_buf) =
            metal.cross_entropy_forward_backward(&m_logits, &m_targets, 2, 3, 99);
        let m_grad = metal.download(&m_grad_buf);

        let c_logits = cpu.upload(&logits);
        let c_targets = cpu.upload_u32(&targets);
        let (c_loss, c_grad_buf) =
            cpu.cross_entropy_forward_backward(&c_logits, &c_targets, 2, 3, 99);
        let c_grad = cpu.download(&c_grad_buf);

        assert!(
            (m_loss - c_loss).abs() < 1e-3,
            "cross_entropy loss mismatch: metal={} cpu={}",
            m_loss,
            c_loss
        );
        for i in 0..m_grad.len() {
            assert!(
                (m_grad[i] - c_grad[i]).abs() < 1e-3,
                "cross_entropy grad mismatch at {i}: metal={} cpu={}",
                m_grad[i],
                c_grad[i]
            );
        }
    }

    #[test]
    fn metal_rms_norm_backward_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0];
        let grad_out = vec![1.0, 0.0, 0.0, 1.0];

        let m_input = metal.upload(&input);
        let m_weight = metal.upload(&weight);
        let m_grad = metal.upload(&grad_out);
        let (m_gi, m_gw) = metal.rms_norm_backward(&m_input, &m_weight, &m_grad, 2, 2, 1e-5);
        let m_gi_v = metal.download(&m_gi);
        let m_gw_v = metal.download(&m_gw);

        let c_input = cpu.upload(&input);
        let c_weight = cpu.upload(&weight);
        let c_grad = cpu.upload(&grad_out);
        let (c_gi, c_gw) = cpu.rms_norm_backward(&c_input, &c_weight, &c_grad, 2, 2, 1e-5);
        let c_gi_v = cpu.download(&c_gi);
        let c_gw_v = cpu.download(&c_gw);

        for i in 0..m_gi_v.len() {
            assert!(
                (m_gi_v[i] - c_gi_v[i]).abs() < 1e-3,
                "rms_norm_backward grad_input mismatch at {i}: metal={} cpu={}",
                m_gi_v[i],
                c_gi_v[i]
            );
        }
        for i in 0..m_gw_v.len() {
            assert!(
                (m_gw_v[i] - c_gw_v[i]).abs() < 1e-3,
                "rms_norm_backward grad_weight mismatch at {i}: metal={} cpu={}",
                m_gw_v[i],
                c_gw_v[i]
            );
        }
    }

    #[test]
    fn metal_causal_attention_backward_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let seq_len = 4;
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        // Deterministic pseudo-random data
        let q_data: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i * 7 + 3) % 13) as f32 / 13.0 - 0.5)
            .collect();
        let k_data: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i * 11 + 5) % 17) as f32 / 17.0 - 0.5)
            .collect();
        let v_data: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i * 13 + 7) % 19) as f32 / 19.0 - 0.5)
            .collect();
        let go_data: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i * 3 + 11) % 23) as f32 / 23.0 - 0.5)
            .collect();

        // Metal
        let m_go = metal.upload(&go_data);
        let m_q = metal.upload(&q_data);
        let m_k = metal.upload(&k_data);
        let m_v = metal.upload(&v_data);
        let (m_gq, m_gk, m_gv) = metal.causal_attention_backward(
            &m_go, &m_q, &m_k, &m_v, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let m_gq_v = metal.download(&m_gq);
        let m_gk_v = metal.download(&m_gk);
        let m_gv_v = metal.download(&m_gv);

        // CPU
        let c_go = cpu.upload(&go_data);
        let c_q = cpu.upload(&q_data);
        let c_k = cpu.upload(&k_data);
        let c_v = cpu.upload(&v_data);
        let (c_gq, c_gk, c_gv) = cpu.causal_attention_backward(
            &c_go, &c_q, &c_k, &c_v, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let c_gq_v = cpu.download(&c_gq);
        let c_gk_v = cpu.download(&c_gk);
        let c_gv_v = cpu.download(&c_gv);

        for i in 0..m_gq_v.len() {
            assert!(
                (m_gq_v[i] - c_gq_v[i]).abs() < 1e-3,
                "grad_Q mismatch at {i}: metal={} cpu={}",
                m_gq_v[i],
                c_gq_v[i]
            );
        }
        for i in 0..m_gk_v.len() {
            assert!(
                (m_gk_v[i] - c_gk_v[i]).abs() < 1e-3,
                "grad_K mismatch at {i}: metal={} cpu={}",
                m_gk_v[i],
                c_gk_v[i]
            );
        }
        for i in 0..m_gv_v.len() {
            assert!(
                (m_gv_v[i] - c_gv_v[i]).abs() < 1e-3,
                "grad_V mismatch at {i}: metal={} cpu={}",
                m_gv_v[i],
                c_gv_v[i]
            );
        }
    }

    #[test]
    fn metal_causal_attention_backward_mha_vs_cpu() {
        // Test with n_heads == n_kv_heads (standard MHA, no GQA)
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let seq_len = 8;
        let n_heads = 4;
        let n_kv_heads = 4;
        let head_dim = 8;
        let total_dim = n_heads * head_dim;

        let q_data: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i * 7 + 3) % 13) as f32 / 13.0 - 0.5)
            .collect();
        let k_data: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i * 11 + 5) % 17) as f32 / 17.0 - 0.5)
            .collect();
        let v_data: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i * 13 + 7) % 19) as f32 / 19.0 - 0.5)
            .collect();
        let go_data: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i * 3 + 11) % 23) as f32 / 23.0 - 0.5)
            .collect();

        let m_go = metal.upload(&go_data);
        let m_q = metal.upload(&q_data);
        let m_k = metal.upload(&k_data);
        let m_v = metal.upload(&v_data);
        let (m_gq, m_gk, m_gv) = metal.causal_attention_backward(
            &m_go, &m_q, &m_k, &m_v, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let m_gq_v = metal.download(&m_gq);
        let m_gk_v = metal.download(&m_gk);
        let m_gv_v = metal.download(&m_gv);

        let c_go = cpu.upload(&go_data);
        let c_q = cpu.upload(&q_data);
        let c_k = cpu.upload(&k_data);
        let c_v = cpu.upload(&v_data);
        let (c_gq, c_gk, c_gv) = cpu.causal_attention_backward(
            &c_go, &c_q, &c_k, &c_v, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let c_gq_v = cpu.download(&c_gq);
        let c_gk_v = cpu.download(&c_gk);
        let c_gv_v = cpu.download(&c_gv);

        for i in 0..m_gq_v.len() {
            assert!(
                (m_gq_v[i] - c_gq_v[i]).abs() < 1e-3,
                "MHA grad_Q mismatch at {i}: metal={} cpu={}",
                m_gq_v[i],
                c_gq_v[i]
            );
        }
        for i in 0..m_gk_v.len() {
            assert!(
                (m_gk_v[i] - c_gk_v[i]).abs() < 1e-3,
                "MHA grad_K mismatch at {i}: metal={} cpu={}",
                m_gk_v[i],
                c_gk_v[i]
            );
        }
        for i in 0..m_gv_v.len() {
            assert!(
                (m_gv_v[i] - c_gv_v[i]).abs() < 1e-3,
                "MHA grad_V mismatch at {i}: metal={} cpu={}",
                m_gv_v[i],
                c_gv_v[i]
            );
        }
    }

    #[test]
    fn metal_kv_attention_batched_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let q_len = 4;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q_data: Vec<f32> = (0..q_len * total_dim)
            .map(|i| ((i * 7 + 3) % 13) as f32 / 13.0)
            .collect();
        let kv_data: Vec<f32> = (0..q_len * kv_dim)
            .map(|i| ((i * 11 + 5) % 17) as f32 / 17.0)
            .collect();
        let v_data: Vec<f32> = (0..q_len * kv_dim)
            .map(|i| ((i * 13 + 7) % 19) as f32 / 19.0)
            .collect();

        let m_q = metal.upload(&q_data);
        let m_k = metal.upload(&kv_data);
        let m_v = metal.upload(&v_data);
        let m_out = metal.download(
            &metal.kv_attention(&m_q, &m_k, &m_v, 0, q_len, n_heads, n_kv_heads, head_dim),
        );

        let c_q = cpu.upload(&q_data);
        let c_k = cpu.upload(&kv_data);
        let c_v = cpu.upload(&v_data);
        let c_out = cpu
            .download(&cpu.kv_attention(&c_q, &c_k, &c_v, 0, q_len, n_heads, n_kv_heads, head_dim));

        assert_eq!(m_out.len(), c_out.len());
        for i in 0..m_out.len() {
            assert!(
                (m_out[i] - c_out[i]).abs() < 1e-3,
                "batched kv_attention mismatch at {i}: metal={} cpu={}",
                m_out[i],
                c_out[i]
            );
        }
    }

    #[test]
    fn metal_simdgroup_matmul_aligned() {
        // Test the simdgroup matmul path (32-aligned dimensions)
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let m = 64;
        let k = 32;
        let n = 64;
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 3) % 13) as f32 / 13.0 - 0.5)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i * 11 + 5) % 17) as f32 / 17.0 - 0.5)
            .collect();

        let m_a = metal.upload(&a_data);
        let m_b = metal.upload(&b_data);
        let m_c = metal.matmul(&m_a, &m_b, m, k, n);
        let metal_out = metal.download(&m_c);

        let c_a = cpu.upload(&a_data);
        let c_b = cpu.upload(&b_data);
        let c_c = cpu.matmul(&c_a, &c_b, m, k, n);
        let cpu_out = cpu.download(&c_c);

        for i in 0..m * n {
            assert!(
                (metal_out[i] - cpu_out[i]).abs() < 1e-2,
                "simdgroup matmul mismatch at {i}: metal={} cpu={}",
                metal_out[i],
                cpu_out[i]
            );
        }
    }

    #[test]
    fn metal_simdgroup_matmul_training_dims() {
        // Test training-sized dimensions (768x768, 768x3072)
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        for (m, k, n) in [(32, 768, 768), (32, 768, 3072), (32, 3072, 768)] {
            let a_data: Vec<f32> = (0..m * k)
                .map(|i| ((i * 7 + 3) % 37) as f32 / 37.0 - 0.5)
                .collect();
            let b_data: Vec<f32> = (0..k * n)
                .map(|i| ((i * 11 + 5) % 41) as f32 / 41.0 - 0.5)
                .collect();

            let m_a = metal.upload(&a_data);
            let m_b = metal.upload(&b_data);
            let m_c = metal.matmul(&m_a, &m_b, m, k, n);
            let metal_out = metal.download(&m_c);

            let c_a = cpu.upload(&a_data);
            let c_b = cpu.upload(&b_data);
            let c_c = cpu.matmul(&c_a, &c_b, m, k, n);
            let cpu_out = cpu.download(&c_c);

            let max_err = metal_out
                .iter()
                .zip(cpu_out.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_err < 0.1, "matmul {m}x{k}x{n}: max error {max_err}");
        }
    }

    #[test]
    fn metal_matmul_b_transposed_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let m = 64;
        let k = 64;
        let n = 32;
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 3) % 13) as f32 / 13.0 - 0.5)
            .collect();
        let b_data: Vec<f32> = (0..n * k)
            .map(|i| ((i * 11 + 5) % 17) as f32 / 17.0 - 0.5)
            .collect();

        let m_a = metal.upload(&a_data);
        let m_b = metal.upload(&b_data);
        let m_c = metal.matmul_b_transposed(&m_a, &m_b, m, k, n);
        let metal_out = metal.download(&m_c);

        let c_a = cpu.upload(&a_data);
        let c_b = cpu.upload(&b_data);
        let c_c = cpu.matmul_b_transposed(&c_a, &c_b, m, k, n);
        let cpu_out = cpu.download(&c_c);

        for i in 0..m * n {
            assert!(
                (metal_out[i] - cpu_out[i]).abs() < 1e-2,
                "matmul_bt mismatch at {i}: metal={} cpu={}",
                metal_out[i],
                cpu_out[i]
            );
        }
    }

    #[test]
    fn metal_matmul_a_transposed_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let m = 32;
        let k = 64;
        let n = 64;
        let a_data: Vec<f32> = (0..k * m)
            .map(|i| ((i * 7 + 3) % 13) as f32 / 13.0 - 0.5)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i * 11 + 5) % 17) as f32 / 17.0 - 0.5)
            .collect();

        let m_a = metal.upload(&a_data);
        let m_b = metal.upload(&b_data);
        let m_c = metal.matmul_a_transposed(&m_a, &m_b, m, k, n);
        let metal_out = metal.download(&m_c);

        let c_a = cpu.upload(&a_data);
        let c_b = cpu.upload(&b_data);
        let c_c = cpu.matmul_a_transposed(&c_a, &c_b, m, k, n);
        let cpu_out = cpu.download(&c_c);

        for i in 0..m * n {
            assert!(
                (metal_out[i] - cpu_out[i]).abs() < 1e-2,
                "matmul_at mismatch at {i}: metal={} cpu={}",
                metal_out[i],
                cpu_out[i]
            );
        }
    }

    #[test]
    fn metal_matmul_accumulate_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let m = 32;
        let k = 64;
        let n = 32;
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 3) % 13) as f32 / 13.0 - 0.5)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i * 11 + 5) % 17) as f32 / 17.0 - 0.5)
            .collect();
        let c_init: Vec<f32> = (0..m * n)
            .map(|i| ((i * 3 + 1) % 11) as f32 / 11.0)
            .collect();

        let m_a = metal.upload(&a_data);
        let m_b = metal.upload(&b_data);
        let mut m_c = metal.upload(&c_init);
        metal.matmul_accumulate(&m_a, &m_b, &mut m_c, m, k, n);
        let metal_out = metal.download(&m_c);

        let c_a = cpu.upload(&a_data);
        let c_b = cpu.upload(&b_data);
        let mut c_c = cpu.upload(&c_init);
        cpu.matmul_accumulate(&c_a, &c_b, &mut c_c, m, k, n);
        let cpu_out = cpu.download(&c_c);

        for i in 0..m * n {
            assert!(
                (metal_out[i] - cpu_out[i]).abs() < 1e-2,
                "matmul_acc mismatch at {i}: metal={} cpu={}",
                metal_out[i],
                cpu_out[i]
            );
        }
    }

    #[test]
    fn metal_gpu_add_assign() {
        let metal = get_metal_device();
        let mut dst = metal.upload(&[1.0, 2.0, 3.0, 4.0]);
        let src = metal.upload(&[10.0, 20.0, 30.0, 40.0]);
        metal.add_assign(&mut dst, &src);
        let out = metal.download(&dst);
        assert!((out[0] - 11.0).abs() < 1e-5);
        assert!((out[1] - 22.0).abs() < 1e-5);
        assert!((out[2] - 33.0).abs() < 1e-5);
        assert!((out[3] - 44.0).abs() < 1e-5);
    }

    #[test]
    fn metal_gpu_zero_buffer() {
        let metal = get_metal_device();
        let mut buf = metal.upload(&[1.0, 2.0, 3.0, 4.0]);
        metal.zero_buffer(&mut buf);
        let out = metal.download(&buf);
        for v in &out {
            assert!(*v == 0.0, "zero_buffer: got {v}");
        }
    }

    #[test]
    fn metal_gpu_adamw_step() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let param_data = vec![1.0, 2.0, 3.0, 4.0];
        let grad_data = vec![0.1, 0.2, 0.3, 0.4];
        let m_data = vec![0.0; 4];
        let v_data = vec![0.0; 4];
        let lr = 0.001;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let wd = 0.01;
        let step_t = 1;

        let mut m_param = metal.upload(&param_data);
        let m_grad = metal.upload(&grad_data);
        let mut m_m = metal.upload(&m_data);
        let mut m_v = metal.upload(&v_data);
        metal.adamw_step(
            &mut m_param,
            &m_grad,
            &mut m_m,
            &mut m_v,
            lr,
            beta1,
            beta2,
            eps,
            wd,
            step_t,
        );
        let metal_param = metal.download(&m_param);

        let mut c_param = cpu.upload(&param_data);
        let c_grad = cpu.upload(&grad_data);
        let mut c_m = cpu.upload(&m_data);
        let mut c_v = cpu.upload(&v_data);
        cpu.adamw_step(
            &mut c_param,
            &c_grad,
            &mut c_m,
            &mut c_v,
            lr,
            beta1,
            beta2,
            eps,
            wd,
            step_t,
        );
        let cpu_param = cpu.download(&c_param);

        for i in 0..4 {
            assert!(
                (metal_param[i] - cpu_param[i]).abs() < 1e-5,
                "adamw mismatch at {i}: metal={} cpu={}",
                metal_param[i],
                cpu_param[i]
            );
        }
    }

    #[test]
    fn metal_gpu_slice_buffer() {
        let metal = get_metal_device();
        let buf = metal.upload(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let sliced = metal.slice_buffer(&buf, 1, 3);
        let out = metal.download(&sliced);
        assert_eq!(out, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn metal_gpu_write_into() {
        let metal = get_metal_device();
        let mut dst = metal.upload(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        let src = metal.upload(&[7.0, 8.0]);
        metal.write_into(&mut dst, 2, &src);
        let out = metal.download(&dst);
        assert_eq!(out, vec![0.0, 0.0, 7.0, 8.0, 0.0]);
    }

    #[test]
    fn metal_linear_vs_cpu_decode_and_prefill() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();
        // Decode (GEMV), unaligned prefill (padded), aligned prefill, and K not a multiple of 4.
        for &(m, k, n) in &[
            (1, 64, 96),
            (5, 64, 96),
            (37, 64, 96),
            (64, 64, 32),
            (3, 30, 40),
        ] {
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i * 7 + 3) % 13) as f32 / 13.0 - 0.5)
                .collect();
            let w: Vec<f32> = (0..n * k)
                .map(|i| ((i * 11 + 5) % 17) as f32 / 17.0 - 0.5)
                .collect();
            let got = metal.download(&metal.linear(&metal.upload(&x), &metal.upload(&w), m, k, n));
            let want =
                cpu.download(&cpu.matmul_b_transposed(&cpu.upload(&x), &cpu.upload(&w), m, k, n));
            assert_eq!(got.len(), m * n);
            for i in 0..m * n {
                assert!(
                    (got[i] - want[i]).abs() < 1e-3,
                    "({m},{k},{n}) at {i}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn metal_rope_half_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();
        let (seq, heads, hd, start, max_pos) = (3, 2, 8, 4, 16);
        let x: Vec<f32> = (0..seq * heads * hd)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();
        let cos: Vec<f32> = (0..max_pos * hd / 2)
            .map(|i| (i as f32 * 0.1).cos())
            .collect();
        let sin: Vec<f32> = (0..max_pos * hd / 2)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let got = metal.download(&metal.rope_half_cached(
            &metal.upload(&x),
            &metal.upload(&cos),
            &metal.upload(&sin),
            seq,
            heads,
            hd,
            start,
        ));
        let want = cpu.download(&cpu.rope_half_cached(
            &cpu.upload(&x),
            &cpu.upload(&cos),
            &cpu.upload(&sin),
            seq,
            heads,
            hd,
            start,
        ));
        for i in 0..x.len() {
            assert!(
                (got[i] - want[i]).abs() < 1e-5,
                "at {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn metal_bf16_linear_and_embedding_vs_f32() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();
        let bf = |v: f32| (v.to_bits() >> 16) as u16;
        for &(m, k, n) in &[(1, 64, 40), (6, 36, 40), (45, 70, 33), (64, 64, 64)] {
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i * 7 + 3) % 13) as f32 / 13.0 - 0.5)
                .collect();
            let bits: Vec<u16> = (0..n * k)
                .map(|i| bf(((i * 11 + 5) % 17) as f32 / 17.0 - 0.5))
                .collect();
            let w: Vec<f32> = bits
                .iter()
                .map(|&b| f32::from_bits((b as u32) << 16))
                .collect();
            let got = metal.download(&metal.linear(
                &metal.upload(&x),
                &metal.upload_bf16(&bits),
                m,
                k,
                n,
            ));
            let want =
                cpu.download(&cpu.matmul_b_transposed(&cpu.upload(&x), &cpu.upload(&w), m, k, n));
            for i in 0..m * n {
                assert!(
                    (got[i] - want[i]).abs() < 1e-3,
                    "({m},{k},{n}) at {i}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
        let (vocab, dim) = (10, 6);
        let bits: Vec<u16> = (0..vocab * dim).map(|i| bf(i as f32 * 0.25)).collect();
        let ids = metal.upload_u32(&[3, 0, 9]);
        let got = metal.download(&metal.embedding(&metal.upload_bf16(&bits), &ids, 3, dim));
        let want: Vec<f32> = [3, 0, 9]
            .iter()
            .flat_map(|&t| (0..dim).map(move |d| (t * dim + d) as f32 * 0.25))
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn metal_flash_attention_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();
        let (nh, nkv, d) = (4, 2, 64);
        // Decode with several splits, and a prefill chunk after cached tokens.
        for &(cache_start, q_len) in &[(700usize, 1usize), (5, 9), (0, 70), (40, 33)] {
            let total = (cache_start + q_len).next_multiple_of(32);
            let q: Vec<f32> = (0..q_len * nh * d)
                .map(|i| ((i * 13 + 1) % 23) as f32 / 23.0 - 0.5)
                .collect();
            let k: Vec<f32> = (0..total * nkv * d)
                .map(|i| ((i * 7 + 3) % 19) as f32 / 19.0 - 0.5)
                .collect();
            let v: Vec<f32> = (0..total * nkv * d)
                .map(|i| ((i * 5 + 2) % 17) as f32 / 17.0 - 0.5)
                .collect();
            let got = metal.download(&metal.kv_attention(
                &metal.upload(&q),
                &metal.upload(&k),
                &metal.upload(&v),
                cache_start,
                q_len,
                nh,
                nkv,
                d,
            ));
            let want = cpu.download(&cpu.kv_attention(
                &cpu.upload(&q),
                &cpu.upload(&k),
                &cpu.upload(&v),
                cache_start,
                q_len,
                nh,
                nkv,
                d,
            ));
            for i in 0..want.len() {
                assert!(
                    (got[i] - want[i]).abs() < 1e-4,
                    "({cache_start},{q_len}) at {i}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn metal_q4_linear_and_embedding_vs_dequantized() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();
        let bf = |v: f32| (v.to_bits() >> 16) as u16;
        let group = 64;
        for &(m, k, n) in &[(1, 128, 40), (3, 256, 24), (41, 192, 33)] {
            let packed: Vec<u32> = (0..n * k / 8)
                .map(|i| (i as u32).wrapping_mul(2654435761))
                .collect();
            let scales: Vec<u16> = (0..n * k / group)
                .map(|i| bf(0.01 + (i % 7) as f32 * 0.003))
                .collect();
            let biases: Vec<u16> = (0..n * k / group)
                .map(|i| bf(-0.05 + (i % 5) as f32 * 0.01))
                .collect();
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i * 7 + 3) % 13) as f32 / 13.0 - 0.5)
                .collect();
            let wq = metal.upload_q4(&packed, &scales, &biases, group);
            let w = cpu.upload_q4(&packed, &scales, &biases, group);
            assert_eq!(wq.to_vec(), cpu.download(&w), "dequantization");
            let got = metal.download(&metal.linear(&metal.upload(&x), &wq, m, k, n));
            let want = cpu.download(&cpu.matmul_b_transposed(&cpu.upload(&x), &w, m, k, n));
            for i in 0..m * n {
                assert!(
                    (got[i] - want[i]).abs() < 1e-3,
                    "({m},{k},{n}) at {i}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
            if m == 1 {
                let ids = [5u32, 0, 39];
                let got = metal.download(&metal.embedding(&wq, &metal.upload_u32(&ids), 3, k));
                let all = cpu.download(&w);
                let want: Vec<f32> = ids
                    .iter()
                    .flat_map(|&t| all[t as usize * k..(t as usize + 1) * k].to_vec())
                    .collect();
                assert_eq!(got, want);
            }
        }
    }

    #[test]
    fn metal_implicit_convolution_matches_cpu_at_edges_and_vae_widths() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();
        for (input, output, h, w, kernel) in [
            (3, 5, 3, 7, 1),
            (3, 5, 3, 7, 3),
            (2, 1, 1, 1, 3),
            (16, 512, 16, 16, 3),
            (512, 512, 4, 4, 3),
            (128, 3, 17, 19, 3),
        ] {
            let x: Vec<f32> = (0..input * h * w)
                .map(|i| ((i * 17) % 97) as f32 / 41. - 1.)
                .collect();
            let weight: Vec<f32> = (0..output * input * kernel * kernel)
                .map(|i| {
                    (((i * 13) % 89) as f32 / 47. - 1.) / ((input * kernel * kernel) as f32).sqrt()
                })
                .collect();
            let bias: Vec<f32> = (0..output).map(|i| (i % 7) as f32 / 7. - 0.5).collect();
            let actual = metal.download(&metal.conv2d_nchw(
                &metal.upload(&x),
                &metal.upload(&weight),
                &metal.upload(&bias),
                input,
                output,
                h,
                w,
                kernel,
            ));
            let expected = cpu.download(&cpu.conv2d_nchw(
                &cpu.upload(&x),
                &cpu.upload(&weight),
                &cpu.upload(&bias),
                input,
                output,
                h,
                w,
                kernel,
            ));
            for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    a.is_finite() && (a - b).abs() < 3e-5 * b.abs().max(1.),
                    "{input}->{output} {h}x{w} kernel {kernel}, index {i}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn metal_vae_normalization_and_upsampling_match_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();
        for (channels, h, w, groups) in [(12, 3, 5, 3), (512, 16, 16, 32), (128, 128, 128, 32)] {
            for offset in [0., 10000.] {
                let data: Vec<f32> = (0..channels * h * w)
                    .map(|i| ((i * 17) % 97) as f32 / 13. - 3. + offset)
                    .collect();
                let weight: Vec<f32> = (0..channels).map(|i| 0.5 + (i % 11) as f32 / 11.).collect();
                let bias: Vec<f32> = (0..channels).map(|i| (i % 7) as f32 / 7. - 0.5).collect();
                let expected = cpu.download(&cpu.group_norm_affine(
                    &cpu.upload(&data),
                    &cpu.upload(&weight),
                    &cpu.upload(&bias),
                    channels,
                    h * w,
                    groups,
                    1e-6,
                ));
                let actual = metal.download(&metal.group_norm_affine(
                    &metal.upload(&data),
                    &metal.upload(&weight),
                    &metal.upload(&bias),
                    channels,
                    h * w,
                    groups,
                    1e-6,
                ));
                for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        a.is_finite() && (a - b).abs() < 2e-5 * b.abs().max(1.),
                        "{channels}x{h}x{w} offset {offset}, index {i}: {a} vs {b}"
                    );
                }
            }
            let data: Vec<f32> = (0..channels * h * w).map(|i| (i % 97) as f32).collect();
            assert_eq!(
                metal.download(&metal.upsample_nearest_2x(&metal.upload(&data), channels, h, w)),
                cpu.download(&cpu.upsample_nearest_2x(&cpu.upload(&data), channels, h, w))
            );
        }
    }

    #[test]
    fn metal_channel_modulation_matches_cpu_without_expanded_scales() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();
        for (rows, dim, offset) in [(3, 7, 2), (5, 128, 0), (2, 3840, 3840)] {
            let data: Vec<f32> = (0..rows * dim)
                .map(|i| ((i * 17) % 97) as f32 / 13. - 3.)
                .collect();
            let mut scale: Vec<f32> = (0..offset + dim + 3)
                .map(|i| ((i * 11) % 31) as f32 / 7. - 2.)
                .collect();
            scale[offset] = -1.;
            scale[offset + 1] = 1000.;
            let actual = metal.download(&metal.modulate_channels(
                &metal.upload(&data),
                &metal.upload(&scale),
                offset,
                rows,
                dim,
            ));
            let expected = cpu.download(&cpu.modulate_channels(
                &cpu.upload(&data),
                &cpu.upload(&scale),
                offset,
                rows,
                dim,
            ));
            assert_eq!(actual, expected, "{rows}x{dim} offset {offset}");
        }
    }

    #[test]
    fn metal_modulated_rms_matches_cpu_at_dit_widths_and_large_gates() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();
        for (dim, rows) in [(24, 3), (128, 5), (3840, 2)] {
            let data: Vec<f32> = (0..rows * dim)
                .map(|i| ((i * 17) % 97) as f32 / 13. - 3.)
                .collect();
            let weight: Vec<f32> = (0..dim).map(|i| 0.5 + (i % 13) as f32 / 13.).collect();
            let residual: Vec<f32> = (0..rows * dim)
                .map(|i| ((i * 11) % 31) as f32 / 7. - 2.)
                .collect();
            let mut modulation: Vec<f32> = (0..4 * dim)
                .map(|i| ((i * 13) % 27) as f32 / 9. - 1.5)
                .collect();
            modulation[dim] = -1000.;
            modulation[dim + 1] = 1000.;
            let (x, w, m, r) = (
                metal.upload(&data),
                metal.upload(&weight),
                metal.upload(&modulation),
                metal.upload(&residual),
            );
            let (cx, cw, cm, cr) = (
                cpu.upload(&data),
                cpu.upload(&weight),
                cpu.upload(&modulation),
                cpu.upload(&residual),
            );
            let actual = [
                metal.download(&metal.rms_norm_scale(&x, &w, &m, 0, rows, dim, 1e-6)),
                metal.download(&metal.rms_norm_gate_residual(&x, &w, &m, &r, dim, rows, dim, 1e-6)),
            ];
            let expected = [
                cpu.download(&cpu.rms_norm_scale(&cx, &cw, &cm, 0, rows, dim, 1e-6)),
                cpu.download(&cpu.rms_norm_gate_residual(&cx, &cw, &cm, &cr, dim, rows, dim, 1e-6)),
            ];
            for (a, b) in actual.iter().flatten().zip(expected.iter().flatten()) {
                assert!(
                    a.is_finite() && (a - b).abs() <= 2e-5 * b.abs().max(1.),
                    "dim {dim}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn metal_softmax_large_logits_at_256_columns_is_stable() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();
        let (rows, cols) = (32, 256);
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7) % 251) as f32 - 125.)
            .collect();
        let expected = cpu.download(&cpu.softmax(&cpu.upload(&values), rows, cols));
        for _ in 0..4 {
            let actual = metal.download(&metal.softmax(&metal.upload(&values), rows, cols));
            for (index, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    a.is_finite() && (a - b).abs() < 1e-6,
                    "softmax index {index}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn metal_unary_silu_matches_cpu_and_previous_swiglu_path() {
        let Some(metal) = MetalDevice::new() else {
            return;
        };
        let cpu = crate::CpuDevice::new();
        let data = [-1000., -100., -20., -5., -1., 0., 1., 5., 20., 100., 1000.];
        let x = metal.upload(&data);
        let actual = metal.download(&metal.silu_buf(&x, data.len()));
        let previous = metal.download(&metal.swiglu_fused_buf(
            &x,
            &metal.upload(&vec![1.; data.len()]),
            data.len(),
        ));
        let expected = cpu.download(&cpu.silu_buf(&cpu.upload(&data), data.len()));
        assert_eq!(actual, previous);
        for (a, b) in actual.iter().zip(expected) {
            assert!(a.is_finite() && (a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn metal_swiglu_large_negative_gates_remain_finite() {
        let metal = get_metal_device();
        let gates = [-1000., -100., -90., -80., -20., 0., 20., 100., 1000.];
        let up = vec![1f32; gates.len()];
        let actual = metal.download(&metal.swiglu_fused_buf(
            &metal.upload(&gates),
            &metal.upload(&up),
            gates.len(),
        ));
        for (&gate, &value) in gates.iter().zip(&actual) {
            let expected = (gate as f64 / (1. + (-(gate as f64)).exp())) as f32;
            assert!(value.is_finite(), "nonfinite SiLU at {gate}: {value}");
            assert!(
                (value - expected).abs() < 1e-5,
                "SiLU at {gate}: {value} vs {expected}"
            );
        }
    }

    #[test]
    fn metal_fused_attention_prep_and_swiglu_vs_default() {
        let metal = get_metal_device();
        let (seq, nh, nkv, hd, pos, max) = (3, 4, 2, 64, 2, 8);
        let row = (nh + 2 * nkv) * hd;
        let qkv: Vec<f32> = (0..seq * row)
            .map(|i| ((i * 13 + 1) % 23) as f32 / 23.0 - 0.5)
            .collect();
        let qn: Vec<f32> = (0..hd).map(|i| 1.0 + i as f32 * 0.01).collect();
        let kn: Vec<f32> = (0..hd).map(|i| 0.5 + i as f32 * 0.02).collect();
        let cos: Vec<f32> = (0..max * hd / 2).map(|i| (i as f32 * 0.1).cos()).collect();
        let sin: Vec<f32> = (0..max * hd / 2).map(|i| (i as f32 * 0.1).sin()).collect();
        let up = |v: &[f32]| metal.upload(v);
        let run = |fused: bool| {
            let (mut kc, mut vc) = (metal.alloc(max * nkv * hd), metal.alloc(max * nkv * hd));
            let (a, b, c, d, e) = (up(&qkv), up(&qn), up(&kn), up(&cos), up(&sin));
            let q = if fused {
                metal.attention_prep(
                    &a,
                    Some(&b),
                    Some(&c),
                    &d,
                    &e,
                    &mut kc,
                    &mut vc,
                    seq,
                    (nh, nkv, hd),
                    pos,
                    1e-6,
                )
            } else {
                crate::device::attention_prep_default(
                    &metal,
                    &a,
                    Some(&b),
                    Some(&c),
                    &d,
                    &e,
                    &mut kc,
                    &mut vc,
                    seq,
                    (nh, nkv, hd),
                    pos,
                    1e-6,
                )
            };
            (metal.download(&q), metal.download(&kc), metal.download(&vc))
        };
        let (got, want) = (run(true), run(false));
        for (g, w) in [(&got.0, &want.0), (&got.1, &want.1), (&got.2, &want.2)] {
            for i in 0..w.len() {
                assert!((g[i] - w[i]).abs() < 1e-5, "at {i}: {} vs {}", g[i], w[i]);
            }
        }
        let gu: Vec<f32> = (0..2 * 2 * 5).map(|i| i as f32 * 0.3 - 2.0).collect();
        let got = metal.download(&metal.swiglu_split(&up(&gu), 2, 5));
        let silu = |g: f32| g / (1.0 + (-g).exp());
        for r in 0..2 {
            for i in 0..5 {
                let want = silu(gu[r * 10 + i]) * gu[r * 10 + 5 + i];
                assert!((got[r * 5 + i] - want).abs() < 1e-5);
            }
        }
    }
}
