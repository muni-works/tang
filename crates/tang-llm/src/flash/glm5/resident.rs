//! The device-resident decode step (`TANG_GLM5_RESIDENT`, default on when every stage is
//! resident; `=0` forces the host-staged path).
//!
//! One decode token, slot 0, one device, every stage on ONE stream (the fused mHC's):
//!
//! - embed: the host gathers the token's row and `MhcFused::begin` uploads the 4 streams
//!   (the step's only H2D);
//! - every half block: `pre_dev` writes the normed `x` straight into the consumer's input
//!   buffer (KDA `xb`, the MLA scratch `x`, or [`Resident::x`] for the FFN); the consumer
//!   runs async on the same stream (KDA: its captured decode graph; DSA:
//!   `MlaProj::window_dev`, `kv_len`/`pos0` passed as per-call kernel args; FFN:
//!   `ffn_async`); `post_dev` hands the producer's output pointer to the next pre;
//! - head: `end_dev` (last post + stream mean) → `rs_norm` (output_norm, f64 sum like the
//!   host) → k5_q8 + k5_gemv8 (the LM head) → `rs_argmax` (first max, like `argmax`) →
//!   ONE sync and a 4-byte read: the next token.
//!
//! So per token: 1 H2D, 1 D2H, 1 host sync (was ~182 / 226 / ~226). The DSA host mirrors
//! are only resized; [`DsaCache::stale_from`] marks them and the host paths refresh them
//! from the device before any host use.

use super::*;
use tang_moe::gpu::{DevBuf, Module};

const SRC: &str = r#"
// output_norm: one block (1024 threads), x[n] -> y[n] = x * inv * w; the sum of squares
// in f64 (the host's inv_rms), mean in f64, cast to f32, + eps, 1/sqrt in f32.
extern "C" __global__ void rs_norm(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ double red[1024];
    double s = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += (double)x[i] * (double)x[i];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf((float)(red[0] / (double)n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv * w[i];
}

// argmax over v[n]: the FIRST maximum (strict >, like the host's argmax). One block.
// out[0] = the token; `inc` (the whole-step graph) also advances the position out[1].
extern "C" __global__ void rs_argmax(const float* v, int n, unsigned* out, int inc) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    float best = v[0];
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float x = v[i];
        if (x > best || (x == best && i < idx)) { best = x; idx = i; }
    }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = idx;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) {
            const float a = bv[threadIdx.x], b = bv[threadIdx.x + o];
            const int ia = bi[threadIdx.x], ib = bi[threadIdx.x + o];
            if (b > a || (b == a && ib < ia)) { bv[threadIdx.x] = b; bi[threadIdx.x] = ib; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[0] = (unsigned)bi[0];
        if (inc) out[1] += 1u;
    }
}

// x[i] += y[i] (the MTP draft cell's residual adds)
extern "C" __global__ void rs_add(float* x, const float* y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) x[i] += y[i];
}

// ---- the device embed gather (TANG_GLM5_STEP_GRAPH=2) ----
// Row tok[0] of the GGUF token_embd table (its own quant type), dequantised exactly as the
// host's gguf::dequantize (explicit _rn ops: no FMA contraction), broadcast to the hc
// streams: r[c*n + j]. Types: 0 F32, 1 F16, 2 BF16, 8 Q8_0, 12 Q4_K, 13 Q5_K, 14 Q6_K.
__device__ __forceinline__ unsigned rs_u16(const unsigned char* p) { return (unsigned)p[0] | ((unsigned)p[1] << 8); }
__device__ __forceinline__ float rs_h2f(unsigned h) {
    const unsigned s = (h >> 15) & 1u, e = (h >> 10) & 31u, m = h & 1023u;
    if (e == 0) {
        const float v = __fmul_rn((float)m, 5.9604644775390625e-8f);  // m * 2^-24, exact
        return s ? -v : v;
    }
    const unsigned bits = (s << 31) | (e == 31 ? (0xffu << 23) : ((e + 112u) << 23)) | (m << 13);
    return __uint_as_float(bits);
}
__device__ __forceinline__ void rs_smk4(int j, const unsigned char* q, int& sc, int& mn) {
    if (j < 4) { sc = q[j] & 63; mn = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
__device__ float rs_deq(unsigned ty, const unsigned char* row, int j) {
    if (ty == 0) return __uint_as_float((unsigned)row[4*j] | ((unsigned)row[4*j+1] << 8) | ((unsigned)row[4*j+2] << 16) | ((unsigned)row[4*j+3] << 24));
    if (ty == 1) return rs_h2f(rs_u16(row + 2 * j));
    if (ty == 2) return __uint_as_float(rs_u16(row + 2 * j) << 16);
    if (ty == 8) {
        const unsigned char* b = row + (j / 32) * 34;
        return __fmul_rn(rs_h2f(rs_u16(b)), (float)(signed char)b[2 + j % 32]);
    }
    if (ty == 12 || ty == 13) {
        const unsigned char* b = row + (j / 256) * (ty == 12 ? 144 : 176);
        const int e = j % 256, g = e / 64, w = e % 64, hi = w / 32, l = w % 32;
        int sc, mn;
        rs_smk4(2 * g + hi, b + 4, sc, mn);
        const unsigned char* qs = b + (ty == 12 ? 16 : 48);
        int q = (qs[g * 32 + l] >> (4 * hi)) & 0xF;
        if (ty == 13 && (b[16 + l] & (1u << (2 * g + hi)))) q += 16;
        const float d1 = __fmul_rn(rs_h2f(rs_u16(b)), (float)sc);
        const float m1 = __fmul_rn(rs_h2f(rs_u16(b + 2)), (float)mn);
        return __fsub_rn(__fmul_rn(d1, (float)q), m1);
    }
    if (ty == 14) {
        const unsigned char* b = row + (j / 256) * 210;
        const int e = j % 256, n = e / 128, r = e % 128, qd = r / 32, l = r % 32;
        const unsigned char* ql = b + n * 64;
        const unsigned char* qh = b + 128 + n * 32;
        const int lo = (ql[l + (qd & 1) * 32] >> ((qd & 2) ? 4 : 0)) & 0xF;
        const int h2 = (qh[l] >> (2 * qd)) & 3;
        const int sc = (signed char)b[192 + n * 8 + (l / 16) + 2 * qd];
        return __fmul_rn(__fmul_rn(rs_h2f(rs_u16(b + 208)), (float)sc), (float)((lo | (h2 << 4)) - 32));
    }
    return __uint_as_float(0x7fc00000u);
}
extern "C" __global__ void rs_embed(const unsigned char* tab, unsigned long long row_bytes, unsigned ty,
    const unsigned* tok, float* r, int n, int hc) {
    const unsigned char* row = tab + (unsigned long long)tok[0] * row_bytes;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        const float v = rs_deq(ty, row, j);
        for (int c = 0; c < hc; c++) r[(size_t)c * n + j] = v;
    }
}
"#;

/// `rs_embed`'s type id for a GGUF type it decodes (`None`: the host gathers).
fn embed_ty(t: crate::gguf::GgmlType) -> Option<u32> {
    use crate::gguf::GgmlType::*;
    Some(match t {
        F32 => 0,
        F16 => 1,
        Bf16 => 2,
        Q8_0 => 8,
        Q4K => 12,
        Q5K => 13,
        Q6K => 14,
        _ => return None,
    })
}

/// The resident step's own device state: the FFN's x / y, output_norm's weight, the token,
/// and (when there is no Q8_0 [`HeadDev`], e.g. the f32 tiny fixture) its own LM head on
/// the generic `k5_gemv`: weights, the normed row, the logits.
pub(super) struct Resident {
    m: Module,
    x: DevBuf,
    y: DevBuf,
    norm_w: DevBuf,
    /// `[token, position]` u32: `rs_argmax` writes the token; on the whole-step graph it
    /// also advances the position, which every DSA kernel of the next replay reads.
    tok: DevBuf,
    head: Option<(
        super::super::moe::gpu::DevW,
        super::super::moe::gpu::Kernels5,
        DevBuf,
        DevBuf,
    )>,
    /// One eager resident token has run since the last host window (every buffer the
    /// step touches is allocated): the step graph may be captured.
    warm: bool,
    /// `TANG_GLM5_STEP_GRAPH=1`: the captured segments (see [`Glm5Engine::step_resident`]).
    pub(super) graphs: Option<Vec<tang_moe::gpu::Graph>>,
    /// The whole-step graphs (default; `TANG_GLM5_STEP_GRAPH=2`), `[dense, sparse]` (the indexer
    /// branch; picked on the host from `pos0`, no transfer).
    pub(super) whole: [Option<tang_moe::gpu::Graph>; 2],
    /// The DSA cache rows the whole graphs may reach (their baked cache bases stay valid
    /// while `pos0 + 1 <= whole_cap`).
    whole_cap: usize,
    /// Replays per whole graph `[dense, sparse]` (tests).
    pub(super) whole_replays: [usize; 2],
    /// What `tok` holds on the device (`(token, position)`), when known: a replay uploads
    /// the pair only when it is out of step (after an eager token or a host window).
    dev_tp: Option<(u32, usize)>,
    /// The GGUF `token_embd` table on the device and its row bytes / `rs_embed` type, for
    /// the whole graph's embed gather (`None` until first use, or the type is unsupported).
    embd: Option<Option<(DevBuf, u64, u32)>>,
    /// The resident verify window's buffers ([`Glm5Engine::window_resident`]).
    win: Option<WinBufs>,
    /// Resident windows run / rewinds applied (tests, the MTP report).
    pub(super) win_count: [usize; 2],
    /// The window lengths that have run one eager window (buffers allocated), and the
    /// captured window segments per length (split at the DSA cores, which stay eager).
    wwarm: Vec<usize>,
    wgraphs: Vec<(usize, Vec<tang_moe::gpu::Graph>)>,
}

/// [`Glm5Engine::window_resident`]'s device buffers, sized for `cap` rows: the KDA rows'
/// input / output (each KDA layer runs its t=1 decode once per row), the FFN rows, the
/// normed head rows, the per-row argmax ids, and per KDA layer `cap - 1` state checkpoints
/// (conv history + recurrent state after each row but the last) for the rewind.
struct WinBufs {
    cap: usize,
    kx: DevBuf,
    ky: DevBuf,
    fx: DevBuf,
    fy: DevBuf,
    hn: DevBuf,
    ids: DevBuf,
    /// The window's tokens when they come from the device ([`Glm5Engine::window_resident_dev`]).
    wtok: DevBuf,
    /// Per layer (`None` for DSA): `cap - 1` checkpoints of `ck_bytes[l]` bytes each.
    ck: Vec<Option<DevBuf>>,
    ck_bytes: Vec<usize>,
}

impl Resident {
    fn new(
        gpu: &tang_moe::gpu::Gpu,
        g: &crate::gguf::Gguf,
        norm_w: &[f32],
        own_head: bool,
    ) -> Result<Self> {
        let ge = |e: tang_moe::gpu::Error| anyhow::anyhow!("glm5 resident: {e}");
        let n = norm_w.len();
        let head = if own_head {
            let info = match g.get("output.weight") {
                Some(t) => t,
                None => g.info("token_embd.weight")?,
            };
            let (rows, cols) = (info.dims[1] as usize, info.dims[0] as usize);
            ensure!(cols == n, "lm head: {cols} cols, hidden {n}");
            let dw = super::super::moe::gpu::upload(
                &super::super::moe::W::Gguf(g, info),
                1,
                rows,
                cols,
            )?;
            let k5 = super::super::moe::gpu::Kernels5::new(gpu)?;
            Some((
                dw,
                k5,
                DevBuf::zeroed_elems::<f32>(n).map_err(ge)?,
                DevBuf::zeroed_elems::<f32>(rows).map_err(ge)?,
            ))
        } else {
            None
        };
        Ok(Self {
            m: gpu.module(SRC).map_err(ge)?,
            x: DevBuf::zeroed_elems::<f32>(n).map_err(ge)?,
            y: DevBuf::zeroed_elems::<f32>(n).map_err(ge)?,
            norm_w: DevBuf::from_slice(norm_w).map_err(ge)?,
            tok: DevBuf::zeroed_elems::<u32>(2).map_err(ge)?,
            head,
            warm: false,
            graphs: None,
            whole: [None, None],
            whole_cap: 0,
            whole_replays: [0, 0],
            dev_tp: None,
            embd: None,
            win: None,
            win_count: [0, 0],
            wwarm: Vec::new(),
            wgraphs: Vec::new(),
        })
    }
}

/// `TANG_GLM5_RESIDENT=0`: force the host-staged decode.
fn resident_env() -> bool {
    !std::env::var("TANG_GLM5_RESIDENT").is_ok_and(|v| v == "0")
}

impl Glm5Engine {
    /// Can slot `slot`'s next decode token take the resident step? Every stage must be
    /// resident on one device, no shadow / dump, slot 0 of a single-slot engine, and every
    /// KDA layer's device state live.
    fn resident_ready(&self, slot: usize) -> bool {
        if !resident_env() || slot != 0 || self.n_slots != 1 || self.dump.is_some() {
            return false;
        }
        if self.kda_dbg.is_some() || self.moe_dbg.is_some() || self.mla_dbg.is_some() {
            return false;
        }
        let Some(g) = self.gpu.as_ref() else {
            return false;
        };
        if g.devs.len() != 1 || !g.devs[0].1.fast || g.mhc_f.is_none() {
            return false;
        }
        let hp = &self.m.hp;
        (0..hp.n_layer).all(|l| {
            g.mhc_g[l].is_some()
                && g.has_ffn(l)
                && if hp.is_kda[l] {
                    g.kda_l[l].as_ref().is_some_and(|k| k.resident_ok()) && g.kda_dev_fresh[slot][l]
                } else {
                    g.mla_k[l].is_some() && g.mla_p[l].is_some()
                }
        })
    }

    /// Before a host path touches DSA layer `l`'s mirrors: pull the rows the resident step
    /// left on the device only.
    pub(super) fn refresh_stale_dsa(&mut self, slot: usize, l: usize) -> Result<()> {
        let Some(from) = self.dsa[slot][l].stale_from else {
            return Ok(());
        };
        let Some(g) = self.gpu.as_ref() else {
            return Ok(());
        };
        let Some(cu) = g.mla_k[l].as_ref() else {
            return Ok(());
        };
        g.devs[g.dev_of[l]]
            .0
            .bind()
            .map_err(|e| anyhow::anyhow!("bind: {e}"))?;
        cu.set_seq(slot as u64);
        let c = &mut self.dsa[slot][l];
        let rows = c.valid.len();
        if rows > from {
            cu.refresh_mirrors(
                from,
                rows - from,
                &mut c.lat16,
                &mut c.idx_k,
                &mut c.idx_g,
                &mut c.valid,
            )
            .map_err(|e| anyhow::anyhow!("glm5 dsa mirror refresh: {e}"))?;
        }
        c.stale_from = None;
        Ok(())
    }

    /// One greedy decode token on the resident step, or `None` when it doesn't apply
    /// (the caller runs the host path).
    ///
    /// The step graph (`TANG_GLM5_STEP_GRAPH=2` whole step, the default; `1` segments; `0` = eager):
    /// after one eager token (every buffer allocated), with `=1` the step
    /// is captured as `n_dsa + 1` graph segments split at the DSA attention cores, which
    /// stay eager between the segment launches (their `pos0`/`kv_len`/cache-row pointers
    /// move every token). Any host window drops the segments ([`Self::drop_step_graph`]):
    /// a prefill can reallocate the scratch they point at.
    pub(super) fn step_resident(&mut self, slot: usize, tok: u32) -> Result<Option<u32>> {
        if !self.resident_ready(slot) {
            return Ok(None);
        }
        let hp = self.m.hp.clone();
        let (n, hc) = (hp.n_embd, hp.hc);
        ensure!((tok as usize) < hp.n_vocab, "token {tok} out of range");
        let pos0 = self.slot_tokens(slot).len();
        use super::super::trace::span;
        let tr_root = super::super::trace::window_begin(1, pos0);
        let tr = span("embed");
        // the DSA weights (Arc'd; the window_dev path reads only scalars + i_ape)
        let mut mla_w: Vec<Option<std::sync::Arc<MlaWeights>>> = vec![None; hp.n_layer];
        for l in 0..hp.n_layer {
            if !hp.is_kda[l] {
                mla_w[l] = Some(self.mla_weights(l)?);
            }
        }
        let norm_w = if self.gpu.as_ref().is_some_and(|g| g.resident.is_none()) {
            Some(self.m.vec("output_norm.weight")?)
        } else {
            None
        };
        let embd = self.m.g.info("token_embd.weight")?;
        let gguf = &self.m.g;
        let host_embed = || -> Result<Vec<f32>> {
            let e = gguf.rows(embd, tok as usize, 1)?;
            let mut r = vec![0f32; hc * n];
            for c in 0..hc {
                r[c * n..(c + 1) * n].copy_from_slice(&e);
            }
            Ok(r)
        };
        // `TANG_GLM5_STEP_GRAPH`: unset / `2` = whole step (default), `1` = segments, `0` = eager
        let mode = std::env::var("TANG_GLM5_STEP_GRAPH").unwrap_or_default();
        let whole_on = mode.is_empty() || mode == "2";
        let graph_on = mode == "1";
        // the indexer branch of this token (every DSA layer shares topk / kpool)
        let dsa_l: Vec<usize> = (0..hp.n_layer).filter(|&l| !hp.is_kda[l]).collect();
        let (topk, kpool) = dsa_l.first().map_or((usize::MAX, 1), |&l| {
            let w = mla_w[l].as_ref().unwrap();
            (w.topk, w.kpool)
        });
        ensure!(
            dsa_l.iter().all(|&l| mla_w[l]
                .as_ref()
                .is_some_and(|w| w.topk == topk && w.kpool == kpool)),
            "resident step: DSA layers disagree on topk / kpool"
        );
        let sparse = !super::super::mla::indexer_is_dense(pos0 + 1, topk);
        let g = self.gpu.as_mut().context("resident step without a GPU")?;
        g.devs[0]
            .0
            .bind()
            .map_err(|e| anyhow::anyhow!("bind: {e}"))?;
        if let Some(w) = norm_w {
            let own = g.head_dev.is_none();
            g.resident = Some(Resident::new(&g.devs[0].0, &self.m.g, &w, own)?);
        }
        let ge = |e: tang_moe::gpu::Error| anyhow::anyhow!("glm5 resident: {e}");
        let mut f = g.mhc_f.take().context("fused mHC: no instance")?;
        // a non-owning alias of the step stream (the fused mHC's): the launches below
        // borrow `f` mutably while they enqueue on its stream
        let s = std::mem::ManuallyDrop::new(tang_moe::gpu::Stream(f.stream().0));
        let mut res = g.resident.take().unwrap();
        let items = plan(&hp);
        // debug: TANG_GLM5_STEP_GRAPH_EAGER=kda,ffn also runs those items eagerly between
        // the segments (bisecting a replay mismatch)
        let eager_kinds = std::env::var("TANG_GLM5_STEP_GRAPH_EAGER").unwrap_or_default();
        let boundary = |it: &Item| match it {
            Item::Dsa(_) => true,
            Item::Kda(_) => eager_kinds.contains("kda"),
            Item::Ffn(_) => eager_kinds.contains("ffn"),
            _ => false,
        };
        let dsa = &mut self.dsa[slot];
        let out = (|| -> Result<u32> {
            if whole_on && res.warm {
                if let Some(t) = step_whole(
                    &mut f,
                    g,
                    &mut res,
                    dsa,
                    &mla_w,
                    &hp,
                    &items,
                    &dsa_l,
                    pos0,
                    tok,
                    sparse,
                    &s,
                    gguf,
                    embd,
                    &host_embed,
                )? {
                    return Ok(t);
                }
                // past the graphs' cache capacity: this token runs eagerly (and regrows)
            }
            res.dev_tp = None;
            if graph_on && res.warm && res.graphs.is_none() {
                // capture pass: records only, so the mHC state machine is walked once with
                // the DSA cores skipped; the replay below runs everything
                let _tr = span("capture");
                f.set_window(1)?;
                let mut graphs = Vec::new();
                let mut rest: &[Item] = &items;
                loop {
                    let cut = rest.iter().position(&boundary).unwrap_or(rest.len());
                    let seg = &rest[..cut];
                    let gr = tang_moe::gpu::Graph::capture(&s, |cs| {
                        for it in seg {
                            exec(
                                *it,
                                &mut f,
                                g,
                                &res,
                                dsa,
                                &mla_w,
                                &hp,
                                pos0,
                                cs,
                                Pass::Capture,
                            )
                            .map_err(|e| tang_moe::gpu::Error(format!("{e:#}")))?;
                        }
                        Ok(())
                    })
                    .map_err(|e| anyhow::anyhow!("step graph capture: {e}"))?;
                    graphs.push(gr);
                    if cut == rest.len() {
                        break;
                    }
                    // the boundary item's host side (its mHC post) in capture order
                    exec(
                        rest[cut],
                        &mut f,
                        g,
                        &res,
                        dsa,
                        &mla_w,
                        &hp,
                        pos0,
                        &s,
                        Pass::HostOnly,
                    )?;
                    rest = &rest[cut + 1..];
                }
                res.graphs = Some(graphs);
                eprintln!(
                    "glm5: resident step graph captured ({} segments)",
                    res.graphs.as_ref().unwrap().len()
                );
            }
            f.begin_async(&host_embed()?)?;
            drop(tr);
            if let Some(graphs) = res.graphs.as_ref().filter(|_| graph_on) {
                let mut gi = graphs.iter();
                gi.next().unwrap().launch(&s).map_err(ge)?;
                for it in items.iter().filter(|it| boundary(it)) {
                    exec(
                        *it,
                        &mut f,
                        g,
                        &res,
                        dsa,
                        &mla_w,
                        &hp,
                        pos0,
                        &s,
                        Pass::Replay,
                    )?;
                    gi.next()
                        .context("step graph: segment count")?
                        .launch(&s)
                        .map_err(ge)?;
                }
                // the replay skipped the host side of the mHC state machine
                f.set_window(0)?;
            } else {
                for it in &items {
                    exec(
                        *it,
                        &mut f,
                        g,
                        &res,
                        dsa,
                        &mla_w,
                        &hp,
                        pos0,
                        &s,
                        Pass::Eager,
                    )?;
                }
                res.warm = true;
            }
            if graph_on && std::env::var("TANG_GLM5_STEP_GRAPH_RECAPTURE").is_ok_and(|v| v == "1") {
                res.graphs = None; // debug: a fresh capture every token
            }
            let tr = span("head");
            let _tr = span("sync+read");
            s.sync().map_err(ge)?;
            let t = res.tok.read::<u32>(0, 1).map_err(ge)?[0];
            drop(tr);
            Ok(t)
        })();
        g.mhc_f = Some(f);
        g.resident = Some(res);
        let next = out?;
        self.slot_tokens_mut(slot).push(tok);
        super::super::trace::window_end(tr_root, 1, pos0);
        Ok(Some(next))
    }

    /// The MTP verify window on the resident path: `toks` (t rows, `t <= ` the fused mHC's
    /// window cap) run as ONE pass over the layers, every weight read once per stage except
    /// the KDA layers, whose t=1 decode runs once per row (bit-identical to the decode step,
    /// as the host window's WIN_STEP path is). DSA attends the t rows causally over the cache
    /// (`MlaProj::window_dev` at `t`), the FFN routes all t rows in one `ffn_async`, the LM
    /// head + argmax run per row (the t=1 head numerics). Each KDA layer's state after every
    /// row but the last is checkpointed on the device for [`Self::rewind_resident`].
    /// Returns the per-row greedy ids and the normed h rows `[t][n]` (also `last_h`), or
    /// `None` when the resident path does not apply (the caller runs [`Self::window`]).
    /// The tokens are appended to slot 0.
    pub(crate) fn window_resident(&mut self, toks: &[u32]) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
        Ok(self
            .window_resident_src(toks, None)?
            .map(|(_, ids, hn)| (ids, hn)))
    }

    /// [`Self::window_resident`] whose rows after `toks` come from the device: `dev =
    /// (ids ptr, k, event)` appends k rows whose tokens are the u32s at `ids` (the MTP draft
    /// cells' output, ready once `event` fires). The embed is gathered on the device, so
    /// the host does not wait for the drafts. Returns (all t tokens, ids, hn).
    pub(crate) fn window_resident_dev(
        &mut self,
        cur: u32,
        ids: u64,
        k: usize,
        ev: &tang_moe::gpu::Event,
    ) -> Result<Option<(Vec<u32>, Vec<u32>, Vec<f32>)>> {
        self.window_resident_src(&[cur], Some((ids, k, ev)))
    }

    #[allow(clippy::type_complexity)]
    fn window_resident_src(
        &mut self,
        toks: &[u32],
        dev: Option<(u64, usize, &tang_moe::gpu::Event)>,
    ) -> Result<Option<(Vec<u32>, Vec<u32>, Vec<f32>)>> {
        let slot = 0;
        let t = toks.len() + dev.map_or(0, |d| d.1);
        if t == 0
            || std::env::var("TANG_GLM5_RES_WINDOW").is_ok_and(|v| v == "0")
            || !self.resident_ready(slot)
        {
            return Ok(None);
        }
        let hp = self.m.hp.clone();
        let (n, hc, nv) = (hp.n_embd, hp.hc, hp.n_vocab);
        for &tk in toks {
            ensure!((tk as usize) < nv, "token {tk} out of range");
        }
        let pos0 = self.slot_tokens(slot).len();
        let mut mla_w: Vec<Option<std::sync::Arc<MlaWeights>>> = vec![None; hp.n_layer];
        for l in 0..hp.n_layer {
            if !hp.is_kda[l] {
                mla_w[l] = Some(self.mla_weights(l)?);
            }
        }
        let norm_w = if self.gpu.as_ref().is_some_and(|g| g.resident.is_none()) {
            Some(self.m.vec("output_norm.weight")?)
        } else {
            None
        };
        let embd = self.m.g.info("token_embd.weight")?;
        if dev.is_some() && embed_ty(embd.ty).is_none() {
            return Ok(None);
        }
        let gguf = &self.m.g;
        let mut r = vec![0f32; if dev.is_some() { 0 } else { t * hc * n }];
        for (i, &tk) in toks.iter().enumerate().filter(|_| dev.is_none()) {
            let e = self.m.g.rows(embd, tk as usize, 1)?;
            for c in 0..hc {
                r[(i * hc + c) * n..(i * hc + c + 1) * n].copy_from_slice(&e);
            }
        }
        let g = self.gpu.as_mut().context("resident window without a GPU")?;
        if g.mhc_f.as_ref().is_none_or(|f| f.max_t() < t) {
            return Ok(None);
        }
        g.devs[0]
            .0
            .bind()
            .map_err(|e| anyhow::anyhow!("bind: {e}"))?;
        if let Some(w) = norm_w {
            let own = g.head_dev.is_none();
            g.resident = Some(Resident::new(&g.devs[0].0, &self.m.g, &w, own)?);
        }
        let ge = |e: tang_moe::gpu::Error| anyhow::anyhow!("glm5 resident window: {e}");
        let mut res = g.resident.take().unwrap();
        // (re)size the window buffers
        if res.win.as_ref().is_none_or(|w| w.cap < t) {
            let cap = t.max(4);
            let f4 = |k: usize| DevBuf::zeroed_elems::<f32>(k).map_err(ge);
            let mut ck = Vec::with_capacity(hp.n_layer);
            let mut ck_bytes = Vec::with_capacity(hp.n_layer);
            for l in 0..hp.n_layer {
                match g.kda_l[l].as_ref().filter(|_| hp.is_kda[l]) {
                    Some(k) => {
                        let b: usize = k.state_regions(0).iter().map(|r| r.1).sum();
                        ck.push(Some(DevBuf::alloc(b * (cap - 1)).map_err(ge)?));
                        ck_bytes.push(b);
                    }
                    None => {
                        ck.push(None);
                        ck_bytes.push(0);
                    }
                }
            }
            res.win = Some(WinBufs {
                cap,
                kx: f4(cap * n)?,
                ky: f4(cap * n)?,
                fx: f4(cap * n)?,
                fy: f4(cap * n)?,
                hn: f4(cap * n)?,
                ids: DevBuf::zeroed_elems::<u32>(cap).map_err(ge)?,
                wtok: DevBuf::zeroed_elems::<u32>(cap).map_err(ge)?,
                ck,
                ck_bytes,
            });
        }
        let mut f = g.mhc_f.take().unwrap();
        let s = std::mem::ManuallyDrop::new(tang_moe::gpu::Stream(f.stream().0));
        let dsa = &mut self.dsa[slot];
        let out = (|| -> Result<(Vec<u32>, Vec<u32>, Vec<f32>)> {
            res.dev_tp = None;
            // a DSA regrow moves the cache bases the whole-step graphs baked in
            let mut regrow = false;
            for l in (0..hp.n_layer).filter(|&l| !hp.is_kda[l]) {
                let cu = g.mla_k[l].as_ref().context("mla not resident")?;
                cu.set_seq(0);
                regrow |= !cu.rows_fit(pos0 + t);
            }
            if regrow {
                res.whole = [None, None];
                res.whole_cap = 0;
            }
            let items = plan(&hp);
            let is_dsa = |it: &Item| matches!(it, Item::Dsa(_));
            let graph_on =
                !prof_on() && !std::env::var("TANG_GLM5_RES_WINDOW_GRAPH").is_ok_and(|v| v == "0");
            let have = res.wgraphs.iter().any(|(tt, _)| *tt == t);
            if graph_on && !have && res.wwarm.contains(&t) {
                // capture pass: records only (the mHC host state walked once, DSA cores
                // skipped); the replay below runs everything
                f.set_window(t)?;
                let mut graphs = Vec::new();
                let mut rest: &[Item] = &items;
                let res_ro: &Resident = &res;
                loop {
                    let cut = rest.iter().position(is_dsa).unwrap_or(rest.len());
                    let seg = &rest[..cut];
                    let gr = tang_moe::gpu::Graph::capture(&s, |cs| {
                        for it in seg {
                            wexec(
                                *it,
                                &mut f,
                                g,
                                res_ro,
                                dsa,
                                &mla_w,
                                &hp,
                                pos0,
                                t,
                                cs,
                                Pass::Capture,
                            )
                            .map_err(|e| tang_moe::gpu::Error(format!("{e:#}")))?;
                        }
                        Ok(())
                    })
                    .map_err(|e| anyhow::anyhow!("window graph capture: {e}"))?;
                    graphs.push(gr);
                    if cut == rest.len() {
                        break;
                    }
                    wexec(
                        rest[cut],
                        &mut f,
                        g,
                        res_ro,
                        dsa,
                        &mla_w,
                        &hp,
                        pos0,
                        t,
                        &s,
                        Pass::HostOnly,
                    )?;
                    rest = &rest[cut + 1..];
                }
                res.wgraphs.push((t, graphs));
                eprintln!(
                    "glm5: resident window graph captured (t {t}, {} segments)",
                    res.wgraphs.last().unwrap().1.len()
                );
            }
            match dev {
                None => f.begin_async(&r)?,
                Some((ids, k, ev)) => {
                    if res.embd.is_none() {
                        let rb = embd.row_bytes()? as u64;
                        let ty = embed_ty(embd.ty).unwrap();
                        res.embd = Some(Some((
                            DevBuf::from_slice(gguf.bytes(embd)).map_err(ge)?,
                            rb,
                            ty,
                        )));
                    }
                    let (tab, rbb, ty) = {
                        let e = res.embd.as_ref().unwrap().as_ref().context("embed table")?;
                        (e.0.ptr, e.1, e.2)
                    };
                    let wb = res.win.as_ref().unwrap();
                    s.wait(ev).map_err(ge)?;
                    wb.wtok.write_async(0, &[toks[0]], &s).map_err(ge)?;
                    tang_moe::gpu::copy_d2d_async(wb.wtok.ptr + 4, ids, 4 * k, &s).map_err(ge)?;
                    let ef = res.m.func("rs_embed").map_err(ge)?;
                    let (ni, hci) = (n as i32, hc as i32);
                    for i in 0..t {
                        let (tp, rp) = (
                            wb.wtok.ptr + 4 * i as u64,
                            f.r_ptr() + (i * hc * n * 4) as u64,
                        );
                        unsafe {
                            launch(
                                ef,
                                (n.div_ceil(256) as u32, 1, 1),
                                (256, 1, 1),
                                0,
                                &s,
                                args![tab, rbb, ty, tp, rp, ni, hci],
                            )?
                        };
                    }
                    f.set_window(t)?;
                }
            }
            let gi = res
                .wgraphs
                .iter()
                .position(|(tt, _)| *tt == t)
                .filter(|_| graph_on);
            if let Some(gi) = gi {
                let graphs = &res.wgraphs[gi].1;
                let mut it_g = graphs.iter();
                it_g.next().unwrap().launch(&s).map_err(ge)?;
                for it in items.iter().filter(|it| is_dsa(it)) {
                    wexec(
                        *it,
                        &mut f,
                        g,
                        &res,
                        dsa,
                        &mla_w,
                        &hp,
                        pos0,
                        t,
                        &s,
                        Pass::Replay,
                    )?;
                    it_g.next()
                        .context("window graph: segment count")?
                        .launch(&s)
                        .map_err(ge)?;
                }
                f.set_window(0)?;
            } else {
                if !res.wwarm.contains(&t) {
                    // a new window length may grow shared scratch (the FFN pool): every
                    // captured graph that could point at the old buffers goes
                    GRAPH_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    res.wgraphs.clear();
                    res.whole = [None, None];
                    res.whole_cap = 0;
                }
                for it in &items {
                    let t0 = prof_on().then(|| {
                        let _ = s.sync();
                        std::time::Instant::now()
                    });
                    wexec(
                        *it,
                        &mut f,
                        g,
                        &res,
                        dsa,
                        &mla_w,
                        &hp,
                        pos0,
                        t,
                        &s,
                        Pass::Eager,
                    )?;
                    if let Some(t0) = t0 {
                        s.sync().map_err(ge)?;
                        let k = match it {
                            Item::PreAttn(_) | Item::PreFfn(_) => "win.mhc",
                            Item::Kda(_) => "win.kda",
                            Item::Dsa(_) => "win.dsa",
                            Item::Ffn(_) => "win.moe",
                            Item::Head => "win.head",
                        };
                        prof_add(k, t0.elapsed().as_secs_f64());
                        // the routed experts' union over the t rows (one line per layer)
                        if let (Item::Ffn(l), true) = (
                            it,
                            std::env::var("TANG_GLM5_WIN_PROF_UNION").is_ok_and(|v| v == "1"),
                        ) {
                            if let Some(m) = g.layers[*l].as_ref().filter(|m| m.exps.is_some()) {
                                let nr = m.p.top_k;
                                if let Some(ids) =
                                    super::super::moe::gpu::last_route_ids(&g.devs[0].1, t * nr)?
                                {
                                    let mut u = ids.clone();
                                    u.sort_unstable();
                                    u.dedup();
                                    eprintln!(
                                        "prof union l{l} t {t}: {} distinct of {} routed",
                                        u.len(),
                                        t * nr
                                    );
                                    prof_add("win.moe_union", u.len() as f64 / (t * nr) as f64);
                                }
                            }
                        }
                    }
                }
                if !res.wwarm.contains(&t) {
                    res.wwarm.push(t);
                }
            }
            let wb = res.win.as_ref().unwrap();
            s.sync().map_err(ge)?;
            let ids = wb.ids.read::<u32>(0, t).map_err(ge)?;
            let hn = wb.hn.read::<f32>(0, t * n).map_err(ge)?;
            let all = if dev.is_some() {
                wb.wtok.read::<u32>(0, t).map_err(ge)?
            } else {
                toks.to_vec()
            };
            Ok((all, ids, hn))
        })();
        if out.is_err() {
            // the mHC state machine may be mid-window
            let _ = f.set_window(0);
        }
        res.win_count[0] += 1;
        g.mhc_f = Some(f);
        g.resident = Some(res);
        let (all, ids, hn) = out?;
        self.slot_tokens_mut(slot).extend_from_slice(&all);
        self.last_h.clear();
        self.last_h.extend_from_slice(&hn);
        Ok(Some((all, ids, hn)))
    }

    /// After a [`Self::window_resident`] of `t` rows: keep only the first `keep` (1..=t).
    /// The KDA layers restore the device checkpoint after row `keep - 1` (D2D), the DSA
    /// layers drop their device watermarks (latent / indexer rows, pools) and host mirrors
    /// back to the kept position, and slot 0's tokens are truncated. Nothing re-runs.
    pub(crate) fn rewind_resident(&mut self, t: usize, keep: usize) -> Result<()> {
        ensure!(
            keep >= 1 && keep <= t,
            "rewind_resident: keep {keep} of {t}"
        );
        if keep == t {
            return Ok(());
        }
        let hp = self.m.hp.clone();
        let pos = self.tokens.len() - t + keep;
        let mut kpool = vec![0usize; hp.n_layer];
        for l in 0..hp.n_layer {
            if !hp.is_kda[l] {
                kpool[l] = self.mla_weights(l)?.kpool;
            }
        }
        let g = self.gpu.as_mut().context("rewind_resident without a GPU")?;
        g.devs[0]
            .0
            .bind()
            .map_err(|e| anyhow::anyhow!("bind: {e}"))?;
        let ge = |e: tang_moe::gpu::Error| anyhow::anyhow!("glm5 resident rewind: {e}");
        let f = g.mhc_f.as_ref().context("fused mHC: no instance")?;
        let s = f.stream();
        let res = g
            .resident
            .as_mut()
            .context("rewind_resident: no resident window")?;
        let wb = res
            .win
            .as_ref()
            .context("rewind_resident: no window buffers")?;
        ensure!(wb.cap >= t, "rewind_resident: window cap {}", wb.cap);
        for l in 0..hp.n_layer {
            if hp.is_kda[l] {
                let k = g.kda_l[l].as_ref().context("kda not resident")?;
                let regs = k.state_regions(0);
                let base = wb.ck[l].as_ref().unwrap().ptr + ((keep - 1) * wb.ck_bytes[l]) as u64;
                tang_moe::gpu::copy_d2d_async(regs[0].0, base, regs[0].1, s).map_err(ge)?;
                tang_moe::gpu::copy_d2d_async(regs[1].0, base + regs[0].1 as u64, regs[1].1, s)
                    .map_err(ge)?;
            } else {
                let cu = g.mla_k[l].as_ref().context("mla not resident")?;
                cu.set_seq(0);
                cu.append_done(pos, 0);
                cu.set_pool_upto(cu.pool_upto().min(pos / kpool[l].max(1)));
            }
        }
        s.sync().map_err(ge)?;
        res.dev_tp = None;
        res.win_count[1] += 1;
        let (kl, idd) = (hp.kv_lora_rank, hp.idx_dim);
        for c in &mut self.dsa[0] {
            c.truncate_rows(pos, kl, idd);
        }
        self.tokens.truncate(pos);
        Ok(())
    }

    /// The LM head's greedy id for each normed row of `hn` (`[t][n]`, already normed), on
    /// the device head (q8 + gemv8 + `rs_argmax` per row: the t=1 head numerics); one
    /// upload, one sync, `4 * t` bytes back. `None` without the device head or before the
    /// resident state exists (the caller reads the logits back instead). The MTP drafter.
    pub(crate) fn head_top(&self, hn: &[f32]) -> Result<Option<Vec<u32>>> {
        let (n, nv) = (self.m.hp.n_embd, self.m.hp.n_vocab);
        let Some(g) = self.gpu.as_ref() else {
            return Ok(None);
        };
        let (Some(hd), Some(res)) = (g.head_dev.as_ref(), g.resident.as_ref()) else {
            return Ok(None);
        };
        let Some(wb) = res.win.as_ref() else {
            return Ok(None);
        };
        let t = hn.len() / n;
        if t == 0 || t > wb.cap {
            return Ok(None);
        }
        let ge = |e: tang_moe::gpu::Error| anyhow::anyhow!("glm5 head_top: {e}");
        g.devs[0]
            .0
            .bind()
            .map_err(|e| anyhow::anyhow!("bind: {e}"))?;
        let s = &hd.s;
        // the draft rows go through the window's h buffer (free between windows)
        wb.hn.write_async(0, hn, s).map_err(ge)?;
        let af = res.m.func("rs_argmax").map_err(ge)?;
        for i in 0..t {
            let xi = wb.hn.ptr + (i * n * 4) as u64;
            unsafe {
                launch(
                    hd.k5.q8,
                    ((n / 32) as u32, 1, 1),
                    (32, 1, 1),
                    0,
                    s,
                    args![xi, hd.xq8.ptr, n as u32, 1u32],
                )?;
                launch(
                    hd.k5.gemv8,
                    (nv.div_ceil(K5_WARPS) as u32, 1, 1),
                    (128, 1, 1),
                    (K5_WARPS * ((hd.dw.w.row_bytes as usize).div_ceil(16) + 1) * 16) as u32,
                    s,
                    args![
                        hd.dw.w,
                        0u32,
                        hd.xq8.ptr,
                        hd.xq8.ptr + n as u64,
                        hd.out.ptr,
                        nv as u32,
                        1u32,
                        nv as u32,
                        n as u32
                    ],
                )?;
                let (lg, nvi, tp, inc) = (hd.out.ptr, nv as i32, wb.ids.ptr + 4 * i as u64, 0i32);
                launch(af, (1, 1, 1), (1024, 1, 1), 0, s, args![lg, nvi, tp, inc])?;
            }
        }
        s.sync().map_err(ge)?;
        let ids = wb.ids.read::<u32>(0, t).map_err(ge)?;
        if prof_on() {
            prof_add("draft.head_top", 0.0);
        }
        Ok(Some(ids))
    }

    /// After a resident decode step ([`Self::step_resident`]): the device pointers of its
    /// greedy token (`Resident::tok[0]`) and its normed h row (the device head's input).
    pub(crate) fn step_out_ptrs(&self) -> Option<(u64, u64)> {
        let g = self.gpu.as_ref()?;
        Some((g.resident.as_ref()?.tok.ptr, g.head_dev.as_ref()?.xf.ptr))
    }

    /// The last resident step's normed h row, read back (the host drafter's fallback).
    pub(crate) fn step_h_read(&self) -> Result<Option<Vec<f32>>> {
        let Some(g) = self.gpu.as_ref() else {
            return Ok(None);
        };
        let Some(hd) = g.head_dev.as_ref() else {
            return Ok(None);
        };
        Ok(Some(
            hd.xf
                .read::<f32>(0, self.m.hp.n_embd)
                .map_err(|e| anyhow::anyhow!("step h: {e}"))?,
        ))
    }

    /// Order the next trunk launch (the fused mHC stream) after `ev`.
    pub(crate) fn trunk_wait(&self, ev: &tang_moe::gpu::Event) -> Result<()> {
        if let Some(f) = self.gpu.as_ref().and_then(|g| g.mhc_f.as_ref()) {
            f.stream()
                .wait(ev)
                .map_err(|e| anyhow::anyhow!("trunk wait: {e}"))?;
        }
        Ok(())
    }

    /// After a resident step: copy its greedy token and normed h row into `dst` (`[tok u32]`
    /// then `[h f32; n]`) ON the trunk stream (in order: the next step cannot overwrite them
    /// first), and return an event recorded after the copy.
    pub(crate) fn step_out_stage(
        &self,
        dst_tok: u64,
        dst_h: u64,
    ) -> Result<Option<tang_moe::gpu::Event>> {
        let ge = |e: tang_moe::gpu::Error| anyhow::anyhow!("step stage: {e}");
        let Some((tp, hp)) = self.step_out_ptrs() else {
            return Ok(None);
        };
        let Some(f) = self.gpu.as_ref().and_then(|g| g.mhc_f.as_ref()) else {
            return Ok(None);
        };
        let s = f.stream();
        tang_moe::gpu::copy_d2d_async(dst_tok, tp, 4, s).map_err(ge)?;
        tang_moe::gpu::copy_d2d_async(dst_h, hp, self.m.hp.n_embd * 4, s).map_err(ge)?;
        let ev = tang_moe::gpu::Event::new(false).map_err(ge)?;
        ev.record(s).map_err(ge)?;
        Ok(Some(ev))
    }

    /// The last resident window's device outputs: (argmax ids `[t]` u32, normed h `[t][n]`).
    pub(crate) fn window_out_ptrs(&self) -> Option<(u64, u64)> {
        let wb = self.gpu.as_ref()?.resident.as_ref()?.win.as_ref()?;
        Some((wb.ids.ptr, wb.hn.ptr))
    }

    /// Drop the captured step segments (any host window: it may reallocate their scratch).
    pub(super) fn drop_step_graph(&mut self) {
        GRAPH_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(r) = self.gpu.as_mut().and_then(|g| g.resident.as_mut()) {
            r.graphs = None;
            r.whole = [None, None];
            r.whole_cap = 0;
            r.dev_tp = None;
            r.warm = false;
            r.wgraphs.clear();
            r.wwarm.clear();
        }
    }
}

/// `TANG_GLM5_STEP_GRAPH=2`: the whole decode token as ONE graph replay. The device embed
/// gather reads the token `rs_argmax` left in `res.tok[0]`; every DSA kernel reads the
/// position from `res.tok[1]` (advanced by the graph's last kernel); the host only keeps
/// its watermarks / mirror sizes in step. Two graphs, dense and sparse (the indexer
/// branch), captured lazily the first time each is needed. `None`: the token is past the
/// graphs' DSA cache capacity (the caller runs it eagerly, which regrows the caches; the
/// next token recaptures).
#[allow(clippy::too_many_arguments)]
fn step_whole(
    f: &mut super::super::mhc_fused::MhcFused,
    g: &mut GpuFfn,
    res: &mut Resident,
    dsa: &mut [DsaCache],
    mla_w: &[Option<std::sync::Arc<MlaWeights>>],
    hp: &super::super::Hparams,
    items: &[Item],
    dsa_l: &[usize],
    pos0: usize,
    tok: u32,
    sparse: bool,
    s: &tang_moe::gpu::Stream,
    gguf: &crate::gguf::Gguf,
    embd: &crate::gguf::TensorInfo,
    host_embed: &dyn Fn() -> Result<Vec<f32>>,
) -> Result<Option<u32>> {
    use super::super::trace::span;
    let ge = |e: tang_moe::gpu::Error| anyhow::anyhow!("glm5 step graph 2: {e}");
    let kv_len = pos0 + 1;
    if res.whole_cap < kv_len {
        res.whole = [None, None];
    }
    if res.embd.is_none() {
        res.embd = Some(match embed_ty(embd.ty) {
            Some(ty) => {
                let rb = embd.row_bytes()? as u64;
                Some((DevBuf::from_slice(gguf.bytes(embd)).map_err(ge)?, rb, ty))
            }
            None => {
                eprintln!(
                    "glm5: step graph 2: token_embd is {}, embed stays a host gather",
                    embd.ty.name()
                );
                None
            }
        });
    }
    let sp = usize::from(sparse);
    if res.whole[sp].is_none() {
        let mut cap = usize::MAX;
        for &l in dsa_l {
            let cu = g.mla_k[l].as_ref().context("mla not resident")?;
            cu.set_seq(0);
            cap = cap.min(cu.row_cap());
        }
        if cap < kv_len {
            return Ok(None);
        }
        let mut grew = false;
        for &l in dsa_l {
            let w = mla_w[l].as_ref().unwrap();
            let cu = g.mla_k[l].as_ref().unwrap();
            grew |= cu.reserve_pool(cap / w.kpool + 1).map_err(ge)?;
            cu.stage_ape(&w.i_ape).map_err(ge)?;
        }
        if grew || cap != res.whole_cap {
            res.whole[1 - sp] = None;
        }
        res.whole_cap = cap;
        let _tr = span("capture");
        f.set_window(1)?;
        let dev_embed = res
            .embd
            .as_ref()
            .unwrap()
            .as_ref()
            .map(|(b, rb, ty)| (b.ptr, *rb, *ty));
        let rp = f.r_ptr();
        let res_ro: &Resident = res;
        let gr = tang_moe::gpu::Graph::capture(s, |cs| {
            let mut run = || -> Result<()> {
                if let Some((tab, rb, ty)) = dev_embed {
                    let ef = res_ro.m.func("rs_embed").map_err(ge)?;
                    let (tp, ni, hci) = (res_ro.tok.ptr, hp.n_embd as i32, hp.hc as i32);
                    let grid = hp.n_embd.div_ceil(256) as u32;
                    unsafe {
                        launch(
                            ef,
                            (grid, 1, 1),
                            (256, 1, 1),
                            0,
                            cs,
                            args![tab, rb, ty, tp, rp, ni, hci],
                        )?
                    };
                }
                for it in items {
                    exec(
                        *it,
                        f,
                        g,
                        res_ro,
                        dsa,
                        mla_w,
                        hp,
                        pos0,
                        cs,
                        Pass::Whole(sparse),
                    )?;
                }
                Ok(())
            };
            run().map_err(|e| tang_moe::gpu::Error(format!("{e:#}")))
        })
        .map_err(|e| anyhow::anyhow!("whole step graph capture: {e}"))?;
        f.set_window(0)?;
        res.whole[sp] = Some(gr);
        eprintln!(
            "glm5: whole step graph captured ({}, DSA cache cap {cap} rows)",
            if sparse { "sparse" } else { "dense" }
        );
    }
    let tr = span("embed");
    if res.dev_tp != Some((tok, pos0)) {
        // out of step (first replay, or after an eager token / host window): 8 bytes up
        res.tok.write_async(0, &[tok, pos0 as u32], s).map_err(ge)?;
    }
    if res.embd.as_ref().unwrap().is_none() {
        f.begin_async(&host_embed()?)?;
        f.set_window(0)?;
    }
    drop(tr);
    if sparse {
        // pools completed before this token that no replay computed (the dense graph
        // skips the indexer): the eager incremental pool, a no-op once caught up
        let _tr = span("dsa/pool_catchup");
        for &l in dsa_l {
            let (cu, w) = (g.mla_k[l].as_ref().unwrap(), mla_w[l].as_ref().unwrap());
            cu.set_seq(0);
            let np = pos0 / w.kpool;
            if cu.pool_upto() < np {
                let c = &dsa[l];
                cu.idx_pool(&c.idx_k, &c.idx_g, &w.i_ape, pos0, 0, np, &mut [], s)
                    .map_err(ge)?;
            }
        }
    }
    {
        let _tr = span("graph");
        res.whole[sp].as_ref().unwrap().launch(s).map_err(ge)?;
        res.whole_replays[sp] += 1;
    }
    // the host's view of what the replay did to the DSA caches
    for &l in dsa_l {
        let (cu, w) = (g.mla_k[l].as_ref().unwrap(), mla_w[l].as_ref().unwrap());
        let c = &mut dsa[l];
        let (dl, idd) = (w.kv_lora, w.idx_dim);
        ensure!(c.lat16.len() == pos0 * dl, "latent cache out of step");
        c.lat16.resize(kv_len * dl, 0);
        c.idx_k.resize(kv_len * idd, 0.0);
        c.idx_g.resize(kv_len * idd, 0.0);
        c.valid.resize(kv_len, 0);
        c.stale_from = Some(c.stale_from.map_or(pos0, |s| s.min(pos0)));
        cu.set_seq(0);
        cu.append_done(pos0, 1);
        if sparse {
            cu.set_pool_upto(kv_len / w.kpool);
        }
    }
    let _tr = span("sync+read");
    s.sync().map_err(ge)?;
    let t = res.tok.read::<u32>(0, 1).map_err(ge)?[0];
    res.dev_tp = Some((t, kv_len));
    Ok(Some(t))
}

/// `TANG_GLM5_RES_KDA_ROWS=1` (default off): the window's KDA layers batch their
/// projections over the t rows (`DevLayer::rows_enqueue`). Measured SLOWER on the B300
/// (13.5 vs 6.9 ms of KDA per t=3 window: the t>1 projection / gate kernels are not the
/// tuned t=1 ones) and not bit-identical; off = the per-row t=1 decodes (= plain decode).
fn kda_rows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TANG_GLM5_RES_KDA_ROWS").is_ok_and(|v| v == "1"))
}

/// `TANG_GLM5_WIN_PROF=1`: the resident window runs eagerly with a sync around every item
/// and the MTP draft times its stages; [`prof_report`] prints per-stage ms per call.
pub(crate) fn prof_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TANG_GLM5_WIN_PROF").is_ok_and(|v| v == "1"))
}

static PROF: std::sync::Mutex<Vec<(&'static str, f64, usize)>> = std::sync::Mutex::new(Vec::new());

/// Accumulate `secs` under `k` (one call).
pub(crate) fn prof_add(k: &'static str, secs: f64) {
    let mut p = PROF.lock().unwrap();
    match p.iter_mut().find(|e| e.0 == k) {
        Some(e) => {
            e.1 += secs;
            e.2 += 1;
        }
        None => p.push((k, secs, 1)),
    }
}

/// The accumulated stages: `name total_ms calls`, one line each.
pub(crate) fn prof_report() -> String {
    PROF.lock()
        .unwrap()
        .iter()
        .map(|(k, s, c)| format!("prof {k}: {:.2} ms total, {c} calls\n", s * 1e3))
        .collect()
}

/// One [`Item`] of [`Glm5Engine::window_resident`] (t rows) on `s`. `Pass` as for
/// [`exec`]: `Capture` records (KDA enqueues its kernels instead of its own graph), a DSA
/// item is the segment boundary (`HostOnly`: its mHC post; `Replay`: its device work).
#[allow(clippy::too_many_arguments)]
fn wexec(
    it: Item,
    f: &mut super::super::mhc_fused::MhcFused,
    g: &mut GpuFfn,
    res: &Resident,
    dsa: &mut [DsaCache],
    mla_w: &[Option<std::sync::Arc<MlaWeights>>],
    hp: &super::super::Hparams,
    pos0: usize,
    t: usize,
    s: &tang_moe::gpu::Stream,
    pass: Pass,
) -> Result<()> {
    use tang_moe::gpu::copy_d2d_async as d2d;
    let ge = |e: tang_moe::gpu::Error| anyhow::anyhow!("glm5 resident window: {e}");
    let wb = res.win.as_ref().context("window buffers")?;
    let (n, nv) = (hp.n_embd, hp.n_vocab);
    let rb = n * 4;
    if pass == Pass::HostOnly {
        let Item::Dsa(l) = it else {
            anyhow::bail!("window graph: only DSA items are boundaries")
        };
        return f.post_dev(g.mla_p[l].as_ref().context("mla not resident")?.y_ptr(t)?);
    }
    let capture = pass == Pass::Capture;
    match it {
        Item::PreAttn(l) => {
            let dst = if hp.is_kda[l] {
                let k = g.kda_l[l].as_ref().context("kda not resident")?;
                if kda_rows() || k.rows_fast_ok(t) {
                    k.x_ptr()
                } else {
                    wb.kx.ptr
                }
            } else {
                g.mla_p[l].as_ref().context("mla not resident")?.x_ptr(t)?
            };
            f.pre_dev(g.mhc_g[l].as_ref().context("mhc not resident")?, l, 0, dst)?;
        }
        Item::Kda(l) => {
            let k = g.kda_l[l].as_mut().context("kda not resident")?;
            if kda_rows() || k.rows_fast_ok(t) {
                let ck = wb.ck[l].as_ref().map(|b| (b.ptr, wb.ck_bytes[l]));
                k.rows_enqueue(s, t, ck)?;
                return f.post_dev(k.y_ptr());
            }
            let regs = k.state_regions(0);
            for i in 0..t {
                d2d(k.x_ptr(), wb.kx.ptr + (i * rb) as u64, rb, s).map_err(ge)?;
                if capture {
                    k.decode_enqueue(s)?;
                } else {
                    k.decode_dev(s)?;
                }
                d2d(wb.ky.ptr + (i * rb) as u64, k.y_ptr(), rb, s).map_err(ge)?;
                if i + 1 < t {
                    let base = wb.ck[l].as_ref().unwrap().ptr + (i * wb.ck_bytes[l]) as u64;
                    d2d(base, regs[0].0, regs[0].1, s).map_err(ge)?;
                    d2d(base + regs[0].1 as u64, regs[1].0, regs[1].1, s).map_err(ge)?;
                }
            }
            f.post_dev(wb.ky.ptr)?;
        }
        Item::Dsa(l) => {
            ensure!(!capture, "window graph: a DSA item inside a capture");
            let (cu, p) = (
                g.mla_k[l].as_ref().unwrap(),
                g.mla_p[l].as_ref().context("mla not resident")?,
            );
            let w = mla_w[l].as_ref().unwrap();
            cu.set_seq(0);
            let c = &mut dsa[l];
            let (dl, idd) = (w.kv_lora, w.idx_dim);
            ensure!(c.lat16.len() == pos0 * dl, "latent cache out of step");
            let kv_len = pos0 + t;
            if !cu.rows_fit(kv_len) {
                if let Some(from) = c.stale_from.take().filter(|&fr| fr < pos0) {
                    cu.refresh_mirrors(
                        from,
                        pos0 - from,
                        &mut c.lat16,
                        &mut c.idx_k,
                        &mut c.idx_g,
                        &mut c.valid,
                    )
                    .map_err(|e| anyhow::anyhow!("glm5 dsa mirror refresh: {e}"))?;
                }
            }
            c.lat16.resize(kv_len * dl, 0);
            c.idx_k.resize(kv_len * idd, 0.0);
            c.idx_g.resize(kv_len * idd, 0.0);
            c.valid.resize(kv_len, 0);
            c.stale_from = Some(c.stale_from.map_or(pos0, |s| s.min(pos0)));
            let y = p.window_dev(cu, w, t, pos0, &c.lat16, &c.idx_k, &c.idx_g, &c.valid, s)?;
            if pass != Pass::Replay {
                f.post_dev(y)?;
            }
        }
        Item::PreFfn(l) => {
            f.pre_dev(
                g.mhc_g[l].as_ref().context("mhc not resident")?,
                l,
                1,
                wb.fx.ptr,
            )?;
        }
        Item::Ffn(l) => {
            let m = g.layers[l].as_ref().context("FFN not resident")?;
            super::super::moe::gpu::ffn_async(&g.devs[0].1, s, m, wb.fx.ptr, wb.fy.ptr, t)?;
            f.post_dev(wb.fy.ptr)?;
        }
        Item::Head => {
            let h = f.end_dev()?;
            let nf = res.m.func("rs_norm").map_err(ge)?;
            let af = res.m.func("rs_argmax").map_err(ge)?;
            let (nw, ni, eps) = (res.norm_w.ptr, n as i32, hp.eps);
            for i in 0..t {
                let hi = h + (i * rb) as u64;
                let (xf, logits) = unsafe {
                    if let Some(hd) = g.head_dev.as_ref() {
                        let xf = hd.xf.ptr;
                        launch(
                            nf,
                            (1, 1, 1),
                            (1024, 1, 1),
                            0,
                            s,
                            args![hi, nw, xf, ni, eps],
                        )?;
                        launch(
                            hd.k5.q8,
                            ((n / 32) as u32, 1, 1),
                            (32, 1, 1),
                            0,
                            s,
                            args![hd.xf.ptr, hd.xq8.ptr, n as u32, 1u32],
                        )?;
                        launch(
                            hd.k5.gemv8,
                            (nv.div_ceil(K5_WARPS) as u32, 1, 1),
                            (128, 1, 1),
                            (K5_WARPS * ((hd.dw.w.row_bytes as usize).div_ceil(16) + 1) * 16)
                                as u32,
                            s,
                            args![
                                hd.dw.w,
                                0u32,
                                hd.xq8.ptr,
                                hd.xq8.ptr + n as u64,
                                hd.out.ptr,
                                nv as u32,
                                1u32,
                                nv as u32,
                                n as u32
                            ],
                        )?;
                        (xf, hd.out.ptr)
                    } else {
                        let (dw, k5, xb, ob) = res.head.as_ref().context("resident: no LM head")?;
                        let xf = xb.ptr;
                        launch(
                            nf,
                            (1, 1, 1),
                            (1024, 1, 1),
                            0,
                            s,
                            args![hi, nw, xf, ni, eps],
                        )?;
                        launch(
                            k5.gemv,
                            (nv.div_ceil(128) as u32, 1, 1),
                            (128, 1, 1),
                            0,
                            s,
                            args![dw.w, 0u32, xb.ptr, ob.ptr, nv as u32, 1u32, nv as u32, n as u32],
                        )?;
                        (xf, ob.ptr)
                    }
                };
                d2d(wb.hn.ptr + (i * rb) as u64, xf, rb, s).map_err(ge)?;
                let (nvi, tp, inc) = (nv as i32, wb.ids.ptr + 4 * i as u64, 0i32);
                unsafe {
                    launch(
                        af,
                        (1, 1, 1),
                        (1024, 1, 1),
                        0,
                        s,
                        args![logits, nvi, tp, inc],
                    )?
                };
            }
        }
    }
    Ok(())
}

/// One step of the resident decode, in order. `Dsa` is the eager part of a DSA layer (the
/// graph segment boundary); everything else is position-independent.
#[derive(Clone, Copy)]
enum Item {
    PreAttn(usize),
    Kda(usize),
    Dsa(usize),
    PreFfn(usize),
    Ffn(usize),
    Head,
}

fn plan(hp: &super::super::Hparams) -> Vec<Item> {
    let mut v = Vec::new();
    for l in 0..hp.n_layer {
        v.push(Item::PreAttn(l));
        v.push(if hp.is_kda[l] {
            Item::Kda(l)
        } else {
            Item::Dsa(l)
        });
        v.push(Item::PreFfn(l));
        v.push(Item::Ffn(l));
    }
    v.push(Item::Head);
    v
}

/// How [`exec`] runs an item.
#[derive(Clone, Copy, PartialEq)]
enum Pass {
    /// Everything, eagerly (the no-graph step and the warm-up token).
    Eager,
    /// Inside a segment capture: KDA enqueues its fused kernels instead of launching its
    /// own graph; a DSA item only does its mHC post (host state).
    Capture,
    /// Between segment launches: a boundary item runs its device work only.
    Replay,
    /// The capture pass at a boundary item (outside any capture): its mHC post only, so
    /// the next segment's pre applies it (the replay runs its device work).
    HostOnly,
    /// Inside the whole-step capture (`TANG_GLM5_STEP_GRAPH=2`; `true` = the sparse
    /// indexer branch): as `Capture`, and a DSA item enqueues its position-counter window.
    Whole(bool),
}

/// Enqueue one [`Item`] on `s`.
#[allow(clippy::too_many_arguments)]
fn exec(
    it: Item,
    f: &mut super::super::mhc_fused::MhcFused,
    g: &mut GpuFfn,
    res: &Resident,
    dsa: &mut [DsaCache],
    mla_w: &[Option<std::sync::Arc<MlaWeights>>],
    hp: &super::super::Hparams,
    pos0: usize,
    s: &tang_moe::gpu::Stream,
    pass: Pass,
) -> Result<()> {
    let capture = matches!(pass, Pass::Capture | Pass::Whole(_));
    if pass == Pass::HostOnly {
        let y = match it {
            Item::Kda(l) => g.kda_l[l].as_ref().context("kda not resident")?.y_ptr(),
            Item::Dsa(l) => g.mla_p[l].as_ref().context("mla not resident")?.y_ptr(1)?,
            Item::Ffn(_) => res.y.ptr,
            _ => anyhow::bail!("step graph: only kda / dsa / ffn items can be boundaries"),
        };
        return f.post_dev(y);
    }
    use super::super::trace::span;
    let ge = |e: tang_moe::gpu::Error| anyhow::anyhow!("glm5 resident: {e}");
    let mh = |l: usize| g.mhc_g[l].as_ref().context("mhc not resident");
    match it {
        Item::PreAttn(l) => {
            let _tr = span("mhc_pre.attn");
            let dst = if hp.is_kda[l] {
                g.kda_l[l].as_ref().context("kda not resident")?.x_ptr()
            } else {
                g.mla_p[l].as_ref().context("mla not resident")?.x_ptr(1)?
            };
            f.pre_dev(mh(l)?, l, 0, dst)?;
        }
        Item::Kda(l) => {
            let _tr = span("kda");
            let k = g.kda_l[l].as_mut().context("kda not resident")?;
            if capture {
                k.decode_enqueue(s)?;
            } else {
                k.decode_dev(s)?;
            }
            if pass != Pass::Replay {
                f.post_dev(k.y_ptr())?;
            }
        }
        Item::Dsa(l) => {
            let (cu, p) = (g.mla_k[l].as_ref().unwrap(), g.mla_p[l].as_ref().unwrap());
            if let Pass::Whole(sparse) = pass {
                let _tr = span("dsa");
                cu.set_seq(0);
                let y =
                    p.window_dev_pos(cu, mla_w[l].as_ref().unwrap(), res.tok.ptr + 4, sparse, s)?;
                debug_assert_eq!(y, p.y_ptr(1)?);
                return f.post_dev(y);
            }
            ensure!(!capture, "step graph: a DSA item inside a capture");
            {
                let _tr = span("dsa");
                let w = mla_w[l].as_ref().unwrap();
                cu.set_seq(0);
                let c = &mut dsa[l];
                let (dl, idd) = (w.kv_lora, w.idx_dim);
                ensure!(c.lat16.len() == pos0 * dl, "latent cache out of step");
                let kv_len = pos0 + 1;
                if !cu.rows_fit(kv_len) {
                    // the append regrows the device caches and re-uploads [0, pos0) from the
                    // host mirrors: bring the stale rows over first (rare: capacity doubles)
                    if let Some(from) = c.stale_from.take().filter(|&f| f < pos0) {
                        cu.refresh_mirrors(
                            from,
                            pos0 - from,
                            &mut c.lat16,
                            &mut c.idx_k,
                            &mut c.idx_g,
                            &mut c.valid,
                        )
                        .map_err(|e| anyhow::anyhow!("glm5 dsa mirror refresh: {e}"))?;
                    }
                }
                c.lat16.resize(kv_len * dl, 0);
                c.idx_k.resize(kv_len * idd, 0.0);
                c.idx_g.resize(kv_len * idd, 0.0);
                c.valid.resize(kv_len, 0);
                c.stale_from = Some(c.stale_from.map_or(pos0, |s| s.min(pos0)));
                let y = p.window_dev(cu, w, 1, pos0, &c.lat16, &c.idx_k, &c.idx_g, &c.valid, s)?;
                debug_assert_eq!(y, p.y_ptr(1)?);
            }
            // replay: the host mHC state is not walked (set_window resets it after)
            if pass != Pass::Replay {
                f.post_dev(p.y_ptr(1)?)?;
            }
        }
        Item::PreFfn(l) => {
            let _tr = span("mhc_pre.ffn");
            f.pre_dev(mh(l)?, l, 1, res.x.ptr)?;
        }
        Item::Ffn(l) => {
            let _tr = span("ffn");
            let m = g.layers[l].as_ref().context("FFN not resident")?;
            super::super::moe::gpu::ffn_async(&g.devs[0].1, s, m, res.x.ptr, res.y.ptr, 1)?;
            if pass != Pass::Replay {
                f.post_dev(res.y.ptr)?;
            }
        }
        Item::Head => {
            let _tr = span("head");
            let h = f.end_dev()?;
            let n = hp.n_embd;
            let nv = hp.n_vocab;
            let nf = res.m.func("rs_norm").map_err(ge)?;
            let af = res.m.func("rs_argmax").map_err(ge)?;
            let (nw, ni, eps) = (res.norm_w.ptr, n as i32, hp.eps);
            let logits = unsafe {
                if let Some(hd) = g.head_dev.as_ref() {
                    let xf = hd.xf.ptr;
                    launch(nf, (1, 1, 1), (1024, 1, 1), 0, s, args![h, nw, xf, ni, eps])?;
                    launch(
                        hd.k5.q8,
                        ((n / 32) as u32, 1, 1),
                        (32, 1, 1),
                        0,
                        s,
                        args![hd.xf.ptr, hd.xq8.ptr, n as u32, 1u32],
                    )?;
                    launch(
                        hd.k5.gemv8,
                        (nv.div_ceil(K5_WARPS) as u32, 1, 1),
                        (128, 1, 1),
                        (K5_WARPS * ((hd.dw.w.row_bytes as usize).div_ceil(16) + 1) * 16) as u32,
                        s,
                        args![
                            hd.dw.w,
                            0u32,
                            hd.xq8.ptr,
                            hd.xq8.ptr + n as u64,
                            hd.out.ptr,
                            nv as u32,
                            1u32,
                            nv as u32,
                            n as u32
                        ],
                    )?;
                    hd.out.ptr
                } else {
                    let (dw, k5, xb, ob) = res.head.as_ref().context("resident: no LM head")?;
                    let xf = xb.ptr;
                    launch(nf, (1, 1, 1), (1024, 1, 1), 0, s, args![h, nw, xf, ni, eps])?;
                    launch(
                        k5.gemv,
                        (nv.div_ceil(128) as u32, 1, 1),
                        (128, 1, 1),
                        0,
                        s,
                        args![dw.w, 0u32, xb.ptr, ob.ptr, nv as u32, 1u32, nv as u32, n as u32],
                    )?;
                    ob.ptr
                }
            };
            let (nvi, tp, inc) = (
                nv as i32,
                res.tok.ptr,
                i32::from(matches!(pass, Pass::Whole(_))),
            );
            unsafe {
                launch(
                    af,
                    (1, 1, 1),
                    (1024, 1, 1),
                    0,
                    s,
                    args![logits, nvi, tp, inc],
                )?
            };
        }
    }
    Ok(())
}

/// Bumped whenever shared scratch a captured graph may point at can be reallocated (a host
/// window, a new resident window length): the MTP draft-cell graphs recapture after it.
static GRAPH_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The MTP draft cell on the device (`TANG_GLM5_MTP_CELL_DEV`, default on): one NextN cell
/// at the drafter cache's next position as ONE graph replay — embed gather of the token in
/// `tp[0]` (`rs_embed`), enorm / hnorm (`rs_norm`) into `[e ; h]`, eh_proj on the fast
/// q8 + `g5f_gemv` (the MLA projections' kernel), attn_norm, the DSA block on the
/// device-position kernels (`MlaProj::window_dev_pos`, position `tp[1]`), the FFN
/// (`ffn_async`), shared_head_norm, the LM head (q8 + gemv8) and `rs_argmax` (writes the
/// draft into `tp[0]`, advances `tp[1]`), and `hin <- hout` for the next chained cell. No
/// host sync inside; the host only keeps the cache watermarks. Graphs `[dense, sparse]`.
pub(crate) struct DraftCell {
    /// The cell's own MoE kernels (their scratch pool is private: a cell may run
    /// concurrently with a trunk step, `run`'s `after`).
    mk: super::super::moe::gpu::Kernels,
    stage_t: DevBuf,
    stage_h: DevBuf,
    stage_i: usize,
    warm: bool,
    own_head: Option<super::super::moe::gpu::DevW>,
    m: Module,
    s: tang_moe::gpu::Stream,
    k5: super::super::moe::gpu::Kernels5,
    eh: super::super::moe::gpu::DevW,
    embd: (DevBuf, u64, u32),
    norms: [DevBuf; 5],
    eps: f32,
    pub(crate) tp: DevBuf,
    pub(crate) ids: DevBuf,
    hin: DevBuf,
    cat: DevBuf,
    cq: DevBuf,
    cq2: DevBuf,
    cd: DevBuf,
    x: DevBuf,
    fx: DevBuf,
    fy: DevBuf,
    hout: DevBuf,
    hq8: DevBuf,
    logits: DevBuf,
    graphs: [Option<tang_moe::gpu::Graph>; 2],
    gen: u64,
    cap: usize,
    pub(crate) replays: usize,
    /// What `tp[1]` holds on the device once the queued cells ran (`None`: unknown).
    dev_pos: Option<usize>,
}

/// [`DraftCell::stage_next`]'s ring length.
const STAGE: usize = 8;

/// `TANG_GLM5_MTP_CELL_DEV=0`: the drafter keeps the host-staged cell.
pub(crate) fn cell_dev_on() -> bool {
    !std::env::var("TANG_GLM5_MTP_CELL_DEV").is_ok_and(|v| v == "0")
}

impl DraftCell {
    /// `norms` = enorm, hnorm, attn_norm, ffn_norm, shared_head_norm (host f32). `None` when
    /// a piece is not supported (no device head, an embed / eh_proj type the kernels don't
    /// read): the caller keeps the host cell.
    pub(crate) fn new(
        e: &Glm5Engine,
        layer: usize,
        norms: [&[f32]; 5],
        eps: f32,
    ) -> Result<Option<Self>> {
        let Some(g) = e.gpu.as_ref() else {
            return Ok(None);
        };
        let ge = |er: tang_moe::gpu::Error| anyhow::anyhow!("mtp draft cell: {er}");
        let gpu = &g.devs[0].0;
        gpu.bind().map_err(ge)?;
        let (n, nv) = (e.m.hp.n_embd, e.m.hp.n_vocab);
        let gg = &e.m.g;
        let embd = gg.info("token_embd.weight")?;
        let Some(ty) = embed_ty(embd.ty) else {
            return Ok(None);
        };
        let info = gg.info(&super::super::t(layer, "nextn.eh_proj.weight"))?;
        let eh =
            super::super::moe::gpu::upload(&super::super::moe::W::Gguf(gg, info), 1, n, 2 * n)?;
        eprintln!(
            "glm5: MTP draft cell on the device (eh_proj {} {}x{} row_bytes {}, embed {})",
            info.ty.name(),
            n,
            2 * n,
            eh.w.row_bytes,
            embd.ty.name()
        );
        let f4 = |k: usize| DevBuf::zeroed_elems::<f32>(k).map_err(ge);
        let b = |k: usize| DevBuf::zeroed(k).map_err(ge);
        let up = |v: &[f32]| DevBuf::from_slice(v).map_err(ge);
        // no Q8_0 device head (the tiny f32 fixture): the cell's own head on k5_gemv
        let own_head = if g.head_dev.is_none() {
            let info = match gg.get("output.weight") {
                Some(t) => t,
                None => gg.info("token_embd.weight")?,
            };
            Some(super::super::moe::gpu::upload(
                &super::super::moe::W::Gguf(gg, info),
                1,
                nv,
                n,
            )?)
        } else {
            None
        };
        Ok(Some(Self {
            mk: super::super::moe::gpu::Kernels::new(gpu)?,
            stage_t: DevBuf::zeroed_elems::<u32>(STAGE).map_err(ge)?,
            stage_h: DevBuf::zeroed_elems::<f32>(STAGE * n).map_err(ge)?,
            stage_i: 0,
            warm: false,
            own_head,
            m: gpu.module(SRC).map_err(ge)?,
            s: tang_moe::gpu::Stream::blocking().map_err(ge)?,
            k5: super::super::moe::gpu::Kernels5::new(gpu)?,
            eh,
            embd: (
                DevBuf::from_slice(gg.bytes(embd)).map_err(ge)?,
                embd.row_bytes()? as u64,
                ty,
            ),
            norms: [
                up(norms[0])?,
                up(norms[1])?,
                up(norms[2])?,
                up(norms[3])?,
                up(norms[4])?,
            ],
            eps,
            tp: DevBuf::zeroed_elems::<u32>(2).map_err(ge)?,
            ids: DevBuf::zeroed_elems::<u32>(16).map_err(ge)?,
            hin: f4(n)?,
            // g5f_gemv's token tile may touch 8 activation rows: size for 8
            cat: f4(8 * 2 * n)?,
            cq: b(8 * 2 * n)?,
            cq2: b(8 * 2 * n)?,
            cd: f4(8 * 2 * n / 32)?,
            x: f4(n)?,
            fx: f4(n)?,
            fy: f4(n)?,
            hout: f4(n)?,
            hq8: b(n + n / 32 * 4 + 64)?,
            logits: f4(nv)?,
            graphs: [None, None],
            gen: u64::MAX,
            cap: 0,
            replays: 0,
            dev_pos: None,
        }))
    }

    pub(crate) fn stream(&self) -> &tang_moe::gpu::Stream {
        &self.s
    }

    /// The next staging slot a concurrent observe reads (`[tok]`, `[h; n]`): a ring of
    /// `STAGE` slots, so a later step's stage copy never overwrites a row a queued cell has
    /// not read yet (a cell is ~0.5 ms GPU against a ~9 ms step).
    pub(crate) fn stage_next(&mut self, n: usize) -> (u64, u64) {
        let i = self.stage_i % STAGE;
        self.stage_i += 1;
        (
            self.stage_t.ptr + 4 * i as u64,
            self.stage_h.ptr + (i * n * 4) as u64,
        )
    }

    /// The cell's launches on `s` (capturable).
    #[allow(clippy::too_many_arguments)]
    fn enqueue(
        &self,
        g: &GpuFfn,
        cu: &super::super::mla::cuda::MlaCuda,
        p: &super::super::mla_dev::MlaProj,
        w: &MlaWeights,
        moe: &super::super::moe::gpu::DevMoe,
        n: usize,
        nv: usize,
        sparse: bool,
        s: &tang_moe::gpu::Stream,
    ) -> Result<()> {
        let ge = |er: tang_moe::gpu::Error| anyhow::anyhow!("mtp draft cell: {er}");
        let f = |k: &str| self.m.func(k).map_err(ge);
        let (nf, af, ad, ef) = (f("rs_norm")?, f("rs_argmax")?, f("rs_add")?, f("rs_embed")?);
        let (ni, eps) = (n as i32, self.eps);
        let addg = (n.div_ceil(256) as u32, 1, 1);
        let ni2 = ni;
        let (tab, rb, ty) = (self.embd.0.ptr, self.embd.1, self.embd.2);
        let (tpp, cat, cat_h, hin) = (
            self.tp.ptr,
            self.cat.ptr,
            self.cat.ptr + (n * 4) as u64,
            self.hin.ptr,
        );
        let one = 1i32;
        let nw = |i: usize| self.norms[i].ptr;
        let dbg = std::env::var("TANG_GLM5_MTP_CELL_EAGER").is_ok_and(|v| v == "1");
        let chk = |what: &str| -> Result<()> {
            if dbg {
                s.sync()
                    .map_err(|er| anyhow::anyhow!("draft cell stage {what}: {er}"))?;
            }
            Ok(())
        };
        unsafe {
            launch(
                ef,
                addg,
                (256, 1, 1),
                0,
                s,
                args![tab, rb, ty, tpp, cat, ni, one],
            )?;
            let (w0, w1) = (nw(0), nw(1));
            launch(
                nf,
                (1, 1, 1),
                (1024, 1, 1),
                0,
                s,
                args![cat, w0, cat, ni, eps],
            )?;
            launch(
                nf,
                (1, 1, 1),
                (1024, 1, 1),
                0,
                s,
                args![hin, w1, cat_h, ni, eps],
            )?;
        }
        chk("embed+norms")?;
        // eh_proj: Q8_0 on the k5 q8 + gemv8 pair (the trunk's KDA / head projection
        // kernels; g5f_gemv faulted on this 4096x8192 Q8_0 matrix, not diagnosed), else g5f_gemv
        if self.eh.w.ty == 8 && !std::env::var("TANG_GLM5_MTP_EH_G5F").is_ok_and(|v| v == "1") {
            let (cq, xp) = (self.cq.ptr, self.x.ptr);
            let n2 = (2 * n) as u32;
            unsafe {
                launch(
                    self.k5.q8,
                    ((2 * n / 32) as u32, 1, 1),
                    (32, 1, 1),
                    0,
                    s,
                    args![cat, cq, n2, 1u32],
                )?;
                launch(
                    self.k5.gemv8,
                    (n.div_ceil(K5_WARPS) as u32, 1, 1),
                    (128, 1, 1),
                    (K5_WARPS * ((self.eh.w.row_bytes as usize).div_ceil(16) + 1) * 16) as u32,
                    s,
                    args![
                        self.eh.w,
                        0u32,
                        cq,
                        cq + 2 * n as u64,
                        xp,
                        n as u32,
                        1u32,
                        n as u32,
                        n2
                    ],
                )?;
            }
        } else {
            let quant = self.eh.w.ty > 1;
            if quant {
                self.k5
                    .fast
                    .q8(s, cat, self.cq.ptr, self.cq2.ptr, self.cd.ptr, 2 * n, 1)?;
            }
            let a = super::super::moe::gpu::Act {
                xf: cat,
                xq: if quant { self.cq.ptr } else { 0 },
                xq2: if quant { self.cq2.ptr } else { 0 },
                xd: if quant { self.cd.ptr } else { 0 },
                xs: 2 * n,
            };
            self.k5
                .fast
                .gemv(s, self.eh.w, 0, a, self.x.ptr, n, 1, n, 2 * n)?;
        }
        chk("eh_proj")?;
        let (xp, xa, w2) = (self.x.ptr, p.x_ptr(1)?, nw(2));
        unsafe {
            launch(
                nf,
                (1, 1, 1),
                (1024, 1, 1),
                0,
                s,
                args![xp, w2, xa, ni, eps],
            )?
        };
        cu.set_seq(0);
        let y = p.window_dev_pos(cu, w, tpp + 4, sparse, s)?;
        chk("dsa")?;
        let (fx, fy, w3, w4, ho) = (self.fx.ptr, self.fy.ptr, nw(3), nw(4), self.hout.ptr);
        unsafe {
            launch(ad, addg, (256, 1, 1), 0, s, args![xp, y, ni2])?;
            launch(
                nf,
                (1, 1, 1),
                (1024, 1, 1),
                0,
                s,
                args![xp, w3, fx, ni, eps],
            )?;
        }
        super::super::moe::gpu::ffn_async(&self.mk, s, moe, fx, fy, 1)?;
        chk("ffn")?;
        let hq = self.hq8.ptr;
        let lg = self.logits.ptr;
        unsafe {
            launch(ad, addg, (256, 1, 1), 0, s, args![xp, fy, ni2])?;
            launch(
                nf,
                (1, 1, 1),
                (1024, 1, 1),
                0,
                s,
                args![xp, w4, ho, ni, eps],
            )?;
            match (g.head_dev.as_ref(), self.own_head.as_ref()) {
                (Some(hd), _) => {
                    launch(
                        hd.k5.q8,
                        ((n / 32) as u32, 1, 1),
                        (32, 1, 1),
                        0,
                        s,
                        args![ho, hq, n as u32, 1u32],
                    )?;
                    launch(
                        hd.k5.gemv8,
                        (nv.div_ceil(K5_WARPS) as u32, 1, 1),
                        (128, 1, 1),
                        (K5_WARPS * ((hd.dw.w.row_bytes as usize).div_ceil(16) + 1) * 16) as u32,
                        s,
                        args![
                            hd.dw.w,
                            0u32,
                            hq,
                            hq + n as u64,
                            lg,
                            nv as u32,
                            1u32,
                            nv as u32,
                            n as u32
                        ],
                    )?;
                }
                (None, Some(oh)) => {
                    launch(
                        self.k5.gemv,
                        (nv.div_ceil(128) as u32, 1, 1),
                        (128, 1, 1),
                        0,
                        s,
                        args![oh.w, 0u32, ho, lg, nv as u32, 1u32, nv as u32, n as u32],
                    )?;
                }
                (None, None) => anyhow::bail!("draft cell: no head"),
            }
            let (nvi, inc) = (nv as i32, 1i32);
            launch(af, (1, 1, 1), (1024, 1, 1), 0, s, args![lg, nvi, tpp, inc])?;
        }
        tang_moe::gpu::copy_d2d_async(hin, ho, n * 4, s).map_err(ge)?;
        Ok(())
    }

    /// Run one cell at the drafter cache position `pos` (`c`'s rows): when `tok_h` is given
    /// the token / `h` row are uploaded first (a teacher-forced cell), else the chain
    /// continues from the device (`tp[0]`, `hin`). Async (no sync). `false`: the cell can't
    /// run here (cache capacity); nothing was enqueued.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &mut self,
        e: &Glm5Engine,
        cu: &super::super::mla::cuda::MlaCuda,
        p: &super::super::mla_dev::MlaProj,
        w: &MlaWeights,
        moe: &super::super::moe::gpu::DevMoe,
        c: &mut DsaCache,
        tok_h: Option<(u32, &[f32])>,
        dev_src: Option<(u64, u64)>,
        after: Option<&tang_moe::gpu::Event>,
    ) -> Result<bool> {
        let ge = |er: tang_moe::gpu::Error| anyhow::anyhow!("mtp draft cell: {er}");
        let g = e.gpu.as_ref().context("draft cell: no GPU")?;
        let (n, nv) = (e.m.hp.n_embd, e.m.hp.n_vocab);
        let (dl, idd) = (w.kv_lora, w.idx_dim);
        let pos = c.valid.len();
        if c.lat16.len() != pos * dl {
            return Ok(false);
        }
        let kv_len = pos + 1;
        cu.set_seq(0);
        let gen = GRAPH_GEN.load(std::sync::atomic::Ordering::Relaxed);
        if gen != self.gen || self.cap < kv_len {
            self.graphs = [None, None];
            self.gen = gen;
            let cap = cu.row_cap();
            if cap < kv_len {
                return Ok(false);
            }
            cu.reserve_pool(cap / w.kpool + 1).map_err(ge)?;
            cu.stage_ape(&w.i_ape).map_err(ge)?;
            self.cap = cap;
        }
        let sparse = !super::super::mla::indexer_is_dense(kv_len, w.topk);
        let sp = usize::from(sparse);
        // the first cell runs eagerly: its private MoE pool allocates (not capturable)
        let eager = std::env::var("TANG_GLM5_MTP_CELL_EAGER").is_ok_and(|v| v == "1") || !self.warm;
        if self.graphs[sp].is_none() && !eager {
            let me: &Self = self;
            let gr = tang_moe::gpu::Graph::capture(&me.s, |cs| {
                me.enqueue(g, cu, p, w, moe, n, nv, sparse, cs)
                    .map_err(|er| tang_moe::gpu::Error(format!("{er:#}")))
            })
            .map_err(|er| anyhow::anyhow!("draft cell capture: {er}"))?;
            self.graphs[sp] = Some(gr);
        }
        let s = &self.s;
        if let Some(ev) = after {
            s.wait(ev).map_err(ge)?;
        }
        match tok_h {
            _ if dev_src.is_some() => {
                // teacher-forced from device rows (the verify window's ids / hn)
                let (tk, hp) = dev_src.unwrap();
                tang_moe::gpu::copy_d2d_async(self.tp.ptr, tk, 4, s).map_err(ge)?;
                tang_moe::gpu::copy_d2d_async(self.hin.ptr, hp, n * 4, s).map_err(ge)?;
                if self.dev_pos != Some(pos) {
                    self.tp.write_async(4, &[pos as u32], s).map_err(ge)?;
                }
            }
            Some((tok, h)) => {
                self.tp.write_async(0, &[tok, pos as u32], s).map_err(ge)?;
                self.hin.write_async(0, h, s).map_err(ge)?;
            }
            None => {
                // the chain: tp[1] was advanced by the last cell's argmax. A pageable H2D
                // here would wait for the stream (one GPU cell of host stall per cell), so
                // the position is written only when it is out of step (after a truncate).
                if self.dev_pos != Some(pos) {
                    self.tp.write_async(4, &[pos as u32], s).map_err(ge)?;
                }
            }
        }
        if sparse {
            let np = pos / w.kpool;
            if cu.pool_upto() < np {
                cu.idx_pool(&c.idx_k, &c.idx_g, &w.i_ape, pos, 0, np, &mut [], s)
                    .map_err(ge)?;
            }
        }
        if eager {
            self.enqueue(g, cu, p, w, moe, n, nv, sparse, s)?;
            s.sync().map_err(|er| {
                anyhow::anyhow!("draft cell (eager, pos {pos}, sparse {sparse}): {er}")
            })?;
        } else {
            self.graphs[sp].as_ref().unwrap().launch(s).map_err(ge)?;
        }
        self.replays += 1;
        self.warm = true;
        self.dev_pos = Some(kv_len);
        c.lat16.resize(kv_len * dl, 0);
        c.idx_k.resize(kv_len * idd, 0.0);
        c.idx_g.resize(kv_len * idd, 0.0);
        c.valid.resize(kv_len, 0);
        c.stale_from = Some(c.stale_from.map_or(pos, |f| f.min(pos)));
        cu.append_done(pos, 1);
        if sparse {
            cu.set_pool_upto(kv_len / w.kpool);
        }
        Ok(true)
    }
}

/// Before a host-staged drafter cell: pull the rows the device cells left only on the
/// device into the drafter cache's mirrors.
pub(crate) fn refresh_cell_mirrors(
    cu: &super::super::mla::cuda::MlaCuda,
    c: &mut DsaCache,
) -> Result<()> {
    let Some(from) = c.stale_from else {
        return Ok(());
    };
    let rows = c.valid.len();
    cu.set_seq(0);
    if rows > from {
        cu.refresh_mirrors(
            from,
            rows - from,
            &mut c.lat16,
            &mut c.idx_k,
            &mut c.idx_g,
            &mut c.valid,
        )
        .map_err(|er| anyhow::anyhow!("mtp mirror refresh: {er}"))?;
    }
    c.stale_from = None;
    Ok(())
}

#[cfg(test)]
mod draft_cell_tests {
    use super::super::super::moe::gpu::{Act, Kernels5, G5W};
    use tang_moe::gpu::{DevBuf, Gpu, Stream};

    /// The MTP eh_proj shape (4096 x 8192 Q8_0) on `g5f_gemv` (one token, the draft cell's
    /// call) against a host dot over the same int8 activations: the round-4 fault repro.
    #[test]
    fn g5f_gemv_q8_4096x8192() {
        let Some(gpu) = Gpu::for_test() else { return };
        let (rows, cols) = (4096usize, 8192usize);
        let rb = cols / 32 * 34;
        let mut w = vec![0u8; rows * rb];
        let mut st = 12345u32;
        let mut rnd = || {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            st
        };
        for b in w.chunks_mut(34) {
            b[0] = 0x00;
            b[1] = 0x20; // f16 2^-7
            for q in &mut b[2..] {
                *q = (rnd() % 31) as u8;
            }
        }
        let x: Vec<f32> = (0..cols)
            .map(|_| (rnd() % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let k5 = Kernels5::new(&gpu).unwrap();
        let wd = DevBuf::from_slice(&w).unwrap();
        let g5 = G5W {
            base: wd.ptr,
            estride: (rows * rb) as u64,
            row_bytes: rb as u32,
            ty: 8,
        };
        let xb = DevBuf::from_slice(&x).unwrap();
        let (xq, xq2, xd) = (
            DevBuf::zeroed(8 * cols).unwrap(),
            DevBuf::zeroed(8 * cols).unwrap(),
            DevBuf::zeroed(8 * cols / 8).unwrap(),
        );
        let y = DevBuf::zeroed(rows * 4).unwrap();
        let s = Stream::blocking().unwrap();
        k5.fast
            .q8(&s, xb.ptr, xq.ptr, xq2.ptr, xd.ptr, cols, 1)
            .unwrap();
        let a = Act {
            xf: xb.ptr,
            xq: xq.ptr,
            xq2: xq2.ptr,
            xd: xd.ptr,
            xs: cols,
        };
        k5.fast
            .gemv(&s, g5, 0, a, y.ptr, rows, 1, rows, cols)
            .unwrap();
        s.sync().expect("g5f_gemv 4096x8192 Q8_0");
        let got = y.read::<f32>(0, rows).unwrap();
        // host reference in f32 over the dequantised weights
        let mut worst = 0f32;
        for r in 0..rows {
            let mut acc = 0f64;
            for c in 0..cols {
                let b = &w[r * rb + (c / 32) * 34..];
                let q = b[2 + c % 32] as i8 as f64;
                acc += q * (1.0 / 128.0) * x[c] as f64;
            }
            worst = worst.max((got[r] - acc as f32).abs() / (acc.abs() as f32).max(1.0));
        }
        eprintln!("g5f_gemv 4096x8192 Q8_0: worst rel {worst:.3e}");
        assert!(worst < 2e-2, "worst {worst}");
    }
}
