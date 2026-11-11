// Device-side selection (appended to kernels.cu in one NVRTC module): the indexer's masks,
// candidate blocks and top-k, MoE routing lists, and attention index rows, so a decode step
// has no host round trip for them. Each mirrors forward.rs (`indexer`, `candidates`,
// `topk_stable`, `moe`): ties to the LOWEST index, selections emitted in ascending order.

#define DS_SEL_THREADS 512

// Order-preserving key: a > b as floats <=> key(a) > key(b). ±0 share a key (they tie in
// Rust's partial_cmp).
__device__ __forceinline__ unsigned int ds_key(float x) {
    if (x == 0.f) x = 0.f;
    unsigned int u = __float_as_uint(x);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Prefix lens of row `ti`: causal prefill sees (ti + 1) / r compressed entries, decode all.
__device__ __forceinline__ int ds_len(int ti, int width, int causal, int r) {
    return causal ? (ti + 1) / r : width;
}

// Block-wide exclusive prefix of `flag` (blockDim.x == DS_SEL_THREADS); returns the block
// total in *total. `ws` holds DS_SEL_THREADS / 32 + 1 ints.
__device__ int ds_block_scan(int flag, int* ws, int* total) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    unsigned int b = __ballot_sync(0xffffffffu, flag);
    int pre = __popc(b & ((1u << lane) - 1u));
    __syncthreads();
    if (lane == 0) ws[w] = __popc(b);
    __syncthreads();
    if (threadIdx.x == 0) {
        int s = 0;
        for (int i = 0; i < nw; i++) {
            int c = ws[i];
            ws[i] = s;
            s += c;
        }
        ws[nw] = s;
    }
    __syncthreads();
    pre += ws[w];
    *total = ws[nw];
    return pre;
}

// One block: the `k` largest of v[0..n) (ties to the lowest index), written to sel[0..k)
// in ascending index order. Radix select on the keys finds the k-th largest key T; then
// everything above T plus the first (k - #above) equal to T, in index order.
__device__ void ds_select_row(const float* v, int n, int k, int* sel) {
    __shared__ int hist[256];
    __shared__ int ws[DS_SEL_THREADS / 32 + 1];
    __shared__ unsigned int s_prefix;
    __shared__ int s_kk;
    if (k <= 0) return;
    unsigned int prefix = 0, mask = 0;
    int kk = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            unsigned int key = ds_key(v[i]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int b = 255;
            for (; b > 0; b--) {
                if (hist[b] >= kk) break;
                kk -= hist[b];
            }
            s_prefix = prefix | ((unsigned int)b << shift);
            s_kk = kk;
        }
        __syncthreads();
        prefix = s_prefix;
        kk = s_kk;
        mask |= 255u << shift;
        __syncthreads();
    }
    // prefix = T exactly; kk = how many of the keys equal to T to take
    int base = 0, eq_seen = 0;
    for (int c0 = 0; c0 < n; c0 += blockDim.x) {
        int i = c0 + threadIdx.x;
        unsigned int key = i < n ? ds_key(v[i]) : 0u;
        int eq = i < n && key == prefix;
        int eq_tot;
        int eq_pre = ds_block_scan(eq, ws, &eq_tot);
        int take = i < n && (key > prefix || (eq && eq_seen + eq_pre < kk));
        int tot;
        int pos = ds_block_scan(take, ws, &tot);
        if (take) sel[base + pos] = i;
        base += tot;
        eq_seen += eq_tot;
    }
    __syncthreads();
}

// In place on score [t][width]: -inf past each row's causal prefix (when `causal`), and
// where the candidate mask cand [t][width] (u8; null for none) is 0.
extern "C" __global__ void ds_mask_scores(float* score, const unsigned char* cand, int t, int width, int causal,
                                          int r) { DS_PDL_ENTRY
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long long)t * width) return;
    int ti = (int)(e / width), j = (int)(e % width);
    if ((causal && j >= ds_len(ti, width, causal, r)) || (cand && !cand[e])) score[e] = -DS_INF;
}

// Candidate blocks, one block per row: block maxima of `bs` (scratch sc [t][nb]), the block
// holding the row's newest position pinned to +inf, the top `kb` blocks kept (-inf blocks
// never), expanded to a mask cand [t][width].
extern "C" __global__ void ds_candidates(const float* score, float* sc, int* scratch, unsigned char* cand, int t,
                                         int width, int causal, int r, int bs, int kb) { DS_PDL_ENTRY
    int ti = blockIdx.x;
    if (ti >= t) return;
    int nb = (width + bs - 1) / bs;
    const float* row = score + (long long)ti * width;
    float* s = sc + (long long)ti * nb;
    int* sel = scratch + (long long)ti * nb;
    int len = ds_len(ti, width, causal, r);
    int last = len >= 1 ? (len - 1) / bs : -1;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        float m = -DS_INF;
        int hi = min((b + 1) * bs, width);
        for (int j = b * bs; j < hi; j++) m = fmaxf(m, row[j]);
        s[b] = b == last ? DS_INF : m;
    }
    __syncthreads();
    int k = min(kb, nb);
    ds_select_row(s, nb, k, sel);
    unsigned char* out = cand + (long long)ti * width;
    for (int j = threadIdx.x; j < width; j += blockDim.x) out[j] = 0;
    __syncthreads();
    for (int q = 0; q < k; q++) {
        int b = sel[q];
        if (!(s[b] > -DS_INF)) continue;
        int hi = min((b + 1) * bs, width);
        for (int j = b * bs + threadIdx.x; j < hi; j += blockDim.x) out[j] = 1;
    }
}

// Indexer top-k, one block per row: the k largest of score [t][width] in ascending index
// order, as out[ti * stride + q] = idx + offset, or -1 when idx is past the row's prefix.
// `scratch` holds [t][k] ints.
extern "C" __global__ void ds_topk(const float* score, int* scratch, int* out, int t, int width, int k, int stride,
                                   int offset, int causal, int r) { DS_PDL_ENTRY
    int ti = blockIdx.x;
    if (ti >= t) return;
    int* sel = scratch + (long long)ti * k;
    ds_select_row(score + (long long)ti * width, width, k, sel);
    int len = ds_len(ti, width, causal, r);
    for (int q = threadIdx.x; q < k; q += blockDim.x) {
        int i = sel[q];
        out[(long long)ti * stride + q] = i < len ? i + offset : -1;
    }
}

// Attention window rows into out [t][stride] at column 0:
//   mode 0, prefill: row e sees positions max(e - win + 1, 0) .. e (n = min(t, win) columns,
//     -1 past e);
//   mode 1, decode at sp: the ring oldest first, -1 for slots not yet written;
//   mode 3, verify window at sp (kv = the pre-chunk ring then the chunk's t rows): row e sees
//     positions max(sp + e - win + 1, 0) .. sp + e ascending, -1 past them;
//   mode 2, DSpark at sp: ring slots 0 .. min(win, sp + 1), then win .. win + b (the block).
extern "C" __global__ void ds_win_idx(int* out, int t, int stride, int n, int mode, int sp, int win, int b) { DS_PDL_ENTRY
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= t * n) return;
    int row = e / n, j = e % n, v;
    if (mode == 0) {
        int lo = max(row - win + 1, 0);
        v = lo + j > row ? -1 : lo + j;
    } else if (mode == 1) {
        int i = (sp % win + 1 + j) % win;
        v = i > sp ? -1 : i;
    } else if (mode == 3) {
        int q = sp + row, cnt = min(win, q + 1), pos = q + 1 - cnt + j;
        v = j >= cnt ? -1 : (pos < sp ? pos % win : win + pos - sp);
    } else {
        int n1 = min(win, sp + 1);
        v = j < n1 ? j : win + (j - n1);
    }
    out[(long long)row * stride + j] = v;
}

// dst[row * dstride + col + j] = src[row * k + j], j < k.
extern "C" __global__ void ds_idx_cat(int* dst, const int* src, int t, int k, int dstride, int col) { DS_PDL_ENTRY
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= t * k) return;
    dst[(long long)(e / k) * dstride + col + e % k] = src[e];
}

// MoE routing lists, one block: ids/wts [t][k] -> per-expert CSR, rows ascending within an
// expert: counts[ne], off[ne + 1], rows[t * k] (token), rw[t * k] (route weight).
#define ROUTE_NE 512  // experts at most for ds_route (256 threads, 2 each)
// ds_route's body for one block of 256 threads: counts in shared memory and a block scan for the
// offsets (the old serial prefix over ne global reads was ~15 µs alone), then the stable fill.
__device__ __forceinline__ void route_body(const int* ids, const float* wts, int* counts, int* off, int* rows,
                                           float* rw, int t, int k, int ne) {
    __shared__ int sc[ROUTE_NE];
    __shared__ int wsum[8];
    int n = t * k, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    for (int e = tid; e < ne; e += blockDim.x) {
        int c = 0;
        for (int i = 0; i < n; i++) c += ids[i] == e;
        sc[e] = c;
        counts[e] = c;
    }
    __syncthreads();
    // each thread owns ROUTE_NE / 256 consecutive experts: local sums, warp scan, block offsets
    const int per = ROUTE_NE / 256;
    int e0 = tid * per, loc = 0;
    for (int j = 0; j < per; j++) loc += e0 + j < ne ? sc[e0 + j] : 0;
    int inc = loc;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int u = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += u;
    }
    if (lane == 31) wsum[wid] = inc;
    __syncthreads();
    int base = 0;
    for (int q = 0; q < wid; q++) base += wsum[q];
    int run = base + inc - loc;
    for (int j = 0; j < per; j++) {
        int e = e0 + j;
        if (e < ne) {
            off[e] = run;
            run += sc[e];
        }
    }
    if (tid == 255) off[ne] = run;
    __syncthreads();
    for (int e = tid; e < ne; e += blockDim.x) {
        int o = off[e];
        if (!sc[e]) continue;
        for (int i = 0; i < n; i++)
            if (ids[i] == e) {
                rows[o] = i / k;
                rw[o] = wts[i];
                o++;
            }
    }
}

extern "C" __global__ void ds_route(const int* ids, const float* wts, int* counts, int* off, int* rows, float* rw,
                                    int t, int k, int ne) { DS_PDL_ENTRY
    route_body(ids, wts, counts, off, rows, rw, t, k, ne);
}

// ds_moe_plan's list for one block of any size: the live local experts (counts > 0, mask) in
// ascending id into elist[0..cap), then -1. Each thread owns a run of consecutive experts; a
// block scan of the run counts gives the positions (the old one-thread loop over ne global
// reads was serial).
__device__ __forceinline__ void plan_body(const int* counts, const int* mask, int* elist, int ne, int cap) {
    __shared__ int pw[32];
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, nt = blockDim.x;
    int per = (ne + nt - 1) / nt, e0 = tid * per, loc = 0;
    for (int j = 0; j < per; j++) {
        int e = e0 + j;
        loc += e < ne && counts[e] > 0 && mask[e];
    }
    int inc = loc;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int u = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += u;
    }
    if (lane == 31) pw[wid] = inc;
    __syncthreads();
    int base = 0;
    for (int q = 0; q < wid; q++) base += pw[q];
    int pos = base + inc - loc, tot = 0;
    for (int q = 0; q < (nt + 31) / 32; q++) tot += pw[q];
    for (int j = 0; j < per && pos < cap; j++) {
        int e = e0 + j;
        if (e < ne && counts[e] > 0 && mask[e]) elist[pos++] = e;
    }
    for (int j = tot + tid; j < cap; j += nt) elist[j] = -1;
}

// ds_route, then ds_moe_plan, in block 0; blocks 1.. are ds_ks_quant of x's ng 32-groups (the
// grouped K-split GU's activation, independent of the routing): three launches in one.
extern "C" __global__ void ds_route_plan_q(const int* ids, const float* wts, int* counts, int* off, int* rows,
                                           float* rw, int t, int k, int ne, const int* mask, int* elist, int cap,
                                           const float* x, unsigned char* codes, float* sx, int ng) { DS_PDL_ENTRY
    if (blockIdx.x == 0) {
        route_body(ids, wts, counts, off, rows, rw, t, k, ne);
        __syncthreads();
        plan_body(counts, mask, elist, ne, cap);
        return;
    }
    int gw = ((blockIdx.x - 1) * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (gw >= ng) return;
    float xv = x[(long long)gw * 32 + lane];
    float a = fabsf(xv);
#pragma unroll
    for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    float sc = pow2_ceil(fmaxf(a, 1e-4f) / 448.f);
    codes[(long long)gw * 32 + lane] = (unsigned char)e4m3_byte(round_e4m3(fminf(fmaxf(xv / sc, -448.f), 448.f)));
    if (lane == 0) sx[gw] = sc;
}

// Expert e's rows of x [.][d] into xs [cap][d] and its route weights into w [cap]; rows past
// the expert's count are zero (cap is the host's bound, t).
extern "C" __global__ void ds_gather_csr(const float* x, const int* off, const int* rows, const float* rw, float* xs,
                                         float* w, int e, int cap, int d) { DS_PDL_ENTRY
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)cap * d) return;
    int j = (int)(i / d), n = off[e + 1] - off[e];
    bool live = j < n;
    xs[i] = live ? x[(long long)rows[off[e] + j] * d + i % d] : 0.f;
    if (i % d == 0) w[j] = live ? rw[off[e] + j] : 0.f;
}

// y[rows[j]] += v[j] for expert e's rows only.
extern "C" __global__ void ds_scatter_csr(float* y, const float* v, const int* off, const int* rows, int e, int cap,
                                          int d) { DS_PDL_ENTRY
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int n = off[e + 1] - off[e];
    if (i >= (long long)n * d) return;
    y[(long long)rows[off[e] + i / d] * d + i % d] += v[i];
}

// ds_linear limited to expert e's row count (rows past it are left unwritten).
extern "C" __global__ void ds_linear_csr(DsW w, const float* x, float* y, int t, int round, const int* off, int e) { DS_PDL_ENTRY
    int n_rows = min(t, off[e + 1] - off[e]);
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (warp >= n_rows * w.rows) return;
    int ti = warp / w.rows, n = warp % w.rows;
    const float* xr = x + (long long)ti * w.cols;
    float s = 0.f;
    for (int k = lane; k < w.cols; k += 32) s += xr[k] * wget(w, n, k);
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) y[(long long)ti * w.rows + n] = round ? bf(s) : s;
}

// Grouped MoE plan (plan_body, any block size): elist[0..cap) = the experts with rows that this rank owns
// (mask[e] != 0), ascending, then -1.
extern "C" __global__ void ds_moe_plan(const int* counts, const int* mask, int* elist, int ne, int cap) { DS_PDL_ENTRY
    if (blockIdx.x) return;
    plan_body(counts, mask, elist, ne, cap);
}

// The 8-row tiles of the elist experts' CSR segments, one thread: tiles[0..cap) = e | c << 16
// for each slot's expert e and chunk c < ceil(rows / 8), then -1 (cap >= slots + rows / 8).
extern "C" __global__ void ds_moe_plan_tiles(const int* elist, const int* off, int* tiles, int slots, int cap) { DS_PDL_ENTRY
    if (threadIdx.x || blockIdx.x) return;
    int j = 0;
    for (int s = 0; s < slots; s++) {
        int e = elist[s];
        if (e < 0) break;
        int nc = (off[e + 1] - off[e] + 7) / 8;
        for (int c = 0; c < nc && j < cap; c++) tiles[j++] = e | (c << 16);
    }
    for (; j < cap; j++) tiles[j] = -1;
}

// y [t][d] = Σ over token ti's local experts in ascending id of part[slot] (slot = the token's
// CSR position under that expert): the per-expert loop's scatter order, from 0. ds_moe_reduce_sh
// then adds the shared expert's row part[t * k + ti] (moe_finish's routed + shared, unrounded).
__device__ __forceinline__ void moe_reduce_at(float* y, const float* part, const int* ids, const int* off,
                                              const int* rows, const int* mask, int t, int k, int d, int sh) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)t * d) return;
    int ti = (int)(i / d), c = (int)(i % d);
    const int* id = ids + (long long)ti * k;
    float s = 0.f;
    int done = -1;
    for (int q = 0; q < k; q++) {
        int best = 0x7fffffff;
        for (int r = 0; r < k; r++)
            if (id[r] > done && id[r] < best) best = id[r];
        if (best == 0x7fffffff) break;
        done = best;
        if (!mask[best]) continue;
        int o = off[best], n = off[best + 1] - o, p = o;
        for (int j = 0; j < n; j++)
            if (rows[o + j] == ti) {
                p = o + j;
                break;
            }
        s += part[(long long)p * d + c];
    }
    if (sh) s += part[((long long)t * k + ti) * d + c];
    y[i] = s;
}
extern "C" __global__ void ds_moe_reduce(float* y, const float* part, const int* ids, const int* off, const int* rows,
                                         const int* mask, int t, int k, int d) { DS_PDL_ENTRY
    moe_reduce_at(y, part, ids, off, rows, mask, t, k, d, 0);
}
extern "C" __global__ void ds_moe_reduce_sh(float* y, const float* part, const int* ids, const int* off,
                                            const int* rows, const int* mask, int t, int k, int d, int sh) { DS_PDL_ENTRY
    moe_reduce_at(y, part, ids, off, rows, mask, t, k, d, sh);
}

// n ints set to v (a verify window's [t][k] selection, -1 = no entry).
extern "C" __global__ void ds_fill_i32(int* p, int v, long long n) { DS_PDL_ENTRY
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

// n bytes (candidate mask rows).
extern "C" __global__ void ds_copy_u8(unsigned char* dst, const unsigned char* src, long long n) { DS_PDL_ENTRY
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// DSpark's greedy Markov draw on the device: row += bias (when given; the Markov bias, as
// ds_add), ids[slot] = argmax(row) (the first maximum, as the host `argmax`), or with
// row == nullptr keep ids[slot] (first set to set[0] when given); then, when memb != nullptr, the
// Markov embedding row of that id: memb[0..rank] = tab[id * rank ..]. One block of 1024.
extern "C" __global__ void ds_spark_pick(float* row, int v, int* ids, int slot, const float* tab, int rank,
                                         float* memb, const float* bias, const int* set) { DS_PDL_ENTRY
    __shared__ float sv[1024];
    __shared__ int si[1024];
    __shared__ int pick;
    int tid = threadIdx.x;
    if (row) {
        float bv = -DS_INF;
        int bi = 0x7fffffff;
        for (int j = tid; j < v; j += blockDim.x) {
            float x = row[j];
            if (bias) row[j] = x = x + bias[j];
            if (x > bv) {
                bv = x;
                bi = j;
            }
        }
        sv[tid] = bv;
        si[tid] = bi;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float ov = sv[tid + s];
                int oi = si[tid + s];
                if (ov > sv[tid] || (ov == sv[tid] && oi < si[tid])) {
                    sv[tid] = ov;
                    si[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            int a = si[0] == 0x7fffffff ? 0 : si[0];
            ids[slot] = a;
            pick = a;
        }
    } else if (tid == 0) {
        if (set) ids[slot] = set[0];
        pick = ids[slot];
    }
    __syncthreads();
    if (memb)
        for (int j = tid; j < rank; j += blockDim.x) memb[j] = tab[(long long)pick * rank + j];
}

// Block-wide exclusive scan of v over 128 threads (4 warps); returns the total in *tot.
__device__ __forceinline__ int gate_scan128(int v, int* wsum, int* tot) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int u = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += u;
    }
    if (lane == 31) wsum[wid] = inc;
    __syncthreads();
    int base = 0;
    for (int q = 0; q < wid; q++) base += wsum[q];
    *tot = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    __syncthreads();
    return base + inc - v;
}

#define GATE_NE 512   // experts at most (16 a lane)
#define GATE_N 128    // t * k routes at most

// The decode router in one launch (t <= 8, k <= 16, ne <= 512, 128 threads a block):
// 1. the gate GEMV raw [t][ne] = x · wᵀ, warp per expert row with ds_linear_t8's per-lane
//    order and shuffle tree (so raw is bit-identical to ds_linear / ds_linear_t8);
// 2. the last block to arrive (cnt, left at zero) runs ds_router per token (warp per token,
//    scores in registers; the same score math, ties to the lower id, the weight sum serial in
//    selection order), then ds_route's CSR (counts, off, rows ascending within an expert, rw)
//    with a block scan, then (elist non-null) ds_moe_plan's live local expert list.
extern "C" __global__ void __launch_bounds__(128) ds_moe_gate(DsW w, const float* __restrict__ x, float* raw,
                                                             const float* __restrict__ bias, int* ids, float* wts,
                                                             int* counts, int* off, int* rows, float* rw,
                                                             const int* __restrict__ mask, int* elist, int cap, int t,
                                                             int ne, int k, int norm, float route_scale, int* cnt) { DS_PDL_ENTRY
    __shared__ int sid[GATE_N];
    __shared__ float swt[GATE_N];
    __shared__ int soff[GATE_NE + 1];
    __shared__ int wsum[4];
    __shared__ int last;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    {
        int n = blockIdx.x * 4 + wid;
        if (n < w.rows) {
            row_prefetch(w, n, lane);
            float s[8];
#pragma unroll
            for (int i = 0; i < 8; i++) s[i] = 0.f;
            int kk = lane;
            if (w.fmt <= 1) {
                const int U = 8;
                for (; kk + 32 * (U - 1) < w.cols; kk += 32 * U) {
                    float wv[U];
#pragma unroll
                    for (int u = 0; u < U; u++) wv[u] = wget(w, n, kk + 32 * u);
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        if (i >= t) break;
                        const float* xr = x + (long long)i * w.cols + kk;
                        float xv[U];
#pragma unroll
                        for (int u = 0; u < U; u++) xv[u] = xr[32 * u];
#pragma unroll
                        for (int u = 0; u < U; u++) s[i] = fmaf(xv[u], wv[u], s[i]);
                    }
                }
            }
            for (; kk < w.cols; kk += 32) {
                float wv = wget(w, n, kk);
#pragma unroll
                for (int i = 0; i < 8; i++)
                    if (i < t) s[i] = fmaf(x[(long long)i * w.cols + kk], wv, s[i]);
            }
#pragma unroll
            for (int i = 0; i < 8; i++) {
                if (i >= t) break;
                float v = s[i];
                for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                if (lane == 0) raw[(long long)i * w.rows + n] = v;
            }
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(cnt, 1) == (int)gridDim.x - 1;
        if (last) *cnt = 0;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    // router: warp per token, lane holds experts lane + 32 j
    for (int ti = wid; ti < t; ti += 4) {
        const float* r = raw + (long long)ti * ne;
        float sv[GATE_NE / 32];
        unsigned used = 0;
#pragma unroll
        for (int j = 0; j < GATE_NE / 32; j++) {
            int e = lane + 32 * j;
            sv[j] = 0.f;
            if (e < ne) {
                float xv = __ldcg(r + e);
                float sc = sqrtf(xv > 20.f ? xv : log1pf(expf(xv)));
                sv[j] = sc + bias[e];
            }
        }
        int sel_l = -1;  // lane j holds selection j
        for (int j = 0; j < k; j++) {
            int be = -1;
            float bv = 0.f;
#pragma unroll
            for (int q = 0; q < GATE_NE / 32; q++) {
                int e = lane + 32 * q;
                if (e < ne && !(used >> q & 1) && router_better(sv[q], e, bv, be)) { be = e; bv = sv[q]; }
            }
            for (int o = 16; o; o >>= 1) {
                float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                int oe = __shfl_xor_sync(0xffffffffu, be, o);
                if (router_better(ov, oe, bv, be)) { be = oe; bv = ov; }
            }
            if (be >= 0 && (be & 31) == lane) used |= 1u << (be >> 5);
            if (lane == j) sel_l = be;
        }
        // gather selections to every lane, weights serial in lane 0
        int selv[16];
#pragma unroll
        for (int j = 0; j < 16; j++) selv[j] = __shfl_sync(0xffffffffu, sel_l, j);
        if (lane == 0) {
            float wl[16], sum = 0.f;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                if (j >= k) break;
                float xv = __ldcg(r + selv[j]);
                wl[j] = sqrtf(xv > 20.f ? xv : log1pf(expf(xv)));
                sum += wl[j];
            }
#pragma unroll
            for (int j = 0; j < 16; j++) {
                if (j >= k) break;
                float v = (norm && k > 1) ? wl[j] / (sum + 1e-20f) : wl[j];
                ids[ti * k + j] = selv[j];
                wts[ti * k + j] = v * route_scale;
                sid[ti * k + j] = selv[j];
                swt[ti * k + j] = v * route_scale;
            }
        }
    }
    __syncthreads();
    // route: counts, exclusive offsets (each thread a run of consecutive experts), rows
    int n = t * k, per = (ne + 127) / 128, e0 = threadIdx.x * per;
    int loc = 0;
    for (int e = e0; e < min(ne, e0 + per); e++) {
        int c = 0;
        for (int i = 0; i < n; i++) c += sid[i] == e;
        counts[e] = c;
        soff[e] = c;
        loc += c;
    }
    int tot;
    int base = gate_scan128(loc, wsum, &tot);
    for (int e = e0; e < min(ne, e0 + per); e++) {
        int c = soff[e];
        soff[e] = base;
        off[e] = base;
        base += c;
    }
    if (threadIdx.x == 0) off[ne] = tot;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int e = sid[i], p = soff[e];
        for (int j = 0; j < i; j++) p += sid[j] == e;
        rows[p] = i / k;
        rw[p] = swt[i];
    }
    if (!elist) return;
    // plan: live local experts ascending, then -1
    int lv = 0;
    for (int e = e0; e < min(ne, e0 + per); e++) {
        int c = (e + 1 < ne ? soff[e + 1] : tot) - soff[e];
        lv += c > 0 && mask[e];
    }
    int nl;
    int j = gate_scan128(lv, wsum, &nl);
    for (int e = e0; e < min(ne, e0 + per); e++) {
        int c = (e + 1 < ne ? soff[e + 1] : tot) - soff[e];
        if (c > 0 && mask[e]) {
            if (j < cap) elist[j] = e;
            j++;
        }
    }
    for (int q = min(nl, cap) + threadIdx.x; q < cap; q += blockDim.x) elist[q] = -1;
}

// ---- DSpark's sampled Markov draw on the device (`SpecSampler::sample`, same transform) ----

// The 15 sums of a thread's `loc` over the block (1024 threads), in a fixed order (so every
// EP rank gets the same bits); the results in `out` for all threads.
__device__ void ds_ss_sum15(double* loc, double (*red)[15], double* out) {
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    for (int k = 0; k < 15; k++) {
        double x = loc[k];
        for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffff, x, o);
        if (lane == 0) red[wid][k] = x;
    }
    __syncthreads();
    if (tid < 15) {
        double s = 0.0;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) s += red[w][tid];
        out[tid] = s;
    }
    __syncthreads();
}

// The largest key k (w's bits: w >= 0, so they order as w does) with G(k) = sum of w over
// keys >= k at least `target` and positive, 4 bits a pass.
__device__ unsigned long long ds_ss_search(const double* w, int v, double target, double (*red)[15],
                                  double* out) {
    unsigned long long kap = 0;
    double loc[15];
    for (int sh = 60; sh >= 0; sh -= 4) {
        for (int k = 0; k < 15; k++) loc[k] = 0.0;
        for (int j = threadIdx.x; j < v; j += blockDim.x) {
            unsigned long long key = (unsigned long long)__double_as_longlong(w[j]);
            double x = w[j];
            for (int k = 0; k < 15; k++)
                if (key >= (kap | ((unsigned long long)(k + 1) << sh))) loc[k] += x;
        }
        ds_ss_sum15(loc, red, out);
        int best = 0;
        for (int k = 0; k < 15; k++)
            if (out[k] >= target && out[k] > 0.0) best = k + 1;
        kap |= (unsigned long long)best << sh;
        __syncthreads();
    }
    return kap;
}

// G(> kap) and the number of keys == kap.
__device__ void ds_ss_above(const double* w, int v, unsigned long long kap, double (*red)[15], double* out,
                            double* above, int* ties) {
    double loc[15];
    for (int k = 0; k < 15; k++) loc[k] = 0.0;
    for (int j = threadIdx.x; j < v; j += blockDim.x) {
        unsigned long long key = (unsigned long long)__double_as_longlong(w[j]);
        if (key > kap) loc[0] += w[j];
        if (key == kap) loc[1] += 1.0;
    }
    ds_ss_sum15(loc, red, out);
    *above = out[0];
    *ties = (int)out[1];
    __syncthreads();
}

// The n-th (1-based) token, in id order, with key == kap.
__device__ int ds_ss_nth(const double* w, int v, unsigned long long kap, int n, int* cnt, int* res) {
    int tid = threadIdx.x, nt = blockDim.x;
    int cs = (v + nt - 1) / nt, lo = tid * cs, hi = min(v, lo + cs);
    int c = 0;
    for (int j = lo; j < hi; j++) c += (unsigned long long)__double_as_longlong(w[j]) == kap;
    cnt[tid] = c;
    if (tid == 0) *res = -1;
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < nt; i++) {
            int x = cnt[i];
            cnt[i] = acc;
            acc += x;
        }
    }
    __syncthreads();
    int before = cnt[tid];
    if (before < n && before + c >= n) {
        int k = before;
        for (int j = lo; j < hi; j++)
            if ((unsigned long long)__double_as_longlong(w[j]) == kap && ++k == n) {
                *res = j;
                break;
            }
    }
    __syncthreads();
    return *res;
}

// The fast transformed distribution (SpecSampler::dist on the device). The kept top-p prefix
// is almost always tiny, so instead of searching the whole vocab by key bits, one pass
// buckets w by exponent (w >= 2^-e for the DS_SS_NB thresholds e below), the smallest bucket
// holding the top-p mass (with a 1e-9 margin, so the host's running sum surely reaches it)
// is gathered (at most DS_SS_CAP tokens: an exact prefix of the host's order, since it is
// every token with w above a threshold), bitonic-sorted by (w desc, id asc), and one
// thread runs the host's cut, total and renormalisation in its order. Returns the kept
// count, or 0 when it does not apply (top_p >= 1, or the bucket is too big): the caller then
// takes the full-vocab path. Always leaves w = exp((row - max) / temp) and z = sum w.
#define DS_SS_CAP 2048
#define DS_SS_NB 10
__constant__ int ds_ss_e[DS_SS_NB] = {1, 2, 3, 4, 6, 8, 11, 14, 20, 32};

struct DsSs {
    double red[32][32];
    double out[32];
    double cw[DS_SS_CAP];
    int ci[DS_SS_CAP];
    float smx[32];
    double z;
    int n, keep;
};

// The N sums of each thread's `loc` over the block, in a fixed order; into out for all.
template <int N>
__device__ void ds_ss_sumn(double* loc, DsSs& S) {
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
#pragma unroll
    for (int k = 0; k < N; k++) {
        double x = loc[k];
        for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffff, x, o);
        if (lane == 0) S.red[wid][k] = x;
    }
    __syncthreads();
    if (tid < N) {
        double s = 0.0;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) s += S.red[w][tid];
        S.out[tid] = s;
    }
    __syncthreads();
}

// Bias (when given) added to row in place, then w, z and the fast prefix (see above): the
// kept count (cw = probabilities, ci = ids, by decreasing probability), or 0.
__device__ int ds_ss_dist(float* row, int v, const float* bias, float temp, float top_p, double* w,
                          DsSs& S) {
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    float mx = -DS_INF;
    for (int j = tid; j < v; j += blockDim.x) {
        float x = row[j];
        if (bias) row[j] = x = x + bias[j];
        mx = fmaxf(mx, x);
    }
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
    if (lane == 0) S.smx[wid] = mx;
    __syncthreads();
    if (tid == 0) {
        float m = S.smx[0];
        for (int i = 1; i < (int)(blockDim.x >> 5); i++) m = fmaxf(m, S.smx[i]);
        S.smx[0] = m;
        S.n = 0;
    }
    __syncthreads();
    mx = S.smx[0];
    double loc[DS_SS_NB + 1];
    int cnt[DS_SS_NB];
#pragma unroll
    for (int k = 0; k <= DS_SS_NB; k++) loc[k] = 0.0;
#pragma unroll
    for (int k = 0; k < DS_SS_NB; k++) cnt[k] = 0;
    for (int j = tid; j < v; j += blockDim.x) {
        double x = exp((double)__fdiv_rn(__fsub_rn(row[j], mx), temp));
        w[j] = x;
        loc[DS_SS_NB] += x;
        // w in [2^-kk, 2^-kk+1): w >= 2^-e iff kk <= e
        int kk = 1023 - (int)(((unsigned long long)__double_as_longlong(x) >> 52) & 0x7ff);
        int b = 0;
        while (b < DS_SS_NB && ds_ss_e[b] < kk) b++;
#pragma unroll
        for (int k = 0; k < DS_SS_NB; k++)
            if (b == k) {
                loc[k] += x;
                cnt[k]++;
            }
    }
    ds_ss_sumn<DS_SS_NB + 1>(loc, S);
    double z = S.out[DS_SS_NB], mass[DS_SS_NB];
#pragma unroll
    for (int k = 0; k < DS_SS_NB; k++) {
        mass[k] = S.out[k];
        loc[k] = (double)cnt[k];
    }
    __syncthreads();
    ds_ss_sumn<DS_SS_NB>(loc, S);
    if (tid == 0) S.z = z;
    if (!(top_p < 1.0f)) return 0;
    double tgt = (double)top_p * z * (1.0 + 1e-9), cm = 0.0, cc = 0.0;
    int b = -1;
    for (int k = 0; k < DS_SS_NB; k++) {
        cm += mass[k];
        cc += S.out[k];
        if (cm >= tgt && cm > 0.0) {
            b = k;
            break;
        }
    }
    if (b < 0 || cc > DS_SS_CAP) return 0;
    double thr = ldexp(1.0, -ds_ss_e[b]);
    __syncthreads();
    for (int j = tid; j < v; j += blockDim.x)
        if (w[j] >= thr) {
            int p = atomicAdd(&S.n, 1);
            if (p < DS_SS_CAP) {
                S.cw[p] = w[j];
                S.ci[p] = j;
            }
        }
    __syncthreads();
    int n = min(S.n, DS_SS_CAP), P = 1;
    while (P < n) P <<= 1;
    for (int j = n + tid; j < P; j += blockDim.x) {
        S.cw[j] = -1.0;
        S.ci[j] = 0x7fffffff;
    }
    __syncthreads();
    for (int k = 2; k <= P; k <<= 1)
        for (int jj = k >> 1; jj > 0; jj >>= 1) {
            for (int i = tid; i < P; i += blockDim.x) {
                int l = i ^ jj;
                if (l > i) {
                    double wi = S.cw[i], wl = S.cw[l];
                    int ii = S.ci[i], il = S.ci[l];
                    bool lfirst = wl > wi || (wl == wi && il < ii);
                    bool up = (i & k) == 0;
                    if (up ? lfirst : !lfirst) {
                        S.cw[i] = wl;
                        S.cw[l] = wi;
                        S.ci[i] = il;
                        S.ci[l] = ii;
                    }
                }
            }
            __syncthreads();
        }
    if (tid == 0) {
        double acc = 0.0;
        int keep = 0;
        for (int j = 0; j < n; j++) {
            acc += S.cw[j] / z;
            if (acc >= (double)top_p) {
                keep = j + 1;
                break;
            }
        }
        if (keep) {
            double total = 0.0;
            for (int j = 0; j < keep; j++) total += S.cw[j];
            for (int j = 0; j < keep; j++) S.cw[j] = S.cw[j] / total;
        }
        S.keep = keep;
    }
    __syncthreads();
    return S.keep;
}

// SpecSampler::draw over n (id, prob) entries, probabilities by decreasing size.
__device__ int ds_ss_draw(const int* ci, const double* cp, int n, double u) {
    double s = 0.0;
    for (int j = 0; j < n; j++) s += cp[j];
    double r = u * s;
    for (int j = 0; j < n; j++) {
        r -= cp[j];
        if (r <= 0.0) return ci[j];
    }
    return ci[n - 1];
}

__device__ unsigned long long ds_ss_next(unsigned long long* st) {
    unsigned long long s = *st + 0x9E3779B97F4A7C15ull;
    *st = s;
    unsigned long long z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

__device__ double ds_ss_uniform(unsigned long long* st) {
    return (double)(ds_ss_next(st) >> 11) / (double)(1ull << 53);
}

// ids[slot] = a draw from SpecSampler::dist(row) (temperature `temp` > 0, top-p `top_p`):
// w = exp((row - max) / temp) in f64, the smallest prefix by decreasing w (ties by id) holding
// top_p of the mass, one uniform from the splitmix64 state *rng (advanced), the first prefix
// token whose cumulative mass reaches it. `w` is [v] f64 scratch; `bias` (when given) is
// first added to row, as ds_spark_pick. The kept distribution goes to (qi, qp, qn) row
// slot - 1 (DS_SS_CAP entries a row; qn = 0 when the full-vocab path ran) for the device
// verify. Then, as ds_spark_pick, the Markov embedding row of the draw. One block of 1024.
extern "C" __global__ void __launch_bounds__(1024) ds_spark_sample(float* row, int v, int* ids, int slot,
                                           const float* tab, int rank, float* memb, float temp,
                                           float top_p, unsigned long long* rng, double* w,
                                           const float* bias, int* qi, double* qp, int* qn) { DS_PDL_ENTRY
    __shared__ DsSs S;
    __shared__ double red[32][15];
    __shared__ double out[15];
    __shared__ int cnt[1024];
    __shared__ int res;
    __shared__ double su;
    __shared__ int pick;
    int tid = threadIdx.x;
    int keep = ds_ss_dist(row, v, bias, temp, top_p, w, S);
    if (tid == 0) su = ds_ss_uniform(rng);
    __syncthreads();
    double z = S.z, u = su;
    int tok;
    if (keep > 0) {
        if (qn) {
            for (int j = tid; j < keep; j += blockDim.x) {
                qi[(slot - 1) * DS_SS_CAP + j] = S.ci[j];
                qp[(slot - 1) * DS_SS_CAP + j] = S.cw[j];
            }
        }
        if (tid == 0) pick = ds_ss_draw(S.ci, S.cw, keep, u);
        __syncthreads();
        tok = pick;
    } else {
        // the full-vocab path: the kept prefix: keys above kc, then the first m ties at kc
        unsigned long long kc = 0;
        double above = 0.0, total = z;
        int ties = 0, m = 0;
        bool cut = top_p < 1.0f;
        if (cut) {
            double tgt = (double)top_p * z;
            kc = ds_ss_search(w, v, tgt, red, out);
            ds_ss_above(w, v, kc, red, out, &above, &ties);
            double wc = __longlong_as_double((long long)kc);
            m = wc > 0.0 ? (int)ceil((tgt - above) / wc) : ties;
            m = max(1, min(m, ties));
            total = above + m * wc;
        }
        double r = fmin(u * total, total);
        if (cut && r > above) {
            double wc = __longlong_as_double((long long)kc);
            int n = wc > 0.0 ? (int)ceil((r - above) / wc) : m;
            tok = ds_ss_nth(w, v, kc, max(1, min(n, m)), cnt, &res);
        } else {
            unsigned long long kd = ds_ss_search(w, v, r, red, out);
            double ab;
            int nt;
            ds_ss_above(w, v, kd, red, out, &ab, &nt);
            double wd = __longlong_as_double((long long)kd);
            int n = wd > 0.0 ? (int)ceil((r - ab) / wd) : nt;
            tok = ds_ss_nth(w, v, kd, max(1, min(n, nt)), cnt, &res);
        }
    }
    if (tid == 0) {
        if (tok < 0) tok = 0;
        ids[slot] = tok;
        pick = tok;
        if (qn) qn[slot - 1] = keep;
    }
    __syncthreads();
    if (memb)
        for (int j = tid; j < rank; j += blockDim.x) memb[j] = tab[(long long)pick * rank + j];
}

// The transformed distribution of each logits row (blockIdx.x) of a verify window, as
// ds_spark_sample keeps it: (pi, pp, pn) row-major with DS_SS_CAP entries a row, pn = 0 when
// the fast path does not apply. `w` is [rows][v] f64 scratch.
extern "C" __global__ void __launch_bounds__(1024) ds_spec_dist(float* logits, int v, float temp,
                                                                float top_p, double* w, int* pi,
                                                                double* pp, int* pn) { DS_PDL_ENTRY
    __shared__ DsSs S;
    int r = blockIdx.x, tid = threadIdx.x;
    int keep = ds_ss_dist(logits + (long long)r * v, v, nullptr, temp, top_p, w + (long long)r * v, S);
    for (int j = tid; j < keep; j += blockDim.x) {
        pi[r * DS_SS_CAP + j] = S.ci[j];
        pp[r * DS_SS_CAP + j] = S.cw[j];
    }
    if (tid == 0) pn[r] = keep;
}

__device__ double ds_ss_at(const int* ci, const double* cp, int n, int t) {
    for (int j = 0; j < n; j++)
        if (ci[j] == t) return cp[j];
    return 0.0;
}

// SpecSampler::verify's walk over a window of n rows (ids: the fed token, then n - 1 drafts):
// P rows from ds_spec_dist, Q rows from the draft's ds_spark_sample, the RNG from rng0.
// out = [ok, k, tokens[n], rng lo, rng hi]; ok = 0 (nothing else written) when a row the
// walk reaches has no fast distribution, and the host walks instead. One block.
extern "C" __global__ void __launch_bounds__(1024) ds_spec_walk(const int* ids, int n, const int* pi,
                                                                const double* pp, const int* pn,
                                                                const int* qi, const double* qp,
                                                                const int* qn, unsigned long long rng0,
                                                                int* out) { DS_PDL_ENTRY
    __shared__ double res[DS_SS_CAP];
    __shared__ unsigned long long st;
    __shared__ int sk, acc;
    int tid = threadIdx.x;
    if (tid == 0) {
        st = rng0;
        sk = 0;
        out[0] = 0;
    }
    __syncthreads();
    while (true) {
        int k = sk;
        int np = pn[k];
        bool last = k + 1 == n;
        int nq = last ? 1 : qn[k];
        if (np <= 0 || nq <= 0) return;
        const int* Pi = pi + k * DS_SS_CAP;
        const double* Pp = pp + k * DS_SS_CAP;
        const int* Qi = qi + k * DS_SS_CAP;
        const double* Qp = qp + k * DS_SS_CAP;
        int tok = -1;
        if (tid == 0) {
            if (last) {
                tok = ds_ss_draw(Pi, Pp, np, ds_ss_uniform(&st));
                acc = 2;
            } else {
                int d = ids[k + 1];
                double pd = ds_ss_at(Pi, Pp, np, d), qd = ds_ss_at(Qi, Qp, nq, d);
                acc = qd > 0.0 && ds_ss_uniform(&st) * qd < pd;
                if (acc) {
                    out[2 + k] = d;
                    sk = k + 1;
                }
            }
        }
        __syncthreads();
        int a = acc;
        __syncthreads();
        if (a == 1) continue;
        if (a == 0) {
            // the residual norm(max(0, P - Q)) over P's order
            for (int j = tid; j < np; j += blockDim.x)
                res[j] = fmax(Pp[j] - ds_ss_at(Qi, Qp, nq, Pi[j]), 0.0);
            __syncthreads();
            if (tid == 0) {
                double s = 0.0;
                int lastp = -1;
                for (int j = 0; j < np; j++)
                    if (res[j] > 0.0) {
                        s += res[j];
                        lastp = j;
                    }
                if (lastp < 0) {
                    tok = ds_ss_draw(Pi, Pp, np, ds_ss_uniform(&st));
                } else {
                    double r = ds_ss_uniform(&st) * s;
                    tok = Pi[lastp];
                    for (int j = 0; j < np; j++)
                        if (res[j] > 0.0) {
                            r -= res[j];
                            if (r <= 0.0) {
                                tok = Pi[j];
                                break;
                            }
                        }
                }
            }
        }
        if (tid == 0) {
            out[2 + k] = tok;
            out[1] = k;
            out[2 + n] = (int)(unsigned)(st & 0xffffffffull);
            out[3 + n] = (int)(unsigned)(st >> 32);
            out[0] = 1;
        }
        return;
    }
}
