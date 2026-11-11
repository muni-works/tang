// CUDA-graph variants of the position-dependent kernels of a decode step (engine.rs,
// graph.rs). The step's start position is read on the device from `P[0]` (written by a
// captured memcpy before the step) instead of baked into the launch, so one captured graph
// serves every position. Bodies are copies of kernels.cu / kernels_select.cu with the same
// arithmetic, bit for bit. Shapes that grow with the position (the indexer's width, the
// candidate mask, top-k) are launched for their maximum and bounded on the device.
//
// Its own small NVRTC module (not appended to kernels.cu), so it carries its own helpers.
// `add` is a row's offset from the step's start (a verify window runs its positions one by
// one); `r` the layer's compress ratio.

// Programmatic dependent launch (DS_PDL=1, pdl.rs): every kernel lets its stream successor
// launch at its entry and waits for its predecessor (and its memory) before its first read;
// both are no-ops for a plain launch and below sm_90.
#ifndef DS_PDL_ENTRY
#if __CUDA_ARCH__ >= 900
#define DS_PDL_TRIGGER asm volatile("griddepcontrol.launch_dependents;");
#define DS_PDL_WAIT asm volatile("griddepcontrol.wait;" ::: "memory");
#else
#define DS_PDL_TRIGGER
#define DS_PDL_WAIT
#endif
#define DS_PDL_ENTRY DS_PDL_TRIGGER DS_PDL_WAIT
#endif


#define DG_INF __int_as_float(0x7f800000)
#define DG_SEL_THREADS 512

__device__ __forceinline__ float dg_bf(float x) {
    unsigned int b = __float_as_uint(x);
    if ((b & 0x7f800000u) == 0x7f800000u && (b & 0x7fffffu)) return x;  // NaN
    b += 0x7fffu + ((b >> 16) & 1u);
    return __uint_as_float(b & 0xffff0000u);
}

// As ds_rope: row r at P[0] + add + r * pstride, or (fixed_r > 0) every row at the
// compressed group's position P[0] + add + 1 - fixed_r.
extern "C" __global__ void dg_rope(float* v, const float2* tab, int rows, int per_row, int d, int rd, const int* P,
                                   int add, int pstride, int fixed_r, int inverse) { DS_PDL_ENTRY
    int half = rd / 2;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * per_row * half) return;
    int j = i % half;
    long long vec = i / half;
    int r = vec / per_row;
    int pos = fixed_r > 0 ? max(P[0] + add + 1 - fixed_r, 0) : P[0] + add + r * pstride;
    float2 cs = tab[(long long)pos * half + j];
    float s = inverse ? -cs.y : cs.y;
    float* t = v + vec * d + (d - rd);
    float a = t[2 * j], b = t[2 * j + 1];
    t[2 * j] = dg_bf(a * cs.x - b * s);
    t[2 * j + 1] = dg_bf(a * s + b * cs.x);
}

// As ds_win_idx modes 1 (decode) and 3 (verify window) at sp = P[0]; with src, also its [t][k]
// rows into columns n.. (as ds_idx_cat), so the whole index row in one launch.
extern "C" __global__ void dg_win_idx(int* out, int t, int stride, int n, int mode, const int* P, int win,
                                      const int* src, int k) { DS_PDL_ENTRY
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (src && e >= t * n && e < t * (n + k)) {
        int f = e - t * n;
        out[(long long)(f / k) * stride + n + f % k] = src[f];
        return;
    }
    if (e >= t * n) return;
    int sp = P[0];
    int row = e / n, j = e % n, v;
    if (mode == 1) {
        int i = (sp % win + 1 + j) % win;
        v = i > sp ? -1 : i;
    } else {
        int q = sp + row, cnt = min(win, q + 1), pos = q + 1 - cnt + j;
        v = j >= cnt ? -1 : (pos < sp ? pos % win : win + pos - sp);
    }
    out[(long long)row * stride + j] = v;
}

// t rows of hd into a ring of `win` slots at positions P[0] + add + e.
extern "C" __global__ void dg_ring_write(float* ring, const float* src, int t, int hd, int win, const int* P,
                                         int add) { DS_PDL_ENTRY
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)t * hd) return;
    int e = i / hd, c = i % hd;
    ring[(long long)((P[0] + add + e) % win) * hd + c] = src[i];
}

// One row of n into dst's row (P[0] + add) / r (a compressed cache).
extern "C" __global__ void dg_row_write(float* dst, const float* src, int n, const int* P, int add, int r) { DS_PDL_ENTRY
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    dst[(long long)((P[0] + add) / r) * n + c] = src[c];
}

__device__ __forceinline__ int dg_width(const int* P, int add, int r) { return (P[0] + add + 1) / r; }

// As ds_index_score for one row at position P[0] + add (width (pos + 1) / r of at most the
// launch's).
// Row blockIdx.y of a batch: q [y][nh][ihd], w [y][nh], score [y][sst], position add + y.
extern "C" __global__ void dg_index_score(const float* q, const float* ik, const float* w, float* score, const int* P,
                                          int add, int r, int nh, int ihd, int sst) { DS_PDL_ENTRY
    int y = blockIdx.y;
    q += (long long)y * nh * ihd;
    w += y * nh;
    score += (long long)y * sst;
    add += y;
    int j = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (j >= dg_width(P, add, r)) return;
    const float* kj = ik + (long long)j * ihd;
    float s = 0.f;
    for (int h = 0; h < nh; h++) {
        const float* qv = q + (long long)h * ihd;
        float d = 0.f;
        for (int i = lane; i < ihd; i += 32) d += qv[i] * kj[i];
        for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        s += dg_bf(fmaxf(dg_bf(d), 0.f) * w[h]);
    }
    if (lane == 0) score[j] = dg_bf(s);
}

// As ds_mask_scores (no causal part), row blockIdx.y of a batch.
extern "C" __global__ void dg_mask_scores(float* score, const unsigned char* cand, const int* P, int add, int r,
                                          int sst, int cst) { DS_PDL_ENTRY
    int y = blockIdx.y;
    score += (long long)y * sst;
    cand += (long long)y * cst;
    add += y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= dg_width(P, add, r)) return;
    if (!cand[j]) score[j] = -DG_INF;
}

__device__ __forceinline__ unsigned int dg_key(float x) {
    if (x == 0.f) x = 0.f;
    unsigned int u = __float_as_uint(x);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ int dg_block_scan(int flag, int* ws, int* total) {
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

// As ds_select_row.
__device__ void dg_select_row(const float* v, int n, int k, int* sel) {
    __shared__ int hist[256];
    __shared__ int ws[DG_SEL_THREADS / 32 + 1];
    __shared__ unsigned int s_prefix;
    __shared__ int s_kk;
    if (k <= 0) return;
    unsigned int prefix = 0, mask = 0;
    int kk = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            unsigned int key = dg_key(v[i]);
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
    int base = 0, eq_seen = 0;
    for (int c0 = 0; c0 < n; c0 += blockDim.x) {
        int i = c0 + threadIdx.x;
        unsigned int key = i < n ? dg_key(v[i]) : 0u;
        int eq = i < n && key == prefix;
        int eq_tot;
        int eq_pre = dg_block_scan(eq, ws, &eq_tot);
        int take = i < n && (key > prefix || (eq && eq_seen + eq_pre < kk));
        int tot;
        int pos = dg_block_scan(take, ws, &tot);
        if (take) sel[base + pos] = i;
        base += tot;
        eq_seen += eq_tot;
    }
    __syncthreads();
}

// As ds_candidates (decode: no causal prefix), block per row of a batch (position add + row).
extern "C" __global__ void dg_candidates(const float* score, float* s, int* sel, unsigned char* out, const int* P,
                                         int add, int r, int bs, int kb, int sst, int nbs, int ost) { DS_PDL_ENTRY
    int y = blockIdx.x;
    score += (long long)y * sst;
    s += (long long)y * nbs;
    sel += (long long)y * nbs;
    out += (long long)y * ost;
    add += y;
    int width = dg_width(P, add, r);
    int nb = (width + bs - 1) / bs;
    int len = width;
    int last = len >= 1 ? (len - 1) / bs : -1;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        float m = -DG_INF;
        int hi = min((b + 1) * bs, width);
        for (int j = b * bs; j < hi; j++) m = fmaxf(m, score[j]);
        s[b] = b == last ? DG_INF : m;
    }
    __syncthreads();
    int k = min(kb, nb);
    dg_select_row(s, nb, k, sel);
    for (int j = threadIdx.x; j < width; j += blockDim.x) out[j] = 0;
    __syncthreads();
    for (int q = 0; q < k; q++) {
        int b = sel[q];
        if (!(s[b] > -DG_INF)) continue;
        int hi = min((b + 1) * bs, width);
        for (int j = b * bs + threadIdx.x; j < hi; j += blockDim.x) out[j] = 1;
    }
}

// As ds_topk, block per row of a batch (position add + row; no causal prefix): the min(kmax, width) largest, ascending, as
// idx + offset, then -1 up to kmax (holes ds_sparse_attn skips exactly).
extern "C" __global__ void dg_topk(const float* score, int* sel, int* out, const int* P, int add, int r, int kmax,
                                   int offset, int sst) { DS_PDL_ENTRY
    int y = blockIdx.x;
    score += (long long)y * sst;
    sel += (long long)y * kmax;
    out += (long long)y * kmax;
    add += y;
    int width = dg_width(P, add, r);
    int k = min(kmax, width);
    dg_select_row(score, width, k, sel);
    for (int q = threadIdx.x; q < kmax; q += blockDim.x) out[q] = q < k ? sel[q] + offset : -1;
}

// As ds_sparse_attn, with the keys in two places: row r < split of kv0, else row r - split of
// kv1 (the window, then a source layer's compressed cache), so neither is copied.
extern "C" __global__ void dg_sparse_attn(const float* q, const float* kv0, int split, const float* kv1,
                                          const float* sink, const int* idxs, float* o, int h, int d, int k,
                                          float scale) { DS_PDL_ENTRY
    extern __shared__ float sc[];
    __shared__ float red[32];
    __shared__ float den_s;
    int t = blockIdx.x / h, hi = blockIdx.x % h;
    const float* qv = q + ((long long)t * h + hi) * d;
    const int* ix = idxs + (long long)t * k;
    for (int j = 0; j < k; j++) {
        int r = ix[j];
        float part = 0.f;
        if (r >= 0) {
            const float* kr = r < split ? kv0 + (long long)r * d : kv1 + (long long)(r - split) * d;
            for (int i = threadIdx.x; i < d; i += blockDim.x) part += qv[i] * kr[i];
        }
        for (int o2 = 16; o2; o2 >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o2);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = part;
        __syncthreads();
        if (threadIdx.x == 0) {
            float s = 0.f;
            for (int w = 0; w < (blockDim.x + 31) / 32; w++) s += red[w];
            sc[j] = r >= 0 ? s * scale : -DG_INF;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float mx = -DG_INF;
        for (int j = 0; j < k; j++) mx = fmaxf(mx, sc[j]);
        mx = fmaxf(mx, -1e30f);
        float den = 0.f;
        for (int j = 0; j < k; j++) {
            float p = expf(sc[j] - mx);
            den += p;
            sc[j] = p;
        }
        den_s = den + expf(sink[hi] - mx);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < k; j++) {
            int r = ix[j];
            if (r >= 0) {
                const float* kr = r < split ? kv0 + (long long)r * d : kv1 + (long long)(r - split) * d;
                acc += dg_bf(sc[j]) * kr[i];
            }
        }
        o[((long long)t * h + hi) * d + i] = dg_bf(acc / den_s);
    }
}

// As ds_win_idx mode 2 (a DSpark block of t rows: the ring's min(win, sp + 1) rows, then the
// block's t) at sp = P[0] + add, for n = win + t columns: the rest -1.
extern "C" __global__ void dg_spark_idx(int* out, int t, int n, const int* P, int add, int win) { DS_PDL_ENTRY
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= t * n) return;
    int j = e % n, n1 = min(win, P[0] + add + 1), v;
    v = j < n1 ? j : (j - n1 < t ? win + (j - n1) : -1);
    out[e] = v;
}
