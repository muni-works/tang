// Prefill-shaped GEMM throughput through cuBLASLt: int8 (i32 accumulate),
// fp8 e4m3 (f32 accumulate, bf16 out) and bf16, on the coder-next and
// DeepSeek-V4.1-Flash layer shapes at M = 2k..32k tokens. Routed experts are
// a strided batch of E GEMMs of M*topk/E rows each (perfect balance).
// Prints one line per (model, M, dtype): GEMM-only tok/s for the whole model
// and the achieved TFLOP/s, plus per-shape rows.
//
//   nvcc -O2 -arch=native -o prefill_gemm_bench prefill_gemm_bench.cu -lcublasLt
//   ./prefill_gemm_bench [iters]
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

#define CK(x) do { auto e = (x); if (e != 0) { fprintf(stderr, "%s:%d %s = %d\n", __FILE__, __LINE__, #x, (int)e); exit(1);} } while (0)

struct Shape { const char* name; int n, k; int batch; double topk_frac; int count; };
// count: how many times per forward (layers). batch>1: routed experts, rows = M*topk/E.
struct Model { const char* name; std::vector<Shape> s; };

enum Dt { I8, F8, BF16 };
static const char* dtn[] = {"int8", "fp8", "bf16"};

static cublasLtHandle_t lt;
static void* ws; static size_t wsz = 256u << 20;

// time C[n x m] = A[k x n]^T * B[k x m] (col-major, TN), batch copies
static double run(Dt dt, int m, int n, int k, int batch, int iters, void* A, void* B, void* C) {
  cudaDataType_t ab, cd, sc; cublasComputeType_t ct;
  if (dt == I8) { ab = CUDA_R_8I; cd = CUDA_R_32I; ct = CUBLAS_COMPUTE_32I; sc = CUDA_R_32I; }
  else if (dt == F8) { ab = CUDA_R_8F_E4M3; cd = CUDA_R_16BF; ct = CUBLAS_COMPUTE_32F; sc = CUDA_R_32F; }
  else { ab = CUDA_R_16BF; cd = CUDA_R_16BF; ct = CUBLAS_COMPUTE_32F; sc = CUDA_R_32F; }
  cublasLtMatmulDesc_t op; CK(cublasLtMatmulDescCreate(&op, ct, sc));
  cublasOperation_t T = CUBLAS_OP_T, N = CUBLAS_OP_N;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &N, sizeof N));
  cublasLtMatrixLayout_t la, lb, lc;
  CK(cublasLtMatrixLayoutCreate(&la, ab, k, n, k));
  CK(cublasLtMatrixLayoutCreate(&lb, ab, k, m, k));
  CK(cublasLtMatrixLayoutCreate(&lc, cd, n, m, n));
  if (batch > 1) {
    long long sa = (long long)k * n, sb = (long long)k * m, scd = (long long)n * m;
    cublasLtMatrixLayout_t ls[3] = {la, lb, lc}; long long ss[3] = {sa, sb, scd};
    for (int i = 0; i < 3; i++) {
      CK(cublasLtMatrixLayoutSetAttribute(ls[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof batch));
      CK(cublasLtMatrixLayoutSetAttribute(ls[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &ss[i], sizeof ss[i]));
    }
  }
  cublasLtMatmulPreference_t pref; CK(cublasLtMatmulPreferenceCreate(&pref));
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof wsz));
  cublasLtMatmulHeuristicResult_t h[8]; int got = 0;
  cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 8, h, &got);
  double best = -1;
  int ia = 1; float fa = 1.f; int ib = 0; float fb = 0.f;
  void* al = dt == I8 ? (void*)&ia : (void*)&fa; void* be = dt == I8 ? (void*)&ib : (void*)&fb;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int a = 0; a < got; a++) {  // best of the heuristic's top candidates
    if (cublasLtMatmul(lt, op, al, A, la, B, lb, be, C, lc, C, lc, &h[a].algo, ws, wsz, 0)) continue;
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++) cublasLtMatmul(lt, op, al, A, la, B, lb, be, C, lc, C, lc, &h[a].algo, ws, wsz, 0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= iters;
    if (best < 0 || ms < best) best = ms;
  }
  CK(cudaGetLastError());
  cublasLtMatmulPreferenceDestroy(pref); cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
  return best;  // ms, -1 if unsupported
}

// pseudo-random bytes: constant operands draw less power and clock higher
__global__ void fill(unsigned char* p, size_t n, unsigned seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned x = (unsigned)i * 2654435761u ^ seed; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    p[i] = (x & 0x77);  // small positive/negative magnitudes, no fp8 NaN/inf codes
  }
}

int main(int argc, char** argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 20;
  CK(cublasLtCreate(&lt)); CK(cudaMalloc(&ws, wsz));
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("# %s sm%d%d\n", p.name, p.major, p.minor);
  // coder-next (Qwen3-Next-80B-A3B): 48 layers = 36 GDN + 12 gated attention, all MoE (512 x top-10, inter 512) + shared expert 512
  Model coder{"coder-next", {
    {"gdn.in_qkvz+ba", 12352, 2048, 1, 0, 36}, {"gdn.out", 2048, 4096, 1, 0, 36},
    {"attn.q+gate", 8192, 2048, 1, 0, 12}, {"attn.kv", 1024, 2048, 1, 0, 12}, {"attn.o", 2048, 4096, 1, 0, 12},
    {"router", 512, 2048, 1, 0, 48}, {"shared.gate_up", 1024, 2048, 1, 0, 48}, {"shared.down", 2048, 512, 1, 0, 48},
    {"expert.gate_up", 1024, 2048, 512, 10.0 / 512, 48}, {"expert.down", 2048, 512, 512, 10.0 / 512, 48}}};
  // DeepSeek-V4.1-Flash: 40 layers, dim 5120, MLA-ish q/o low-rank, 384 x top-6 inter 2304 + shared 2304
  Model ds{"ds41-flash", {
    {"wq_a", 1280, 5120, 1, 0, 40}, {"wq_b/2", 16384, 1280, 1, 0, 80}, {"wkv", 512, 5120, 1, 0, 40},
    {"wo_a", 8192, 4096, 1, 0, 40}, {"wo_b", 5120, 8192, 1, 0, 40}, {"gate", 384, 5120, 1, 0, 40},
    {"shared.gate_up", 4608, 5120, 1, 0, 40}, {"shared.down", 5120, 2304, 1, 0, 40},
    {"expert.gate_up", 4608, 5120, 384, 6.0 / 384, 40}, {"expert.down", 5120, 2304, 384, 6.0 / 384, 40}}};
  int Ms[] = {2048, 8192, 32768};
  size_t cap = 1500u << 20;  // per-operand bytes (bf16 worst case)
  void *A, *B, *C; CK(cudaMalloc(&A, cap)); CK(cudaMalloc(&B, cap)); CK(cudaMalloc(&C, 2 * cap));
  fill<<<1024, 256>>>((unsigned char*)A, cap, 1); fill<<<1024, 256>>>((unsigned char*)B, cap, 2);
  CK(cudaDeviceSynchronize());
  for (Model* md : {&coder, &ds})
    for (int M : Ms)
      for (Dt dt : {I8, F8, BF16}) {
        double tot_ms = 0, tot_fl = 0; bool ok = true;
        int esz = dt == BF16 ? 2 : 1;
        for (auto& s : md->s) {
          int m = s.batch > 1 ? (int)(M * s.topk_frac + 0.5) : M;
          m = (m + 15) / 16 * 16;
          int bt = s.batch;
          // big expert batches: time a slice and scale, to fit the buffers
          int slice = bt;
          while ((size_t)slice * s.k * std::max(m, s.n) * esz > cap || (size_t)slice * s.n * m * 4 > 2 * cap) slice /= 2;
          int mr = m;  // dense: time a row slice and scale when C would not fit
          while (slice == 1 && ((size_t)mr * s.n > (1u << 28) || (size_t)mr * s.k * esz > cap)) mr /= 2;
          int nr = s.n;  // cuBLASLt faults (700) on 32768-wide int8 at M = 32k: split n
          while (nr > 16384) nr /= 2;
          double ms = run(dt, mr, nr, s.k, slice, iters, A, B, C);
          if (ms < 0) { ok = false; break; }
          ms *= (double)bt / slice * ((double)m / mr) * ((double)s.n / nr);
          double fl = 2.0 * m * s.n * s.k * bt;
          printf("  %-10s M=%-6d %-5s %-16s m=%-6d n=%-6d k=%-5d x%-3d %8.3f ms %7.1f TFLOP/s\n", md->name, M, dtn[dt],
                 s.name, m, s.n, s.k, bt, ms, fl / ms / 1e9);
          tot_ms += ms * s.count; tot_fl += fl * s.count;
        }
        if (!ok) { printf("%-10s M=%-6d %-5s unsupported\n", md->name, M, dtn[dt]); continue; }
        printf("%-10s M=%-6d %-5s GEMMs %8.2f ms/forward  %9.0f tok/s  %6.1f TFLOP/s  %.2f GFLOP/tok\n", md->name, M,
               dtn[dt], tot_ms, M / tot_ms * 1e3, tot_fl / tot_ms / 1e9, tot_fl / M / 1e9);
        fflush(stdout);
      }
}
