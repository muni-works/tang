// Symbols cudarc 0.16 resolves eagerly that the libraries seen under ncu's injection may
// lack (newer NVRTC / driver entry points tang never calls). Each returns an error code.
// Build: cc -shared -fPIC -O2 -o cudarc_stubs.so cudarc_stubs.c; run under ncu with
// LD_PRELOAD=.../cudarc_stubs.so (ncu_kernels.sh does this when STUBS is set).
#define STUB(name) int name() { return 7; }
STUB(nvrtcGetNVVM)
STUB(nvrtcGetNVVMSize)
STUB(nvrtcGetPCHCreateStatus)
STUB(nvrtcGetPCHHeapSize)
STUB(nvrtcGetPCHHeapSizeRequired)
STUB(nvrtcSetFlowCallback)
STUB(nvrtcSetPCHHeapSize)
STUB(cuTensorMapEncodeIm2colWide)
