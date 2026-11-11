// Read bandwidth vs working-set size on the Apple GPU (Metal), independent
// of any tang kernel. Each thread streams float4s through a buffer of
// `size` bytes, many passes; time from GPUStart/EndTime, best of 5.
// swiftc -O cachebw.swift -o /tmp/cachebw && /tmp/cachebw
import Metal
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void rd(device const float4* x [[buffer(0)]], device float* o [[buffer(1)]],
               constant uint& n4 [[buffer(2)]], constant uint& passes [[buffer(3)]],
               uint tid [[thread_position_in_grid]], uint nt [[threads_per_grid]]) {
  float4 a = 0;
  for (uint p = 0; p < passes; p++)
    for (uint i = tid; i < n4; i += nt) a += x[(i + p * 64) % n4];
  if (a.x == 1234.5f) o[tid] = a.y + a.z + a.w;
}
"""
let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "rd")!)
let q = dev.makeCommandQueue()!
let threads = 10 * 1024 * 4
let out = dev.makeBuffer(length: threads * 4)!
for kb in [256, 1024, 4096, 8192, 16384, 32768, 131072, 524288] {
  let size = kb * 1024
  let x = dev.makeBuffer(length: size, options: .storageModeShared)!
  let p = x.contents().bindMemory(to: Float.self, capacity: size / 4)
  for i in 0..<(size / 4) { p[i] = Float(i & 1023) }  // touch every page
  var n4 = UInt32(size / 16)
  var passes = UInt32(max(1, (2 << 30) / size))
  var best = Double.infinity
  for _ in 0..<5 {
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso)
    e.setBuffer(x, offset: 0, index: 0); e.setBuffer(out, offset: 0, index: 1)
    e.setBytes(&n4, length: 4, index: 2); e.setBytes(&passes, length: 4, index: 3)
    e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    best = min(best, cb.gpuEndTime - cb.gpuStartTime)
  }
  let gbs = Double(size) * Double(passes) / best / 1e9
  print("\(kb) KB  \(String(format: "%.0f", gbs)) GB/s")
}
