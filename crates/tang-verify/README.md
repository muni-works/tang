# tang-verify

Every GPU kernel checked against its CPU oracle, at write time.

- **shadow**: the CPU reference runs on the same input, from the device's own state; every
  intermediate is compared in full (rel-L2, max abs with a decoded location, cosine, exact-zero
  and NaN counts).
- **poison**: device outputs are NaN-filled before the kernel runs, so unwritten memory reads
  NaN, never a plausible 0.0.
- **attribute**: the failing kernel is named by provenance ("these values belong to a sibling,
  written at offset 2048 instead of 8192: an element offset added to a byte address"), then
  unwritten/zero anomalies, then pipeline order.
- **never skip silently**: under `TANG_REQUIRE_CUDA=1` a GPU test without a GPU fails.

`TANG_KERNEL_VERIFY=1` (or `=kda,mhc`) switches it on; off, nothing runs. The method, the four
GLM-5.3 bugs it was built from and the API: [`docs/kernel-verify.md`](../../docs/kernel-verify.md).
