//! The Flash-Next decode ops on CUDA (kernels in `kernels::flash_cuda`, contracts in
//! `crate::flash`). Every launch here writes into caller buffers and takes per-window values
//! from the `win` record, so a window of them can be captured into one CUDA graph.

use std::collections::HashMap;

use cudarc::driver::{CudaFunction, DevicePtr, LaunchConfig, PushKernelArg};

use super::{CudaBuffer, CudaComputeDevice, CudaStorage, Q2Weight};
use crate::flash::shape::*;
use crate::flash::{
    GdnMode, GdnParams, Geometry, HcPending, HcWeights, MoePlan, QAct, QsaCache, QsaNorms, MAX_T,
};

/// A null device pointer for kernel arguments a mode never reads.
const NULL: u64 = 0;

/// Words of the per-device sync counters, and each fused kernel's slot in them.
const SYNC_WORDS: usize = 1024;
const SYNC_HC: usize = 0;
/// `fl_moe_fused`: PLAN_CAP group counters, then its last-block counter.
const SYNC_MOE: usize = 64;
const _: () = assert!(SYNC_MOE + MoePlan::WIDE_CAP + 1 <= SYNC_WORDS);

/// `TANG_FLASH_UNFUSED=1`: the multi-launch kernels instead of the barrier-merged ones (A/B).
fn unfused() -> bool {
    static U: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *U.get_or_init(|| std::env::var("TANG_FLASH_UNFUSED").is_ok_and(|v| v == "1"))
}

fn grid(g: (usize, usize, usize), threads: u32) -> LaunchConfig {
    LaunchConfig {
        block_dim: (threads, 1, 1),
        grid_dim: (g.0 as u32, g.1 as u32, g.2 as u32),
        shared_mem_bytes: 0,
    }
}

/// `TANG_FLASH_WIDE_PLAIN=1`: wide windows on the plain T=16/32/64 instantiations instead of
/// the token-tiled kernels (A/B).
fn wide_plain() -> bool {
    static U: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *U.get_or_init(|| std::env::var("TANG_FLASH_WIDE_PLAIN").is_ok_and(|v| v == "1"))
}

/// `TANG_FLASH_WIDE_V1=1`: wide native GEMV on the re-decoding tile loop (`fl_natw`, A/B).
fn wide_v1() -> bool {
    static U: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *U.get_or_init(|| std::env::var("TANG_FLASH_NATW").is_ok_and(|v| v == "v1"))
}

/// Wide native GEMV on int8 tensor cores (`fl_natmma`) unless `TANG_FLASH_NATW` picks a dp4a
/// variant.
fn wide_mma() -> bool {
    static U: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *U.get_or_init(|| std::env::var("TANG_FLASH_NATW").is_ok_and(|v| v == "mma"))
}

/// `TANG_FLASH_NATW=16x4`: the wide native GEMV's token tile and rows a warp (A/B; instantiated
/// 8x2 8x4 8x8 16x2 16x4 12x4 32x1 32x2).
fn natw_variant() -> Option<(usize, usize)> {
    static U: std::sync::OnceLock<Option<(usize, usize)>> = std::sync::OnceLock::new();
    // Default 16x4 (measured best at T=32: Q3K/Q4K 2560->10240 96/105 us vs 8x4 146/112);
    // TANG_FLASH_NATW=v2 for fl_natw2 (decoded-weight reuse, slower), =v1 for fl_natw.
    *U.get_or_init(|| match std::env::var("TANG_FLASH_NATW") {
        Err(_) => Some((16, 4)),
        Ok(v) if v == "mma" => None,
        Ok(v) => {
            let (a, b) = v.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        }
    })
}

/// `TANG_FLASH_NATST=TTxGR`: wide stacked native GEMV as token tiles of TT columns, GR rows a
/// warp (default 16x4, `fl_natw`'s best); `off` runs the plain T-column body (A/B).
fn natst_tile() -> Option<(usize, usize)> {
    static U: std::sync::OnceLock<Option<(usize, usize)>> = std::sync::OnceLock::new();
    *U.get_or_init(|| match std::env::var("TANG_FLASH_NATST") {
        Err(_) => Some((16, 4)),
        Ok(v) if v == "off" => None,
        Ok(v) => {
            let (a, b) = v.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        }
    })
}

/// Prefill-only window widths beyond `MAX_T` (instantiated for some kernels only).
pub const WIDE_T: [usize; 3] = [16, 32, 64]; // hc_fused: 16, 32 only (t64 exceeds 48 KB static smem)

/// Widths a launch accepts: 1..=MAX_T everywhere; wider (prefill) where the op has a wide path
/// (token-tiled kernels take any width up to 64; the plain instantiations only WIDE_T).
fn width_ok(t: usize) -> bool {
    (1..=MAX_T).contains(&t) || (!wide_plain() && (1..=64).contains(&t)) || WIDE_T.contains(&t)
}

/// `{prefix}{t}` as a static name (cached).
fn wide_name(prefix: &'static str, t: usize) -> &'static str {
    use std::sync::Mutex;
    static NAMES: Mutex<Vec<(&'static str, usize, &'static str)>> = Mutex::new(Vec::new());
    let mut v = NAMES.lock().unwrap();
    if let Some(n) = v.iter().find(|x| x.0 == prefix && x.1 == t) {
        return n.2;
    }
    let n: &'static str = Box::leak(format!("{prefix}{t}").into_boxed_str());
    v.push((prefix, t, n));
    n
}

const NATST_NAMES: [&str; 8] = [
    "fl_natst_t1",
    "fl_natst_t2",
    "fl_natst_t3",
    "fl_natst_t4",
    "fl_natst_t5",
    "fl_natst_t6",
    "fl_natst_t7",
    "fl_natst_t8",
];

const NAT_NAMES: [&str; 8] = [
    "fl_nat_t1",
    "fl_nat_t2",
    "fl_nat_t3",
    "fl_nat_t4",
    "fl_nat_t5",
    "fl_nat_t6",
    "fl_nat_t7",
    "fl_nat_t8",
];
const GEMV_FMTS: [&str; 4] = ["bf16_gemv", "q2_gemv", "q4x_gemv", "q8x_gemv"];
const GEMV_NAMES: [[&str; 8]; 4] = [
    [
        "fl_bf16_gemv_t1",
        "fl_bf16_gemv_t2",
        "fl_bf16_gemv_t3",
        "fl_bf16_gemv_t4",
        "fl_bf16_gemv_t5",
        "fl_bf16_gemv_t6",
        "fl_bf16_gemv_t7",
        "fl_bf16_gemv_t8",
    ],
    [
        "fl_q2_gemv_t1",
        "fl_q2_gemv_t2",
        "fl_q2_gemv_t3",
        "fl_q2_gemv_t4",
        "fl_q2_gemv_t5",
        "fl_q2_gemv_t6",
        "fl_q2_gemv_t7",
        "fl_q2_gemv_t8",
    ],
    [
        "fl_q4x_gemv_t1",
        "fl_q4x_gemv_t2",
        "fl_q4x_gemv_t3",
        "fl_q4x_gemv_t4",
        "fl_q4x_gemv_t5",
        "fl_q4x_gemv_t6",
        "fl_q4x_gemv_t7",
        "fl_q4x_gemv_t8",
    ],
    [
        "fl_q8x_gemv_t1",
        "fl_q8x_gemv_t2",
        "fl_q8x_gemv_t3",
        "fl_q8x_gemv_t4",
        "fl_q8x_gemv_t5",
        "fl_q8x_gemv_t6",
        "fl_q8x_gemv_t7",
        "fl_q8x_gemv_t8",
    ],
];

impl CudaComputeDevice {
    /// Compile the flash kernels for `g` from now on (drops the flash kernels already loaded).
    pub fn set_flash_geometry(&self, g: &Geometry) {
        *self.flash_src.borrow_mut() = Some(Box::leak(g.cuda_source().into_boxed_str()));
        self.llm_funcs
            .borrow_mut()
            .retain(|k, _| !(k.starts_with("fl_") || k.starts_with("wide:fl_")));
    }

    /// The flash module's source (Flash-Next unless `set_flash_geometry` said otherwise).
    fn flash_source(&self) -> &'static str {
        *self
            .flash_src
            .borrow_mut()
            .get_or_insert_with(|| Box::leak(Geometry::FLASH_NEXT.cuda_source().into_boxed_str()))
    }

    /// Kernel `name` of the flash module (compiled once for sm_86, as the tensor-core kernels are:
    /// `__dp4a` needs an explicit architecture).
    fn fl(&self, name: &'static str) -> CudaFunction {
        if let Some(f) = self.llm_funcs.borrow().get(name) {
            return f.clone();
        }
        // TANG_FL_DEFS="#define X;#define Y": experiment defines prepended to the flash source.
        let flash = self.flash_source();
        let src: &'static str = match std::env::var("TANG_FL_DEFS") {
            Ok(d) => Box::leak(format!("{}\n{flash}", d.replace(';', "\n")).into_boxed_str()),
            Err(_) => flash,
        };
        let (_module, f) = self.get_func_with_arch(src, name, "sm_86");
        self.llm_funcs.borrow_mut().insert(name, f.clone());
        f
    }

    /// A kernel of the wide (prefill) MoE module: the flash source with the wide plan capacity.
    fn fl_wide(&self, name: &'static str) -> CudaFunction {
        let key: &'static str = Box::leak(format!("wide:{name}").into_boxed_str());
        if let Some(f) = self.llm_funcs.borrow().get(key) {
            return f.clone();
        }
        let src: &'static str = Box::leak(
            format!(
                "{}\n#define FLASH_WIDE 1\n#define PLAN_CAP {}\n#define SHARED_ROW {}\n{}",
                std::env::var("TANG_FL_DEFS")
                    .unwrap_or_default()
                    .replace(';', "\n"),
                MoePlan::WIDE_CAP,
                MoePlan::shared_row(MAX_T + 1),
                self.flash_source()
            )
            .into_boxed_str(),
        );
        let (_module, f) = self.get_func_with_arch(src, name, "sm_86");
        self.llm_funcs.borrow_mut().insert(key, f.clone());
        f
    }

    /// `fl` for the MoE kernels: the wide module above `MAX_T`.
    fn fl_moe(&self, name: &'static str, t: usize) -> CudaFunction {
        if t > MAX_T {
            self.fl_wide(name)
        } else {
            self.fl(name)
        }
    }

    pub(super) fn upload_bytes_impl(&self, bytes: &[u8]) -> CudaBuffer {
        self.upload_f32(&crate::flash::words_of(&crate::flash::bytes_to_words(
            bytes,
        )))
    }

    pub(super) fn buffer_addr_impl(&self, buf: &CudaBuffer) -> u64 {
        let (p, _sync) = match buf.storage() {
            CudaStorage::F32(s) => s.device_ptr(&self.stream),
            CudaStorage::Bf16(s) => s.device_ptr(&self.stream),
            CudaStorage::Q2(q) => q.data.device_ptr(&self.stream),
            CudaStorage::Q4(q) => q.packed.device_ptr(&self.stream),
        };
        p
    }

    pub(super) fn upload_q2_impl(&self, raw: &[u8], n: usize, k: usize) -> CudaBuffer {
        let bytes = crate::flash::q2_repack(raw, n, k);
        let len = bytes.len().div_ceil(4);
        let data = self.stream.clone_htod(&bytes).unwrap();
        Self::make_buf_unpooled(CudaStorage::Q2(Box::new(Q2Weight { data, n, k })), len)
    }

    /// `linear` into `out` for `m <= 8` with the bf16 / 4-bit GEMVs; anything else allocates
    /// through `linear` and copies.
    pub(super) fn linear_into_impl(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        use crate::ComputeDevice;
        let direct = m <= 8 && k.is_multiple_of(8) && !x.is_bf16() && !out.is_bf16();
        let (mu, ku, nu) = (m as u32, k as u32, n as u32);
        let cfg = grid((n.div_ceil(8), 1, 1), 256);
        match w.storage() {
            CudaStorage::Q4(q) if direct => {
                let f = self.llm_func(crate::kernels::llm_cuda::GEMV_CUDA, "gemv_q4");
                let group = q.group as u32;
                unsafe {
                    self.stream
                        .launch_builder(&f)
                        .arg(x.f32_data())
                        .arg(&q.packed)
                        .arg(&q.scales)
                        .arg(&q.biases)
                        .arg(out.f32_data_mut())
                        .arg(&mu)
                        .arg(&ku)
                        .arg(&nu)
                        .arg(&group)
                        .launch(cfg)
                        .unwrap();
                }
            }
            CudaStorage::Bf16(s) if direct && k.is_multiple_of(8) && m >= 1 => {
                self.gemv_rows("bf16_gemv", Some(x), None, s, out, (0, n), m, k, n);
            }
            // Prefill widths: the plain wide instantiation, else 8-column tiles (same bits).
            CudaStorage::Bf16(s)
                if m > 8
                    && width_ok(m)
                    && k.is_multiple_of(8)
                    && !x.is_bf16()
                    && !out.is_bf16() =>
            {
                if WIDE_T.contains(&m) {
                    self.gemv_rows("bf16_gemv", Some(x), None, s, out, (0, n), m, k, n);
                } else {
                    for t0 in (0..m).step_by(8) {
                        let mt = (m - t0).min(8);
                        let xs = self.slice_buffer(x, t0 * k, mt * k);
                        self.gemv_rows("bf16_gemv", Some(&xs), None, s, out, (t0 * n, n), mt, k, n);
                    }
                }
            }
            _ => {
                let y = self.linear(x, w, m, k, n);
                self.write_into(out, 0, &y);
            }
        }
    }

    /// bf16 GEMV into a slice of a wider output (`out[off + t · ostride + o]`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn bf16_linear_out_impl(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        off: usize,
        ostride: usize,
        m: usize,
        k: usize,
        n: usize,
    ) {
        assert!(
            k.is_multiple_of(8) && w.len >= n * k,
            "bf16_linear_out_into: weight"
        );
        self.gemv_rows(
            "bf16_gemv",
            Some(x),
            None,
            w.bf16_data(),
            out,
            (off, ostride),
            m,
            k,
            n,
        );
    }

    pub(super) fn quantize_act_impl(
        &self,
        x: &CudaBuffer,
        xq: &mut CudaBuffer,
        m: usize,
        k: usize,
    ) {
        assert!(k.is_multiple_of(32) && xq.len >= QAct { m, k }.words());
        let f = self.fl("fl_quantize");
        let (mu, ku) = (m as u32, k as u32);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(x.f32_data())
                .arg(xq.f32_data_mut())
                .arg(&mu)
                .arg(&ku)
                .launch(grid((k.div_ceil(256), m, 1), 256))
                .unwrap();
        }
    }

    /// Launch `fl_{name}_t{m}` (the multi-column GEMV) for `[n, k]` weights at `w`.
    #[allow(clippy::too_many_arguments)]
    fn gemv_rows(
        &self,
        name: &str,
        x: Option<&CudaBuffer>,
        xq: Option<&CudaBuffer>,
        w: &cudarc::driver::CudaSlice<impl cudarc::driver::DeviceRepr>,
        out: &mut CudaBuffer,
        (off, ostride): (usize, usize),
        m: usize,
        k: usize,
        n: usize,
    ) {
        assert!(width_ok(m) && ostride >= n && out.len >= off + (m - 1) * ostride + n);
        // Split K over 1..8 warps until there are ~2 blocks per SM, keeping 32+ weight vectors
        // per warp (4 rows a warp, 8 / ks row groups a 256-thread block).
        let epv = match name {
            "bf16_gemv" => 8,
            "q2_gemv" => 64,
            "q8x_gemv" => 16,
            _ => 32,
        };
        // Rows per warp, as the kernel picks it: 2 for one column, 4 for more. The K split is
        // chosen from (n, k) alone (at 2 rows a warp), so a row's summation order doesn't depend
        // on the window width m.
        let gr = if m == 1 {
            2
        } else if m <= MAX_T {
            4
        } else {
            1
        };
        let mut ks = 1;
        while ks < 8
            && n.div_ceil(2 * 8 / ks) < 2 * super::llm::sm_count()
            && k / epv / (2 * ks) >= 32
        {
            ks *= 2;
        }
        let f = if m <= MAX_T {
            self.fl(GEMV_NAMES[GEMV_FMTS.iter().position(|&f| f == name).unwrap()][m - 1])
        } else {
            self.fl(wide_name(
                [
                    "fl_bf16_gemv_t",
                    "fl_q2_gemv_t",
                    "fl_q4x_gemv_t",
                    "fl_q8x_gemv_t",
                ][GEMV_FMTS.iter().position(|&f| f == name).unwrap()],
                m,
            ))
        };
        let (ku, nu, ksu, osu) = (k as u32, n as u32, ks as u32, ostride as u32);
        let any = xq.or(x).unwrap();
        let mut y = out.f32_data_mut().slice_mut(off..);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(x.unwrap_or(any).f32_data())
                .arg(xq.unwrap_or(any).f32_data())
                .arg(w)
                .arg(&mut y)
                .arg(&ku)
                .arg(&nu)
                .arg(&ksu)
                .arg(&osu)
                .launch(grid((n.div_ceil(gr * 8 / ks), 1, 1), 256))
                .unwrap();
        }
    }

    pub(super) fn q2_linear_impl(
        &self,
        xq: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let CudaStorage::Q2(q) = w.storage() else {
            panic!("q2_linear_into needs a Q2 weight (upload_q2)")
        };
        assert!(q.n == n && q.k == k);
        self.gemv_rows("q2_gemv", None, Some(xq), &q.data, out, (0, n), m, k, n);
    }

    /// `fl_nat_t{m}` of the NatX module for `ty`, compiled on first use.
    fn nat_func(&self, ty: crate::flash_native::NatType, m: usize) -> CudaFunction {
        static KEYS: std::sync::OnceLock<std::sync::Mutex<HashMap<(u32, usize), &'static str>>> =
            std::sync::OnceLock::new();
        let key = *KEYS
            .get_or_init(Default::default)
            .lock()
            .unwrap()
            .entry((ty.ggml(), m))
            .or_insert_with(|| Box::leak(format!("nat{}_t{m}", ty.ggml()).into_boxed_str()));
        if let Some(f) = self.llm_funcs.borrow().get(key) {
            return f.clone();
        }
        let src: &'static str = Box::leak(
            format!(
                "#define NAT_TY {}\n{}",
                ty.ggml(),
                crate::kernels::native_cuda::NATIVE_CUDA
            )
            .into_boxed_str(),
        );
        let (_module, f) = self.get_func_with_arch(
            src,
            if m <= MAX_T {
                NAT_NAMES[m - 1]
            } else if wide_plain() {
                wide_name("fl_nat_t", m)
            } else if wide_mma() {
                "fl_natmma"
            } else if let Some((tt, g)) = natw_variant() {
                Box::leak(format!("fl_natw_{tt}x{g}").into_boxed_str())
            } else if wide_v1() {
                "fl_natw"
            } else {
                "fl_natw2"
            },
            "sm_86",
        );
        self.llm_funcs.borrow_mut().insert(key, f.clone());
        f
    }

    /// K split of a stacked segment: the single-segment GEMVs' rule, from (rows, k) alone.
    fn stack_ks(ty: Option<crate::flash_native::NatType>, n: usize, k: usize) -> usize {
        let epv = if ty.is_some() { 32 } else { 8 };
        let mut ks = 1;
        while ks < 8
            && n.div_ceil(2 * 8 / ks) < 2 * super::llm::sm_count()
            && k / epv / (2 * ks) >= 32
        {
            ks *= 2;
        }
        ks
    }

    pub(super) fn native_stack_impl(
        &self,
        segs: Vec<crate::flash_native::NatSeg<CudaBuffer>>,
        n: usize,
        k: usize,
    ) -> crate::flash_native::NatStack<CudaBuffer> {
        let mut words = vec![];
        for g in &segs {
            assert!(g.off + g.rows <= n, "native_stack: segment out of range");
            let p = self.buffer_addr_impl(&g.w);
            let (ty, ho, so) = match g.ty {
                Some(ty) => {
                    assert!(k.is_multiple_of(ty.block().0));
                    let (_, h, s, total) = crate::flash_native::nat_layout(ty, g.rows, k);
                    assert!(g.w.len * 4 >= total, "native_stack: weight size");
                    (ty.ggml(), h as u64, s as u64)
                }
                None => {
                    assert!(k.is_multiple_of(8) && g.w.len >= g.rows * k);
                    (0, 0, 0)
                }
            };
            let ks = Self::stack_ks(g.ty, g.rows, k) as u32;
            for v in [p, ho, so] {
                words.extend([v as u32, (v >> 32) as u32]);
            }
            words.extend([ty, g.rows as u32, ks, g.off as u32]);
        }
        let table = Some(crate::ComputeDevice::upload_u32(self, &words));
        crate::flash_native::NatStack { segs, table, n, k }
    }

    pub(super) fn native_stack_into_impl(
        &self,
        s: &crate::flash_native::NatStack<CudaBuffer>,
        x: &CudaBuffer,
        xq: &CudaBuffer,
        out: &mut CudaBuffer,
        m: usize,
    ) {
        assert!(width_ok(m) && out.len >= m * s.n);
        let blocks: usize = s
            .segs
            .iter()
            .map(|g| {
                let gr = g.ty.map_or(
                    if m == 1 {
                        2
                    } else if m <= MAX_T {
                        4
                    } else {
                        1
                    },
                    |t| match natst_tile() {
                        Some((_, gr)) if m > MAX_T => gr,
                        _ => t.rows_per_warp(m),
                    },
                );
                g.rows.div_ceil(gr * 8 / Self::stack_ks(g.ty, g.rows, s.k))
            })
            .sum();
        let key = if m <= MAX_T {
            NATST_NAMES[m - 1]
        } else {
            wide_name("fl_natst_t", m)
        };
        let f = match self.llm_funcs.borrow().get(key) {
            Some(f) => Some(f.clone()),
            None => None,
        };
        let f = f.unwrap_or_else(|| {
            let src: &'static str = Box::leak(
                format!(
                    "#define NAT_STACK\n{}{}",
                    natst_tile().map_or(String::new(), |(tt, gr)| format!(
                        "#define NATST_TT {tt}\n#define NATST_GR {gr}\n"
                    )),
                    crate::kernels::native_cuda::NATIVE_CUDA
                )
                .into_boxed_str(),
            );
            let (_module, f) = self.get_func_with_arch(src, key, "sm_86");
            self.llm_funcs.borrow_mut().insert(key, f.clone());
            f
        });
        let (nseg, ku, os) = (s.segs.len() as u32, s.k as u32, s.n as u32);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(xq.f32_data())
                .arg(x.f32_data())
                .arg(s.table.as_ref().expect("native_stack table").f32_data())
                .arg(&nseg)
                .arg(out.f32_data_mut())
                .arg(&ku)
                .arg(&os)
                .launch(grid((blocks, 1, 1), 256))
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn native_linear_impl(
        &self,
        ty: crate::flash_native::NatType,
        xq: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        off: usize,
        ostride: usize,
        m: usize,
        k: usize,
        n: usize,
    ) {
        assert!(width_ok(m) && k.is_multiple_of(32) && ostride >= n);
        assert!(
            out.len >= off + (m - 1) * ostride + n,
            "native_linear: output size"
        );
        let (_, hoff, soff, total) = crate::flash_native::nat_layout(ty, n, k);
        assert!(w.len * 4 >= total, "native_linear_into: weight size");
        // Rows a warp: 2 for one column, 4 for more (as the kernel's GR). K is split over 1..8
        // warps until there are ~2 blocks per SM at 2 rows a warp, keeping 32+ chunks per warp:
        // chosen from (n, k) alone, so a row's summation order (and its bits) doesn't depend on
        // the window width m.
        let tiled = m > MAX_T && !wide_plain();
        let v2 = tiled && !wide_v1() && natw_variant().is_none();
        let gr = match natw_variant() {
            Some((_, g)) if tiled => g,
            _ if v2 => 1,
            _ => ty.rows_per_warp(if tiled { MAX_T } else { m }),
        };
        let mut ks = 1;
        while ks < 8
            && n.div_ceil(2 * 8 / ks) < 2 * super::llm::sm_count()
            && k / 32 / (2 * ks) >= 32
        {
            ks *= 2;
        }
        let f = self.nat_func(ty, m);
        let (ku, nu, ksu, ho, so, os) = (
            k as u32,
            n as u32,
            ks as u32,
            hoff as u64,
            soff as u64,
            ostride as u32,
        );
        let mu = m as u32;
        let mut y = out.f32_data_mut().slice_mut(off..);
        if tiled && wide_mma() {
            // 16 rows a warp, 8 / ks row groups a block.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(xq.f32_data())
                    .arg(w.f32_data())
                    .arg(&mut y)
                    .arg(&ku)
                    .arg(&nu)
                    .arg(&ksu)
                    .arg(&ho)
                    .arg(&so)
                    .arg(&os)
                    .arg(&mu)
                    .launch(grid((n.div_ceil(16), 1, 1), 256))
                    .unwrap();
            }
            return;
        }
        if v2 {
            // Up to 32 columns a launch (the kernel's shared accumulators).
            for t0 in (0..m).step_by(32) {
                let (t0u, mt) = (t0 as u32, (m - t0).min(32) as u32);
                unsafe {
                    self.stream
                        .launch_builder(&f)
                        .arg(xq.f32_data())
                        .arg(w.f32_data())
                        .arg(&mut y)
                        .arg(&ku)
                        .arg(&nu)
                        .arg(&ksu)
                        .arg(&ho)
                        .arg(&so)
                        .arg(&os)
                        .arg(&mu)
                        .arg(&t0u)
                        .arg(&mt)
                        .launch(grid((n.div_ceil(8 / ks), 1, 1), 256))
                        .unwrap();
                }
            }
            return;
        }
        unsafe {
            let mut lb = self.stream.launch_builder(&f);
            lb.arg(xq.f32_data())
                .arg(w.f32_data())
                .arg(&mut y)
                .arg(&ku)
                .arg(&nu)
                .arg(&ksu)
                .arg(&ho)
                .arg(&so)
                .arg(&os);
            if tiled {
                lb.arg(&mu);
            }
            lb.launch(grid((n.div_ceil(gr * 8 / ks), 1, 1), 256))
                .unwrap();
        }
    }

    pub(super) fn q4x_linear_impl(
        &self,
        xq: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        assert!(
            w.len * 4 >= n * k / 2 + 4 * n * k / 64,
            "q4x_linear_into: weight size"
        );
        self.gemv_rows(
            "q4x_gemv",
            None,
            Some(xq),
            w.f32_data(),
            out,
            (0, n),
            m,
            k,
            n,
        );
    }

    pub(super) fn q8x_linear_impl(
        &self,
        xq: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        assert!(
            w.len * 4 >= n * k + 2 * n * k / 32,
            "q8x_linear_into: weight size"
        );
        self.gemv_rows(
            "q8x_gemv",
            None,
            Some(xq),
            w.f32_data(),
            out,
            (0, n),
            m,
            k,
            n,
        );
    }

    pub(super) fn hc_write_impl(
        &self,
        r: &mut CudaBuffer,
        y: &CudaBuffer,
        inj: &CudaBuffer,
        t: usize,
    ) {
        let f = self.fl("fl_hc_write");
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(r.f32_data_mut())
                .arg(y.f32_data())
                .arg(inj.f32_data())
                .launch(grid((HIDDEN / 256, HC, t), 256))
                .unwrap();
        }
    }

    /// The self-resetting counters the fused kernels synchronize through (zeroed once).
    fn sync_words(&self) -> std::cell::RefMut<'_, cudarc::driver::CudaSlice<u32>> {
        let mut s = self.flash_sync.borrow_mut();
        if s.is_none() {
            *s = Some(self.stream.alloc_zeros::<u32>(SYNC_WORDS).unwrap());
        }
        std::cell::RefMut::map(s, |s| s.as_mut().unwrap())
    }

    /// Blocks for a grid that must be co-resident (it meets at `grid_bar`): up to `per_sm`
    /// per SM, as the occupancy calculator allows.
    fn coresident(&self, f: &CudaFunction, threads: u32, per_sm: u32) -> usize {
        let occ = f
            .occupancy_max_active_blocks_per_multiprocessor(threads, 0, None)
            .unwrap();
        assert!(occ >= 1, "fused kernel does not fit on an SM");
        occ.min(per_sm) as usize * super::llm::sm_count()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_read_impl(
        &self,
        r: &mut CudaBuffer,
        pending: Option<HcPending<'_, CudaBuffer>>,
        w: &HcWeights<'_, CudaBuffer>,
        x: &mut CudaBuffer,
        xq: Option<&mut CudaBuffer>,
        inj: Option<&mut CudaBuffer>,
        scratch: &mut CudaBuffer,
        t: usize,
        eps: f32,
    ) {
        assert!(width_ok(t) && scratch.len >= crate::flash::hc_scratch_words(t));
        // The q8 weights have only the fused kernel.
        if !unfused() || w.q8 {
            return self.hc_fused_impl(r, pending, w, x, xq, inj, scratch, t, eps);
        }
        let xn_len = t * HC * HIDDEN;
        let (mut xn, mut lo) = scratch.f32_data_mut().split_at_mut(xn_len);
        // Unused pointer arguments get `w.norm`; the kernel never reads them in that mode.
        let any = w.norm;
        let (mode, yp, ip, parts, wr, lg, stride, sg) = match pending {
            None => (0u32, any, any, any, any, any, 0usize, -1i32),
            Some(HcPending::Write { y, inj }) => (1, y, inj, any, any, any, 0, -1),
            Some(HcPending::Moe {
                parts,
                w: wr,
                logits,
                stride,
                sg,
                inj,
            }) => (
                2,
                any,
                inj,
                parts,
                wr,
                logits,
                stride,
                sg.map_or(-1, |c| c as i32),
            ),
        };
        let su = stride as u32;
        let f = self.fl("fl_hc_norm");
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(r.f32_data_mut())
                .arg(yp.f32_data())
                .arg(ip.f32_data())
                .arg(&mode)
                .arg(w.norm.f32_data())
                .arg(&mut xn)
                .arg(&eps)
                .arg(parts.f32_data())
                .arg(wr.f32_data())
                .arg(lg.f32_data())
                .arg(&su)
                .arg(&sg)
                .launch(grid((HC, t, 1), 256))
                .unwrap();
        }
        let tu = t as u32;
        let f = self.fl("fl_hc_down");
        let rows = HC_LR + if w.inject.is_some() { HC } else { 0 };
        let wi = w.inject.unwrap_or(w.down);
        unsafe {
            let mut l = self.stream.launch_builder(&f);
            l.arg(&mut xn)
                .arg(w.down.bf16_data())
                .arg(wi.bf16_data())
                .arg(&mut lo);
            // Without injection rows the kernel never writes INJ; any pointer will do.
            match inj {
                Some(i) => l.arg(i.f32_data_mut()),
                None => l.arg(r.f32_data_mut()),
            };
            let ru = rows as u32;
            l.arg(&tu)
                .arg(&ru)
                .launch(grid((rows.div_ceil(4), 1, 1), 256))
                .unwrap();
        }
        const UP: [&str; 8] = [
            "fl_hc_up_t1",
            "fl_hc_up_t2",
            "fl_hc_up_t3",
            "fl_hc_up_t4",
            "fl_hc_up_t5",
            "fl_hc_up_t6",
            "fl_hc_up_t7",
            "fl_hc_up_t8",
        ];
        let f = self.fl(UP[t - 1]);
        let quant = xq.is_some() as u32;
        unsafe {
            let mut l = self.stream.launch_builder(&f);
            l.arg(&mut xn).arg(&mut lo).arg(w.up.bf16_data());
            match xq {
                Some(q) => {
                    assert!(q.len >= QAct { m: t, k: HIDDEN }.words());
                    l.arg(x.f32_data_mut()).arg(q.f32_data_mut())
                }
                None => l.arg(x.f32_data_mut()).arg(&NULL),
            };
            l.arg(&quant)
                .launch(grid((HIDDEN / 32, 1, 1), 1024))
                .unwrap();
        }
    }

    /// `hc_read_impl` as one launch of `fl_hc_fused_t{t}` on a co-resident grid.
    #[allow(clippy::too_many_arguments)]
    fn hc_fused_impl(
        &self,
        r: &mut CudaBuffer,
        pending: Option<HcPending<'_, CudaBuffer>>,
        w: &HcWeights<'_, CudaBuffer>,
        x: &mut CudaBuffer,
        xq: Option<&mut CudaBuffer>,
        inj: Option<&mut CudaBuffer>,
        scratch: &mut CudaBuffer,
        t: usize,
        eps: f32,
    ) {
        const NAMES: [&str; 8] = [
            "fl_hc_fused_t1",
            "fl_hc_fused_t2",
            "fl_hc_fused_t3",
            "fl_hc_fused_t4",
            "fl_hc_fused_t5",
            "fl_hc_fused_t6",
            "fl_hc_fused_t7",
            "fl_hc_fused_t8",
        ];
        const Q8: [&str; 8] = [
            "fl_hc_fused_q8_t1",
            "fl_hc_fused_q8_t2",
            "fl_hc_fused_q8_t3",
            "fl_hc_fused_q8_t4",
            "fl_hc_fused_q8_t5",
            "fl_hc_fused_q8_t6",
            "fl_hc_fused_q8_t7",
            "fl_hc_fused_q8_t8",
        ];
        let wide = t > MAX_T && !wide_plain();
        let f = if wide {
            self.fl(if w.q8 {
                "fl_hc_fusedw_q8"
            } else {
                "fl_hc_fusedw"
            })
        } else {
            self.fl(match (w.q8, t <= MAX_T) {
                (true, true) => Q8[t - 1],
                (false, true) => NAMES[t - 1],
                (true, false) => wide_name("fl_hc_fused_q8_t", t),
                (false, false) => wide_name("fl_hc_fused_t", t),
            })
        };
        let blocks = self.coresident(&f, 512, 2);
        let (mut xn, mut lo) = scratch.f32_data_mut().split_at_mut(t * HC * HIDDEN);
        let any = w.norm;
        let (mode, yp, ip, parts, wr, lg, stride, sg) = match pending {
            None => (0u32, any, any, any, any, any, 0usize, -1i32),
            Some(HcPending::Write { y, inj }) => (1, y, inj, any, any, any, 0, -1),
            Some(HcPending::Moe {
                parts,
                w: wr,
                logits,
                stride,
                sg,
                inj,
            }) => (
                2,
                any,
                inj,
                parts,
                wr,
                logits,
                stride,
                sg.map_or(-1, |c| c as i32),
            ),
        };
        let rows = (HC_LR + if w.inject.is_some() { HC } else { 0 }) as u32;
        let wi = w.inject.unwrap_or(w.down);
        let (su, quant) = (stride as u32, xq.is_some() as u32);
        let mut bar = self.sync_words();
        let mut bar = bar.slice_mut(SYNC_HC..SYNC_HC + 2);
        unsafe {
            let mut l = self.stream.launch_builder(&f);
            l.arg(r.f32_data_mut())
                .arg(yp.f32_data())
                .arg(ip.f32_data())
                .arg(&mode)
                .arg(w.norm.f32_data())
                .arg(&eps)
                .arg(parts.f32_data())
                .arg(wr.f32_data())
                .arg(lg.f32_data())
                .arg(&su)
                .arg(&sg);
            if w.q8 {
                l.arg(w.down.f32_data())
                    .arg(wi.f32_data())
                    .arg(&rows)
                    .arg(w.up.f32_data());
            } else {
                l.arg(w.down.bf16_data())
                    .arg(wi.bf16_data())
                    .arg(&rows)
                    .arg(w.up.bf16_data());
            }
            l.arg(x.f32_data_mut());
            match xq {
                Some(q) => {
                    assert!(q.len >= QAct { m: t, k: HIDDEN }.words());
                    l.arg(q.f32_data_mut())
                }
                None => l.arg(&NULL),
            };
            l.arg(&quant);
            match inj {
                Some(i) => l.arg(i.f32_data_mut()),
                None => l.arg(&NULL),
            };
            let tu = t as u32;
            l.arg(&mut xn).arg(&mut lo).arg(&mut bar);
            if wide {
                l.arg(&tu);
            }
            l.launch(grid((blocks, 1, 1), 512)).unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_conv_impl(
        &self,
        proj: &CudaBuffer,
        stride: usize,
        hist: &CudaBuffer,
        conv: &CudaBuffer,
        h: &mut CudaBuffer,
        t: usize,
        eps: f32,
    ) {
        assert!(width_ok(t));
        let f = self.fl("fl_gdn_conv");
        let (s, tu) = (stride as u32, t as u32);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(proj.f32_data())
                .arg(&s)
                .arg(hist.f32_data())
                .arg(conv.f32_data())
                .arg(h.f32_data_mut())
                .arg(&tu)
                .arg(&eps)
                .launch(grid((GDN_CONV / 128, 1, 1), 128))
                .unwrap();
        }
    }

    pub(super) fn gdn_conv_commit_impl(
        &self,
        hist: &mut CudaBuffer,
        proj: &CudaBuffer,
        stride: usize,
        win: &CudaBuffer,
        t: usize,
    ) {
        let f = self.fl("fl_gdn_conv_commit");
        let (s, tu) = (stride as u32, t as u32);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(hist.f32_data_mut())
                .arg(proj.f32_data())
                .arg(&s)
                .arg(win.f32_data())
                .arg(&tu)
                .launch(grid((GDN_CONV.div_ceil(256), 1, 1), 256))
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_step_impl(
        &self,
        state: &mut CudaBuffer,
        h: &CudaBuffer,
        proj: &CudaBuffer,
        stride: usize,
        p: &GdnParams<'_, CudaBuffer>,
        y: &mut CudaBuffer,
        yq: Option<&mut CudaBuffer>,
        t: usize,
        mode: GdnMode<'_, CudaBuffer>,
        eps: f32,
    ) {
        assert!(width_ok(t));
        let quant = yq.is_some() as u32;
        let (win, commit) = match mode {
            GdnMode::ReadOnly => (h, 0u32),
            GdnMode::Commit { win } => (win, 1u32),
        };
        let f = self.fl("fl_gdn_step");
        let (s, tu) = (stride as u32, t as u32);
        unsafe {
            let mut l = self.stream.launch_builder(&f);
            l.arg(state.f32_data_mut())
                .arg(h.f32_data())
                .arg(proj.f32_data())
                .arg(&s)
                .arg(p.dt_bias.f32_data())
                .arg(p.ssm_a.f32_data())
                .arg(p.norm.f32_data())
                .arg(y.f32_data_mut())
                .arg(&tu)
                .arg(win.f32_data())
                .arg(&commit)
                .arg(&eps);
            match yq {
                Some(q) => l.arg(q.f32_data_mut()),
                None => l.arg(&NULL),
            };
            l.arg(&quant).launch(grid((GDN_HV, 1, 1), 512)).unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_conv_step_impl(
        &self,
        state: &mut CudaBuffer,
        proj: &CudaBuffer,
        stride: usize,
        hist: &CudaBuffer,
        p: &GdnParams<'_, CudaBuffer>,
        y: &mut CudaBuffer,
        yq: Option<&mut CudaBuffer>,
        t: usize,
        mode: GdnMode<'_, CudaBuffer>,
        eps: f32,
    ) {
        assert!(width_ok(t));
        let quant = yq.is_some() as u32;
        let (win, commit) = match mode {
            GdnMode::ReadOnly => (proj, 0u32),
            GdnMode::Commit { win } => (win, 1u32),
        };
        let f = self.fl("fl_gdn_conv_step");
        let (s, tu) = (stride as u32, t as u32);
        unsafe {
            let mut l = self.stream.launch_builder(&f);
            l.arg(state.f32_data_mut())
                .arg(proj.f32_data())
                .arg(&s)
                .arg(p.dt_bias.f32_data())
                .arg(p.ssm_a.f32_data())
                .arg(p.norm.f32_data())
                .arg(y.f32_data_mut())
                .arg(&tu)
                .arg(win.f32_data())
                .arg(&commit)
                .arg(&eps);
            match yq {
                Some(q) => l.arg(q.f32_data_mut()),
                None => l.arg(&NULL),
            };
            l.arg(&quant)
                .arg(hist.f32_data())
                .arg(p.conv.f32_data())
                .launch(grid((GDN_HV, 1, 1), 512))
                .unwrap();
        }
    }

    pub(super) fn router_topk_impl(
        &self,
        logits: &CudaBuffer,
        stride: usize,
        n_expert: usize,
        ids: &mut CudaBuffer,
        w: &mut CudaBuffer,
        t: usize,
    ) {
        assert!((TOPK..=512).contains(&n_expert));
        let f = self.fl("fl_router_topk");
        let (s, ne) = (stride as u32, n_expert as u32);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(logits.f32_data())
                .arg(&s)
                .arg(&ne)
                .arg(ids.f32_data_mut())
                .arg(w.f32_data_mut())
                .launch(grid((t, 1, 1), 32))
                .unwrap();
        }
    }

    pub(super) fn moe_plan_impl(
        &self,
        ids: &CudaBuffer,
        table: &CudaBuffer,
        shared: u64,
        plan: &mut CudaBuffer,
        t: usize,
    ) {
        assert!(
            width_ok(t)
                && t <= MoePlan::WIDE_MAX
                && plan.len >= MoePlan::words(t)
                && table.len >= 2 * EXPERTS
        );
        let f = self.fl_moe("fl_moe_plan", t);
        let threads = if t > MAX_T { 1024 } else { 128 };
        let tu = t as u32;
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(ids.f32_data())
                .arg(table.f32_data())
                .arg(&shared)
                .arg(plan.f32_data_mut())
                .arg(&tu)
                .launch(grid((1, 1, 1), threads))
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn moe_route_impl(
        &self,
        logits: &CudaBuffer,
        stride: usize,
        n_expert: usize,
        forced: Option<&CudaBuffer>,
        table: &CudaBuffer,
        shared: u64,
        ids: &mut CudaBuffer,
        w: &mut CudaBuffer,
        plan: &mut CudaBuffer,
        t: usize,
    ) {
        assert!(width_ok(t) && t <= MoePlan::WIDE_MAX && (TOPK..=512).contains(&n_expert));
        assert!(plan.len >= MoePlan::words(t) && table.len >= 2 * EXPERTS);
        let f = self.fl_moe("fl_moe_route", t);
        let threads = if t > MAX_T { 1024 } else { 256 };
        let (s, ne, tu, fo) = (
            stride as u32,
            n_expert as u32,
            t as u32,
            forced.is_some() as u32,
        );
        unsafe {
            let mut l = self.stream.launch_builder(&f);
            l.arg(logits.f32_data()).arg(&s).arg(&ne);
            match forced {
                Some(b) => l.arg(b.f32_data()),
                None => l.arg(&NULL),
            };
            l.arg(&fo)
                .arg(table.f32_data())
                .arg(&shared)
                .arg(ids.f32_data_mut())
                .arg(w.f32_data_mut())
                .arg(plan.f32_data_mut())
                .arg(&tu)
                .launch(grid((1, 1, 1), threads))
                .unwrap();
        }
    }

    pub(super) fn moe_grouped_impl(
        &self,
        xq: &CudaBuffer,
        plan: &CudaBuffer,
        scratch: &mut CudaBuffer,
        parts: &mut CudaBuffer,
        t: usize,
    ) {
        assert!(
            scratch.len >= MoePlan::scratch_words_for(t)
                && parts.len >= MoePlan::parts_rows(t) * HIDDEN
        );
        const GU: [&str; 8] = [
            "fl_moe_gu_t1",
            "fl_moe_gu_t2",
            "fl_moe_gu_t3",
            "fl_moe_gu_t4",
            "fl_moe_gu_t5",
            "fl_moe_gu_t6",
            "fl_moe_gu_t7",
            "fl_moe_gu_t8",
        ];
        const DOWN: [&str; 8] = [
            "fl_moe_down_t1",
            "fl_moe_down_t2",
            "fl_moe_down_t3",
            "fl_moe_down_t4",
            "fl_moe_down_t5",
            "fl_moe_down_t6",
            "fl_moe_down_t7",
            "fl_moe_down_t8",
        ];
        // A group has at most t entries (an expert appears once per token; the shared group has t).
        let tu = t as u32;
        // Wide windows: the two-launch path (measured faster with tile_mma: T=64 8.4 vs 10.4 us/token).
        if !unfused() && t <= MAX_T {
            const FUSED: [&str; 8] = [
                "fl_moe_fused_t1",
                "fl_moe_fused_t2",
                "fl_moe_fused_t3",
                "fl_moe_fused_t4",
                "fl_moe_fused_t5",
                "fl_moe_fused_t6",
                "fl_moe_fused_t7",
                "fl_moe_fused_t8",
            ];
            let f = if t > MAX_T {
                self.fl_wide("fl_moe_fused_t64")
            } else {
                self.fl(FUSED[t - 1])
            };
            let blocks = self.coresident(&f, 128, 8);
            let mut sync = self.sync_words();
            let mut cnt = sync.slice_mut(SYNC_MOE..SYNC_MOE + MoePlan::cap(t) + 1);
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(xq.f32_data())
                    .arg(&tu)
                    .arg(plan.f32_data())
                    .arg(scratch.f32_data_mut())
                    .arg(parts.f32_data_mut())
                    .arg(&mut cnt)
                    .launch(grid((blocks, 1, 1), 128))
                    .unwrap();
            }
            return;
        }
        let f = if t > MAX_T {
            self.fl_wide("fl_moe_gu_t64")
        } else {
            self.fl(GU[t - 1])
        };
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(xq.f32_data())
                .arg(&tu)
                .arg(plan.f32_data())
                .arg(scratch.f32_data_mut())
                .launch(grid((16 * super::llm::sm_count(), 1, 1), 128))
                .unwrap();
        }
        let f = if t > MAX_T {
            self.fl_wide("fl_moe_down_t64")
        } else {
            self.fl(DOWN[t - 1])
        };
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(scratch.f32_data())
                .arg(plan.f32_data())
                .arg(parts.f32_data_mut())
                .launch(grid((8 * super::llm::sm_count(), 1, 1), 256))
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn moe_combine_impl(
        &self,
        parts: &CudaBuffer,
        w: &CudaBuffer,
        logits: &CudaBuffer,
        stride: usize,
        sg: Option<usize>,
        y: &mut CudaBuffer,
        t: usize,
    ) {
        let f = self.fl_moe("fl_moe_combine", t);
        let (s, sgi) = (stride as u32, sg.map_or(-1i32, |c| c as i32));
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(parts.f32_data())
                .arg(w.f32_data())
                .arg(logits.f32_data())
                .arg(&s)
                .arg(&sgi)
                .arg(y.f32_data_mut())
                .launch(grid((HIDDEN / 256, t, 1), 256))
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qsa_prep_impl(
        &self,
        proj: &CudaBuffer,
        stride: usize,
        win: &CudaBuffer,
        norms: &QsaNorms<'_, CudaBuffer>,
        rope: (&CudaBuffer, &CudaBuffer),
        q: &mut CudaBuffer,
        cache: QsaCache<'_, CudaBuffer>,
        t: usize,
        eps: f32,
    ) {
        assert!(width_ok(t) && q.len >= crate::flash::qsa_q_words(t));
        let f = self.fl("fl_qsa_prep");
        let (s, tu) = (stride as u32, t as u32);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(proj.f32_data())
                .arg(&s)
                .arg(win.f32_data())
                .arg(norms.q.f32_data())
                .arg(norms.k.f32_data())
                .arg(norms.iq.f32_data())
                .arg(norms.ik.f32_data())
                .arg(rope.0.f32_data())
                .arg(rope.1.f32_data())
                .arg(q.f32_data_mut())
                .arg(cache.k.bf16_data_mut())
                .arg(cache.v.bf16_data_mut())
                .arg(cache.ring.f32_data_mut())
                .arg(cache.pooled.f32_data_mut())
                .arg(&tu)
                .arg(&eps)
                .launch(grid((34, t, 1), 256))
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qsa_select_impl(
        &self,
        pooled: &CudaBuffer,
        q: &CudaBuffer,
        win: &CudaBuffer,
        scores: &mut CudaBuffer,
        ids: &mut CudaBuffer,
        max_blocks: usize,
        t: usize,
    ) {
        assert!(max_blocks <= 8192, "qsa_select: context above 32K cells");
        assert!(scores.len >= t * max_blocks && ids.len >= t * QSA_WIDTH);
        let iq = q.f32_data().slice(t * QSA_HEADS * QSA_D..);
        let mb = max_blocks as u32;
        let f = self.fl("fl_qsa_scores");
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(pooled.f32_data())
                .arg(&iq)
                .arg(win.f32_data())
                .arg(scores.f32_data_mut())
                .arg(&mb)
                .launch(grid((max_blocks.div_ceil(8), t, 1), 256))
                .unwrap();
        }
        let f = self.fl("fl_qsa_select");
        let mask = 0u32;
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(scores.f32_data())
                .arg(win.f32_data())
                .arg(ids.f32_data_mut())
                .arg(&mb)
                .arg(&NULL)
                .arg(&mask)
                .launch(grid((t, 1, 1), 1024))
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qsa_select_union_impl(
        &self,
        pooled: &CudaBuffer,
        q: &CudaBuffer,
        win: &CudaBuffer,
        scores: &mut CudaBuffer,
        ids: &mut CudaBuffer,
        union: &mut CudaBuffer,
        max_blocks: usize,
        t: usize,
    ) {
        assert!(max_blocks <= 8192 && union.len >= crate::flash::qsa_union_words(max_blocks, t));
        assert!(scores.len >= t * max_blocks && ids.len >= t * QSA_WIDTH);
        let iq = q.f32_data().slice(t * QSA_HEADS * QSA_D..);
        let mb = max_blocks as u32;
        let f = self.fl("fl_qsa_scores");
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(pooled.f32_data())
                .arg(&iq)
                .arg(win.f32_data())
                .arg(scores.f32_data_mut())
                .arg(&mb)
                .launch(grid((max_blocks.div_ceil(8), t, 1), 256))
                .unwrap();
        }
        let cap = crate::flash::qsa_union_cap(t);
        let (mut bm, mut rest) = union.f32_data_mut().split_at_mut(max_blocks);
        let (mut ub, mut rest) = rest.split_at_mut(cap);
        let (mut um, mut uc) = rest.split_at_mut(cap);
        let f = self.fl("fl_qsa_select");
        let mask = 1u32;
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(scores.f32_data())
                .arg(win.f32_data())
                .arg(ids.f32_data_mut())
                .arg(&mb)
                .arg(&mut bm)
                .arg(&mask)
                .launch(grid((t, 1, 1), 1024))
                .unwrap();
        }
        let f = self.fl("fl_qsa_union");
        let tu = t as u32;
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(&mut bm)
                .arg(win.f32_data())
                .arg(&tu)
                .arg(&mut ub)
                .arg(&mut um)
                .arg(&mut uc)
                .launch(grid((1, 1, 1), 1024))
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qsa_attend_union_impl(
        &self,
        q: &CudaBuffer,
        k_cache: &CudaBuffer,
        v_cache: &CudaBuffer,
        union: &CudaBuffer,
        max_blocks: usize,
        proj: &CudaBuffer,
        stride: usize,
        win: &CudaBuffer,
        scratch: &mut CudaBuffer,
        out: &mut CudaBuffer,
        outq: Option<&mut CudaBuffer>,
        t: usize,
    ) {
        assert!(scratch.len >= crate::flash::qsa_union_scratch_words(t));
        let cap = crate::flash::qsa_union_cap(t);
        let nchu = cap.div_ceil(16);
        let u = union.f32_data();
        let (ub, um, uc) = (
            u.slice(max_blocks..max_blocks + cap),
            u.slice(max_blocks + cap..max_blocks + 2 * cap),
            u.slice(max_blocks + 2 * cap..max_blocks + 2 * cap + 1),
        );
        let (tu, nu) = (t as u32, nchu as u32);
        let f = self.fl("fl_qsa_attend_union");
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(q.f32_data())
                .arg(k_cache.bf16_data())
                .arg(v_cache.bf16_data())
                .arg(&ub)
                .arg(&um)
                .arg(&uc)
                .arg(win.f32_data())
                .arg(scratch.f32_data_mut())
                .arg(&tu)
                .arg(&nu)
                .launch(grid((nchu, QSA_KV, 1), 256))
                .unwrap();
        }
        let f = self.fl("fl_qsa_union_merge");
        let (s, quant) = (stride as u32, outq.is_some() as u32);
        unsafe {
            let mut l = self.stream.launch_builder(&f);
            l.arg(scratch.f32_data())
                .arg(proj.f32_data())
                .arg(&s)
                .arg(&uc)
                .arg(out.f32_data_mut());
            match outq {
                Some(q) => l.arg(q.f32_data_mut()),
                None => l.arg(&NULL),
            };
            l.arg(&quant)
                .arg(&tu)
                .arg(&nu)
                .launch(grid((QSA_HEADS, t, 1), 256))
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qsa_attend_impl(
        &self,
        q: &CudaBuffer,
        k_cache: &CudaBuffer,
        v_cache: &CudaBuffer,
        ids: &CudaBuffer,
        proj: &CudaBuffer,
        stride: usize,
        win: &CudaBuffer,
        scratch: &mut CudaBuffer,
        out: &mut CudaBuffer,
        outq: Option<&mut CudaBuffer>,
        t: usize,
    ) {
        assert!(scratch.len >= crate::flash::qsa_attend_scratch_words(t));
        let nch = QSA_WIDTH.div_ceil(crate::flash::QSA_CHUNK);
        let f = self.fl("fl_qsa_attend");
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(q.f32_data())
                .arg(k_cache.bf16_data())
                .arg(v_cache.bf16_data())
                .arg(ids.f32_data())
                .arg(win.f32_data())
                .arg(scratch.f32_data_mut())
                .launch(grid((nch, QSA_KV, t), 256))
                .unwrap();
        }
        let f = self.fl("fl_qsa_merge");
        let s = stride as u32;
        let quant = outq.is_some() as u32;
        let tu = t as u32;
        unsafe {
            let mut l = self.stream.launch_builder(&f);
            l.arg(scratch.f32_data())
                .arg(proj.f32_data())
                .arg(&s)
                .arg(win.f32_data())
                .arg(out.f32_data_mut());
            match outq {
                Some(q) => l.arg(q.f32_data_mut()),
                None => l.arg(&NULL),
            };
            l.arg(&quant)
                .arg(&tu)
                .launch(grid((QSA_HEADS, t, 1), 256))
                .unwrap();
        }
    }
}

/// Raw handles for an engine that launches its own kernels and mapped-memory protocol on this
/// device's stream (tang-llm's Flash-Next engine and tang-moe's doorbell).
impl CudaComputeDevice {
    /// The driver handle of this device's (only) stream.
    pub fn cu_stream(&self) -> cudarc::driver::sys::CUstream {
        self.stream.cu_stream()
    }

    /// This device's stream, for `launch_builder` on [`custom_func`](Self::custom_func)s.
    pub fn cuda_stream(&self) -> &std::sync::Arc<cudarc::driver::CudaStream> {
        &self.stream
    }

    /// This device's context.
    pub fn cuda_context(&self) -> &std::sync::Arc<cudarc::driver::CudaContext> {
        &self.ctx
    }

    /// Kernel `name` of CUDA C `src`, compiled once for sm_86 and cached by name.
    pub fn custom_func(&self, src: &str, name: &'static str) -> CudaFunction {
        if let Some(f) = self.llm_funcs.borrow().get(name) {
            return f.clone();
        }
        let (_module, f) = self.get_func_with_arch(src, name, "sm_86");
        self.llm_funcs.borrow_mut().insert(name, f.clone());
        f
    }
}

/// GPU timing for benchmarks (`flash-kernel-bench`).
impl CudaComputeDevice {
    /// GPU milliseconds between CUDA events recorded before and after `f` (whatever `f`
    /// enqueues on this device's stream, including any idle time between its launches).
    pub fn event_ms(&self, f: &mut dyn FnMut()) -> f32 {
        use cudarc::driver::sys::CUevent_flags;
        let a = self
            .ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        let b = self
            .ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        a.record(&self.stream).unwrap();
        f();
        b.record(&self.stream).unwrap();
        a.elapsed_ms(&b).unwrap()
    }

    /// Capture `f` into a CUDA graph (run it eagerly once first, so modules are loaded and
    /// pools warm), then return the graph.
    pub fn capture(&self, f: &mut dyn FnMut()) -> cudarc::driver::CudaGraph {
        f();
        self.stream.synchronize().unwrap();
        self.begin_capture();
        f();
        self.end_capture().expect("empty graph")
    }
}

/// Parity against `CpuDevice` (the reference in `cpu::flash`) for windows of 1, 2, 4 and 8
/// tokens. Skips without a GPU unless `TANG_REQUIRE_CUDA=1`. Run in release: the reference is
/// plain loops at the model's real shapes.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash::{self as fl, u32s, ExpertBlob, HcPending};
    use crate::{ComputeBuffer, ComputeDevice, CpuDevice};

    fn on_gpu(check: fn(&CudaComputeDevice)) {
        match std::panic::catch_unwind(CudaComputeDevice::new) {
            Ok(Ok(dev)) => check(&dev),
            _ if std::env::var("TANG_REQUIRE_CUDA").is_ok_and(|v| v == "1") => {
                panic!("TANG_REQUIRE_CUDA=1 but no CUDA device")
            }
            _ => eprintln!("no CUDA device: skipping"),
        }
    }

    const TS: [usize; 4] = [1, 2, 4, 8];

    /// xorshift values in [-scale, scale).
    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f(&mut self, scale: f32) -> f32 {
            ((self.u() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        }
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.f(scale)).collect()
        }
        fn bf16(&mut self, n: usize, scale: f32) -> Vec<u16> {
            (0..n)
                .map(|_| (self.f(scale).to_bits() >> 16) as u16)
                .collect()
        }
        /// A random GGUF Q2_0 matrix.
        fn q2(&mut self, n: usize, k: usize) -> Vec<u8> {
            let mut b = Vec::with_capacity(fl::q2_bytes(n, k));
            for _ in 0..n * k / 64 {
                let d = fl::f32_to_f16(0.01 + 0.03 * (self.f(1.0) + 1.0) / 2.0);
                b.extend(d.to_le_bytes());
                for _ in 0..16 {
                    b.push(self.u() as u8);
                }
            }
            b
        }
    }

    /// |got − want| <= tol · (scale + |want|), `scale` the mean |want|.
    fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let scale = want.iter().map(|v| v.abs()).sum::<f32>() / want.len().max(1) as f32;
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol * (scale + w.abs()),
                "{what} at {i}: {g} vs {w} (scale {scale})"
            );
        }
    }

    fn same_bits(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(g.to_bits() == w.to_bits(), "{what} at {i}: {g} vs {w}");
        }
    }

    fn win<D: ComputeDevice>(d: &D, pos0: usize, n_keep: usize) -> D::Buffer {
        d.upload_u32(&[pos0 as u32, n_keep as u32, 0])
    }

    // ---- Q2 / int8 / dense GEMVs ----

    fn q2_linear_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for &(k, n) in &[(2560, 300), (640, 2560), (10240, 64)] {
            let raw = rng.q2(n, k);
            let (gw, cw) = (g.upload_q2(&raw, n, k), c.upload_q2(&raw, n, k));
            same_bits(&gw.to_vec(), &c.download(&cw), "q2 to_vec");
            for t in TS {
                let mut x = rng.vec(t * k, 3.0);
                x[5] = 0.0; // a zero chunk element, and below a whole zero chunk
                for v in x.iter_mut().skip(64).take(32) {
                    *v = 0.0;
                }
                let qa = QAct { m: t, k };
                let (mut gq, mut cq) = (g.alloc_f32(qa.words()), c.alloc_f32(qa.words()));
                g.quantize_act_into(&g.upload_f32(&x), &mut gq, t, k);
                c.quantize_act_into(&c.upload_f32(&x), &mut cq, t, k);
                same_bits(
                    &g.download(&gq),
                    &c.download(&cq),
                    &format!("quantize t={t} k={k}"),
                );
                let (mut gy, mut cy) = (g.alloc_f32(t * n), c.alloc_f32(t * n));
                g.q2_linear_into(&gq, &gw, &mut gy, t, k, n);
                c.q2_linear_into(&cq, &cw, &mut cy, t, k, n);
                close(
                    &g.download(&gy),
                    &c.download(&cy),
                    1e-5,
                    &format!("q2 linear t={t} {k}->{n}"),
                );
            }
        }
    }

    fn q4x_linear_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(0x51);
        for &(k, n) in &[(2560, 300), (6144, 129), (10240, 40)] {
            let packed: Vec<u32> = (0..n * k / 8).map(|_| rng.u() as u32).collect();
            let sc: Vec<u16> = (0..n * k / 64)
                .map(|_| (rng.f(0.02).to_bits() >> 16) as u16)
                .collect();
            let bi: Vec<u16> = (0..n * k / 64)
                .map(|_| (rng.f(0.1).to_bits() >> 16) as u16)
                .collect();
            let (gw, cw) = (
                g.upload_q4x(&packed, &sc, &bi, n, k),
                c.upload_q4x(&packed, &sc, &bi, n, k),
            );
            // The same weights through tang-Q4 `linear` (f32 activations) agree to int8 rounding.
            let cq4 = c.upload_q4(&packed, &sc, &bi, 64);
            for t in TS {
                let x = rng.vec(t * k, 2.0);
                let qa = QAct { m: t, k };
                let (mut gq, mut cq) = (g.alloc_f32(qa.words()), c.alloc_f32(qa.words()));
                g.quantize_act_into(&g.upload_f32(&x), &mut gq, t, k);
                c.quantize_act_into(&c.upload_f32(&x), &mut cq, t, k);
                let (mut gy, mut cy) = (g.alloc_f32(t * n), c.alloc_f32(t * n));
                g.q4x_linear_into(&gq, &gw, &mut gy, t, k, n);
                c.q4x_linear_into(&cq, &cw, &mut cy, t, k, n);
                let want = c.download(&cy);
                close(
                    &g.download(&gy),
                    &want,
                    1e-5,
                    &format!("q4x linear t={t} {k}->{n}"),
                );
                let f32_path = c.download(&c.linear(&c.upload_f32(&x), &cq4, t, k, n));
                close(&want, &f32_path, 2e-2, &format!("q4x vs f32 q4 t={t}"));
            }
        }
    }

    fn q8x_linear_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(0x88);
        for &(k, n) in &[(2560, 300), (6144, 129), (10240, 40)] {
            let mut raw = Vec::with_capacity(n * k / 32 * fl::Q8_BLOCK_BYTES);
            for _ in 0..n * k / 32 {
                raw.extend(fl::f32_to_f16(0.001 + 0.002 * (rng.f(1.0) + 1.0)).to_le_bytes());
                raw.extend((0..32).map(|_| rng.u() as u8));
            }
            let (gw, cw) = (g.upload_q8x(&raw, n, k), c.upload_q8x(&raw, n, k));
            for t in TS {
                let x = rng.vec(t * k, 2.0);
                let qa = QAct { m: t, k };
                let (mut gq, mut cq) = (g.alloc_f32(qa.words()), c.alloc_f32(qa.words()));
                g.quantize_act_into(&g.upload_f32(&x), &mut gq, t, k);
                c.quantize_act_into(&c.upload_f32(&x), &mut cq, t, k);
                let (mut gy, mut cy) = (g.alloc_f32(t * n), c.alloc_f32(t * n));
                g.q8x_linear_into(&gq, &gw, &mut gy, t, k, n);
                c.q8x_linear_into(&cq, &cw, &mut cy, t, k, n);
                let want = c.download(&cy);
                close(
                    &g.download(&gy),
                    &want,
                    1e-5,
                    &format!("q8x linear t={t} {k}->{n}"),
                );
                // Against the dequantized weights and the f32 activations: int8 rounding only.
                let mut deq = vec![0f32; n * k];
                for (b, blk) in raw.chunks(fl::Q8_BLOCK_BYTES).enumerate() {
                    let d = fl::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    for i in 0..32 {
                        deq[b * 32 + i] = d * blk[2 + i] as i8 as f32;
                    }
                }
                let f32_path =
                    c.download(&c.linear(&c.upload_f32(&x), &c.upload_f32(&deq), t, k, n));
                close(&want, &f32_path, 2e-2, &format!("q8x vs f32 t={t}"));
            }
        }
    }

    /// Native GGUF types: GPU against the CPU reference (same pinned per-chunk arithmetic), and
    /// both within f32 rounding of dequant(W) · dequant(q8(x)) in f64 (the engine's gate).
    fn native_linear_vs_cpu<D: ComputeDevice>(g: &D) {
        use crate::flash_native::{
            nat_decode,
            tests::{random_gguf, ALL},
        };
        let c = CpuDevice::new();
        let mut rng = Rng(0x7a7);
        for ty in ALL {
            for &(k, n) in &[(2560, 300), (6144, 129)] {
                let raw = random_gguf(ty, n, k, rng.u());
                let (gw, cw) = (
                    g.upload_native(ty, &raw, n, k),
                    c.upload_native(ty, &raw, n, k),
                );
                let ir = nat_decode(ty, &raw, n, k);
                for t in TS {
                    let x = rng.vec(t * k, 2.0);
                    let qa = QAct { m: t, k };
                    let (mut gq, mut cq) = (g.alloc_f32(qa.words()), c.alloc_f32(qa.words()));
                    g.quantize_act_into(&g.upload_f32(&x), &mut gq, t, k);
                    c.quantize_act_into(&c.upload_f32(&x), &mut cq, t, k);
                    let (mut gy, mut cy) = (g.alloc_f32(t * n), c.alloc_f32(t * n));
                    g.native_linear_into(ty, &gq, &gw, &mut gy, t, k, n);
                    c.native_linear_into(ty, &cq, &cw, &mut cy, t, k, n);
                    let (got, want) = (g.download(&gy), c.download(&cy));
                    close(&got, &want, 1e-5, &format!("native {ty:?} t={t} {k}->{n}"));
                    // Into a slice of a wider output: the same bits, the rest untouched.
                    let (off, os) = (5, n + 11);
                    let mut gs = g.upload_f32(&vec![7.0; off + t * os]);
                    g.native_linear_out_into(ty, &gq, &gw, &mut gs, off, os, t, k, n);
                    let s = g.download(&gs);
                    for r in 0..t {
                        assert!(s[off + r * os..off + r * os + n]
                            .iter()
                            .zip(&got[r * n..(r + 1) * n])
                            .all(|(a, b)| a.to_bits() == b.to_bits()));
                        assert!(s[off + r * os + n..off + (r + 1) * os]
                            .iter()
                            .all(|&v| v == 7.0));
                    }
                    assert!(s[..off].iter().all(|&v| v == 7.0));
                    // Window-width invariance: column 0 alone gives the same bits.
                    if t > 1 {
                        let mut g1 = g.alloc_f32(QAct { m: 1, k }.words());
                        g.quantize_act_into(&g.upload_f32(&x[..k]), &mut g1, 1, k);
                        let mut y1 = g.alloc_f32(n);
                        g.native_linear_into(ty, &g1, &gw, &mut y1, 1, k, n);
                        let y1 = g.download(&y1);
                        assert!(
                            y1.iter()
                                .zip(&got[..n])
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "native {ty:?} t={t}: column 0 differs from the T=1 result"
                        );
                    }
                    // The gate: against the dequantized product in f64.
                    let qw = u32s(&c.download(&cq));
                    for r in 0..t {
                        let xr = crate::cpu::flash::QRow::decode(&qw, qa, r);
                        for o in (0..n).step_by(17) {
                            let (mut dot, mut mag) = (0f64, 0f64);
                            for e in 0..k {
                                let w = ir.scale[(o * k + e) / 16] as f64
                                    * ir.code[o * k + e] as f64
                                    - ir.min[(o * k + e) / 32] as f64;
                                let xv = xr.q[e] as f64 * xr.d[e / 32] as f64;
                                dot += w * xv;
                                mag += (w * xv).abs();
                            }
                            let gv = got[r * n + o] as f64;
                            assert!(
                                (gv - dot).abs() <= 2e-6 * mag + 1e-30,
                                "native {ty:?} gate t={t} row {o}: {gv} vs {dot} (mag {mag})"
                            );
                        }
                    }
                }
            }
        }
    }

    fn linear_into_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(7);
        let (k, n) = (2560, 333);
        let wb = rng.bf16(n * k, 0.5);
        let packed: Vec<u32> = (0..n * k / 8).map(|_| rng.u() as u32).collect();
        let (sc, bi) = (rng.bf16(n * k / 64, 0.02), rng.bf16(n * k / 64, 0.1));
        for t in TS {
            let x = rng.vec(t * k, 1.0);
            for (gw, cw) in [
                (g.upload_bf16(&wb), c.upload_bf16(&wb)),
                (
                    g.upload_q4(&packed, &sc, &bi, 64),
                    c.upload_q4(&packed, &sc, &bi, 64),
                ),
            ] {
                let mut gy = g.alloc_f32(t * n);
                g.linear_into(&g.upload_f32(&x), &gw, &mut gy, t, k, n);
                let cy = c.linear(&c.upload_f32(&x), &cw, t, k, n);
                close(
                    &g.download(&gy),
                    &c.download(&cy),
                    1e-4,
                    &format!("linear_into t={t}"),
                );
            }
        }
    }

    /// A stacked projection (mixed native types and bf16, one launch) against the per-segment
    /// calls: bitwise, and each column of a T=8 window bitwise its T=1 result.
    fn native_stack_vs_segments(g: &CudaComputeDevice) {
        use crate::flash_native::{tests::random_gguf, NatSeg, NatType};
        let mut rng = Rng(0x57ac);
        for k in [2560usize, 6144] {
            // Segment layout of a GDN w_in (shrunk): two native types, then two bf16 strips.
            let parts: Vec<(Option<NatType>, usize)> = vec![
                (Some(NatType::Iq4Xs), 1000),
                (Some(NatType::Q4K), 600),
                (None, 48),
                (None, 48),
                (Some(NatType::Q3K), 130),
                (Some(NatType::Q2_0), 64),
            ];
            let n: usize = parts.iter().map(|p| p.1).sum();
            let (mut segs, mut off) = (vec![], 0);
            for &(ty, rows) in &parts {
                let w = match ty {
                    Some(ty) => g.upload_native(ty, &random_gguf(ty, rows, k, rng.u()), rows, k),
                    None => g.upload_bf16(&rng.bf16(rows * k, 0.5)),
                };
                segs.push(NatSeg { ty, w, rows, off });
                off += rows;
            }
            let st = g.native_stack(segs, n, k);
            let x = rng.vec(TW * k, 1.0);
            let run = |m: usize, xs: &[f32], stacked: bool| -> Vec<f32> {
                let xb = g.upload_f32(xs);
                let mut xq = g.alloc_f32(QAct { m, k }.words());
                g.quantize_act_into(&xb, &mut xq, m, k);
                let mut y = g.upload_f32(&vec![0.0; m * n]);
                if stacked {
                    g.native_stack_into(&st, &xb, &xq, &mut y, m);
                } else {
                    for sg in &st.segs {
                        match sg.ty {
                            Some(ty) => g.native_linear_out_into(
                                ty, &xq, &sg.w, &mut y, sg.off, n, m, k, sg.rows,
                            ),
                            None => {
                                g.bf16_linear_out_into(&xb, &sg.w, &mut y, sg.off, n, m, k, sg.rows)
                            }
                        }
                    }
                }
                g.download(&y)
            };
            for m in TS {
                let a = run(m, &x[..m * k], true);
                let b = run(m, &x[..m * k], false);
                same_bits(&a, &b, &format!("stack k={k} m={m}"));
                for j in 0..m {
                    let c = run(1, &x[j * k..(j + 1) * k], true);
                    same_bits(
                        &a[j * n..(j + 1) * n],
                        &c,
                        &format!("stack k={k} m={m} col {j} vs T=1"),
                    );
                }
            }
        }
    }

    /// The bf16 GEMV into a slice of a wider output: the plain GEMV's bits, the rest untouched.
    fn bf16_linear_out<D: ComputeDevice>(g: &D) {
        let mut rng = Rng(77);
        for &(k, n) in &[(2560, 513), (2560, 48), (6144, 2560)] {
            let wb = g.upload_bf16(&rng.bf16(n * k, 0.5));
            for t in TS {
                let x = g.upload_f32(&rng.vec(t * k, 1.0));
                let mut y = g.alloc_f32(t * n);
                g.linear_into(&x, &wb, &mut y, t, k, n);
                let y = g.download(&y);
                let (off, os) = (3, n + 29);
                let mut o = g.upload_f32(&vec![5.0; off + t * os]);
                g.bf16_linear_out_into(&x, &wb, &mut o, off, os, t, k, n);
                let o = g.download(&o);
                for r in 0..t {
                    same_bits(
                        &o[off + r * os..off + r * os + n],
                        &y[r * n..(r + 1) * n],
                        "bf16 out rows",
                    );
                    assert!(o[off + r * os + n..off + (r + 1) * os]
                        .iter()
                        .all(|&v| v == 5.0));
                }
                assert!(o[..off].iter().all(|&v| v == 5.0));
            }
        }
    }

    // ---- hyper-connections ----

    /// (norm, down, up, inject, r0) on the host.
    type HcWeightsHost<'a> = (&'a [f32], &'a [u16], &'a [u16], &'a [u16], &'a [f32]);
    /// (y, inj, parts, router weights, logits) on the host.
    type HcInputsHost<'a> = (&'a [f32], &'a [f32], &'a [f32], &'a [f32], &'a [f32]);

    fn hc_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(11);
        let w = HC * HIDDEN;
        let norm: Vec<f32> = rng.vec(w, 1.0).iter().map(|v| 1.0 + 0.2 * v).collect();
        let (down, up, inject) = (
            rng.bf16(HC_LR * w, 0.05),
            fl::hc_up_repack(&rng.bf16(w * HC_LR, 0.2)),
            rng.bf16(HC * w, 0.05),
        );
        let ls = EXPERTS + 1;
        // (inject rows, pending: 0 none, 1 write, 2 MoE write)
        for t in TS {
            for (with_inject, mode) in [(true, 0), (true, 1), (false, 1), (true, 2)] {
                let r0 = rng.vec(t * w, 2.0);
                let (y, injp) = (rng.vec(t * HIDDEN, 1.0), rng.vec(t * HC, 3.0));
                let parts = rng.vec(MoePlan::PARTS_ROWS * HIDDEN, 1.0);
                let (rw, logits) = (rng.vec(t * TOPK, 0.3), rng.vec(t * ls, 2.0));
                // (x, inj, r, xq) after one read on `d`; `split` does the write as its own op.
                fn go<D: ComputeDevice>(
                    d: &D,
                    split: bool,
                    a: HcWeightsHost,
                    b: HcInputsHost,
                    with_inject: bool,
                    mode: u32,
                    t: usize,
                ) -> [Vec<f32>; 4] {
                    let (norm, down, up, inject, r0) = a;
                    let (y, injp, parts, rw, logits) = b;
                    let ls = EXPERTS + 1;
                    let (n, dn, u, i) = (
                        d.upload_f32(norm),
                        d.upload_bf16(down),
                        d.upload_bf16(up),
                        d.upload_bf16(inject),
                    );
                    let hw = HcWeights {
                        norm: &n,
                        down: &dn,
                        up: &u,
                        inject: with_inject.then_some(&i),
                        q8: false,
                    };
                    let mut r = d.upload_f32(r0);
                    let (yb, ib) = (d.upload_f32(y), d.upload_f32(injp));
                    let (pb, wb, lb) =
                        (d.upload_f32(parts), d.upload_f32(rw), d.upload_f32(logits));
                    let pending = match mode {
                        0 => None,
                        1 => Some(HcPending::Write { y: &yb, inj: &ib }),
                        _ => Some(HcPending::Moe {
                            parts: &pb,
                            w: &wb,
                            logits: &lb,
                            stride: ls,
                            sg: Some(EXPERTS),
                            inj: &ib,
                        }),
                    };
                    let pending = if split && mode > 0 {
                        let mut yc = d.alloc_f32(t * HIDDEN);
                        let yref = if mode == 2 {
                            d.moe_combine_into(&pb, &wb, &lb, ls, Some(EXPERTS), &mut yc, t);
                            &yc
                        } else {
                            &yb
                        };
                        d.hc_write(&mut r, yref, &ib, t);
                        None
                    } else {
                        pending
                    };
                    let (mut x, mut inj, mut sc, mut xq) = (
                        d.alloc_f32(t * HIDDEN),
                        d.alloc_f32(t * HC),
                        d.alloc_f32(fl::hc_scratch_words(t)),
                        d.alloc_f32(QAct { m: t, k: HIDDEN }.words()),
                    );
                    d.hc_read_into(
                        &mut r,
                        pending,
                        &hw,
                        &mut x,
                        Some(&mut xq),
                        with_inject.then_some(&mut inj),
                        &mut sc,
                        t,
                        1e-6,
                    );
                    [
                        d.download(&x),
                        d.download(&inj),
                        d.download(&r),
                        d.download(&xq),
                    ]
                }
                let a = (&norm[..], &down[..], &up[..], &inject[..], &r0[..]);
                let b = (&y[..], &injp[..], &parts[..], &rw[..], &logits[..]);
                let got = go(g, false, a, b, with_inject, mode, t);
                let want = go(&c, false, a, b, with_inject, mode, t);
                let what = format!("hc t={t} inject={with_inject} pending={mode}");
                close(&got[2], &want[2], 1e-6, &format!("{what} R"));
                close(&got[0], &want[0], 2e-4, &format!("{what} x"));
                if with_inject {
                    close(&got[1], &want[1], 2e-4, &format!("{what} inj"));
                }
                // The fused int8 output is the quantization of x.
                let mut q = g.alloc_f32(QAct { m: t, k: HIDDEN }.words());
                g.quantize_act_into(&g.upload_f32(&got[0]), &mut q, t, HIDDEN);
                same_bits(&got[3], &g.download(&q), &format!("{what} xq"));
                if mode > 0 {
                    // Fused write-then-read is bitwise the separate ops.
                    let split = go(g, true, a, b, with_inject, mode, t);
                    same_bits(&split[2], &got[2], &format!("{what} unfused R"));
                    same_bits(&split[0], &got[0], &format!("{what} unfused x"));
                }
            }
        }
    }

    /// The 8-bit hc weights: GPU against the CPU reference on the dequantized weights.
    fn hc_q8_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(13);
        let w = HC * HIDDEN;
        let norm: Vec<f32> = rng.vec(w, 1.0).iter().map(|v| 1.0 + 0.2 * v).collect();
        let (db, ub, ib) = (
            rng.bf16(HC_LR * w, 0.05),
            rng.bf16(w * HC_LR, 0.2),
            rng.bf16(HC * w, 0.05),
        );
        let (qd, qu, qi) = (
            fl::hc_q8(&db, HC_LR, w, false),
            fl::hc_q8(&ub, w, HC_LR, true),
            fl::hc_q8(&ib, HC, w, false),
        );
        // The dequantized weights are within half a step of bf16 ones.
        let back = fl::hc_q8_dequant(&qd, HC_LR, w, false);
        for (blk, chunk) in db.chunks(32).enumerate() {
            let f: Vec<f32> = chunk
                .iter()
                .map(|&b| f32::from_bits((b as u32) << 16))
                .collect();
            let d = f.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
            for (i, v) in f.iter().enumerate() {
                // Half a step, plus the f16 rounding of d times up to 127.
                assert!((back[blk * 32 + i] - v).abs() <= 0.5 * d + 127.0 * d / 1024.0 + 1e-9);
            }
        }
        for t in TS {
            for (with_inject, mode) in [(true, 0), (true, 1), (false, 1)] {
                let r0 = rng.vec(t * w, 2.0);
                let (y, injp) = (rng.vec(t * HIDDEN, 1.0), rng.vec(t * HC, 3.0));
                macro_rules! go {
                    ($d:expr) => {{
                        let d = $d;
                        let (n, dn, u, i) = (
                            d.upload_f32(&norm),
                            d.upload_bytes(&qd),
                            d.upload_bytes(&qu),
                            d.upload_bytes(&qi),
                        );
                        let hw = HcWeights {
                            norm: &n,
                            down: &dn,
                            up: &u,
                            inject: with_inject.then_some(&i),
                            q8: true,
                        };
                        let mut r = d.upload_f32(&r0);
                        let (yb, ib) = (d.upload_f32(&y), d.upload_f32(&injp));
                        let pending = (mode == 1).then_some(HcPending::Write { y: &yb, inj: &ib });
                        let (mut x, mut inj, mut sc, mut xq) = (
                            d.alloc_f32(t * HIDDEN),
                            d.alloc_f32(t * HC),
                            d.alloc_f32(fl::hc_scratch_words(t)),
                            d.alloc_f32(QAct { m: t, k: HIDDEN }.words()),
                        );
                        d.hc_read_into(
                            &mut r,
                            pending,
                            &hw,
                            &mut x,
                            Some(&mut xq),
                            with_inject.then_some(&mut inj),
                            &mut sc,
                            t,
                            1e-6,
                        );
                        (d.download(&x), d.download(&inj), d.download(&r))
                    }};
                }
                let got = go!(g);
                let want = go!(&c);
                let what = format!("hc q8 t={t} inject={with_inject} pending={mode}");
                close(&got.2, &want.2, 1e-6, &format!("{what} R"));
                close(&got.0, &want.0, 2e-4, &format!("{what} x"));
                if with_inject {
                    close(&got.1, &want.1, 2e-4, &format!("{what} inj"));
                }
            }
        }
    }

    // ---- GDN ----

    struct GdnCase {
        proj: Vec<f32>,
        hist: Vec<f32>,
        conv: Vec<f32>,
        dt: Vec<f32>,
        a: Vec<f32>,
        norm: Vec<f32>,
        state: Vec<f32>,
    }

    fn gdn_case(rng: &mut Rng, t: usize) -> GdnCase {
        let mut proj = rng.vec(t * GDN_PROJ, 1.0);
        for tt in 0..t {
            for h in 0..GDN_HV {
                proj[tt * GDN_PROJ + GDN_A + h] = rng.f(3.0);
                proj[tt * GDN_PROJ + GDN_B + h] = rng.f(3.0);
            }
        }
        // One head's a well past the softplus cutoff.
        proj[GDN_A + 5] = 30.0;
        GdnCase {
            proj,
            hist: rng.vec(fl::GDN_HIST, 1.0),
            conv: rng.vec(GDN_CONV * GDN_TAPS, 0.6),
            dt: rng.vec(GDN_HV, 1.0),
            a: rng
                .vec(GDN_HV, 1.0)
                .iter()
                .map(|v| -(v.abs() + 0.05))
                .collect(),
            norm: rng.vec(GDN_D, 1.0).iter().map(|v| 1.0 + 0.3 * v).collect(),
            state: rng.vec(fl::GDN_STATE, 0.3),
        }
    }

    /// (h, y readonly, y commit, state after commit, hist after commit)
    fn gdn_run<D: ComputeDevice>(d: &D, cs: &GdnCase, t: usize, n_keep: usize) -> [Vec<f32>; 5] {
        let (proj, hist, conv) = (
            d.upload_f32(&cs.proj),
            d.upload_f32(&cs.hist),
            d.upload_f32(&cs.conv),
        );
        let (dt, a, norm) = (
            d.upload_f32(&cs.dt),
            d.upload_f32(&cs.a),
            d.upload_f32(&cs.norm),
        );
        let p = GdnParams {
            conv: &conv,
            dt_bias: &dt,
            ssm_a: &a,
            norm: &norm,
        };
        let mut state = d.upload_f32(&cs.state);
        let mut h = d.alloc_f32(t * GDN_CONV);
        d.gdn_conv_into(&proj, GDN_PROJ, &hist, &conv, &mut h, t, 1e-6);
        let mut y = d.alloc_f32(t * GDN_V);
        let mut yq = d.alloc_f32(QAct { m: t, k: GDN_V }.words());
        d.gdn_step(
            &mut state,
            &h,
            &proj,
            GDN_PROJ,
            &p,
            &mut y,
            Some(&mut yq),
            t,
            GdnMode::ReadOnly,
            1e-6,
        );
        let mut q = d.alloc_f32(QAct { m: t, k: GDN_V }.words());
        d.quantize_act_into(&y, &mut q, t, GDN_V);
        same_bits(&d.download(&yq), &d.download(&q), "gdn fused int8 output");
        // Conv and step in one launch: bitwise the two.
        let mut yf = d.alloc_f32(t * GDN_V);
        d.gdn_conv_step(
            &mut state,
            &proj,
            GDN_PROJ,
            &hist,
            &p,
            &mut yf,
            None,
            t,
            GdnMode::ReadOnly,
            1e-6,
        );
        same_bits(&d.download(&yf), &d.download(&y), "gdn conv+step fused");
        let unchanged = d.download(&state);
        same_bits(&unchanged, &cs.state, "readonly leaves the state");
        let w = win(d, 0, n_keep);
        let mut yc = d.alloc_f32(t * GDN_V);
        d.gdn_step(
            &mut state,
            &h,
            &proj,
            GDN_PROJ,
            &p,
            &mut yc,
            None,
            t,
            GdnMode::Commit { win: &w },
            1e-6,
        );
        let mut hist2 = d.upload_f32(&cs.hist);
        d.gdn_conv_commit(&mut hist2, &proj, GDN_PROJ, &w, t);
        [
            d.download(&h),
            d.download(&y),
            d.download(&yc),
            d.download(&state),
            d.download(&hist2),
        ]
    }

    fn gdn_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(23);
        for t in TS {
            let cs = gdn_case(&mut rng, t);
            for n_keep in [0, 1, t.div_ceil(2), t, t + 3] {
                let got = gdn_run(g, &cs, t, n_keep);
                let want = gdn_run(&c, &cs, t, n_keep);
                let what = format!("gdn t={t} n_keep={n_keep}");
                close(&got[0], &want[0], 1e-5, &format!("{what} conv"));
                close(&got[1], &want[1], 2e-4, &format!("{what} y"));
                close(&got[3], &want[3], 1e-4, &format!("{what} state"));
                same_bits(&got[4], &want[4], &format!("{what} conv commit"));
                let k = n_keep.min(t);
                // Commit replays exactly what verify computed.
                same_bits(
                    &got[2][..k * GDN_V],
                    &got[1][..k * GDN_V],
                    &format!("{what} commit y"),
                );
                if k < t {
                    // And the next window, from the committed state, continues bitwise.
                    let mut rest = GdnCase {
                        proj: cs.proj[k * GDN_PROJ..].to_vec(),
                        state: got[3].clone(),
                        ..gdn_case(&mut Rng(1), 1)
                    };
                    rest.hist = cs.hist.clone();
                    let (proj, h) = (
                        g.upload_f32(&rest.proj),
                        g.upload_f32(&got[0][k * GDN_CONV..]),
                    );
                    let (dt, a, norm, conv) = (
                        g.upload_f32(&cs.dt),
                        g.upload_f32(&cs.a),
                        g.upload_f32(&cs.norm),
                        g.upload_f32(&cs.conv),
                    );
                    let p = GdnParams {
                        conv: &conv,
                        dt_bias: &dt,
                        ssm_a: &a,
                        norm: &norm,
                    };
                    let mut state = g.upload_f32(&got[3]);
                    let mut y = g.alloc_f32((t - k) * GDN_V);
                    g.gdn_step(
                        &mut state,
                        &h,
                        &proj,
                        GDN_PROJ,
                        &p,
                        &mut y,
                        None,
                        t - k,
                        GdnMode::ReadOnly,
                        1e-6,
                    );
                    same_bits(
                        &g.download(&y),
                        &got[1][k * GDN_V..],
                        &format!("{what} continue"),
                    );
                }
            }
        }
    }

    // ---- MoE ----

    fn router_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(31);
        let stride = EXPERTS + 1;
        for t in TS {
            let mut l = rng.vec(t * stride, 4.0);
            // Ties straddling the cut, and a run of equal logits.
            for e in [3, 40, 77, 300, 301] {
                l[e] = 5.5;
            }
            if t > 1 {
                l[stride + 9] = 9.0;
                l[stride + 10] = 9.0;
            }
            let (mut gi, mut gw, mut ci, mut cw) = (
                g.alloc_f32(t * TOPK),
                g.alloc_f32(t * TOPK),
                c.alloc_f32(t * TOPK),
                c.alloc_f32(t * TOPK),
            );
            g.router_topk_into(&g.upload_f32(&l), stride, EXPERTS, &mut gi, &mut gw, t);
            c.router_topk_into(&c.upload_f32(&l), stride, EXPERTS, &mut ci, &mut cw, t);
            assert_eq!(
                u32s(&g.download(&gi)),
                u32s(&c.download(&ci)),
                "router ids t={t}"
            );
            close(
                &g.download(&gw),
                &c.download(&cw),
                1e-6,
                &format!("router w t={t}"),
            );
        }
    }

    fn moe_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(41);
        // Eight distinct blobs; experts map onto them, some not resident.
        let blobs: Vec<Vec<u8>> = (0..8)
            .map(|_| {
                ExpertBlob::from_gguf(
                    &rng.q2(FF, HIDDEN),
                    &rng.q2(FF, HIDDEN),
                    &rng.q2(HIDDEN, FF),
                )
            })
            .collect();
        let shared_blob = ExpertBlob::from_gguf(
            &rng.q2(FF, HIDDEN),
            &rng.q2(FF, HIDDEN),
            &rng.q2(HIDDEN, FF),
        );
        let (gb, cb): (Vec<_>, Vec<_>) = blobs
            .iter()
            .map(|b| (g.upload_bytes(b), c.upload_bytes(b)))
            .unzip();
        let (gs, cs) = (g.upload_bytes(&shared_blob), c.upload_bytes(&shared_blob));
        let table = |d: &dyn Fn(usize) -> u64| -> Vec<u64> {
            (0..EXPERTS)
                .map(|e| if e % 7 == 3 { 0 } else { d(e % 8) })
                .collect()
        };
        let gt = table(&|i| g.buffer_addr(&gb[i]));
        let ct = table(&|i| c.buffer_addr(&cb[i]));
        let tw = |t: &[u64]| -> Vec<u32> {
            t.iter()
                .flat_map(|&p| [p as u32, (p >> 32) as u32])
                .collect()
        };
        for t in TS {
            // Few distinct experts per window so groups have several entries.
            let mut ids = vec![];
            for tt in 0..t {
                let mut row: Vec<u32> = vec![];
                while row.len() < TOPK {
                    let e = (rng.u() % 24) as u32 * 5 + (tt as u32 % 2);
                    if !row.contains(&e) {
                        row.push(e);
                    }
                }
                ids.extend(row);
            }
            let x = rng.vec(t * HIDDEN, 1.0);
            let w = rng.vec(t * TOPK, 0.3);
            let logits = rng.vec(t * (EXPERTS + 1), 2.0);
            for shared in [false, true] {
                // Plan: the device kernel against the reference on the same table.
                let gsh = if shared { g.buffer_addr(&gs) } else { 0 };
                let mut gplan = g.alloc_f32(MoePlan::WORDS);
                g.moe_plan_into(
                    &g.upload_u32(&ids),
                    &g.upload_u32(&tw(&gt)),
                    gsh,
                    &mut gplan,
                    t,
                );
                let want = crate::cpu::flash::moe_plan(&ids, &gt, gsh, t);
                let got = u32s(&g.download(&gplan));
                let (ng, ne, nm) = (want[0] as usize, want[1] as usize, want[2] as usize);
                assert_eq!(&got[..3], &want[..3], "plan header t={t}");
                assert_eq!(
                    got[MoePlan::GROUP_PTR..][..2 * ng],
                    want[MoePlan::GROUP_PTR..][..2 * ng],
                    "plan ptrs"
                );
                assert_eq!(
                    got[MoePlan::GROUP_START..][..ng + 1],
                    want[MoePlan::GROUP_START..][..ng + 1],
                    "plan starts"
                );
                assert_eq!(
                    got[MoePlan::ENT_TOK..][..ne],
                    want[MoePlan::ENT_TOK..][..ne],
                    "plan tok"
                );
                assert_eq!(
                    got[MoePlan::ENT_DST..][..ne],
                    want[MoePlan::ENT_DST..][..ne],
                    "plan dst"
                );
                assert_eq!(
                    got[MoePlan::MISSING..][..nm],
                    want[MoePlan::MISSING..][..nm],
                    "plan missing"
                );
                assert!(nm > 0 && ng > 0);

                // The fused route: router ids/weights as router_topk, plan as moe_plan on them,
                // and with forced ids, the plan above.
                let gl = g.upload_f32(&logits);
                let (mut ri, mut rw, mut rp) = (
                    g.alloc_f32(t * TOPK),
                    g.alloc_f32(t * TOPK),
                    g.alloc_f32(MoePlan::WORDS),
                );
                let gtab = g.upload_u32(&tw(&gt));
                g.moe_route_into(
                    &gl,
                    EXPERTS + 1,
                    EXPERTS,
                    None,
                    &gtab,
                    gsh,
                    &mut ri,
                    &mut rw,
                    &mut rp,
                    t,
                );
                let (mut ti, mut tw2) = (g.alloc_f32(t * TOPK), g.alloc_f32(t * TOPK));
                g.router_topk_into(&gl, EXPERTS + 1, EXPERTS, &mut ti, &mut tw2, t);
                same_bits(&g.download(&ri), &g.download(&ti), "route ids");
                same_bits(&g.download(&rw), &g.download(&tw2), "route w");
                let want_r = crate::cpu::flash::moe_plan(&u32s(&g.download(&ti)), &gt, gsh, t);
                let got_r = u32s(&g.download(&rp));
                let ner = want_r[1] as usize;
                assert_eq!(&got_r[..3], &want_r[..3], "route plan header");
                assert_eq!(
                    got_r[MoePlan::ENT_DST..][..ner],
                    want_r[MoePlan::ENT_DST..][..ner],
                    "route plan dst"
                );
                let gids = g.upload_u32(&ids);
                g.moe_route_into(
                    &gl,
                    EXPERTS + 1,
                    EXPERTS,
                    Some(&gids),
                    &gtab,
                    gsh,
                    &mut ri,
                    &mut rw,
                    &mut rp,
                    t,
                );
                let got_f = u32s(&g.download(&rp));
                assert_eq!(&got_f[..3], &want[..3], "forced plan header");
                assert_eq!(
                    got_f[MoePlan::ENT_TOK..][..ne],
                    want[MoePlan::ENT_TOK..][..ne],
                    "forced plan tok"
                );

                // Grouped experts and combine, each device with its own addresses.
                let run = |d: &dyn Fn() -> (Vec<f32>, Vec<f32>)| d();
                macro_rules! go {
                    ($d:expr, $tab:expr, $sh:expr) => {{
                        let d = $d;
                        let sh = if shared { $sh } else { 0 };
                        let mut plan = d.alloc_f32(MoePlan::WORDS);
                        d.moe_plan_into(
                            &d.upload_u32(&ids),
                            &d.upload_u32(&tw($tab)),
                            sh,
                            &mut plan,
                            t,
                        );
                        let mut xq = d.alloc_f32(QAct { m: t, k: HIDDEN }.words());
                        d.quantize_act_into(&d.upload_f32(&x), &mut xq, t, HIDDEN);
                        let (mut sc, mut parts) = (
                            d.alloc_f32(MoePlan::scratch_words()),
                            d.alloc_f32(MoePlan::PARTS_ROWS * HIDDEN),
                        );
                        unsafe { d.moe_grouped_into(&xq, &plan, &mut sc, &mut parts, t) };
                        let mut y = d.alloc_f32(t * HIDDEN);
                        d.moe_combine_into(
                            &parts,
                            &d.upload_f32(&w),
                            &d.upload_f32(&logits),
                            EXPERTS + 1,
                            shared.then_some(EXPERTS),
                            &mut y,
                            t,
                        );
                        (d.download(&parts), d.download(&y))
                    }};
                }
                let gotp = run(&|| go!(g, &gt, g.buffer_addr(&gs)));
                let wantp = run(&|| go!(&c, &ct, c.buffer_addr(&cs)));
                // h is requantized to int8 between gate/up and down, so a summation-order
                // difference can move one code by 1 at a rounding tie: a few outputs may move by
                // d_h · w; nearly all agree to fp32 noise.
                // The expert order is pinned (ExpertBlob::ORDER): GPU rows are the reference's
                // bit for bit, so a CPU-served miss and a VRAM hit agree exactly.
                let what = format!("moe parts t={t} shared={shared}");
                same_bits(&gotp.0, &wantp.0, &what);
                // Combine on the same parts.
                let mut cy = c.alloc_f32(t * HIDDEN);
                let ls = EXPERTS + 1;
                let sg = shared.then_some(EXPERTS);
                let (cp, cw, cl) = (
                    c.upload_f32(&gotp.0),
                    c.upload_f32(&w),
                    c.upload_f32(&logits),
                );
                c.moe_combine_into(&cp, &cw, &cl, ls, sg, &mut cy, t);
                let what = format!("moe combine t={t} shared={shared}");
                close(&gotp.1, &c.download(&cy), 1e-5, &what);
            }
        }
    }

    // ---- QSA ----

    fn qsa_prep_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(53);
        let max_ctx = 128;
        let (cos, sin) = fl::rope_table(max_ctx, 1e7);
        let norms: Vec<Vec<f32>> = [QSA_D, QSA_D, IDX_D, IDX_D]
            .iter()
            .map(|&n| rng.vec(n, 1.0).iter().map(|v| 1.0 + 0.3 * v).collect())
            .collect();
        for t in TS {
            // A run of windows from position 0, so blocks complete inside and across windows.
            let windows: Vec<(usize, Vec<f32>)> = (0..6)
                .map(|i| (i * t, rng.vec(t * QSA_PROJ, 1.0)))
                .collect();
            let run = |d: &dyn Fn() -> [Vec<f32>; 5]| d();
            macro_rules! go {
                ($d:expr) => {{
                    let d = $d;
                    let nb: Vec<_> = norms.iter().map(|v| d.upload_f32(v)).collect();
                    let qn = QsaNorms {
                        q: &nb[0],
                        k: &nb[1],
                        iq: &nb[2],
                        ik: &nb[3],
                    };
                    let (cb, sb) = (d.upload_f32(&cos), d.upload_f32(&sin));
                    let (mut kc, mut vc) = (
                        d.alloc_bf16(max_ctx * QSA_KV * QSA_D),
                        d.alloc_bf16(max_ctx * QSA_KV * QSA_D),
                    );
                    let (mut ring, mut pooled) = (
                        d.alloc_f32(QSA_RING * IDX_D),
                        d.alloc_f32(max_ctx / 4 * IDX_D),
                    );
                    let mut q = d.alloc_f32(fl::qsa_q_words(t));
                    for (pos0, proj) in &windows {
                        let cache = QsaCache {
                            k: &mut kc,
                            v: &mut vc,
                            ring: &mut ring,
                            pooled: &mut pooled,
                        };
                        d.qsa_prep(
                            &d.upload_f32(proj),
                            QSA_PROJ,
                            &win(d, *pos0, 0),
                            &qn,
                            (&cb, &sb),
                            &mut q,
                            cache,
                            t,
                            1e-6,
                        );
                    }
                    [
                        d.download(&q),
                        d.download(&kc),
                        d.download(&vc),
                        d.download(&ring),
                        d.download(&pooled),
                    ]
                }};
            }
            let got = run(&|| go!(g));
            let want = run(&|| go!(&c));
            let n = 6 * t;
            let what = format!("qsa prep t={t}");
            close(&got[0], &want[0], 1e-4, &format!("{what} q"));
            close(
                &got[1][..n * 512],
                &want[1][..n * 512],
                1e-2,
                &format!("{what} k cache"),
            );
            same_bits(
                &got[2][..n * 512],
                &want[2][..n * 512],
                &format!("{what} v cache"),
            );
            same_bits(&got[3], &want[3], &format!("{what} ring"));
            close(
                &got[4][..n / 4 * IDX_D],
                &want[4][..n / 4 * IDX_D],
                1e-4,
                &format!("{what} pooled"),
            );
        }
    }

    fn qsa_select_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(59);
        let max_blocks = 1024;
        let pooled = rng.vec(max_blocks * IDX_D, 1.0);
        for t in TS {
            let mut q = rng.vec(fl::qsa_q_words(t), 1.0);
            // Make heads share signs so plenty of scores tie at a few values.
            for v in q.iter_mut().skip(t * QSA_HEADS * QSA_D) {
                *v = (*v * 4.0).round() / 4.0;
            }
            for &pos0 in &[0usize, 2040, 2048, 2049, 2050, 2051, 2052, 3000, 4093 - t] {
                if pos0 + t > max_blocks * 4 {
                    continue;
                }
                let run = |d: &dyn Fn() -> (Vec<f32>, Vec<f32>)| d();
                macro_rules! go {
                    ($d:expr) => {{
                        let d = $d;
                        let (mut sc, mut ids) =
                            (d.alloc_f32(t * max_blocks), d.alloc_f32(t * QSA_WIDTH));
                        d.qsa_select_into(
                            &d.upload_f32(&pooled),
                            &d.upload_f32(&q),
                            &win(d, pos0, 0),
                            &mut sc,
                            &mut ids,
                            max_blocks,
                            t,
                        );
                        (d.download(&sc), d.download(&ids))
                    }};
                }
                let got = run(&|| go!(g));
                let want = run(&|| go!(&c));
                for tt in 0..t {
                    let n_kv = pos0 + tt + 1;
                    let nb = n_kv / 4;
                    same_bits(
                        &got.0[tt * max_blocks..tt * max_blocks + nb],
                        &want.0[tt * max_blocks..tt * max_blocks + nb],
                        &format!("qsa scores t={t} pos0={pos0} row {tt}"),
                    );
                    let w = crate::cpu::flash::qsa_n_sel(n_kv);
                    assert_eq!(
                        u32s(&got.1[tt * QSA_WIDTH..tt * QSA_WIDTH + w]),
                        u32s(&want.1[tt * QSA_WIDTH..tt * QSA_WIDTH + w]),
                        "qsa ids t={t} pos0={pos0} row {tt}"
                    );
                }
            }
        }
    }

    fn qsa_attend_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(61);
        let max_ctx = 4096;
        let kv: Vec<f32> = rng
            .vec(max_ctx * QSA_KV * QSA_D, 1.0)
            .iter()
            .map(|&v| fl::bf16_round(v))
            .collect();
        let vv: Vec<f32> = rng
            .vec(max_ctx * QSA_KV * QSA_D, 1.0)
            .iter()
            .map(|&v| fl::bf16_round(v))
            .collect();
        let to_bits =
            |v: &[f32]| -> Vec<u16> { v.iter().map(|x| (x.to_bits() >> 16) as u16).collect() };
        for t in TS {
            let q = rng.vec(fl::qsa_q_words(t), 1.0);
            let proj = rng.vec(t * QSA_PROJ, 2.0);
            for &pos0 in &[0usize, 63, 2045, 3500] {
                // Arbitrary ascending selections.
                let mut ids = vec![0u32; t * QSA_WIDTH];
                for tt in 0..t {
                    let n_kv = pos0 + tt + 1;
                    let w = crate::cpu::flash::qsa_n_sel(n_kv);
                    let mut cells: Vec<u32> = (0..n_kv as u32).collect();
                    while cells.len() > w {
                        let i = (rng.u() as usize) % cells.len();
                        cells.remove(i);
                    }
                    ids[tt * QSA_WIDTH..tt * QSA_WIDTH + w].copy_from_slice(&cells);
                }
                let gk = g.upload_bf16(&to_bits(&kv));
                let gv = g.upload_bf16(&to_bits(&vv));
                let mut gs = g.alloc_f32(fl::qsa_attend_scratch_words(t));
                let mut go = g.alloc_f32(t * QSA_OUT);
                let mut gq = g.alloc_f32(QAct { m: t, k: QSA_OUT }.words());
                g.qsa_attend_into(
                    &g.upload_f32(&q),
                    &gk,
                    &gv,
                    &g.upload_u32(&ids),
                    &g.upload_f32(&proj),
                    QSA_PROJ,
                    &win(g, pos0, 0),
                    &mut gs,
                    &mut go,
                    Some(&mut gq),
                    t,
                );
                let mut qq = g.alloc_f32(QAct { m: t, k: QSA_OUT }.words());
                g.quantize_act_into(&go, &mut qq, t, QSA_OUT);
                same_bits(&g.download(&gq), &g.download(&qq), "qsa fused int8 output");
                let mut co = c.alloc_f32(t * QSA_OUT);
                let mut cs = c.alloc_f32(1);
                c.qsa_attend_into(
                    &c.upload_f32(&q),
                    &c.upload_f32(&kv),
                    &c.upload_f32(&vv),
                    &c.upload_u32(&ids),
                    &c.upload_f32(&proj),
                    QSA_PROJ,
                    &win(&c, pos0, 0),
                    &mut cs,
                    &mut co,
                    None,
                    t,
                );
                close(
                    &g.download(&go),
                    &c.download(&co),
                    1e-4,
                    &format!("qsa attend t={t} pos0={pos0}"),
                );
            }
        }
    }

    fn qsa_union_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(67);
        let (max_ctx, max_blocks) = (4096, 1024);
        let to_bits =
            |v: &[f32]| -> Vec<u16> { v.iter().map(|x| (x.to_bits() >> 16) as u16).collect() };
        let kv: Vec<f32> = rng
            .vec(max_ctx * QSA_KV * QSA_D, 1.0)
            .iter()
            .map(|&v| fl::bf16_round(v))
            .collect();
        let vv: Vec<f32> = rng
            .vec(max_ctx * QSA_KV * QSA_D, 1.0)
            .iter()
            .map(|&v| fl::bf16_round(v))
            .collect();
        let pooled = rng.vec(max_blocks * IDX_D, 1.0);
        let (gk, gv, gp) = (
            g.upload_bf16(&to_bits(&kv)),
            g.upload_bf16(&to_bits(&vv)),
            g.upload_f32(&pooled),
        );
        let mut un = g.alloc_f32(fl::qsa_union_words(max_blocks, 8));
        for t in TS {
            let q = rng.vec(fl::qsa_q_words(t), 1.0);
            let proj = rng.vec(t * QSA_PROJ, 2.0);
            for &pos0 in &[0usize, 61, 2047, 2049, 3001, 4095 - t] {
                let w = win(g, pos0, 0);
                let gq = g.upload_f32(&q);
                let (mut sc, mut ids) = (g.alloc_f32(t * max_blocks), g.alloc_f32(t * QSA_WIDTH));
                g.qsa_select_union_into(&gp, &gq, &w, &mut sc, &mut ids, &mut un, max_blocks, t);
                let mut s = g.alloc_f32(fl::qsa_union_scratch_words(t));
                let (mut out, mut oq) = (
                    g.alloc_f32(t * QSA_OUT),
                    g.alloc_f32(QAct { m: t, k: QSA_OUT }.words()),
                );
                let gpr = g.upload_f32(&proj);
                g.qsa_attend_union_into(
                    &gq,
                    &gk,
                    &gv,
                    &ids,
                    &un,
                    max_blocks,
                    &gpr,
                    QSA_PROJ,
                    &w,
                    &mut s,
                    &mut out,
                    Some(&mut oq),
                    t,
                );
                let mut co = c.alloc_f32(t * QSA_OUT);
                let mut cs = c.alloc_f32(1);
                let cw = win(&c, pos0, 0);
                let (cq, ck, cv, ci, cp) = (
                    c.upload_f32(&q),
                    c.upload_f32(&kv),
                    c.upload_f32(&vv),
                    c.upload_f32(&g.download(&ids)),
                    c.upload_f32(&proj),
                );
                c.qsa_attend_into(
                    &cq, &ck, &cv, &ci, &cp, QSA_PROJ, &cw, &mut cs, &mut co, None, t,
                );
                close(
                    &g.download(&out),
                    &c.download(&co),
                    1e-4,
                    &format!("qsa union attend t={t} pos0={pos0}"),
                );
                let mut qq = g.alloc_f32(QAct { m: t, k: QSA_OUT }.words());
                g.quantize_act_into(&out, &mut qq, t, QSA_OUT);
                same_bits(&g.download(&oq), &g.download(&qq), "qsa union int8 output");
            }
        }
        // The mask region is left clear for the next use.
        let left = g.download(&un);
        assert!(
            left[..max_blocks].iter().all(|v| v.to_bits() == 0),
            "union masks not cleared"
        );
    }

    // ---- window-width invariance ----
    //
    // A token's result must not depend on the window it is computed in (speculative verify
    // windows must reproduce T=1 decoding bit for bit): every op, run on a T=8 window, gives
    // each column (token) exactly what it gives that column alone at T=1 -- or, for the stateful
    // ops (GDN, QSA prep), what eight successive T=1 windows give.

    const TW: usize = 8;

    /// Column `j` of a `[m][w]` buffer.
    fn col(v: &[f32], j: usize, w: usize) -> &[f32] {
        &v[j * w..(j + 1) * w]
    }

    /// Column `j` of a QAct (codes, scales, sums) as one vector.
    fn qact_col(v: &[f32], m: usize, k: usize, j: usize) -> Vec<f32> {
        let (kb, nch) = (k / 4, k / 32);
        let mut o = v[j * kb..(j + 1) * kb].to_vec();
        o.extend_from_slice(&v[m * kb + j * nch..m * kb + (j + 1) * nch]);
        o.extend_from_slice(&v[m * kb + m * nch + j * nch..m * kb + m * nch + (j + 1) * nch]);
        o
    }

    struct Inv(Vec<String>);
    impl Inv {
        fn eq(&mut self, a: &[f32], b: &[f32], what: &str) {
            assert_eq!(a.len(), b.len(), "{what}: lengths");
            if let Some(i) = a
                .iter()
                .zip(b)
                .position(|(x, y)| x.to_bits() != y.to_bits())
            {
                self.0.push(format!(
                    "{what}: first difference at {i}: {} vs {}",
                    a[i], b[i]
                ));
            }
        }
    }

    fn t_invariance_w(g: &CudaComputeDevice, tw: usize) {
        let mut inv = Inv(vec![]);
        let mut rng = Rng(0x7177);
        let quant = |x: &[f32], m: usize, k: usize| {
            let mut q = g.alloc_f32(QAct { m, k }.words());
            g.quantize_act_into(&g.upload_f32(x), &mut q, m, k);
            q
        };
        // quantize_act
        for k in [HIDDEN, GDN_V, FF] {
            let x = rng.vec(tw * k, 2.0);
            let q8 = g.download(&quant(&x, tw, k));
            for j in 0..tw {
                let q1 = g.download(&quant(col(&x, j, k), 1, k));
                inv.eq(
                    &qact_col(&q8, tw, k, j),
                    &q1,
                    &format!("quantize k={k} col {j}"),
                );
            }
        }
        // Dense GEMVs: (name, k, n, run(xq or x, m) -> y).
        type Gemv<'a> = Box<dyn Fn(&D, &B, &B, usize) -> B + 'a>;
        type D = CudaComputeDevice;
        type B = CudaBuffer;
        let mut gemvs: Vec<(String, usize, usize, B, Gemv)> = vec![];
        // Shapes either side of the K-split heuristics (router, ple, mixer, head-sized).
        for (k, n) in [
            (2560, 300),
            (6144, 129),
            (2560, 512),
            (2560, 2560),
            (6144, 2560),
            (2560, 10240),
        ] {
            let wb = rng.bf16(n * k, 0.5);
            gemvs.push((
                format!("bf16 linear {k}->{n}"),
                k,
                n,
                g.upload_bf16(&wb),
                Box::new(move |d: &D, x: &B, w: &B, m| {
                    let mut y = d.alloc_f32(m * n);
                    d.linear_into(x, w, &mut y, m, k, n);
                    y
                }),
            ));
            let raw = rng.q2(n, k);
            gemvs.push((
                format!("q2 {k}->{n}"),
                k,
                n,
                g.upload_q2(&raw, n, k),
                Box::new(move |d: &D, xq: &B, w: &B, m| {
                    let mut y = d.alloc_f32(m * n);
                    d.q2_linear_into(xq, w, &mut y, m, k, n);
                    y
                }),
            ));
            let packed: Vec<u32> = (0..n * k / 8).map(|_| rng.u() as u32).collect();
            let (sc, bi) = (rng.bf16(n * k / 64, 0.02), rng.bf16(n * k / 64, 0.1));
            gemvs.push((
                format!("q4x {k}->{n}"),
                k,
                n,
                g.upload_q4x(&packed, &sc, &bi, n, k),
                Box::new(move |d: &D, xq: &B, w: &B, m| {
                    let mut y = d.alloc_f32(m * n);
                    d.q4x_linear_into(xq, w, &mut y, m, k, n);
                    y
                }),
            ));
            let q8raw: Vec<u8> = (0..n * k / 32)
                .flat_map(|_| {
                    let mut b = vec![0u8; fl::Q8_BLOCK_BYTES];
                    b[..2]
                        .copy_from_slice(&fl::f32_to_f16(0.01 + rng.f(0.005).abs()).to_le_bytes());
                    for v in b.iter_mut().skip(2) {
                        *v = rng.u() as u8;
                    }
                    b
                })
                .collect();
            gemvs.push((
                format!("q8x {k}->{n}"),
                k,
                n,
                g.upload_q8x(&q8raw, n, k),
                Box::new(move |d: &D, xq: &B, w: &B, m| {
                    let mut y = d.alloc_f32(m * n);
                    d.q8x_linear_into(xq, w, &mut y, m, k, n);
                    y
                }),
            ));
            for ty in crate::flash_native::tests::ALL {
                if k % ty.block().0 != 0 {
                    continue;
                }
                let raw = crate::flash_native::tests::random_gguf(ty, n, k, rng.u());
                gemvs.push((
                    format!("native {ty:?} {k}->{n}"),
                    k,
                    n,
                    g.upload_native(ty, &raw, n, k),
                    Box::new(move |d: &D, xq: &B, w: &B, m| {
                        let mut y = d.alloc_f32(m * n);
                        d.native_linear_into(ty, xq, w, &mut y, m, k, n);
                        y
                    }),
                ));
            }
        }
        for (name, k, n, w, run) in &gemvs {
            let (k, n) = (*k, *n);
            let x = rng.vec(tw * k, 1.5);
            let f32_in = name.starts_with("bf16");
            let in8 = if f32_in {
                g.upload_f32(&x)
            } else {
                quant(&x, tw, k)
            };
            let y8 = g.download(&run(g, &in8, w, tw));
            for j in 0..tw {
                let in1 = if f32_in {
                    g.upload_f32(col(&x, j, k))
                } else {
                    quant(col(&x, j, k), 1, k)
                };
                let y1 = g.download(&run(g, &in1, w, 1));
                inv.eq(col(&y8, j, n), &y1, &format!("{name} col {j}"));
            }
        }
        // Hyper-connection reads (no pending, pending write, pending MoE write).
        for q8 in [false, true] {
            let w = HC * HIDDEN;
            let norm: Vec<f32> = rng.vec(w, 1.0).iter().map(|v| 1.0 + 0.2 * v).collect();
            let (db, ub, ib) = (
                rng.bf16(HC_LR * w, 0.05),
                rng.bf16(w * HC_LR, 0.2),
                rng.bf16(HC * w, 0.05),
            );
            let (n, dn, u, i) = if q8 {
                (
                    g.upload_f32(&norm),
                    g.upload_bytes(&fl::hc_q8(&db, HC_LR, w, false)),
                    g.upload_bytes(&fl::hc_q8(&ub, w, HC_LR, true)),
                    g.upload_bytes(&fl::hc_q8(&ib, HC, w, false)),
                )
            } else {
                (
                    g.upload_f32(&norm),
                    g.upload_bf16(&db),
                    g.upload_bf16(&fl::hc_up_repack(&ub)),
                    g.upload_bf16(&ib),
                )
            };
            let hw = HcWeights {
                norm: &n,
                down: &dn,
                up: &u,
                inject: Some(&i),
                q8,
            };
            let ls = EXPERTS + 1;
            let r0 = rng.vec(tw * w, 2.0);
            let (y, injp) = (rng.vec(tw * HIDDEN, 1.0), rng.vec(tw * HC, 3.0));
            let parts = rng.vec(MoePlan::parts_rows(tw) * HIDDEN, 1.0);
            let (rw, logits) = (rng.vec(tw * TOPK, 0.3), rng.vec(tw * ls, 2.0));
            // One read over columns `cols` (parts rows remapped to the window's own rows).
            let read = |cols: &[usize], mode: u32| -> [Vec<f32>; 4] {
                let m = cols.len();
                let pick = |v: &[f32], wd: usize| -> Vec<f32> {
                    cols.iter().flat_map(|&j| col(v, j, wd).to_vec()).collect()
                };
                let mut pr = vec![0f32; MoePlan::parts_rows(m) * HIDDEN];
                for (t, &j) in cols.iter().enumerate() {
                    for kk in 0..TOPK {
                        pr[(t * TOPK + kk) * HIDDEN..][..HIDDEN].copy_from_slice(col(
                            &parts,
                            j * TOPK + kk,
                            HIDDEN,
                        ));
                    }
                    pr[(MoePlan::shared_row(m) + t) * HIDDEN..][..HIDDEN].copy_from_slice(col(
                        &parts,
                        MoePlan::shared_row(tw) + j,
                        HIDDEN,
                    ));
                }
                let mut r = g.upload_f32(&pick(&r0, w));
                let (yb, ib) = (
                    g.upload_f32(&pick(&y, HIDDEN)),
                    g.upload_f32(&pick(&injp, HC)),
                );
                let (pb, wb, lb) = (
                    g.upload_f32(&pr),
                    g.upload_f32(&pick(&rw, TOPK)),
                    g.upload_f32(&pick(&logits, ls)),
                );
                let pending = match mode {
                    0 => None,
                    1 => Some(HcPending::Write { y: &yb, inj: &ib }),
                    _ => Some(HcPending::Moe {
                        parts: &pb,
                        w: &wb,
                        logits: &lb,
                        stride: ls,
                        sg: Some(EXPERTS),
                        inj: &ib,
                    }),
                };
                let (mut x, mut inj, mut sc, mut xq) = (
                    g.alloc_f32(m * HIDDEN),
                    g.alloc_f32(m * HC),
                    g.alloc_f32(fl::hc_scratch_words(m)),
                    g.alloc_f32(QAct { m, k: HIDDEN }.words()),
                );
                g.hc_read_into(
                    &mut r,
                    pending,
                    &hw,
                    &mut x,
                    Some(&mut xq),
                    Some(&mut inj),
                    &mut sc,
                    m,
                    1e-6,
                );
                [
                    g.download(&x),
                    g.download(&inj),
                    g.download(&r),
                    g.download(&xq),
                ]
            };
            let all: Vec<usize> = (0..tw).collect();
            for mode in 0..3 {
                let o8 = read(&all, mode);
                for j in 0..tw {
                    let o1 = read(&[j], mode);
                    let what = format!("hc_read q8={q8} pending={mode} col {j}");
                    inv.eq(col(&o8[0], j, HIDDEN), &o1[0], &format!("{what} x"));
                    inv.eq(col(&o8[1], j, HC), &o1[1], &format!("{what} inj"));
                    inv.eq(col(&o8[2], j, w), &o1[2], &format!("{what} r"));
                    inv.eq(
                        &qact_col(&o8[3], tw, HIDDEN, j),
                        &o1[3],
                        &format!("{what} xq"),
                    );
                }
            }
        }
        // Router top-k.
        {
            let stride = EXPERTS + 1;
            let l = rng.vec(tw * stride, 4.0);
            let topk = |v: &[f32], m: usize| {
                let (mut i, mut w) = (g.alloc_f32(m * TOPK), g.alloc_f32(m * TOPK));
                g.router_topk_into(&g.upload_f32(v), stride, EXPERTS, &mut i, &mut w, m);
                (g.download(&i), g.download(&w))
            };
            let (i8, w8) = topk(&l, tw);
            for j in 0..tw {
                let (i1, w1) = topk(col(&l, j, stride), 1);
                inv.eq(col(&i8, j, TOPK), &i1, &format!("router ids col {j}"));
                inv.eq(col(&w8, j, TOPK), &w1, &format!("router w col {j}"));
            }
        }
        // Grouped experts (parts rows per token) and combine.
        {
            let blobs: Vec<B> = (0..6)
                .map(|_| {
                    g.upload_bytes(&ExpertBlob::from_gguf(
                        &rng.q2(FF, HIDDEN),
                        &rng.q2(FF, HIDDEN),
                        &rng.q2(HIDDEN, FF),
                    ))
                })
                .collect();
            let shared = g.upload_bytes(&ExpertBlob::from_gguf(
                &rng.q2(FF, HIDDEN),
                &rng.q2(FF, HIDDEN),
                &rng.q2(HIDDEN, FF),
            ));
            let table: Vec<u32> = (0..EXPERTS)
                .flat_map(|e| {
                    let p = g.buffer_addr(&blobs[e % 6]);
                    [p as u32, (p >> 32) as u32]
                })
                .collect();
            let gtab = g.upload_u32(&table);
            // Overlapping experts across tokens, so the T=8 groups hold several entries.
            let mut ids = vec![];
            for tt in 0..tw {
                let mut row: Vec<u32> = vec![];
                while row.len() < TOPK {
                    let e = (rng.u() % 16) as u32 * 3 + (tt as u32 % 2);
                    if !row.contains(&e) {
                        row.push(e);
                    }
                }
                ids.extend(row);
            }
            let x = rng.vec(tw * HIDDEN, 1.0);
            let w = rng.vec(tw * TOPK, 0.3);
            let logits = rng.vec(tw * (EXPERTS + 1), 2.0);
            let run = |cols: &[usize]| -> (Vec<f32>, Vec<f32>) {
                let m = cols.len();
                let pid: Vec<u32> = cols
                    .iter()
                    .flat_map(|&j| ids[j * TOPK..(j + 1) * TOPK].to_vec())
                    .collect();
                let px: Vec<f32> = cols
                    .iter()
                    .flat_map(|&j| col(&x, j, HIDDEN).to_vec())
                    .collect();
                let pw: Vec<f32> = cols
                    .iter()
                    .flat_map(|&j| col(&w, j, TOPK).to_vec())
                    .collect();
                let pl: Vec<f32> = cols
                    .iter()
                    .flat_map(|&j| col(&logits, j, EXPERTS + 1).to_vec())
                    .collect();
                let mut plan = g.alloc_f32(MoePlan::words(m));
                g.moe_plan_into(
                    &g.upload_u32(&pid),
                    &gtab,
                    g.buffer_addr(&shared),
                    &mut plan,
                    m,
                );
                let xq = quant(&px, m, HIDDEN);
                let (mut sc, mut parts) = (
                    g.alloc_f32(MoePlan::scratch_words_for(m)),
                    g.alloc_f32(MoePlan::parts_rows(m) * HIDDEN),
                );
                // SAFETY: the plan's addresses are the live blobs above.
                unsafe { g.moe_grouped_into(&xq, &plan, &mut sc, &mut parts, m) };
                let mut y = g.alloc_f32(m * HIDDEN);
                g.moe_combine_into(
                    &parts,
                    &g.upload_f32(&pw),
                    &g.upload_f32(&pl),
                    EXPERTS + 1,
                    Some(EXPERTS),
                    &mut y,
                    m,
                );
                (g.download(&parts), g.download(&y))
            };
            let (p8, y8) = run(&(0..tw).collect::<Vec<_>>());
            for j in 0..tw {
                let (p1, y1) = run(&[j]);
                for kk in 0..TOPK {
                    inv.eq(
                        col(&p8, j * TOPK + kk, HIDDEN),
                        col(&p1, kk, HIDDEN),
                        &format!("moe parts col {j} slot {kk}"),
                    );
                }
                inv.eq(
                    col(&p8, MoePlan::shared_row(tw) + j, HIDDEN),
                    col(&p1, MoePlan::shared_row(1), HIDDEN),
                    &format!("moe shared col {j}"),
                );
                inv.eq(col(&y8, j, HIDDEN), &y1, &format!("moe combine col {j}"));
            }
        }
        // GDN: one T=8 commit window vs eight T=1 commit windows (both launch shapes).
        {
            let cs = gdn_case(&mut rng, tw);
            let (dt, a, norm, conv) = (
                g.upload_f32(&cs.dt),
                g.upload_f32(&cs.a),
                g.upload_f32(&cs.norm),
                g.upload_f32(&cs.conv),
            );
            let p = GdnParams {
                conv: &conv,
                dt_bias: &dt,
                ssm_a: &a,
                norm: &norm,
            };
            for fused in [false, true] {
                let run = |m: usize, steps: usize| -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
                    let mut state = g.upload_f32(&cs.state);
                    let mut hist = g.upload_f32(&cs.hist);
                    let (mut ys, mut yqs) = (vec![], vec![]);
                    for s in 0..steps {
                        let proj = g.upload_f32(&cs.proj[s * m * GDN_PROJ..(s + 1) * m * GDN_PROJ]);
                        let w = win(g, s * m, m);
                        let (mut y, mut yq) = (
                            g.alloc_f32(m * GDN_V),
                            g.alloc_f32(QAct { m, k: GDN_V }.words()),
                        );
                        if fused {
                            g.gdn_conv_step(
                                &mut state,
                                &proj,
                                GDN_PROJ,
                                &hist,
                                &p,
                                &mut y,
                                Some(&mut yq),
                                m,
                                GdnMode::Commit { win: &w },
                                1e-6,
                            );
                        } else {
                            let mut h = g.alloc_f32(m * GDN_CONV);
                            g.gdn_conv_into(&proj, GDN_PROJ, &hist, &conv, &mut h, m, 1e-6);
                            g.gdn_step(
                                &mut state,
                                &h,
                                &proj,
                                GDN_PROJ,
                                &p,
                                &mut y,
                                Some(&mut yq),
                                m,
                                GdnMode::Commit { win: &w },
                                1e-6,
                            );
                        }
                        g.gdn_conv_commit(&mut hist, &proj, GDN_PROJ, &w, m);
                        ys.extend(g.download(&y));
                        let q = g.download(&yq);
                        for j in 0..m {
                            yqs.extend(qact_col(&q, m, GDN_V, j));
                        }
                    }
                    (ys, yqs, g.download(&state), g.download(&hist))
                };
                let a8 = run(tw, 1);
                let a1 = run(1, tw);
                let what = format!("gdn fused={fused}");
                inv.eq(&a8.0, &a1.0, &format!("{what} y"));
                inv.eq(&a8.1, &a1.1, &format!("{what} yq"));
                inv.eq(&a8.2, &a1.2, &format!("{what} state"));
                inv.eq(&a8.3, &a1.3, &format!("{what} hist"));
            }
        }
        // QSA: prep over successive windows (T=8 vs T=1 from position 0), then select and
        // attend per column at its own position.
        {
            let max_ctx = 4096;
            let (cos, sin) = fl::rope_table(max_ctx, 1e7);
            let (cb, sb) = (g.upload_f32(&cos), g.upload_f32(&sin));
            let nb: Vec<B> = [QSA_D, QSA_D, IDX_D, IDX_D]
                .iter()
                .map(|&n| {
                    g.upload_f32(
                        &rng.vec(n, 1.0)
                            .iter()
                            .map(|v| 1.0 + 0.3 * v)
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            let qn = QsaNorms {
                q: &nb[0],
                k: &nb[1],
                iq: &nb[2],
                ik: &nb[3],
            };
            // Three single cells first, so the windows straddle pooled blocks (pos0 % 4 == 3) and
            // read pre-window ring cells while writing their own.
            let n_tok = 3 + 2 * tw;
            let proj = rng.vec(n_tok * QSA_PROJ, 1.0);
            let prep = |m: usize| -> [Vec<f32>; 5] {
                let (mut kc, mut vc) = (
                    g.alloc_bf16(max_ctx * QSA_KV * QSA_D),
                    g.alloc_bf16(max_ctx * QSA_KV * QSA_D),
                );
                let (mut ring, mut pooled) = (
                    g.alloc_f32(QSA_RING * IDX_D),
                    g.alloc_f32(max_ctx / 4 * IDX_D),
                );
                let mut qs = vec![];
                let starts: Vec<(usize, usize)> = [(0, 1), (1, 1), (2, 1)]
                    .into_iter()
                    .chain((3..n_tok).step_by(m).map(|p| (p, m)))
                    .collect();
                for (p0, m) in starts {
                    let mut q = g.alloc_f32(fl::qsa_q_words(m));
                    let cache = QsaCache {
                        k: &mut kc,
                        v: &mut vc,
                        ring: &mut ring,
                        pooled: &mut pooled,
                    };
                    g.qsa_prep(
                        &g.upload_f32(&proj[p0 * QSA_PROJ..(p0 + m) * QSA_PROJ]),
                        QSA_PROJ,
                        &win(g, p0, 0),
                        &qn,
                        (&cb, &sb),
                        &mut q,
                        cache,
                        m,
                        1e-6,
                    );
                    let q = g.download(&q);
                    let (a, b) = (QSA_HEADS * QSA_D, IDX_HEADS * IDX_D);
                    for j in 0..m {
                        qs.extend_from_slice(&q[j * a..(j + 1) * a]);
                        qs.extend_from_slice(&q[m * a + j * b..m * a + (j + 1) * b]);
                    }
                }
                let n = n_tok * QSA_KV * QSA_D;
                [
                    qs,
                    g.download(&kc)[..n].to_vec(),
                    g.download(&vc)[..n].to_vec(),
                    g.download(&ring),
                    g.download(&pooled)[..n_tok / 4 * IDX_D].to_vec(),
                ]
            };
            let (p8, p1) = (prep(tw), prep(1));
            for (i, nm) in ["q", "k cache", "v cache", "ring", "pooled"]
                .iter()
                .enumerate()
            {
                inv.eq(&p8[i], &p1[i], &format!("qsa prep {nm}"));
            }
            // Select and attend at a long position (selection active), per column.
            let max_blocks = 1024;
            let pooled = rng.vec(max_blocks * IDX_D, 1.0);
            let to_bits = |v: &[f32]| -> Vec<u16> {
                v.iter()
                    .map(|x| (fl::bf16_round(*x).to_bits() >> 16) as u16)
                    .collect()
            };
            let (gk, gv, gp) = (
                g.upload_bf16(&to_bits(&rng.vec(max_ctx * QSA_KV * QSA_D, 1.0))),
                g.upload_bf16(&to_bits(&rng.vec(max_ctx * QSA_KV * QSA_D, 1.0))),
                g.upload_f32(&pooled),
            );
            let q8 = rng.vec(fl::qsa_q_words(tw), 1.0);
            let pr8 = rng.vec(tw * QSA_PROJ, 2.0);
            let (a, b) = (QSA_HEADS * QSA_D, IDX_HEADS * IDX_D);
            let qcol = |j: usize| -> Vec<f32> {
                let mut v = q8[j * a..(j + 1) * a].to_vec();
                v.extend_from_slice(&q8[tw * a + j * b..tw * a + (j + 1) * b]);
                v
            };
            for pos0 in [100usize, 3000] {
                let sel_att = |q: &[f32],
                               pr: &[f32],
                               m: usize,
                               pos: usize|
                 -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
                    let (gq, w) = (g.upload_f32(q), win(g, pos, 0));
                    let (mut sc, mut ids) =
                        (g.alloc_f32(m * max_blocks), g.alloc_f32(m * QSA_WIDTH));
                    g.qsa_select_into(&gp, &gq, &w, &mut sc, &mut ids, max_blocks, m);
                    let mut s = g.alloc_f32(fl::qsa_attend_scratch_words(m));
                    let (mut out, mut oq) = (
                        g.alloc_f32(m * QSA_OUT),
                        g.alloc_f32(QAct { m, k: QSA_OUT }.words()),
                    );
                    g.qsa_attend_into(
                        &gq,
                        &gk,
                        &gv,
                        &ids,
                        &g.upload_f32(pr),
                        QSA_PROJ,
                        &w,
                        &mut s,
                        &mut out,
                        Some(&mut oq),
                        m,
                    );
                    (
                        g.download(&sc),
                        g.download(&ids),
                        g.download(&out),
                        g.download(&oq),
                    )
                };
                let o8 = sel_att(&q8, &pr8, tw, pos0);
                for j in 0..tw {
                    let o1 = sel_att(&qcol(j), col(&pr8, j, QSA_PROJ), 1, pos0 + j);
                    let nbk = (pos0 + j + 1) / 4;
                    let what = format!("qsa pos0={pos0} col {j}");
                    inv.eq(
                        &col(&o8.0, j, max_blocks)[..nbk],
                        &o1.0[..nbk],
                        &format!("{what} scores"),
                    );
                    inv.eq(col(&o8.1, j, QSA_WIDTH), &o1.1, &format!("{what} ids"));
                    inv.eq(col(&o8.2, j, QSA_OUT), &o1.2, &format!("{what} out"));
                    inv.eq(
                        &qact_col(&o8.3, tw, QSA_OUT, j),
                        &o1.3,
                        &format!("{what} outq"),
                    );
                }
            }
        }
        if !inv.0.is_empty() {
            // One line per (op, field): the first column that differs.
            let mut seen = std::collections::BTreeSet::new();
            let lines: Vec<&String> = inv
                .0
                .iter()
                .filter(|l| seen.insert(l.split(" col ").next().unwrap().to_string()))
                .collect();
            panic!(
                "window-width dependence:\n{}",
                lines
                    .iter()
                    .map(|l| l.as_str())
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
    }

    /// Wide (prefill) windows of the native GEMV: every column bitwise its T=1 result.
    fn native_wide(g: &CudaComputeDevice) {
        use crate::flash_native::{tests::random_gguf, NatType};
        let mut rng = Rng(0x31de);
        for ty in [NatType::Iq4Xs, NatType::Q3K, NatType::Q4K, NatType::Q2_0] {
            for &(k, n) in &[(2560usize, 300usize), (6144, 129)] {
                let w = g.upload_native(ty, &random_gguf(ty, n, k, rng.u()), n, k);
                for m in [13usize, 16, 32, 64] {
                    let x = rng.vec(m * k, 1.0);
                    let mut xq = g.alloc_f32(QAct { m, k }.words());
                    g.quantize_act_into(&g.upload_f32(&x), &mut xq, m, k);
                    let mut y = g.alloc_f32(m * n);
                    g.native_linear_into(ty, &xq, &w, &mut y, m, k, n);
                    let y = g.download(&y);
                    for j in 0..m {
                        let mut q1 = g.alloc_f32(QAct { m: 1, k }.words());
                        g.quantize_act_into(&g.upload_f32(&x[j * k..(j + 1) * k]), &mut q1, 1, k);
                        let mut y1 = g.alloc_f32(n);
                        g.native_linear_into(ty, &q1, &w, &mut y1, 1, k, n);
                        same_bits(
                            &y[j * n..(j + 1) * n],
                            &g.download(&y1),
                            &format!("native wide {ty:?} m={m} col {j}"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cuda_native_wide() {
        on_gpu(native_wide);
    }

    #[test]
    fn cuda_native_stack() {
        on_gpu(native_stack_vs_segments);
    }

    #[test]
    fn cuda_bf16_linear_out() {
        on_gpu(bf16_linear_out);
    }

    #[test]
    fn cuda_hc_q8_vs_cpu() {
        on_gpu(hc_q8_vs_cpu);
    }

    #[test]
    fn cuda_window_width_invariance() {
        on_gpu(|g| t_invariance_w(g, TW));
    }

    /// Prefill widths: `TANG_TEST_WIDE=32,64` (default 16,32,64).
    #[test]
    fn cuda_window_width_invariance_wide() {
        on_gpu(|g| {
            let ws: Vec<usize> = std::env::var("TANG_TEST_WIDE")
                .ok()
                .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
                .unwrap_or(vec![16, 32, 64]);
            ws.iter().for_each(|&w| t_invariance_w(g, w))
        });
    }

    #[test]
    fn cuda_qsa_union_vs_cpu() {
        on_gpu(qsa_union_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_q2_linear_vs_cpu() {
        on_gpu(q2_linear_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_q4x_linear_vs_cpu() {
        on_gpu(q4x_linear_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_q8x_linear_vs_cpu() {
        on_gpu(q8x_linear_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_native_linear_vs_cpu() {
        on_gpu(native_linear_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_linear_into_vs_cpu() {
        on_gpu(linear_into_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_hc_vs_cpu() {
        on_gpu(hc_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_gdn_vs_cpu() {
        on_gpu(gdn_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_router_vs_cpu() {
        on_gpu(router_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_moe_vs_cpu() {
        on_gpu(moe_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_qsa_prep_vs_cpu() {
        on_gpu(qsa_prep_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_qsa_select_vs_cpu() {
        on_gpu(qsa_select_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_qsa_attend_vs_cpu() {
        on_gpu(qsa_attend_vs_cpu::<CudaComputeDevice>);
    }
}
