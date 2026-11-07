//! The hand-written Coder-Next MoE kernels on one layer's bytes, as the
//! engine launches them, against the tensor-IR derivation of the same layer
//! (`tang-bench --example coder_moe_ir`, `CODER_MOE_DUMP=dir`).
//!
//! Reads `dir/b{B}/` (inputs: x, logits, ids, wts, the Q4_K gate|up and
//! Q5_K down expert bytes; the derived outputs), runs the engine's kernels
//! for that window width, checks them against the derived outputs, and
//! times each kernel (min of warm launches; median with the L2 flushed
//! before each launch, as `coder_moe_ir` does):
//!
//! - B <= 8 (narrow windows): `cx_moe_gu_d2` / `_d4` (per pair, direct
//!   rows, dp4a), `cx_moe_down_d2f` / `_d4f` (SiLU + int8 fused), then
//!   `cx_moe_add_norm` (the combine, fused with the residual and next norm).
//! - B > 8 (batch windows): `cx_moe_plan_w`, `cx_moe_gu_q` (GGUF-native int8
//!   MMA, the §8 kernel), `cx_silu_q`, `cx_moe_down_q`, `cx_moe_add_norm`.
//!
//! `cargo run --release -p tang-llm --features cuda --example coder_moe_hand -- dir [B...]`

use std::ffi::c_void;
use std::path::Path;

use cudarc::driver::sys;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use tang_llm::flash::coder_kernels::{self as ck, Shapes};
use tang_llm::flash::kernels;
use tang_moe::gpu::Gpu;

#[repr(C)]
#[derive(Clone, Copy)]
struct ExW {
    base: u64,
    estride: u64,
    ty: i32,
    row_bytes: i32,
}

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

struct Dev {
    stream: std::sync::Arc<CudaStream>,
    gpu: Gpu,
}

impl Dev {
    fn up<T: cudarc::driver::DeviceRepr>(&self, v: &[T]) -> CudaSlice<T> {
        self.stream.clone_htod(v).unwrap()
    }
    fn zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        n: usize,
    ) -> CudaSlice<T> {
        self.stream.alloc_zeros::<T>(n.max(1)).unwrap()
    }
    fn ptr<T>(&self, s: &CudaSlice<T>) -> u64 {
        let (p, _sync) = s.device_ptr(&self.stream);
        p
    }
    fn down(&self, s: &CudaSlice<f32>) -> Vec<f32> {
        self.stream.clone_dtoh(s).unwrap()
    }
}

/// One launch: function, grid, block, and its arguments as raw bytes.
struct Launch {
    name: &'static str,
    f: sys::CUfunction,
    grid: (u32, u32, u32),
    block: u32,
    args: Vec<Vec<u8>>,
}

fn a<T: Copy>(v: T) -> Vec<u8> {
    let p = &v as *const T as *const u8;
    unsafe { std::slice::from_raw_parts(p, std::mem::size_of::<T>()).to_vec() }
}

impl Launch {
    fn go(&self, s: &CudaStream) {
        let mut ptrs: Vec<*mut c_void> = self
            .args
            .iter()
            .map(|b| b.as_ptr() as *mut c_void)
            .collect();
        let r = unsafe {
            sys::cuLaunchKernel(
                self.f,
                self.grid.0,
                self.grid.1,
                self.grid.2,
                self.block,
                1,
                1,
                0,
                s.cu_stream(),
                ptrs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "{}", self.name);
    }
}

fn max_rel(got: &[f32], want: &[f32]) -> f64 {
    let scale = want
        .iter()
        .fold(0f64, |m, &v| m.max(v.abs() as f64))
        .max(1e-30);
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (g as f64 - w as f64).abs() / scale)
        .fold(0.0, f64::max)
}

fn run(dev: &Dev, dir: &Path) {
    let meta = std::fs::read_to_string(dir.join("meta.txt")).unwrap();
    let v: Vec<usize> = meta
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect();
    let (t, d, e, k, f) = (v[0], v[1], v[2], v[3], v[4]);
    let batch = t > 8;
    let maxt = if batch {
        t.max(16)
    } else {
        tang_llm::flash::coder::MAX_T
    };
    let sh = Shapes {
        hidden: d,
        n_head: 16,
        n_kv: 2,
        head_dim: 256,
        n_rot: 64,
        ff: f,
        topk: k,
        n_expert: e,
    };
    let src = format!(
        "{}{}{}{}",
        if batch { "#define CX_GCHUNK 8\n" } else { "" },
        ck::prelude(&sh, maxt),
        kernels::GEMV_SRC,
        ck::SRC
    );
    let m = dev.gpu.module(&src).expect("compile the coder kernels");
    let func = |n: &str| m.func(n).unwrap_or_else(|e| panic!("{n}: {e}"));
    let np_max = maxt * (k + 1);
    let n = t * (k + 1);

    let x = read_f32(&dir.join("x.f32"));
    let logits = read_f32(&dir.join("logits.f32")); // [t][e + 1]
    let ids: Vec<u32> = std::fs::read(dir.join("ids.u32"))
        .unwrap()
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let wts = read_f32(&dir.join("wts.f32"));
    let wgu = std::fs::read(dir.join("wgu.q4k")).unwrap();
    let wd = std::fs::read(dir.join("wd.q5k")).unwrap();
    // x as the norm kernel leaves it: int8 codes [maxt][d], then f32 scales [maxt][d / 32]
    let mut xq = vec![0u8; maxt * d + maxt * d / 32 * 4];
    for ti in 0..t {
        for b in 0..d / 32 {
            let blk = &x[ti * d + b * 32..][..32];
            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let s = amax / 127.0;
            for (j, &v) in blk.iter().enumerate() {
                let q = if s == 0.0 {
                    0
                } else {
                    (v / s).round().clamp(-127.0, 127.0) as i32
                };
                xq[ti * d + b * 32 + j] = q as i8 as u8;
            }
            xq[maxt * d + (ti * (d / 32) + b) * 4..][..4].copy_from_slice(&s.to_le_bytes());
        }
    }
    let (rb_gu, rb_d) = (d / 256 * 144, f / 256 * 176);
    let (g_wgu, g_wd) = (dev.up(&wgu), dev.up(&wd));
    let (pgu, pd) = (dev.ptr(&g_wgu), dev.ptr(&g_wd));
    let es_gu = (2 * f * rb_gu) as u64;
    let es_d = (d * rb_d) as u64;
    let gw = ExW {
        base: pgu,
        estride: es_gu,
        ty: 12,
        row_bytes: rb_gu as i32,
    };
    let uw = ExW {
        base: pgu + (f * rb_gu) as u64,
        ..gw
    };
    let sg = ExW {
        base: pgu + e as u64 * es_gu,
        ..gw
    };
    let su = ExW {
        base: sg.base + (f * rb_gu) as u64,
        ..gw
    };
    let dw = ExW {
        base: pd,
        estride: es_d,
        ty: 13,
        row_bytes: rb_d as i32,
    };
    let sdw = ExW {
        base: pd + e as u64 * es_d,
        ..dw
    };

    let g_ids = dev.up(&ids);
    let g_xq = dev.up(&xq);
    let g_egu = dev.zeros::<f32>(np_max * 2 * f);
    let g_hq = dev.zeros::<u8>(np_max * f + np_max * f / 32 * 4);
    let g_parts = dev.zeros::<f32>(np_max * d);
    let g_plan = dev.zeros::<u32>(2 * (4 + 2 * np_max));
    let g_wts = dev.up(&wts);
    let g_lg = dev.up(&logits);
    let g_r = dev.zeros::<f32>(t * d);
    let g_y = dev.zeros::<f32>(t * d);
    let g_wn = dev.up(&vec![1f32; d]);
    let g_xo = dev.zeros::<f32>(t * d);
    let g_xq2 = dev.zeros::<u8>(maxt * d + maxt * d / 32 * 4);
    let p = |s: &CudaSlice<f32>| dev.ptr(s);
    let (ids_p, xq_p, egu_p, hq_p, parts_p, plan_p) = (
        dev.ptr(&g_ids),
        dev.ptr(&g_xq),
        p(&g_egu),
        dev.ptr(&g_hq),
        p(&g_parts),
        dev.ptr(&g_plan),
    );
    let (ti, fastk, pb, r16) = (t as i32, 2i32, 0i32, 0i32);
    let ls = (e + 1) as i32;
    let mut ls_: Vec<Launch> = Vec::new();
    if batch {
        let (sgu, sdn) = ck::tcx_split();
        let groups = n.min(e + 1 + n / 8) as u32;
        ls_.push(Launch {
            name: "cx_moe_plan_w",
            f: func("cx_moe_plan_w"),
            grid: (1, 1, 1),
            block: 1024,
            args: vec![a(ids_p), a(ti), a(pb), a(n as i32), a(plan_p)],
        });
        ls_.push(Launch {
            name: "cx_moe_gu_q",
            f: func("cx_moe_gu_q"),
            grid: ((2 * f * sgu / 128) as u32, groups, 1),
            block: 256,
            args: vec![
                a(gw),
                a(uw),
                a(sg),
                a(su),
                a(ids_p),
                a(xq_p),
                a(egu_p),
                a(ti),
                a(fastk),
                a(plan_p),
            ],
        });
        ls_.push(Launch {
            name: "cx_silu_q",
            f: func("cx_silu_q"),
            grid: ((f / 32) as u32, n as u32, 1),
            block: 32,
            args: vec![a(egu_p), a(hq_p), a(pb)],
        });
        ls_.push(Launch {
            name: "cx_moe_down_q",
            f: func("cx_moe_down_q"),
            grid: ((d * sdn / 128) as u32, groups, 1),
            block: 256,
            args: vec![
                a(dw),
                a(sdw),
                a(ids_p),
                a(hq_p),
                a(parts_p),
                a(ti),
                a(fastk),
                a(plan_p),
            ],
        });
    } else {
        let (gu_n, dn_n, rows) = if t <= 4 {
            ("cx_moe_gu_d2", "cx_moe_down_d2f", 16)
        } else {
            ("cx_moe_gu_d4", "cx_moe_down_d4f", 32)
        };
        ls_.push(Launch {
            name: gu_n,
            f: func(gu_n),
            grid: ((2 * f / rows) as u32, n as u32, 1),
            block: 256,
            args: vec![
                a(gw),
                a(uw),
                a(sg),
                a(su),
                a(ids_p),
                a(xq_p),
                a(egu_p),
                a(ti),
                a(r16),
                a(fastk),
                a(pb),
            ],
        });
        ls_.push(Launch {
            name: dn_n,
            f: func(dn_n),
            grid: ((d / rows) as u32, n as u32, 1),
            block: 256,
            args: vec![
                a(dw),
                a(sdw),
                a(ids_p),
                a(egu_p),
                a(parts_p),
                a(ti),
                a(r16),
                a(fastk),
                a(pb),
            ],
        });
    }
    ls_.push(Launch {
        name: "cx_moe_add_norm",
        f: func("cx_moe_add_norm"),
        grid: (t as u32, 1, 1),
        block: 1024,
        args: vec![
            a(p(&g_r)),
            a(parts_p),
            a(dev.ptr(&g_wts)),
            a(p(&g_lg)),
            a(ls),
            a(ti),
            a(p(&g_y)),
            a(p(&g_wn)),
            a(p(&g_xo)),
            a(dev.ptr(&g_xq2)),
            a(1e-6f32),
        ],
    });
    let s = &dev.stream;
    for l in &ls_ {
        l.go(s);
    }
    s.synchronize().unwrap();

    // check against the derived outputs (pair order: routed t·K + k, then shared T·K + t)
    let ir_gu = read_f32(&dir.join("ir_gu.f32"));
    let ir_dn = read_f32(&dir.join("ir_dn.f32"));
    let ir_y = read_f32(&dir.join("ir_y.f32"));
    let ref_y = read_f32(&dir.join("ref_y.f32"));
    let gu = dev.down(&g_egu);
    let parts = dev.down(&g_parts);
    let y = dev.down(&g_y);
    let gu_rel = max_rel(&gu[..n * 2 * f], &ir_gu);
    let dn_rel = max_rel(&parts[..n * d], &ir_dn);
    println!(
        "B={t}: hand vs derived max rel: gate|up {gu_rel:.2e}, down {dn_rel:.2e}, combine {:.2e}; hand vs f64 reference: combine {:.2e}",
        max_rel(&y, &ir_y),
        max_rel(&y, &ref_y)
    );

    // timings
    let mut flush = dev.zeros::<u8>(256 << 20);
    let iters = 100;
    let ev = |s: &CudaStream| {
        s.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap()
    };
    let (mut tw, mut tc) = (0.0, 0.0);
    for l in &ls_ {
        for _ in 0..3 {
            l.go(s);
        }
        s.synchronize().unwrap();
        let warm = (0..5 * iters)
            .map(|_| {
                let e0 = ev(s);
                l.go(s);
                let e1 = ev(s);
                e1.synchronize().unwrap();
                e0.elapsed_ms(&e1).unwrap() as f64
            })
            .fold(f64::INFINITY, f64::min);
        let mut cold: Vec<f64> = (0..iters)
            .map(|_| {
                s.memset_zeros(&mut flush).unwrap();
                let e0 = ev(s);
                l.go(s);
                let e1 = ev(s);
                e1.synchronize().unwrap();
                e0.elapsed_ms(&e1).unwrap() as f64
            })
            .collect();
        cold.sort_by(f64::total_cmp);
        let c = cold[cold.len() / 2];
        tw += warm;
        tc += c;
        println!("  {:>8.4} ms warm  {:>8.4} ms cold  {}", warm, c, l.name);
    }
    println!("  hand expert half: {tw:.4} warm / {tc:.4} cold ms");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = Path::new(args.first().expect("usage: coder_moe_hand dir [B...]"));
    let bs: Vec<String> = if args.len() > 1 {
        args[1..].to_vec()
    } else {
        vec!["1".into(), "8".into(), "64".into()]
    };
    let gpu = Gpu::new().expect("cuda");
    let stream = gpu.ctx.default_stream();
    let dev = Dev { stream, gpu };
    for b in bs {
        run(&dev, &dir.join(format!("b{b}")));
    }
}
