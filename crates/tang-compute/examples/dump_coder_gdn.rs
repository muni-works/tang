//! Print the flash CUDA module for Qwen3-Coder-Next's GDN heads (16 k heads, 32 v heads,
//! d 128, grouped), as `coder.rs` compiles it (f32 state, one slot). Used by
//! `scripts/gdn_hand_bench.cu` to time the hand prefill recurrence (`fl_gdn_wide`).
fn main() {
    let geo = tang_compute::flash::Geometry {
        gdn_hk: 16,
        gdn_hv: 32,
        gdn_d: 128,
        gdn_grouped: true,
        ..tang_compute::flash::Geometry::FLASH_NEXT
    };
    print!("#define GDN_SLOTS 1\n{}", geo.cuda_source());
}
