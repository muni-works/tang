//! The dense lead-in FFN against Hugging Face fp32 on a recorded fixture
//! (`tests/fixtures/glm5/dense_ffn/`, generated from `Glm5NextTextMLP.forward`).

use std::fs;
use std::path::Path;

fn read_f32(p: &Path) -> Vec<f32> {
    let b = fs::read(p).unwrap();
    b.chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn dense_ffn_matches_hf() {
    let dir = Path::new("tests/fixtures/glm5/dense_ffn");
    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("fixture.json")).unwrap()).unwrap();
    let n = meta["hidden"].as_u64().unwrap() as usize;
    let ff = meta["ff"].as_u64().unwrap() as usize;
    assert_eq!(meta["swiglu_limit"].as_f64().unwrap(), 10.0);

    // tang's Mat layout: data[row * cols..], row = output, col = input.
    let gate = read_f32(&dir.join("gate_proj.f32")); // [ff, n] row-major HF = Mat rows=ff cols=n
    let up = read_f32(&dir.join("up_proj.f32"));
    let down = read_f32(&dir.join("down_proj.f32")); // [n, ff]
    let x = read_f32(&dir.join("x.f32")); // [t, n]
    let want = read_f32(&dir.join("y.f32"));
    let t_len = x.len() / n;
    assert_eq!(want.len(), t_len * n);

    let mat = |rows: usize, cols: usize, data: Vec<f32>| tang_llm::flash::reference::Mat {
        rows,
        cols,
        data,
        act: tang_llm::flash::reference::Act::F32,
    };
    let g = mat(ff, n, gate).apply(&x, t_len, true);
    let u = mat(ff, n, up).apply(&x, t_len, true);
    let silu = |x: f32| x / (1.0 + (-x).exp());
    let h: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(&gv, &uv)| silu(gv.min(10.0)) * uv.clamp(-10.0, 10.0))
        .collect();
    let y = mat(n, ff, down).apply(&h, t_len, true);

    // fp32 vs fp32, different summation orders: 1e-4 abs, per COMMON.md
    let mut max = 0f32;
    for (a, b) in y.iter().zip(&want) {
        max = max.max((a - b).abs());
    }
    assert!(
        max <= 1e-4 * (1.0 + want.iter().map(|v| v.abs()).fold(0f32, f32::max)),
        "max abs err {max}"
    );
}
