/-
Real semantics of the tang-expr ops that appear in exported traces.
`Exp2 x = 2^x`; `Select c a b = a if c > 0 else b` (`tang-expr/src/node.rs`).
`log2e` is the *exact* `log₂ e = 1 / ln 2`; tang's literal is its f64
rounding (`std::f64::consts::LOG2_E`), so the trace theorems hold for the
real-number reading of the graph, not bit-for-bit for f64.
-/
import Mathlib

namespace Tang.TraceOps

open Real

noncomputable def exp2 (x : ℝ) : ℝ := (2 : ℝ) ^ x

noncomputable def sel (c a b : ℝ) : ℝ := if 0 < c then a else b

noncomputable def log2e : ℝ := 1 / Real.log 2

theorem exp2_log2e (a b : ℝ) : exp2 ((a + -b) * log2e) = Real.exp (a - b) := by
  have h2 : Real.log 2 ≠ 0 := by positivity
  unfold exp2 log2e
  rw [Real.rpow_def_of_pos (by norm_num : (0 : ℝ) < 2)]
  congr 1
  field_simp
  ring

theorem sel_max (a b : ℝ) : sel (a + -b) a b = max a b := by
  unfold sel
  split_ifs with h
  · rw [max_eq_left]; linarith
  · rw [max_eq_right]; linarith

end Tang.TraceOps
