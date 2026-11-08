/-
The exported `stabilize_softmax` combine (`Trace.lean`, generated from
tang-compile) *is* the online-softmax combine proved associative in
`OnlineSoftmax.lean`. So the rewrite tang actually performed inherits
Theorems 4–6 (associativity, tiling, flash = two-pass softmax).
-/
import Mathlib
import TangProofs.OnlineSoftmax
import TangProofs.Trace

namespace Tang.Trace

open Tang.TraceOps

theorem combine_eq_osm (v0 v1 v2 v3 v4 v5 : ℝ) :
    combine v0 v1 v2 v3 v4 v5 =
      let r := OSM.combine ⟨v0, v1, v2⟩ ⟨v3, v4, v5⟩
      (r.m, r.l, r.o) := by
  simp only [combine, OSM.combine, sel_max]
  rw [show v0 + -max v0 v3 = v0 + -max v0 v3 from rfl, exp2_log2e, exp2_log2e]

end Tang.Trace
