# tang proofs (Lean 4 + Mathlib)

Mechanized soundness of tang's core rewrites. Statements and proof
sketches in paper form: [`docs/proofs.md`](../docs/proofs.md).

| file | theorems |
|---|---|
| `TangProofs/Fusion.lean` | `Form.eval_compose`, `View.read_compose`, `substitute_map_sound(₂)`, `split_reduction` |
| `TangProofs/OnlineSoftmax.lean` | `OSM.combine_assoc`, `OSM.combine_comm`, `OSM.sem_combine`, `OSM.run_eq_softmax`, `OSM.run_append`, `softmaxShift_eq`, `lse_shift` |
| `TangProofs/Grad.lean` | `vjp_sum_comp`, `vjp_fiber`, `stabilized_softmax_eq`, `stabilized_softmax_hasFDerivAt`, `softmaxShift_deriv_shift`, `lse_deriv_shift`, `softmax_avg_deriv` |

Build (first time downloads the Mathlib cache, ~5 GB):

    lake exe cache get
    lake build

No `sorry` anywhere. Everything is over ℝ; see "What is not proven" in
the doc for floating point and the other gaps.
