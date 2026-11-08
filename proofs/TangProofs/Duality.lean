/-
The IO bound and the tile as LP duals (`tensor_io_lp.rs`).

Coordinates `ι`, arrays `κ`, array `j` reads coordinates `φ j`.
* `weak_duality`: any HBL exponents `s` covering every coordinate and any
  tile exponents `x` whose every array footprint fits satisfy
  `Σ x ≤ Σ s` — the tile can never beat the bound's exponent.
* `box_hbl`: for a box tile with side lengths `T q ≥ 1`,
  `|E| = ∏ T ≤ ∏_j |φ_j(E)|^{s_j}`; with every footprint `≤ M`,
  `|E| ≤ M^{Σ s}`. Hence a schedule that executes the iteration box in
  box tiles whose footprints fit in fast memory runs at least
  `|V| / M^{Σ s}` tiles, each loading its footprint: no *box-tiled*
  schedule moves fewer words than the primal bound (`tiles_needed`).
  General (non-box) schedules need the full discrete HBL inequality, which
  is not mechanized here.
-/
import Mathlib

namespace Tang.Duality

open Finset BigOperators Real

variable {ι κ : Type*} [Fintype ι] [Fintype κ] [DecidableEq ι]

/-- Exchange the coordinate / array double sum. -/
theorem exchange (φ : κ → Finset ι) (s : κ → ℝ) (f : ι → ℝ) :
    ∑ q, f q * ∑ j ∈ univ.filter (fun j => q ∈ φ j), s j = ∑ j, s j * ∑ q ∈ φ j, f q := by
  simp only [Finset.mul_sum, Finset.sum_filter]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp only [mul_ite, mul_zero]
  rw [Finset.sum_ite_mem, Finset.univ_inter]
  exact Finset.sum_congr rfl fun _ _ => mul_comm _ _

/-- LP weak duality for the HBL / tiling pair. -/
theorem weak_duality (φ : κ → Finset ι) (s : κ → ℝ) (x : ι → ℝ)
    (hs : ∀ j, 0 ≤ s j) (hx : ∀ q, 0 ≤ x q)
    (cover : ∀ q, 1 ≤ ∑ j ∈ univ.filter (fun j => q ∈ φ j), s j)
    (fits : ∀ j, ∑ q ∈ φ j, x q ≤ 1) :
    ∑ q, x q ≤ ∑ j, s j := by
  calc ∑ q, x q ≤ ∑ q, x q * ∑ j ∈ univ.filter (fun j => q ∈ φ j), s j :=
        Finset.sum_le_sum fun q _ => le_mul_of_one_le_right (hx q) (cover q)
    _ = ∑ j, s j * ∑ q ∈ φ j, x q := exchange φ s x
    _ ≤ ∑ j, s j * 1 := Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (fits j) (hs j)
    _ = ∑ j, s j := by simp

/-- Hölder–Brascamp–Lieb for boxes: `∏ T ≤ ∏_j (∏_{q∈φ j} T q)^{s j}`. -/
theorem box_hbl (φ : κ → Finset ι) (s : κ → ℝ) (T : ι → ℝ)
    (hs : ∀ j, 0 ≤ s j) (hT : ∀ q, 1 ≤ T q)
    (cover : ∀ q, 1 ≤ ∑ j ∈ univ.filter (fun j => q ∈ φ j), s j) :
    ∏ q, T q ≤ ∏ j, (∏ q ∈ φ j, T q) ^ s j := by
  have hpos : ∀ q, 0 < T q := fun q => lt_of_lt_of_le one_pos (hT q)
  have hfp : ∀ j, 0 < ∏ q ∈ φ j, T q := fun j => Finset.prod_pos fun q _ => hpos q
  rw [← Real.log_le_log_iff (Finset.prod_pos fun q _ => hpos q)
    (Finset.prod_pos fun j _ => Real.rpow_pos_of_pos (hfp j) _)]
  rw [Real.log_prod (fun q _ => (hpos q).ne'),
    Real.log_prod (fun j _ => (Real.rpow_pos_of_pos (hfp j) _).ne')]
  simp only [Real.log_rpow (hfp _), Real.log_prod (fun q _ => (hpos q).ne')]
  -- Σ_q log T_q ≤ Σ_j s_j Σ_{q∈φ j} log T_q, since log T ≥ 0 and coverage ≥ 1
  have key := weak_duality φ s (fun q => Real.log (T q)) hs
    (fun q => Real.log_nonneg (hT q)) cover
  calc ∑ q, Real.log (T q)
      ≤ ∑ q, Real.log (T q) * ∑ j ∈ univ.filter (fun j => q ∈ φ j), s j :=
        Finset.sum_le_sum fun q _ => le_mul_of_one_le_right (Real.log_nonneg (hT q)) (cover q)
    _ = ∑ j, s j * ∑ q ∈ φ j, Real.log (T q) := exchange φ s _

/-- Every footprint `≤ M` bounds the tile's volume by `M^{Σ s}`. -/
theorem box_volume (φ : κ → Finset ι) (s : κ → ℝ) (T : ι → ℝ) (M : ℝ)
    (hs : ∀ j, 0 ≤ s j) (hT : ∀ q, 1 ≤ T q)
    (cover : ∀ q, 1 ≤ ∑ j ∈ univ.filter (fun j => q ∈ φ j), s j)
    (fits : ∀ j, ∏ q ∈ φ j, T q ≤ M) :
    ∏ q, T q ≤ M ^ (∑ j, s j) := by
  have hpos : ∀ q, 0 < T q := fun q => lt_of_lt_of_le one_pos (hT q)
  have hfp : ∀ j, 0 < ∏ q ∈ φ j, T q := fun j => Finset.prod_pos fun q _ => hpos q
  have hM : ∀ j, 0 < M := fun j => lt_of_lt_of_le (hfp j) (fits j)
  calc ∏ q, T q ≤ ∏ j, (∏ q ∈ φ j, T q) ^ s j := box_hbl φ s T hs hT cover
    _ ≤ ∏ j, M ^ s j := by
        apply Finset.prod_le_prod₀
        · intro j _; exact Real.rpow_nonneg (hfp j).le _
        · intro j _; exact Real.rpow_le_rpow (hfp j).le (fits j) (hs j)
    _ = M ^ (∑ j, s j) := by
        by_cases hk : Nonempty κ
        · obtain ⟨j0⟩ := hk
          rw [Real.rpow_sum_of_pos (hM j0)]
        · simp [not_nonempty_iff.mp hk]

/-- **No box-tiled schedule beats the primal.** If `n` box tiles, each of
volume `≤ M^σ` (by `box_volume`), cover an iteration space of `V` points,
then `n ≥ V / M^σ`; each tile loads at least its own footprint, so the
traffic is at least `(V / M^σ) · (one footprint)`. -/
theorem tiles_needed (V Mσ : ℝ) (vol : ℕ → ℝ) (n : ℕ) (hMσ : 0 < Mσ)
    (hvol : ∀ i < n, vol i ≤ Mσ) (hcover : V ≤ ∑ i ∈ range n, vol i) :
    V / Mσ ≤ n := by
  rw [div_le_iff₀ hMσ]
  calc V ≤ ∑ i ∈ range n, vol i := hcover
    _ ≤ ∑ i ∈ range n, Mσ := Finset.sum_le_sum fun i hi => hvol i (Finset.mem_range.mp hi)
    _ = n * Mσ := by simp

end Tang.Duality
