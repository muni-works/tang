/-
Gradients of comprehensions (`tensor_grad.rs`).

* The vector-Jacobian product of a sum comprehension with respect to one
  input coordinate is again a sum comprehension over the *same* iteration
  space, gated by `[view(p) = a]`; summing out the gate leaves a sum over
  the fiber of the view, which for a pick view is a box (an ordinary
  comprehension) and for a gather view is a segmented sum.
* The stabilizing max contributes nothing: softmax / log-sum-exp with any
  shift `c(x)` are the same functions as without it, so their derivatives
  coincide even where `c = max` is not differentiable.
* The online-softmax output's gradient is `pⱼ (vⱼ - y)`: FlashAttention-2's
  `dS = P ∘ (dP - D)` with `D = Σ dO ∘ O`.
-/
import Mathlib
import TangProofs.OnlineSoftmax

namespace Tang

open Real Finset BigOperators

section VJP

variable {ι κ σ : Type*} [Fintype ι] [Fintype κ] [DecidableEq σ]

/-- `x` with coordinate `a` moved by `t`. -/
def bump (x : σ → ℝ) (a : σ) (t : ℝ) : σ → ℝ := Function.update x a (x a + t)

theorem term_deriv (f f' : ℝ → ℝ) (hd : ∀ z, HasDerivAt f (f' z) z)
    (x : σ → ℝ) (b a : σ) :
    HasDerivAt (fun t => f (bump x a t b)) (if b = a then f' (x a) else 0) 0 := by
  by_cases hv : b = a
  · subst hv
    simp only [bump, Function.update_self, ite_true]
    have := (hd (x b + 0)).comp_const_add (x b) 0
    simpa using this
  · simp only [bump, Function.update_of_ne hv, hv, ite_false]
    exact hasDerivAt_const _ _

/-- **Gradients of comprehensions are comprehensions.** Let
`y i = Σ_k h i k (x (v i k))` (the body `h i k` closes over the other
arguments, so multi-argument bodies are included) and `L = Σ_i g i · y i`.
Then `∂L/∂x_a = Σ_{i,k} [v i k = a] · g i · h' i k (x a)`: a sum
comprehension over the same space with a gated body. -/
theorem vjp_sum_comp (h h' : ι → κ → ℝ → ℝ)
    (hd : ∀ i k z, HasDerivAt (h i k) (h' i k z) z)
    (x : σ → ℝ) (v : ι → κ → σ) (g : ι → ℝ) (a : σ) :
    HasDerivAt (fun t => ∑ i, g i * ∑ k, h i k (bump x a t (v i k)))
      (∑ i, ∑ k, if v i k = a then g i * h' i k (x a) else 0) 0 := by
  have hterm : ∀ i, HasDerivAt (fun t => g i * ∑ k, h i k (bump x a t (v i k)))
      (∑ k, if v i k = a then g i * h' i k (x a) else 0) 0 := by
    intro i
    have hs : HasDerivAt (fun t => ∑ k, h i k (bump x a t (v i k)))
        (∑ k, if v i k = a then h' i k (x a) else 0) 0 :=
      HasDerivAt.fun_sum (fun k _ => term_deriv (h i k) (h' i k) (hd i k) x (v i k) a)
    have := hs.const_mul (g i)
    simpa [Finset.mul_sum, mul_ite] using this
  exact HasDerivAt.fun_sum (fun i _ => hterm i)

/-- The gate sums out: the gradient is a sum over the fiber `v⁻¹(a)` of
the view — a box for a pick view (an ordinary comprehension over the
eliminated coordinates), a run of a sorted key for a gather (a segmented
sum, `TNode::segment`). -/
theorem vjp_fiber (h' : ι → κ → ℝ → ℝ) (x : σ → ℝ) (v : ι → κ → σ) (g : ι → ℝ)
    (a : σ) :
    (∑ i, ∑ k, if v i k = a then g i * h' i k (x a) else 0)
      = ∑ p ∈ (Finset.univ : Finset (ι × κ)).filter (fun p => v p.1 p.2 = a),
          g p.1 * h' p.1 p.2 (x a) := by
  rw [Finset.sum_filter, ← Finset.univ_product_univ, Finset.sum_product]

end VJP

section Stabilize

variable {n : ℕ}

/-- **The stabilizing max has zero gradient.** For *any* shift function
`c` — in particular `c = max`, which is not differentiable at ties —
shifted softmax is literally softmax, so it has exactly softmax's
derivative; no derivative of `c` ever enters. -/
theorem stabilized_softmax_eq (c : (Fin n → ℝ) → ℝ) (i : Fin n) :
    (fun x => softmaxShift x (c x) i) = fun x => softmax x i := by
  funext x
  rw [softmaxShift_eq]

theorem stabilized_softmax_hasFDerivAt (c : (Fin n → ℝ) → ℝ) (i : Fin n)
    {x : Fin n → ℝ} {F' : (Fin n → ℝ) →L[ℝ] ℝ}
    (h : HasFDerivAt (fun x => softmax x i) F' x) :
    HasFDerivAt (fun x => softmaxShift x (c x) i) F' x := by
  rw [stabilized_softmax_eq]; exact h

/-- Partial derivative with respect to the shift is zero everywhere. -/
theorem softmaxShift_deriv_shift (x : Fin n → ℝ) (i : Fin n) (c : ℝ) :
    HasDerivAt (fun c => softmaxShift x c i) 0 c := by
  have : (fun c => softmaxShift x c i) = fun _ => softmax x i := by
    funext c; rw [softmaxShift_eq]
  rw [this]; exact hasDerivAt_const _ _

/-- Same for log-sum-exp (the saved statistic of the flash backward). -/
theorem lse_deriv_shift [NeZero n] (x : Fin n → ℝ) (c : ℝ) :
    HasDerivAt (fun c => c + log (∑ j, exp (x j - c))) 0 c := by
  have : (fun c => c + log (∑ j, exp (x j - c))) = fun _ => log (∑ j, exp (x j)) := by
    funext c; rw [lse_shift]
  rw [this]; exact hasDerivAt_const _ _

end Stabilize

section SoftmaxBackward

variable {n : ℕ} [NeZero n]

/-- **Online-softmax gradient (FlashAttention-2 backward).** With
`y = Σᵢ pᵢ vᵢ`, `p = softmax x`: `∂y/∂xⱼ = pⱼ (vⱼ − y)`. Contracting with
an upstream `dO` gives `dSⱼ = Pⱼ (dPⱼ − D)` with `D = Σ dO·O`, the identity
tang derives as the gradient of the online-softmax monoid. -/
theorem softmax_avg_deriv (x v : Fin n → ℝ) (j : Fin n) :
    HasDerivAt
      (fun t => (∑ i, exp (bump x j t i) * v i) / ∑ i, exp (bump x j t i))
      (softmax x j * (v j - (∑ i, exp (x i) * v i) / ∑ i, exp (x i))) 0 := by
  have hb : bump x j 0 = x := by simp [bump]
  have hN : HasDerivAt (fun t => ∑ i, exp (bump x j t i) * v i)
      (∑ i, (if i = j then exp (x j) else 0) * v i) 0 :=
    HasDerivAt.fun_sum (fun i _ =>
      (term_deriv exp exp (fun z => hasDerivAt_exp z) x i j).mul_const (v i))
  have hD : HasDerivAt (fun t => ∑ i, exp (bump x j t i))
      (∑ i, (if i = j then exp (x j) else 0)) 0 :=
    HasDerivAt.fun_sum (fun i _ => term_deriv exp exp (fun z => hasDerivAt_exp z) x i j)
  have hpos : 0 < ∑ i, exp (x i) :=
    Finset.sum_pos (fun i _ => exp_pos _) Finset.univ_nonempty
  have hD0 : (∑ i, exp (bump x j 0 i)) ≠ 0 := by rw [hb]; exact hpos.ne'
  have := hN.div hD hD0
  simp only [hb, ite_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ, ite_true] at this
  convert this using 1
  simp only [softmax]
  field_simp

end SoftmaxBackward

end Tang
