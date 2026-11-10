/-
Fusion soundness for `substitute_maps`.

tang addresses every input of a comprehension through a *view*: one affine
`Form` (`offset + Σ coef·idx`) per source dimension, over the consumer's
iteration space (`crates/tang-compile/src/tensor.rs`, `Form`, `View`).
`substitute_maps` splices a map's body into every reader and composes the
reader's view with the map's own views (`Form::compose`). This file proves
that the composed view reads exactly what reading through both views does,
and hence that splicing a map into any monoid comprehension is sound.
-/
import Mathlib

namespace Tang

open Finset BigOperators

/-- `offset + Σ coefs d · idx d` over an `r`-dimensional iteration space. -/
structure Form (r : ℕ) where
  coefs : Fin r → ℤ
  offset : ℤ

namespace Form

variable {r s : ℕ}

def eval (f : Form r) (idx : Fin r → ℤ) : ℤ :=
  f.offset + ∑ d, f.coefs d * idx d

/-- `Form::compose`: substitute coordinate `d` of `f` by `inner d`. -/
def compose (f : Form s) (inner : Fin s → Form r) : Form r where
  coefs e := ∑ d, f.coefs d * (inner d).coefs e
  offset := f.offset + ∑ d, f.coefs d * (inner d).offset

theorem eval_compose (f : Form s) (inner : Fin s → Form r) (idx : Fin r → ℤ) :
    (f.compose inner).eval idx = f.eval (fun d => (inner d).eval idx) := by
  simp only [eval, compose, mul_add, Finset.sum_add_distrib, Finset.mul_sum,
    Finset.sum_mul]
  rw [Finset.sum_comm (f := fun x y => f.coefs y * (inner y).coefs x * idx x)]
  simp only [mul_assoc]
  ring

end Form

/-- A view of an `s`-dimensional source from an `r`-dimensional space. -/
abbrev View (r s : ℕ) := Fin s → Form r

/-- A tensor with `s` dims, as a function of its (integer) coordinates. -/
abbrev Tensor (α : Type*) (s : ℕ) := (Fin s → ℤ) → α

namespace View

variable {r m s : ℕ} {α : Type*}

def read (v : View r s) (x : Tensor α s) : Tensor α r :=
  fun idx => x (fun d => (v d).eval idx)

/-- Compose an inner view `v` (map space → source) with an outer view `w`
(consumer space → map space). -/
def compose (v : View m s) (w : View r m) : View r s :=
  fun d => (v d).compose w

theorem read_compose (v : View m s) (w : View r m) (x : Tensor α s) :
    (v.compose w).read x = fun idx => v.read x (fun e => (w e).eval idx) := by
  funext idx
  simp [read, compose, Form.eval_compose]

/-- Reading a map through a view = applying the map's body to the composed
read. This is the whole content of `substitute_maps` for one argument. -/
theorem read_map {β : Type*} (f : α → β) (v : View m s) (w : View r m)
    (x : Tensor α s) :
    w.read (fun j => f (v.read x j)) = fun idx => f ((v.compose w).read x idx) := by
  funext idx
  rw [read_compose]
  rfl

end View

/-- A monoid comprehension `out[i] = ⨁_{k ∈ K} body(read(i, k))`, with the
reduce coordinates appended to the point coordinates by `pt`. Commutative
monoids model `sum`, `max` (on `WithBot`), `min`, `prod`. -/
def comp {r p : ℕ} {κ α β : Type*} [CommMonoid β] (K : Finset κ)
    (pt : (Fin p → ℤ) → κ → (Fin r → ℤ)) (body : α → β) (a : Tensor α r) :
    (Fin p → ℤ) → β :=
  fun i => ∏ k ∈ K, body (a (pt i k))

/-- **Fusion soundness.** Substituting a map `m = f ∘ (x read through v)`
into a comprehension that reads `m` through `w` gives the comprehension that
reads `x` through the composed view `v ∘ w` with body `body ∘ f`. -/
theorem substitute_map_sound {r m s p : ℕ} {κ α γ β : Type*} [CommMonoid β]
    (K : Finset κ) (pt : (Fin p → ℤ) → κ → (Fin r → ℤ)) (body : γ → β)
    (f : α → γ) (v : View m s) (w : View r m) (x : Tensor α s) :
    comp K pt body (w.read (fun j => f (v.read x j)))
      = comp K pt (body ∘ f) ((v.compose w).read x) := by
  rw [View.read_map]
  rfl

/-- Two-argument bodies (e.g. a product in a contraction): substituting a
map into one argument leaves the other untouched. -/
theorem substitute_map_sound₂ {r m s p : ℕ} {κ α γ δ β : Type*} [CommMonoid β]
    (K : Finset κ) (pt : (Fin p → ℤ) → κ → (Fin r → ℤ)) (body : γ → δ → β)
    (f : α → γ) (v : View m s) (w : View r m) (x : Tensor α s) (y : Tensor δ r) :
    (fun i => ∏ k ∈ K, body (w.read (fun j => f (v.read x j)) (pt i k)) (y (pt i k)))
      = fun i => ∏ k ∈ K, body (f ((v.compose w).read x (pt i k))) (y (pt i k)) := by
  rw [View.read_map]

end Tang

namespace Tang

/-- **Split / tiled reductions** (`split_reduction`, `schedule_split_k`,
workgroup tree combines): reducing each chunk and then the partials gives
the same value as reducing the whole, for any monoid (associativity alone;
the chunks keep their order, so no commutativity is needed). -/
theorem split_reduction {β : Type*} [Monoid β] (chunks : List (List β)) :
    (chunks.map List.prod).prod = chunks.flatten.prod :=
  (List.prod_flatten).symm

end Tang
