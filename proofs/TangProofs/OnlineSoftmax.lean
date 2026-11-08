/-
The online-softmax monoid (`stabilize_softmax`, FlashAttention).

State `(m, l, o)`: running max, running `Σ exp(x - m)`, running
`Σ exp(x - m) · v`. Combine rescales both sides to the larger max. We prove:
* the combine is associative and commutative (so any tiling / split /
  reordering of the reduction gives the same state),
* its meaning `⟦(m,l,o)⟧ = (l·eᵐ, o·eᵐ)` is a homomorphism to `(ℝ × ℝ, +)`,
* folding it over the scores gives `(max x, Σ e^{x-max}, Σ e^{x-max} v)`, so
  `o / l` is exactly two-pass (and naive) softmax-weighted averaging,
* softmax is invariant under any shift `c`, which is why the stabilizing
  max is free (see `Stabilize.lean` for the gradient statement).
The value channel is scalar; a `d`-vector of values is `d` independent
copies (the combine acts componentwise with shared `m`, `l`).
-/
import Mathlib

namespace Tang

open Real Finset BigOperators

/-- Online softmax state. -/
@[ext] structure OSM where
  m : ℝ
  l : ℝ
  o : ℝ

namespace OSM

noncomputable def combine (a b : OSM) : OSM :=
  let M := max a.m b.m
  ⟨M, a.l * exp (a.m - M) + b.l * exp (b.m - M),
      a.o * exp (a.m - M) + b.o * exp (b.m - M)⟩

/-- One score `x` with value `v`. -/
def single (x v : ℝ) : OSM := ⟨x, 1, v⟩

/-- What a state means: the unnormalized `(Σ eˣ, Σ eˣ v)`. -/
noncomputable def sem (a : OSM) : ℝ × ℝ := (a.l * exp a.m, a.o * exp a.m)

theorem sem_combine (a b : OSM) : (combine a b).sem = a.sem + b.sem := by
  simp only [sem, combine, Prod.mk_add_mk, Prod.mk.injEq]
  constructor <;>
  · rw [add_mul, mul_assoc, mul_assoc, ← exp_add, ← exp_add]
    simp

/-- A state is determined by its max and its meaning. -/
theorem eq_of_sem {a b : OSM} (hm : a.m = b.m) (hs : a.sem = b.sem) : a = b := by
  simp only [sem, Prod.mk.injEq] at hs
  have h := (exp_pos a.m).ne'
  ext
  · exact hm
  · rw [hm] at hs; exact mul_right_cancel₀ (exp_pos b.m).ne' hs.1
  · rw [hm] at hs; exact mul_right_cancel₀ (exp_pos b.m).ne' hs.2

theorem combine_m (a b : OSM) : (combine a b).m = max a.m b.m := rfl

/-- **Associativity**: the online-softmax combine is a semigroup. -/
theorem combine_assoc (a b c : OSM) :
    combine (combine a b) c = combine a (combine b c) := by
  apply eq_of_sem
  · simp [combine_m, max_assoc]
  · simp [sem_combine, add_assoc]

theorem combine_comm (a b : OSM) : combine a b = combine b a := by
  apply eq_of_sem
  · simp [combine_m, max_comm]
  · simp [sem_combine, add_comm]

/-- Left fold of the combine over a nonempty list of scores and values,
seeded with the first element (no `-∞` identity needed). -/
noncomputable def run : List (ℝ × ℝ) → OSM
  | [] => ⟨0, 0, 0⟩
  | (x, v) :: rest => rest.foldl (fun s p => combine s (single p.1 p.2)) (single x v)

theorem foldl_sem (rest : List (ℝ × ℝ)) (s : OSM) :
    (rest.foldl (fun s p => combine s (single p.1 p.2)) s).sem
      = s.sem + (rest.map (fun p => (exp p.1, exp p.1 * p.2))).sum := by
  induction rest generalizing s with
  | nil => simp
  | cons p rest ih =>
    simp only [List.foldl_cons, ih, sem_combine, List.map_cons, List.sum_cons]
    ext <;> simp [sem, single] <;> ring

theorem foldl_m (rest : List (ℝ × ℝ)) (s : OSM) :
    (rest.foldl (fun s p => combine s (single p.1 p.2)) s).m
      = rest.foldl (fun acc p => max acc p.1) s.m := by
  induction rest generalizing s with
  | nil => simp
  | cons p rest ih =>
    simp only [List.foldl_cons]
    rw [ih]
    rfl

theorem foldl_max_assoc (a b : ℝ) (l : List (ℝ × ℝ)) :
    l.foldl (fun acc p => max acc p.1) (max a b)
      = max a (l.foldl (fun acc p => max acc p.1) b) := by
  induction l generalizing b with
  | nil => rfl
  | cons p l ih =>
    simp only [List.foldl_cons]
    rw [max_assoc, ih]

/-- The fold computes the unnormalized sums exactly. -/
theorem run_sem (x v : ℝ) (rest : List (ℝ × ℝ)) :
    (run ((x, v) :: rest)).sem
      = (((x, v) :: rest).map (fun p => (exp p.1, exp p.1 * p.2))).sum := by
  simp only [run]
  rw [foldl_sem]
  simp [sem, single, mul_comm]

theorem sum_pair {α : Type*} (f g : α → ℝ) (l : List α) :
    (l.map (fun p => (f p, g p))).sum = ((l.map f).sum, (l.map g).sum) := by
  induction l with
  | nil => rfl
  | cons a l ih => simp [ih]

/-- **Flash = two-pass softmax.** For any nonempty block of scores, the
online state's `o / l` equals `Σᵢ softmax(x)ᵢ · vᵢ` computed with the
global normalizer `Σ eˣ`, and its `m` is the running max. -/
theorem run_eq_softmax (x v : ℝ) (rest : List (ℝ × ℝ)) :
    let s := run ((x, v) :: rest)
    let ps := (x, v) :: rest
    s.o / s.l = (ps.map (fun p => exp p.1 * p.2)).sum / (ps.map (fun p => exp p.1)).sum := by
  intro s ps
  have h := run_sem x v rest
  simp only [sem] at h
  rw [sum_pair] at h
  have hfst := congrArg Prod.fst h
  have hsnd := congrArg Prod.snd h
  simp only at hfst hsnd
  have hpos := (exp_pos s.m).ne'
  rw [← hfst, ← hsnd, mul_div_mul_right _ _ hpos]

/-- Tiling: folding a concatenation of tiles is combining the tiles' states. -/
theorem run_append (p q : ℝ × ℝ) (r1 r2 : List (ℝ × ℝ)) :
    run ((p :: r1) ++ (q :: r2)) = combine (run (p :: r1)) (run (q :: r2)) := by
  apply eq_of_sem
  · -- maxima agree: both are the max of all scores
    simp only [run, List.cons_append, List.foldl_append, List.foldl_cons]
    simp only [foldl_m, combine_m]
    simp only [single]
    rw [foldl_max_assoc]
  · simp only [List.cons_append, sem_combine]
    rw [show p :: (r1 ++ q :: r2) = (p.1, p.2) :: (r1 ++ q :: r2) from rfl, run_sem,
      show p :: r1 = (p.1, p.2) :: r1 from rfl, run_sem,
      show q :: r2 = (q.1, q.2) :: r2 from rfl, run_sem]
    simp [List.map_append, List.sum_append, add_assoc]

end OSM

/-- Softmax with an arbitrary shift `c`. -/
noncomputable def softmaxShift {n : ℕ} (x : Fin n → ℝ) (c : ℝ) (i : Fin n) : ℝ :=
  exp (x i - c) / ∑ j, exp (x j - c)

noncomputable def softmax {n : ℕ} (x : Fin n → ℝ) (i : Fin n) : ℝ :=
  exp (x i) / ∑ j, exp (x j)

/-- **Shift invariance**: `stabilize_softmax` may subtract any `c`, in
particular the row max, without changing the value. -/
theorem softmaxShift_eq {n : ℕ} (x : Fin n → ℝ) (c : ℝ) :
    softmaxShift x c = softmax x := by
  funext i
  simp only [softmaxShift, softmax, exp_sub, ← Finset.sum_div]
  rw [div_div_div_cancel_right₀ (exp_pos c).ne']

/-- Same for log-sum-exp: `c + log Σ e^{x-c} = log Σ eˣ` (nonempty rows). -/
theorem lse_shift {n : ℕ} [NeZero n] (x : Fin n → ℝ) (c : ℝ) :
    c + log (∑ j, exp (x j - c)) = log (∑ j, exp (x j)) := by
  have hpos : 0 < ∑ j, exp (x j) :=
    Finset.sum_pos (fun j _ => exp_pos _) Finset.univ_nonempty
  simp only [exp_sub, ← Finset.sum_div]
  rw [log_div hpos.ne' (exp_pos c).ne', log_exp]
  ring

end Tang
