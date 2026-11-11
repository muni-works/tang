//! Expression node types and ExprId handle.

use std::fmt;

/// Handle into the expression graph. Lightweight (4 bytes), Copy.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(pub(crate) u32);

/// Well-known node indices, pre-populated in every graph.
impl ExprId {
    /// The constant 0.0 (index 0).
    pub const ZERO: Self = Self(0);
    /// The constant 1.0 (index 1).
    pub const ONE: Self = Self(1);
    /// The constant 2.0 (index 2).
    pub const TWO: Self = Self(2);

    /// Create an ExprId from a raw index.
    #[inline]
    pub fn from_index(index: u32) -> Self {
        Self(index)
    }

    /// The raw index of this expression in the graph.
    #[inline]
    pub fn index(&self) -> u32 {
        self.0
    }
}

impl fmt::Debug for ExprId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "e{}", self.0)
    }
}

impl fmt::Display for ExprId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "e{}", self.0)
    }
}

impl Default for ExprId {
    fn default() -> Self {
        Self::ZERO
    }
}

impl PartialOrd for ExprId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.0.cmp(&other.0))
    }
}

/// A node in the expression graph.
///
/// 9 RISC primitive operations + 2 atom types. Every higher-level math
/// operation decomposes into these primitives via the `Scalar` impl.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Node {
    // Atoms
    /// Input variable by index.
    Var(u16),
    /// Literal f64 value stored as bits for Hash/Eq.
    Lit(u64),

    // RISC ops (9 primitives)
    /// Addition.
    Add(ExprId, ExprId),
    /// Multiplication.
    Mul(ExprId, ExprId),
    /// Negation.
    Neg(ExprId),
    /// Reciprocal (1/x).
    Recip(ExprId),
    /// Square root.
    Sqrt(ExprId),
    /// Sine (only trig primitive).
    Sin(ExprId),
    /// Two-argument arctangent atan2(y, x).
    Atan2(ExprId, ExprId),
    /// Base-2 exponential (2^x).
    Exp2(ExprId),
    /// Base-2 logarithm.
    Log2(ExprId),
    /// Floor: the largest integer <= x. Piecewise constant, so its
    /// derivative is 0. `t mod r` is built from it (see `ExprGraph::modulo`).
    Floor(ExprId),
    /// Branchless select: returns `a` if `cond > 0`, else `b`.
    Select(ExprId, ExprId, ExprId),
}

impl Node {
    /// Create a `Lit` node from an f64 value.
    #[inline]
    pub fn lit(v: f64) -> Self {
        Self::Lit(v.to_bits())
    }

    /// Extract f64 value from a `Lit` node, or `None`.
    #[inline]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Lit(bits) => Some(f64::from_bits(*bits)),
            _ => None,
        }
    }
}
