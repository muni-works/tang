//! tang-expr — RISC expression graph for symbolic computation.
//!
//! Builds computation graphs from generic `Scalar` code via a thread-local
//! graph. Enables symbolic differentiation, sparsity detection, simplification,
//! and multi-backend compilation (CPU closures, WGSL shaders).
//!
//! # Quick start
//!
//! ```
//! use tang::Vec3;
//! use tang_expr::{trace, ExprId};
//!
//! let (mut g, dot) = trace(|| {
//!     let a = Vec3::new(ExprId::var(0), ExprId::var(1), ExprId::var(2));
//!     let b = Vec3::new(ExprId::var(3), ExprId::var(4), ExprId::var(5));
//!     a.dot(b)
//! });
//!
//! // Evaluate with concrete values
//! let result: f64 = g.eval(dot, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//! assert!((result - 32.0).abs() < 1e-10);
//!
//! // Symbolic differentiation
//! let ddot_dx0 = g.diff(dot, 0);
//! let ddot_dx0 = g.simplify(ddot_dx0);
//! ```
//!
//! # Integer dialect
//!
//! [`int::IntGraph`] is a separate, bit-exact 32-bit integer graph
//! (`i32`/`u32`/`bool`, wrapping arithmetic, `mul_wide`, structured loops)
//! with three lowerings that compute the same bits: the CPU interpreter
//! ([`int::IntGraph::eval`] / [`int::Program`]), WGSL and Rust source.
//!
//! ```
//! use tang_expr::int::{IType, IntGraph};
//!
//! // sum of i*i for i in 0..n
//! let mut g = IntGraph::new();
//! let n = g.param(0, IType::U32);
//! let zero = g.u32(0);
//! let s = g.for_range(n, &[zero], |g, i, s| {
//!     let sq = g.mul(i, i);
//!     vec![g.add(s[0], sq)]
//! });
//! assert_eq!(g.eval(&s, &[10], &[]), vec![285]);
//!
//! let wgsl = g.to_wgsl_fn("sum_sq", &s);
//! assert!(wgsl.contains("fn sum_sq"));
//! ```

pub mod codegen;
pub mod compile;
pub mod diff;
pub mod display;
pub mod eval;
pub mod graph;
pub mod int;
pub mod node;
pub mod reverse;
mod scalar;
pub mod simplify;
pub mod sparsity;
pub mod split;
pub mod wgsl;

pub use graph::ExprGraph;
pub use node::{ExprId, Node};

use std::cell::RefCell;

thread_local! {
    static GRAPH: RefCell<ExprGraph> = RefCell::new(ExprGraph::new());
}

/// Access the thread-local graph.
pub fn with_graph<F, R>(f: F) -> R
where
    F: FnOnce(&mut ExprGraph) -> R,
{
    GRAPH.with(|g| f(&mut g.borrow_mut()))
}

/// Run a closure with a fresh graph, returning the graph and result.
///
/// Installs a new empty graph, runs `f` (which builds the expression via
/// `ExprId` arithmetic / `Scalar` calls), then extracts the graph.
pub fn trace<F, R>(f: F) -> (ExprGraph, R)
where
    F: FnOnce() -> R,
{
    // Swap in a fresh graph
    GRAPH.with(|g| {
        let old = std::mem::take(&mut *g.borrow_mut());
        let result = f();
        let graph = std::mem::replace(&mut *g.borrow_mut(), old);
        (graph, result)
    })
}

impl ExprId {
    /// Create a variable node in the thread-local graph.
    #[inline]
    pub fn var(n: u16) -> Self {
        with_graph(|g| g.var(n))
    }
}

#[cfg(test)]
mod floor_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use tang::{Scalar, Vec3};

    #[test]
    fn trace_vec3_dot() {
        let (g, dot) = trace(|| {
            let a = Vec3::new(ExprId::var(0), ExprId::var(1), ExprId::var(2));
            let b = Vec3::new(ExprId::var(3), ExprId::var(4), ExprId::var(5));
            a.dot(b)
        });

        // Evaluate: [1,2,3] . [4,5,6] = 4+10+18 = 32
        let result: f64 = g.eval(dot, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!((result - 32.0).abs() < 1e-10);
    }

    #[test]
    fn trace_vec3_norm() {
        let (g, norm) = trace(|| {
            let v = Vec3::new(ExprId::var(0), ExprId::var(1), ExprId::var(2));
            v.norm()
        });

        // norm([3,4,0]) = 5
        let result: f64 = g.eval(norm, &[3.0, 4.0, 0.0]);
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn trace_isolation() {
        // Traces should be isolated
        let (g1, _) = trace(|| {
            let _x = ExprId::var(0);
        });
        let (g2, _) = trace(|| {
            let _x = ExprId::var(0);
        });
        // Both graphs should have same size (3 pre-populated + 1 var)
        assert_eq!(g1.len(), g2.len());
    }

    #[test]
    fn from_f64_creates_lit() {
        let (g, v) = trace(|| ExprId::from_f64(42.0));
        let result: f64 = g.eval(v, &[]);
        assert!((result - 42.0).abs() < 1e-10);
    }

    #[test]
    fn scalar_constants() {
        let (g, (zero, one, two)) = trace(|| {
            let z: ExprId = Scalar::from_f64(0.0);
            let o: ExprId = Scalar::from_f64(1.0);
            let t: ExprId = Scalar::from_f64(2.0);
            (z, o, t)
        });
        assert_eq!(g.eval::<f64>(zero, &[]), 0.0);
        assert_eq!(g.eval::<f64>(one, &[]), 1.0);
        assert_eq!(g.eval::<f64>(two, &[]), 2.0);
    }
}
