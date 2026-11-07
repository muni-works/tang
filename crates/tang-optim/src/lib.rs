//! Optimization algorithms — Adam, L-BFGS, Newton, Levenberg-Marquardt.
//!
//! [`minimize_lbfgs`] (projected two-loop L-BFGS with line search) and
//! [`levenberg_marquardt`] (adaptive damping over a [`LeastSquares`] system)
//! are full minimizers; [`Lbfgs`] and the other optimizers are steppers.

#![no_std]

extern crate alloc;

mod adam;
mod lbfgs;
mod lbfgs_projected;
mod least_squares;
mod line_search;
mod lm;
mod newton;
mod sgd;

pub use adam::{Adam, AdamW};
pub use lbfgs::Lbfgs;
pub use lbfgs_projected::{
    minimize_lbfgs, project_box, LbfgsIterate, LbfgsOptions, LbfgsResult, LbfgsStop,
};
pub use least_squares::{levenberg_marquardt, LeastSquares, LmConfig, LmResult, LmStatus};
pub use line_search::armijo;
pub use lm::LevenbergMarquardt;
pub use newton::Newton;
pub use sgd::Sgd;
