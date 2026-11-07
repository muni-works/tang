//! CSR/CSC sparse matrices and sparse solvers.

#![no_std]

extern crate alloc;

mod coo;
mod csc;
mod csr;
pub mod pcg;

pub use coo::CooMatrix;
pub use csc::CscMatrix;
pub use csr::CsrMatrix;
pub use pcg::{
    csr_apply, csr_diagonal, pcg, Ic0, Ic0Error, Identity, Jacobi, PcgOptions, PcgSolution,
    PcgStatus, Preconditioner,
};
