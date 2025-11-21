#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![deny(missing_docs)]

//! PIOP Library
//! This library provides the PIOP (Polynomial Interactive Oracle Proof) protocol.
//! It includes operations such as:
//! 1. Modular Addition
//! 2. NTT

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};
use std::fmt;

/// 
pub mod mod_addition;

/// Trait for PIOP instance info.
pub trait PIOPInstanceInfo {
    type Info: fmt::Display + Clone;
    fn num_vars(&self) -> usize;
}

/// Trait for PIOP instance.
pub trait PIOPInstance<F: Field> {
    /// The type of the instance.
    type Instance: PIOPInstanceInfo;
    /// The type of the instance info.
    type Info;
    /// The type of the evaluation of the instance at a specific point.
    type Eval;

    /// Gets the information about the instance that is sent to the verifier.
    fn info(&self) -> Self::Info;

    /// Gets the number of variables in each polynomial involved in the instance.
    fn num_vars(&self) -> usize;

    /// Gets the number of polynomials involved in the instance.
    fn num_polys(&self) -> usize;

    /// Gets the logarithm of the number of polynomials,
    /// which corresponds to the added number of variables
    /// when packing all polynomials into a larger polynomial.
    fn log_num_polys(&self) -> usize {
        self.num_polys().next_power_of_two().ilog2() as usize
    }

    /// Evaluates all the polynomials involved in the instance at a base-field point.
    fn eval(&self) -> Self::Eval;

    /// Evaluates all the polynomials involoved in the instance at an extension-field point.
    fn eval_ef<EF: AbstractExtensionField<F>>(&self, point: &[EF]) -> Self::Eval;

    /// Flatten all dense multilinear polynomials into a single vector.
    fn pack_all_polys(&self) -> Vec<F>;

    /// Generate the oracle extracted from this instance, which will be committed by the prover.
    /// This oracle contains all input/output/witness polynomials, followed by zero polynomials to
    /// pad the total number of polynomials to a power of two.
    fn generate_oracle(&self) -> DenseMultilinearExtension<F> {
        let num_vars_extended = self.log_num_polys();
        let num_vars_final = self.num_vars() + self.log_num_polys();
        let num_zeros_padded = ((1 << num_vars_extended) - self.num_polys()) << self.num_vars();
        
        let mut evals = self.pack_all_polys();
        evals.reserve(num_zeros_padded);
        evals.resize(1 << num_vars_final, F::zero());
        <DenseMultilinearExtension<F>>::from_evaluations_vec(num_vars_final, evals)
    }


}