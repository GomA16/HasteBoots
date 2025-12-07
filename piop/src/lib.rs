pub mod ntt;

use algebra::{Field, ListOfProductsOfPolynomials};

pub struct SumcheckClaim<F: Field> {
    poly: ListOfProductsOfPolynomials<F>,
    sum: F,
}
use serde::Serialize;
use sumcheck::verifier::SubClaim as SumcheckSubclaim;

/// PIOP trait for sumcheck-based protocols
pub trait SumcheckPIOP<F: Field> {
    type Instance;
    type Info: Serialize;
    type Proof;
    type ProverState; // State stored for prover to generate evaluation proofs later.
    type VerifierSubclaim; // Subclaim stored for verifier to check evaluation proofs later.
    type FSTranscript;

    /// Generate the PIOP proof (with transcript) for given instance
    /// and store the prover state to generate evaluation proofs later.
    fn prover(
        trans: &mut Self::FSTranscript,
        instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState);

    /// Verify the PIOP proof (with transcript) for given info (statement),
    /// and store the subclaim to be check the evaluation proofs later.
    fn verifier(
        trans: &mut Self::FSTranscript,
        info: &Self::Info,
        proof: &Self::Proof,
    ) -> (bool, Self::VerifierSubclaim);

    /// Batch sumcheck protocols with given randomness.
    fn prover_batch_sumcheck(
        instance: &Self::Instance,
        claim: &mut SumcheckClaim<F>,
        randomness: &[F],
    ) -> Self::ProverState;

    /// Verify the subclaim for the batched sumcheck protocol.
    fn verifier_compute_subclaim(
        proof: &Self::Proof,
        subclaim: &mut SumcheckSubclaim<F>,
        randomness: &[F],
    );
}

impl<F: Field> SumcheckClaim<F> {
    pub fn new(num_vars: usize) -> Self {
        Self {
            poly: ListOfProductsOfPolynomials::new(num_vars),
            sum: F::zero(),
        }
    }

    pub fn poly_mut(&mut self) -> &mut ListOfProductsOfPolynomials<F> {
        &mut self.poly
    }

    pub fn sum_mut(&mut self) -> &mut F {
        &mut self.sum
    }

    pub fn poly_ref(&self) -> &ListOfProductsOfPolynomials<F> {
        &self.poly
    }

    pub fn sum_ref(&self) -> &F {
        &self.sum
    }
}
