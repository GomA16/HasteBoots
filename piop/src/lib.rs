pub mod lookup;
pub mod ntt;

use algebra::{DenseMultilinearExtension, Field, ListOfProductsOfPolynomials};
use helper::{Transcript, utils::gen_identity_evaluations};
use std::rc::Rc;

pub struct SumcheckClaim<F: Field> {
    poly: ListOfProductsOfPolynomials<F>,
    sum: F,
}

/// When proving the \sum a(x)b(x) = c(x) on a hypercube, it can be reduced 
/// to a sumchek by applying a Lagrange kernel eq( , r) for a random point r.
pub struct LagrangeKernel<F: Field> {
    pub point: Vec<F>,
    pub eq_at_point: Rc<DenseMultilinearExtension<F>>,
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
    /// # Parameters
    /// - `instance`: The instance for the batched sumcheck protocol.
    /// - `claim`: The sumcheck claim to be filled in the batched sumcheck
    /// - `randomness`: The randomness used to batch the sumcheck protocols.
    /// - `lagrange_kernel`: (Optional) The Lagrange kernel used to reduce
    ///   the sum of products into a standard sumcheck.
    fn prover_batch_sumcheck(
        instance: &Self::Instance,
        claim: &mut SumcheckClaim<F>,
        randomness: &[F],
        lagrange_kernel: Option<&LagrangeKernel<F>>,
    ) -> Self::ProverState;

    /// Verify the subclaim for the batched sumcheck protocol.
    fn verifier_compute_subclaim(
        proof: &Self::Proof,
        subclaim: &mut SumcheckSubclaim<F>,
        randomness: &[F],
        lagrange_kernel: Option<&LagrangeKernel<F>>,
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

impl<F: Field> LagrangeKernel<F> {
    pub fn new(&self, num_vars: usize, trans: &mut Transcript<F>)-> Self {
        let point = trans.get_vec_challenge(
            b"Sample random point for a batch of sumchecks over products",
            num_vars,
        );
        let eq_at_point = Rc::new(gen_identity_evaluations(&point));
        Self {
            point,
            eq_at_point,
        }
    }
}
