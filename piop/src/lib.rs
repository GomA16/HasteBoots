pub mod ntt;

use algebra::{Field, ListOfProductsOfPolynomials};

pub struct SumcheckClaim<F: Field> {
    poly: ListOfProductsOfPolynomials<F>,
    sum: F,
}
use sumcheck::verifier::SubClaim as SumcheckSubclaim;

pub trait SumcheckPIOP<F: Field> {
    type Instance;
    type Info;
    type Proof;
    type ProverState;
    type FSTranscript;

    fn prover(trans: &mut Self::FSTranscript, instance: &Self::Instance) -> Self::Proof;
    fn verifier(trans: &mut Self::FSTranscript, info: &Self::Info, proof: &Self::Proof) -> bool;
    fn prover_batch_sumcheck(
        instance: &Self::Instance,
        claim: &mut SumcheckClaim<F>,
        randomness: &[F],
    ) -> Self::ProverState;
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
