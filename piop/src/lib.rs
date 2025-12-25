pub mod hadamard;
pub mod lookup;
pub mod ntt;
pub mod permutation;
pub mod sparse_matrix_eval;

use algebra::{DenseMultilinearExtension, Field, ListOfProductsOfPolynomials, PolynomialInfo};
use helper::{
    FiatShamirTranscript, Transcript,
    utils::{eval_identity_function, gen_identity_evaluations},
};
use std::rc::Rc;
use sumcheck::{MLSumcheck, Proof, prover::ProverState, verifier::SubClaim};

pub struct SumcheckClaim<F: Field> {
    pub poly: ListOfProductsOfPolynomials<F>,
    pub sum: F,
}

/// When proving the \sum a(x)b(x) = c(x) on a hypercube, it can be reduced
/// to a sumchek by applying a Lagrange kernel eq( , r) for a random point r.
pub struct LagrangeKernel<F: Field> {
    pub point: Vec<F>,
    pub eq_at_point: Rc<DenseMultilinearExtension<F>>,
}

impl<F: Field> LagrangeKernel<F> {
    // generate a random Lagrange kernel
    pub fn random(trans: &mut Transcript<F>, num_vars: usize) -> Self {
        let point = trans.get_vec_challenge(
            b"Sample random point for a batch of sumchecks over products",
            num_vars,
        );
        let eq_at_point = Rc::new(gen_identity_evaluations(&point));
        Self { point, eq_at_point }
    }

    // generate a random Lagrange kernel but only return the point
    pub fn random_point(trans: &mut Transcript<F>, num_vars: usize) -> Vec<F> {
        let point = trans.get_vec_challenge(
            b"Sample random point for a batch of sumchecks over products",
            num_vars,
        );
        point
    }

    // generate the Lagrange kernel for a given point
    pub fn from_point(point: &[F]) -> Self {
        let eq_at_point = Rc::new(gen_identity_evaluations(point));
        Self {
            point: point.to_vec(),
            eq_at_point,
        }
    }

    pub fn evaluate(&self, x: &[F]) -> F {
        self.eq_at_point.evaluate(x)
    }
}

use serde::Serialize;

pub trait SumcheckInstance<F: Field + Serialize> {
    type Info: SumcheckInfo<F> + Serialize;

    fn info(&self) -> Self::Info;
    fn num_vars(&self) -> usize;
}

pub trait SumcheckInfo<F: Field> {
    fn num_vars(&self) -> usize;
    fn num_sumchecks(&self) -> usize;
    fn sumcheck_num_vars(&self) -> usize;
    fn sample_randomness_for_sumcheck(&self, trans: &mut Transcript<F>) -> Vec<F> {
        trans.get_vec_challenge(
            b"Sample random coefficients for a batch of sumchecks",
            self.num_sumchecks(),
        )
    }
}

pub trait SumcheckPureProof<F: Field> {
    fn from_sumcheck(sumcheck_claim: &SumcheckClaim<F>, proof: Proof<F>) -> Self;
    fn get_poly_info(&self) -> &PolynomialInfo;
    fn get_sumcheck_proof(&self) -> &Proof<F>;
}

pub trait SumcheckPureSubclaim<F: Field> {
    fn from_sumcheck(sumcheck_subclaim: SubClaim<F>) -> Self;
}

pub trait SumcheckPureProverState<F: Field> {
    fn from_sumcheck(sumcheck_prover_state: ProverState<F>) -> Self;
}

/// PIOP trait for sumcheck-based protocols
pub trait SumcheckPIOP<F: Field + Serialize> {
    type Instance: SumcheckInstance<F>;
    type Info: SumcheckInfo<F> + Serialize;
    type Proof: SumcheckPureProof<F>; // Proof stored for verifier to check evaluation proofs.
    type ProverState: SumcheckPureProverState<F>; // State stored for prover to generate evaluation proofs later.
    type VerifierSubclaim: SumcheckPureSubclaim<F>; // Subclaim stored for verifier to check evaluation proofs later.
    // type FSTranscript: FiatShamirTranscript<F>;

    /// Generate the PIOP proof (with transcript) for given instance
    /// and store the prover state to generate evaluation proofs later.
    fn prover(
        trans: &mut Transcript<F>,
        instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState);

    /// Verify the PIOP proof (with transcript) for given info (statement),
    /// and store the subclaim to be check the evaluation proofs later.
    fn verifier(
        trans: &mut Transcript<F>,
        info: &Self::Info,
        proof: &Self::Proof,
    ) -> (bool, Self::VerifierSubclaim) {
        trans.append_message(b"[Statement]", &info);

        let mut res = true;
        let kernel_point = LagrangeKernel::random_point(trans, info.num_vars());
        let randomness_batch = info.sample_randomness_for_sumcheck(trans);

        let mut sumcheck_subclaim = MLSumcheck::verify(
            trans,
            proof.get_poly_info(),
            MLSumcheck::extract_sum(&proof.get_sumcheck_proof()),
            &proof.get_sumcheck_proof(),
        )
        .expect("[SumcheckIOP - Verifier] Fail to verify the sumcheck");

        let kernel_at_r = eval_identity_function(&kernel_point, &sumcheck_subclaim.point);

        Self::verifier_compute_subclaim(
            info,
            proof,
            &mut sumcheck_subclaim,
            &randomness_batch,
            Some(kernel_at_r),
        );
        res &= sumcheck_subclaim.expected_evaluations.is_zero();

        let subclaim = Self::VerifierSubclaim::from_sumcheck(sumcheck_subclaim);
        (res, subclaim)
    }

    /// Generate the PIOP proof (with transcript) for given instance
    /// but without generating the evaluations.
    /// This is useful when we only need to generate the proof from the base field instance.
    fn prover_without_evals(
        trans: &mut Transcript<F>,
        instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState) {
        let info = instance.info();
        trans.append_message(b"[Statement]", &info);

        let mut sumcheck_claim = SumcheckClaim::new(info.num_vars());
        let lagrange_kernel = Some(&LagrangeKernel::random(trans, instance.num_vars()));
        let randomness_batch = info.sample_randomness_for_sumcheck(trans);
        Self::prover_batch_sumcheck(
            instance,
            &mut sumcheck_claim,
            &randomness_batch,
            lagrange_kernel,
        );
        let (sumcheck_proof, prover_state) = MLSumcheck::prove(trans, &sumcheck_claim.poly)
            .expect("[SumcheckIOP] Fail to generate sumcheck proof");

        let proof = Self::Proof::from_sumcheck(&sumcheck_claim, sumcheck_proof);
        let state = Self::ProverState::from_sumcheck(prover_state);
        (proof, state)
    }

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
    ) -> Option<Self::ProverState>;

    /// Verify the subclaim for the batched sumcheck protocol.
    fn verifier_compute_subclaim(
        info: &Self::Info,
        proof: &Self::Proof,
        subclaim: &mut SubClaim<F>,
        randomness: &[F],
        kernel_at_r: Option<F>,
    );
}

impl<F: Field> SumcheckClaim<F> {
    pub fn new(num_vars: usize) -> Self {
        Self {
            poly: ListOfProductsOfPolynomials::new(num_vars),
            sum: F::zero(),
        }
    }
}
