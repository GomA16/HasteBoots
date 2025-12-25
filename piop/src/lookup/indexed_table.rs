use std::rc::Rc;

use algebra::{Field, PolynomialInfo};
use helper::{FiatShamirTranscript, Transcript, utils::eval_identity_function};
use serde::Serialize;
use sumcheck::{MLSumcheck, Proof, prover::ProverState, verifier::SubClaim};
use trace::lookup_trace::indexed_table::{
    IndexedLookupTraceMLE, IndexedLookupWitness, IndexedLookupWitnessHelper,
};

use crate::{
    LagrangeKernel, SumcheckClaim, SumcheckInfo, SumcheckInstance, SumcheckPIOP, SumcheckPureProof,
    SumcheckPureProverState, SumcheckPureSubclaim,
};

/// This implementation proves the indexed logup with two separate sumchecks,
/// since the number of variables for input and table are usually difference.
pub struct IndexedLogUpInputIOP<F: Field> {
    _marker: std::marker::PhantomData<F>,
}
pub struct IndexedLogUpTableIOP<F: Field> {
    _marker: std::marker::PhantomData<F>,
}
pub struct IndexedLogUpInputInstance<F: Field> {
    pub num_input_vars: usize,
    pub trace: IndexedLookupTraceMLE<F>,
    pub helper: IndexedLookupWitnessHelper<F>,
    pub sum: F,
}

pub struct IndexedLogUpTableInstance<F: Field> {
    pub num_table_vars: usize,
    pub witness: IndexedLookupWitness<F>,
    pub helper: IndexedLookupWitnessHelper<F>,
    pub sum: F,
}

#[derive(Serialize)]
pub struct IndexedLogUpInputInstanceInfo<F: Field> {
    pub num_input_vars: usize,
    pub random_value: F,
    pub random_s_hash: F,
}

#[derive(Serialize)]
pub struct IndexedLogUpTableInstanceInfo<F: Field> {
    pub num_table_vars: usize,
    pub table_point: Vec<F>,
    pub random_value: F,
    pub random_s_hash: F,
}

#[derive(Serialize)]
pub struct IndexedLogUpInputProof<F: Field> {
    pub poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<F>,
    pub input_at_rx: F,
    pub index_at_rx: F,
    pub phi_input_at_rx: F,
    pub helper_input_at_rx: F,
}

#[derive(Serialize)]
pub struct IndexedLogUpTableProof<F: Field> {
    pub poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<F>,
    pub multiplicity_at_ry: F,
    // table_at_ry can be efficiently computed
    // table_at_ry = eq(ry, table_point)
    pub table_at_ry: F,
    pub phi_table_at_ry: F,
    pub helper_table_at_ry: F,
}

pub struct IndexedLogUpProverState<F: Field> {
    pub point_r: Vec<F>,
}

pub struct IndexedLogUpVerifierSubclaim<F: Field> {
    pub point_r: Vec<F>,
}

impl<F: Field> IndexedLogUpInputInstance<F> {
    pub fn from(
        trace: &IndexedLookupTraceMLE<F>,
        helper: &IndexedLookupWitnessHelper<F>,
    ) -> Self {
        Self {
            num_input_vars: trace.num_input_vars,
            trace: trace.clone(),
            helper: helper.clone(),
            sum: helper.sum,
        }
    }

    pub fn add_helper_input_into_sumcheck(&self, claim: &mut SumcheckClaim<F>, random_lambda: F) {
        claim
            .poly
            .add_product(vec![Rc::clone(&self.helper.helper_input)], random_lambda);
        claim.sum += random_lambda * self.helper.sum;
    }

    // helper_input = 1 / phi_input
    pub fn add_helper_input_identity_into_sumcheck(
        &self,
        claim: &mut SumcheckClaim<F>,
        random_lambda: F,
        kernel: &LagrangeKernel<F>,
    ) {
        claim.poly.add_product(
            vec![
                Rc::clone(&self.helper.helper_input),
                Rc::clone(&self.helper.phi_input),
                Rc::clone(&kernel.eq_at_point),
            ],
            random_lambda,
        );
        claim
            .poly
            .add_product(vec![Rc::clone(&kernel.eq_at_point)], -random_lambda);
    }
}

impl<F: Field> IndexedLogUpTableInstance<F> {
    pub fn from(
        witness: &IndexedLookupWitness<F>,
        helper: &IndexedLookupWitnessHelper<F>,
    ) -> Self {
        Self {
            num_table_vars: witness.num_table_vars,
            witness: witness.clone(),
            helper: helper.clone(),
            sum: helper.sum,
        }
    }

    pub fn add_helper_table_into_sumcheck(&self, claim: &mut SumcheckClaim<F>, random_lambda: F) {
        claim
            .poly
            .add_product(vec![Rc::clone(&self.helper.helper_table)], random_lambda);
        claim.sum += random_lambda * self.helper.sum;
    }

    // helper_table = multiplicity / phi_table
    pub fn add_helper_table_identity_into_sumcheck(
        &self,
        claim: &mut SumcheckClaim<F>,
        random_lambda: F,
        kernel: &LagrangeKernel<F>,
    ) {
        claim.poly.add_product(
            vec![
                Rc::clone(&self.helper.helper_table),
                Rc::clone(&self.helper.phi_table),
                Rc::clone(&kernel.eq_at_point),
            ],
            random_lambda,
        );
        claim.poly.add_product(
            vec![
                Rc::clone(&kernel.eq_at_point),
                Rc::clone(&self.witness.multiplicity),
            ],
            -random_lambda,
        );
    }
}

impl<F: Field + Serialize> SumcheckInstance<F> for IndexedLogUpInputInstance<F> {
    type Info = IndexedLogUpInputInstanceInfo<F>;

    fn num_vars(&self) -> usize {
        self.num_input_vars
    }

    fn info(&self) -> Self::Info {
        IndexedLogUpInputInstanceInfo {
            num_input_vars: self.num_input_vars,
            random_value: self.helper.random_value,
            random_s_hash: self.helper.random_s_hash,
        }
    }
}

impl<F: Field + Serialize> SumcheckInstance<F> for IndexedLogUpTableInstance<F> {
    type Info = IndexedLogUpTableInstanceInfo<F>;

    fn num_vars(&self) -> usize {
        self.num_table_vars
    }

    fn info(&self) -> Self::Info {
        IndexedLogUpTableInstanceInfo {
            num_table_vars: self.num_table_vars,
            random_value: self.helper.random_value,
            random_s_hash: self.helper.random_s_hash,
            table_point: self.witness.table_point.clone(),
        }
    }
}

impl<F: Field> SumcheckInfo<F> for IndexedLogUpInputInstanceInfo<F> {
    fn num_vars(&self) -> usize {
        self.num_input_vars
    }

    fn num_sumchecks(&self) -> usize {
        2
    }

    fn sumcheck_num_vars(&self) -> usize {
        self.num_vars()
    }
}

impl<F: Field> SumcheckInfo<F> for IndexedLogUpTableInstanceInfo<F> {
    fn num_vars(&self) -> usize {
        self.num_table_vars
    }

    fn num_sumchecks(&self) -> usize {
        2
    }

    fn sumcheck_num_vars(&self) -> usize {
        self.num_vars()
    }
}

impl<F: Field> SumcheckPureProof<F> for IndexedLogUpInputProof<F> {
    fn from_sumcheck(sumcheck_claim: &SumcheckClaim<F>, proof: Proof<F>) -> Self {
        IndexedLogUpInputProof {
            poly_info: sumcheck_claim.poly.info(),
            sumcheck_proof: proof,
            input_at_rx: F::zero(),
            index_at_rx: F::zero(),
            phi_input_at_rx: F::zero(),
            helper_input_at_rx: F::zero(),
        }
    }

    fn get_poly_info(&self) -> &PolynomialInfo {
        &self.poly_info
    }

    fn get_sumcheck_proof(&self) -> &Proof<F> {
        &self.sumcheck_proof
    }
}

impl<F: Field> SumcheckPureProof<F> for IndexedLogUpTableProof<F> {
    fn from_sumcheck(sumcheck_claim: &SumcheckClaim<F>, proof: Proof<F>) -> Self {
        IndexedLogUpTableProof {
            poly_info: sumcheck_claim.poly.info(),
            sumcheck_proof: proof,
            multiplicity_at_ry: F::zero(),
            table_at_ry: F::zero(),
            phi_table_at_ry: F::zero(),
            helper_table_at_ry: F::zero(),
        }
    }

    fn get_poly_info(&self) -> &PolynomialInfo {
        &self.poly_info
    }

    fn get_sumcheck_proof(&self) -> &Proof<F> {
        &self.sumcheck_proof
    }
}

impl<F: Field> SumcheckPureProverState<F> for IndexedLogUpProverState<F> {
    fn from_sumcheck(sumcheck_prover_state: ProverState<F>) -> Self {
        IndexedLogUpProverState {
            point_r: sumcheck_prover_state.randomness.clone(),
        }
    }
}

impl<F: Field> SumcheckPureSubclaim<F> for IndexedLogUpVerifierSubclaim<F> {
    fn from_sumcheck(sumcheck_subclaim: SubClaim<F>) -> Self {
        IndexedLogUpVerifierSubclaim {
            point_r: sumcheck_subclaim.point.clone(),
        }
    }
}

impl<F: Field> IndexedLogUpInputProof<F> {
    pub fn compute_helper_subclaim(&self, random_lambda: F) -> F {
        random_lambda * self.helper_input_at_rx
    }

    pub fn compute_helper_identity_subcliam(&self, random_lambda: F, kernel_at_r: F) -> F {
        random_lambda * (self.helper_input_at_rx * self.phi_input_at_rx * kernel_at_r - kernel_at_r)
    }
}

impl<F: Field> IndexedLogUpTableProof<F> {
    pub fn compute_helper_subclaim(&self, random_lambda: F) -> F {
        random_lambda * self.helper_table_at_ry
    }

    pub fn compute_helper_identity_subcliam(&self, random_lambda: F, kernel_at_r: F) -> F {
        random_lambda
            * (self.helper_table_at_ry * self.phi_table_at_ry * kernel_at_r
                - self.multiplicity_at_ry * kernel_at_r)
    }
}

impl<F: Field + Serialize> SumcheckPIOP<F> for IndexedLogUpInputIOP<F> {
    type Instance = IndexedLogUpInputInstance<F>;
    type Info = IndexedLogUpInputInstanceInfo<F>;
    type Proof = IndexedLogUpInputProof<F>;
    type ProverState = IndexedLogUpProverState<F>;
    type VerifierSubclaim = IndexedLogUpVerifierSubclaim<F>;

    fn prover(
        trans: &mut Transcript<F>,
        instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState) {
        let (mut proof, state) = Self::prover_without_evals(trans, instance);

        proof.phi_input_at_rx = instance.helper.phi_input.evaluate(&state.point_r);
        proof.input_at_rx = instance.trace.input.evaluate(&state.point_r);
        proof.index_at_rx = instance.trace.index.evaluate(&state.point_r);
        proof.helper_input_at_rx = instance.helper.helper_input.evaluate(&state.point_r);
        (proof, state)
    }

    fn prover_batch_sumcheck(
        instance: &Self::Instance,
        claim: &mut SumcheckClaim<F>,
        randomness: &[F],
        kernel: Option<&LagrangeKernel<F>>,
    ) -> Option<Self::ProverState> {
        assert!(kernel.is_some());
        let kernel = kernel.unwrap();

        assert_eq!(randomness.len(), 2);
        instance.add_helper_input_into_sumcheck(claim, randomness[0]);
        instance.add_helper_input_identity_into_sumcheck(claim, randomness[1], kernel);

        None
    }

    fn verifier_compute_subclaim(
            _info: &Self::Info,
            proof: &Self::Proof,
            subclaim: &mut SubClaim<F>,
            randomness: &[F],
            kernel_at_r: Option<F>,
        ) {
        assert!(kernel_at_r.is_some());
        let kernel_at_r = kernel_at_r.unwrap();
        subclaim.expected_evaluations -= proof.compute_helper_subclaim(randomness[0]);
        subclaim.expected_evaluations -= proof.compute_helper_identity_subcliam(randomness[1], kernel_at_r);
    }
}

impl<F: Field + Serialize> SumcheckPIOP<F> for IndexedLogUpTableIOP<F> {
    type Instance = IndexedLogUpTableInstance<F>;
    type Info = IndexedLogUpTableInstanceInfo<F>;
    type Proof = IndexedLogUpTableProof<F>;
    type ProverState = IndexedLogUpProverState<F>;
    type VerifierSubclaim = IndexedLogUpVerifierSubclaim<F>;

    fn prover(
        trans: &mut Transcript<F>,
        instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState) {
        let (mut proof, state) = Self::prover_without_evals(trans, instance);
        proof.phi_table_at_ry = instance.helper.phi_table.evaluate(&state.point_r);
        proof.multiplicity_at_ry = instance.witness.multiplicity.evaluate(&state.point_r);
        proof.table_at_ry = eval_identity_function(&instance.witness.table_point, &state.point_r);
        proof.helper_table_at_ry = instance.helper.helper_table.evaluate(&state.point_r);
        (proof, state)
    }

    fn prover_batch_sumcheck(
            instance: &Self::Instance,
            claim: &mut SumcheckClaim<F>,
            randomness: &[F],
            lagrange_kernel: Option<&LagrangeKernel<F>>,
        ) -> Option<Self::ProverState> {
        assert!(lagrange_kernel.is_some());
        let kernel = lagrange_kernel.unwrap();
        assert_eq!(randomness.len(), 2);
        instance.add_helper_table_into_sumcheck(claim, randomness[0]);
        instance.add_helper_table_identity_into_sumcheck(claim, randomness[1], kernel);
        None
    }

    fn verifier_compute_subclaim(
            _info: &Self::Info,
            proof: &Self::Proof,
            subclaim: &mut SubClaim<F>,
            randomness: &[F],
            kernel_at_r: Option<F>,
        ) {
        assert!(kernel_at_r.is_some());
        let kernel_at_r = kernel_at_r.unwrap();
        subclaim.expected_evaluations -= proof.compute_helper_subclaim(randomness[0]);
        subclaim.expected_evaluations -= proof.compute_helper_identity_subcliam(randomness[1], kernel_at_r);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{SumcheckInstance};
    use algebra::{
        FieldUniformSampler,
        derive::{Field, Prime},
    };
    use helper::Transcript;
    use rand_distr::Distribution;
    use trace::lookup_trace::indexed_table::{IndexedLookupTrace};

    #[derive(Field, Prime)]
    #[modulus = 132120577]
    pub struct Fp32(u32);
    // field type
    type FF = Fp32;

    #[test]
    fn test_logup_iop() {
        let mut rng = rand::rng();
        let num_input_vars = 2;
        let num_table_vars = 5;

        let lookup_trace = IndexedLookupTrace::<FF>::random(&mut rng, num_input_vars, num_table_vars);
        let lookup_mle: IndexedLookupTraceMLE<FF> = lookup_trace.into();
        let lookup_witness = lookup_mle.compute_witness();

        let uniform = FieldUniformSampler::<FF>::new();
        let random_value = uniform.sample(&mut rng);
        let random_s_hash = uniform.sample(&mut rng);
        let lookup_helper = lookup_mle.compute_helper_functions(&lookup_witness, random_value, random_s_hash);

        let lookup_input_instance = IndexedLogUpInputInstance::<FF>::from(&lookup_mle, &lookup_helper);
        let lookup_input_info = lookup_input_instance.info();
        let lookup_table_instance = IndexedLogUpTableInstance::<FF>::from(&lookup_witness, &lookup_helper);
        let lookup_table_info = lookup_table_instance.info();

        let mut prover_transcript = Transcript::<FF>::new();
        let (proof1, _) = IndexedLogUpInputIOP::<FF>::prover(&mut prover_transcript, &lookup_input_instance);
        let (proof2, _) = IndexedLogUpTableIOP::<FF>::prover(&mut prover_transcript, &lookup_table_instance);
        let mut verifier_transcript = Transcript::<FF>::new();
        let (res1, _) = IndexedLogUpInputIOP::<FF>::verifier(&mut verifier_transcript, &lookup_input_info, &proof1);
        let (res2, _) = IndexedLogUpTableIOP::<FF>::verifier(&mut verifier_transcript, &lookup_table_info, &proof2);
        assert!(res1 && res2);
    }
}