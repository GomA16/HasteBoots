use algebra::{DenseMultilinearExtension, Field, ListOfProductsOfPolynomials, PolynomialInfo};
use helper::Transcript;
use serde::Serialize;
use std::{iter::Sum, rc::Rc};
use sumcheck::Proof;
use sumcheck::{prover::ProverState, verifier::SubClaim};
use trace::{LookupTraceMLE, LookupWitness, LookupWitnessHelper};

use crate::{LagrangeKernel, SumcheckClaim, SumcheckPIOP};

pub struct LogUpInstance<F: Field> {
    pub num_vars: usize,
    pub lookup_trace: LookupTraceMLE<F>,
    pub witness: LookupWitness<F>,
    pub helper: LookupWitnessHelper<F>,
}

#[derive(Serialize)]
pub struct LogUpInstanceEvalInfo {
    pub num_vars: usize,
    pub block_size: usize,
    pub num_blocks: usize,
    pub num_columns: usize,
}

pub struct LogUpProof<F: Field> {
    pub poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<F>,
    pub lookup_input_at_r: Vec<F>,
    // evaluations of (table(r), multiplicity(r))
    pub witness_at_r: (F, F),
    pub helper_at_r: Vec<F>,
}

pub struct LogUpProverState<F: Field> {
    pub random_value: F,
    pub random_sumcheck: Vec<F>,
    pub point_r: Vec<F>,
}

pub struct LogUpVerifierSubclaim<F: Field> {
    pub random_value: F,
    pub random_sumcheck: Vec<F>,
    pub point_r: Vec<F>,
}

impl<F: Field> LogUpInstance<F> {
    pub fn from(
        trace: &LookupTraceMLE<F>,
        witness: &LookupWitness<F>,
        helper: &LookupWitnessHelper<F>,
    ) -> Self {
        Self {
            num_vars: trace.num_vars,
            lookup_trace: trace.clone(),
            witness: witness.clone(),
            helper: helper.clone(),
        }
    }

    pub fn sumcheck_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn info(&self) -> LogUpInstanceEvalInfo {
        LogUpInstanceEvalInfo {
            num_vars: self.num_vars,
            block_size: self.helper.block_size,
            num_blocks: self.helper.num_blocks,
            num_columns: self.lookup_trace.vec_input.len(),
        }
    }

    // \sum h_1 + ... + h_k = 0
    pub fn add_helper_into_sumcheck(&self, claim: &mut SumcheckClaim<F>, randomness: F) {
        let add_assign = |acc: &mut [F], vec: &[F]| {
            for (a, b) in acc.iter_mut().zip(vec.iter()) {
                *a += *b;
            }
        };

        let h_sum = self.helper.helper_functions.iter().fold(
            vec![F::zero(); 1 << self.num_vars],
            |mut acc, hf| {
                add_assign(&mut acc, hf.as_slice());
                acc
            },
        );
        let h_sum = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            self.num_vars,
            h_sum,
        ));

        claim.poly_mut().add_product(vec![h_sum], randomness);
    }

    pub fn add_helper_identity_into_sumcheck(
        &self,
        idx: usize,
        claim: &mut SumcheckClaim<F>,
        randomness: F,
        kernel: &LagrangeKernel<F>,
    ) {
        assert_eq!(idx, 0);
        assert!(self.helper.block_size >= 1);

        let idx_start = idx * self.helper.block_size;
        let idx_end = ((idx + 1) * self.helper.block_size).min(self.helper.phi_functions.len());

        let blk_len = idx_end - idx_start;

        let phi_block = self.helper.phi_functions[idx_start..idx_end]
            .iter()
            .map(|phi| Rc::clone(phi))
            .collect::<Vec<_>>();


        // randomness * h * L * \prod phi_i
        let mut prod: Vec<Rc<DenseMultilinearExtension<F>>> = Vec::with_capacity(blk_len + 2);
        prod.extend_from_slice(&phi_block);
        prod.push(Rc::clone(&kernel.eq_at_point));
        prod.push(Rc::clone(&self.helper.helper_functions[idx]));
        claim.poly_mut().add_product(prod, randomness);

        for off_idx in 0..blk_len {
            match idx {
                0 => {
                    // - randomness * L * m * \prod phi_i / phi_{off_idx}
                    let mut prod_i = Vec::with_capacity(blk_len + 1);
                    prod_i.extend_from_slice(&phi_block);
                    prod_i[off_idx] = Rc::clone(&self.witness.multiplicity);
                    claim.poly_mut().add_product(prod_i, -randomness);
                }
                _ => {
                    // randomness * L * \prod phi_i / phi_{off_idx}
                    let mut prod_i = Vec::with_capacity(blk_len);
                    prod_i.extend_from_slice(&phi_block);
                    prod_i[off_idx] = Rc::clone(&kernel.eq_at_point);
                    claim.poly_mut().add_product(prod_i, randomness);
                }
            }
        }
    }
}

impl<F: Field> LogUpInstance<F> {
    pub fn num_sumchecks(&self) -> usize {
        self.helper.num_blocks + 1
    }
}

impl<F: Field + Serialize> SumcheckPIOP<F> for LogUpIOP<F> {
    type Instance = LogUpInstance<F>;
    type Info = LogUpInstanceEvalInfo;
    type Proof = LogUpProof<F>;
    type ProverState = LogUpProverState<F>;
    type VerifierSubclaim = LogUpVerifierSubclaim<F>;
    type FSTranscript = Transcript<F>;

    fn prover(
        trans: &mut Self::FSTranscript,
        instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState) {
        let statement = instance.info();
        trans.append_message(b"[Lookup Statement]", &statement);

        let mut sumcheck_claim = SumcheckClaim::new(instance.sumcheck_num_vars());
    }

    fn verifier(
        trans: &mut Self::FSTranscript,
        info: &Self::Info,
        proof: &Self::Proof,
    ) -> (bool, Self::VerifierSubclaim) {
        unimplemented!()
    }

    fn prover_batch_sumcheck(
        instance: &Self::Instance,
        claim: &mut crate::SumcheckClaim<F>,
        randomness: &[F],
        kernel: Option<&LagrangeKernel<F>>,
    ) -> Self::ProverState {
        
    }

    fn verifier_compute_subclaim(
        proof: &Self::Proof,
        subclaim: &mut SubClaim<F>,
        randomness: &[F],
        kernel: Option<&LagrangeKernel<F>>,
    ) {
        unimplemented!()
    }
}
