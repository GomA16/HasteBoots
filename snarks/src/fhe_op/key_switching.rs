//! External Product SNARKs proving FHE operation Mid * RGSW(s_i)
//!     where Mid = (X^{a_i} - 1) * ACC_Input
use core::time;
use std::rc::Rc;

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field, PolynomialInfo};
use bincode::config::standard;
use helper::utils::{compute_oracle_evals, eval_identity_function};
use helper::{FiatShamirTranscript, Transcript};
use pcs::PolynomialCommitmentScheme;
use piop::hadamard::{BatchedSumHadamardProof, HadamardPIOP, SumHadamardInfo, SumHadamardInstance};
use piop::lookup::small_table::{LogUpIOP, LogUpInstance, LogUpInstanceInfo, LogUpProof};
use piop::ntt::{NTTMatrixEvalIOP, NTTMatrixEvalInfo, NTTMatrixEvalInstance, NTTMatrixEvalProof};
use piop::{
    BatchedSumcheckPIOP, LagrangeKernel, SumcheckClaim, SumcheckInfo, SumcheckInstance,
    SumcheckPIOP,
};
use serde::Serialize;
use sumcheck::{MLSumcheck, Proof};
use trace::basic_ops::{SumHadamardTraceEval, SumHadamardTraceMLE};
use trace::key_switching_trace::{KeySwitchingTrace, KeySwitchingTraceMLE};
use trace::lookup_trace::small_table::LookupWitnessHelperEval;
use trace::{ConvertToEF, EvaluableTraceEF, PackableTrace};
use trace::{EvaluableTrace, PackableEval};

use crate::fhe_op::external_product::{
    ExternalProductParams, ExternalProductProof, ExternalProductSnarks,
};
use crate::fhe_op::row_permutation::{RowPermutationSignedProof, RowPermutationSignedSnarks};

#[derive(Default)]
pub struct KeySwitchingSnarks<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    _marker_f: std::marker::PhantomData<F>,
    _marker_ef: std::marker::PhantomData<EF>,
    _marker_s: std::marker::PhantomData<S>,
    _marker_pcs: std::marker::PhantomData<PCS>,
}

pub struct KeySwitchingParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub external_product_params: ExternalProductParams<F, EF, S, PCS>,
}

impl<F, EF, S, PCS> KeySwitchingParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub fn new(
        code_spec: S,
        ntt_table: Vec<F>,
        blk_size: usize,
        basis: usize,
        trace: &KeySwitchingTraceMLE<F>,
    ) -> Self {
        KeySwitchingParams {
            external_product_params: ExternalProductParams::new(
                code_spec,
                ntt_table,
                blk_size,
                basis,
                &trace.hadamard_trace,
            ),
        }
    }
}

#[derive(Serialize)]
pub struct KeySwitchingProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub log_lwe_dim: usize,
    pub log_rlwe_dim: usize,
    pub external_product_proof: ExternalProductProof<F, EF, S, PCS>,
    pub permutation_proof: Option<RowPermutationSignedProof<F, EF, S, PCS>>,
}

impl<F, EF, S, PCS> KeySwitchingSnarks<F, EF, S, PCS>
where
    F: Field + Serialize,
    EF: AbstractExtensionField<F> + Serialize,
    S: Clone,
    PCS: PolynomialCommitmentScheme<
            F,
            EF,
            S,
            Polynomial = DenseMultilinearExtension<F>,
            EFPolynomial = DenseMultilinearExtension<EF>,
            Point = EF,
        >,
{
    pub fn prove(
        &self,
        trans: &mut Transcript<EF>,
        trace_mle: &KeySwitchingTraceMLE<F>,
        params: &KeySwitchingParams<F, EF, S, PCS>,
    ) -> KeySwitchingProof<F, EF, S, PCS> {
        KeySwitchingSnarks::prove_as_subprotocol(trans, trace_mle, params)
    }

    pub fn verify(
        &self,
        trans: &mut Transcript<EF>,
        proof: &KeySwitchingProof<F, EF, S, PCS>,
    ) -> bool {
        KeySwitchingSnarks::verify_as_subprotocol(trans, proof)
    }

    pub fn prove_as_subprotocol(
        trans: &mut Transcript<EF>,
        trace_mle: &KeySwitchingTraceMLE<F>,
        params: &KeySwitchingParams<F, EF, S, PCS>,
    ) -> KeySwitchingProof<F, EF, S, PCS> {
        let external_product_proof = ExternalProductSnarks::prove_as_subprotocol(
            trans,
            &trace_mle.hadamard_trace,
            &params.external_product_params,
        );

        let time = std::time::Instant::now();
        let mut permutation_proof: Option<RowPermutationSignedProof<F, EF, S, PCS>> = None;
        if trace_mle.permutation_trace.is_some() {
            permutation_proof.replace(RowPermutationSignedSnarks::prove_as_subprotocol(
                trans,
                trace_mle.permutation_trace.as_ref().unwrap(),
            ));
        }
        println!("[PermutationTrace] Proof time: {:?}", time.elapsed());

        KeySwitchingProof {
            log_lwe_dim: trace_mle.log_lwe_dim,
            log_rlwe_dim: trace_mle.log_rlwe_dim,
            external_product_proof,
            permutation_proof,
        }
    }

    pub fn verify_as_subprotocol(
        trans: &mut Transcript<EF>,
        proof: &KeySwitchingProof<F, EF, S, PCS>,
    ) -> bool {
        let mut res = true;
        res &= ExternalProductSnarks::verify_as_subprotocol(trans, &proof.external_product_proof);

        let time = std::time::Instant::now();
        if proof.permutation_proof.is_some() {
            res &= RowPermutationSignedSnarks::verify_as_subprotocol(
                trans,
                proof.permutation_proof.as_ref().unwrap(),
            );
        }
        println!("[PermutationTrace] Verification time: {:?}", time.elapsed());
        res
    }
}
