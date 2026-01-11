//! External Product SNARKs proving FHE operation Mid * RGSW(s_i)
//!     where Mid = (X^{a_i} - 1) * ACC_Input
use core::time;

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};
use bincode::config::standard;
use helper::Transcript;
use log::info;
use pcs::PolynomialCommitmentScheme;
use serde::Serialize;
use trace::key_switching_trace::KeySwitchingTraceMLE;

use crate::fhe_op::decomposition::{
    DecompositionParams, DecompositionSnarks, DecompositionSnarksProof,
};
use crate::fhe_op::hadmard_product::{
    HadamardProductParams, HadamardProductProof, HadamardProductSnarks,
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
    S: Clone + Serialize,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub hadamard_product_params: HadamardProductParams<F, EF, S, PCS>,
    pub code_spec: S,
}

impl<F, EF, S, PCS> KeySwitchingParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone + Serialize,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub fn new(code_spec: S, ntt_table: Vec<F>, trace: &KeySwitchingTraceMLE<F>) -> Self {
        KeySwitchingParams {
            hadamard_product_params: HadamardProductParams::new(
                code_spec.clone(),
                ntt_table,
                &trace.hadamard_trace,
            ),
            code_spec,
        }
    }
}

#[derive(Serialize)]
pub struct KeySwitchingProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone + Serialize,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub log_lwe_dim: usize,
    pub log_rlwe_dim: usize,
    pub hadamard_proof: HadamardProductProof<F, EF, S, PCS>,
    pub decomp_proof: DecompositionSnarksProof<F, EF, S, PCS>,
    pub permutation_proof: Option<RowPermutationSignedProof<F, EF, S, PCS>>,
}

impl<F, EF, S, PCS> KeySwitchingProof<F, EF, S, PCS>
where
    F: Field + Serialize,
    EF: AbstractExtensionField<F> + Serialize,
    S: Clone + Serialize,
    PCS: PolynomialCommitmentScheme<
            F,
            EF,
            S,
            Polynomial = DenseMultilinearExtension<F>,
            EFPolynomial = DenseMultilinearExtension<EF>,
            Point = EF,
        > + Serialize,
{
    pub fn piop_proof_len(&self) -> usize {
        let mut len = if self.permutation_proof.is_some() {
            bincode::serde::encode_to_vec(&self.permutation_proof.as_ref().unwrap(), standard())
                .unwrap()
                .len()
        } else {
            0
        };
        len += self.hadamard_proof.piop_proof_len() + self.decomp_proof.piop_proof_len();
        len
    }

    pub fn pcs_proof_len(&self) -> usize {
        let mut len = if self.permutation_proof.is_some() {
            bincode::serde::encode_to_vec(&self.permutation_proof.as_ref().unwrap(), standard())
                .unwrap()
                .len()
        } else {
            0
        };
        len += self.hadamard_proof.pcs_proof_len() + self.decomp_proof.pcs_proof_len();
        len
    }
}

impl<F, EF, S, PCS> KeySwitchingSnarks<F, EF, S, PCS>
where
    F: Field + Serialize,
    EF: AbstractExtensionField<F> + Serialize,
    S: Clone + Serialize,
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
        statistics: &mut Option<&mut crate::SnarkStatistics>,
    ) -> KeySwitchingProof<F, EF, S, PCS> {
        KeySwitchingSnarks::prove_as_subprotocol(trans, trace_mle, params, statistics)
    }

    pub fn verify(
        &self,
        trans: &mut Transcript<EF>,
        proof: &KeySwitchingProof<F, EF, S, PCS>,
        statistics: &mut Option<&mut crate::SnarkStatistics>,
    ) -> bool {
        KeySwitchingSnarks::verify_as_subprotocol(trans, proof, statistics)
    }

    pub fn prove_as_subprotocol(
        trans: &mut Transcript<EF>,
        trace_mle: &KeySwitchingTraceMLE<F>,
        params: &KeySwitchingParams<F, EF, S, PCS>,
        statistics: &mut Option<&mut crate::SnarkStatistics>,
    ) -> KeySwitchingProof<F, EF, S, PCS> {
        info!("[P] Start Key Switching Proof Generation...");
        let hadamard_proof = HadamardProductSnarks::prove_as_subprotocol(
            trans,
            &trace_mle.hadamard_trace,
            &params.hadamard_product_params,
            statistics,
        );

        let decomp_instances = trace_mle.extract_decomposition_traces();
        let decomp_params =
            DecompositionParams::new(params.code_spec.clone(), &trace_mle.lt_tables);
        let decomp_proof = DecompositionSnarks::<F, EF, S, PCS>::prove_as_subprotocol(
            trans,
            &decomp_instances,
            &decomp_params,
            statistics,
        );

        let mut permutation_proof: Option<RowPermutationSignedProof<F, EF, S, PCS>> = None;
        if trace_mle.permutation_trace.is_some() {
            permutation_proof.replace(RowPermutationSignedSnarks::prove_as_subprotocol(
                trans,
                trace_mle.permutation_trace.as_ref().unwrap(),
            ));
        }

        KeySwitchingProof {
            log_lwe_dim: trace_mle.log_lwe_dim,
            log_rlwe_dim: trace_mle.log_rlwe_dim,
            hadamard_proof,
            permutation_proof,
            decomp_proof,
        }
    }

    pub fn verify_as_subprotocol(
        trans: &mut Transcript<EF>,
        proof: &KeySwitchingProof<F, EF, S, PCS>,
        statistics: &mut Option<&mut crate::SnarkStatistics>,
    ) -> bool {
        let mut res = true;
        res &=
            HadamardProductSnarks::verify_as_subprotocol(trans, &proof.hadamard_proof, statistics);
        assert!(res, "Hadamard Product verification failed.");

        let decomp_res = DecompositionSnarks::<F, EF, S, PCS>::verify_as_subprotocol(
            trans,
            &proof.decomp_proof,
            statistics,
        );
        res &= decomp_res;
        assert!(res, "Decomposition verification failed.");

        if proof.permutation_proof.is_some() {
            res &= RowPermutationSignedSnarks::verify_as_subprotocol(
                trans,
                proof.permutation_proof.as_ref().unwrap(),
                statistics,
            );
            assert!(res, "Row Permutation verification failed.");
        }
        res
    }
}
