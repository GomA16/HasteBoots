//! This snarks prove the correctness of the row permutation with sign flip.
//!
//! Besides proving it with RowPermPIOP, it reduces to the evaluation on the permutation
//! matrix and input matrix.
//!
//! We use SparseRowEvalSnarks to prove the evaluation on the permutation matrix.
//! The input and output matrices are committed in the main snarks protocol, so their
//! evaluations can be directly verified.
use std::rc::Rc;

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field, MultilinearExtension};
use helper::{FiatShamirTranscript, Transcript};
use pcs::PolynomialCommitmentScheme;
use piop::{
    BatchedSumcheckPIOP, LagrangeKernel, SumcheckInstance, SumcheckPIOP,
    permutation::row_perm::{
        BatchedRowPermProof, RowPermInfo, RowPermInstance, RowPermPIOP, RowPermProof,
    },
    sparse_matrix_eval::sparse_row::SparseRowEvalInstance,
};
use serde::Serialize;
use trace::{BlindRotationTraceMLE, ConvertToEF, EvaluableTraceEF, basic_ops::RowPermTraceMLE};

use crate::sparse_matrix_eval::{SparseRowEvalSnarks, sparse_row::SparseRowEvalSnarksProof};

#[derive(Default)]
pub struct RowPermutationSignedSnarks<F, EF, S, PCS>
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

#[derive(Serialize)]
pub struct RowPermutationSignedProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub log_num_rows: usize,
    pub log_num_cols: usize,
    pub permutation_info: RowPermInfo<EF>,
    pub permutation_proof: RowPermProof<EF>,
    #[serde(skip)]
    pub input_commitment: PCS::Polynomial,
    #[serde(skip)]
    pub output_commitment: PCS::Polynomial,
    pub permutation_eval_proof: SparseRowEvalSnarksProof<F, EF, S, PCS>,
    pub output_ry_rx: EF,
    pub input_ry_r: EF,
}

impl<F, EF, S, PCS> RowPermutationSignedSnarks<F, EF, S, PCS>
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
        trace_mle: &RowPermTraceMLE<F>,
    ) -> RowPermutationSignedProof<F, EF, S, PCS> {
        RowPermutationSignedSnarks::<F, EF, S, PCS>::prove_as_subprotocol(trans, trace_mle)
    }

    pub fn verify(
        &self,
        trans: &mut Transcript<EF>,
        proof: &RowPermutationSignedProof<F, EF, S, PCS>,
    ) -> bool {
        RowPermutationSignedSnarks::<F, EF, S, PCS>::verify_as_subprotocol(trans, proof)
    }

    pub fn prove_as_subprotocol(
        trans: &mut Transcript<EF>,
        trace_mle: &RowPermTraceMLE<F>,
        // permutaion_info: &PermutationSignedInfo<F>,
    ) -> RowPermutationSignedProof<F, EF, S, PCS> {
        trans.append_message(b"Commit Phase", trace_mle.input.as_ref());
        trans.append_message(b"Commit Phase", trace_mle.output.as_ref());

        let log_num_rows = trace_mle.log_num_rows;
        let log_num_cols = trace_mle.log_num_cols;
        let kernel_rx = LagrangeKernel::random(trans, log_num_rows);
        let point_ry = LagrangeKernel::random_point(trans, log_num_cols);

        let trace_ef = trace_mle.to_ef();
        let row_per_instance =
            RowPermInstance::from_permutation_trace(&trace_ef, &kernel_rx.point, &point_ry);
        let (piop_proof, piop_state) = RowPermPIOP::prover(trans, &row_per_instance);

        // prove the evaluation of the permutation matrix using SparseRowEvalSnarks
        let kernel_r = LagrangeKernel::from_point(&piop_state.randomness);
        let eval = row_per_instance.perm_rx.evaluate(&kernel_r.point);

        let sparse_instance = SparseRowEvalInstance::<EF>::from_subclaim_permutation(
            &trace_ef.permutation_info,
            &kernel_rx,
            &kernel_r,
            eval,
        );
        let sparse_proof =
            SparseRowEvalSnarks::<F, EF, S, PCS>::prove_as_subprotocol(trans, &sparse_instance);

        // other evaluation proof
        let output_ry_rx = row_per_instance.output_ry_rx;
        let input_ry_r = piop_proof.input_at_ry_r;
        RowPermutationSignedProof {
            log_num_rows,
            log_num_cols,
            permutation_info: row_per_instance.info(),
            permutation_proof: piop_proof,
            input_commitment: trace_mle.input.as_ref().clone(),
            output_commitment: trace_mle.output.as_ref().clone(),
            permutation_eval_proof: sparse_proof,
            output_ry_rx,
            input_ry_r,
        }
    }

    pub fn verify_as_subprotocol(
        trans: &mut Transcript<EF>,
        proof: &RowPermutationSignedProof<F, EF, S, PCS>,
    ) -> bool {
        trans.append_message(b"Commit Phase", &proof.input_commitment);
        trans.append_message(b"Commit Phase", &proof.output_commitment);

        let point_rx = LagrangeKernel::random_point(trans, proof.log_num_rows);
        let point_ry = LagrangeKernel::random_point(trans, proof.log_num_cols);

        let mut res = true;
        let (permutation_res, permutation_subclaim) =
            RowPermPIOP::verifier(trans, &proof.permutation_info, &proof.permutation_proof);
        res &= permutation_res;

        let sparse_res = SparseRowEvalSnarks::<F, EF, S, PCS>::verify_as_subprotocol(
            trans,
            &proof.permutation_eval_proof,
        );
        res &= sparse_res;

        let point_ry_rx = point_ry
            .iter()
            .cloned()
            .chain(point_rx.iter().cloned())
            .collect::<Vec<EF>>();
        res &= proof.output_ry_rx == proof.output_commitment.evaluate_ext(&point_ry_rx);
        let point_ry_r = point_ry
            .iter()
            .cloned()
            .chain(permutation_subclaim.randomness.iter().cloned())
            .collect::<Vec<EF>>();
        res &= proof.input_ry_r == proof.input_commitment.evaluate_ext(&point_ry_r);

        res
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use algebra::{BabyBear, BabyBearExetension};
    use helper::Transcript;
    use pcs::{
        multilinear::BrakedownPCS,
        utils::code::{ExpanderCode, ExpanderCodeSpec},
    };
    use trace::basic_ops::{RowPermTrace, row_perm_trace::PermutationInfo};

    type FF = BabyBear;
    type EF = BabyBearExetension;
    type Hash = sha2::Sha256;

    #[test]
    fn test_row_permutation_signed() {
        let log_num_rows = 10;
        let log_num_cols = 0;
        let log_blk_size = 9;

        let rng = &mut rand::rng();
        let permutation_info =
            PermutationInfo::new_ks_permutation(1 << log_num_rows, 1 << log_blk_size);
        let trace = RowPermTrace::<FF>::random_ks_permutation(
            rng,
            log_num_rows,
            log_num_cols,
            log_blk_size,
        );

        let test_output =
            permutation_info.permute_row_wise_signed(log_num_rows, log_num_cols, &trace.input);
        assert_eq!(test_output, trace.output);

        let trace_mle: RowPermTraceMLE<FF> = trace.into();
        let snarks = RowPermutationSignedSnarks::<
            FF,
            EF,
            ExpanderCodeSpec,
            BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
        >::default();

        let prover_trans = &mut Transcript::<EF>::default();
        let proof = snarks.prove(prover_trans, &trace_mle);
        let verifier_trans = &mut Transcript::<EF>::default();
        let res = snarks.verify(verifier_trans, &proof);
        assert!(res);
    }
}
