//! This snarks prove the correctness of the accumulation iteration step in FHE bootstrapping.
//! (This imeplementation is only for testing purpose.)
//!
//! Based on the AccTrace, we need to prove the correctness of the following relations:
//! 1. inital_acc (a) is the first row of input_acc (A) => a(ry) = A(0..0, ry)
//! 2. final_acc (d) is the last row of output_acc (D) => d(ry) = D(1..1, ry)
//! 3. i-th row of input_acc is (i-1)-th row of output_acc
//!
//! For the third relation, we can rewrite it as a matrix permutation relation:
//! input_acc_permuted(denoted by A') = output_acc(D) + Zero matrix (Z)
//! => A'(rx, ry) = D(rx, ry) + Z(rx, ry)
//! where Zero matrix is a matrix where only the last row is inital_acc - final_acc
//! => Z(ry, rx) = eq(rx, 1...1) * row(ry) = eq(rx, 1...1) * (a(ry) - d(ry))
//! input_acc_permuted = permutation_matrix * input_acc
//! => A'(rx, ry) = sum_{k} P(rx, k) * A(k, ry)
use std::rc::Rc;

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field, MultilinearExtension};
use helper::{FiatShamirTranscript, Transcript};
use piop::{
    BatchedSumcheckPIOP, LagrangeKernel, SumcheckInstance, SumcheckPIOP,
    permutation::row_perm::{
        BatchedRowPermProof, RowPermInfo, RowPermInstance, RowPermPIOP,
        compute_permutation_at_point,
    },
};
use serde::Serialize;
use trace::{ConvertToEF, EvaluableTraceEF, PBSTraceMLE, pbs_trace::PBSTraceEval};

#[derive(Default)]
pub struct AccIterationSnarks<F, EF>
where
    F: Field,
    EF: AbstractExtensionField<F>,
{
    _marker_f: std::marker::PhantomData<F>,
    _marker_ef: std::marker::PhantomData<EF>,
}

#[derive(Serialize)]
pub struct AccIterationSnarksProof<EF: Field> {
    pub log_num_rows: usize,
    pub log_num_cols: usize,
    pub permutation_proof: BatchedRowPermProof<EF>,
    pub permutation_infos: Vec<RowPermInfo<EF>>,
}

impl<F, EF> AccIterationSnarks<F, EF>
where
    F: Field,
    EF: AbstractExtensionField<F> + Serialize,
{
    pub fn prove(
        &self,
        trans: &mut Transcript<EF>,
        trace_mle: &PBSTraceMLE<F>,
    ) -> AccIterationSnarksProof<EF> {
        let log_num_rows = trace_mle.log_num_round;
        let log_num_cols = trace_mle.log_coeff_count;
        let kernel_rx = LagrangeKernel::random(trans, log_num_rows);
        let kernel_ry = LagrangeKernel::random(trans, log_num_cols);

        let point = kernel_ry
            .point
            .iter()
            .chain(kernel_rx.point.iter())
            .cloned()
            .collect::<Vec<EF>>();

        let trace_eval = trace_mle.evaluate_ef(&point);
        let trace_ef = trace_mle.to_ef();
        let permutation = Rc::new(compute_permutation_at_point(
            log_num_rows,
            &trace_mle.acc_trace.permutation_info,
            &kernel_rx.point,
        ));
        AccIterationSnarks::prove_as_subprotocol(
            trans,
            &trace_ef,
            &trace_eval,
            &kernel_rx.point,
            &kernel_ry.point,
            &permutation,
        )
    }

    pub fn verify(&self, trans: &mut Transcript<EF>, proof: &AccIterationSnarksProof<EF>) -> bool {
        let point_rx = LagrangeKernel::random_point(trans, proof.log_num_rows);
        let point_ry = LagrangeKernel::random_point(trans, proof.log_num_cols);

        AccIterationSnarks::verify_as_subprotocol(trans, proof)
    }

    pub fn prove_as_subprotocol(
        trans: &mut Transcript<EF>,
        trace_ef: &PBSTraceMLE<EF>,
        trace_eval: &PBSTraceEval<EF>,
        point_rx: &[EF],
        point_ry: &[EF],
        permutation: &Rc<DenseMultilinearExtension<EF>>,
    ) -> AccIterationSnarksProof<EF> {
        // let poly = trace_mle.generate_oracle();
        // let (commitment, _commitment_state) = PCS::commit(&params.pcs_params, &poly);

        // 1. initial_acc (a) is the first row of input_acc (A) => a(ry) = A(0..0, ry)
        // 2. final_acc (d) is the last row of output_acc (D) => d(ry) = D(1..1, ry)
        // 3. i-th row of input_acc is (i-1)-th row of output_acc

        // input_acc_permuted(denoted by A') = output_acc(D) + Zero matrix (Z)
        // => A'(rx, ry) = D(rx, ry) + Z(rx, ry)
        // where Zero matrix is a matrix where only the last row is inital_acc - final_acc
        // => Z(ry, rx) = eq(rx, 1...1) * row(ry) = eq(rx, 1...1) * (a(ry) - d(ry))
        // input_acc_permuted = permutation_matrix * input_acc
        // => A'(rx, ry) = sum_{k} P(rx, k) * A(k, ry)

        let permutation_instances =
            RowPermInstance::from_subclaim(trace_ef, &trace_eval, &permutation, point_rx, point_ry);

        // Prove the permutation;
        let (piop_proof, _piop_state) = RowPermPIOP::prover_batch(trans, &permutation_instances);

        AccIterationSnarksProof {
            log_num_rows: trace_ef.log_num_round,
            log_num_cols: trace_ef.log_coeff_count,
            permutation_proof: piop_proof,
            permutation_infos: permutation_instances
                .iter()
                .map(|inst| inst.info())
                .collect::<Vec<RowPermInfo<EF>>>(),
        }
    }

    pub fn verify_as_subprotocol(
        trans: &mut Transcript<EF>,
        proof: &AccIterationSnarksProof<EF>,
    ) -> bool {
        let mut res = true;

        let (piop_res, _piop_subclaim) =
            RowPermPIOP::verifier_batch(trans, &proof.permutation_infos, &proof.permutation_proof);
        res &= piop_res;

        res
    }
}
