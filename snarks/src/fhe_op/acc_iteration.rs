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

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};
use bincode::config::standard;
use helper::Transcript;
use pcs::PolynomialCommitmentScheme;
use piop::{
    BatchedSumcheckPIOP, LagrangeKernel, SumcheckInstance,
    permutation::row_perm::{BatchedRowPermProof, RowPermInfo, RowPermInstance, RowPermPIOP},
};
use serde::Serialize;
use trace::{
    BlindRotationTraceMLE, ConvertToEF, EvaluableTraceEF,
    blind_rotation_trace::BlindRotationTraceEval,
    lookup_trace::indexed_table::IndexedLookupTraceMLE,
};

use crate::lookup::indexed_table::{IndexedLogUpSnarks, IndexedLogUpSnarksProof};

#[derive(Default)]
pub struct AccIterationSnarks<F, EF, S, PCS>
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
pub struct AccIterationSnarksProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub log_num_rows: usize,
    pub log_num_cols: usize,
    pub lookup_proof: IndexedLogUpSnarksProof<F, EF, S, PCS>,
    pub permutation_proof: BatchedRowPermProof<EF>,
    pub permutation_infos: Vec<RowPermInfo<EF>>,
}

impl<F, EF, S, PCS> AccIterationSnarksProof<F, EF, S, PCS>
where
    F: Field + Serialize,
    EF: AbstractExtensionField<F> + Serialize,
    S: Clone + Serialize,
    PCS: PolynomialCommitmentScheme<F, EF, S> + Serialize,
{
    pub fn piop_proof_len(&self) -> usize {
        bincode::serde::encode_to_vec(&self.permutation_infos, standard())
            .unwrap()
            .len()
            + bincode::serde::encode_to_vec(&self.permutation_proof, standard())
                .unwrap()
                .len()
            + self.lookup_proof.piop_proof_len()
    }

    pub fn pcs_proof_len(&self) -> usize {
        self.lookup_proof.pcs_proof_len()
    }
}

impl<F, EF, S, PCS> AccIterationSnarks<F, EF, S, PCS>
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
        trace_mle: &BlindRotationTraceMLE<F>,
    ) -> AccIterationSnarksProof<F, EF, S, PCS> {
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
        let indexed_lookup_trace = trace_ef
            .acc_trace
            .permutation_info
            .extract_indexed_lookup_trace(&kernel_rx.point);

        AccIterationSnarks::<F, EF, S, PCS>::prove_as_subprotocol(
            trans,
            &trace_ef,
            &trace_eval,
            &kernel_rx.point,
            &kernel_ry.point,
            &indexed_lookup_trace,
        )
    }

    pub fn verify(
        &self,
        trans: &mut Transcript<EF>,
        proof: &AccIterationSnarksProof<F, EF, S, PCS>,
    ) -> bool {
        let _point_rx = LagrangeKernel::random_point(trans, proof.log_num_rows);
        let _point_ry = LagrangeKernel::random_point(trans, proof.log_num_cols);

        AccIterationSnarks::<F, EF, S, PCS>::verify_as_subprotocol(trans, proof, &mut None)
    }

    pub fn prove_as_subprotocol(
        trans: &mut Transcript<EF>,
        trace_ef: &BlindRotationTraceMLE<EF>,
        trace_eval: &BlindRotationTraceEval<EF>,
        point_rx: &[EF],
        point_ry: &[EF],
        indexed_lookup_mle: &IndexedLookupTraceMLE<EF>,
    ) -> AccIterationSnarksProof<F, EF, S, PCS> {
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

        // Prove P(rx, k) = eq(rx, perm_inver(k))
        #[cfg(feature = "br-profiling")]
        let lookup_scope = crate::profiling::scope(crate::profiling::BrPhase::AccumulatorLookup);
        let lookup_proof =
            IndexedLogUpSnarks::<F, EF, S, PCS>::prove_as_subprotocol(trans, indexed_lookup_mle);
        #[cfg(feature = "br-profiling")]
        drop(lookup_scope);

        // Prove the permutation;
        let (permutation_instances, piop_proof) = {
            #[cfg(feature = "br-profiling")]
            let _scope =
                crate::profiling::scope(crate::profiling::BrPhase::AccumulatorPermutation);
            let permutation_instances = RowPermInstance::from_subclaim(
                trace_ef,
                trace_eval,
                indexed_lookup_mle,
                point_rx,
                point_ry,
            );
            let (piop_proof, _piop_state) =
                RowPermPIOP::prover_batch(trans, &permutation_instances);
            #[cfg(feature = "br-profiling")]
            {
                crate::profiling::add_work(
                    crate::profiling::BrPhase::AccumulatorPermutation,
                    "instances",
                    permutation_instances.len() as u64,
                );
                crate::profiling::add_work(
                    crate::profiling::BrPhase::AccumulatorPermutation,
                    "variables_rounds",
                    trace_ef.log_num_round as u64,
                );
            }
            (permutation_instances, piop_proof)
        };

        AccIterationSnarksProof {
            log_num_rows: trace_ef.log_num_round,
            log_num_cols: trace_ef.log_coeff_count,
            lookup_proof,
            permutation_proof: piop_proof,
            permutation_infos: permutation_instances
                .iter()
                .map(|inst| inst.info())
                .collect::<Vec<RowPermInfo<EF>>>(),
        }
    }

    pub fn verify_as_subprotocol(
        trans: &mut Transcript<EF>,
        proof: &AccIterationSnarksProof<F, EF, S, PCS>,
        statistics: &mut Option<&mut crate::SnarkStatistics>,
    ) -> bool {
        let mut res = true;

        let res_lookup = IndexedLogUpSnarks::<F, EF, S, PCS>::verify_as_subprotocol(
            trans,
            &proof.lookup_proof,
            statistics,
        );
        res &= res_lookup;

        let (piop_res, _piop_subclaim) =
            RowPermPIOP::verifier_batch(trans, &proof.permutation_infos, &proof.permutation_proof);
        res &= piop_res;

        res
    }
}
