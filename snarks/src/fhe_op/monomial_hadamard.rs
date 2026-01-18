//! SNARKs proving FHE operation (X^{a_i} - 1) * ACC
use std::rc::Rc;

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};
use helper::utils::compute_oracle_evals;
use helper::{FiatShamirTranscript, Transcript};
use pcs::PolynomialCommitmentScheme;
use piop::hadamard::{BatchedSumHadamardProof, HadamardPIOP, SumHadamardInfo, SumHadamardInstance};
use piop::ntt::{
    BatchedNTTMatrixEvalProof, NTTMatrixEvalIOP, NTTMatrixEvalInfo, NTTMatrixEvalInstance,
};
use piop::sparse_matrix_eval::sparse_row::SparseRowEvalInstance;
use piop::{BatchedSumcheckPIOP, LagrangeKernel, SumcheckInstance};
use serde::Serialize;
use trace::{AccTraceMLE, PackableEval, PackableTrace};
use trace::{ConvertToEF, EvaluableTraceEF};

use crate::sparse_matrix_eval::SparseRowEvalSnarks;
use crate::sparse_matrix_eval::sparse_row::SparseRowEvalSnarksProof;
// Snarks for Monomial * RLWE
// Containing:
// - PIOP for Hadamard Product: reduce to evaluations on NTT Matrix
// - NTT Matrix Evaluation: reduce to evaluations on Coefficient Matrix
// - Opening Proofs for Dense Coefficient Matrix
// - Sparse Matrix Evaluation Proofs for Monomial Representation, which has Sparse Coefficient Matrix
#[derive(Default)]
pub struct MonomialHadamardSnarks<F, EF, S, PCS>
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

pub struct MonomialHadamardParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub pcs_params: PCS::Parameters,
    pub ntt_table: Rc<Vec<EF>>,
}

impl<F, EF, S, PCS> MonomialHadamardParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub fn new(code_spec: S, ntt_table: &Rc<Vec<EF>>, trace: &AccTraceMLE<F>) -> Self {
        let num_oracle_vars = trace.num_vars() + trace.log_num_oracles();
        let pcs_params = PCS::setup(num_oracle_vars, &code_spec);
        MonomialHadamardParams {
            pcs_params,
            ntt_table: ntt_table.clone(),
        }
    }
}

pub struct MonomialHadamardProof<F, EF, S, PCS>
where
    F: Field + Serialize,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub log_coeff_count: usize,
    pub log_num_oracles: usize,
    pub pcs_params: PCS::Parameters,
    pub commitment: PCS::Commitment,
    pub hadamard_info: Vec<SumHadamardInfo<EF>>,
    pub hadamard_proof: BatchedSumHadamardProof<EF>,
    pub ntt_infos: Vec<NTTMatrixEvalInfo<EF>>,
    pub ntt_proof: BatchedNTTMatrixEvalProof<EF>,
    pub eval_proof: PCS::Proof,
    pub sparse_eval_proof: SparseRowEvalSnarksProof<F, EF, S, PCS>,
}

impl<F, EF, S, PCS> MonomialHadamardSnarks<F, EF, S, PCS>
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
        trace_mle: &AccTraceMLE<F>,
        params: &MonomialHadamardParams<F, EF, S, PCS>,
    ) -> MonomialHadamardProof<F, EF, S, PCS> {
        let poly = trace_mle.generate_oracle();
        let (commitment, commitment_state) = PCS::commit(&params.pcs_params, &poly);
        trans.append_message(b"Commit Phase", &commitment);

        // Extract all Hadamard trace from the PBS trace
        let hadamard_trace = trace_mle.extract_hadamard_trace();
        let hadamard_instance = SumHadamardInstance::from(&hadamard_trace.to_ef());
        let hadamard_info = hadamard_instance
            .iter()
            .map(SumcheckInstance::info)
            .collect::<Vec<_>>();
        let (mut hadamard_piop_proof, hadamard_piop_state) =
            HadamardPIOP::prover_batch_without_evals(trans, &hadamard_instance);

        let acc_eval = trace_mle.evaluate_ef(&hadamard_piop_state.randomness);
        let hadamard_evals = acc_eval.extract_hadamard_eval();
        hadamard_piop_proof.append_eval(&hadamard_evals);
        trans.append_message(b"[PIOP Phase]", &hadamard_piop_proof);

        // Subclaim from Hadamard PIOP are evaluations on NTT Matrix
        let point_u = hadamard_piop_state.randomness[..trace_mle.log_coeff_count].to_vec();
        let point_v = hadamard_piop_state.randomness[trace_mle.log_coeff_count..].to_vec();

        // NTT Sparse Matrix Evaluation
        let monomial_poly = Rc::new(trace_mle.monomial.poly.to_ef());
        let ntt_sparse_instance = NTTMatrixEvalInstance::from_subclaim(
            &monomial_poly,
            &params.ntt_table,
            &point_u,
            &point_v,
            acc_eval.monomial.ntt,
        );

        // Normal NTT Matrix Evaluation
        let point_bit_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            trace_mle.log_num_oracles(),
        );

        let bit_poly = Rc::new(poly.to_ef());
        let poly_ntt_evals = acc_eval.pack_ntt_to_vec();
        let eval = compute_oracle_evals(&poly_ntt_evals, &point_bit_oracle);

        let mut point_v_prime = Vec::with_capacity(point_v.len() + point_bit_oracle.len());
        point_v_prime.extend_from_slice(&point_v);
        point_v_prime.extend_from_slice(&point_bit_oracle);
        let ntt_normal_instance = NTTMatrixEvalInstance::from_subclaim(
            &bit_poly,
            &params.ntt_table,
            &point_u,
            &point_v_prime,
            eval,
        );

        // Prove two NTT instances using [`BatchedSumcheckPIOP`]
        let infos = vec![ntt_sparse_instance.info(), ntt_normal_instance.info()];
        let instances = vec![ntt_sparse_instance, ntt_normal_instance];
        let (ntt_proof, ntt_state) = NTTMatrixEvalIOP::prover_batch(trans, &instances);
        trans.append_message(b"[PIOP Phase]", &ntt_proof);

        // Open the coeffcient matrix evaluation `ntt_proof.coeff_eval_at_r_v[1]` at point_r_v_prime
        let mut point_r_v_prime =
            Vec::with_capacity(ntt_state.randomness.len() + point_v_prime.len());
        point_r_v_prime.extend_from_slice(&ntt_state.randomness);
        point_r_v_prime.extend_from_slice(&point_v_prime);
        let eval_proof = PCS::open(
            &params.pcs_params,
            &commitment,
            &commitment_state,
            &point_r_v_prime,
            trans,
        );

        // Open the sparse coefficient matrix evaluation `ntt_proof.coeff_eval_at_r_v[0]` at point_r_v using SparseMatrix
        let kernel_rx = LagrangeKernel::from_point(&point_v);
        let kernel_ry = LagrangeKernel::from_point(&ntt_state.randomness);
        let sparse_matrix_eval_instance = SparseRowEvalInstance::from_subclaim::<F>(
            &trace_mle.monomial_representation,
            &kernel_rx,
            &kernel_ry,
            ntt_proof.coeff_eval_at_r_v[0],
        );

        let sparse_eval_proof = SparseRowEvalSnarks::<F, EF, S, PCS>::prove_as_subprotocol(
            trans,
            &sparse_matrix_eval_instance,
        );

        MonomialHadamardProof {
            log_coeff_count: trace_mle.log_coeff_count,
            log_num_oracles: trace_mle.log_num_oracles(),
            pcs_params: params.pcs_params.clone(),
            commitment,
            hadamard_info,
            hadamard_proof: hadamard_piop_proof,
            ntt_infos: infos,
            ntt_proof,
            eval_proof,
            sparse_eval_proof,
        }
    }

    pub fn verify(
        &self,
        trans: &mut Transcript<EF>,
        proof: &MonomialHadamardProof<F, EF, S, PCS>,
    ) -> bool {
        trans.append_message(b"Commit Phase", &proof.commitment);
        let mut res = true;

        let (hadamard_res, hadamard_subclaim) =
            HadamardPIOP::verifier_batch(trans, &proof.hadamard_info, &proof.hadamard_proof);
        res &= hadamard_res;
        trans.append_message(b"[PIOP Phase]", &proof.hadamard_proof);

        let _point_u = hadamard_subclaim.randomness[..proof.log_coeff_count].to_vec();
        let point_v = hadamard_subclaim.randomness[proof.log_coeff_count..].to_vec();
        let point_bit_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            proof.log_num_oracles,
        );

        let mut point_v_prime = Vec::with_capacity(point_v.len() + point_bit_oracle.len());
        point_v_prime.extend_from_slice(&point_v);
        point_v_prime.extend_from_slice(&point_bit_oracle);

        let (ntt_res, ntt_subclaim) =
            NTTMatrixEvalIOP::verifier_batch(trans, &proof.ntt_infos, &proof.ntt_proof);
        trans.append_message(b"[PIOP Phase]", &proof.ntt_proof);
        res &= ntt_res;

        // Verify the coeffcient matrix evaluation `ntt_proof.coeff_eval_at_r_v[1]` at point_r_v_prime
        let mut point_r_v_prime =
            Vec::with_capacity(ntt_subclaim.randomness.len() + point_v_prime.len());
        point_r_v_prime.extend_from_slice(&ntt_subclaim.randomness);
        point_r_v_prime.extend_from_slice(&point_v_prime);

        let eval_res = PCS::verify(
            &proof.pcs_params,
            &proof.commitment,
            &point_r_v_prime,
            ntt_subclaim.coeff_eval_at_r_v[1],
            &proof.eval_proof,
            trans,
        );
        res &= eval_res;

        // Verify the coeffcient matrix evaluation `ntt_proof.coeff_eval_at_r_v[0]` at point_r_v
        let sparse_eval_res = SparseRowEvalSnarks::<F, EF, S, PCS>::verify_as_subprotocol(
            trans,
            &proof.sparse_eval_proof,
            &mut None,
        );
        res &= sparse_eval_res;

        res
    }
}
