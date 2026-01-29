//! Blind Rotation SNARKs proving FHE operation (X^{a_i} - 1) * ACC * RGSW(s_i)
//!
//! This mainly consists of two parts:
//! 1. Proving the correctness of Mid = (X^{a_i} - 1) * ACC_Input
//!     This is separately proven in `monomial_hadamard.rs`
//! 2. Proving the correctness of ACC_Output = Mid * RGSW(s_i)
//!     This is separately proven in `external_product.rs`

use std::rc::Rc;

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field, PolynomialInfo};
use bincode::config::standard;
use helper::utils::{compute_oracle_evals, eval_identity_function};
use helper::{FiatShamirTranscript, Transcript};
use log::info;
use pcs::PolynomialCommitmentScheme;
use piop::hadamard::{BatchedSumHadamardProof, HadamardPIOP, SumHadamardInfo, SumHadamardInstance};
use piop::ntt::{
    BatchedNTTMatrixEvalProof, NTTMatrixEvalIOP, NTTMatrixEvalInfo, NTTMatrixEvalInstance,
};
use piop::sparse_matrix_eval::sparse_row::SparseRowEvalInstance;
use piop::{BatchedSumcheckPIOP, LagrangeKernel, SumcheckClaim, SumcheckInfo, SumcheckInstance};
use serde::Serialize;
use sumcheck::{MLSumcheck, Proof};
use trace::blind_rotation_trace::BlindRotationTraceEval;
use trace::{
    BlindRotationTrace, BlindRotationTraceMLE, ConvertToEF, EvaluableTraceEF, PackableTrace,
};
use trace::{SeparatelyPackableEval, SeparatelyPackableTrace};

use crate::fhe_op::acc_iteration::{AccIterationSnarks, AccIterationSnarksProof};
use crate::fhe_op::decomposition::{
    DecompositionParams, DecompositionSnarks, DecompositionSnarksProof,
};
use crate::sparse_matrix_eval::SparseRowEvalSnarks;
use crate::sparse_matrix_eval::sparse_row::SparseRowEvalSnarksProof;

#[derive(Default)]
pub struct BlindRotationSnarks<F, EF, S, PCS>
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

pub struct BlindRotationParams<'a, F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub code_spec: S,
    pub pcs_params: PCS::Parameters,
    pub key_commit: &'a KeyCommitment<F, EF, S, PCS>,
    pub ntt_table: Rc<Vec<EF>>,
}

pub struct KeyCommitment<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub commitment: PCS::Commitment,
    pub commitment_state: PCS::CommitmentState,
    pub params: PCS::Parameters,
    pub poly: Rc<DenseMultilinearExtension<EF>>,
}

impl<F, EF, S, PCS> KeyCommitment<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S, Polynomial = DenseMultilinearExtension<F>>,
{
    pub fn new(code_spec: &S, trace: &BlindRotationTrace<F>) -> Self {
        let num_vars = trace.num_vars() + trace.log_num_key_oracles();
        info!(
            "[Preprocessing] Commit to a key oracle of {} variables",
            num_vars
        );
        let time = std::time::Instant::now();
        let params = PCS::setup(num_vars, code_spec);
        let poly = trace.generate_key_oracle();
        let (commitment, commitment_state) = PCS::commit(&params, &poly);
        let poly = Rc::new(poly.to_ef());
        info!(
            "[Preprocessing] Key Commitment done in {:?}",
            time.elapsed()
        );
        KeyCommitment {
            commitment,
            commitment_state,
            params,
            poly,
        }
    }
}

impl<'a, F, EF, S, PCS> BlindRotationParams<'a, F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub fn new(
        code_spec: S,
        ntt_table: Vec<F>,
        trace: &BlindRotationTrace<F>,
        key_commit: &'a KeyCommitment<F, EF, S, PCS>,
    ) -> Self {
        let oracle_num_vars = trace.num_vars() + trace.log_num_bit_oracles();
        let pcs_params = PCS::setup(oracle_num_vars, &code_spec);

        BlindRotationParams {
            code_spec,
            pcs_params,
            key_commit,
            ntt_table: Rc::new(ntt_table.to_ef()),
        }
    }
}

#[derive(Serialize)]
pub struct BlindRotationProof<F, EF, S, PCS>
where
    F: Field + Serialize,
    EF: AbstractExtensionField<F>,
    S: Clone + Serialize,
    PCS: PolynomialCommitmentScheme<F, EF, S> + Serialize,
{
    pub log_coeff_count: usize,
    pub log_num_bit_oracle: usize,
    pub log_num_key_oracle: usize,
    pub pcs_params: PCS::Parameters,
    pub commitment: PCS::Commitment,
    pub sumcheck_poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<EF>,
    pub hadamard_info: Vec<SumHadamardInfo<EF>>,
    pub hadamard_proof: BatchedSumHadamardProof<EF>,
    pub ntt_infos: Vec<NTTMatrixEvalInfo<EF>>,
    pub ntt_proof: BatchedNTTMatrixEvalProof<EF>,
    pub acc_iteration_proof: AccIterationSnarksProof<F, EF, S, PCS>,
    pub decomp_proof: DecompositionSnarksProof<F, EF, S, PCS>,

    pub eval_proof: PCS::Proof,
    pub eval_proof_key: PCS::Proof,
    pub sparse_eval_proof: SparseRowEvalSnarksProof<F, EF, S, PCS>,

    // Redudant fields for ease of implementation
    #[serde(skip)]
    pub trace_evals: BlindRotationTraceEval<EF>,
    pub key_pcs_params: PCS::Parameters,
    pub key_commitment: PCS::Commitment,
}

impl<F, EF, S, PCS> BlindRotationProof<F, EF, S, PCS>
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
        bincode::serde::encode_to_vec(&self.sumcheck_poly_info, standard())
            .unwrap()
            .len()
            + bincode::serde::encode_to_vec(&self.sumcheck_proof, standard())
                .unwrap()
                .len()
            + bincode::serde::encode_to_vec(&self.hadamard_info, standard())
                .unwrap()
                .len()
            + bincode::serde::encode_to_vec(&self.hadamard_proof, standard())
                .unwrap()
                .len()
            + bincode::serde::encode_to_vec(&self.ntt_infos, standard())
                .unwrap()
                .len()
            + bincode::serde::encode_to_vec(&self.ntt_proof, standard())
                .unwrap()
                .len()
            + self.acc_iteration_proof.piop_proof_len()
            + self.decomp_proof.piop_proof_len()
            + self.sparse_eval_proof.piop_proof_len()
    }

    pub fn pcs_proof_len(&self) -> usize {
        bincode::serde::encode_to_vec(&self.eval_proof, standard())
            .unwrap()
            .len()
            + bincode::serde::encode_to_vec(&self.commitment, standard())
                .unwrap()
                .len()
            + self.acc_iteration_proof.pcs_proof_len()
            + self.decomp_proof.pcs_proof_len()
            + self.sparse_eval_proof.pcs_proof_len()
    }
}

impl<F, EF, S, PCS> BlindRotationSnarks<F, EF, S, PCS>
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
    pub fn prove(
        &self,
        trans: &mut Transcript<EF>,
        blind_rotation_trace: BlindRotationTrace<F>,
        params: &BlindRotationParams<F, EF, S, PCS>,
        statistics: &mut Option<&mut crate::SnarkStatistics>,
    ) -> BlindRotationProof<F, EF, S, PCS> {
        info!("[P] Start Blind Rotation Proof Generation...");
        // [Commit Phase] commit to the trace polynomial
        let bit_poly = blind_rotation_trace.generate_bit_oracle();
        let blind_rotation_trace_mle = BlindRotationTraceMLE::from(blind_rotation_trace);
        let pcs_commit_time = std::time::Instant::now();
        let (commitment, commitment_state) = PCS::commit(&params.pcs_params, &bit_poly);
        trans.append_message(b"[Commit Phase]", &commitment);
        info!(
            // These elements are doubly committed in snarks proving decompositon (batched indexed log-up)
            "[P]-[PCS] (duplicated) Committing to a polynomial of {} variables in {:?}",
            bit_poly.num_vars(),
            pcs_commit_time.elapsed()
        );
        if let Some(stats) = statistics {
            stats.add_prover_pcs_time(pcs_commit_time.elapsed());
        }

        // [PIOP Phase] extract all the Hadamard instances and prove them via one single sumcheck
        let piop_hadamard_time = std::time::Instant::now();
        let blind_rotation_trace_ef = blind_rotation_trace_mle.to_ef();

        // prepare Hadamard instances
        let external_product_hadamard_instance =
            SumHadamardInstance::from(&blind_rotation_trace_ef.hadamard_trace);
        let acc_hadamard_instance =
            SumHadamardInstance::from(&blind_rotation_trace_ef.acc_trace.extract_hadamard_trace());
        let hadamard_instances = [external_product_hadamard_instance, acc_hadamard_instance]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let hadamard_instance_infos = hadamard_instances
            .iter()
            .map(SumcheckInstance::info)
            .collect::<Vec<_>>();
        trans.append_message(b"[Hadamard Statement]", &hadamard_instance_infos);

        // batch all Hadamard instances into one sumcheck claim
        let sumcheck_num_vars = hadamard_instance_infos[0].sumcheck_num_vars();
        let mut sumcheck_claim = SumcheckClaim::new(sumcheck_num_vars);
        let lagrange_kernel = Some(&LagrangeKernel::random(trans, sumcheck_num_vars));
        let randomness_hadamard = hadamard_instance_infos[0]
            .sample_randomness_for_sumcheck_batch(trans, hadamard_instances.len());
        HadamardPIOP::prover_batch_add_sumcheck(
            &hadamard_instances,
            &mut sumcheck_claim,
            &randomness_hadamard,
            lagrange_kernel,
        );

        // run the sumcheck protocol and generate proof
        let (sumcheck_proof, sumcheck_state) = MLSumcheck::prove(trans, &sumcheck_claim.poly)
            .expect("[External Product PIOP] Fail to generate sumcheck proof.");

        // generate evaluations for verifier to check the final subclaim of the sumcheck protocol
        let eval_table = sumcheck_state.fast_evaluate();
        let mut trace_evals = BlindRotationTraceEval::<EF>::default();
        blind_rotation_trace_mle.evaluate_ef_ntt_only(
            &mut trace_evals,
            &sumcheck_state.randomness,
            &blind_rotation_trace_ef,
            &sumcheck_claim.poly,
            &eval_table,
        );
        let hadamard_eval_proof =
            BatchedSumHadamardProof::from_blind_rotation_trace_eval(&trace_evals);
        trans.append_message(b"[Hadamard Evals]", &hadamard_eval_proof);

        info!(
            "[P]-[PIOP] Proving Hadamard via Sumcheck in {:?}",
            piop_hadamard_time.elapsed()
        );

        // [PIOP Phase] prove the validity of NTT evaluations since we consider all NTT oracles as virtual oracles
        let piop_ntt_time = std::time::Instant::now();
        let point_u = sumcheck_state.randomness
            [..blind_rotation_trace_mle.hadamard_trace.log_coeff_count]
            .to_vec();
        let point_v = sumcheck_state.randomness
            [blind_rotation_trace_mle.hadamard_trace.log_coeff_count..]
            .to_vec();

        // prepare the NTT equality instance for monomials used in Hadamard, where the coefficient matrix is sparse
        let monomial_poly = blind_rotation_trace_ef.acc_trace.monomial.poly.clone();
        let ntt_sparse_instance = NTTMatrixEvalInstance::from_subclaim(
            &monomial_poly,
            &params.ntt_table,
            &point_u,
            &point_v,
            trace_evals.acc_trace.monomial.ntt,
        );

        // parepare the NTT equality instance for normal polynomials used in Hadamard, where the coefficient matrix is dense
        let point_bit_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            trace_evals.log_num_bit_evals(),
        );
        let bit_poly = Rc::new(bit_poly.to_ef());
        let bit_ntt_evals = trace_evals.pack_bit_ntt_to_vec();
        let eval = compute_oracle_evals(&bit_ntt_evals, &point_bit_oracle);

        let mut point_v_prime = Vec::with_capacity(point_v.len() + point_bit_oracle.len());
        point_v_prime.extend_from_slice(&point_v);
        point_v_prime.extend_from_slice(&point_bit_oracle);
        let ntt_dense_instance = NTTMatrixEvalInstance::from_subclaim(
            &bit_poly,
            &params.ntt_table,
            &point_u,
            &point_v_prime,
            eval,
        );
        // parepare the NTT equality instance for normal [key] polynomials used in Hadamard
        let point_key_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            trace_evals.log_num_key_evals(),
        );
        let key_poly = params.key_commit.poly.clone();
        let key_ntt_evals = trace_evals.pack_key_ntt_to_vec();
        let eval = compute_oracle_evals(&key_ntt_evals, &point_key_oracle);

        let mut point_v_prime_key = Vec::with_capacity(point_v.len() + point_key_oracle.len());
        point_v_prime_key.extend_from_slice(&point_v);
        point_v_prime_key.extend_from_slice(&point_key_oracle);
        let ntt_dense_instance_key = NTTMatrixEvalInstance::from_subclaim(
            &key_poly,
            &params.ntt_table,
            &point_u,
            &point_v_prime_key,
            eval,
        );

        // prove both NTT instances in one sumcheck protocol
        let ntt_infos = vec![
            ntt_sparse_instance.info(),
            ntt_dense_instance.info(),
            ntt_dense_instance_key.info(),
        ];
        let ntt_instances = vec![
            ntt_sparse_instance,
            ntt_dense_instance,
            ntt_dense_instance_key,
        ];
        let (ntt_proof, ntt_state) = NTTMatrixEvalIOP::prover_batch(trans, &ntt_instances);
        info!(
            "[P]-[PIOP] Proving NTT Equality in {:?}",
            piop_ntt_time.elapsed()
        );

        // [PCS Phase] Open the dense coeffcient matrix evaluation `ntt_proof.coeff_eval_at_r_v[1]` at point_r_v_prime
        // this is the final subclaim of the NTT equality for dense polynomial
        let pcs_poly_open_time = std::time::Instant::now();
        let mut open_point = Vec::with_capacity(ntt_state.randomness.len() + point_v_prime.len());
        open_point.extend_from_slice(&ntt_state.randomness);
        open_point.extend_from_slice(&point_v_prime);
        let eval_proof = PCS::open(
            &params.pcs_params,
            &commitment,
            &commitment_state,
            &open_point,
            trans,
        );
        info!(
            "[P]-[PCS] Generating evaluation proof for one point of {} variables in {:?}",
            open_point.len(),
            pcs_poly_open_time.elapsed()
        );
        if let Some(stats) = statistics {
            stats.add_prover_pcs_time(pcs_poly_open_time.elapsed());
        }

        let pcs_poly_open_time = std::time::Instant::now();
        let mut open_point_key =
            Vec::with_capacity(ntt_state.randomness.len() + point_v_prime_key.len());
        open_point_key.extend_from_slice(&ntt_state.randomness);
        open_point_key.extend_from_slice(&point_v_prime_key);
        let eval_proof_key = PCS::open(
            &params.key_commit.params,
            &params.key_commit.commitment,
            &params.key_commit.commitment_state,
            &open_point_key,
            trans,
        );
        info!(
            "[P]-[PCS] Generating evaluation proof for one point of {} variables in {:?}",
            open_point_key.len(),
            pcs_poly_open_time.elapsed()
        );
        if let Some(stats) = statistics {
            stats.add_prover_pcs_time(pcs_poly_open_time.elapsed());
        }

        // [PIOP Phase] Open the sparse coefficient matrix evaluation `ntt_proof.coeff_eval_at_r_v[0]` at point_r_v
        // the pcs part is skipped in this since the polynomial to be committed is too small
        let piop_sparse_open_time = std::time::Instant::now();
        let kernel_rx = LagrangeKernel::from_point(&point_v);
        let kernel_ry = LagrangeKernel::from_point(&ntt_state.randomness);
        let sparse_matrix_eval_instance = SparseRowEvalInstance::from_subclaim::<F>(
            &blind_rotation_trace_mle.acc_trace.monomial_representation,
            &kernel_rx,
            &kernel_ry,
            ntt_proof.coeff_eval_at_r_v[0],
        );
        let sparse_eval_proof = SparseRowEvalSnarks::<F, EF, S, PCS>::prove_as_subprotocol(
            trans,
            &sparse_matrix_eval_instance,
        );
        info!(
            "[P]-PIOP Generating evaluation proof for a sparse polynomial in {:?}",
            piop_sparse_open_time.elapsed()
        );

        // [PIOP Phase] Prove the correctness of Accumulator Iteration Structure
        // the pcs part is skipped in this since the polynomial to be committed is too small
        let piop_acc_iteration_time = std::time::Instant::now();
        let indexed_lookup_permutation = blind_rotation_trace_ef
            .acc_trace
            .permutation_info
            .extract_indexed_lookup_trace(&point_v);
        let acc_iteration_proof = AccIterationSnarks::<F, EF, S, PCS>::prove_as_subprotocol(
            trans,
            &blind_rotation_trace_ef,
            &trace_evals,
            &point_v,
            &point_u,
            &indexed_lookup_permutation,
        );
        info!(
            "[P]-[PIOP] Proving Accumulator Iteration Structure in {:?}",
            piop_acc_iteration_time.elapsed()
        );

        // [Prover] Prove decomposition relation via lookup PIOP
        // For better modularity, we commit to the decomposition trace again in the decomposition SNARKs,
        // which is actually committed here already.
        let decomposition_trace = blind_rotation_trace_mle.extract_decomposition_traces();
        let decomp_params = DecompositionParams::new(
            params.code_spec.clone(),
            &blind_rotation_trace_mle.lt_tables,
        );
        let decomp_proof = DecompositionSnarks::<F, EF, S, PCS>::prove_as_subprotocol(
            trans,
            &decomposition_trace,
            &decomp_params,
            statistics,
        );

        BlindRotationProof {
            log_coeff_count: blind_rotation_trace_mle.log_coeff_count,
            log_num_bit_oracle: trace_evals.log_num_bit_evals(),
            log_num_key_oracle: trace_evals.log_num_key_evals(),
            pcs_params: params.pcs_params.clone(),
            commitment,
            sumcheck_poly_info: sumcheck_claim.poly.info(),
            sumcheck_proof,
            hadamard_info: hadamard_instance_infos,
            hadamard_proof: hadamard_eval_proof,
            ntt_infos,
            ntt_proof,
            acc_iteration_proof,
            eval_proof,
            eval_proof_key,
            sparse_eval_proof,
            decomp_proof,

            trace_evals,
            key_pcs_params: params.key_commit.params.clone(),
            key_commitment: params.key_commit.commitment.clone(),
        }
    }

    pub fn verify(
        &self,
        trans: &mut Transcript<EF>,
        proof: &BlindRotationProof<F, EF, S, PCS>,
        statistics: &mut Option<&mut crate::SnarkStatistics>,
    ) -> bool {
        let mut res = true;
        info!("[V] Start Blind Rotation Proof Verification...");
        trans.append_message(b"[Commit Phase]", &proof.commitment);

        // [PIOP Phase] verify all Hadamard instances via one single sumcheck
        let piop_hadamard_time = std::time::Instant::now();
        trans.append_message(b"[Hadamard Statement]", &proof.hadamard_info);
        let sumcheck_num_vars = proof.hadamard_info[0].sumcheck_num_vars();
        let lagrange_point = LagrangeKernel::random_point(trans, sumcheck_num_vars);

        // combine all sumcheck from Hadamard protocol into one with randomization
        let randomness_hadamard = proof.hadamard_info[0]
            .sample_randomness_for_sumcheck_batch(trans, proof.hadamard_info.len());

        // run the sumcheck verification
        let mut sumcheck_subclaim = MLSumcheck::verify(
            trans,
            &proof.sumcheck_poly_info,
            MLSumcheck::extract_sum(&proof.sumcheck_proof),
            &proof.sumcheck_proof,
        )
        .expect("[Sumcheck Protocol -Verifier] Fail to verify the sumcheck");

        // in the final round, compute the polynomial at the random point
        let kernel_at_r = eval_identity_function(&lagrange_point, &sumcheck_subclaim.point);
        trans.append_message(b"[Hadamard Evals]", &proof.hadamard_proof);
        HadamardPIOP::verifier_batch_compute_subclaim(
            &proof.hadamard_info,
            &proof.hadamard_proof,
            &mut sumcheck_subclaim,
            &randomness_hadamard,
            Some(kernel_at_r),
        );
        res &= sumcheck_subclaim.expected_evaluations.is_zero();
        assert!(res, "Hadamard Sumcheck verification failed.");

        info!(
            "[V]-[PIOP] Verifying Hadamard via Sumcheck in {:?}",
            piop_hadamard_time.elapsed()
        );

        // [PIOP Phase] verify the validity of NTT evaluations since we consider all NTT oracles as virtual oracles
        let piop_ntt_time = std::time::Instant::now();
        let _point_u = sumcheck_subclaim.point[..proof.log_coeff_count].to_vec();
        let point_v = sumcheck_subclaim.point[proof.log_coeff_count..].to_vec();
        let point_bit_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            proof.log_num_bit_oracle,
        );
        let point_key_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            proof.log_num_key_oracle,
        );

        // point_v.extend_from_slice(&point_bit_oracle);
        let (ntt_res, ntt_subclaim) =
            NTTMatrixEvalIOP::verifier_batch(trans, &proof.ntt_infos, &proof.ntt_proof);
        let open_evals = &proof.ntt_proof.coeff_eval_at_r_v;
        res &= ntt_res;
        assert!(res, "NTT Equality verification failed.");
        info!(
            "[V]-[PIOP] Verifying NTT Equality in {:?}",
            piop_ntt_time.elapsed()
        );

        // [PCS Phase] Verify the opening proof for the dense coeffcient matrix
        // evaluation `ntt_proof.coeff_eval_at_r_v[1]` at point_r_v_prime
        let pcs_poly_open_time = std::time::Instant::now();
        let mut open_point = Vec::with_capacity(
            ntt_subclaim.randomness.len() + point_v.len() + point_bit_oracle.len(),
        );
        open_point.extend_from_slice(&ntt_subclaim.randomness);
        open_point.extend_from_slice(&point_v);
        open_point.extend_from_slice(&point_bit_oracle);
        let eval_res = PCS::verify(
            &proof.pcs_params,
            &proof.commitment,
            &open_point,
            open_evals[1],
            &proof.eval_proof,
            trans,
        );
        res &= eval_res;
        assert!(res, "PCS Opening verification failed.");
        info!(
            "[V]-[PCS] Verifying evaluation proof for one point of {} variables in {:?}",
            open_point.len(),
            pcs_poly_open_time.elapsed()
        );
        if let Some(stats) = statistics {
            stats.add_verifier_pcs_time(pcs_poly_open_time.elapsed());
        }

        let pcs_poly_open_time = std::time::Instant::now();
        let mut open_point = Vec::with_capacity(
            ntt_subclaim.randomness.len() + point_v.len() + point_key_oracle.len(),
        );
        open_point.extend_from_slice(&ntt_subclaim.randomness);
        open_point.extend_from_slice(&point_v);
        open_point.extend_from_slice(&point_key_oracle);
        let eval_res = PCS::verify(
            &proof.key_pcs_params,
            &proof.key_commitment,
            &open_point,
            open_evals[2],
            &proof.eval_proof_key,
            trans,
        );
        res &= eval_res;
        assert!(res, "PCS Opening verification failed.");
        info!(
            "[V]-[PCS] Verifying evaluation proof for one point of {} variables in {:?}",
            open_point.len(),
            pcs_poly_open_time.elapsed()
        );
        if let Some(stats) = statistics {
            stats.add_verifier_pcs_time(pcs_poly_open_time.elapsed());
        }

        // [PIOP Phase] Verify the opening proof for the sparse coefficient matrix
        // evaluation `ntt_proof.coeff_eval_at_r_v[0]` at point_r_v
        let piop_sparse_open_time = std::time::Instant::now();
        let sparse_eval_res = SparseRowEvalSnarks::<F, EF, S, PCS>::verify_as_subprotocol(
            trans,
            &proof.sparse_eval_proof,
            statistics,
        );
        res &= sparse_eval_res;
        assert!(res, "Sparse Matrix Evaluation verification failed.");
        info!(
            "[V]-[PIOP] Verifying evaluation proof for a sparse polynomial in {:?}",
            piop_sparse_open_time.elapsed()
        );

        // [PIOP Phase] Verify the correctness of Accumulator Iteration Structure
        let piop_acc_iteration_time = std::time::Instant::now();
        let acc_iteration_res = AccIterationSnarks::<F, EF, S, PCS>::verify_as_subprotocol(
            trans,
            &proof.acc_iteration_proof,
            statistics,
        );
        res &= acc_iteration_res;
        assert!(res, "Acc Iteration verification failed.");
        info!(
            "[V]-[PIOP] Verifying Accumulator Iteration Structure in {:?}",
            piop_acc_iteration_time.elapsed()
        );

        // [Verifier] Verify decomposition relation via lookup PIOP
        let decomp_res = DecompositionSnarks::<F, EF, S, PCS>::verify_as_subprotocol(
            trans,
            &proof.decomp_proof,
            statistics,
        );
        res &= decomp_res;
        assert!(res, "Decomposition verification failed.");

        res
    }
}
