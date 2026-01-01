//! Blind Rotation SNARKs proving FHE operation (X^{a_i} - 1) * ACC * RGSW(s_i)
//!
//! This mainly consists of two parts:
//! 1. Proving the correctness of Mid = (X^{a_i} - 1) * ACC_Input
//!     This is separately proven in `monomial_hadamard.rs`
//! 2. Proving the correctness of ACC_Output = Mid * RGSW(s_i)
//!     This is separately proven in `external_product.rs`
use core::time;
use std::rc::Rc;

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field, PolynomialInfo};
use bincode::config::standard;
use helper::utils::{compute_oracle_evals, eval_identity_function};
use helper::{FiatShamirTranscript, Transcript};
use pcs::PolynomialCommitmentScheme;
use piop::hadamard::{BatchedSumHadamardProof, HadamardPIOP, SumHadamardInfo, SumHadamardInstance};
use piop::lookup::small_table::{LogUpIOP, LogUpInstance, LogUpInstanceInfo, LogUpProof};
use piop::ntt::{
    BatchedNTTMatrixEvalProof, NTTMatrixEvalIOP, NTTMatrixEvalInfo, NTTMatrixEvalInstance,
    NTTMatrixEvalProof,
};
use piop::permutation::row_perm::compute_permutation_at_point;
use piop::sparse_matrix_eval::sparse_row::SparseRowEvalInstance;
use piop::{
    BatchedSumcheckPIOP, LagrangeKernel, SumcheckClaim, SumcheckInfo, SumcheckInstance,
    SumcheckPIOP,
};
use serde::Serialize;
use sumcheck::{MLSumcheck, Proof};
use trace::lookup_trace::small_table::LookupWitnessHelperEval;
use trace::pbs_trace::PBSTraceEval;
use trace::{
    ConvertToEF, EvaluableTraceEF, PBSTrace, PBSTraceMLE, PackableTrace, SumHadamardTraceEval,
    acc_trace, pbs_trace,
};
use trace::{EvaluableTrace, PackableEval, SumHadamardTraceMLE};

use crate::fhe_op::acc_iteration::{self, AccIterationSnarks, AccIterationSnarksProof};
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

pub struct BlindRotationParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub blk_size: usize,
    // basis is the range size of the lookup table
    pub basis: usize,
    pub pcs_params: PCS::Parameters,
    pub pcs_params_ef: PCS::Parameters,
    pub ntt_table: Rc<Vec<EF>>,
}

impl<F, EF, S, PCS> BlindRotationParams<F, EF, S, PCS>
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
        trace: &PBSTrace<F>,
    ) -> Self {
        let oracle_num_vars = trace.num_vars() + trace.log_num_oracles();
        let pcs_params = PCS::setup(oracle_num_vars, Some(code_spec.clone()));
        let helper_num_vars = trace.num_vars() + trace.hadamard_trace.log_num_helper_poly(blk_size);
        let pcs_params_ef = PCS::setup(helper_num_vars, Some(code_spec.clone()));

        BlindRotationParams {
            blk_size,
            basis,
            pcs_params,
            pcs_params_ef,
            ntt_table: Rc::new(ntt_table.to_ef()),
        }
    }
}

#[derive(Serialize)]
pub struct BlindRotationProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub log_coeff_count: usize,
    pub log_num_oracle: usize,
    pub log_num_helper_poly: usize,
    #[serde(skip)]
    pub pcs_params: PCS::Parameters,
    #[serde(skip)]
    pub commitment: PCS::Commitment,
    #[serde(skip)]
    pub pcs_params_ef: PCS::Parameters,
    #[serde(skip)]
    pub helper_commitment: PCS::Commitment,
    pub sumcheck_poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<EF>,
    pub lookup_info: LogUpInstanceInfo<EF>,
    pub lookup_proof: LogUpProof<EF>,
    pub hadamard_info: Vec<SumHadamardInfo<EF>>,
    pub hadamard_proof: BatchedSumHadamardProof<EF>,
    pub ntt_infos: Vec<NTTMatrixEvalInfo<EF>>,
    pub ntt_proof: BatchedNTTMatrixEvalProof<EF>,
    pub acc_iteration_proof: AccIterationSnarksProof<EF>,

    #[serde(skip)]
    pub eval_proof: Vec<PCS::Proof>,
    #[serde(skip)]
    pub eval_ef_proof: PCS::ProofEF,
    pub sparse_eval_proof: SparseRowEvalSnarksProof<F, EF, S, PCS>,

    // Redudant fields for ease of implementation
    #[serde(skip)]
    pub trace_evals: PBSTraceEval<EF>,
    #[serde(skip)]
    pub helper_evals: LookupWitnessHelperEval<EF>,
}

impl<F, EF, S, PCS> BlindRotationProof<F, EF, S, PCS>
where
    F: Field,
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
    pub fn piop_proof_len(&self) -> usize {
        bincode::serde::encode_to_vec(self.sumcheck_poly_info, standard())
            .unwrap()
            .len()
            + bincode::serde::encode_to_vec(&self.sumcheck_proof, standard())
                .unwrap()
                .len()
            + bincode::serde::encode_to_vec(&self.lookup_info, standard())
                .unwrap()
                .len()
            + bincode::serde::encode_to_vec(&self.lookup_proof, standard())
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
            + bincode::serde::encode_to_vec(&self.trace_evals, standard())
                .unwrap()
                .len()
            + bincode::serde::encode_to_vec(&self.helper_evals, standard())
                .unwrap()
                .len()
    }

    pub fn pcs_proof_len(&self) -> usize {
        let mut len = 0;
        for proof in &self.eval_proof {
            len += bincode::serde::encode_to_vec(proof, standard())
                .unwrap()
                .len();
        }
        len + bincode::serde::encode_to_vec(&self.eval_ef_proof, standard())
            .unwrap()
            .len()
            + bincode::serde::encode_to_vec(&self.commitment, standard())
                .unwrap()
                .len()
            + bincode::serde::encode_to_vec(&self.helper_commitment, standard())
                .unwrap()
                .len()
    }
}

impl<F, EF, S, PCS> BlindRotationSnarks<F, EF, S, PCS>
where
    F: Field,
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
        // trace_mle: &SumHadamardTraceMLE<F>,
        pbs_trace: PBSTrace<F>,
        params: &BlindRotationParams<F, EF, S, PCS>,
    ) -> BlindRotationProof<F, EF, S, PCS> {
        let time = std::time::Instant::now();
        // [Commit Phase] commit to the trace polynomial
        let bit_poly = pbs_trace.generate_oracle();
        let pbs_trace_mle = PBSTraceMLE::from(pbs_trace);
        let (commitment, commitment_state) = PCS::commit(&params.pcs_params, &bit_poly);
        trans.append_message(b"[Commit Phase]", &commitment);

        // [Commit Phase] commit to the helper polynomial for lookup
        let random_value =
            trans.get_challenge(b"[Challenge] random value used in the rational identity");
        let lookup_trace = pbs_trace_mle
            .hadamard_trace
            .extract_lookup_trace_mle_small_table(params.basis);
        let lookup_witness = lookup_trace.compute_witness_pure();
        let lookup_helper = lookup_trace.compute_helper_functions_ef::<EF>(
            params.blk_size,
            random_value,
            &lookup_witness,
        );
        let helper_poly = lookup_helper.generate_oracle();
        let (helper_commitment, helper_commitment_state) =
            PCS::commit_ef(&params.pcs_params_ef, &helper_poly);
        println!("[P] Commit Phase time: {:?}", time.elapsed());

        // [PIOP Phase] extract the Hadamard instances
        let piop_time = std::time::Instant::now();
        let time = std::time::Instant::now();
        let pbs_ef = pbs_trace_mle.to_ef();
        let lookup_trace_ef = lookup_trace.to_ef();
        let hadamard_instance = SumHadamardInstance::from(&pbs_ef.hadamard_trace);
        let acc_hadamard_instance =
            SumHadamardInstance::from(&pbs_ef.acc_trace.extract_hadamard_trace());
        let hadamard_instance = [hadamard_instance, acc_hadamard_instance]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        // [PIOP Phase] extract the Lookup instance
        let lookup_instance = LogUpInstance::from(&lookup_trace_ef, &lookup_helper);
        let hadamard_instance_info = hadamard_instance
            .iter()
            .map(SumcheckInstance::info)
            .collect::<Vec<_>>();
        trans.append_message(b"[Hadamard Statement]", &hadamard_instance_info);
        let lookup_instance_info = lookup_instance.info();
        trans.append_message(b"[Lookup Statement]", &lookup_instance_info);

        assert_eq!(
            hadamard_instance_info[0].sumcheck_num_vars(),
            lookup_instance_info.sumcheck_num_vars()
        );
        let sumcheck_num_vars = hadamard_instance_info[0].sumcheck_num_vars();
        let mut sumcheck_claim = SumcheckClaim::new(sumcheck_num_vars);
        let lagrange_kernel = Some(&LagrangeKernel::random(trans, sumcheck_num_vars));

        // [PIOP Phase] prove Hadamard and Lookup via one sumcheck protocol
        // [PIOP Phase] add sumchecks from Hadamard instance
        let randomness_hadamard = hadamard_instance_info[0]
            .sample_randomness_for_sumcheck_batch(trans, hadamard_instance.len());
        HadamardPIOP::prover_batch_add_sumcheck(
            &hadamard_instance,
            &mut sumcheck_claim,
            &randomness_hadamard,
            lagrange_kernel,
        );
        // [PIOP Phase] add sumchecks from Lookup instance
        let randomness_lookup = lookup_instance_info.sample_randomness_for_sumcheck(trans);
        LogUpIOP::prover_add_sumcheck(
            &lookup_instance,
            &mut sumcheck_claim,
            &randomness_lookup,
            lagrange_kernel,
        );
        // [PIOP Phase] run the sumcheck protocol
        let (sumcheck_proof, sumcheck_state) = MLSumcheck::prove(trans, &sumcheck_claim.poly)
            .expect("[External Product PIOP] Fail to generate sumcheck proof.");
        trans.append_message(b"[Sumcheck Protocol]", &sumcheck_proof);

        let eval_table = sumcheck_state.fast_evaluate();
        println!("[P] PIOP Phase: Proving All Hadamard and Lookup Relation in {:?}", time.elapsed());

        let time = std::time::Instant::now();
        let trace_evals = pbs_trace_mle.evaluate_ef_with_lookup(
            &sumcheck_state.randomness,
            &pbs_ef,
            &sumcheck_claim.poly,
            &eval_table,
        );
        let helper_evals = lookup_helper.evaluate_with_lookup(
            &sumcheck_state.randomness,
            &sumcheck_claim.poly,
            &eval_table,
        );
        println!(
            "[P] PIOP Phase: Evaluating the remaining oracle in {:?}",
            time.elapsed()
        );

        // [PIOP Phase] evaluate the polynomials and append them into proof
        // let trace_evals = pbs_trace_mle.evaluate_ef(&sumcheck_state.randomness);
        // let helper_evals = lookup_helper.evaluate(&sumcheck_state.randomness);
        let hadamard_eval_proof = BatchedSumHadamardProof::from_pbs_trace_eval(&trace_evals);
        let lookup_eval_proof = LogUpProof::from_hadamard_trace_eval(
            &trace_evals.hadamard_trace,
            &helper_evals,
            random_value,
        );
        trans.append_message(b"[Hadamard Evals]", &hadamard_eval_proof);
        trans.append_message(b"[Lookup Evals]", &lookup_eval_proof);

        let point_u =
            sumcheck_state.randomness[..pbs_trace_mle.hadamard_trace.log_coeff_count].to_vec();
        let mut point_v =
            sumcheck_state.randomness[pbs_trace_mle.hadamard_trace.log_coeff_count..].to_vec();

        let time = std::time::Instant::now();
        // NTT Sparse Matrix Evaluation
        let monomial_poly = pbs_ef.acc_trace.monomial.poly.clone();
        let ntt_sparse_instance = NTTMatrixEvalInstance::from_subclaim(
            &monomial_poly,
            &params.ntt_table,
            &point_u,
            &point_v,
            trace_evals.acc_trace.monomial.ntt,
        );

        // Normal NTT Matrix Evaluation
        let point_bit_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            pbs_trace_mle.log_num_oracles(),
        );
        let point_helper_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            lookup_trace.log_num_helper_oracles(params.blk_size),
        );
        // Reduced subclaim from Lookup: query the coefficient polynomial at `open_point_1`
        let mut open_point_1 =
            Vec::with_capacity(point_u.len() + point_v.len() + point_bit_oracle.len());
        let mut open_point_ef =
            Vec::with_capacity(point_u.len() + point_v.len() + point_helper_oracle.len());
        open_point_1.extend_from_slice(&sumcheck_state.randomness);
        open_point_1.extend_from_slice(&point_bit_oracle);
        open_point_ef.extend_from_slice(&sumcheck_state.randomness);
        open_point_ef.extend_from_slice(&point_helper_oracle);

        // Reduced subclaim from Hadamard: query the evluation polynomial at `open_point_1`,
        // which can be further proven by NTT PIOP
        // TODO : optimize the conversion (101.199208ms here)
        let bit_poly = Rc::new(bit_poly.to_ef());
        let bit_ntt_evals = trace_evals.pack_ntt_to_vec();
        let eval = compute_oracle_evals(&bit_ntt_evals, &point_bit_oracle);

        let mut point_v_prime = Vec::with_capacity(point_v.len() + point_bit_oracle.len());
        point_v_prime.extend_from_slice(&point_v);
        point_v_prime.extend_from_slice(&point_bit_oracle);
        let ntt_instance = NTTMatrixEvalInstance::from_subclaim(
            &bit_poly,
            &params.ntt_table,
            &point_u,
            &point_v_prime,
            eval,
        );

        let ntt_infos = vec![ntt_sparse_instance.info(), ntt_instance.info()];
        let ntt_instances = vec![ntt_sparse_instance, ntt_instance];
        let (ntt_proof, ntt_state) = NTTMatrixEvalIOP::prover_batch(trans, &ntt_instances);

        println!("[P] PIOP Phase: Proving NTT Virtual Oracle Equality in {:?}", time.elapsed());
        trans.append_message(b"[PIOP Phase]", &ntt_proof);

        let time = std::time::Instant::now();
        // Acc Iteration Structure
        let permutation = Rc::new(compute_permutation_at_point(
            pbs_ef.log_num_round,
            &pbs_ef.acc_trace.permutation_info,
            &point_v,
        ));
        let acc_iteration_proof = AccIterationSnarks::prove_as_subprotocol(
            trans,
            &pbs_ef,
            &trace_evals,
            &point_v,
            &point_u,
            &permutation,
        );
        println!(
            "[P] PIOP Phase: Proving Accumulator Iteration Structure in {:?}",
            time.elapsed()
        );

        println!("[P] PIOP Phase total time: {:?}", piop_time.elapsed());

        let time = std::time::Instant::now();
        // Open the coeffcient matrix evaluation `ntt_proof.coeff_eval_at_r_v[1]` at point_r_v_prime
        let mut open_point_2 = Vec::with_capacity(ntt_state.randomness.len() + point_v_prime.len());
        open_point_2.extend_from_slice(&ntt_state.randomness);
        open_point_2.extend_from_slice(&point_v_prime);
        let open_points = vec![open_point_1, open_point_2];
        let eval_proof = PCS::batch_open(
            &params.pcs_params,
            &commitment,
            &commitment_state,
            &open_points,
            trans,
        );

        let eval_ef_proof = PCS::open_ef(
            &params.pcs_params_ef,
            &helper_commitment,
            &helper_commitment_state,
            &open_point_ef,
            trans,
        );

        println!("[P] PCS Opening 3 Points on 2 Oracles in {:?}", time.elapsed());

        // Open the sparse coefficient matrix evaluation `ntt_proof.coeff_eval_at_r_v[0]` at point_r_v using SparseMatrix
        let time = std::time::Instant::now();
        let kernel_rx = LagrangeKernel::from_point(&point_v);
        let kernel_ry = LagrangeKernel::from_point(&ntt_state.randomness);
        let sparse_matrix_eval_instance = SparseRowEvalInstance::from_subclaim::<F>(
            &pbs_trace_mle.acc_trace.monomial_representation,
            &kernel_rx,
            &kernel_ry,
            ntt_proof.coeff_eval_at_r_v[0],
        );
        let sparse_eval_proof = SparseRowEvalSnarks::<F, EF, S, PCS>::prove_as_subprotocol(
            trans,
            &sparse_matrix_eval_instance,
        );
        println!("[P] Sparse PCS Opening Phase time: {:?}", time.elapsed());

        BlindRotationProof {
            log_coeff_count: pbs_trace_mle.log_coeff_count,
            log_num_oracle: pbs_trace_mle.log_num_oracles(),
            log_num_helper_poly: lookup_trace.log_num_helper_oracles(params.blk_size),
            pcs_params: params.pcs_params.clone(),
            commitment,
            pcs_params_ef: params.pcs_params_ef.clone(),
            helper_commitment,
            sumcheck_poly_info: sumcheck_claim.poly.info(),
            sumcheck_proof,
            lookup_info: lookup_instance_info,
            lookup_proof: lookup_eval_proof,
            hadamard_info: hadamard_instance_info,
            hadamard_proof: hadamard_eval_proof,
            ntt_infos,
            ntt_proof,
            acc_iteration_proof,
            eval_proof,
            eval_ef_proof,
            sparse_eval_proof,

            trace_evals,
            helper_evals,
        }
    }

    pub fn verify(
        &self,
        trans: &mut Transcript<EF>,
        proof: &BlindRotationProof<F, EF, S, PCS>,
    ) -> bool {
        trans.append_message(b"[Commit Phase]", &proof.commitment);
        let mut res = true;

        let _random_value =
            trans.get_challenge(b"[Challenge] random value used in the rational identity");

        let time = std::time::Instant::now();
        // PIOP Phase
        trans.append_message(b"[Hadamard Statement]", &proof.hadamard_info);
        trans.append_message(b"[Lookup Statement]", &proof.lookup_info);

        assert_eq!(
            proof.hadamard_info[0].sumcheck_num_vars(),
            proof.lookup_info.sumcheck_num_vars()
        );
        let sumcheck_num_vars = proof.hadamard_info[0].sumcheck_num_vars();
        let lagrange_point = LagrangeKernel::random_point(trans, sumcheck_num_vars);

        // Combine all sumcheck from Hadamard protocol into one
        let randomness_hadamard = proof.hadamard_info[0]
            .sample_randomness_for_sumcheck_batch(trans, proof.hadamard_info.len());
        // Combine all sumcheck form Lookup protocol into one
        let randomness_lookup = proof.lookup_info.sample_randomness_for_sumcheck(trans);

        let mut sumcheck_subclaim = MLSumcheck::verify(
            trans,
            &proof.sumcheck_poly_info,
            MLSumcheck::extract_sum(&proof.sumcheck_proof),
            &proof.sumcheck_proof,
        )
        .expect("[Sumcheck Protocol -Verifier] Fail to verify the sumcheck");
        trans.append_message(b"[Sumcheck Protocol]", &proof.sumcheck_proof);

        let kernel_at_r = eval_identity_function(&lagrange_point, &sumcheck_subclaim.point);

        trans.append_message(b"[Hadamard Evals]", &proof.hadamard_proof);
        trans.append_message(b"[Lookup Evals]", &proof.lookup_proof);
        HadamardPIOP::verifier_batch_compute_subclaim(
            &proof.hadamard_info,
            &proof.hadamard_proof,
            &mut sumcheck_subclaim,
            &randomness_hadamard,
            Some(kernel_at_r),
        );

        LogUpIOP::verifier_compute_subclaim(
            &proof.lookup_info,
            &proof.lookup_proof,
            &mut sumcheck_subclaim,
            &randomness_lookup,
            Some(kernel_at_r),
        );

        res &= sumcheck_subclaim.expected_evaluations.is_zero();
        assert!(res, "Sumcheck verification failed.");

        let point_u = sumcheck_subclaim.point[..proof.log_coeff_count].to_vec();
        let mut point_v = sumcheck_subclaim.point[proof.log_coeff_count..].to_vec();
        let point_bit_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            proof.log_num_oracle,
        );
        let point_helper_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            proof.log_num_helper_poly,
        );
        // Reduced subclaim from Lookup: query the coefficient polynomial at `open_point_1`
        let mut open_point_1 =
            Vec::with_capacity(point_u.len() + point_v.len() + point_bit_oracle.len());
        let mut open_point_ef =
            Vec::with_capacity(point_u.len() + point_v.len() + point_helper_oracle.len());
        open_point_1.extend_from_slice(&point_u);
        open_point_1.extend_from_slice(&point_v);
        open_point_1.extend_from_slice(&point_bit_oracle);
        open_point_ef.extend_from_slice(&sumcheck_subclaim.point);
        open_point_ef.extend_from_slice(&point_helper_oracle);

        let bit_poly_evals = proof.trace_evals.pack_poly_to_vec();
        let open_eval_1 = compute_oracle_evals(&bit_poly_evals, &point_bit_oracle);

        let helper_poly_evals = proof.helper_evals.pack_to_vec();
        let open_helper_eval = compute_oracle_evals(&helper_poly_evals, &point_helper_oracle);

        point_v.extend_from_slice(&point_bit_oracle);
        let (ntt_res, ntt_subclaim) =
            NTTMatrixEvalIOP::verifier_batch(trans, &proof.ntt_infos, &proof.ntt_proof);
        let open_eval_2 = &proof.ntt_proof.coeff_eval_at_r_v;
        trans.append_message(b"[PIOP Phase]", &proof.ntt_proof);
        res &= ntt_res;
        assert!(res, "NTT Matrix Evaluation verification failed.");

        // Acc Iteration Verification
        let acc_iteration_res = AccIterationSnarks::verify_as_subprotocol(
            trans,
            &proof.acc_iteration_proof,
        );
        res &= acc_iteration_res;
        assert!(res, "Acc Iteration verification failed.");

        println!("[V] PIOP Phase total time: {:?}", time.elapsed());

        let time = std::time::Instant::now();
        let mut open_point_2 = Vec::with_capacity(ntt_subclaim.randomness.len() + point_v.len());
        open_point_2.extend_from_slice(&ntt_subclaim.randomness);
        open_point_2.extend_from_slice(&point_v);

        // Verify the coeffcient matrix evaluation `ntt_proof.coeff_eval_at_r_v[1]` at point_r_v_prime
        let open_points = vec![open_point_1, open_point_2];
        let open_evals = vec![open_eval_1, open_eval_2[1]];

        // PCS Opening Phase
        let eval_res = PCS::batch_verify(
            &proof.pcs_params,
            &proof.commitment,
            &open_points,
            &open_evals,
            &proof.eval_proof,
            trans,
        );
        res &= eval_res;
        assert!(res, "PCS Opening verification failed.");

        let eval_ef_res = PCS::verify_ef(
            &proof.pcs_params_ef,
            &proof.helper_commitment,
            &open_point_ef,
            open_helper_eval,
            &proof.eval_ef_proof,
            trans,
        );
        res &= eval_ef_res;
        assert!(res, "PCS EF Opening verification failed.");

        println!("[V] PCS Opening Phase time: {:?}", time.elapsed());

        // Verify the coeffcient matrix evaluation `ntt_proof.coeff_eval_at_r_v[0]` at point_r_v
        let time = std::time::Instant::now();
        let sparse_eval_res = SparseRowEvalSnarks::<F, EF, S, PCS>::verify_as_subprotocol(
            trans,
            &proof.sparse_eval_proof,
        );
        println!("[V] Sparse PCS Opening Phase time: {:?}", time.elapsed());
        res &= sparse_eval_res;
        assert!(res, "Sparse Matrix Evaluation verification failed.");

        res
    }
}
