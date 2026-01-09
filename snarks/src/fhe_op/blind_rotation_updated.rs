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
use piop::sparse_matrix_eval::sparse_row::SparseRowEvalInstance;
use piop::{
    BatchedSumcheckPIOP, LagrangeKernel, SumcheckClaim, SumcheckInfo, SumcheckInstance,
    SumcheckPIOP,
};
use serde::Serialize;
use sumcheck::{MLSumcheck, Proof};
use trace::basic_ops::{SumHadamardTraceEval, SumHadamardTraceMLE};
use trace::blind_rotation_trace::BlindRotationTraceEval;
use trace::lookup_trace::small_table::LookupWitnessHelperEval;
use trace::{
    BlindRotationTrace, BlindRotationTraceMLE, ConvertToEF, EvaluableTraceEF, PackableTrace,
    acc_trace, blind_rotation_trace,
};
use trace::{EvaluableTrace, PackableEval};

use crate::fhe_op::acc_iteration::{self, AccIterationSnarks, AccIterationSnarksProof};
use crate::fhe_op::decomposition::{
    DecompositionParams, DecompositionSnarks, DecompositionSnarksProof,
};
use crate::sparse_matrix_eval::SparseRowEvalSnarks;
use crate::sparse_matrix_eval::sparse_row::SparseRowEvalSnarksProof;

#[derive(Default)]
pub struct BlindRotationSnarksUpdated<F, EF, S, PCS>
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

pub struct BlindRotationParamsUpdated<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub code_spec: S,
    pub blk_size: usize,
    // basis is the range size of the lookup table
    pub basis: usize,
    pub pcs_params: PCS::Parameters,
    pub pcs_params_ef: PCS::Parameters,
    pub ntt_table: Rc<Vec<EF>>,
}

impl<F, EF, S, PCS> BlindRotationParamsUpdated<F, EF, S, PCS>
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
        trace: &BlindRotationTrace<F>,
    ) -> Self {
        let oracle_num_vars = trace.num_vars() + trace.log_num_oracles();
        let pcs_params = PCS::setup(oracle_num_vars, Some(&code_spec));
        let helper_num_vars = trace.num_vars() + trace.hadamard_trace.log_num_helper_poly(blk_size);
        let pcs_params_ef = PCS::setup(helper_num_vars, Some(&code_spec));

        BlindRotationParamsUpdated {
            code_spec,
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
    // pub log_num_helper_poly: usize,
    #[serde(skip)]
    pub pcs_params: PCS::Parameters,
    #[serde(skip)]
    pub commitment: PCS::Commitment,
    #[serde(skip)]
    pub sumcheck_poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<EF>,
    pub hadamard_info: Vec<SumHadamardInfo<EF>>,
    pub hadamard_proof: BatchedSumHadamardProof<EF>,
    pub ntt_infos: Vec<NTTMatrixEvalInfo<EF>>,
    pub ntt_proof: BatchedNTTMatrixEvalProof<EF>,
    pub acc_iteration_proof: AccIterationSnarksProof<F, EF, S, PCS>,
    pub decomp_proof: DecompositionSnarksProof<F, EF, S, PCS>,

    #[serde(skip)]
    pub eval_proof: PCS::Proof,
    pub sparse_eval_proof: SparseRowEvalSnarksProof<F, EF, S, PCS>,

    // Redudant fields for ease of implementation
    #[serde(skip)]
    pub trace_evals: BlindRotationTraceEval<EF>,
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
    }

    pub fn pcs_proof_len(&self) -> usize {
        let mut len = 0;
        len += bincode::serde::encode_to_vec(&self.eval_proof, standard())
            .unwrap()
            .len();
        len + bincode::serde::encode_to_vec(&self.commitment, standard())
            .unwrap()
            .len()
    }
}

impl<F, EF, S, PCS> BlindRotationSnarksUpdated<F, EF, S, PCS>
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
        blind_rotation_trace: BlindRotationTrace<F>,
        params: &BlindRotationParamsUpdated<F, EF, S, PCS>,
    ) -> BlindRotationProof<F, EF, S, PCS> {
        let time = std::time::Instant::now();
        // [Commit Phase] commit to the trace polynomial
        let bit_poly = blind_rotation_trace.generate_oracle();
        let blind_rotation_trace_mle = BlindRotationTraceMLE::from(blind_rotation_trace);
        let (commitment, commitment_state) = PCS::commit(&params.pcs_params, &bit_poly);
        trans.append_message(b"[Commit Phase]", &commitment);

        // [PIOP Phase] extract all the Hadamard instances
        let piop_time = std::time::Instant::now();
        let time = std::time::Instant::now();
        let blind_rotation_trace_ef = blind_rotation_trace_mle.to_ef();

        let hadamard_instance = SumHadamardInstance::from(&blind_rotation_trace_ef.hadamard_trace);
        let acc_hadamard_instance =
            SumHadamardInstance::from(&blind_rotation_trace_ef.acc_trace.extract_hadamard_trace());
        let hadamard_instance = [hadamard_instance, acc_hadamard_instance]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let hadamard_instance_info = hadamard_instance
            .iter()
            .map(SumcheckInstance::info)
            .collect::<Vec<_>>();
        trans.append_message(b"[Hadamard Statement]", &hadamard_instance_info);

        let sumcheck_num_vars = hadamard_instance_info[0].sumcheck_num_vars();
        let mut sumcheck_claim = SumcheckClaim::new(sumcheck_num_vars);
        let lagrange_kernel = Some(&LagrangeKernel::random(trans, sumcheck_num_vars));

        // [PIOP Phase] prove all Hadamard via one sumcheck protocol
        // [PIOP Phase] add sumchecks from Hadamard instance
        let randomness_hadamard = hadamard_instance_info[0]
            .sample_randomness_for_sumcheck_batch(trans, hadamard_instance.len());
        HadamardPIOP::prover_batch_add_sumcheck(
            &hadamard_instance,
            &mut sumcheck_claim,
            &randomness_hadamard,
            lagrange_kernel,
        );

        // [PIOP Phase] run the sumcheck protocol
        let (sumcheck_proof, sumcheck_state) = MLSumcheck::prove(trans, &sumcheck_claim.poly)
            .expect("[External Product PIOP] Fail to generate sumcheck proof.");
        trans.append_message(b"[Sumcheck Protocol]", &sumcheck_proof);

        let eval_table = sumcheck_state.fast_evaluate();
        println!(
            "[P] PIOP Phase: Proving All Hadamard and Lookup Relation in {:?}",
            time.elapsed()
        );

        let time = std::time::Instant::now();
        let mut trace_evals = BlindRotationTraceEval::<EF>::default();
        blind_rotation_trace_mle.evaluate_ef_ntt_only(
            &mut trace_evals,
            &sumcheck_state.randomness,
            &blind_rotation_trace_ef,
            &sumcheck_claim.poly,
            &eval_table,
        );

        println!(
            "[P] PIOP Phase: Evaluating the remaining oracle in {:?}",
            time.elapsed()
        );

        // [PIOP Phase] evaluate the polynomials and append them into proof
        let hadamard_eval_proof =
            BatchedSumHadamardProof::from_blind_rotation_trace_eval(&trace_evals);

        trans.append_message(b"[Hadamard Evals]", &hadamard_eval_proof);

        let point_u = sumcheck_state.randomness
            [..blind_rotation_trace_mle.hadamard_trace.log_coeff_count]
            .to_vec();
        let point_v = sumcheck_state.randomness
            [blind_rotation_trace_mle.hadamard_trace.log_coeff_count..]
            .to_vec();

        let time = std::time::Instant::now();
        // NTT Sparse Matrix Evaluation
        let monomial_poly = blind_rotation_trace_ef.acc_trace.monomial.poly.clone();
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
            blind_rotation_trace_mle.log_num_oracles(),
        );

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

        println!(
            "[P] PIOP Phase: Proving NTT Virtual Oracle Equality in {:?}",
            time.elapsed()
        );
        trans.append_message(b"[PIOP Phase]", &ntt_proof);

        let time = std::time::Instant::now();
        // Proving Acc Iteration Structure
        let indexed_lookup_permutation = blind_rotation_trace_ef
            .acc_trace
            .permutation_info
            .extract_indexed_lookup_trace(&point_v);

        let acc_iteration_proof = AccIterationSnarks::prove_as_subprotocol(
            trans,
            &blind_rotation_trace_ef,
            &trace_evals,
            &point_v,
            &point_u,
            &indexed_lookup_permutation,
        );
        println!(
            "[P] PIOP Phase: Proving Accumulator Iteration Structure in {:?}",
            time.elapsed()
        );

        println!("[P] PIOP Phase total time: {:?}", piop_time.elapsed());

        let time = std::time::Instant::now();
        // Open the coeffcient matrix evaluation `ntt_proof.coeff_eval_at_r_v[1]` at point_r_v_prime
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

        println!(
            "[P] PCS Opening one point on Oracles in {:?}",
            time.elapsed()
        );

        // Open the sparse coefficient matrix evaluation `ntt_proof.coeff_eval_at_r_v[0]` at point_r_v using SparseMatrix
        let time = std::time::Instant::now();
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
        println!("[P] Sparse PCS Opening Phase time: {:?}", time.elapsed());

        // Prove decomposition relation via lookup PIOP
        let time = std::time::Instant::now();
        let decomposition_trace = blind_rotation_trace_mle.extract_decomposition_traces();
        let decomp_params = DecompositionParams::new(
            params.code_spec.clone(),
            &blind_rotation_trace_mle.lt_tables,
        );
        let decomp_proof = DecompositionSnarks::<F, EF, S, PCS>::prove_as_subprotocol(
            trans,
            &decomposition_trace,
            &decomp_params,
        );
        println!("[P] Decomposition time: {:?}", time.elapsed());

        BlindRotationProof {
            log_coeff_count: blind_rotation_trace_mle.log_coeff_count,
            log_num_oracle: blind_rotation_trace_mle.log_num_oracles(),
            pcs_params: params.pcs_params.clone(),
            commitment,
            sumcheck_poly_info: sumcheck_claim.poly.info(),
            sumcheck_proof,
            hadamard_info: hadamard_instance_info,
            hadamard_proof: hadamard_eval_proof,
            ntt_infos,
            ntt_proof,
            acc_iteration_proof,
            eval_proof,
            sparse_eval_proof,
            decomp_proof,

            trace_evals,
        }
    }

    pub fn verify(
        &self,
        trans: &mut Transcript<EF>,
        proof: &BlindRotationProof<F, EF, S, PCS>,
    ) -> bool {
        trans.append_message(b"[Commit Phase]", &proof.commitment);
        let mut res = true;

        let time = std::time::Instant::now();
        // PIOP Phase
        trans.append_message(b"[Hadamard Statement]", &proof.hadamard_info);

        let sumcheck_num_vars = proof.hadamard_info[0].sumcheck_num_vars();
        let lagrange_point = LagrangeKernel::random_point(trans, sumcheck_num_vars);

        // Combine all sumcheck from Hadamard protocol into one
        let randomness_hadamard = proof.hadamard_info[0]
            .sample_randomness_for_sumcheck_batch(trans, proof.hadamard_info.len());

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
        HadamardPIOP::verifier_batch_compute_subclaim(
            &proof.hadamard_info,
            &proof.hadamard_proof,
            &mut sumcheck_subclaim,
            &randomness_hadamard,
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

        point_v.extend_from_slice(&point_bit_oracle);
        let (ntt_res, ntt_subclaim) =
            NTTMatrixEvalIOP::verifier_batch(trans, &proof.ntt_infos, &proof.ntt_proof);
        let open_eval_2 = &proof.ntt_proof.coeff_eval_at_r_v;
        trans.append_message(b"[PIOP Phase]", &proof.ntt_proof);
        res &= ntt_res;
        assert!(res, "NTT Matrix Evaluation verification failed.");

        // Acc Iteration Verification
        let acc_iteration_res =
            AccIterationSnarks::verify_as_subprotocol(trans, &proof.acc_iteration_proof);
        res &= acc_iteration_res;
        assert!(res, "Acc Iteration verification failed.");

        println!("[V] PIOP Phase total time: {:?}", time.elapsed());

        let time = std::time::Instant::now();
        let mut open_point_2 = Vec::with_capacity(ntt_subclaim.randomness.len() + point_v.len());
        open_point_2.extend_from_slice(&ntt_subclaim.randomness);
        open_point_2.extend_from_slice(&point_v);

        // PCS Opening Phase
        let eval_res = PCS::verify(
            &proof.pcs_params,
            &proof.commitment,
            &open_point_2,
            open_eval_2[1],
            &proof.eval_proof,
            trans,
        );
        res &= eval_res;
        assert!(res, "PCS Opening verification failed.");

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

        // Verify decomposition relation via lookup PIOP
        let decomp_res =
            DecompositionSnarks::<F, EF, S, PCS>::verify_as_subprotocol(trans, &proof.decomp_proof);
        res &= decomp_res;
        assert!(res, "Decomposition verification failed.");

        res
    }
}
