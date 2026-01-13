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
use log::info;
use pcs::PolynomialCommitmentScheme;
use piop::hadamard::{BatchedSumHadamardProof, HadamardPIOP, SumHadamardInfo, SumHadamardInstance};
use piop::ntt::{NTTMatrixEvalIOP, NTTMatrixEvalInfo, NTTMatrixEvalInstance, NTTMatrixEvalProof};
use piop::{
    BatchedSumcheckPIOP, LagrangeKernel, SumcheckClaim, SumcheckInfo, SumcheckInstance,
    SumcheckPIOP,
};
use serde::Serialize;
use sumcheck::{MLSumcheck, Proof};
use trace::PackableEval;
use trace::basic_ops::{SumHadamardTraceEval, SumHadamardTraceMLE};
use trace::{ConvertToEF, EvaluableTraceEF, PackableTrace};

#[derive(Default)]
pub struct HadamardProductSnarks<F, EF, S, PCS>
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

pub struct HadamardProductParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub code_spec: S,
    pub pcs_params: PCS::Parameters,
    pub ntt_table: Rc<Vec<EF>>,
}

impl<F, EF, S, PCS> HadamardProductParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub fn new(code_spec: S, ntt_table: Vec<F>, trace: &SumHadamardTraceMLE<F>) -> Self {
        let oracle_num_vars = trace.num_vars() + trace.log_num_oracles();
        let pcs_params = PCS::setup(oracle_num_vars, Some(&code_spec));

        HadamardProductParams {
            code_spec,
            pcs_params,
            ntt_table: Rc::new(ntt_table.to_ef()),
        }
    }
}

#[derive(Serialize)]
pub struct HadamardProductProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub log_coeff_count: usize,
    pub log_num_oracle: usize,
    // pub log_num_helper_poly: usize,
    pub pcs_params: PCS::Parameters,
    pub commitment: PCS::Commitment,
    pub sumcheck_poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<EF>,
    pub hadamard_info: Vec<SumHadamardInfo<EF>>,
    pub hadamard_proof: BatchedSumHadamardProof<EF>,
    pub ntt_info: NTTMatrixEvalInfo<EF>,
    pub ntt_proof: NTTMatrixEvalProof<EF>,
    pub eval_proof: PCS::Proof,

    // Redudant fields for ease of implementation
    #[serde(skip)]
    pub trace_evals: SumHadamardTraceEval<EF>,
}

impl<F, EF, S, PCS> HadamardProductProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F> + Serialize,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
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
            + bincode::serde::encode_to_vec(&self.ntt_info, standard())
                .unwrap()
                .len()
            + bincode::serde::encode_to_vec(&self.ntt_proof, standard())
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

impl<F, EF, S, PCS> HadamardProductSnarks<F, EF, S, PCS>
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
        trace_mle: &SumHadamardTraceMLE<F>,
        params: &HadamardProductParams<F, EF, S, PCS>,
    ) -> HadamardProductProof<F, EF, S, PCS> {
        Self::prove_as_subprotocol(trans, trace_mle, params, &mut None)
    }

    pub fn verify(
        &self,
        trans: &mut Transcript<EF>,
        proof: &HadamardProductProof<F, EF, S, PCS>,
    ) -> bool {
        Self::verify_as_subprotocol(trans, proof, &mut None)
    }

    pub fn prove_as_subprotocol(
        trans: &mut Transcript<EF>,
        trace_mle: &SumHadamardTraceMLE<F>,
        params: &HadamardProductParams<F, EF, S, PCS>,
        statistics: &mut Option<&mut crate::SnarkStatistics>,
    ) -> HadamardProductProof<F, EF, S, PCS> {
        // [Commit Phase] commit to the trace polynomial
        let bit_poly = trace_mle.generate_oracle();
        let pcs_commit_time = std::time::Instant::now();
        let (commitment, commitment_state) = PCS::commit(&params.pcs_params, &bit_poly);
        trans.append_message(b"[Commit Phase]", &commitment);
        info!(
            "[P]-[PCS] Committing to a polynomial of {} variables in {:?}",
            bit_poly.num_vars(),
            pcs_commit_time.elapsed()
        );
        if let Some(stats) = statistics {
            stats.add_prover_pcs_time(pcs_commit_time.elapsed());
        }

        // [PIOP Phase] extract all the Hadamard instances and prove them via one single sumcheck
        let piop_hadamard_time = std::time::Instant::now();
        let trace_ef = trace_mle.to_ef();

        // prepare Hadamard instances
        let hadamard_instances = SumHadamardInstance::from(&trace_ef);
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
        let mut trace_evals = SumHadamardTraceEval::<EF>::default();
        trace_mle.evaluate_ef_ntt_only(
            &mut trace_evals,
            &sumcheck_state.randomness,
            &trace_ef,
            &sumcheck_claim.poly,
            &eval_table,
        );
        let hadamard_eval_proof = BatchedSumHadamardProof::from_hadamard_trace_eval(&trace_evals);
        trans.append_message(b"[Hadamard Evals]", &hadamard_eval_proof);

        info!(
            "[P]-[PIOP] Proving Hadamard via Sumcheck in {:?}",
            piop_hadamard_time.elapsed()
        );

        // [PIOP Phase] prove the validity of NTT evaluations since we consider all NTT oracles as virtual oracles
        let piop_ntt_time = std::time::Instant::now();
        let point_u = sumcheck_state.randomness[..trace_mle.log_coeff_count].to_vec();
        let point_v = sumcheck_state.randomness[trace_mle.log_coeff_count..].to_vec();

        // parepare the NTT equality instance for normal polynomials used in Hadamard, where the coefficient matrix is dense
        let point_bit_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            trace_mle.log_num_oracles(),
        );
        let bit_poly = Rc::new(bit_poly.to_ef());
        let bit_ntt_evals = trace_evals.pack_ntt_to_vec();
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
        // prove both NTT instances in one sumcheck protocol
        let (ntt_proof, ntt_state) = NTTMatrixEvalIOP::prover(trans, &ntt_dense_instance);
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

        HadamardProductProof {
            log_coeff_count: trace_mle.log_coeff_count,
            log_num_oracle: trace_mle.log_num_oracles(),
            pcs_params: params.pcs_params.clone(),
            commitment,
            sumcheck_poly_info: sumcheck_claim.poly.info(),
            sumcheck_proof,
            hadamard_info: hadamard_instance_infos,
            hadamard_proof: hadamard_eval_proof,
            ntt_info: ntt_dense_instance.info(),
            ntt_proof,
            eval_proof,
            trace_evals,
        }
    }

    pub fn verify_as_subprotocol(
        trans: &mut Transcript<EF>,
        proof: &HadamardProductProof<F, EF, S, PCS>,
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
        let point_u = sumcheck_subclaim.point[..proof.log_coeff_count].to_vec();
        let mut point_v = sumcheck_subclaim.point[proof.log_coeff_count..].to_vec();
        let point_bit_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            proof.log_num_oracle,
        );

        point_v.extend_from_slice(&point_bit_oracle);
        let (ntt_res, ntt_subclaim) =
            NTTMatrixEvalIOP::verifier(trans, &proof.ntt_info, &proof.ntt_proof);
        let open_evals = proof.ntt_proof.coeff_eval_at_r_v;
        res &= ntt_res;
        assert!(res, "NTT Equality verification failed.");
        info!(
            "[V]-[PIOP] Verifying NTT Equality in {:?}",
            piop_ntt_time.elapsed()
        );

        // [PCS Phase] Verify the opening proof for the dense coeffcient matrix
        // evaluation `ntt_proof.coeff_eval_at_r_v[1]` at point_r_v_prime
        let pcs_poly_open_time = std::time::Instant::now();
        let mut open_point = Vec::with_capacity(ntt_subclaim.randomness.len() + point_v.len());
        open_point.extend_from_slice(&ntt_subclaim.randomness);
        open_point.extend_from_slice(&point_v);
        let eval_res = PCS::verify(
            &proof.pcs_params,
            &proof.commitment,
            &open_point,
            open_evals,
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

        res
    }
}
