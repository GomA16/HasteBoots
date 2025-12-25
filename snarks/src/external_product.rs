//! This snarks implementation includes the proof generation for Hadamard product
//! along with all NTT evaluations.
//!
//! When considering the multiplication-related relation between polynomials,
//! we are able to use Hadamard product to represent the element-wise relation
//! of their NTT evaluations.
//!
//! To reduce the elements to be committed as more as possible and also to simplify
//! the proof structure, we only commit to the coefficient form of the polynomials.
//! After running the protocol for Hadamard product, it is reduced to querying the
//! evaluations of these polynomials at some random points.
//! All these queries are answered by the NTT PIOP, reducing to the queries of
//! their coefficient forms.
use std::rc::Rc;

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field, PolynomialInfo};
use bincode::config::standard;
use bincode::{Decode, Encode};
use helper::utils::{compute_oracle_evals, eval_identity_function};
use helper::{FiatShamirTranscript, Transcript};
use pcs::PolynomialCommitmentScheme;
use piop::hadamard::{
    BatchedSumHadamardInfo, BatchedSumHadamardInstance, BatchedSumHadamardProof, HadamardPIOP,
};
use piop::lookup::small_table::{LogUpIOP, LogUpInstance, LogUpInstanceInfo, LogUpProof};
use piop::ntt::{NTTMatrixEvalIOP, NTTMatrixEvalInfo, NTTMatrixEvalInstance, NTTMatrixEvalProof};
use piop::{LagrangeKernel, SumcheckClaim, SumcheckInfo, SumcheckInstance, SumcheckPIOP};
use serde::{Serialize, ser};
use sumcheck::{MLSumcheck, Proof};
use trace::lookup_trace::small_table::LookupWitnessHelperEval;
use trace::{ConvertToEF, EvaluableTraceEF, PackableTrace, SumHadamardTraceEval};
use trace::{EvaluableTrace, PackableEval, SumHadamardTraceMLE};

#[derive(Default)]
pub struct ExternalProductSnarks<F, EF, S, PCS>
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

pub struct ExternalProductParams<F, EF, S, PCS>
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

impl<F, EF, S, PCS> ExternalProductParams<F, EF, S, PCS>
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
        trace: &SumHadamardTraceMLE<F>,
    ) -> Self {
        let oracle_num_vars = trace.num_vars() + trace.log_num_all_poly();
        let pcs_params = PCS::setup(oracle_num_vars, Some(code_spec.clone()));
        let helper_num_vars = trace.num_vars() + trace.log_num_helper_poly(blk_size);
        let pcs_params_ef = PCS::setup(helper_num_vars, Some(code_spec.clone()));

        ExternalProductParams {
            blk_size,
            basis,
            pcs_params,
            pcs_params_ef,
            ntt_table: Rc::new(ntt_table.to_ef()),
        }
    }
}

#[derive(Serialize)]
pub struct ExternalProductProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub log_coeff_count: usize,
    pub log_num_all_poly: usize,
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
    pub hadamard_info: BatchedSumHadamardInfo<EF>,
    pub hadamard_proof: BatchedSumHadamardProof<EF>,
    pub ntt_info: NTTMatrixEvalInfo<EF>,
    pub ntt_proof: NTTMatrixEvalProof<EF>,
    #[serde(skip)]
    pub eval_proof: Vec<PCS::Proof>,
    #[serde(skip)]
    pub eval_ef_proof: PCS::ProofEF,
    // Redudant fields for ease of implementation
    #[serde(skip)]
    pub trace_evals: SumHadamardTraceEval<EF>,
    #[serde(skip)]
    pub helper_evals: LookupWitnessHelperEval<EF>,
}

impl<F, EF, S, PCS> ExternalProductProof<F, EF, S, PCS>
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
        bincode::serde::encode_to_vec(&self.sumcheck_poly_info, standard())
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
            + bincode::serde::encode_to_vec(&self.ntt_info, standard())
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

impl<F, EF, S, PCS> ExternalProductSnarks<F, EF, S, PCS>
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
        params: &ExternalProductParams<F, EF, S, PCS>,
    ) -> ExternalProductProof<F, EF, S, PCS> {
        let time = std::time::Instant::now();
        // Commit to the trace polynomial
        let bit_poly = trace_mle.generate_all_oracle();
        let (commitment, commitment_state) = PCS::commit(&params.pcs_params, &bit_poly);
        trans.append_message(b"[Commit Phase]", &commitment);

        // Commit to the helper polynomial
        let random_value =
            trans.get_challenge(b"[Challenge] random value used in the rational identity");
        let lookup_trace = trace_mle.extract_lookup_trace_mle_small_table(params.basis);
        let lookup_witness = lookup_trace.compute_witness_pure();
        let lookup_helper = lookup_trace.compute_helper_functions_ef::<EF>(
            params.blk_size,
            random_value,
            &lookup_witness,
        );
        let helper_poly = lookup_helper.generate_oracle();
        let (helper_commitment, helper_commitment_state) =
            PCS::commit_ef(&params.pcs_params_ef, &helper_poly);
        println!("Commit Phase time: {:?}", time.elapsed());

        // PIOP Phase
        let time = std::time::Instant::now();
        let trace_ef = trace_mle.to_ef();
        let lookup_trace_ef = lookup_trace.to_ef();
        let hadamard_instance = BatchedSumHadamardInstance::from(&trace_ef);
        let lookup_instance = LogUpInstance::from(&lookup_trace_ef, &lookup_helper);
        let hadamard_instance_info = hadamard_instance.info();
        trans.append_message(b"[Hadamard Statement]", &hadamard_instance_info);
        let lookup_instance_info = lookup_instance.info();
        trans.append_message(b"[Lookup Statement]", &lookup_instance_info);

        assert_eq!(
            hadamard_instance_info.sumcheck_num_vars(),
            lookup_instance_info.sumcheck_num_vars()
        );
        let sumcheck_num_vars = hadamard_instance_info.sumcheck_num_vars();
        let mut sumcheck_claim = SumcheckClaim::new(sumcheck_num_vars);
        let lagrange_kernel = Some(&LagrangeKernel::random(trans, sumcheck_num_vars));

        // Combine all sumcheck from Hadamard protocol into one
        let randomness_hadamard = hadamard_instance_info.sample_randomness_for_sumcheck(trans);
        HadamardPIOP::prover_batch_sumcheck(
            &hadamard_instance,
            &mut sumcheck_claim,
            &randomness_hadamard,
            lagrange_kernel,
        );
        // Combine all sumcheck form Lookup protocol into one
        let randomness_lookup = lookup_instance_info.sample_randomness_for_sumcheck(trans);
        LogUpIOP::prover_batch_sumcheck(
            &lookup_instance,
            &mut sumcheck_claim,
            &randomness_lookup,
            lagrange_kernel,
        );
        // Execute the sumcheck protocol
        let (sumcheck_proof, sumcheck_state) = MLSumcheck::prove(trans, &sumcheck_claim.poly)
            .expect("[External Product PIOP] Fail to generate sumcheck proof.");
        trans.append_message(b"[Sumcheck Protocol]", &sumcheck_proof);
        // Evaluate the polynomials and append them into proof
        let trace_evals = trace_mle.evaluate_ef(&sumcheck_state.randomness);
        let helper_evals = lookup_helper.evaluate(&sumcheck_state.randomness);
        let hadamard_eval_proof = BatchedSumHadamardProof::from_hadamard_trace_eval(&trace_evals);
        let lookup_eval_proof =
            LogUpProof::from_hadamard_trace_eval(&trace_evals, &helper_evals, random_value);
        trans.append_message(b"[Hadamard Evals]", &hadamard_eval_proof);
        trans.append_message(b"[Lookup Evals]", &lookup_eval_proof);

        let point_u = sumcheck_state.randomness[..trace_mle.log_coeff_count].to_vec();
        let mut point_v = sumcheck_state.randomness[trace_mle.log_coeff_count..].to_vec();
        let point_bit_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            trace_mle.log_num_all_poly(),
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
        open_point_1.extend_from_slice(&point_u);
        open_point_1.extend_from_slice(&point_v);
        open_point_1.extend_from_slice(&point_bit_oracle);
        open_point_ef.extend_from_slice(&sumcheck_state.randomness);
        open_point_ef.extend_from_slice(&point_helper_oracle);

        // Reduced subclaim from Hadamard: query the evluation polynomial at `open_point_1`,
        // which can be further proven by NTT PIOP
        let bit_poly = Rc::new(bit_poly.to_ef());
        let bit_ntt_evals = trace_evals.pack_all_ntt_to_vec();
        let eval = compute_oracle_evals(&bit_ntt_evals, &point_bit_oracle);

        point_v.extend_from_slice(&point_bit_oracle);
        let ntt_instance = NTTMatrixEvalInstance::from_subclaim(
            &bit_poly,
            &params.ntt_table,
            &point_u,
            &point_v,
            eval,
        );
        let (ntt_piop_proof, ntt_piop_state) = NTTMatrixEvalIOP::prover(trans, &ntt_instance);
        // let open_eval_2 = ntt_piop_proof.coeff_eval_at_r_v;

        trans.append_message(b"[PIOP Phase]", &ntt_piop_proof);

        println!("PIOP Phase time: {:?}", time.elapsed());

        let time = std::time::Instant::now();
        let mut open_point_2 = Vec::with_capacity(ntt_piop_state.point_r.len() + point_v.len());
        open_point_2.extend_from_slice(&ntt_piop_state.point_r);
        open_point_2.extend_from_slice(&point_v);
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

        println!("PCS Opening Phase time: {:?}", time.elapsed());

        ExternalProductProof {
            log_coeff_count: trace_mle.log_coeff_count,
            log_num_all_poly: trace_mle.log_num_all_poly(),
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
            ntt_info: ntt_instance.info(),
            ntt_proof: ntt_piop_proof,
            eval_proof,
            // eval_proof_1,
            // eval_proof_2,
            eval_ef_proof,
            trace_evals,
            helper_evals,
        }
    }

    pub fn verify(
        &self,
        trans: &mut Transcript<EF>,
        proof: &ExternalProductProof<F, EF, S, PCS>,
    ) -> bool {
        trans.append_message(b"[Commit Phase]", &proof.commitment);
        let mut res = true;

        let random_value =
            trans.get_challenge(b"[Challenge] random value used in the rational identity");

        let time = std::time::Instant::now();
        // PIOP Phase
        trans.append_message(b"[Hadamard Statement]", &proof.hadamard_info);
        trans.append_message(b"[Lookup Statement]", &proof.lookup_info);

        assert_eq!(
            proof.hadamard_info.sumcheck_num_vars(),
            proof.lookup_info.sumcheck_num_vars()
        );
        let sumcheck_num_vars = proof.hadamard_info.sumcheck_num_vars();
        let lagrange_point = LagrangeKernel::random_point(trans, sumcheck_num_vars);

        // Combine all sumcheck from Hadamard protocol into one
        let randomness_hadamard = proof.hadamard_info.sample_randomness_for_sumcheck(trans);
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
        HadamardPIOP::verifier_compute_subclaim(
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

        let point_u = sumcheck_subclaim.point[..proof.log_coeff_count].to_vec();
        let mut point_v = sumcheck_subclaim.point[proof.log_coeff_count..].to_vec();
        let point_bit_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            proof.log_num_all_poly,
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

        let bit_poly_evals = proof.trace_evals.pack_all_poly_to_vec();
        let open_eval_1 = compute_oracle_evals(&bit_poly_evals, &point_bit_oracle);

        let helper_poly_evals = proof.helper_evals.pack_to_vec();
        let open_helper_eval = compute_oracle_evals(&helper_poly_evals, &point_helper_oracle);

        point_v.extend_from_slice(&point_bit_oracle);
        let (ntt_res, ntt_subclaim) =
            NTTMatrixEvalIOP::verifier(trans, &proof.ntt_info, &proof.ntt_proof);
        let open_eval_2 = proof.ntt_proof.coeff_eval_at_r_v;
        trans.append_message(b"[PIOP Phase]", &proof.ntt_proof);
        res &= ntt_res;

        println!("PIOP Phase time: {:?}", time.elapsed());

        let time = std::time::Instant::now();
        let mut open_point_2 = Vec::with_capacity(ntt_subclaim.point_r.len() + point_v.len());
        open_point_2.extend_from_slice(&ntt_subclaim.point_r);
        open_point_2.extend_from_slice(&point_v);

        let open_points = vec![open_point_1, open_point_2];
        let open_evals = vec![open_eval_1, open_eval_2];

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

        let eval_ef_res = PCS::verify_ef(
            &proof.pcs_params_ef,
            &proof.helper_commitment,
            &open_point_ef,
            open_helper_eval,
            &proof.eval_ef_proof,
            trans,
        );
        res &= eval_ef_res;

        println!("PCS Opening Phase time: {:?}", time.elapsed());

        res
    }
}
