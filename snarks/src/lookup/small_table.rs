use core::time;

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};
use helper::utils::compute_oracle_evals;
use helper::{FiatShamirTranscript, Transcript};
use pcs::PolynomialCommitmentScheme;
use pcs::utils::code;
use piop::lookup::small_table::LogUpInstanceInfo;
use piop::lookup::small_table::{LogUpIOP, LogUpInstance, LogUpProof};
use piop::{SumcheckInstance, SumcheckPIOP};
use serde::Serialize;
use trace::{EvaluableTrace, EvaluableTraceEF, PackableEval, PackableTrace};
use trace::{ConvertToEF};
use trace::lookup_trace::small_table::{LookupTrace, LookupTraceEval, LookupTraceMLE, LookupWitnessHelper, LookupWitnessHelperEval};

#[derive(Default)]
pub struct LogUpSnarks<F, EF, S, PCS>
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
pub struct LogUpParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub blk_size: usize,
    pub pcs_params: PCS::Parameters,
    pub pcs_params_ef: PCS::Parameters,
}

impl<F, EF, S, PCS> LogUpParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub fn new(code_spec: S, blk_size: usize, trace: &LookupTraceMLE<F>) -> Self {
        Self {
            // code_spec,
            blk_size,
            pcs_params: PCS::setup(trace.num_vars() + trace.log_num_oracles(), Some(code_spec.clone())),
            pcs_params_ef: PCS::setup(trace.num_vars() + trace.log_num_helper_oracles(blk_size), Some(code_spec.clone())),
        }
    }
}

impl<F, EF, S, PCS> Clone for LogUpParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
    PCS::Parameters: Clone,
{
    fn clone(&self) -> Self {
        Self {
            blk_size: self.blk_size,
            pcs_params: self.pcs_params.clone(),
            pcs_params_ef: self.pcs_params_ef.clone(),
        }
    }
}

pub struct LogUpSnarksProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub params: LogUpParams<F, EF, S, PCS>,
    pub trace_commitment: PCS::Commitment,
    pub helper_commitment: PCS::Commitment,
    pub info: LogUpInstanceInfo<EF>,
    pub piop_proof: LogUpProof<EF>,
    pub eval_trace_proof: PCS::Proof,
    pub eval_helper_proof: PCS::ProofEF,
    // redudant info for ease of verification
    pub log_num_oracles: usize,
    pub log_num_helper_oracles: usize,
    pub trace_eval: LookupTraceEval<EF>,
    pub helper_eval: LookupWitnessHelperEval<EF>,
}

impl<F, EF, S, PCS> LogUpSnarks<F, EF, S, PCS>
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
    pub fn commit_witness(
        &self,
        params: &LogUpParams<F, EF, S, PCS>,
        trace: &LookupTraceMLE<F>,
    ) -> (PCS::Commitment, PCS::CommitmentState) {
        let trace_poly = trace.generate_oracle();
        PCS::commit(&params.pcs_params, &trace_poly)
    }

    pub fn commit_witness_ef(
        &self,
        params: &LogUpParams<F, EF, S, PCS>,
        witness: &LookupWitnessHelper<EF>,
    ) -> (PCS::Commitment, PCS::CommitmentStateEF) {
        let witness_poly = witness.generate_oracle();
        PCS::commit_ef(&params.pcs_params_ef, &witness_poly)
    }

    pub fn compute_oracle_evaluation(&self, eval: &LookupTraceEval<EF>, point: &[EF]) -> EF {
        let reduced_eval = eval.pack_to_vec();
        compute_oracle_evals(&reduced_eval, point)
    }

    pub fn compute_oracle_evaluation_ef(&self, eval: &LookupWitnessHelperEval<EF>, point: &[EF]) -> EF {
        let reduced_eval = eval.pack_to_vec();
        compute_oracle_evals(&reduced_eval, point)
    }

    pub fn prove(
        &self,
        trans: &mut Transcript<EF>,
        trace_mle: LookupTraceMLE<F>,
        params: &mut LogUpParams<F, EF, S, PCS>,
    ) -> LogUpSnarksProof<F, EF, S, PCS> {
        let witness = trace_mle.compute_witness_pure();

        // Commit to the trace polynomial
        let (trace_commitment, commitment_state) = self.commit_witness(params, &trace_mle);
        trans.append_message(b"[Commit Phase]", &trace_commitment);

        // Commit to the helper polynomial
        let random_value =
            trans.get_challenge(b"[Challenge] random value used in the rational identity");
        let helper = trace_mle.compute_helper_functions_ef::<EF>(params.blk_size, random_value, &witness);
        let (helper_commitment, helper_commitment_state) = self.commit_witness_ef(params, &helper);
        trans.append_message(b"[Commit Phase]", &helper_commitment);

        // PIOP Phase
        let trace_ef = trace_mle.to_ef();
        let instance = LogUpInstance::from(&trace_ef, &helper);
        // let (piop_proof, piop_state) = LogUpIOP::prover(trans, &instance);
        // Prover can also use the following line to separate the sumcheck proof generation from the evaluation phase.
        let (mut piop_proof, piop_state) = LogUpIOP::prover_without_evals(trans, &instance);

        let trace_eval = trace_mle.evaluate_ef(&piop_state.point_r);
        let helper_eval = helper.evaluate(&piop_state.point_r);
        piop_proof.append_eval(&trace_eval, &helper_eval, random_value);
        trans.append_message(b"[PIOP Phase]", &piop_proof);

        // Reduce the evaluations to one evaluation on the oracle
        let point_trace_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            trace_mle.log_num_oracles(),
        );
        let point_helper_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            trace_mle.log_num_helper_oracles(params.blk_size),
        );

        let oracle_evaluation = self.compute_oracle_evaluation(&trace_eval, &point_trace_oracle);
        trans.append_message(b"[PIOP Phase]", &oracle_evaluation);
        let oracle_evaluation_ef =
            self.compute_oracle_evaluation_ef(&helper_eval, &point_helper_oracle);
        trans.append_message(b"[PIOP Phase]", &oracle_evaluation_ef);

        // PCS Phase
        let eval_trace_proof = PCS::open(
            &params.pcs_params,
            &trace_commitment,
            &commitment_state,
            &piop_state
                .point_r
                .iter()
                .chain(point_trace_oracle.iter())
                .copied()
                .collect::<Vec<_>>(),
            trans,
        );
        let eval_helper_proof = PCS::open_ef(
            &params.pcs_params_ef,
            &helper_commitment,
            &helper_commitment_state,
            &piop_state
                .point_r
                .iter()
                .chain(point_helper_oracle.iter())
                .copied()
                .collect::<Vec<_>>(),
            trans,
        );

        LogUpSnarksProof {
            params: params.clone(),
            trace_commitment,
            helper_commitment,
            info: instance.info(),
            piop_proof,
            log_num_oracles: trace_mle.log_num_oracles(),
            log_num_helper_oracles: trace_mle.log_num_helper_oracles(params.blk_size),
            eval_trace_proof,
            eval_helper_proof,
            trace_eval,
            helper_eval,
        }
    }

    pub fn verifier(
        &self,
        trans: &mut Transcript<EF>,
        proof: &LogUpSnarksProof<F, EF, S, PCS>,
    ) -> bool {
        let mut res = true;

        trans.append_message(b"[Commit Phase]", &proof.trace_commitment);

        let random_value =
            trans.get_challenge(b"[Challenge] random value used in the rational identity");
        assert_eq!(random_value, proof.info.random_value);

        trans.append_message(b"[Commit Phase]", &proof.helper_commitment);

        let (piop_res, piop_subclaim) = LogUpIOP::verifier(trans, &proof.info, &proof.piop_proof);
        trans.append_message(b"[PIOP Phase]", &proof.piop_proof);
        res &= piop_res;

        // Reduce the evaluations to one evaluation on the oracle
        let point_trace_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            proof.log_num_oracles,
        );
        let point_helper_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            proof.log_num_helper_oracles,
        );

        let oracle_evaluation =
            self.compute_oracle_evaluation(&proof.trace_eval, &point_trace_oracle);
        trans.append_message(b"[PIOP Phase]", &oracle_evaluation);
        let oracle_evaluation_ef =
            self.compute_oracle_evaluation_ef(&proof.helper_eval, &point_helper_oracle);
        trans.append_message(b"[PIOP Phase]", &oracle_evaluation_ef);

        let pcs_res = PCS::verify(
            &proof.params.pcs_params,
            &proof.trace_commitment,
            &piop_subclaim
                .point_r
                .iter()
                .chain(point_trace_oracle.iter())
                .copied()
                .collect::<Vec<_>>(),
            oracle_evaluation,
            &proof.eval_trace_proof,
            trans,
        );
        let pcs_res_ef = PCS::verify_ef(
            &proof.params.pcs_params_ef,
            &proof.helper_commitment,
            &piop_subclaim
                .point_r
                .iter()
                .chain(point_helper_oracle.iter())
                .copied()
                .collect::<Vec<_>>(),
            oracle_evaluation_ef,
            &proof.eval_helper_proof,
            trans,
        );
        res &= pcs_res && pcs_res_ef;
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
    use trace::lookup_trace::small_table::LookupTrace;

    type FF = BabyBear;
    type EF = BabyBearExetension;
    type Hash = sha2::Sha256;
    const BASE_FIELD_BITS: usize = 31;

    #[test]
    fn test_logup_snarks() {
        let mut rng = rand::rng();
        let num_vars = 10;
        let num_vec = 2;
        let range = 8;
        let blk_size = 2;

        let lookup_trace = LookupTrace::<FF>::random(&mut rng, num_vars, num_vec, range);
        let trace: LookupTraceMLE<_> = lookup_trace.into();
        let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
        let snarks = LogUpSnarks::<
            FF,
            EF,
            ExpanderCodeSpec,
            BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
        >::default();
        let params = &mut LogUpParams::new(code_spec, blk_size, &trace);

        let prover_trans = &mut Transcript::<EF>::default();
        let proof = snarks.prove(prover_trans, trace, params);

        let verifier_trans = &mut Transcript::<EF>::default();
        let res = snarks.verifier(verifier_trans, &proof);
        assert!(res);
    }
}
