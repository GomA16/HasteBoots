use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};
use helper::utils::compute_oracle_evals;
use helper::{FiatShamirTranscript, Transcript};
use pcs::PolynomialCommitmentScheme;
use piop::lookup::logup::LogUpInstanceInfo;
use piop::lookup::{LogUpIOP, LogUpInstance, LogUpProof};
use piop::{PackableEFProof, PackableProof, SumcheckInstance, SumcheckPIOP};
use serde::Serialize;
use trace::PackableTrace;
use trace::{ConvertToEF, LookupTrace, LookupTraceMLE, LookupWitness, LookupWitnessHelper};

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
    pub code_spec: S,
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
    pub fn new(code_spec: S, blk_size: usize) -> Self {
        Self {
            code_spec,
            blk_size,
            pcs_params: PCS::Parameters::default(),
            pcs_params_ef: PCS::Parameters::default(),
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
            code_spec: self.code_spec.clone(),
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
        params: &mut LogUpParams<F, EF, S, PCS>,
        witness: &LookupWitness<F>,
    ) -> (PCS::Commitment, PCS::CommitmentState) {
        let witness_poly = witness.generate_oracle();
        let pcs_params = PCS::setup(witness_poly.num_vars, Some(params.code_spec.clone()));
        params.pcs_params = pcs_params.clone();
        PCS::commit(&pcs_params, &witness_poly)
    }

    pub fn commit_witness_ef(
        &self,
        params: &mut LogUpParams<F, EF, S, PCS>,
        witness: &LookupWitnessHelper<EF>,
    ) -> (PCS::Commitment, PCS::CommitmentStateEF) {
        let witness_poly = witness.generate_oracle();
        let pcs_params = PCS::setup(witness_poly.num_vars, Some(params.code_spec.clone()));
        params.pcs_params_ef = pcs_params.clone();
        PCS::commit_ef(&pcs_params, &witness_poly)
    }

    pub fn compute_oracle_evaluation(&self, proof: &LogUpProof<EF>, point: &[EF]) -> EF {
        let evals = proof.pack_to_vec();
        compute_oracle_evals(&evals, point)
    }

    pub fn compute_oracle_evaluation_ef(&self, proof: &LogUpProof<EF>, point: &[EF]) -> EF {
        let evals = proof.pack_to_vec_ef();
        compute_oracle_evals(&evals, point)
    }

    pub fn prove(
        &self,
        trans: &mut Transcript<EF>,
        trace: LookupTrace<F>,
        params: &mut LogUpParams<F, EF, S, PCS>,
        // oracle: &EvalOracle<F, EF, S, PCS>,
    ) -> LogUpSnarksProof<F, EF, S, PCS> {
        let trace_mle: LookupTraceMLE<F> = trace.into();
        let witness: LookupWitness<F> = trace_mle.into();

        // Commit to the trace polynomial
        let (trace_commitment, commitment_state) = self.commit_witness(params, &witness);
        trans.append_message(b"[Commit Phase]", &trace_commitment);

        // Commit to the helper polynomial
        let random_value =
            trans.get_challenge(b"[Challenge] random value used in the rational identity");
        let witness_ef: LookupWitness<EF> = witness.to_ef();
        let helper = witness_ef.compute_helper_functions(params.blk_size, random_value);
        let (helper_commitment, helper_commitment_state) = self.commit_witness_ef(params, &helper);
        trans.append_message(b"[Commit Phase]", &helper_commitment);

        // PIOP Phase
        let instance = LogUpInstance::from(&witness_ef, &helper);
        let (piop_proof, piop_state) = LogUpIOP::prover(trans, &instance);
        trans.append_message(b"[PIOP Phase]", &piop_proof);

        // Reduce the evaluations to one evaluation on the oracle
        let point_trace_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            piop_proof.log_num_evals(),
        );
        let point_helper_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            piop_proof.log_num_evals_ef(),
        );

        let oracle_evaluation = self.compute_oracle_evaluation(&piop_proof, &point_trace_oracle);
        trans.append_message(b"[PIOP Phase]", &oracle_evaluation);
        let oracle_evaluation_ef =
            self.compute_oracle_evaluation_ef(&piop_proof, &point_helper_oracle);
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
            eval_trace_proof,
            eval_helper_proof,
        }
    }

    fn verifier(
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
            proof.piop_proof.log_num_evals(),
        );
        let point_helper_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            proof.piop_proof.log_num_evals_ef(),
        );

        let oracle_evaluation =
            self.compute_oracle_evaluation(&proof.piop_proof, &point_trace_oracle);
        trans.append_message(b"[PIOP Phase]", &oracle_evaluation);
        let oracle_evaluation_ef =
            self.compute_oracle_evaluation_ef(&proof.piop_proof, &point_helper_oracle);
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
    use bincode::config::standard;
    use helper::Transcript;
    use pcs::{
        multilinear::BrakedownPCS,
        utils::code::{ExpanderCode, ExpanderCodeSpec},
    };
    use trace::LookupTrace;

    type FF = BabyBear;
    type EF = BabyBearExetension;
    type Hash = sha2::Sha256;
    const BASE_FIELD_BITS: usize = 31;

    #[test]
    fn test_logup_snarks() {
        let mut rng = rand::rng();
        let num_vars = 3;
        let num_vec = 2;
        let range = 8;
        let blk_size = 1;

        let lookup_trace = LookupTrace::<FF>::random(&mut rng, num_vars, num_vec, range);
        let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
        let snarks = LogUpSnarks::<
            FF,
            EF,
            ExpanderCodeSpec,
            BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
        >::default();
        let params = &mut LogUpParams::new(code_spec, blk_size);

        let prover_trans = &mut Transcript::<EF>::default();
        let proof = snarks.prove(prover_trans, lookup_trace, params);

        let verifier_trans = &mut Transcript::<EF>::default();
        let res = snarks.verifier(verifier_trans, &proof);
        assert!(res);
    }
}
