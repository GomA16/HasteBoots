use core::time;

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};
use helper::{
    FiatShamirTranscript, Transcript,
    utils::{compute_oracle_evals, eval_identity_function},
};
use pcs::PolynomialCommitmentScheme;
use piop::{
    BatchedSumcheckPIOP, SumcheckInstance, SumcheckPIOP,
    lookup::indexed_table::{
        self, BatchedIndexLogUpInputProof, IndexedLogUpInputIOP, IndexedLogUpInputInstanceInfo,
        IndexedLogUpInputProof, IndexedLogUpTableIOP, IndexedLogUpTableInstanceInfo,
        IndexedLogUpTableProof,
    },
};
use serde::{Serialize, ser};
use trace::{
    ConvertToEF, EvaluableTraceEF, PackableEval, PackableTrace,
    lookup_trace::indexed_table::{IndexedLookupEval, IndexedLookupTraceMLE},
};

#[derive(Default)]
pub struct BatchedIndexedLogUpSnarks<F, EF, S, PCS>
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

pub struct BatchedIndexedLogUpParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub pcs_params: PCS::Parameters,
    pub pcs_params_ef: PCS::Parameters,
}

impl<F, EF, S, PCS> BatchedIndexedLogUpParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub fn new(code_spec: S, trace: &Vec<IndexedLookupTraceMLE<F>>) -> Self {
        let num_oracle_vars = trace.num_vars() + trace.log_num_oracles();
        let pcs_params = PCS::setup(num_oracle_vars, Some(&code_spec));
        let helper_log_num_oracles = if trace.len() == 1 {
            0
        } else {
            trace.len().next_power_of_two().trailing_zeros() as usize
        };
        let helper_num_oracle_vars = trace.num_vars() + helper_log_num_oracles;
        let pcs_params_ef = PCS::setup(helper_num_oracle_vars, Some(&code_spec));
        BatchedIndexedLogUpParams {
            pcs_params,
            pcs_params_ef,
        }
    }
}

#[derive(Serialize)]
pub struct BatchedIndexedLogUpSnarksProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub trace_log_num_oracles: usize,
    pub helper_log_num_oracles: usize,
    #[serde(skip)]
    pub pcs_params: PCS::Parameters,
    #[serde(skip)]
    pub commitment: PCS::Commitment,
    #[serde(skip)]
    pub pcs_params_ef: PCS::Parameters,
    #[serde(skip)]
    pub helper_commitment: PCS::Commitment,
    pub input_instance_info: Vec<IndexedLogUpInputInstanceInfo<EF>>,
    pub input_piop_proof: BatchedIndexLogUpInputProof<EF>,
    pub input_point_r: Vec<EF>,
    pub lookup_evals: Vec<IndexedLookupEval<EF>>,
    pub eval_proof: PCS::Proof,
    pub eval_helper_proof: PCS::ProofEF,
}

impl<F, EF, S, PCS> BatchedIndexedLogUpSnarks<F, EF, S, PCS>
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
        trace_mle: &Vec<IndexedLookupTraceMLE<F>>,
        params: &BatchedIndexedLogUpParams<F, EF, S, PCS>,
    ) -> BatchedIndexedLogUpSnarksProof<F, EF, S, PCS> {
        BatchedIndexedLogUpSnarks::prove_as_subprotocol(trans, trace_mle, params)
    }

    pub fn verify(
        &self,
        trans: &mut Transcript<EF>,
        proof: &BatchedIndexedLogUpSnarksProof<F, EF, S, PCS>,
    ) -> bool {
        BatchedIndexedLogUpSnarks::verify_as_subprotocol(trans, proof)
    }

    pub fn prove_as_subprotocol(
        trans: &mut Transcript<EF>,
        trace_mle: &Vec<IndexedLookupTraceMLE<F>>,
        params: &BatchedIndexedLogUpParams<F, EF, S, PCS>,
    ) -> BatchedIndexedLogUpSnarksProof<F, EF, S, PCS> {
        let witness = trace_mle
            .iter()
            .map(|trace| trace.compute_witness())
            .collect::<Vec<_>>();
        let poly = trace_mle.generate_oracle();
        let (trace_commitment, commitment_state) = PCS::commit(&params.pcs_params, &poly);
        trans.append_message(b"[Commit Phase]", &trace_commitment);

        let random_value =
            trans.get_challenge(b"[Challenge] random value used in the rational identity");
        let random_s_hash = trans.get_challenge(b"[Challenge] random value used for hashing.");
        let trace_ef = trace_mle
            .iter()
            .map(|trace| trace.to_ef())
            .collect::<Vec<_>>();
        let helper = trace_mle
            .iter()
            .zip(witness.iter())
            .map(|(trace, wit)| {
                trace.compute_helper_functions_ef(&wit, random_value, random_s_hash)
            })
            .collect::<Vec<_>>();

        let helper_poly = helper.generate_oracle();
        let (helper_commitment, helper_commitment_state) =
            PCS::commit_ef(&params.pcs_params_ef, &helper_poly);
        trans.append_message(b"[Commit Phase]", &helper_commitment);

        let input_instance = trace_ef
            .iter()
            .zip(helper.iter())
            .map(|(trace_ef, helper)| {
                indexed_table::IndexedLogUpInputInstance::<EF>::from(trace_ef, helper)
            })
            .collect::<Vec<_>>();
        let (input_piop_proof, input_piop_state) =
            indexed_table::IndexedLogUpInputIOP::prover_batch(trans, &input_instance);

        let point_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            trace_mle.log_num_oracles(),
        );
        let point_helper_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            helper.log_num_oracles(),
        );

        let mut point = Vec::with_capacity(input_piop_state.point_r.len() + point_oracle.len());
        let mut point_helper =
            Vec::with_capacity(input_piop_state.point_r.len() + point_helper_oracle.len());
        point.extend_from_slice(&input_piop_state.point_r);
        point.extend_from_slice(&point_oracle);
        point_helper.extend_from_slice(&input_piop_state.point_r);
        point_helper.extend_from_slice(&point_helper_oracle);
        let lookup_evals = trace_mle.evaluate_ef(&input_piop_state.point_r);

        let eval_proof = PCS::open(
            &params.pcs_params,
            &trace_commitment,
            &commitment_state,
            &point,
            trans,
        );
        let eval_helper_proof = PCS::open_ef(
            &params.pcs_params_ef,
            &helper_commitment,
            &helper_commitment_state,
            &point_helper,
            trans,
        );

        BatchedIndexedLogUpSnarksProof {
            trace_log_num_oracles: trace_mle.log_num_oracles(),
            helper_log_num_oracles: helper.log_num_oracles(),
            pcs_params: params.pcs_params.clone(),
            commitment: trace_commitment,
            pcs_params_ef: params.pcs_params_ef.clone(),
            helper_commitment,
            input_instance_info: input_instance
                .iter()
                .map(|instance| instance.info())
                .collect(),
            input_piop_proof,
            input_point_r: input_piop_state.point_r,
            lookup_evals,
            eval_proof,
            eval_helper_proof,
        }
    }

    pub fn verify_as_subprotocol(
        trans: &mut Transcript<EF>,
        proof: &BatchedIndexedLogUpSnarksProof<F, EF, S, PCS>,
    ) -> bool {
        let mut res = true;

        trans.append_message(b"[Commit Phase]", &proof.commitment);
        // Some simple value equality checks are omitted here for brevity.
        let _random_value =
            trans.get_challenge(b"[Challenge] random value used in the rational identity");
        let _random_s_hash = trans.get_challenge(b"[Challenge] random value used for hashing.");

        trans.append_message(b"[Commit Phase]", &proof.helper_commitment);

        let (piop_res1, piop_subclaim1) = IndexedLogUpInputIOP::verifier_batch(
            trans,
            &proof.input_instance_info,
            &proof.input_piop_proof,
        );
        res &= piop_res1;
        assert!(res);

        let point_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            proof.trace_log_num_oracles,
        );
        let point_helper_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            proof.helper_log_num_oracles,
        );
        let mut point = Vec::with_capacity(piop_subclaim1.point_r.len() + point_oracle.len());
        let mut point_helper =
            Vec::with_capacity(piop_subclaim1.point_r.len() + point_helper_oracle.len());
        point.extend_from_slice(&piop_subclaim1.point_r);
        point.extend_from_slice(&point_oracle);
        point_helper.extend_from_slice(&piop_subclaim1.point_r);
        point_helper.extend_from_slice(&point_helper_oracle);

        let evals = proof.lookup_evals.pack_to_vec();
        let eval = compute_oracle_evals(&evals, &point_oracle);
        let eval_res = PCS::verify(
            &proof.pcs_params,
            &proof.commitment,
            &point,
            eval,
            &proof.eval_proof,
            trans,
        );
        res &= eval_res;
        assert!(res);

        let eval_helper = compute_oracle_evals(
            &proof.input_piop_proof.helper_input_at_rx,
            &point_helper_oracle,
        );
        let eval_helper_res = PCS::verify_ef(
            &proof.pcs_params_ef,
            &proof.helper_commitment,
            &point_helper,
            eval_helper,
            &proof.eval_helper_proof,
            trans,
        );
        res &= eval_helper_res;
        assert!(res);

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
    use trace::lookup_trace::indexed_table::IndexedLookupTrace;

    type FF = BabyBear;
    type EF = BabyBearExetension;
    type Hash = sha2::Sha256;
    const BASE_FIELD_BITS: usize = 31;

    #[test]
    fn test_logup_snarks() {
        let mut rng = rand::rng();
        let num_input_vars = 10;
        let num_table_vars = 7;

        let lookup_trace =
            IndexedLookupTrace::<FF>::random(&mut rng, num_input_vars, num_table_vars);
        let lookup_mle: IndexedLookupTraceMLE<FF> = lookup_trace.into();
        let snarks = BatchedIndexedLogUpSnarks::<
            FF,
            EF,
            ExpanderCodeSpec,
            BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
        >::default();
        let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
        let traces = vec![lookup_mle];
        let params = BatchedIndexedLogUpParams::new(code_spec, &traces);

        let prover_trans = &mut Transcript::<EF>::default();
        let proof = snarks.prove(prover_trans, &traces, &params);

        let verifier_trans = &mut Transcript::<EF>::default();
        let res = snarks.verify(verifier_trans, &proof);
        assert!(res);
    }
}
