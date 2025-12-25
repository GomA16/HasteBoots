use std::ops::Index;

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};
use helper::{FiatShamirTranscript, Transcript};
use pcs::PolynomialCommitmentScheme;
use piop::{
    SumcheckInstance, SumcheckPIOP,
    lookup::indexed_table::{
        self, IndexedLogUpInputIOP, IndexedLogUpInputInstanceInfo, IndexedLogUpInputProof,
        IndexedLogUpTableIOP, IndexedLogUpTableInstanceInfo, IndexedLogUpTableProof,
    },
};
use rand::rand_core::le;
use serde::Serialize;
use trace::lookup_trace::indexed_table::IndexedLookupTraceMLE;

#[derive(Default)]
pub struct IndexedLogUpSnarksPlusPCS<F, EF, S, PCS>
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
pub struct IndexedLogUpParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub pcs_input_params: PCS::Parameters,
    pub pcs_table_params: PCS::Parameters,
}

impl<F, EF, S, PCS> IndexedLogUpParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub fn new(code_spec: S, trace: &IndexedLookupTraceMLE<EF>) -> Self {
        Self {
            pcs_input_params: PCS::setup(trace.num_input_vars, Some(code_spec.clone())),
            pcs_table_params: PCS::setup(trace.num_table_vars, Some(code_spec.clone())),
        }
    }
}

impl<F, EF, S, PCS> Clone for IndexedLogUpParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    fn clone(&self) -> Self {
        Self {
            pcs_input_params: self.pcs_input_params.clone(),
            pcs_table_params: self.pcs_table_params.clone(),
        }
    }
}

pub struct IndexedLogUpSnarksProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub params: IndexedLogUpParams<F, EF, S, PCS>,
    pub input_commitment: PCS::Commitment,
    pub index_commitment: PCS::Commitment,
    pub helper_input_commitment: PCS::Commitment,
    pub multiplicity_commitment: PCS::Commitment,
    pub helper_table_commitment: PCS::Commitment,
    pub input_instance_info: IndexedLogUpInputInstanceInfo<EF>,
    pub input_piop_proof: IndexedLogUpInputProof<EF>,
    pub table_instance_info: IndexedLogUpTableInstanceInfo<EF>,
    pub table_piop_proof: IndexedLogUpTableProof<EF>,
    pub input_eval_proof: PCS::ProofEF,
    pub index_eval_proof: PCS::ProofEF,
    pub helper_input_eval_proof: PCS::ProofEF,
    pub multiplicity_eval_proof: PCS::ProofEF,
    pub helper_table_eval_proof: PCS::ProofEF,
}

impl<F, EF, S, PCS> IndexedLogUpSnarksPlusPCS<F, EF, S, PCS>
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
        trace_mle: &IndexedLookupTraceMLE<EF>,
        params: &IndexedLogUpParams<F, EF, S, PCS>,
    ) -> IndexedLogUpSnarksProof<F, EF, S, PCS> {
        let witness = trace_mle.compute_witness();

        let comm_input = |poly: &PCS::EFPolynomial| PCS::commit_ef(&params.pcs_input_params, poly);
        let comm_table = |poly: &PCS::EFPolynomial| PCS::commit_ef(&params.pcs_table_params, poly);
        let (input_comm, input_comm_state) = comm_input(trace_mle.input.as_ref());
        let (index_comm, index_comm_state) = comm_input(trace_mle.index.as_ref());
        let (multiplicity_comm, multiplicity_comm_state) =
            comm_table(witness.multiplicity.as_ref());
        trans.append_message(b"[Commit Phase]", &input_comm);
        trans.append_message(b"[Commit Phase]", &index_comm);
        trans.append_message(b"[Commit Phase]", &multiplicity_comm);

        let random_value =
            trans.get_challenge(b"[Challenge] random value used in the rational identity");
        let random_s_hash = trans.get_challenge(b"[Challenge] random value used for hashing.");
        let helper = trace_mle.compute_helper_functions(&witness, random_value, random_s_hash);
        let (helper_input_comm, helper_input_comm_state) = comm_input(helper.helper_input.as_ref());
        let (helper_table_comm, helper_table_comm_state) = comm_table(helper.helper_table.as_ref());
        trans.append_message(b"[Commit Phase]", &helper_input_comm);
        trans.append_message(b"[Commit Phase]", &helper_table_comm);

        let input_instance =
            indexed_table::IndexedLogUpInputInstance::<EF>::from(trace_mle, &helper);
        let (input_piop_proof, input_piop_state) =
            indexed_table::IndexedLogUpInputIOP::prover(trans, &input_instance);
        let table_instance =
            indexed_table::IndexedLogUpTableInstance::<EF>::from(&witness, &helper);
        let (table_piop_proof, table_piop_state) =
            indexed_table::IndexedLogUpTableIOP::prover(trans, &table_instance);
        trans.append_message(b"[PIOP Phase]", &input_piop_proof);
        trans.append_message(b"[PIOP Phase]", &table_piop_proof);

        let mut open_input = |comm: &PCS::Commitment, state: &PCS::CommitmentStateEF| {
            PCS::open_ef(
                &params.pcs_input_params,
                comm,
                state,
                &input_piop_state.point_r,
                trans,
            )
        };

        let input_eval_proof = open_input(&input_comm, &input_comm_state);
        let index_eval_proof = open_input(&index_comm, &index_comm_state);
        let helper_input_eval_proof = open_input(&helper_input_comm, &helper_input_comm_state);

        let mut open_table = |comm: &PCS::Commitment, state: &PCS::CommitmentStateEF| {
            PCS::open_ef(
                &params.pcs_table_params,
                comm,
                state,
                &table_piop_state.point_r,
                trans,
            )
        };
        let multiplicity_eval_proof = open_table(&multiplicity_comm, &multiplicity_comm_state);
        let helper_table_eval_proof = open_table(&helper_table_comm, &helper_table_comm_state);
        IndexedLogUpSnarksProof {
            params: params.clone(),
            input_commitment: input_comm,
            index_commitment: index_comm,
            helper_input_commitment: helper_input_comm,
            multiplicity_commitment: multiplicity_comm,
            helper_table_commitment: helper_table_comm,
            input_instance_info: input_instance.info(),
            input_piop_proof,
            table_instance_info: table_instance.info(),
            table_piop_proof,
            input_eval_proof,
            index_eval_proof,
            helper_input_eval_proof,
            multiplicity_eval_proof,
            helper_table_eval_proof,
        }
    }

    pub fn verify(
        &self,
        trans: &mut Transcript<EF>,
        proof: &IndexedLogUpSnarksProof<F, EF, S, PCS>,
    ) -> bool {
        let mut res = true;
        trans.append_message(b"[Commit Phase]", &proof.input_commitment);
        trans.append_message(b"[Commit Phase]", &proof.index_commitment);
        trans.append_message(b"[Commit Phase]", &proof.multiplicity_commitment);

        let random_value =
            trans.get_challenge(b"[Challenge] random value used in the rational identity");
        let random_s_hash = trans.get_challenge(b"[Challenge] random value used for hashing.");

        trans.append_message(b"[Commit Phase]", &proof.helper_input_commitment);
        trans.append_message(b"[Commit Phase]", &proof.helper_table_commitment);

        let (piop_res1, piop_subclaim1) = IndexedLogUpInputIOP::verifier(
            trans,
            &proof.input_instance_info,
            &proof.input_piop_proof,
        );
        res &= piop_res1;
        let (piop_res2, piop_subclaim2) = IndexedLogUpTableIOP::verifier(
            trans,
            &proof.table_instance_info,
            &proof.table_piop_proof,
        );
        res &= piop_res2;
        trans.append_message(b"[PIOP Phase]", &proof.input_piop_proof);
        trans.append_message(b"[PIOP Phase]", &proof.table_piop_proof);

        let time = std::time::Instant::now();
        let mut verify_input = |comm: &PCS::Commitment, eval: EF, eval_proof: &PCS::ProofEF| {
            PCS::verify_ef(
                &proof.params.pcs_input_params,
                comm,
                &piop_subclaim1.point_r,
                eval,
                eval_proof,
                trans,
            )
        };
        let input_res = verify_input(
            &proof.input_commitment,
            proof.input_piop_proof.input_at_rx,
            &proof.input_eval_proof,
        );
        res &= input_res;
        let index_res = verify_input(
            &proof.index_commitment,
            proof.input_piop_proof.index_at_rx,
            &proof.index_eval_proof,
        );
        res &= index_res;
        let helper_input_res = verify_input(
            &proof.helper_input_commitment,
            proof.input_piop_proof.helper_input_at_rx,
            &proof.helper_input_eval_proof,
        );
        res &= helper_input_res;

        let mut verify_table = |comm: &PCS::Commitment, eval: EF, eval_proof: &PCS::ProofEF| {
            PCS::verify_ef(
                &proof.params.pcs_table_params,
                comm,
                &piop_subclaim2.point_r,
                eval,
                eval_proof,
                trans,
            )
        };
        let multiplicity_res = verify_table(
            &proof.multiplicity_commitment,
            proof.table_piop_proof.multiplicity_at_ry,
            &proof.multiplicity_eval_proof,
        );
        res &= multiplicity_res;
        let helper_table_res = verify_table(
            &proof.helper_table_commitment,
            proof.table_piop_proof.helper_table_at_ry,
            &proof.helper_table_eval_proof,
        );
        res &= helper_table_res;
        println!("PCS verify time: {:?}", time.elapsed());
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
    use trace::lookup_trace::{indexed_table::IndexedLookupTrace, small_table::LookupTrace};

    type FF = BabyBear;
    type EF = BabyBearExetension;
    type Hash = sha2::Sha256;
    const BASE_FIELD_BITS: usize = 31;

    #[test]
    fn test_logup_snarks() {
        let mut rng = rand::rng();
        let num_input_vars = 5;
        let num_table_vars = 10;

        let lookup_trace = IndexedLookupTrace::<EF>::random(&mut rng, num_input_vars, num_table_vars);
        let lookup_mle: IndexedLookupTraceMLE<EF> = lookup_trace.into();
        let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
        let snarks = IndexedLogUpSnarksPlusPCS::<
            FF,
            EF,
            ExpanderCodeSpec,
            BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
        >::default();
        let params = &mut IndexedLogUpParams::new(code_spec, &lookup_mle);

        let prover_trans = &mut Transcript::<EF>::default();
        let proof = snarks.prove(prover_trans, &lookup_mle, params);

        let verifier_trans = &mut Transcript::<EF>::default();
        let res = snarks.verify(verifier_trans, &proof);
        assert!(res);
    }
}

