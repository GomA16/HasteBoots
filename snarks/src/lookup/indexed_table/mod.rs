//! In our case, the polynomial is so small and the number of these small
//! oracles is also small. For better efficiency when proving a single
//! bootstrapping operation, we instead send the polynomial directly to
//! the verifier and the verifier just evaluates the evaluation of the
//! random point on his own.
//!
//! When proving multiple bootstrapping operations in a batch, we can use
//! PCS to boost efficiency of the evaluation.
pub mod indexed_batch;
// pub mod indexed_table_pcs;

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};
use helper::{FiatShamirTranscript, Transcript, utils::eval_identity_function};
use pcs::PolynomialCommitmentScheme;
use piop::{
    SumcheckInstance, SumcheckPIOP,
    lookup::indexed_table::{
        self, IndexedLogUpInputIOP, IndexedLogUpInputInstanceInfo, IndexedLogUpInputProof,
        IndexedLogUpTableIOP, IndexedLogUpTableInstanceInfo, IndexedLogUpTableProof,
    },
};
use serde::Serialize;
use trace::lookup_trace::indexed_table::IndexedLookupTraceMLE;

#[derive(Default)]
pub struct IndexedLogUpSnarks<F, EF, S, PCS>
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
pub struct IndexedLogUpSnarksProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    #[serde(skip)]
    pub input_commitment: PCS::EFPolynomial,
    #[serde(skip)]
    pub index_commitment: PCS::EFPolynomial,
    #[serde(skip)]
    pub helper_input_commitment: PCS::EFPolynomial,
    #[serde(skip)]
    pub multiplicity_commitment: PCS::EFPolynomial,
    #[serde(skip)]
    pub helper_table_commitment: PCS::EFPolynomial,
    pub input_instance_info: IndexedLogUpInputInstanceInfo<EF>,
    pub input_piop_proof: IndexedLogUpInputProof<EF>,
    pub table_instance_info: IndexedLogUpTableInstanceInfo<EF>,
    pub table_piop_proof: IndexedLogUpTableProof<EF>,
}

impl<F, EF, S, PCS> IndexedLogUpSnarks<F, EF, S, PCS>
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
    ) -> IndexedLogUpSnarksProof<F, EF, S, PCS> {
        IndexedLogUpSnarks::prove_as_subprotocol(trans, trace_mle)
    }

    pub fn verify(
        &self,
        trans: &mut Transcript<EF>,
        proof: &IndexedLogUpSnarksProof<F, EF, S, PCS>,
    ) -> bool {
        IndexedLogUpSnarks::verify_as_subprotocol(trans, proof)
    }

    pub fn prove_as_subprotocol(
        trans: &mut Transcript<EF>,
        trace_mle: &IndexedLookupTraceMLE<EF>,
    ) -> IndexedLogUpSnarksProof<F, EF, S, PCS> {
        let witness = trace_mle.compute_witness();
        // Commit phase: send the polynomials directly to the verifier
        trans.append_message(b"[Commit Phase]", trace_mle.input.as_ref());
        trans.append_message(b"[Commit Phase]", trace_mle.index.as_ref());
        trans.append_message(b"[Commit Phase]", witness.multiplicity.as_ref());

        let random_value =
            trans.get_challenge(b"[Challenge] random value used in the rational identity");
        let random_s_hash = trans.get_challenge(b"[Challenge] random value used for hashing.");
        let helper = trace_mle.compute_helper_functions(&witness, random_value, random_s_hash);
        trans.append_message(b"[Commit Phase]", helper.helper_input.as_ref());
        trans.append_message(b"[Commit Phase]", helper.helper_table.as_ref());

        let input_instance =
            indexed_table::IndexedLogUpInputInstance::<EF>::from(trace_mle, &helper);
        let (input_piop_proof, _input_piop_state) =
            indexed_table::IndexedLogUpInputIOP::prover(trans, &input_instance);
        let table_instance =
            indexed_table::IndexedLogUpTableInstance::<EF>::from(&witness, &helper);
        let (table_piop_proof, _table_piop_state) =
            indexed_table::IndexedLogUpTableIOP::prover(trans, &table_instance);
        trans.append_message(b"[PIOP Phase]", &input_piop_proof);
        trans.append_message(b"[PIOP Phase]", &table_piop_proof);

        IndexedLogUpSnarksProof {
            input_commitment: trace_mle.input.as_ref().clone(),
            index_commitment: trace_mle.index.as_ref().clone(),
            helper_input_commitment: helper.helper_input.as_ref().clone(),
            multiplicity_commitment: witness.multiplicity.as_ref().clone(),
            helper_table_commitment: helper.helper_table.as_ref().clone(),
            input_instance_info: input_instance.info(),
            input_piop_proof,
            table_instance_info: table_instance.info(),
            table_piop_proof,
        }
    }

    pub fn verify_as_subprotocol(
        trans: &mut Transcript<EF>,
        proof: &IndexedLogUpSnarksProof<F, EF, S, PCS>,
    ) -> bool {
        let mut res = true;
        trans.append_message(b"[Commit Phase]", &proof.input_commitment);
        trans.append_message(b"[Commit Phase]", &proof.index_commitment);
        trans.append_message(b"[Commit Phase]", &proof.multiplicity_commitment);

        // Some simple value equality checks are omitted here for brevity.
        let _random_value =
            trans.get_challenge(b"[Challenge] random value used in the rational identity");
        let _random_s_hash = trans.get_challenge(b"[Challenge] random value used for hashing.");

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

        let evaluate_input_at_r = |poly: &PCS::EFPolynomial| poly.evaluate(&piop_subclaim1.point_r);
        res &= evaluate_input_at_r(&proof.input_commitment) == proof.input_piop_proof.input_at_rx;
        res &= evaluate_input_at_r(&proof.index_commitment) == proof.input_piop_proof.index_at_rx;
        res &= evaluate_input_at_r(&proof.helper_input_commitment)
            == proof.input_piop_proof.helper_input_at_rx;

        let evaluate_table_at_r = |poly: &PCS::EFPolynomial| poly.evaluate(&piop_subclaim2.point_r);
        res &= evaluate_table_at_r(&proof.multiplicity_commitment)
            == proof.table_piop_proof.multiplicity_at_ry;
        res &= evaluate_table_at_r(&proof.helper_table_commitment)
            == proof.table_piop_proof.helper_table_at_ry;

        if proof.table_instance_info.table_point.is_some() {
            res &= eval_identity_function(
                proof.table_instance_info.table_point.as_ref().unwrap(),
                &piop_subclaim2.point_r,
            ) == proof.table_piop_proof.table_at_ry;
        }

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
    use trace::{ConvertToEF, lookup_trace::indexed_table::IndexedLookupTrace};

    type FF = BabyBear;
    type EF = BabyBearExetension;
    type Hash = sha2::Sha256;

    #[test]
    fn test_logup_ef_snarks() {
        let mut rng = rand::rng();
        let num_input_vars = 5;
        let num_table_vars = 10;

        let lookup_trace =
            IndexedLookupTrace::<EF>::random(&mut rng, num_input_vars, num_table_vars);
        let lookup_mle: IndexedLookupTraceMLE<EF> = lookup_trace.into();
        let snarks = IndexedLogUpSnarks::<
            FF,
            EF,
            ExpanderCodeSpec,
            BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
        >::default();

        let prover_trans = &mut Transcript::<EF>::default();
        let proof = snarks.prove(prover_trans, &lookup_mle);

        let verifier_trans = &mut Transcript::<EF>::default();
        let res = snarks.verify(verifier_trans, &proof);
        assert!(res);
    }

    #[test]
    fn test_logup_snarks() {
        let mut rng = rand::rng();
        let num_input_vars = 5;
        let num_table_vars = 10;

        let lookup_trace =
            IndexedLookupTrace::<FF>::random(&mut rng, num_input_vars, num_table_vars);
        let lookup_mle: IndexedLookupTraceMLE<FF> = lookup_trace.into();
        let snarks = IndexedLogUpSnarks::<
            FF,
            EF,
            ExpanderCodeSpec,
            BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
        >::default();

        let prover_trans = &mut Transcript::<EF>::default();
        let proof = snarks.prove(prover_trans, &lookup_mle.to_ef());

        let verifier_trans = &mut Transcript::<EF>::default();
        let res = snarks.verify(verifier_trans, &proof);
        assert!(res);
    }
}
