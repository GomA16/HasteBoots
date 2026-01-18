//! In our case, the polynomial is so small and the number of these small
//! oracles is also small. For better efficiency when proving a single
//! bootstrapping operation, we instead send the polynomial directly to
//! the verifier and the verifier just evaluates the evaluation of the
//! random point on his own.
//!
//! When proving multiple bootstrapping operations in a batch, we can use
//! PCS to boost efficiency of the evaluation.
//!
use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};
use bincode::config::standard;
use helper::{FiatShamirTranscript, Transcript};
use log::info;
use pcs::PolynomialCommitmentScheme;
use piop::{
    SumcheckInstance, SumcheckPIOP,
    sparse_matrix_eval::sparse_row::{
        SparseRowEvalInfo, SparseRowEvalInstance, SparseRowEvalPIOP, SparseRowEvalProof,
    },
};
use serde::Serialize;

use crate::lookup::indexed_table::{IndexedLogUpSnarks, IndexedLogUpSnarksProof, indexed_batch::BatchedIndexedLogUpSnarks};

#[derive(Default)]
pub struct SparseRowEvalSnarks<F, EF, S, PCS>
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
pub struct SparseRowEvalSnarksProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub val_commitment: PCS::EFPolynomial,
    pub eval_mle_ry_commitment: PCS::EFPolynomial,
    pub indexed_lookup_proof: IndexedLogUpSnarksProof<F, EF, S, PCS>,
    pub sparse_row_instance_info: SparseRowEvalInfo<EF>,
    pub piop_proof: SparseRowEvalProof<EF>,
}

impl<F, EF, S, PCS> SparseRowEvalSnarksProof<F, EF, S, PCS>
where
    F: Field + Serialize,
    EF: AbstractExtensionField<F> + Serialize,
    S: Clone + Serialize,
    PCS: PolynomialCommitmentScheme<F, EF, S> + Serialize,
{
    pub fn piop_proof_len(&self) -> usize {
        bincode::serde::encode_to_vec(&self.indexed_lookup_proof, standard())
            .unwrap()
            .len()
            + bincode::serde::encode_to_vec(&self.sparse_row_instance_info, standard())
                .unwrap()
                .len()
            + bincode::serde::encode_to_vec(&self.piop_proof, standard())
                .unwrap()
                .len()
    }

    pub fn pcs_proof_len(&self) -> usize {
        bincode::serde::encode_to_vec(&self.val_commitment, standard())
            .unwrap()
            .len()
            + bincode::serde::encode_to_vec(&self.eval_mle_ry_commitment, standard())
                .unwrap()
                .len()
    }
}

impl<F, EF, S, PCS> SparseRowEvalSnarks<F, EF, S, PCS>
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
        instance: &SparseRowEvalInstance<EF>,
    ) -> SparseRowEvalSnarksProof<F, EF, S, PCS> {
        SparseRowEvalSnarks::prove_as_subprotocol(trans, instance)
    }

    pub fn verify(
        &self,
        trans: &mut Transcript<EF>,
        proof: &SparseRowEvalSnarksProof<F, EF, S, PCS>,
    ) -> bool {
        SparseRowEvalSnarks::verify_as_subprotocol(trans, proof, &mut None)
    }

    pub fn prove_as_subprotocol(
        trans: &mut Transcript<EF>,
        instance: &SparseRowEvalInstance<EF>,
    ) -> SparseRowEvalSnarksProof<F, EF, S, PCS> {
        // [PCS Phase] send the polynomial directly to the verifier (since thery are small polynomials)
        trans.append_message(b"[Commit Phase]", instance.val.as_ref());

        // Prove the eval_mle_ry = eq(to-bits(col), ry) using IndexedLogUpSnarks
        let indexed_lookup_trace = instance.extract_indexed_lookup_trace();
        let indexed_lookup_proof =
            IndexedLogUpSnarks::<F, EF, S, PCS>::prove_as_subprotocol(trans, &indexed_lookup_trace);
        trans.append_message(
            b"[PIOP Phase] Proving E_ry is well formed",
            &indexed_lookup_proof,
        );

        // Prove the sparse row evaluation using SparseRowEvalPIOP
        let (piop_proof, _) = SparseRowEvalPIOP::prover(trans, instance);
        trans.append_message(b"[PIOP Phase]", &piop_proof);

        SparseRowEvalSnarksProof {
            val_commitment: instance.val.as_ref().clone(),
            eval_mle_ry_commitment: instance.eval_mle_ry.as_ref().clone(),
            indexed_lookup_proof,
            sparse_row_instance_info: instance.info(),
            piop_proof,
        }
    }

    pub fn verify_as_subprotocol(
        trans: &mut Transcript<EF>,
        proof: &SparseRowEvalSnarksProof<F, EF, S, PCS>,
        statistics: &mut Option<&mut crate::SnarkStatistics>,
    ) -> bool {
        let mut res = true;
        let time = std::time::Instant::now();
        trans.append_message(b"[Commit Phase]", &proof.val_commitment);
        info!("[V]-[PCS] Receiving small polynomial time in {:?}", time.elapsed());
        if let Some(stats) = statistics {
            stats.add_verifier_pcs_time(time.elapsed());
        }

        let res_lookup = IndexedLogUpSnarks::<F, EF, S, PCS>::verify_as_subprotocol(
            trans,
            &proof.indexed_lookup_proof,
            statistics,
        );
        trans.append_message(
            b"[PIOP Phase] Proving E_ry is well formed",
            &proof.indexed_lookup_proof,
        );
        res &= res_lookup;

        let time = std::time::Instant::now();
        let (piop_res, piop_subclaim) =
            SparseRowEvalPIOP::verifier(trans, &proof.sparse_row_instance_info, &proof.piop_proof);
        trans.append_message(b"[PIOP Phase]", &proof.piop_proof);
        res &= piop_res;
        info!(
            "[V]-[PIOP] Sparse row evaluation proof verification in {:?}",
            time.elapsed()
        );

        let time = std::time::Instant::now();
        let evaluate_at_r = |poly: &PCS::EFPolynomial| poly.evaluate(&piop_subclaim.randomness);
        res &= evaluate_at_r(&proof.eval_mle_ry_commitment) == proof.piop_proof.eval_mle_at_r;
        res &= evaluate_at_r(&proof.val_commitment) == proof.piop_proof.val_at_r;
        info!(
            "[V]-[PCS] Sparse row evaluation consistency check in {:?}",
            time.elapsed()
        );
        if let Some(stats) = statistics {
            stats.add_verifier_pcs_time(time.elapsed());
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

    type FF = BabyBear;
    type EF = BabyBearExetension;
    type Hash = sha2::Sha256;

    #[test]
    fn test_sparse_row_eval_snarks() {
        let mut rng = rand::rng();
        let num_x_vars = 4;
        let num_y_vars = 10;

        let instance = SparseRowEvalInstance::<EF>::random(&mut rng, num_x_vars, num_y_vars);
        let snarks = SparseRowEvalSnarks::<
            FF,
            EF,
            ExpanderCodeSpec,
            BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
        >::default();
        let prover_trans = &mut Transcript::<EF>::default();
        let proof = snarks.prove(prover_trans, &instance);
        let verifier_trans = &mut Transcript::<EF>::default();
        let res = snarks.verify(verifier_trans, &proof);
        assert!(res);
    }
}
