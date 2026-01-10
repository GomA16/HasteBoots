use core::time;

use algebra::{AbstractExtensionField, AsInto, DenseMultilinearExtension, Field};
use bincode::de;
use helper::{FiatShamirTranscript, Transcript, utils::compute_oracle_evals};
use pcs::{PolynomialCommitmentScheme, utils::code};
use serde::{Serialize, ser};
use trace::{
    PackableTrace, basic_ops::decomp_trace::DecompTraceMLE, cmp_trace::lt_trace::LTTablesMLE,
};

use crate::lookup::indexed_table::indexed_batch::{
    BatchedIndexedLogUpParams, BatchedIndexedLogUpSnarks, BatchedIndexedLogUpSnarksProof,
};

#[derive(Default)]
pub struct DecompositionSnarks<F, EF, S, PCS>
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

pub struct DecompositionParams<'a, F, S>
where
    F: Field,
    S: Clone,
{
    pub lt_tables: &'a LTTablesMLE<F>,
    pub code_spec: S,
}

#[derive(Serialize)]
pub struct DecompositionSnarksProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub log_num_oracles: usize,
    pub basis: F,
    pub decomp_len: usize,
    #[serde(skip)]
    pub input_params: PCS::Parameters,
    #[serde(skip)]
    pub input_commitment: PCS::Commitment,
    pub lookup_proof: BatchedIndexedLogUpSnarksProof<F, EF, S, PCS>,
    pub inputs_eval: Vec<EF>,
    pub eval_proof: PCS::Proof,
}

impl<'a, F, S> DecompositionParams<'a, F, S>
where
    F: Field,
    S: Clone,
{
    pub fn new(code_spec: S, lt_tables: &'a LTTablesMLE<F>) -> Self {
        Self {
            code_spec: code_spec.clone(),
            lt_tables,
        }
    }
}

impl<F, EF, S, PCS> DecompositionSnarks<F, EF, S, PCS>
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
        traces: &Vec<DecompTraceMLE<F>>,
        params: &DecompositionParams<F, S>,
    ) -> DecompositionSnarksProof<F, EF, S, PCS> {
        Self::prove_as_subprotocol(trans, traces, params)
    }

    pub fn verify(
        &self,
        trans: &mut Transcript<EF>,
        proof: &DecompositionSnarksProof<F, EF, S, PCS>,
    ) -> bool {
        Self::verify_as_subprotocol(trans, proof)
    }

    pub fn prove_as_subprotocol(
        trans: &mut Transcript<EF>,
        traces: &Vec<DecompTraceMLE<F>>,
        params: &DecompositionParams<F, S>,
    ) -> DecompositionSnarksProof<F, EF, S, PCS> {
        let poly = traces.generate_oracle();
        let input_params = PCS::setup(poly.num_vars(), Some(&params.code_spec));

        let (input_commitment, input_comm_state) = PCS::commit(&input_params, &poly);
        trans.append_message(b"[Commit Phase]", &input_commitment);

        // Prove the decomposition consistency via batched indexed log-up proofs
        // 1. each bit x_i is in range [0, 2^k)
        // 2. \sum_{i} x_i * 2^{i*k} x_i = x
        // The second part is garanteed by ensuring \sum_{i} x_i * 2^{i*k} < p via lookups.
        // refer: https://www.usenix.org/conference/usenixsecurity24/presentation/hao-meng-scalable
        let lookup_trace = traces
            .iter()
            .map(|trace| trace.extract_lt_general_lookup_trace(&params.lt_tables))
            .flatten()
            .collect::<Vec<_>>();
        let lookup_params = BatchedIndexedLogUpParams::new(params.code_spec.clone(), &lookup_trace);
        let lookup_proof =
            BatchedIndexedLogUpSnarks::prove_as_subprotocol(trans, &lookup_trace, &lookup_params);

        let point_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            traces.log_num_oracles(),
        );
        let inputs_evals = traces
            .iter()
            .map(|trace| trace.input.evaluate_ext(&lookup_proof.input_point_r))
            .collect::<Vec<_>>();

        trans.append_message(b"[PIOP Phase]", &inputs_evals);
        let mut point = Vec::with_capacity(lookup_proof.input_point_r.len() + point_oracle.len());
        point.extend_from_slice(&lookup_proof.input_point_r);
        point.extend_from_slice(&point_oracle);

        let eval_proof = PCS::open(
            &input_params,
            &input_commitment,
            &input_comm_state,
            &point,
            trans,
        );

        let basis = 1 << params.lt_tables.basis_bits;
        DecompositionSnarksProof {
            log_num_oracles: traces.log_num_oracles(),
            basis: F::new(basis.as_into()),
            decomp_len: params.lt_tables.decomp_len,
            input_params,
            input_commitment,
            lookup_proof,
            inputs_eval: inputs_evals,
            eval_proof,
        }
    }

    pub fn verify_as_subprotocol(
        trans: &mut Transcript<EF>,
        proof: &DecompositionSnarksProof<F, EF, S, PCS>,
    ) -> bool {
        trans.append_message(b"[Commit Phase]", &proof.input_commitment);

        // Verify the decomposition consistency via batched indexed log-up proofs
        // It ensures that the decomposition is in range of [0, p)
        // Each bit is in range [0, 2^k)
        let mut res = true;
        let lookup_res =
            BatchedIndexedLogUpSnarks::verify_as_subprotocol(trans, &proof.lookup_proof);
        res &= lookup_res;
        assert!(lookup_res, "Decomposition lookup proof verification failed");

        // Verify the decomposition relation
        let res_decomp = proof
            .inputs_eval
            .iter()
            .zip(
                proof
                    .lookup_proof
                    .lookup_evals
                    .chunks_exact(proof.decomp_len),
            )
            .all(|(input, bits)| {
                let mut sum = EF::zero();
                let mut basis = EF::one();
                for bit in bits {
                    sum += bit.index_at_r * basis;
                    basis = basis * proof.basis;
                }
                sum == *input
            });
        res &= res_decomp;
        assert!(res_decomp, "Decomposition relation verification failed");

        let point_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            proof.log_num_oracles,
        );

        let eval = compute_oracle_evals(&proof.inputs_eval, &point_oracle);
        trans.append_message(b"[PIOP Phase]", &proof.inputs_eval);

        let mut point =
            Vec::with_capacity(proof.lookup_proof.input_point_r.len() + point_oracle.len());
        point.extend_from_slice(&proof.lookup_proof.input_point_r);
        point.extend_from_slice(&point_oracle);

        let eval_res = PCS::verify(
            &proof.input_params,
            &proof.input_commitment,
            &point,
            eval,
            &proof.eval_proof,
            trans,
        );
        res &= eval_res;
        assert!(
            eval_res,
            "Decomposition evaluation proof verification failed"
        );
        res
    }
}
