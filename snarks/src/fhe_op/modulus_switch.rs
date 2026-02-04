use algebra::{AbstractExtensionField, DecomposableField, DenseMultilinearExtension, Field};
use helper::{FiatShamirTranscript, Transcript};
use pcs::PolynomialCommitmentScheme;
use serde::Serialize;
use trace::{ConvertToEF, modulus_switching_trace::ModulusSwitchingTraceMLE};

use crate::{
    cmp::{
        cmp_eq::{ComputeEqualityParams, ComputeEqualityProof, ComputeEqualitySnarks},
        cmp_lt::{ComputeLessThanParams, ComputeLessThanProof, ComputeLessThanSnarks},
    },
    lookup::indexed_table::{
        IndexedLogUpSnarks, IndexedLogUpSnarksProof,
        indexed_batch::{
            BatchedIndexedLogUpParams, BatchedIndexedLogUpSnarks, BatchedIndexedLogUpSnarksProof,
        },
    },
};

#[derive(Default)]
pub struct ModulusSwitchingSnarks<F, EF, S, PCS>
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

pub struct ModulusSwitchingParams<'a, S> {
    pub code_spec: &'a S,
}

#[derive(Serialize)]
pub struct ModulusSwitchingProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone + Serialize,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub log_num: usize,
    pub b_eq_b_prime_proof: IndexedLogUpSnarksProof<F, EF, S, PCS>,
    pub a_eq_k_proof: ComputeEqualityProof<F, EF, S, PCS>,
    pub e_lt_2k_plus_1_proof: ComputeLessThanProof<F, EF, S, PCS>,
    pub e_eq_2k_proof: ComputeEqualityProof<F, EF, S, PCS>,
    pub a_eq_k_result: PCS::Polynomial,
    pub e_eq_2k_result: PCS::Polynomial,
}

#[derive(Serialize)]
pub struct ModulusSwitchingProofOpt<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone + Serialize,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub log_num: usize,
    pub lookup_proof: BatchedIndexedLogUpSnarksProof<F, EF, S, PCS>,
    pub a_eq_k_result: PCS::Polynomial,
    pub e_eq_2k_result: PCS::Polynomial,
}

impl<F, EF, S, PCS> ModulusSwitchingProof<F, EF, S, PCS>
where
    F: Field + Serialize,
    EF: AbstractExtensionField<F> + Serialize,
    S: Clone + Serialize,
    PCS: PolynomialCommitmentScheme<
            F,
            EF,
            S,
            Polynomial = DenseMultilinearExtension<F>,
            EFPolynomial = DenseMultilinearExtension<EF>,
            Point = EF,
        > + Serialize,
{
    pub fn piop_proof_len(&self) -> usize {
        self.b_eq_b_prime_proof.piop_proof_len()
            + self.a_eq_k_proof.piop_proof_len()
            + self.e_lt_2k_plus_1_proof.piop_proof_len()
            + self.e_eq_2k_proof.piop_proof_len()
    }

    pub fn pcs_proof_len(&self) -> usize {
        self.b_eq_b_prime_proof.pcs_proof_len()
            + self.a_eq_k_proof.pcs_proof_len()
            + self.e_lt_2k_plus_1_proof.pcs_proof_len()
            + self.e_eq_2k_proof.pcs_proof_len()
    }
}

impl<F, EF, S, PCS> ModulusSwitchingProofOpt<F, EF, S, PCS>
where
    F: Field + Serialize,
    EF: AbstractExtensionField<F> + Serialize,
    S: Clone + Serialize,
    PCS: PolynomialCommitmentScheme<
            F,
            EF,
            S,
            Polynomial = DenseMultilinearExtension<F>,
            EFPolynomial = DenseMultilinearExtension<EF>,
            Point = EF,
        > + Serialize,
{
    pub fn piop_proof_len(&self) -> usize {
        self.lookup_proof.piop_proof_len()
    }

    pub fn pcs_proof_len(&self) -> usize {
        self.lookup_proof.pcs_proof_len()
    }
}

impl<'a, S> ModulusSwitchingParams<'a, S>
where
    S: Clone,
{
    pub fn new(code_spec: &'a S) -> Self {
        Self { code_spec }
    }
}

impl<F, EF, S, PCS> ModulusSwitchingSnarks<F, EF, S, PCS>
where
    F: Field + DecomposableField + Serialize,
    EF: AbstractExtensionField<F> + Serialize,
    S: Clone + Serialize,
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
        trace: &ModulusSwitchingTraceMLE<F>,
        params: &ModulusSwitchingParams<S>,
        statistics: &mut Option<&mut crate::SnarkStatistics>,
    ) -> ModulusSwitchingProofOpt<F, EF, S, PCS> {
        Self::prove_as_subprotocol_opt(trans, trace, params, statistics)
    }

    pub fn verify(
        &self,
        trans: &mut Transcript<EF>,
        proof: &ModulusSwitchingProofOpt<F, EF, S, PCS>,
        statistics: &mut Option<&mut crate::SnarkStatistics>,
    ) -> bool {
        Self::verify_as_subprotocol_opt(trans, proof, statistics)
    }

    pub fn prove_as_subprotocol_opt(
        trans: &mut Transcript<EF>,
        trace: &ModulusSwitchingTraceMLE<F>,
        params: &ModulusSwitchingParams<S>,
        statistics: &mut Option<&mut crate::SnarkStatistics>,
    ) -> ModulusSwitchingProofOpt<F, EF, S, PCS> {
        // Prove b = b' mod q via batched indexed log-up proofs
        let b_eq_b_prime = trace.extract_output_eq_output_witness_trace();

        // Prove e < 2k + 1 via less-than proof
        let e_lt_2k_plus_1_trace = trace.extract_helper_lt_2k_plus_1();
        let indexed_lookups_e =
            e_lt_2k_plus_1_trace.extract_lt_lookup_traces(&trace.e_lt_2k_plus_1_tables);

        // Prove a = k mod p via equality proof
        let a_eq_k_trace = trace.extract_a_eq_k_trace();
        let indexed_lookups_a = a_eq_k_trace.extract_eq_lookup_traces(&trace.a_eq_k_tables);
        trans.append_message(b"[Commit]", a_eq_k_trace.eq_result.as_ref());

        // Prove e = 2k mod p via equality proof
        let e_eq_2k_trace = trace.extract_e_eq_2k_trace();
        let indexed_lookups_e_eq = e_eq_2k_trace.extract_eq_lookup_traces(&trace.e_eq_2k_tables);
        trans.append_message(b"[Commit]", e_eq_2k_trace.eq_result.as_ref());

        let mut all_indexed_lookups = indexed_lookups_e
            .into_iter()
            .chain(indexed_lookups_a)
            .chain(indexed_lookups_e_eq)
            .collect::<Vec<_>>();
        all_indexed_lookups.push(b_eq_b_prime);

        let lookup_params =
            BatchedIndexedLogUpParams::new(params.code_spec.clone(), &all_indexed_lookups);
        let lookup_proof = BatchedIndexedLogUpSnarks::prove_as_subprotocol(
            trans,
            &all_indexed_lookups,
            &lookup_params,
            statistics,
        );

        let _point = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            trace.log_num,
        );

        ModulusSwitchingProofOpt {
            log_num: trace.log_num,
            lookup_proof,
            a_eq_k_result: a_eq_k_trace.eq_result.as_ref().clone(),
            e_eq_2k_result: e_eq_2k_trace.eq_result.as_ref().clone(),
        }
    }

    pub fn prove_as_subprotocol(
        trans: &mut Transcript<EF>,
        trace: &ModulusSwitchingTraceMLE<F>,
        params: &ModulusSwitchingParams<S>,
        statistics: &mut Option<&mut crate::SnarkStatistics>,
    ) -> ModulusSwitchingProof<F, EF, S, PCS> {
        // Prove b = b' mod q via batched indexed log-up proofs
        let b_eq_b_prime = trace.extract_output_eq_output_witness_trace().to_ef();
        let b_eq_b_prime_proof =
            IndexedLogUpSnarks::<F, EF, S, PCS>::prove_as_subprotocol(trans, &b_eq_b_prime);

        // Prove e < 2k + 1 via less-than proof
        let e_lt_2k_plus_1_trace = trace.extract_helper_lt_2k_plus_1();
        let e_lt_2k_plus_1_params =
            ComputeLessThanParams::new(params.code_spec.clone(), &trace.e_lt_2k_plus_1_tables);
        let e_lt_2k_plus_1_proof = ComputeLessThanSnarks::prove_as_subprotocol(
            trans,
            &e_lt_2k_plus_1_trace,
            &e_lt_2k_plus_1_params,
            statistics,
        );

        // Prove a = k mod p via equality proof
        let a_eq_k_trace = trace.extract_a_eq_k_trace();
        trans.append_message(b"[Commit]", a_eq_k_trace.eq_result.as_ref());
        let a_eq_k_params = ComputeEqualityParams::new(params.code_spec, &trace.a_eq_k_tables);
        let a_eq_k_proof = ComputeEqualitySnarks::prove_as_subprotocol(
            trans,
            &a_eq_k_trace,
            &a_eq_k_params,
            statistics,
        );

        // Prove e = 2k mod p via equality proof
        let e_eq_2k_trace = trace.extract_e_eq_2k_trace();
        trans.append_message(b"[Commit]", e_eq_2k_trace.eq_result.as_ref());
        let e_eq_2k_params = ComputeEqualityParams::new(params.code_spec, &trace.e_eq_2k_tables);
        let e_eq_2k_proof = ComputeEqualitySnarks::prove_as_subprotocol(
            trans,
            &e_eq_2k_trace,
            &e_eq_2k_params,
            statistics,
        );

        let _point = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            trace.log_num,
        );

        ModulusSwitchingProof {
            log_num: trace.log_num,
            b_eq_b_prime_proof,
            a_eq_k_proof,
            e_lt_2k_plus_1_proof,
            e_eq_2k_proof,
            a_eq_k_result: a_eq_k_trace.eq_result.as_ref().clone(),
            e_eq_2k_result: e_eq_2k_trace.eq_result.as_ref().clone(),
        }
    }

    pub fn verify_as_subprotocol_opt(
        trans: &mut Transcript<EF>,
        proof: &ModulusSwitchingProofOpt<F, EF, S, PCS>,
        statistics: &mut Option<&mut crate::SnarkStatistics>,
    ) -> bool {
        let mut res = true;

        trans.append_message(b"[Commit]", &proof.a_eq_k_result);
        trans.append_message(b"[Commit]", &proof.e_eq_2k_result);

        let lookup_proof_res = BatchedIndexedLogUpSnarks::verify_as_subprotocol(
            trans,
            &proof.lookup_proof,
            statistics,
        );
        res &= lookup_proof_res;
        assert!(lookup_proof_res, "lookup proofs failed");

        let time = std::time::Instant::now();
        // a = k && e = 2k
        let point = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            proof.log_num,
        );
        res &=
            proof.a_eq_k_result.evaluate_ext(&point) == proof.e_eq_2k_result.evaluate_ext(&point);
        if let Some(statistics) = statistics {
            statistics.add_verifier_pcs_time(time.elapsed());
        }

        res
    }

    pub fn verify_as_subprotocol(
        trans: &mut Transcript<EF>,
        proof: &ModulusSwitchingProof<F, EF, S, PCS>,
        statistics: &mut Option<&mut crate::SnarkStatistics>,
    ) -> bool {
        let mut res = true;
        let b_eq_b_prime_res = IndexedLogUpSnarks::<F, EF, S, PCS>::verify_as_subprotocol(
            trans,
            &proof.b_eq_b_prime_proof,
            statistics,
        );
        res &= b_eq_b_prime_res;
        assert!(b_eq_b_prime_res, "b = b' mod q proof failed");

        let e_lt_2k_plus_1_res = ComputeLessThanSnarks::verify_as_subprotocol(
            trans,
            &proof.e_lt_2k_plus_1_proof,
            statistics,
        );
        res &= e_lt_2k_plus_1_res;
        assert!(e_lt_2k_plus_1_res, "e < 2k + 1 proof failed");

        trans.append_message(b"[Commit]", &proof.a_eq_k_result);
        let a_eq_k_res =
            ComputeEqualitySnarks::verify_as_subprotocol(trans, &proof.a_eq_k_proof, statistics);
        res &= a_eq_k_res;
        assert!(a_eq_k_res, "a = k mod p proof failed");

        trans.append_message(b"[Commit]", &proof.e_eq_2k_result);
        let e_eq_2k_res =
            ComputeEqualitySnarks::verify_as_subprotocol(trans, &proof.e_eq_2k_proof, statistics);
        res &= e_eq_2k_res;
        assert!(e_eq_2k_res, "e = 2k mod p proof failed");

        // a = k && e = 2k
        let point = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            proof.log_num,
        );
        res &=
            proof.a_eq_k_result.evaluate_ext(&point) == proof.e_eq_2k_result.evaluate_ext(&point);

        res
    }
}
