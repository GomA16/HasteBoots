use algebra::{
    AbstractExtensionField, AsInto, DecomposableField, DenseMultilinearExtension, Field,
};
use bincode::config::standard;
use helper::{FiatShamirTranscript, Transcript};
use pcs::PolynomialCommitmentScheme;
use serde::Serialize;
use trace::cmp_trace::lt_trace::{LTTablesMLE, LTTraceMLE};

use crate::{
    SnarkStatistics,
    lookup::indexed_table::indexed_batch::{
        BatchedIndexedLogUpParams, BatchedIndexedLogUpSnarks, BatchedIndexedLogUpSnarksProof,
    },
};

// Note: It omits one lookup out of k + 1 lookup to check the less-than relation.
#[derive(Default)]
pub struct ComputeLessThanSnarks<F, EF, S, PCS>
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

pub struct ComputeLessThanParams<'a, F, S>
where
    F: Field,
    S: Clone,
{
    pub lt_tables: &'a LTTablesMLE<F>,
    pub code_spec: S,
}

#[derive(Serialize)]
pub struct ComputeLessThanProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone + Serialize,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub lookup_proof: BatchedIndexedLogUpSnarksProof<F, EF, S, PCS>,
    pub input: PCS::Polynomial,
    pub basis: F,
}

impl<F, EF, S, PCS> ComputeLessThanProof<F, EF, S, PCS>
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
            + bincode::serde::encode_to_vec(&self.input, standard())
                .unwrap()
                .len()
    }
}

impl<'a, F, S> ComputeLessThanParams<'a, F, S>
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

impl<F, EF, S, PCS> ComputeLessThanSnarks<F, EF, S, PCS>
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
        trace: &LTTraceMLE<F>,
        params: &ComputeLessThanParams<F, S>,
    ) -> ComputeLessThanProof<F, EF, S, PCS> {
        Self::prove_as_subprotocol(trans, trace, params, &mut None)
    }

    pub fn verify(
        &self,
        trans: &mut Transcript<EF>,
        proof: &ComputeLessThanProof<F, EF, S, PCS>,
    ) -> bool {
        Self::verify_as_subprotocol(trans, proof, &mut None)
    }

    pub fn prove_as_subprotocol(
        trans: &mut Transcript<EF>,
        trace: &LTTraceMLE<F>,
        params: &ComputeLessThanParams<F, S>,
        statistic: &mut Option<&mut SnarkStatistics>,
    ) -> ComputeLessThanProof<F, EF, S, PCS> {
        trans.append_message(b"[Commit]", trace.input.as_ref());
        // Prove the decomposition consistency via batched indexed log-up proofs
        // 1. each bit x_i is in range [0, 2^k)
        // 2. \sum_{i} x_i * 2^{i*k} x_i = x
        // The second part is garanteed by ensuring \sum_{i} x_i * 2^{i*k} < p via lookups.
        // refer: https://www.usenix.org/conference/usenixsecurity24/presentation/hao-meng-scalable
        let lookup_trace = trace.extract_lt_lookup_traces(params.lt_tables);
        let lookup_params = BatchedIndexedLogUpParams::new(params.code_spec.clone(), &lookup_trace);
        let lookup_proof = BatchedIndexedLogUpSnarks::prove_as_subprotocol(
            trans,
            &lookup_trace,
            &lookup_params,
            statistic,
        );

        let basis = 1 << params.lt_tables.basis_bits;
        ComputeLessThanProof {
            lookup_proof,
            input: trace.input.as_ref().to_owned(),
            basis: F::new(basis.as_into()),
        }
    }

    pub fn verify_as_subprotocol(
        trans: &mut Transcript<EF>,
        proof: &ComputeLessThanProof<F, EF, S, PCS>,
        statistic: &mut Option<&mut SnarkStatistics>,
    ) -> bool {
        trans.append_message(b"[Commit]", &proof.input);

        // Verify the decomposition consistency via batched indexed log-up proofs
        // It ensures that the decomposition is in range of [0, p)
        // Each bit is in range [0, 2^k)
        let mut res = true;
        let lookup_res =
            BatchedIndexedLogUpSnarks::verify_as_subprotocol(trans, &proof.lookup_proof, statistic);
        res &= lookup_res;
        assert!(lookup_res, "Decomposition lookup proof verification failed");

        let time = std::time::Instant::now();
        let input_eval = proof.input.evaluate_ext(&proof.lookup_proof.input_point_r);
        if let Some(stats) = statistic {
            stats.add_verifier_pcs_time(time.elapsed());
        }

        let mut sum = EF::zero();
        let mut power = EF::one();
        for bit in proof.lookup_proof.lookup_evals.iter() {
            sum += bit.index_at_r * power;
            power = power * proof.basis;
        }
        res &= input_eval == sum;
        assert!(
            input_eval == sum,
            "Decomposition sumcheck verification failed"
        );

        res
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use std::rc::Rc;

    use algebra::{BabyBear, BabyBearExetension, Basis};
    use helper::Transcript;
    use num_traits::One;
    use pcs::{
        multilinear::BrakedownPCS,
        utils::code::{ExpanderCode, ExpanderCodeSpec},
    };
    use trace::cmp_trace::lt_trace::LTTables;

    type FF = BabyBear;
    type EF = BabyBearExetension;
    type Hash = sha2::Sha256;
    const BASE_FIELD_BITS: usize = 31;

    #[test]
    fn test_cmp_lt_snarks() {
        let mut rng = rand::rng();
        let num_vars = 2;
        let scale = FF::new(1 << 10);
        let lt_constant = (-FF::one()) / scale;
        let basis = Basis::new(8);

        let lt_tables = LTTables::<FF>::new(&basis, Some(lt_constant)).into();

        let input = Rc::new(DenseMultilinearExtension::<FF>::random(num_vars, &mut rng));
        let lt_trace = LTTraceMLE::<FF>::from(&input, &lt_tables);
        let lt_tables_mle = LTTablesMLE::<FF>::from(lt_tables);

        let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
        let params = ComputeLessThanParams::new(code_spec, &lt_tables_mle);
        let snarks = ComputeLessThanSnarks::<
            FF,
            EF,
            ExpanderCodeSpec,
            BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
        >::default();

        let prover_trans = &mut Transcript::<EF>::default();
        let proof = snarks.prove(prover_trans, &lt_trace, &params);

        let verifier_trans = &mut Transcript::<EF>::default();
        let res = snarks.verify(verifier_trans, &proof);
        assert!(res);
    }
}
