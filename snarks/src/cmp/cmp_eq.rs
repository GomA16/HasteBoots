use core::time;

use algebra::{
    AbstractExtensionField, AsInto, DecomposableField, DenseMultilinearExtension, Field,
};
use bincode::{config::standard, de};
use helper::{FiatShamirTranscript, Transcript, utils::compute_oracle_evals};
use pcs::{PolynomialCommitmentScheme, utils::code};
use piop::{
    SumcheckInstance, SumcheckPIOP,
    grand_prod::{GrandProdInfo, GrandProdInstance, GrandProdPIOP, GrandProdProof},
};
use serde::{Serialize, ser};
use trace::{
    ConvertToEF, PackableTrace,
    basic_ops::decomp_trace::DecompTraceMLE,
    cmp_trace::{
        eq_trace::{self, EQTablesMLE, EQTraceMLE},
        lt_trace::LTTablesMLE,
    },
};

use crate::lookup::indexed_table::indexed_batch::{
    BatchedIndexedLogUpParams, BatchedIndexedLogUpSnarks, BatchedIndexedLogUpSnarksProof,
};

#[derive(Default)]
pub struct ComputeEqualitySnarks<F, EF, S, PCS>
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

pub struct ComputeEqualityParams<'a, F, S>
where
    F: Field,
{
    pub eq_tables: &'a EQTablesMLE<F>,
    pub code_spec: &'a S,
}

#[derive(Serialize)]
pub struct ComputeEqualityProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone + Serialize,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub lookup_proof: BatchedIndexedLogUpSnarksProof<F, EF, S, PCS>,
    pub equality_info: GrandProdInfo<EF>,
    pub equality_proof: GrandProdProof<EF>,
    pub input: PCS::Polynomial,
    pub basis: F,
}

impl<F, EF, S, PCS> ComputeEqualityProof<F, EF, S, PCS>
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
            + bincode::serde::encode_to_vec(&self.equality_info, standard())
                .unwrap()
                .len()
            + bincode::serde::encode_to_vec(&self.equality_proof, standard())
                .unwrap()
                .len()
    }

    pub fn pcs_proof_len(&self) -> usize {
        self.lookup_proof.pcs_proof_len()
            + bincode::serde::encode_to_vec(&self.input, standard())
                .unwrap()
                .len()
    }
}

impl<'a, F, S> ComputeEqualityParams<'a, F, S>
where
    F: Field,
{
    pub fn new(code_spec: &'a S, eq_tables: &'a EQTablesMLE<F>) -> Self {
        Self {
            code_spec,
            eq_tables,
        }
    }
}

impl<F, EF, S, PCS> ComputeEqualitySnarks<F, EF, S, PCS>
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
        trace: &EQTraceMLE<F>,
        params: &ComputeEqualityParams<F, S>,
    ) -> ComputeEqualityProof<F, EF, S, PCS> {
        Self::prove_as_subprotocol(trans, trace, params)
    }

    pub fn verify(
        &self,
        trans: &mut Transcript<EF>,
        proof: &ComputeEqualityProof<F, EF, S, PCS>,
    ) -> bool {
        Self::verify_as_subprotocol(trans, proof)
    }

    pub fn prove_as_subprotocol(
        trans: &mut Transcript<EF>,
        trace: &EQTraceMLE<F>,
        params: &ComputeEqualityParams<F, S>,
    ) -> ComputeEqualityProof<F, EF, S, PCS> {
        trans.append_message(b"[Commit]", trace.input.as_ref());
        // Prove the decomposition consistency via batched indexed log-up proofs
        // 1. each bit x_i is in range [0, 2^k)
        // 2. \sum_{i} x_i * 2^{i*k} x_i = x
        // The second part is garanteed by ensuring \sum_{i} x_i * 2^{i*k} < p via lookups.
        // refer: https://www.usenix.org/conference/usenixsecurity24/presentation/hao-meng-scalable
        let lookup_trace = trace.extract_eq_lookup_traces(params.eq_tables);
        let lookup_params = BatchedIndexedLogUpParams::new(params.code_spec.clone(), &lookup_trace);
        let lookup_proof =
            BatchedIndexedLogUpSnarks::prove_as_subprotocol(trans, &lookup_trace, &lookup_params);

        let trace_ef: EQTraceMLE<EF> = trace.to_ef();
        let equality_instance = GrandProdInstance::from_eq_trace(&trace_ef);
        let (piop_proof, piop_state) = GrandProdPIOP::<EF>::prover(trans, &equality_instance);

        let basis = 1 << params.eq_tables.basis_bits;
        ComputeEqualityProof {
            lookup_proof,
            equality_info: equality_instance.info(),
            equality_proof: piop_proof,
            input: trace.input.as_ref().to_owned(),
            basis: F::new(basis.as_into()),
        }
    }

    pub fn verify_as_subprotocol(
        trans: &mut Transcript<EF>,
        proof: &ComputeEqualityProof<F, EF, S, PCS>,
    ) -> bool {
        trans.append_message(b"[Commit]", &proof.input);

        // Verify the decomposition consistency via batched indexed log-up proofs
        // It ensures that the decomposition is in range of [0, p)
        // Each bit is in range [0, 2^k)
        let mut res = true;
        let lookup_res =
            BatchedIndexedLogUpSnarks::verify_as_subprotocol(trans, &proof.lookup_proof);
        res &= lookup_res;
        assert!(lookup_res, "Decomposition lookup proof verification failed");

        let input_eval = proof.input.evaluate_ext(&proof.lookup_proof.input_point_r);
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

        let (res_piop, _res_subclaim) =
            GrandProdPIOP::<EF>::verifier(trans, &proof.equality_info, &proof.equality_proof);
        res &= res_piop;
        assert!(res_piop, "Equality grand product proof verification failed");

        res
    }
}

#[cfg(test)]
mod test {
    use core::num;
    use std::rc::Rc;

    use super::*;
    use algebra::{BabyBear, BabyBearExetension, Basis};
    use helper::Transcript;
    use num_traits::One;
    use pcs::{
        multilinear::BrakedownPCS,
        utils::code::{ExpanderCode, ExpanderCodeSpec},
    };
    use trace::cmp_trace::eq_trace::{EQTable, EQTables, EQTrace};

    type FF = BabyBear;
    type EF = BabyBearExetension;
    type Hash = sha2::Sha256;
    const BASE_FIELD_BITS: usize = 31;

    #[test]
    fn test_cmp_eq_snarks() {
        let mut rng = rand::rng();
        let num_vars = 2;
        let scale = FF::new(1 << 10);
        let eq_constant = (-FF::one()) / scale;
        let basis = Basis::<FF>::new(7);

        let eq_tables = EQTables::<FF>::new(eq_constant, &basis);
        let eq_trace = EQTrace::<FF>::random(&mut rng, num_vars, &eq_tables);
        let eq_trace_mle = eq_trace.into();
        let eq_tables_mle = eq_tables.into();

        let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, 31, 10);
        let params = ComputeEqualityParams::new(&code_spec, &eq_tables_mle);
        let snarks = ComputeEqualitySnarks::<
            FF,
            EF,
            ExpanderCodeSpec,
            BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
        >::default();

        let prover_trans = &mut Transcript::<EF>::default();
        let proof = snarks.prove(prover_trans, &eq_trace_mle, &params);

        let verifier_trans = &mut Transcript::<EF>::default();
        let res = snarks.verify(verifier_trans, &proof);
        assert!(res);
    }

    #[test]
    fn test_cmp_custome_eq_snarks() {
        let mut rng = rand::rng();
        let num_vars = 2;
        let scale = FF::new(1 << 10);
        let eq_constant = (-FF::one()) / scale;
        let basis = Basis::<FF>::new(7);

        let input = Rc::new(DenseMultilinearExtension::<FF>::random(num_vars, &mut rng));

        let eq_tables = EQTables::<FF>::new(eq_constant, &basis).into();
        let eq_trace = EQTraceMLE::<FF>::from(&input, &eq_tables);
        let eq_trace_mle = eq_trace.into();
        let eq_tables_mle = eq_tables.into();

        let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, 31, 10);
        let params = ComputeEqualityParams::new(&code_spec, &eq_tables_mle);
        let snarks = ComputeEqualitySnarks::<
            FF,
            EF,
            ExpanderCodeSpec,
            BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
        >::default();

        let prover_trans = &mut Transcript::<EF>::default();
        let proof = snarks.prove(prover_trans, &eq_trace_mle, &params);

        let verifier_trans = &mut Transcript::<EF>::default();
        let res = snarks.verify(verifier_trans, &proof);
        assert!(res);
    }
}
