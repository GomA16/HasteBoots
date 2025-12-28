use algebra::{
    AbstractExtensionField, DenseMultilinearExtension, Field, ListOfProductsOfPolynomials,
};

pub mod acc_trace;
pub mod hadamard_trace;
pub mod lookup_trace;
pub mod ntt_trace;
pub mod pbs_trace;
pub mod rlwe_trace;
pub mod row_perm_trace;

pub use acc_trace::{AccTrace, AccTraceMLE};
pub use hadamard_trace::{
    HadamardTrace, HadamardTraceMLE, SumHadamardTrace, SumHadamardTraceEval, SumHadamardTraceMLE,
};
pub use ntt_trace::{NTTTrace, NTTTraceMLE};
pub use pbs_trace::{PBSTrace, PBSTraceMLE};
use sumcheck::prover::ProverState;

pub trait ConvertToEF<F: Field, EF: AbstractExtensionField<F>> {
    type Output;
    fn to_ef(&self) -> Self::Output;
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for Vec<F> {
    type Output = Vec<EF>;

    fn to_ef(&self) -> Self::Output {
        self.iter().map(|&b| EF::from_base(b)).collect()
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for DenseMultilinearExtension<F> {
    type Output = DenseMultilinearExtension<EF>;

    fn to_ef(&self) -> Self::Output {
        DenseMultilinearExtension::from_evaluations_vec(self.num_vars, self.evaluations.to_ef())
    }
}

pub trait PackableTrace<F: Field> {
    fn num_vars(&self) -> usize;
    fn num_oracles(&self) -> usize;
    fn log_num_oracles(&self) -> usize {
        self.num_oracles().next_power_of_two().trailing_zeros() as usize
    }
    fn pack_to_vec(&self) -> Vec<F>;
    fn generate_oracle(&self) -> DenseMultilinearExtension<F> {
        let new_nvs = self.num_vars() + self.log_num_oracles();
        let num_zeros = (1 << new_nvs) - (self.num_oracles() << self.num_vars());

        let mut packed_values = self.pack_to_vec();
        packed_values.extend(vec![F::zero(); num_zeros]);
        DenseMultilinearExtension::from_evaluations_vec(new_nvs, packed_values)
    }
}

pub trait PackableEval<F: Field> {
    fn num_evals(&self) -> usize;
    fn pack_to_vec(&self) -> Vec<F>;
    // These two functions are used when the polynomials are stored both in coefficient form and NTT form
    fn pack_poly_to_vec(&self) -> Vec<F>;
    fn pack_ntt_to_vec(&self) -> Vec<F>;
}

pub trait EvaluableTrace<F: Field> {
    type TraceEval;
    fn evaluate(&self, point: &[F]) -> Self::TraceEval;
    // Lookup evaluation if it has be computed in ProverState of the sumcheck protocol.
    // Otherwise, evaluate it normally.
    fn evaluate_with_lookup(
        &self,
        point: &[F],
        hash_table: &ListOfProductsOfPolynomials<F>,
        eval_table: &[F],
    ) -> Self::TraceEval;
}

pub trait EvaluableTraceEF<F: Field, EF: AbstractExtensionField<F>> {
    type TraceMLEEF;
    type TraceEvalEF;
    fn evaluate_ef(&self, point: &[EF]) -> Self::TraceEvalEF;
    // Lookup evaluation if it has be computed in ProverState of the sumcheck protocol.
    // Otherwise, evaluate it normally.
    // [Optimized with base field * extension field evaluations]
    fn evaluate_ef_with_lookup(
        &self,
        point: &[EF],
        trace_ef: &Self::TraceMLEEF,
        hash_table: &ListOfProductsOfPolynomials<EF>,
        eval_table: &[EF],
    ) -> Self::TraceEvalEF;
}

pub trait LookupableTraceEF<F: Field, EF: AbstractExtensionField<F>> {
    type TraceEvalEF;
}
