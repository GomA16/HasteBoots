use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};

mod acc_trace;
mod hadamard_trace;
mod lookup_trace;
mod ntt_trace;
mod pbs_trace;

pub use acc_trace::{AccTrace, AccTraceMLE};
pub use hadamard_trace::{
    BatchedHadamardTrace, BatchedHadamardTraceMLE, HadamardTrace, HadamardTraceMLE,
};
pub use lookup_trace::normal_table::{
    LookupTrace, LookupTraceMLE, LookupWitness, LookupWitnessHelper,
};
pub use ntt_trace::{NTTTrace, NTTTraceInfo, NTTTraceMLE};
pub use pbs_trace::{PBSTrace, PBSTraceMLE};

pub trait ConvertToEF<F: Field, EF: AbstractExtensionField<F>> {
    type Output;
    fn into_ef(self) -> Self::Output;
    fn to_ef(&self) -> Self::Output;
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for Vec<F> {
    type Output = Vec<EF>;

    fn into_ef(self) -> Self::Output {
        self.into_iter().map(EF::from_base).collect()
    }

    fn to_ef(&self) -> Self::Output {
        self.iter().map(|&b| EF::from_base(b)).collect()
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for DenseMultilinearExtension<F> {
    type Output = DenseMultilinearExtension<EF>;

    fn into_ef(self) -> Self::Output {
        DenseMultilinearExtension::from_evaluations_vec(self.num_vars, self.evaluations.into_ef())
    }

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

pub trait EvaluableTrace<F: Field> {
    type TraceEval;
    fn evaluate(&self, point: &[F]) -> Self::TraceEval;
}
