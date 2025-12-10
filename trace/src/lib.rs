use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};

mod acc_trace;
mod hadamard_prod_trace;
mod ntt_trace;

// pub use hadamard_prod_trace::{
//     HadamardProdTrace, HadamardProdTraceMLE, HadamardProdsTrace,
// };
pub use acc_trace::AccTrace;
pub use hadamard_prod_trace::{
    BatchedHadamardTrace, BatchedHadamardTraceMLE, HadamardTrace, HadamardTraceMLE,
};
pub use ntt_trace::{NTTInstanceInfo, NTTTrace, NTTTraceInfo, NTTTraceMLE};

pub trait FieldTrace<F: Field> {
    type EFInfo;
    fn get_commit_poly(&self) -> DenseMultilinearExtension<F>;
    fn info_ef<EF: AbstractExtensionField<F>>(&self) -> Self::EFInfo;
}

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

pub trait PackTrace<F: Field> {
    type TraceType;

    fn num_vars(&self) -> usize;
    fn num_oracles(&self) -> usize;
    fn log_num_oracles(&self) -> usize {
        self.num_oracles().next_power_of_two().trailing_zeros() as usize
    }
    fn pack_to_vec(&self) -> Vec<F>;
    fn generate_oracle(&self) -> DenseMultilinearExtension<F>;
}
