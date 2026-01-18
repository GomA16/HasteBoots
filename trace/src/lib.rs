use algebra::{
    AbstractExtensionField, DenseMultilinearExtension, Field, ListOfProductsOfPolynomials,
};

pub mod acc_trace;
pub mod basic_ops;
pub mod blind_rotation_trace;
pub mod cmp_trace;
pub mod key_switching_trace;
pub mod lookup_trace;
pub mod modulus_switching_trace;
pub mod pbs_trace;

pub use acc_trace::{AccTrace, AccTraceEval, AccTraceMLE};
pub use blind_rotation_trace::{BlindRotationTrace, BlindRotationTraceMLE};

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
    fn num_oracles(&self) -> usize {
        unimplemented!()
    }
    fn log_num_oracles(&self) -> usize {
        match self.num_oracles() {
            1 => 0,
            _ => self.num_oracles().next_power_of_two().trailing_zeros() as usize,
        }
    }
    fn pack_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }
    fn generate_oracle(&self) -> DenseMultilinearExtension<F> {
        let new_nvs = self.num_vars() + self.log_num_oracles();
        let num_zeros = (1 << new_nvs) - (self.num_oracles() << self.num_vars());

        let mut packed_values = self.pack_to_vec();
        packed_values.extend(vec![F::zero(); num_zeros]);
        DenseMultilinearExtension::from_evaluations_vec(new_nvs, packed_values)
    }
}

pub trait SeparatelyPackableTrace<F: Field>: PackableTrace<F> {
    fn num_bit_oracles(&self) -> usize;
    fn num_key_oracles(&self) -> usize {
        unimplemented!()
    }
    fn log_num_bit_oracles(&self) -> usize {
        match self.num_bit_oracles() {
            1 => 0,
            _ => self.num_bit_oracles().next_power_of_two().trailing_zeros() as usize,
        }
    }

    fn log_num_key_oracles(&self) -> usize {
        match self.num_key_oracles() {
            1 => 0,
            _ => self.num_key_oracles().next_power_of_two().trailing_zeros() as usize,
        }
    }
    fn pack_bit_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }
    fn pack_key_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }
    fn generate_bit_oracle(&self) -> DenseMultilinearExtension<F> {
        let new_nvs = self.num_vars() + self.log_num_bit_oracles();
        let num_zeros = (1 << new_nvs) - (self.num_bit_oracles() << self.num_vars());

        let mut packed_values = self.pack_bit_to_vec();
        packed_values.extend(vec![F::zero(); num_zeros]);
        DenseMultilinearExtension::from_evaluations_vec(new_nvs, packed_values)
    }
    fn generate_key_oracle(&self) -> DenseMultilinearExtension<F> {
        let new_nvs = self.num_vars() + self.log_num_key_oracles();
        let num_zeros = (1 << new_nvs) - (self.num_key_oracles() << self.num_vars());

        let mut packed_values = self.pack_key_to_vec();
        packed_values.extend(vec![F::zero(); num_zeros]);
        DenseMultilinearExtension::from_evaluations_vec(new_nvs, packed_values)
    }
}

pub trait PackableEval<F: Field> {
    fn num_evals(&self) -> usize;
    fn pack_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }
    // These two functions are used when the polynomials are stored both in coefficient form and NTT form
    fn pack_poly_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }
    fn pack_ntt_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }
}

pub trait SeparatelyPackableEval<F: Field>: PackableEval<F> {
    fn num_bit_evals(&self) -> usize;
    fn num_key_evals(&self) -> usize {
        unimplemented!()
    }
    fn log_num_bit_evals(&self) -> usize {
        match self.num_bit_evals() {
            1 => 0,
            _ => self.num_bit_evals().next_power_of_two().trailing_zeros() as usize,
        }
    }
    fn log_num_key_evals(&self) -> usize {
        match self.num_key_evals() {
            1 => 0,
            _ => self.num_key_evals().next_power_of_two().trailing_zeros() as usize,
        }
    }
    fn pack_bit_to_vec(&self) -> Vec<F> {
        unimplemented!();
    }
    fn pack_key_to_vec(&self) -> Vec<F> {
        unimplemented!();
    }
    // These two functions are used when the polynomials are stored both in coefficient form and NTT form
    fn pack_bit_poly_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }
    fn pack_bit_ntt_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }
    fn pack_key_poly_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }
    fn pack_key_ntt_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }
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
    fn evaluate_ef(&self, point: &[EF]) -> Self::TraceEvalEF {
        unimplemented!()
    }

    // Lookup evaluation if it has be computed in ProverState of the sumcheck protocol.
    // Otherwise, evaluate it normally.
    // [Optimized with base field * extension field evaluations]
    fn evaluate_ef_with_lookup(
        &self,
        point: &[EF],
        trace_ef: &Self::TraceMLEEF,
        hash_table: &ListOfProductsOfPolynomials<EF>,
        eval_table: &[EF],
    ) -> Self::TraceEvalEF {
        unimplemented!()
    }

    fn evaluate_ef_ntt_only(
        &self,
        eval: &mut Self::TraceEvalEF,
        point: &[EF],
        trace_ef: &Self::TraceMLEEF,
        hash_table: &ListOfProductsOfPolynomials<EF>,
        eval_table: &[EF],
    ) {
        unimplemented!()
    }
}
