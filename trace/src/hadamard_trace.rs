use std::rc::Rc;

use algebra::{
    AbstractExtensionField, DenseMultilinearExtension, Field, NTTField, transformation::AbstractNTT,
};
use serde::Serialize;

use crate::lookup_trace::normal_table::LookupTraceMLE as LookupTraceMLENormalTable;
use crate::lookup_trace::small_table::LookupTraceMLE as LookupTraceMLESmallTable;
use crate::rlwe_trace::{
    PolynomialEval, PolynomialTrace, PolynomialTraceMLE, RLWEEval, RLWETrace, RLWETraceMLE,
};
use crate::{ConvertToEF, EvaluableTraceEF, PackableEval, PackableTrace};

/// Store the traces of the multiplication between a bit polynomial and an RLWE ciphertext
#[derive(Clone)]
pub struct HadamardTrace<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_poly: usize,
    // bit is a small polynomial
    pub bit: PolynomialTrace<F>,
    // rlwe is an RLWE ciphertext
    pub rlwe: RLWETrace<F>,
}

/// Store the traces of the multiplications between multiple bit polynomials and RLWE ciphertexts
/// sum_prod = \sum bit * rlwe
pub struct SumHadamardTrace<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_poly: usize,
    pub num_hadamard: usize,
    pub vec_hadamard: Vec<HadamardTrace<F>>,
    // sum_prod = \sum bit * rlwe
    pub sum_prod: RLWETrace<F>,
}

pub struct HadamardTraceMLE<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_poly: usize,
    pub bit: PolynomialTraceMLE<F>,
    pub rlwe: RLWETraceMLE<F>,
}

#[derive(Serialize)]
pub struct HadamardTraceEval<F: Field> {
    pub bit: PolynomialEval<F>,
    pub rlwe: RLWEEval<F>,
}

pub struct SumHadamardTraceMLE<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_poly: usize,
    pub num_hadamard: usize,
    pub vec_hadamard: Vec<HadamardTraceMLE<F>>,
    pub sum_prod: RLWETraceMLE<F>,
}

#[derive(Serialize)]
pub struct SumHadamardTraceEval<F: Field> {
    pub vec_hadamard: Vec<HadamardTraceEval<F>>,
    pub sum_prod: RLWEEval<F>,
}

impl<F: NTTField> From<HadamardTrace<F>> for HadamardTraceMLE<F> {
    #[inline]
    fn from(trace: HadamardTrace<F>) -> Self {
        Self {
            log_coeff_count: trace.log_coeff_count,
            log_num_poly: trace.log_num_poly,
            bit: PolynomialTraceMLE::from(trace.bit),
            rlwe: RLWETraceMLE::from(trace.rlwe),
        }
    }
}

impl<F: NTTField> From<SumHadamardTrace<F>> for SumHadamardTraceMLE<F> {
    #[inline]
    fn from(trace: SumHadamardTrace<F>) -> Self {
        Self {
            log_coeff_count: trace.log_coeff_count,
            log_num_poly: trace.log_num_poly,
            num_hadamard: trace.num_hadamard,
            vec_hadamard: trace
                .vec_hadamard
                .into_iter()
                .map(HadamardTraceMLE::from)
                .collect(),
            sum_prod: RLWETraceMLE::from(trace.sum_prod),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for HadamardTraceMLE<F> {
    type Output = HadamardTraceMLE<EF>;

    #[inline]
    fn to_ef(&self) -> Self::Output {
        Self::Output {
            log_coeff_count: self.log_coeff_count,
            log_num_poly: self.log_num_poly,
            bit: self.bit.to_ef(),
            rlwe: self.rlwe.to_ef(),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for SumHadamardTraceMLE<F> {
    type Output = SumHadamardTraceMLE<EF>;

    #[inline]
    fn to_ef(&self) -> Self::Output {
        Self::Output {
            log_coeff_count: self.log_coeff_count,
            log_num_poly: self.log_num_poly,
            num_hadamard: self.num_hadamard,
            vec_hadamard: self
                .vec_hadamard
                .iter()
                .map(|trace| trace.to_ef())
                .collect(),
            sum_prod: self.sum_prod.to_ef(),
        }
    }
}

impl<F: Field> SumHadamardTraceMLE<F> {
    pub fn iter(&self) -> impl Iterator<Item = &HadamardTraceMLE<F>> {
        self.vec_hadamard.iter()
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> EvaluableTraceEF<F, EF> for HadamardTraceMLE<F> {
    type TraceEvalEF = HadamardTraceEval<EF>;

    #[inline]
    fn evaluate_ef(&self, point: &[EF]) -> Self::TraceEvalEF {
        Self::TraceEvalEF {
            bit: self.bit.evaluate_ef(point),
            rlwe: self.rlwe.evaluate_ef(point),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> EvaluableTraceEF<F, EF> for SumHadamardTraceMLE<F> {
    type TraceEvalEF = SumHadamardTraceEval<EF>;

    #[inline]
    fn evaluate_ef(&self, point: &[EF]) -> Self::TraceEvalEF {
        Self::TraceEvalEF {
            vec_hadamard: self
                .vec_hadamard
                .iter()
                .map(|trace| trace.evaluate_ef(point))
                .collect(),
            sum_prod: self.sum_prod.evaluate_ef(point),
        }
    }
}

impl<F: Field> HadamardTrace<F> {
    pub fn new(log_coeff_count: usize, log_num_poly: usize) -> Self {
        Self {
            log_coeff_count,
            log_num_poly,
            bit: PolynomialTrace::new(log_coeff_count, log_num_poly),
            rlwe: RLWETrace::new(log_coeff_count, log_num_poly),
        }
    }

    #[inline]
    pub fn append_bit_poly(&mut self, poly: &[F]) {
        self.bit.poly.extend_from_slice(poly);
    }

    #[inline]
    pub fn append_bit_ntt(&mut self, bit_ntt: &[F]) {
        self.bit.ntt.extend_from_slice(bit_ntt);
    }

    #[inline]
    pub fn finalize(&mut self, num_round: usize) {
        self.bit.finalize(num_round);
        self.rlwe.finalize(num_round);
    }
}

impl<F: NTTField> HadamardTrace<F> {
    #[inline]
    pub fn append_rlwe_ntt(&mut self, rlwe_ntt: (&[F], &[F])) {
        self.rlwe.ntt.0.extend_from_slice(rlwe_ntt.0);
        self.rlwe.ntt.1.extend_from_slice(rlwe_ntt.1);
        let ntt_table = F::get_ntt_table(self.log_coeff_count as u32).unwrap();
        let mut rlwe_poly_0 = rlwe_ntt.0.to_vec();
        let mut rlwe_poly_1 = rlwe_ntt.1.to_vec();
        ntt_table.inverse_transform_slice(&mut rlwe_poly_0);
        ntt_table.inverse_transform_slice(&mut rlwe_poly_1);
        self.rlwe.poly.0.extend_from_slice(rlwe_poly_0.as_slice());
        self.rlwe.poly.1.extend_from_slice(rlwe_poly_1.as_slice());
    }
}

impl<F: Field> SumHadamardTrace<F> {
    #[inline]
    pub fn new(num_trace: usize, log_coeff_count: usize, log_num_poly: usize) -> Self {
        Self {
            log_coeff_count,
            log_num_poly,
            num_hadamard: num_trace,
            vec_hadamard: vec![HadamardTrace::new(log_coeff_count, log_num_poly); num_trace],
            sum_prod: RLWETrace::new(log_coeff_count, log_num_poly),
        }
    }

    #[inline]
    pub fn get_trace_mul(&mut self, trace_idx: usize) -> &mut HadamardTrace<F> {
        &mut self.vec_hadamard[trace_idx]
    }

    #[inline]
    pub fn add_sum_prod_ntt(&mut self, sum_prod: (&[F], &[F])) {
        self.sum_prod.ntt.0.extend_from_slice(sum_prod.0);
        self.sum_prod.ntt.1.extend_from_slice(sum_prod.1);
    }

    #[inline]
    pub fn add_sum_prod_poly(&mut self, sum_prod: (&[F], &[F])) {
        self.sum_prod.poly.0.extend_from_slice(sum_prod.0);
        self.sum_prod.poly.1.extend_from_slice(sum_prod.1);
    }

    #[inline]
    pub fn finalize(&mut self, num_round: usize) {
        for trace in self.vec_hadamard.iter_mut() {
            trace.finalize(num_round);
        }
        self.sum_prod.finalize(num_round);
    }
}

impl<F: Field> SumHadamardTraceMLE<F> {
    #[inline]
    pub fn num_bit_poly(&self) -> usize {
        self.num_hadamard
    }

    #[inline]
    pub fn log_num_helper_poly(&self, blk_size: usize) -> usize {
        let num_lookup = self.num_bit_poly();
        let num_helper = (num_lookup + blk_size - 1) / blk_size;
        num_helper.next_power_of_two().trailing_zeros() as usize
    }

    #[inline]
    pub fn extract_lookup_trace_mle_small_table(
        &self,
        range: usize,
    ) -> LookupTraceMLESmallTable<F> {
        let vec_input = self
            .vec_hadamard
            .iter()
            .map(|trace| trace.bit.poly.clone())
            .collect::<Vec<_>>();
        let num_vars = self.log_coeff_count + self.log_num_poly;
        LookupTraceMLESmallTable {
            num_vars,
            range,
            vec_input,
        }
    }

    #[inline]
    pub fn extract_lookup_trace_mle_normal_table(
        &self,
        range: usize,
    ) -> LookupTraceMLENormalTable<F> {
        let vec_input = self
            .vec_hadamard
            .iter()
            .map(|trace| trace.bit.poly.clone())
            .collect::<Vec<_>>();
        let num_vars = self.log_coeff_count + self.log_num_poly;
        LookupTraceMLENormalTable {
            num_vars,
            range,
            vec_input,
        }
    }
}

impl<F: Field> PackableTrace<F> for HadamardTraceMLE<F> {
    fn num_vars(&self) -> usize {
        self.log_coeff_count + self.log_num_poly
    }

    fn num_oracles(&self) -> usize {
        3
    }

    fn pack_to_vec(&self) -> Vec<F> {
        self.bit
            .pack_to_vec()
            .into_iter()
            .chain(self.rlwe.pack_to_vec().into_iter())
            .collect()
    }
}

impl<F: Field> PackableEval<F> for HadamardTraceEval<F> {
    #[inline]
    fn num_evals(&self) -> usize {
        3
    }

    #[inline]
    fn pack_ntt_to_vec(&self) -> Vec<F> {
        self.bit
            .pack_ntt_to_vec()
            .into_iter()
            .chain(self.rlwe.pack_ntt_to_vec().into_iter())
            .collect()
    }

    #[inline]
    fn pack_poly_to_vec(&self) -> Vec<F> {
        self.bit
            .pack_poly_to_vec()
            .into_iter()
            .chain(self.rlwe.pack_poly_to_vec().into_iter())
            .collect()
    }

    #[inline]
    fn pack_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }
}

impl<F: Field> PackableTrace<F> for SumHadamardTraceMLE<F> {
    fn num_vars(&self) -> usize {
        self.log_coeff_count + self.log_num_poly
    }

    fn num_oracles(&self) -> usize {
        self.num_hadamard * self.vec_hadamard[0].num_oracles() + self.sum_prod.num_oracles()
    }

    fn pack_to_vec(&self) -> Vec<F> {
        self.vec_hadamard
            .iter()
            .flat_map(|trace| trace.pack_to_vec().into_iter())
            .chain(self.sum_prod.pack_to_vec().into_iter())
            .collect()
    }
}

impl<F: Field> PackableEval<F> for SumHadamardTraceEval<F> {
    #[inline]
    fn num_evals(&self) -> usize {
        self.vec_hadamard[0].num_evals() * self.vec_hadamard.len() + self.sum_prod.num_evals()
    }

    #[inline]
    fn pack_ntt_to_vec(&self) -> Vec<F> {
        self.vec_hadamard
            .iter()
            .flat_map(|trace| trace.pack_ntt_to_vec().into_iter())
            .chain(self.sum_prod.pack_ntt_to_vec().into_iter())
            .collect()
    }

    #[inline]
    fn pack_poly_to_vec(&self) -> Vec<F> {
        self.vec_hadamard
            .iter()
            .flat_map(|trace| trace.pack_poly_to_vec().into_iter())
            .chain(self.sum_prod.pack_poly_to_vec().into_iter())
            .collect()
    }

    #[inline]
    fn pack_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }
}
