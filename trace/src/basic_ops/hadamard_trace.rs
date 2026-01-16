use std::iter::Sum;
use std::rc::Rc;

use algebra::AsInto;
use algebra::{
    AbstractExtensionField, DenseMultilinearExtension, Field, NTTField, transformation::AbstractNTT,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::Serialize;

use super::rlwe_trace::{
    PolynomialEval, PolynomialTrace, PolynomialTraceMLE, RLWEEval, RLWETrace, RLWETraceMLE,
};
use crate::basic_ops::decomp_trace::DecompTraceMLE;
use crate::cmp_trace::lt_trace::{LTTables, LTTablesMLE};
use crate::lookup_trace::indexed_table::IndexedLookupTraceMLE;
use crate::lookup_trace::normal_table::LookupTraceMLE as LookupTraceMLENormalTable;
use crate::lookup_trace::small_table::LookupTraceMLE as LookupTraceMLESmallTable;
use crate::{ConvertToEF, EvaluableTrace, EvaluableTraceEF, PackableEval, PackableTrace, SeparatelyPackableEval, SeparatelyPackableTrace};

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

#[derive(Clone)]
pub struct HadamardTraceMLE<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_poly: usize,
    pub bit: PolynomialTraceMLE<F>,
    pub rlwe: RLWETraceMLE<F>,
}

#[derive(Serialize, Default, Clone)]
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

#[derive(Serialize, Default)]
pub struct SumHadamardTraceEval<F: Field> {
    pub vec_hadamard: Vec<HadamardTraceEval<F>>,
    pub sum_prod: RLWEEval<F>,
}

impl<F: Field> From<HadamardTrace<F>> for HadamardTraceMLE<F> {
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

impl<F: Field> From<SumHadamardTrace<F>> for SumHadamardTraceMLE<F> {
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
    type TraceMLEEF = HadamardTraceMLE<EF>;
    type TraceEvalEF = HadamardTraceEval<EF>;

    #[inline]
    fn evaluate_ef(&self, point: &[EF]) -> Self::TraceEvalEF {
        Self::TraceEvalEF {
            bit: self.bit.evaluate_ef(point),
            rlwe: self.rlwe.evaluate_ef(point),
        }
    }

    fn evaluate_ef_with_lookup(
        &self,
        point: &[EF],
        trace_ef: &Self::TraceMLEEF,
        hash_table: &algebra::ListOfProductsOfPolynomials<EF>,
        eval_table: &[EF],
    ) -> Self::TraceEvalEF {
        Self::TraceEvalEF {
            bit: self
                .bit
                .evaluate_ef_with_lookup(point, &trace_ef.bit, hash_table, eval_table),
            rlwe: self
                .rlwe
                .evaluate_ef_with_lookup(point, &trace_ef.rlwe, hash_table, eval_table),
        }
    }

    fn evaluate_ef_ntt_only(
        &self,
        eval: &mut Self::TraceEvalEF,
        point: &[EF],
        trace_ef: &Self::TraceMLEEF,
        hash_table: &algebra::ListOfProductsOfPolynomials<EF>,
        eval_table: &[EF],
    ) {
        self.bit
            .evaluate_ef_ntt_only(&mut eval.bit, point, &trace_ef.bit, hash_table, eval_table);
        self.rlwe.evaluate_ef_ntt_only(
            &mut eval.rlwe,
            point,
            &trace_ef.rlwe,
            hash_table,
            eval_table,
        );
    }
}

impl<F: Field> EvaluableTrace<F> for HadamardTraceMLE<F> {
    type TraceEval = HadamardTraceEval<F>;

    #[inline]
    fn evaluate(&self, point: &[F]) -> Self::TraceEval {
        Self::TraceEval {
            bit: self.bit.evaluate(point),
            rlwe: self.rlwe.evaluate(point),
        }
    }

    fn evaluate_with_lookup(
        &self,
        point: &[F],
        hash_table: &algebra::ListOfProductsOfPolynomials<F>,
        eval_table: &[F],
    ) -> Self::TraceEval {
        Self::TraceEval {
            bit: self.bit.evaluate_with_lookup(point, hash_table, eval_table),
            rlwe: self
                .rlwe
                .evaluate_with_lookup(point, hash_table, eval_table),
        }
    }
}

impl<F: Field> EvaluableTrace<F> for SumHadamardTraceMLE<F> {
    type TraceEval = SumHadamardTraceEval<F>;

    #[inline]
    fn evaluate(&self, point: &[F]) -> Self::TraceEval {
        Self::TraceEval {
            vec_hadamard: self
                .vec_hadamard
                .iter()
                .map(|trace| trace.evaluate(point))
                .collect(),
            sum_prod: self.sum_prod.evaluate(point),
        }
    }

    fn evaluate_with_lookup(
        &self,
        point: &[F],
        hash_table: &algebra::ListOfProductsOfPolynomials<F>,
        eval_table: &[F],
    ) -> Self::TraceEval {
        Self::TraceEval {
            vec_hadamard: self
                .vec_hadamard
                .iter()
                .map(|trace| trace.evaluate_with_lookup(point, hash_table, eval_table))
                .collect(),
            sum_prod: self
                .sum_prod
                .evaluate_with_lookup(point, hash_table, eval_table),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> EvaluableTraceEF<F, EF> for SumHadamardTraceMLE<F> {
    type TraceMLEEF = SumHadamardTraceMLE<EF>;
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

    #[inline]
    fn evaluate_ef_with_lookup(
        &self,
        point: &[EF],
        trace_ef: &Self::TraceMLEEF,
        hash_table: &algebra::ListOfProductsOfPolynomials<EF>,
        eval_table: &[EF],
    ) -> Self::TraceEvalEF {
        Self::TraceEvalEF {
            vec_hadamard: self
                .vec_hadamard
                .iter()
                .zip(trace_ef.vec_hadamard.iter())
                .map(|(trace, trace_ef)| {
                    trace.evaluate_ef_with_lookup(point, trace_ef, hash_table, eval_table)
                })
                .collect(),
            sum_prod: self.sum_prod.evaluate_ef_with_lookup(
                point,
                &trace_ef.sum_prod,
                hash_table,
                eval_table,
            ),
        }
    }

    fn evaluate_ef_ntt_only(
        &self,
        eval: &mut Self::TraceEvalEF,
        point: &[EF],
        trace_ef: &Self::TraceMLEEF,
        hash_table: &algebra::ListOfProductsOfPolynomials<EF>,
        eval_table: &[EF],
    ) {
        if eval.vec_hadamard.is_empty() {
            eval.vec_hadamard = vec![HadamardTraceEval::<EF>::default(); self.vec_hadamard.len()];
        }
        assert_eq!(eval.vec_hadamard.len(), self.vec_hadamard.len());
        for (i, trace) in self.vec_hadamard.iter().enumerate() {
            trace.evaluate_ef_ntt_only(
                &mut eval.vec_hadamard[i],
                point,
                &trace_ef.vec_hadamard[i],
                hash_table,
                eval_table,
            );
        }
        self.sum_prod.evaluate_ef_ntt_only(
            &mut eval.sum_prod,
            point,
            &trace_ef.sum_prod,
            hash_table,
            eval_table,
        );
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

    #[inline]
    pub fn extract_indexed_lookup_trace_mle(
        &self,
        tables: &LTTablesMLE<F>,
    ) -> Vec<IndexedLookupTraceMLE<F>> {
        assert_eq!(self.num_bit_poly(), tables.decomp_len * 2);
        let extract_traces = |start: usize| {
            assert!(start + tables.decomp_len <= self.num_bit_poly());
            self.vec_hadamard[start..start + tables.decomp_len]
                .iter()
                .enumerate()
                .map(|(i, trace)| {
                    let index = trace.bit.poly.clone();
                    let input = tables.lookup(i, &index);
                    IndexedLookupTraceMLE {
                        num_input_vars: self.log_coeff_count + self.log_num_poly,
                        num_table_vars: tables.tables[i].num_vars(),
                        index,
                        input,
                        table: tables.get_table(i),
                        table_point: None,
                    }
                })
                .collect::<Vec<_>>()
        };
        let traces_0 = extract_traces(0);
        let traces_1 = extract_traces(tables.decomp_len);
        traces_0.into_iter().chain(traces_1.into_iter()).collect()
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

impl<F: Field> PackableTrace<F> for HadamardTrace<F> {
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

impl<F: Field> SeparatelyPackableTrace<F> for HadamardTrace<F> {

    fn num_bit_oracles(&self) -> usize {
        1
    }

    fn num_key_oracles(&self) -> usize {
        2
    }

    fn pack_bit_to_vec(&self) -> Vec<F> {
        self.bit.pack_to_vec()
    }

    fn pack_key_to_vec(&self) -> Vec<F> {
        self.rlwe.pack_to_vec()
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
}

impl<F: Field> SeparatelyPackableEval<F> for HadamardTraceEval<F> {
    #[inline]
    fn num_bit_evals(&self) -> usize {
        1
    }

    #[inline]
    fn num_key_evals(&self) -> usize {
        2
    }

    #[inline]
    fn pack_bit_ntt_to_vec(&self) -> Vec<F> {
        self.bit.pack_ntt_to_vec()
    }

    #[inline]
    fn pack_key_ntt_to_vec(&self) -> Vec<F> {
        self.rlwe.pack_ntt_to_vec()
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

impl<F: Field> PackableTrace<F> for SumHadamardTrace<F> {
    fn num_vars(&self) -> usize {
        self.log_coeff_count + self.log_num_poly
    }

    fn num_oracles(&self) -> usize {
        self.num_hadamard * self.vec_hadamard[0].num_oracles() + self.sum_prod.num_oracles()
    }

    fn pack_to_vec(&self) -> Vec<F> {
        let mut result = self
            .vec_hadamard
            .par_iter()
            .flat_map(|trace| trace.pack_to_vec())
            .collect::<Vec<F>>();
        result.extend(self.sum_prod.pack_to_vec());
        result
    }
}

impl<F: Field> SeparatelyPackableTrace<F> for SumHadamardTrace<F> {

    fn num_bit_oracles(&self) -> usize {
        self.num_hadamard * self.vec_hadamard[0].num_bit_oracles() + self.sum_prod.num_oracles()
    }

    fn num_key_oracles(&self) -> usize {
        self.num_hadamard * self.vec_hadamard[0].num_key_oracles()
    }

    fn pack_bit_to_vec(&self) -> Vec<F> {
        self.vec_hadamard
            .iter()
            .flat_map(|trace| trace.pack_bit_to_vec().into_iter())
            .chain(self.sum_prod.pack_to_vec().into_iter())
            .collect()
    }

    fn pack_key_to_vec(&self) -> Vec<F> {
        self.vec_hadamard
            .iter()
            .flat_map(|trace| trace.pack_key_to_vec().into_iter())
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
}

impl<F: Field> SeparatelyPackableEval<F> for SumHadamardTraceEval<F> {
    #[inline]
    fn num_bit_evals(&self) -> usize {
        self.vec_hadamard[0].num_bit_evals() * self.vec_hadamard.len() + self.sum_prod.num_evals()
    }

    #[inline]
    fn num_key_evals(&self) -> usize {
        self.vec_hadamard[0].num_key_evals() * self.vec_hadamard.len()
    }

    #[inline]
    fn pack_bit_ntt_to_vec(&self) -> Vec<F> {
        self.vec_hadamard
            .iter()
            .flat_map(|trace| trace.pack_bit_ntt_to_vec().into_iter())
            .chain(self.sum_prod.pack_ntt_to_vec().into_iter())
            .collect()
    }

    #[inline]
    fn pack_key_ntt_to_vec(&self) -> Vec<F> {
        self.vec_hadamard
            .iter()
            .flat_map(|trace| trace.pack_key_ntt_to_vec().into_iter())
            .collect()
    }
}