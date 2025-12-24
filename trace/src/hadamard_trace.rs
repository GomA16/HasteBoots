use std::rc::Rc;

use algebra::{
    AbstractExtensionField, DenseMultilinearExtension, Field, NTTField, transformation::AbstractNTT,
};
use rayon::iter::IntoParallelRefIterator;
use serde::Serialize;

use crate::lookup_trace::normal_table::LookupTraceMLE as LookupTraceMLENormalTable;
use crate::lookup_trace::small_table::LookupTraceMLE as LookupTraceMLESmallTable;
use crate::{ConvertToEF, EvaluableTraceEF, NTTTraceMLE, PackableEval, PackableTrace};

/// Store the traces of each round of Hadamard product during blind rotation.
#[derive(Debug, Clone)]
pub struct HadamardTrace<F: NTTField> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub bit_poly: Vec<F>,
    pub bit_ntt: Vec<F>,
    pub key_ntt: (Vec<F>, Vec<F>),
    pub key_poly: (Vec<F>, Vec<F>),
}

pub struct SumHadamardTrace<F: NTTField> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub num_trace: usize,
    pub vec_trace: Vec<HadamardTrace<F>>,
    // sum_prod_ntt = \sum bit_ntt * key_ntt
    pub sum_prod_ntt: (Vec<F>, Vec<F>),
    pub sum_prod_poly: (Vec<F>, Vec<F>),
}

#[derive(Clone)]
pub struct HadamardTraceMLE<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub bit_poly: Rc<DenseMultilinearExtension<F>>,
    pub bit_ntt: Rc<DenseMultilinearExtension<F>>,
    pub key_ntt: (
        Rc<DenseMultilinearExtension<F>>,
        Rc<DenseMultilinearExtension<F>>,
    ),
    pub key_poly: (
        Rc<DenseMultilinearExtension<F>>,
        Rc<DenseMultilinearExtension<F>>,
    ),
}

#[derive(Serialize)]
pub struct HadamardTraceEval<F: Field> {
    pub bit_poly: F,
    pub bit_ntt: F,
    pub key_poly: (F, F),
    pub key_ntt: (F, F),
}

pub struct SumHadamardTraceMLE<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub num_trace: usize,
    pub vec_trace: Vec<HadamardTraceMLE<F>>,
    pub sum_prod_ntt: (
        Rc<DenseMultilinearExtension<F>>,
        Rc<DenseMultilinearExtension<F>>,
    ),
    pub sum_prod_poly: (
        Rc<DenseMultilinearExtension<F>>,
        Rc<DenseMultilinearExtension<F>>,
    ),
}

#[derive(Serialize)]
pub struct SumHadamardTraceEval<F: Field> {
    pub vec_trace: Vec<HadamardTraceEval<F>>,
    pub sum_prod_ntt: (F, F),
    pub sum_prod_poly: (F, F),
}

impl<F: Field> SumHadamardTraceEval<F> {
    #[inline]
    pub fn pack_bit_ntt_to_vec(&self) -> Vec<F> {
        self.vec_trace
            .iter()
            .map(|trace| &trace.bit_ntt)
            .cloned()
            .collect()
    }

    #[inline]
    pub fn pack_bit_poly_to_vec(&self) -> Vec<F> {
        self.vec_trace
            .iter()
            .map(|trace| &trace.bit_poly)
            .cloned()
            .collect()
    }

    #[inline]
    pub fn pack_key_ntt_to_vec(&self) -> Vec<F> {
        self.vec_trace
            .iter()
            .flat_map(|trace| [trace.key_ntt.0, trace.key_ntt.1])
            .collect()
    }

    #[inline]
    pub fn pack_key_poly_to_vec(&self) -> Vec<F> {
        self.vec_trace
            .iter()
            .flat_map(|trace| [trace.key_poly.0, trace.key_poly.1])
            .collect()
    }

    #[inline]
    pub fn pack_all_ntt_to_vec(&self) -> Vec<F> {
        let mut overall_vec = self.pack_bit_ntt_to_vec();
        overall_vec.extend(self.pack_key_ntt_to_vec());
        overall_vec.extend(vec![self.sum_prod_ntt.0, self.sum_prod_ntt.1]);
        overall_vec
    }

    #[inline]
    pub fn pack_all_poly_to_vec(&self) -> Vec<F> {
        let mut overall_vec = self.pack_bit_poly_to_vec();
        overall_vec.extend(self.pack_key_poly_to_vec());
        overall_vec.extend(vec![self.sum_prod_poly.0, self.sum_prod_poly.1]);
        overall_vec
    }
}

impl<F: NTTField> From<HadamardTrace<F>> for HadamardTraceMLE<F> {
    #[inline]
    fn from(trace: HadamardTrace<F>) -> Self {
        let num_vars = trace.log_coeff_count + trace.log_num_round;
        let bit_poly_mle =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, trace.bit_poly);
        let bit_ntt_mle = DenseMultilinearExtension::from_evaluations_vec(num_vars, trace.bit_ntt);
        let key_mle_0 = DenseMultilinearExtension::from_evaluations_vec(num_vars, trace.key_ntt.0);
        let key_mle_1 = DenseMultilinearExtension::from_evaluations_vec(num_vars, trace.key_ntt.1);
        let key_poly_mle_0 =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, trace.key_poly.0);
        let key_poly_mle_1 =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, trace.key_poly.1);
        Self {
            log_coeff_count: trace.log_coeff_count,
            log_num_round: trace.log_num_round,
            bit_poly: Rc::new(bit_poly_mle),
            bit_ntt: Rc::new(bit_ntt_mle),
            key_ntt: (Rc::new(key_mle_0), Rc::new(key_mle_1)),
            key_poly: (Rc::new(key_poly_mle_0), Rc::new(key_poly_mle_1)),
        }
    }
}

impl<F: NTTField> From<SumHadamardTrace<F>> for SumHadamardTraceMLE<F> {
    #[inline]
    fn from(trace: SumHadamardTrace<F>) -> Self {
        Self {
            log_coeff_count: trace.log_coeff_count,
            log_num_round: trace.log_num_round,
            num_trace: trace.num_trace,
            vec_trace: trace
                .vec_trace
                .into_iter()
                .map(HadamardTraceMLE::from)
                .collect(),
            sum_prod_ntt: (
                Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                    trace.log_coeff_count + trace.log_num_round,
                    trace.sum_prod_ntt.0,
                )),
                Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                    trace.log_coeff_count + trace.log_num_round,
                    trace.sum_prod_ntt.1,
                )),
            ),
            sum_prod_poly: (
                Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                    trace.log_coeff_count + trace.log_num_round,
                    trace.sum_prod_poly.0,
                )),
                Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                    trace.log_coeff_count + trace.log_num_round,
                    trace.sum_prod_poly.1,
                )),
            ),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for HadamardTraceMLE<F> {
    type Output = HadamardTraceMLE<EF>;

    #[inline]
    fn into_ef(self) -> Self::Output {
        unimplemented!("into_ef for HadamardTraceMLE is not supported yet");
    }

    #[inline]
    fn to_ef(&self) -> Self::Output {
        Self::Output {
            log_coeff_count: self.log_coeff_count,
            log_num_round: self.log_num_round,
            bit_poly: Rc::new(self.bit_poly.to_ef()),
            bit_ntt: Rc::new(self.bit_ntt.to_ef()),
            key_ntt: (
                Rc::new(self.key_ntt.0.to_ef()),
                Rc::new(self.key_ntt.1.to_ef()),
            ),
            key_poly: (
                Rc::new(self.key_poly.0.to_ef()),
                Rc::new(self.key_poly.1.to_ef()),
            ),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for SumHadamardTraceMLE<F> {
    type Output = SumHadamardTraceMLE<EF>;
    #[inline]
    fn into_ef(self) -> Self::Output {
        unimplemented!("into_ef for SumHadamardTraceMLE is not supported yet");
    }

    #[inline]
    fn to_ef(&self) -> Self::Output {
        Self::Output {
            log_coeff_count: self.log_coeff_count,
            log_num_round: self.log_num_round,
            num_trace: self.num_trace,
            vec_trace: self.vec_trace.iter().map(|trace| trace.to_ef()).collect(),
            sum_prod_ntt: (
                Rc::new(self.sum_prod_ntt.0.to_ef()),
                Rc::new(self.sum_prod_ntt.1.to_ef()),
            ),
            sum_prod_poly: (
                Rc::new(self.sum_prod_poly.0.to_ef()),
                Rc::new(self.sum_prod_poly.1.to_ef()),
            ),
        }
    }
}

impl<F: Field> SumHadamardTraceMLE<F> {
    pub fn iter(&self) -> impl Iterator<Item = &HadamardTraceMLE<F>> {
        self.vec_trace.iter()
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> EvaluableTraceEF<F, EF> for HadamardTraceMLE<F> {
    type TraceEval = HadamardTraceEval<F>;
    type TraceEvalEF = HadamardTraceEval<EF>;
    #[inline]
    fn evaluate(&self, point: &[F]) -> Self::TraceEval {
        Self::TraceEval {
            bit_poly: self.bit_poly.evaluate(point),
            bit_ntt: self.bit_ntt.evaluate(point),
            key_ntt: (
                self.key_ntt.0.evaluate(point),
                self.key_ntt.1.evaluate(point),
            ),
            key_poly: (
                self.key_poly.0.evaluate(point),
                self.key_poly.1.evaluate(point),
            ),
        }
    }

    #[inline]
    fn evaluate_ef(&self, point: &[EF]) -> Self::TraceEvalEF {
        Self::TraceEvalEF {
            bit_poly: self.bit_poly.evaluate_ext(point),
            bit_ntt: self.bit_ntt.evaluate_ext(point),
            key_ntt: (
                self.key_ntt.0.evaluate_ext(point),
                self.key_ntt.1.evaluate_ext(point),
            ),
            key_poly: (
                self.key_poly.0.evaluate_ext(point),
                self.key_poly.1.evaluate_ext(point),
            ),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> EvaluableTraceEF<F, EF> for SumHadamardTraceMLE<F> {
    type TraceEval = SumHadamardTraceEval<F>;
    type TraceEvalEF = SumHadamardTraceEval<EF>;

    #[inline]
    fn evaluate(&self, point: &[F]) -> Self::TraceEval {
        Self::TraceEval {
            vec_trace: self
                .vec_trace
                .iter()
                .map(|trace| {
                    <HadamardTraceMLE<F> as EvaluableTraceEF<F, EF>>::evaluate(trace, point)
                })
                .collect(),
            sum_prod_ntt: (
                self.sum_prod_ntt.0.evaluate(point),
                self.sum_prod_ntt.1.evaluate(point),
            ),
            sum_prod_poly: (
                self.sum_prod_poly.0.evaluate(point),
                self.sum_prod_poly.1.evaluate(point),
            ),
        }
    }

    #[inline]
    fn evaluate_ef(&self, point: &[EF]) -> Self::TraceEvalEF {
        Self::TraceEvalEF {
            vec_trace: self
                .vec_trace
                .iter()
                .map(|trace| trace.evaluate_ef(point))
                .collect(),
            sum_prod_ntt: (
                self.sum_prod_ntt.0.evaluate_ext(point),
                self.sum_prod_ntt.1.evaluate_ext(point),
            ),
            sum_prod_poly: (
                self.sum_prod_poly.0.evaluate_ext(point),
                self.sum_prod_poly.1.evaluate_ext(point),
            ),
        }
    }
}

impl<F: NTTField> HadamardTrace<F> {
    pub fn new(log_coeff_count: usize, log_num_poly: usize) -> Self {
        Self {
            log_coeff_count,
            log_num_round: log_num_poly,
            bit_poly: Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
            bit_ntt: Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
            key_ntt: (
                Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
                Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
            ),
            key_poly: (
                Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
                Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
            ),
        }
    }

    #[inline]
    pub fn append_bit_poly(&mut self, bit_poly: &[F]) {
        self.bit_poly.extend_from_slice(bit_poly);
    }

    #[inline]
    pub fn append_bit_ntt(&mut self, bit_ntt: &[F]) {
        self.bit_ntt.extend_from_slice(bit_ntt);
    }

    #[inline]
    pub fn append_key_ntt(&mut self, key_ntt: (&[F], &[F])) {
        self.key_ntt.0.extend_from_slice(key_ntt.0);
        self.key_ntt.1.extend_from_slice(key_ntt.1);
        let ntt_table = F::get_ntt_table(self.log_coeff_count as u32).unwrap();
        let mut key_poly_0 = key_ntt.0.to_vec();
        let mut key_poly_1 = key_ntt.1.to_vec();
        ntt_table.inverse_transform_slice(&mut key_poly_0);
        ntt_table.inverse_transform_slice(&mut key_poly_1);
        self.key_poly.0.extend_from_slice(key_poly_0.as_slice());
        self.key_poly.1.extend_from_slice(key_poly_1.as_slice());
    }

    #[inline]
    pub fn finalize(&mut self, num_round: usize) {
        if !num_round.is_power_of_two() {
            let num_zeros = ((1 << self.log_num_round) - num_round) * (1 << self.log_coeff_count);
            self.bit_poly.extend(vec![F::zero(); num_zeros]);
            self.bit_ntt.extend(vec![F::zero(); num_zeros]);
            self.key_ntt.0.extend(vec![F::zero(); num_zeros]);
            self.key_ntt.1.extend(vec![F::zero(); num_zeros]);
            self.key_poly.0.extend(vec![F::zero(); num_zeros]);
            self.key_poly.1.extend(vec![F::zero(); num_zeros]);
        }
    }
}

impl<F: NTTField> SumHadamardTrace<F> {
    #[inline]
    pub fn new(num_trace: usize, log_coeff_count: usize, log_num_poly: usize) -> Self {
        Self {
            log_coeff_count,
            log_num_round: log_num_poly,
            num_trace,
            vec_trace: vec![HadamardTrace::new(log_coeff_count, log_num_poly); num_trace],
            sum_prod_ntt: (
                Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
                Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
            ),
            sum_prod_poly: (
                Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
                Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
            ),
        }
    }

    #[inline]
    pub fn get_trace_mul(&mut self, trace_idx: usize) -> &mut HadamardTrace<F> {
        &mut self.vec_trace[trace_idx]
    }

    #[inline]
    pub fn add_sum_prod_ntt(&mut self, sum_prod: (&[F], &[F])) {
        self.sum_prod_ntt.0.extend_from_slice(sum_prod.0);
        self.sum_prod_ntt.1.extend_from_slice(sum_prod.1);
    }

    #[inline]
    pub fn add_sum_prod_poly(&mut self, sum_prod: (&[F], &[F])) {
        self.sum_prod_poly.0.extend_from_slice(sum_prod.0);
        self.sum_prod_poly.1.extend_from_slice(sum_prod.1);
    }

    #[inline]
    pub fn finalize(&mut self, num_round: usize) {
        for trace in self.vec_trace.iter_mut() {
            trace.finalize(num_round);
        }
        if !num_round.is_power_of_two() {
            let num_zeros = ((1 << self.log_num_round) - num_round) * (1 << self.log_coeff_count);
            self.sum_prod_ntt.0.extend(vec![F::zero(); num_zeros]);
            self.sum_prod_ntt.1.extend(vec![F::zero(); num_zeros]);
            self.sum_prod_poly.0.extend(vec![F::zero(); num_zeros]);
            self.sum_prod_poly.1.extend(vec![F::zero(); num_zeros]);
        }
    }
}

impl<F: Field> SumHadamardTraceMLE<F> {
    #[inline]
    pub fn num_vars(&self) -> usize {
        self.log_coeff_count + self.log_num_round
    }

    #[inline]
    pub fn num_bit_poly(&self) -> usize {
        self.num_trace
    }

    #[inline]
    pub fn num_key_poly(&self) -> usize {
        self.num_trace * 2
    }

    #[inline]
    pub fn num_all_poly(&self) -> usize {
        // #bit_poly = num_trace
        // #key_ntt = num_trace * 2
        // #sum_prod = 2
        self.num_trace * 3 + 2
    }

    #[inline]
    pub fn log_num_bit_poly(&self) -> usize {
        self.num_bit_poly().next_power_of_two().trailing_zeros() as usize
    }

    #[inline]
    pub fn log_num_key_poly(&self) -> usize {
        self.num_key_poly().next_power_of_two().trailing_zeros() as usize
    }

    #[inline]
    pub fn log_num_all_poly(&self) -> usize {
        (self.num_bit_poly() + self.num_key_poly() + 2)
            .next_power_of_two()
            .trailing_zeros() as usize
    }

    #[inline]
    pub fn log_num_helper_poly(&self, blk_size: usize) -> usize {
        let num_lookup = self.num_bit_poly();
        let num_helper = (num_lookup + blk_size - 1) / blk_size;
        num_helper.next_power_of_two().trailing_zeros() as usize
    }

    #[inline]
    pub fn generate_bit_vec(&self) -> Vec<F> {
        self.vec_trace
            .iter()
            .flat_map(|trace| trace.bit_poly.iter())
            .cloned()
            .collect()
    }

    #[inline]
    pub fn generate_key_vec(&self) -> Vec<F> {
        self.vec_trace
            .iter()
            .flat_map(|trace| trace.key_poly.0.iter().chain(trace.key_poly.1.iter()))
            .cloned()
            .collect()
    }

    #[inline]
    pub fn generate_sum_prod_vec(&self) -> Vec<F> {
        self.sum_prod_poly
            .0
            .iter()
            .cloned()
            .chain(self.sum_prod_poly.1.iter().cloned())
            .collect()
    }

    #[inline]
    pub fn generate_bit_oracle(&self) -> DenseMultilinearExtension<F> {
        let mut bit_polys = self.generate_bit_vec();
        let num_vars = bit_polys.len().next_power_of_two().trailing_zeros() as usize;
        let num_zeros = (1 << num_vars) - bit_polys.len();
        bit_polys.extend(vec![F::zero(); num_zeros]);
        DenseMultilinearExtension::from_evaluations_vec(num_vars, bit_polys)
    }

    #[inline]
    pub fn generate_key_oracle(&self) -> DenseMultilinearExtension<F> {
        let mut key_ntts = self.generate_key_vec();
        let num_vars = key_ntts.len().next_power_of_two().trailing_zeros() as usize;
        let num_zeros = (1 << num_vars) - key_ntts.len();
        key_ntts.extend(vec![F::zero(); num_zeros]);
        DenseMultilinearExtension::from_evaluations_vec(num_vars, key_ntts)
    }

    #[inline]
    pub fn generate_all_oracle(&self) -> DenseMultilinearExtension<F> {
        let mut overall_vec = self.generate_bit_vec();
        overall_vec.extend(self.generate_key_vec());
        overall_vec.extend(self.generate_sum_prod_vec());
        let num_vars = overall_vec.len().next_power_of_two().trailing_zeros() as usize;
        let num_zeros = (1 << num_vars) - overall_vec.len();
        overall_vec.extend(vec![F::zero(); num_zeros]);
        DenseMultilinearExtension::from_evaluations_vec(num_vars, overall_vec)
    }

    #[inline]
    pub fn extract_lookup_trace_mle_small_table(
        &self,
        range: usize,
    ) -> LookupTraceMLESmallTable<F> {
        let vec_input = self
            .vec_trace
            .iter()
            .map(|trace| trace.bit_poly.clone())
            .collect::<Vec<_>>();
        let num_vars = self.log_coeff_count + self.log_num_round;
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
            .vec_trace
            .iter()
            .map(|trace| trace.bit_poly.clone())
            .collect::<Vec<_>>();
        let num_vars = self.log_coeff_count + self.log_num_round;
        LookupTraceMLENormalTable {
            num_vars,
            range,
            vec_input,
        }
    }
}
