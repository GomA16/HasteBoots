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
use crate::{ConvertToEF, EvaluableTraceEF};

/// Store the traces of each round of Hadamard product during blind rotation.
#[derive(Clone)]
pub struct HadamardTrace<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub poly: PolynomialTrace<F>,
    pub rlwe: RLWETrace<F>,
}

pub struct SumHadamardTrace<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub num_trace: usize,
    pub vec_trace: Vec<HadamardTrace<F>>,
    // sum_prod = \sum poly * rlwe
    pub sum_prod: RLWETrace<F>,
}

pub struct HadamardTraceMLE<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub poly: PolynomialTraceMLE<F>,
    pub rlwe: RLWETraceMLE<F>,
}

#[derive(Serialize)]
pub struct HadamardTraceEval<F: Field> {
    pub poly_eval: PolynomialEval<F>,
    pub rlwe_eval: RLWEEval<F>,
}

pub struct SumHadamardTraceMLE<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub num_trace: usize,
    pub vec_trace: Vec<HadamardTraceMLE<F>>,
    pub sum_prod: RLWETraceMLE<F>,
}

#[derive(Serialize)]
pub struct SumHadamardTraceEval<F: Field> {
    pub vec_trace: Vec<HadamardTraceEval<F>>,
    pub sum_prod: RLWEEval<F>,
}

impl<F: Field> SumHadamardTraceEval<F> {
    #[inline]
    pub fn pack_bit_ntt_to_vec(&self) -> Vec<F> {
        self.vec_trace
            .iter()
            .map(|trace| &trace.poly_eval.ntt)
            .cloned()
            .collect()
    }

    #[inline]
    pub fn pack_bit_poly_to_vec(&self) -> Vec<F> {
        self.vec_trace
            .iter()
            .map(|trace| &trace.poly_eval.poly)
            .cloned()
            .collect()
    }

    #[inline]
    pub fn pack_key_ntt_to_vec(&self) -> Vec<F> {
        self.vec_trace
            .iter()
            .flat_map(|trace| [trace.rlwe_eval.ntt.0, trace.rlwe_eval.ntt.1])
            .collect()
    }

    #[inline]
    pub fn pack_key_poly_to_vec(&self) -> Vec<F> {
        self.vec_trace
            .iter()
            .flat_map(|trace| [trace.rlwe_eval.poly.0, trace.rlwe_eval.poly.1])
            .collect()
    }

    #[inline]
    pub fn pack_all_ntt_to_vec(&self) -> Vec<F> {
        let mut overall_vec = self.pack_bit_ntt_to_vec();
        overall_vec.extend(self.pack_key_ntt_to_vec());
        overall_vec.extend(vec![self.sum_prod.ntt.0, self.sum_prod.ntt.1]);
        overall_vec
    }

    #[inline]
    pub fn pack_all_poly_to_vec(&self) -> Vec<F> {
        let mut overall_vec = self.pack_bit_poly_to_vec();
        overall_vec.extend(self.pack_key_poly_to_vec());
        overall_vec.extend(vec![self.sum_prod.poly.0, self.sum_prod.poly.1]);
        overall_vec
    }
}

impl<F: NTTField> From<HadamardTrace<F>> for HadamardTraceMLE<F> {
    #[inline]
    fn from(trace: HadamardTrace<F>) -> Self {
        Self {
            log_coeff_count: trace.log_coeff_count,
            log_num_round: trace.log_num_round,
            poly: PolynomialTraceMLE::from(trace.poly),
            rlwe: RLWETraceMLE::from(trace.rlwe),
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
            sum_prod: RLWETraceMLE::from(trace.sum_prod),
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
            poly: self.poly.to_ef(),
            rlwe: self.rlwe.to_ef(),
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
            sum_prod: self.sum_prod.to_ef(),
        }
    }
}

impl<F: Field> SumHadamardTraceMLE<F> {
    pub fn iter(&self) -> impl Iterator<Item = &HadamardTraceMLE<F>> {
        self.vec_trace.iter()
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> EvaluableTraceEF<F, EF> for HadamardTraceMLE<F> {
    type TraceEvalEF = HadamardTraceEval<EF>;

    #[inline]
    fn evaluate_ef(&self, point: &[EF]) -> Self::TraceEvalEF {
        Self::TraceEvalEF {
            poly_eval: self.poly.evaluate_ef(point),
            rlwe_eval: self.rlwe.evaluate_ef(point),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> EvaluableTraceEF<F, EF> for SumHadamardTraceMLE<F> {
    type TraceEvalEF = SumHadamardTraceEval<EF>;

    #[inline]
    fn evaluate_ef(&self, point: &[EF]) -> Self::TraceEvalEF {
        Self::TraceEvalEF {
            vec_trace: self
                .vec_trace
                .iter()
                .map(|trace| trace.evaluate_ef(point))
                .collect(),
            sum_prod: self.sum_prod.evaluate_ef(point),
        }
    }
}

impl<F: NTTField> HadamardTrace<F> {
    pub fn new(log_coeff_count: usize, log_num_poly: usize) -> Self {
        Self {
            log_coeff_count,
            log_num_round: log_num_poly,
            poly: PolynomialTrace::new(log_coeff_count, log_num_poly),
            rlwe: RLWETrace::new(log_coeff_count, log_num_poly),
        }
    }

    #[inline]
    pub fn append_bit_poly(&mut self, poly: &[F]) {
        self.poly.poly.extend_from_slice(poly);
    }

    #[inline]
    pub fn append_bit_ntt(&mut self, bit_ntt: &[F]) {
        self.poly.ntt.extend_from_slice(bit_ntt);
    }

    #[inline]
    pub fn append_key_ntt(&mut self, key_ntt: (&[F], &[F])) {
        self.rlwe.ntt.0.extend_from_slice(key_ntt.0);
        self.rlwe.ntt.1.extend_from_slice(key_ntt.1);
        let ntt_table = F::get_ntt_table(self.log_coeff_count as u32).unwrap();
        let mut rlwe_poly_0 = key_ntt.0.to_vec();
        let mut rlwe_poly_1 = key_ntt.1.to_vec();
        ntt_table.inverse_transform_slice(&mut rlwe_poly_0);
        ntt_table.inverse_transform_slice(&mut rlwe_poly_1);
        self.rlwe.poly.0.extend_from_slice(rlwe_poly_0.as_slice());
        self.rlwe.poly.1.extend_from_slice(rlwe_poly_1.as_slice());
    }

    #[inline]
    pub fn finalize(&mut self, num_round: usize) {
        if !num_round.is_power_of_two() {
            let num_zeros = ((1 << self.log_num_round) - num_round) * (1 << self.log_coeff_count);
            self.poly.poly.extend(vec![F::zero(); num_zeros]);
            self.poly.ntt.extend(vec![F::zero(); num_zeros]);
            self.rlwe.ntt.0.extend(vec![F::zero(); num_zeros]);
            self.rlwe.ntt.1.extend(vec![F::zero(); num_zeros]);
            self.rlwe.poly.0.extend(vec![F::zero(); num_zeros]);
            self.rlwe.poly.1.extend(vec![F::zero(); num_zeros]);
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
            sum_prod: RLWETrace::new(log_coeff_count, log_num_poly),
        }
    }

    #[inline]
    pub fn get_trace_mul(&mut self, trace_idx: usize) -> &mut HadamardTrace<F> {
        &mut self.vec_trace[trace_idx]
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
        for trace in self.vec_trace.iter_mut() {
            trace.finalize(num_round);
        }
        if !num_round.is_power_of_two() {
            let num_zeros = ((1 << self.log_num_round) - num_round) * (1 << self.log_coeff_count);
            self.sum_prod.ntt.0.extend(vec![F::zero(); num_zeros]);
            self.sum_prod.ntt.1.extend(vec![F::zero(); num_zeros]);
            self.sum_prod.poly.0.extend(vec![F::zero(); num_zeros]);
            self.sum_prod.poly.1.extend(vec![F::zero(); num_zeros]);
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
            .flat_map(|trace| trace.poly.poly.iter())
            .cloned()
            .collect()
    }

    #[inline]
    pub fn generate_key_vec(&self) -> Vec<F> {
        self.vec_trace
            .iter()
            .flat_map(|trace| trace.rlwe.poly.0.iter().chain(trace.rlwe.poly.1.iter()))
            .cloned()
            .collect()
    }

    #[inline]
    pub fn generate_sum_prod_vec(&self) -> Vec<F> {
        self.sum_prod
            .poly
            .0
            .iter()
            .cloned()
            .chain(self.sum_prod.poly.1.iter().cloned())
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
            .map(|trace| trace.poly.poly.clone())
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
            .map(|trace| trace.poly.poly.clone())
            .collect::<Vec<_>>();
        let num_vars = self.log_coeff_count + self.log_num_round;
        LookupTraceMLENormalTable {
            num_vars,
            range,
            vec_input,
        }
    }
}
