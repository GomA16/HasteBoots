use std::rc::Rc;

use algebra::{DenseMultilinearExtension, Field, NTTField, transformation::AbstractNTT};
use rayon::iter::IntoParallelRefIterator;

use crate::{EvaluableTrace, LookupTraceMLE, NTTTraceMLE};

/// Store the traces of each round of Hadamard product during blind rotation.
#[derive(Debug, Clone)]
pub struct HadamardTrace<F: NTTField> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub bit_poly: Vec<F>,
    pub bit_ntt: Vec<F>,
    pub key_ntt: (Vec<F>, Vec<F>),
}

pub struct BatchedHadamardTrace<F: NTTField> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub num_trace: usize,
    pub vec_trace: Vec<HadamardTrace<F>>,
    // sum_prod_ntt = \sum bit_ntt * key_ntt
    pub sum_prod_ntt: (Vec<F>, Vec<F>),
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
}

pub struct HadamardTraceEval<F: Field> {
    pub bit_poly: F,
    pub bit_ntt: F,
    pub key_ntt: (F, F),
}

pub struct BatchedHadamardTraceMLE<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub num_trace: usize,
    pub vec_trace: Vec<HadamardTraceMLE<F>>,
    pub sum_prod_ntt: (
        Rc<DenseMultilinearExtension<F>>,
        Rc<DenseMultilinearExtension<F>>,
    ),
}

pub struct BatchedHadamardTraceEval<F: Field> {
    pub vec_trace: Vec<HadamardTraceEval<F>>,
    pub sum_prod_ntt: (F, F),
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

        Self {
            log_coeff_count: trace.log_coeff_count,
            log_num_round: trace.log_num_round,
            bit_poly: Rc::new(bit_poly_mle),
            bit_ntt: Rc::new(bit_ntt_mle),
            key_ntt: (Rc::new(key_mle_0), Rc::new(key_mle_1)),
        }
    }
}

impl<F: NTTField> From<BatchedHadamardTrace<F>> for BatchedHadamardTraceMLE<F> {
    #[inline]
    fn from(trace: BatchedHadamardTrace<F>) -> Self {
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
        }
    }
}

impl<F: Field> BatchedHadamardTraceMLE<F> {
    pub fn iter(&self) -> impl Iterator<Item = &HadamardTraceMLE<F>> {
        self.vec_trace.iter()
    }
}

impl<F: Field> EvaluableTrace<F> for HadamardTraceMLE<F> {
    type TraceEval = HadamardTraceEval<F>;
    #[inline]
    fn evaluate(&self, point: &[F]) -> Self::TraceEval {
        Self::TraceEval {
            bit_poly: self.bit_poly.evaluate(point),
            bit_ntt: self.bit_ntt.evaluate(point),
            key_ntt: (
                self.key_ntt.0.evaluate(point),
                self.key_ntt.1.evaluate(point),
            ),
        }
    }
}

impl<F: Field> EvaluableTrace<F> for BatchedHadamardTraceMLE<F> {
    type TraceEval = BatchedHadamardTraceEval<F>;

    #[inline]
    fn evaluate(&self, point: &[F]) -> Self::TraceEval {
        Self::TraceEval {
            vec_trace: self
                .vec_trace
                .iter()
                .map(|trace| trace.evaluate(point))
                .collect(),
            sum_prod_ntt: (
                self.sum_prod_ntt.0.evaluate(point),
                self.sum_prod_ntt.1.evaluate(point),
            ),
        }
    }
}

impl<F: NTTField> HadamardTrace<F> {
    pub fn new(log_coeff_count: usize, log_num_poly: usize) -> Self {
        Self {
            log_coeff_count,
            log_num_round: log_num_poly,
            // ntt_table: F::get_ntt_table(log_coeff_count as u32).unwrap().root_powers(),
            bit_poly: Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
            bit_ntt: Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
            key_ntt: (
                Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
                Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
            ),
        }
    }

    pub fn append_bit_poly(&mut self, bit_poly: &[F]) {
        self.bit_poly.extend_from_slice(bit_poly);
    }

    pub fn append_bit_ntt(&mut self, bit_ntt: &[F]) {
        self.bit_ntt.extend_from_slice(bit_ntt);
    }

    pub fn append_key_ntt(&mut self, key_poly: (&[F], &[F])) {
        self.key_ntt.0.extend_from_slice(key_poly.0);
        self.key_ntt.1.extend_from_slice(key_poly.1);
    }

    pub fn export_mles(
        self,
    ) -> (
        (DenseMultilinearExtension<F>, DenseMultilinearExtension<F>),
        (DenseMultilinearExtension<F>, DenseMultilinearExtension<F>),
    ) {
        let num_vars = self.log_coeff_count + self.log_num_round;
        let bit_poly_mle = DenseMultilinearExtension::from_evaluations_vec(num_vars, self.bit_poly);
        let bit_ntt_mle = DenseMultilinearExtension::from_evaluations_vec(num_vars, self.bit_ntt);
        let key_mle_0 = DenseMultilinearExtension::from_evaluations_vec(num_vars, self.key_ntt.0);
        let key_mle_1 = DenseMultilinearExtension::from_evaluations_vec(num_vars, self.key_ntt.1);

        ((bit_poly_mle, bit_ntt_mle), (key_mle_0, key_mle_1))
    }
}

impl<F: NTTField> HadamardTraceMLE<F> {
    pub fn extract_ntt_trace_mle(&self) -> NTTTraceMLE<F> {
        NTTTraceMLE {
            log_coeff_count: self.log_coeff_count,
            log_num_ntt: self.log_num_round,
            ntt_table: Rc::new(
                F::get_ntt_table(self.log_coeff_count as u32)
                    .unwrap()
                    .root_powers(),
            ),
            coefficients: Rc::clone(&self.bit_ntt),
            evaluations: Rc::clone(&self.bit_poly),
        }
    }
}

impl<F: NTTField> BatchedHadamardTrace<F> {
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
        }
    }

    pub fn get_trace_mul(&mut self, trace_idx: usize) -> &mut HadamardTrace<F> {
        &mut self.vec_trace[trace_idx]
    }

    pub fn add_sum_prod(&mut self, sum_prod: (&[F], &[F])) {
        self.sum_prod_ntt.0.extend_from_slice(sum_prod.0);
        self.sum_prod_ntt.1.extend_from_slice(sum_prod.1);
    }
}

impl<F: NTTField> BatchedHadamardTraceMLE<F> {
    pub fn extract_random_ntt_trace_mle(&self, randomness: &[F]) -> NTTTraceMLE<F> {
        let size = 1 << (self.log_coeff_count + self.log_num_round);
        let mut rand_coeffs = vec![F::zero(); size];
        let mut rand_evals = vec![F::zero(); size];

        let add_assign = |acc: &mut [F], vec: &[F], r: F| {
            for (a, b) in acc.iter_mut().zip(vec.iter()) {
                *a += r.mul(*b);
            }
        };

        self.vec_trace
            .iter()
            .zip(randomness)
            .for_each(|(trace, r)| {
                add_assign(&mut rand_coeffs, trace.bit_ntt.as_slice(), *r);
                add_assign(&mut rand_evals, trace.bit_poly.as_slice(), *r);
            });

        NTTTraceMLE {
            log_coeff_count: self.log_coeff_count,
            log_num_ntt: self.log_num_round,
            ntt_table: Rc::new(
                F::get_ntt_table(self.log_coeff_count as u32)
                    .unwrap()
                    .root_powers(),
            ),
            coefficients: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                self.log_coeff_count + self.log_num_round,
                rand_coeffs,
            )),
            evaluations: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                self.log_coeff_count + self.log_num_round,
                rand_evals,
            )),
        }
    }

    pub fn extract_lookup_trace_mle(&self, range: usize) -> LookupTraceMLE<F> {
        let vec_input = self
            .vec_trace
            .iter()
            .map(|trace| trace.bit_poly.clone())
            .collect::<Vec<_>>();
        let num_vars = self.log_coeff_count + self.log_num_round;
        LookupTraceMLE {
            num_vars,
            range,
            vec_input,
        }
    }
}
