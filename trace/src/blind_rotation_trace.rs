use algebra::{AbstractExtensionField, Field, NTTField};
use serde::Serialize;

use crate::basic_ops::decomp_trace::DecompTraceMLE;
use crate::basic_ops::{RLWEEval, SumHadamardTrace, SumHadamardTraceEval, SumHadamardTraceMLE};
use crate::cmp_trace::lt_trace::{LTTables, LTTablesMLE};
use crate::{
    AccTrace, AccTraceEval, AccTraceMLE, ConvertToEF, EvaluableTrace, EvaluableTraceEF,
    PackableEval, PackableTrace, SeparatelyPackableEval, SeparatelyPackableTrace,
};

pub struct BlindRotationParams {
    // log of polynomial coefficient count, denoted as N=2^{log_coeff_count}
    pub log_coeff_count: usize,
    // log of number of blind rotation rounds, denoted as M=2^{log_num_round}
    pub log_num_round: usize,
    // the length of the vector of the decomposed gadgets based on the basis
    pub decomposed_len: usize,
    // the number of all decomposed gadgets
    pub num_bit_poly: usize,
    // the number of key polynomials in NTT form
    pub num_key_ntt: usize,
    // the basis for decomposition of the Field
    pub basis: usize,
}

#[derive(Clone)]
pub struct BlindRotationTrace<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub acc_trace: AccTrace<F>,
    pub hadamard_trace: SumHadamardTrace<F>,
    pub tables: LTTables<F>,
}

pub struct BlindRotationTraceMLE<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub acc_trace: AccTraceMLE<F>,
    pub hadamard_trace: SumHadamardTraceMLE<F>,
    pub lt_tables: LTTablesMLE<F>,
}

#[derive(Serialize, Default)]
pub struct BlindRotationTraceEval<F: Field> {
    pub acc_trace: AccTraceEval<F>,
    pub hadamard_trace: SumHadamardTraceEval<F>,
    pub output_acc: RLWEEval<F>,
}

impl BlindRotationParams {
    pub fn new(
        log_coeff_count: usize,
        log_num_round: usize,
        decomposed_len: usize,
        basis: usize,
    ) -> Self {
        let num_bit_poly = decomposed_len * 2;
        let num_key_ntt = decomposed_len * 4;
        Self {
            log_coeff_count,
            log_num_round,
            decomposed_len,
            num_bit_poly,
            num_key_ntt,
            basis,
        }
    }
}

impl<F: Field> From<BlindRotationTrace<F>> for BlindRotationTraceMLE<F> {
    fn from(trace: BlindRotationTrace<F>) -> Self {
        Self {
            log_coeff_count: trace.acc_trace.log_coeff_count,
            log_num_round: trace.acc_trace.log_num_round,
            acc_trace: AccTraceMLE::from(trace.acc_trace),
            hadamard_trace: SumHadamardTraceMLE::from(trace.hadamard_trace),
            lt_tables: LTTablesMLE::from(trace.tables),
        }
    }
}

impl<F: NTTField> BlindRotationTrace<F> {
    pub fn finalize(&mut self, num_round: usize) {
        self.acc_trace.finalize(num_round);
        self.hadamard_trace.finalize(num_round);
    }

    // Aggregate multiple traces into one trace
    pub fn from_batch_trace(traces: &Vec<BlindRotationTrace<F>>, num_each_poly: usize) -> Self {
        let log_coeff_count = traces[0].log_coeff_count;
        let num_polys = num_each_poly * traces.len();
        let log_num_poly = num_polys.next_power_of_two().trailing_zeros() as usize;

        let mut new_trace = Self {
            log_coeff_count,
            log_num_round: log_num_poly,
            acc_trace: AccTrace::new(log_coeff_count, log_num_poly),
            hadamard_trace: SumHadamardTrace::new(
                traces[0].hadamard_trace.num_hadamard,
                log_coeff_count,
                log_num_poly,
            ),
            tables: traces[0].tables.clone(),
        };

        for trace in traces.iter() {
            new_trace.acc_trace.append_trace(&trace.acc_trace);
            new_trace.hadamard_trace.append_trace(&trace.hadamard_trace);
        }
        new_trace
    }
}

impl<F: Field> PackableTrace<F> for BlindRotationTraceMLE<F> {
    #[inline]
    fn num_vars(&self) -> usize {
        self.log_coeff_count + self.log_num_round
    }

    #[inline]
    fn num_oracles(&self) -> usize {
        self.hadamard_trace.num_oracles() + self.acc_trace.num_oracles()
    }

    #[inline]
    fn pack_to_vec(&self) -> Vec<F> {
        self.hadamard_trace
            .pack_to_vec()
            .into_iter()
            .chain(self.acc_trace.pack_to_vec().into_iter())
            .collect()
    }
}

impl<F: Field> PackableTrace<F> for BlindRotationTrace<F> {
    #[inline]
    fn num_vars(&self) -> usize {
        self.log_coeff_count + self.log_num_round
    }

    #[inline]
    fn num_oracles(&self) -> usize {
        self.hadamard_trace.num_oracles() + self.acc_trace.num_oracles()
    }

    #[inline]
    fn pack_to_vec(&self) -> Vec<F> {
        self.hadamard_trace
            .pack_to_vec()
            .into_iter()
            .chain(self.acc_trace.pack_to_vec().into_iter())
            .collect()
    }
}

impl<F: Field> SeparatelyPackableTrace<F> for BlindRotationTrace<F> {
    #[inline]
    fn num_bit_oracles(&self) -> usize {
        self.hadamard_trace.num_bit_oracles() + self.acc_trace.num_oracles()
    }

    #[inline]
    fn num_key_oracles(&self) -> usize {
        self.hadamard_trace.num_key_oracles()
    }

    #[inline]
    fn pack_bit_to_vec(&self) -> Vec<F> {
        self.hadamard_trace
            .pack_bit_to_vec()
            .into_iter()
            .chain(self.acc_trace.pack_to_vec().into_iter())
            .collect()
    }

    #[inline]
    fn pack_key_to_vec(&self) -> Vec<F> {
        self.hadamard_trace.pack_key_to_vec()
    }
}

impl<F: Field> PackableTrace<F> for Vec<BlindRotationTrace<F>> {
    #[inline]
    fn num_vars(&self) -> usize {
        self[0].num_vars()
    }
}

impl<F: Field> SeparatelyPackableTrace<F> for Vec<BlindRotationTrace<F>> {
    #[inline]
    fn num_bit_oracles(&self) -> usize {
        self.iter().map(|trace| trace.num_bit_oracles()).sum()
    }

    #[inline]
    fn pack_bit_to_vec(&self) -> Vec<F> {
        self.iter()
            .flat_map(|trace| trace.pack_bit_to_vec().into_iter())
            .collect()
    }
}

impl<F: Field> PackableEval<F> for BlindRotationTraceEval<F> {
    #[inline]
    fn num_evals(&self) -> usize {
        self.hadamard_trace.num_evals() + self.acc_trace.num_evals()
    }

    #[inline]
    fn pack_ntt_to_vec(&self) -> Vec<F> {
        self.hadamard_trace
            .pack_ntt_to_vec()
            .into_iter()
            .chain(self.acc_trace.pack_ntt_to_vec().into_iter())
            .collect()
    }

    #[inline]
    fn pack_poly_to_vec(&self) -> Vec<F> {
        self.hadamard_trace
            .pack_poly_to_vec()
            .into_iter()
            .chain(self.acc_trace.pack_poly_to_vec().into_iter())
            .collect()
    }

    #[inline]
    fn pack_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }
}

impl<F: Field> SeparatelyPackableEval<F> for BlindRotationTraceEval<F> {
    #[inline]
    fn num_bit_evals(&self) -> usize {
        self.hadamard_trace.num_bit_evals() + self.acc_trace.num_evals()
    }

    #[inline]
    fn num_key_evals(&self) -> usize {
        self.hadamard_trace.num_key_evals()
    }

    #[inline]
    fn pack_bit_ntt_to_vec(&self) -> Vec<F> {
        self.hadamard_trace
            .pack_bit_ntt_to_vec()
            .into_iter()
            .chain(self.acc_trace.pack_ntt_to_vec().into_iter())
            .collect()
    }

    #[inline]
    fn pack_key_ntt_to_vec(&self) -> Vec<F> {
        self.hadamard_trace.pack_key_ntt_to_vec()
    }
}

impl<F: Field> PackableEval<F> for Vec<BlindRotationTraceEval<F>> {
    #[inline]
    fn num_evals(&self) -> usize {
        self.iter().map(|eval| eval.num_evals()).sum()
    }
}

impl<F: Field> SeparatelyPackableEval<F> for Vec<BlindRotationTraceEval<F>> {
    #[inline]
    fn num_bit_evals(&self) -> usize {
        self.iter().map(|eval| eval.num_bit_evals()).sum()
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> EvaluableTraceEF<F, EF> for BlindRotationTraceMLE<F> {
    type TraceMLEEF = BlindRotationTraceMLE<EF>;
    type TraceEvalEF = BlindRotationTraceEval<EF>;
    fn evaluate_ef(&self, point: &[EF]) -> BlindRotationTraceEval<EF> {
        let acc_trace = self.acc_trace.evaluate_ef(point);
        let hadamard_trace = self.hadamard_trace.evaluate_ef(point);
        let output_acc = RLWEEval {
            poly: (
                acc_trace.input_acc.poly.0 + hadamard_trace.sum_prod.poly.0,
                acc_trace.input_acc.poly.1 + hadamard_trace.sum_prod.poly.1,
            ),
            ntt: (
                acc_trace.input_acc.ntt.0 + hadamard_trace.sum_prod.ntt.0,
                acc_trace.input_acc.ntt.1 + hadamard_trace.sum_prod.ntt.1,
            ),
        };
        BlindRotationTraceEval {
            acc_trace,
            hadamard_trace,
            output_acc,
        }
    }

    fn evaluate_ef_with_lookup(
        &self,
        point: &[EF],
        trace_ef: &Self::TraceMLEEF,
        hash_table: &algebra::ListOfProductsOfPolynomials<EF>,
        eval_table: &[EF],
    ) -> Self::TraceEvalEF {
        let acc_trace = self.acc_trace.evaluate_ef_with_lookup(
            point,
            &trace_ef.acc_trace,
            hash_table,
            eval_table,
        );
        let hadamard_trace = self.hadamard_trace.evaluate_ef_with_lookup(
            point,
            &trace_ef.hadamard_trace,
            hash_table,
            eval_table,
        );
        let output_acc = RLWEEval {
            poly: (
                acc_trace.input_acc.poly.0 + hadamard_trace.sum_prod.poly.0,
                acc_trace.input_acc.poly.1 + hadamard_trace.sum_prod.poly.1,
            ),
            ntt: (
                acc_trace.input_acc.ntt.0 + hadamard_trace.sum_prod.ntt.0,
                acc_trace.input_acc.ntt.1 + hadamard_trace.sum_prod.ntt.1,
            ),
        };
        BlindRotationTraceEval {
            acc_trace,
            hadamard_trace,
            output_acc,
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
        self.acc_trace.evaluate_ef_ntt_only(
            &mut eval.acc_trace,
            point,
            &trace_ef.acc_trace,
            hash_table,
            eval_table,
        );
        self.hadamard_trace.evaluate_ef_ntt_only(
            &mut eval.hadamard_trace,
            point,
            &trace_ef.hadamard_trace,
            hash_table,
            eval_table,
        );
        eval.output_acc.ntt = (
            eval.acc_trace.input_acc.ntt.0 + eval.hadamard_trace.sum_prod.ntt.0,
            eval.acc_trace.input_acc.ntt.1 + eval.hadamard_trace.sum_prod.ntt.1,
        );
    }
}

impl<F: Field> EvaluableTrace<F> for BlindRotationTraceMLE<F> {
    type TraceEval = BlindRotationTraceEval<F>;
    fn evaluate(&self, point: &[F]) -> BlindRotationTraceEval<F> {
        let acc_trace = self.acc_trace.evaluate(point);
        let hadamard_trace = self.hadamard_trace.evaluate(point);
        let out_acc = RLWEEval {
            poly: (
                acc_trace.input_acc.poly.0 + hadamard_trace.sum_prod.poly.0,
                acc_trace.input_acc.poly.1 + hadamard_trace.sum_prod.poly.1,
            ),
            ntt: (
                acc_trace.input_acc.ntt.0 + hadamard_trace.sum_prod.ntt.0,
                acc_trace.input_acc.ntt.1 + hadamard_trace.sum_prod.ntt.1,
            ),
        };
        BlindRotationTraceEval {
            acc_trace,
            hadamard_trace,
            output_acc: out_acc,
        }
    }

    fn evaluate_with_lookup(
        &self,
        point: &[F],
        poly: &algebra::ListOfProductsOfPolynomials<F>,
        eval_table: &[F],
    ) -> Self::TraceEval {
        let acc_trace = self.acc_trace.evaluate_with_lookup(point, poly, eval_table);
        let hadamard_trace = self
            .hadamard_trace
            .evaluate_with_lookup(point, poly, eval_table);
        let output_acc = RLWEEval {
            poly: (
                acc_trace.input_acc.poly.0 + hadamard_trace.sum_prod.poly.0,
                acc_trace.input_acc.poly.1 + hadamard_trace.sum_prod.poly.1,
            ),
            ntt: (
                acc_trace.input_acc.ntt.0 + hadamard_trace.sum_prod.ntt.0,
                acc_trace.input_acc.ntt.1 + hadamard_trace.sum_prod.ntt.1,
            ),
        };
        BlindRotationTraceEval {
            acc_trace,
            hadamard_trace,
            output_acc,
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for BlindRotationTraceMLE<F> {
    type Output = BlindRotationTraceMLE<EF>;

    fn to_ef(&self) -> Self::Output {
        BlindRotationTraceMLE {
            log_coeff_count: self.log_coeff_count,
            log_num_round: self.log_num_round,
            acc_trace: self.acc_trace.to_ef(),
            hadamard_trace: self.hadamard_trace.to_ef(),
            lt_tables: self.lt_tables.to_ef(),
        }
    }
}

impl<F: Field> BlindRotationTraceMLE<F> {
    pub fn extract_decomposition_traces(&self) -> Vec<DecompTraceMLE<F>> {
        let lt_tables = &self.lt_tables;
        assert_eq!(
            self.hadamard_trace.vec_hadamard.len(),
            lt_tables.decomp_len * 2
        );
        let log_num = self.log_coeff_count + self.log_num_round;
        let decomp_len = lt_tables.decomp_len;
        let extract_bits = |start_idx: usize| {
            self.hadamard_trace.vec_hadamard[start_idx..start_idx + decomp_len]
                .iter()
                .map(|hadamard| hadamard.bit.poly.clone())
                .collect::<Vec<_>>()
        };

        let bits_0 = extract_bits(0);
        let bits_1 = extract_bits(decomp_len);

        let trace_0 = DecompTraceMLE {
            log_num,
            basis_bits: self.lt_tables.basis_bits,
            decomp_len,
            bits: bits_0,
            input: self.acc_trace.external_product_input.poly.0.clone(),
        };
        let trace_1 = DecompTraceMLE {
            log_num,
            basis_bits: self.lt_tables.basis_bits,
            decomp_len,
            bits: bits_1,
            input: self.acc_trace.external_product_input.poly.1.clone(),
        };
        vec![trace_0, trace_1]
    }
}
