use crate::{
    AccTrace, AccTraceMLE, ConvertToEF, EvaluableTrace, EvaluableTraceEF, HadamardTrace,
    PackableEval, PackableTrace, SumHadamardTrace, SumHadamardTraceEval, SumHadamardTraceMLE,
    acc_trace::AccTraceEval,
};
use algebra::{AbstractExtensionField, Field, NTTField};
use serde::Serialize;
use std::{iter::chain, rc::Rc};

pub struct PBSParameters {
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

pub struct PBSTrace<F: Field> {
    pub acc_trace: AccTrace<F>,
    pub hadamard_trace: SumHadamardTrace<F>,
    // pub params: PBSParameters,
}

pub struct PBSTraceMLE<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub acc_trace: AccTraceMLE<F>,
    pub hadamard_trace: SumHadamardTraceMLE<F>,
    // pub params: PBSParameters,
}

#[derive(Serialize)]
pub struct PBSTraceEval<F: Field> {
    pub acc_trace: AccTraceEval<F>,
    pub hadamard_trace: SumHadamardTraceEval<F>,
}

impl PBSParameters {
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

impl<F: NTTField> From<PBSTrace<F>> for PBSTraceMLE<F> {
    fn from(trace: PBSTrace<F>) -> Self {
        Self {
            log_coeff_count: trace.acc_trace.log_coeff_count,
            log_num_round: trace.acc_trace.log_num_round,
            acc_trace: AccTraceMLE::from(trace.acc_trace),
            hadamard_trace: SumHadamardTraceMLE::from(trace.hadamard_trace),
            // params: trace.params,
        }
    }
}

impl<F: Field> PBSTrace<F> {
    pub fn finalize(&mut self, num_round: usize) {
        self.acc_trace.finalize(num_round);
        self.hadamard_trace.finalize(num_round);
    }
}

impl<F: Field> PackableTrace<F> for PBSTraceMLE<F> {
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

impl<F: Field> PackableEval<F> for PBSTraceEval<F> {
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

impl<F: Field, EF: AbstractExtensionField<F>> EvaluableTraceEF<F, EF> for PBSTraceMLE<F> {
    type TraceEvalEF = PBSTraceEval<EF>;
    fn evaluate_ef(&self, point: &[EF]) -> PBSTraceEval<EF> {
        PBSTraceEval {
            acc_trace: self.acc_trace.evaluate_ef(point),
            hadamard_trace: self.hadamard_trace.evaluate_ef(point),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for PBSTraceMLE<F> {
    type Output = PBSTraceMLE<EF>;

    fn to_ef(&self) -> Self::Output {
        PBSTraceMLE {
            log_coeff_count: self.log_coeff_count,
            log_num_round: self.log_num_round,
            acc_trace: self.acc_trace.to_ef(),
            hadamard_trace: self.hadamard_trace.to_ef(),
        }
    }
}
