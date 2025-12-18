use crate::{AccTrace, AccTraceMLE, BatchedHadamardTrace, BatchedHadamardTraceMLE, HadamardTrace};
use crate::{LookupTraceMLE, LookupWitnessHelper};
use algebra::{Field, NTTField};
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

pub struct PBSTrace<F: NTTField> {
    pub acc_trace: AccTrace<F>,
    pub hadamard_trace_a: BatchedHadamardTrace<F>,
    pub hadamard_trace_b: BatchedHadamardTrace<F>,
    pub params: PBSParameters,
}

pub struct PBSTraceMLE<F: NTTField> {
    pub acc_trace: AccTraceMLE<F>,
    pub hadamard_trace_a: BatchedHadamardTraceMLE<F>,
    pub hadamard_trace_b: BatchedHadamardTraceMLE<F>,
    pub params: PBSParameters,
}

impl PBSParameters {
    pub fn new(log_coeff_count: usize, log_num_round: usize, decomposed_len:usize, basis: usize) -> Self {
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
            acc_trace: AccTraceMLE::from(trace.acc_trace),
            hadamard_trace_a: BatchedHadamardTraceMLE::from(trace.hadamard_trace_a),
            hadamard_trace_b: BatchedHadamardTraceMLE::from(trace.hadamard_trace_b),
            params: trace.params,
        }
    }
}

// TODO: use 
impl<F: NTTField> PBSTraceMLE<F> {
    pub fn num_vars(&self) -> usize {
        self.params.log_coeff_count + self.params.log_num_round
    }

    // pub fn get_key_ntt_oracle(&self) -> Vec<F> {
    // }

    pub fn num_bit_poly(&self) -> usize {
        self.params.decomposed_len * 2
    }

    // helper functions for lookup
    pub fn helper_num_vars(&self, blk_size: usize) -> usize {
        let total = self.num_bit_poly() + 1;
        let num_blks = (total + blk_size - 1) / blk_size;
        self.num_vars() + num_blks.next_power_of_two().trailing_zeros() as usize
    }

    pub fn get_lookup_trace(&self) -> LookupTraceMLE<F> {
        let vec_input = self
            .hadamard_trace_a
            .iter()
            .chain(self.hadamard_trace_b.iter())
            .map(|trace| Rc::clone(&trace.bit_poly))
            .collect();

        LookupTraceMLE {
            num_vars: self.params.log_coeff_count + self.params.log_num_round,
            range: self.params.basis,
            vec_input,
        }
    }
}
