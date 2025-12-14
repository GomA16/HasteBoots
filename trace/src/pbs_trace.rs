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

// The witness generated from PBS trace.
pub struct PBSWitness<F: NTTField> {
    pub lookup_witness: LookupTraceMLE<F>,
}

// The witness generated with help of randomness
pub struct PBSRandomWitness<F: NTTField> {
    pub lookup_witness_helper: LookupWitnessHelper<F>,
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

impl<F: NTTField> PBSTraceMLE<F> {
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
