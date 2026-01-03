use algebra::{AbstractExtensionField, Field};

use crate::{
    ConvertToEF,
    basic_ops::{
        RowPermTrace, RowPermTraceMLE, SumHadamardTrace, SumHadamardTraceMLE,
        row_perm_trace::PermutationSignedInfo,
    },
};

pub struct KeySwitchingTrace<F: Field> {
    pub log_lwe_dim: usize,
    pub log_rlwe_dim: usize,
    pub log_coeff_count: usize,
    pub hadamard_trace: SumHadamardTrace<F>,
    pub permutation_trace: Option<RowPermTrace<F>>,
}

pub struct KeySwitchingTraceMLE<F: Field> {
    pub log_lwe_dim: usize,
    pub log_rlwe_dim: usize,
    pub log_coeff_count: usize,
    pub hadamard_trace: SumHadamardTraceMLE<F>,
    pub permutation_trace: Option<RowPermTraceMLE<F>>,
}

impl<F: Field> From<KeySwitchingTrace<F>> for KeySwitchingTraceMLE<F> {
    fn from(trace: KeySwitchingTrace<F>) -> Self {
        KeySwitchingTraceMLE {
            log_lwe_dim: trace.log_lwe_dim,
            log_rlwe_dim: trace.log_rlwe_dim,
            log_coeff_count: trace.log_coeff_count,
            hadamard_trace: trace.hadamard_trace.into(),
            permutation_trace: trace.permutation_trace.map(|t| t.into()),
        }
    }
}
