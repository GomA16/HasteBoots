use algebra::Field;

use crate::{
    basic_ops::{
        RowPermTrace, RowPermTraceMLE, SumHadamardTrace, SumHadamardTraceMLE,
        decomp_trace::DecompTraceMLE,
        rlwe_trace::{PolynomialTrace, PolynomialTraceMLE},
    },
    cmp_trace::lt_trace::{LTTables, LTTablesMLE},
};

pub struct KeySwitchingTrace<F: Field> {
    pub log_lwe_dim: usize,
    pub log_rlwe_dim: usize,
    pub log_coeff_count: usize,
    pub log_trace: usize, // for batching
    pub decomposed_polys: Vec<PolynomialTrace<F>>,
    pub hadamard_trace: SumHadamardTrace<F>,
    pub permutation_trace: Option<RowPermTrace<F>>,
    // TODO: same table as in blind rotation
    pub lt_tables: LTTables<F>,
}

pub struct KeySwitchingTraceMLE<F: Field> {
    pub log_lwe_dim: usize,
    pub log_rlwe_dim: usize,
    pub log_coeff_count: usize,
    pub log_trace: usize, // for batching
    pub decomposed_polys: Vec<PolynomialTraceMLE<F>>,
    pub hadamard_trace: SumHadamardTraceMLE<F>,
    pub permutation_trace: Option<RowPermTraceMLE<F>>,
    pub lt_tables: LTTablesMLE<F>,
}

impl<F: Field> Clone for KeySwitchingTrace<F> {
    fn clone(&self) -> Self {
        KeySwitchingTrace {
            log_lwe_dim: self.log_lwe_dim,
            log_rlwe_dim: self.log_rlwe_dim,
            log_coeff_count: self.log_coeff_count,
            log_trace: self.log_trace,
            decomposed_polys: self.decomposed_polys.clone(),
            hadamard_trace: self.hadamard_trace.clone(),
            permutation_trace: None,
            lt_tables: self.lt_tables.clone(),
        }
    }
}

impl<F: Field> KeySwitchingTrace<F> {
    pub fn from_batch_trace(traces: Vec<KeySwitchingTrace<F>>) -> Self {
        let num_trace = traces.len();
        assert!(num_trace.is_power_of_two());
        let log_num_trace = num_trace.trailing_zeros() as usize;
        let decom_poly = PolynomialTrace::new(traces[0].log_coeff_count, log_num_trace);
        let mut decomposed_polys = vec![decom_poly; traces[0].decomposed_polys.len()];
        let mut hadamard_trace = SumHadamardTrace::new(
            traces[0].hadamard_trace.num_hadamard,
            traces[0].log_coeff_count,
            log_num_trace,
        );
        let log_lwe_dim = traces[0].log_lwe_dim;
        let log_rlwe_dim = traces[0].log_rlwe_dim;
        let log_coeff_count = traces[0].log_coeff_count;
        let lt_tables = traces[0].lt_tables.clone();
        traces.into_iter().for_each(|trace| {
            decomposed_polys
                .iter_mut()
                .zip(trace.decomposed_polys.iter())
                .for_each(|(dst, src)| {
                    dst.append_trace(src);
                });
            hadamard_trace.append_trace(&trace.hadamard_trace);
        });
        KeySwitchingTrace {
            log_lwe_dim,
            log_rlwe_dim,
            log_coeff_count,
            log_trace: log_num_trace,
            decomposed_polys,
            hadamard_trace,
            permutation_trace: None,
            lt_tables,
        }
    }
}

impl<F: Field> From<KeySwitchingTrace<F>> for KeySwitchingTraceMLE<F> {
    fn from(trace: KeySwitchingTrace<F>) -> Self {
        KeySwitchingTraceMLE {
            log_lwe_dim: trace.log_lwe_dim,
            log_rlwe_dim: trace.log_rlwe_dim,
            log_coeff_count: trace.log_coeff_count,
            log_trace: trace.log_trace,
            decomposed_polys: trace
                .decomposed_polys
                .into_iter()
                .map(|p| p.into())
                .collect(),
            hadamard_trace: trace.hadamard_trace.into(),
            permutation_trace: trace.permutation_trace.map(|t| t.into()),
            lt_tables: trace.lt_tables.into(),
        }
    }
}

impl<F: Field> KeySwitchingTraceMLE<F> {
    pub fn extract_decomposition_traces(&self) -> Vec<DecompTraceMLE<F>> {
        let lt_tables = &self.lt_tables;
        assert_eq!(
            self.hadamard_trace.vec_hadamard.len(),
            lt_tables.decomp_len * self.decomposed_polys.len()
        );

        let log_num = self.log_coeff_count + self.log_trace;
        let decomp_len = lt_tables.decomp_len;
        let extract_bits = |start_idx: usize| {
            self.hadamard_trace.vec_hadamard[start_idx..start_idx + decomp_len]
                .iter()
                .map(|hadamard| hadamard.bit.poly.clone())
                .collect::<Vec<_>>()
        };

        let bits = (0..self.decomposed_polys.len())
            .map(|i| extract_bits(i * decomp_len))
            .collect::<Vec<_>>();

        bits.into_iter()
            .zip(self.decomposed_polys.iter())
            .map(|(bit_polys, poly)| DecompTraceMLE {
                log_num,
                basis_bits: lt_tables.basis_bits,
                decomp_len,
                bits: bit_polys,
                input: poly.poly.clone(),
            })
            .collect()
    }
}
