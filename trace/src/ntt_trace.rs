use std::rc::Rc;

use algebra::{
    AbstractExtensionField, DenseMultilinearExtension, Field, FieldUniformSampler, NTTField,
    transformation::AbstractNTT,
};
use rand_distr::Distribution;
use serde::Serialize;

use crate::{
    ConvertToEF, PackableTrace,
    rlwe_trace::{MonomialTrace, MonomialTraceMLE},
};

pub struct NTTTrace<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_ntt: usize,
    pub ntt_table: Vec<F>,
    pub coefficients: Vec<F>,
    pub evaluations: Vec<F>,
    pub is_monomial: Option<MonomialTrace<F>>,
}

/// NTT instance to be proved
pub struct NTTTraceMLE<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_ntt: usize,
    pub ntt_table: Rc<Vec<F>>,
    pub coefficients: Rc<DenseMultilinearExtension<F>>,
    pub evaluations: Rc<DenseMultilinearExtension<F>>,
    pub is_monomial: Option<MonomialTraceMLE<F>>,
}

impl<F: Field> NTTTrace<F> {
    /// Create a new empty NTT trace
    #[inline]
    pub fn new(log_coeff_count: usize, log_num_ntt: usize, ntt_table: Vec<F>) -> Self {
        Self {
            log_coeff_count,
            log_num_ntt,
            ntt_table,
            coefficients: Vec::with_capacity(1 << log_coeff_count),
            evaluations: Vec::with_capacity(1 << log_coeff_count),
            is_monomial: None,
        }
    }

    #[inline]
    pub fn get_commit_poly(&self) -> DenseMultilinearExtension<F> {
        DenseMultilinearExtension::from_evaluations_vec(
            self.log_coeff_count + self.log_num_ntt,
            self.coefficients.clone(),
        )
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for NTTTraceMLE<F> {
    type Output = NTTTraceMLE<EF>;
    fn into_ef(self) -> Self::Output {
        unimplemented!()
    }

    fn to_ef(&self) -> Self::Output {
        NTTTraceMLE {
            log_coeff_count: self.log_coeff_count,
            log_num_ntt: self.log_num_ntt,
            ntt_table: Rc::new(self.ntt_table.to_ef()),
            coefficients: Rc::new(self.coefficients.to_ef()),
            evaluations: Rc::new(self.evaluations.to_ef()),
            is_monomial: match &self.is_monomial {
                Some(mono) => Some(mono.to_ef()),
                None => None,
            },
        }
    }
}

impl<F: NTTField> NTTTrace<F> {
    /// Generate a random NTT trace
    #[inline]
    pub fn random<R: rand::Rng + rand::CryptoRng>(
        log_coeff_count: usize,
        log_num_ntt: usize,
        rng: &mut R,
    ) -> Self {
        let size = 1 << (log_coeff_count + log_num_ntt);
        let coefficients = FieldUniformSampler::new()
            .sample_iter(rng)
            .take(size)
            .collect::<Vec<F>>();

        let mut evaluations = coefficients.clone();
        let ntt_table = F::get_ntt_table(log_coeff_count as u32).unwrap();

        evaluations
            .chunks_exact_mut(1 << log_coeff_count)
            .for_each(|chunk| ntt_table.transform_slice(chunk));

        Self {
            log_coeff_count,
            log_num_ntt,
            ntt_table: F::get_ntt_table(log_coeff_count as u32)
                .unwrap()
                .root_powers(),
            coefficients,
            evaluations,
            is_monomial: None,
        }
    }
}

impl<F: Field> From<NTTTrace<F>> for NTTTraceMLE<F> {
    /// Convert NTT trace to NTT instance
    #[inline]
    fn from(trace: NTTTrace<F>) -> Self {
        let coefficients = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            trace.log_coeff_count + trace.log_num_ntt,
            trace.coefficients,
        ));
        let evaluations = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            trace.log_coeff_count + trace.log_num_ntt,
            trace.evaluations,
        ));
        let ntt_table = Rc::new(trace.ntt_table);
        Self {
            log_coeff_count: trace.log_coeff_count,
            log_num_ntt: trace.log_num_ntt,
            ntt_table,
            coefficients,
            evaluations,
            is_monomial: match trace.is_monomial {
                Some(mono) => Some(MonomialTraceMLE::from(mono)),
                None => None,
            },
        }
    }
}

impl<F: Field> PackableTrace<F> for NTTTraceMLE<F> {
    fn num_vars(&self) -> usize {
        self.log_coeff_count + self.log_num_ntt
    }

    fn num_oracles(&self) -> usize {
        1
    }

    fn pack_to_vec(&self) -> Vec<F> {
        self.coefficients.evaluations.clone()
    }
}
