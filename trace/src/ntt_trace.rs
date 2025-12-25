use std::rc::Rc;

use algebra::{
    AbstractExtensionField, DenseMultilinearExtension, Field, FieldUniformSampler, NTTField,
    transformation::AbstractNTT,
};
use rand_distr::Distribution;
use serde::Serialize;

use crate::{ConvertToEF, PackableTrace, rlwe_trace::MonomialTrace};

pub struct NTTTrace<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_ntt: usize,
    pub ntt_table: Vec<F>,
    pub coefficients: Vec<F>,
    pub evaluations: Vec<F>,
    pub is_monomial: Option<MonomialTrace<F>>,
}

#[derive(Serialize)]
pub struct NTTTraceInfo<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_ntt: usize,
    #[serde(skip)]
    pub ntt_table: Rc<Vec<F>>,
}

pub struct BatchedNTTTraceMLE<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_ntt: usize,
    pub ntt_table: Rc<Vec<F>>,
    pub coefficients: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub evaluations: Vec<Rc<DenseMultilinearExtension<F>>>,
}

/// NTT instance to be proved
pub struct NTTTraceMLE<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_ntt: usize,
    pub ntt_table: Rc<Vec<F>>,
    pub coefficients: Rc<DenseMultilinearExtension<F>>,
    pub evaluations: Rc<DenseMultilinearExtension<F>>,
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

    pub fn info_ef<EF: AbstractExtensionField<F>>(&self) -> NTTTraceInfo<EF> {
        NTTTraceInfo {
            log_coeff_count: self.log_coeff_count,
            log_num_ntt: self.log_num_ntt,
            ntt_table: Rc::new(self.ntt_table.iter().map(|x| EF::from_base(*x)).collect()),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for NTTTrace<F> {
    type Output = NTTTrace<EF>;
    fn into_ef(self) -> Self::Output {
        NTTTrace {
            log_coeff_count: self.log_coeff_count,
            log_num_ntt: self.log_num_ntt,
            ntt_table: self.ntt_table.into_ef(),
            coefficients: self.coefficients.into_ef(),
            evaluations: self.evaluations.into_ef(),
            is_monomial: match self.is_monomial {
                Some(mono) => Some(mono.into_ef()),
                None => None,
            },
        }
    }

    fn to_ef(&self) -> Self::Output {
        NTTTrace {
            log_coeff_count: self.log_coeff_count,
            log_num_ntt: self.log_num_ntt,
            ntt_table: self.ntt_table.to_ef(),
            coefficients: self.coefficients.to_ef(),
            evaluations: self.evaluations.to_ef(),
            is_monomial: match &self.is_monomial {
                Some(mono) => Some(mono.to_ef()),
                None => None,
            },
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for MonomialTrace<F> {
    type Output = MonomialTrace<EF>;
    fn into_ef(self) -> Self::Output {
        MonomialTrace {
            log_num_poly: self.log_num_poly,
            degree: self.degree.into_ef(),
            coefficient: self.coefficient.into_ef(),
        }
    }

    fn to_ef(&self) -> Self::Output {
        MonomialTrace {
            log_num_poly: self.log_num_poly,
            degree: self.degree.to_ef(),
            coefficient: self.coefficient.to_ef(),
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

impl<F: Field> BatchedNTTTraceMLE<F> {
    #[inline]
    pub fn to_random_trace(&self, randomness: &[F]) -> NTTTraceMLE<F> {
        let size = 1 << self.log_coeff_count;
        let mut rand_coeffs = vec![F::zero(); size];
        let mut rand_evals = vec![F::zero(); size];

        let add_assign = |acc: &mut [F], vec: &[F], r: F| {
            for (a, b) in acc.iter_mut().zip(vec.iter()) {
                *a += r.mul(*b);
            }
        };

        self.coefficients
            .iter()
            .zip(randomness.iter())
            .for_each(|(coeffs, r)| add_assign(&mut rand_coeffs, coeffs.as_slice(), *r));
        self.evaluations
            .iter()
            .zip(randomness.iter())
            .for_each(|(evals, r)| add_assign(&mut rand_evals, evals.as_slice(), *r));

        NTTTraceMLE {
            log_coeff_count: self.log_coeff_count,
            log_num_ntt: self.log_num_ntt,
            ntt_table: self.ntt_table.clone(),
            coefficients: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                self.log_coeff_count + self.log_num_ntt,
                rand_coeffs,
            )),
            evaluations: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                self.log_coeff_count + self.log_num_ntt,
                rand_evals,
            )),
        }
        .into()
    }

    #[inline]
    pub fn to_random_ef_instance<EF: AbstractExtensionField<F>>(
        &self,
        randomness: &[EF],
    ) -> NTTTraceMLE<EF> {
        let size = 1 << self.log_coeff_count;
        let mut rand_coeffs = vec![EF::zero(); size];
        let mut rand_evals = vec![EF::zero(); size];

        let add_assign = |acc: &mut [EF], vec: &[F], r: EF| {
            for (a, b) in acc.iter_mut().zip(vec.iter()) {
                *a += r.mul(*b);
            }
        };

        self.coefficients
            .iter()
            .zip(randomness.iter())
            .for_each(|(coeffs, r)| add_assign(&mut rand_coeffs, coeffs.as_slice(), *r));
        self.evaluations
            .iter()
            .zip(randomness.iter())
            .for_each(|(evals, r)| add_assign(&mut rand_evals, evals.as_slice(), *r));

        NTTTraceMLE::<EF> {
            log_coeff_count: self.log_coeff_count,
            log_num_ntt: self.log_num_ntt,
            ntt_table: Rc::new(self.ntt_table.to_ef()),
            coefficients: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                self.log_coeff_count + self.log_num_ntt,
                rand_coeffs,
            )),
            evaluations: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                self.log_coeff_count + self.log_num_ntt,
                rand_evals,
            )),
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

impl<F: Field> BatchedNTTTraceMLE<F> {
    fn num_vars(&self) -> usize {
        self.log_coeff_count + self.log_num_ntt
    }

    fn num_oracles(&self) -> usize {
        self.coefficients.len()
    }

    fn pack_to_vec(&self) -> Vec<F> {
        self.coefficients
            .iter()
            .flat_map(|mle| mle.iter())
            .cloned()
            .collect::<Vec<F>>()
    }
}
