use std::rc::Rc;

use algebra::{
    AbstractExtensionField, DenseMultilinearExtension, Field, FieldUniformSampler, NTTField,
    transformation::AbstractNTT,
};
use rand_distr::Distribution;
use serde::Serialize;

use crate::{ConvertToEF, FieldTrace};

pub struct NTTTrace<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_ntt: usize,
    pub ntt_table: Vec<F>,
    pub coefficients: Vec<F>,
    pub evaluations: Vec<F>,
}

#[derive(Serialize)]
pub struct NTTTraceInfo<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_ntt: usize,
    #[serde(skip)]
    pub ntt_table: Rc<Vec<F>>,
}

pub struct NTTBatchedTrace<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_ntt: usize,
    pub ntt_table: Vec<F>,
    pub coefficients: Vec<Vec<F>>,
    pub evaluations: Vec<Vec<F>>,
}

/// NTT instance to be proved
pub struct NTTTraceMLE<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_ntt: usize,
    pub ntt_table: Rc<Vec<F>>,
    pub coefficients: Rc<DenseMultilinearExtension<F>>,
    pub evaluations: Rc<DenseMultilinearExtension<F>>,
}

#[derive(Clone)]
pub struct NTTInstanceInfo<F: Field> {
    pub log_coeff_count: usize,
    pub ntt_table: Rc<Vec<F>>,
    pub num_instances: usize,
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
        }
    }

    fn to_ef(&self) -> Self::Output {
        NTTTrace {
            log_coeff_count: self.log_coeff_count,
            log_num_ntt: self.log_num_ntt,
            ntt_table: self.ntt_table.to_ef(),
            coefficients: self.coefficients.to_ef(),
            evaluations: self.evaluations.to_ef(),
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

        // F::get_ntt_table(log_coeff_count as u32)
        //     .unwrap()
        //     .transform_slice(&mut evaluations);
        Self {
            log_coeff_count,
            log_num_ntt,
            ntt_table: F::get_ntt_table(log_coeff_count as u32)
                .unwrap()
                .root_powers(),
            coefficients,
            evaluations,
        }
    }
}

impl<F: NTTField> NTTBatchedTrace<F> {
    /// Generate random batched NTT trace
    #[inline]
    pub fn random<R: rand::Rng + rand::CryptoRng>(
        log_coeff_count: usize,
        num_instances: usize,
        rng: &mut R,
    ) -> Self {
        todo!("need to fix batched ntt trace generation");
        let size = 1 << log_coeff_count;
        let mut coefficients = Vec::with_capacity(num_instances);
        let mut evaluations = Vec::with_capacity(num_instances);

        let uniform = FieldUniformSampler::new();

        let mut sample_coeffs = || {
            uniform
                .sample_iter(&mut *rng)
                .take(size)
                .collect::<Vec<F>>()
        };

        for _ in 0..num_instances {
            let coeffs: Vec<F> = sample_coeffs();
            let mut evals = coeffs.clone();
            F::get_ntt_table(log_coeff_count as u32)
                .unwrap()
                .transform_slice(&mut evals);
            coefficients.push(coeffs);
            evaluations.push(evals);
        }
        Self {
            log_coeff_count,
            log_num_ntt: num_instances.trailing_zeros() as usize,
            ntt_table: F::get_ntt_table(log_coeff_count as u32)
                .unwrap()
                .root_powers(),
            coefficients,
            evaluations,
        }
    }
}

impl<F: Field> NTTBatchedTrace<F> {
    #[inline]
    pub fn to_random_instance(&self, randomness: &[F]) -> NTTTraceMLE<F> {
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
            .for_each(|(coeffs, r)| add_assign(&mut rand_coeffs, coeffs, *r));
        self.evaluations
            .iter()
            .zip(randomness.iter())
            .for_each(|(evals, r)| add_assign(&mut rand_evals, evals, *r));

        NTTTrace {
            log_coeff_count: self.log_coeff_count,
            log_num_ntt: self.log_num_ntt,
            ntt_table: self.ntt_table.clone(),
            coefficients: rand_coeffs,
            evaluations: rand_evals,
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
            .for_each(|(coeffs, r)| add_assign(&mut rand_coeffs, coeffs, *r));
        self.evaluations
            .iter()
            .zip(randomness.iter())
            .for_each(|(evals, r)| add_assign(&mut rand_evals, evals, *r));

        NTTTrace::<EF> {
            log_coeff_count: self.log_coeff_count,
            log_num_ntt: self.log_num_ntt,
            ntt_table: self.ntt_table.to_ef(),
            coefficients: rand_coeffs,
            evaluations: rand_evals,
        }
        .into()
    }
}

impl<F: Serialize + Field> Serialize for NTTInstanceInfo<F> {
    /// Serialize only the necessary fields
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (self.log_coeff_count, self.num_instances).serialize(serializer)
    }
}

impl<F: Field> NTTInstanceInfo<F> {
    /// Get number of variables
    #[inline]
    pub fn num_vars(&self) -> usize {
        self.log_coeff_count
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

impl<F: Field> NTTTraceMLE<F> {
    // /// Get NTT instance info
    // #[inline]
    // pub fn info(&self) -> NTTInstanceInfo<F> {
    //     NTTInstanceInfo {
    //         log_coeff_count: self.log_coeff_count,
    //         ntt_table: Rc::clone(&self.ntt_table),
    //         num_instances: 1,
    //     }
    // }

    pub fn num_vars(&self) -> usize {
        assert_eq!(self.coefficients.num_vars(), self.evaluations.num_vars());
        self.coefficients.num_vars()
    }

    pub fn coefficients(&self) -> Rc<DenseMultilinearExtension<F>> {
        Rc::clone(&self.coefficients)
    }

    pub fn evaluations(&self) -> Rc<DenseMultilinearExtension<F>> {
        Rc::clone(&self.evaluations)
    }
}
