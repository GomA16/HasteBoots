use std::rc::Rc;

use algebra::{
    AbstractExtensionField, DenseMultilinearExtension, Field, NTTField, transformation::AbstractNTT,
};
use serde::Serialize;

use crate::{ConvertToEF, EvaluableTrace, EvaluableTraceEF, PackableTrace};
#[derive(Clone)]
pub struct PolynomialTrace<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_poly: usize,
    pub poly: Vec<F>,
    pub ntt: Vec<F>,
}

#[derive(Clone)]
pub struct PolynomialTraceMLE<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_poly: usize,
    pub poly: Rc<DenseMultilinearExtension<F>>,
    pub ntt: Rc<DenseMultilinearExtension<F>>,
}

#[derive(Serialize, Clone)]
pub struct PolynomialEval<F: Field> {
    pub poly: F,
    pub ntt: F,
}

pub struct MonomialTrace<F: Field> {
    pub log_num_poly: usize,
    pub degree: Vec<F>,
    pub coefficient: Vec<F>,
}

pub struct MonomialTraceMLE<F: Field> {
    pub log_num_poly: usize,
    pub degree: Rc<DenseMultilinearExtension<F>>,
    pub coefficient: Rc<DenseMultilinearExtension<F>>,
}

#[derive(Serialize)]
pub struct MonomialEval<F: Field> {
    pub degree: F,
    pub coefficient: F,
}

#[derive(Clone)]
pub struct RLWETrace<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_poly: usize,
    pub poly: (Vec<F>, Vec<F>),
    pub ntt: (Vec<F>, Vec<F>),
}

#[derive(Clone)]
pub struct RLWETraceMLE<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_poly: usize,
    pub poly: (
        Rc<DenseMultilinearExtension<F>>,
        Rc<DenseMultilinearExtension<F>>,
    ),
    pub ntt: (
        Rc<DenseMultilinearExtension<F>>,
        Rc<DenseMultilinearExtension<F>>,
    ),
}

#[derive(Serialize, Clone)]
pub struct RLWEEval<F: Field> {
    pub poly: (F, F),
    pub ntt: (F, F),
}

impl<F: Field> MonomialTrace<F> {
    pub fn new(log_num_poly: usize) -> Self {
        Self {
            log_num_poly,
            degree: Vec::with_capacity(1 << log_num_poly),
            coefficient: Vec::with_capacity(1 << log_num_poly),
        }
    }

    pub fn append(&mut self, degree: F, coefficient: F) {
        self.degree.push(degree);
        self.coefficient.push(coefficient);
    }
}

impl<F: Field> PolynomialTrace<F> {
    #[inline]
    pub fn new(log_coeff_count: usize, log_num_poly: usize) -> Self {
        Self {
            log_coeff_count,
            log_num_poly,
            poly: Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
            ntt: Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
        }
    }

    #[inline]
    pub fn finalize(&mut self, num_poly: usize) {
        if !num_poly.is_power_of_two() {
            let num_zeros = ((1 << self.log_num_poly) - num_poly) * (1 << self.log_coeff_count);
            self.poly.extend(vec![F::zero(); num_zeros]);
            self.ntt.extend(vec![F::zero(); num_zeros]);
        }
    }
}

impl<F: NTTField> PolynomialTrace<F> {
    #[inline]
    pub fn append_poly(&mut self, poly: &[F]) {
        self.poly.extend_from_slice(poly);

        let ntt_table = F::get_ntt_table(self.log_coeff_count as u32).unwrap();

        let mut ntt_poly = poly.to_vec();
        ntt_table.transform_slice(&mut ntt_poly);

        self.ntt.extend_from_slice(&ntt_poly);
    }
}

impl<F: Field> RLWETrace<F> {
    #[inline]
    pub fn new(log_coeff_count: usize, log_num_poly: usize) -> Self {
        Self {
            log_coeff_count,
            log_num_poly,
            poly: (
                Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
                Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
            ),
            ntt: (
                Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
                Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
            ),
        }
    }

    #[inline]
    pub fn append(&mut self, rlwe: (&[F], &[F]), ntt_rlwe: (&[F], &[F])) {
        self.poly.0.extend_from_slice(rlwe.0);
        self.poly.1.extend_from_slice(rlwe.1);
        self.ntt.0.extend_from_slice(ntt_rlwe.0);
        self.ntt.1.extend_from_slice(ntt_rlwe.1);
    }

    #[inline]
    pub fn finalize(&mut self, num_poly: usize) {
        if !num_poly.is_power_of_two() {
            let num_zeros = ((1 << self.log_num_poly) - num_poly) * (1 << self.log_coeff_count);
            self.poly.0.extend(vec![F::zero(); num_zeros]);
            self.poly.1.extend(vec![F::zero(); num_zeros]);
            self.ntt.0.extend(vec![F::zero(); num_zeros]);
            self.ntt.1.extend(vec![F::zero(); num_zeros]);
        }
    }
}

impl<F: NTTField> RLWETrace<F> {
    #[inline]
    pub fn append_poly(&mut self, rlwe: (&[F], &[F])) {
        self.poly.0.extend_from_slice(rlwe.0);
        self.poly.1.extend_from_slice(rlwe.1);

        let ntt_table = F::get_ntt_table(self.log_coeff_count as u32).unwrap();

        let mut ntt_a = rlwe.0.to_vec();
        let mut ntt_b = rlwe.1.to_vec();
        ntt_table.transform_slice(&mut ntt_a);
        ntt_table.transform_slice(&mut ntt_b);

        self.ntt.0.extend_from_slice(&ntt_a);
        self.ntt.1.extend_from_slice(&ntt_b);
    }
}

impl<F: Field> From<RLWETrace<F>> for RLWETraceMLE<F> {
    #[inline]
    fn from(trace: RLWETrace<F>) -> Self {
        Self {
            log_coeff_count: trace.log_coeff_count,
            log_num_poly: trace.log_num_poly,
            poly: (
                Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                    trace.log_coeff_count + trace.log_num_poly,
                    trace.poly.0,
                )),
                Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                    trace.log_coeff_count + trace.log_num_poly,
                    trace.poly.1,
                )),
            ),
            ntt: (
                Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                    trace.log_coeff_count + trace.log_num_poly,
                    trace.ntt.0,
                )),
                Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                    trace.log_coeff_count + trace.log_num_poly,
                    trace.ntt.1,
                )),
            ),
        }
    }
}

impl<F: Field> From<PolynomialTrace<F>> for PolynomialTraceMLE<F> {
    #[inline]
    fn from(trace: PolynomialTrace<F>) -> Self {
        Self {
            log_coeff_count: trace.log_coeff_count,
            log_num_poly: trace.log_num_poly,
            poly: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                trace.log_coeff_count + trace.log_num_poly,
                trace.poly,
            )),
            ntt: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                trace.log_coeff_count + trace.log_num_poly,
                trace.ntt,
            )),
        }
    }
}

impl<F: Field> From<MonomialTrace<F>> for MonomialTraceMLE<F> {
    #[inline]
    fn from(trace: MonomialTrace<F>) -> Self {
        Self {
            log_num_poly: trace.log_num_poly,
            degree: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                trace.log_num_poly,
                trace.degree,
            )),
            coefficient: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                trace.log_num_poly,
                trace.coefficient,
            )),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for PolynomialTraceMLE<F> {
    type Output = PolynomialTraceMLE<EF>;
    fn into_ef(self) -> Self::Output {
        unimplemented!()
    }
    fn to_ef(&self) -> Self::Output {
        PolynomialTraceMLE {
            log_coeff_count: self.log_coeff_count,
            log_num_poly: self.log_num_poly,
            poly: Rc::new(self.poly.to_ef()),
            ntt: Rc::new(self.ntt.to_ef()),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for MonomialTraceMLE<F> {
    type Output = MonomialTraceMLE<EF>;
    fn into_ef(self) -> Self::Output {
        unimplemented!()
    }
    fn to_ef(&self) -> Self::Output {
        MonomialTraceMLE {
            log_num_poly: self.log_num_poly,
            degree: Rc::new(self.degree.to_ef()),
            coefficient: Rc::new(self.coefficient.to_ef()),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for RLWETraceMLE<F> {
    type Output = RLWETraceMLE<EF>;
    fn into_ef(self) -> Self::Output {
        unimplemented!()
    }
    fn to_ef(&self) -> Self::Output {
        RLWETraceMLE {
            log_coeff_count: self.log_coeff_count,
            log_num_poly: self.log_num_poly,
            poly: (Rc::new(self.poly.0.to_ef()), Rc::new(self.poly.1.to_ef())),
            ntt: (Rc::new(self.ntt.0.to_ef()), Rc::new(self.ntt.1.to_ef())),
        }
    }
}

impl<F: Field> EvaluableTrace<F> for PolynomialTraceMLE<F> {
    type TraceEval = PolynomialEval<F>;
    fn evaluate(&self, point: &[F]) -> Self::TraceEval {
        Self::TraceEval {
            poly: self.poly.evaluate(point),
            ntt: self.ntt.evaluate(point),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> EvaluableTraceEF<F, EF> for PolynomialTraceMLE<F> {
    type TraceEvalEF = PolynomialEval<EF>;
    fn evaluate_ef(&self, point: &[EF]) -> Self::TraceEvalEF {
        Self::TraceEvalEF {
            poly: self.poly.evaluate_ext(point),
            ntt: self.ntt.evaluate_ext(point),
        }
    }
}

impl<F: Field> EvaluableTrace<F> for MonomialTraceMLE<F> {
    type TraceEval = MonomialEval<F>;
    fn evaluate(&self, point: &[F]) -> Self::TraceEval {
        Self::TraceEval {
            degree: self.degree.evaluate(point),
            coefficient: self.coefficient.evaluate(point),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> EvaluableTraceEF<F, EF> for MonomialTraceMLE<F> {
    type TraceEvalEF = MonomialEval<EF>;
    fn evaluate_ef(&self, point: &[EF]) -> Self::TraceEvalEF {
        Self::TraceEvalEF {
            degree: self.degree.evaluate_ext(point),
            coefficient: self.coefficient.evaluate_ext(point),
        }
    }
}

impl<F: Field> EvaluableTrace<F> for RLWETraceMLE<F> {
    type TraceEval = RLWEEval<F>;
    fn evaluate(&self, point: &[F]) -> Self::TraceEval {
        Self::TraceEval {
            poly: (self.poly.0.evaluate(point), self.poly.1.evaluate(point)),
            ntt: (self.ntt.0.evaluate(point), self.ntt.1.evaluate(point)),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> EvaluableTraceEF<F, EF> for RLWETraceMLE<F> {
    type TraceEvalEF = RLWEEval<EF>;
    fn evaluate_ef(&self, point: &[EF]) -> Self::TraceEvalEF {
        Self::TraceEvalEF {
            poly: (
                self.poly.0.evaluate_ext(point),
                self.poly.1.evaluate_ext(point),
            ),
            ntt: (
                self.ntt.0.evaluate_ext(point),
                self.ntt.1.evaluate_ext(point),
            ),
        }
    }
}

impl<F: Field> PackableTrace<F> for RLWETraceMLE<F> {
    fn num_vars(&self) -> usize {
        self.log_coeff_count + self.log_num_poly
    }

    fn num_oracles(&self) -> usize {
        2
    }

    fn pack_to_vec(&self) -> Vec<F> {
        self.poly
            .0
            .iter()
            .chain(self.poly.1.iter())
            .cloned()
            .collect::<Vec<F>>()
    }
}
