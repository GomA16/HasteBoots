use algebra::{DenseMultilinearExtension, Field, ListOfProductsOfPolynomials};
use std::rc::Rc;
use trace::{NTTTraceMLE};

pub struct NTTPolyEvalInstance<F: Field> {
    pub num_vars: usize,
    pub coefficients: Rc<DenseMultilinearExtension<F>>,
    pub ntt_table: Rc<Vec<F>>,
    pub point_u: Vec<F>,
    pub evaluations_at_u: F,
}


impl<F: Field> NTTPolyEvalInstance<F> {
    pub fn from(trace: &NTTTraceMLE<F>, point_u: Vec<F>) -> Self {
        NTTPolyEvalInstance {
            num_vars: trace.num_vars(),
            coefficients: Rc::clone(&trace.coefficients),
            ntt_table: Rc::clone(&trace.ntt_table),
            point_u: point_u.clone(),
            evaluations_at_u: trace.evaluations.evaluate(&point_u),
        }
    }

    pub fn update_sumcheck_poly(&self, poly: &mut ListOfProductsOfPolynomials<F>) {

    }
}