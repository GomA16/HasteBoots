pub mod ntt;
pub mod ntt_fourier;
// pub mod ntt_eval;

use algebra::{Field, ListOfProductsOfPolynomials};
pub use ntt_fourier::NTTFourierEvalIOP;
use sumcheck::Proof;

pub struct SumcheckClaim<F: Field> {
    poly: ListOfProductsOfPolynomials<F>,
    sum: F,
}

impl<F: Field> SumcheckClaim<F> {
    pub fn new(num_vars: usize) -> Self {
        Self {
            poly: ListOfProductsOfPolynomials::new(num_vars),
            sum: F::zero(),
        }
    }

    pub fn poly_mut(&mut self) -> &mut ListOfProductsOfPolynomials<F> {
        &mut self.poly
    }

    pub fn sum_mut(&mut self) -> &mut F {
        &mut self.sum
    }

    pub fn poly_ref(&self) -> &ListOfProductsOfPolynomials<F> {
        &self.poly
    }

    pub fn sum_ref(&self) -> &F {
        &self.sum
    }
}