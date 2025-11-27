use crate::{
    Field, Polynomial, NTTPolynomial,
};

pub enum PolyRep<F: Field> {
    Coeff(Polynomial<F>),
    NTT(NTTPolynomial<F>),
}

impl<F: Field> PolyRep<F> {
    pub fn as_slice(&self) -> &[F] {
        match self {
            PolyRep::Coeff(p) => p.as_slice(),
            PolyRep::NTT(p) => p.as_slice(),
        }
    }
}

/// Stores a multilinear extension in dense evaluation matrix form,
/// where each row stores a polynomial in either coefficient or NTT form.
#[derive(Default)]
pub struct DensePolyMultilinearExtension<F: Field> {
    pub evaluations: Vec<PolyRep<F>>,
    pub log_num_polys: u32,
    pub log_poly_degree: u32,
    pub num_vars: u32,
}

impl<F:Field> DensePolyMultilinearExtension<F> {
    #[inline]
    pub fn new_poly(
        evaluations: Vec<PolyRep<F>>,
        log_num_polys: u32,
        log_poly_degree: u32,
        num_vars: u32,
    ) -> Self {
        Self {
            evaluations,
            log_num_polys,
            log_poly_degree,
            num_vars,
        }
    }
}