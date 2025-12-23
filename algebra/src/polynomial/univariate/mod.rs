mod native_polynomial;
mod ntt_polynomial;

pub use native_polynomial::{Polynomial, ntt_mul_assign_fast};
pub use ntt_polynomial::{
    NTTPolynomial, ntt_add_mul_assign, ntt_add_mul_assign_fast, ntt_add_mul_inplace,
};
