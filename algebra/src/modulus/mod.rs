//! This module implements some functions and methods for
//! modular arithmetic.

mod baby_bear;
mod barrett;
mod goldilocks;
mod powof2;
mod shoup;

pub use baby_bear::{
    BabyBearModulus, MONTY_NEG_ONE, MONTY_ONE, MONTY_TWO, MONTY_ZERO, P as BABY_BEAR_P, from_monty,
    to_monty,
};
pub use barrett::BarrettModulus;
pub use goldilocks::{GoldilocksModulus, P as GOLDILOCKS_P, to_canonical_u64};
pub use powof2::PowOf2Modulus;
pub use shoup::ShoupFactor;
