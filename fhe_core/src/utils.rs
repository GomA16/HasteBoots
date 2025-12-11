//! utility

use algebra::UnsignedInteger;

/// NOT
#[inline]
pub fn not<T: UnsignedInteger>(a: T) -> T {
    if a.is_zero() { T::ONE } else { T::ZERO }
}

/// AND
#[inline]
pub fn and<T: UnsignedInteger>(a: T, b: T) -> T {
    a & b
}

/// NAND
#[inline]
pub fn nand<T: UnsignedInteger>(a: T, b: T) -> T {
    not(and(a, b))
}

/// OR
#[inline]
pub fn or<T: UnsignedInteger>(a: T, b: T) -> T {
    a | b
}

/// NOR
#[inline]
pub fn nor<T: UnsignedInteger>(a: T, b: T) -> T {
    not(or(a, b))
}

/// XOR
#[inline]
pub fn xor<T: UnsignedInteger>(a: T, b: T) -> T {
    a ^ b
}

/// XNOR
#[inline]
pub fn xnor<T: UnsignedInteger>(a: T, b: T) -> T {
    not(xor(a, b))
}

/// MAJ
#[inline]
pub fn majority<T: UnsignedInteger>(a: T, b: T, c: T) -> T {
    (a & b) | (b & c) | (a & c)
}
