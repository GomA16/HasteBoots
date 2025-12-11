use std::{
    fmt::{Debug, Display},
    ops::{
        BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, ShlAssign, Shr,
        ShrAssign,
    },
};

use num_traits::{ConstOne, ConstZero, FromBytes, MulAdd, MulAddAssign, NumAssign, Pow, ToBytes};
use rand_distr::uniform::SampleUniform;

/// A trait for big number calculation
pub trait Widening: Sized {
    /// A wider type for multiplication
    type WideT;

    /// Calculates `self` + `rhs` + `carry` and checks for overflow.
    ///
    /// Performs “ternary addition” of two integer operands and a carry-in bit,
    /// and returns a tuple of the sum along with a boolean indicating
    /// whether an arithmetic overflow would occur. On overflow, the wrapped value is returned.
    ///
    /// This allows chaining together multiple additions to create a wider addition,
    /// and can be useful for bignum addition.
    /// This method should only be used for the most significant word.
    ///
    /// The output boolean returned by this method is not a carry flag,
    /// and should not be added to a more significant word.
    ///
    /// If the input carry is false, this method is equivalent to `overflowing_add`.
    fn carry_add(self, rhs: Self, carry: bool) -> (Self, bool);

    /// Calculates `self` - `rhs` - `borrow` and returns a tuple containing
    /// the difference and the output borrow.
    ///
    /// Performs "ternary subtraction" by subtracting both an integer operand and a borrow-in bit from self,
    /// and returns an output integer and a borrow-out bit. This allows chaining together multiple subtractions
    /// to create a wider subtraction, and can be useful for bignum subtraction.
    fn borrow_sub(self, rhs: Self, borrow: bool) -> (Self, bool);

    /// Calculates the complete product `self` * `rhs` without the possibility to overflow.
    ///
    /// This returns the low-order (wrapping) bits and the high-order (overflow) bits
    /// of the result as two separate values, in that order.
    fn widen_mul(self, rhs: Self) -> (Self, Self);

    /// Calculates the "full multiplication" `self` * `rhs` + `carry` without
    /// the possibility to overflow.
    ///
    /// This returns the low-order (wrapping) bits and the high-order (overflow) bits
    /// of the result as two separate values, in that order.
    ///
    /// Performs "long multiplication" which takes in an extra amount to add, and may return
    /// an additional amount of overflow. This allows for chaining together multiple multiplications
    /// to create "big integers" which represent larger values.
    fn carry_mul(self, rhs: Self, carry: Self) -> (Self, Self);
}

macro_rules! uint_widening_impl {
    ($SelfT:ty, $WideT:ty) => {
        impl Widening for $SelfT {
            type WideT = $WideT;

            #[inline]
            fn carry_add(self, rhs: Self, carry: bool) -> (Self, bool) {
                let (a, b) = self.overflowing_add(rhs);
                let (c, d) = a.overflowing_add(carry as Self);
                (c, b || d)
            }

            #[inline]
            fn borrow_sub(self, rhs: Self, borrow: bool) -> (Self, bool) {
                let (a, b) = self.overflowing_sub(rhs);
                let (c, d) = a.overflowing_sub(borrow as Self);
                (c, b || d)
            }

            #[inline]
            fn widen_mul(self, rhs: Self) -> (Self, Self) {
                let wide = (self as Self::WideT) * (rhs as Self::WideT);
                (wide as Self, (wide >> Self::BITS) as Self)
            }

            #[inline]
            fn carry_mul(self, rhs: Self, carry: Self) -> (Self, Self) {
                let wide = (self as Self::WideT) * (rhs as Self::WideT) + (carry as Self::WideT);
                (wide as Self, (wide >> Self::BITS) as Self)
            }
        }
    };
}

uint_widening_impl! { u8, u16 }
uint_widening_impl! { u16, u32 }
uint_widening_impl! { u32, u64 }
uint_widening_impl! { u64, u128 }

/// Extension trait to provide access to bits of integers.
pub trait Bits {
    /// The number of bits this type has.
    const BITS: u32;

    /// Returns the number of ones in the binary representation of `self`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let n = 0b01001100u8;
    ///
    /// assert_eq!(n.count_ones(), 3);
    /// ```
    fn count_ones(self) -> u32;

    /// Returns the number of zeros in the binary representation of `self`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let n = 0b01001100u8;
    ///
    /// assert_eq!(n.count_zeros(), 5);
    /// ```
    fn count_zeros(self) -> u32;

    /// Returns the number of leading zeros in the binary representation
    /// of `self`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let n = 0b0101000u16;
    ///
    /// assert_eq!(n.leading_zeros(), 10);
    /// ```
    fn leading_zeros(self) -> u32;

    /// Returns the number of leading ones in the binary representation
    /// of `self`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let n = 0xF00Du16;
    ///
    /// assert_eq!(n.leading_ones(), 4);
    /// ```
    fn leading_ones(self) -> u32;

    /// Returns the number of trailing zeros in the binary representation
    /// of `self`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let n = 0b0101000u16;
    ///
    /// assert_eq!(n.trailing_zeros(), 3);
    /// ```
    fn trailing_zeros(self) -> u32;

    /// Returns the number of trailing ones in the binary representation
    /// of `self`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let n = 0xBEEFu16;
    ///
    /// assert_eq!(n.trailing_ones(), 4);
    /// ```
    fn trailing_ones(self) -> u32;
}

macro_rules! impl_bits {
    ($($T:ty),*) => {
        $(
            impl Bits for $T {
                const BITS: u32 = <$T>::BITS;

                #[inline]
                fn count_ones(self) -> u32 {
                    <$T>::count_ones(self)
                }

                #[inline]
                fn count_zeros(self) -> u32 {
                    <$T>::count_zeros(self)
                }

                #[inline]
                fn leading_zeros(self) -> u32 {
                    <$T>::leading_zeros(self)
                }

                #[inline]
                fn leading_ones(self) -> u32 {
                    <$T>::leading_ones(self)
                }


                #[inline]
                fn trailing_zeros(self) -> u32 {
                    <$T>::trailing_zeros(self)
                }

                #[inline]
                fn trailing_ones(self) -> u32 {
                    <$T>::trailing_ones(self)
                }
            }
        )*
    };
}

impl_bits! {i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize}

#[doc = " Calculates the quotient of `self` and `rhs`, rounding the result towards positive infinity."]
#[doc = ""]
#[doc = " # Panics"]
#[doc = ""]
#[doc = " This function will panic if `rhs` is zero."]
#[doc = ""]
#[doc = " ## Overflow behavior"]
#[doc = ""]
#[doc = " On overflow, this function will panic if overflow checks are enabled (default in debug"]
#[doc = " mode) and wrap if overflow checks are disabled (default in release mode)."]
pub const fn div_ceil(lhs: u32, rhs: u32) -> u32 {
    let d = lhs / rhs;
    let r = lhs % rhs;
    if r > 0 { d + 1 } else { d }
}

/// A trait for big number calculation
pub trait WrappingOps: Sized + Copy {
    /// Wrapping addition. Computes `self + rhs`, wrapping around at the boundary of the type.
    fn wrapping_add(self, rhs: Self) -> Self;

    /// Wrapping subtraction. Computes `self - rhs`, wrapping around at the boundary of the type.
    fn wrapping_sub(self, rhs: Self) -> Self;

    /// Wrapping negation. Computes `-self`, wrapping around at the boundary of the type.
    ///
    /// Since unsigned types do not have negative equivalents
    /// all applications of this function will wrap (except for `-0`).
    /// For values smaller than the corresponding signed type's maximum the result
    /// is the same as casting the corresponding signed value.
    /// Any larger values are equivalent to `MAX + 1 - (val - MAX - 1)` where `MAX` is the corresponding signed type's maximum.
    fn wrapping_neg(self) -> Self;

    /// Wrapping multiplication. Computes `self * rhs`, wrapping around at the boundary of the type.
    fn wrapping_mul(self, rhs: Self) -> Self;
}

macro_rules! wrapping_impl {
    ($($SelfT:ty),*) => {$(
        impl WrappingOps for $SelfT {
            #[inline]
            fn wrapping_add(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline]
            fn wrapping_sub(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline]
            fn wrapping_neg(self) -> Self {
                self.wrapping_neg()
            }

            #[inline]
            fn wrapping_mul(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }
        })*
    };
}

wrapping_impl!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

/// A trait to convert from type `T` by `as`.
pub trait AsFrom<T: Copy>: Copy {
    /// Convert `value` from type `T` into `Self` by `as`.
    fn as_from(value: T) -> Self;
}

/// A trait to convert `self` into type `T` by `as`.
pub trait AsInto<T: Copy>: Copy {
    /// Convert `self` from type `Self` into `T` by `as`.
    fn as_into(self) -> T;
}

impl<T: Copy, U: Copy> AsInto<T> for U
where
    T: AsFrom<U>,
{
    #[inline]
    fn as_into(self) -> T {
        T::as_from(self)
    }
}

impl<T: Copy> AsFrom<T> for T {
    #[inline(always)]
    fn as_from(value: T) -> Self {
        value
    }
}

macro_rules! impl_as_from {
    (@ $T: ty => $(#[$cfg:meta])* impl $U: ty ) => {
        $(#[$cfg])*
        impl AsFrom<$T> for $U {
            #[inline] fn as_from(value: $T) -> $U { value as $U }
        }
    };
    ($T: ty => { $( $U: ty ),* } ) => {$(
        impl_as_from!(@ $T => impl $U);
    )*};
}

impl_as_from!(u8 => { char, f32, f64, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize });
impl_as_from!(i8 => { f32, f64, u8, u16, u32, u64, u128, usize, i16, i32, i64, i128, isize });
impl_as_from!(u16 => { f32, f64, u8, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize });
impl_as_from!(i16 => { f32, f64, u8, u16, u32, u64, u128, usize, i8, i32, i64, i128, isize });
impl_as_from!(u32 => { f32, f64, u8, u16, u64, u128, usize, i8, i16, i32, i64, i128, isize });
impl_as_from!(i32 => { f32, f64, u8, u16, u32, u64, u128, usize, i8, i16, i64, i128, isize });
impl_as_from!(u64 => { f32, f64, u8, u16, u32, u128, usize, i8, i16, i32, i64, i128, isize });
impl_as_from!(i64 => { f32, f64, u8, u16, u32, u64, u128, usize, i8, i16, i32, i128, isize });
impl_as_from!(u128 => { f32, f64, u8, u16, u32, u64, usize, i8, i16, i32, i64, i128, isize });
impl_as_from!(i128 => { f32, f64, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, isize });
impl_as_from!(usize => { f32, f64, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, isize });
impl_as_from!(isize => { f32, f64, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128 });
impl_as_from!(f32 => { f64, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize });
impl_as_from!(f64 => { f32, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize });
impl_as_from!(char => { u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize });
impl_as_from!(bool => { u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize });

/// A helper trait defines all `as` cast between all primitive integer types.
pub trait AsCast:
    AsFrom<i8>
    + AsFrom<u8>
    + AsFrom<i16>
    + AsFrom<u16>
    + AsFrom<i32>
    + AsFrom<u32>
    + AsFrom<i64>
    + AsFrom<u64>
    + AsFrom<i128>
    + AsFrom<u128>
    + AsFrom<isize>
    + AsFrom<usize>
    + AsFrom<f32>
    + AsFrom<f64>
    + AsInto<i8>
    + AsInto<u8>
    + AsInto<i16>
    + AsInto<u16>
    + AsInto<i32>
    + AsInto<u32>
    + AsInto<i64>
    + AsInto<u64>
    + AsInto<i128>
    + AsInto<u128>
    + AsInto<isize>
    + AsInto<usize>
    + AsInto<f32>
    + AsInto<f64>
{
}

macro_rules! impl_as_cast {
    ($($T: ty),*) => {$(
        impl AsCast for $T {}
    )*};
}

impl_as_cast! {u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize}

/// Defines an associated constant representing `2` for `Self`.
pub trait ConstTwo {
    /// `2`
    const TWO: Self;
}

macro_rules! impl_two {
    ($($T:ty),*) => {
        $(
            impl ConstTwo for $T {
                const TWO: Self = 2;
            }
        )*
    };
}

impl_two! {i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize}

/// Numbers which have upper and lower bounds
pub trait ConstBounded {
    /// The smallest finite number this type can represent
    const MIN: Self;
    /// The largest finite number this type can represent
    const MAX: Self;
}

macro_rules! impl_bounded {
    ($($T:ty),*) => {
        $(
            impl ConstBounded for $T {
                const MIN: Self = <$T>::MIN;
                const MAX: Self = <$T>::MAX;
            }
        )*
    };
}

impl_bounded! {i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize}

/// An abstraction over integer types
pub trait Integer:
    Sized
    + Send
    + Sync
    + Clone
    + Copy
    + Default
    + PartialOrd
    + Ord
    + PartialEq
    + Eq
    + Debug
    + Display
    + Bits
    + ToBytes
    + FromBytes
    + ConstZero
    + ConstOne
    + ConstTwo
    + ConstBounded
    + AsCast
    + AsFrom<bool>
    + NumAssign
    + WrappingOps
    // + WrappingAdd
    // + WrappingSub
    // + WrappingNeg
    // + WrappingMul
    // + WrappingShl
    // + WrappingShr
    // + OverflowingAdd
    // + OverflowingSub
    // + OverflowingMul
    // + CheckedAdd
    // + CheckedSub
    // + CheckedMul
    // + CheckedDiv
    // + CheckedNeg
    // + CheckedRem
    // + CheckedShl
    // + CheckedShr
    + MulAdd
    + MulAddAssign
    + Not<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + BitAndAssign
    + BitOrAssign
    + BitXorAssign
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
    + Shl<u32, Output = Self>
    + Shr<u32, Output = Self>
    + ShlAssign<u32>
    + ShrAssign<u32>
    + Pow<u32, Output = Self>
    + Pow<usize, Output = Self>
    + SampleUniform<Sampler: Copy>
{
}

macro_rules! empty_trait_impl {
    ($name:ident for $($t:ty)*) => ($(
        impl $name for $t {}
    )*)
}

empty_trait_impl!(Integer for u8 u16 u32 u64 u128 i8 i16 i32 i64 i128);

/// An abstract over unsigned integer type.
pub trait UnsignedInteger:
    Integer + num_traits::Unsigned + Widening + TryFrom<usize> + TryInto<usize>
{
    /// signed type
    type SignedInteger: Integer;

    /// Returns `true` if and only if `self == 2^k` for some `k`.
    #[must_use]
    #[inline(always)]
    fn is_power_of_two(self) -> bool {
        self.count_ones() == 1
    }

    /// cast from signed type
    fn cast_from_signed(value: Self::SignedInteger) -> Self;

    /// Wrapping (modular) addition with a signed integer. Computes `self + rhs`, wrapping around at the boundary of the type.
    fn wrapping_add_signed(self, rhs: Self::SignedInteger) -> Self;
}

macro_rules! impl_unsigned_integer {
    ($t:ty, $i:ty) => {
        impl UnsignedInteger for $t {
            type SignedInteger = $i;

            #[inline]
            fn cast_from_signed(value: Self::SignedInteger) -> Self {
                value as $t
            }

            #[inline(always)]
            fn wrapping_add_signed(self, rhs: Self::SignedInteger) -> Self {
                <$t>::wrapping_add_signed(self, rhs)
            }
        }
    };
}

impl_unsigned_integer! {u8, i8}
impl_unsigned_integer! {u16, i16}
impl_unsigned_integer! {u32, i32}
impl_unsigned_integer! {u64, i64}
