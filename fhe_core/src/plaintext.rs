use algebra::{AsInto, UnsignedInteger};

// pub trait Shrink {
//     /// shrink to small container.
//     fn shrink(c: u64) -> Self;
// }

// macro_rules! shrink_impl {
//     (@ bool) => {
//         impl Shrink for bool {
//             #[inline(always)]
//             fn shrink(c: u64) -> bool {
//                 match c {
//                     0 => false,
//                     1 => true,
//                     _ => panic!("shrink error!")
//                 }
//             }
//         }
//     };
//     (@ u64) => {
//         impl Shrink for u64 {
//             #[inline(always)]
//             fn shrink(c: u64) -> u64 {
//                 c
//             }
//         }
//     };
//     (@@ $($M:ty),*) => {
//         $(
//             impl Shrink for $M {
//                 #[inline(always)]
//                 fn shrink(c: u64) -> $M {
//                     if c > <$M>::MAX as u64 {
//                         panic!("shrink error!")
//                     } else {
//                         c as $M
//                     }
//                 }
//             }
//         )*
//     };
//     () => {
//         shrink_impl!(@ bool);
//         shrink_impl!(@ u64);
//         shrink_impl!(@@ u8, u16, u32);
//     }
// }

// shrink_impl!();

/// Encodes a message.
///
/// # Parameters
///
/// - `t` is message space
/// - `q` is LWE modulus value.
/// - This function needs `q` and `t` are power of 2.
///
/// # Panic
///
/// Panics if the message exceeds the message space.
#[inline]
pub fn encode<T: UnsignedInteger>(message: T, t: T, q: T) -> T {
    assert!(
        message < t,
        "message {message} is bigger than the message space"
    );
    let message: f64 = message.as_into();
    let t: f64 = t.as_into();
    let q: f64 = q.as_into();

    (message * (q / t)).round().as_into()
}

/// Decodes an encode value.
///
/// # Parameters
///
/// - `t` is message space
/// - `q` is LWE modulus value.
/// - This function needs `q` and `t` are power of 2.
///
/// # Panic
///
/// Panics if the decoded message cannot fit in `M`.
#[inline]
pub fn decode<T: UnsignedInteger>(cipher: T, t: T, q: T) -> T {
    let cipher: f64 = cipher.as_into();
    let tf: f64 = t.as_into();
    let q: f64 = q.as_into();
    let res: T = (cipher * tf / q).round().as_into();
    if res < t { res } else { res - t }
}

// /// Trait for LWE message type.
// pub trait LWEMsgType: Copy + Send + Sync + AsInto<u64> + Shrink {}

// macro_rules! plain_impl {
//     (@ $($M:ty),*) => {
//         $(
//             impl LWEMsgType for $M {}
//         )*
//     };
//     () =>{
//         plain_impl!(@ bool, u8, u16, u32, u64);
//     }
// }

// plain_impl!();

// /// Trait for LWE cipher text modulus value type.
// pub trait LWEModulusType:
//     PrimInt
//     + Send
//     + Sync
//     + Display
//     + ConstOne
//     + ConstZero
//     + Bits
//     + Shl<u32, Output = Self>
//     + Shr<u32, Output = Self>
//     + AsFrom<u32>
//     + AsFrom<u64>
//     + AsFrom<f64>
//     + AsInto<f64>
//     + AsInto<usize>
//     + AsInto<u64>
//     + TryFrom<u64>
//     + TryInto<usize>
//     + SampleUniform
//     + AddReduce<PowOf2Modulus<Self>, Output = Self>
//     + SubReduce<PowOf2Modulus<Self>, Output = Self>
//     + MulReduce<PowOf2Modulus<Self>, Output = Self>
//     + AddReduceAssign<PowOf2Modulus<Self>>
//     + MulReduceAssign<PowOf2Modulus<Self>>
//     + NegReduce<PowOf2Modulus<Self>, Output = Self>
//     + NegReduceAssign<PowOf2Modulus<Self>>
//     + DotProductReduce<PowOf2Modulus<Self>, Output = Self>
// {
//     /// 2
//     const TWO: Self;
//     /// Generate the corresponding power of 2 modulus.
//     fn to_power_of_2_modulus(self) -> PowOf2Modulus<Self>;
// }

// macro_rules! cipher_impl {
//     ($($T:ty),*) => {
//         $(
//             impl LWEModulusType for $T {
//                 const TWO: Self = 2;
//                 #[inline]
//                 fn to_power_of_2_modulus(self) -> PowOf2Modulus<Self> {
//                     PowOf2Modulus::<$T>::new(self)
//                 }
//             }
//         )*
//     };
// }

// cipher_impl!(u8, u16, u32, u64);
