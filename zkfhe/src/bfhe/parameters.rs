use algebra::{BabyBear, Field, Goldilocks};
use fhe_core::{ConstParameters, DefaultFieldU32, LWESecretKeyType, Parameters, RingSecretKeyType};
use once_cell::sync::Lazy;
use pcs::utils::code::ExpanderCodeSpec;

/// Default 128-bits security Parameters
pub static DEFAULT_TERNARY_128_BITS_PARAMETERS: Lazy<Parameters<DefaultFieldU32>> =
    Lazy::new(|| {
        Parameters::<DefaultFieldU32>::new(ConstParameters {
            lwe_dimension: 1024,
            lwe_plain_modulus: 4,
            lwe_noise_standard_deviation: 3.20,
            lwe_secret_key_type: LWESecretKeyType::Binary,
            ring_dimension: 1024,
            ring_modulus: DefaultFieldU32::MODULUS_VALUE,
            ring_noise_standard_deviation: 3.20 * ((1 << 1) as f64),
            ring_secret_key_type: RingSecretKeyType::Binary,
            blind_rotation_basis_bits: 3,
            key_switching_basis_bits: 1,
            key_switching_standard_deviation: 3.2 * ((1 << 1) as f64),
        })
        .unwrap()
    });

/// Default 128-bits security Parameters
pub static ZAMA_GOLDILOCKS_PARAMETERS: Lazy<Parameters<Goldilocks>> = Lazy::new(|| {
    Parameters::<Goldilocks>::new(ConstParameters {
        lwe_dimension: 728,
        lwe_plain_modulus: 4,
        lwe_noise_standard_deviation: 2.9 * ((1u64 << 45) as f64),
        lwe_secret_key_type: LWESecretKeyType::Binary,
        ring_dimension: 1024,
        ring_modulus: Goldilocks::MODULUS_VALUE,
        ring_noise_standard_deviation: 2.3 * ((1u64 << 37) as f64),
        ring_secret_key_type: RingSecretKeyType::Ternary,
        blind_rotation_basis_bits: 5,
        key_switching_basis_bits: 5,
        key_switching_standard_deviation: 2.9 * ((1u64 << 45) as f64),
    })
    .unwrap()
});

/// Default 128-bits security for Goldilocks Parameters
pub static GOLDILOCKS_BINARY_128_BITS_PARAMETERS: Lazy<Parameters<Goldilocks>> = Lazy::new(|| {
    Parameters::<Goldilocks>::new(ConstParameters {
        lwe_dimension: 728,
        lwe_plain_modulus: 4,
        lwe_noise_standard_deviation: 2.9 * ((1u64 << 45) as f64),
        lwe_secret_key_type: LWESecretKeyType::Binary,
        ring_dimension: 1024,
        ring_modulus: Goldilocks::MODULUS_VALUE,
        ring_noise_standard_deviation: 2.3 * ((1u64 << 37) as f64),
        ring_secret_key_type: RingSecretKeyType::Ternary,
        blind_rotation_basis_bits: 8,
        key_switching_basis_bits: 5,
        key_switching_standard_deviation: 2.9 * ((1u64 << 45) as f64),
    })
    .unwrap()
});

/// Default 128-bits security for BabyBear Parameters
pub static BABYBEAR_BINARY_128_BITS_PARAMETERS: Lazy<Parameters<BabyBear>> = Lazy::new(|| {
    Parameters::<BabyBear>::new(ConstParameters {
        lwe_dimension: 728,
        lwe_plain_modulus: 4,
        lwe_noise_standard_deviation: 11000.0,
        lwe_secret_key_type: LWESecretKeyType::Binary,
        ring_dimension: 1024,
        ring_modulus: BabyBear::MODULUS_VALUE,
        ring_noise_standard_deviation: 41.9,
        ring_secret_key_type: RingSecretKeyType::Ternary,
        blind_rotation_basis_bits: 8,
        key_switching_basis_bits: 5,
        key_switching_standard_deviation: 11000.0,
    })
    .unwrap()
});