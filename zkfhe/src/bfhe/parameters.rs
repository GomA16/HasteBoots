use algebra::{BabyBear, Field};
use fhe_core::{ConstParameters, DefaultFieldU32, LWESecretKeyType, Parameters, RingSecretKeyType};
use once_cell::sync::Lazy;

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
pub static CUSTOM_TERNARY_128_BITS_PARAMETERS: Lazy<Parameters<DefaultFieldU32>> =
    Lazy::new(|| {
        Parameters::<DefaultFieldU32>::new(ConstParameters {
            lwe_dimension: 512,
            lwe_plain_modulus: 4,
            lwe_noise_standard_deviation: 3.20,
            lwe_secret_key_type: LWESecretKeyType::Binary,
            ring_dimension: 1024,
            ring_modulus: DefaultFieldU32::MODULUS_VALUE,
            ring_noise_standard_deviation: 3.20 * ((1 << 1) as f64),
            ring_secret_key_type: RingSecretKeyType::Binary,
            blind_rotation_basis_bits: 7,
            key_switching_basis_bits: 7,
            key_switching_standard_deviation: 3.2 * ((1 << 1) as f64),
        })
        .unwrap()
    });

/// Default 128-bits security Parameters
pub static BABYBEAR_BINARY_128_BITS_PARAMETERS: Lazy<Parameters<BabyBear>> = Lazy::new(|| {
    Parameters::<BabyBear>::new(ConstParameters {
        lwe_dimension: 512,
        lwe_plain_modulus: 4,
        lwe_noise_standard_deviation: 3.20,
        lwe_secret_key_type: LWESecretKeyType::Binary,
        ring_dimension: 1024,
        ring_modulus: BabyBear::MODULUS_VALUE,
        ring_noise_standard_deviation: 3.20 * ((1 << 1) as f64),
        ring_secret_key_type: RingSecretKeyType::Binary,
        blind_rotation_basis_bits: 7,
        key_switching_basis_bits: 7,
        key_switching_standard_deviation: 3.2 * ((1 << 1) as f64),
    })
    .unwrap()
});
