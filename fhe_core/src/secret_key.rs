use std::cell::RefCell;

use algebra::{
    AsInto, Field, NTTField, NTTPolynomial, Polynomial,
    utils::{Prg, sample_binary_field_vec, sample_ternary_field_vec},
};

use crate::{Parameters, ciphertext::LWECiphertext, decode, encode, parameter::LWEParameters};

/// The distribution type of the LWE Secret Key.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum LWESecretKeyType {
    /// Binary SecretKey Distribution.
    Binary,
    /// Ternary SecretKey Distribution.
    #[default]
    Ternary,
}

/// The distribution type of the Ring Secret Key.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum RingSecretKeyType {
    /// Binary SecretKey Distribution.
    Binary,
    /// Ternary SecretKey Distribution.
    #[default]
    Ternary,
}

/// Ring Secret key
pub type RingSecretKey<F> = Polynomial<F>;

/// NTT version Ring Secret key
pub type NTTRingSecretKey<F> = NTTPolynomial<F>;

/// Boolean fhe's secret keys pack.
///
/// This struct contains the LWE secret key,
/// ring secret key, ntt version ring secret key
/// and boolean fhe's parameters.
#[derive(Clone)]
pub struct SecretKeyPack<Q: NTTField> {
    /// LWE secret key
    lwe_secret_key: Vec<Q>,

    /// ring secret key
    ring_secret_key: RingSecretKey<Q>,
    /// ntt version ring secret key
    ntt_ring_secret_key: NTTRingSecretKey<Q>,

    /// boolean fhe's parameters
    parameters: Parameters<Q>,

    /// cryptographically secure random number generator
    csrng: RefCell<Prg>,
}

impl<Q: NTTField> SecretKeyPack<Q> {
    fn create_lwe_secret_key(params: &LWEParameters<Q>, csrng: &mut Prg) -> Vec<Q> {
        match params.secret_key_type {
            LWESecretKeyType::Binary => sample_binary_field_vec(params.dimension, csrng),
            LWESecretKeyType::Ternary => sample_ternary_field_vec(params.dimension, csrng),
        }
    }

    /// Creates a new [`SecretKeyPack<Q>`].
    pub fn new(params: Parameters<Q>) -> Self {
        let mut csrng = Prg::new();

        let lwe_secret_key = Self::create_lwe_secret_key(&params.lwe_params(), &mut csrng);

        let ring_dimension = params.ring_dimension();

        let ring_secret_key = match params.ring_secret_key_type() {
            RingSecretKeyType::Binary => Polynomial::random_with_binary(ring_dimension, &mut csrng),
            RingSecretKeyType::Ternary => {
                Polynomial::random_with_ternary(ring_dimension, &mut csrng)
            }
        };
        let ntt_ring_secret_key = ring_secret_key.clone().into_ntt_polynomial();

        Self {
            lwe_secret_key,
            ring_secret_key,
            ntt_ring_secret_key,
            parameters: params,
            csrng: RefCell::new(csrng),
        }
    }

    /// Returns the lwe secret key of this [`SecretKeyPack<Q>`].
    #[inline]
    pub fn lwe_secret_key(&self) -> &[Q] {
        &self.lwe_secret_key
    }

    /// Returns the ring secret key of this [`SecretKeyPack<Q>`].
    #[inline]
    pub fn ring_secret_key(&self) -> &RingSecretKey<Q> {
        &self.ring_secret_key
    }

    /// Returns the ntt ring secret key of this [`SecretKeyPack<Q>`].
    #[inline]
    pub fn ntt_ring_secret_key(&self) -> &NTTRingSecretKey<Q> {
        &self.ntt_ring_secret_key
    }

    /// Returns the parameters of this [`SecretKeyPack<Q>`].
    #[inline]
    pub fn parameters(&self) -> &Parameters<Q> {
        &self.parameters
    }

    /// Returns the csrng of this [`SecretKeyPack<Q>`].
    #[inline]
    pub fn csrng(&self) -> std::cell::Ref<'_, Prg> {
        self.csrng.borrow()
    }

    /// Returns the csrng of this [`SecretKeyPack<Q>`].
    #[inline]
    pub fn csrng_mut(&self) -> std::cell::RefMut<'_, Prg> {
        self.csrng.borrow_mut()
    }

    /// Encrypts message into [`LWECiphertext`].
    #[inline]
    pub fn encrypt(&self, message: <Q as Field>::Value) -> LWECiphertext<Q> {
        let cipher_modulus_value = self.parameters.lwe_cipher_modulus_value();
        let noise_distribution = self.parameters.lwe_noise_distribution();
        let mut csrng = self.csrng_mut();

        let mut ciphertext = LWECiphertext::generate_random_zero_sample_field(
            self.lwe_secret_key(),
            noise_distribution,
            &mut *csrng,
        );

        let msg: <Q as Field>::Value = encode(
            message,
            self.parameters.lwe_plain_modulus(),
            cipher_modulus_value.as_into(),
        );

        ciphertext.b_mut().add_assign(Q::new(msg));

        ciphertext
    }

    /// Decrypts the [`LWECiphertext`] back to message.
    #[inline]
    pub fn decrypt(&self, cipher_text: &LWECiphertext<Q>) -> <Q as Field>::Value {
        let a_mul_s = Q::dot_product(cipher_text.a(), self.lwe_secret_key());
        let plaintext = cipher_text.b() - a_mul_s;

        decode(
            plaintext.value(),
            self.parameters.lwe_plain_modulus(),
            self.parameters.lwe_cipher_modulus_value().as_into(),
        )
    }

    /// Decrypts the [`LWECiphertext`] back to message.
    #[inline]
    pub fn decrypt_with_noise(
        &self,
        cipher_text: &LWECiphertext<Q>,
    ) -> (<Q as Field>::Value, <Q as Field>::Value) {
        let t: <Q as Field>::Value = self.parameters.lwe_plain_modulus();
        let q: <Q as Field>::Value = self.parameters.lwe_cipher_modulus_value();

        let a_mul_s = Q::dot_product(cipher_text.a(), self.lwe_secret_key());

        let plaintext = cipher_text.b() - a_mul_s;

        let p = plaintext.value();

        let message = decode(p, t, q);

        let fresh = encode(message, t, q);

        let pp = Q::new(fresh);
        (
            message,
            (plaintext - pp).value().min((pp - plaintext).value()),
        )
    }
}
