//! implementation of encryption.

use algebra::{Field, NTTField};
use fhe_core::{LWECiphertext, SecretKeyPack};

/// Encryptor
pub struct Encryptor<Q: NTTField> {
    sk: SecretKeyPack<Q>,
}

impl<Q: NTTField> Encryptor<Q> {
    /// New a Encryptor instance.
    #[inline]
    pub fn new(sk: SecretKeyPack<Q>) -> Self {
        Self { sk }
    }

    /// Encrypt a bool message.
    #[inline]
    pub fn encrypt(&self, m: <Q as Field>::Value) -> LWECiphertext<Q> {
        self.sk.encrypt(m)
    }
}
