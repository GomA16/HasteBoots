//! implementation of decryption.

use algebra::{Field, NTTField};
use fhe_core::{LWECiphertext, SecretKeyPack};

/// Encryptor
pub struct Decryptor<Q: NTTField> {
    sk: SecretKeyPack<Q>,
}

impl<Q: NTTField> Decryptor<Q> {
    /// Create a Decryptor instance.
    #[inline]
    pub fn new(sk: SecretKeyPack<Q>) -> Self {
        Self { sk }
    }

    /// Decrypt a ciphertext into a message.
    #[inline]
    pub fn decrypt(&self, c: &LWECiphertext<Q>) -> <Q as Field>::Value {
        self.sk.decrypt(c)
    }

    /// Decrypt a ciphertext into a bool message and an error.
    #[inline]
    pub fn decrypt_with_noise(
        &self,
        c: &LWECiphertext<Q>,
    ) -> (<Q as Field>::Value, <Q as Field>::Value) {
        self.sk.decrypt_with_noise(c)
    }
}
