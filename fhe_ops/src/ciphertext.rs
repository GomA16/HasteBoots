use std::ops::{Add, Sub};

use algebra::{Field, NTTField};

/// LWE ciphertext structure.
/// This structure represents a ciphertext in the LWE (Learning With Errors) scheme,
/// which is a foundational primitive in fully homomorphic encryption (FHE).
/// It contains a vector of field elements `a` and a single field element `b`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LWECiphertext<F: Field> {
    a: Vec<F>,
    b: F,
}

/// RLWE ciphertext structure.
/// 
pub struct RLWECiphertext<F: NTTField> {
    a: Vec<F>,
    b: Vec<F>,
}

pub struct NTTRLWECiphertext<F: NTTField> {
    a: Vec<F>,
    b: Vec<F>,
}



impl<F: Field> LWECiphertext<F> {
    /// Creates a new LWE ciphertext with the given vector `a` and element `b`.
    pub fn new(a: Vec<F>, b: F) -> Self {
        LWECiphertext { a, b }
    }

    /// Returns the vector `a` of the ciphertext.
    pub fn a(&self) -> &[F] {
        self.a.as_ref()
    }

    /// Returns the element `b` of the ciphertext.
    pub fn b(&self) -> F {
        self.b
    }

    /// Performs addition of two LWE ciphertexts modulo a given modulus.
    pub fn add_modulo(&self, other: &Self, modulus: F) -> Self {
        let a = self.a.iter().zip(other.a.iter())
            .map(|(x, y)| {
                let sum = *x + *y;
                if sum >= modulus {
                    sum - modulus
                } else {
                    sum
                }
            }).collect();
        let b = if self.b + other.b >= modulus {
            self.b + other.b - modulus
        } else {
            self.b + other.b
        };
        LWECiphertext::new(a, b)
    }
}

