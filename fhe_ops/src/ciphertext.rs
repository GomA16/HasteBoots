use algebra::{Field, FieldUniformSampler, NTTField, NTTPolynomial, Polynomial};
use rand::{CryptoRng, Rng};

/// LWE ciphertext structure.
/// This structure represents a ciphertext in the LWE (Learning With Errors) scheme,
/// which is a foundational primitive in fully homomorphic encryption (FHE).
/// It contains a vector of field elements `a` and a single field element `b`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LWECiphertext<F: Field> {
    /// Represents the first component (vector of field elements).
    a: Vec<F>,
    /// Represents the second component (a single field element), computed as
    /// the dot product of `a` with a secret vector, plus message and some noise.
    b: F,
}

/// RLWE ciphertext structure.
/// This structure represents a ciphertext in the RLWE (Ring Learning With Errors) scheme,
/// which is a variant of LWE that operates over polynomial rings.
/// It contains two vectors of field elements `a` and `b`, where `a` and `b` represent the polynomial coefficients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RLWECiphertext<F: NTTField> {
    /// Represents the first component (polynomial coefficients).
    a: Polynomial<F>,
    /// Represents the second component (polynomial coefficients).
    b: Polynomial<F>,
}

/// NTTRLWE ciphertext structure.
pub struct NTTRLWECiphertext<F: NTTField> {
    /// Represents the first component (NTT representation of a polynomial).
    a: NTTPolynomial<F>,
    /// Represents the first component (NTT representation of a polynomial).
    b: NTTPolynomial<F>,
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
        let a = self
            .a
            .iter()
            .zip(other.a.iter())
            .map(|(x, y)| {
                let sum = *x + *y;
                if sum >= modulus {
                    sum - modulus
                } else {
                    sum
                }
            })
            .collect();
        let b = if self.b + other.b >= modulus {
            self.b + other.b - modulus
        } else {
            self.b + other.b
        };
        LWECiphertext::new(a, b)
    }

    /// Generates a random LWE ciphertext with the given modulus `q` and length `len`.
    pub fn random<R: Rng + CryptoRng>(rng: &mut R, q: F, len: usize) -> Self {
        let uniform = FieldUniformSampler::<F>::new();
        let a = (0..len)
            .map(|_| uniform.sample_range(rng, q.value()))
            .collect();
        let b = uniform.sample_range(rng, q.value());
        LWECiphertext::new(a, b)
    }
}
