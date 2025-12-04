use algebra::Field;
use rand::{CryptoRng, Rng};

use crate::ciphertext::LWECiphertext;

/// Represents a batch of LWE ciphertext additions under a smaller modulus `q`.
///
/// Operation:
/// For each pair of ciphertexts `input1[i]` and `input2[i]`,
/// computes the result as:
/// `output[i].a = (input1[i].a + input2[i].a) mod q`
/// and `output[i].b = (input1[i].b + input2[i].b) mod q`.
#[derive(Debug)]
pub struct LWEAdditionOpInstance<F: Field> {
    /// Vector of LWE ciphertexts.
    pub input1: Vec<LWECiphertext<F>>,
    /// Vector of LWE ciphertexts.
    pub input2: Vec<LWECiphertext<F>>,
    /// Result ciphertexts of the additions: `input1`[i] + `input2`[i] mod `q`.
    pub output: Vec<LWECiphertext<F>>,
    /// Number of ciphertext additions in this batch.
    pub num: usize,
    /// Length of the vector of LWE ciphertexts.
    /// Assumes `len` is a power of two.
    pub len: usize,
    /// Modulus under which the addition is performed.
    pub q: F,
}

impl<F: Field> LWEAdditionOpInstance<F> {
    /// Creates an empty instance of LWE Addition operation.
    pub fn new(q: F, len: usize) -> Self {
        LWEAdditionOpInstance {
            input1: Vec::new(),
            input2: Vec::new(),
            output: Vec::new(),
            num: 0,
            len,
            q,
        }
    }

    /// Creates `num` instances of LWE Addition operation with the given modulus `q`.
    pub fn from(
        q: F,
        num: usize,
        len: usize,
        lhs: Vec<LWECiphertext<F>>,
        rhs: Vec<LWECiphertext<F>>,
        result: Vec<LWECiphertext<F>>,
    ) -> Self {
        assert!(num > 0);
        assert!(len > 0);
        assert_eq!(num, lhs.len());
        assert_eq!(num, rhs.len());
        assert_eq!(num, result.len());
        lhs.iter().for_each(|item| assert_eq!(item.a().len(), len));
        rhs.iter().for_each(|item| assert_eq!(item.a().len(), len));
        result
            .iter()
            .for_each(|item| assert_eq!(item.a().len(), len));
        LWEAdditionOpInstance {
            q,
            num,
            len,
            input1: lhs,
            input2: rhs,
            output: result,
        }
    }

    /// Adds a pair of ciphertexts to the operation instance.
    pub fn add_pair(
        &mut self,
        lhs: LWECiphertext<F>,
        rhs: LWECiphertext<F>,
        result: LWECiphertext<F>,
    ) {
        assert_eq!(self.input1.len(), self.input2.len());
        self.num += 1;
        self.input1.push(lhs);
        self.input2.push(rhs);
        self.output.push(result);
    }

    /// Generates `num` random instances of LWE Addition operation with the given modulus `q`.
    ///
    /// # Arguments
    /// * `rng` - Random number generator.
    /// * `q` - Modulus under which the addition is emulated in the field.
    /// * `num` - Number of instances to generate.
    /// * `len` - Length of the vector of LWE ciphertexts.
    pub fn random<R: Rng + CryptoRng>(rng: &mut R, q: F, num: usize, len: usize) -> Self {
        assert!(num > 0);
        assert!(len > 0);
        let lhs: Vec<LWECiphertext<F>> = (0..num)
            .map(|_| LWECiphertext::random(rng, q, len))
            .collect();
        let rhs: Vec<LWECiphertext<F>> = (0..num)
            .map(|_| LWECiphertext::random(rng, q, len))
            .collect();
        let result: Vec<LWECiphertext<F>> = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(l, r)| l.add_modulo(r, q))
            .collect();
        LWEAdditionOpInstance::from(q, num, len, lhs, rhs, result)
    }
}

#[test]
fn test_lwe_addition_op_instance() {
    use algebra_derive::Field;

    #[derive(Field)]
    #[modulus = 23]
    pub struct FF(u32);

    let mut rng = rand::rng();
    let q = FF::new(7);

    let len: usize = 5;

    let mut inst1 = LWEAdditionOpInstance::<FF>::new(q, len);
    assert_eq!(inst1.num, 0);
    let lwe_cipher1 = LWECiphertext::<FF>::random(&mut rng, q, len);
    let lwe_cipher2 = LWECiphertext::<FF>::random(&mut rng, q, len);
    let lwe_cipher3 = lwe_cipher1.add_modulo(&lwe_cipher2, q);
    inst1.add_pair(lwe_cipher1, lwe_cipher2, lwe_cipher3);
    println!("{:?}", inst1);

    let num = 2;
    let len = 2;
    let inst2 = LWEAdditionOpInstance::<FF>::random(&mut rng, q, num, len);
    println!("{:?}", inst2);
}
