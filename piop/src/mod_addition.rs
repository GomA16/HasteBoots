use algebra::{DenseMultilinearExtension, Field};
use fhe_ops::LWEAdditionOpInstance;
use itertools::izip;
use core::fmt;
use std::rc::Rc;

/// Represents multiple instances of modular addition operations.
///
/// # Operation:
/// `a` + `b` = `c` mod `q`.
/// * `a` - in the range [0, q)
/// * `b` - in the range [0, q)
/// * `c` - in the range [0, q)
/// * `q` - modulus
///
/// # Relation:
/// `a` + `b` = `k` * q + `c`
/// * `k` - in the range [0, 1], computed by the prover as the witness.
///
/// # Arrangement
/// There are 1 << (`num_vars` + `num_batches`) modular addition instances of each [`ModularAdditionInstance`].
/// The input/output and witness can be considered as a matrix with `num_vars` columns and `num_batches` rows,
/// where each row will be considered as a batch and encoded into a multivariate polynomial.
pub struct ModularAdditionInstance<F: Field> {
    /// Vector of input `a` values encoded as `num_batches` dense multilinear polynomials of `num_vars` variables.
    pub input_a: Vec<Rc<DenseMultilinearExtension<F>>>,
    /// Vector of input `b` values encoded as `num_batches` dense multilinear polynomials of `num_vars` variables.
    pub input_b: Vec<Rc<DenseMultilinearExtension<F>>>,
    /// Vector of output `c` values encoded as `num_batches` dense multilinear polynomials of `num_vars` variables.
    pub output_c: Vec<Rc<DenseMultilinearExtension<F>>>,
    /// Vector of witness `k` values encoded as `num_batches` dense multilinear polynomials of `num_vars` variables.
    pub witness_k: Vec<Rc<DenseMultilinearExtension<F>>>,
    /// Number of variables in each dense multilinear polynomial.
    pub num_vars: usize,
    /// Number of polynomials in each input/output/witness vector.
    pub num_batches: usize,
    /// Modulus for the addition operation.
    pub q: F,
}

impl<F: Field> fmt::Debug for ModularAdditionInstance<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ModularAdditionInstance {{")?;
        writeln!(f, "num_vars: {}, num_batches: {}, q: {}", self.num_vars, self.num_batches, self.q)?;

        writeln!(f, "input_a: ")?;
        for a in &self.input_a {
            writeln!(f, "{}", a)?;
        }
        writeln!(f, "input_b: ")?;
        for b in &self.input_b {
            writeln!(f, "{}", b)?;
        }
        writeln!(f, "output_c: ")?;
        for c in &self.output_c {
            writeln!(f, "{}", c)?;
        }
        writeln!(f, "witness_k: ")?;
        for k in &self.witness_k {
            writeln!(f, "{}", k)?;
        }
        writeln!(f, "}}")
    }
}

impl<F: Field> ModularAdditionInstance<F> {
    pub fn new(q: F, num_vars: usize) -> Self {
        ModularAdditionInstance {
            input_a: Vec::new(),
            input_b: Vec::new(),
            output_c: Vec::new(),
            witness_k: Vec::new(),
            num_vars,
            num_batches: 0,
            q,
        }
    }

    /// Generates a [`ModularAdditionInstance`] from an [`LWEAdditionOpInstance`] in the default arrangement.
    ///
    /// The first component (vector `a`) of each LWE ciphertext is encoded into a dense multilinear polynomial,
    /// and the second componet (element `b`) is **disregarded** for ease of alignment. (If needed, all elements `b`s
    /// can be gathered together and encoded into another dense multilinear polynomial. Alternatively, a single
    /// modular addition instance can be quickly verified by the verifier.)
    pub fn from_op(op_instance: LWEAdditionOpInstance<F>, num_vars: usize) -> Self {
        assert_eq!(1 << num_vars, op_instance.len);
        let input_a: Vec<Rc<DenseMultilinearExtension<F>>> = op_instance
            .input1
            .iter()
            .map(|ct| {
                Rc::new(DenseMultilinearExtension::from_evaluations_slice(
                    num_vars,
                    ct.a(),
                ))
            })
            .collect();
        let input_b: Vec<Rc<DenseMultilinearExtension<F>>> = op_instance
            .input2
            .iter()
            .map(|ct| {
                Rc::new(DenseMultilinearExtension::from_evaluations_slice(
                    num_vars,
                    ct.a(),
                ))
            })
            .collect();
        let output_c: Vec<Rc<DenseMultilinearExtension<F>>> = op_instance
            .output
            .iter()
            .map(|ct| {
                Rc::new(DenseMultilinearExtension::from_evaluations_slice(
                    num_vars,
                    ct.a(),
                ))
            })
            .collect();

        // Computes the witness `k`
        let witness_k: Vec<Rc<DenseMultilinearExtension<F>>> =
            izip!(input_a.iter(), input_b.iter(), output_c.iter())
                .map(|(poly_a, poly_b, poly_c)| {
                    let vec_c: Vec<F> = izip!(poly_a.iter(), poly_b.iter(), poly_c.iter())
                        .map(|(&a, &b, &c)| match a + b == c {
                            true => F::zero(),
                            false => F::one(),
                        })
                        .collect();
                    Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                        num_vars, vec_c,
                    ))
                })
                .collect();

        Self {
            input_a,
            input_b,
            output_c,
            witness_k,
            num_vars,
            num_batches: op_instance.num,
            q: op_instance.q,
        }
    }

    /// Generates a [`ModularAdditionInstance`] from an [`LWEAdditionOpInstance`] in a flattened arrangement.
    /// 
    /// The first component (vector `a`) of each LWE ciphertext is encoded into a dense multilinear polynomial,
    /// and the second componet (element `b`) is **disregarded** for ease of alignment. (If needed, all elements `b`s
    /// can be gathered together and encoded into another dense multilinear polynomial. Alternatively, a single
    /// modular addition instance can be quickly verified by the verifier.)
    pub fn from_op_flatten(op_instance: LWEAdditionOpInstance<F>, num_vars: usize) -> Self {
        assert_eq!(1 << num_vars, op_instance.len * op_instance.num);
        let vec_a: Vec<F> = op_instance
            .input1
            .iter()
            .flat_map(|ct| ct.a().iter().copied()).collect();
        let vec_b: Vec<F> = op_instance
            .input2
            .iter()
            .flat_map(|ct| ct.a().iter().copied()).collect();
        let vec_c: Vec<F> = op_instance
            .output
            .iter()
            .flat_map(|ct| ct.a().iter().copied()).collect();
        let vec_k: Vec<F> = izip!(vec_a.iter(), vec_b.iter(), vec_c.iter())
            .map(|(&a, &b, &c)| match a + b == c {
                true => F::zero(),
                false => F::one(),
            })
            .collect();

        Self {
            input_a: vec![Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars, vec_a,
            ))],
            input_b: vec![Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars, vec_b,
            ))],
            output_c: vec![Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars, vec_c,
            ))],
            witness_k: vec![Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars, vec_k,
            ))],
            num_vars,
            num_batches: 1,
            q: op_instance.q,
        }
    }
}

#[test]
fn test_modular_addition_instance_from_op() {
    use algebra_derive::Field;
    use fhe_ops::LWEAdditionOpInstance;

    #[derive(Field)]
    #[modulus = 23]
    struct FF(u32);

    let q = FF::new(7u32);
    let mut rng = rand::thread_rng();

    let len = 4; // power of 2
    let num = 3; // number of batches
    let op_instance = LWEAdditionOpInstance::random(&mut rng, q, num, len);
    
    let num_vars = 2; // log2(len)
    let instance = ModularAdditionInstance::from_op(op_instance, num_vars);
    println!("{:?}", instance);
}

#[test]
fn test_modular_addition_instance_from_op_flatten() {
    use algebra_derive::Field;
    use fhe_ops::LWEAdditionOpInstance;

    #[derive(Field)]
    #[modulus = 23]
    struct FF(u32);

    let q = FF::new(7u32);
    let mut rng = rand::thread_rng();

    let len = 4; // power of 2
    let num = 4; // number of batches
    let op_instance = LWEAdditionOpInstance::random(&mut rng, q, num, len);
    
    let num_vars = 4; // log2(len * num)
    let instance = ModularAdditionInstance::from_op_flatten(op_instance, num_vars);
    println!("{:?}", instance);
}