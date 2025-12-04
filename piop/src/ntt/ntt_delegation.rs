//! PIOP for NTT Bare without delegation the evaluation of F(u, r)

use algebra::{DenseMultilinearExtension, Field};
use std::rc::Rc;
use helper::Transcript;

/// store the intermediate mles generated in each iteration in the `init_fourier_table_overall` algorithm
pub struct IntermediateMLEs<F: Field> {
    f_mles: Vec<Rc<DenseMultilinearExtension<F>>>,
    w_mles: Vec<Rc<DenseMultilinearExtension<F>>>,
}

impl<F: Field> IntermediateMLEs<F> {
    /// Initiate the vector
    pub fn new(n_rounds: u32) -> Self {
        IntermediateMLEs {
            f_mles: Vec::with_capacity(n_rounds as usize),
            w_mles: Vec::with_capacity(n_rounds as usize),
        }
    }

    /// Add the intermediate mles generated in each round
    pub fn add_round_mles(&mut self, num_vars: usize, f_mle: &[F], w_mle: Vec<F>) {
        self.f_mles
            .push(Rc::new(DenseMultilinearExtension::from_evaluations_slice(
                num_vars, f_mle,
            )));
        self.w_mles
            .push(Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars, w_mle,
            )));
    }

    /// Convert into the stored mles
    pub fn into_mles(self) -> (Vec<Rc<DenseMultilinearExtension<F>>>, Vec<Rc<DenseMultilinearExtension<F>>>) {
        (self.f_mles, self.w_mles)
    }

    /// Get f_mles
    pub fn get_f_mles(&self, idx: usize) -> Rc<DenseMultilinearExtension<F>> {
        self.f_mles[idx].clone()
    }
}


/// This function is the same as the `init_fourier_table` in `ntt_bare.rs` but it also stores the intermediate mles for delegation.
/// 
/// # Arguments
/// * u: the random point
/// * ntt_table: It stores the NTT table
/// 
/// # Returns
/// * IntermediateMLEs: it stores all the intermediate mles generated in each iteration
pub fn init_fourier_table_with_mle<F: Field>(u: &[F], ntt_table: &[F]) -> IntermediateMLEs<F> {
    let log_n = u.len(); // N = 1 << dim
    let m = ntt_table.len(); // M = 2N = 2 * (1 << dim)

    let mut evaluations: Vec<_> = vec![F::zero(); 1 << log_n];
    evaluations[0] = F::one();

    // stores all the intermediate evaluations of the table (i.e. F(u, x)) and the term ω^{2^{i + 1} * X} in each iteration
    let mut intermediate_mles = <IntermediateMLEs<F>>::new(log_n as u32);

    for (i, u_i) in u.iter().enumerate() {
        let this_round_dim = i + 1;
        let last_round_dim = this_round_dim - 1;
        let this_round_table_size = 1 << this_round_dim;
        let last_round_table_size = 1 << last_round_dim;

        let mut evaluations_w_term = vec![F::zero(); this_round_table_size];
        for x in (0..this_round_table_size).rev() {
            let idx = (1 << (log_n - i)) * x % m;
            // the bit index in this iteration is last_round_dim = this_round_dim - 1
            // If x >= last_round_table_size, meaning the bit = 1, we need to multiply by ω^{2^last_round_dim * 1}
            if x >= last_round_table_size {
                evaluations[x] = evaluations[x % last_round_table_size]
                    * (F::one() - *u_i + *u_i * ntt_table[idx])
                    * ntt_table[1 << last_round_dim];
            }
            // the bit index in this iteration is last_round_dim = this_round_dim - 1
            // If x < last_round_table_size, meaning the bit = 0, we do not need to multiply because ω^{2^last_round_dim * 0} = 1
            else {
                evaluations[x] = evaluations[x % last_round_table_size]
                    * (F::one() - *u_i + *u_i * ntt_table[idx]);
            }
            // store the term ω_{2^{i + 1}} ^ x in this iteration
            evaluations_w_term[x] = ntt_table[idx];
        }
        intermediate_mles.add_round_mles(
            this_round_dim,
            &evaluations[..this_round_table_size],
            evaluations_w_term,
        );
    }

    intermediate_mles
}

/// Naive implementation for computing the MLE: w_{i+1} (x)= w^{ M / {2^{i+1}}  * X} = w^{ N / 2^i * X}
/// for x \in \{0, 1\}^x_dim in a naive method (for testing purpose)
/// Or, it is the power of the 2^{i+1}-th root of unity.
///
/// # Arguments:
///
/// * ntt_table: NTT table for w (M-th root of unity) containing {1, w, w^1, ..., w^{M-1}}
/// * sub: the exponent of the function defined above (i+1) = x_dim
pub fn naive_w_power_times_x_table<F: Field>(
    ntt_table: &[F],
    subscript: usize,
) -> DenseMultilinearExtension<F> {
    let m = ntt_table.len();
    assert!(m.is_power_of_two());

    let x_dim = subscript;
    let mut evaluations = vec![F::zero(); 1 << x_dim];
    for x in 0..(1 << x_dim) {
        evaluations[x] = ntt_table[(m >> x_dim) * x % m];
    }
    DenseMultilinearExtension::from_evaluations_vec(x_dim, evaluations)
}

/// Evaluate the mle w_{i+1} (x) for a random point r \in F^{x_dim} where w_{i+1} denotes the 2^{i+1}-th root of unity
///
/// w_{i+1} (x)= w^{ M / {2^{i+1}} * X} = w^{ N / 2^i * X} for a random point r
///               = \prod_k (1 - r_k + r_k * w_{i+1}^{2^k})
///
/// # Arguments:
///
/// * ntt_table: NTT table for w (M-th root of unity) containing {1, w, w^1, ..., w^{M-1}}
/// * log_m: log of M
/// * x_dim: dimension of x or the num of variables of the outputted mle, which is also the subscript of w_{i+1}
/// * subscript: the subscript of the function w_{i+1} (x) which equals the dimension of this MLE
/// * r: random point in F^{x_dim}
pub fn eval_w_power_times_x<F: Field>(
    ntt_table: &[F],
    log_m: usize,
    x_dim: usize,
    subscript: usize,
    r: &[F],
) -> F {
    assert_eq!(ntt_table.len(), 1 << log_m);
    assert_eq!(x_dim, r.len());
    assert_eq!(x_dim, subscript);
    let mut prod = F::one();

    for (i, &r_i) in r.iter().enumerate() {
        let log_exp = (log_m - subscript + i) % log_m;
        prod *= F::one() - r_i + r_i * ntt_table[1 << log_exp];
    }

    prod
}

#[cfg(test)]
mod test {
    use super::{eval_w_power_times_x, naive_w_power_times_x_table};
    use algebra::{
        DenseMultilinearExtension, FieldUniformSampler, NTTField, derive::{DecomposableField, FheField, Field, NTT, Prime}, transformation::AbstractNTT
    };
    use num_traits::{One, Zero};
    use rand::thread_rng;
    use rand_distr::Distribution;

    use super::init_fourier_table_with_mle;

    #[derive(Field, DecomposableField, FheField, Prime, NTT)]
    #[modulus = 132120577]
    pub struct Fp32(u32);
    // field type
    type FF = Fp32;

    /// Given an `index` of `len` bits, output a new index where the bits are reversed.
    fn reverse_bits(index: usize, len: usize) -> usize {
        let mut tmp = index;
        let mut reverse_index = 0;
        let mut pow = 1 << (len - 1);
        for _ in 0..len {
            reverse_index += pow * (1 & tmp);
            pow >>= 1;
            tmp >>= 1;
        }
        reverse_index
    }

    #[test]
    fn test_init_fourier_table_overall() {
        let uniform = <FieldUniformSampler<FF>>::new();
        let mut rng = thread_rng();
        

        let dim = 10;
        let m = 1 << (dim + 1); // M = 2N = 2 * (1 << dim)
        let u: Vec<_> = (0..dim).map(|_| uniform.sample(&mut rng)).collect();
        let v: Vec<_> = (0..dim).map(|_| uniform.sample(&mut rng)).collect();

        let mut u_v: Vec<_> = Vec::with_capacity(dim << 1);
        u_v.extend(&u);
        u_v.extend(&v);

        // root is the M-th root of unity
        let root = FF::try_minimal_primitive_root(m).unwrap();

        let mut fourier_matrix = vec![FF::zero(); (1 << dim) * (1 << dim)];
        let mut ntt_table = Vec::with_capacity(m as usize);

        let mut power = FF::one();
        for _ in 0..m {
            ntt_table.push(power);
            power *= root;
        }

        let ntt_table = FF::get_ntt_table(dim as u32).unwrap().root_powers();

        // The special structure of the fourier matrix is defined as:
        // F[i, j] = w^{(2 * rev_i + 1) * j} where w is the M-th root of unity.
        // With little endian representation, we have:
        // F[x_0, x_1, ..., x_{\logN-1} || y_0, y_1, ..., y_{\logN-1}] = F_matrix[i, j] = w^{(2 * rev_i + 1) * j}
        // where i = \sum_k 2^k * x_k and j = \sum_k 2^k * y_k
        // and rev_i = \sum_k 2^{\logN-1-k} x_k 
        for i in 0..1 << dim {
            for j in 0..1 << dim {
                // ! Modified Formula
                let rev_i = reverse_bits(i, dim);
                let idx_power = ((2 * rev_i + 1) * j) as u32 % m;
                let idx_fourier = i + (j << dim);
                fourier_matrix[idx_fourier] = ntt_table[idx_power as usize];
            }
        }

        let fourier_mle = DenseMultilinearExtension::from_evaluations_vec(dim << 1, fourier_matrix);
        let partial_fourier_mle = &init_fourier_table_with_mle(&u, &ntt_table).get_f_mles(dim - 1);

        assert_eq!(fourier_mle.evaluate(&u_v), partial_fourier_mle.evaluate(&v));
    }

    #[test]
    fn test_w_power_x() {
        let dim = 10; // meaning x\in \{0, 1\}^{dim} and N = 1 << dim
        let log_m = dim + 1;
        let m = 1 << log_m; // M = 2N

        // root is the M-th root of unity
        let root = FF::try_minimal_primitive_root(m).unwrap();

        let mut ntt_table = Vec::with_capacity(m as usize);

        let mut power = FF::one();
        for _ in 0..m {
            ntt_table.push(power);
            power *= root;
        }

        let sampler = <FieldUniformSampler<FF>>::new();
        let mut rng = thread_rng();

        for x_dim in 0..=dim {
            let r: Vec<_> = (0..x_dim).map(|_| sampler.sample(&mut rng)).collect();
            let w_mle = naive_w_power_times_x_table(&ntt_table, x_dim);
            let w_eval = eval_w_power_times_x(&ntt_table, log_m, x_dim, x_dim, &r);
            assert_eq!(w_eval, w_mle.evaluate(&r));
        }
    }
}

