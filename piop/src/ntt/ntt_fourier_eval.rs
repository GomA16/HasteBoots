use algebra::{Field, ListOfProductsOfPolynomials, PolynomialInfo};
use helper::Transcript;
use serde::Serialize;
use sumcheck::{Proof, prover::ProverState, verifier::SubClaim};
use std::{marker::PhantomData, rc::Rc};
use algebra::DenseMultilinearExtension;
use helper::utils::{gen_identity_evaluations, eval_identity_function};
use sumcheck::MLSumcheck;

/// Fourier Matrix evaluation instance
/// 
/// Definition of Fourier Matrix in NTT:
/// [Matrix Definition]
/// F is a matrix where F(Y, X) denotes the X-th row and Y-th column
/// in little endian format.
/// F(Y, X) = w^{(2Y^R + 1) * X} where Y^R is the bit-reversal of Y
/// 
/// [MLE Definition]
/// F(y, x) = F(y_0, y_1, ..., y_{k-1}, x_0, x_1, ..., x_{k-1}) = F(Y, X)
/// where Y = y_0 * 2^0 + y_1 * 2^1 + ... + y_{k-1} * 2^{k-1}
/// and X = x_0 * 2^0 + x_1 * 2^1 + ... + x_{k-1} * 2^{k-1}
/// for y, x \in {0, 1}^k where k = log_2(N) and N is the size of the NTT.
/// Note that Y^R = y_{k-1} * 2^0 + y_{k-2} * 2^1 + ... + y_0 * 2^{k-1}.
/// 
/// F(u, v) = eval 
pub struct NTTFourierEvalInfo<F: Field> {
    pub num_vars: usize,
    pub ntt_table: Rc<Vec<F>>,
    pub point_u: Vec<F>,
    pub point_v: Vec<F>,
    pub eval: F,
}

pub struct NTTFourierProof<F: Field> {
    pub num_rounds: usize,
    pub sumcheck_proofs: Vec<Proof<F>>,
    pub sub_claims: Vec<F>,
    pub final_claim: F,
}

pub struct NTTFourierEvalIOP<F: Field>(PhantomData<F>);

/// store the intermediate mles generated in each iteration in the `init_fourier_table_overall` algorithm
pub struct IntermediateMLEs<F: Field> {
    f_mles: Vec<Rc<DenseMultilinearExtension<F>>>,
    w_mles: Vec<Rc<DenseMultilinearExtension<F>>>,
}

impl<F: Field + Serialize> NTTFourierEvalIOP<F> {

    pub fn prove(
        info: &NTTFourierEvalInfo<F>,
        trans: &mut Transcript<F>,
    ) -> NTTFourierProof<F> {
        let ntt_bare_randomness = &info.point_v;
        let u = &info.point_u;
        let log_n = info.num_vars;

        let (f_mles, w_mles) = init_fourier_table_with_mle(u, &info.ntt_table).into_mles();

        // 1. (detached) prove a(u) = \sum_{x\in \{0, 1\}^{\log N} c(x)\cdot F(u, x) } for a random point u

        // the above sumcheck is reduced to prove F(u, v) where v is the requested point
        // Note that the delegated value F(u, v) is stored in proof.delegation_claimed_sums[0].
        let mut requested_point = ntt_bare_randomness.to_owned();
        let mut reduced_claim = f_mles[log_n - 1].evaluate(&requested_point);

        // 2. prove the computation of F(u, v) in log_n - 1 rounds

        // store the sumcheck proof in each round
        let mut delegation_sumcheck_msgs = Vec::with_capacity(log_n - 1);
        // store the claimed sum of the sumcheck protocol in each round
        let mut delegation_claimed_sums = Vec::with_capacity(log_n - 1);
        // ! Modified Formula
        for k in (1..log_n).rev() {
            // start form log_n - 1;
            // let i = log_n - 1 - k;
            delegation_claimed_sums.push(reduced_claim);

            let w_coeff = info.ntt_table[1 << k];
            let f = &f_mles[k - 1];
            let (proof_round, state_round) = Self::delegation_prover_round(
                trans,
                k,
                &requested_point,
                // ! Modified Formula
                u[k],
                w_coeff,
                f,
                &w_mles[k],
            );
            delegation_sumcheck_msgs.push(proof_round);

            // the requested point returned from this round of sumcheck protocol, which initiates the claimed sum of the next round
            requested_point = state_round.randomness;
            reduced_claim = f.evaluate(&requested_point);
        }

        NTTFourierProof {
            num_rounds: log_n - 1,
            sumcheck_proofs: delegation_sumcheck_msgs,
            sub_claims: delegation_claimed_sums,
            final_claim: reduced_claim,
        }
    }

    /// The delegation of F(u, v) consists of logN - 1 rounds, each of which is a sumcheck protocol.
    ///
    /// We define $A_{F}^{(k)}:\{0,1\}^{k+1} -> \mathbb{F}$ and $ω^{(k)}_{i+1}:\{0,1\}^{k+1} -> \mathbb{F}$.
    /// The prover asserts the following sum = \tilde{A}_{F}^{(k)}(x, b) at a random point $(x, b)\in \mathbb{F}^{k+1}$:
    /// sum = \sum_{z\in \{0,1\}}^k
    ///         \tilde{\beta}((x, b),(z,0)) * \tilde{A}_{F}^{(k-1)}(z) ( (1-u_{i})+u_{i} * \tilde{ω}^{(k)}_{i+1}(z, 0)
    ///       + \tilde{\beta}((x, b),(z,1)) * \tilde{A}_{F}^{(k-1)}(z) ( (1-u_{i})+u_{i} * \tilde{ω}^{(k)}_{i+1}(z, 1) * ω^{2^k}
    /// where $\ω^{(k)}_{i+1}(x,b ) = \ω^{2^{i+1}\cdot j}$ for $j = X+2^{i+1}\cdot b$.
    ///
    /// In the term of the data structure, the polynomial to be sumed can be viewed as the sum of two products,
    /// one has coefficient one, and the other has coefficient ω^{2^k}.
    ///
    /// # Arguments
    /// * round: round number denoted by k, which is iterated in a reverse order as described in the algorithm
    /// * point: the random point $(x, b)\in \mathbb{F}^{k+1}$ reduced from the last sumcheck, used to prove the sum in the round
    /// * u_i: parameter in this round as described in the formula
    /// * w_coeff: the coefficient ω^{2^k} of the second product
    /// * f: MLE \tilde{A}_{F}^{(k-1)}(z) for z\in \{0,1\}^k
    /// * w: MLE \tilde{ω}^{(k)}_{i+1}(z, b) for z\in \{0,1\}^k  and b\in \{0, 1\}, which will be divided into two smaller MLEs \tilde{ω}^{(k)}_{i+1}(z, 0) and \tilde{ω}^{(k)}_{i+1}(z, 1)
    pub fn delegation_prover_round(
        trans: &mut Transcript<F>,
        round: usize,
        point: &[F],
        u_i: F,
        w_coeff: F,
        f: &Rc<DenseMultilinearExtension<F>>,
        w: &Rc<DenseMultilinearExtension<F>>,
    ) -> (Proof<F>, ProverState<F>) {
        assert_eq!(f.num_vars, round);
        assert_eq!(w.num_vars, round + 1);

        let mut poly = <ListOfProductsOfPolynomials<F>>::new(round);

        // the equality function defined by the random point $(x, b)\in \mathbb{F}^{k+1}$
        // it is divided into two MLEs \tilde{\beta}((x, b),(z,0)) and \tilde{\beta}((x, b),(z,1))
        let eq_func = gen_identity_evaluations(point);
        let (eq_func_left, eq_func_right) = eq_func.split_halves();

        // two divided MLEs: \tilde{ω}^{(k)}_{i+1}(z, 0) and \tilde{ω}^{(k)}_{i+1}(z, 1)
        let (w_left, w_right) = w.split_halves();

        // construct the polynomial to be sumed
        // left product is \tilde{\beta}((x, b),(z,0)) * \tilde{A}_{F}^{(k-1)}(z) ( (1-u_{i})+u_{i} * \tilde{ω}^{(k)}_{i+1}(z, 0)
        // right product is \tilde{\beta}((x, b),(z,1)) * \tilde{A}_{F}^{(k-1)}(z) ( (1-u_{i})+u_{i} * \tilde{ω}^{(k)}_{i+1}(z, 1) * ω^{2^k}
        poly.add_product_with_linear_op(
            [Rc::new(eq_func_left), Rc::clone(f), Rc::new(w_left)],
            &[
                (F::one(), F::zero()),
                (F::one(), F::zero()),
                (u_i, F::one() - u_i),
            ],
            F::one(),
        );

        poly.add_product_with_linear_op(
            [Rc::new(eq_func_right), Rc::clone(f), Rc::new(w_right)],
            &[
                (F::one(), F::zero()),
                (F::one(), F::zero()),
                (u_i, F::one() - u_i),
            ],
            w_coeff,
        );

        MLSumcheck::prove(trans, &poly).expect("ntt proof of delegation failed in round {round}")
    }

    /// Compared to the `prove` functionality, we remove the phase to prove NTT bare.
    /// Also, after detaching the verification of NTT bare, verifier can directly check the recursive proofs.
    pub fn verify_recursive(
        trans: &mut Transcript<F>,
        proof: &NTTFourierProof<F>,
        info: &NTTFourierEvalInfo<F>,
    ) -> bool {
        let log_n = info.num_vars;
        assert_eq!(proof.sumcheck_proofs.len(), log_n - 1);
        assert_eq!(proof.sub_claims.len(), log_n - 1);

        let u = &info.point_u;

        // 1. [detached] verify a(u) = \sum_{x\in \{0, 1\}^{\log N} c(x)\cdot F(u, x) } for a random point u
        // Note that the delegated value F(u, v) is stored in proof.delegation_claimed_sums[0].

        // 2. verify the computation of F(u, v) in log_n - 1 rounds
        let mut requested_point = info.point_v.clone();
        for (cnt, k) in (1..log_n).rev().enumerate() {
            // let i = log_n - 1 - k;

            // verify the proof of the sumcheck protocol
            let poly_info = PolynomialInfo {
                max_multiplicands: 3,
                num_variables: k,
            };
            let subclaim = MLSumcheck::verify(
                trans,
                &poly_info,
                proof.sub_claims[cnt],
                &proof.sumcheck_proofs[cnt],
            )
            .expect("ntt verification failed in round {cnt}");

            // In the last round of the sumcheck protocol, the verify needs to check the equality of the evaluation of the polynomial to be summed at a random point z = r \in \{0,1\}}^k.
            // The verifier is given the evaluation of \tilde{A}_{F}^{(k-1)}(z = r) instead of computing on his own, so he can use it to check.
            // If the equality holds, it is reduced to check the evaluation of \tilde{A}_{F}^{(k-1)}(z = r).
            let reduced_claim = if cnt < log_n - 2 {
                proof.sub_claims[cnt + 1]
            } else {
                proof.final_claim
            };
            // check the equality
            if !Self::delegation_verify_round(
                k,
                &requested_point,
                // ! Modified Formula
                u[k],
                &subclaim,
                reduced_claim,
                info,
            ) {
                panic!("ntt verification failed in round {cnt}");
            }
            requested_point = subclaim.point;
        }

        let delegation_final_claim = proof.final_claim;
        let final_point = requested_point;
        // TODO: handle the case that log = 1
        assert_eq!(final_point.len(), 1);

        // check the final claim returned from the last round of delegation
        // ! Modified Formula
        let idx = 1 << (info.num_vars);
        let eval = eval_identity_function(&final_point, &[F::zero()])
            + eval_identity_function(&final_point, &[F::one()])
                * (F::one() - u[0] + u[0] * info.ntt_table[idx])
                * info.ntt_table[1];

        delegation_final_claim == eval
    }

    /// The delegation of F(u, v) consists of logN - 1 rounds, each of which is a sumcheck protocol.
    ///
    /// We define $A_{F}^{(k)}:\{0,1\}^{k+1} -> \mathbb{F}$ and $ω^{(k)}_{i+1}:\{0,1\}^{k+1} -> \mathbb{F}$.
    /// The prover asserts the following sum = \tilde{A}_{F}^{(k)}(x, b) at a random point $(x, b)\in \mathbb{F}^{k+1}$:
    /// sum = \sum_{z\in \{0,1\}}^k
    ///         \tilde{\beta}((x, b),(z,0)) * \tilde{A}_{F}^{(k-1)}(z) ( (1-u_{i})+u_{i} * \tilde{ω}^{(k)}_{i+1}(z, 0)
    ///       + \tilde{\beta}((x, b),(z,1)) * \tilde{A}_{F}^{(k-1)}(z) ( (1-u_{i})+u_{i} * \tilde{ω}^{(k)}_{i+1}(z, 1) * ω^{2^k}
    /// where $\ω^{(k)}_{i+1}(x,b ) = \ω^{2^{i+1}\cdot j}$ for $j = X+2^{i+1}\cdot b$.
    ///
    /// The verify needs to check the equality of the evaluation of the polynomial to be summed at a random point z = r \in \{0,1\}}^k.
    /// In verification, the verifier is given the evaluation of \tilde{A}_{F}^{(k-1)}(z = r) instead of computing on his own, so he can use it to check.
    /// If the equality holds, it is reduced to check the evaluation of \tilde{A}_{F}^{(k-1)}(z = r).
    ///
    /// # Arguments
    /// * round: round number denoted by k, which is iterated in a reverse order as described in the algorithm
    /// * x_b_point: the random point $(x, b)\in \mathbb{F}^{k+1}$ reduced from the last sumcheck
    /// * u_i: parameter in this round as described in the formula
    /// * subclaim: the subclaim returned from this round of the sumcheck, containing the random point r used for equality check
    /// * reduced_claim: the given evaluation of \tilde{A}_{F}^{(k-1)}(z = r) so verify does not need to compute on his own
    pub fn delegation_verify_round(
        round: usize,
        x_b_point: &[F],
        u_i: F,
        subclaim: &SubClaim<F>,
        reduced_claim: F,
        ntt_instance_info: &NTTFourierEvalInfo<F>,
    ) -> bool {
        let log_n = ntt_instance_info.num_vars;
        let ntt_table = &ntt_instance_info.ntt_table;

        // r_left = (r, 0) and r_right = (r, 0)
        let mut r_left: Vec<_> = Vec::with_capacity(round + 1);
        let mut r_right: Vec<_> = Vec::with_capacity(round + 1);
        r_left.extend(&subclaim.point);
        r_right.extend(&subclaim.point);
        r_left.push(F::zero());
        r_right.push(F::one());

        // compute $\ω^{(k)}_{i+1}(x,b ) = \ω^{2^{i+1}\cdot j}$ for $j = X+2^{i+1}\cdot b$ at point (r, 0) and (r, 1)
        // exp: i + 1 = n - k
        // let exp = log_n - round;
        // ! Modified Formula
        let sub = round + 1;
        // w_left = \tilde{ω}^{(k)}_{i+1}(r, 0) and w_right = \tilde{ω}^{(k)}_{i+1}(r, 0)
        let w_left = eval_w_power_times_x(ntt_table, log_n + 1, round + 1, sub, &r_left);
        let w_right = eval_w_power_times_x(ntt_table, log_n + 1, round + 1, sub, &r_right);

        let eval = eval_identity_function(x_b_point, &r_left)
            * reduced_claim
            * (F::one() - u_i + u_i * w_left)
            + eval_identity_function(x_b_point, &r_right)
                * reduced_claim
                * (F::one() - u_i + u_i * w_right)
                * ntt_table[1 << round];

        eval == subclaim.expected_evaluations
    }
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

