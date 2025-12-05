use algebra::{DenseMultilinearExtension, Field, ListOfProductsOfPolynomials, PolynomialInfo};
use helper::Transcript;
use serde::Serialize;
use sumcheck::{MLSumcheck, Proof};
use std::rc::Rc;
use trace::{NTTTraceMLE};

use crate::{NTTFourierEvalIOP, SumcheckClaim, ntt::ntt_fourier_eval::NTTFourierProof};

pub struct NTTPolyEvalIOP<F: Field> {
    _marker: std::marker::PhantomData<F>,
}

pub struct NTTPolyEvalInstance<F: Field> {
    pub num_vars: usize,
    pub coefficients: Rc<DenseMultilinearExtension<F>>,
    pub ntt_table: Rc<Vec<F>>,
    pub point_u: Vec<F>,
    pub evaluations_at_u: F,
}

pub struct NTTPolyEvalProof<F: Field> {
    // claimed sum in the sumcheck protocol
    pub claimed_sum: F,
    // polynomial info in the sumcheck protocol
    pub poly_info: PolynomialInfo,
    // sumcheck proofs for the NTT evaluation
    pub sumcheck_proof: Proof<F>,
    // subclaim1: evaluation of the coeffient polynomial at point_v
    pub coeff_eval_at_v: F,
    // subclaim2: evaluation of the fourier polynomial at point_v
    pub fourier_eval_at_v: F,
    // fourier evaluation proof for subclaim 2
    pub fourier_eval_proof: NTTFourierProof<F>,
}

impl<F: Field> NTTPolyEvalInstance<F> {
    pub fn from(trace: &NTTTraceMLE<F>, point_u: Vec<F>) -> Self {
        NTTPolyEvalInstance {
            num_vars: trace.num_vars(),
            coefficients: Rc::clone(&trace.coefficients),
            ntt_table: Rc::clone(&trace.ntt_table),
            point_u: point_u.clone(),
            evaluations_at_u: trace.evaluations.evaluate(&point_u),
        }
    }
}

impl<F: Field + Serialize> NTTPolyEvalIOP<F> {
    pub fn prove(
        instance: &NTTPolyEvalInstance<F>,
        trans: &mut Transcript<F>,
    ) -> NTTPolyEvalProof<F> {
        let sumcheck_claim = SumcheckClaim::new(instance.num_vars);

        Self::update_sumcheck_claim(instance, &mut sumcheck_claim, &[F::one()]);
        let (sumcheck_proof, sumcheck_state) = 
            MLSumcheck::<F>::prove(trans, sumcheck_claim.poly_ref()).expect("[NTTPolyEvalIOP] Fail to generate sumcheck proof");
        
        let coeff_eval_at_v = instance
            .coefficients
            .evaluate(&sumcheck_state.randomness);
        
    }

    pub fn update_sumcheck_claim(
        instance: &NTTPolyEvalInstance<F>,
        claim: &mut SumcheckClaim<F>,
        randomness: &[F]
    ) {
        assert_eq!(randomness.len(), 1);
        let fourier_at_u = Rc::new(init_fourier_table(&instance.point_u, &instance.ntt_table));
        claim.poly_mut().add_product(
            [Rc::clone(&fourier_at_u), Rc::clone(&instance.coefficients)],
            randomness[0],
        );
        *claim.sum_mut() += instance.evaluations_at_u * randomness[0]
    }
}

// Naive implementation for initializing F(u, x) in NTT (for testing purpose).
// In negacyclic NTT where the ring is defined as R = F[x] / (x^N + 1), the Fourier matrix is different since we choose these points: 
// X = ω^1, ω^3, ..., ω^{2N-1} such that X^N = -1.
//
// # Arguments
// * u: the random point
// * ntt_table: It stores the NTT table: ω^0, ω^1, ..., ω^{2N - 1}
//
// # Returns
// * The MLE for F(u, x)
pub fn naive_init_fourier_table<F: Field>(
    u: &[F],
    ntt_table: &[F],
) -> DenseMultilinearExtension<F> {
    let log_n = u.len();
    let m = ntt_table.len(); // M = 2N = 2 * (1 << dim)

    let mut evaluations = vec![F::one(); 1 << log_n];

    // F^R(u, x) = \prod_{i=0}^{\log{N-1}} ((1 - u_i) + u_i * {ω_{2^{i + 1}} ^ X}) * ω^{2^i * x_i}
    // ω_{2^{i + 1}} = ω^ (M / (2^{i+1})), which is the 2^{i+1}-th root of unity
    for (x, eval_at_x) in evaluations.iter_mut().enumerate() {
        for (i, &u_i) in u.iter().enumerate().take(log_n) {
            let idx = (1 << (log_n - i)) * x % m;
            // use little-endian representation so that x_i is the i-th bit of x
            let x_i = (x >> i) & 1;
            let x_i_idx = (1 << i) * x_i;
            *eval_at_x *= ((F::one() - u_i) + u_i * ntt_table[idx]) * ntt_table[x_i_idx];
        }
    }

    DenseMultilinearExtension::from_evaluations_vec(log_n, evaluations)
}

/// Generate MLE for the Fourier function F(u, x) for x \in \{0, 1\}^dim where u is the random point.
/// Dynamic programming implementation for initializing F(u, x) in NTT with linear time complexity O(N).
///
/// # Arguments
/// * u: the random point
/// * ω: It stores the NTT table: ω^0, ω^1, ..., ω^{2N - 1}
pub fn init_fourier_table<F: Field>(u: &[F], ntt_table: &[F]) -> DenseMultilinearExtension<F> {
    let log_n = u.len(); // n = 1 << dim
    let m = ntt_table.len(); // m = 2n = 2 * (1 << dim)

    // It stores the evaluations of all F(u, x) for x \in {0, 1}^dim.
    // Note that in our implementation, we use little endian form, so the index `0b1011`
    // represents the point `P(1,1,0,1)` in {0,1}^`dim` where x_0 = 1, x_1 = 1, x_2 = 0, x_3 = 1.
    let mut evaluations: Vec<_> = vec![F::zero(); 1 << log_n];
    evaluations[0] = F::one();

    // * Compute \prod_{i=0}^{\log{N-1}} ((1 - u_i) + u_i * {ω_{2^{i + 1}} ^ X}) * ω^{2^i * x_i}
    // We need to update the table from i = 0 to log_n - 1 since ω_{2^{i + 1}} ^ X takes on only 2^{i + 1} distinct values.
    for (i, u_i) in u.iter().enumerate() {
        let last_table_size = 1 << i;

        for j in (0..1 << (i + 1)).rev() {
            // `idx` denotes the index for ω^{2^{i+1} * X} stored in the NTT table where X shares the same last i + 1 bits with j
            let idx = (1 << (log_n - i)) * j % m;
            // `bit` denotes the most significant bit of j. If bit = 1, we need to multiply by ω^{2^i}
            let bit = j >> i;
            if bit == 1 {
                evaluations[j] = evaluations[j % last_table_size]
                    * (F::one() - *u_i + *u_i * ntt_table[idx])
                    * ntt_table[last_table_size];
            }
            else {
                evaluations[j] =
                    evaluations[j % last_table_size] * (F::one() - u_i + *u_i * ntt_table[idx]);
            }
        }
    }
    DenseMultilinearExtension::from_evaluations_vec(log_n, evaluations)
}