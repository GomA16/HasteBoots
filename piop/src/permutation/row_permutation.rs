//! Row Permutation IOP
//! Prove that matrix A is a row-permuted version of matrix C under permutation ρ
//!
//! Suppose A, C are matrices with dimension N x M, and A[X][Y] denotes the entry
//! at row X and column Y of matrix A. The row permutation under permutation ρ
//! is defined as:
//!```text         
//!         `A[X][Y] = C[ρ(X)][Y] for all X in [N], Y in [M]`.
//!```text
//! This can be represented as a matrix multiplication between the permutation matrix P_ρ
//! and matrix C, i.e. A = P_ρ * C, where P_ρ is an N x N matrix defined as:
//! P_ρ[X][Y] = 1 if Y = ρ(X), else 0.
//!
//! We can also represent this in terms of multilinear extensions:
//! Let A(y, x), C(y, x) be the multilinear extensions of matrices A and C respectively,
//! where y represents the column variables and x represents the row variables.
//!```text     
//!     A(y, x) = \sum_k P(k, x) * C(y, k) where k iterates over \{0, 1\}^logN
//!```text
//! Here, P(k, x) is the multilinear extension of the permutation matrix P_ρ.
//! The relation can be verified using the sumcheck protocol on the above equation:
//!```text
//!      A(r_y, r_x) = \sum_k P(k, r_x) * C(r_y, k) where k iterates over \{0, 1\}^logN
//! ```text
//! for random points r_y, r_x.
//!
//! One observation is that P(k, r_x) can be computed efficiently in O(N) time since
//! each row of P has exactly one 1. Specifically,
//!     P(k, r_x) = eq(r_x, ρ_inv(k)) for all k in \{0, 1\}^logN
//! which means to perform the inverse permutation on the equality function evaluations.
//!
use algebra::DenseMultilinearExtension;
use algebra::Field;
use algebra::MultilinearExtension;
use algebra::PolynomialInfo;
use bincode::config;
use helper::utils::gen_identity_evaluations;
use num_traits::{Zero, ops::inv, sign};
use rand::random;
use serde::Serialize;
use std::{mem::Discriminant, rc::Rc};
use sumcheck::Proof;
use trace::row_perm_trace::RowPermTrace;
use trace::row_perm_trace::RowPermTraceMLE;

use crate::SumcheckInfo;
use crate::SumcheckInstance;
use crate::SumcheckPIOP;
use crate::SumcheckPureProof;
use crate::SumcheckPureProverState;
use crate::SumcheckPureSubclaim;

pub struct RowPermPIOP<F: Field> {
    _marker: std::marker::PhantomData<F>,
}

pub struct RowPermInstance<F: Field> {
    pub log_num_cols: usize,
    pub log_num_rows: usize,
    pub point_rx: Vec<F>,
    pub point_ry: Vec<F>,
    pub input_ry: Rc<DenseMultilinearExtension<F>>,
    pub perm_rx: Rc<DenseMultilinearExtension<F>>,
    pub output_ry_rx: F,
    pub perm: Vec<usize>,
}

#[derive(Serialize)]
pub struct RowPermInfo<F: Field> {
    pub log_num_cols: usize,
    pub log_num_rows: usize,
    pub perm: Vec<usize>,
    _marker: std::marker::PhantomData<F>,
}

pub struct RowPermProof<F: Field> {
    pub poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<F>,
    pub perm_at_r_rx: F,
    pub input_at_ry_r: F,
}

pub struct RowPermProverState<F: Field> {
    pub randomness: Vec<F>,
}

pub struct RowPermVerifierSubclaim<F: Field> {
    pub randomness: Vec<F>,
}

impl<F: Field> RowPermInstance<F> {
    pub fn random<R: rand::Rng + rand::CryptoRng>(
        rng: &mut R,
        log_num_rows: usize,
        log_num_cols: usize,
    ) -> Self {
        let random_trace: RowPermTraceMLE<F> =
            RowPermTrace::random(rng, log_num_rows, log_num_cols).into();
        let point_rx: Vec<F> = (0..log_num_rows).map(|_| F::random(rng)).collect();
        let point_ry: Vec<F> = (0..log_num_cols).map(|_| F::random(rng)).collect();
        let perm_rx = compute_permutation_at_point::<F>(
            log_num_rows,
            &random_trace.permutation_info,
            &point_rx,
        );
        let input_ry = random_trace.input.fix_variables(&point_ry);
        let point_ry_rx = point_ry
            .iter()
            .chain(point_rx.iter())
            .cloned()
            .collect::<Vec<F>>();
        let output_ry_rx = random_trace.output.evaluate(&point_ry_rx);

        Self {
            log_num_cols,
            log_num_rows,
            point_rx,
            point_ry,
            input_ry: Rc::new(input_ry),
            perm_rx: Rc::new(perm_rx),
            output_ry_rx,
            perm: random_trace.permutation_info,
        }
    }
}

impl<F: Field + Serialize> SumcheckInstance<F> for RowPermInstance<F> {
    type Info = RowPermInfo<F>;
    fn info(&self) -> Self::Info {
        RowPermInfo {
            log_num_cols: self.log_num_cols,
            log_num_rows: self.log_num_rows,
            perm: self.perm.clone(),
            _marker: std::marker::PhantomData,
        }
    }
    fn num_vars(&self) -> usize {
        self.log_num_rows
    }
}

impl<F: Field> SumcheckInfo<F> for RowPermInfo<F> {
    fn num_vars(&self) -> usize {
        self.log_num_rows
    }
    fn num_sumchecks(&self) -> usize {
        1
    }
    fn sumcheck_num_vars(&self) -> usize {
        self.log_num_rows
    }
}

impl<F: Field> SumcheckPureProof<F> for RowPermProof<F> {
    fn from_sumcheck(sumcheck_claim: &crate::SumcheckClaim<F>, proof: sumcheck::Proof<F>) -> Self {
        RowPermProof {
            poly_info: sumcheck_claim.poly.info(),
            sumcheck_proof: proof,
            perm_at_r_rx: F::zero(),
            input_at_ry_r: F::zero(),
        }
    }
    fn get_poly_info(&self) -> &PolynomialInfo {
        &self.poly_info
    }
    fn get_sumcheck_proof(&self) -> &Proof<F> {
        &self.sumcheck_proof
    }
}

impl<F: Field> SumcheckPureProverState<F> for RowPermProverState<F> {
    fn from_sumcheck(
        sumcheck_prover_state: sumcheck::prover::ProverState<F>,
        claim: crate::SumcheckClaim<F>,
    ) -> Self {
        RowPermProverState {
            randomness: sumcheck_prover_state.randomness.clone(),
        }
    }
}

impl<F: Field> SumcheckPureSubclaim<F> for RowPermVerifierSubclaim<F> {
    fn from_sumcheck(sumcheck_subclaim: sumcheck::verifier::SubClaim<F>) -> Self {
        RowPermVerifierSubclaim {
            randomness: sumcheck_subclaim.point.clone(),
        }
    }
}

impl<F: Field + Serialize> SumcheckPIOP<F> for RowPermPIOP<F> {
    type Instance = RowPermInstance<F>;
    type Info = RowPermInfo<F>;
    type Proof = RowPermProof<F>;
    type ProverState = RowPermProverState<F>;
    type VerifierSubclaim = RowPermVerifierSubclaim<F>;

    fn prover(
        trans: &mut helper::Transcript<F>,
        instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState) {
        let (mut proof, state) = Self::prover_without_evals(trans, instance);
        proof.perm_at_r_rx = instance.perm_rx.evaluate(&state.randomness);
        proof.input_at_ry_r = instance.input_ry.evaluate(&state.randomness);
        (proof, state)
    }

    fn prover_add_sumcheck(
        instance: &Self::Instance,
        claim: &mut crate::SumcheckClaim<F>,
        randomness: &[F],
        _lagrange_kernel: Option<&crate::LagrangeKernel<F>>,
    ) -> Option<Self::ProverState> {
        assert_eq!(randomness.len(), 1);
        claim.poly.add_product(
            vec![Rc::clone(&instance.perm_rx), Rc::clone(&instance.input_ry)],
            randomness[0],
        );
        claim.sum += instance.output_ry_rx * randomness[0];
        None
    }

    fn verifier_compute_subclaim(
        info: &Self::Info,
        proof: &Self::Proof,
        subclaim: &mut sumcheck::verifier::SubClaim<F>,
        randomness: &[F],
        _kernel_at_r: Option<F>,
    ) {
        assert!(randomness.len() == 1);
        subclaim.expected_evaluations -= proof.perm_at_r_rx * proof.input_at_ry_r * randomness[0];
    }
}

pub fn compute_inverse_permutation(perm: &Vec<usize>) -> Vec<usize> {
    let n = perm.len();
    let mut inv_perm = vec![0; n];
    for (i, &p) in perm.iter().enumerate() {
        inv_perm[p] = i;
    }
    inv_perm
}

pub fn perform_permutation<F: Zero + Clone + Copy>(
    dim: usize,
    perm: &Vec<usize>,
    input: &[F],
) -> Vec<F> {
    assert_eq!(1 << dim, perm.len());
    let mut output_evals = vec![F::zero(); 1 << dim];
    for x in 0..1 << dim {
        let y = perm[x];
        output_evals[x] = input[y];
    }
    output_evals
}

// Compute P(y, r) for all y in the hypercube, where r is a random point in F^n
pub fn compute_permutation_at_point<F: Field>(
    dim: usize,
    perm: &Vec<usize>,
    point: &[F],
) -> DenseMultilinearExtension<F> {
    assert_eq!(1 << dim, perm.len());
    let eq_mle = gen_identity_evaluations(point);
    let inverse_perm = compute_inverse_permutation(perm);
    DenseMultilinearExtension::from_evaluations_vec(
        dim,
        perform_permutation(dim, &inverse_perm, eq_mle.as_slice()),
    )
}

// Compute P(y, r) for all y in the hypercube, where r is a random point in F^n
pub fn compute_permutation_at_point_w_sign<F: Field>(
    dim: usize,
    perm: &Vec<usize>,
    sign: &Vec<usize>,
    point: &[F],
) -> DenseMultilinearExtension<F> {
    assert_eq!(1 << dim, perm.len());
    let eq_mle = gen_identity_evaluations(point);
    let inverse_perm = compute_inverse_permutation(perm);
    let sign_permutated = perform_permutation(dim, &inverse_perm, sign);
    let mut eq_mle_permutated = perform_permutation(dim, &inverse_perm, eq_mle.as_slice());
    eq_mle_permutated
        .iter_mut()
        .zip(sign_permutated.iter())
        .for_each(|(v, s)| {
            if *s == 0 {
                *v = -*v;
            }
        });
    DenseMultilinearExtension::from_evaluations_vec(dim, eq_mle_permutated)
}

#[cfg(test)]
mod test {
    use super::*;
    use algebra::{DenseMultilinearExtension, Field, MultilinearExtension, derive::Field};
    use helper::Transcript;
    use rayon::vec;

    #[derive(Field)]
    #[modulus = 132120577]
    pub struct Fp32(u32);
    // field type
    type FF = Fp32;

    // P[X][Y] denotes the permutation matrix entry at row i and column j
    // P(y, x) = P(y_0, y_1, ..., y_{n-1}, x_0, x_1, ..., x_{n-1}) = P[X][Y]
    // where X = \sum 2^i x_i, Y = \sum 2^i y_i.
    //
    // Permutation phi(X) = Y is represented as P[X][Y] = 1 if Y = phi(X), else 0.
    // Hence, each row of P has exactly one 1.
    fn generate_permutation_matrix<F: Field>(
        dim: usize,
        perm: &Vec<usize>,
    ) -> DenseMultilinearExtension<F> {
        assert_eq!(1 << dim, perm.len());
        let mut perm_matrix = vec![F::zero(); (1 << dim) * (1 << dim)];

        for x in 0..1 << dim {
            for y in 0..1 << dim {
                let idx = x * (1 << dim) + y;
                if perm[x] == y {
                    perm_matrix[idx] = F::one();
                }
            }
        }
        DenseMultilinearExtension::from_evaluations_vec(dim * 2, perm_matrix)
    }

    fn generate_permutation_matrix_w_sign<F: Field>(
        dim: usize,
        perm: &Vec<usize>,
        sig: &Vec<bool>,
    ) -> DenseMultilinearExtension<F> {
        assert_eq!(1 << dim, perm.len());
        let mut perm_matrix = vec![F::zero(); (1 << dim) * (1 << dim)];

        for x in 0..1 << dim {
            for y in 0..1 << dim {
                let idx = x * (1 << dim) + y;
                if perm[x] == y {
                    perm_matrix[idx] = if sig[x] == true { F::one() } else { -F::one() };
                }
            }
        }
        DenseMultilinearExtension::from_evaluations_vec(dim * 2, perm_matrix)
    }

    #[test]
    fn test_permutation_at_point() {
        let dim = 2 as usize;
        let perm = vec![1, 2, 3, 0]; // permutation on 4 elements
        let sig = vec![1, 1, 1, 1];

        let perm_matrix_mle = generate_permutation_matrix::<FF>(dim, &perm);
        let rng = &mut rand::rng();
        let point: Vec<FF> = (0..dim).map(|_| FF::random(rng)).collect();

        let perm_mle = perm_matrix_mle.fix_variables_back(&point);
        let computed_perm_mle = compute_permutation_at_point::<FF>(dim, &perm, &point);

        assert_eq!(perm_mle, computed_perm_mle);
    }

    #[test]
    fn test_permutation_matrix() {
        let mut rng = rand::rng();
        let log_num_rows = 2;
        let log_num_cols = 2;

        let random_trace = RowPermTrace::<FF>::random(&mut rng, log_num_rows, log_num_cols);
        let permutation_matrix =
            generate_permutation_matrix::<FF>(log_num_rows, &random_trace.permutation_info);

        let mut product = vec![FF::zero(); 1 << (log_num_rows + log_num_cols)];
        for x in 0..(1 << log_num_rows) {
            for y in 0..(1 << log_num_rows) {
                let prod_idx = y + (x << log_num_rows);
                for z in 0..(1 << log_num_rows) {
                    let perm_idx = z + (x << log_num_rows);
                    let input_idx = y + (z << log_num_rows);
                    product[prod_idx] +=
                        permutation_matrix.evaluations[perm_idx] * random_trace.input[input_idx];
                }
            }
        }
        assert_eq!(product, random_trace.output);
    }

    #[test]
    fn test_row_perm_piop() {
        let mut rng = rand::rng();
        let log_num_rows = 2;
        let log_num_cols = 2;

        let instance = RowPermInstance::<FF>::random(&mut rng, log_num_rows, log_num_cols);
        let instance_info = instance.info();
        let mut prover_trans = Transcript::default();
        let (proof, _prover_state) = RowPermPIOP::<FF>::prover(&mut prover_trans, &instance);
        let mut verifier_trans = Transcript::default();
        let (res, _) = RowPermPIOP::<FF>::verifier(&mut verifier_trans, &instance_info, &proof);
        assert!(res);
    }
}
