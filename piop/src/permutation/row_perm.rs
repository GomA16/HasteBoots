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
use arc_swap::strategy;
use bincode::config;
use helper::utils::eval_identity_function;
use helper::utils::gen_identity_evaluations;
use num_traits::{Zero, ops::inv, sign};
use rand::rand_core::le;
use rand::random;
use rayon::vec;
use serde::Serialize;
use sha2::digest::typenum::type_operators;
use std::collections::HashMap;
use std::os::macos::raw::stat;
use std::{mem::Discriminant, rc::Rc};
use sumcheck::Proof;
use sumcheck::verifier::SubClaim;
use trace::PBSTraceMLE;
use trace::pbs_trace::PBSTraceEval;
use trace::row_perm_trace::RowPermTrace;
use trace::row_perm_trace::RowPermTraceMLE;

use crate::BatchedSumcheckPIOP;
use crate::SumcheckInfo;
use crate::SumcheckInstance;
use crate::SumcheckPIOP;
use crate::SumcheckPureBatchedProof;
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
    // pub perm: Vec<usize>,
}

#[derive(Serialize)]
pub struct RowPermInfo<F: Field> {
    pub log_num_cols: usize,
    pub log_num_rows: usize,
    // pub perm: Vec<usize>,
    _marker: std::marker::PhantomData<F>,
}

pub struct RowPermProof<F: Field> {
    pub poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<F>,
    pub perm_at_r_rx: F,
    pub input_at_ry_r: F,
}

#[derive(Serialize)]
pub struct BatchedRowPermProof<F: Field> {
    pub poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<F>,
    pub perm_at_r_rx: Vec<F>,
    pub input_at_ry_r: Vec<F>,
}

pub struct RowPermProverState<F: Field> {
    pub randomness: Vec<F>,
    pub flattened_mle_evals: Vec<F>,
    raw_pointers_lookup_table: HashMap<*const DenseMultilinearExtension<F>, usize>,
}

pub struct RowPermVerifierSubclaim<F: Field> {
    pub randomness: Vec<F>,
}

impl<F: Field> RowPermInstance<F> {
    pub fn add_into_sumcheck(&self, claim: &mut crate::SumcheckClaim<F>, random_lambda: F) {
        claim.poly.add_product(
            vec![Rc::clone(&self.perm_rx), Rc::clone(&self.input_ry)],
            random_lambda,
        );
        claim.sum += self.output_ry_rx * random_lambda;
    }

    // Cheetsheet for AccInteration:
    // 1. initial_acc (a) is the first row of input_acc (A) => a(ry) = A(0..0, ry)
    // 2. final_acc (d) is the last row of output_acc (D) => d(ry) = D(1..1, ry)
    // 3. i-th row of input_acc is (i-1)-th row of output_acc
    //
    // The matrix permutation relation:
    // input_acc_permuted(denoted by A') = output_acc(D) + Zero matrix (Z)
    // => A'(rx, ry) = D(rx, ry) + Z(rx, ry)
    // where Zero matrix is a matrix where only the last row is inital_acc - final_acc
    // => Z(ry, rx) = eq(rx, 1...1) * row(ry) = eq(rx, 1...1) * (a(ry) - d(ry))
    // input_acc_permuted = permutation_matrix * input_acc
    // => A'(rx, ry) = sum_{k} P(rx, k) * A(k, ry)
    pub fn from_subclaim(
        trace: &PBSTraceMLE<F>,
        trace_eval: &PBSTraceEval<F>,
        permutation: &Rc<DenseMultilinearExtension<F>>,
        point_rx: &[F],
        point_ry: &[F],
    ) -> Vec<Self> {
        let log_num_rows = trace.log_num_round;
        let log_num_cols = trace.log_coeff_count;
        // output_acc: D(rx, ry)
        let output_acc_0 = trace_eval.output_acc.poly.0 + trace_eval.hadamard_trace.sum_prod.poly.0;
        let output_acc_1 = trace_eval.output_acc.poly.1 + trace_eval.hadamard_trace.sum_prod.poly.1;
        // A'(rx, ry) = D(rx, ry) + Z(rx, ry)
        // Z(ry, rx) = eq(rx, 1...1) * row(ry) = eq(rx, 1...1) * (a(ry) - d(ry))
        let eq_rx_ones = eval_identity_function(&vec![F::one(); log_num_rows], point_rx);
        let input_acc_permuted_0 = output_acc_0
            + eq_rx_ones
                * (trace_eval.acc_trace.initial_acc.poly.0 - trace_eval.acc_trace.final_acc.poly.0);
        let input_acc_permuted_1 = output_acc_1
            + eq_rx_ones
                * (trace_eval.acc_trace.initial_acc.poly.1 - trace_eval.acc_trace.final_acc.poly.1);

        let input_ry_0 = Rc::new(trace.acc_trace.input_acc.poly.0.fix_variables(&point_ry));
        let input_ry_1 = Rc::new(trace.acc_trace.input_acc.poly.1.fix_variables(&point_ry));

        let instance_0 = Self {
            log_num_cols,
            log_num_rows,
            point_rx: point_rx.to_vec(),
            point_ry: point_ry.to_vec(),
            input_ry: input_ry_0,
            perm_rx: Rc::clone(permutation),
            output_ry_rx: input_acc_permuted_0,
        };
        let instance_1 = Self {
            log_num_cols,
            log_num_rows,
            point_rx: point_rx.to_vec(),
            point_ry: point_ry.to_vec(),
            input_ry: input_ry_1,
            perm_rx: Rc::clone(permutation),
            output_ry_rx: input_acc_permuted_1,
        };
        vec![instance_0, instance_1]
    }

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
            // perm: random_trace.permutation_info,
        }
    }
}

impl<F: Field + Serialize> SumcheckInstance<F> for RowPermInstance<F> {
    type Info = RowPermInfo<F>;
    fn info(&self) -> Self::Info {
        RowPermInfo {
            log_num_cols: self.log_num_cols,
            log_num_rows: self.log_num_rows,
            // perm: self.perm.clone(),
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

impl<F: Field> SumcheckPureProof<F> for BatchedRowPermProof<F> {
    fn from_sumcheck(sumcheck_claim: &crate::SumcheckClaim<F>, proof: sumcheck::Proof<F>) -> Self {
        BatchedRowPermProof {
            poly_info: sumcheck_claim.poly.info(),
            sumcheck_proof: proof,
            perm_at_r_rx: vec![],
            input_at_ry_r: vec![],
        }
    }
    fn get_poly_info(&self) -> &PolynomialInfo {
        &self.poly_info
    }
    fn get_sumcheck_proof(&self) -> &Proof<F> {
        &self.sumcheck_proof
    }
}
impl<F: Field + Serialize> SumcheckPureBatchedProof<F> for BatchedRowPermProof<F> {
    type Info = RowPermInfo<F>;
    type Instance = RowPermInstance<F>;
    type ProverState = RowPermProverState<F>;

    fn append_evaluations(
        &mut self,
        instances: &[Self::Instance],
        prover_state: &Self::ProverState,
    ) {
        let lookup = |m: &Rc<DenseMultilinearExtension<F>>| {
            let m_ptr: *const DenseMultilinearExtension<F> = Rc::as_ptr(m);
            let index = prover_state.raw_pointers_lookup_table.get(&m_ptr).unwrap();
            prover_state.flattened_mle_evals[*index]
        };
        self.input_at_ry_r = instances
            .iter()
            .map(|instance| lookup(&instance.input_ry))
            .collect();
        self.perm_at_r_rx = instances
            .iter()
            .map(|instance| lookup(&instance.perm_rx))
            .collect();
    }

    fn compute_subclaim(
        &self,
        infos: &[Self::Info],
        subclaim: &mut SubClaim<F>,
        randomness: &Vec<Vec<F>>,
        _kernel_at_r: Option<F>,
    ) {
        assert_eq!(randomness.len(), infos.len());
        for (r, &input_at_ry_r, &perm_at_r_rx) in itertools::izip!(
            randomness.iter(),
            self.input_at_ry_r.iter(),
            self.perm_at_r_rx.iter()
        ) {
            subclaim.expected_evaluations -= perm_at_r_rx * input_at_ry_r * r[0];
        }
    }
}

impl<F: Field> SumcheckPureProverState<F> for RowPermProverState<F> {
    fn from_sumcheck(
        sumcheck_prover_state: sumcheck::prover::ProverState<F>,
        claim: crate::SumcheckClaim<F>,
    ) -> Self {
        let flattened_mle_evals = sumcheck_prover_state.fast_evaluate();
        Self {
            randomness: sumcheck_prover_state.randomness,
            flattened_mle_evals,
            raw_pointers_lookup_table: claim.poly.raw_pointers_lookup_table,
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
        instance.add_into_sumcheck(claim, randomness[0]);
        None
    }

    fn verifier_compute_subclaim(
        _info: &Self::Info,
        proof: &Self::Proof,
        subclaim: &mut sumcheck::verifier::SubClaim<F>,
        randomness: &[F],
        _kernel_at_r: Option<F>,
    ) {
        assert!(randomness.len() == 1);
        subclaim.expected_evaluations -= proof.perm_at_r_rx * proof.input_at_ry_r * randomness[0];
    }
}

impl<F: Field + Serialize> BatchedSumcheckPIOP<F> for RowPermPIOP<F> {
    type BatchedProof = BatchedRowPermProof<F>;
    type BatchedProverState = RowPermProverState<F>;
    type BatchedVerifierSubclaim = RowPermVerifierSubclaim<F>;
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

    #[test]
    fn test_row_perm_piop_batched() {
        let mut rng = rand::rng();
        let log_num_rows = 2;
        let log_num_cols = 2;
        let num_instances = 3;

        let instances = (0..num_instances)
            .map(|_| RowPermInstance::<FF>::random(&mut rng, log_num_rows, log_num_cols))
            .collect::<Vec<_>>();
        let infos = instances
            .iter()
            .map(|instance| instance.info())
            .collect::<Vec<_>>();

        let mut prover_trans = Transcript::default();
        let (proof, _prover_state) = RowPermPIOP::<FF>::prover_batch(&mut prover_trans, &instances);
        let mut verifier_trans = Transcript::default();
        let (res, _) = RowPermPIOP::<FF>::verifier_batch(&mut verifier_trans, &infos, &proof);
        assert!(res);
    }
}
