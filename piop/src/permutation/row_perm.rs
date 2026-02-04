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
use helper::utils::eval_identity_function;
use serde::Serialize;

use std::rc::Rc;
use sumcheck::Proof;
use sumcheck::verifier::SubClaim;
use trace::BlindRotationTraceMLE;
use trace::basic_ops::{RowPermTrace, RowPermTraceMLE};
use trace::blind_rotation_trace::BlindRotationTraceEval;
use trace::lookup_trace::indexed_table::IndexedLookupTraceMLE;

use crate::{
    BatchedSumcheckPIOP, SumcheckInfo, SumcheckInstance, SumcheckPIOP, SumcheckProverState,
    SumcheckPureBatchedProof, SumcheckPureProof, SumcheckSubclaim,
};

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
}

#[derive(Serialize)]
pub struct RowPermInfo<F: Field> {
    pub log_num_cols: usize,
    pub log_num_rows: usize,
    _marker: std::marker::PhantomData<F>,
}

#[derive(Serialize)]
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
        trace: &BlindRotationTraceMLE<F>,
        trace_eval: &BlindRotationTraceEval<F>,
        indexed_permutation: &IndexedLookupTraceMLE<F>,
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

        let input_ry_0 = Rc::new(trace.acc_trace.input_acc.poly.0.fix_variables(point_ry));
        let input_ry_1 = Rc::new(trace.acc_trace.input_acc.poly.1.fix_variables(point_ry));

        let instance_0 = Self {
            log_num_cols,
            log_num_rows,
            point_rx: point_rx.to_vec(),
            point_ry: point_ry.to_vec(),
            input_ry: input_ry_0,
            perm_rx: Rc::clone(&indexed_permutation.input),
            output_ry_rx: input_acc_permuted_0,
        };
        let instance_1 = Self {
            log_num_cols,
            log_num_rows,
            point_rx: point_rx.to_vec(),
            point_ry: point_ry.to_vec(),
            input_ry: input_ry_1,
            perm_rx: Rc::clone(&indexed_permutation.input),
            output_ry_rx: input_acc_permuted_1,
        };
        vec![instance_0, instance_1]
    }

    pub fn from_permutation_trace(
        trace: &RowPermTraceMLE<F>,
        point_rx: &[F],
        point_ry: &[F],
    ) -> Self {
        let input_ry = trace.input.fix_variables(point_ry);

        let point_ry_rx = point_ry
            .iter()
            .chain(point_rx.iter())
            .cloned()
            .collect::<Vec<F>>();
        let output_ry_rx = trace.output.evaluate(&point_ry_rx);

        let permutation_rx = match trace.permutation_info.signed {
            None => {
                let lookup_trace = trace
                    .permutation_info
                    .extract_indexed_lookup_trace(point_rx);
                lookup_trace.input
            }
            Some(_) => {
                let mle = trace.permutation_info.fixed_variable(point_rx);
                Rc::new(mle)
            }
        };

        Self {
            log_num_cols: trace.log_num_cols,
            log_num_rows: trace.log_num_rows,
            point_rx: point_rx.to_vec(),
            point_ry: point_ry.to_vec(),
            input_ry: Rc::new(input_ry),
            perm_rx: Rc::clone(&permutation_rx),
            output_ry_rx,
        }
    }

    pub fn random_rotation_left<R: rand::Rng + rand::CryptoRng>(
        rng: &mut R,
        log_num_rows: usize,
        log_num_cols: usize,
    ) -> Self {
        let random_trace: RowPermTraceMLE<F> =
            RowPermTrace::random_rotation_left(rng, log_num_rows, log_num_cols).into();

        let point_rx: Vec<F> = (0..log_num_rows).map(|_| F::random(rng)).collect();
        let point_ry: Vec<F> = (0..log_num_cols).map(|_| F::random(rng)).collect();
        RowPermInstance::from_permutation_trace(&random_trace, &point_rx, &point_ry)
    }

    pub fn random_ks_permutation<R: rand::Rng + rand::CryptoRng>(
        rng: &mut R,
        log_num_rows: usize,
        log_num_cols: usize,
        log_blk_size: usize,
    ) -> Self {
        let random_trace: RowPermTraceMLE<F> =
            RowPermTrace::random_ks_permutation(rng, log_num_rows, log_num_cols, log_blk_size)
                .into();
        let point_rx: Vec<F> = (0..log_num_rows).map(|_| F::random(rng)).collect();
        let point_ry: Vec<F> = (0..log_num_cols).map(|_| F::random(rng)).collect();
        RowPermInstance::from_permutation_trace(&random_trace, &point_rx, &point_ry)
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
    type ProverState = SumcheckProverState<F>;

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
        randomness: &[Vec<F>],
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

impl<F: Field + Serialize> SumcheckPIOP<F> for RowPermPIOP<F> {
    type Instance = RowPermInstance<F>;
    type Info = RowPermInfo<F>;
    type Proof = RowPermProof<F>;
    type ProverState = SumcheckProverState<F>;
    type VerifierSubclaim = SumcheckSubclaim<F>;

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
    type BatchedProverState = SumcheckProverState<F>;
    type BatchedVerifierSubclaim = SumcheckSubclaim<F>;
}

#[cfg(test)]
mod test {

    use super::*;
    use algebra::derive::Field;
    use helper::Transcript;

    #[derive(Field)]
    #[modulus = 132120577]
    pub struct Fp32(u32);
    // field type
    type FF = Fp32;

    #[test]
    fn test_row_perm_piop() {
        let mut rng = rand::rng();
        let log_num_rows = 2;
        let log_num_cols = 2;

        let instance =
            RowPermInstance::<FF>::random_rotation_left(&mut rng, log_num_rows, log_num_cols);
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
            .map(|_| {
                RowPermInstance::<FF>::random_rotation_left(&mut rng, log_num_rows, log_num_cols)
            })
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

    #[test]
    fn test_row_perm_signed_piop() {
        let mut rng = rand::rng();
        let log_num_rows = 10;
        let log_num_cols = 0;
        let log_blk_size = 9;

        let instance = RowPermInstance::<FF>::random_ks_permutation(
            &mut rng,
            log_num_rows,
            log_num_cols,
            log_blk_size,
        );
        let instance_info = instance.info();
        let mut prover_trans = Transcript::default();
        let (proof, _prover_state) = RowPermPIOP::<FF>::prover(&mut prover_trans, &instance);
        let mut verifier_trans = Transcript::default();
        let (res, _) = RowPermPIOP::<FF>::verifier(&mut verifier_trans, &instance_info, &proof);
        assert!(res);
    }
}
