//! Consider a sparse matrix C with dimensions M x N, where each row contains only
//! a single non-zero entry.
//!
//! Matrix indexing: C[Y][X] where X is the row index and Y is the column index.
//!
//! MLE indexing: `C(y, x) = C(y0, y1, ..., y\logN, x0, x1, ..., x\logM)` where
//! `Y = y0*2^0 + y1*2^1 + ... + yN*2^\logN` and `X = x0*2^0 + x1*2^1 + ...
//! + xM*2^\logM`. This is the little-endian representation.
//!
//! This instance is related to evaluating the sparse matrix on a random point
//! (r_y, r_x) as follows:
//! C(r_y, r_x) =
//!     \sum_k val(k) * eq(to-bits(col(k)), r_y) * eq(to-bits(row(k)), r_x).
//!     k iterates the hybercube of dimension logM.
//! where r_y is a random point on the y axis with logN variables, and r_x is a
//! random point on the x axis with logM variables. Here, col(k) and row(k)
//! represent the column and row indices of the k-th non-zero entry in the
//! sparse matrix.
//!
//! This equation is from Eq(8) in Lasso: https://eprint.iacr.org/2023/1216.pdf.
//!
//! Here, `row(k)` = k, so eq(to-bits(row(k)), r_x) = eq(k, r_x). The evaluation
//! can be simplied to:
//! C(r_y, r_x) =
//!    \sum_k val(k) * eq(to-bits(col(k)), r_y) * eq(k, r_x).
//!
//! To prove this, it contains a sumcheck on the above equation:
//! `C(r_y, r_x) = \sum_{k} val(k) * E[k] * eq(k,r_x)`
//!  along with an indexed lookup argument. The indexed lookup argument is defined
//! as follows:
//! - Table T[y] = eq(y, r_y) for y in [N]
//! - Input E[x] = eq(to-bits(col(x)), r_y) for x in [M]
//! - Index I[x] = col(x) for x in [M]
//! We don't prove the indexed lookup argument here. Instead, we prove it in the
//! snarks layer using the IndexedLogUpSnarks.
use algebra::{DenseMultilinearExtension, Field, PolynomialInfo};
use helper::utils::eval_identity_function;
use serde::Serialize;
use std::rc::Rc;
use sumcheck::{Proof, prover::ProverState, verifier::SubClaim};
use trace::lookup_trace::indexed_table::{IndexedLookupTrace, IndexedLookupTraceMLE};

use crate::{
    LagrangeKernel, SumcheckClaim, SumcheckInfo, SumcheckInstance, SumcheckPIOP, SumcheckPureProof,
    SumcheckPureProverState, SumcheckPureSubclaim,
};
/// We don't prove the indexed lookup argument here. Instead, we prove it in the
/// snarks layer using the IndexedLogUpSnarks.
pub struct SparseRowEvalPIOP<F: Field> {
    _marker: std::marker::PhantomData<F>,
}

pub struct SparseRowEvalInstance<F: Field> {
    // denoted by logM in the above description
    pub num_x_vars: usize,
    // denoted by logN in the above description
    pub num_y_vars: usize,
    pub point_ry: Vec<F>,
    pub point_rx: Vec<F>,
    /// sparse representation
    pub col: Rc<DenseMultilinearExtension<F>>,
    pub val: Rc<DenseMultilinearExtension<F>>,
    /// helper MLE E_ry(k) = eq(to-bits(col(k)), ry)
    pub eval_mle_ry: Rc<DenseMultilinearExtension<F>>,
    /// lagrange kernel eq(k, r_x)
    pub kernel_rx: LagrangeKernel<F>,
    pub eval: F,
}

#[derive(Serialize)]
pub struct SparseRowEvalInstanceInfo<F: Field> {
    pub num_x_vars: usize,
    #[serde(skip)]
    pub point_rx: Vec<F>,
    #[serde(skip)]
    pub point_ry: Vec<F>,
    pub eval: F,
}

#[derive(Serialize)]
pub struct SparseRowEvalProof<F: Field> {
    pub poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<F>,
    pub eval_mle_at_r: F,
    pub val_at_r: F,
}

pub struct SparseRowProverState<F: Field> {
    pub sumcheck_point_r: Vec<F>,
}

pub struct SparseRowVerifierSubclaim<F: Field> {
    pub sumcheck_point_r: Vec<F>,
}

impl<F: Field + Serialize> SumcheckInstance<F> for SparseRowEvalInstance<F> {
    type Info = SparseRowEvalInstanceInfo<F>;

    fn num_vars(&self) -> usize {
        self.num_x_vars
    }

    fn info(&self) -> Self::Info {
        SparseRowEvalInstanceInfo {
            num_x_vars: self.num_x_vars,
            point_rx: self.point_rx.clone(),
            point_ry: self.point_ry.clone(),
            eval: self.eval,
        }
    }
}

impl<F: Field> SumcheckInfo<F> for SparseRowEvalInstanceInfo<F> {
    fn num_vars(&self) -> usize {
        self.num_x_vars
    }

    fn num_sumchecks(&self) -> usize {
        1
    }

    fn sumcheck_num_vars(&self) -> usize {
        self.num_x_vars
    }
}

impl<F: Field> SparseRowEvalInstance<F> {
    pub fn random<R: rand::Rng + rand::CryptoRng>(
        rng: &mut R,
        num_x_vars: usize,
        num_y_vars: usize,
    ) -> Self {
        let point_ry = (0..num_y_vars).map(|_| F::random(rng)).collect::<Vec<F>>();
        let point_rx = (0..num_x_vars).map(|_| F::random(rng)).collect::<Vec<F>>();

        let num_nonzero = 1 << num_x_vars;
        let val = (0..num_nonzero).map(|_| F::random(rng)).collect::<Vec<F>>();
        let kernel_rx = LagrangeKernel::from_point(&point_rx);

        let lookup_trace = IndexedLookupTrace::<F>::random(rng, num_x_vars, num_y_vars);
        Self {
            num_x_vars,
            num_y_vars,
            point_ry,
            point_rx,
            col: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_x_vars,
                lookup_trace.index,
            )),
            val: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_x_vars, val,
            )),
            eval_mle_ry: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_x_vars,
                lookup_trace.input,
            )),
            kernel_rx,
            eval: F::zero(),
        }
    }

    /// extract the indexed lookup trace from the instance
    pub fn extract_indexed_lookup_trace(&self) -> IndexedLookupTraceMLE<F> {
        IndexedLookupTraceMLE {
            num_input_vars: self.num_x_vars,
            num_table_vars: self.num_y_vars,
            index: Rc::clone(&self.col),
            input: Rc::clone(&self.eval_mle_ry),
            table_point: self.point_ry.clone(),
            table: self.kernel_rx.eq_at_point.clone(),
        }
    }

    pub fn add_into_sumcheck(&self, claim: &mut SumcheckClaim<F>, random_lambda: F) {
        claim.poly.add_product(
            vec![
                Rc::clone(&self.val),
                Rc::clone(&self.eval_mle_ry),
                Rc::clone(&self.kernel_rx.eq_at_point),
            ],
            random_lambda,
        );
    }
}

impl<F: Field> SumcheckPureProof<F> for SparseRowEvalProof<F> {
    fn from_sumcheck(sumcheck_claim: &SumcheckClaim<F>, proof: Proof<F>) -> SparseRowEvalProof<F> {
        SparseRowEvalProof {
            poly_info: sumcheck_claim.poly.info(),
            sumcheck_proof: proof,
            eval_mle_at_r: F::zero(),
            val_at_r: F::zero(),
        }
    }

    fn get_poly_info(&self) -> &PolynomialInfo {
        &self.poly_info
    }

    fn get_sumcheck_proof(&self) -> &Proof<F> {
        &self.sumcheck_proof
    }
}

impl<F: Field> SumcheckPureProverState<F> for SparseRowProverState<F> {
    fn from_sumcheck(sumcheck_prover_state: ProverState<F>) -> Self {
        SparseRowProverState {
            sumcheck_point_r: sumcheck_prover_state.randomness.clone(),
        }
    }
}

impl<F: Field> SumcheckPureSubclaim<F> for SparseRowVerifierSubclaim<F> {
    fn from_sumcheck(sumcheck_subclaim: SubClaim<F>) -> Self {
        SparseRowVerifierSubclaim {
            sumcheck_point_r: sumcheck_subclaim.point.clone(),
        }
    }
}

impl<F: Field + Serialize> SumcheckPIOP<F> for SparseRowEvalPIOP<F> {
    type Instance = SparseRowEvalInstance<F>;
    type Info = SparseRowEvalInstanceInfo<F>;
    type Proof = SparseRowEvalProof<F>;
    type ProverState = SparseRowProverState<F>;
    type VerifierSubclaim = SparseRowVerifierSubclaim<F>;

    fn prover(
        trans: &mut helper::Transcript<F>,
        instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState) {
        let (mut proof, state) = Self::prover_without_evals(trans, instance);

        proof.eval_mle_at_r = instance.eval_mle_ry.evaluate(&state.sumcheck_point_r);
        proof.val_at_r = instance.val.evaluate(&state.sumcheck_point_r);
        (proof, state)
    }

    fn prover_batch_sumcheck(
        instance: &Self::Instance,
        claim: &mut SumcheckClaim<F>,
        randomness: &[F],
        _lagrange_kernel: Option<&LagrangeKernel<F>>,
    ) -> Option<Self::ProverState> {
        assert_eq!(randomness.len(), 1);
        instance.add_into_sumcheck(claim, randomness[0]);
        None
    }

    fn verifier_compute_subclaim(
        info: &Self::Info,
        proof: &Self::Proof,
        subclaim: &mut SubClaim<F>,
        randomness: &[F],
        _kernel_at_r: Option<F>,
    ) {
        assert_eq!(randomness.len(), 1);
        let kernel_at_r = eval_identity_function(&info.point_rx, &subclaim.point);
        subclaim.expected_evaluations -=
            proof.eval_mle_at_r * proof.val_at_r * kernel_at_r * randomness[0];
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use algebra::derive::{Field, Prime};
    use helper::Transcript;

    #[derive(Field, Prime)]
    #[modulus = 132120577]
    pub struct Fp32(u32);
    // field type
    type FF = Fp32;

    #[test]
    fn test_sparse_row_eval_piop() {
        let mut rng = rand::rng();
        let num_x_vars = 4;
        let num_y_vars = 4;

        let instance = SparseRowEvalInstance::<FF>::random(&mut rng, num_x_vars, num_y_vars);
        let instance_info = instance.info();
        let mut prover_trans = Transcript::<FF>::new();
        let (proof, _) = SparseRowEvalPIOP::prover(&mut prover_trans, &instance);
        let mut verifier_trans = Transcript::<FF>::new();
        let (res, _) = SparseRowEvalPIOP::verifier(&mut verifier_trans, &instance_info, &proof);
        assert!(res);
    }
}
