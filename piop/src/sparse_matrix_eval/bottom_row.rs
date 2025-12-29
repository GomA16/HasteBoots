//! This matrix is all zeros but the bottom row.
//! This is useful fo building relation on complicated iteration while
//! fully exploiting the repeating structure of the iteration.

use std::rc::Rc;

use algebra::{DenseMultilinearExtension, Field, PolynomialInfo};
use helper::utils::eval_identity_function;
use rand::rand_core::le;
use serde::Serialize;
use sumcheck::Proof;
use trace::acc_trace::AccIterationTraceMLE;

use crate::{
    LagrangeKernel, SumcheckClaim, SumcheckInfo, SumcheckInstance, SumcheckPIOP, SumcheckPureProof,
    SumcheckPureProverState, SumcheckPureSubclaim,
};

pub struct BottomRowEvalPIOP<F: Field> {
    _marker: std::marker::PhantomData<F>,
}

// Z(r_y,r_x) = \sum_{y} eq(r_y, y) * eq(r_x, 1111) * row(y)
pub struct BottomRowEvalInstance<F: Field> {
    pub log_num_rows: usize,
    pub log_num_cols: usize,
    pub row: Rc<DenseMultilinearExtension<F>>,
    pub point_rx: Vec<F>,
    pub kernel_ry: LagrangeKernel<F>,
    pub eval_ry_rx: F,
}

#[derive(Serialize)]
pub struct BottomRowEvalinfo<F: Field> {
    pub log_num_rows: usize,
    pub log_num_cols: usize,
    pub point_rx: Vec<F>,
    pub point_ry: Vec<F>,
}

#[derive(Serialize)]
pub struct BottomRowEvalProof<F: Field> {
    pub poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<F>,
    pub row_at_r: F,
}

pub struct BottomRowEvalProverState<F: Field> {
    pub randomness: Vec<F>,
}

pub struct BottomRowVerifierSubclaim<F: Field> {
    pub randomness: Vec<F>,
}

impl<F: Field> SumcheckInfo<F> for BottomRowEvalinfo<F> {
    fn num_vars(&self) -> usize {
        self.log_num_cols
    }

    fn num_sumchecks(&self) -> usize {
        1
    }

    fn sumcheck_num_vars(&self) -> usize {
        self.log_num_cols
    }
}

impl<F: Field + Serialize> SumcheckInstance<F> for BottomRowEvalInstance<F> {
    type Info = BottomRowEvalinfo<F>;
    fn info(&self) -> Self::Info {
        BottomRowEvalinfo {
            log_num_rows: self.log_num_rows,
            log_num_cols: self.log_num_cols,
            point_rx: self.point_rx.clone(),
            point_ry: self.kernel_ry.point.clone(),
        }
    }
    fn num_vars(&self) -> usize {
        self.log_num_cols
    }
}

impl<F: Field> BottomRowEvalInstance<F> {
    pub fn random<R: rand::Rng + rand::CryptoRng>(
        rng: &mut R,
        log_num_rows: usize,
        log_num_cols: usize,
    ) -> Self {
        let num_rows = 1 << log_num_rows;
        let row_size = 1 << log_num_cols;
        let row = (0..row_size).map(|_| F::random(rng)).collect::<Vec<F>>();
        let point_rx = (0..log_num_rows)
            .map(|_| F::random(rng))
            .collect::<Vec<F>>();
        let point_ry = (0..log_num_cols)
            .map(|_| F::random(rng))
            .collect::<Vec<F>>();
        let kernel_ry = LagrangeKernel::from_point(&point_ry);
        let mut matrix = vec![F::zero(); num_rows * row_size];
        matrix[(num_rows - 1) * row_size..].copy_from_slice(&row);

        let point = point_ry
            .iter()
            .chain(point_rx.iter())
            .cloned()
            .collect::<Vec<F>>();
        let eval_ry_rx =
            DenseMultilinearExtension::from_evaluations_vec(log_num_rows + log_num_cols, matrix)
                .evaluate(&point);

        let row = DenseMultilinearExtension::from_evaluations_vec(log_num_cols, row);
        let coeff = eval_identity_function(&vec![F::one(); log_num_rows], &point_rx);
        assert_eq!(eval_ry_rx, coeff * row.evaluate(&point_ry));

        Self {
            log_num_rows,
            log_num_cols,
            row: Rc::new(row),
            point_rx,
            kernel_ry,
            eval_ry_rx,
        }
    }

    pub fn add_into_sumcheck(&self, claim: &mut SumcheckClaim<F>, random_lambda: F) {
        let coeff = eval_identity_function(&vec![F::one(); self.log_num_rows], &self.point_rx);
        claim.poly.add_product(
            [Rc::clone(&self.kernel_ry.eq_at_point), Rc::clone(&self.row)],
            coeff * random_lambda,
        );
        claim.sum += self.eval_ry_rx * random_lambda;
    }
}

impl<F: Field> SumcheckPureProof<F> for BottomRowEvalProof<F> {
    fn from_sumcheck(sumcheck_claim: &SumcheckClaim<F>, proof: Proof<F>) -> Self {
        BottomRowEvalProof {
            poly_info: sumcheck_claim.poly.info(),
            sumcheck_proof: proof,
            row_at_r: F::zero(),
        }
    }

    fn get_poly_info(&self) -> &PolynomialInfo {
        &self.poly_info
    }

    fn get_sumcheck_proof(&self) -> &Proof<F> {
        &self.sumcheck_proof
    }
}

impl<F: Field> SumcheckPureProverState<F> for BottomRowEvalProverState<F> {
    fn from_sumcheck(
        sumcheck_prover_state: sumcheck::prover::ProverState<F>,
        claim: SumcheckClaim<F>,
    ) -> Self {
        BottomRowEvalProverState {
            randomness: sumcheck_prover_state.randomness,
        }
    }
}

impl<F: Field> SumcheckPureSubclaim<F> for BottomRowVerifierSubclaim<F> {
    fn from_sumcheck(sumcheck_subclaim: sumcheck::verifier::SubClaim<F>) -> Self {
        BottomRowVerifierSubclaim {
            randomness: sumcheck_subclaim.point,
        }
    }
}

impl<F: Field + Serialize> SumcheckPIOP<F> for BottomRowEvalPIOP<F> {
    type Instance = BottomRowEvalInstance<F>;
    type Info = BottomRowEvalinfo<F>;
    type Proof = BottomRowEvalProof<F>;
    type ProverState = BottomRowEvalProverState<F>;
    type VerifierSubclaim = BottomRowVerifierSubclaim<F>;

    fn prover(
        trans: &mut helper::Transcript<F>,
        instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState) {
        let (mut proof, state) = Self::prover_without_evals(trans, instance);
        let row_at_r = instance.row.evaluate(&state.randomness);
        proof.row_at_r = row_at_r;
        (proof, state)
    }
    fn prover_add_sumcheck(
        instance: &Self::Instance,
        claim: &mut SumcheckClaim<F>,
        randomness: &[F],
        _lagrange_kernel: Option<&LagrangeKernel<F>>,
    ) -> Option<Self::ProverState> {
        instance.add_into_sumcheck(claim, randomness[0]);
        None
    }

    fn verifier_compute_subclaim(
        info: &Self::Info,
        proof: &Self::Proof,
        subclaim: &mut sumcheck::verifier::SubClaim<F>,
        randomness: &[F],
        _kernel_at_r: Option<F>,
    ) {
        let kernel_at_r = eval_identity_function(&info.point_ry, &subclaim.point);
        let coeff = eval_identity_function(&vec![F::one(); info.log_num_rows], &info.point_rx);
        subclaim.expected_evaluations -= kernel_at_r * coeff * randomness[0] * proof.row_at_r;
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
    fn test_bottom_row_eval_piop() {
        let mut rng = rand::rng();
        let log_num_rows = 4;
        let log_num_cols = 10;
        let instance = BottomRowEvalInstance::<FF>::random(&mut rng, log_num_rows, log_num_cols);
        let info = instance.info();
        let mut prover_trans = Transcript::<FF>::default();
        let (proof, _state) = BottomRowEvalPIOP::<FF>::prover(&mut prover_trans, &instance);
        let mut verifier_trans = Transcript::<FF>::default();
        let (res, _) = BottomRowEvalPIOP::<FF>::verifier(&mut verifier_trans, &info, &proof);
        assert!(res);
    }
}
