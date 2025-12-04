//! Sparse Eval IOP

use core::fmt;
use std::rc::Rc;

use algebra::{
    AbstractExtensionField, DenseMultilinearExtension, Field, ListOfProductsOfPolynomials,
    utils::Transcript,
};
use serde::{Deserialize, Serialize};
use sumcheck::{MLSumcheck, verifier::SubClaim};
use sumcheck::{ProofWrapper, SumcheckKit};

use super::LookupInstance;
use super::lookup::LookupInstanceInfo;

use crate::utils::{eval_identity_function, gen_identity_evaluations};

/// Sparse Matrix Instance used as prover keys
pub struct SparseEvalInstance<F: Field> {
    /// number of variables on x axis
    pub num_x_vars: usize,
    /// number of variables on y axis
    pub num_y_vars: usize,
    /// sparse representation
    pub row: Rc<DenseMultilinearExtension<F>>,
    /// sparse representation
    pub val: Rc<DenseMultilinearExtension<F>>,
    /// evaluation vector defined by E_rx(k) = eq(to-bits(row(k)), rx)
    pub eval_rx: DenseMultilinearExtension<F>,
    /// evaluation table defined by eq(x, rx)
    pub table: DenseMultilinearExtension<F>,
}

/// Info of Sparse Matrix used as verifier key
pub struct SparseEvalInstanceInfo {
    /// number of variables on x axis
    pub num_x_vars: usize,
    /// number of variables on y axis
    pub num_y_vars: usize,
}

/// Evaluations of all MLEs involved in the instance at a random point
#[derive(Serialize, Deserialize)]
pub struct SparseEvalInstanceEval<F: Field> {
    /// evaluation of val
    pub val: F,
    /// evaluation of eval_rx
    pub eval_rx: F,
}

/// IOP for sparse matrix evaluation
pub struct SparseEvalIOP<F: Field> {
    /// The random point on x axis
    pub r_x: Vec<F>,
    /// The random point on y axis
    pub r_y: Vec<F>,
    /// The evaluation on the above random point
    pub eval: F,
}

impl<F: Field> SparseEvalInstance<F> {
    /// Construct a new instance from slice
    #[inline]
    pub fn from_slice(
        num_x_vars: usize,
        num_y_vars: usize,
        row: &Rc<DenseMultilinearExtension<F>>,
        val: &Rc<DenseMultilinearExtension<F>>,
    ) -> Self {
        assert_eq!(row.num_vars, num_y_vars);
        assert_eq!(val.num_vars, num_y_vars);
        Self {
            num_x_vars,
            num_y_vars,
            row: Rc::clone(row),
            val: Rc::clone(val),
            eval_rx: Default::default(),
            table: Default::default(),
        }
    }

    /// info
    #[inline]
    pub fn info(&self) -> SparseEvalInstanceInfo {
        SparseEvalInstanceInfo {
            num_x_vars: self.num_x_vars,
            num_y_vars: self.num_y_vars,
        }
    }
    /// Construct a EF version
    pub fn to_ef<EF: AbstractExtensionField<F>>(&self) -> SparseEvalInstance<EF> {
        SparseEvalInstance::<EF> {
            num_x_vars: self.num_x_vars,
            num_y_vars: self.num_y_vars,
            row: Rc::new(self.row.to_ef()),
            val: Rc::new(self.val.to_ef()),
            eval_rx: Default::default(),
            table: Default::default(),
        }
    }

    /// Evaluate at a random point defined over Field
    #[inline]
    pub fn evaluate(&self, point: &[F]) -> SparseEvalInstanceEval<F> {
        SparseEvalInstanceEval::<F> {
            eval_rx: self.eval_rx.evaluate(point),
            val: self.val.evaluate(point),
        }
    }

    /// Extract the lookup instance returned in the subclaim
    #[inline]
    pub fn extract_lookup_instance(&self) -> LookupInstance<F> {
        //FIXME support lookup with different size
        assert_eq!(self.eval_rx.num_vars, self.table.num_vars);
        LookupInstance::from_slice(std::slice::from_ref(&self.eval_rx), self.table.clone(), 1)
    }
}

impl fmt::Display for SparseEvalInstanceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "An instance of SparseEvalInstance: #vars_x = {}, #vars_y = {}",
            self.num_x_vars, self.num_y_vars,
        )
    }
}

impl SparseEvalInstanceInfo {
    /// extract the info of the lookup instance
    #[inline]
    pub fn extract_lookup_info(&self) -> LookupInstanceInfo {
        LookupInstanceInfo {
            num_vars: self.num_y_vars,
            num_batch: 1,
            block_size: 1,
            block_num: 1,
        }
    }
}

impl<F: Field + Serialize> SparseEvalIOP<F> {
    /// The prover initiates the evaluation vector defined by r_x
    #[inline]
    pub fn prover_generate_eval_vector(&self, instance: &mut SparseEvalInstance<F>) {
        let eq_rx = gen_identity_evaluations(&self.r_x);
        let eval_rx = <DenseMultilinearExtension<F>>::from_evaluations_vec(
            instance.num_y_vars,
            instance
                .row
                .iter()
                .map(|idx| eq_rx[idx.value().into() as usize])
                .collect(),
        );
        instance.eval_rx = eval_rx;
        instance.table = eq_rx;
    }

    /// prepare the polynomial in the sumcheck protocol
    pub fn prove_as_subprotocol(
        &self,
        poly: &mut ListOfProductsOfPolynomials<F>,
        subclaim: &mut F,
        instance: &SparseEvalInstance<F>,
        eq_at_ry: &Rc<DenseMultilinearExtension<F>>,
    ) {
        poly.add_product(
            [
                Rc::clone(&instance.val),
                Rc::new(instance.eval_rx.clone()),
                Rc::clone(eq_at_ry),
            ],
            F::one(),
        );
        *subclaim += self.eval;
    }

    /// SparseEvalIOP prover
    pub fn prove(&self, instance: &SparseEvalInstance<F>) -> SumcheckKit<F> {
        let mut trans = Transcript::<F>::new();
        let eq_at_u = Rc::new(gen_identity_evaluations(&self.r_y));

        let mut poly = ListOfProductsOfPolynomials::<F>::new(instance.num_y_vars);

        let mut claimed_sum = F::zero();
        Self::prove_as_subprotocol(self, &mut poly, &mut claimed_sum, instance, &eq_at_u);

        let (proof, state) =
            MLSumcheck::prove(&mut trans, &poly).expect("fail to prove the sumcheck protocol");

        SumcheckKit {
            proof,
            claimed_sum,
            info: poly.info(),
            u: self.r_y.clone(),
            randomness: state.randomness,
        }
    }

    /// Verify the Sparse Matrix Evaluation
    pub fn verify(&self, wrapper: &ProofWrapper<F>, evals: &SparseEvalInstanceEval<F>) -> bool {
        let mut trans = Transcript::new();
        let mut subclaim = MLSumcheck::verify(
            &mut trans,
            &wrapper.info,
            wrapper.claimed_sum,
            &wrapper.proof,
        )
        .expect("fail to verify the sumcheck protocol");

        let eq_at_u_r = eval_identity_function(&self.r_y, &subclaim.point);

        // check the sumcheck evaluation
        if !Self::verify_as_subprotocol(&mut subclaim, evals, eq_at_u_r) {
            return false;
        }

        subclaim.expected_evaluations == F::zero()
    }

    /// Verify the subclaim.
    pub fn verify_as_subprotocol(
        subclaim: &mut SubClaim<F>,
        evals: &SparseEvalInstanceEval<F>,
        eq_at_u_r: F,
    ) -> bool {
        subclaim.expected_evaluations -= evals.val * evals.eval_rx * eq_at_u_r;
        true
    }
}
