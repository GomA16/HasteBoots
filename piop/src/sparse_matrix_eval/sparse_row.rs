use algebra::{DenseMultilinearExtension, Field, PolynomialInfo};
use serde::Serialize;
use sumcheck::Proof;
use std::rc::Rc;

use crate::{SumcheckInfo, SumcheckInstance};

/// Consider a sparse matrix C with dimensions M x N, where each row contains only
/// a single non-zero entry.
/// 
/// Matrix indexing: C[Y][X] where X is the row index and Y is the column index.
/// 
/// MLE indexing: `C(y, x) = C(y0, y1, ..., y\logN, x0, x1, ..., x\logM)` where 
/// `Y = y0*2^0 + y1*2^1 + ... + yN*2^\logN` and `X = x0*2^0 + x1*2^1 + ... 
/// + xM*2^\logM`. This is the little-endian representation.
/// 
/// This instance is related to evaluating the sparse matrix on a random point
/// (r_y, r_x) as follows:
/// C(r_y, r_x) = 
///     \sum_k val(k) * eq(to-bits(col(k)), r_y) * eq(to-bits(row(k)), r_x).
///     k iterates the hybercube of dimension logM.
/// where r_y is a random point on the y axis with logN variables, and r_x is a
/// random point on the x axis with logM variables. Here, col(k) and row(k) 
/// represent the column and row indices of the k-th non-zero entry in the 
/// sparse matrix.
/// 
/// This equation is from Eq(8) in Lasso: https://eprint.iacr.org/2023/1216.pdf.
/// 
/// Here, `row(k)` = k, so eq(to-bits(row(k)), r_x) = eq(k, r_x). The evaluation
/// can be simplied to:
/// C(r_y, r_x) =
///    \sum_k val(k) * eq(to-bits(col(k)), r_y) * eq(k, r_x).
pub struct SparseRowEvalInstance<F: Field> {
    // denoted by logM in the above description
    pub num_x_vars: usize,
    // denoted by logN in the above description
    pub num_y_vars: usize,
    /// sparse representation
    pub col: Rc<DenseMultilinearExtension<F>>,
    pub val: Rc<DenseMultilinearExtension<F>>,
    /// helper MLE E_ry(k) = eq(to-bits(col(k)), ry)
    pub eval_mle_ry: DenseMultilinearExtension<F>,
    pub point_ry: Vec<F>,
    pub point_rx: Vec<F>,
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

pub struct SparseRowEvalProof<F: Field> {
    pub poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<F>,
    pub eval_mle_at_r: F,
    pub val_at_r: F,
}

pub struct SparseRowProverState<F: Field> {
    /// Lagrange Kernel eq(k, r_x)
    pub kernel_rx: Rc<DenseMultilinearExtension<F>>,
    /// E_ry(k) = eq(to-bits(col(k)), ry)
    pub eval_mle_ry: F,
    pub point_ry: Vec<F>,
}

pub struct SparseRowVerifierSubclaim<F: Field> {
    /// query point_r on val(k) and E_ry(k) with PCS
    pub sumcheck_point: Vec<F>,
    /// check E_ry(k) = eq(to-bits(col(k)), ry) with lookup argument
    pub point_ry: Vec<F>,
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