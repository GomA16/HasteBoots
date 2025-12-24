use algebra::DenseMultilinearExtension;
use algebra::Field;
use bincode::config;
use helper::utils::gen_identity_evaluations;
use num_traits::{Zero, ops::inv, sign};
use std::{mem::Discriminant, rc::Rc};

pub struct RowPermInstance<F: Field> {
    pub log_num_cols: usize,
    pub log_num_rows: usize,
    pub input: Rc<DenseMultilinearExtension<F>>,
    pub output: Rc<DenseMultilinearExtension<F>>,
    pub perm: Vec<usize>,
}

pub struct RowPermInfo {
    pub log_num_cols: usize,
    pub log_num_rows: usize,
    pub perm: Vec<usize>,
}

pub fn compute_inverse_permutation(perm: &Vec<usize>) -> Vec<usize> {
    let n = perm.len();
    let mut inv_perm = vec![0; n];
    for (i, &p) in perm.iter().enumerate() {
        inv_perm[p] = i;
    }
    inv_perm
}

// pub fn perform_permutation<F: Field>(
//     dim: usize,
//     perm: &Vec<usize>,
//     input: &DenseMultilinearExtension<F>,
// ) -> DenseMultilinearExtension<F> {
//     assert_eq!(1 << dim, perm.len());
//     let mut output_evals = vec![F::zero(); 1 << dim];
//     for x in 0..1 << dim {
//         let y = perm[x];
//         output_evals[x] = input.evaluations[y];
//     }
//     DenseMultilinearExtension::from_evaluations_vec(dim, output_evals)
// }

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
        let perm = vec![3, 0, 1, 2]; // permutation on 4 elements
        let sig = vec![-1, 1, 1, 1];

        let perm_matrix_mle = generate_permutation_matrix::<FF>(dim, &perm);
        let rng = &mut rand::rng();
        let point: Vec<FF> = (0..dim).map(|_| FF::random(rng)).collect();

        let perm_mle = perm_matrix_mle.fix_variables_back(&point);
        let computed_perm_mle = compute_permutation_at_point::<FF>(dim, &perm, &point);

        assert_eq!(perm_mle, computed_perm_mle);
    }
}
