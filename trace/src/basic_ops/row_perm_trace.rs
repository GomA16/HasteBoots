use std::rc::Rc;

use algebra::{AbstractExtensionField, AsInto, DenseMultilinearExtension, Field};
use helper::utils::gen_identity_evaluations;
use rand_distr::num_traits::Zero;

use crate::{ConvertToEF, lookup_trace::indexed_table::IndexedLookupTraceMLE};

// Trace for row permutation operation
#[derive(Clone)]
pub struct RowPermTrace<F: Field> {
    pub log_num_rows: usize,
    pub log_num_cols: usize,
    pub input: Vec<F>,
    pub output: Vec<F>,
    pub permutation_info: PermutationInfo<F>,
}

pub struct RowPermTraceMLE<F: Field> {
    pub log_num_rows: usize,
    pub log_num_cols: usize,
    pub input: Rc<DenseMultilinearExtension<F>>,
    pub output: Rc<DenseMultilinearExtension<F>>,
    pub permutation_info: PermutationInfo<F>,
}

#[derive(Clone)]
pub struct PermutationInfo<F: Field> {
    pub log_num: usize,
    pub permutation_table: Vec<usize>,
    pub signed: Option<PermutationSignedInfo<F>>,
}

#[derive(Clone)]
pub struct PermutationSignedInfo<F: Field> {
    pub log_dim: usize,
    pub permutation: Vec<F>,
    pub sign: Vec<F>,
}

impl<F: Field> PermutationInfo<F> {
    // rotation left by offset
    pub fn new_rotation_left(log_num: usize, range: usize, offset: usize) -> Self {
        let mut permuation_table = (0..1 << log_num).map(|x| x).collect::<Vec<usize>>();
        permuation_table[0..range].rotate_left(offset);

        Self {
            log_num,
            permutation_table: permuation_table,
            signed: None,
        }
    }

    // key switching permutation
    pub fn new_ks_permutation(num: usize, blk_size: usize) -> Self {
        assert!(num.is_power_of_two());
        assert!(blk_size.is_power_of_two());
        assert!(num % blk_size == 0);
        let mut permutation_table = (0..num).map(|x| x).collect::<Vec<usize>>();
        let mut sign = vec![F::one(); num];
        sign[0] = -sign[0];
        permutation_table[1..].reverse();

        permutation_table
            .chunks_exact_mut(blk_size)
            .zip(sign.chunks_exact_mut(blk_size))
            .for_each(|(perm_chunk, sign_chunk)| {
                sign_chunk[0] = -sign_chunk[0];
                perm_chunk[1..].reverse();
            });

        let permutation_field = permutation_table
            .iter()
            .map(|&i| F::new((i as u32).as_into()))
            .collect::<Vec<F>>();
        let trace = PermutationSignedInfo {
            log_dim: num.trailing_zeros() as usize,
            permutation: permutation_field,
            sign,
        };

        Self {
            log_num: num.trailing_zeros() as usize,
            permutation_table,
            signed: Some(trace),
        }
    }

    // sample extraction permutation
    pub fn new_sample_extraction_permutation(num: usize) -> Self {
        assert!(num.is_power_of_two());
        let mut permutation_table = (0..num).map(|x| x).collect::<Vec<usize>>();
        let mut sign = vec![-F::one(); num];
        sign[0] = F::one();
        permutation_table[1..].reverse();
        let permutation_field = permutation_table
            .iter()
            .map(|&i| F::new((i as u32).as_into()))
            .collect::<Vec<F>>();
        let trace = PermutationSignedInfo {
            log_dim: num.trailing_zeros() as usize,
            permutation: permutation_field,
            sign,
        };
        Self {
            log_num: num.trailing_zeros() as usize,
            permutation_table,
            signed: Some(trace),
        }
    }

    pub fn get_inverse_permutation(&self) -> Self {
        let mut inversed_permutation_table = vec![0; 1 << self.log_num];
        for (i, &p) in self.permutation_table.iter().enumerate() {
            inversed_permutation_table[p] = i;
        }
        Self {
            log_num: self.log_num,
            permutation_table: inversed_permutation_table,
            signed: None,
        }
    }

    pub fn permute<Q: Zero + Clone + Copy>(&self, input: &[Q]) -> Vec<Q> {
        assert_eq!(1 << self.log_num, input.len());
        assert!(self.signed.is_none());
        let mut output = vec![Q::zero(); 1 << self.log_num];
        for x in 0..1 << self.log_num {
            let y = self.permutation_table[x];
            output[x] = input[y];
        }
        output
    }

    pub fn permute_signed(&self, input: &[F]) -> Vec<F> {
        assert_eq!(1 << self.log_num, input.len());
        assert!(self.signed.is_some());
        let sign = &self.signed.as_ref().unwrap().sign;
        let mut output = vec![F::zero(); 1 << self.log_num];
        for x in 0..1 << self.log_num {
            let y = self.permutation_table[x];
            output[x] = input[y] * sign[x];
        }
        output
    }

    pub fn permute_row_wise<Q: Zero + Clone + Copy>(
        &self,
        log_num_rows: usize,
        log_num_cols: usize,
        input: &[Q],
    ) -> Vec<Q> {
        assert_eq!(1 << self.log_num, 1 << log_num_rows);
        assert_eq!(1 << (log_num_rows + log_num_cols), input.len());
        assert!(self.signed.is_none());
        let mut output = vec![Q::zero(); 1 << (log_num_rows + log_num_cols)];
        for y in 0..1 << log_num_cols {
            for x in 0..1 << log_num_rows {
                let z = self.permutation_table[x];
                output[y | (x << log_num_cols)] = input[y | (z << log_num_cols)];
            }
        }
        output
    }

    pub fn permute_row_wise_signed(
        &self,
        log_num_rows: usize,
        log_num_cols: usize,
        input: &[F],
    ) -> Vec<F> {
        assert_eq!(1 << self.log_num, 1 << log_num_rows);
        assert_eq!(1 << (log_num_rows + log_num_cols), input.len());
        assert!(self.signed.is_some());
        let sign = &self.signed.as_ref().unwrap().sign;
        let mut output = vec![F::zero(); 1 << (log_num_rows + log_num_cols)];
        for y in 0..1 << log_num_cols {
            for x in 0..1 << log_num_rows {
                let z = self.permutation_table[x];
                output[y | (x << log_num_cols)] = input[y | (z << log_num_cols)] * sign[x];
            }
        }
        output
    }

    // P(k, rx) = eq(rx, \rho_inv(k)) for all k in {0, 1}^n
    // Considering it as an indexed lookup table trace.
    pub fn extract_indexed_lookup_trace(&self, point: &[F]) -> IndexedLookupTraceMLE<F> {
        let eq_at_rx = gen_identity_evaluations(point);
        let inversed_permutation = self.get_inverse_permutation();
        let invered_permutation_field = inversed_permutation
            .permutation_table
            .iter()
            .map(|&i| F::new((i as u32).as_into()))
            .collect::<Vec<F>>();

        // P(k, rx) = eq(rx, \rho_inv(k)) for all k in {0, 1}^n
        let permutation_at_rx = inversed_permutation.permute(eq_at_rx.as_slice());

        // indexed lookup trace
        let lookup_input =
            DenseMultilinearExtension::from_evaluations_vec(self.log_num, permutation_at_rx);
        let lookup_index = DenseMultilinearExtension::from_evaluations_vec(
            self.log_num,
            invered_permutation_field,
        );

        IndexedLookupTraceMLE {
            num_input_vars: self.log_num,
            num_table_vars: self.log_num,
            index: Rc::new(lookup_index),
            input: Rc::new(lookup_input),
            table: Rc::new(eq_at_rx),
            table_point: Some(point.to_vec()),
        }
    }

    // P(k, rx) = \sum_k eq(rx, ρ_inv(k)) * s(k) for all k in {0, 1}^n
    // Compute P(y, r) for all y in the hypercube, where r is a random point in F^n
    pub fn fixed_variable(&self, point: &[F]) -> DenseMultilinearExtension<F> {
        let eq_at_rx = gen_identity_evaluations(point);
        let inversed_permutation = self.get_inverse_permutation();

        assert!(self.signed.is_some());
        let sign_info = self.signed.as_ref().unwrap();
        let sign_permutated = inversed_permutation.permute(&sign_info.sign);
        let mut eq_at_rx_permutated = inversed_permutation.permute(eq_at_rx.as_slice());
        eq_at_rx_permutated
            .iter_mut()
            .zip(sign_permutated.iter())
            .for_each(|(v, s)| {
                *v *= *s;
            });

        DenseMultilinearExtension::from_evaluations_vec(self.log_num, eq_at_rx_permutated)
    }
}

impl<F: Field> RowPermTrace<F> {
    // Generate a random row-permutated matrix trace
    pub fn random_rotation_left<R: rand::Rng + rand::CryptoRng>(
        rng: &mut R,
        log_num_rows: usize,
        log_num_cols: usize,
    ) -> Self {
        // Special row permutation used in our case: cyclic shift by 1
        let permutation_info =
            PermutationInfo::new_rotation_left(log_num_rows, 1 << log_num_rows, 1);
        let input = (0..1 << (log_num_rows + log_num_cols))
            .map(|_| F::random(rng))
            .collect::<Vec<F>>();
        let output = permutation_info.permute_row_wise(log_num_rows, log_num_cols, &input);

        Self {
            log_num_rows,
            log_num_cols,
            input,
            output,
            permutation_info,
        }
    }

    // Generate a random trace, which is permuted according to key-switching permutation
    pub fn random_ks_permutation<R: rand::Rng + rand::CryptoRng>(
        rng: &mut R,
        log_num_rows: usize,
        log_num_cols: usize,
        // permutation params
        log_blk_size: usize,
    ) -> Self {
        let num = 1 << log_num_rows;
        let blk_size = 1 << log_blk_size;

        let permutation_info = PermutationInfo::new_ks_permutation(num, blk_size);
        let input = (0..1 << (log_num_rows + log_num_cols))
            .map(|_| F::random(rng))
            .collect::<Vec<F>>();
        let output = permutation_info.permute_row_wise_signed(log_num_rows, log_num_cols, &input);

        Self {
            log_num_rows,
            log_num_cols,
            input,
            output,
            permutation_info,
        }
    }

    pub fn from_batch_trace(traces: Vec<RowPermTrace<F>>) -> Self {
        let num_trace = traces.len();
        assert!(num_trace.is_power_of_two());
        let log_num_trace = num_trace.trailing_zeros() as usize;
        let log_num_rows = traces[0].log_num_rows;
        let mut input = vec![F::zero(); 1 << (log_num_rows + log_num_trace)];
        let mut output = vec![F::zero(); 1 << (log_num_rows + log_num_trace)];
        let permutation_info = traces[0].permutation_info.clone();
        traces.iter().enumerate().for_each(|(i, trace)| {
            for r in 0..1 << log_num_rows {
                input[i | (r << log_num_trace)] = trace.input[r];
                output[i | (r << log_num_trace)] = trace.output[r];
            }
        });
        RowPermTrace {
            log_num_rows,
            log_num_cols: traces[0].log_num_cols + log_num_trace,
            input,
            output,
            permutation_info,
        }
    }
}

impl<F: Field> From<RowPermTrace<F>> for RowPermTraceMLE<F> {
    fn from(trace: RowPermTrace<F>) -> Self {
        let num_vars = trace.log_num_rows + trace.log_num_cols;
        let input = DenseMultilinearExtension::from_evaluations_vec(num_vars, trace.input);
        let output = DenseMultilinearExtension::from_evaluations_vec(num_vars, trace.output);
        Self {
            log_num_rows: trace.log_num_rows,
            log_num_cols: trace.log_num_cols,
            input: Rc::new(input),
            output: Rc::new(output),
            permutation_info: trace.permutation_info,
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for RowPermTraceMLE<F> {
    type Output = RowPermTraceMLE<EF>;
    fn to_ef(&self) -> Self::Output {
        RowPermTraceMLE {
            log_num_rows: self.log_num_rows,
            log_num_cols: self.log_num_cols,
            input: Rc::new(self.input.to_ef()),
            output: Rc::new(self.output.to_ef()),
            permutation_info: self.permutation_info.to_ef(),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for PermutationSignedInfo<F> {
    type Output = PermutationSignedInfo<EF>;
    fn to_ef(&self) -> Self::Output {
        PermutationSignedInfo {
            log_dim: self.log_dim,
            permutation: self.permutation.to_ef(),
            sign: self.sign.to_ef(),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for PermutationInfo<F> {
    type Output = PermutationInfo<EF>;
    fn to_ef(&self) -> Self::Output {
        PermutationInfo {
            log_num: self.log_num,
            permutation_table: self.permutation_table.clone(),
            signed: match self.signed {
                None => None,
                Some(ref s) => Some(s.to_ef()),
            },
        }
    }
}

impl<F: Field> Default for PermutationInfo<F> {
    fn default() -> Self {
        Self {
            log_num: 0,
            permutation_table: vec![],
            signed: None,
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use algebra::{DenseMultilinearExtension, Field, MultilinearExtension, derive::Field};
    use num_traits::One;

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
        permutation_table: &Vec<usize>,
    ) -> DenseMultilinearExtension<F> {
        assert_eq!(1 << dim, permutation_table.len());
        let mut perm_matrix = vec![F::zero(); (1 << dim) * (1 << dim)];

        for x in 0..1 << dim {
            for y in 0..1 << dim {
                let idx = x * (1 << dim) + y;
                if permutation_table[x] == y {
                    perm_matrix[idx] = F::one();
                }
            }
        }
        DenseMultilinearExtension::from_evaluations_vec(dim * 2, perm_matrix)
    }

    fn generate_permutation_matrix_signed<F: Field>(
        dim: usize,
        permutation_table: &Vec<usize>,
        sign: &Vec<F>,
    ) -> DenseMultilinearExtension<F> {
        assert_eq!(1 << dim, permutation_table.len());
        let mut perm_matrix = vec![F::zero(); (1 << dim) * (1 << dim)];

        for x in 0..1 << dim {
            for y in 0..1 << dim {
                let idx = x * (1 << dim) + y;
                if permutation_table[x] == y {
                    perm_matrix[idx] = sign[x];
                }
            }
        }
        DenseMultilinearExtension::from_evaluations_vec(dim * 2, perm_matrix)
    }

    #[test]
    fn test_extract_indexed_lookup_in_permutation() {
        let dim = 2 as usize;
        let permutataion_table = vec![1, 2, 3, 0]; // permutation on 4 elements
        let permutation_info = PermutationInfo::<FF> {
            log_num: dim,
            permutation_table: permutataion_table.clone(),
            signed: None,
        };

        let perm_matrix_mle =
            generate_permutation_matrix::<FF>(dim, &permutation_info.permutation_table);
        let rng = &mut rand::rng();
        let point: Vec<FF> = (0..dim).map(|_| FF::random(rng)).collect();

        let permutation_at_point = perm_matrix_mle.fix_variables_back(&point);
        let computed_perm_mle = permutation_info.extract_indexed_lookup_trace(&point);

        assert_eq!(permutation_at_point, *computed_perm_mle.input);
    }

    #[test]
    fn test_fixed_variable() {
        let log_num = 2 as usize;
        let permutation_table = vec![1, 2, 3, 0]; // permutation on 4 elements
        let sign = vec![true, false, true, true];
        let sign = sign
            .iter()
            .map(|&s| match s {
                true => FF::one(),
                false => -FF::one(),
            })
            .collect::<Vec<FF>>();

        let sign_info = PermutationSignedInfo::<FF> {
            log_dim: log_num,
            permutation: permutation_table
                .iter()
                .map(|&i| FF::new((i as u32).as_into()))
                .collect::<Vec<FF>>(),
            sign: sign.clone(),
        };
        let permutation_info = PermutationInfo::<FF> {
            log_num,
            permutation_table: permutation_table.clone(),
            signed: Some(sign_info),
        };

        let permutation_matrix =
            generate_permutation_matrix_signed::<FF>(log_num, &permutation_table, &sign);
        let rng = &mut rand::rng();
        let point: Vec<FF> = (0..log_num).map(|_| FF::random(rng)).collect();

        let permutation_at_point = permutation_matrix.fix_variables_back(&point);
        let computed_perm_mle = permutation_info.fixed_variable(&point);

        assert_eq!(permutation_at_point, computed_perm_mle);
    }

    #[test]
    fn test_permutation_matrix() {
        let mut rng = rand::rng();
        let log_num_rows = 2;
        let log_num_cols = 2;

        let random_trace =
            RowPermTrace::<FF>::random_rotation_left(&mut rng, log_num_rows, log_num_cols);
        let permutation_matrix = generate_permutation_matrix::<FF>(
            log_num_rows,
            &random_trace.permutation_info.permutation_table,
        );

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
}
