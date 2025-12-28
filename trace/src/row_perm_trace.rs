use std::rc::Rc;

use algebra::{DenseMultilinearExtension, Field};
use rand::CryptoRng;

// Trace for row permutation operation
pub struct RowPermTrace<F: Field> {
    pub log_num_rows: usize,
    pub log_num_cols: usize,
    pub input: Vec<F>,
    pub output: Vec<F>,
    pub permutation_info: Vec<usize>,
}

pub struct RowPermTraceMLE<F: Field> {
    pub log_num_rows: usize,
    pub log_num_cols: usize,
    pub input: Rc<DenseMultilinearExtension<F>>,
    pub output: Rc<DenseMultilinearExtension<F>>,
    pub permutation_info: Vec<usize>,
}

impl<F: Field> RowPermTrace<F> {
    pub fn random<R: rand::Rng + rand::CryptoRng>(
        rng: &mut R,
        log_num_rows: usize,
        log_num_cols: usize,
    ) -> Self {
        let num_rows = 1 << log_num_rows;
        let row_size = 1 << log_num_cols;
        let mut perm = (0..num_rows).map(|x| x).collect::<Vec<usize>>();
        perm.rotate_left(1);
        let initial_input = (0..row_size).map(|_| F::random(rng)).collect::<Vec<F>>();
        let mut input = Vec::with_capacity(1 << (log_num_rows + log_num_cols));
        let mut output = Vec::with_capacity(1 << (log_num_rows + log_num_cols));

        input.extend_from_slice(&initial_input);

        for _ in 1..num_rows {
            let row = (0..row_size).map(|_| F::random(rng)).collect::<Vec<F>>();
            output.extend_from_slice(&row);
            input.extend_from_slice(&row);
        }

        output.extend_from_slice(&initial_input);

        Self {
            log_num_rows,
            log_num_cols,
            input,
            output,
            permutation_info: perm,
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
