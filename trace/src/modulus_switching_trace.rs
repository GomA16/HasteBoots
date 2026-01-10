use std::rc::Rc;

use algebra::{AsFrom, AsInto, DenseMultilinearExtension, Field};
use num_traits::Zero;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

// Trace for Modulus Switching operation
// From Field::MODULUS_VALUE (denoted by Q) to modulus_after (denoted by q)
// Relation between b = b' mod q
// - Indexed Lookup Trace:
//   Table is q, 1, ..., q-1
//   Input is b' \in [1, q]
//   Index is b \in [0, q-1]
// - Lookup Trace (large table):
//   Table is [0, 2k]
//   Input is e
pub struct ModulusSwitchingTrace<F: Field> {
    pub log_num: usize,
    // q is the modulus after switching s.t. 2q | Q - 1
    pub modulus_after: F,
    // parameter k = Q - 1 / 2q
    pub blk_param: F,
    // denoted by a
    pub input: Vec<F>,
    // denoted by b = round (a * q / Q) \in Z_q
    pub output: Vec<F>,
    // denoted by b' s.t. b' = b mod q AND b' \in [1, q]
    pub output_witness: Vec<F>,

    // denoted by e = a - (2b' - 1) * k - 1 \in [0, 2k)
    pub helper: Vec<F>, // (no need to commit)
    // 2k = (Q - 1) / q
    pub helper_range: usize,
    // tunable parameter
    pub helper_basis_len: usize,
}

pub struct ModulusSwitchingTraceMLE<F: Field> {
    pub log_num: usize,
    // q is the modulus after switching s.t. 2q | Q - 1
    pub modulus_after: F,
    // parameter k = Q - 1 / 2q
    pub blk_param: F,
    // denoted by a
    pub input: Rc<DenseMultilinearExtension<F>>,
    // denoted by b = round (a * q / Q) \in Z_q
    pub output: Rc<DenseMultilinearExtension<F>>,
    // denoted by b' s.t. b' = b mod q AND b' \in [1, q]
    pub output_witness: Rc<DenseMultilinearExtension<F>>,

    // denoted by e = a - (2b' - 1) * k - 1 \in [0, 2k] < (2k + 1)
    // a = k: e = 2k
    // otherwise: a < 2k
    pub helper: Rc<DenseMultilinearExtension<F>>, // (no need to commit)
    // 2k = (Q - 1) / q
    pub helper_range: usize,
    // tunable parameter
    pub helper_basis_len: usize,
}

impl<F: Field> ModulusSwitchingTrace<F> {
    pub fn new(log_num: usize, modulus_after: F) -> Self {
        // k = (Q - 1) / (2q)
        let blk_size = (-F::one()) / (modulus_after + modulus_after);
        // helper range = 2k = (Q - 1) / q
        let helper_range: usize = ((-F::one()) / modulus_after).value().as_into();
        Self {
            log_num,
            modulus_after,
            // k = (Q - 1) / (2q)
            blk_param: blk_size,
            input: Vec::with_capacity(1 << log_num),
            output: Vec::with_capacity(1 << log_num),
            output_witness: Vec::with_capacity(1 << log_num),
            helper: Vec::with_capacity(1 << log_num),
            helper_range,
            helper_basis_len: 10,
        }
    }

    pub fn append_input(&mut self, input: &[F]) {
        self.input.extend_from_slice(input);
    }

    pub fn append_output(&mut self, output: &[F]) {
        self.output.extend_from_slice(output);
    }

    pub fn finalize(&mut self, num: usize) {
        self.output_witness = self
            .output
            .par_iter()
            .map(|b| if b.is_one() { self.modulus_after } else { *b })
            .collect();
        // e = a - (2b' - 1) * k - 1 \in [0, 2k]
        self.helper = self
            .input
            .par_iter()
            .zip(self.output_witness.par_iter())
            .map(|(a, b_wit)| {
                let two_b_minus_1 = *b_wit + *b_wit - F::one();
                *a - two_b_minus_1 * self.blk_param - F::one()
            })
            .collect();

        let num_zeros = (1 << self.log_num) - num;
        self.input.extend(vec![F::zero(); num_zeros]);
        self.output.extend(vec![F::zero(); num_zeros]);
        self.output_witness.extend(vec![F::zero(); num_zeros]);
        self.helper.extend(vec![F::zero(); num_zeros]);
    }
}

impl<F: Field> From<ModulusSwitchingTrace<F>> for ModulusSwitchingTraceMLE<F> {
    fn from(trace: ModulusSwitchingTrace<F>) -> Self {
        Self {
            log_num: trace.log_num,
            modulus_after: trace.modulus_after,
            blk_param: trace.blk_param,
            input: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                trace.log_num,
                trace.input,
            )),
            output: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                trace.log_num,
                trace.output,
            )),
            output_witness: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                trace.log_num,
                trace.output_witness,
            )),
            helper: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                trace.log_num,
                trace.helper,
            )),
            helper_range: trace.helper_range,
            helper_basis_len: trace.helper_basis_len,
        }
    }
}
