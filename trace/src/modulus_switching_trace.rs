use std::rc::Rc;

use algebra::{AsInto, Basis, DecomposableField, DenseMultilinearExtension, Field};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    cmp_trace::{
        eq_trace::{EQTables, EQTablesMLE, EQTraceMLE},
        lt_trace::{LTTables, LTTablesMLE, LTTraceMLE},
    },
    lookup_trace::indexed_table::IndexedLookupTraceMLE,
};

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

    // denoted by e = a - (2b' - 1) * k - 1 \in [0, 2k] < (2k + 1)
    // a = k: e = 2k
    // otherwise: e < 2k
    pub helper: Vec<F>, // (no need to commit)
    // upper_bound = 2k = (Q - 1) / q
    pub helper_upper_bound: F,

    // comparison tables
    pub a_eq_k_tables: EQTables<F>,
    pub e_lt_2k_plus_1_tables: LTTables<F>,
    pub e_eq_2k_tables: EQTables<F>,
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
    // otherwise: e < 2k
    pub helper: Rc<DenseMultilinearExtension<F>>, // (no need to commit)
    // upper_bound = 2k = (Q - 1) / q
    pub helper_upper_bound: F,

    // comparison tables
    pub a_eq_k_tables: EQTablesMLE<F>,
    pub e_lt_2k_plus_1_tables: LTTablesMLE<F>,
    pub e_eq_2k_tables: EQTablesMLE<F>,
}

impl<F: DecomposableField> ModulusSwitchingTrace<F> {
    pub fn new(log_num: usize, modulus_after: F) -> Self {
        // k = (Q - 1) / (2q)
        let blk_size = (-F::one()) / (modulus_after + modulus_after);
        // helper_upper_bound = 2k = (Q - 1) / q
        let helper_upper_bound = (-F::one()) / modulus_after;
        let basis = Basis::<F>::new(10);
        let a_eq_k_tables = EQTables::<F>::new(blk_size, &basis);
        let e_lt_2k_plus_1_tables = LTTables::<F>::new(&basis, Some(helper_upper_bound + F::one()));
        let e_eq_2k_tables = EQTables::<F>::new(helper_upper_bound, &basis);
        Self {
            log_num,
            modulus_after,
            // k = (Q - 1) / (2q)
            blk_param: blk_size,
            input: Vec::with_capacity(1 << log_num),
            output: Vec::with_capacity(1 << log_num),
            output_witness: Vec::with_capacity(1 << log_num),
            helper: Vec::with_capacity(1 << log_num),
            helper_upper_bound,

            a_eq_k_tables,
            e_lt_2k_plus_1_tables,
            e_eq_2k_tables,
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
            .map(|b| if b.is_zero() { self.modulus_after } else { *b })
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
        self.output_witness
            .extend(vec![self.modulus_after; num_zeros]);
        self.helper.extend(vec![self.blk_param; num_zeros]);
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
            helper_upper_bound: trace.helper_upper_bound,
            a_eq_k_tables: trace.a_eq_k_tables.into(),
            e_lt_2k_plus_1_tables: trace.e_lt_2k_plus_1_tables.into(),
            e_eq_2k_tables: trace.e_eq_2k_tables.into(),
        }
    }
}

impl<F: DecomposableField> ModulusSwitchingTraceMLE<F> {
    pub fn extract_output_eq_output_witness_trace(&self) -> IndexedLookupTraceMLE<F> {
        let modulus_after: usize = self.modulus_after.value().as_into();
        assert!(modulus_after.is_power_of_two());
        let num_table_vars = modulus_after.trailing_zeros() as usize;

        // table: [q, 1, ..., q-1]
        let mut table = Vec::with_capacity(modulus_after);
        table.push(self.modulus_after);
        table.extend((1..modulus_after).map(|i| F::new(i.as_into())));
        let table = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_table_vars,
            table,
        ));

        let input = Rc::clone(&self.output_witness);
        let index = Rc::clone(&self.output);

        IndexedLookupTraceMLE {
            num_table_vars,
            num_input_vars: self.log_num,
            index,
            input,
            table,
            table_point: None,
        }
    }

    pub fn extract_helper_lt_2k_plus_1(&self) -> LTTraceMLE<F> {
        LTTraceMLE::from(&self.helper, &self.e_lt_2k_plus_1_tables)
    }

    pub fn extract_a_eq_k_trace(&self) -> EQTraceMLE<F> {
        EQTraceMLE::from(&self.input, &self.a_eq_k_tables)
    }

    pub fn extract_e_eq_2k_trace(&self) -> EQTraceMLE<F> {
        EQTraceMLE::from(&self.helper, &self.e_eq_2k_tables)
    }
}
