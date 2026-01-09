use std::rc::Rc;

use algebra::{AsInto, Basis, DecomposableField, DenseMultilinearExtension, Field};

// When we prove the decomposition of x < p where p is the field, we compute x < p - 1 or x = p - 1.

// In this implementation, we only consider p - 1 in the form of 2^k1 - 2^k2 for k1 > k2 >= 0.
// (x < p - 1) <=> 1 - (x >= p - 1)

// When c = p - 1 which is in form of 111100000
// Each bit_constant is 11111...1, 11..10...0, and 000.00
// bit_gt_eq = 1 if x_i >= c_i
// \prod {bit_gt_eq} = 1 if and only if x >= c
// Hence, lt_result = 1 - \prod {bit_gt_eq}
#[derive(Clone)]
pub struct GTEQTable<F: Field> {
    pub num_table_vars: usize,
    pub bit_constant: F,
    pub bit_position: usize,
    pub table: Vec<F>,
}

#[derive(Clone)]
pub struct GTEQTables<F: Field> {
    pub gt_eq_constant: F,
    pub basis_bits: usize,
    pub decomp_len: usize,
    pub tables: Vec<GTEQTable<F>>,
}

pub struct GTEQTablesMLE<F: Field> {
    pub eq_constant: F,
    pub basis_bits: usize,
    pub decomp_len: usize,
    pub tables: Vec<Rc<DenseMultilinearExtension<F>>>,
}

pub struct LTTrace<F: Field> {
    pub num_vars: usize,
    pub num_bits: usize,

    pub input: Vec<F>,
    // lt_result = 1 - \prod {bit_gt_eq}
    pub lt_result: Vec<F>,
    pub bits: Vec<Vec<F>>,
    pub bit_gt_eq: Vec<Vec<F>>,
}

pub struct LTTraceMLE<F: Field> {
    pub num_vars: usize,
    pub num_bits: usize,

    pub input: Rc<DenseMultilinearExtension<F>>,
    pub lt_result: Rc<DenseMultilinearExtension<F>>,
    pub bits: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub bit_gt_eq: Vec<Rc<DenseMultilinearExtension<F>>>,
}

impl<F: Field> GTEQTable<F> {
    pub fn new(num_table_vars: usize, bit_constant: F, bit_position: usize) -> Self {
        let table_size = 1 << num_table_vars;
        let mut table = vec![F::zero(); table_size];
        let bit_pivot: usize = bit_constant.value().as_into();

        table[bit_pivot..].iter_mut().for_each(|t| *t = F::one());

        Self {
            num_table_vars,
            bit_constant,
            bit_position,
            table,
        }
    }
}

impl<F: DecomposableField> GTEQTables<F> {
    pub fn new(basis_bits: usize) -> Self {
        let mut decomposed_constant = -F::one();
        let mut decomp_len = 0;
        let mut tables = Vec::new();
        while !decomposed_constant.is_zero() {
            let bit_constant = decomposed_constant
                .decompose_lsb_bits(F::mask(basis_bits as u32), basis_bits as u32);
            let table = GTEQTable::new(basis_bits, bit_constant, decomp_len);
            tables.push(table);
            decomp_len += 1;
        }
        GTEQTables {
            gt_eq_constant: -F::one(),
            basis_bits,
            decomp_len,
            tables,
        }
    }
}

impl<F: DecomposableField> LTTrace<F> {
    pub fn random<R: rand::Rng + rand::CryptoRng>(
        rng: &mut R,
        num_vars: usize,
        gt_eq_tables: &GTEQTables<F>,
    ) -> Self {
        let num_bits = gt_eq_tables.decomp_len;
        let input_size = 1 << num_vars;
        let mut input = vec![F::zero(); input_size];
        for i in 0..input_size {
            input[i] = F::random(rng);
        }
        input[0] = gt_eq_tables.gt_eq_constant;

        let mut bits = vec![vec![F::zero(); input_size]; num_bits];
        let mut bit_gt_eq = vec![vec![F::zero(); input_size]; num_bits];
        let mut lt_result = vec![F::zero(); input_size];

        for i in 0..input_size {
            let mut x = input[i];
            for (j, table) in gt_eq_tables.tables.iter().enumerate() {
                let bit = x.decompose_lsb_bits(
                    F::mask(gt_eq_tables.basis_bits as u32),
                    gt_eq_tables.basis_bits as u32,
                );
                bits[j][i] = bit;
                let table_index: usize = bit.value().as_into();
                let gt_eq_bit = table.table[table_index];
                bit_gt_eq[j][i] = gt_eq_bit;
            }

            if input[i] < gt_eq_tables.gt_eq_constant {
                lt_result[i] = F::one();
            }
        }

        LTTrace {
            num_vars,
            num_bits,
            input,
            lt_result,
            bits,
            bit_gt_eq,
        }
    }
}

impl<F: Field> From<LTTrace<F>> for LTTraceMLE<F> {
    fn from(trace: LTTrace<F>) -> Self {
        let num_vars = trace.num_vars;
        let num_bits = trace.num_bits;

        let input = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            trace.input,
        ));
        let lt_result = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            trace.lt_result,
        ));
        let bits = trace
            .bits
            .into_iter()
            .map(|b| Rc::new(DenseMultilinearExtension::from_evaluations_vec(num_vars, b)))
            .collect();
        let bit_gt_eq = trace
            .bit_gt_eq
            .into_iter()
            .map(|b| Rc::new(DenseMultilinearExtension::from_evaluations_vec(num_vars, b)))
            .collect();

        LTTraceMLE {
            num_vars,
            num_bits,
            input,
            lt_result,
            bits,
            bit_gt_eq,
        }
    }
}
