use std::rc::Rc;

use algebra::{AsInto, Basis, DecomposableField, DenseMultilinearExtension, Field};

#[derive(Clone)]
pub struct EQTable<F: Field> {
    pub num_table_vars: usize,
    pub bit_constant: F,
    pub bit_position: usize,
    pub table: Vec<F>,
}

#[derive(Clone)]
pub struct EQTables<F: Field> {
    pub eq_constant: F,
    pub basis_bits: usize,
    pub decomp_len: usize,
    pub tables: Vec<EQTable<F>>,
}

pub struct EQTablesMLE<F: Field> {
    pub eq_constant: F,
    pub basis_bits: usize,
    pub decomp_len: usize,
    pub tables: Vec<Rc<DenseMultilinearExtension<F>>>,
}

pub struct EQTrace<F: Field> {
    pub num_vars: usize,
    pub num_bits: usize,

    pub input: Vec<F>,
    pub eq_result: Vec<F>,
    pub bits: Vec<Vec<F>>,
    pub bit_eq: Vec<Vec<F>>,
}

pub struct EQTraceMLE<F: Field> {
    pub num_vars: usize,
    pub num_bits: usize,

    pub input: Rc<DenseMultilinearExtension<F>>,
    pub eq_result: Rc<DenseMultilinearExtension<F>>,
    pub bits: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub bit_eq: Vec<Rc<DenseMultilinearExtension<F>>>,
}

impl<F: Field> EQTable<F> {
    pub fn new(num_table_vars: usize, bit_constant: F, bit_position: usize) -> Self {
        let table_size = 1 << num_table_vars;
        let mut table = vec![F::zero(); table_size];
        let bit_pivot: usize = bit_constant.value().as_into();

        table[bit_pivot] = F::one();

        Self {
            num_table_vars,
            bit_constant,
            bit_position,
            table,
        }
    }
}

impl<F: DecomposableField> EQTables<F> {
    pub fn new(eq_constant: F, basis_bits: usize) -> Self {
        let mut decomposed_constant = eq_constant;
        let mut decomp_len = 0;
        let mut tables = Vec::new();
        while !decomposed_constant.is_zero() {
            let bit_constant = decomposed_constant
                .decompose_lsb_bits(F::mask(basis_bits as u32), basis_bits as u32);
            let table = EQTable::new(basis_bits, bit_constant, decomp_len);
            tables.push(table);
            decomp_len += 1;
        }
        EQTables {
            eq_constant,
            basis_bits,
            decomp_len,
            tables,
        }
    }
}

impl<F: DecomposableField> EQTrace<F> {
    pub fn random<R: rand::Rng + rand::CryptoRng>(
        rng: &mut R,
        num_vars: usize,
        eq_tables: &EQTables<F>,
    ) -> Self {
        let num_bits = eq_tables.decomp_len;
        let input_size = 1 << num_vars;
        let mut input = vec![F::zero(); input_size];
        for i in 0..input_size {
            input[i] = F::random(rng);
        }

        let mut bits = vec![vec![F::zero(); input_size]; num_bits];
        let mut bit_eq = vec![vec![F::zero(); input_size]; num_bits];
        let mut eq_result = vec![F::one(); input_size];

        for i in 0..input_size {
            let mut x = input[i];
            for (j, table) in eq_tables.tables.iter().enumerate() {
                let bit = x.decompose_lsb_bits(
                    F::mask(eq_tables.basis_bits as u32),
                    eq_tables.basis_bits as u32,
                );
                bits[j][i] = bit;
                let table_index: usize = bit.value().as_into();
                let eq_bit = table.table[table_index];
                bit_eq[j][i] = eq_bit;
                eq_result[i] *= eq_bit;
            }
        }

        EQTrace {
            num_vars,
            num_bits,
            input,
            eq_result,
            bits,
            bit_eq,
        }
    }
}

impl<F: Field> From<EQTrace<F>> for EQTraceMLE<F> {
    fn from(trace: EQTrace<F>) -> Self {
        let num_vars = trace.num_vars;
        let num_bits = trace.num_bits;

        let input = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            trace.input,
        ));
        let eq_result = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            trace.eq_result,
        ));
        let bits = trace
            .bits
            .into_iter()
            .map(|b| Rc::new(DenseMultilinearExtension::from_evaluations_vec(num_vars, b)))
            .collect();
        let bit_eq = trace
            .bit_eq
            .into_iter()
            .map(|b| Rc::new(DenseMultilinearExtension::from_evaluations_vec(num_vars, b)))
            .collect();

        EQTraceMLE {
            num_vars,
            num_bits,
            input,
            eq_result,
            bits,
            bit_eq,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use algebra::{BabyBear, derive::Field};
    use num_traits::One;

    // field type
    type FF = BabyBear;

    #[test]
    fn test_cmp_table_trace() {
        let basis_bits = 7;
        let eq_tables = EQTables::<FF>::new(-FF::one(), basis_bits);
        println!("RHS constant: {:b}", (-FF::one()).value());
        eq_tables.tables.iter().for_each(|table| {
            println!(
                "{:07b}: {}",
                table.bit_constant.value(),
                table.bit_constant.value()
            );
        });
    }
}
