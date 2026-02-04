use std::rc::Rc;

use algebra::{
    AbstractExtensionField, AsInto, Basis, DecomposableField, DenseMultilinearExtension, Field,
};

use crate::{ConvertToEF, lookup_trace::indexed_table::IndexedLookupTraceMLE};

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
    pub fn new(eq_constant: F, basis: &Basis<F>) -> Self {
        let mut decomposed_constant = eq_constant;

        let mut tables = Vec::with_capacity(basis.decompose_len());
        let mask = basis.mask();
        let basis_bits = basis.bits() as usize;
        for i in 0..basis.decompose_len() {
            let bit_constant =
                (&mut decomposed_constant).decompose_lsb_bits(mask, basis_bits as u32);
            let table = EQTable::new(basis_bits, bit_constant, i);
            tables.push(table);
        }

        EQTables {
            eq_constant,
            basis_bits,
            decomp_len: basis.decompose_len(),
            tables,
        }
    }
}

impl<F: Field> EQTablesMLE<F> {
    pub fn get_table(&self, index: usize) -> Rc<DenseMultilinearExtension<F>> {
        Rc::clone(&self.tables[index])
    }
}

impl<F: Field> From<EQTables<F>> for EQTablesMLE<F> {
    fn from(tables: EQTables<F>) -> Self {
        let eq_constant = tables.eq_constant;
        let basis_bits = tables.basis_bits;
        let decomp_len = tables.decomp_len;
        let mle_tables = tables
            .tables
            .into_iter()
            .map(|table| {
                Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                    table.num_table_vars,
                    table.table,
                ))
            })
            .collect();

        EQTablesMLE {
            eq_constant,
            basis_bits,
            decomp_len,
            tables: mle_tables,
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

        input[0] = eq_tables.eq_constant; // ensure at least one equal case

        let mut bits = vec![vec![F::zero(); input_size]; num_bits];
        let mut bit_eq = vec![vec![F::zero(); input_size]; num_bits];
        let mut eq_result = vec![F::one(); input_size];

        for i in 0..input_size {
            let mut x = input[i];

            for (j, eq_table) in eq_tables.tables.iter().enumerate() {
                let bit = (&mut x).decompose_lsb_bits(
                    F::mask(eq_tables.basis_bits as u32),
                    eq_tables.basis_bits as u32,
                );
                bits[j][i] = bit;
                let table_index: usize = bit.value().as_into();
                let eq_bit = eq_table.table[table_index];
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

impl<F: DecomposableField> EQTraceMLE<F> {
    pub fn from(input: &Rc<DenseMultilinearExtension<F>>, eq_tables: &EQTablesMLE<F>) -> Self {
        let num_vars = input.num_vars();
        let num_bits = eq_tables.decomp_len;
        let input_size = 1 << num_vars;

        let mut bits = vec![vec![F::zero(); input_size]; num_bits];
        let mut bit_eq = vec![vec![F::zero(); input_size]; num_bits];
        let mut eq_result_evals = vec![F::one(); 1 << num_vars];

        input
            .evaluations
            .iter()
            .zip(eq_result_evals.iter_mut())
            .enumerate()
            .for_each(|(i, (input, eq_result))| {
                *eq_result = match *input == eq_tables.eq_constant {
                    true => F::one(),
                    false => F::zero(),
                };
                let mut x = *input;
                eq_tables.tables.iter().enumerate().for_each(|(j, table)| {
                    let bit = (&mut x).decompose_lsb_bits(
                        F::mask(eq_tables.basis_bits as u32),
                        eq_tables.basis_bits as u32,
                    );
                    bits[j][i] = bit;
                    let table_index: usize = bits[j][i].value().as_into();
                    bit_eq[j][i] = table.evaluations[table_index];
                });
            });

        let eq_result = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            eq_result_evals,
        ));
        let bits = bits
            .into_iter()
            .map(|b| Rc::new(DenseMultilinearExtension::from_evaluations_vec(num_vars, b)))
            .collect();
        let bit_eq = bit_eq
            .into_iter()
            .map(|b| Rc::new(DenseMultilinearExtension::from_evaluations_vec(num_vars, b)))
            .collect();

        EQTraceMLE {
            num_vars,
            num_bits,
            input: Rc::clone(input),
            eq_result,
            bits,
            bit_eq,
        }
    }

    pub fn extract_eq_lookup_traces(
        &self,
        eq_tables: &EQTablesMLE<F>,
    ) -> Vec<IndexedLookupTraceMLE<F>> {
        self.bits
            .iter()
            .zip(self.bit_eq.iter())
            .enumerate()
            .map(|(i, (bits, bit_eq))| IndexedLookupTraceMLE {
                num_input_vars: self.num_vars,
                num_table_vars: eq_tables.tables[i].num_vars(),
                index: Rc::clone(bits),
                input: Rc::clone(bit_eq),
                table: eq_tables.get_table(i),
                table_point: None,
            })
            .collect()
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for EQTraceMLE<F> {
    type Output = EQTraceMLE<EF>;
    fn to_ef(&self) -> Self::Output {
        let input = Rc::new(self.input.to_ef());
        let eq_result = Rc::new(self.eq_result.to_ef());
        let bits = self.bits.iter().map(|b| Rc::new(b.to_ef())).collect();
        let bit_eq = self.bit_eq.iter().map(|b| Rc::new(b.to_ef())).collect();

        EQTraceMLE {
            num_vars: self.num_vars,
            num_bits: self.num_bits,
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
    use algebra::BabyBear;
    use num_traits::One;

    // field type
    type FF = BabyBear;

    #[test]
    fn test_cmp_table_trace() {
        let basis = Basis::<FF>::new(7);
        let eq_tables = EQTables::<FF>::new(-FF::one(), &basis);
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
