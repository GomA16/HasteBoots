// trace for compute x < c where c <= the prime

use std::rc::Rc;

use algebra::{
    AbstractExtensionField, AsInto, Basis, DecomposableField, DenseMultilinearExtension, Field,
};

use crate::ConvertToEF;

// Comparison table for x_i < g_i and x_i == g_i
// Construct the table of size 2^len where len = basis.bits().:
// For each table corresponding to gadget i (denoted by g_i), we compute lt_i = (x_i < g_i) and z_i = (x_i == g_i).
// The two bits are encoded as follows:
// 0 0 0 ... lt_i ... 0 0 0 || 0 0 0 ... z_i ... 0 0 0
// Hence, the i-th table has 2^(len) entries but only 3 different values:
// [0, p_i) for lt_i = 1, z_i = 0: 1 << k + i
// [p_i, p_i + 1) for lt_i = 0, z_i = 1: 1 << i
// [p_i + 1, 2^len) for lt_i = 0, z_i = 0: 0
pub struct LTGeneralTable<F: Field> {
    // denoted by len
    pub num_table_vars: usize,
    // denoted by k
    pub num_bits: usize,
    // denoted by g_i
    pub bit_constant: F,
    // denoted by i
    pub bit_position: usize,
    // [0, p_i) for lt_i = 1, z_i = 0: 1 << k + i
    pub less_than_codeword: F,
    // [p_i, p_i + 1) for lt_i = 0, z_i = 1: 1 << i
    pub equal_codeword: F,
    // [p_i + 1, 2^len) for lt_i = 0, z_i = 0: 0
    pub greater_than_codeword: F,
    pub table: Vec<F>,
}

pub struct LTGeneralTables<F: Field> {
    pub lt_constant: Option<F>,
    pub basis_bits: usize,
    pub decomp_len: usize,
    pub tables: Vec<LTGeneralTable<F>>,
}

pub struct LTGeneralTablesMLE<F: Field> {
    // pub lt_constant: Option<F>,
    pub basis_bits: usize,
    pub decomp_len: usize,
    pub tables: Vec<Rc<DenseMultilinearExtension<F>>>,
}

impl<F: Field> LTGeneralTable<F> {
    pub fn new(
        num_table_vars: usize,
        bit_constant: F,
        bit_position: usize,
        num_bits: usize,
    ) -> Self {
        let table_size = 1 << num_table_vars;
        let mut table = vec![F::zero(); table_size];
        let less_than_codeword = F::new((1 << (num_bits + bit_position)).as_into());
        let equal_codeword = F::new((1 << bit_position).as_into());
        let greater_than_codeword = F::zero();
        let gadget_pivot: usize = bit_constant.value().as_into();
        table[..gadget_pivot].iter_mut().for_each(|v| {
            *v = less_than_codeword;
        });
        table[gadget_pivot] = equal_codeword;

        Self {
            num_table_vars,
            num_bits,
            bit_constant,
            bit_position,
            less_than_codeword,
            equal_codeword,
            greater_than_codeword,
            table,
        }
    }
}

impl<F: DecomposableField> LTGeneralTables<F> {
    pub fn new(
        basis: &Basis<F>,
        // If None, the table is constructed for x < p.
        lt_constant: Option<F>,
    ) -> Self {
        let mut decomposed_constant = match lt_constant {
            Some(c) => c,
            // p - 1
            None => -F::one(),
        };

        let mut num_bits = 0;
        let mut bit_constants = Vec::with_capacity(basis.decompose_len() as usize);
        while !decomposed_constant.is_zero() {
            let bit = decomposed_constant.decompose_lsb_bits(basis.mask(), basis.bits());
            bit_constants.push(bit);
            num_bits += 1;
        }

        if lt_constant.is_none() {
            // if no constant is provided, we set it to p - 1, so we need to add 1 to the LSB gadget
            bit_constants[0] += F::one();
        }

        // Construct the table of size 2^len where len = basis.bits().:
        // For each table corresponding to gadget i (denoted by g_i), we compute lt_i = (x_i < g_i) and z_i = (x_i == g_i).
        // The two bits are encoded as follows:
        // 0 0 0 ... lt_i ... 0 0 0 || 0 0 0 ... z_i ... 0 0 0
        // Hence, the i-th table has 2^(2 * basis.bits()) entries but only 3 different values:
        // [0, p_i) for lt_i = 1, z_i = 0: 1 << k + i
        // [p_i, p_i + 1) for lt_i = 0, z_i = 1: 1 << i
        // [p_i + 1, 2^len) for lt_i = 0, z_i = 0: 0
        let tables = bit_constants
            .iter()
            .enumerate()
            .map(|(i, &gadget_constant)| {
                LTGeneralTable::new(basis.bits() as usize, gadget_constant, i, num_bits)
            })
            .collect::<Vec<_>>();

        Self {
            lt_constant,
            basis_bits: basis.bits() as usize,
            decomp_len: num_bits,
            tables,
        }
    }
}

impl<F: Field> LTGeneralTablesMLE<F> {
    pub fn lookup(
        &self,
        table_index: usize,
        index: &Rc<DenseMultilinearExtension<F>>,
    ) -> Rc<DenseMultilinearExtension<F>> {
        let table = &self.tables[table_index];
        let evaluations = index
            .iter()
            .map(|&idx| {
                let table_index: usize = idx.value().as_into();
                table.evaluations[table_index]
            })
            .collect::<Vec<_>>();
        Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            index.num_vars(),
            evaluations,
        ))
    }

    pub fn get_table(&self, table_index: usize) -> Rc<DenseMultilinearExtension<F>> {
        Rc::clone(&self.tables[table_index])
    }
}

impl<F: Field> From<LTGeneralTables<F>> for LTGeneralTablesMLE<F> {
    fn from(tables: LTGeneralTables<F>) -> Self {
        let mle_tables = tables
            .tables
            .iter()
            .map(|table| {
                Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                    table.num_table_vars,
                    table.table.clone(),
                ))
            })
            .collect::<Vec<_>>();

        Self {
            basis_bits: tables.basis_bits,
            decomp_len: tables.decomp_len,
            tables: mle_tables,
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for LTGeneralTablesMLE<F> {
    type Output = LTGeneralTablesMLE<EF>;
    fn to_ef(&self) -> LTGeneralTablesMLE<EF> {
        let mle_tables = self
            .tables
            .iter()
            .map(|table| Rc::new(table.to_ef()))
            .collect::<Vec<_>>();

        LTGeneralTablesMLE {
            basis_bits: self.basis_bits,
            decomp_len: self.decomp_len,
            tables: mle_tables,
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
        let basis = Basis::<FF>::new(7);
        let cmp_table_trace = LTGeneralTables::<FF>::new(&basis, None);
        println!("RHS constant: {:b}", (-FF::one()).value() + 1);
        cmp_table_trace.tables.iter().for_each(|table| {
            println!(
                "{:07b}: {}",
                table.bit_constant.value(),
                table.bit_constant.value()
            );
        });

        println!("");
        println!(
            "RHS constant: {:b}",
            (-FF::one() / FF::from(1 << 10)).value()
        );
        cmp_table_trace.tables.iter().for_each(|table| {
            println!(
                "{:07b}: {}",
                table.bit_constant.value(),
                table.bit_constant.value()
            );
        });
    }
}
