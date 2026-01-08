// trace for compute x < c where c <= the prime

use algebra::{AsInto, Basis, DecomposableField, DenseMultilinearExtension, Field};

use crate::cmp_trace::table_trace;

// Comparison table for x_i < g_i and x_i == g_i
// Construct the table of size 2^len where len = basis.bits().:
// For each table corresponding to gadget i (denoted by g_i), we compute lt_i = (x_i < g_i) and z_i = (x_i == g_i).
// The two bits are encoded as follows:
// 0 0 0 ... lt_i ... 0 0 0 || 0 0 0 ... z_i ... 0 0 0
// Hence, the i-th table has 2^(len) entries but only 3 different values:
// [0, p_i) for lt_i = 1, z_i = 0: 1 << k + i
// [p_i, p_i + 1) for lt_i = 0, z_i = 1: 1 << i
// [p_i + 1, 2^len) for lt_i = 0, z_i = 0: 0
pub struct CMPTable<F: Field> {
    // denoted by len
    pub num_table_vars: usize,
    // denoted by k
    pub num_gadgets: usize,
    // denoted by g_i
    pub gadget_constant: F,
    // denoted by i
    pub gadget_position: usize,
    // [0, p_i) for lt_i = 1, z_i = 0: 1 << k + i
    pub less_than_codeword: F,
    // [p_i, p_i + 1) for lt_i = 0, z_i = 1: 1 << i
    pub equal_codeword: F,
    // [p_i + 1, 2^len) for lt_i = 0, z_i = 0: 0
    pub greater_than_codeword: F,
    pub table: Vec<F>,
}

pub struct LessThanPrimeTrace<F: DecomposableField> {
    pub rhs_constant: F,
    pub num_gadgets: usize,
    pub basis: Basis<F>,
    pub tables: Vec<CMPTable<F>>,
}

impl<F: Field> CMPTable<F> {
    pub fn new(
        num_table_vars: usize,
        gadget_constant: F,
        gadget_position: usize,
        num_gadgets: usize,
    ) -> Self {
        let table_size = 1 << num_table_vars;
        let mut table = vec![F::zero(); table_size];
        let less_than_codeword = F::new((1 << (num_gadgets + gadget_position)).as_into());
        let equal_codeword = F::new((1 << gadget_position).as_into());
        let greater_than_codeword = F::zero();
        let gadget_pivot: usize = gadget_constant.value().as_into();
        table[..gadget_pivot].iter_mut().for_each(|v| {
            *v = less_than_codeword;
        });
        table[gadget_pivot] = equal_codeword;

        Self {
            num_table_vars,
            num_gadgets,
            gadget_constant,
            gadget_position,
            less_than_codeword,
            equal_codeword,
            greater_than_codeword,
            table,
        }
    }
}

impl<F: DecomposableField> LessThanPrimeTrace<F> {
    pub fn new(
        basis: &Basis<F>,
        // If None, the table is constructed for x < p.
        rhs_constant: Option<F>,
    ) -> Self {
        let mut decomposed_constant = match rhs_constant {
            Some(c) => c,
            // p - 1
            None => -F::one(),
        };

        let mut num_gadgets = 0;
        let mut rhs_gadgets = Vec::with_capacity(basis.decompose_len() as usize);
        while !decomposed_constant.is_zero() {
            let gadget = decomposed_constant.decompose_lsb_bits(basis.mask(), basis.bits());
            rhs_gadgets.push(gadget);
            num_gadgets += 1;
        }

        if rhs_constant.is_none() {
            // if no constant is provided, we set it to p - 1, so we need to add 1 to the LSB gadget
            rhs_gadgets[0] += F::one();
        }

        // Construct the table of size 2^len where len = basis.bits().:
        // For each table corresponding to gadget i (denoted by g_i), we compute lt_i = (x_i < g_i) and z_i = (x_i == g_i).
        // The two bits are encoded as follows:
        // 0 0 0 ... lt_i ... 0 0 0 || 0 0 0 ... z_i ... 0 0 0
        // Hence, the i-th table has 2^(2 * basis.bits()) entries but only 3 different values:
        // [0, p_i) for lt_i = 1, z_i = 0: 1 << k + i
        // [p_i, p_i + 1) for lt_i = 0, z_i = 1: 1 << i
        // [p_i + 1, 2^len) for lt_i = 0, z_i = 0: 0
        let tables = rhs_gadgets
            .iter()
            .enumerate()
            .map(|(i, &gadget_constant)| {
                CMPTable::new(basis.bits() as usize, gadget_constant, i, num_gadgets)
            })
            .collect::<Vec<_>>();

        Self {
            rhs_constant: decomposed_constant,
            num_gadgets,
            basis: *basis,
            tables,
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
        let cmp_table_trace = LessThanPrimeTrace::<FF>::new(&basis, None);
        println!("RHS constant: {:b}", (-FF::one()).value() + 1);
        cmp_table_trace.tables.iter().for_each(|table| {
            println!(
                "{:07b}: {}",
                table.gadget_constant.value(),
                table.gadget_constant.value()
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
                table.gadget_constant.value(),
                table.gadget_constant.value()
            );
        });
    }
}
