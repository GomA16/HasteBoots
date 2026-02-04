// trace for compute x < c where c's next power of 2 > p.

use std::rc::Rc;

use algebra::{
    AbstractExtensionField, AsInto, Basis, DecomposableField, DenseMultilinearExtension, Field,
};

use crate::{ConvertToEF, lookup_trace::indexed_table::IndexedLookupTraceMLE};

// Comparison table for x_i < g_i and x_i == g_i
// Construct the table of size 2^len where len = basis.bits().:
// For each table corresponding to gadget i (denoted by g_i), we compute lt_i = (x_i < g_i) and z_i = (x_i == g_i).
// The two bits are encoded as follows:
// 0 0 0 ... lt_i ... 0 0 0 || 0 0 0 ... z_i ... 0 0 0
// Hence, the i-th table has 2^(len) entries but only 3 different values:
// [0, p_i) for lt_i = 1, z_i = 0: 1 << k + i
// [p_i, p_i + 1) for lt_i = 0, z_i = 1: 1 << i
// [p_i + 1, 2^len) for lt_i = 0, z_i = 0: 0
#[derive(Clone)]
pub struct LTTable<F: Field> {
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

#[derive(Clone)]
pub struct LTTables<F: Field> {
    pub lt_constant: Option<F>,
    pub basis_bits: usize,
    pub decomp_len: usize,
    pub tables: Vec<LTTable<F>>,
}

pub struct LTTablesMLE<F: Field> {
    pub lt_constant: Option<F>,
    pub basis_bits: usize,
    pub decomp_len: usize,
    pub tables: Vec<Rc<DenseMultilinearExtension<F>>>,
}

pub struct LTTrace<F: Field> {
    pub num_vars: usize,
    pub num_bits: usize,

    pub input: Vec<F>,
    pub lt_result: Vec<F>,
    pub bits: Vec<Vec<F>>,
    pub bit_lt: Vec<Vec<F>>,
}

pub struct LTTraceMLE<F: Field> {
    pub num_vars: usize,
    pub num_bits: usize,

    pub input: Rc<DenseMultilinearExtension<F>>,
    pub lt_result: Rc<DenseMultilinearExtension<F>>,
    pub bits: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub bit_lt: Vec<Rc<DenseMultilinearExtension<F>>>,
}

impl<F: Field> LTTable<F> {
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

    pub fn num_vars(&self) -> usize {
        self.num_table_vars
    }
}

impl<F: DecomposableField> LTTables<F> {
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

        let mut tables = Vec::with_capacity(basis.decompose_len());
        for i in 0..basis.decompose_len() {
            let mut bit_constant =
                (&mut decomposed_constant).decompose_lsb_bits(basis.mask(), basis.bits());
            if lt_constant.is_none() && i == 0 {
                // if no constant is provided, we set it to p - 1, so we need to add 1 to the LSB gadget
                bit_constant += F::one();
            }

            let table = LTTable::new(
                basis.bits() as usize,
                bit_constant,
                i,
                basis.decompose_len(),
            );
            tables.push(table);
        }

        Self {
            lt_constant,
            basis_bits: basis.bits() as usize,
            decomp_len: basis.decompose_len(),
            tables,
        }
    }

    pub fn get_table(&self, index: usize) -> &LTTable<F> {
        &self.tables[index]
    }
}

impl<F: Field> LTTablesMLE<F> {
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

impl<F: Field> From<LTTables<F>> for LTTablesMLE<F> {
    fn from(tables: LTTables<F>) -> Self {
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
            lt_constant: tables.lt_constant,
            basis_bits: tables.basis_bits,
            decomp_len: tables.decomp_len,
            tables: mle_tables,
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for LTTablesMLE<F> {
    type Output = LTTablesMLE<EF>;
    fn to_ef(&self) -> LTTablesMLE<EF> {
        let mle_tables = self
            .tables
            .iter()
            .map(|table| Rc::new(table.to_ef()))
            .collect::<Vec<_>>();

        LTTablesMLE {
            lt_constant: match self.lt_constant {
                Some(c) => Some(EF::from_base(c)),
                None => None,
            },
            basis_bits: self.basis_bits,
            decomp_len: self.decomp_len,
            tables: mle_tables,
        }
    }
}

impl<F: Field> From<LTTrace<F>> for LTTraceMLE<F> {
    fn from(trace: LTTrace<F>) -> Self {
        Self {
            num_vars: trace.num_vars,
            num_bits: trace.num_bits,
            input: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                trace.num_vars,
                trace.input,
            )),
            lt_result: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                trace.num_vars,
                trace.lt_result,
            )),
            bits: trace
                .bits
                .into_iter()
                .map(|b| {
                    Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                        trace.num_vars,
                        b,
                    ))
                })
                .collect(),
            bit_lt: trace
                .bit_lt
                .into_iter()
                .map(|b| {
                    Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                        trace.num_vars,
                        b,
                    ))
                })
                .collect(),
        }
    }
}

impl<F: DecomposableField> LTTraceMLE<F> {
    pub fn from(input: &Rc<DenseMultilinearExtension<F>>, lt_tables: &LTTablesMLE<F>) -> Self {
        let num_vars = input.num_vars();
        let num_bits = lt_tables.decomp_len;
        let input_size = 1 << num_vars;

        let mut bits = vec![vec![F::zero(); input_size]; num_bits];
        let mut bit_lt = vec![vec![F::zero(); input_size]; num_bits];
        let mut lt_result = vec![F::zero(); input_size];

        assert!(lt_tables.lt_constant.is_some());
        let lt_constant = lt_tables.lt_constant.unwrap();

        input
            .evaluations
            .iter()
            .zip(lt_result.iter_mut())
            .enumerate()
            .for_each(|(i, (input, lt_result))| {
                let mut x = *input;
                *lt_result = if x < lt_constant { F::one() } else { F::zero() };
                lt_tables
                    .tables
                    .iter()
                    .enumerate()
                    .for_each(|(j, lt_table)| {
                        let bit = (&mut x).decompose_lsb_bits(
                            F::mask(lt_tables.basis_bits as u32),
                            lt_tables.basis_bits as u32,
                        );
                        bits[j][i] = bit;
                        let table_index: usize = bit.value().as_into();
                        bit_lt[j][i] = lt_table.evaluations[table_index];
                    });
            });
        let lt_result = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, lt_result,
        ));
        let bits = bits
            .into_iter()
            .map(|b| Rc::new(DenseMultilinearExtension::from_evaluations_vec(num_vars, b)))
            .collect();
        let bit_lt = bit_lt
            .into_iter()
            .map(|b| Rc::new(DenseMultilinearExtension::from_evaluations_vec(num_vars, b)))
            .collect();
        LTTraceMLE {
            num_vars,
            num_bits,
            input: Rc::clone(input),
            lt_result,
            bits,
            bit_lt,
        }
    }

    pub fn extract_lt_lookup_traces(
        &self,
        lt_tables: &LTTablesMLE<F>,
    ) -> Vec<IndexedLookupTraceMLE<F>> {
        self.bits
            .iter()
            .zip(self.bit_lt.iter())
            .enumerate()
            .map(|(i, (bits, bit_lt))| IndexedLookupTraceMLE {
                num_input_vars: self.num_vars,
                num_table_vars: lt_tables.tables[i].num_vars(),
                index: Rc::clone(bits),
                input: Rc::clone(bit_lt),
                table: lt_tables.get_table(i),
                table_point: None,
            })
            .collect()
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
        let cmp_table_trace = LTTables::<FF>::new(&basis, None);
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
