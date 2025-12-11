use algebra::{AbstractExtensionField, AsFrom, AsInto};
use algebra::{DenseMultilinearExtension, Field};
use helper::utils::batch_inverse;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use rayon::vec;
use std::{collections::HashMap, rc::Rc};

use crate::ConvertToEF;
use helper::batch_inverse;
use log::{debug, info};

/// Lookup trace specifically for range checking
#[derive(Clone)]
pub struct LookupTrace<F: Field> {
    pub num_vars: usize,
    pub num_vec: usize,
    pub vec_input: Vec<Vec<F>>,
    pub range: usize,
}

pub enum TableKind<F: Field> {
    // range table: (range value)
    RangeTable(usize),
    NormalTable(Rc<DenseMultilinearExtension<F>>),
}

pub struct LookupTraceMLE<F: Field> {
    pub num_vars: usize,
    pub vec_input: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub table: Rc<DenseMultilinearExtension<F>>,
    pub multiplicity: Rc<DenseMultilinearExtension<F>>,
}

pub struct LookupWitness<F: Field> {
    pub block_size: usize,
    pub num_of_blocks: usize,
    pub helper_functions: Vec<Vec<F>>,
}

// implementation for BabyBear using the Montgomery representation
// TODO: optimize it without hashmap when using Goldilocks
impl<F: Field> From<LookupTrace<F>> for LookupTraceMLE<F> {
    #[inline]
    fn from(trace: LookupTrace<F>) -> Self {
        assert!(trace.range <= 1 << trace.num_vars);

        // let padding_element = F::zero();
        let num_padding = (1 << trace.num_vars) - trace.range;
        let factor_for_padding_element = F::new((num_padding as u32 + 1).as_into());

        let mut multiplicity_hashmap = HashMap::new();

        trace.vec_input.iter().for_each(|input| {
            input.iter().for_each(|&elem| {
                multiplicity_hashmap
                    .entry(elem)
                    .and_modify(|cnt| *cnt += 1u32)
                    .or_insert(1u32);
            });
        });

        // compute multiplicity
        let mut multiplicity = vec![F::zero(); 1 << trace.num_vars];
        let mut table = vec![F::zero(); 1 << trace.num_vars];
        let mut ele = F::zero();

        for (t_i, m_i) in table
            .iter_mut()
            .take(trace.range)
            .zip(multiplicity.iter_mut().take(trace.range))
        {
            *t_i = ele;
            let count = multiplicity_hashmap.remove(&ele).unwrap_or(0u32);
            *m_i = F::new((count as u32).as_into());
            ele += F::one();
        }

        // normalize the multiplicity for the padding element
        if num_padding > 0 {
            let multiplicity_of_zero = multiplicity[0];
            let multi_normalized = multiplicity_of_zero / factor_for_padding_element;
            multiplicity[0] = multi_normalized;
            for (t_i, m_i) in table
                .iter_mut()
                .skip(trace.range)
                .zip(multiplicity.iter_mut().skip(trace.range))
            {
                *t_i = F::zero();
                *m_i = multi_normalized;
            }
        }

        LookupTraceMLE {
            num_vars: trace.num_vars,
            vec_input: trace
                .vec_input
                .into_iter()
                .map(|input| {
                    Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                        trace.num_vars,
                        input,
                    ))
                })
                .collect(),
            table: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                trace.num_vars,
                table,
            )),
            multiplicity: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                trace.num_vars,
                multiplicity,
            )),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for LookupTraceMLE<F> {
    type Output = LookupTraceMLE<EF>;
    fn into_ef(self) -> Self::Output {
        assert!(false, "into_ef not supported for LookupTraceMLE");
        unimplemented!()
    }

    fn to_ef(&self) -> Self::Output {
        LookupTraceMLE {
            num_vars: self.num_vars,
            vec_input: self
                .vec_input
                .iter()
                .map(|input| Rc::new(input.to_ef()))
                .collect(),
            table: Rc::new(self.table.to_ef()),
            multiplicity: Rc::new(self.multiplicity.to_ef()),
        }
    }
}

impl<F: Field> LookupTraceMLE<F> {
    pub fn compute_helper_functions(&self, block_size: usize, randomness: F) -> LookupWitness<F> {
        assert_eq!(self.table.num_vars, self.num_vars);

        let num_blocks = (self.vec_input.len() + block_size) / block_size;
        let mut helper_functions = Vec::with_capacity(num_blocks);

        // t(x) + r and f(x) + r
        let table_and_inputs = self
            .table
            .iter()
            .chain(self.vec_input.iter().flat_map(|input| input.iter()))
            .map(|&x| x + randomness)
            .collect::<Vec<F>>();

        // let flatten_inputs_and_table = self
        //     .vec_input
        //     .iter()
        //     .flat_map(|input| input.iter())
        //     .chain(self.table.iter())
        //     .map(|&x| x + randomness)
        //     .collect::<Vec<F>>();

        let num_threads = rayon::current_num_threads();
        info!("Computing helper functions using {} threads", num_threads);
        let chunk_size = table_and_inputs.len() / num_threads;

        // 1 / (t(x) + r) and 1 / (f(x) + r)
        let mut inversed_values = table_and_inputs
            .par_chunks(chunk_size)
            .map(|chunk| batch_inverse(chunk))
            .flatten()
            .collect::<Vec<F>>();

        // m(x) / (t(x) + r)
        for (t_i, m_i) in inversed_values
            .iter_mut()
            .take(1 << self.num_vars)
            .zip(self.multiplicity.iter())
        {
            *t_i *= *m_i;
        }

        let chunks_in_helper_functions = inversed_values.chunks(block_size * (1 << self.num_vars));

        let add_assign = |acc: &mut [F], vec: &[F]| {
            for (a, b) in acc.iter_mut().zip(vec.iter()) {
                *a += *b;
            }
        };

        helper_functions = chunks_in_helper_functions
            .map(|block| {
                let helper_function = block.chunks_exact(1 << self.num_vars).fold(vec![F::zero(); 1 << self.num_vars], |mut helper, one_mle| {
                    add_assign(&mut helper, one_mle);
                    helper
                });
                
            }).collect();
        
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn hash_map() {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        fn random_stat_buff() -> u8 {
            42
        }

        let w = map.entry("two").or_insert(2);

        println!("two of map: {}", map.get("two").unwrap());
    }
}
