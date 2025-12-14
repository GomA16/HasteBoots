use algebra::{AbstractExtensionField, AsInto};
use algebra::{DenseMultilinearExtension, Field};
use helper::utils::batch_inverse;
use itertools::Itertools;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use std::{collections::HashMap, rc::Rc};

use crate::{ConvertToEF, PackableTrace};
use log::info;

// Conversion Chain: LookupTrace => LookupTraceMLE => LookupWitness
// LookupWitnessHelper is computed from LookupWitness with a random value

/// Lookup trace specifically for range checking
#[derive(Clone)]
pub struct LookupTrace<F: Field> {
    pub num_vars: usize,
    pub num_vec: usize,
    pub vec_input: Vec<Vec<F>>,
    pub range: usize,
}

#[derive(Clone)]
pub struct LookupTraceMLE<F: Field> {
    pub num_vars: usize,
    pub range: usize,
    pub vec_input: Vec<Rc<DenseMultilinearExtension<F>>>,
    // pub table: Rc<DenseMultilinearExtension<F>>,
    // pub multiplicity: Rc<DenseMultilinearExtension<F>>,
}

#[derive(Clone)]
pub struct LookupWitness<F: Field> {
    pub num_vars: usize,
    pub trace: LookupTraceMLE<F>,
    pub table: Rc<DenseMultilinearExtension<F>>,
    pub multiplicity: Rc<DenseMultilinearExtension<F>>,
}

#[derive(Clone)]
pub struct LookupWitnessHelper<F: Field> {
    pub block_size: usize,
    pub num_blocks: usize,
    pub randomness: F,
    pub helper_functions: Vec<Rc<DenseMultilinearExtension<F>>>,
    // only for efficiency of prover
    pub phi_functions: Vec<Rc<DenseMultilinearExtension<F>>>,
}

impl<F: Field> LookupTrace<F> {
    pub fn random<R: rand::Rng + rand::CryptoRng>(
        rng: &mut R,
        num_vars: usize,
        num_vec: usize,
        range: usize,
    ) -> Self {
        let size = 1 << num_vars;
        let vec_input = (0..num_vec)
            .map(|_| {
                (0..size)
                    .map(|_| F::new(rng.random_range(0..range).as_into()))
                    .collect::<Vec<F>>()
            })
            .collect::<Vec<Vec<F>>>();
        Self {
            num_vars,
            num_vec,
            vec_input,
            range,
        }
    }
}

impl<F: Field> From<LookupTrace<F>> for LookupTraceMLE<F> {
    #[inline]
    fn from(trace: LookupTrace<F>) -> Self {
        assert!(trace.range <= 1 << trace.num_vars);
        Self {
            num_vars: trace.num_vars,
            range: trace.range,
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
        }
    }
}

impl<F: Field> From<LookupTraceMLE<F>> for LookupWitness<F> {
    #[inline]
    fn from(trace: LookupTraceMLE<F>) -> Self {
        assert!(trace.range <= 1 << trace.num_vars);

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

        LookupWitness {
            num_vars: trace.num_vars,
            trace: trace.clone(),
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

// impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for LookupTraceMLE<F> {
//     type Output = LookupTraceMLE<EF>;
//     fn into_ef(self) -> Self::Output {
//         assert!(false, "into_ef not supported for LookupTraceMLE");
//         unimplemented!()
//     }

//     fn to_ef(&self) -> Self::Output {
//         LookupTraceMLE {
//             num_vars: self.num_vars,
//             range: self.range,
//             vec_input: self
//                 .vec_input
//                 .iter()
//                 .map(|input| Rc::new(input.to_ef()))
//                 .collect(),
//         }
//     }
// }

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for LookupTraceMLE<F> {
    type Output = LookupTraceMLE<EF>;
    fn into_ef(self) -> Self::Output {
        assert!(false, "into_ef not supported for LookupTraceMLE");
        unimplemented!()
    }

    fn to_ef(&self) -> Self::Output {
        LookupTraceMLE {
            num_vars: self.num_vars,
            range: self.range,
            vec_input: self
                .vec_input
                .iter()
                .map(|input| Rc::new(input.to_ef()))
                .collect(),
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for LookupWitness<F> {
    type Output = LookupWitness<EF>;
    fn into_ef(self) -> Self::Output {
        assert!(false, "into_ef not supported for LookupWitness");
        unimplemented!()
    }

    fn to_ef(&self) -> Self::Output {
        LookupWitness {
            num_vars: self.num_vars,
            trace: self.trace.to_ef(),
            table: Rc::new(self.table.to_ef()),
            multiplicity: Rc::new(self.multiplicity.to_ef()),
        }
    }
}

impl<F: Field> LookupWitness<F> {
    pub fn compute_helper_functions(
        &self,
        block_size: usize,
        randomness: F,
    ) -> LookupWitnessHelper<F> {
        // divide vec_input || table into blocks of size block_size
        let num_blocks = (self.trace.vec_input.len() + block_size) / block_size;

        // t(x) + r and f(x) + r
        let table_and_inputs = self
            .table
            .iter()
            .chain(self.trace.vec_input.iter().flat_map(|input| input.iter()))
            .map(|&x| x + randomness)
            .collect::<Vec<F>>();

        let num_threads = rayon::current_num_threads();
        info!("Computing helper functions using {} threads", num_threads);
        let chunk_size = table_and_inputs.len() / num_threads;

        // 1 / (t(x) + r) and 1 / (f(x) + r)
        let mut inversed_values = table_and_inputs
            .par_chunks(chunk_size)
            .map(|chunk| batch_inverse(chunk))
            .flatten()
            .collect::<Vec<F>>();

        // -1 / (t(x) + r) and -1 / (f(x) + r)
        inversed_values.iter_mut().for_each(|x| *x = -*x);

        // m(x) / (t(x) + r)
        for (t_i, m_i) in inversed_values
            .iter_mut()
            .take(1 << self.num_vars)
            .zip(self.multiplicity.iter())
        {
            *t_i *= -*m_i;
        }

        let chunks_in_helper_functions = inversed_values.chunks(block_size * (1 << self.num_vars));

        let add_assign = |acc: &mut [F], vec: &[F]| {
            for (a, b) in acc.iter_mut().zip(vec.iter()) {
                *a += *b;
            }
        };

        let helper_functions = chunks_in_helper_functions
            .map(|block| {
                block.chunks_exact(1 << self.num_vars).fold(
                    vec![F::zero(); 1 << self.num_vars],
                    |mut helper, one_mle| {
                        add_assign(&mut helper, one_mle);
                        helper
                    },
                )
            })
            .collect::<Vec<_>>();

        let phi = table_and_inputs
            .into_iter()
            .chunks(1 << self.num_vars)
            .into_iter()
            .map(|chunk| chunk.collect::<Vec<_>>())
            .collect::<Vec<_>>();

        LookupWitnessHelper {
            block_size,
            num_blocks,
            randomness,
            helper_functions: helper_functions
                .into_iter()
                .map(|hf| {
                    Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                        self.num_vars,
                        hf,
                    ))
                })
                .collect(),
            phi_functions: phi
                .into_iter()
                .map(|phi| {
                    Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                        self.num_vars,
                        phi,
                    ))
                })
                .collect(),
        }
    }
}

impl<F: Field> PackableTrace<F> for LookupWitness<F> {
    fn num_oracles(&self) -> usize {
        self.trace.vec_input.len() + 2
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn pack_to_vec(&self) -> Vec<F> {
        self.trace
            .vec_input
            .iter()
            .flat_map(|input: &Rc<DenseMultilinearExtension<F>>| input.iter())
            .chain(self.table.iter())
            .chain(self.multiplicity.iter())
            .cloned()
            .collect::<Vec<F>>()
    }
}

impl<F: Field> PackableTrace<F> for LookupWitnessHelper<F> {
    fn num_oracles(&self) -> usize {
        self.helper_functions.len()
    }

    fn num_vars(&self) -> usize {
        self.helper_functions[0].num_vars
    }

    fn pack_to_vec(&self) -> Vec<F> {
        self.helper_functions
            .iter()
            .flat_map(|input| input.iter())
            .cloned()
            .collect::<Vec<F>>()
    }
}

// impl<F: Field> LookupTraceMLE<F> {
//     pub fn compute_multiplicity(&self) -> LookupWitness<F> {
//         assert!(self.range <= 1 << self.num_vars);

//         let num_padding = (1 << self.num_vars) - self.range;
//         let factor_for_padding_element = F::new((num_padding as u32 + 1).as_into());

//         let mut multiplicity_hashmap = HashMap::new();

//         self.vec_input.iter().for_each(|input| {
//             input.iter().for_each(|&elem| {
//                 multiplicity_hashmap
//                     .entry(elem)
//                     .and_modify(|cnt| *cnt += 1u32)
//                     .or_insert(1u32);
//             });
//         });

//         // compute multiplicity
//         let mut multiplicity = vec![F::zero(); 1 << self.num_vars];
//         let mut table = vec![F::zero(); 1 << self.num_vars];
//         let mut ele = F::zero();

//         for (t_i, m_i) in table
//             .iter_mut()
//             .take(self.range)
//             .zip(multiplicity.iter_mut().take(self.range))
//         {
//             *t_i = ele;
//             let count = multiplicity_hashmap.remove(&ele).unwrap_or(0u32);
//             *m_i = F::new((count as u32).as_into());
//             ele += F::one();
//         }

//         // normalize the multiplicity for the padding element
//         if num_padding > 0 {
//             let multiplicity_of_zero = multiplicity[0];
//             let multi_normalized = multiplicity_of_zero / factor_for_padding_element;
//             multiplicity[0] = multi_normalized;
//             for (t_i, m_i) in table
//                 .iter_mut()
//                 .skip(self.range)
//                 .zip(multiplicity.iter_mut().skip(self.range))
//             {
//                 *t_i = F::zero();
//                 *m_i = multi_normalized;
//             }
//         }

//         LookupWitness {
//             num_vars: self.num_vars,
//             trace: self.clone(),
//             table: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
//                 self.num_vars,
//                 table,
//             )),
//             multiplicity: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
//                 self.num_vars,
//                 multiplicity,
//             )),
//         }
//     }

//     pub fn compute_helper_functions(
//         &self,
//         witness: &LookupWitness<F>,
//         block_size: usize,
//         randomness: F,
//     ) -> LookupWitnessHelper<F> {
//         assert_eq!(witness.table.num_vars, self.num_vars);
//         assert_eq!(witness.multiplicity.num_vars, self.num_vars);

//         // divide vec_input || table into blocks of size block_size
//         // let num_blocks = (self.vec_input.len() + 1 + block_size - 1) / block_size;
//         let num_blocks = (self.vec_input.len() + block_size) / block_size;

//         // t(x) + r and f(x) + r
//         let table_and_inputs = witness
//             .table
//             .iter()
//             .chain(self.vec_input.iter().flat_map(|input| input.iter()))
//             .map(|&x| x + randomness)
//             .collect::<Vec<F>>();

//         let num_threads = rayon::current_num_threads();
//         info!("Computing helper functions using {} threads", num_threads);
//         let chunk_size = table_and_inputs.len() / num_threads;

//         // 1 / (t(x) + r) and 1 / (f(x) + r)
//         let mut inversed_values = table_and_inputs
//             .par_chunks(chunk_size)
//             .map(|chunk| batch_inverse(chunk))
//             .flatten()
//             .collect::<Vec<F>>();

//         // -1 / (t(x) + r) and -1 / (f(x) + r)
//         inversed_values.iter_mut().for_each(|x| *x = -*x);

//         // m(x) / (t(x) + r)
//         for (t_i, m_i) in inversed_values
//             .iter_mut()
//             .take(1 << self.num_vars)
//             .zip(witness.multiplicity.iter())
//         {
//             *t_i *= -*m_i;
//         }

//         let chunks_in_helper_functions = inversed_values.chunks(block_size * (1 << self.num_vars));

//         let add_assign = |acc: &mut [F], vec: &[F]| {
//             for (a, b) in acc.iter_mut().zip(vec.iter()) {
//                 *a += *b;
//             }
//         };

//         let helper_functions = chunks_in_helper_functions
//             .map(|block| {
//                 block.chunks_exact(1 << self.num_vars).fold(
//                     vec![F::zero(); 1 << self.num_vars],
//                     |mut helper, one_mle| {
//                         add_assign(&mut helper, one_mle);
//                         helper
//                     },
//                 )
//             })
//             .collect::<Vec<_>>();

//         let phi = table_and_inputs
//             .into_iter()
//             .chunks(1 << self.num_vars)
//             .into_iter()
//             .map(|chunk| chunk.collect::<Vec<_>>())
//             .collect::<Vec<_>>();

//         LookupWitnessHelper {
//             block_size,
//             num_blocks,
//             randomness,
//             helper_functions: helper_functions
//                 .into_iter()
//                 .map(|hf| {
//                     Rc::new(DenseMultilinearExtension::from_evaluations_vec(
//                         self.num_vars,
//                         hf,
//                     ))
//                 })
//                 .collect(),
//             phi_functions: phi
//                 .into_iter()
//                 .map(|phi| {
//                     Rc::new(DenseMultilinearExtension::from_evaluations_vec(
//                         self.num_vars,
//                         phi,
//                     ))
//                 })
//                 .collect(),
//         }
//     }
// }
