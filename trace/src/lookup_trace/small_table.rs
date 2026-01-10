//! Compared to normal_table.rs, this implementation is optimized for small lookup tables.
//! We directly use vectors to store the lookup table and multiplicity information instead of
//! DenseMultilinearExtension.
//! We don't need to commit multiplicity and the table separately, as they will be sent as part
//! of the proof.
use algebra::{AbstractExtensionField, AsInto};
use algebra::{DenseMultilinearExtension, Field};
use core::num;
use helper::utils::batch_inverse;
use itertools::Itertools;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use rayon::vec;
use serde::Serialize;
use std::sync::Arc;
use std::{collections::HashMap, rc::Rc};

use crate::{ConvertToEF, EvaluableTrace, EvaluableTraceEF, PackableEval, PackableTrace};
use log::{debug, info};

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
}

pub struct LookupTraceEval<F: Field> {
    pub num_vars: usize,
    pub range: usize,
    pub vec_input_at_r: Vec<F>,
}

pub struct LookupWitnessPure<F: Field> {
    pub num_vars: usize,
    pub table: Vec<F>,
    pub multiplicity: Vec<F>,
}

#[derive(Clone)]
pub struct LookupWitnessHelper<F: Field> {
    pub block_size: usize,
    pub num_blocks: usize,
    pub random_value: F,
    // sum over m(x) / (t(x) + r)
    pub sum: F,
    pub helper_functions: Vec<Rc<DenseMultilinearExtension<F>>>,
    // only for efficiency of prover
    pub phi_functions: Vec<Rc<DenseMultilinearExtension<F>>>,
}

#[derive(Serialize)]
pub struct LookupWitnessHelperEval<F: Field> {
    pub block_size: usize,
    pub num_blocks: usize,
    pub random_value: F,
    pub sum: F,
    pub helper_functions_at_r: Vec<F>,
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
        // assert!(trace.range <= 1 << trace.num_vars);
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

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for LookupTraceMLE<F> {
    type Output = LookupTraceMLE<EF>;

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

impl<F: Field> LookupTraceMLE<F> {
    pub fn num_oracles(&self) -> usize {
        self.vec_input.len()
    }

    pub fn num_helper_oracles(&self, blk_size: usize) -> usize {
        let total = self.num_oracles();
        let num_blks = (total + blk_size - 1) / blk_size;
        num_blks
    }

    pub fn log_num_oracles(&self) -> usize {
        self.num_oracles().next_power_of_two().trailing_zeros() as usize
    }

    pub fn log_num_helper_oracles(&self, blk_size: usize) -> usize {
        self.num_helper_oracles(blk_size)
            .next_power_of_two()
            .trailing_zeros() as usize
    }

    pub fn compute_witness_pure(&self) -> LookupWitnessPure<F> {
        assert!(self.range.is_power_of_two());
        let range_bit_len = self.range.trailing_zeros() as usize;
        // let factor_for_padding_element = F::new((num_padding as u32 + 1).as_into());

        let mut multiplicity_hashmap = HashMap::new();

        self.vec_input.iter().for_each(|input| {
            input.iter().for_each(|&elem| {
                multiplicity_hashmap
                    .entry(elem)
                    .and_modify(|cnt| *cnt += 1u32)
                    .or_insert(1u32);
            });
        });

        // compute multiplicity
        let mut multiplicity = vec![F::zero(); 1 << range_bit_len];
        let mut table = vec![F::zero(); 1 << range_bit_len];
        let mut ele = F::zero();

        for (t_i, m_i) in table
            .iter_mut()
            .take(self.range)
            .zip(multiplicity.iter_mut().take(self.range))
        {
            *t_i = ele;
            let count = multiplicity_hashmap.remove(&ele).unwrap_or(0u32);
            *m_i = F::new((count as u32).as_into());
            ele += F::one();
        }

        LookupWitnessPure {
            num_vars: self.num_vars,
            table,
            multiplicity,
        }
    }

    pub fn compute_helper_functions(
        &self,
        block_size: usize,
        randomness: F,
        witness: &LookupWitnessPure<F>,
    ) -> LookupWitnessHelper<F> {
        let mle_size = 1 << self.num_vars;
        let blk_span = block_size << self.num_vars;

        // divide vec_input into blocks of size block_size
        let total = self.vec_input.len();
        let num_blocks = (total + block_size - 1) / block_size;

        // f(x) + r
        let all_inputs_plus_r = self
            .vec_input
            .iter()
            .flat_map(|input| input.iter())
            .map(|&x| x + randomness)
            .collect::<Vec<F>>();
        // t(x) + r
        let table_plus_r = witness
            .table
            .iter()
            .map(|&x| randomness + x)
            .collect::<Vec<F>>();

        let num_threads = rayon::current_num_threads();
        debug!("Computing helper functions using {} threads", num_threads);
        let chunk_size =
            std::cmp::max(1, (all_inputs_plus_r.len() + num_threads - 1) / num_threads);

        // 1 / (f(x) + r)
        let inversed_values = all_inputs_plus_r
            .par_chunks(chunk_size)
            .map(|chunk| batch_inverse(chunk))
            .flatten()
            .collect::<Vec<F>>();

        // 1 / (t(x) + r)
        let table_inversed_values = table_plus_r
            .par_chunks(chunk_size)
            .map(|chunk| batch_inverse(chunk))
            .flatten()
            .collect::<Vec<F>>();

        // sum = \sum m(x) / (t(x) + r)
        let mut sum = F::zero();
        for (t_i, m_i) in table_inversed_values
            .iter()
            .take(1 << self.num_vars)
            .zip(witness.multiplicity.iter())
        {
            sum += *t_i * *m_i;
        }

        let add_assign = |acc: &mut [F], vec: &[F]| {
            for (a, b) in acc.iter_mut().zip(vec.iter()) {
                *a += *b;
            }
        };

        let helper_functions = inversed_values
            .par_chunks(blk_span)
            .map(|block| {
                let mut acc = vec![F::zero(); mle_size];
                for one_mle in block.chunks_exact(mle_size) {
                    add_assign(&mut acc, one_mle);
                }
                acc
            })
            .collect::<Vec<_>>();

        // f(x) + r
        let phi = all_inputs_plus_r
            .chunks_exact(1 << self.num_vars)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();

        LookupWitnessHelper {
            block_size,
            num_blocks,
            random_value: randomness,
            sum,
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

    pub fn compute_helper_functions_ef<EF: AbstractExtensionField<F>>(
        &self,
        block_size: usize,
        randomness: EF,
        witness: &LookupWitnessPure<F>,
    ) -> LookupWitnessHelper<EF> {
        let mle_size = 1 << self.num_vars;
        let blk_span = block_size << self.num_vars;

        // divide vec_input into blocks of size block_size
        let total = self.vec_input.len();
        let num_blocks = (total + block_size - 1) / block_size;

        // f(x) + r
        let all_inputs_plus_r = self
            .vec_input
            .iter()
            .flat_map(|input| input.iter())
            .map(|&x| randomness + x)
            .collect::<Vec<EF>>();
        // t(x) + r
        let table_plus_r = witness
            .table
            .iter()
            .map(|&x| randomness + x)
            .collect::<Vec<EF>>();

        let num_threads = rayon::current_num_threads();
        debug!("Computing helper functions using {} threads", num_threads);
        let chunk_size =
            std::cmp::max(1, (all_inputs_plus_r.len() + num_threads - 1) / num_threads);

        // 1 / (f(x) + r)
        let inversed_values = all_inputs_plus_r
            .par_chunks(chunk_size)
            .map(|chunk| batch_inverse(chunk))
            .flatten()
            .collect::<Vec<EF>>();

        // 1 / (t(x) + r)
        let mut table_inversed_values = table_plus_r
            .par_chunks(chunk_size)
            .map(|chunk| batch_inverse(chunk))
            .flatten()
            .collect::<Vec<EF>>();

        // sum = \sum m(x) / (t(x) + r)
        let mut sum = EF::zero();
        for (t_i, m_i) in table_inversed_values
            .iter_mut()
            .take(1 << self.num_vars)
            .zip(witness.multiplicity.iter())
        {
            sum += *t_i * *m_i;
        }

        let add_assign = |acc: &mut [EF], vec: &[EF]| {
            for (a, b) in acc.iter_mut().zip(vec.iter()) {
                *a += *b;
            }
        };

        let helper_functions = inversed_values
            .par_chunks(blk_span)
            .map(|block| {
                let mut acc = vec![EF::zero(); mle_size];
                for one_mle in block.chunks_exact(mle_size) {
                    add_assign(&mut acc, one_mle);
                }
                acc
            })
            .collect::<Vec<_>>();

        // f(x) + r
        let phi = all_inputs_plus_r
            .chunks_exact(1 << self.num_vars)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();

        LookupWitnessHelper {
            block_size,
            num_blocks,
            random_value: randomness,
            sum,
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

impl<F: Field> PackableTrace<F> for LookupTraceMLE<F> {
    fn num_oracles(&self) -> usize {
        self.vec_input.len()
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn pack_to_vec(&self) -> Vec<F> {
        self.vec_input
            .iter()
            .flat_map(|input: &Rc<DenseMultilinearExtension<F>>| input.iter())
            .cloned()
            .collect::<Vec<F>>()
    }
}

impl<F: Field> EvaluableTrace<F> for LookupTraceMLE<F> {
    type TraceEval = LookupTraceEval<F>;
    fn evaluate(&self, point: &[F]) -> Self::TraceEval {
        let vec_input_at_r = self
            .vec_input
            .iter()
            .map(|input| input.evaluate(point))
            .collect::<Vec<F>>();
        LookupTraceEval {
            num_vars: self.num_vars,
            range: self.range,
            vec_input_at_r,
        }
    }

    fn evaluate_with_lookup(
        &self,
        point: &[F],
        hash_table: &algebra::ListOfProductsOfPolynomials<F>,
        eval_table: &[F],
    ) -> Self::TraceEval {
        let vec_input_at_r = self
            .vec_input
            .iter()
            .map(|input| hash_table.lookup_mle_eval(input, eval_table, point))
            .collect::<Vec<F>>();
        LookupTraceEval {
            num_vars: self.num_vars,
            range: self.range,
            vec_input_at_r,
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> EvaluableTraceEF<F, EF> for LookupTraceMLE<F> {
    type TraceMLEEF = LookupTraceMLE<EF>;
    type TraceEvalEF = LookupTraceEval<EF>;
    fn evaluate_ef(&self, point: &[EF]) -> Self::TraceEvalEF {
        let vec_input_at_r = self
            .vec_input
            .iter()
            .map(|input| input.evaluate_ext(point))
            .collect::<Vec<EF>>();
        LookupTraceEval {
            num_vars: self.num_vars,
            range: self.range,
            vec_input_at_r,
        }
    }

    fn evaluate_ef_with_lookup(
        &self,
        point: &[EF],
        trace_ef: &Self::TraceMLEEF,
        hash_table: &algebra::ListOfProductsOfPolynomials<EF>,
        eval_table: &[EF],
    ) -> Self::TraceEvalEF {
        let vec_input_at_r = self
            .vec_input
            .iter()
            .zip(trace_ef.vec_input.iter())
            .map(|(input, input_ef)| {
                hash_table.lookup_mle_eval_ef(input, input_ef, eval_table, point)
            })
            .collect::<Vec<EF>>();
        LookupTraceEval {
            num_vars: self.num_vars,
            range: self.range,
            vec_input_at_r,
        }
    }

    fn evaluate_ef_ntt_only(
        &self,
        eval: &mut Self::TraceEvalEF,
        point: &[EF],
        trace_ef: &Self::TraceMLEEF,
        hash_table: &algebra::ListOfProductsOfPolynomials<EF>,
        eval_table: &[EF],
    ) {
        unimplemented!()
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
            .flat_map(|input: &Rc<DenseMultilinearExtension<F>>| input.iter())
            .cloned()
            .collect::<Vec<F>>()
    }
}

impl<F: Field> EvaluableTrace<F> for LookupWitnessHelper<F> {
    type TraceEval = LookupWitnessHelperEval<F>;

    fn evaluate(&self, point: &[F]) -> Self::TraceEval {
        let helper_functions_at_r = self
            .helper_functions
            .iter()
            .map(|input| input.evaluate(point))
            .collect::<Vec<F>>();
        LookupWitnessHelperEval {
            block_size: self.block_size,
            num_blocks: self.num_blocks,
            random_value: self.random_value,
            sum: self.sum,
            helper_functions_at_r,
        }
    }

    fn evaluate_with_lookup(
        &self,
        point: &[F],
        hash_table: &algebra::ListOfProductsOfPolynomials<F>,
        eval_table: &[F],
    ) -> Self::TraceEval {
        let helper_functions_at_r = self
            .helper_functions
            .iter()
            .map(|hf| hash_table.lookup_mle_eval(hf, eval_table, point))
            .collect::<Vec<F>>();
        LookupWitnessHelperEval {
            block_size: self.block_size,
            num_blocks: self.num_blocks,
            random_value: self.random_value,
            sum: self.sum,
            helper_functions_at_r,
        }
    }
}

impl<F: Field> PackableEval<F> for LookupTraceEval<F> {
    fn num_evals(&self) -> usize {
        self.vec_input_at_r.len()
    }

    fn pack_to_vec(&self) -> Vec<F> {
        self.vec_input_at_r.clone()
    }

    fn pack_poly_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }

    fn pack_ntt_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }
}

impl<F: Field> PackableEval<F> for LookupWitnessHelperEval<F> {
    fn num_evals(&self) -> usize {
        self.helper_functions_at_r.len()
    }

    fn pack_to_vec(&self) -> Vec<F> {
        self.helper_functions_at_r.clone()
    }

    fn pack_poly_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }

    fn pack_ntt_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }
}
