use algebra::{AbstractExtensionField, AsInto};
use algebra::{DenseMultilinearExtension, Field};
use core::num;
use helper::utils::batch_inverse;
use itertools::Itertools;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use rayon::vec;
use std::sync::Arc;
use std::{collections::HashMap, rc::Rc};

use crate::{ConvertToEF, EvaluableTrace, EvaluableTraceEF, PackableEval, PackableTrace};
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
}

#[derive(Clone)]
pub struct LookupWitness<F: Field> {
    pub num_vars: usize,
    pub trace: LookupTraceMLE<F>,
    pub table: Rc<DenseMultilinearExtension<F>>,
    pub multiplicity: Rc<DenseMultilinearExtension<F>>,
}

pub struct LookupWitnessEval<F: Field> {
    pub vec_input_at_r: Vec<F>,
    pub table_at_r: F,
    pub multiplicity_at_r: F,
}

#[derive(Clone)]
pub struct LookupWitnessHelper<F: Field> {
    pub block_size: usize,
    pub num_blocks: usize,
    pub random_value: F,
    pub helper_functions: Vec<Rc<DenseMultilinearExtension<F>>>,
    // only for efficiency of prover
    pub phi_functions: Vec<Rc<DenseMultilinearExtension<F>>>,
}

pub struct LookupWitnessHelperEval<F: Field> {
    pub helper_at_r: Vec<F>,
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
    pub fn witness_num_vars(&self) -> usize {
        let num_oracles = self.vec_input.len() + 2;
        self.num_vars + num_oracles.next_power_of_two().trailing_zeros() as usize
    }

    pub fn compute_helper_num_vars(num_vars: usize, num_vec: usize, blk_size: usize) -> usize {
        let total = 1 + num_vec;
        let num_blks = (total + blk_size - 1) / blk_size;
        num_vars + num_blks.next_power_of_two().trailing_zeros() as usize
    }

    pub fn helper_num_vars(&self, blk_size: usize) -> usize {
        LookupTraceMLE::<F>::compute_helper_num_vars(self.num_vars, self.vec_input.len(), blk_size)
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for LookupWitness<F> {
    type Output = LookupWitness<EF>;

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
        let mle_size = 1 << self.num_vars;
        let blk_span = block_size << self.num_vars;

        // divide vec_input || table into blocks of size block_size
        let total = 1 + self.trace.vec_input.len();
        let num_blocks = (total + block_size - 1) / block_size;

        // t(x) + r and f(x) + r
        let table_and_inputs = self
            .table
            .iter()
            .chain(self.trace.vec_input.iter().flat_map(|input| input.iter()))
            .map(|&x| x + randomness)
            .collect::<Vec<F>>();

        let num_threads = rayon::current_num_threads();
        info!("Computing helper functions using {} threads", num_threads);
        let chunk_size = std::cmp::max(1, (table_and_inputs.len() + num_threads - 1) / num_threads);

        // 1 / (t(x) + r) and 1 / (f(x) + r)
        let mut inversed_values = table_and_inputs
            .par_chunks(chunk_size)
            .map(|chunk| batch_inverse(chunk))
            .flatten()
            .collect::<Vec<F>>();

        // - m(x) / (t(x) + r)
        for (t_i, m_i) in inversed_values
            .iter_mut()
            .take(1 << self.num_vars)
            .zip(self.multiplicity.iter())
        {
            *t_i *= -*m_i;
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

        let phi = table_and_inputs
            .chunks_exact(1 << self.num_vars)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();

        LookupWitnessHelper {
            block_size,
            num_blocks,
            random_value: randomness,
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
            .flat_map(|input: &Rc<DenseMultilinearExtension<F>>| input.iter())
            .cloned()
            .collect::<Vec<F>>()
    }
}

impl<F: Field> PackableEval<F> for LookupWitnessEval<F> {
    fn num_evals(&self) -> usize {
        self.vec_input_at_r.len() + 2
    }

    fn pack_to_vec(&self) -> Vec<F> {
        self.vec_input_at_r
            .iter()
            .cloned()
            .chain(std::iter::once(self.table_at_r))
            .chain(std::iter::once(self.multiplicity_at_r))
            .collect::<Vec<F>>()
    }

    fn pack_ntt_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }

    fn pack_poly_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }
}

impl<F: Field> PackableEval<F> for LookupWitnessHelperEval<F> {
    fn num_evals(&self) -> usize {
        self.helper_at_r.len()
    }

    fn pack_to_vec(&self) -> Vec<F> {
        self.helper_at_r.iter().cloned().collect::<Vec<F>>()
    }

    fn pack_ntt_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }

    fn pack_poly_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> EvaluableTraceEF<F, EF> for LookupWitness<F> {
    type TraceEvalEF = LookupWitnessEval<EF>;
    fn evaluate_ef(&self, point: &[EF]) -> Self::TraceEvalEF {
        let vec_input_at_r = self
            .trace
            .vec_input
            .iter()
            .map(|input| input.evaluate_ext(point))
            .collect::<Vec<EF>>();
        let table_at_r = self.table.evaluate_ext(point);
        let multiplicity_at_r = self.multiplicity.evaluate_ext(point);
        LookupWitnessEval {
            vec_input_at_r,
            table_at_r,
            multiplicity_at_r,
        }
    }
}

impl<F: Field> EvaluableTrace<F> for LookupWitnessHelper<F> {
    type TraceEval = LookupWitnessHelperEval<F>;
    fn evaluate(&self, point: &[F]) -> Self::TraceEval {
        let helper_at_r = self
            .helper_functions
            .iter()
            .map(|hf| hf.evaluate(point))
            .collect::<Vec<F>>();
        LookupWitnessHelperEval { helper_at_r }
    }
}
