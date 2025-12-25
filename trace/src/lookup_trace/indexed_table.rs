//! Indexed Lookup Trace, specialy used in the sparse matrix evaluation.
//!
//! The indexed table consists of n pairs (y, T[y]) for y in [n].
//! The input are m pairs (I[x], E[x]) for x in [m] where I[x] is the index
//! to the table. The relation of the indexed lookup is to prove that:
//!     E[x] = T[I[x]] for all x in [m].
//! Assume m <= n in the following.
//!
//! Hence, the indexed lookup can be reduced to a normal unindexed lookup argument.
//! Each pair can be hashed into a single field element using a random element `s`.
//!
//! The unindexed table consists of all T'[y] = T[y] + s * y for y in [n].
//! The input is E[x] + s * I[x] for x in [m].
//!
//! In sparse matrix evaluation, one is to prove E(k) = eq(to-bits(col(k)), ry) with
//! E(k) and col(k) committed. This can be viewed as a indexed lookup problem where
//! the table is T[y] = eq(y, ry) of size n and the input are m pairs (col(k), E(k))
//! for k in [m]. Here col(k) is the index for each E(k).
//! It satisfies E[k] = T[col(k)].
use algebra::{AbstractExtensionField, AsInto};
use algebra::{DenseMultilinearExtension, Field};
use core::num;
use helper::utils::{batch_inverse, gen_identity_evaluations};
use itertools::Itertools;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use rayon::vec;
use std::sync::Arc;
use std::{collections::HashMap, rc::Rc};

use crate::{ConvertToEF, EvaluableTrace, EvaluableTraceEF, PackableEval, PackableTrace};
use log::info;

// Conversion Chain: LookupTraceMLE => LookupWitness
// LookupWitnessHelper is computed from LookupWitness with a random value

pub struct IndexedLookupTrace<F: Field> {
    pub num_input_vars: usize,
    pub num_table_vars: usize,
    // indexed input (I[x], E[x]) for x in [m]
    pub index: Vec<F>,
    pub input: Vec<F>,
    pub table: Vec<F>,
    // table T[y] = eq(y, ry) for a random point ry
    pub table_point: Vec<F>,
}

#[derive(Clone)]
pub struct IndexedLookupTraceMLE<F: Field> {
    pub num_input_vars: usize,
    pub num_table_vars: usize,
    // indexed input (I[x], E[x]) for x in [m]
    pub index: Rc<DenseMultilinearExtension<F>>,
    pub input: Rc<DenseMultilinearExtension<F>>,
    pub table: Rc<DenseMultilinearExtension<F>>,
    // table T[y] = eq(y, ry) for a random point ry
    pub table_point: Vec<F>,
}

#[derive(Clone)]
pub struct IndexedLookupWitness<F: Field> {
    // pub num_input_vars: usize,
    pub num_table_vars: usize,
    pub table_point: Vec<F>,
    // pub trace: IndexedLookupTraceMLE<F>,
    pub multiplicity: Rc<DenseMultilinearExtension<F>>,
}

#[derive(Clone)]
pub struct IndexedLookupWitnessHelper<F: Field> {
    pub random_value: F,
    pub random_s_hash: F,
    // helper_input = 1 / phi_input
    pub helper_input: Rc<DenseMultilinearExtension<F>>,
    // helper_table = multiplicity / phi_table
    pub helper_table: Rc<DenseMultilinearExtension<F>>,
    pub sum: F,
    // only for efficiency of prover
    // phi_input = E[x] + s * I[x] + r
    pub phi_input: Rc<DenseMultilinearExtension<F>>,
    // phi_table = T[y] + s * y + r
    pub phi_table: Rc<DenseMultilinearExtension<F>>,
}

pub struct LookupWitnessHelperEval<F: Field> {
    pub helper_input_at_r: F,
    pub helper_table_at_r: F,
}

impl<F: Field> IndexedLookupTrace<F> {
    pub fn random<R: rand::Rng + rand::CryptoRng>(
        rng: &mut R,
        num_input_vars: usize,
        num_table_vars: usize,
    ) -> Self {
        let table_point = (0..num_table_vars)
            .map(|_| F::random(rng))
            .collect::<Vec<F>>();

        let table = gen_identity_evaluations(&table_point).evaluations;

        let num_inputs = 1 << num_input_vars;
        let table_len = 1 << num_table_vars;
        let index = (0..num_inputs)
            .map(|_| rng.random_range(0..table_len))
            .collect::<Vec<usize>>();

        let mut input = vec![F::zero(); num_inputs];
        for i in 0..num_inputs {
            input[i] = table[index[i]];
        }

        let index = index
            .iter()
            .map(|&i| F::new((i as u32).as_into()))
            .collect::<Vec<F>>();

        Self {
            num_input_vars,
            num_table_vars,
            index,
            input,
            table,
            table_point,
        }
    }
}

impl<F: Field> From<IndexedLookupTrace<F>> for IndexedLookupTraceMLE<F> {
    fn from(trace: IndexedLookupTrace<F>) -> Self {
        Self {
            num_input_vars: trace.num_input_vars,
            num_table_vars: trace.num_table_vars,
            index: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                trace.num_input_vars,
                trace.index,
            )),
            input: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                trace.num_input_vars,
                trace.input,
            )),
            table: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                trace.num_table_vars,
                trace.table,
            )),
            table_point: trace.table_point,
        }
    }
}

impl<F: Field> IndexedLookupTraceMLE<F> {
    pub fn compute_witness(&self) -> IndexedLookupWitness<F> {
        let mut multiplicity_hashmap = HashMap::new();

        self.input.iter().for_each(|&elem| {
            multiplicity_hashmap
                .entry(elem)
                .and_modify(|cnt| *cnt += 1u32)
                .or_insert(1u32);
        });

        // compute multiplicity
        let mut multiplicity = vec![F::zero(); 1 << self.num_table_vars];

        for (t_i, m_i) in self.table.iter().zip(multiplicity.iter_mut()) {
            let count = multiplicity_hashmap.remove(t_i).unwrap_or(0u32);
            *m_i = F::new((count as u32).as_into());
        }

        IndexedLookupWitness {
            num_table_vars: self.num_table_vars,
            multiplicity: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                self.num_table_vars,
                multiplicity,
            )),
            table_point: self.table_point.clone(),
        }
    }

    pub fn compute_helper_functions(
        &self,
        witness: &IndexedLookupWitness<F>,
        randomness: F,
        s_hash: F,
    ) -> IndexedLookupWitnessHelper<F> {
        assert_eq!(self.num_table_vars, witness.num_table_vars);
        // phi_input = E[x] + s * I[x] + r
        let hashed_inputs_plus_r = self
            .input
            .iter()
            .zip(self.index.iter())
            .map(|(&e_x, &i_x)| e_x + s_hash * i_x + randomness)
            .collect::<Vec<F>>();

        // phi_table =T[y] + s * y + r
        let hashed_table_plus_r = self
            .table
            .iter()
            .enumerate()
            .map(|(y, &t_y)| t_y + s_hash * F::new((y as u32).as_into()) + randomness)
            .collect::<Vec<F>>();

        let num_threads = rayon::current_num_threads();
        info!("Computing helper functions using {} threads", num_threads);
        let chunk_size = std::cmp::max(
            1,
            (hashed_inputs_plus_r.len() + num_threads - 1) / num_threads,
        );

        // helper_input = 1 / phi_input
        let helper_input = hashed_inputs_plus_r
            .par_chunks(chunk_size)
            .map(|chunk| batch_inverse(chunk))
            .flatten()
            .collect::<Vec<F>>();

        let num_threads = rayon::current_num_threads();
        info!("Computing helper functions using {} threads", num_threads);
        let chunk_size = std::cmp::max(
            1,
            (hashed_table_plus_r.len() + num_threads - 1) / num_threads,
        );

        // 1 / phi_table
        let table_inversed_values = hashed_table_plus_r
            .par_chunks(chunk_size)
            .map(|chunk| batch_inverse(chunk))
            .flatten()
            .collect::<Vec<F>>();

        // helper_table = multiplicity / phi_table
        // sum = \sum m(y) / phi_table(y)
        let mut sum = F::zero();
        let helper_table = table_inversed_values
            .iter()
            .zip(witness.multiplicity.iter())
            .map(|(&t_i, &m_i)| {
                let val = t_i * m_i;
                sum += val;
                val
            })
            .collect::<Vec<F>>();

        IndexedLookupWitnessHelper {
            random_value: randomness,
            random_s_hash: s_hash,
            helper_input: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                self.num_input_vars,
                helper_input,
            )),
            helper_table: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                self.num_table_vars,
                helper_table,
            )),
            sum,
            phi_input: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                self.num_input_vars,
                hashed_inputs_plus_r,
            )),
            phi_table: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                self.num_table_vars,
                hashed_table_plus_r,
            )),
        }
    }
}
