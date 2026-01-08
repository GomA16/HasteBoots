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
use algebra::{AsInto};
use algebra::{DenseMultilinearExtension, Field};
use helper::utils::{batch_inverse, gen_identity_evaluations};
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use std::{collections::HashMap, rc::Rc};

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
    pub table_point: Option<Vec<F>>,
}