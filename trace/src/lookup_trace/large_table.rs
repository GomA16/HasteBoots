use std::rc::Rc;

use algebra::{DecomposableField, DenseMultilinearExtension, Field};

use crate::lookup_trace::normal_table::LookupTraceMLE;

pub struct LookupLargeTableTrace<F: Field> {
    pub num_input_vars: usize,
    pub input: Vec<F>,
    pub range: usize,
    pub num_range_bits: usize,
}

pub struct LookupLargeTableTraceMLE<F: Field> {
    pub num_input_vars: usize,
    pub input: Rc<DenseMultilinearExtension<F>>,
    pub range: usize,
    pub range_bits: usize,
}

pub struct LookupLargeTableWitness<F: DecomposableField> {
    pub input_bits: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub basis_bits: usize,
    pub decompose_len: usize,
}

impl<F: DecomposableField> LookupLargeTableTraceMLE<F> {
    pub fn compute_lookup_large_table_witness(
        &self,
        // basis_len is the bit length of basis (we only support power of 2 basis)
        basis_bits: usize,
    ) -> LookupLargeTableWitness<F> {
        let decompose_len = self.range_bits.div_ceil(basis_bits) + 1;
        let decomposed_input = self.input.get_decomposed_mles(basis_bits, decompose_len);
        LookupLargeTableWitness {
            input_bits: decomposed_input,
            basis_bits,
            decompose_len,
        }
    }

    pub fn extract_lookup_trace(&self, witness: &LookupLargeTableWitness<F>) -> LookupTraceMLE<F> {
        let range = 1 << witness.basis_bits;
        LookupTraceMLE {
            num_vars: self.num_input_vars,
            vec_input: witness.input_bits.clone(),
            range,
        }
    }
}
