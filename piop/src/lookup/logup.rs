use algebra::{DenseMultilinearExtension, Field};
use std::rc::Rc;

pub struct LogUpInstance<F: Field> {
    pub num_vars: usize,
    pub num_vars_of_table: usize,
    pub num_columns: usize,
    pub columns: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub table: Rc<DenseMultilinearExtension<F>>,
    pub multiplicity: Rc<DenseMultilinearExtension<F>>,
    pub block_size: usize,
    pub num_of_blocks: usize,
}
