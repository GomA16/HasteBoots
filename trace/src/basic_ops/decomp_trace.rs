use std::rc::Rc;

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};

use crate::{
    EvaluableTraceEF, PackableTrace,
    cmp_trace::lt_general_trace::{LTGeneralTable, LTGeneralTables, LTGeneralTablesMLE},
    lookup_trace::indexed_table::IndexedLookupTraceMLE,
};

pub struct DecompTraceMLE<F: Field> {
    pub log_num: usize,
    pub basis_bits: usize,
    pub decomp_len: usize,
    pub input: Rc<DenseMultilinearExtension<F>>,
    pub bits: Vec<Rc<DenseMultilinearExtension<F>>>,
}

pub struct DecompEval<F: Field> {
    pub input: F,
    pub bits: Vec<F>,
}

impl<F: Field> DecompTraceMLE<F> {
    pub fn num_lookup(&self) -> usize {
        self.bits.len()
    }

    pub fn extract_lt_general_lookup_trace(
        &self,
        lt_tables: &LTGeneralTablesMLE<F>,
    ) -> Vec<IndexedLookupTraceMLE<F>> {
        assert_eq!(self.bits.len(), lt_tables.decomp_len);

        self.bits
            .iter()
            .enumerate()
            .map(|(i, bit)| {
                let index = Rc::clone(bit);
                let input = lt_tables.lookup(i, &index);
                IndexedLookupTraceMLE {
                    num_input_vars: index.num_vars(),
                    num_table_vars: lt_tables.tables[i].num_vars(),
                    index,
                    input,
                    table: lt_tables.get_table(i),
                    table_point: None,
                }
            })
            .collect()
    }
}

impl<F: Field> PackableTrace<F> for Vec<DecompTraceMLE<F>> {
    fn num_vars(&self) -> usize {
        self[0].log_num
    }

    fn num_oracles(&self) -> usize {
        self.len()
    }

    fn pack_to_vec(&self) -> Vec<F> {
        self.iter()
            .flat_map(|trace| trace.input.iter().cloned())
            .collect()
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> EvaluableTraceEF<F, EF> for DecompTraceMLE<F> {
    type TraceMLEEF = DecompTraceMLE<EF>;
    type TraceEvalEF = DecompEval<EF>;
    fn evaluate_ef(&self, point: &[EF]) -> Self::TraceEvalEF {
        let bits = self
            .bits
            .iter()
            .map(|bit| bit.evaluate_ext(point))
            .collect();
        let input = self.input.evaluate_ext(point);
        DecompEval { input, bits }
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

    fn evaluate_ef_with_lookup(
        &self,
        point: &[EF],
        trace_ef: &Self::TraceMLEEF,
        hash_table: &algebra::ListOfProductsOfPolynomials<EF>,
        eval_table: &[EF],
    ) -> Self::TraceEvalEF {
        let bits = self
            .bits
            .iter()
            .zip(trace_ef.bits.iter())
            .map(|(bit, trace_bit)| {
                hash_table.lookup_mle_eval_ef(&bit, &trace_bit, eval_table, point)
            })
            .collect();
        let input = hash_table.lookup_mle_eval_ef(&self.input, &trace_ef.input, eval_table, point);
        DecompEval { input, bits }
    }
}
