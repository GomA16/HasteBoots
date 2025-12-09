use std::rc::Rc;

use algebra::{
    AbstractExtensionField, DenseMultilinearExtension, Field, FieldUniformSampler, NTTField,
    transformation::AbstractNTT,
};

use crate::{ConvertToEF, NTTTraceMLE};

pub struct AccTrace<F: NTTField> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub input_acc: RLWETrace<F>,
    pub output_acc: RLWETrace<F>,
}

pub struct RLWETrace<F: NTTField> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub poly: (Vec<F>, Vec<F>),
    pub ntt: (Vec<F>, Vec<F>),
}

impl<F: NTTField> RLWETrace<F> {
    pub fn new(log_coeff_count: usize, log_num_round: usize) -> Self {
        Self {
            log_coeff_count,
            log_num_round,
            poly: (
                Vec::with_capacity(1 << (log_coeff_count + log_num_round)),
                Vec::with_capacity(1 << (log_coeff_count + log_num_round)),
            ),
            ntt: (
                Vec::with_capacity(1 << (log_coeff_count + log_num_round)),
                Vec::with_capacity(1 << (log_coeff_count + log_num_round)),
            ),
        }
    }

    pub fn append_acc_poly(&mut self, acc_poly: (&[F], &[F])) {
        self.poly.0.extend_from_slice(acc_poly.0);
        self.poly.1.extend_from_slice(acc_poly.1);
    }

    pub fn ntt_append_acc_ntt(&mut self, acc_poly: (&[F], &[F])) {
        let ntt_table = F::get_ntt_table(self.log_coeff_count as u32).unwrap();

        let mut ntt_a = acc_poly.0.to_vec();
        let mut ntt_b = acc_poly.1.to_vec();
        ntt_table.transform_slice(&mut ntt_a);
        ntt_table.transform_slice(&mut ntt_b);

        self.ntt.0.extend_from_slice(&ntt_a);
        self.ntt.1.extend_from_slice(&ntt_b);
    }

    pub fn append_acc_ntt(&mut self, acc_ntt: (&[F], &[F])) {
        self.ntt.0.extend_from_slice(acc_ntt.0);
        self.ntt.1.extend_from_slice(acc_ntt.1);
    }
}



impl<F: NTTField> AccTrace<F> {
    pub fn new(log_coeff_count: usize, log_num_round: usize) -> Self {
        Self {
            log_coeff_count,
            log_num_round,
            input_acc: RLWETrace::new(log_coeff_count, log_num_round),
            output_acc: RLWETrace::new(log_coeff_count, log_num_round),
        }
    }

    // First round
    pub fn append_acc_input(&mut self, acc_poly: (&[F], &[F])) {
        self.input_acc.poly.0.extend_from_slice(acc_poly.0);
        self.input_acc.poly.1.extend_from_slice(acc_poly.1);

        let ntt_table = F::get_ntt_table(self.log_coeff_count as u32).unwrap();

        let mut ntt_a = acc_poly.0.to_vec();
        let mut ntt_b = acc_poly.1.to_vec();
        ntt_table.transform_slice(&mut ntt_a);
        ntt_table.transform_slice(&mut ntt_b);

        self.input_acc.ntt.0.extend_from_slice(&ntt_a);
        self.input_acc.ntt.1.extend_from_slice(&ntt_b);
    }

    // Last round
    pub fn append_acc_output(&mut self, acc_poly: (&[F], &[F])) {
        self.output_acc.poly.0.extend_from_slice(acc_poly.0);
        self.output_acc.poly.1.extend_from_slice(acc_poly.1);

        let ntt_table = F::get_ntt_table(self.log_coeff_count as u32).unwrap();

        let mut ntt_a = acc_poly.0.to_vec();
        let mut ntt_b = acc_poly.1.to_vec();
        ntt_table.transform_slice(&mut ntt_a);
        ntt_table.transform_slice(&mut ntt_b);

        self.output_acc.ntt.0.extend_from_slice(&ntt_a);
        self.output_acc.ntt.1.extend_from_slice(&ntt_b);
    }

    // Intermediate rounds
    pub fn append_acc_round(&mut self, acc_poly: (&[F], &[F])) {
        self.input_acc.poly.0.extend_from_slice(acc_poly.0);
        self.input_acc.poly.1.extend_from_slice(acc_poly.1);
        self.output_acc.poly.0.extend_from_slice(acc_poly.0);
        self.output_acc.poly.1.extend_from_slice(acc_poly.1);

        let ntt_table = F::get_ntt_table(self.log_coeff_count as u32).unwrap();

        let mut ntt_a = acc_poly.0.to_vec();
        let mut ntt_b = acc_poly.1.to_vec();
        ntt_table.transform_slice(&mut ntt_a);
        ntt_table.transform_slice(&mut ntt_b);

        self.input_acc.ntt.0.extend_from_slice(&ntt_a);
        self.input_acc.ntt.1.extend_from_slice(&ntt_b);
        self.output_acc.ntt.0.extend_from_slice(&ntt_a);
        self.output_acc.ntt.1.extend_from_slice(&ntt_b);
    }
}
