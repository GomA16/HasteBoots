use algebra::{DenseMultilinearExtension, Field, NTTField, transformation::AbstractNTT};

use crate::HadamardTrace;
use crate::rlwe_trace::{
    MonomialTrace, MonomialTraceMLE, PolynomialTrace, PolynomialTraceMLE, RLWETrace, RLWETraceMLE,
};

pub struct AccTrace<F: NTTField> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    // initial acc is the acc value input into the blind rotation
    // final acc is the acc value output from the blind rotation
    pub initial_acc: RLWETrace<F>,
    pub final_acc: RLWETrace<F>,
    // all monomials used in the blind rotation
    pub monomial: PolynomialTrace<F>,
    pub monomial_representation: MonomialTrace<F>,
    // all acc values input into each round of blind rotation
    pub input_acc: RLWETrace<F>,
    // all acc values output from each round of blind rotation
    pub output_acc: RLWETrace<F>,
    // all products computed during the blind rotation
    // monomial_times_acc = monomial * input_acc
    pub monomial_times_acc: RLWETrace<F>,
    // intermediate values input into external product (can be ommitted in practice)
    // external_product = monomial_times_acc - input_acc
    pub external_product_input: RLWETrace<F>,
}

pub struct AccTraceMLE<F: NTTField> {
    pub log_coeff_count: usize,
    pub log_num_poly: usize,
    pub initial_acc: RLWETraceMLE<F>,
    pub final_acc: RLWETraceMLE<F>,
    pub input_acc: RLWETraceMLE<F>,
    pub output_acc: RLWETraceMLE<F>,
    pub monomial: PolynomialTraceMLE<F>,
    pub monomial_representation: MonomialTraceMLE<F>,
    pub monomial_times_acc: RLWETraceMLE<F>,
    pub external_product_input: RLWETraceMLE<F>,
}

impl<F: NTTField> AccTrace<F> {
    pub fn new(log_coeff_count: usize, log_num_round: usize) -> Self {
        Self {
            log_coeff_count,
            log_num_round,
            initial_acc: RLWETrace::new(log_coeff_count, 1),
            final_acc: RLWETrace::new(log_coeff_count, 1),
            monomial: PolynomialTrace::new(log_coeff_count, log_num_round),
            monomial_representation: MonomialTrace::new(log_num_round),
            input_acc: RLWETrace::new(log_coeff_count, log_num_round),
            output_acc: RLWETrace::new(log_coeff_count, log_num_round),
            monomial_times_acc: RLWETrace::new(log_coeff_count, log_num_round),
            external_product_input: RLWETrace::new(log_coeff_count, log_num_round),
        }
    }

    // pub fn extract_hadamard_trace(&self) -> HadamardTraceMLE<F> {
    //     let monomial_times_acc = HadamardTraceMLE {
    //         log_coeff_count: self.log_coeff_count,
    //         log_num_round: self.log_num_round,
    //         bit_poly: self.monomial.clone(),

    //     };
    // }

    // First round
    pub fn append_acc_initial(&mut self, acc_poly: (&[F], &[F])) {
        self.initial_acc.append_poly(acc_poly);
        self.input_acc.append_poly(acc_poly);
    }

    // Last round
    pub fn append_acc_output(&mut self, acc_poly: (&[F], &[F])) {
        self.final_acc.append_poly(acc_poly);
        self.output_acc.append_poly(acc_poly);
    }

    // Intermediate rounds
    pub fn append_acc_round(&mut self, acc_poly: (&[F], &[F])) {
        self.input_acc.append_poly(acc_poly);
        self.output_acc.append_poly(acc_poly);
    }

    pub fn append_monomial(&mut self, monomial: &[F], degree: F, coefficient: F) {
        self.monomial.append_poly(monomial);
        self.monomial_representation.append(degree, coefficient);
    }

    pub fn append_product(&mut self, product: (&[F], &[F])) {
        self.monomial_times_acc.append_poly(product);
    }

    pub fn append_external_product_input(&mut self, ext_prod_input: (&[F], &[F])) {
        self.external_product_input.append_poly(ext_prod_input);
    }
}

impl<F: NTTField> From<AccTrace<F>> for AccTraceMLE<F> {
    #[inline]
    fn from(trace: AccTrace<F>) -> Self {
        Self {
            log_coeff_count: trace.log_coeff_count,
            log_num_poly: trace.log_num_round,
            initial_acc: RLWETraceMLE::from(trace.initial_acc),
            final_acc: RLWETraceMLE::from(trace.final_acc),
            input_acc: RLWETraceMLE::from(trace.input_acc),
            output_acc: RLWETraceMLE::from(trace.output_acc),
            monomial: PolynomialTraceMLE::from(trace.monomial),
            monomial_representation: MonomialTraceMLE::from(trace.monomial_representation),
            monomial_times_acc: RLWETraceMLE::from(trace.monomial_times_acc),
            external_product_input: RLWETraceMLE::from(trace.external_product_input),
        }
    }
}
