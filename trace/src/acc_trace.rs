use algebra::AbstractExtensionField;
use algebra::{DenseMultilinearExtension, Field, NTTField, transformation::AbstractNTT};
use serde::Serialize;

use crate::hadamard_trace::HadamardTraceEval;
use crate::rlwe_trace::{
    MonomialTrace, MonomialTraceMLE, PolynomialEval, PolynomialTrace, PolynomialTraceMLE, RLWEEval,
    RLWETrace, RLWETraceMLE,
};
use crate::{
    ConvertToEF, EvaluableTrace, EvaluableTraceEF, HadamardTrace, HadamardTraceMLE, PackableEval,
    PackableTrace, SumHadamardTraceEval, SumHadamardTraceMLE,
};

pub struct AccTrace<F: Field> {
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

pub struct AccTraceMLE<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub initial_acc: RLWETraceMLE<F>,
    pub final_acc: RLWETraceMLE<F>,
    pub input_acc: RLWETraceMLE<F>,
    pub output_acc: RLWETraceMLE<F>,
    pub monomial: PolynomialTraceMLE<F>,
    pub monomial_representation: MonomialTraceMLE<F>,
    pub monomial_times_acc: RLWETraceMLE<F>,
    pub external_product_input: RLWETraceMLE<F>,
}

#[derive(Serialize)]
pub struct AccTraceEval<F: Field> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub input_acc: RLWEEval<F>,
    pub monomial: PolynomialEval<F>,
    pub monomial_times_acc: RLWEEval<F>,
    pub external_product_input: RLWEEval<F>,
}

impl<F: NTTField> AccTrace<F> {
    #[inline]
    pub fn new(log_coeff_count: usize, log_num_round: usize) -> Self {
        Self {
            log_coeff_count,
            log_num_round,
            initial_acc: RLWETrace::new(log_coeff_count, 0),
            final_acc: RLWETrace::new(log_coeff_count, 0),
            monomial: PolynomialTrace::new(log_coeff_count, log_num_round),
            monomial_representation: MonomialTrace::new(log_coeff_count, log_num_round),
            input_acc: RLWETrace::new(log_coeff_count, log_num_round),
            output_acc: RLWETrace::new(log_coeff_count, log_num_round),
            monomial_times_acc: RLWETrace::new(log_coeff_count, log_num_round),
            external_product_input: RLWETrace::new(log_coeff_count, log_num_round),
        }
    }

    // First round
    #[inline]
    pub fn append_acc_initial(&mut self, acc_poly: (&[F], &[F])) {
        self.initial_acc.append_poly(acc_poly);
        self.input_acc.append_poly(acc_poly);
    }

    // Last round
    #[inline]
    pub fn append_acc_output(&mut self, acc_poly: (&[F], &[F])) {
        self.final_acc.append_poly(acc_poly);
        self.output_acc.append_poly(acc_poly);
    }

    // Intermediate rounds
    #[inline]
    pub fn append_acc_round(&mut self, acc_poly: (&[F], &[F])) {
        self.input_acc.append_poly(acc_poly);
        self.output_acc.append_poly(acc_poly);
    }

    #[inline]
    pub fn append_monomial(&mut self, monomial: &[F], degree: F, coefficient: F) {
        self.monomial.append_poly(monomial);
        self.monomial_representation.append(degree, coefficient);
    }

    #[inline]
    pub fn append_product(&mut self, product: (&[F], &[F])) {
        self.monomial_times_acc.append_poly(product);
    }

    #[inline]
    pub fn append_external_product_input(&mut self, ext_prod_input: (&[F], &[F])) {
        self.external_product_input.append_poly(ext_prod_input);
    }
}

impl<F: Field> AccTrace<F> {
    #[inline]
    pub fn finalize(&mut self, num_round: usize) {
        self.initial_acc.finalize(1);
        self.final_acc.finalize(1);
        self.monomial.finalize(num_round);
        self.input_acc.finalize(num_round);
        self.output_acc.finalize(num_round);
        self.monomial_times_acc.finalize(num_round);
        self.external_product_input.finalize(num_round);
    }
}

impl<F: Field> From<AccTrace<F>> for AccTraceMLE<F> {
    #[inline]
    fn from(trace: AccTrace<F>) -> Self {
        Self {
            log_coeff_count: trace.log_coeff_count,
            log_num_round: trace.log_num_round,
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

impl<F: Field> PackableTrace<F> for AccTraceMLE<F> {
    fn num_vars(&self) -> usize {
        self.log_coeff_count + self.log_num_round
    }

    fn num_oracles(&self) -> usize {
        // input_acc
        // monomial_times_acc
        // monomial will be committed in other places
        // other polynomials are derived from these:
        // external_product_input = monomial_times_acc - input_acc
        // output_acc = input_acc + sum_prod of SumHadamardTrace
        self.input_acc.num_oracles() + self.monomial_times_acc.num_oracles()
    }

    fn pack_to_vec(&self) -> Vec<F> {
        self.input_acc
            .pack_to_vec()
            .into_iter()
            .chain(self.monomial_times_acc.pack_to_vec())
            .collect()
    }
}

impl<F: Field> PackableTrace<F> for AccTrace<F> {
    fn num_vars(&self) -> usize {
        self.log_coeff_count + self.log_num_round
    }

    fn num_oracles(&self) -> usize {
        // input_acc
        // monomial_times_acc
        // monomial will be committed in other places
        // other polynomials are derived from these:
        // external_product_input = monomial_times_acc - input_acc
        // output_acc = input_acc + sum_prod of SumHadamardTrace
        self.input_acc.num_oracles() + self.monomial_times_acc.num_oracles()
    }

    fn pack_to_vec(&self) -> Vec<F> {
        self.input_acc
            .pack_to_vec()
            .into_iter()
            .chain(self.monomial_times_acc.pack_to_vec())
            .collect()
    }
}

impl<F: Field> PackableEval<F> for AccTraceEval<F> {
    #[inline]
    fn num_evals(&self) -> usize {
        self.input_acc.num_evals() + self.monomial_times_acc.num_evals()
    }

    #[inline]
    fn pack_ntt_to_vec(&self) -> Vec<F> {
        self.input_acc
            .pack_ntt_to_vec()
            .into_iter()
            .chain(self.monomial_times_acc.pack_ntt_to_vec())
            .collect()
    }

    #[inline]
    fn pack_poly_to_vec(&self) -> Vec<F> {
        self.input_acc
            .pack_poly_to_vec()
            .into_iter()
            .chain(self.monomial_times_acc.pack_poly_to_vec())
            .collect()
    }

    #[inline]
    fn pack_to_vec(&self) -> Vec<F> {
        unimplemented!()
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for AccTraceMLE<F> {
    type Output = AccTraceMLE<EF>;
    fn to_ef(&self) -> AccTraceMLE<EF> {
        AccTraceMLE {
            log_coeff_count: self.log_coeff_count,
            log_num_round: self.log_num_round,
            initial_acc: self.initial_acc.to_ef(),
            final_acc: self.final_acc.to_ef(),
            input_acc: self.input_acc.to_ef(),
            output_acc: self.output_acc.to_ef(),
            monomial: self.monomial.to_ef(),
            monomial_representation: self.monomial_representation.to_ef(),
            monomial_times_acc: self.monomial_times_acc.to_ef(),
            external_product_input: self.external_product_input.to_ef(),
        }
    }
}

impl<F: Field> AccTraceMLE<F> {
    #[inline]
    pub fn extract_hadamard_trace(&self) -> SumHadamardTraceMLE<F> {
        let monomial_times_acc = HadamardTraceMLE {
            log_coeff_count: self.log_coeff_count,
            log_num_poly: self.log_num_round,
            bit: self.monomial.clone(),
            rlwe: self.input_acc.clone(),
        };
        let sum_prod = self.monomial_times_acc.clone();
        SumHadamardTraceMLE {
            log_coeff_count: self.log_coeff_count,
            log_num_poly: self.log_num_round,
            num_hadamard: 1,
            vec_hadamard: vec![monomial_times_acc],
            sum_prod,
        }
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> EvaluableTraceEF<F, EF> for AccTraceMLE<F> {
    type TraceMLEEF = AccTraceMLE<EF>;
    type TraceEvalEF = AccTraceEval<EF>;
    fn evaluate_ef(&self, point: &[EF]) -> Self::TraceEvalEF {
        let input_acc = self.input_acc.evaluate_ef(point);
        let monomial = self.monomial.evaluate_ef(point);
        let monomial_times_acc = self.monomial_times_acc.evaluate_ef(point);
        let rlwe_sub = |a: &RLWEEval<EF>, b: &RLWEEval<EF>| -> RLWEEval<EF> {
            let c_poly = (a.poly.0 - b.poly.0, a.poly.1 - b.poly.1);
            let c_ntt = (a.ntt.0 - b.ntt.0, a.ntt.1 - b.ntt.1);
            RLWEEval {
                poly: c_poly,
                ntt: c_ntt,
            }
        };
        let external_product_input = rlwe_sub(&monomial_times_acc, &input_acc);
        AccTraceEval {
            log_coeff_count: self.log_coeff_count,
            log_num_round: self.log_num_round,
            input_acc,
            monomial,
            monomial_times_acc,
            external_product_input,
        }
    }

    fn evaluate_ef_with_lookup(
        &self,
        point: &[EF],
        trace_ef: &Self::TraceMLEEF,
        hash_table: &algebra::ListOfProductsOfPolynomials<EF>,
        eval_table: &[EF],
    ) -> Self::TraceEvalEF {
        let input_acc = self.input_acc.evaluate_ef_with_lookup(
            point,
            &trace_ef.input_acc,
            hash_table,
            eval_table,
        );
        let monomial = self.monomial.evaluate_ef_with_lookup(
            point,
            &trace_ef.monomial,
            hash_table,
            eval_table,
        );
        let monomial_times_acc = self.monomial_times_acc.evaluate_ef_with_lookup(
            point,
            &trace_ef.monomial_times_acc,
            hash_table,
            eval_table,
        );
        let rlwe_sub = |a: &RLWEEval<EF>, b: &RLWEEval<EF>| -> RLWEEval<EF> {
            let c_poly = (a.poly.0 - b.poly.0, a.poly.1 - b.poly.1);
            let c_ntt = (a.ntt.0 - b.ntt.0, a.ntt.1 - b.ntt.1);
            RLWEEval {
                poly: c_poly,
                ntt: c_ntt,
            }
        };
        let external_product_input = rlwe_sub(&monomial_times_acc, &input_acc);
        AccTraceEval {
            log_coeff_count: self.log_coeff_count,
            log_num_round: self.log_num_round,
            input_acc,
            monomial,
            monomial_times_acc,
            external_product_input,
        }
    }
}

impl<F: Field> EvaluableTrace<F> for AccTraceMLE<F> {
    type TraceEval = AccTraceEval<F>;
    fn evaluate(&self, point: &[F]) -> Self::TraceEval {
        let input_acc = self.input_acc.evaluate(point);
        let monomial = self.monomial.evaluate(point);
        let monomial_times_acc = self.monomial_times_acc.evaluate(point);
        let rlwe_sub = |a: &RLWEEval<F>, b: &RLWEEval<F>| -> RLWEEval<F> {
            let c_poly = (a.poly.0 - b.poly.0, a.poly.1 - b.poly.1);
            let c_ntt = (a.ntt.0 - b.ntt.0, a.ntt.1 - b.ntt.1);
            RLWEEval {
                poly: c_poly,
                ntt: c_ntt,
            }
        };
        let external_product_input = rlwe_sub(&monomial_times_acc, &input_acc);
        AccTraceEval {
            log_coeff_count: self.log_coeff_count,
            log_num_round: self.log_num_round,
            input_acc,
            monomial,
            monomial_times_acc,
            external_product_input,
        }
    }

    fn evaluate_with_lookup(
        &self,
        point: &[F],
        hash_table: &algebra::ListOfProductsOfPolynomials<F>,
        eval_table: &[F],
    ) -> Self::TraceEval {
        let input_acc = self
            .input_acc
            .evaluate_with_lookup(point, hash_table, eval_table);
        let monomial = self
            .monomial
            .evaluate_with_lookup(point, hash_table, eval_table);
        let monomial_times_acc = self
            .monomial_times_acc
            .evaluate_with_lookup(point, hash_table, eval_table);
        let rlwe_sub = |a: &RLWEEval<F>, b: &RLWEEval<F>| -> RLWEEval<F> {
            let c_poly = (a.poly.0 - b.poly.0, a.poly.1 - b.poly.1);
            let c_ntt = (a.ntt.0 - b.ntt.0, a.ntt.1 - b.ntt.1);
            RLWEEval {
                poly: c_poly,
                ntt: c_ntt,
            }
        };
        let external_product_input = rlwe_sub(&monomial_times_acc, &input_acc);
        AccTraceEval {
            log_coeff_count: self.log_coeff_count,
            log_num_round: self.log_num_round,
            input_acc,
            monomial,
            monomial_times_acc,
            external_product_input,
        }
    }
}

impl<F: Field> AccTraceEval<F> {
    pub fn extract_hadamard_eval(&self) -> SumHadamardTraceEval<F> {
        let hadamard_eval = HadamardTraceEval {
            bit: self.monomial.clone(),
            rlwe: self.input_acc.clone(),
        };
        SumHadamardTraceEval {
            vec_hadamard: vec![hadamard_eval],
            sum_prod: self.monomial_times_acc.clone(),
        }
    }
}
