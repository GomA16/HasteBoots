use algebra::{DenseMultilinearExtension, NTTField, transformation::AbstractNTT};

pub struct AccTrace<F: NTTField> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    // initial acc is the acc value input into the blind rotation
    // final acc is the acc value output from the blind rotation
    pub initial_acc: RLWETrace<F>,
    pub final_acc: RLWETrace<F>,
    // all monomials used in the blind rotation
    pub monomial: PolynomialTrace<F>,
    // all acc values input into each round of blind rotation
    pub input_acc: RLWETrace<F>,
    // all acc values output from each round of blind rotation
    pub output_acc: RLWETrace<F>,
    // all products computed during the blind rotation
    // product = monomial * input_acc
    pub product: RLWETrace<F>,
    // all intermediate values input into external product
    // external_product = product - input_acc
    pub external_product_input: RLWETrace<F>,
}

pub struct RLWETrace<F: NTTField> {
    pub log_coeff_count: usize,
    pub log_num_poly: usize,
    pub poly: (Vec<F>, Vec<F>),
    pub ntt: (Vec<F>, Vec<F>),
}

pub struct PolynomialTrace<F: NTTField> {
    pub log_coeff_count: usize,
    pub log_num_poly: usize,
    pub poly: Vec<F>,
    pub ntt: Vec<F>,
}

pub struct AccTraceMLE<F: NTTField> {
    pub log_coeff_count: usize,
    pub log_num_poly: usize,
    pub initial_acc: RLWETraceMLE<F>,
    pub final_acc: RLWETraceMLE<F>,
    pub input_acc: RLWETraceMLE<F>,
    pub output_acc: RLWETraceMLE<F>,
    pub monomial: PolynomialTraceMLE<F>,
    pub product: RLWETraceMLE<F>,
    pub external_product_input: RLWETraceMLE<F>,
}

pub struct RLWETraceMLE<F: NTTField> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub poly: (DenseMultilinearExtension<F>, DenseMultilinearExtension<F>),
    pub ntt: (DenseMultilinearExtension<F>, DenseMultilinearExtension<F>),
}

pub struct PolynomialTraceMLE<F: NTTField> {
    pub log_coeff_count: usize,
    pub log_num_poly: usize,
    pub poly: DenseMultilinearExtension<F>,
    pub ntt: DenseMultilinearExtension<F>,
}

impl<F: NTTField> PolynomialTrace<F> {
    pub fn new(log_coeff_count: usize, log_num_poly: usize) -> Self {
        Self {
            log_coeff_count,
            log_num_poly,
            poly: Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
            ntt: Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
        }
    }

    pub fn append_poly(&mut self, poly: &[F]) {
        self.poly.extend_from_slice(poly);

        let ntt_table = F::get_ntt_table(self.log_coeff_count as u32).unwrap();

        let mut ntt_poly = poly.to_vec();
        ntt_table.transform_slice(&mut ntt_poly);

        self.ntt.extend_from_slice(&ntt_poly);
    }
}

impl<F: NTTField> RLWETrace<F> {
    pub fn new(log_coeff_count: usize, log_num_round: usize) -> Self {
        Self {
            log_coeff_count,
            log_num_poly: log_num_round,
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

    pub fn append_poly(&mut self, rlwe: (&[F], &[F])) {
        self.poly.0.extend_from_slice(rlwe.0);
        self.poly.1.extend_from_slice(rlwe.1);

        let ntt_table = F::get_ntt_table(self.log_coeff_count as u32).unwrap();

        let mut ntt_a = rlwe.0.to_vec();
        let mut ntt_b = rlwe.1.to_vec();
        ntt_table.transform_slice(&mut ntt_a);
        ntt_table.transform_slice(&mut ntt_b);

        self.ntt.0.extend_from_slice(&ntt_a);
        self.ntt.1.extend_from_slice(&ntt_b);
    }

    pub fn append(&mut self, rlwe: (&[F], &[F]), ntt_rlwe: (&[F], &[F])) {
        self.poly.0.extend_from_slice(rlwe.0);
        self.poly.1.extend_from_slice(rlwe.1);
        self.ntt.0.extend_from_slice(ntt_rlwe.0);
        self.ntt.1.extend_from_slice(ntt_rlwe.1);
    }
}

impl<F: NTTField> AccTrace<F> {
    pub fn new(log_coeff_count: usize, log_num_round: usize) -> Self {
        Self {
            log_coeff_count,
            log_num_round,
            initial_acc: RLWETrace::new(log_coeff_count, 1),
            final_acc: RLWETrace::new(log_coeff_count, 1),
            input_acc: RLWETrace::new(log_coeff_count, log_num_round),
            output_acc: RLWETrace::new(log_coeff_count, log_num_round),
            monomial: PolynomialTrace::new(log_coeff_count, log_num_round),
            product: RLWETrace::new(log_coeff_count, log_num_round),
            external_product_input: RLWETrace::new(log_coeff_count, log_num_round),
        }
    }

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

    pub fn append_monomial(&mut self, monomial: &[F]) {
        self.monomial.append_poly(monomial);
    }

    pub fn append_product(&mut self, product: (&[F], &[F])) {
        self.product.append_poly(product);
    }

    pub fn append_external_product_input(&mut self, ext_prod_input: (&[F], &[F])) {
        self.external_product_input.append_poly(ext_prod_input);
    }
}

impl<F: NTTField> From<RLWETrace<F>> for RLWETraceMLE<F> {
    #[inline]
    fn from(trace: RLWETrace<F>) -> Self {
        Self {
            log_coeff_count: trace.log_coeff_count,
            log_num_round: trace.log_num_poly,
            poly: (
                DenseMultilinearExtension::from_evaluations_vec(
                    trace.log_coeff_count + trace.log_num_poly,
                    trace.poly.0,
                ),
                DenseMultilinearExtension::from_evaluations_vec(
                    trace.log_coeff_count + trace.log_num_poly,
                    trace.poly.1,
                ),
            ),
            ntt: (
                DenseMultilinearExtension::from_evaluations_vec(
                    trace.log_coeff_count + trace.log_num_poly,
                    trace.ntt.0,
                ),
                DenseMultilinearExtension::from_evaluations_vec(
                    trace.log_coeff_count + trace.log_num_poly,
                    trace.ntt.1,
                ),
            ),
        }
    }
}

impl<F: NTTField> From<PolynomialTrace<F>> for PolynomialTraceMLE<F> {
    #[inline]
    fn from(trace: PolynomialTrace<F>) -> Self {
        Self {
            log_coeff_count: trace.log_coeff_count,
            log_num_poly: trace.log_num_poly,
            poly: DenseMultilinearExtension::from_evaluations_vec(
                trace.log_coeff_count + trace.log_num_poly,
                trace.poly,
            ),
            ntt: DenseMultilinearExtension::from_evaluations_vec(
                trace.log_coeff_count + trace.log_num_poly,
                trace.ntt,
            ),
        }
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
            product: RLWETraceMLE::from(trace.product),
            external_product_input: RLWETraceMLE::from(trace.external_product_input),
        }
    }
}
