use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field, NTTField, NTTPolynomial, transformation::AbstractNTT};

mod ntt_trace;

pub use ntt_trace::{NTTTraceMLE, NTTInstanceInfo, NTTTrace};

pub trait ConvertToEF<F: Field, EF: AbstractExtensionField<F>> {
    type Output;
    fn into_ef(self) -> Self::Output;
    fn to_ef(&self) -> Self::Output;
}

impl<F: Field, EF: AbstractExtensionField<F>> ConvertToEF<F, EF> for Vec<F> {
    type Output = Vec<EF>;

    fn into_ef(self) -> Self::Output {
        self.into_iter().map(EF::from_base).collect()
    }

    fn to_ef(&self) -> Self::Output {
        self.iter().map(|&b| EF::from_base(b)).collect()
    }
}

pub trait PackTrace<F: Field> {
    type TraceType;

    fn num_vars(&self) -> usize;
    fn num_oracles(&self) -> usize;
    fn log_num_oracles(&self) -> usize {
        self.num_oracles().next_power_of_two().trailing_zeros() as usize
    }
    fn pack_to_vec(&self) -> Vec<F>;
    fn generate_oracle(&self) -> DenseMultilinearExtension<F>;
}

/// Store the traces of each round of Hadamard product during blind rotation.
#[derive(Debug, Clone)]
pub struct HadmardProdTrace<F: NTTField> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub bit_poly: Vec<F>,
    pub bit_ntt: Vec<F>,
    pub key_ntt: (Vec<F>, Vec<F>),
}

impl<F: NTTField> HadmardProdTrace<F> {
    pub fn new(log_coeff_count: usize, log_num_poly: usize) -> Self {
        Self {
            log_coeff_count,
            log_num_round: log_num_poly,
            bit_poly: Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
            bit_ntt: Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
            key_ntt: (
                Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
                Vec::with_capacity(1 << (log_coeff_count + log_num_poly)),
            ),
        }
    }

    pub fn append_bit_poly(&mut self, bit_poly: &[F]) {
        self.bit_poly.extend_from_slice(bit_poly);
    }

    pub fn append_bit_ntt(&mut self, bit_ntt: &[F]) {
        self.bit_ntt.extend_from_slice(bit_ntt);
    }

    pub fn append_key_ntt(&mut self, key_poly: (&[F], &[F])) {
        self.key_ntt.0.extend_from_slice(key_poly.0);
        self.key_ntt.1.extend_from_slice(key_poly.1);
    }

    pub fn export_mles(
        self,
    ) -> (
        (DenseMultilinearExtension<F>, DenseMultilinearExtension<F>),
        (DenseMultilinearExtension<F>, DenseMultilinearExtension<F>),
    ) {
        let num_vars = self.log_coeff_count + self.log_num_round;
        let bit_poly_mle = DenseMultilinearExtension::from_evaluations_vec(num_vars, self.bit_poly);
        let bit_ntt_mle = DenseMultilinearExtension::from_evaluations_vec(num_vars, self.bit_ntt);
        let key_mle_0 = DenseMultilinearExtension::from_evaluations_vec(num_vars, self.key_ntt.0);
        let key_mle_1 = DenseMultilinearExtension::from_evaluations_vec(num_vars, self.key_ntt.1);

        ((bit_poly_mle, bit_ntt_mle), (key_mle_0, key_mle_1))
    }
}

pub struct HadamardProdsTrace<F: NTTField> {
    pub num_trace: usize,
    pub vec_trace: Vec<HadmardProdTrace<F>>,
}

impl<F: NTTField> HadamardProdsTrace<F> {
    pub fn new(num_trace: usize, log_coeff_count: usize, log_num_poly: usize) -> Self {
        Self {
            num_trace,
            vec_trace: vec![HadmardProdTrace::new(log_coeff_count, log_num_poly); num_trace],
        }
    }

    pub fn get_trace_mul(&mut self, trace_idx: usize) -> &mut HadmardProdTrace<F> {
        &mut self.vec_trace[trace_idx]
    }
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

pub struct AccTrace<F: NTTField> {
    pub log_coeff_count: usize,
    pub log_num_round: usize,
    pub input_acc: RLWETrace<F>,
    pub output_acc: RLWETrace<F>,
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
