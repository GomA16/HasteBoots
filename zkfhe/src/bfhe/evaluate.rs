use algebra::{Basis, NTTField, Polynomial};
use fhe_core::{
    KeySwitchingKeyEnum, KeySwitchingRLWEKey, LWECiphertext, Parameters, RLWEBlindRotationKey,
    SecretKeyPack, lwe_modulus_switch_w_trace,
};
use trace::basic_ops::SumHadamardTrace;
use trace::cmp_trace::lt_trace::LTTables;
use trace::key_switching_trace::KeySwitchingTrace;
use trace::pbs_trace::PBSTrace;
use trace::{AccTrace, BlindRotationTrace};

/// The evaluator of the homomorphic encryption scheme.
#[derive(Debug, Clone)]
pub struct EvaluationKey<Q: NTTField> {
    /// Blind rotation key.
    blind_rotation_key: RLWEBlindRotationKey<Q>,
    /// Key switching key.
    key_switching_key: KeySwitchingKeyEnum<Q>,
    /// The parameters of the fully homomorphic encryption scheme.
    parameters: Parameters<Q>,
}

impl<Q: NTTField> EvaluationKey<Q> {
    /// Returns the parameters of this [`EvaluationKey<F>`].
    #[inline]
    pub fn parameters(&self) -> &Parameters<Q> {
        &self.parameters
    }

    /// Creates a new [`EvaluationKey`] from the given [`SecretKeyPack`].
    pub fn new(secret_key_pack: &SecretKeyPack<Q>) -> Self {
        let parameters = secret_key_pack.parameters();

        let blind_rotation_key = RLWEBlindRotationKey::generate(secret_key_pack);

        let key_switching_key =
            KeySwitchingKeyEnum::RLWE(KeySwitchingRLWEKey::generate(secret_key_pack));

        Self {
            blind_rotation_key,
            key_switching_key,
            parameters: *parameters,
        }
    }

    /// Complete the bootstrapping operation with LWE Ciphertext *`c`* and lookup table `lut`.
    pub fn bootstrap(
        &self,
        c: LWECiphertext<Q>,
        lut: Polynomial<Q>,
    ) -> (LWECiphertext<Q>, PBSTrace<Q>) {
        let parameters = self.parameters();
        let pre = parameters.process_before_blind_rotation();

        let (c_prime, modulus_switching_trace) =
            lwe_modulus_switch_w_trace(&c, pre.twice_ring_dimension_value());

        // -- Blind Rotation with Trace --
        let log_coeff_count = parameters.ring_dimension().trailing_zeros() as usize;
        let log_num_round = parameters
            .lwe_dimension()
            .next_power_of_two()
            .trailing_zeros() as usize;
        let mut acc_trace = AccTrace::<Q>::new(log_coeff_count, log_num_round);
        let mut hadamard_trace = SumHadamardTrace::<Q>::new(
            parameters.blind_rotation_basis().decompose_len() << 1,
            log_coeff_count,
            log_num_round,
        );

        let mut acc = self.blind_rotation_key.blind_rotate_w_trace(
            lut,
            &c_prime,
            parameters.blind_rotation_basis(),
            &mut acc_trace,
            &mut hadamard_trace,
        );

        let lt_general_tables = LTTables::<Q>::new(&parameters.blind_rotation_basis(), None);

        let blind_rotation_trace = BlindRotationTrace {
            log_coeff_count,
            log_num_round,
            acc_trace,
            hadamard_trace,
            tables: lt_general_tables,
        };
        // blind_rotation_trace.finalize(parameters.lwe_dimension());
        // -- End --

        acc.b_mut()[0] += Q::new(Q::MODULUS_VALUE >> 3u32);

        let ksk = match self.key_switching_key {
            KeySwitchingKeyEnum::RLWE(ref ksk) => ksk,
            _ => panic!("Unable to get the corresponding key switching key!"),
        };

        // -- Key Switching & Sample Extraction with Trace --
        let ks_log_coeff_count = parameters
            .lwe_dimension()
            .next_power_of_two()
            .trailing_zeros() as usize;
        let ks_log_rounds = 0;
        let ks_hadamard_len = ksk.num_rlwes_in_key();
        let mut ks_hadamard_trace =
            SumHadamardTrace::<Q>::new(ks_hadamard_len, ks_log_coeff_count, ks_log_rounds);
        let mut decomposed_polys = Vec::new();
        let mut permutation_trace = None;
        // perform key switching, along with sample extraction
        let (output_lwe, sample_extraction_trace) = ksk.key_switch_for_rlwe_w_trace(
            acc,
            &mut ks_hadamard_trace,
            &mut decomposed_polys,
            &mut permutation_trace,
        );

        let ks_basis = Basis::new(parameters.key_switching_basis_bits());
        let ks_lt_tables = LTTables::<Q>::new(&ks_basis, None);
        let key_switching_trace = KeySwitchingTrace {
            log_lwe_dim: ks_log_coeff_count,
            log_rlwe_dim: log_coeff_count,
            log_coeff_count: ks_log_coeff_count,
            hadamard_trace: ks_hadamard_trace,
            permutation_trace,
            decomposed_polys,
            lt_tables: ks_lt_tables,
        };
        // -- End --

        let pbs_trace = PBSTrace {
            modulus_switching_trace,
            blind_rotation_trace,
            key_switching_trace,
            sample_extraction_trace,
        };

        (output_lwe, pbs_trace)
    }
}

/// Evaluator
#[derive(Debug, Clone)]
pub struct Evaluator<Q: NTTField> {
    ek: EvaluationKey<Q>,
}

impl<Q: NTTField> Evaluator<Q> {
    /// Create a new instance.
    #[inline]
    pub fn new(sk: &SecretKeyPack<Q>) -> Self {
        Self {
            ek: EvaluationKey::new(sk),
        }
    }

    /// Returns a reference to the parameters of this [`Evaluator<F>`].
    #[inline]
    pub fn parameters(&self) -> &Parameters<Q> {
        self.ek.parameters()
    }

    /// Complete the bootstrapping operation with LWE Ciphertext *`c`* and lookup table `lut`.
    #[inline]
    pub fn bootstrap(
        &self,
        c: LWECiphertext<Q>,
        lut: Polynomial<Q>,
    ) -> (LWECiphertext<Q>, PBSTrace<Q>) {
        self.ek.bootstrap(c, lut)
    }

    /// Performs the homomorphic not operation.
    ///
    /// # Arguments
    ///
    /// * Input: ciphertext `c`, with message `true`(resp. `false`).
    /// * Output: ciphertext with message `false`(resp. `true`).
    ///
    /// Link: <https://eprint.iacr.org/2020/086>
    pub fn not(&self, c: &LWECiphertext<Q>) -> LWECiphertext<Q> {
        let parameters = self.parameters();

        let mut neg = c.neg();
        *neg.b_mut() += Q::new(parameters.lwe_cipher_modulus_value() >> 2u32);
        neg
    }

    /// Performs the homomorphic nand operation.
    ///
    /// # Arguments
    ///
    /// * Input: ciphertext `c0`, with message `a`.
    /// * Input: ciphertext `c1`, with message `b`.
    /// * Output: ciphertext with message `not(a and b)`.
    pub fn nand(
        &self,
        c0: &LWECiphertext<Q>,
        c1: &LWECiphertext<Q>,
    ) -> (LWECiphertext<Q>, PBSTrace<Q>) {
        let parameters = self.parameters();

        let add = c0.add_component_wise_ref(c1);

        let lut = init_nand_lut(parameters.ring_dimension(), parameters.lut_step());

        self.bootstrap(add, lut)
    }

    /// Performs the homomorphic and operation.
    ///
    /// # Arguments
    ///
    /// * Input: ciphertext `c0`, with message `a`.
    /// * Input: ciphertext `c1`, with message `b`.
    /// * Output: ciphertext with message `a and b`.
    pub fn and(
        &self,
        c0: &LWECiphertext<Q>,
        c1: &LWECiphertext<Q>,
    ) -> (LWECiphertext<Q>, PBSTrace<Q>) {
        let parameters = self.parameters();

        let add = c0.add_component_wise_ref(c1);

        let lut: Polynomial<Q> =
            init_and_majority_lut(parameters.ring_dimension(), parameters.lut_step());

        self.bootstrap(add, lut)
    }

    /// Performs the homomorphic or operation.
    ///
    /// # Arguments
    ///
    /// * Input: ciphertext `c0`, with message `a`.
    /// * Input: ciphertext `c1`, with message `b`.
    /// * Output: ciphertext with message `a or b`.
    pub fn or(
        &self,
        c0: &LWECiphertext<Q>,
        c1: &LWECiphertext<Q>,
    ) -> (LWECiphertext<Q>, PBSTrace<Q>) {
        let parameters = self.parameters();

        let add = c0.add_component_wise_ref(c1);

        let lut = init_or_lut(parameters.ring_dimension(), parameters.lut_step());

        self.bootstrap(add, lut)
    }

    /// Performs the homomorphic nor operation.
    ///
    /// # Arguments
    ///
    /// * Input: ciphertext `c0`, with message `a`.
    /// * Input: ciphertext `c1`, with message `b`.
    /// * Output: ciphertext with message `not(a or b)`.
    pub fn nor(
        &self,
        c0: &LWECiphertext<Q>,
        c1: &LWECiphertext<Q>,
    ) -> (LWECiphertext<Q>, PBSTrace<Q>) {
        let parameters = self.parameters();

        let add = c0.add_component_wise_ref(c1);

        let lut = init_nor_lut(parameters.ring_dimension(), parameters.lut_step());

        self.bootstrap(add, lut)
    }

    /// Performs the homomorphic xor operation.
    ///
    /// # Arguments
    ///
    /// * Input: ciphertext `c0`, with message `a`.
    /// * Input: ciphertext `c1`, with message `b`.
    /// * Output: ciphertext with message `a xor b`.
    pub fn xor(
        &self,
        c0: &LWECiphertext<Q>,
        c1: &LWECiphertext<Q>,
    ) -> (LWECiphertext<Q>, PBSTrace<Q>) {
        let parameters = self.parameters();

        let mut sub = c0.sub_component_wise_ref(c1);
        sub.double_inplace();

        let lut = init_xor_lut(parameters.ring_dimension(), parameters.lut_step());

        self.bootstrap(sub, lut)
    }

    /// Performs the homomorphic xnor operation.
    ///
    /// # Arguments
    ///
    /// * Input: ciphertext `c0`, with message `a`.
    /// * Input: ciphertext `c1`, with message `b`.
    /// * Output: ciphertext with message `not(a xor b)`.
    pub fn xnor(
        &self,
        c0: &LWECiphertext<Q>,
        c1: &LWECiphertext<Q>,
    ) -> (LWECiphertext<Q>, PBSTrace<Q>) {
        let parameters = self.parameters();

        let mut sub = c0.sub_component_wise_ref(c1);
        sub.double_inplace();

        let lut = init_xnor_lut(parameters.ring_dimension(), parameters.lut_step());

        self.bootstrap(sub, lut)
    }

    /// Performs the homomorphic majority operation.
    ///
    /// # Arguments
    ///
    /// * Input: ciphertext `c0`, with message `a`.
    /// * Input: ciphertext `c1`, with message `b`.
    /// * Input: ciphertext `c2`, with message `c`.
    /// * Output: ciphertext with message `(a & b) | (b & c) | (a & c)`.
    ///   If there are two or three `true`(resp. `false`) in `a`, `b` and `c`, it will return `true`(resp. `false`).
    pub fn majority(
        &self,
        c0: &LWECiphertext<Q>,
        c1: &LWECiphertext<Q>,
        c2: &LWECiphertext<Q>,
    ) -> (LWECiphertext<Q>, PBSTrace<Q>) {
        let parameters = self.parameters();

        let mut add = c0.add_component_wise_ref(c1);
        add.add_inplace_component_wise(c2);

        let lut = init_and_majority_lut(parameters.ring_dimension(), parameters.lut_step());

        self.bootstrap(add, lut)
    }

    /// Performs the homomorphic mux operation.
    ///
    /// # Arguments
    ///
    /// * Input: ciphertext `c0`, with message `a`.
    /// * Input: ciphertext `c1`, with message `b`.
    /// * Input: ciphertext `c2`, with message `c`.
    /// * Output: ciphertext with message `if a {b} else {c}`.
    ///   If `a` is `true`, it will return `b`. If `a` is `false`, it will return `c`.
    pub fn mux(
        &self,
        c0: &LWECiphertext<Q>,
        c1: &LWECiphertext<Q>,
        c2: &LWECiphertext<Q>,
    ) -> (LWECiphertext<Q>, PBSTrace<Q>) {
        let parameters = self.parameters();

        let not_c0 = self.not(c0);

        let ((mut t0, _), (t1, _)) = rayon::join(|| self.and(c0, c1), || self.and(&not_c0, c2));

        // (a & b) | (!a & c)
        t0.add_inplace_component_wise(&t1);

        let lut = init_or_lut(parameters.ring_dimension(), parameters.lut_step());

        self.bootstrap(t0, lut)
    }
}

/// init lut for bootstrapping which performs homomorphic `nand`.
fn init_nand_lut<F>(
    rlwe_dimension: usize,
    twice_rlwe_dimension_div_lwe_modulus: usize,
) -> Polynomial<F>
where
    F: NTTField,
{
    let q = F::MODULUS_VALUE;
    let q_div_8 = F::new(q >> 3u32);
    let neg_q_div_8 = F::new(q - q_div_8.value());

    init_nand_and_majority_lut(
        rlwe_dimension,
        twice_rlwe_dimension_div_lwe_modulus,
        q_div_8,
        neg_q_div_8,
    )
}

/// init lut for bootstrapping which performs homomorphic `and` or `majority`.
fn init_and_majority_lut<F>(
    rlwe_dimension: usize,
    twice_rlwe_dimension_div_lwe_modulus: usize,
) -> Polynomial<F>
where
    F: NTTField,
{
    let q = F::MODULUS_VALUE;
    let q_div_8 = F::new(q >> 3u32);
    let neg_q_div_8 = F::new(q - q_div_8.value());

    init_nand_and_majority_lut(
        rlwe_dimension,
        twice_rlwe_dimension_div_lwe_modulus,
        neg_q_div_8,
        q_div_8,
    )
}

/// init lut for bootstrapping which performs homomorphic `nand`, `and` or `majority`.
fn init_nand_and_majority_lut<F>(
    rlwe_dimension: usize,
    twice_rlwe_dimension_div_lwe_modulus: usize,
    value_0_1: F, // [−q/8, 3q/8)
    value_2_3: F, // [3q/8, 7q/8)
) -> Polynomial<F>
where
    F: NTTField,
{
    let mut v = Polynomial::zero(rlwe_dimension);

    let mid = (rlwe_dimension >> 1) + (rlwe_dimension >> 2); // 3N/4

    v[..mid]
        .iter_mut()
        .step_by(twice_rlwe_dimension_div_lwe_modulus)
        .for_each(|a| *a = value_0_1);

    v[mid..]
        .iter_mut()
        .step_by(twice_rlwe_dimension_div_lwe_modulus)
        .for_each(|a| *a = value_2_3);

    v
}

/// init lut for bootstrapping which performs homomorphic `or` or `xor`.
fn init_or_lut<F>(
    rlwe_dimension: usize,
    twice_rlwe_dimension_div_lwe_modulus: usize,
) -> Polynomial<F>
where
    F: NTTField,
{
    let q = F::MODULUS_VALUE;
    let q_div_8 = F::new(q >> 3u32);
    let neg_q_div_8 = F::new(q - q_div_8.value());

    init_or_nor_lut(
        rlwe_dimension,
        twice_rlwe_dimension_div_lwe_modulus,
        q_div_8,
        neg_q_div_8,
    )
}

/// init lut for bootstrapping which performs homomorphic `nor` or `xnor`.
fn init_nor_lut<F>(
    rlwe_dimension: usize,
    twice_rlwe_dimension_div_lwe_modulus: usize,
) -> Polynomial<F>
where
    F: NTTField,
{
    let q = F::MODULUS_VALUE;
    let q_div_8 = F::new(q >> 3u32);
    let neg_q_div_8 = F::new(q - q_div_8.value());

    init_or_nor_lut(
        rlwe_dimension,
        twice_rlwe_dimension_div_lwe_modulus,
        neg_q_div_8,
        q_div_8,
    )
}

/// init lut for bootstrapping which performs homomorphic `xor`.
fn init_xor_lut<F>(
    rlwe_dimension: usize,
    twice_rlwe_dimension_div_lwe_modulus: usize,
) -> Polynomial<F>
where
    F: NTTField,
{
    let q = F::MODULUS_VALUE;
    let q_div_8 = F::new(q >> 3u32);
    let neg_q_div_8 = F::new(q - q_div_8.value());

    init_xor_xnor_lut(
        rlwe_dimension,
        twice_rlwe_dimension_div_lwe_modulus,
        q_div_8,
        neg_q_div_8,
    )
}

/// init lut for bootstrapping which performs homomorphic `xnor`.
fn init_xnor_lut<F>(
    rlwe_dimension: usize,
    twice_rlwe_dimension_div_lwe_modulus: usize,
) -> Polynomial<F>
where
    F: NTTField,
{
    let q = F::MODULUS_VALUE;
    let q_div_8 = F::new(q >> 3u32);
    let neg_q_div_8 = F::new(q - q_div_8.value());

    init_xor_xnor_lut(
        rlwe_dimension,
        twice_rlwe_dimension_div_lwe_modulus,
        neg_q_div_8,
        q_div_8,
    )
}

/// init lut for bootstrapping which performs homomorphic `or`, `nor`, `xor` or `xnor`.
fn init_or_nor_lut<F>(
    rlwe_dimension: usize,
    twice_rlwe_dimension_div_lwe_modulus: usize,
    value_1_2: F, // [q/8, 5q/8)
    value_3_0: F, // [−3q/8, q/8)
) -> Polynomial<F>
where
    F: NTTField,
{
    let mut v = Polynomial::zero(rlwe_dimension);

    let mid = rlwe_dimension >> 2; // N/4

    v[..mid]
        .iter_mut()
        .step_by(twice_rlwe_dimension_div_lwe_modulus)
        .for_each(|a| *a = value_3_0);

    v[mid..]
        .iter_mut()
        .step_by(twice_rlwe_dimension_div_lwe_modulus)
        .for_each(|a| *a = value_1_2);

    v
}

/// init lut for bootstrapping which performs homomorphic `xor` or `xnor`.
fn init_xor_xnor_lut<F>(
    rlwe_dimension: usize,
    twice_rlwe_dimension_div_lwe_modulus: usize,
    value_2: F, // [q/4, 3q/4)
    value_0: F, // [−q/4, q/4)
) -> Polynomial<F>
where
    F: NTTField,
{
    let mut v = Polynomial::zero(rlwe_dimension);

    let mid = rlwe_dimension >> 1; // N/2

    v[..mid]
        .iter_mut()
        .step_by(twice_rlwe_dimension_div_lwe_modulus)
        .for_each(|a| *a = value_0);

    v[mid..]
        .iter_mut()
        .step_by(twice_rlwe_dimension_div_lwe_modulus)
        .for_each(|a| *a = value_2);

    v
}
