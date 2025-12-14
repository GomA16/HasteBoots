use algebra::{DenseMultilinearExtension, Field, PolynomialInfo};
use helper::{FiatShamirTranscript, Transcript};
use serde::Serialize;
use std::rc::Rc;
use sumcheck::{MLSumcheck, Proof, verifier::SubClaim};
use trace::NTTTraceMLE;

use crate::LagrangeKernel;
use crate::SumcheckClaim;
use crate::SumcheckInfo;
use crate::SumcheckInstance;
use crate::SumcheckPIOP;
use crate::ntt::fourier_eval::NTTFourierProof;
use crate::ntt::{NTTFourierEvalIOP, NTTFourierEvalInfo};

pub struct NTTEvalIOP<F: Field> {
    _marker: std::marker::PhantomData<F>,
}

pub struct NTTEvalInstance<F: Field> {
    pub log_coeff_count: usize,
    pub coefficients: Rc<DenseMultilinearExtension<F>>,
    pub ntt_table: Rc<Vec<F>>,
    pub point_u: Vec<F>,
    pub evaluations_at_u: F,
}

#[derive(Serialize)]
pub struct NTTEvalInfo<F: Field> {
    pub log_coeff_count: usize,
    #[serde(skip)]
    pub ntt_table: Rc<Vec<F>>,
    #[serde(skip)]
    pub point_u: Vec<F>,
    pub evaluations_at_u: F,
}

#[derive(Serialize)]
pub struct NTTEvalProof<F: Field> {
    // claimed sum in the sumcheck protocol
    // #[serde(skip)]
    // pub claimed_sum: F,
    // polynomial info in the sumcheck protocol
    #[serde(skip)]
    pub poly_info: PolynomialInfo,
    // sumcheck proofs for the NTT evaluation
    pub sumcheck_proof: Proof<F>,
    // subclaim1: evaluation of the coeffient polynomial at point_v
    pub coeff_eval_at_v: F,
    // subclaim2: evaluation of the fourier polynomial at point_v
    pub fourier_eval_at_v: F,
    // fourier evaluation proof for subclaim 2
    pub fourier_eval_proof: NTTFourierProof<F>,
}

pub struct NTTEvalProverState<F: Field> {
    pub fourier_at_u: Rc<DenseMultilinearExtension<F>>,
    pub point_v: Vec<F>,
}

pub struct NTTEvalVerifierSubclaim<F: Field> {
    pub point_v: Vec<F>,
    pub coeff_eval_at_v: F,
}

impl<F: Field> NTTEvalInstance<F> {
    pub fn from(trace: &NTTTraceMLE<F>, point_u: &[F]) -> Self {
        NTTEvalInstance {
            log_coeff_count: trace.num_vars(),
            coefficients: Rc::clone(&trace.coefficients),
            ntt_table: Rc::clone(&trace.ntt_table),
            point_u: point_u.to_vec(),
            evaluations_at_u: trace.evaluations.evaluate(point_u),
        }
    }
}

impl<F: Field> SumcheckInstance<F> for NTTEvalInstance<F> {
    type Info = NTTEvalInfo<F>;

    fn info(&self) -> Self::Info {
        NTTEvalInfo {
            log_coeff_count: self.log_coeff_count,
            ntt_table: Rc::clone(&self.ntt_table),
            point_u: self.point_u.clone(),
            evaluations_at_u: self.evaluations_at_u,
        }
    }
}

impl<F: Field> SumcheckInfo<F> for NTTEvalInfo<F> {
    fn num_sumchecks(&self) -> usize {
        1
    }

    fn sumcheck_num_vars(&self) -> usize {
        self.log_coeff_count
    }
}

impl<F: Field + Serialize> SumcheckPIOP<F> for NTTEvalIOP<F> {
    type Instance = NTTEvalInstance<F>;
    type Info = NTTEvalInfo<F>;
    type Proof = NTTEvalProof<F>;
    type ProverState = NTTEvalProverState<F>;
    type VerifierSubclaim = NTTEvalVerifierSubclaim<F>;
    type FSTranscript = Transcript<F>;

    fn prover(
        trans: &mut Self::FSTranscript,
        instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState) {
        // let statement = instance.info();
        let info = instance.info();
        trans.append_message(b"[NTT Evaluation Statement]", &info);

        let mut sumcheck_claim = SumcheckClaim::new(instance.log_coeff_count);
        let prover_state =
            Self::prover_batch_sumcheck(instance, &mut sumcheck_claim, &[F::one()], None).unwrap();
        let (sumcheck_proof, sumcheck_state) =
            MLSumcheck::<F>::prove(trans, sumcheck_claim.poly_ref())
                .expect("[NTTPolyIOP - Prover] Fail to generate sumcheck proof");

        let coeff_eval_at_v = instance.coefficients.evaluate(&sumcheck_state.randomness);
        let fourier_eval_at_v = prover_state
            .fourier_at_u
            .evaluate(&sumcheck_state.randomness);

        let fourier_eval_subclaim = NTTFourierEvalInfo {
            log_coeff_count: instance.log_coeff_count,
            ntt_table: Rc::clone(&instance.ntt_table),
            point_u: instance.point_u.clone(),
            point_v: sumcheck_state.randomness.clone(),
            eval: fourier_eval_at_v,
        };
        let fourier_eval_proof = NTTFourierEvalIOP::<F>::prover(trans, &fourier_eval_subclaim);

        (
            Self::Proof {
                poly_info: sumcheck_claim.poly_ref().info(),
                sumcheck_proof,
                coeff_eval_at_v,
                fourier_eval_at_v,
                fourier_eval_proof,
            },
            prover_state,
        )
    }

    fn verifier(
        trans: &mut Self::FSTranscript,
        info: &Self::Info,
        proof: &Self::Proof,
    ) -> (bool, Self::VerifierSubclaim) {
        trans.append_message(b"[NTT Evaluation Statement]", &info);

        let mut res = true;

        let mut sumcheck_subclaim = MLSumcheck::verify(
            trans,
            &proof.poly_info,
            MLSumcheck::extract_sum(&proof.sumcheck_proof),
            &proof.sumcheck_proof,
        )
        .expect("[NTTEvalIOP - Verifier] Fail to verify the sumcheck");

        Self::verifier_compute_subclaim(info, proof, &mut sumcheck_subclaim, &[F::one()], None);
        res &= sumcheck_subclaim.expected_evaluations.is_zero();

        let ntt_fourier_eval_info = NTTFourierEvalInfo {
            log_coeff_count: info.point_u.len(),
            ntt_table: Rc::clone(&info.ntt_table),
            point_u: info.point_u.clone(),
            point_v: sumcheck_subclaim.point.clone(),
            eval: proof.fourier_eval_at_v,
        };

        res &=
            NTTFourierEvalIOP::verifier(trans, &ntt_fourier_eval_info, &proof.fourier_eval_proof);
        (
            res,
            Self::VerifierSubclaim {
                point_v: sumcheck_subclaim.point,
                coeff_eval_at_v: proof.coeff_eval_at_v,
            },
        )
    }

    fn prover_batch_sumcheck(
        instance: &Self::Instance,
        claim: &mut SumcheckClaim<F>,
        randomness: &[F],
        lagrange_kernel: Option<&LagrangeKernel<F>>,
    ) -> Option<Self::ProverState> {
        assert!(
            lagrange_kernel.is_none(),
            "Lagrange kernel is not supported in NTTEvalIOP"
        );

        let fourier_at_u = Rc::new(init_fourier_table(&instance.point_u, &instance.ntt_table));
        claim.poly_mut().add_product(
            [Rc::clone(&fourier_at_u), Rc::clone(&instance.coefficients)],
            randomness[0],
        );
        *claim.sum_mut() += instance.evaluations_at_u * randomness[0];
        Some(Self::ProverState {
            fourier_at_u,
            point_v: Vec::with_capacity(instance.log_coeff_count),
        })
    }

    fn verifier_compute_subclaim(
        _info: &Self::Info,
        proof: &Self::Proof,
        subclaim: &mut SubClaim<F>,
        randomness: &[F],
        kernel_at_r: Option<F>,
    ) {
        assert!(
            kernel_at_r.is_none(),
            "Lagrange kernel is not supported in NTTEvalIOP"
        );

        assert_eq!(randomness.len(), 1);
        subclaim.expected_evaluations -=
            proof.coeff_eval_at_v * proof.fourier_eval_at_v * randomness[0];
    }
}

// Naive implementation for initializing F(u, x) in NTT (for testing purpose).
// In negacyclic NTT where the ring is defined as R = F[x] / (x^N + 1), the Fourier matrix is different since we choose these points:
// X = ω^1, ω^3, ..., ω^{2N-1} such that X^N = -1.
//
// # Arguments
// * u: the random point
// * ntt_table: It stores the NTT table: ω^0, ω^1, ..., ω^{2N - 1}
//
// # Returns
// * The MLE for F(u, x)
pub fn naive_init_fourier_table<F: Field>(
    u: &[F],
    ntt_table: &[F],
) -> DenseMultilinearExtension<F> {
    let log_n = u.len();
    let m = ntt_table.len(); // M = 2N = 2 * (1 << dim)

    let mut evaluations = vec![F::one(); 1 << log_n];

    // F^R(u, x) = \prod_{i=0}^{\log{N-1}} ((1 - u_i) + u_i * {ω_{2^{i + 1}} ^ X}) * ω^{2^i * x_i}
    // ω_{2^{i + 1}} = ω^ (M / (2^{i+1})), which is the 2^{i+1}-th root of unity
    for (x, eval_at_x) in evaluations.iter_mut().enumerate() {
        for (i, &u_i) in u.iter().enumerate().take(log_n) {
            let idx = (1 << (log_n - i)) * x % m;
            // use little-endian representation so that x_i is the i-th bit of x
            let x_i = (x >> i) & 1;
            let x_i_idx = (1 << i) * x_i;
            *eval_at_x *= ((F::one() - u_i) + u_i * ntt_table[idx]) * ntt_table[x_i_idx];
        }
    }

    DenseMultilinearExtension::from_evaluations_vec(log_n, evaluations)
}

/// Generate MLE for the Fourier function F(u, x) for x \in \{0, 1\}^dim where u is the random point.
/// Dynamic programming implementation for initializing F(u, x) in NTT with linear time complexity O(N).
///
/// # Arguments
/// * u: the random point
/// * ω: It stores the NTT table: ω^0, ω^1, ..., ω^{2N - 1}
pub fn init_fourier_table<F: Field>(u: &[F], ntt_table: &[F]) -> DenseMultilinearExtension<F> {
    let log_n = u.len(); // n = 1 << dim
    let m = ntt_table.len(); // m = 2n = 2 * (1 << dim)

    // It stores the evaluations of all F(u, x) for x \in {0, 1}^dim.
    // Note that in our implementation, we use little endian form, so the index `0b1011`
    // represents the point `P(1,1,0,1)` in {0,1}^`dim` where x_0 = 1, x_1 = 1, x_2 = 0, x_3 = 1.
    let mut evaluations: Vec<_> = vec![F::zero(); 1 << log_n];
    evaluations[0] = F::one();

    // * Compute \prod_{i=0}^{\log{N-1}} ((1 - u_i) + u_i * {ω_{2^{i + 1}} ^ X}) * ω^{2^i * x_i}
    // We need to update the table from i = 0 to log_n - 1 since ω_{2^{i + 1}} ^ X takes on only 2^{i + 1} distinct values.
    for (i, u_i) in u.iter().enumerate() {
        let last_table_size = 1 << i;

        for j in (0..1 << (i + 1)).rev() {
            // `idx` denotes the index for ω^{2^{i+1} * X} stored in the NTT table where X shares the same last i + 1 bits with j
            let idx = (1 << (log_n - i)) * j % m;
            // `bit` denotes the most significant bit of j. If bit = 1, we need to multiply by ω^{2^i}
            let bit = j >> i;
            if bit == 1 {
                evaluations[j] = evaluations[j % last_table_size]
                    * (F::one() - *u_i + *u_i * ntt_table[idx])
                    * ntt_table[last_table_size];
            } else {
                evaluations[j] =
                    evaluations[j % last_table_size] * (F::one() - u_i + *u_i * ntt_table[idx]);
            }
        }
    }
    DenseMultilinearExtension::from_evaluations_vec(log_n, evaluations)
}

#[cfg(test)]
mod test {
    use crate::ntt::{NTTEvalInstance, ntt_eval::init_fourier_table};

    use super::NTTEvalIOP;
    use crate::{SumcheckInstance, SumcheckPIOP};
    use algebra::{
        FieldUniformSampler, NTTField,
        derive::{DecomposableField, FheField, Field, NTT, Prime},
        transformation::AbstractNTT,
    };
    use helper::Transcript;
    use rand_distr::Distribution;
    use trace::NTTTrace;

    #[derive(Field, DecomposableField, FheField, Prime, NTT)]
    #[modulus = 132120577]
    pub struct Fp32(u32);
    // field type
    type FF = Fp32;

    #[test]
    fn test_ntt_eval_iop() {
        let log_coeff_count = 10;
        let log_num_ntt = 0;
        let uniform = <FieldUniformSampler<FF>>::new();
        let mut rng = rand::rng();
        let ntt_trace = NTTTrace::<FF>::random(log_coeff_count, log_num_ntt, &mut rng);

        let point_u = uniform
            .sample_iter(&mut rng)
            .take(log_coeff_count)
            .collect::<Vec<_>>();
        let ntt_eval_instance = NTTEvalInstance::from(&ntt_trace.into(), &point_u);
        let ntt_eval_info = ntt_eval_instance.info();

        let mut prover_trans = Transcript::<FF>::default();
        let (proof, _) = NTTEvalIOP::<FF>::prover(&mut prover_trans, &ntt_eval_instance);

        let mut verifier_trans = Transcript::<FF>::default();
        let (res, _) = NTTEvalIOP::<FF>::verifier(&mut verifier_trans, &ntt_eval_info, &proof);
        assert!(res);
    }

    #[test]
    fn test_init_fourier_table() {
        let uniform = <FieldUniformSampler<FF>>::new();
        let mut rng = rand::rng();

        let dim = 10;
        let u = uniform.sample_iter(&mut rng).take(dim).collect::<Vec<_>>();
        let v = uniform.sample_iter(&mut rng).take(dim).collect::<Vec<_>>();

        let ntt_table = FF::get_ntt_table(dim as u32).unwrap().root_powers();
        let naive_fourier = super::naive_init_fourier_table(&u, &ntt_table);
        let fourier_mle = init_fourier_table(&u, &ntt_table);

        assert_eq!(fourier_mle.evaluate(&v), naive_fourier.evaluate(&v));
    }
}
