use algebra::{DenseMultilinearExtension, Field, MultilinearExtension, PolynomialInfo};
use helper::{FiatShamirTranscript, Transcript};
use serde::Serialize;
use std::rc::Rc;
use sumcheck::{MLSumcheck, Proof, verifier::SubClaim};
use trace::NTTTraceMLE;

use crate::{
    LagrangeKernel, SumcheckClaim, SumcheckInfo, SumcheckInstance, SumcheckPIOP, SumcheckPureProof,
    SumcheckPureProverState, SumcheckPureSubclaim,
    ntt::{
        NTTFourierEvalIOP, NTTFourierEvalInfo, fourier_eval::NTTFourierProof,
        ntt_eval::init_fourier_table,
    },
};

pub struct NTTMatrixEvalIOP<F: Field> {
    _marker: std::marker::PhantomData<F>,
}

/// Consider a coefficient matrix `C` where each row is the coefficients of a polynomial.
/// Matrix indexing: `C[Y][X]` where `X` is the row index and `Y` is the column index.
/// MLE indexing: `C(y0, y1, ..., yN, x0, x1, ..., xM)` where `Y = y0*2^0 + y1*2^1 + ...
/// + yN*2^N` and `X = x0*2^0 + x1*2^1 + ... + xM*2^M`. This is the little-endian
/// representation.
///
/// Consider the corresponding NTT matrix `A` where each row is the NTT evaluation of the
/// corresponding polynomial in `C`.
///
/// The NTT Matrix Evaluation Instance denotes a specific evaluation query on the NTT
/// matrix `A`. Specifically, given two points `u` and `v`, the instance denotes the
/// evaluation relation of the NTT matrix `A` at the point `(u, v)`. That is:
/// `A(u, v) = \sum_{k \in {0, 1}^log N} C(k, v) * F(u, k)` where `F` is the NTT Fourier
/// matrix defined in `fourier_eval.rs`.
pub struct NTTMatrixEvalInstance<F: Field> {
    // column dimension of the NTT matrix denoted by `N`
    pub log_coeff_count: usize,
    // row dimension of the NTT matrix denoted by `M`
    pub log_num_ntt: usize,
    /// MLE encoding the coefficient matrix
    pub coefficients: Rc<DenseMultilinearExtension<F>>,
    /// NTT table storing {1, w, w^2, ..., w^{2N-1}}
    /// where w is the primitive 2N-th root of unity
    pub ntt_table: Rc<Vec<F>>,
    /// evaluation points (u, v) where `u \in F^{log N}`
    pub point_u: Vec<F>,
    /// evaluation points (u, v) where `v \in F^{log M}`
    pub point_v: Vec<F>,
    /// claimed evaluation `A(u, v)`
    pub evaluations_at_u_v: F,
}

#[derive(Serialize)]
pub struct NTTMatrixEvalInfo<F: Field> {
    pub log_coeff_count: usize,
    #[serde(skip)]
    pub ntt_table: Rc<Vec<F>>,
    #[serde(skip)]
    pub point_u: Vec<F>,
    #[serde(skip)]
    pub point_v: Vec<F>,
    pub evaluations_at_u_v: F,
}

#[derive(Serialize)]
pub struct NTTMatrixEvalProof<F: Field> {
    pub poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<F>,
    pub coeff_eval_at_r_v: F,
    pub fourier_eval_at_u_r: F,
    pub fourier_eval_proof: NTTFourierProof<F>,
}

pub struct NTTMatrixEvalProverState<F: Field> {
    pub fourier_at_u: Rc<DenseMultilinearExtension<F>>,
    pub coeffs_at_v_back: Rc<DenseMultilinearExtension<F>>,
    pub point_r: Vec<F>,
}

pub struct NTTMatrixEvalVerifierSubclaim<F: Field> {
    pub point_r: Vec<F>,
    pub coeff_eval_at_r_v: F,
}

impl<F: Field> NTTMatrixEvalInstance<F> {
    /// Create an NTT Matrix Evaluation Instance from NTT Trace MLE and evaluation points
    pub fn from(trace: &NTTTraceMLE<F>, point_u: &[F], point_v: &[F]) -> Self {
        let point_u_v: Vec<F> = [point_u, point_v].concat();
        Self {
            log_coeff_count: trace.log_coeff_count,
            log_num_ntt: trace.log_num_ntt,
            coefficients: Rc::clone(&trace.coefficients),
            ntt_table: Rc::clone(&trace.ntt_table),
            point_u: point_u.to_vec(),
            point_v: point_v.to_vec(),
            evaluations_at_u_v: trace.evaluations.evaluate(&point_u_v),
        }
    }
}

impl<F: Field + Serialize> SumcheckInstance<F> for NTTMatrixEvalInstance<F> {
    type Info = NTTMatrixEvalInfo<F>;

    fn info(&self) -> Self::Info {
        NTTMatrixEvalInfo {
            log_coeff_count: self.log_coeff_count,
            ntt_table: Rc::clone(&self.ntt_table),
            point_u: self.point_u.clone(),
            point_v: self.point_v.clone(),
            evaluations_at_u_v: self.evaluations_at_u_v,
        }
    }

    fn num_vars(&self) -> usize {
        self.log_coeff_count
    }
}

impl<F: Field> SumcheckInfo<F> for NTTMatrixEvalInfo<F> {
    fn sumcheck_num_vars(&self) -> usize {
        self.log_coeff_count
    }

    fn num_sumchecks(&self) -> usize {
        1
    }

    fn num_vars(&self) -> usize {
        self.log_coeff_count
    }
}

impl<F: Field> SumcheckPureProof<F> for NTTMatrixEvalProof<F> {
    fn from_sumcheck(_sumcheck_claim: &SumcheckClaim<F>, _proof: Proof<F>) -> Self {
        unimplemented!("from_sumcheck is not implemented for NTTMatrixEvalProof");
    }

    fn poly_info(&self) -> &PolynomialInfo {
        &self.poly_info
    }

    fn sumcheck_proof(&self) -> &Proof<F> {
        &self.sumcheck_proof
    }
}

impl<F: Field> SumcheckPureProverState<F> for NTTMatrixEvalProverState<F> {
    fn from_sumcheck(_sumcheck_prover_state: sumcheck::prover::ProverState<F>) -> Self {
        unimplemented!("from_sumcheck is not implemented for NTTMatrixEvalProverState");
    }
}

impl<F: Field> SumcheckPureSubclaim<F> for NTTMatrixEvalVerifierSubclaim<F> {
    fn from_sumcheck(_sumcheck_subclaim: SubClaim<F>) -> Self {
        unimplemented!("from_sumcheck is not implemented for NTTMatrixEvalVerifierSubclaim");
    }
}

impl<F: Field + Serialize> SumcheckPIOP<F> for NTTMatrixEvalIOP<F> {
    type Instance = NTTMatrixEvalInstance<F>;
    type Info = NTTMatrixEvalInfo<F>;
    type Proof = NTTMatrixEvalProof<F>;
    type ProverState = NTTMatrixEvalProverState<F>;
    type VerifierSubclaim = NTTMatrixEvalVerifierSubclaim<F>;
    // type FSTranscript = Transcript<F>;

    fn prover(
        trans: &mut Transcript<F>,
        instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState) {
        let info = instance.info();
        trans.append_message(b"[NTT Matrix Evaluation Statement", &info);

        let mut sumcheck_claim = SumcheckClaim::new(info.sumcheck_num_vars());
        let mut prover_state =
            Self::prover_batch_sumcheck(instance, &mut sumcheck_claim, &[F::one()], None).unwrap();
        let (sumcheck_proof, sumcheck_state) = MLSumcheck::prove(trans, sumcheck_claim.poly_ref())
            .expect("[NTTMatrixEvalIOP] Fail to generate sumcheck proof");
        prover_state.point_r = sumcheck_state.randomness.clone();

        let coeff_eval_at_r_v = prover_state
            .coeffs_at_v_back
            .evaluate(&sumcheck_state.randomness);
        let fourier_eval_at_u_r = prover_state
            .fourier_at_u
            .evaluate(&sumcheck_state.randomness);

        let fourier_eval_subclaim = NTTFourierEvalInfo {
            log_coeff_count: instance.log_coeff_count,
            ntt_table: Rc::clone(&instance.ntt_table),
            point_u: instance.point_u.clone(),
            point_v: sumcheck_state.randomness.clone(),
            eval: fourier_eval_at_u_r,
        };

        let fourier_eval_proof = NTTFourierEvalIOP::<F>::prover(trans, &fourier_eval_subclaim);

        (
            Self::Proof {
                poly_info: sumcheck_claim.poly_ref().info(),
                sumcheck_proof,
                coeff_eval_at_r_v,
                fourier_eval_at_u_r,
                fourier_eval_proof,
            },
            prover_state,
        )
    }

    fn prover_without_evals(
        _trans: &mut Transcript<F>,
        _instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState) {
        unimplemented!("prover_wo_evals is not implemented for NTTMatrixEvalIOP");
    }

    fn verifier(
        trans: &mut Transcript<F>,
        info: &Self::Info,
        proof: &Self::Proof,
    ) -> (bool, Self::VerifierSubclaim) {
        trans.append_message(b"[NTT Matrix Evaluation Statement", &info);

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
            eval: proof.fourier_eval_at_u_r,
        };
        res &=
            NTTFourierEvalIOP::verifier(trans, &ntt_fourier_eval_info, &proof.fourier_eval_proof);

        (
            res,
            NTTMatrixEvalVerifierSubclaim {
                point_r: sumcheck_subclaim.point,
                coeff_eval_at_r_v: proof.coeff_eval_at_r_v,
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
            "Lagrange kernel is not supported in NTTMatrixEvalIOP"
        );

        // TODO optimize the init_fourier_table with intermediate storage
        let fourier_at_u = Rc::new(init_fourier_table(&instance.point_u, &instance.ntt_table));
        let coeffs_at_v_back = Rc::new(instance.coefficients.fix_variables_back(&instance.point_v));
        claim.poly_mut().add_product(
            [Rc::clone(&fourier_at_u), Rc::clone(&coeffs_at_v_back)],
            randomness[0],
        );
        *claim.sum_mut() += instance.evaluations_at_u_v * randomness[0];
        Some(Self::ProverState {
            fourier_at_u,
            coeffs_at_v_back,
            point_r: Vec::with_capacity(instance.log_coeff_count),
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
            "Lagrange kernel is not supported in NTTMatrixEvalIOP"
        );
        assert_eq!(randomness.len(), 1);
        subclaim.expected_evaluations -=
            proof.coeff_eval_at_r_v * proof.fourier_eval_at_u_r * randomness[0];
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::SumcheckPIOP;
    use algebra::{
        FieldUniformSampler,
        derive::{DecomposableField, FheField, Field, NTT, Prime},
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
    fn test_ntt_matrix_eval_iop() {
        let mut rng = rand::rng();
        let log_coeff_count = 10;
        let log_num_ntt = 12;
        let uniform = <FieldUniformSampler<FF>>::new();

        let ntt_trace = NTTTrace::<FF>::random(log_coeff_count, log_num_ntt, &mut rng);
        let point_u = uniform
            .sample_iter(&mut rng)
            .take(log_coeff_count)
            .collect::<Vec<_>>();
        let point_v = uniform
            .sample_iter(&mut rng)
            .take(log_num_ntt)
            .collect::<Vec<_>>();

        let ntt_matrix_eval_instance =
            NTTMatrixEvalInstance::from(&ntt_trace.into(), &point_u, &point_v);
        let ntt_eval_info = ntt_matrix_eval_instance.info();

        let mut prover_trans = Transcript::<FF>::default();
        let (proof, _) =
            NTTMatrixEvalIOP::<FF>::prover(&mut prover_trans, &ntt_matrix_eval_instance);

        let mut verifier_trans = Transcript::<FF>::default();
        let (res, _) =
            NTTMatrixEvalIOP::<FF>::verifier(&mut verifier_trans, &ntt_eval_info, &proof);
        assert!(res);
    }
}
