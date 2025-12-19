use core::time;
use std::rc::Rc;

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};
use helper::utils::compute_oracle_evals;
use helper::{FiatShamirTranscript, Transcript};
use pcs::PolynomialCommitmentScheme;
use pcs::utils::code;
use piop::hadamard::{
    BatchedSumHadamardInfo, BatchedSumHadamardInstance, BatchedSumHadamardProof, HadamardPIOP,
};
use piop::lookup::logup::LogUpInstanceInfo;
use piop::lookup::{LogUpIOP, LogUpInstance, LogUpProof};
use piop::ntt::{NTTFourierEvalInfo, NTTMatrixEvalInfo, NTTMatrixEvalInstance, NTTMatrixEvalProof};
use piop::{PackableEFProof, PackableProof, SumcheckInstance, SumcheckPIOP};
use rayon::iter::IntoParallelRefIterator;
use serde::Serialize;
use trace::{
    ConvertToEF, EvaluableTraceEF, LookupTrace, LookupTraceMLE, LookupWitness, LookupWitnessHelper,
};
use trace::{PackableTrace, SumHadamardTraceMLE};

pub struct PBSSnarks<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    _marker_f: std::marker::PhantomData<F>,
    _marker_ef: std::marker::PhantomData<EF>,
    _marker_s: std::marker::PhantomData<S>,
    _marker_pcs: std::marker::PhantomData<PCS>,
}

#[derive(Serialize)]
pub struct KeyNTTCommitment<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub num_vars: usize,
    pub key_ntt_0_commitment: Vec<PCS::Commitment>,
    pub key_ntt_1_commitment: Vec<PCS::Commitment>,
}

#[derive(Serialize)]
pub struct KeyNTTEvalProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub num_vars: usize,
    pub key_ntt_0_commitment: Vec<PCS::Proof>,
    pub key_ntt_1_commitment: Vec<PCS::Proof>,
}

#[derive(Serialize)]
pub struct PolyCommitment<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub num_vars: usize,
    pub bit_poly_commitment: Vec<PCS::Commitment>,
    pub sum_prod_commitment_0: PCS::Commitment,
    pub sum_prod_commitment_1: PCS::Commitment,
}

#[derive(Serialize)]
pub struct PolyEvalProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub num_vars: usize,
    pub bit_poly_commitment: Vec<PCS::Proof>,
    pub sum_prod_commitment_0: PCS::Proof,
    pub sum_prod_commitment_1: PCS::Proof,
}

impl<F, EF, S, PCS> KeyNTTCommitment<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<
            F,
            EF,
            S,
            Polynomial = DenseMultilinearExtension<F>,
            EFPolynomial = DenseMultilinearExtension<EF>,
            Point = EF,
        >,
{
    pub fn from(params: &PCS::Parameters, trace: &SumHadamardTraceMLE<F>) -> Self {
        let mut key_ntt_0_commitment = Vec::with_capacity(trace.num_trace);
        let mut key_ntt_1_commitment = Vec::with_capacity(trace.num_trace);

        for i in 0..trace.num_trace {
            let key_poly_0 = &trace.vec_trace[i].key_ntt.0;
            let key_poly_1 = &trace.vec_trace[i].key_ntt.1;

            let (commitment_0, _state_0) = PCS::commit(params, key_poly_0.as_ref());
            let (commitment_1, _state_1) = PCS::commit(params, key_poly_1.as_ref());

            key_ntt_0_commitment.push(commitment_0);
            key_ntt_1_commitment.push(commitment_1);
        }

        KeyNTTCommitment {
            num_vars: trace.log_coeff_count,
            key_ntt_0_commitment,
            key_ntt_1_commitment,
        }
    }
}

impl<F, EF, S, PCS> PolyCommitment<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<
            F,
            EF,
            S,
            Polynomial = DenseMultilinearExtension<F>,
            EFPolynomial = DenseMultilinearExtension<EF>,
            Point = EF,
        >,
{
    pub fn from(params: &PCS::Parameters, trace: &SumHadamardTraceMLE<F>) -> Self {
        let mut poly_commitment = Vec::with_capacity(trace.num_trace);
        trace.vec_trace.iter().for_each(|trace| {
            let (commitment, _state) = PCS::commit(params, trace.bit_poly.as_ref());
            poly_commitment.push(commitment);
        });
        let (sum_prod_commitment_0, _state_0) = PCS::commit(params, trace.sum_prod_poly.0.as_ref());
        let (sum_prod_commitment_1, _state_1) = PCS::commit(params, trace.sum_prod_poly.1.as_ref());
        PolyCommitment {
            num_vars: trace.log_coeff_count + trace.log_num_round,
            bit_poly_commitment: poly_commitment,
            sum_prod_commitment_0,
            sum_prod_commitment_1,
        }
    }
}

pub struct PBSSnarksProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub params: PBSSnarksParams<F, EF, S, PCS>,
    pub coeffs_commitment: PCS::Commitment,
    pub hadmard_info: BatchedSumHadamardInfo<EF>,
    pub hadmard_piop_proof: BatchedSumHadamardProof<EF>,
    pub ntt_eval_info: NTTMatrixEvalInfo<EF>,
    pub ntt_eval_piop_proof: NTTMatrixEvalProof<EF>,
    pub ntt_fourier_eval_info: NTTFourierEvalInfo<EF>,
    pub ntt_fourier_eval_piop_proof: NTTMatrixEvalProof<EF>,
    pub coeffs_eval_proof: PCS::Proof,
}

#[derive(Serialize)]
pub struct PBSSnarksParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub blk_size: usize,
    pub pcs_params: PCS::Parameters,
    pub key_ntt_commitment: KeyNTTCommitment<F, EF, S, PCS>,
    #[serde(skip)]
    pub ntt_table: Rc<Vec<EF>>,
}

impl<F, EF, S, PCS> PBSSnarks<F, EF, S, PCS>
where
    F: Field + Serialize,
    EF: AbstractExtensionField<F> + Serialize,
    S: Clone + Serialize,
    PCS: PolynomialCommitmentScheme<
            F,
            EF,
            S,
            Polynomial = DenseMultilinearExtension<F>,
            EFPolynomial = DenseMultilinearExtension<EF>,
            Point = EF,
        >,
{
    pub fn prove(
        &self,
        trans: &mut Transcript<EF>,
        trace_mle: &SumHadamardTraceMLE<F>,
        params: &PBSSnarksParams<F, EF, S, PCS>,
    ) -> PBSSnarksProof<F, EF, S, PCS> {
        let poly_commitment = PolyCommitment::<F, EF, S, PCS>::from(&params.pcs_params, trace_mle);

        trans.append_message(b"[Commit Phase]", &poly_commitment);

        let trace_ef = trace_mle.to_ef();
        let hadamard_instance = BatchedSumHadamardInstance::from(&trace_ef);
        let (mut hadamard_piop_proof, hadamard_piop_state) =
            HadamardPIOP::prover_without_evals(trans, &hadamard_instance);
        trans.append_message(b"[PIOP Phase]", &hadamard_piop_proof);
        let hadamard_trace_eval = trace_mle.evaluate_ef(&hadamard_piop_state.point_r);
        trans.append_message(b"[PIOP Eval Phase]", &hadamard_trace_eval);
        hadamard_piop_proof.append_eval(&hadamard_trace_eval);

        let point_u = hadamard_piop_state.point_r[..trace_mle.log_coeff_count].to_vec();
        let point_v = hadamard_piop_state.point_r[trace_mle.log_coeff_count..].to_vec();
        

        todo!()
    }
}
