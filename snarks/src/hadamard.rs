//! This snarks implementation includes the proof generation for Hadamard product 
//! along with all NTT evaluations.
//! 
//! When considering the multiplication-related relation between polynomials, 
//! we are able to use Hadamard product to represent the element-wise relation
//! of their NTT evaluations.
//! 
//! To reduce the elements to be committed as more as possible and also to simplify
//! the proof structure, we only commit to the coefficient form of the polynomials.
//! After running the protocol for Hadamard product, it is reduced to querying the
//! evaluations of these polynomials at some random points. 
//! All these queries are answered by the NTT PIOP, reducing to the queries of 
//! their coefficient forms.
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
use piop::ntt::{
    NTTFourierEvalInfo, NTTFourierProof, NTTMatrixEvalIOP, NTTMatrixEvalInfo,
    NTTMatrixEvalInstance, NTTMatrixEvalProof,
};
use piop::{SumcheckInstance, SumcheckPIOP};
use serde::Serialize;
use trace::{
    ConvertToEF, EvaluableTraceEF, LookupTrace, LookupTraceMLE, LookupWitness, LookupWitnessHelper,
};
use trace::{SumHadamardTraceMLE};

#[derive(Default)]
pub struct HadamardSnarks<F, EF, S, PCS>
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

pub struct HadamardParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub pcs_params: PCS::Parameters,
    pub ntt_table: Rc<Vec<EF>>,
}

pub struct HadamardProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub log_num_overall_poly: usize,
    pub pcs_params: PCS::Parameters,
    pub commitment: PCS::Commitment,
    pub hadamard_info: BatchedSumHadamardInfo<EF>,
    pub hadamard_proof: BatchedSumHadamardProof<EF>,
    pub ntt_info: NTTMatrixEvalInfo<EF>,
    pub ntt_proof: NTTMatrixEvalProof<EF>,
    pub eval_proof: PCS::Proof,
}

impl<F, EF, S, PCS> HadamardSnarks<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F> + Serialize,
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
    pub fn setup(
        &self,
        trace: &SumHadamardTraceMLE<F>,
        code_spec: S,
        ntt_table: Vec<F>,
    ) -> HadamardParams<F, EF, S, PCS> {
        let num_oracle_vars = trace.num_vars() + trace.log_num_overall_poly();
        let pcs_params = PCS::setup(num_oracle_vars, Some(code_spec.clone()));
        
        HadamardParams {
            pcs_params,
            ntt_table: Rc::new(ntt_table.to_ef()),
        }
    }

    pub fn prove(
        &self,
        trans: &mut Transcript<EF>,
        trace_mle: &SumHadamardTraceMLE<F>,
        params: &HadamardParams<F, EF, S, PCS>,
    ) -> HadamardProof<F, EF, S, PCS> {
        let bit_poly = trace_mle.generate_overall_oracle();
        let (commitment, commitment_state) = PCS::commit(&params.pcs_params, &bit_poly);
        trans.append_message(b"Commit Phase", &commitment);

        let trace_ef = trace_mle.to_ef();
        let hadamard_instance = BatchedSumHadamardInstance::from(&trace_ef);
        let (mut hadamard_piop_proof, hadamard_piop_state) =
            HadamardPIOP::prover_without_evals(trans, &hadamard_instance);
        let hadamard_evals = trace_mle.evaluate_ef(&hadamard_piop_state.point_r);
        hadamard_piop_proof.append_eval(&hadamard_evals);
        trans.append_message(b"[PIOP Phase]", &hadamard_piop_proof);

        let point_u = hadamard_piop_state.point_r[..trace_mle.log_coeff_count].to_vec();
        let mut point_v = hadamard_piop_state.point_r[trace_mle.log_coeff_count..].to_vec();
        let point_bit_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            hadamard_evals.log_num_overall_poly(),
        );

        let bit_poly = Rc::new(bit_poly.to_ef());
        let bit_ntt_evals = hadamard_evals.pack_overall_poly_to_vec();
        let eval = compute_oracle_evals(&bit_ntt_evals, &point_bit_oracle);

        point_v.extend_from_slice(&point_bit_oracle);
        let ntt_instance = NTTMatrixEvalInstance::from_subclaim(
            &bit_poly,
            &params.ntt_table,
            &point_u,
            &point_v,
            eval,
        );
        let (ntt_piop_proof, ntt_piop_state) = NTTMatrixEvalIOP::prover(trans, &ntt_instance);
        trans.append_message(b"[PIOP Phase]", &ntt_piop_proof);

        let mut point_r_v = Vec::with_capacity(ntt_piop_state.point_r.len() + point_v.len());
        point_r_v.extend_from_slice(&ntt_piop_state.point_r);
        point_r_v.extend_from_slice(&point_v);
        let eval_proof = PCS::open(
            &params.pcs_params,
            &commitment,
            &commitment_state,
            &point_r_v,
            trans,
        );

        HadamardProof {
            log_num_overall_poly: trace_mle.log_num_overall_poly(),
            pcs_params: params.pcs_params.clone(),
            commitment,
            hadamard_info: hadamard_instance.info(),
            hadamard_proof: hadamard_piop_proof,
            ntt_info: ntt_instance.info(),
            ntt_proof: ntt_piop_proof,
            eval_proof,
        }
    }

    pub fn verify(&self, trans: &mut Transcript<EF>, proof: &HadamardProof<F, EF, S, PCS>) -> bool {
        trans.append_message(b"Commit Phase", &proof.commitment);
        let mut res = true;

        let (hadamard_res, hadamard_subclaim) =
            HadamardPIOP::verifier(trans, &proof.hadamard_info, &proof.hadamard_proof);
        res &= hadamard_res;

        trans.append_message(b"[PIOP Phase]", &proof.hadamard_proof);

        let point_u = hadamard_subclaim.point_r[..proof.ntt_info.log_coeff_count].to_vec();
        let mut point_v = hadamard_subclaim.point_r[proof.ntt_info.log_coeff_count..].to_vec();
        let point_bit_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            proof.log_num_overall_poly,
        );
        point_v.extend_from_slice(&point_bit_oracle);

        let (ntt_res, ntt_subclaim) =
            NTTMatrixEvalIOP::verifier(trans, &proof.ntt_info, &proof.ntt_proof);
        trans.append_message(b"[PIOP Phase]", &proof.ntt_proof);
        res &= ntt_res;

        let mut point_r_v = Vec::with_capacity(ntt_subclaim.point_r.len() + point_v.len());
        point_r_v.extend_from_slice(&ntt_subclaim.point_r);
        point_r_v.extend_from_slice(&point_v);
        
        let eval_res = PCS::verify(
            &proof.pcs_params,
            &proof.commitment,
            &point_r_v,
            proof.ntt_proof.coeff_eval_at_r_v,
            &proof.eval_proof,
            trans,
        );
        res &= eval_res;

        res
    }
}
