use core::num;
use std::rc::Rc;

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};
use helper::utils::compute_oracle_evals;
use helper::{FiatShamirTranscript, Transcript};
use pcs::PolynomialCommitmentScheme;
use piop::hadamard::{BatchedSumHadamardProof, HadamardPIOP, SumHadamardInfo, SumHadamardInstance};
use piop::ntt::{NTTMatrixEvalIOP, NTTMatrixEvalInfo, NTTMatrixEvalInstance, NTTMatrixEvalProof};
use piop::{BatchedSumcheckPIOP, SumcheckInstance, SumcheckPIOP};
use serde::Serialize;
use trace::basic_ops::{SumHadamardTrace, SumHadamardTraceEval, SumHadamardTraceMLE};
use trace::{ConvertToEF, PackableEval, PackableTrace};

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

impl<F, EF, S, PCS> HadamardParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub fn new(code_spec: S, ntt_table: &Rc<Vec<EF>>, trace: &SumHadamardTrace<F>) -> Self {
        let num_oracle_vars = trace.num_vars() + trace.log_num_oracles();
        let pcs_params = PCS::setup(num_oracle_vars, Some(&code_spec));
        HadamardParams {
            pcs_params,
            ntt_table: ntt_table.clone(),
        }
    }
}

pub struct HadamardProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub log_num_oracles: usize,
    pub pcs_params: PCS::Parameters,
    pub commitment: PCS::Commitment,
    pub hadamard_info: Vec<SumHadamardInfo<EF>>,
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
    pub fn prove(
        &self,
        trans: &mut Transcript<EF>,
        trace: SumHadamardTrace<F>,
        params: &HadamardParams<F, EF, S, PCS>,
    ) -> HadamardProof<F, EF, S, PCS> {
        let bit_poly = trace.generate_oracle();
        let (commitment, commitment_state) = PCS::commit(&params.pcs_params, &bit_poly);
        trans.append_message(b"Commit Phase", &commitment);

        // [Hadamard PIOP] extract hadamard instances
        let trace_mle: SumHadamardTraceMLE<F> = trace.into();
        let trace_ef = trace_mle.to_ef();
        let hadamard_instance = SumHadamardInstance::from(&trace_ef);
        let hadamard_info = hadamard_instance
            .iter()
            .map(SumcheckInstance::info)
            .collect::<Vec<_>>();
        // [Hadamard PIOP] prove these instances using sumcheck-based protocol
        let (mut hadamard_piop_proof, hadamard_piop_state) =
            HadamardPIOP::prover_batch(trans, &hadamard_instance);
        let time = std::time::Instant::now();
        let mut hadamard_evals: SumHadamardTraceEval<EF> = SumHadamardTraceEval::default();
        hadamard_piop_proof.export_eval(0, 2, &mut hadamard_evals);
        // let hadamard_evals = trace_mle.evaluate_ef(&hadamard_piop_state.point_r);
        // hadamard_piop_proof.append_eval(&hadamard_evals);
        println!("Hadamard evals computation time: {:?}\n", time.elapsed());
        trans.append_message(b"[PIOP Phase]", &hadamard_piop_proof);
        //[Hadamard PIOP] reduce to queries on NTT evaluation matrix

        // [NTT PIOP] compute the common query point on each NTT evaluation matrix
        let point_u = hadamard_piop_state.randomness[..trace_mle.log_coeff_count].to_vec();
        let mut point_v = hadamard_piop_state.randomness[trace_mle.log_coeff_count..].to_vec();

        // [NTT PIOP] compute the combined query point on the (virtual) large NTT evaluation matrix
        // This virtual NTT evaluation matrix normally corresponds to the coefficient matrix that is committed at the beginning.
        let point_bit_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            trace_mle.log_num_oracles(),
        );
        // This is the coefficient matrix of the virtual NTT evaluation matrix
        let bit_poly = Rc::new(bit_poly.to_ef());
        let bit_ntt_evals = hadamard_evals.pack_ntt_to_vec();
        // The evaluation can be computed via the evaluations of component NTT evaluation matrices
        let eval = compute_oracle_evals(&bit_ntt_evals, &point_bit_oracle);

        // [NTT PIOP] obtain the NTT Matrix Evaluation Instance
        point_v.extend_from_slice(&point_bit_oracle);
        let ntt_instance = NTTMatrixEvalInstance::from_subclaim(
            &bit_poly,
            &params.ntt_table,
            &point_u,
            &point_v,
            eval,
        );
        // [NTT PIOP] prove it using sumcheck-based protocol
        let (ntt_piop_proof, ntt_piop_state) = NTTMatrixEvalIOP::prover(trans, &ntt_instance);
        trans.append_message(b"[PIOP Phase]", &ntt_piop_proof);
        // [NTT PIOP] reduce to the query on the coefficient matrix that is committed at the beginning

        // [PCS] generate the evaluation proof for the query on the committed coefficient matrix
        let mut point_r_v = Vec::with_capacity(ntt_piop_state.randomness.len() + point_v.len());
        point_r_v.extend_from_slice(&ntt_piop_state.randomness);
        point_r_v.extend_from_slice(&point_v);
        let eval_proof = PCS::open(
            &params.pcs_params,
            &commitment,
            &commitment_state,
            &point_r_v,
            trans,
        );

        HadamardProof {
            log_num_oracles: trace_mle.log_num_oracles(),
            pcs_params: params.pcs_params.clone(),
            commitment,
            hadamard_info,
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
            HadamardPIOP::verifier_batch(trans, &proof.hadamard_info, &proof.hadamard_proof);
        res &= hadamard_res;

        trans.append_message(b"[PIOP Phase]", &proof.hadamard_proof);

        let _point_u = hadamard_subclaim.randomness[..proof.ntt_info.log_coeff_count].to_vec();
        let mut point_v = hadamard_subclaim.randomness[proof.ntt_info.log_coeff_count..].to_vec();
        let point_bit_oracle = trans.get_vec_challenge(
            b"[Challenge] random point used to verify evaluations",
            proof.log_num_oracles,
        );
        point_v.extend_from_slice(&point_bit_oracle);

        let (ntt_res, ntt_subclaim) =
            NTTMatrixEvalIOP::verifier(trans, &proof.ntt_info, &proof.ntt_proof);
        trans.append_message(b"[PIOP Phase]", &proof.ntt_proof);
        res &= ntt_res;

        let mut point_r_v = Vec::with_capacity(ntt_subclaim.randomness.len() + point_v.len());
        point_r_v.extend_from_slice(&ntt_subclaim.randomness);
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
