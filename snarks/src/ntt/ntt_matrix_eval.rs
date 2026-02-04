use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};
use helper::{FiatShamirTranscript, Transcript};
use pcs::PolynomialCommitmentScheme;
use piop::ntt::{NTTMatrixEvalIOP, NTTMatrixEvalInfo, NTTMatrixEvalInstance, NTTMatrixEvalProof};
use piop::{SumcheckInstance, SumcheckPIOP};
use serde::Serialize;
use trace::basic_ops::NTTTraceMLE;
use trace::{ConvertToEF, PackableTrace};

pub struct NTTMatrixEvalSnarksParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pcs_params: PCS::Parameters,
}

#[derive(Default)]
pub struct NTTMatrixEvalSnarks<F, EF, S, PCS>
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
pub struct NTTMatrixEvalSnarksProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub params: PCS::Parameters,
    pub commitment: PCS::Commitment,
    pub instance_info: NTTMatrixEvalInfo<EF>,
    pub piop_proof: NTTMatrixEvalProof<EF>,
    pub eval_proof: PCS::Proof,
}

impl<F, EF, S, PCS> NTTMatrixEvalSnarksParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub fn new(code_spec: S, trace: &NTTTraceMLE<F>) -> Self {
        let pcs_params = PCS::setup(trace.num_vars(), &code_spec);
        Self { pcs_params }
    }
}

impl<F, EF, S, PCS> NTTMatrixEvalSnarks<F, EF, S, PCS>
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
    pub fn prover(
        &self,
        trans: &mut Transcript<EF>,
        trace: &NTTTraceMLE<F>,
        params: &NTTMatrixEvalSnarksParams<F, EF, S, PCS>,
    ) -> NTTMatrixEvalSnarksProof<F, EF, S, PCS> {
        // Commit Phase
        let poly = trace.generate_oracle();
        let (commitment, comm_state) = PCS::commit(&params.pcs_params, &poly);
        trans.append_message(b"[Commit Phase]", &commitment);

        let point_u = trans.get_vec_challenge(b"random point", trace.log_coeff_count);
        let point_v = trans.get_vec_challenge(b"random point", trace.log_num_ntt);
        let trace_ef: NTTTraceMLE<EF> = trace.to_ef();
        let ntt_eval_instance = &NTTMatrixEvalInstance::from(&trace_ef, &point_u, &point_v);

        let (piop_proof, piop_state) = NTTMatrixEvalIOP::prover(trans, ntt_eval_instance);

        let mut point_r_v = Vec::with_capacity(piop_state.randomness.len() + point_v.len());
        point_r_v.extend_from_slice(&piop_state.randomness);
        point_r_v.extend_from_slice(&point_v);

        let eval_proof = PCS::open(
            &params.pcs_params,
            &commitment,
            &comm_state,
            &point_r_v,
            trans,
        );

        NTTMatrixEvalSnarksProof {
            params: params.pcs_params.clone(),
            commitment,
            instance_info: ntt_eval_instance.info(),
            piop_proof,
            eval_proof,
        }
    }

    pub fn verifier(
        &self,
        trans: &mut Transcript<EF>,
        proof: &NTTMatrixEvalSnarksProof<F, EF, S, PCS>,
    ) -> bool {
        trans.append_message(b"[Commit Phase]", &proof.commitment);

        let mut res = true;

        let _point_u =
            trans.get_vec_challenge(b"random point", proof.instance_info.log_coeff_count);
        let point_v = trans.get_vec_challenge(b"random point", proof.instance_info.log_num_ntt);

        let (piop_res, piop_subclaim) =
            NTTMatrixEvalIOP::verifier(trans, &proof.instance_info, &proof.piop_proof);
        res &= piop_res;

        let mut point_r_v = Vec::with_capacity(piop_subclaim.randomness.len() + point_v.len());
        point_r_v.extend_from_slice(&piop_subclaim.randomness);
        point_r_v.extend_from_slice(&point_v);

        let pcs_res = PCS::verify(
            &proof.params,
            &proof.commitment,
            &point_r_v,
            proof.piop_proof.coeff_eval_at_r_v,
            &proof.eval_proof,
            trans,
        );

        res & pcs_res
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use algebra::{BabyBear, BabyBearExetension};
    use bincode::config::standard;
    use helper::Transcript;
    use pcs::{
        multilinear::BrakedownPCS,
        utils::code::{ExpanderCode, ExpanderCodeSpec},
    };
    use trace::basic_ops::NTTTrace;

    type FF = BabyBear;
    type EF = BabyBearExetension;
    type Hash = sha2::Sha256;
    const BASE_FIELD_BITS: usize = 31;

    #[test]
    fn test_ntt_matrix_eval_snarks() {
        let mut rng = rand::rng();
        let log_coeff_count = 10;
        let log_num_ntt = 10;

        let ntt_trace = NTTTrace::<FF>::random(log_coeff_count, log_num_ntt, &mut rng);
        let trace_mle: NTTTraceMLE<FF> = ntt_trace.into();

        let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
        let snarks = NTTMatrixEvalSnarks::<
            FF,
            EF,
            ExpanderCodeSpec,
            BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
        >::default();
        let params = NTTMatrixEvalSnarksParams::new(code_spec.clone(), &trace_mle);

        let prover_trans = &mut Transcript::<EF>::default();
        let proof = snarks.prover(prover_trans, &trace_mle, &params);

        let proof_length = bincode::serde::encode_to_vec(&proof, standard())
            .unwrap()
            .len();

        let verifier_trans = &mut Transcript::<EF>::default();
        let res = snarks.verifier(verifier_trans, &proof);
        assert!(res);

        println!("Proof size: {} bytes", proof_length);
        println!(
            "Proof size in piop: {} bytes",
            bincode::serde::encode_to_vec(&proof.piop_proof, standard())
                .unwrap()
                .len()
        );
        println!(
            "Proof size in pcs: {} bytes",
            bincode::serde::encode_to_vec(&proof.eval_proof, standard())
                .unwrap()
                .len()
        );
    }
}
