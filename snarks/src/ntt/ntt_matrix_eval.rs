use crate::EvalOracle;
use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};
use helper::Transcript;
use pcs::PolynomialCommitmentScheme;
use piop::SumcheckPIOP;
use piop::ntt::{NTTMatrixEvalIOP, NTTMatrixEvalInfo, NTTMatrixEvalInstance, NTTMatrixEvalProof};
use serde::Serialize;
use std::rc::Rc;
use trace::{ConvertToEF, NTTTrace, NTTTraceInfo};

#[derive(Default)]
pub struct NTTMatrixEvalSnarks<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    code_spec: S,
    pcs_params: PCS::Parameters,
}

#[derive(Serialize)]
pub struct NTTMatrixEvalSnarksProof<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub comm_coefficients: PCS::Commitment,
    pub piop_proof: NTTMatrixEvalProof<EF>,
    pub evaluations_at_u_v: EF,
    pub coefficients_at_r_v: EF,
    pub eval_proof: PCS::Proof,
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
    pub fn setup(&mut self, trace: &NTTTrace<F>, code_spec: S) -> EvalOracle<F, EF, S, PCS> {
        self.code_spec = code_spec.clone();
        let trace_poly = trace.get_commit_poly();
        let pcs_params = PCS::setup(trace_poly.num_vars(), Some(code_spec.clone()));
        self.pcs_params = pcs_params.clone();
        EvalOracle {
            poly: trace_poly,
            params: pcs_params,
        }
    }

    pub fn prover(
        &self,
        trans: &mut Transcript<EF>,
        trace: NTTTrace<F>,
        statement: &NTTTraceInfo<EF>,
        oracle: &EvalOracle<F, EF, S, PCS>,
    ) -> NTTMatrixEvalSnarksProof<F, EF, S, PCS> {
        trans.append_message(b"[NTT Relation Snarks]", b"Init");
        trans.append_message(b"[NTT Statement]", &statement);

        let (commitment, comm_state) = oracle.commit();
        trans.append_message(b"[Commit Phase]", &commitment);

        let point_u = trans.get_vec_challenge(b"random point", statement.log_coeff_count);
        let point_v = trans.get_vec_challenge(b"random point", statement.log_num_ntt);
        let trace_ef: NTTTrace<EF> = trace.into_ef();
        let ntt_eval_instance = &NTTMatrixEvalInstance::from(&trace_ef.into(), &point_u, &point_v);

        let (piop_proof, piop_state) = NTTMatrixEvalIOP::prover(trans, ntt_eval_instance);

        let mut point_r_v = Vec::with_capacity(piop_state.point_r.len() + point_v.len());
        point_r_v.extend_from_slice(&piop_state.point_r);
        point_r_v.extend_from_slice(&point_v);
        let eval_proof = oracle.open(trans, &commitment, &comm_state, &point_r_v);

        NTTMatrixEvalSnarksProof {
            comm_coefficients: commitment,
            piop_proof,
            evaluations_at_u_v: ntt_eval_instance.evaluations_at_u_v,
            coefficients_at_r_v: piop_state.coeffs_at_v_back.evaluate(&piop_state.point_r),
            eval_proof,
        }
    }

    pub fn verifier(
        &self,
        trans: &mut Transcript<EF>,
        statement: &NTTTraceInfo<EF>,
        proof: &NTTMatrixEvalSnarksProof<F, EF, S, PCS>,
    ) -> bool {
        trans.append_message(b"[NTT Relation Snarks]", b"Init");
        trans.append_message(b"[NTT Statement]", &statement);

        let mut res = true;

        trans.append_message(b"[Commit Phase]", &proof.comm_coefficients);
        let point_u = trans.get_vec_challenge(b"random point", statement.log_coeff_count);
        let point_v = trans.get_vec_challenge(b"random point", statement.log_num_ntt);

        let ntt_eval_statement = NTTMatrixEvalInfo {
            ntt_table: Rc::clone(&statement.ntt_table),
            point_u,
            point_v: point_v.clone(),
            evaluations_at_u_v: proof.evaluations_at_u_v,
        };

        let (piop_res, piop_subclaim) =
            NTTMatrixEvalIOP::verifier(trans, &ntt_eval_statement, &proof.piop_proof);
        res &= piop_res;

        let mut point_r_v = Vec::with_capacity(piop_subclaim.point_r.len() + point_v.len());
        point_r_v.extend_from_slice(&piop_subclaim.point_r);
        point_r_v.extend_from_slice(&point_v);

        let pcs_res = PCS::verify(
            &self.pcs_params,
            &proof.comm_coefficients,
            &point_r_v,
            proof.coefficients_at_r_v,
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
    use trace::NTTTrace;

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
        let ntt_trace_info = ntt_trace.info_ef();

        let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
        let mut snarks = NTTMatrixEvalSnarks::<
            FF,
            EF,
            ExpanderCodeSpec,
            BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
        >::default();

        let oracle = snarks.setup(&ntt_trace, code_spec.clone());

        let prover_trans = &mut Transcript::<EF>::default();
        let proof = snarks.prover(prover_trans, ntt_trace, &ntt_trace_info, &oracle);

        let proof_length = bincode::serde::encode_to_vec(&proof, standard())
            .unwrap()
            .len();

        let verifier_trans = &mut Transcript::<EF>::default();
        let res = snarks.verifier(verifier_trans, &ntt_trace_info, &proof);
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
