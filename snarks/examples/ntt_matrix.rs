use std::time::Instant;

use algebra::{BabyBear, BabyBearExetension};
use bincode::config::standard;
use helper::Transcript;
use pcs::{
    multilinear::BrakedownPCS,
    utils::code::{ExpanderCode, ExpanderCodeSpec},
};
use snarks::ntt::{NTTMatrixEvalSnarks, ntt_matrix_eval::NTTMatrixEvalSnarksParams};
use trace::{NTTTrace, NTTTraceMLE};

type FF = BabyBear;
type EF = BabyBearExetension;
type Hash = sha2::Sha256;
const BASE_FIELD_BITS: usize = 31;

fn main() {
    let mut rng = rand::rng();
    let log_coeff_count = 10;
    let log_num_ntt = 14;

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
