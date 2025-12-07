use std::time::Instant;

use algebra::{BabyBear, BabyBearExetension};
use bincode::config::standard;
use helper::Transcript;
use pcs::{
    multilinear::BrakedownPCS,
    utils::code::{ExpanderCode, ExpanderCodeSpec},
};
use piop::SumcheckPIOP;
use snarks::ntt::NTTMatrixEvalSnarks;
use trace::NTTTrace;

type FF = BabyBear;
type EF = BabyBearExetension;
type Hash = sha2::Sha256;
const BASE_FIELD_BITS: usize = 31;

fn main() {
    let mut rng = rand::rng();
    let log_coeff_count = 10;
    let log_num_ntt = 14;

    let ntt_trace = NTTTrace::<FF>::random(log_coeff_count, log_num_ntt, &mut rng);
    let ntt_trace_info = ntt_trace.info_ef::<EF>();

    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
    let mut snarks = NTTMatrixEvalSnarks::<
        FF,
        EF,
        ExpanderCodeSpec,
        BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
    >::default();

    let oracle = snarks.setup(&ntt_trace, code_spec.clone());

    let prover_trans = &mut Transcript::<EF>::default();
    let start = Instant::now();
    let proof = snarks.prover(prover_trans, ntt_trace, &ntt_trace_info, &oracle);
    println!("Proving time: {:?}", start.elapsed());

    let proof_length = bincode::serde::encode_to_vec(&proof, standard())
        .unwrap()
        .len();

    let verifier_trans = &mut Transcript::<EF>::default();

    let start = Instant::now();
    let res = snarks.verifier(verifier_trans, &ntt_trace_info, &proof);
    println!("Verification time: {:?}", start.elapsed());

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
