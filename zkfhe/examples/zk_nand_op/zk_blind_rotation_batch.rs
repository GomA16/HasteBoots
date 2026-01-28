use algebra::transformation::AbstractNTT;
use algebra::{AsInto, BabyBear, BabyBearExetension, NTTField};
use fhe_core::utils::*;
use helper::Transcript;
use pcs::multilinear::BrakedownPCS;
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use rand::Rng;
use snarks::fhe_batch_op::batch_blind_rotation::{BatchBlindRotationParams, BatchBlindRotationSnarks};
use snarks::fhe_op::blind_rotation::{BlindRotationParams, BlindRotationSnarks, KeyCommitment};
use trace::BlindRotationTraceMLE;
// use trace::HadamardProdTraceMLE;
use zkfhe::bfhe::{
    BABYBEAR_BINARY_128_BITS_PARAMETERS, Evaluator,
};
use zkfhe::{Decryptor, Encryptor, KeyGen};

type FF = BabyBear;
type EF = BabyBearExetension;
type Hash = sha2::Sha256;
const BASE_FIELD_BITS: usize = 31;
const LOG_BATCH_SIZE: usize = 1; // batch size = 2^LOG_BATCH_SIZE
fn main() {
    env_logger::init();
    // set random generator
    let mut rng = rand::rng();

    // set parameter
    let params = *BABYBEAR_BINARY_128_BITS_PARAMETERS;
    println!("Parameters: {params:#?}\n");

    let noise_max = (params.lwe_cipher_modulus_value() as f64 / 16.0).as_into();

    let check_noise = |noise, op: &str| {
        assert!(
            noise < noise_max,
            "Type: {op}\nNoise: {noise} >= {noise_max}"
        );
        println!("{op:4.4} Noise: {noise:3} < {noise_max:3}");
    };

    // generate keys
    let sk = KeyGen::generate_secret_key(params);
    println!("Secret Key Generation done!\n");

    let enc = Encryptor::new(sk.clone());
    let eval = Evaluator::new(&sk);
    let dec = Decryptor::new(sk);
    println!("Evaluation Key Generation done!\n");

    let a: bool = rng.random();
    let b: bool = rng.random();
    // let mut c = rng.random();

    let a = a.as_into();
    let b = b.as_into();

    let x = enc.encrypt(a);
    let y = enc.encrypt(b);
    // let mut z = enc.encrypt(c);

    let start = std::time::Instant::now();
    // let (ct_nand, trace) = eval.nand(&x, &y);
    let (ct_nand, mut trace) = eval.nand(&x, &y);
    println!("NAND Evaluation Time is : {:?}\n", start.elapsed());

    // nand
    let (m, noise) = dec.decrypt_with_noise(&ct_nand);
    assert_eq!(m, nand(a, b), "Noise: {noise}");
    check_noise(noise, "nand");


    // Generate SNARKs for nand
    println!("");
    println!("Starting verification of {} instances of blind rotation.\n", 1 << LOG_BATCH_SIZE);

    let mut trace = trace.blind_rotation_trace;
    trace.finalize(params.lwe_dimension());
    let traces = vec![trace; 1 << LOG_BATCH_SIZE];

    let ntt_table = FF::get_ntt_table(traces[0].log_coeff_count as u32)
        .unwrap()
        .root_powers();
    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);

    let key_commitment = KeyCommitment::new(&code_spec, &traces[0]);
    let params = BatchBlindRotationParams::new(code_spec, ntt_table, &traces, &key_commitment);
    let snarks = BatchBlindRotationSnarks::<
        FF,
        EF,
        ExpanderCodeSpec,
        BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
    >::default();

    let mut prover_trans = Transcript::default();
    let time = std::time::Instant::now();
    let proof = snarks.prove(&mut prover_trans, traces, &params, &mut None);
    println!("Proofs generation done!\n");
    println!("Proof generation time: {:?}\n", time.elapsed());

    let mut verifier_trans = Transcript::default();
    let time = std::time::Instant::now();
    let res = snarks.verify(&mut verifier_trans, &proof, &mut None);
    println!("Proofs verification done!\n");
    println!("Proof verification time: {:?}\n", time.elapsed());
    println!(
        "PIOP Proof Size: {} MB",
        proof.piop_proof_len() as f64 / (1000 * 1000) as f64
    );
    println!(
        "PCS Proof Size: {} MB",
        proof.pcs_proof_len() as f64 / (1000 * 1000) as f64
    );
    assert!(res);
}

// fn main() {}
