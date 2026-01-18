use algebra::transformation::AbstractNTT;
use algebra::{AsInto, BabyBear, BabyBearExetension, NTTField};
use fhe_core::utils::*;
use helper::Transcript;
use pcs::multilinear::BrakedownPCS;
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use rand::Rng;
use snarks::SnarkStatistics;
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
const LOG_BATCH_SIZE: usize = 5; // batch size = 2^LOG_BATCH_SIZE
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
    println!("Starting verification of nand.\n");

    let mut trace = trace.blind_rotation_trace;
    trace.finalize(params.lwe_dimension());

    let traces = vec![trace; 1 << LOG_BATCH_SIZE]; // batch size 2


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
    let pcs_statistics = &mut Some(&mut SnarkStatistics::default());

    let mut prover_trans = Transcript::default();
    let time = std::time::Instant::now();
    let proof = snarks.prove(&mut prover_trans, traces, &params, pcs_statistics);
    let prover_total_time = time.elapsed();
    println!("Proofs generation done!\n");
    println!("Proof generation time: {:?}\n", prover_total_time);

    let mut verifier_trans = Transcript::default();
    let time = std::time::Instant::now();
    let res = snarks.verify(&mut verifier_trans, &proof, pcs_statistics);
    let verifier_total_time = time.elapsed();
    assert!(res);
    println!("Proofs verification done!\n");
    println!("Proof verification time: {:?}\n", verifier_total_time);
    
    println!("--- SNARK Statistics Summary ---\n");
    println!("Prover Total Time: {:.2?} s", prover_total_time.as_secs_f64());
    if let Some(stats) = pcs_statistics {
        let pcs_ratio =
            stats.prover_pcs_time.as_secs_f64() / prover_total_time.as_secs_f64() * 100.0;
        println!(
            "Prover PCS Time (including commit and open): {:.2?} s, accounts for {:.2} %",
            stats.prover_pcs_time.as_secs_f64(), pcs_ratio
        );
        println!(
            "Prover PIOP Time: {:.2?}\n",
            (prover_total_time - stats.prover_pcs_time).as_secs_f64()
        );
    }
    println!("Verifier Total Time: {:.2?} ms", verifier_total_time.as_secs_f64() * 1000.0);
    if let Some(stats) = pcs_statistics {
        let pcs_ratio =
            stats.verifier_pcs_time.as_secs_f64() / verifier_total_time.as_secs_f64() * 100.0;
        println!(
            "Verifier PCS Time (including commit and open): {:.2?} ms, accounts for {:.2} %",
            stats.verifier_pcs_time.as_secs_f64() * 1000.0, pcs_ratio
        );
        println!(
            "Verifier PIOP Time: {:.2?} ms\n",
            (verifier_total_time - stats.verifier_pcs_time).as_secs_f64() * 1000.0
        );
    }

    let piop_size = proof.piop_proof_len();
    let pcs_size = proof.pcs_proof_len();
    println!(
        "Proof Sizes: {} MB total",
        (piop_size + pcs_size) as f64 / (1024 * 1024) as f64
    );
    println!(
        "PCS Proof Sizes: {:.2} MB, accounts for {:.2} %",
        (pcs_size) as f64 / (1024 * 1024) as f64,
        pcs_size as f64 / (piop_size + pcs_size) as f64 * 100.0
    );
    println!(
        "PIOP Proof Sizes: {:.2} MB",
        piop_size as f64 / (1024 * 1024) as f64,
    );
}

// fn main() {}
