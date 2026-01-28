use algebra::transformation::AbstractNTT;
use algebra::{AsInto, BabyBear, BabyBearExetension, NTTField};
use fhe_core::utils::*;
use helper::Transcript;
use pcs::multilinear::BrakedownPCS;
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use rand::Rng;
use snarks::fhe_op::key_switching::{KeySwitchingParams, KeySwitchingSnarks};
use snarks::fhe_op::row_permutation::RowPermutationSignedSnarks;
use trace::basic_ops::{RowPermTrace, RowPermTraceMLE, SumHadamardTraceMLE};
use trace::key_switching_trace::KeySwitchingTraceMLE;
use zkfhe::bfhe::{
    BABYBEAR_BINARY_128_BITS_PARAMETERS, Evaluator,
};
use zkfhe::{Decryptor, Encryptor, KeyGen};

type FF = BabyBear;
type EF = BabyBearExetension;
type Hash = sha2::Sha256;
const BASE_FIELD_BITS: usize = 31;
const LOG_BATCH_SIZE: usize = 2; // batch size = 2^LOG_BATCH_SIZE
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
    println!("Starting verification of {} instances of sample extraction.\n", 1 << LOG_BATCH_SIZE);

    let trace = trace.sample_extraction_trace;
    let traces = vec![trace; 1 << LOG_BATCH_SIZE]; // batch size 2^LOG_BATCH_SIZE
    let trace = RowPermTrace::from_batch_trace(traces);
    let trace_mle: RowPermTraceMLE<_> = trace.into();

    let snarks = RowPermutationSignedSnarks::<
        FF,
        EF,
        ExpanderCodeSpec,
        BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
    >::default();

    let mut prover_trans = Transcript::default();
    let time = std::time::Instant::now();
    let proof = snarks.prove(&mut prover_trans, &trace_mle);
    println!("Proofs generation done!\n");
    println!("Proof generation time: {:?}\n", time.elapsed());

    let mut verifier_trans = Transcript::default();
    let time = std::time::Instant::now();
    let res = snarks.verify(&mut verifier_trans, &proof, &mut None);
    println!("Proofs verification done!\n");
    println!("Proof verification time: {:?}\n", time.elapsed());

    assert!(res);
}

// fn main() {}
