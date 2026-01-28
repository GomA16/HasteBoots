use algebra::transformation::AbstractNTT;
use algebra::{AsInto, BabyBear, BabyBearExetension, NTTField};
use fhe_core::utils::*;
use helper::Transcript;
use pcs::multilinear::BrakedownPCS;
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use rand::Rng;
use snarks::SnarkStatistics;
use snarks::fhe_op::key_switching::{KeySwitchingParams, KeySwitchingSnarks};
use snarks::fhe_op::modulus_switch::{ModulusSwitchingParams, ModulusSwitchingSnarks};
use trace::basic_ops::SumHadamardTraceMLE;
use trace::key_switching_trace::KeySwitchingTraceMLE;
use trace::modulus_switching_trace::ModulusSwitchingTrace;
use zkfhe::bfhe::{BABYBEAR_BINARY_128_BITS_PARAMETERS, Evaluator};
use zkfhe::{Decryptor, Encryptor, KeyGen};

type FF = BabyBear;
type EF = BabyBearExetension;
type Hash = sha2::Sha256;
const BASE_FIELD_BITS: usize = 31;
const LOG_BATCH_SIZE: usize = 0; // batch size = 2^LOG_BATCH_SIZE
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

    let modulus_switching_trace = trace.modulus_switching_trace;
    let traces = vec![modulus_switching_trace; 1 << LOG_BATCH_SIZE];
    let batched_trace =
        ModulusSwitchingTrace::from_batch_trace(traces, params.lwe_dimension() + 1);
    let trace = batched_trace.into();


    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
    let params = ModulusSwitchingParams::new(&code_spec);
    let snarks = ModulusSwitchingSnarks::<
        FF,
        EF,
        ExpanderCodeSpec,
        BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
    >::default();

    let pcs_statistics = &mut Some(&mut SnarkStatistics::default());
    let mut prover_trans = Transcript::default();
    let time = std::time::Instant::now();
    let proof = snarks.prove(&mut prover_trans, &trace, &params, pcs_statistics);
    println!("Proofs generation done!\n");
    println!("Proof generation time: {:?}\n", time.elapsed());

    let mut verifier_trans = Transcript::default();
    let time = std::time::Instant::now();
    let res = snarks.verify(&mut verifier_trans, &proof, pcs_statistics);
    println!("Proofs verification done!\n");
    println!("Proof verification time: {:?}\n", time.elapsed());

    assert!(res);
}

// fn main() {}
