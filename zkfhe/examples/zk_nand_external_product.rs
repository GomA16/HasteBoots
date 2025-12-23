use core::time;

use algebra::transformation::AbstractNTT;
use algebra::{AsInto, BabyBear, BabyBearExetension, FieldUniformSampler, NTTField};
use fhe_core::utils::*;
use helper::Transcript;
use pcs::multilinear::BrakedownPCS;
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use piop::ntt::{NTTMatrixEvalIOP, NTTMatrixEvalInstance};
use piop::{SumcheckInstance, SumcheckPIOP};
use rand::Rng;
use rand_distr::Distribution;
use snarks::external_product::{ExternalProductParams, ExternalProductSnarks};
use snarks::hadamard::{HadamardParams, HadamardSnarks};
use trace::SumHadamardTraceMLE;
// use trace::HadamardProdTraceMLE;
use zkfhe::bfhe::{CUSTOM_TERNARY_128_BITS_PARAMETERS, Evaluator, BABYBEAR_BINARY_128_BITS_PARAMETERS};
use zkfhe::{Decryptor, Encryptor, KeyGen};

type FF = BabyBear;
type EF = BabyBearExetension;
type Hash = sha2::Sha256;
const BASE_FIELD_BITS: usize = 31;
fn main() {
    env_logger::init();
    // set random generator
    let mut rng = rand::rng();

    // set parameter
    let params = *BABYBEAR_BINARY_128_BITS_PARAMETERS;
    println!("Parameters: {params:?}\n");

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

    let mut a = a.as_into();
    let mut b = b.as_into();

    let x = enc.encrypt(a);
    let y = enc.encrypt(b);
    // let mut z = enc.encrypt(c);

    let _start = std::time::Instant::now();
    // let (ct_nand, trace) = eval.nand(&x, &y);
    let (ct_nand, trace) = eval.nand(&x, &y);

    // nand
    let (m, noise) = dec.decrypt_with_noise(&ct_nand);
    assert_eq!(m, nand(a, b), "Noise: {noise}");
    check_noise(noise, "nand");

    // Generate SNARKs for nand
    println!("");
    println!("Starting verification of nand.\n");
    let trace_mle: SumHadamardTraceMLE<_> = trace.into();
    let ntt_table = FF::get_ntt_table(trace_mle.log_coeff_count as u32).unwrap().root_powers();
    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
    let blk_size = 3;
    let basis = BABYBEAR_BINARY_128_BITS_PARAMETERS.blind_rotation_basis().basis() as usize;
    let params = ExternalProductParams::new(code_spec, ntt_table, blk_size, basis, &trace_mle);
    let snarks = ExternalProductSnarks::<
        FF,
        EF,
        ExpanderCodeSpec,
        BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
    >::default();

    let mut prover_trans = Transcript::default();
    let time = std::time::Instant::now();
    let proof = snarks.prove(&mut prover_trans, &trace_mle, &params);
    println!("Proofs generation done!\n");
    println!("Proof generation time: {:?}\n", time.elapsed());

    let mut verifier_trans = Transcript::default();
    let time = std::time::Instant::now();
    let res = snarks.verify(&mut verifier_trans, &proof);
    println!("Proofs verification done!\n");
    println!("Proof verification time: {:?}\n", time.elapsed());
    assert!(res);
}

// fn main() {}
