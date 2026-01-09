use core::time;

use algebra::{AsInto, BabyBear, BabyBearExetension, Field, FieldUniformSampler};
use fhe_core::{DefaultFieldU32, utils::*};
use helper::Transcript;
use pcs::multilinear::BrakedownPCS;
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use piop::lookup::normal_table::{LogUpIOP, LogUpInstance};
use piop::ntt::{NTTMatrixEvalIOP, NTTMatrixEvalInstance};
use piop::{SumcheckInstance, SumcheckPIOP};
use rand::Rng;
use snarks::fhe_op::decomposition::{self, DecompositionParams, DecompositionSnarks};
use snarks::lookup::indexed_table::indexed_batch::{
    BatchedIndexedLogUpParams, BatchedIndexedLogUpSnarks,
};
use trace::BlindRotationTraceMLE;
use trace::basic_ops::SumHadamardTraceMLE;
use trace::lookup_trace::normal_table::LookupWitness;
// use trace::HadamardProdTraceMLE;
use zkfhe::bfhe::{BABYBEAR_BINARY_128_BITS_PARAMETERS, BABYBEAR_CODE_SPEC, Evaluator};
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
    let (ct_nand, mut trace) = eval.nand(&x, &y);

    // nand
    let (m, noise) = dec.decrypt_with_noise(&ct_nand);
    assert_eq!(m, nand(a, b), "Noise: {noise}");
    check_noise(noise, "nand");

    // Generate SNARKs for nand
    println!("Starting verification of nand.\n");

    let mut blind_rotation_trace = trace.blind_rotation_trace;
    blind_rotation_trace.finalize(params.lwe_dimension());
    let trace_mle: BlindRotationTraceMLE<_> = blind_rotation_trace.into();
    let decomp_trace_mle = trace_mle.extract_decomposition_traces();

    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, 31, 10);
    let snarks = DecompositionSnarks::<
        FF,
        EF,
        ExpanderCodeSpec,
        BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
    >::default();

    let params = DecompositionParams::new(code_spec, &trace_mle.lt_tables);
    let mut prover_trans = Transcript::new();
    let time = std::time::Instant::now();
    let proof = snarks.prove(&mut prover_trans, &decomp_trace_mle, &params);
    println!("Prover time: {:?}", time.elapsed());
    let mut verifier_trans = Transcript::new();
    let time = std::time::Instant::now();
    let res = snarks.verify(&mut verifier_trans, &proof);
    println!("Verifier time: {:?}", time.elapsed());
    assert!(res);
    println!("Verification of nand done!\n");
}

// fn main() {}
