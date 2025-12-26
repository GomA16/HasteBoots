use core::time;
use std::rc::Rc;

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
use snarks::fhe_op::hadamard::{HadamardParams, HadamardSnarks};
use snarks::fhe_op::monomial_hadamard::{MonomialHadamardParams, MonomialHadamardSnarks};
use trace::{AccTraceMLE, ConvertToEF, SumHadamardTraceMLE};
// use trace::HadamardProdTraceMLE;
use zkfhe::bfhe::{
    BABYBEAR_BINARY_128_BITS_PARAMETERS, CUSTOM_TERNARY_128_BITS_PARAMETERS, Evaluator,
};
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
    trace.finalize(params.lwe_dimension() as usize);

    // Two hadamard trace
    let hadamard_trace = trace.hadamard_trace;
    let hadamard_trace_mle: SumHadamardTraceMLE<_> = hadamard_trace.into();
    let acc_trace = trace.acc_trace;
    let acc_mle: AccTraceMLE<FF> = acc_trace.into();

    let ntt_table = FF::get_ntt_table(hadamard_trace_mle.log_coeff_count as u32)
        .unwrap()
        .root_powers();
    let ntt_table = Rc::new(ntt_table.to_ef());

    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
    let params = HadamardParams::new(code_spec.clone(), &ntt_table, &hadamard_trace_mle);
    let snarks = HadamardSnarks::<
        FF,
        EF,
        ExpanderCodeSpec,
        BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
    >::default();
    let acc_params = MonomialHadamardParams::new(code_spec, &ntt_table, &acc_mle);
    let acc_snarks = MonomialHadamardSnarks::<
        FF,
        EF,
        ExpanderCodeSpec,
        BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
    >::default();

    let mut prover_trans = Transcript::default();
    let time = std::time::Instant::now();
    let proof = snarks.prove(&mut prover_trans, &hadamard_trace_mle, &params);
    let acc_proof = acc_snarks.prove(&mut prover_trans, &acc_mle, &acc_params);
    println!("Proofs generation done!\n");
    println!("Proof generation time: {:?}\n", time.elapsed());

    let mut verifier_trans = Transcript::default();
    let time = std::time::Instant::now();
    let res = snarks.verify(&mut verifier_trans, &proof);
    let acc_res = acc_snarks.verify(&mut verifier_trans, &acc_proof);

    println!("Proofs verification done!\n");
    println!("Proof verification time: {:?}\n", time.elapsed());
    assert!(res && acc_res);
}

// fn main() {}
