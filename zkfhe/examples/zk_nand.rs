use algebra::{AsInto, FieldUniformSampler};
use fhe_core::utils::*;
use helper::Transcript;
use piop::{SumcheckPIOP, SumcheckInstance};
use piop::ntt::{NTTMatrixEvalIOP, NTTMatrixEvalInstance};
use rand::Rng;
use rand_distr::Distribution;
// use trace::HadamardProdTraceMLE;
use zkfhe::bfhe::{CUSTOM_TERNARY_128_BITS_PARAMETERS, Evaluator};
use zkfhe::{Decryptor, Encryptor, KeyGen};

fn main() {
    env_logger::init();
    // set random generator
    let mut rng = rand::rng();

    // set parameter
    let params = *CUSTOM_TERNARY_128_BITS_PARAMETERS;
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

    let a:bool = rng.random();
    let b:bool = rng.random();
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
    println!("Starting verification of nand.\n");
    let uniform = FieldUniformSampler::new();
    let randomness = uniform
        .sample_iter(&mut rng)
        .take(trace.vec_trace.len())
        .collect::<Vec<_>>();

    let ntt_trace = trace.extract_random_ntt_trace_mle(&randomness);

    let log_coeff_count = trace.log_coeff_count;
    let log_num_ntt = trace.log_num_round;

    let point_u = uniform
        .sample_iter(&mut rng)
        .take(log_coeff_count)
        .collect::<Vec<_>>();
    let point_v = uniform
        .sample_iter(&mut rng)
        .take(log_num_ntt)
        .collect::<Vec<_>>();
    let ntt_matrix_eval_instance = NTTMatrixEvalInstance::from(&ntt_trace, &point_u, &point_v);

    let ntt_eval_info = ntt_matrix_eval_instance.info();

    let mut prover_trans = Transcript::default();
    let (proof, _) = NTTMatrixEvalIOP::prover(&mut prover_trans, &ntt_matrix_eval_instance);
    println!("Proofs generation done!\n");

    let mut verifier_trans = Transcript::default();
    let (res, _) = NTTMatrixEvalIOP::verifier(&mut verifier_trans, &ntt_eval_info, &proof);
    println!("Proofs verification done!\n");
    assert!(res);
}

// fn main() {}
