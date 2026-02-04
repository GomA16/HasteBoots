use algebra::{AsInto, BabyBear, Field};
use fhe_core::utils::*;
use helper::Transcript;
use piop::lookup::normal_table::{LogUpIOP, LogUpInstance};
use piop::{SumcheckInstance, SumcheckPIOP};
use rand::Rng;
use trace::basic_ops::SumHadamardTraceMLE;
use trace::lookup_trace::normal_table::LookupWitness;
use vfhe::bfhe::{BABYBEAR_BINARY_128_BITS_PARAMETERS, Evaluator};
use vfhe::{Decryptor, Encryptor, KeyGen};

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

    let a = a.as_into();
    let b = b.as_into();

    let x = enc.encrypt(a);
    let y = enc.encrypt(b);

    let _start = std::time::Instant::now();
    // let (ct_nand, trace) = eval.nand(&x, &y);
    let (ct_nand, trace) = eval.nand(&x, &y);

    // nand
    let (m, noise) = dec.decrypt_with_noise(&ct_nand);
    assert_eq!(m, nand(a, b), "Noise: {noise}");
    check_noise(noise, "nand");

    // Generate SNARKs for nand
    println!("Starting verification of nand.\n");
    let blk_size = 1;
    let randomness = BabyBear::random(&mut rng);

    let trace = trace.blind_rotation_trace;
    let trace = trace.hadamard_trace;
    let trace_mle: SumHadamardTraceMLE<_> = trace.into();
    let range = 1 << params.blind_rotation_basis().bits() as usize;
    let lookup_trace_mle = trace_mle.extract_lookup_trace_mle_normal_table(range);
    let lookup_witness: LookupWitness<_> = lookup_trace_mle.into();
    let lookup_helper = lookup_witness.compute_helper_functions(blk_size, randomness);

    let instance = LogUpInstance::from(&lookup_witness, &lookup_helper);
    let info = instance.info();

    let mut prover_trans = Transcript::new();
    let time = std::time::Instant::now();
    let (proof, _) = LogUpIOP::prover(&mut prover_trans, &instance);
    println!("Prover time: {:?}", time.elapsed());
    let mut verifier_trans = Transcript::new();
    let time = std::time::Instant::now();
    let (res, _) = LogUpIOP::verifier(&mut verifier_trans, &info, &proof);
    println!("Verifier time: {:?}", time.elapsed());
    assert!(res);
    println!("Verification of nand done!\n");
    println!(
        "Lookup Info: num_vars = {}, block_size = {}, num_blks = {}\n",
        info.num_vars, info.block_size, info.num_blocks
    );
    println!("Lookup num columns: {}\n", info.num_columns);
    println!("range is {}\n", range);
    println!(
        "num_vars is {} and num_round is {}",
        trace_mle.log_coeff_count, trace_mle.log_num_poly
    );
}

// fn main() {}
