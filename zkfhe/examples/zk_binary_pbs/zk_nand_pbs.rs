use algebra::transformation::AbstractNTT;
use algebra::{
    AbstractExtensionField, AsInto, BabyBear, BabyBearExetension, Field, Goldilocks,
    GoldilocksExtension, NTTField,
};
use fhe_core::utils::*;
use helper::Transcript;
use pcs::PolynomialCommitmentScheme;
use pcs::multilinear::BrakedownPCS;
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use rand::Rng;
use sha2::digest::crypto_common::Key;
use snarks::SnarkStatistics;
use snarks::fhe_op::blind_rotation::{BlindRotationParams, BlindRotationSnarks, KeyCommitment};
use snarks::fhe_op::key_switching::{KeySwitchingParams, KeySwitchingSnarks};
use snarks::fhe_op::modulus_switch::{self, ModulusSwitchingSnarks};
use snarks::fhe_op::row_permutation::RowPermutationSignedSnarks;
use trace::pbs_trace::PBSTrace;
// use trace::HadamardProdTraceMLE;
use zkfhe::bfhe::{BABYBEAR_BINARY_128_BITS_PARAMETERS, Evaluator, ZAMA_GOLDILOCKS_PARAMETERS};
use zkfhe::{Decryptor, Encryptor, KeyGen};

type FF = BabyBear;
type EF = BabyBearExetension;
type Hash = sha2::Sha256;
const BASE_FIELD_BITS: usize = 64;

#[derive(Default)]
pub struct PBSSnarks<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub modulus_switching: ModulusSwitchingSnarks<F, EF, S, PCS>,
    pub blind_rotation: BlindRotationSnarks<F, EF, S, PCS>,
    pub key_switching: KeySwitchingSnarks<F, EF, S, PCS>,
    pub sample_extraction: RowPermutationSignedSnarks<F, EF, S, PCS>,
}

fn main() {
    env_logger::init();
    // ------------------ zkfhe nand pbs with storing trace ------------------
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

    // ------------------ generate snarks nand pbs ------------------

    // Generate SNARKs for nand
    println!("");
    println!("--- Starting verification of nand ---\n");

    // Perepare parameters and traces
    let time = std::time::Instant::now();
    let PBSTrace {
        modulus_switching_trace,
        mut blind_rotation_trace,
        key_switching_trace,
        sample_extraction_trace,
    } = trace;

    let blind_rotation_ntt_table = FF::get_ntt_table(blind_rotation_trace.log_coeff_count as u32)
        .unwrap()
        .root_powers();
    let key_switching_ntt_table = FF::get_ntt_table(key_switching_trace.log_coeff_count as u32)
        .unwrap()
        .root_powers();

    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);

    let bs_key_commitment = KeyCommitment::new(&code_spec, &blind_rotation_trace);
    let blind_rotation_params = BlindRotationParams::new(
        code_spec.clone(),
        blind_rotation_ntt_table,
        &blind_rotation_trace,
        &bs_key_commitment,
    );

    let key_switching_trace = key_switching_trace.into();
    let key_switching_params = KeySwitchingParams::new(
        code_spec.clone(),
        key_switching_ntt_table,
        &key_switching_trace,
    );

    let modulus_switching_trace = modulus_switching_trace.into();
    let modulus_switching_params = modulus_switch::ModulusSwitchingParams::new(&code_spec);

    let snarks = PBSSnarks::<
        FF,
        EF,
        ExpanderCodeSpec,
        BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
    >::default();
    println!("Preparing parameters time: {:?}\n", time.elapsed());

    let pcs_statistics = &mut Some(&mut SnarkStatistics::default());

    let mut prover_trans = Transcript::default();
    let prover_total_time = std::time::Instant::now();

    println!("[Prover] Starting to generate proofs for modulus switching.");
    let time = std::time::Instant::now();
    let modulus_switching_proof = snarks.modulus_switching.prove(
        &mut prover_trans,
        &modulus_switching_trace,
        &modulus_switching_params,
        pcs_statistics,
    );
    println!(
        "[Prover] Modulus switching proof generation time: {:?}\n",
        time.elapsed()
    );

    println!("[Prover] Starting to generate proofs for blind rotation.");
    let time = std::time::Instant::now();
    let blind_rotation_proof = snarks.blind_rotation.prove(
        &mut prover_trans,
        blind_rotation_trace,
        &blind_rotation_params,
        pcs_statistics,
    );
    println!(
        "[Prover] Blind rotation proof generation time: {:?}\n",
        time.elapsed()
    );

    println!("[Prover] Starting to generate proofs for key switching.");
    let time = std::time::Instant::now();
    let key_switching_proof = snarks.key_switching.prove(
        &mut prover_trans,
        &key_switching_trace,
        &key_switching_params,
        pcs_statistics,
    );
    println!(
        "[Prover] Key switching proof generation time: {:?}\n",
        time.elapsed()
    );

    println!("[Prover] Starting to generate proofs for sample extraction.");
    let time = std::time::Instant::now();
    let sample_extraction_proof = snarks
        .sample_extraction
        .prove(&mut prover_trans, &sample_extraction_trace.into());
    println!(
        "[Prover] Sample extraction proof generation time: {:?}\n",
        time.elapsed()
    );

    println!("--- Proofs generation done! ---\n");
    let prover_total_time = prover_total_time.elapsed();
    println!("Proof generation time: {:?}\n", prover_total_time);

    let mut verifier_trans = Transcript::default();
    let mut res = true;
    let verifier_total_time = std::time::Instant::now();

    println!("[Verifier] Starting to check modulus switching.");
    let time = std::time::Instant::now();
    res &= snarks.modulus_switching.verify(
        &mut verifier_trans,
        &modulus_switching_proof,
        pcs_statistics,
    );
    println!(
        "[Verifier] Modulus switching verification time: {:?}\n",
        time.elapsed()
    );

    println!("[Verifier] Starting to check blind rotation.");
    let time = std::time::Instant::now();
    res &= snarks
        .blind_rotation
        .verify(&mut verifier_trans, &blind_rotation_proof, pcs_statistics);
    println!(
        "[Verifier] Blind rotation verification time: {:?}\n",
        time.elapsed()
    );

    println!("[Verifier] Starting to check key switching.");
    let time = std::time::Instant::now();
    res &= snarks
        .key_switching
        .verify(&mut verifier_trans, &key_switching_proof, pcs_statistics);
    println!(
        "[Verifier] Key switching verification time: {:?}\n",
        time.elapsed()
    );

    println!("[Verifier] Starting to check sample extraction.");
    let time = std::time::Instant::now();
    res &= snarks.sample_extraction.verify(
        &mut verifier_trans,
        &sample_extraction_proof,
        pcs_statistics,
    );
    println!(
        "[Verifier] Sample extraction verification time: {:?}\n",
        time.elapsed()
    );
    assert!(res);

    println!("--- Proofs verification done! ---\n");
    let verifier_total_time = verifier_total_time.elapsed();
    println!("Proof verification total time: {:?}\n", verifier_total_time);

    // ------------ Statistics --------------------
    println!("--- SNARK Statistics Summary ---\n");
    println!("Prover Total Time: {:?}", prover_total_time);
    if let Some(stats) = pcs_statistics {
        let pcs_ratio =
            stats.prover_pcs_time.as_secs_f64() / prover_total_time.as_secs_f64() * 100.0;
        println!(
            "Prover PCS Time (including commit and open): {:?}, accounts for {:.2}%",
            stats.prover_pcs_time, pcs_ratio
        );
        println!(
            "Prover PIOP Time: {:?}\n",
            prover_total_time - stats.prover_pcs_time
        );
    }
    println!("Verifier Total Time: {:?}", verifier_total_time);
    if let Some(stats) = pcs_statistics {
        let pcs_ratio =
            stats.verifier_pcs_time.as_secs_f64() / verifier_total_time.as_secs_f64() * 100.0;
        println!(
            "Verifier PCS Time (including commit and open): {:?}, accounts for {:.2}%",
            stats.verifier_pcs_time, pcs_ratio
        );
        println!(
            "Verifier PIOP Time: {:?}\n",
            verifier_total_time - stats.verifier_pcs_time
        );
    }

    let piop_size = modulus_switching_proof.piop_proof_len()
        + blind_rotation_proof.piop_proof_len()
        + key_switching_proof.piop_proof_len()
        + sample_extraction_proof.piop_proof_len();
    let pcs_size = modulus_switching_proof.pcs_proof_len()
        + blind_rotation_proof.pcs_proof_len()
        + key_switching_proof.pcs_proof_len()
        + sample_extraction_proof.pcs_proof_len();
    println!(
        "Proof Sizes: {} MB total",
        (piop_size + pcs_size) as f64 / (1024 * 1024) as f64
    );
    println!(
        "PCS Proof Sizes: {} MB, accounts for {:.2}%",
        (pcs_size) as f64 / (1024 * 1024) as f64,
        pcs_size as f64 / (piop_size + pcs_size) as f64 * 100.0
    );
    println!(
        "PIOP Proof Sizes: {} MB",
        piop_size as f64 / (1024 * 1024) as f64,
    );
}

// fn main() {}
