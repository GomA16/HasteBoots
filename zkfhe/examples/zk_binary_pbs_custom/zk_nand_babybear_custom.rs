use std::fs::OpenOptions;
use std::path::Path;

use algebra::transformation::AbstractNTT;
use algebra::{AbstractExtensionField, AsInto, BabyBear, BabyBearExetension, Field, NTTField};
use fhe_core::utils::*;
use helper::Transcript;
use pcs::PolynomialCommitmentScheme;
use pcs::multilinear::BrakedownPCS;
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use rand::Rng;
use snarks::SnarkStatistics;
use snarks::fhe_op::blind_rotation::{BlindRotationParams, BlindRotationSnarks, KeyCommitment};
use snarks::fhe_op::key_switching::{KeySwitchingParams, KeySwitchingSnarks};
use snarks::fhe_op::modulus_switch::{self, ModulusSwitchingSnarks};
use snarks::fhe_op::row_permutation::RowPermutationSignedSnarks;
use std::io::Write;
use trace::pbs_trace::PBSTrace;
use zkfhe::bfhe::{CUSTOM_BABYBEAR_BINARY_128_BITS_PARAMETERS, Evaluator};
use zkfhe::{Decryptor, Encryptor, KeyGen};

type FF = BabyBear;
type EF = BabyBearExetension;
type Hash = sha2::Sha256;
const BASE_FIELD_BITS: usize = 32;

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

fn run_single_verification() -> (f64, f64, f64, f64, f64, f64) {
    // set random generator
    let mut rng = rand::rng();

    // set parameter
    let params = *CUSTOM_BABYBEAR_BINARY_128_BITS_PARAMETERS;
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
    let (ct_nand, trace) = eval.nand(&x, &y);
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
        blind_rotation_trace,
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
    println!("Prover Total Time: {:.2?}", prover_total_time);
    if let Some(stats) = pcs_statistics {
        let pcs_ratio =
            stats.prover_pcs_time.as_secs_f64() / prover_total_time.as_secs_f64() * 100.0;
        println!(
            "Prover PCS Time (including commit and open): {:?}, accounts for {:.2}%",
            stats.prover_pcs_time, pcs_ratio
        );
        println!(
            "Prover PIOP Time: {:.2?}\n",
            prover_total_time - stats.prover_pcs_time
        );
    }
    println!("Verifier Total Time: {:.2?}", verifier_total_time);
    if let Some(stats) = pcs_statistics {
        let pcs_ratio =
            stats.verifier_pcs_time.as_secs_f64() / verifier_total_time.as_secs_f64() * 100.0;
        println!(
            "Verifier PCS Time (including commit and open): {:?}, accounts for {:.2}%",
            stats.verifier_pcs_time, pcs_ratio
        );
        println!(
            "Verifier PIOP Time: {:.2?}\n",
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
        "Proof Sizes: {:.2} MB total",
        (piop_size + pcs_size) as f64 / (1024 * 1024) as f64
    );
    println!(
        "PCS Proof Sizes: {:.2} MB, accounts for {:.2}%",
        (pcs_size) as f64 / (1024 * 1024) as f64,
        pcs_size as f64 / (piop_size + pcs_size) as f64 * 100.0
    );
    println!(
        "PIOP Proof Sizes: {:.2} MB",
        piop_size as f64 / (1024 * 1024) as f64,
    );

    // Return metrics
    let total_size_mb = (piop_size + pcs_size) as f64 / (1024 * 1024) as f64;
    let piop_size_mb = piop_size as f64 / (1024 * 1024) as f64;
    let prover_total_s = prover_total_time.as_secs_f64();

    let (prover_piop_s, verifier_piop_ms) = if let Some(stats) = pcs_statistics {
        (
            (prover_total_time - stats.prover_pcs_time).as_secs_f64(),
            (verifier_total_time - stats.verifier_pcs_time).as_secs_f64() * 1000.0,
        )
    } else {
        (0.0, 0.0)
    };

    let verifier_total_ms = verifier_total_time.as_secs_f64() * 1000.0;
    (
        prover_total_s,
        prover_piop_s,
        verifier_total_ms,
        verifier_piop_ms,
        total_size_mb,
        piop_size_mb,
    )
}

fn main() {
    env_logger::init();

    println!("Running SNARK verification 3 times to calculate average...\n");

    let mut prover_totals = Vec::new();
    let mut prover_piops = Vec::new();
    let mut verifier_totals = Vec::new();
    let mut verifier_piops = Vec::new();
    let mut total_sizes = Vec::new();
    let mut piop_sizes = Vec::new();

    // Run 3 times
    for i in 1..=3 {
        println!("========================================");
        println!("Run #{}", i);
        println!("========================================\n");

        let (prover_total, prover_piop, verifier_total, verifier_piop, total_size, piop_size) =
            run_single_verification();

        prover_totals.push(prover_total);
        prover_piops.push(prover_piop);
        verifier_totals.push(verifier_total);
        verifier_piops.push(verifier_piop);
        total_sizes.push(total_size);
        piop_sizes.push(piop_size);

        println!("\n");
    }

    // Calculate averages
    let avg_prover_total = prover_totals.iter().sum::<f64>() / 3.0;
    let avg_prover_piop = prover_piops.iter().sum::<f64>() / 3.0;
    let avg_verifier_total = verifier_totals.iter().sum::<f64>() / 3.0;
    let avg_verifier_piop = verifier_piops.iter().sum::<f64>() / 3.0;
    let avg_total_size = total_sizes.iter().sum::<f64>() / 3.0;
    let avg_piop_size = piop_sizes.iter().sum::<f64>() / 3.0;

    println!("========================================");
    println!("Average Results (3 runs)");
    println!("========================================");
    println!("Prover Total: {:.2} s", avg_prover_total);
    println!("Prover PIOP: {:.2} s", avg_prover_piop);
    println!("Verifier Total: {:.2} ms", avg_verifier_total);
    println!("Verifier PIOP: {:.2} ms", avg_verifier_piop);
    println!("Proof Total: {:.4} MB", avg_total_size);
    println!("Proof PIOP: {:.4} MB", avg_piop_size);

    // ------------ Output to CSV --------------------
    let csv_path = "snark_statistics.csv";
    let file_exists = Path::new(csv_path).exists();

    let mut csv_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(csv_path)
        .expect("Failed to open CSV file");

    // Write header only if file is new
    if !file_exists {
        writeln!(csv_file, "Run,Prover Total (s),Prover PIOP (s),Verifier Total (s),Verifier PIOP (s),Proof Total (MB),Proof PIOP (MB)")
            .expect("Failed to write header");
    }

    // Get run number from file line count
    let run_number = if file_exists {
        std::fs::read_to_string(csv_path)
            .map(|content| content.lines().count())
            .unwrap_or(1)
    } else {
        1
    };

    writeln!(
        csv_file,
        "{},{:.2},{:.2},{:.2},{:.2},{:.4},{:.4}",
        run_number,
        avg_prover_total,
        avg_prover_piop,
        avg_verifier_total,
        avg_verifier_piop,
        avg_total_size,
        avg_piop_size
    )
    .expect("Failed to write data");

    println!(
        "\n✓ Average statistics appended to: {} (Entry #{})",
        csv_path, run_number
    );
}

// fn main() {}
