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
use snarks::fhe_batch_op::batch_blind_rotation::{
    BatchBlindRotationParams, BatchBlindRotationSnarks,
};
use snarks::fhe_op::blind_rotation::KeyCommitment;
use snarks::fhe_op::key_switching::{KeySwitchingParams, KeySwitchingSnarks};
use snarks::fhe_op::modulus_switch::{ModulusSwitchingParams, ModulusSwitchingSnarks};
use snarks::fhe_op::row_permutation::RowPermutationSignedSnarks;
use std::io::Write;
use trace::pbs_trace::PBSTrace;
use vfhe::bfhe::{BABYBEAR_BINARY_128_BITS_PARAMETERS, Evaluator};
use vfhe::{Decryptor, Encryptor, KeyGen};

type FF = BabyBear;
type EF = BabyBearExetension;
type Hash = sha2::Sha256;
const BASE_FIELD_BITS: usize = 31;
const LOG_BATCH_SIZE: usize = 2; // batch size = 2^LOG_BATCH_SIZE

#[derive(Default)]
pub struct PBSSnarks<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub modulus_switching: ModulusSwitchingSnarks<F, EF, S, PCS>,
    pub blind_rotation: BatchBlindRotationSnarks<F, EF, S, PCS>,
    pub key_switching: KeySwitchingSnarks<F, EF, S, PCS>,
    pub sample_extraction: RowPermutationSignedSnarks<F, EF, S, PCS>,
}

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
    let (ct_nand, trace) = eval.nand(&x, &y);
    println!("NAND Evaluation Time is : {:?}\n", start.elapsed());

    // nand
    let (m, noise) = dec.decrypt_with_noise(&ct_nand);
    assert_eq!(m, nand(a, b), "Noise: {noise}");
    check_noise(noise, "nand");

    // Prepare Batched Traces
    let batched_trace = trace.generate_batched_trace(LOG_BATCH_SIZE);
    let PBSTrace {
        modulus_switching_trace,
        mut blind_rotation_trace,
        key_switching_trace,
        sample_extraction_trace,
    } = batched_trace;
    blind_rotation_trace.finalize(params.lwe_dimension());

    let blind_rotation_traces = vec![blind_rotation_trace; 1 << LOG_BATCH_SIZE]; // batch size 2

    // Generate SNARKs for nand
    println!("");
    println!("Starting verification of {} nand.\n", 1 << LOG_BATCH_SIZE);

    // Perepare parameters and traces
    // let time = std::time::Instant::now();
    let blind_rotation_ntt_table =
        FF::get_ntt_table(blind_rotation_traces[0].log_coeff_count as u32)
            .unwrap()
            .root_powers();
    let key_switching_ntt_table = FF::get_ntt_table(key_switching_trace.log_coeff_count as u32)
        .unwrap()
        .root_powers();

    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);

    let bs_keys_commitment = KeyCommitment::new(&code_spec, &blind_rotation_traces[0]);
    let blind_rotation_params = BatchBlindRotationParams::new(
        code_spec.clone(),
        blind_rotation_ntt_table,
        &blind_rotation_traces,
        &bs_keys_commitment,
    );

    let key_switching_trace = key_switching_trace.into();
    let key_switching_params = KeySwitchingParams::new(
        code_spec.clone(),
        key_switching_ntt_table,
        &key_switching_trace,
    );

    let modulus_switching_trace = modulus_switching_trace.into();
    let modulus_switching_params = ModulusSwitchingParams::new(&code_spec);

    let snarks = PBSSnarks::<
        FF,
        EF,
        ExpanderCodeSpec,
        BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
    >::default();
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
        blind_rotation_traces,
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

    let prover_total_time = prover_total_time.elapsed();
    println!("--- Proofs generation done! ---\n");
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
    println!("Prover Total Time: {:?} s", prover_total_time.as_secs_f64());
    if let Some(stats) = pcs_statistics {
        let pcs_ratio =
            stats.prover_pcs_time.as_secs_f64() / prover_total_time.as_secs_f64() * 100.0;
        println!(
            "Prover PCS Time (including commit and open): {:?} s, accounts for {:.2}%",
            stats.prover_pcs_time.as_secs_f64(),
            pcs_ratio
        );
        println!(
            "Prover PIOP Time: {:?} s\n",
            (prover_total_time - stats.prover_pcs_time).as_secs_f64()
        );
    }
    println!(
        "Verifier Total Time: {:?} ms",
        verifier_total_time.as_secs_f64() * 1000.0
    );
    if let Some(stats) = pcs_statistics {
        let pcs_ratio =
            stats.verifier_pcs_time.as_secs_f64() / verifier_total_time.as_secs_f64() * 100.0;
        println!(
            "Verifier PCS Time (including commit and open): {:?} ms, accounts for {:.2}%",
            stats.verifier_pcs_time.as_secs_f64() * 1000.0,
            pcs_ratio
        );
        println!(
            "Verifier PIOP Time: {:?} ms\n",
            (verifier_total_time - stats.verifier_pcs_time).as_secs_f64() * 1000.0
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

    // ------------ Output to CSV --------------------
    let csv_path = "statistics/batch_statistics.csv";
    let file_exists = Path::new(csv_path).exists();

    let mut csv_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(csv_path)
        .expect("Failed to open CSV file");

    // Write header only if file is new
    if !file_exists {
        writeln!(csv_file, "Run,Prover Total (ms),Prover PCS (ms),Prover PCS Ratio (%),Prover PIOP (ms),Verifier Total (ms),Verifier PCS (ms),Verifier PCS Ratio (%),Verifier PIOP (ms),Total Size (MB),PCS Size (MB),PCS Size Ratio (%),PIOP Size (MB)")
            .expect("Failed to write header");
    }

    // Calculate all metrics
    let total_size_mb = (piop_size + pcs_size) as f64 / (1024 * 1024) as f64;
    let pcs_size_mb = pcs_size as f64 / (1024 * 1024) as f64;
    let piop_size_mb = piop_size as f64 / (1024 * 1024) as f64;
    let pcs_size_ratio = pcs_size as f64 / (piop_size + pcs_size) as f64 * 100.0;

    let prover_total_s = prover_total_time.as_secs_f64();
    let verifier_total_ms = verifier_total_time.as_secs_f64() * 1000.0;

    // Get run number from file line count
    let run_number = if file_exists {
        std::fs::read_to_string(csv_path)
            .map(|content| content.lines().count())
            .unwrap_or(1)
    } else {
        1
    };

    if let Some(stats) = pcs_statistics {
        let prover_pcs_s = stats.prover_pcs_time.as_secs_f64();
        let prover_pcs_ratio =
            stats.prover_pcs_time.as_secs_f64() / prover_total_time.as_secs_f64() * 100.0;
        let prover_piop_s = (prover_total_time - stats.prover_pcs_time).as_secs_f64();

        let verifier_pcs_ms = stats.verifier_pcs_time.as_secs_f64() * 1000.0;
        let verifier_pcs_ratio =
            stats.verifier_pcs_time.as_secs_f64() / verifier_total_time.as_secs_f64() * 100.0;
        let verifier_piop_ms =
            (verifier_total_time - stats.verifier_pcs_time).as_secs_f64() * 1000.0;

        writeln!(
            csv_file,
            "{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.4},{:.4},{:.2},{:.4}",
            run_number,
            prover_total_s,
            prover_pcs_s,
            prover_pcs_ratio,
            prover_piop_s,
            verifier_total_ms,
            verifier_pcs_ms,
            verifier_pcs_ratio,
            verifier_piop_ms,
            total_size_mb,
            pcs_size_mb,
            pcs_size_ratio,
            piop_size_mb
        )
        .expect("Failed to write data");
    }

    println!(
        "\n✓ Statistics appended to: {} (Run #{})",
        csv_path, run_number
    );
}

// fn main() {}
