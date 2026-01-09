use algebra::transformation::AbstractNTT;
use algebra::{AbstractExtensionField, AsInto, BabyBear, BabyBearExetension, Field, NTTField};
use fhe_core::utils::*;
use helper::Transcript;
use pcs::PolynomialCommitmentScheme;
use pcs::multilinear::BrakedownPCS;
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use rand::Rng;
use snarks::fhe_op::blind_rotation::{BlindRotationParams, BlindRotationSnarks};
use snarks::fhe_op::blind_rotation_updated::{
    BlindRotationParamsUpdated, BlindRotationSnarksUpdated,
};
use snarks::fhe_op::external_product::{ExternalProductParams, ExternalProductSnarks};
use snarks::fhe_op::key_switching::{self, KeySwitchingParams, KeySwitchingSnarks};
use snarks::fhe_op::row_permutation::RowPermutationSignedSnarks;
use trace::BlindRotationTraceMLE;
use trace::pbs_trace::PBSTrace;
// use trace::HadamardProdTraceMLE;
use zkfhe::bfhe::{
    BABYBEAR_BINARY_128_BITS_PARAMETERS, CUSTOM_TERNARY_128_BITS_PARAMETERS, Evaluator,
};
use zkfhe::{Decryptor, Encryptor, KeyGen};

type FF = BabyBear;
type EF = BabyBearExetension;
type Hash = sha2::Sha256;
const BASE_FIELD_BITS: usize = 31;

#[derive(Default)]
pub struct PBSSnarks<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub blind_rotation: BlindRotationSnarksUpdated<F, EF, S, PCS>,
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
    let (ct_nand, mut trace) = eval.nand(&x, &y);
    println!("NAND Evaluation Time is : {:?}\n", start.elapsed());

    // nand
    let (m, noise) = dec.decrypt_with_noise(&ct_nand);
    assert_eq!(m, nand(a, b), "Noise: {noise}");
    check_noise(noise, "nand");

    // Generate SNARKs for nand
    println!("");
    println!("Starting verification of nand.\n");

    // Perepare parameters and traces
    let PBSTrace {
        mut blind_rotation_trace,
        mut key_switching_trace,
        mut sample_extraction_trace,
    } = trace;

    blind_rotation_trace.finalize(params.lwe_dimension());
    let blind_rotation_ntt_table = FF::get_ntt_table(blind_rotation_trace.log_coeff_count as u32)
        .unwrap()
        .root_powers();
    let key_switching_ntt_table = FF::get_ntt_table(key_switching_trace.log_coeff_count as u32)
        .unwrap()
        .root_powers();

    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);

    let blk_size = 3;
    let blind_rotation_basis = params.blind_rotation_basis().basis() as usize;
    let blind_rotation_params = BlindRotationParamsUpdated::new(
        code_spec.clone(),
        blind_rotation_ntt_table,
        blk_size,
        blind_rotation_basis,
        &blind_rotation_trace,
    );

    let key_switching_trace = key_switching_trace.into();
    let key_switching_basis = 1 << params.key_switching_basis_bits() as usize;
    let key_switching_params = KeySwitchingParams::new(
        code_spec.clone(),
        key_switching_ntt_table,
        blk_size,
        key_switching_basis,
        &key_switching_trace,
    );

    let snarks = PBSSnarks::<
        FF,
        EF,
        ExpanderCodeSpec,
        BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
    >::default();

    let mut prover_trans = Transcript::default();
    let time = std::time::Instant::now();
    let blind_rotation_proof = snarks.blind_rotation.prove(
        &mut prover_trans,
        blind_rotation_trace,
        &blind_rotation_params,
    );
    let key_switching_proof = snarks.key_switching.prove(
        &mut prover_trans,
        &key_switching_trace,
        &key_switching_params,
    );
    let sample_extraction_proof = snarks
        .sample_extraction
        .prove(&mut prover_trans, &sample_extraction_trace.into());
    println!("Proofs generation done!\n");
    println!("Proof generation time: {:?}\n", time.elapsed());

    let mut verifier_trans = Transcript::default();
    let mut res = true;
    let time = std::time::Instant::now();
    res &= snarks
        .blind_rotation
        .verify(&mut verifier_trans, &blind_rotation_proof);
    res &= snarks
        .key_switching
        .verify(&mut verifier_trans, &key_switching_proof);
    res &= snarks
        .sample_extraction
        .verify(&mut verifier_trans, &sample_extraction_proof);
    println!("Proofs verification done!\n");

    println!("Proof verification time: {:?}\n", time.elapsed());
    println!(
        "PIOP Proof Size: {} MB",
        blind_rotation_proof.piop_proof_len() as f64 / (1000 * 1000) as f64
    );
    println!(
        "PCS Proof Size: {} MB",
        blind_rotation_proof.pcs_proof_len() as f64 / (1000 * 1000) as f64
    );
    assert!(res);
}

// fn main() {}
