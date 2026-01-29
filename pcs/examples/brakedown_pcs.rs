use std::time::Instant;

use algebra::{BabyBear, BabyBearExetension, DenseMultilinearExtension, FieldUniformSampler};
use helper::Transcript;
use pcs::{
    PolynomialCommitmentScheme,
    multilinear::brakedown::BrakedownPCS,
    utils::code::{ExpanderCode, ExpanderCodeSpec},
};
use rand::Rng;
use sha2::Sha256;

type FF = BabyBear;
type EF = BabyBearExetension;
type Hash = Sha256;
const BASE_FIELD_BITS: usize = 31;

fn main() {
    let num_vars = 20;
    let evaluations: Vec<EF> = rand::rng()
        .sample_iter(FieldUniformSampler::new())
        .take(1 << num_vars)
        .collect();

    let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations);

    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0284, 1.9, BASE_FIELD_BITS, 10);

    let start = Instant::now();
    let pp = BrakedownPCS::<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>::setup(
        num_vars, &code_spec,
    );
    println!("setup time: {:?} ms", start.elapsed().as_millis());

    let mut trans = Transcript::<EF>::new();

    let start = Instant::now();
    let (comm, state) =
        BrakedownPCS::<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>::commit_ef(&pp, &poly);
    println!("commit time: {:?} ms", start.elapsed().as_millis());

    let point: Vec<EF> = rand::rng()
        .sample_iter(FieldUniformSampler::new())
        .take(num_vars)
        .collect();
    let point2: Vec<EF> = rand::rng()
        .sample_iter(FieldUniformSampler::new())
        .take(num_vars)
        .collect();
    let points = vec![point.clone(), point2.clone()];

    let start = Instant::now();
    // let proof = BrakedownPCS::<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>::open(
    //     &pp, &comm, &state, &point, &mut trans,
    // );
    let proof = BrakedownPCS::<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>::open_ef(
        &pp, &comm, &state, &point, &mut trans,
    );
    println!("open time: {:?} ms", start.elapsed().as_millis());

    let eval = poly.evaluate(&point);
    // let eval2 = poly.evaluate_ext(&point2);

    let mut trans = Transcript::<EF>::new();

    let start = Instant::now();
    let check = BrakedownPCS::<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>::verify_ef(
        &pp, &comm, &point, eval, &proof, &mut trans,
    );
    println!("verify time: {:?} ms", start.elapsed().as_millis());

    println!("proof size: {:?} Bytes", proof.to_bytes().unwrap().len());

    assert!(check);
}
