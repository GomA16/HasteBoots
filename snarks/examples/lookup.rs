use core::time;
use std::time::Instant;

use algebra::{BabyBear, BabyBearExetension};
use bincode::config::standard;
use helper::Transcript;
use pcs::{
    multilinear::BrakedownPCS,
    utils::code::{ExpanderCode, ExpanderCodeSpec},
};
use snarks::lookup::{LogUpParams, LogUpSnarks};
use trace::LookupTrace;

type FF = BabyBear;
type EF = BabyBearExetension;
type Hash = sha2::Sha256;
const BASE_FIELD_BITS: usize = 31;

fn main() {
    let mut rng = rand::rng();
    let num_vars = 20;
    let num_vec = 10;
    let range = 1 << 7;
    let blk_size = 3;

    let lookup_trace = LookupTrace::<FF>::random(&mut rng, num_vars, num_vec, range);
    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
    let snarks = LogUpSnarks::<
        FF,
        EF,
        ExpanderCodeSpec,
        BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
    >::default();
    let time = Instant::now();
    let params = &mut LogUpParams::new(code_spec, blk_size, &lookup_trace);
    println!("Setup time: {:?}", time.elapsed());

    let prover_trans = &mut Transcript::<EF>::default();
    let start = Instant::now();
    let proof = snarks.prove(prover_trans, lookup_trace, params);
    println!("Prove time: {:?}", start.elapsed());

    let verifier_trans = &mut Transcript::<EF>::default();
    let start = Instant::now();
    let res = snarks.verifier(verifier_trans, &proof);
    println!("Verify time: {:?}", start.elapsed());
    assert!(res);
}
