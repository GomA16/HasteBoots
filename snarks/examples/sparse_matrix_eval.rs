use core::time;

use algebra::{BabyBear, BabyBearExetension};
use helper::Transcript;
use pcs::{
    multilinear::BrakedownPCS,
    utils::code::{ExpanderCode, ExpanderCodeSpec},
};
use piop::sparse_matrix_eval::sparse_row::SparseRowEvalInstance;
use snarks::{
    sparse_matrix_eval::SparseRowEvalSnarks,
};

type FF = BabyBear;
type EF = BabyBearExetension;
type Hash = sha2::Sha256;

fn main() {
    let mut rng = rand::rng();
    let num_x_vars = 10;
    let num_y_vars = 10;

    let instance = SparseRowEvalInstance::<EF>::random(&mut rng, num_x_vars, num_y_vars);
    let snarks = SparseRowEvalSnarks::<
        FF,
        EF,
        ExpanderCodeSpec,
        BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
    >::default();
    let prover_trans = &mut Transcript::<EF>::default();
    let time = std::time::Instant::now();
    let proof = snarks.prove(prover_trans, &instance);
    println!("Prove time: {:?}", time.elapsed());
    let verifier_trans = &mut Transcript::<EF>::default();
    let time = std::time::Instant::now();
    let res = snarks.verify(verifier_trans, &proof);
    println!("Verify time: {:?}", time.elapsed());
    assert!(res);
}
