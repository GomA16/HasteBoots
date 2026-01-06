use algebra::{BabyBear, BabyBearExetension};
use helper::Transcript;
use pcs::{
    multilinear::BrakedownPCS,
    utils::code::{ExpanderCode, ExpanderCodeSpec},
};
use snarks::lookup::indexed_table::IndexedLogUpSnarks;
use trace::lookup_trace::indexed_table::{IndexedLookupTrace, IndexedLookupTraceMLE};

type FF = BabyBear;
type EF = BabyBearExetension;
type Hash = sha2::Sha256;

fn main() {
    let mut rng = rand::rng();
    let num_input_vars = 10;
    let num_table_vars = 10;

    let lookup_trace = IndexedLookupTrace::<EF>::random(&mut rng, num_input_vars, num_table_vars);
    let lookup_mle: IndexedLookupTraceMLE<EF> = lookup_trace.into();
    let snarks = IndexedLogUpSnarks::<
        FF,
        EF,
        ExpanderCodeSpec,
        BrakedownPCS<FF, Hash, ExpanderCode<FF>, ExpanderCodeSpec, EF>,
    >::default();

    let prover_trans = &mut Transcript::<EF>::default();
    let time = std::time::Instant::now();
    let proof = snarks.prove(prover_trans, &lookup_mle);
    println!("Prove time: {:?}", time.elapsed());

    let verifier_trans = &mut Transcript::<EF>::default();
    let time = std::time::Instant::now();
    let res = snarks.verify(verifier_trans, &proof);
    println!("Verify time: {:?}", time.elapsed());
    assert!(res);
}
