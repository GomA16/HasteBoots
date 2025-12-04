use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::prelude::*;
use rayon::vec;
use sha2::Sha256;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use algebra::{BabyBear, BabyBearExetension, NTTPolynomial};
use algebra::{DecomposableField, DenseMultilinearExtension, Field};
use algebra::{NTTField, Polynomial, transformation::AbstractNTT};
use num_traits::Zero;
use pcs::utils::code::{self, ExpanderCode, ExpanderCodeSpec};
use zkp::piop::NTTBareIOP;
use zkp::piop::ntt::ntt_bare::init_fourier_table;
use zkp::piop::ntt_revision::NTTSnarks;
use zkp::piop::ntt_revision::{NTTInstance, NTTInstances};

type FF = BabyBear;
type EF = BabyBearExetension;
type Hash = Sha256;
const BASE_FIELD_BITS: usize = 31;

fn build_instances(log_n: u32, num_ntt: u32) -> NTTInstances<FF> {
    let m = 1 << (log_n + 1);
    let mut powers: Vec<FF> = vec![FF::zero(); m];
    let plan = FF::get_ntt_table(log_n).unwrap();
    plan.root_powers(&mut powers);
    let ntt_table = Arc::new(powers);

    let mut rng = rand::rng();
    let mut instances = NTTInstances::new(log_n as usize, &ntt_table);

    for _ in 0..num_ntt {
        let coeff = Polynomial::<FF>::random(1 << log_n, &mut rng);
        let evals: NTTPolynomial<BabyBear> = coeff.clone().into();

        let coeff = Rc::new(DenseMultilinearExtension::from_polynomial(
            log_n as usize,
            coeff,
        ));
        let evals = Rc::new(DenseMultilinearExtension::from_ntt_polynomial(
            log_n as usize,
            evals,
        ));
        instances.add_ntt(&coeff, &evals);
    }
    instances
}

// # Parameters
// n = 1024: denotes the dimension of LWE
// N = 1024: denotes the dimension of ring in RLWE
// B = 2^3: denotes the basis used in the bit decomposition
// q = 1024: denotes the modulus in LWE
// Q = DefaultFieldU32: denotes the ciphertext modulus in RLWE
const DIM_LWE: usize = 1024;
const LOG_DIM_RLWE: usize = 10;
const BITS_LEN: usize = 10;

fn bench_ntt_snarks(c: &mut Criterion) {
    let num_vars = LOG_DIM_RLWE;
    let log_n = num_vars;
    // let m = 1 << (log_n + 1);
    let num_ntt = 8;

    let instances = build_instances(log_n as u32, num_ntt as u32);
    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
    c.bench_function("ntt_snarks", |b| {
        b.iter(|| {
            <NTTSnarks<FF, EF>>::snarks::<Hash, ExpanderCode<FF>, ExpanderCodeSpec>(
                &instances, &code_spec,
            );
        });
    });
}

criterion_group!(ntt_snarks_bench, bench_ntt_snarks);
criterion_main!(ntt_snarks_bench);
