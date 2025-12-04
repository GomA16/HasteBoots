use algebra::AsFrom;
use algebra::derive::{DecomposableField, Field};
use algebra::{BabyBear, BabyBearExetension, Basis, DenseMultilinearExtension};
use algebra::{DecomposableField, Field, FieldUniformSampler};
use itertools::izip;
use num_traits::{One, Zero};
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use rand::prelude::*;
use rand_distr::Distribution;
use sha2::Sha256;
use std::rc::Rc;
use zkp::piop::{
    AdditionInZqInstance, AdditionInZqSnarks, AdditionInZqSnarksOpt, BitDecompositionSnarks,
    DecomposedBits, DecomposedBitsInfo,
};

type FF = BabyBear;
type EF = BabyBearExetension;
type Hash = Sha256;
const BASE_FIELD_BITS: usize = 31;

#[derive(Field, DecomposableField)]
#[modulus = 1024]
pub struct Fq(u32);

const DIM_LWE: usize = 1024;
const LOG_DIM_RLWE: usize = 10;
const MOD_LWE: usize = 1024;
const LOG_B: u32 = 1;

fn main() {
    let mut rng = rand::rng();
    let uniform_fq = <FieldUniformSampler<Fq>>::new();
    let num_vars = LOG_DIM_RLWE;
    let q = FF::new(MOD_LWE as u32);

    let base_len = LOG_B as usize;
    let base: FF = FF::new(1 << LOG_B);
    let bits_len = <Basis<Fq>>::new(base_len as u32).decompose_len();

    // Addition in Zq
    let a: Vec<_> = (0..(1 << num_vars))
        .map(|_| uniform_fq.sample(&mut rng))
        .collect();
    let b: Vec<_> = (0..(1 << num_vars))
        .map(|_| uniform_fq.sample(&mut rng))
        .collect();
    let c_k: Vec<_> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            if x.value() + y.value() >= Fq::MODULUS_VALUE {
                (*x + *y, Fq::one())
            } else {
                (*x + *y, Fq::zero())
            }
        })
        .collect();

    let (c, k): (Vec<_>, Vec<_>) = c_k.iter().cloned().unzip();

    let abc = vec![
        Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            // Convert to Fp
            a.iter().map(|x: &Fq| FF::new(x.value())).collect(),
        )),
        Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            b.iter().map(|x: &Fq| FF::new(x.value())).collect(),
        )),
        Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            c.iter().map(|x: &Fq| FF::new(x.value())).collect(),
        )),
    ];

    let k = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
        num_vars,
        k.iter().map(|x: &Fq| FF::new(x.value())).collect(),
    ));

    let bits_info = DecomposedBitsInfo::<FF> {
        base,
        base_len,
        bits_len,
        num_vars,
        num_instances: 3,
    };
    let instance = AdditionInZqInstance::<FF>::from_slice(&abc, &k, q, &bits_info);

    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
    <AdditionInZqSnarks<FF, EF>>::snarks::<Hash, ExpanderCode<FF>, ExpanderCodeSpec>(
        &instance, &code_spec,
    );
}
