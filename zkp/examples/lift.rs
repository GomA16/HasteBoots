use algebra::derive::Field;
use algebra::{transformation::AbstractNTT, NTTField, Polynomial};
use algebra::{BabyBear, BabyBearExetension, Basis, FieldUniformSampler};
use algebra::{DenseMultilinearExtension, Field};
use itertools::izip;
use num_traits::One;
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use rand::prelude::*;
use sha2::Sha256;
use zkp::piop::lift::{LiftInstance, LiftSnarks};
use zkp::piop::ntt_revision::NTTInstanceInfo;
use std::rc::Rc;
use std::sync::Arc;
use std::vec;
use zkp::piop::RlweMultRgswSnarksOpt;
use zkp::piop::{
    DecomposedBitsInfo, RlweCiphertext, RlweCiphertexts, RlweMultRgswInstance,
};

type FF = BabyBear;
type EF = BabyBearExetension;
type Hash = Sha256;
const BASE_FIELD_BITS: usize = 31;

#[derive(Field)]
#[modulus = 2048]
pub struct F32(u32);

// # Parameters
// n = 1024: denotes the dimension of LWE
// N = 1024: denotes the dimension of ring in RLWE
// B = 2^3: denotes the basis used in the bit decomposition
// q = 1024: denotes the modulus in LWE
// Q = DefaultFieldU32: denotes the ciphertext modulus in RLWE
const DIM_LWE: usize = 1024;
const LOG_DIM_RLWE: usize = 10;
const LOG_B: usize = 7;

/// Invoke the existing api to perform ntt transform.
/// The input is in normal order and the output is in the bit-reversed order
fn ntt_transform_reverse_order<F: Field + NTTField>(log_n: u32, coeff: &[F]) -> Vec<F> {
    assert_eq!(coeff.len(), (1 << log_n) as usize);
    let poly = <Polynomial<F>>::from_slice(coeff);
    F::get_ntt_table(log_n).unwrap().transform(&poly).data()
}

fn generate_instance<F: Field + NTTField>(
    num_vars: usize,
    log_N: usize,
    N: F,
    input: Rc<DenseMultilinearExtension<F>>,
    ntt_info: &NTTInstanceInfo<F>,
) -> LiftInstance<F> {
    assert_eq!(num_vars, input.num_vars);

    let mut k = vec![F::zero(); 1 << num_vars];
    let mut row = vec![F::zero(); 1 << num_vars];
    let mut coeffs = Vec::with_capacity(1 << num_vars);
    for (_input, _k, _row) in izip!(input.evaluations.iter(), k.iter_mut(), row.iter_mut()) {
        let mut coeff = vec![F::zero(); 1 << log_N];
        (*_k, *_row) = match _input < &N {
            true => {
                coeff[_input.value().into() as usize] = F::one();
                (F::zero(), *_input)
            }
            false => {
                let idx = *_input - N;
                coeff[idx.value().into() as usize] = -F::one();
                (F::one(), idx)
            }
        };
        let coeff = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, coeff,
        ));
        coeffs.push(coeff);
    }

    let ntt_outpus = coeffs
        .iter()
        .map(|_c| {
            Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                ntt_transform_reverse_order(log_N as u32, &_c.evaluations),
            ))
        })
        .collect();

    let mut ntt_info = ntt_info.clone();
    ntt_info.num_ntt = 1 << num_vars;
    LiftInstance {
        num_vars,
        log_N,
        N,
        input,
        k: Rc::new(DenseMultilinearExtension::from_evaluations_vec(num_vars, k)),
        row: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, row,
        )),
        coeff_cols: coeffs,
        outputs: ntt_outpus,
        ntt_info,
    }
}

fn main() {
    let mut rng = rand::thread_rng();
    let uniform = <FieldUniformSampler<F32>>::new();

    let num_vars = LOG_DIM_RLWE;
    let N = FF::new(DIM_LWE as u32);

    // information used to perform NTT
    let log_n = num_vars;
    let m = 1 << (log_n + 1);
    let mut ntt_table = Vec::with_capacity(m as usize);
    let root = FF::get_ntt_table(log_n as u32).unwrap().root();
    let mut power = FF::one();
    for _ in 0..m {
        ntt_table.push(power);
        power *= root;
    }
    let ntt_table = Arc::new(ntt_table);
    let ntt_info = NTTInstanceInfo {
        num_vars,
        ntt_table,
        num_ntt: 0,
    };

    let input = Rc::new(DenseMultilinearExtension::<FF>::from_evaluations_vec(
        num_vars,
        (0..1 << num_vars)
        .map(|_| FF::new(uniform.sample(&mut rng).value()))
        .collect()
        ,
    ));

    let instance = generate_instance(num_vars, log_n, N, input, &ntt_info);

    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
    <LiftSnarks<FF, EF>>::snarks::<Hash, ExpanderCode<FF>, ExpanderCodeSpec>(
        &instance, &code_spec,
    );
}   


