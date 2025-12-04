use std::rc::Rc;
use std::sync::Arc;

use algebra::derive::Field;
use algebra::{
    BabyBear, BabyBearExetension, DenseMultilinearExtension, Field, FieldUniformSampler, NTTField,
    Polynomial, transformation::AbstractNTT,
};
use itertools::izip;
use num_traits::One;
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use rand_distr::Distribution;
use sha2::Sha256;
use zkp::piop::lift::{LiftIOP, LiftInstance, LiftSnarks};
use zkp::piop::ntt_revision::NTTInstanceInfo;

// field type
type FF = BabyBear;
type EF = BabyBearExetension;
type Hash = Sha256;
const BASE_FIELD_BITS: usize = 31;

macro_rules! field_vec {
    ($t:ty; $elem:expr; $n:expr)=>{
        vec![<$t>::new($elem);$n]
    };
    ($t:ty; $($x:expr),+ $(,)?) => {
        vec![$(<$t>::new($x)),+]
    }
}

#[derive(Field)]
#[modulus = 2048]
pub struct F32(u32);

/// Invoke the existing api to perform ntt transform.
/// The input is in normal order and the output is in the bit-reversed order
fn ntt_transform_reverse_order<F: Field + NTTField>(log_n: u32, coeff: &[F]) -> Vec<F> {
    assert_eq!(coeff.len(), (1 << log_n) as usize);
    let poly = <Polynomial<F>>::from_slice(coeff);
    F::get_ntt_table(log_n).unwrap().transform(&poly).data()
}

fn generate_lift_instance<F: Field + NTTField>(
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

#[test]
fn test_lift_naive_iop() {
    let N = FF::new(1024);
    let log_N = 10;
    let num_vars = 10;

    let mut rng = rand::rng();
    let uniform = <FieldUniformSampler<F32>>::new();
    let input = Rc::new(DenseMultilinearExtension::<FF>::from_evaluations_vec(
        num_vars,
        (0..1 << num_vars)
            .map(|_| FF::new(uniform.sample(&mut rng).value()))
            .collect(),
    ));

    // information used to perform NTT
    let log_n = log_N;
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
    let instance = generate_lift_instance(num_vars, log_N, N, input, &ntt_info);

    let info = instance.info();

    let (kit, recursive_proof) = LiftIOP::<FF>::prove(&instance);

    let evals_at_r = instance.evaluate(&kit.randomness);
    let evals_at_u = instance.evaluate(&kit.u);

    let mut wrapper = kit.extract();
    let check = LiftIOP::<FF>::verify(
        &mut wrapper,
        &evals_at_r,
        &evals_at_u,
        &info,
        &recursive_proof,
    );

    assert!(check);
}

#[test]
fn test_snarks() {
    let N = FF::new(1024);
    let log_N = 10;
    let num_vars = 10;

    let mut rng = rand::rng();
    let uniform = <FieldUniformSampler<F32>>::new();
    let input = Rc::new(DenseMultilinearExtension::<FF>::from_evaluations_vec(
        num_vars,
        (0..1 << num_vars)
            .map(|_| FF::new(uniform.sample(&mut rng).value()))
            .collect(),
    ));

    // information used to perform NTT
    let log_n = log_N;
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
    let instance = generate_lift_instance(num_vars, log_N, N, input, &ntt_info);

    let code_spec = ExpanderCodeSpec::new(0.1195, 0.0248, 1.9, BASE_FIELD_BITS, 10);
    <LiftSnarks<FF, EF>>::snarks::<Hash, ExpanderCode<FF>, ExpanderCodeSpec>(&instance, &code_spec);
}
