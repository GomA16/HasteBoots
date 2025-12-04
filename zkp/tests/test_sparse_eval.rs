use std::rc::Rc;
use std::vec;

use algebra::{
    derive::{DecomposableField, Field, Prime},
    BabyBear, BabyBearExetension, Basis, DecomposableField, DenseMultilinearExtension, Field,
    FieldUniformSampler,
};
use helper::Transcript;
use num_traits::{One, Zero};
use pcs::utils::code::{ExpanderCode, ExpanderCodeSpec};
use rand::prelude::*;
use rand_distr::Distribution;
use sha2::Sha256;
use zkp::piop::{
    LookupIOP,
    sparse_eval::{SparseEvalIOP, SparseEvalInstance},
};

type FF = BabyBear; // field type
type EF = BabyBearExetension;
type Hash = Sha256;
const BASE_FIELD_BITS: usize = 31;
const FP: u32 = FF::MODULUS_VALUE; // ciphertext space
const dim_x: usize = 4;
const dim_y: usize = 4;

#[derive(Field)]
#[modulus = 4]
pub struct F_num(u32);

macro_rules! field_vec {
    ($t:ty; $elem:expr; $n:expr)=>{
        vec![<$t>::new($elem);$n]
    };
    ($t:ty; $($x:expr),+ $(,)?) => {
        vec![$(<$t>::new($x)),+]
    }
}

#[test]
fn test_sparse_eval_naive_iop() {
    let matrix = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
        4,
        field_vec!(FF; 0, 1, 2, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ));
    let row = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
        2,
        field_vec!(FF; 1, 0, 0, 0),
    ));
    let val = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
        2,
        field_vec!(FF; 1, 1, 2, 5),
    ));
    let mut instance = SparseEvalInstance::<FF>::from_slice(1, 2, &row, &val);

    let mut rng = rand::rng();
    let uniform_f = <FieldUniformSampler<FF>>::new();
    let r_x: Vec<_> = (0..2).map(|_| uniform_f.sample(&mut rng)).collect();
    let r_y: Vec<_> = (0..2).map(|_| uniform_f.sample(&mut rng)).collect();
    let mut r = Vec::with_capacity(3);
    r.extend(&r_y);
    r.extend(&r_x);

    let eval = matrix.evaluate(&r);

    let iop = SparseEvalIOP {
        r_x: r_x.clone(),
        r_y: r_y.clone(),
        eval,
    };

    iop.prover_generate_eval_vector(&mut instance);
    let kit = iop.prove(&instance);
    let evals = instance.evaluate(&kit.randomness);

    let wrapper = kit.extract();
    let check = iop.verify(&wrapper, &evals);

    assert!(check);
}

#[test]
fn test_sparse_eval_naive_iop_with_lookup() {
    let matrix = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
        4,
        field_vec!(FF; 0, 1, 2, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ));
    let row = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
        2,
        field_vec!(FF; 1, 0, 0, 0),
    ));
    let val = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
        2,
        field_vec!(FF; 1, 1, 2, 5),
    ));
    let mut instance = SparseEvalInstance::<FF>::from_slice(1, 2, &row, &val);

    let mut rng = rand::rng();
    let uniform_f = <FieldUniformSampler<FF>>::new();
    let r_x: Vec<_> = (0..2).map(|_| uniform_f.sample(&mut rng)).collect();
    let r_y: Vec<_> = (0..2).map(|_| uniform_f.sample(&mut rng)).collect();
    let mut r = Vec::with_capacity(3);
    r.extend(&r_y);
    r.extend(&r_x);

    let eval = matrix.evaluate(&r);

    let iop = SparseEvalIOP {
        r_x: r_x.clone(),
        r_y: r_y.clone(),
        eval,
    };

    iop.prover_generate_eval_vector(&mut instance);
    let mut lookup_instance = instance.extract_lookup_instance();
    let lookup_info = lookup_instance.info();

    let kit = iop.prove(&instance);
    let mut prover_trans = Transcript::<FF>::new();

    let mut lookup = LookupIOP::default();

    lookup.prover_generate_first_randomness(&mut prover_trans, &mut lookup_instance);
    lookup.generate_second_randomness(&mut prover_trans, &lookup_info);
    let lookup_kit = lookup.prove(&mut prover_trans, &mut lookup_instance);

    let evals = instance.evaluate(&kit.randomness);
    let lookup_evals = lookup_instance.evaluate(&lookup_kit.randomness);

    let wrapper = kit.extract();
    let lookup_wrapper = lookup_kit.extract();

    let check = iop.verify(&wrapper, &evals);
    let mut verifier_trans = Transcript::<FF>::new();
    let mut lookup = LookupIOP::default();

    lookup.verifier_generate_first_randomness(&mut verifier_trans);
    lookup.generate_second_randomness(&mut verifier_trans, &lookup_info);
    let (lookup_check, _) = lookup.verify(
        &mut verifier_trans,
        &lookup_wrapper,
        &lookup_evals,
        &lookup_info,
    );

    assert!(check && lookup_check);
}

#[test]
fn test_sparse_eval_random_with_lookupiop() {
    let mut rng = rand::rng();
    let uniform_f = <FieldUniformSampler<FF>>::new();
    let uniform_row = <FieldUniformSampler<F_num>>::new();

    let row_vec_origin: Vec<_> = (0..(1 << dim_y))
        .map(|_| uniform_row.sample(&mut rng))
        .collect();
    let row_vec: Vec<_> = row_vec_origin.iter().map(|x| FF::new(x.value())).collect();
    let val_vec: Vec<FF> = (0..(1 << dim_y))
        .map(|_| uniform_f.sample(&mut rng))
        .collect();
    let mut matrix_vec = vec![FF::zero(); (1 << dim_x) * (1 << dim_y)];
    for (col, (row, val)) in row_vec.iter().zip(val_vec.iter()).enumerate() {
        let idx = (row.value() as usize) * (1 << dim_y) + col;
        matrix_vec[idx] = *val;
    }

    let matrix = DenseMultilinearExtension::from_evaluations_vec(dim_x + dim_y, matrix_vec);
    let row = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
        dim_y, row_vec,
    ));
    let val = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
        dim_y, val_vec,
    ));
    let mut instance = SparseEvalInstance::<FF>::from_slice(dim_x, dim_y, &row, &val);

    let r_x: Vec<_> = (0..dim_x).map(|_| uniform_f.sample(&mut rng)).collect();
    let r_y: Vec<_> = (0..dim_y).map(|_| uniform_f.sample(&mut rng)).collect();
    let mut r = Vec::with_capacity(dim_x + dim_y);
    r.extend(&r_y);
    r.extend(&r_x);

    let eval = matrix.evaluate(&r);

    let iop = SparseEvalIOP {
        r_x: r_x.clone(),
        r_y: r_y.clone(),
        eval,
    };

    iop.prover_generate_eval_vector(&mut instance);
    let mut lookup_instance = instance.extract_lookup_instance();
    let lookup_info = lookup_instance.info();

    let kit = iop.prove(&instance);
    let mut prover_trans = Transcript::<FF>::new();

    let mut lookup = LookupIOP::default();

    lookup.prover_generate_first_randomness(&mut prover_trans, &mut lookup_instance);
    lookup.generate_second_randomness(&mut prover_trans, &lookup_info);
    let lookup_kit = lookup.prove(&mut prover_trans, &mut lookup_instance);

    let evals = instance.evaluate(&kit.randomness);
    let lookup_evals = lookup_instance.evaluate(&lookup_kit.randomness);

    let wrapper = kit.extract();
    let lookup_wrapper = lookup_kit.extract();

    let check = iop.verify(&wrapper, &evals);
    let mut verifier_trans = Transcript::<FF>::new();
    let mut lookup = LookupIOP::default();

    lookup.verifier_generate_first_randomness(&mut verifier_trans);
    lookup.generate_second_randomness(&mut verifier_trans, &lookup_info);
    let (lookup_check, _) = lookup.verify(
        &mut verifier_trans,
        &lookup_wrapper,
        &lookup_evals,
        &lookup_info,
    );

    assert!(check && lookup_check);
}

#[test]
fn test_sparse_eval_random_iop() {
    let mut rng = rand::rng();
    let uniform_f = <FieldUniformSampler<FF>>::new();
    let uniform_row = <FieldUniformSampler<F_num>>::new();

    let row_vec_origin: Vec<_> = (0..(1 << dim_y))
        .map(|_| uniform_row.sample(&mut rng))
        .collect();
    let row_vec: Vec<_> = row_vec_origin.iter().map(|x| FF::new(x.value())).collect();
    let val_vec: Vec<FF> = (0..(1 << dim_y))
        .map(|_| uniform_f.sample(&mut rng))
        .collect();
    let mut matrix_vec = vec![FF::zero(); (1 << dim_x) * (1 << dim_y)];
    for (col, (row, val)) in row_vec.iter().zip(val_vec.iter()).enumerate() {
        let idx = (row.value() as usize) * (1 << dim_y) + col;
        matrix_vec[idx] = *val;
    }

    let matrix = DenseMultilinearExtension::from_evaluations_vec(dim_x + dim_y, matrix_vec);
    let row = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
        dim_y, row_vec,
    ));
    let val = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
        dim_y, val_vec,
    ));
    let mut instance = SparseEvalInstance::<FF>::from_slice(dim_x, dim_y, &row, &val);

    let r_x: Vec<_> = (0..dim_x).map(|_| uniform_f.sample(&mut rng)).collect();
    let r_y: Vec<_> = (0..dim_y).map(|_| uniform_f.sample(&mut rng)).collect();
    let mut r = Vec::with_capacity(dim_x + dim_y);
    r.extend(&r_y);
    r.extend(&r_x);

    let eval = matrix.evaluate(&r);

    let iop = SparseEvalIOP {
        r_x: r_x.clone(),
        r_y: r_y.clone(),
        eval,
    };

    iop.prover_generate_eval_vector(&mut instance);
    let kit = iop.prove(&instance);
    let evals = instance.evaluate(&kit.randomness);

    let wrapper = kit.extract();
    let check = iop.verify(&wrapper, &evals);

    assert!(check);
}
