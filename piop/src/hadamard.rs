use algebra::AbstractExtensionField;
use algebra::DenseMultilinearExtension;
use algebra::Field;
use algebra::PolynomialInfo;
use helper::FiatShamirTranscript;
use helper::Transcript;
use helper::utils::eval_identity_function;
use itertools::izip;
use serde::Serialize;
use std::collections::HashMap;
use std::collections::btree_map::Range;
use std::rc::Rc;
use sumcheck::MLSumcheck;
use sumcheck::Proof;
use sumcheck::prover;
use sumcheck::verifier::SubClaim;
use trace::SumHadamardTraceEval;
use trace::SumHadamardTraceMLE;
use trace::pbs_trace::PBSTraceEval;

use crate::BatchedSumcheckPIOP;
use crate::LagrangeKernel;
use crate::SumcheckClaim;
use crate::SumcheckInfo;
use crate::SumcheckInstance;
use crate::SumcheckPIOP;
use crate::SumcheckPureBatchedProof;
use crate::SumcheckPureProof;
use crate::SumcheckPureProverState;
use crate::SumcheckPureSubclaim;

pub struct HadamardPIOP<F: Field> {
    _marker: std::marker::PhantomData<F>,
}

pub struct SumHadamardInstance<F: Field> {
    pub num_vars: usize,
    pub num_products: usize,
    pub products: Vec<(
        Rc<DenseMultilinearExtension<F>>,
        Rc<DenseMultilinearExtension<F>>,
    )>,
    // result = \sum products[i][0] * products[i][1]
    pub result: Rc<DenseMultilinearExtension<F>>,
}

pub struct HadamardProverState<F: Field> {
    pub point_r: Vec<F>,
    pub flattened_mle_evals: Vec<F>,
    raw_pointers_lookup_table: HashMap<*const DenseMultilinearExtension<F>, usize>,
}

pub struct HadamardVerifierSubclaim<F: Field> {
    pub point_r: Vec<F>,
}

#[derive(Serialize)]
pub struct SumHadamardInfo<F: Field> {
    pub num_vars: usize,
    pub num_products: usize,
    _marker: std::marker::PhantomData<F>,
}

#[derive(Serialize, Default)]
pub struct SumHadamardEval<F: Field> {
    pub products_at_r: Vec<(F, F)>,
    pub result_at_r: F,
}

#[derive(Serialize)]
pub struct SumHadamardProof<F: Field> {
    pub poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<F>,
    pub hadamard_at_r: SumHadamardEval<F>,
}

#[derive(Serialize)]
pub struct BatchedSumHadamardProof<F: Field> {
    pub poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<F>,
    pub hadamard_at_r: Vec<SumHadamardEval<F>>,
}

impl<F: Field> SumHadamardInstance<F> {
    pub fn add_into_sumcheck(
        &self,
        claim: &mut SumcheckClaim<F>,
        random_lambda: F,
        kernel: &LagrangeKernel<F>,
    ) {
        for (a, b) in &self.products {
            let prod: Vec<Rc<DenseMultilinearExtension<F>>> =
                vec![a.clone(), b.clone(), kernel.eq_at_point.clone()];
            claim.poly.add_product(prod, random_lambda);
        }
        let prod = vec![self.result.clone(), kernel.eq_at_point.clone()];
        claim.poly.add_product(prod, -random_lambda);
    }

    pub fn eval_at_point(&self, point_r: &[F]) -> SumHadamardEval<F> {
        let mut products_at_r = Vec::with_capacity(self.num_products);
        for (a, b) in &self.products {
            products_at_r.push((a.evaluate(point_r), b.evaluate(point_r)));
        }
        SumHadamardEval {
            products_at_r,
            result_at_r: self.result.evaluate(point_r),
        }
    }

    pub fn fast_eval_from_prover_state(
        &self,
        state: &HadamardProverState<F>,
    ) -> SumHadamardEval<F> {
        let lookup = |m: &Rc<DenseMultilinearExtension<F>>| {
            let m_ptr: *const DenseMultilinearExtension<F> = Rc::as_ptr(m);
            let index = state.raw_pointers_lookup_table.get(&m_ptr).unwrap();
            state.flattened_mle_evals[*index]
        };

        let mut products_at_r = Vec::with_capacity(self.num_products);
        for (a, b) in &self.products {
            products_at_r.push((lookup(a), lookup(b)));
        }
        SumHadamardEval {
            products_at_r,
            result_at_r: lookup(&self.result),
        }
    }

    pub fn random<R: rand::Rng + rand::CryptoRng>(
        num_vars: usize,
        num_prods: usize,
        rng: &mut R,
    ) -> Self {
        let mut result = vec![F::zero(); 1 << num_vars];
        let mut add_assign_ab = |a: &DenseMultilinearExtension<F>,
                                 b: &DenseMultilinearExtension<F>| {
            result
                .iter_mut()
                .zip(a.iter().zip(b.iter()))
                .for_each(|(res, (a, b))| {
                    *res += *a * *b;
                });
        };
        let products = (0..num_prods)
            .map(|_| {
                let a = DenseMultilinearExtension::random(num_vars, rng);
                let b = DenseMultilinearExtension::random(num_vars, rng);
                add_assign_ab(&a, &b);
                (Rc::new(a), Rc::new(b))
            })
            .collect::<Vec<_>>();
        SumHadamardInstance {
            num_vars,
            num_products: num_prods,
            products,
            result: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars, result,
            )),
        }
    }

    pub fn from(trace: &SumHadamardTraceMLE<F>) -> Vec<Self> {
        let num_vars = trace.log_coeff_count + trace.log_num_poly;
        let num_sum = 2;
        let mut vec_sum = Vec::with_capacity(num_sum);

        let mut add_into_batch = |batch: &SumHadamardTraceMLE<F>| {
            let products = batch
                .vec_hadamard
                .iter()
                .map(|trace| (trace.bit.ntt.clone(), trace.rlwe.ntt.0.clone()))
                .collect::<Vec<_>>();
            vec_sum.push(SumHadamardInstance {
                num_vars,
                num_products: products.len(),
                products,
                result: batch.sum_prod.ntt.0.clone(),
            });

            let products = batch
                .vec_hadamard
                .iter()
                .map(|trace| (trace.bit.ntt.clone(), trace.rlwe.ntt.1.clone()))
                .collect::<Vec<_>>();
            vec_sum.push(SumHadamardInstance {
                num_vars,
                num_products: products.len(),
                products,
                result: batch.sum_prod.ntt.1.clone(),
            });
        };

        add_into_batch(trace);

        vec_sum
    }

    fn random_num<R: rand::Rng + rand::CryptoRng>(
        num_instance: usize,
        num_vars: usize,
        num_prods: usize,
        rng: &mut R,
    ) -> Vec<Self> {
        let mut vec_sum = Vec::with_capacity(num_instance);
        for _ in 0..num_instance {
            vec_sum.push(SumHadamardInstance::random(num_vars, num_prods, rng));
        }
        vec_sum
    }
}

impl<F: Field> SumHadamardEval<F> {
    pub fn compute_subclaim(&self, random_lambda: F, kernel_at_r: F) -> F {
        let mut sum = self
            .products_at_r
            .iter()
            .fold(F::zero(), |sum, &(a, b)| sum + a * b);
        sum -= self.result_at_r;
        sum * random_lambda * kernel_at_r
    }
}

impl<F: Field> BatchedSumHadamardProof<F> {
    pub fn from_hadamard_trace_eval(trace_eval: &SumHadamardTraceEval<F>) -> Self {
        let mut proof = BatchedSumHadamardProof {
            poly_info: PolynomialInfo::default(),
            sumcheck_proof: Proof::default(),
            hadamard_at_r: Vec::with_capacity(2),
        };
        proof.append_eval(trace_eval);
        proof
    }

    pub fn from_pbs_trace_eval(trace_eval: &PBSTraceEval<F>) -> Self {
        let mut proof = BatchedSumHadamardProof {
            poly_info: PolynomialInfo::default(),
            sumcheck_proof: Proof::default(),
            hadamard_at_r: Vec::with_capacity(4),
        };
        proof.append_eval(&trace_eval.hadamard_trace);
        proof.append_eval(&trace_eval.acc_trace.extract_hadamard_eval());
        proof
    }

    pub fn append_eval(&mut self, trace_eval: &SumHadamardTraceEval<F>) {
        let num_sum = 2;
        let mut hadamard_at_r: Vec<SumHadamardEval<F>> = Vec::with_capacity(num_sum);

        let mut add_into_batch = |batch: &SumHadamardTraceEval<F>| {
            let products_at_r = batch
                .vec_hadamard
                .iter()
                .map(|trace_eval| (trace_eval.bit.ntt, trace_eval.rlwe.ntt.0))
                .collect::<Vec<_>>();
            hadamard_at_r.push(SumHadamardEval {
                products_at_r,
                result_at_r: batch.sum_prod.ntt.0,
            });

            let products_at_r = batch
                .vec_hadamard
                .iter()
                .map(|trace_eval| (trace_eval.bit.ntt, trace_eval.rlwe.ntt.1))
                .collect::<Vec<_>>();
            hadamard_at_r.push(SumHadamardEval {
                products_at_r,
                result_at_r: batch.sum_prod.ntt.1,
            });
        };

        add_into_batch(trace_eval);

        self.hadamard_at_r.extend(hadamard_at_r);
    }

    // export evaluations in proof into TraceEval structure
    pub fn export_eval(
        &self,
        idx_start: usize,
        idx_end: usize,
        trace_eval: &mut SumHadamardTraceEval<F>,
    ) {
        let num_sum = 2;
        assert_eq!(idx_end - idx_start, num_sum);
        let [eval_0, eval_2] = &self.hadamard_at_r[idx_start..idx_end] else {
            panic!("Invalid index range for exporting hadamard eval");
        };
        trace_eval.sum_prod.ntt = (eval_0.result_at_r, eval_2.result_at_r);
        for (i, eval) in trace_eval.vec_hadamard.iter_mut().enumerate() {
            eval.bit.ntt = eval_0.products_at_r[i].0;
            eval.rlwe.ntt = (eval_0.products_at_r[i].1, eval_2.products_at_r[i].1);
        }
    }
}

impl<F: Field + Serialize> SumcheckInstance<F> for SumHadamardInstance<F> {
    type Info = SumHadamardInfo<F>;

    fn info(&self) -> Self::Info {
        SumHadamardInfo {
            num_vars: self.num_vars,
            num_products: self.num_products,
            _marker: Default::default(),
        }
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: Field> SumcheckInfo<F> for SumHadamardInfo<F> {
    fn num_sumchecks(&self) -> usize {
        1
    }
    fn sumcheck_num_vars(&self) -> usize {
        self.num_vars
    }
    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: Field> SumcheckPureProof<F> for SumHadamardProof<F> {
    fn from_sumcheck(sumcheck_claim: &SumcheckClaim<F>, proof: Proof<F>) -> Self {
        SumHadamardProof {
            poly_info: sumcheck_claim.poly.info(),
            sumcheck_proof: proof,
            hadamard_at_r: SumHadamardEval::default(),
        }
    }

    fn get_poly_info(&self) -> &PolynomialInfo {
        &self.poly_info
    }

    fn get_sumcheck_proof(&self) -> &Proof<F> {
        &self.sumcheck_proof
    }
}

impl<F: Field> SumcheckPureProof<F> for BatchedSumHadamardProof<F> {
    fn from_sumcheck(sumcheck_claim: &SumcheckClaim<F>, proof: Proof<F>) -> Self {
        BatchedSumHadamardProof {
            poly_info: sumcheck_claim.poly.info(),
            sumcheck_proof: proof,
            hadamard_at_r: Vec::default(),
        }
    }

    fn get_poly_info(&self) -> &PolynomialInfo {
        &self.poly_info
    }

    fn get_sumcheck_proof(&self) -> &Proof<F> {
        &self.sumcheck_proof
    }
}

impl<F: Field + Serialize> SumcheckPureBatchedProof<F> for BatchedSumHadamardProof<F> {
    type Instance = SumHadamardInstance<F>;
    type Info = SumHadamardInfo<F>;
    type ProverState = HadamardProverState<F>;

    fn append_evaluations(
        &mut self,
        instances: &[Self::Instance],
        prover_state: &Self::ProverState,
    ) {
        let lookup = |m: &Rc<DenseMultilinearExtension<F>>| {
            let m_ptr: *const DenseMultilinearExtension<F> = Rc::as_ptr(m);
            let index = prover_state.raw_pointers_lookup_table.get(&m_ptr).unwrap();
            prover_state.flattened_mle_evals[*index]
        };

        self.hadamard_at_r = instances
            .iter()
            .map(|instance| {
                let mut products_at_r = Vec::with_capacity(instance.num_products);
                for (a, b) in &instance.products {
                    products_at_r.push((lookup(a), lookup(b)));
                }
                SumHadamardEval {
                    products_at_r,
                    result_at_r: lookup(&instance.result),
                }
            })
            .collect::<Vec<_>>();
    }

    fn compute_subclaim(
        &self,
        _infos: &[Self::Info],
        subclaim: &mut SubClaim<F>,
        randomness: &Vec<Vec<F>>,
        kernel_at_r: Option<F>,
    ) {
        assert!(kernel_at_r.is_some());
        for (hadamard_eval, r) in izip!(&self.hadamard_at_r, randomness) {
            subclaim.expected_evaluations -=
                hadamard_eval.compute_subclaim(r[0], kernel_at_r.unwrap());
        }
    }
}

impl<F: Field> SumcheckPureProverState<F> for HadamardProverState<F> {
    // Computation Opmitization:
    // Flattened MLE evaluations in prover_state are tables of size 2,
    // which are the evluations of f(r_1, ..., r_n-1, X) for X in {0,1}.
    // To avoid repeated computations, prover can directly compute all
    // f(r_1, ..., r_n) from these tables.
    fn from_sumcheck(
        sumcheck_prover_state: prover::ProverState<F>,
        claim: SumcheckClaim<F>,
    ) -> Self {
        // f(r) = f(0) + r_n * (f(1) - f(0))
        let fast_compute = |mle: &DenseMultilinearExtension<F>| {
            mle.evaluations[0]
                + *sumcheck_prover_state.randomness.last().unwrap()
                    * (mle.evaluations[1] - mle.evaluations[0])
        };
        let flattened_mle_evals = sumcheck_prover_state
            .flattened_ml_extensions
            .iter()
            .map(fast_compute)
            .collect::<Vec<_>>();
        HadamardProverState {
            point_r: sumcheck_prover_state.randomness,
            flattened_mle_evals,
            raw_pointers_lookup_table: claim.poly.raw_pointers_lookup_table,
        }
    }
}

impl<F: Field> SumcheckPureSubclaim<F> for HadamardVerifierSubclaim<F> {
    fn from_sumcheck(sumcheck_subclaim: SubClaim<F>) -> Self {
        HadamardVerifierSubclaim {
            point_r: sumcheck_subclaim.point,
        }
    }
}

impl<F: Field + Serialize> SumcheckPIOP<F> for HadamardPIOP<F> {
    type Instance = SumHadamardInstance<F>;
    type Info = SumHadamardInfo<F>;
    type Proof = SumHadamardProof<F>;
    type ProverState = HadamardProverState<F>;
    type VerifierSubclaim = HadamardVerifierSubclaim<F>;

    fn prover(
        trans: &mut Transcript<F>,
        instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState) {
        let (mut proof, state) = Self::prover_without_evals(trans, instance);

        proof.hadamard_at_r = instance.eval_at_point(&state.point_r);

        // proof.hadamard_at_r = instance.from_flatten_mle_evals(&state);

        // let fast_eval = instance.from_flatten_mle_evals(&state);
        // assert_eq!(proof.hadamard_at_r.result_at_r, fast_eval.result_at_r);

        (proof, state)
    }

    fn prover_add_sumcheck(
        instance: &Self::Instance,
        claim: &mut SumcheckClaim<F>,
        randomness: &[F],
        lagrange_kernel: Option<&LagrangeKernel<F>>,
    ) -> Option<Self::ProverState> {
        assert!(lagrange_kernel.is_some());
        let kernel = lagrange_kernel.unwrap();

        instance.add_into_sumcheck(claim, randomness[0], kernel);
        None
    }

    fn verifier_compute_subclaim(
        _info: &Self::Info,
        proof: &Self::Proof,
        subclaim: &mut SubClaim<F>,
        randomness: &[F],
        kernel_at_r: Option<F>,
    ) {
        let kernel_at_r = kernel_at_r.unwrap();
        assert_eq!(randomness.len(), 1);
        subclaim.expected_evaluations -= proof
            .hadamard_at_r
            .compute_subclaim(randomness[0], kernel_at_r);
    }
}

impl<F: Field + Serialize> BatchedSumcheckPIOP<F> for HadamardPIOP<F> {
    type BatchedProof = BatchedSumHadamardProof<F>;
    type BatchedProverState = HadamardProverState<F>;
    type BatchedVerifierSubclaim = HadamardVerifierSubclaim<F>;
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{SumcheckInstance, SumcheckPIOP};
    use algebra::derive::{DecomposableField, FheField, Field, NTT, Prime};
    use helper::Transcript;

    #[derive(Field, DecomposableField, FheField, Prime, NTT)]
    #[modulus = 132120577]
    pub struct Fp32(u32);
    // field type
    type FF = Fp32;

    #[test]
    fn test_hadamard_piop() {
        let mut rng = &mut rand::rng();
        let num_vars = 10;
        let num_products = 4;

        let instance = SumHadamardInstance::<FF>::random(num_vars, num_products, &mut rng);
        let info = instance.info();

        let mut prover_transcript = Transcript::<FF>::new();
        let (proof, _) = HadamardPIOP::<FF>::prover(&mut prover_transcript, &instance);
        let mut verifier_transcript = Transcript::<FF>::new();
        let (res, _) = HadamardPIOP::<FF>::verifier(&mut verifier_transcript, &info, &proof);
        assert!(res);
    }

    #[test]
    fn test_hadamard_piop_batched() {
        let mut rng = &mut rand::rng();
        let num_vars = 10;
        let num_products = 4;
        let num_instances = 3;

        let instances =
            SumHadamardInstance::<FF>::random_num(num_instances, num_vars, num_products, &mut rng);
        let infos = instances.iter().map(|inst| inst.info()).collect::<Vec<_>>();

        let mut prover_transcript = Transcript::<FF>::new();
        let (proof, _) = HadamardPIOP::<FF>::prover_batch(&mut prover_transcript, &instances);
        let mut verifier_transcript = Transcript::<FF>::new();
        let (res, _) = HadamardPIOP::<FF>::verifier_batch(&mut verifier_transcript, &infos, &proof);
        assert!(res);
    }
}
