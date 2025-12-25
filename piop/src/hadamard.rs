use algebra::DenseMultilinearExtension;
use algebra::Field;
use algebra::PolynomialInfo;
use helper::FiatShamirTranscript;
use helper::Transcript;
use serde::Serialize;
use std::rc::Rc;
use sumcheck::MLSumcheck;
use sumcheck::Proof;
use sumcheck::prover;
use sumcheck::verifier::SubClaim;
use trace::SumHadamardTraceEval;
use trace::SumHadamardTraceMLE;

use crate::LagrangeKernel;
use crate::SumcheckClaim;
use crate::SumcheckInfo;
use crate::SumcheckInstance;
use crate::SumcheckPIOP;
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

pub struct BatchedSumHadamardInstance<F: Field> {
    pub num_vars: usize,
    pub num_sum: usize,
    pub vec_sum: Vec<SumHadamardInstance<F>>,
}

pub struct HadamardProverState<F: Field> {
    pub point_r: Vec<F>,
}

pub struct HadamardVerifierSubclaim<F: Field> {
    pub point_r: Vec<F>,
}

#[derive(Serialize)]
pub struct BatchedSumHadamardInfo<F: Field> {
    pub num_vars: usize,
    pub num_batch: usize,
    pub num_products_each: usize,
    _marker: std::marker::PhantomData<F>,
}

#[derive(Serialize)]
pub struct SumHadamardEval<F: Field> {
    pub products_at_r: Vec<(F, F)>,
    pub result_at_r: F,
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
}

impl<F: Field> BatchedSumHadamardInstance<F> {
    pub fn from(trace: &SumHadamardTraceMLE<F>) -> Self {
        let num_vars = trace.log_coeff_count + trace.log_num_round;
        let num_sum = 2;
        let mut vec_sum = Vec::with_capacity(num_sum);

        let mut add_into_batch = |batch: &SumHadamardTraceMLE<F>| {
            let products = batch
                .vec_trace
                .iter()
                .map(|trace| (trace.poly.ntt.clone(), trace.rlwe.ntt.0.clone()))
                .collect::<Vec<_>>();
            vec_sum.push(SumHadamardInstance {
                num_vars,
                num_products: products.len(),
                products,
                result: batch.sum_prod.ntt.0.clone(),
            });

            let products = batch
                .vec_trace
                .iter()
                .map(|trace| (trace.poly.ntt.clone(), trace.rlwe.ntt.1.clone()))
                .collect::<Vec<_>>();
            vec_sum.push(SumHadamardInstance {
                num_vars,
                num_products: products.len(),
                products,
                result: batch.sum_prod.ntt.1.clone(),
            });
        };

        add_into_batch(trace);

        BatchedSumHadamardInstance {
            num_vars,
            num_sum,
            vec_sum,
        }
    }

    fn random<R: rand::Rng + rand::CryptoRng>(
        num_sum: usize,
        num_vars: usize,
        num_prods: usize,
        rng: &mut R,
    ) -> Self {
        let mut vec_sum = Vec::with_capacity(num_sum);
        for _ in 0..num_sum {
            vec_sum.push(SumHadamardInstance::random(num_vars, num_prods, rng));
        }
        BatchedSumHadamardInstance {
            num_vars,
            num_sum,
            vec_sum,
        }
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
            hadamard_at_r: Vec::new(),
        };
        proof.append_eval(trace_eval);
        proof
    }

    pub fn append_eval(&mut self, trace_eval: &SumHadamardTraceEval<F>) {
        let num_sum = 2;
        let mut hadamard_at_r: Vec<SumHadamardEval<F>> = Vec::with_capacity(num_sum);

        let mut add_into_batch = |batch: &SumHadamardTraceEval<F>| {
            let products_at_r = batch
                .vec_trace
                .iter()
                .map(|trace_eval| (trace_eval.poly_eval.ntt, trace_eval.rlwe_eval.ntt.0))
                .collect::<Vec<_>>();
            hadamard_at_r.push(SumHadamardEval {
                products_at_r,
                result_at_r: batch.sum_prod.ntt.0,
            });

            let products_at_r = batch
                .vec_trace
                .iter()
                .map(|trace_eval| (trace_eval.poly_eval.ntt, trace_eval.rlwe_eval.ntt.1))
                .collect::<Vec<_>>();
            hadamard_at_r.push(SumHadamardEval {
                products_at_r,
                result_at_r: batch.sum_prod.ntt.1,
            });
        };

        add_into_batch(trace_eval);

        self.hadamard_at_r = hadamard_at_r;
    }
}

impl<F: Field + Serialize> SumcheckInstance<F> for BatchedSumHadamardInstance<F> {
    type Info = BatchedSumHadamardInfo<F>;

    fn info(&self) -> Self::Info {
        BatchedSumHadamardInfo {
            num_vars: self.num_vars,
            num_batch: self.num_sum,
            num_products_each: self.vec_sum[0].num_products,
            _marker: Default::default(),
        }
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: Field> SumcheckInfo<F> for BatchedSumHadamardInfo<F> {
    fn num_sumchecks(&self) -> usize {
        self.num_batch
    }
    fn sumcheck_num_vars(&self) -> usize {
        self.num_vars
    }
    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: Field> SumcheckPureProof<F> for BatchedSumHadamardProof<F> {
    fn from_sumcheck(sumcheck_claim: &SumcheckClaim<F>, proof: Proof<F>) -> Self {
        BatchedSumHadamardProof {
            poly_info: sumcheck_claim.poly.info(),
            sumcheck_proof: proof,
            hadamard_at_r: Vec::new(),
        }
    }

    fn get_poly_info(&self) -> &PolynomialInfo {
        &self.poly_info
    }

    fn get_sumcheck_proof(&self) -> &Proof<F> {
        &self.sumcheck_proof
    }
}

impl<F: Field> SumcheckPureProverState<F> for HadamardProverState<F> {
    fn from_sumcheck(sumcheck_prover_state: prover::ProverState<F>) -> Self {
        HadamardProverState {
            point_r: sumcheck_prover_state.randomness,
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
    type Instance = BatchedSumHadamardInstance<F>;
    type Info = BatchedSumHadamardInfo<F>;
    type Proof = BatchedSumHadamardProof<F>;
    type ProverState = HadamardProverState<F>;
    type VerifierSubclaim = HadamardVerifierSubclaim<F>;
    // type FSTranscript = Transcript<F>;

    fn prover(
        trans: &mut Transcript<F>,
        instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState) {
        let info = instance.info();
        trans.append_message(b"[Statement]", &info);

        let mut sumcheck_claim = SumcheckClaim::new(info.num_vars());

        let lagrange_kernel = Some(&LagrangeKernel::random(trans, instance.num_vars));
        let randomness_batch = info.sample_randomness_for_sumcheck(trans);
        Self::prover_batch_sumcheck(
            instance,
            &mut sumcheck_claim,
            &randomness_batch,
            lagrange_kernel,
        );
        let (sumcheck_proof, prover_state) = MLSumcheck::prove(trans, &sumcheck_claim.poly)
            .expect("[HadamardIOP] Fail to generate sumcheck proof");

        let hadamard_at_r = instance
            .vec_sum
            .iter()
            .map(|instance| instance.eval_at_point(&prover_state.randomness))
            .collect::<Vec<_>>();

        let proof = BatchedSumHadamardProof {
            poly_info: sumcheck_claim.poly.info(),
            sumcheck_proof,
            hadamard_at_r,
        };
        let state = HadamardProverState {
            point_r: prover_state.randomness.clone(),
        };
        (proof, state)
    }

    fn prover_batch_sumcheck(
        instance: &Self::Instance,
        claim: &mut SumcheckClaim<F>,
        randomness: &[F],
        lagrange_kernel: Option<&LagrangeKernel<F>>,
    ) -> Option<Self::ProverState> {
        assert!(lagrange_kernel.is_some());
        let kernel = lagrange_kernel.unwrap();

        for (instance, &r) in instance.vec_sum.iter().zip(randomness.iter()) {
            instance.add_into_sumcheck(claim, r, kernel);
        }
        None
    }

    fn verifier_compute_subclaim(
        info: &Self::Info,
        proof: &Self::Proof,
        subclaim: &mut SubClaim<F>,
        randomness: &[F],
        kernel_at_r: Option<F>,
    ) {
        assert_eq!(info.num_batch, proof.hadamard_at_r.len());
        let kernel_at_r = kernel_at_r.unwrap();
        for (proof, &r) in proof.hadamard_at_r.iter().zip(randomness.iter()) {
            subclaim.expected_evaluations -= proof.compute_subclaim(r, kernel_at_r);
        }
    }
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
        let num_sum = 2;

        let instance =
            BatchedSumHadamardInstance::<FF>::random(num_sum, num_vars, num_products, &mut rng);
        let info = instance.info();

        let mut prover_transcript = Transcript::<FF>::new();
        let (proof, _) = HadamardPIOP::<FF>::prover(&mut prover_transcript, &instance);
        let mut verifier_transcript = Transcript::<FF>::new();
        let (res, _) = HadamardPIOP::<FF>::verifier(&mut verifier_transcript, &info, &proof);
        assert!(res);
    }
}
