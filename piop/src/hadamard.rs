use algebra::DenseMultilinearExtension;
use algebra::Field;
use algebra::PolynomialInfo;
use bincode::de;
use helper::FiatShamirTranscript;
use helper::Transcript;
use rayon::vec;
use serde::Serialize;
use std::rc::Rc;
use sumcheck::Proof;

use crate::LagrangeKernel;
use crate::SumcheckClaim;
use crate::SumcheckInfo;
use crate::SumcheckInstance;
use crate::SumcheckPIOP;

pub struct HadamardPIOP<F: Field> {
    _marker: std::marker::PhantomData<F>,
}

pub struct HadamardInstance<F: Field> {
    pub num_vars: usize,
    pub num_products: usize,
    pub products: Vec<(
        Rc<DenseMultilinearExtension<F>>,
        Rc<DenseMultilinearExtension<F>>,
    )>,
    // result = \sum products[i][0] * products[i][1]
    pub result: Rc<DenseMultilinearExtension<F>>,
}

pub struct BatchedHadamardInstance<F: Field> {
    pub num_vars: usize,
    pub num_batch: usize,
    pub vec_batch: Vec<HadamardInstance<F>>,
}

pub struct HadamardProverState<F: Field> {
    pub point_r: Vec<F>,
}

use HadamardProverState as HadamardVerifierSubclaim;

#[derive(Serialize)]
pub struct BatchedHadamardInfo<F: Field> {
    pub num_vars: usize,
    pub num_batch: usize,
    pub num_products_each: usize,
    _marker: std::marker::PhantomData<F>,
}

pub struct HadamardProof<F: Field> {
    pub products_at_r: Vec<(F, F)>,
    pub result_at_r: F,
}

pub struct BatchedHadamardProof<F: Field> {
    pub poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<F>,
    pub hadamard_at_r: Vec<HadamardProof<F>>,
}

impl<F: Field> HadamardInstance<F> {
    pub fn add_into_sumcheck(
        &self,
        claim: &mut SumcheckClaim<F>,
        random_lambda: F,
        kernel: &LagrangeKernel<F>,
    ) {
        for (a, b) in &self.products {
            let prod = vec![a.clone(), b.clone(), kernel.eq_at_point.clone()];
            claim.poly_mut().add_product(prod, random_lambda);
        }
        let prod = vec![self.result.clone(), kernel.eq_at_point.clone()];
        claim.poly_mut().add_product(prod, -random_lambda);
    }

    pub fn eval_at_point(&self, point_r: &[F]) -> HadamardProof<F> {
        let mut products_at_r = Vec::with_capacity(self.num_products);
        for (a, b) in &self.products {
            products_at_r.push((a.evaluate(point_r), b.evaluate(point_r)));
        }
        HadamardProof {
            products_at_r,
            result_at_r: self.result.evaluate(point_r),
        }
    }
}

impl<F: Field> HadamardProof<F> {
    pub fn compute_subclaim(&self, random_lambda: F, kernel_at_r: F) -> F {
        let mut sum = self
            .products_at_r
            .iter()
            .fold(F::zero(), |sum, &(a, b)| sum + a * b);
        sum -= self.result_at_r;
        sum * random_lambda * kernel_at_r
    }
}

impl<F: Field> SumcheckInstance<F> for BatchedHadamardInstance<F> {
    type Info = BatchedHadamardInfo<F>;

    fn info(&self) -> Self::Info {
        BatchedHadamardInfo {
            num_vars: self.num_vars,
            num_batch: self.num_batch,
            num_products_each: self.vec_batch[0].num_products,
            _marker: Default::default(),
        }
    }
}

impl<F: Field> SumcheckInfo<F> for BatchedHadamardInfo<F> {
    fn num_sumchecks(&self) -> usize {
        self.num_batch
    }
    fn sumcheck_num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: Field + Serialize> SumcheckPIOP<F> for HadamardPIOP<F> {
    type Instance = BatchedHadamardInstance<F>;
    type Info = BatchedHadamardInfo<F>;
    type Proof = BatchedHadamardProof<F>;
    type ProverState = HadamardProverState<F>;
    type VerifierSubclaim = HadamardVerifierSubclaim<F>;
    type FSTranscript = Transcript<F>;

    fn prover(
        trans: &mut Self::FSTranscript,
        instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState) {
        let info = instance.info();
        trans.append_message(b"[Hadamard Statement]", &info);

        let sumcheck_claim = SumcheckClaim::new(info.num_sumchecks());

        let lagrange_kernel = Some(&LagrangeKernel::random(trans, instance.num_vars));
        let randomness_batch = Self::sample_randomness_for_sumcheck(&info, trans);
        Self::prover_batch_sumcheck(
            instance,
            &mut sumcheck_claim,
            &randomness_batch,
            lagrange_kernel,
        );

        // let 
    }

    fn prover_batch_sumcheck(
        instance: &Self::Instance,
        claim: &mut SumcheckClaim<F>,
        randomness: &[F],
        lagrange_kernel: Option<&LagrangeKernel<F>>,
    ) -> Option<Self::ProverState> {
        assert!(lagrange_kernel.is_some());
        let kernel = lagrange_kernel.unwrap();

        for (instance, &r) in instance.vec_batch.iter().zip(randomness.iter()) {
            instance.add_into_sumcheck(claim, r, kernel);
        }
        None
    }

    fn verifier_compute_subclaim(
        info: &Self::Info,
        proof: &Self::Proof,
        subclaim: &mut sumcheck::verifier::SubClaim<F>,
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
