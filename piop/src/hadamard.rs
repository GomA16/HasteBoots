use algebra::DenseMultilinearExtension;
use algebra::Field;
use algebra::PolynomialInfo;
use bincode::de;
use helper::FiatShamirTranscript;
use helper::Transcript;
use helper::utils::eval_identity_function;
use rayon::vec;
use serde::Serialize;
use std::rc::Rc;
use sumcheck::MLSumcheck;
use sumcheck::Proof;
use sumcheck::prover;
use sumcheck::verifier::SubClaim;
use trace::BatchedHadamardTraceMLE;

use crate::LagrangeKernel;
use crate::SumcheckClaim;
use crate::SumcheckInfo;
use crate::SumcheckInstance;
use crate::SumcheckPIOP;
use crate::SumcheckPureProof;
use crate::SumcheckPureProverState;
use crate::SumcheckPureSubclaim;
use crate::lookup::LogUpProof;

pub struct HadamardPIOP<F: Field> {
    _marker: std::marker::PhantomData<F>,
}

pub struct SumOfHadamardInstance<F: Field> {
    pub num_vars: usize,
    pub num_products: usize,
    pub products: Vec<(
        Rc<DenseMultilinearExtension<F>>,
        Rc<DenseMultilinearExtension<F>>,
    )>,
    // result = \sum products[i][0] * products[i][1]
    pub result: Rc<DenseMultilinearExtension<F>>,
}

pub struct BatchedSumOfHadamardInstance<F: Field> {
    pub num_vars: usize,
    pub num_sum: usize,
    pub vec_sum: Vec<SumOfHadamardInstance<F>>,
}

pub struct HadamardProverState<F: Field> {
    pub point_r: Vec<F>,
}

pub struct HadamardVerifierSubclaim<F: Field> {
    pub point_r: Vec<F>,
}

#[derive(Serialize)]
pub struct BatchedSumOfHadamardInfo<F: Field> {
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

impl<F: Field> SumOfHadamardInstance<F> {
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

impl<F: Field> BatchedSumOfHadamardInstance<F> {
    pub fn from(batches: &[BatchedHadamardTraceMLE<F>]) -> Self {
        assert!(!batches.is_empty());
        let num_vars = batches[0].log_coeff_count + batches[0].log_coeff_count;
        let num_sum = batches.len() * 2;
        let mut vec_sum = Vec::with_capacity(num_sum);
        for batch in batches.iter() {
            debug_assert_eq!(batch.log_coeff_count + batch.log_coeff_count, num_vars);
            let products = batch
                .vec_trace
                .iter()
                .map(|trace| (trace.bit_poly.clone(), trace.key_ntt.0.clone()))
                .collect::<Vec<_>>();
            vec_sum.push(SumOfHadamardInstance {
                num_vars,
                num_products: products.len(),
                products,
                result: batch.sum_prod_ntt.0.clone(),
            });

            let products = batch
                .vec_trace
                .iter()
                .map(|trace| (trace.bit_poly.clone(), trace.key_ntt.1.clone()))
                .collect::<Vec<_>>();
            vec_sum.push(SumOfHadamardInstance {
                num_vars,
                num_products: products.len(),
                products,
                result: batch.sum_prod_ntt.1.clone(),
            });
        }

        BatchedSumOfHadamardInstance {
            num_vars,
            num_sum,
            vec_sum,
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

impl<F: Field + Serialize> SumcheckInstance<F> for BatchedSumOfHadamardInstance<F> {
    type Info = BatchedSumOfHadamardInfo<F>;

    fn info(&self) -> Self::Info {
        BatchedSumOfHadamardInfo {
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

impl<F: Field> SumcheckInfo<F> for BatchedSumOfHadamardInfo<F> {
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

impl<F: Field> SumcheckPureProof<F> for BatchedHadamardProof<F> {
    fn from_sumcheck(sumcheck_claim: &SumcheckClaim<F>, proof: Proof<F>) -> Self {
        BatchedHadamardProof {
            poly_info: sumcheck_claim.poly_ref().info(),
            sumcheck_proof: proof,
            hadamard_at_r: Vec::new(),
        }
    }

    fn poly_info(&self) -> &PolynomialInfo {
        &self.poly_info
    }

    fn sumcheck_proof(&self) -> &Proof<F> {
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
    type Instance = BatchedSumOfHadamardInstance<F>;
    type Info = BatchedSumOfHadamardInfo<F>;
    type Proof = BatchedHadamardProof<F>;
    type ProverState = HadamardProverState<F>;
    type VerifierSubclaim = HadamardVerifierSubclaim<F>;
    // type FSTranscript = Transcript<F>;

    fn prover(
        trans: &mut Transcript<F>,
        instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState) {
        let info = instance.info();
        trans.append_message(b"[Hadamard Statement]", &info);

        let mut sumcheck_claim = SumcheckClaim::new(info.num_sumchecks());

        let lagrange_kernel = Some(&LagrangeKernel::random(trans, instance.num_vars));
        let randomness_batch = info.sample_randomness_for_sumcheck(trans);
        Self::prover_batch_sumcheck(
            instance,
            &mut sumcheck_claim,
            &randomness_batch,
            lagrange_kernel,
        );
        let (sumcheck_proof, prover_state) = MLSumcheck::prove(trans, sumcheck_claim.poly_ref())
            .expect("[HadamardIOP] Fail to generate sumcheck proof");

        let hadamard_at_r = instance
            .vec_sum
            .iter()
            .map(|instance| instance.eval_at_point(&prover_state.randomness))
            .collect::<Vec<_>>();

        let proof = BatchedHadamardProof {
            poly_info: sumcheck_claim.poly_ref().info(),
            sumcheck_proof,
            hadamard_at_r,
        };
        let state = HadamardProverState {
            point_r: prover_state.randomness.clone(),
        };
        (proof, state)
    }

    fn verifier(
        trans: &mut Transcript<F>,
        info: &Self::Info,
        proof: &Self::Proof,
    ) -> (bool, Self::VerifierSubclaim) {
        trans.append_message(b"[Hadamard Statement]", &info);

        let mut res = true;
        let kernel_point = LagrangeKernel::random_point(trans, info.num_vars);
        let randomness_batch = info.sample_randomness_for_sumcheck(trans);

        let mut sumcheck_subclaim = MLSumcheck::verify(
            trans,
            &proof.poly_info,
            MLSumcheck::extract_sum(&proof.sumcheck_proof),
            &proof.sumcheck_proof,
        )
        .expect("[HadamardIOP - Verifier] Fail to verify the sumcheck");

        let kernel_at_r = eval_identity_function(&kernel_point, &sumcheck_subclaim.point);

        Self::verifier_compute_subclaim(
            info,
            proof,
            &mut sumcheck_subclaim,
            &randomness_batch,
            Some(kernel_at_r),
        );

        res &= sumcheck_subclaim.expected_evaluations.is_zero();

        let subclaim = Self::VerifierSubclaim::from_sumcheck(sumcheck_subclaim);
        (res, subclaim)
    }

    fn prover_without_evals(
        trans: &mut Transcript<F>,
        instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState) {
        let info = instance.info();
        trans.append_message(b"[Hadamard Statement]", &info);

        let mut sumcheck_claim = SumcheckClaim::new(info.num_sumchecks());

        let lagrange_kernel = Some(&LagrangeKernel::random(trans, instance.num_vars));
        let randomness_batch = info.sample_randomness_for_sumcheck(trans);
        Self::prover_batch_sumcheck(
            instance,
            &mut sumcheck_claim,
            &randomness_batch,
            lagrange_kernel,
        );
        let (sumcheck_proof, prover_state) = MLSumcheck::prove(trans, sumcheck_claim.poly_ref())
            .expect("[HadamardIOP] Fail to generate sumcheck proof");

        let proof = Self::Proof::from_sumcheck(&sumcheck_claim, sumcheck_proof);
        let state = Self::ProverState::from_sumcheck(prover_state);
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
