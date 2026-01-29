use algebra::{DenseMultilinearExtension, Field, PolynomialInfo};
use helper::Transcript;
use serde::Serialize;
use std::rc::Rc;
use sumcheck::{Proof, prover::ProverState, verifier::SubClaim};
use trace::basic_ops::SumHadamardTraceEval;
use trace::lookup_trace::small_table::{
    LookupTraceEval, LookupTraceMLE, LookupWitnessHelper, LookupWitnessHelperEval,
};

use crate::{
    LagrangeKernel, SumcheckClaim, SumcheckInfo, SumcheckInstance, SumcheckPIOP,
    SumcheckProverStateTrait, SumcheckPureProof, SumcheckSubclaimTrait,
};

pub struct LogUpIOP<F: Field> {
    _marker: std::marker::PhantomData<F>,
}

pub struct LogUpInstance<F: Field> {
    pub num_vars: usize,
    pub trace: LookupTraceMLE<F>,
    pub sum: F,
    pub helper: LookupWitnessHelper<F>,
}

#[derive(Serialize)]
pub struct LogUpInstanceInfo<F: Field> {
    pub num_vars: usize,
    pub block_size: usize,
    pub num_blocks: usize,
    pub num_columns: usize,
    #[serde(skip)]
    // random value used before sumcheck protocol
    pub random_value: F,
}

#[derive(Serialize)]
pub struct LogUpProof<F: Field> {
    pub poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<F>,
    #[serde(skip)]
    // input_at_r can be derived from phi_at_r and random_value
    pub input_at_r: Vec<F>,
    pub phi_at_r: Vec<F>,
    pub helper_at_r: Vec<F>,
}

pub struct LogUpProverState<F: Field> {
    pub point_r: Vec<F>,
}

pub struct LogUpVerifierSubclaim<F: Field> {
    pub point_r: Vec<F>,
}

impl<F: Field> LogUpInstance<F> {
    pub fn from(trace: &LookupTraceMLE<F>, helper: &LookupWitnessHelper<F>) -> Self {
        Self {
            num_vars: trace.num_vars,
            trace: trace.clone(),
            sum: helper.sum,
            helper: helper.clone(),
        }
    }

    // \sum h_1 + ... + h_k = sum
    pub fn add_helper_sum_into_sumcheck(&self, claim: &mut SumcheckClaim<F>, random_lambda: F) {
        for helper in self.helper.helper_functions.iter() {
            claim
                .poly
                .add_product(vec![Rc::clone(helper)], random_lambda);
        }
        claim.sum += random_lambda * self.sum;

        // let add_assign = |acc: &mut [F], vec: &[F]| {
        //     for (a, b) in acc.iter_mut().zip(vec.iter()) {
        //         *a += *b;
        //     }
        // };

        // let h_sum = self.helper.helper_functions.iter().fold(
        //     vec![F::zero(); 1 << self.num_vars],
        //     |mut acc, hf| {
        //         add_assign(&mut acc, hf.as_slice());
        //         acc
        //     },
        // );
        // let h_sum = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
        //     self.num_vars,
        //     h_sum,
        // ));

        // #[cfg(test)]
        // {
        //     let sum = h_sum.iter().fold(F::zero(), |acc, &x| acc + x);
        //     debug_assert_eq!(sum, self.sum);
        //     println!("Sanity check passed for helper sum function: h_sum");
        // }

        // claim.poly.add_product(vec![h_sum], random_lambda);
        // claim.sum += random_lambda * self.sum;
    }

    pub fn add_helper_identity_into_sumcheck(
        &self,
        idx: usize,
        claim: &mut SumcheckClaim<F>,
        random_lambda: F,
        kernel: &LagrangeKernel<F>,
    ) {
        assert!(self.helper.block_size >= 1);

        let idx_start = idx * self.helper.block_size;
        let idx_end = ((idx + 1) * self.helper.block_size).min(self.helper.phi_functions.len());

        // check h = \sum m_i / phi_i
        #[cfg(test)]
        {
            self.helper.helper_functions[idx]
                .iter()
                .enumerate()
                .for_each(|(x, h)| {
                    let mut sum = F::zero();
                    for i in idx_start..idx_end {
                        let m = F::one();
                        sum += m / self.helper.phi_functions[i].evaluations[x];
                    }
                    assert_eq!(*h, sum);
                });
            println!("Sanity check passed for helper identity function: h{}", idx);
        }

        let blk_len = idx_end - idx_start;

        let phi_block = self.helper.phi_functions[idx_start..idx_end]
            .iter()
            .map(|phi| Rc::clone(phi))
            .collect::<Vec<_>>();

        // randomness * L * h * \prod phi_i
        let mut prod: Vec<Rc<DenseMultilinearExtension<F>>> = Vec::with_capacity(blk_len + 2);
        prod.extend_from_slice(&phi_block);
        prod.push(Rc::clone(&kernel.eq_at_point));
        prod.push(Rc::clone(&self.helper.helper_functions[idx]));
        claim.poly.add_product(prod, random_lambda);

        for off_idx in 0..blk_len {
            // - randomness * L * \prod phi_i / phi_{off_idx}
            let mut prod_i = Vec::with_capacity(blk_len);
            prod_i.extend_from_slice(&phi_block);
            prod_i[off_idx] = Rc::clone(&kernel.eq_at_point);
            claim.poly.add_product(prod_i, -random_lambda);
        }
    }
}

impl<F: Field + Serialize> SumcheckInstance<F> for LogUpInstance<F> {
    type Info = LogUpInstanceInfo<F>;

    fn info(&self) -> Self::Info {
        LogUpInstanceInfo::<F> {
            num_vars: self.num_vars,
            block_size: self.helper.block_size,
            num_blocks: self.helper.num_blocks,
            num_columns: self.trace.vec_input.len(),
            random_value: self.helper.random_value,
        }
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: Field> SumcheckInfo<F> for LogUpInstanceInfo<F> {
    fn sumcheck_num_vars(&self) -> usize {
        self.num_vars
    }

    fn num_sumchecks(&self) -> usize {
        self.num_blocks + 1
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: Field> SumcheckPureProof<F> for LogUpProof<F> {
    fn from_sumcheck(sumcheck_claim: &SumcheckClaim<F>, proof: Proof<F>) -> Self {
        LogUpProof {
            poly_info: sumcheck_claim.poly.info(),
            sumcheck_proof: proof,
            input_at_r: Vec::new(),
            phi_at_r: Vec::new(),
            // witness_at_r: (F::zero(), F::zero()),
            helper_at_r: Vec::new(),
        }
    }

    fn get_poly_info(&self) -> &PolynomialInfo {
        &self.poly_info
    }

    fn get_sumcheck_proof(&self) -> &Proof<F> {
        &self.sumcheck_proof
    }
}

impl<F: Field> SumcheckProverStateTrait<F> for LogUpProverState<F> {
    fn from_sumcheck(sumcheck_prover_state: ProverState<F>, _claim: SumcheckClaim<F>) -> Self {
        LogUpProverState {
            point_r: sumcheck_prover_state.randomness,
        }
    }
}

impl<F: Field> SumcheckSubclaimTrait<F> for LogUpVerifierSubclaim<F> {
    fn from_sumcheck(sumcheck_subclaim: sumcheck::verifier::SubClaim<F>) -> Self {
        LogUpVerifierSubclaim {
            point_r: sumcheck_subclaim.point,
        }
    }
}

impl<F: Field + Serialize> SumcheckPIOP<F> for LogUpIOP<F> {
    type Instance = LogUpInstance<F>;
    type Info = LogUpInstanceInfo<F>;
    type Proof = LogUpProof<F>;
    type ProverState = LogUpProverState<F>;
    type VerifierSubclaim = LogUpVerifierSubclaim<F>;

    fn prover(
        trans: &mut Transcript<F>,
        instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState) {
        let (mut proof, state) = Self::prover_without_evals(trans, instance);

        let mle_eval = |mle: &[Rc<DenseMultilinearExtension<F>>]| {
            mle.iter()
                .map(|m| m.evaluate(&state.point_r))
                .collect::<Vec<_>>()
        };

        proof.phi_at_r = mle_eval(&instance.helper.phi_functions);
        proof.input_at_r = proof.phi_at_r[1..]
            .iter()
            .map(|x| *x - instance.helper.random_value)
            .collect::<Vec<_>>();
        proof.helper_at_r = mle_eval(&instance.helper.helper_functions);
        (proof, state)
    }

    fn prover_add_sumcheck(
        instance: &Self::Instance,
        claim: &mut crate::SumcheckClaim<F>,
        randomness: &[F],
        kernel: Option<&LagrangeKernel<F>>,
    ) -> Option<Self::ProverState> {
        assert!(kernel.is_some());
        let kernel = kernel.unwrap();

        instance.add_helper_sum_into_sumcheck(claim, randomness[0]);
        randomness[1..].iter().enumerate().for_each(|(idx, &r)| {
            instance.add_helper_identity_into_sumcheck(idx, claim, r, kernel);
        });

        None
    }

    fn verifier_compute_subclaim(
        info: &Self::Info,
        proof: &Self::Proof,
        subclaim: &mut SubClaim<F>,
        randomness: &[F],
        kernel_at_r: Option<F>,
    ) {
        let kernerl_at_r = kernel_at_r.unwrap();
        subclaim.expected_evaluations -= proof.compute_helper_subclaim(randomness[0]);
        randomness[1..].iter().enumerate().for_each(|(idx, &r)| {
            subclaim.expected_evaluations -=
                proof.compute_helper_identity_subcliam(info, idx, r, kernerl_at_r)
        });
    }
}

impl<F: Field> LogUpProof<F> {
    pub fn from_hadamard_trace_eval(
        trace_eval: &SumHadamardTraceEval<F>,
        helper_eval: &LookupWitnessHelperEval<F>,
        random_value: F,
    ) -> Self {
        let phi_at_r = trace_eval
            .vec_hadamard
            .iter()
            .map(|x| x.bit.poly + random_value)
            .collect::<Vec<_>>();
        let proof = LogUpProof {
            poly_info: PolynomialInfo::default(),
            sumcheck_proof: Proof::default(),
            input_at_r: Vec::default(),
            phi_at_r,
            helper_at_r: helper_eval.helper_functions_at_r.clone(),
        };
        proof
    }

    pub fn compute_helper_subclaim(&self, random_lambda: F) -> F {
        random_lambda * (self.helper_at_r.iter().fold(F::zero(), |acc, &x| acc + x))
    }

    pub fn compute_helper_identity_subcliam(
        &self,
        info: &LogUpInstanceInfo<F>,
        idx: usize,
        random_lambda: F,
        kernel_at_r: F,
    ) -> F {
        assert!(info.block_size >= 1);
        let idx_start = idx * info.block_size;
        let idx_end = ((idx + 1) * info.block_size).min(self.phi_at_r.len());

        let blk_len = idx_end - idx_start;
        let phi_block = &self.phi_at_r[idx_start..idx_end];

        let grand_prod = phi_block.iter().fold(F::one(), |acc, &x| acc * x);
        let mut sum = F::zero();
        sum += self.helper_at_r[idx] * grand_prod;

        for off_idx in 0..blk_len {
            sum -= grand_prod / phi_block[off_idx];
        }

        sum * random_lambda * kernel_at_r
    }

    pub fn append_eval(
        &mut self,
        trace_eval: &LookupTraceEval<F>,
        helper_eval: &LookupWitnessHelperEval<F>,
        random_value: F,
    ) {
        let phi_at_r = trace_eval
            .vec_input_at_r
            .iter()
            .map(|&x| x + random_value)
            .collect::<Vec<_>>();

        self.input_at_r = trace_eval.vec_input_at_r.clone();
        self.phi_at_r = phi_at_r;
        self.helper_at_r = helper_eval.helper_functions_at_r.clone();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{SumcheckInstance, SumcheckPIOP};
    use algebra::{
        FieldUniformSampler,
        derive::{DecomposableField, FheField, Field, NTT, Prime},
    };
    use helper::Transcript;
    use rand_distr::Distribution;
    use trace::lookup_trace::small_table::LookupTrace;

    #[derive(Field, DecomposableField, FheField, Prime, NTT)]
    #[modulus = 132120577]
    pub struct Fp32(u32);
    // field type
    type FF = Fp32;

    #[test]
    fn test_logup_iop() {
        let mut rng = rand::rng();
        let num_vars = 3;
        let num_vec = 2;
        let range = 8;
        let blk_size = 2;

        let lookup_trace = LookupTrace::<FF>::random(&mut rng, num_vars, num_vec, range);
        let lookup_mle: LookupTraceMLE<FF> = lookup_trace.into();
        let lookup_witness = lookup_mle.compute_witness_pure();

        let uniform = FieldUniformSampler::<FF>::new();
        let random_value = uniform.sample(&mut rng);
        let lookup_helper: LookupWitnessHelper<FF> =
            lookup_mle.compute_helper_functions(blk_size, random_value, &lookup_witness);

        let instance = LogUpInstance::<FF>::from(&lookup_mle, &lookup_helper);
        let info = instance.info();

        let mut prover_transcript = Transcript::<FF>::new();
        let (proof, _) = LogUpIOP::<FF>::prover(&mut prover_transcript, &instance);
        let mut verifier_transcript = Transcript::<FF>::new();
        let (res, _) = LogUpIOP::<FF>::verifier(&mut verifier_transcript, &info, &proof);
        assert!(res);
    }
}
