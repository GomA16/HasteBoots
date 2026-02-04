use std::rc::Rc;

use algebra::{DenseMultilinearExtension, Field, PolynomialInfo};
use serde::Serialize;
use sumcheck::Proof;
use trace::cmp_trace::eq_trace::EQTraceMLE;

use crate::{
    LagrangeKernel, SumcheckClaim, SumcheckInfo, SumcheckInstance, SumcheckPIOP,
    SumcheckProverState, SumcheckPureProof, SumcheckSubclaim,
};

pub struct GrandProdPIOP<F: Field> {
    _marker: std::marker::PhantomData<F>,
}

pub struct GrandProdInstance<F: Field> {
    pub num_vars: usize,
    pub products: Vec<Rc<DenseMultilinearExtension<F>>>,
    pub result: Rc<DenseMultilinearExtension<F>>,
}

#[derive(Serialize)]
pub struct GrandProdInfo<F: Field> {
    pub num_vars: usize,
    pub num_products: usize,
    _marker: std::marker::PhantomData<F>,
}

#[derive(Default, Serialize)]
pub struct GrandProdProof<F: Field> {
    pub poly_info: PolynomialInfo,
    pub sumcheck_proof: Proof<F>,
    pub products_at_r: Vec<F>,
    pub result_at_r: F,
}

impl<F: Field> GrandProdInstance<F> {
    pub fn add_into_sumcheck(
        &self,
        claim: &mut SumcheckClaim<F>,
        kernal: &LagrangeKernel<F>,
        random_lambda: F,
    ) {
        // let mut prod = Vec::with_capacity(self.products.len() + 1);
        let mut prod: Vec<_> = self.products.iter().cloned().collect();
        prod.push(Rc::clone(&kernal.eq_at_point));
        claim.poly.add_product(prod, random_lambda);

        claim.poly.add_product(
            vec![Rc::clone(&self.result), Rc::clone(&kernal.eq_at_point)],
            -random_lambda,
        );
    }

    pub fn from_eq_trace(trace: &EQTraceMLE<F>) -> Self {
        GrandProdInstance {
            num_vars: trace.num_vars,
            products: trace.bit_eq.clone(),
            result: Rc::clone(&trace.eq_result),
        }
    }
}

impl<F: Field> SumcheckInfo<F> for GrandProdInfo<F> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn num_sumchecks(&self) -> usize {
        1
    }

    fn sumcheck_num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: Field + Serialize> SumcheckInstance<F> for GrandProdInstance<F> {
    type Info = GrandProdInfo<F>;

    fn info(&self) -> Self::Info {
        GrandProdInfo {
            num_vars: self.num_vars,
            num_products: self.products.len(),
            _marker: std::marker::PhantomData,
        }
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: Field> SumcheckPureProof<F> for GrandProdProof<F> {
    fn from_sumcheck(sumcheck_claim: &SumcheckClaim<F>, proof: Proof<F>) -> Self {
        GrandProdProof {
            poly_info: sumcheck_claim.poly.info(),
            sumcheck_proof: proof,
            products_at_r: Vec::new(),
            result_at_r: F::zero(),
        }
    }

    fn get_poly_info(&self) -> &PolynomialInfo {
        &self.poly_info
    }

    fn get_sumcheck_proof(&self) -> &Proof<F> {
        &self.sumcheck_proof
    }
}

impl<F: Field + Serialize> SumcheckPIOP<F> for GrandProdPIOP<F> {
    type Instance = GrandProdInstance<F>;
    type Proof = GrandProdProof<F>;
    type Info = GrandProdInfo<F>;
    type ProverState = SumcheckProverState<F>;
    type VerifierSubclaim = SumcheckSubclaim<F>;

    fn prover(
        trans: &mut helper::Transcript<F>,
        instance: &Self::Instance,
    ) -> (Self::Proof, Self::ProverState) {
        let (mut proof, state) = Self::prover_without_evals(trans, instance);

        let lookup = |m: &Rc<DenseMultilinearExtension<F>>| {
            let m_ptr: *const DenseMultilinearExtension<F> = Rc::as_ptr(m);
            let index = state.raw_pointers_lookup_table.get(&m_ptr).unwrap();
            state.flattened_mle_evals[*index]
        };

        proof.products_at_r = instance.products.iter().map(lookup).collect();
        proof.result_at_r = lookup(&instance.result);
        (proof, state)
    }

    fn prover_add_sumcheck(
        instance: &Self::Instance,
        claim: &mut SumcheckClaim<F>,
        randomness: &[F],
        lagrange_kernel: Option<&LagrangeKernel<F>>,
    ) -> Option<Self::ProverState> {
        assert!(lagrange_kernel.is_some());
        let lagrange_kernel = lagrange_kernel.unwrap();
        instance.add_into_sumcheck(claim, lagrange_kernel, randomness[0]);
        None
    }

    fn verifier_compute_subclaim(
        _info: &Self::Info,
        proof: &Self::Proof,
        subclaim: &mut sumcheck::verifier::SubClaim<F>,
        randomness: &[F],
        kernel_at_r: Option<F>,
    ) {
        assert!(kernel_at_r.is_some());
        let kernel_at_r = kernel_at_r.unwrap();
        assert_eq!(randomness.len(), 1);

        subclaim.expected_evaluations -=
            proof.products_at_r.iter().fold(F::one(), |acc, x| acc * *x)
                * kernel_at_r
                * randomness[0];
        subclaim.expected_evaluations -= proof.result_at_r * kernel_at_r * -randomness[0];
    }
}

#[cfg(test)]
mod test {
    use algebra::{BabyBear, Basis};
    use helper::Transcript;
    use trace::cmp_trace::eq_trace::{EQTables, EQTrace};

    use super::*;
    use num_traits::One;
    // field type
    type FF = BabyBear;

    #[test]
    fn test_eq_piop() {
        let mut rng = rand::rng();
        let eq_constant = -FF::one();
        let basis = Basis::<FF>::new(7);
        let eq_tables = EQTables::new(eq_constant, &basis);
        let eq_trace = EQTrace::random(&mut rng, 4, &eq_tables);
        let eq_trace_mle: EQTraceMLE<_> = eq_trace.into();

        let instance = GrandProdInstance::from_eq_trace(&eq_trace_mle);
        let info = instance.info();
        let mut prover_trans = Transcript::default();
        let (proof, _prover_state) = GrandProdPIOP::<FF>::prover(&mut prover_trans, &instance);
        let mut verifier_trans = Transcript::default();
        let (res, _) = GrandProdPIOP::<FF>::verifier(&mut verifier_trans, &info, &proof);
        assert!(res);
    }
}
