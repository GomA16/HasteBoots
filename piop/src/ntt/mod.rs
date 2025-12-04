//! PIOP for NTT with delegation
//! The algorithm is derived from Chap3.1 in zkCNN: https://eprint.iacr.org/2021/673
//! The prover wants to convince that Number Theoretic Transform (NTT) algorithm.
//! NTT is widely used for the multiplication of two polynomials in field.
//!
//! The goal of this IOP is to prove:
//!
//! Given M instances of addition in Zq, the main idea of this IOP is to prove:
//! For y \in \{0, 1\}^N:
//!     $$a(y) = \sum_{x\in \{0, 1\}^{\log N} c(x)\cdot F(y, x) }$$
//! where c represents the coefficients of a degree-{N-1} polynomial and a represents the evaulations at (ω^1, ω^3, ..., ω^{2N-1}),
//!
//! Here ω is the primitive 2N-th root of unity such that ω^{2N} = 1.
//! F is the standard Fourier matrix with only 2N distinct values and F(y, x) = ω^{(2Y-1)X} where Y and X are the field representations for the binary representations y and x, respectively.
//!
//! The LHS and RHS of the above equation are both MLE for y, so it can be reduced to check at a random point due to Schwartz-Zippel Lemma.
//! The remaining thing is to prove $$a(u) = \sum_{x\in \{0, 1\}^{\log N} c(x)\cdot F(u, x) }$$ with the sumcheck protocol
//! where u is the random challenge from the verifier.
//!
//! Without delegation, the verifier needs to compute F(u, v) on his own using the same algorithm as the prover, which costs O(N).
//! In order to keep a succinct verifier, the computation of F(u, v) can be delegated to prover.
//!
//! We define $A_{F}^{(k)}:\{0,1\}^{k+1} -> \mathbb{F}$ and $ω^{(k)}_{i+1}:\{0,1\}^{k+1} -> \mathbb{F}$.
//! Note that k + i + 1= \log N.
//! In each round, the prover wants to prove, for all $x\in \{0,1\}^i$, b\in \{0,1\}:
//! A_{F}^{(k)}(x, b)=A_{F}^{(k-1)}(x) * (1-u_{i} + u_{i} * \ω^{(k)}_{i+1}(x, b)) * ω^{2^k * b}
//! where $\ω^{(k)}_{i+1}(x,b ) = \ω^{2^{i+1}\cdot j}$ for $j = X+2^{i+1}\cdot b$.
//! So, it is reduced to prove the the following sum = \tilde{A}_{F}^{(k)}(x, b) at a random point $(x, b)\in \mathbb{F}^{k+1}$:
//!     =\sum_{z\in \{0,1\}}^k
//!         \tilde{\beta}((x, b),(z,0)) * \tilde{A}_{F}^{(k-1)}(z) ( (1-u_{i})+u_{i} * \tilde{ω}^{(k)}_{i+1}(z, 0)
//!       + \tilde{\beta}((x, b),(z,1)) * \tilde{A}_{F}^{(k-1)}(z) ( (1-u_{i})+u_{i} * \tilde{ω}^{(k)}_{i+1}(z, 1) * ω^{2^k}

use sumcheck::{prover::ProverState, verifier::SubClaim, MLSumcheck, Proof};
use sumcheck::{ProofWrapper, SumcheckKit};
use helper::utils::{
    eval_identity_function, gen_identity_evaluations, print_statistic, verify_oracle_relation,
};
use algebra::{
    AbstractExtensionField, DenseMultilinearExtension, Field,
    ListOfProductsOfPolynomials, PolynomialInfo,
};
use helper::Transcript;
use core::fmt;
use std::sync::Arc;
use itertools::izip;
use pcs::{
    multilinear::brakedown::BrakedownPCS,
    utils::code::{LinearCode, LinearCodeSpec},
    utils::hash::Hash,
    PolynomialCommitmentScheme,
};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::rc::Rc;
use std::time::Instant;

use ntt_bare::NTTBareIOP;
use ntt_delegation::IntermediateMLEs;
use ntt_delegation::{init_fourier_table_with_mle, eval_w_power_times_x};
use trace::{NTTTraceMLE, NTTInstanceInfo};

pub mod ntt_bare;
pub mod ntt_delegation;
pub mod ntt_relation;
pub mod ntt_fourier_eval;
/// IOP for NTT, i.e. $$a(u) = \sum_{x\in \{0, 1\}^{\log N} c(x)\cdot F(u, x) }$$
pub struct NTTIOP<F: Field>(PhantomData<F>);
/// SNARKs for NTT compiled with PCS
pub struct NTTSnarks<F: Field, EF: AbstractExtensionField<F>>(PhantomData<F>, PhantomData<EF>);



/// All the proofs generated only in the recursive phase to prove F(u, v), which does not contain the ntt_bare_proof.
#[derive(Serialize)]
pub struct NTTRecursiveProof<F: Field> {
    /// sumcheck proof for $$a(u) = \sum_{x\in \{0, 1\}^{\log N} c(x)\cdot F(u, x) }$$
    /// collective sumcheck proofs for delegation
    pub delegation_sumcheck_msgs: Vec<Proof<F>>,
    /// collective claimed sums for delegation
    pub delegation_claimed_sums: Vec<F>,
    /// final claim
    pub final_claim: F,
}

impl<F: Field + Serialize> NTTIOP<F> {
    /// sample the random coins before proving sumcheck protocol
    pub fn sample_coins(trans: &mut Transcript<F>, num_ntt: usize) -> Vec<F> {
        trans.get_vec_challenge(
            b"randomness used to obtain the virtual random ntt instance",
            num_ntt,
        )
    }

    /// return the number of coins used in this IOP
    pub fn num_coins(info: &NTTInstanceInfo<F>) -> usize {
        info.num_ntt()
    }

    /// Prove NTT instance with delegation
    pub fn prove(instance: &NTTTraceMLE<F>) -> (SumcheckKit<F>, NTTRecursiveProof<F>) {
        let mut trans = Transcript::new();
        let u = trans.get_vec_challenge(
            b"random point used to instantiate sumcheck protocol",
            instance.num_vars(),
        );

        let mut poly = ListOfProductsOfPolynomials::<F>::new(instance.num_vars);
        let randomness = F::one();
        let mut claimed_sum = F::zero();
        <NTTBareIOP<F>>::prove_as_subprotocol(
            randomness,
            &mut poly,
            &mut claimed_sum,
            instance,
            &u,
        );

        let (proof, state) =
            MLSumcheck::prove(&mut trans, &poly).expect("fail to prove the sumcheck protocol");

        // prove F(u, v) in a recursive manner
        let recursive_proof =
            <NTTIOP<F>>::prove_recursive(&mut trans, &state.randomness, &instance.info(), &u);

        (
            SumcheckKit {
                proof,
                claimed_sum,
                info: poly.info(),
                u,
                randomness: state.randomness,
            },
            recursive_proof,
        )
    }

    /// Verify NTT instance with delegation
    pub fn verify(
        wrapper: &mut ProofWrapper<F>,
        evals_at_r: F,
        evals_at_u: F,
        info: &NTTInstanceInfo<F>,
        recursive_proof: &NTTRecursiveProof<F>,
    ) -> bool {
        let mut trans = Transcript::new();

        let u = trans.get_vec_challenge(
            b"random point used to instantiate sumcheck protocol",
            info.num_vars,
        );

        let randomness = F::one();

        let mut subclaim = MLSumcheck::verify(
            &mut trans,
            &wrapper.info,
            wrapper.claimed_sum,
            &wrapper.proof,
        )
        .expect("fail to verify the sumcheck protocol");

        let f_delegation = recursive_proof.delegation_claimed_sums[0];
        if !<NTTBareIOP<F>>::verify_as_subprotocol(
            randomness,
            &mut subclaim,
            &mut wrapper.claimed_sum,
            evals_at_r,
            evals_at_u,
            f_delegation,
        ) {
            return false;
        }

        if !(subclaim.expected_evaluations == F::zero() && wrapper.claimed_sum == F::zero()) {
            return false;
        }

        <NTTIOP<F>>::verify_recursive(&mut trans, recursive_proof, info, &u, &subclaim)
    }

    /// The delegation of F(u, v) consists of logN - 1 rounds, each of which is a sumcheck protocol.
    ///
    /// We define $A_{F}^{(k)}:\{0,1\}^{k+1} -> \mathbb{F}$ and $ω^{(k)}_{i+1}:\{0,1\}^{k+1} -> \mathbb{F}$.
    /// The prover asserts the following sum = \tilde{A}_{F}^{(k)}(x, b) at a random point $(x, b)\in \mathbb{F}^{k+1}$:
    /// sum = \sum_{z\in \{0,1\}}^k
    ///         \tilde{\beta}((x, b),(z,0)) * \tilde{A}_{F}^{(k-1)}(z) ( (1-u_{i})+u_{i} * \tilde{ω}^{(k)}_{i+1}(z, 0)
    ///       + \tilde{\beta}((x, b),(z,1)) * \tilde{A}_{F}^{(k-1)}(z) ( (1-u_{i})+u_{i} * \tilde{ω}^{(k)}_{i+1}(z, 1) * ω^{2^k}
    /// where $\ω^{(k)}_{i+1}(x,b ) = \ω^{2^{i+1}\cdot j}$ for $j = X+2^{i+1}\cdot b$.
    ///
    /// In the term of the data structure, the polynomial to be sumed can be viewed as the sum of two products,
    /// one has coefficient one, and the other has coefficient ω^{2^k}.
    ///
    /// # Arguments
    /// * round: round number denoted by k, which is iterated in a reverse order as described in the algorithm
    /// * point: the random point $(x, b)\in \mathbb{F}^{k+1}$ reduced from the last sumcheck, used to prove the sum in the round
    /// * u_i: parameter in this round as described in the formula
    /// * w_coeff: the coefficient ω^{2^k} of the second product
    /// * f: MLE \tilde{A}_{F}^{(k-1)}(z) for z\in \{0,1\}^k
    /// * w: MLE \tilde{ω}^{(k)}_{i+1}(z, b) for z\in \{0,1\}^k  and b\in \{0, 1\}, which will be divided into two smaller MLEs \tilde{ω}^{(k)}_{i+1}(z, 0) and \tilde{ω}^{(k)}_{i+1}(z, 1)
    pub fn delegation_prover_round(
        trans: &mut Transcript<F>,
        round: usize,
        point: &[F],
        u_i: F,
        w_coeff: F,
        f: &Rc<DenseMultilinearExtension<F>>,
        w: &Rc<DenseMultilinearExtension<F>>,
    ) -> (Proof<F>, ProverState<F>) {
        assert_eq!(f.num_vars, round);
        assert_eq!(w.num_vars, round + 1);

        let mut poly = <ListOfProductsOfPolynomials<F>>::new(round);

        // the equality function defined by the random point $(x, b)\in \mathbb{F}^{k+1}$
        // it is divided into two MLEs \tilde{\beta}((x, b),(z,0)) and \tilde{\beta}((x, b),(z,1))
        let eq_func = gen_identity_evaluations(point);
        let (eq_func_left, eq_func_right) = eq_func.split_halves();

        // two divided MLEs: \tilde{ω}^{(k)}_{i+1}(z, 0) and \tilde{ω}^{(k)}_{i+1}(z, 1)
        let (w_left, w_right) = w.split_halves();

        // construct the polynomial to be sumed
        // left product is \tilde{\beta}((x, b),(z,0)) * \tilde{A}_{F}^{(k-1)}(z) ( (1-u_{i})+u_{i} * \tilde{ω}^{(k)}_{i+1}(z, 0)
        // right product is \tilde{\beta}((x, b),(z,1)) * \tilde{A}_{F}^{(k-1)}(z) ( (1-u_{i})+u_{i} * \tilde{ω}^{(k)}_{i+1}(z, 1) * ω^{2^k}
        poly.add_product_with_linear_op(
            [Rc::new(eq_func_left), Rc::clone(f), Rc::new(w_left)],
            &[
                (F::one(), F::zero()),
                (F::one(), F::zero()),
                (u_i, F::one() - u_i),
            ],
            F::one(),
        );

        poly.add_product_with_linear_op(
            [Rc::new(eq_func_right), Rc::clone(f), Rc::new(w_right)],
            &[
                (F::one(), F::zero()),
                (F::one(), F::zero()),
                (u_i, F::one() - u_i),
            ],
            w_coeff,
        );

        MLSumcheck::prove(trans, &poly).expect("ntt proof of delegation failed in round {round}")
    }

    /// Compared to the `prove` functionality, we just remove the phase to prove NTT bare.
    ///
    /// * `ntt_bare_state`: stores the prover state after proving the NTT bare
    pub fn prove_recursive(
        trans: &mut Transcript<F>,
        ntt_bare_randomness: &[F],
        info: &NTTInstanceInfo<F>,
        u: &[F],
    ) -> NTTRecursiveProof<F> {
        let log_n = info.num_vars();

        let (f_mles, w_mles) = init_fourier_table_with_mle(u, &info.ntt_table).into_mles();

        // 1. (detached) prove a(u) = \sum_{x\in \{0, 1\}^{\log N} c(x)\cdot F(u, x) } for a random point u

        // the above sumcheck is reduced to prove F(u, v) where v is the requested point
        // Note that the delegated value F(u, v) is stored in proof.delegation_claimed_sums[0].
        let mut requested_point = ntt_bare_randomness.to_owned();
        let mut reduced_claim = f_mles[log_n - 1].evaluate(&requested_point);

        // 2. prove the computation of F(u, v) in log_n - 1 rounds

        // store the sumcheck proof in each round
        let mut delegation_sumcheck_msgs = Vec::with_capacity(log_n - 1);
        // store the claimed sum of the sumcheck protocol in each round
        let mut delegation_claimed_sums = Vec::with_capacity(log_n - 1);
        // ! Modified Formula
        for k in (1..log_n).rev() {
            // start form log_n - 1;
            // let i = log_n - 1 - k;
            delegation_claimed_sums.push(reduced_claim);

            let w_coeff = info.ntt_table[1 << k];
            let f = &f_mles[k - 1];
            let (proof_round, state_round) = Self::delegation_prover_round(
                trans,
                k,
                &requested_point,
                // ! Modified Formula
                u[k],
                w_coeff,
                f,
                &w_mles[k],
            );
            delegation_sumcheck_msgs.push(proof_round);

            // the requested point returned from this round of sumcheck protocol, which initiates the claimed sum of the next round
            requested_point = state_round.randomness;
            reduced_claim = f.evaluate(&requested_point);
        }

        NTTRecursiveProof {
            delegation_sumcheck_msgs,
            delegation_claimed_sums,
            final_claim: reduced_claim,
        }
    }

    /// The delegation of F(u, v) consists of logN - 1 rounds, each of which is a sumcheck protocol.
    ///
    /// We define $A_{F}^{(k)}:\{0,1\}^{k+1} -> \mathbb{F}$ and $ω^{(k)}_{i+1}:\{0,1\}^{k+1} -> \mathbb{F}$.
    /// The prover asserts the following sum = \tilde{A}_{F}^{(k)}(x, b) at a random point $(x, b)\in \mathbb{F}^{k+1}$:
    /// sum = \sum_{z\in \{0,1\}}^k
    ///         \tilde{\beta}((x, b),(z,0)) * \tilde{A}_{F}^{(k-1)}(z) ( (1-u_{i})+u_{i} * \tilde{ω}^{(k)}_{i+1}(z, 0)
    ///       + \tilde{\beta}((x, b),(z,1)) * \tilde{A}_{F}^{(k-1)}(z) ( (1-u_{i})+u_{i} * \tilde{ω}^{(k)}_{i+1}(z, 1) * ω^{2^k}
    /// where $\ω^{(k)}_{i+1}(x,b ) = \ω^{2^{i+1}\cdot j}$ for $j = X+2^{i+1}\cdot b$.
    ///
    /// The verify needs to check the equality of the evaluation of the polynomial to be summed at a random point z = r \in \{0,1\}}^k.
    /// In verification, the verifier is given the evaluation of \tilde{A}_{F}^{(k-1)}(z = r) instead of computing on his own, so he can use it to check.
    /// If the equality holds, it is reduced to check the evaluation of \tilde{A}_{F}^{(k-1)}(z = r).
    ///
    /// # Arguments
    /// * round: round number denoted by k, which is iterated in a reverse order as described in the algorithm
    /// * x_b_point: the random point $(x, b)\in \mathbb{F}^{k+1}$ reduced from the last sumcheck
    /// * u_i: parameter in this round as described in the formula
    /// * subclaim: the subclaim returned from this round of the sumcheck, containing the random point r used for equality check
    /// * reduced_claim: the given evaluation of \tilde{A}_{F}^{(k-1)}(z = r) so verify does not need to compute on his own
    pub fn delegation_verify_round(
        round: usize,
        x_b_point: &[F],
        u_i: F,
        subclaim: &SubClaim<F>,
        reduced_claim: F,
        ntt_instance_info: &NTTInstanceInfo<F>,
    ) -> bool {
        let log_n = ntt_instance_info.num_vars();
        let ntt_table = &ntt_instance_info.ntt_table;

        // r_left = (r, 0) and r_right = (r, 0)
        let mut r_left: Vec<_> = Vec::with_capacity(round + 1);
        let mut r_right: Vec<_> = Vec::with_capacity(round + 1);
        r_left.extend(&subclaim.point);
        r_right.extend(&subclaim.point);
        r_left.push(F::zero());
        r_right.push(F::one());

        // compute $\ω^{(k)}_{i+1}(x,b ) = \ω^{2^{i+1}\cdot j}$ for $j = X+2^{i+1}\cdot b$ at point (r, 0) and (r, 1)
        // exp: i + 1 = n - k
        // let exp = log_n - round;
        // ! Modified Formula
        let sub = round + 1;
        // w_left = \tilde{ω}^{(k)}_{i+1}(r, 0) and w_right = \tilde{ω}^{(k)}_{i+1}(r, 0)
        let w_left = eval_w_power_times_x(ntt_table, log_n + 1, round + 1, sub, &r_left);
        let w_right = eval_w_power_times_x(ntt_table, log_n + 1, round + 1, sub, &r_right);

        let eval = eval_identity_function(x_b_point, &r_left)
            * reduced_claim
            * (F::one() - u_i + u_i * w_left)
            + eval_identity_function(x_b_point, &r_right)
                * reduced_claim
                * (F::one() - u_i + u_i * w_right)
                * ntt_table[1 << round];

        eval == subclaim.expected_evaluations
    }

    /// Compared to the `prove` functionality, we remove the phase to prove NTT bare.
    /// Also, after detaching the verification of NTT bare, verifier can directly check the recursive proofs.
    pub fn verify_recursive(
        trans: &mut Transcript<F>,
        proof: &NTTRecursiveProof<F>,
        info: &NTTInstanceInfo<F>,
        u: &[F],
        subclaim: &SubClaim<F>,
    ) -> bool {
        let log_n = info.num_vars();
        assert_eq!(proof.delegation_sumcheck_msgs.len(), log_n - 1);
        assert_eq!(proof.delegation_claimed_sums.len(), log_n - 1);

        // 1. [detached] verify a(u) = \sum_{x\in \{0, 1\}^{\log N} c(x)\cdot F(u, x) } for a random point u
        // Note that the delegated value F(u, v) is stored in proof.delegation_claimed_sums[0].

        // 2. verify the computation of F(u, v) in log_n - 1 rounds
        let mut requested_point = subclaim.point.clone();
        for (cnt, k) in (1..log_n).rev().enumerate() {
            // let i = log_n - 1 - k;

            // verify the proof of the sumcheck protocol
            let poly_info = PolynomialInfo {
                max_multiplicands: 3,
                num_variables: k,
            };
            let subclaim = MLSumcheck::verify(
                trans,
                &poly_info,
                proof.delegation_claimed_sums[cnt],
                &proof.delegation_sumcheck_msgs[cnt],
            )
            .expect("ntt verification failed in round {cnt}");

            // In the last round of the sumcheck protocol, the verify needs to check the equality of the evaluation of the polynomial to be summed at a random point z = r \in \{0,1\}}^k.
            // The verifier is given the evaluation of \tilde{A}_{F}^{(k-1)}(z = r) instead of computing on his own, so he can use it to check.
            // If the equality holds, it is reduced to check the evaluation of \tilde{A}_{F}^{(k-1)}(z = r).
            let reduced_claim = if cnt < log_n - 2 {
                proof.delegation_claimed_sums[cnt + 1]
            } else {
                proof.final_claim
            };
            // check the equality
            if !Self::delegation_verify_round(
                k,
                &requested_point,
                // ! Modified Formula
                u[k],
                &subclaim,
                reduced_claim,
                info,
            ) {
                panic!("ntt verification failed in round {cnt}");
            }
            requested_point = subclaim.point;
        }

        let delegation_final_claim = proof.final_claim;
        let final_point = requested_point;
        // TODO: handle the case that log = 1
        assert_eq!(final_point.len(), 1);

        // check the final claim returned from the last round of delegation
        // ! Modified Formula
        let idx = 1 << (info.num_vars());
        let eval = eval_identity_function(&final_point, &[F::zero()])
            + eval_identity_function(&final_point, &[F::one()])
                * (F::one() - u[0] + u[0] * info.ntt_table[idx])
                * info.ntt_table[1];

        delegation_final_claim == eval
    }
}

impl<F, EF> NTTSnarks<F, EF>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
{
    /// Generate and check snarks
    pub fn snarks<H, C, S>(instance: &NTTInstances<F>, code_spec: &S)
    where
        H: Hash + Sync + Send,
        C: LinearCode<F> + Serialize + for<'de> Deserialize<'de>,
        S: LinearCodeSpec<F, Code = C> + Clone,
    {
        let instance_info = instance.info();
        println!("Prove {instance_info}\n");
        // This is the actual polynomial to be committed for prover, which consists of all the required small polynomials in the IOP and padded zero polynomials.
        let committed_poly = instance.generate_oracle();

        // 1. Use PCS to commit the above polynomial.
        let start = Instant::now();
        let pp =
            BrakedownPCS::<F, H, C, S, EF>::setup(committed_poly.num_vars, Some(code_spec.clone()));
        let setup_time = start.elapsed().as_millis();

        let start = Instant::now();
        let (comm, comm_state) = BrakedownPCS::<F, H, C, S, EF>::commit(&pp, &committed_poly);
        let commit_time = start.elapsed().as_millis();

        // 2. Prover generates the proof
        let prover_start = Instant::now();
        let mut iop_proof_size = 0;
        let mut prover_trans = Transcript::<EF>::new();

        // 2.1 Generate the random point to instantiate the sumcheck protocol
        let prover_u = prover_trans.get_vec_challenge(
            b"random point used to instantiate sumcheck protocol",
            instance.num_vars,
        );

        // 2.? [one more step] Prover generate the random ntt instance from all instances to be proved
        let prover_r = <NTTIOP<EF>>::sample_coins(&mut prover_trans, instance_info.num_ntt);
        let instance_ef = instance.extract_ntt_instance_to_ef::<EF>(&prover_r);
        let instance_ef_info = instance_ef.info();

        // 2.2 Construct the polynomial and the claimed sum to be proved in the sumcheck protocol
        let mut sumcheck_poly = <ListOfProductsOfPolynomials<EF>>::new(instance.num_vars);
        let mut claimed_sum = EF::zero();
        <NTTBareIOP<EF>>::prove_as_subprotocol(
            EF::one(),
            &mut sumcheck_poly,
            &mut claimed_sum,
            &instance_ef,
            &prover_u,
        );

        let poly_info = sumcheck_poly.info();

        // 2.3 Generate proof of sumcheck protocol
        let (sumcheck_proof, sumcheck_state) =
            <MLSumcheck<EF>>::prove(&mut prover_trans, &sumcheck_poly)
                .expect("Proof generated in Addition In Zq");
        iop_proof_size += bincode::serialize(&sumcheck_proof).unwrap().len();

        // 2.? [one more step] Prover recursive prove the evaluation of F(u, v)
        let recursive_proof = <NTTIOP<EF>>::prove_recursive(
            &mut prover_trans,
            &sumcheck_state.randomness,
            &instance_ef_info,
            &prover_u,
        );
        iop_proof_size += bincode::serialize(&recursive_proof).unwrap().len();
        let iop_prover_time = prover_start.elapsed().as_millis();

        // 2.4 Compute all the evaluations of these small polynomials used in IOP over the random point returned from the sumcheck protocol
        let eq_at_r = gen_identity_evaluations(&sumcheck_state.randomness);
        let eq_at_u = gen_identity_evaluations(&prover_u);
        let coeff_evals_at_r = instance
            .coeffs
            .iter()
            // .map(|x| x.evaluate_ext(&sumcheck_state.randomness))
            .map(|x| x.evaluate_ext_opt(&eq_at_r))
            .collect::<Vec<_>>();
        let point_evals_at_u = instance
            .points
            .iter()
            // .map(|x| x.evaluate_ext(&prover_u))
            .map(|x| x.evaluate_ext_opt(&eq_at_u))
            .collect::<Vec<_>>();

        // 2.5 Reduce the proof of the above evaluations to a single random point over the committed polynomial
        let mut coeff_requested_point = sumcheck_state.randomness.clone();
        let mut point_requested_point = prover_u.clone();
        let oracle_randomness = prover_trans.get_vec_challenge(
            b"random linear combination for evaluations of oracles",
            instance.log_num_oracles(),
        );
        coeff_requested_point.extend(&oracle_randomness);
        point_requested_point.extend(&oracle_randomness);
        coeff_requested_point.push(EF::zero());
        point_requested_point.push(EF::one());

        let oracle_coeff_eval = committed_poly.evaluate_ext(&coeff_requested_point);
        let oracle_point_eval = committed_poly.evaluate_ext(&point_requested_point);

        // 2.6 Generate the evaluation proof of the requested points
        let start = Instant::now();
        // requested point [sumcheck_r, oracle_r, 0]
        let coeff_eval_proof = BrakedownPCS::<F, H, C, S, EF>::open(
            &pp,
            &comm,
            &comm_state,
            &coeff_requested_point,
            &mut prover_trans,
        );
        // requested point [prover_u, oracle_r, 1]
        let point_eval_proof = BrakedownPCS::<F, H, C, S, EF>::open(
            &pp,
            &comm,
            &comm_state,
            &point_requested_point,
            &mut prover_trans,
        );
        let pcs_open_time = start.elapsed().as_millis();

        // 3. Verifier checks the proof
        let verifier_start = Instant::now();
        let mut verifier_trans = Transcript::<EF>::new();

        // 3.1 Generate the random point to instantiate the sumcheck protocol
        let verifier_u = verifier_trans.get_vec_challenge(
            b"random point used to instantiate sumcheck protocol",
            instance_info.num_vars,
        );

        // 3.2 Verifier sample random coins to combine all sumcheck protocols proving ntt instances
        let verifier_r = verifier_trans.get_vec_challenge(
            b"randomness used to obtain the virtual random ntt instance",
            instance_info.num_ntt,
        );

        // 3.3 Check the proof of the sumcheck protocol
        let mut subclaim = <MLSumcheck<EF>>::verify(
            &mut verifier_trans,
            &poly_info,
            claimed_sum,
            &sumcheck_proof,
        )
        .expect("Verify the proof generated in NTT");

        // 3.4 Check the subclaim returned from the sumcheck protocol
        let f_delegation = recursive_proof.delegation_claimed_sums[0];
        let evals_at_r = verifier_r
            .iter()
            .zip(coeff_evals_at_r.iter())
            .fold(EF::zero(), |acc, (r, eval)| acc + *r * *eval);
        let evals_at_u = verifier_r
            .iter()
            .zip(point_evals_at_u.iter())
            .fold(EF::zero(), |acc, (r, eval)| acc + *r * *eval);

        let check_subclaim = <NTTBareIOP<EF>>::verify_as_subprotocol(
            EF::one(),
            &mut subclaim,
            &mut claimed_sum,
            evals_at_r,
            evals_at_u,
            f_delegation,
        );
        assert!(check_subclaim);
        assert_eq!(subclaim.expected_evaluations, EF::zero());
        assert_eq!(claimed_sum, EF::zero());
        // Check the delegation of F(u, v) used in the above check
        let check_recursive = <NTTIOP<EF>>::verify_recursive(
            &mut verifier_trans,
            &recursive_proof,
            &instance_ef_info,
            &verifier_u,
            &subclaim,
        );
        assert!(check_recursive);

        // 3.5 and also check the relation between these small oracles and the committed oracle
        let oracle_randomness = verifier_trans.get_vec_challenge(
            b"random linear combination for evaluations of oracles",
            instance.log_num_oracles(),
        );
        let check_oracle_coeff =
            verify_oracle_relation(&coeff_evals_at_r, oracle_coeff_eval, &oracle_randomness);
        let check_oracle_point =
            verify_oracle_relation(&point_evals_at_u, oracle_point_eval, &oracle_randomness);
        assert!(check_oracle_coeff);
        assert!(check_oracle_point);
        let iop_verifier_time = verifier_start.elapsed().as_millis();

        // 3.5 Check the evaluation of a random point over the committed oracle
        let start = Instant::now();
        let mut pcs_proof_size = 0;
        let check_pcs_coeff = BrakedownPCS::<F, H, C, S, EF>::verify(
            &pp,
            &comm,
            &coeff_requested_point,
            oracle_coeff_eval,
            &coeff_eval_proof,
            &mut verifier_trans,
        );
        let check_pcs_point = BrakedownPCS::<F, H, C, S, EF>::verify(
            &pp,
            &comm,
            &point_requested_point,
            oracle_point_eval,
            &point_eval_proof,
            &mut verifier_trans,
        );
        assert!(check_pcs_coeff);
        assert!(check_pcs_point);
        let pcs_verifier_time = start.elapsed().as_millis();
        pcs_proof_size += bincode::serialize(&coeff_eval_proof).unwrap().len()
            + bincode::serialize(&coeff_evals_at_r).unwrap().len()
            + bincode::serialize(&point_eval_proof).unwrap().len()
            + bincode::serialize(&point_evals_at_u).unwrap().len();

        // 4. print statistic
        print_statistic(
            iop_prover_time + pcs_open_time,
            iop_verifier_time + pcs_verifier_time,
            iop_proof_size + pcs_proof_size,
            iop_prover_time,
            iop_verifier_time,
            iop_proof_size,
            committed_poly.num_vars,
            instance.num_oracles(),
            instance.num_vars,
            setup_time,
            commit_time,
            pcs_open_time,
            pcs_verifier_time,
            pcs_proof_size,
        )
    }
}

