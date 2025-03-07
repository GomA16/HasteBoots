//! Lift
use super::bit_decomposition::BitDecomposition;
use super::bit_decomposition::DecomposedBitsEval;
use super::ntt_revision::ntt_bare::NTTBareIOP;
use super::ntt_revision::NTTRecursiveProof;
use super::ntt_revision::NTTIOP;
use super::ntt_revision::{NTTInstance, NTTInstanceInfo};
use super::sparse_eval::SparseEvalIOP;
use super::sparse_eval::SparseEvalInstance;
use super::sparse_eval::SparseEvalInstanceEval;
use super::LookupInstance;
use super::{DecomposedBits, DecomposedBitsInfo};
use crate::piop::LookupIOP;
use crate::sumcheck::verifier::SubClaim;
use crate::sumcheck::MLSumcheck;
use crate::sumcheck::ProofWrapper;
use crate::sumcheck::SumcheckKit;
use crate::utils::{
    add_assign_ef, eval_identity_function, gen_identity_evaluations, print_statistic,
    verify_oracle_relation,
};
use algebra::DecomposableField;
use algebra::{
    utils::Transcript, AbstractExtensionField, DenseMultilinearExtension, Field,
    ListOfProductsOfPolynomials,
};
use core::fmt;
use std::time::Instant;
use itertools::izip;
use pcs::{
    multilinear::brakedown::BrakedownPCS,
    utils::code::{LinearCode, LinearCodeSpec},
    utils::hash::Hash,
    PolynomialCommitmentScheme,
};
use rand::random;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use std::vec;

/// IOP for Lift
pub struct LiftIOP<F: Field>(PhantomData<F>);

/// SNARKs for Lift compiled with PCS
pub struct LiftSnarks<F: Field, EF: AbstractExtensionField<F>>(PhantomData<F>, PhantomData<EF>);

/// Lift Instance
pub struct LiftInstance<F: Field> {
    /// number of variables = dimension of the input
    pub num_vars: usize,
    /// dimension of RWLE is denoted by N
    pub log_N: usize,
    /// modulus of the input
    pub N: F,
    /// input s in Zq of length M
    pub input: Rc<DenseMultilinearExtension<F>>,

    /// witness k and r such that input = k * N + row where k\in \{0, 1\}
    /// witness val = 1 - 2k
    pub k: Rc<DenseMultilinearExtension<F>>,
    /// witness row
    pub row: Rc<DenseMultilinearExtension<F>>,
    /// (not committed) coefficient matrix C = (c_0, ..., c_{M-1})^T \in F^{N * N}
    pub coeff_cols: Vec<Rc<DenseMultilinearExtension<F>>>,

    /// outpus are composed of (a_0, ...., a_{M-1})^T such that a_i = NTT(c_i)
    pub outputs: Vec<Rc<DenseMultilinearExtension<F>>>,

    /// info for NTT
    pub ntt_info: NTTInstanceInfo<F>,
}

/// Info
pub struct LiftInfo<F: Field> {
    /// number of variables
    pub num_vars: usize,
    /// dimension of RWLE is denoted by N
    pub log_N: usize,
    /// modulus of the input
    pub N: F,

    /// info for NTT
    pub ntt_info: NTTInstanceInfo<F>,
}

impl<F: Field> fmt::Display for LiftInfo<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "An instance of Lift: #vars = {}",
            self.num_vars,
        )?;
        writeln!(f, "- containing 1 single randomized NTT")?;
        writeln!(f, "- containing 1 sparse matrix evaluation")
    }
}

/// Lift Evaluations
pub struct LiftEval<F: Field> {
    /// input s
    pub input: F,
    /// witness
    pub k: F,
    /// witness
    pub row: F,
    /// coeff_cols
    pub coeff_cols: Vec<F>,
    /// outputs
    pub outputs: Vec<F>,
}

impl<F: Field> LiftInstance<F> {
    /// Construct an instance
    pub fn new(
        num_vars: usize,
        log_N: usize,
        N: F,
        input: Rc<DenseMultilinearExtension<F>>,
        k: Rc<DenseMultilinearExtension<F>>,
        row: Rc<DenseMultilinearExtension<F>>,
        coeff_cols: Vec<Rc<DenseMultilinearExtension<F>>>,
        outputs: Vec<Rc<DenseMultilinearExtension<F>>>,
        ntt_info: NTTInstanceInfo<F>,
    ) -> LiftInstance<F> {
        // update num_ntt of ntt_info
        assert_eq!(coeff_cols.len(), outputs.len());
        let ntt_info = NTTInstanceInfo {
            num_ntt: outputs.len(),
            num_vars,
            ntt_table: ntt_info.ntt_table.clone(),
        };
        Self {
            num_vars,
            log_N,
            N,
            input: Rc::clone(&input),
            k: Rc::clone(&k),
            row: Rc::clone(&row),
            coeff_cols,
            outputs,
            ntt_info,
        }
    }

    /// extract info
    #[inline]
    pub fn info(&self) -> LiftInfo<F> {
        LiftInfo {
            num_vars: self.num_vars,
            N: self.N,
            log_N: self.log_N,
            ntt_info: self.ntt_info.clone(),
        }
    }

    /// number of small polynomials sent in iop
    #[inline]
    pub fn num_oracles(&self) -> usize {
        3 + self.outputs.len()
    }

    /// Return the log of the number of small polynomials used in IOP
    #[inline]
    pub fn log_num_oracles(&self) -> usize {
        self.num_oracles().next_power_of_two().ilog2() as usize
    }

    /// return the number of ntt instances contained
    #[inline]
    pub fn num_ntt_contained(&self) -> usize {
        self.ntt_info.num_ntt
    }

    /// Pack all the involved small polynomials into a single vector
    pub fn pack_all_mles(&self) -> Vec<F> {
        self.input
            .iter()
            .chain(self.k.iter())
            .chain(self.row.iter())
            .chain(self.outputs.iter().flat_map(|output| output.iter()))
            .copied()
            .collect::<Vec<F>>()
    }

    /// Generate the oracle to be committed that is composed of all the small oracles used in IOP.
    pub fn generate_oracle(&self) -> DenseMultilinearExtension<F> {
        let num_vars_added = self.log_num_oracles();
        let num_vars = self.num_vars + num_vars_added;
        let num_zeros_padded = ((1 << num_vars_added) - self.num_oracles()) * (1 << self.num_vars);

        // arrangement: all values||all decomposed bits||padded zeros
        let mut evals = self.pack_all_mles();
        evals.append(&mut vec![F::zero(); num_zeros_padded]);
        <DenseMultilinearExtension<F>>::from_evaluations_vec(num_vars, evals)
    }

    /// to instance over extension field
    pub fn to_ef<EF: AbstractExtensionField<F>>(&self) -> LiftInstance<EF> {
        LiftInstance::<EF> {
            num_vars: self.num_vars,
            N: EF::from_base(self.N),
            log_N: self.log_N,
            input: Rc::new(self.input.to_ef()),
            k: Rc::new(self.k.to_ef()),
            row: Rc::new(self.row.to_ef()),
            coeff_cols: self
                .coeff_cols
                .iter()
                .map(|col| Rc::new(col.to_ef()))
                .collect(),
            outputs: self
                .outputs
                .iter()
                .map(|output| Rc::new(output.to_ef()))
                .collect(),
            ntt_info: self.ntt_info.to_ef(),
        }
    }

    /// Evaluate at the same random point
    #[inline]
    pub fn evaluate(&self, point: &[F]) -> LiftEval<F> {
        LiftEval {
            input: self.input.evaluate(point),
            k: self.k.evaluate(point),
            row: self.row.evaluate(point),
            coeff_cols: self
                .coeff_cols
                .iter()
                .map(|col| col.evaluate(point))
                .collect(),
            outputs: self
                .outputs
                .iter()
                .map(|output| output.evaluate(point))
                .collect(),
        }
    }

    /// Evaluate at the same random point
    #[inline]
    pub fn evaluate_ext<EF: AbstractExtensionField<F>>(&self, point: &[EF]) -> LiftEval<EF> {
        LiftEval {
            input: self.input.evaluate_ext(point),
            k: self.k.evaluate_ext(point),
            row: self.row.evaluate_ext(point),
            coeff_cols: self
                .coeff_cols
                .iter()
                .map(|col| col.evaluate_ext(point))
                .collect(),
            outputs: self
                .outputs
                .iter()
                .map(|output| output.evaluate_ext(point))
                .collect(),
        }
    }

    /// extract ntt with Field randomness
    pub fn extract_ntt_instance(&self, randomness: &[F]) -> NTTInstance<F> {
        assert_eq!(randomness.len(), self.coeff_cols.len());
        let mut random_coeffs = <DenseMultilinearExtension<F>>::from_evaluations_vec(
            self.log_N,
            vec![F::zero(); 1 << self.log_N],
        );
        let mut random_points = <DenseMultilinearExtension<F>>::from_evaluations_vec(
            self.log_N,
            vec![F::zero(); 1 << self.log_N],
        );

        for (r, coeff, eval) in izip!(randomness, self.coeff_cols.iter(), self.outputs.iter()) {
            random_coeffs += (*r, coeff.as_ref());
            random_points += (*r, eval.as_ref());
        }

        NTTInstance::<F> {
            num_vars: self.log_N,
            ntt_table: self.ntt_info.ntt_table.clone(),
            coeffs: Rc::new(random_coeffs),
            points: Rc::new(random_points),
        }
    }

    /// extract ntt with Extension Field randomness
    pub fn extract_ntt_instance_to_ef<EF: AbstractExtensionField<F>>(
        &self,
        randomness: &[EF],
    ) -> NTTInstance<EF> {
        assert_eq!(randomness.len(), self.coeff_cols.len());
        let mut random_coeffs = <DenseMultilinearExtension<EF>>::from_evaluations_vec(
            self.log_N,
            vec![EF::zero(); 1 << self.log_N],
        );
        let mut random_points = <DenseMultilinearExtension<EF>>::from_evaluations_vec(
            self.log_N,
            vec![EF::zero(); 1 << self.log_N],
        );

        for (r, coeff, eval) in izip!(randomness, self.coeff_cols.iter(), self.outputs.iter()) {
            add_assign_ef(&mut random_coeffs, r, coeff.as_ref());
            add_assign_ef(&mut random_points, r, eval.as_ref());
        }

        NTTInstance::<EF> {
            num_vars: self.log_N,
            ntt_table: Arc::new(
                self.ntt_info
                    .ntt_table
                    .iter()
                    .map(|x| EF::from_base(*x))
                    .collect::<Vec<EF>>(),
            ),
            coeffs: Rc::new(random_coeffs),
            points: Rc::new(random_points),
        }
    }

    /// Extract the sparse evaluation
    pub fn extract_sparse_eval_instance(
        &self,
        r_y: &[F],
        evals_at_rx: &LiftEval<F>,
    ) -> (SparseEvalInstance<F>, F) {
        let coeff_mle =
            DenseMultilinearExtension::from_evaluations_slice(r_y.len(), &evals_at_rx.coeff_cols);
        let F_two = F::one() + F::one();
        let val = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            self.num_vars,
            self.k.iter().map(|_k| F::one() - F_two * *_k).collect(),
        ));
        (
            SparseEvalInstance {
                num_x_vars: self.log_N,
                num_y_vars: self.num_vars,
                row: self.row.clone(),
                val,
                eval_rx: Default::default(),
                table: Default::default(),
            },
            coeff_mle.evaluate(r_y),
        )
    }
}

impl<F: Field> LiftEval<F> {
    /// number of small polynomials sent in iop
    #[inline]
    pub fn num_oracles(&self) -> usize {
        3 + self.outputs.len()
    }

    /// Return the log of the number of small polynomials used in IOP
    #[inline]
    pub fn log_num_oracles(&self) -> usize {
        self.num_oracles().next_power_of_two().ilog2() as usize
    }

    /// Flatten all evals into a vector with the same arrangement of the committed polynomial
    #[inline]
    pub fn flatten(&self) -> Vec<F> {
        let mut res = Vec::with_capacity(self.num_oracles());
        res.push(self.input);
        res.push(self.k);
        res.push(self.row);
        // no need for packing coeffs_col
        res.extend(self.outputs.iter());
        res
    }

    /// Extract the NTT-Coefficient evaluation
    #[inline]
    pub fn update_ntt_instance_coeff(&self, r_coeff: &mut F, randomness: &[F]) {
        assert_eq!(randomness.len(), self.coeff_cols.len());
        *r_coeff += self
            .coeff_cols
            .iter()
            .zip(randomness)
            .fold(F::zero(), |acc, (coeff, r)| acc + *r * *coeff);
    }

    /// Extract the NTT-Coefficient evaluation
    #[inline]
    pub fn update_ntt_instance_point(&self, r_point: &mut F, randomness: &[F]) {
        assert_eq!(randomness.len(), self.outputs.len());
        *r_point += self
            .outputs
            .iter()
            .zip(randomness)
            .fold(F::zero(), |acc, (coeff, r)| acc + *r * *coeff);
    }
}

impl<F: Field + Serialize> LiftIOP<F> {
    /// sample coins before proving sumcheck protocol
    #[warn(unused_variables)]
    pub fn sample_coins(trans: &mut Transcript<F>, instance: &LiftInstance<F>) -> Vec<F> {
        trans.get_vec_challenge(b"randomness to combine sumcheck protocols", 1)
    }

    /// return the number of coins used in sumcheck protocol
    #[warn(unused_variables)]
    pub fn num_coins(info: &LiftInfo<F>) -> usize {
        1
    }

    /// sample coins for sparse pcs
    pub fn prover_sample_sparse_randomness(trans: &mut Transcript<F>, instance: &LiftInstance<F>) -> Vec<F> {
        trans.get_vec_challenge(b"randomness for sparse eval", instance.num_vars)
    }

    /// sample coins for sparse pcs
    pub fn verifier_sample_sparse_randomness(trans: &mut Transcript<F>, info: &LiftInfo<F>) -> Vec<F> {
        trans.get_vec_challenge(b"randomness for sparse eval", info.num_vars)
    }

    /// Prove aas subprotocol
    pub fn prove_as_subprotocol(
        randomness: &[F],
        poly: &mut ListOfProductsOfPolynomials<F>,
        instance: &LiftInstance<F>,
        eq_at_u: &Rc<DenseMultilinearExtension<F>>,
    ) {
        assert_eq!(randomness.len(), 1);
        poly.add_product_with_linear_op(
            [
                Rc::clone(eq_at_u),
                Rc::clone(&instance.k),
                Rc::clone(&instance.k),
            ],
            &[
                (F::one(), F::zero()),
                (F::one(), F::zero()),
                (-F::one(), F::one()),
            ],
            randomness[0],
        );
    }

    /// prove
    pub fn prove(instance: &LiftInstance<F>) -> (SumcheckKit<F>, NTTRecursiveProof<F>) {
        let mut trans = Transcript::<F>::new();
        let u = trans.get_vec_challenge(
            b"random point used to instantiate sumcheck protocol",
            instance.num_vars,
        );
        let eq_at_u = Rc::new(gen_identity_evaluations(&u));
        let randomness = Self::sample_coins(&mut trans, instance);
        let randomness_ntt = <NTTIOP<F>>::sample_coins(&mut trans, instance.num_ntt_contained());

        let mut poly = ListOfProductsOfPolynomials::<F>::new(instance.num_vars);
        let mut claimed_sum = F::zero();
        // add sumcheck products (without NTT) into poly
        Self::prove_as_subprotocol(&randomness, &mut poly, instance, &eq_at_u);

        // add sumcheck products of NTT into poly
        let ntt_instance = instance.extract_ntt_instance(&randomness_ntt);
        <NTTBareIOP<F>>::prove_as_subprotocol(
            F::one(),
            &mut poly,
            &mut claimed_sum,
            &ntt_instance,
            &u,
        );

        // prove all sumcheck protocol into a large random sumcheck
        let (proof, state) =
            MLSumcheck::prove(&mut trans, &poly).expect("fail to prove the sumcheck protocol");

        // prove F(u, v) in a recursive manner
        let recursive_proof =
            <NTTIOP<F>>::prove_recursive(&mut trans, &state.randomness, &ntt_instance.info(), &u);

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

    /// Verifier
    #[inline]
    pub fn verify(
        wrapper: &mut ProofWrapper<F>,
        evals_at_r: &LiftEval<F>,
        evals_at_u: &LiftEval<F>,
        info: &LiftInfo<F>,
        recursive_proof: &NTTRecursiveProof<F>,
    ) -> bool {
        let mut trans = Transcript::new();

        let u = trans.get_vec_challenge(
            b"random point used to instantiate sumcheck protocol",
            info.num_vars,
        );

        // randomness to combine sumcheck protocols
        let randomness = trans.get_vec_challenge(
            b"randomness to combine sumcheck protocols",
            Self::num_coins(info),
        );
        let randomness_ntt = trans.get_vec_challenge(
            b"randomness used to obtain the virtual random ntt instance",
            <NTTIOP<F>>::num_coins(&info.ntt_info),
        );

        let mut subclaim = MLSumcheck::verify(
            &mut trans,
            &wrapper.info,
            wrapper.claimed_sum,
            &wrapper.proof,
        )
        .expect("fail to verify the sumcheck protocol");
        let eq_at_u_r = eval_identity_function(&u, &subclaim.point);

        // check the sumcheck evaluation (without NTT)
        if !Self::verify_as_subprotocol(&randomness, &mut subclaim, evals_at_r, info, eq_at_u_r) {
            return false;
        }

        let f_delegation = recursive_proof.delegation_claimed_sums[0];
        // one is to evaluate the random linear combination of evaluations at point r returned from sumcheck protocol
        let mut ntt_coeff_evals_at_r = F::zero();
        evals_at_r.update_ntt_instance_coeff(&mut ntt_coeff_evals_at_r, &randomness_ntt);
        // the other is to evaluate the random linear combination of evaluations at point u sampled before the sumcheck protocol
        let mut ntt_point_evals_at_u = F::zero();
        evals_at_u.update_ntt_instance_point(&mut ntt_point_evals_at_u, &randomness_ntt);

        if !<NTTBareIOP<F>>::verify_as_subprotocol(
            F::one(),
            &mut subclaim,
            &mut wrapper.claimed_sum,
            ntt_coeff_evals_at_r,
            ntt_point_evals_at_u,
            f_delegation,
        ) {
            return false;
        }

        if !(subclaim.expected_evaluations == F::zero() && wrapper.claimed_sum == F::zero()) {
            return false;
        }
        <NTTIOP<F>>::verify_recursive(&mut trans, recursive_proof, &info.ntt_info, &u, &subclaim)
    }

    /// verify as subprotocol
    #[inline]
    pub fn verify_as_subprotocol(
        randomness: &[F],
        subclaim: &mut SubClaim<F>,
        evals: &LiftEval<F>,
        info: &LiftInfo<F>,
        eq_at_u_r: F,
    ) -> bool {
        assert_eq!(randomness.len(), 1);
        subclaim.expected_evaluations -= randomness[0] * eq_at_u_r * evals.k * (F::one() - evals.k);
        true
    }
}

impl<F, EF> LiftSnarks<F, EF>
where
    F: Field + Serialize + for<'de> Deserialize<'de>,
    EF: AbstractExtensionField<F> + Serialize + for<'de> Deserialize<'de>,
{
    /// Complied with PCS to get SNARKs
    pub fn snarks<H, C, S>(instance: &LiftInstance<F>, code_spec: &S)
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
        // Convert the original instance into an instance defined over EF
        let instance_ef = instance.to_ef::<EF>();
        let instance_info = instance_ef.info();

        // 2.1 Generate the random point to instantiate the sumcheck protocol
        let prover_u = prover_trans.get_vec_challenge(
            b"random point used to instantiate sumcheck protocol",
            instance.num_vars,
        );
        let eq_at_u = Rc::new(gen_identity_evaluations(&prover_u));

        // 2.2 Construct the polynomial and the claimed sum to be proved in the sumcheck protocol
        let mut sumcheck_poly = ListOfProductsOfPolynomials::<EF>::new(instance.num_vars);
        let mut claimed_sum = EF::zero();
        let randomness = LiftIOP::sample_coins(&mut prover_trans, &instance_ef);
        let randomness_ntt =
            <NTTIOP<EF>>::sample_coins(&mut prover_trans, instance_info.ntt_info.num_ntt);
            LiftIOP::<EF>::prove_as_subprotocol(
            &randomness,
            &mut sumcheck_poly,
            &instance_ef,
            &eq_at_u,
        );

        // 2.? Prover extract the random ntt instance from all ntt instances
        let ntt_instance = instance.extract_ntt_instance_to_ef::<EF>(&randomness_ntt);
        <NTTBareIOP<EF>>::prove_as_subprotocol(
            EF::one(),
            &mut sumcheck_poly,
            &mut claimed_sum,
            &ntt_instance,
            &prover_u,
        );
        let poly_info = sumcheck_poly.info();
        let ntt_instance_info = ntt_instance.info();

        // 2.3 Generate proof of sumcheck protocol
        let (sumcheck_proof, sumcheck_state) =
            <MLSumcheck<EF>>::prove(&mut prover_trans, &sumcheck_poly)
                .expect("Proof generated in Addition In Zq");
        iop_proof_size += bincode::serialize(&sumcheck_proof).unwrap().len();

        // 2.? [one more step] Prover recursive prove the evaluation of F(u, v)
        let recursive_proof = <NTTIOP<EF>>::prove_recursive(
            &mut prover_trans,
            &sumcheck_state.randomness,
            &ntt_instance_info,
            &prover_u,
        );
        iop_proof_size += bincode::serialize(&recursive_proof).unwrap().len();

        // 2.4 Compute all the evaluations of these small polynomials used in IOP over the random point returned from the sumcheck protocol
        let start = Instant::now();
        let evals_at_r = instance.evaluate_ext(&sumcheck_state.randomness);
        let evals_at_u = instance.evaluate_ext(&prover_u);
        // let eq_at_r = gen_identity_evaluations(&sumcheck_state.randomness);
        // let evals_at_r = instance.evaluate_ext_opt(&eq_at_r);
        // let evals_at_u = instance.evaluate_ext_opt(eq_at_u.as_ref());


        // ------ Sparse Evaluation -------
        // 2.? Prove the sparse matrix evaluation
        let prover_v = <LiftIOP<EF>>::prover_sample_sparse_randomness(&mut prover_trans, &instance_ef);
        let (mut sparse_instance, sparse_eval) = instance_ef.extract_sparse_eval_instance(&prover_v, &evals_at_r);

        let sparse_iop = SparseEvalIOP {
            r_x: sumcheck_state.randomness.clone(),
            r_y: prover_v,
            eval: sparse_eval,
        };
        
        sparse_iop.prover_generate_eval_vector(&mut sparse_instance);
        let sparse_kit = sparse_iop.prove(&sparse_instance);
        let sparse_evals = sparse_instance.evaluate(&sparse_kit.randomness);
        let iop_prover_time = prover_start.elapsed().as_millis();
        // -------------------

        // 2.5 Reduce the proof of the above evaluations to a single random point over the committed polynomial
        let mut requested_point_at_r = sumcheck_state.randomness.clone();
        let mut requested_point_at_u = prover_u.clone();
        let oracle_randomness = prover_trans.get_vec_challenge(
            b"random linear combination for evaluations of oracles",
            instance.log_num_oracles(),
        );
        requested_point_at_r.extend(&oracle_randomness);
        requested_point_at_u.extend(&oracle_randomness);
        let oracle_eval_at_r = committed_poly.evaluate_ext(&requested_point_at_r);
        let oracle_eval_at_u = committed_poly.evaluate_ext(&requested_point_at_u);

        // 2.6 Generate the evaluation proof of the requested point
        let eval_proof_at_r = BrakedownPCS::<F, H, C, S, EF>::open(
            &pp,
            &comm,
            &comm_state,
            &requested_point_at_r,
            &mut prover_trans,
        );
        let eval_proof_at_u = BrakedownPCS::<F, H, C, S, EF>::open(
            &pp,
            &comm,
            &comm_state,
            &requested_point_at_u,
            &mut prover_trans,
        );
        let pcs_open_time = start.elapsed().as_millis();

        // 3. Verifier checks the proof
        let verifier_start = Instant::now();
        let mut verifier_trans = Transcript::<EF>::new();

        // 3.1 Generate the random point to instantiate the sumcheck protocol
        let verifier_u = verifier_trans.get_vec_challenge(
            b"random point used to instantiate sumcheck protocol",
            instance.num_vars,
        );

        // 3.2 Generate the randomness used to randomize all the sub-sumcheck protocols
        let randomness = verifier_trans.get_vec_challenge(
            b"randomness to combine sumcheck protocols",
            <LiftIOP<EF>>::num_coins(&instance_info),
        );
        let randomness_ntt = verifier_trans.get_vec_challenge(
            b"randomness used to obtain the virtual random ntt instance",
            <NTTIOP<EF>>::num_coins(&instance_info.ntt_info),
        );

        // 3.3 Check the proof of the sumcheck protocol
        let mut subclaim = <MLSumcheck<EF>>::verify(
            &mut verifier_trans,
            &poly_info,
            claimed_sum,
            &sumcheck_proof,
        )
        .expect("Verify the sumcheck proof generated in RLWE * RGSW");
        let eq_at_u_r = eval_identity_function(&verifier_u, &subclaim.point);

        // 3.4 Check the evaluation over a random point of the polynomial proved in the sumcheck protocol using evaluations over these small oracles used in IOP
        let check_subclaim = LiftIOP::<EF>::verify_as_subprotocol(
            &randomness,
            &mut subclaim,
            &evals_at_r,
            &instance_info,
            eq_at_u_r,
        );
        assert!(check_subclaim);

        // 3.? Check the NTT part
        let f_delegation = recursive_proof.delegation_claimed_sums[0];
        // one is to evaluate the random linear combination of evaluations at point r returned from sumcheck protocol
        let mut ntt_coeff_evals_at_r = EF::zero();
        evals_at_r.update_ntt_instance_coeff(&mut ntt_coeff_evals_at_r, &randomness_ntt);
        // the other is to evaluate the random linear combination of evaluations at point u sampled before the sumcheck protocol
        let mut ntt_point_evals_at_u = EF::zero();
        evals_at_u.update_ntt_instance_point(&mut ntt_point_evals_at_u, &randomness_ntt);

        // check the sumcheck part of NTT
        let check_ntt_bare = <NTTBareIOP<EF>>::verify_as_subprotocol(
            EF::one(),
            &mut subclaim,
            &mut claimed_sum,
            ntt_coeff_evals_at_r,
            ntt_point_evals_at_u,
            f_delegation,
        );
        assert!(check_ntt_bare);
        assert_eq!(subclaim.expected_evaluations, EF::zero());
        assert_eq!(claimed_sum, EF::zero());
        // check the recursive part of NTT
        let check_recursive = <NTTIOP<EF>>::verify_recursive(
            &mut verifier_trans,
            &recursive_proof,
            &ntt_instance_info,
            &verifier_u,
            &subclaim,
        );
        assert!(check_recursive);

        // ------ Sparse Evaluation -------
        // 3.? Prove the sparse matrix evaluation
        let _ = <LiftIOP<EF>>::verifier_sample_sparse_randomness(&mut verifier_trans, &instance_info);

        let sparse_wrapper = sparse_kit.extract();
        let sparse_check = sparse_iop.verify(&sparse_wrapper, &sparse_evals);
        assert!(sparse_check);
        // -------------------

        // 3.5 and also check the relation between these small oracles and the committed oracle
        let start = Instant::now();
        let mut pcs_proof_size = 0;
        let flatten_evals_at_r = evals_at_r.flatten();
        let flatten_evals_at_u = evals_at_u.flatten();
        let oracle_randomness = verifier_trans.get_vec_challenge(
            b"random linear combination for evaluations of oracles",
            evals_at_r.log_num_oracles(),
        );
        let check_oracle_at_r =
            verify_oracle_relation(&flatten_evals_at_r, oracle_eval_at_r, &oracle_randomness);
        let check_oracle_at_u =
            verify_oracle_relation(&flatten_evals_at_u, oracle_eval_at_u, &oracle_randomness);
        assert!(check_oracle_at_r && check_oracle_at_u);
        let iop_verifier_time = verifier_start.elapsed().as_millis();

        // 3.5 Check the evaluation of a random point over the committed oracle
        let check_pcs_at_r = BrakedownPCS::<F, H, C, S, EF>::verify(
            &pp,
            &comm,
            &requested_point_at_r,
            oracle_eval_at_r,
            &eval_proof_at_r,
            &mut verifier_trans,
        );
        let check_pcs_at_u = BrakedownPCS::<F, H, C, S, EF>::verify(
            &pp,
            &comm,
            &requested_point_at_u,
            oracle_eval_at_u,
            &eval_proof_at_u,
            &mut verifier_trans,
        );
        assert!(check_pcs_at_r && check_pcs_at_u);
        let pcs_verifier_time = start.elapsed().as_millis();
        pcs_proof_size += bincode::serialize(&eval_proof_at_r).unwrap().len()
            + bincode::serialize(&eval_proof_at_u).unwrap().len()
            + bincode::serialize(&flatten_evals_at_r).unwrap().len()
            + bincode::serialize(&flatten_evals_at_u).unwrap().len();

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
        );
    }
}
