use algebra::{
    AsInto, Basis, Field, FieldDiscreteGaussianSampler, NTTField, NTTPolynomial, Polynomial,
    ntt_mul_assign_fast,
    transformation::{AbstractNTT, MonomialNTT},
};
use lattice::{
    DecompositionSpace, LWE, NTTPolynomialSpace, NTTRGSW, NTTRLWESpace, PolynomialSpace, RLWE,
    RLWESpace,
};

use num_traits::Zero;
use trace::{AccTrace, SumHadamardTrace};

/// FHE binary blind rotation key
#[derive(Debug, Clone)]
pub struct BinaryBlindRotationKey<F: NTTField> {
    key: Vec<NTTRGSW<F>>,
}

impl<F: NTTField> BinaryBlindRotationKey<F> {
    /// Creates a new [`BinaryBlindRotationKey<F>`].
    #[inline]
    pub fn new(key: Vec<NTTRGSW<F>>) -> Self {
        Self { key }
    }

    /// Performs the blind rotation operation.
    pub fn blind_rotate(&self, mut lut: Polynomial<F>, lwe: &LWE<<F as Field>::Value>) -> RLWE<F> {
        let rlwe_dimension = lut.coeff_count();

        let decompose_space = &mut DecompositionSpace::new(rlwe_dimension);
        let ntt_polynomial_space = &mut NTTPolynomialSpace::new(rlwe_dimension);
        let polynomial_space = &mut PolynomialSpace::new(rlwe_dimension);
        let ntt_rlwe_space = &mut NTTRLWESpace::new(rlwe_dimension);
        let external_product = &mut RLWESpace::new(rlwe_dimension);

        let ntt_table = F::get_ntt_table(rlwe_dimension.trailing_zeros()).unwrap();

        // lut * X^{-b}
        if !lwe.b().is_zero() {
            let neg_b = (rlwe_dimension << 1) - AsInto::<usize>::as_into(lwe.b());
            let lut = lut.as_mut_slice();
            if neg_b < rlwe_dimension {
                ntt_polynomial_space[neg_b] = F::one();
            } else {
                ntt_polynomial_space[neg_b - rlwe_dimension] = F::neg_one();
            }
            ntt_table.transform_slice(ntt_polynomial_space.as_mut_slice());

            ntt_table.transform_slice(lut);
            ntt_mul_assign_fast(lut, ntt_polynomial_space);
            ntt_table.inverse_transform_slice(lut);

            // if neg_b <= rlwe_dimension {
            //     lut.as_mut_slice().rotate_right(neg_b);
            //     lut[..neg_b].iter_mut().for_each(|v| *v = v.neg());
            // } else {
            //     let r = neg_b - rlwe_dimension;
            //     lut.as_mut_slice().rotate_right(r);
            //     lut[r..].iter_mut().for_each(|v| *v = v.neg());
            // }
        }

        let acc = RLWE::new(Polynomial::zero(rlwe_dimension), lut);

        self.key
            .iter()
            .zip(lwe.a())
            .fold(acc, |mut acc, (s_i, &a_i)| {
                if !a_i.is_zero() {
                    // external_product = (X^{a_i} - 1) * ACC
                    acc.transform_inplace(ntt_rlwe_space);
                    ntt_polynomial_space.set_zero();
                    let a_i: usize = a_i.as_into();
                    if a_i < rlwe_dimension {
                        ntt_polynomial_space[a_i] = F::one();
                    } else {
                        ntt_polynomial_space[a_i - rlwe_dimension] = F::neg_one();
                    }
                    ntt_rlwe_space.mul_ntt_polynomial_assign(ntt_polynomial_space);
                    ntt_rlwe_space.inverse_transform_inplace(external_product);
                    external_product.sub_assign_element_wise(&acc);

                    // acc.mul_monic_monomial_sub_one_inplace(
                    //     rlwe_dimension,
                    //     a_i.as_into(),
                    //     external_product,
                    // );

                    // external_product = (X^{a_i} - 1) * ACC * RGSW(s_i)
                    external_product.mul_assign_ntt_rgsw(
                        s_i,
                        decompose_space,
                        polynomial_space,
                        ntt_rlwe_space,
                    );
                    // ACC = ACC + (X^{a_i} - 1) * ACC * RGSW(s_i)
                    acc.add_assign_element_wise(external_product);
                }

                acc
            })
    }

    pub fn blind_rotate_w_trace(
        &self,
        mut lut: Polynomial<F>,
        lwe: &LWE<<F as Field>::Value>,
        // Trace
        acc_trace: &mut AccTrace<F>,
        hadmard_trace: &mut SumHadamardTrace<F>,
    ) -> RLWE<F> {
        println!("Binary Blind Rotation with Trace");
        let rlwe_dimension = lut.coeff_count();
        let lwe_dimension = lwe.a().len();

        let decompose_space = &mut DecompositionSpace::new(rlwe_dimension);
        let ntt_polynomial_space = &mut NTTPolynomialSpace::new(rlwe_dimension);
        let polynomial_space = &mut PolynomialSpace::new(rlwe_dimension);
        let ntt_rlwe_space = &mut NTTRLWESpace::new(rlwe_dimension);
        let external_product = &mut RLWESpace::new(rlwe_dimension);
        

        let ntt_table = F::get_ntt_table(rlwe_dimension.trailing_zeros()).unwrap();

        // lut * X^{-b}
        if !lwe.b().is_zero() {
            let neg_b = (rlwe_dimension << 1) - AsInto::<usize>::as_into(lwe.b());

            let lut = lut.as_mut_slice();

            // TODO: Remove follow line
            polynomial_space.copy_from(&*lut);
            if neg_b < rlwe_dimension {
                ntt_polynomial_space[neg_b] = F::one();
            } else {
                ntt_polynomial_space[neg_b - rlwe_dimension] = F::neg_one();
            }
            ntt_table.transform_slice(ntt_polynomial_space.as_mut_slice());

            ntt_table.transform_slice(lut);
            ntt_mul_assign_fast(lut, ntt_polynomial_space);
            ntt_table.inverse_transform_slice(lut);

            // TODO: Remove follow codes
            #[cfg(test)]
            {
                if neg_b <= rlwe_dimension {
                polynomial_space.as_mut_slice().rotate_right(neg_b);
                polynomial_space[..neg_b]
                    .iter_mut()
                    .for_each(|v| *v = v.neg());
                } else {
                    let r = neg_b - rlwe_dimension;
                    polynomial_space.as_mut_slice().rotate_right(r);
                    polynomial_space[r..].iter_mut().for_each(|v| *v = v.neg());
                }
                assert_eq!(
                    polynomial_space.as_slice(),
                    &*lut,
                    "111111111111111111111111"
                );
                println!("Sanity check for lut * X^-b passed.");
            }
            
        }

        let acc = RLWE::new(Polynomial::zero(rlwe_dimension), lut);

        assert!(rlwe_dimension.is_power_of_two());

        // TODO: Remove the repeated code for computing ntt of acc anc monomials
        acc_trace.append_acc_initial(acc.a_b_slice());

        let mut round = 0;
        self.key
            .iter()
            .zip(lwe.a())
            .fold(acc, |mut acc, (s_i, &a_i)| {
                // external_product = (X^{a_i} - 1) * ACC
                acc.transform_inplace(ntt_rlwe_space);
                ntt_polynomial_space.set_zero();
                let a_i: usize = a_i.as_into();
                if a_i < rlwe_dimension {
                    ntt_polynomial_space[a_i] = F::one();
                } else {
                    ntt_polynomial_space[a_i - rlwe_dimension] = F::neg_one();
                }
                acc_trace.append_monomial(ntt_polynomial_space.as_slice());
                ntt_table.transform_slice(ntt_polynomial_space.as_mut_slice());
                ntt_rlwe_space.mul_ntt_polynomial_assign(ntt_polynomial_space);
                ntt_rlwe_space.inverse_transform_inplace(external_product);
                acc_trace.append_product(external_product.a_b_slice());
                external_product.sub_assign_element_wise(&acc);
                acc_trace.append_external_product_input(external_product.a_b_slice());

                #[cfg(test)]
                {
                    let external_product2 = &mut RLWESpace::new(rlwe_dimension);
                    acc.mul_monic_monomial_sub_one_inplace(
                    rlwe_dimension,
                    a_i.as_into(),
                    external_product2,
                    );
                    assert_eq!(
                        &**external_product, &**external_product2,
                        "22222222222222222222"
                    );
                    println!("Sanity check for (X^a - 1) * ACC passed.");
                }
                

                // external_product = (X^{a_i} - 1) * ACC * RGSW(s_i)
                external_product.mul_assign_ntt_rgsw_w_trace(
                    s_i,
                    decompose_space,
                    polynomial_space,
                    ntt_rlwe_space,
                    hadmard_trace,
                );
                hadmard_trace.add_sum_prod_poly(external_product.a_b_slice());

                // ACC = ACC + (X^{a_i} - 1) * ACC * RGSW(s_i)
                acc.add_assign_element_wise(external_product);

                round += 1;
                if round < lwe_dimension {
                    acc_trace.append_acc_round(acc.a_b_slice());
                } else {
                    acc_trace.append_acc_output(acc.a_b_slice());
                }
                acc
            })
    }

    /// Generates the [`BinaryBlindRotationKey<F>`].
    pub(crate) fn generate<Rng>(
        lwe_secret_key: &[F],
        rlwe_secret_key: &NTTPolynomial<F>,
        blind_rotation_basis: Basis<F>,
        chi: FieldDiscreteGaussianSampler,
        rng: &mut Rng,
    ) -> Self
    where
        Rng: rand::Rng + rand::CryptoRng,
    {
        let key = lwe_secret_key
            .iter()
            .map(|&s| {
                if s.is_zero() {
                    <NTTRGSW<F>>::generate_random_zero_sample(
                        rlwe_secret_key,
                        blind_rotation_basis,
                        chi,
                        rng,
                    )
                } else {
                    <NTTRGSW<F>>::generate_random_one_sample(
                        rlwe_secret_key,
                        blind_rotation_basis,
                        chi,
                        rng,
                    )
                }
            })
            .collect();
        BinaryBlindRotationKey::new(key)
    }
}
