use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{DeriveInput, Result};

use crate::{ast::Input, attr::ModulusValue};

#[inline]
pub(super) fn derive(input: &DeriveInput) -> Result<TokenStream> {
    let input = Input::from_syn(input)?;
    Ok(impl_ntt(input))
}

fn impl_ntt(input: Input) -> TokenStream {
    let name = &input.ident;
    let field_ty = input.field.ty;

    let modulus_value = input.attrs.modulus_value;
    let modulus = modulus_value.into_token_stream();

    #[cfg(feature = "concrete-ntt")]
    let table = match modulus_value {
        ModulusValue::U32(_) => quote! {::algebra::transformation::prime32::ConcreteTable<Self>},
        ModulusValue::U64(_) => quote! {::algebra::transformation::prime64::ConcreteTable<Self>},
        ModulusValue::U8(_) | ModulusValue::U16(_) => {
            quote! {::algebra::transformation::NTTTable<Self>}
        }
    };
    #[cfg(not(feature = "concrete-ntt"))]
    let table = quote! {::algebra::transformation::NTTTable<Self>};

    #[cfg(feature = "concrete-ntt")]
    let root = match modulus_value {
        ModulusValue::U32(_) | ModulusValue::U64(_) => quote! {Self},
        ModulusValue::U8(_) | ModulusValue::U16(_) => {
            quote! {::algebra::modulus::ShoupFactor<<Self as ::algebra::Field>::Value>}
        }
    };
    #[cfg(not(feature = "concrete-ntt"))]
    let root = quote! {::algebra::modulus::ShoupFactor<<Self as ::algebra::Field>::Value>};

    let ntt_table = format_ident!("NTT_TABLE{}", name.to_string().to_uppercase());

    let from_root = impl_from_root(modulus_value);
    let to_root = impl_to_root(modulus_value, &modulus);
    let mul_root = impl_mul_root(modulus_value, &modulus);
    let mul_root_assign = impl_mul_root_assign(modulus_value, &modulus);
    let generate_ntt_table = impl_generate_ntt_table(modulus_value, &table);

    quote! {
        static #ntt_table: ::std::sync::OnceLock<::arc_swap::ArcSwap<Vec<(u32, ::std::sync::Arc<<#name as ::algebra::NTTField>::Table>)>>> = ::std::sync::OnceLock::new();

        impl ::std::convert::From<usize> for #name {
            #[inline]
            fn from(value: usize) -> Self {
                let modulus = #modulus.try_into().unwrap();
                if value < modulus {
                    Self(value as #field_ty)
                } else {
                    Self((value % modulus) as #field_ty)
                }
            }
        }

        impl ::algebra::NTTField for #name {
            type Table = #table;

            type Root = #root;

            type Degree = #field_ty;

            #from_root

            #to_root

            #mul_root

            #mul_root_assign

            #[inline]
            fn is_primitive_root(root: Self, degree: Self::Degree) -> bool {
                debug_assert!(root.0 < #modulus);
                debug_assert!(
                    degree > 1 && degree.is_power_of_two(),
                    "degree must be a power of two and bigger than 1"
                );

                if root.0 == 0 {
                    return false;
                }

                ::num_traits::Pow::pow(root, degree >> 1).0 == #modulus - 1
            }

            fn try_primitive_root(degree: Self::Degree) -> Result<Self, ::algebra::AlgebraError> {
                // p-1
                let modulus_sub_one = #modulus - 1;

                // (p-1)/n
                let quotient = modulus_sub_one / degree;

                // (p-1) must be divisible by n
                if modulus_sub_one != quotient * degree {
                    return Err(::algebra::AlgebraError::NoPrimitiveRoot {
                        degree: degree.to_string(),
                        modulus: #modulus.to_string(),
                    });
                }

                let mut rng = ::rand::rng();
                let distr = ::rand::distr::Uniform::new_inclusive(2, #modulus - 1).unwrap();

                let mut w = Self(0);

                if (0..100).any(|_| {
                    w = ::num_traits::Pow::pow(Self(::rand::Rng::sample(&mut rng, distr)), quotient);
                    Self::is_primitive_root(w, degree)
                }) {
                    Ok(w)
                } else {
                    Err(::algebra::AlgebraError::NoPrimitiveRoot {
                        degree: degree.to_string(),
                        modulus: #modulus.to_string(),
                    })
                }
            }

            fn try_minimal_primitive_root(degree: Self::Degree) -> Result<Self, ::algebra::AlgebraError> {
                let mut root = Self::try_primitive_root(degree)?;

                let generator_sq = (root * root).to_root();
                let mut current_generator = root;

                for _ in 0..degree {
                    if current_generator < root {
                        root = current_generator;
                    }

                    current_generator.mul_root_assign(generator_sq);
                }

                Ok(root)
            }

            #generate_ntt_table

            fn get_ntt_table(log_n: u32) -> Result<::std::sync::Arc<Self::Table>, ::algebra::AlgebraError> {
                let ntt_tables = #ntt_table.get_or_init(|| ::arc_swap::ArcSwap::from_pointee(Vec::with_capacity(2)));

                if let Some(table) = ntt_tables
                    .load()
                    .iter()
                    .find(|(key, _)| *key == log_n)
                    .map(|(_, v)| ::std::sync::Arc::clone(v))
                {
                    Ok(table)
                } else {
                    Self::init_ntt_table(log_n)?;

                    let ntt_tables = #ntt_table.get().unwrap();

                    let table = ntt_tables
                        .load()
                        .iter()
                        .find(|(key, _)| *key == log_n)
                        .map(|(_, v)| ::std::sync::Arc::clone(v))
                        .unwrap();

                    Ok(table)
                }
            }

            fn init_ntt_table(log_n: u32) -> Result<(), ::algebra::AlgebraError> {
                let ntt_tables = #ntt_table.get_or_init(|| ::arc_swap::ArcSwap::from_pointee(Vec::with_capacity(2)));

                if let None = ntt_tables.load().iter().find(|(key, _)| *key == log_n) {
                    ntt_tables.rcu(|inner| {
                        let mut tables = inner.as_ref().clone();
                        let temp_table = Self::generate_ntt_table(log_n).unwrap();
                        tables.push((log_n, ::std::sync::Arc::new(temp_table)));
                        tables
                    });
                }

                Ok(())
            }

            fn dot_product(a: impl AsRef<[Self]>, b: impl AsRef<[Self]>) -> Self {
                /// `c += a * b`
                fn multiply_add(c: &mut [#field_ty; 2], a: #field_ty, b: #field_ty) {
                    let (lw, hw) = a.widen_mul(b);
                    let carry;
                    (c[0], carry) = c[0].overflowing_add(lw);
                    (c[1], _) = c[1].carry_add(hw, carry);
                }
                use ::algebra::Widening;
                use ::algebra::reduce::{AddReduce, Reduce};
                let a = a.as_ref();
                let b = b.as_ref();
                debug_assert_eq!(a.len(), b.len());
                let mut a_iter = a.chunks_exact(16);
                let mut b_iter = b.chunks_exact(16);
                let acc = (&mut a_iter)
                    .zip(&mut b_iter)
                    .map(|(a_s, b_s)| {
                        let mut c: [#field_ty; 2] = [0, 0];
                        for (&a, &b) in a_s.iter().zip(b_s) {
                            multiply_add(&mut c, a.0, b.0);
                        }
                        c.reduce(<Self as ::algebra::ModulusConfig>::MODULUS)
                    })
                    .fold(0, |acc: #field_ty, b| {
                        acc.add_reduce(b, <Self as ::algebra::ModulusConfig>::MODULUS)
                    });

                a_iter.remainder().iter().zip(b_iter.remainder()).fold(#name(acc),|acc,(&x,&y)| x*y+acc)
            }
        }
    }
}

#[allow(unused_variables)]
fn impl_from_root(modulus_value: ModulusValue) -> TokenStream {
    #[cfg(feature = "concrete-ntt")]
    match modulus_value {
        ModulusValue::U32(_) | ModulusValue::U64(_) => {
            quote! {
                #[inline]
                fn from_root(root: Self::Root) -> Self {
                    root
                }
            }
        }
        ModulusValue::U8(_) | ModulusValue::U16(_) => {
            quote! {
                #[inline]
                fn from_root(root: Self::Root) -> Self {
                    Self(root.value())
                }
            }
        }
    }

    #[cfg(not(feature = "concrete-ntt"))]
    quote! {
        #[inline]
        fn from_root(root: Self::Root) -> Self {
            Self(root.value())
        }
    }
}

#[allow(unused_variables)]
fn impl_to_root(modulus_value: ModulusValue, modulus: &TokenStream) -> TokenStream {
    #[cfg(feature = "concrete-ntt")]
    match modulus_value {
        ModulusValue::U32(_) | ModulusValue::U64(_) => {
            quote! {
                #[inline]
                fn to_root(self) -> Self::Root {
                    self
                }
            }
        }
        ModulusValue::U8(_) | ModulusValue::U16(_) => {
            quote! {
                #[inline]
                fn to_root(self) -> Self::Root {
                    Self::Root::new(self.0, #modulus)
                }
            }
        }
    }

    #[cfg(not(feature = "concrete-ntt"))]
    quote! {
        #[inline]
        fn to_root(self) -> Self::Root {
            Self::Root::new(self.0, #modulus)
        }
    }
}

#[allow(unused_variables)]
fn impl_mul_root(modulus_value: ModulusValue, modulus: &TokenStream) -> TokenStream {
    #[cfg(feature = "concrete-ntt")]
    match modulus_value {
        ModulusValue::U32(_) | ModulusValue::U64(_) => {
            quote! {
                #[inline]
                fn mul_root(self, root: Self::Root) -> Self {
                    self * root
                }
            }
        }
        ModulusValue::U8(_) | ModulusValue::U16(_) => {
            quote! {
                #[inline]
                fn mul_root(self, root: Self::Root) -> Self {
                    use ::algebra::reduce::MulReduce;
                    Self(self.0.mul_reduce(root, #modulus))
                }
            }
        }
    }

    #[cfg(not(feature = "concrete-ntt"))]
    quote! {
        #[inline]
        fn mul_root(self, root: Self::Root) -> Self {
            use ::algebra::reduce::MulReduce;
            Self(self.0.mul_reduce(root, #modulus))
        }
    }
}

#[allow(unused_variables)]
fn impl_mul_root_assign(modulus_value: ModulusValue, modulus: &TokenStream) -> TokenStream {
    #[cfg(feature = "concrete-ntt")]
    match modulus_value {
        ModulusValue::U32(_) | ModulusValue::U64(_) => {
            quote! {
                #[inline]
                fn mul_root_assign(&mut self, root: Self::Root) {
                    *self *= root;
                }
            }
        }
        ModulusValue::U8(_) | ModulusValue::U16(_) => {
            quote! {
                #[inline]
                fn mul_root_assign(&mut self, root: Self::Root) {
                    use ::algebra::reduce::MulReduceAssign;
                    self.0.mul_reduce_assign(root, #modulus);
                }
            }
        }
    }

    #[cfg(not(feature = "concrete-ntt"))]
    quote! {
        #[inline]
        fn mul_root_assign(&mut self, root: Self::Root) {
            use ::algebra::reduce::MulReduceAssign;
            self.0.mul_reduce_assign(root, #modulus);
        }
    }
}

#[allow(unused_variables)]
fn impl_generate_ntt_table(modulus_value: ModulusValue, table: &TokenStream) -> TokenStream {
    #[cfg(feature = "concrete-ntt")]
    match modulus_value {
        ModulusValue::U32(_) | ModulusValue::U64(_) => {
            quote! {
                #[inline]
                fn generate_ntt_table(log_n: u32) -> Result<Self::Table, ::algebra::AlgebraError> {
                    <#table>::new(log_n)
                }
            }
        }
        ModulusValue::U8(_) | ModulusValue::U16(_) => {
            quote! {
                fn generate_ntt_table(log_n: u32) -> Result<Self::Table, ::algebra::AlgebraError> {
                    let n = 1usize << log_n;

                    let root_one = Self(1).to_root();

                    let root = Self::try_minimal_primitive_root((n * 2).try_into().unwrap())?;

                    let root_factor = root.to_root();
                    let mut power = root;

                    let mut ordinal_root_powers = vec![Self::Root::default(); n * 2];
                    let mut iter = ordinal_root_powers.iter_mut();
                    *iter.next().unwrap() = root_one;
                    *iter.next().unwrap() = root_factor;
                    for root_power in iter {
                        power.mul_root_assign(root_factor);
                        *root_power = power.to_root();
                    }

                    Ok(Self::Table::new(
                        root,
                        log_n,
                        ordinal_root_powers,
                    ))
                }
            }
        }
    }

    #[cfg(not(feature = "concrete-ntt"))]
    quote! {
        fn generate_ntt_table(log_n: u32) -> Result<Self::Table, ::algebra::AlgebraError> {
            let n = 1usize << log_n;

            let root_one = Self(1).to_root();

            let root = Self::try_minimal_primitive_root((n * 2).try_into().unwrap())?;

            let root_factor = root.to_root();
            let mut power = root;

            let mut ordinal_root_powers = vec![Self::Root::default(); n * 2];
            let mut iter = ordinal_root_powers.iter_mut();
            *iter.next().unwrap() = root_one;
            *iter.next().unwrap() = root_factor;
            for root_power in iter {
                power.mul_root_assign(root_factor);
                *root_power = power.to_root();
            }

            Ok(Self::Table::new(
                root,
                log_n,
                ordinal_root_powers,
            ))
        }
    }
}
