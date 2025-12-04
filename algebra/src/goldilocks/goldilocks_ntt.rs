use std::sync::{Arc, OnceLock};

use arc_swap::ArcSwap;
use num_traits::{Zero, pow};

use crate::{Field, NTTField, transformation::prime64::ConcreteTable};

use super::Goldilocks;

impl From<usize> for Goldilocks {
    #[inline]
    fn from(value: usize) -> Self {
        let modulus = Goldilocks::MODULUS_VALUE as usize;
        if value < modulus {
            Self(value as u64)
        } else {
            Self((value - modulus) as u64)
        }
    }
}

type Table = <Goldilocks as NTTField>::Table;
static NTT_TABLE: OnceLock<ArcSwap<Vec<(u32, Arc<Table>)>>> = OnceLock::new();

impl NTTField for Goldilocks {
    type Table = ConcreteTable<Self>;

    type Root = Self;

    type Degree = u64;

    #[inline]
    fn from_root(root: Self::Root) -> Self {
        root
    }

    #[inline]
    fn to_root(self) -> Self::Root {
        self
    }

    #[inline]
    fn mul_root(self, root: Self::Root) -> Self {
        self * root
    }

    #[inline]
    fn mul_root_assign(&mut self, root: Self::Root) {
        *self *= root
    }

    #[inline]
    fn is_primitive_root(root: Self, degree: Self::Degree) -> bool {
        debug_assert!(
            degree > 1 && degree.is_power_of_two(),
            "degree must be a power of two and bigger than 1"
        );

        if root == Self::zero() {
            return false;
        }

        pow(root, (degree >> 1) as usize) == Self::neg_one()
    }

    fn try_primitive_root(degree: Self::Degree) -> Result<Self, crate::AlgebraError> {
        let modulus_sub_one = Goldilocks::MODULUS_VALUE - 1;
        let quotient = modulus_sub_one / degree;
        if modulus_sub_one != quotient * degree {
            return Err(crate::AlgebraError::NoPrimitiveRoot {
                degree: degree.to_string(),
                modulus: Goldilocks::MODULUS_VALUE.to_string(),
            });
        }

        let mut rng = rand::rng();
        let distr = rand::distr::Uniform::new_inclusive(2, modulus_sub_one).unwrap();

        let mut w = Self::zero();

        if (0..100).any(|_| {
            w = pow(
                Self::new(rand::Rng::sample(&mut rng, distr)),
                quotient as usize,
            );
            Self::is_primitive_root(w, degree)
        }) {
            Ok(w)
        } else {
            Err(crate::AlgebraError::NoPrimitiveRoot {
                degree: degree.to_string(),
                modulus: Goldilocks::MODULUS_VALUE.to_string(),
            })
        }
    }

    fn try_minimal_primitive_root(degree: Self::Degree) -> Result<Self, crate::AlgebraError> {
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

    #[inline]
    fn generate_ntt_table(log_n: u32) -> Result<Self::Table, crate::AlgebraError> {
        Self::Table::new(log_n)
    }

    fn init_ntt_table(log_n: u32) -> Result<(), crate::AlgebraError> {
        let ntt_tables = NTT_TABLE.get_or_init(|| ArcSwap::from_pointee(Vec::with_capacity(2)));

        if let None = ntt_tables.load().iter().find(|(key, _)| *key == log_n) {
            ntt_tables.rcu(|inner| {
                let mut tables = inner.as_ref().clone();
                let temp_table = Self::generate_ntt_table(log_n).unwrap();
                tables.push((log_n, Arc::new(temp_table)));
                tables
            });
        }

        Ok(())
    }

    fn get_ntt_table(log_n: u32) -> Result<Arc<Self::Table>, crate::AlgebraError> {
        let ntt_tables = NTT_TABLE.get_or_init(|| ArcSwap::from_pointee(Vec::with_capacity(2)));

        if let Some(table) = ntt_tables
            .load()
            .iter()
            .find(|(key, _)| *key == log_n)
            .map(|(_, v)| Arc::clone(v))
        {
            Ok(table)
        } else {
            Self::init_ntt_table(log_n)?;

            let ntt_tables = NTT_TABLE.get().unwrap();

            let table = ntt_tables
                .load()
                .iter()
                .find(|(key, _)| *key == log_n)
                .map(|(_, v)| Arc::clone(v))
                .unwrap();

            Ok(table)
        }
    }
}

#[test]
fn ntt_test() {
    use crate::{NTTPolynomial, Polynomial};
    let n = 1 << 10;
    let mut rng = rand::rng();
    let poly = Polynomial::<Goldilocks>::random(n, &mut rng);

    let ntt_poly: NTTPolynomial<Goldilocks> = poly.clone().into();

    let expect_poly: Polynomial<Goldilocks> = ntt_poly.into();
    assert_eq!(poly, expect_poly);
}
