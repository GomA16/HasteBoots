//! This snarks implementation includes the proof generation for FHE operations.
//!
//! When considering the multiplication-related relation between polynomials,
//! we are able to use Hadamard product to represent the element-wise relation
//! of their NTT evaluations.
//!
//! To reduce the number of elements to be committed as much as possible and also
//! to simplify the proof structure, we only commit to the coefficient form of the
//! polynomials.
//!
//! After running the sumcheck protocol for Hadamard product, it is reduced to
//! querying the evaluations of these polynomials at some random points.
//!
//! All these queries are answered by the NTT PIOP, reducing to the queries of
//! their coefficient forms.
//! - If their coefficient forms are committed in PCS, the queries can be answered
//! by the PCS opening proofs.
//! - If their coefficients are sparse (e.g. monomial), the queries can be answered
//! by the sparse PCS opening proofs.

pub mod acc_iteration;
pub mod blind_rotation;
pub mod decomposition;
pub mod external_product;
pub mod hadmard_product;
pub mod key_switching;
pub mod modulus_switching;
pub mod monomial_hadamard;
pub mod row_permutation;
