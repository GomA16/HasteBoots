use algebra::{AbstractExtensionField, Field};
use helper::Transcript;
use pcs::PolynomialCommitmentScheme;

pub mod lookup;
pub mod ntt;
// pub mod pbs;
pub mod external_product;
pub mod hadamard;
pub mod monomial_hadamard;
pub mod sparse_matrix_eval;

/// Oracle used in the NTT matrix evaluation PIOP
pub struct EvalOracle<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub poly: PCS::Polynomial,
    pub params: PCS::Parameters,
}

impl<F, EF, S, PCS> EvalOracle<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub fn commit(&self) -> (PCS::Commitment, PCS::CommitmentState) {
        PCS::commit(&self.params, &self.poly)
    }

    pub fn open(
        &self,
        trans: &mut Transcript<EF>,
        comm: &PCS::Commitment,
        state: &PCS::CommitmentState,
        points: &[PCS::Point],
    ) -> PCS::Proof {
        PCS::open(&self.params, comm, state, points, trans)
    }
}
