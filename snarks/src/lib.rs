use algebra::{AbstractExtensionField, Field};
use helper::Transcript;
use pcs::PolynomialCommitmentScheme;

pub mod cmp;
pub mod fhe_batch_op;
pub mod fhe_op;
pub mod lookup;
pub mod ntt;
pub mod sparse_matrix_eval;

#[derive(Default, Debug)]
pub struct SnarkStatistics {
    pub prover_pcs_time: std::time::Duration,
    pub verifier_pcs_time: std::time::Duration,
}

impl SnarkStatistics {
    pub fn add_prover_pcs_time(&mut self, dur: std::time::Duration) {
        self.prover_pcs_time += dur;
    }

    pub fn add_verifier_pcs_time(&mut self, dur: std::time::Duration) {
        self.verifier_pcs_time += dur;
    }
}

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
