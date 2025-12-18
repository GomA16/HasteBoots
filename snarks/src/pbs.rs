use core::time;

use algebra::{AbstractExtensionField, DenseMultilinearExtension, Field};
use helper::utils::compute_oracle_evals;
use helper::{FiatShamirTranscript, Transcript};
use pcs::PolynomialCommitmentScheme;
use pcs::utils::code;
use piop::lookup::logup::LogUpInstanceInfo;
use piop::lookup::{LogUpIOP, LogUpInstance, LogUpProof};
use piop::{PackableEFProof, PackableProof, SumcheckInstance, SumcheckPIOP};
use serde::Serialize;
use trace::PackableTrace;
use trace::{ConvertToEF, LookupTrace, LookupTraceMLE, LookupWitness, LookupWitnessHelper};


pub struct PBSSnarks<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    _marker_f: std::marker::PhantomData<F>,
    _marker_ef: std::marker::PhantomData<EF>,
    _marker_s: std::marker::PhantomData<S>,
    _marker_pcs: std::marker::PhantomData<PCS>,
}

#[derive(Serialize)]
pub struct PBSSnarksParams<F, EF, S, PCS>
where
    F: Field,
    EF: AbstractExtensionField<F>,
    S: Clone,
    PCS: PolynomialCommitmentScheme<F, EF, S>,
{
    pub blk_size: usize,
    pub pcs_params_key_ntt: PCS::Parameters,
    pub pcs_params_poly: PCS::Parameters,
    pub pcs_params_helper: PCS::Parameters,
}

impl<F, EF, S, PCS> PBSSnarks<F, EF, S, PCS>
where
    F: Field + Serialize,
    EF: AbstractExtensionField<F> + Serialize,
    S: Clone + Serialize,
    PCS: PolynomialCommitmentScheme<F, EF, S> + Serialize,
{


}