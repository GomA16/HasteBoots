//! Utilities for zk-proof systems.
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![deny(missing_docs)]
/// Transcript module for Fiat-Shamir transforms.
pub mod transcript;
/// Utilities for zk-proof systems.
pub mod utils;

use num_traits::ops::bytes;
pub use transcript::Transcript;

use algebra::Field;
use serde::Serialize;

/// A trait for Fiat-Shamir transcripts.
pub trait FiatShamirTranscript<F: Field> {
    /// Appends a message to the transcript.
    fn append_message<M: Serialize>(&mut self, label: &'static [u8], message: &M);
    /// Generate the challenge bytes from the current transcript
    fn get_challenge_bytes(&mut self, label: &'static [u8], bytes: &mut [u8]);
    /// Generate the challenge from the current transcript
    fn get_challenge(&mut self, label: &'static [u8]) -> F;
    /// Generate the challenge vector from the current transcript
    fn get_vec_challenge(&mut self, label: &'static [u8], num: usize) -> Vec<F>;
}