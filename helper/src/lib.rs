//! Utilities for zk-proof systems.
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![deny(missing_docs)]
/// Transcript module for Fiat-Shamir transforms.
pub mod transcript;
/// Utilities for zk-proof systems.
pub mod utils;

pub use transcript::Transcript;
