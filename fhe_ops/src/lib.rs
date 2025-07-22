#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![deny(missing_docs)]

//! FHE Operations Library
//! This libary generates FHE instances for various FHE operations.
//! These FHE instances will be extended to pass to the PIOP. 
//! It includes operations such as:
//! 1. LWE Addition
//! 2. Lift Operation
//! 3. Gadget Decomposition
//! 4. 
//! This library provides various operations for Fully Homomorphic Encryption (FHE).
//! 
//! 

mod ciphertext;
mod ops;

pub use ops::lwe_addition::LWEAdditionOpInstance;