use algebra::NTTField;

// pub use lwe::KeySwitchingLWEKey;
pub use rlwe::KeySwitchingRLWEKey;

// mod lwe;
mod rlwe;

/// A enum type for different key switching purposes.
#[derive(Debug, Clone)]
pub enum KeySwitchingKeyEnum<Q: NTTField> {
    /// The key switching is based on rlwe multiply with gadget rlwe.
    RLWE(KeySwitchingRLWEKey<Q>),
    /// No key switching.
    None,
}
