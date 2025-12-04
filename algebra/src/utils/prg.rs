//! Implement AES-based PRG.

use crate::utils::{Block, aes::Aes};
use rand::Rng;
use rand_core::{
    CryptoRng, RngCore, SeedableRng,
    block::{BlockRng, BlockRngCore},
};

///Struct of PRG Core
#[derive(Clone, Copy, Debug)]
struct PrgCore {
    aes: Aes,
    state: u64,
}

impl RngCore for PrgCore {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        unreachable!()
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        unreachable!()
    }

    #[inline]
    fn fill_bytes(&mut self, _dst: &mut [u8]) {
        unreachable!()
    }
}

impl BlockRngCore for PrgCore {
    type Item = u32;
    type Results = [u32; 32];

    // Compute [AES(state)..AES(state+8)]
    #[inline(always)]
    fn generate(&mut self, results: &mut Self::Results) {
        let states = [0; 8].map(
            #[inline(always)]
            |_| {
                let x = self.state;
                self.state += 1;
                Block::from([x, 0])
            },
        );
        *results = bytemuck::cast(self.aes.encrypt_many_blocks(states))
    }
}

impl SeedableRng for PrgCore {
    type Seed = Block;

    #[inline(always)]
    fn from_seed(seed: Self::Seed) -> Self {
        let aes = Aes::new(seed);
        Self { aes, state: 0u64 }
    }
}

impl CryptoRng for PrgCore {}

/// Struct of PRG
#[derive(Clone, Debug)]
pub struct Prg(BlockRng<PrgCore>);

impl RngCore for Prg {
    #[inline(always)]
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    #[inline(always)]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill_bytes(dest)
    }
}

impl SeedableRng for Prg {
    type Seed = Block;

    #[inline(always)]
    fn from_seed(seed: Self::Seed) -> Self {
        Prg(BlockRng::<PrgCore>::from_seed(seed))
    }

    #[inline(always)]
    fn from_rng(rng: &mut impl RngCore) -> Self {
        Prg(BlockRng::<PrgCore>::from_rng(rng))
    }
}

impl CryptoRng for Prg {}

impl Prg {
    /// New a Prg with random seed.
    #[inline(always)]
    pub fn new() -> Self {
        let seed = rand::random::<Block>();
        Prg::from_seed(seed)
    }

    /// Generate a random bool value.
    #[inline(always)]
    pub fn random_bool(&mut self) -> bool {
        self.random()
    }

    /// Fill a bool slice with random bool values.
    #[inline(always)]
    pub fn random_bools(&mut self, buf: &mut [bool]) {
        self.fill(buf);
    }

    /// Generate a random byte value.
    #[inline(always)]
    pub fn random_byte(&mut self) -> u8 {
        self.random()
    }

    /// Fill a byte slice with random values.
    #[inline(always)]
    pub fn random_bytes(&mut self, buf: &mut [u8]) {
        self.fill_bytes(buf);
    }

    /// Generate a random block.
    #[inline(always)]
    pub fn random_block(&mut self) -> Block {
        self.random()
    }

    /// Fill a block slice with random block values.
    #[inline(always)]
    pub fn random_blocks(&mut self, buf: &mut [Block]) {
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(buf);
        self.fill_bytes(bytes);
    }
}

impl Default for Prg {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

#[test]
fn prg_test() {
    let mut prg = Prg::new();
    let mut x = vec![Block::ZERO; 2];
    prg.random_blocks(&mut x);
    assert_ne!(x[0], x[1]);
}
