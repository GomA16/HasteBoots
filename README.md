# Artifact for HasteBoots

This artifact provides experimental data and the source code of the implementation of HasteBoots for reproducing the experimental results reported in the paper. 

The project can run on both macOS, Windows and Linux environment.

The crate dependency is organized as follows:

- TFHE: api of the TFHE binary operation (without snarks)
- VFHE: api of the TFHE binary operaion with snarks
  - trace: storing the FHE operation trace during executing TFHE bootstrapping
  - piop: PIOP protocol for relations
  - pcs: brakedown PCS implementation
  - helper: transcript and utility function

## Install Rust

This project relies on Rust. Installation can be done by following these steps:

1. Install build tools.
  On Windows, please install [Visual Studio C++ Build tools](https://rust-lang.github.io/rustup/installation/windows-msvc.html).
   On Ubuntu and Debian, please install build-essential according to the instructions below:
2. Install Rust using rustup (the recommended Rust installer):
  ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
   On Windows, one can download and run the installer `rustup-init.exe` from [https://rust-lang.org/tools/install/](https://rust-lang.org/tools/install/).
3. After installation, verify Rust is installed correctly:
  ```bash
   rustc --version
   cargo --version
  ```

For more information, see the [Rust installation guide](https://www.rust-lang.org/tools/install).

## Parameters

Parameters are set in `vfhe/src/bfhe/parameters.rs`. There are three parameter sets listed in paper:

- Zama: `ZAMA_GOLDILOCKS_PARAMETERS` 
Note: This parameter is from [this paper](Towards Verifiable FHE in Practice: Proving Correct Execution of TFHE's Bootstrapping using plonky2) where they use an approximate decomposition. Since we only implement the exact full decomposition, so we choose the same decomposition basis $B=2^5$ with a larger $\ell=13$.
- BabyBear: `BABYBEAR_BINARY_128_BITS_PARAMETERS` 
- Goldilocks: `GOLDILOCKS_BINARY_128_BITS_PARAMETERS`

For each parameter setting, we run the experiment 10 times and report the average proof generation time, verification time, and proof size.

## TFHE Performance in Table 1

This evaluation only evaluates TFHE with bootstrapping **without** generating SNARKS.

To reproduct the results in **table 1** from the paper, simply run

```shell
# 'ZAMA_GOLDILOCKS_PARAMETERS'
cargo r -r -p tfhe --example nand_zama
# 'BABYBEAR_BINARY_128_BITS_PARAMETERS' 
cargo r -r -p tfhe --example nand_babybear
# 'GOLDILOCKS_BINARY_128_BITS_PARAMETERS'
cargo r -r -p tfhe --example nand_goldilocks
```

## VFHE Performance in Table 2

We only implemented and integrated  `Brakedown` PCS in our codebase.

To reproduct the results in **table 2** from the paper, simply run

```shell
# 'ZAMA_GOLDILOCKS_PARAMETERS'
cargo r -r -p vfhe --example zk_nand_zama
# 'BABYBEAR_BINARY_128_BITS_PARAMETERS' 
cargo r -r -p vfhe --example zk_nand_babybear
# 'GOLDILOCKS_BINARY_128_BITS_PARAMETERS'
cargo r -r -p vfhe --example zk_nand_goldilocks

# For more detialed outpus:
# 'ZAMA_GOLDILOCKS_PARAMETERS'
RUST_LOG=info cargo r -r -p vfhe --example zk_nand_zama
# 'BABYBEAR_BINARY_128_BITS_PARAMETERS' 
RUST_LOG=info cargo r -r -p vfhe --example zk_nand_babybear
# 'GOLDILOCKS_BINARY_128_BITS_PARAMETERS'
RUST_LOG=info cargo r -r -p vfhe --example zk_nand_goldilocks
```

Since the PIOP layer and the PCS layer are fully modular and separable, we are able to isolate the performance cost introduced by the PCS from the overall end-to-end performance.
This allows us to instantiate the same PIOP with different PCS choices that commit to and evaluate polynomials of the same size.

Concretely, we integrate our PIOP construction with Dory and BaseFold.
For Dory, we use the implementation provided in [Jolt](https://github.com/a16z/jolt), which is based on elliptic-curve commitments over the BN254 curve and targets a 128-bit security level.
For BaseFold, we use the implementation from [Jolt-b](https://github.com/cysic-labs/jolt-b), which operates over the Goldilocks prime field
$Q = 2^{64} - 2^{32} + 1$ and a degree-2 extension field of size $Q^2$, also corresponding to approximately 128-bit security.

The full experimental statistics are provided in `statistics/HasteBoots_Performance.xlsx` (Table 2). 

## Batched VFHE Performance in Table 4

We only implemented and integrated  `Brakedown` PCS in our codebase.

To reproduct the results with Brakedown in **table 4** from the paper, simply run

```shell
# 'BABYBEAR_BINARY_128_BITS_PARAMETERS' 
cargo r -r -p vfhe --example zk_nand_batch
```

The full experimental statistics are provided in `statistics/HasteBoots_Performance.xlsx` (Table 4). 