# Artifact for HasteBoots

This artifact provides experimental data and the source code of the implementation of HasteBoots for reproducing the experimental results
reported in the paper. 
In particular, the artifacts supports reproducing the performance

The project can run on both macOS, Windows and Linux environment.

## TLDR;

### Parameters

Parameters are set in `VFHE/src/bfhe/parameters.rs`. There are three parameter sets listed in paper:

- Zama: `ZAMA_GOLDILOCKS_PARAMETERS` 
  Note: This parameter is from [this paper](Towards Verifiable FHE in Practice: Proving Correct Execution of TFHE's Bootstrapping using plonky2) where they use an approximate decomposition. Since we only implement the exact full decomposition, so we choose the same decomposition basis $B=2^5$ with a larger $\ell=13$.
- BabyBear: `BABYBEAR_BINARY_128_BITS_PARAMETERS` 
- Goldilocks: `GOLDILOCKS_BINARY_128_BITS_PARAMETERS` 

### TFHE Performance in Table 1

This evaluation only evaluates TFHE with bootstrapping without generating SNARKS.

To reproduct the results in **table 1** from the paper, simply run

```shell
# 'ZAMA_GOLDILOCKS_PARAMETERS'
cargo r -r -p tfhe --example nand_zama
# 'BABYBEAR_BINARY_128_BITS_PARAMETERS' 
cargo r -r -p tfhe --example nand_babybear
# 'GOLDILOCKS_BINARY_128_BITS_PARAMETERS'
cargo r -r -p tfhe --example nand_goldilocks
```



### VFHE Performance in Table 2

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



For the other evaluations

It outputs the proving time and verification time of each phase in pbs.

We also summerize the statistics about the PCS and PIOP costs.

```text
Parameters: Parameters {
    ...
}
--- Starting verification of nand ---

Preparing parameters time: 311.161333ms

[Prover] Starting to generate proofs for modulus switching.
[Prover] Modulus switching proof generation time: 26.734833ms

[Prover] Starting to generate proofs for blind rotation.
[Prover] Blind rotation proof generation time: 3.230655792s

[Prover] Starting to generate proofs for key switching.
[Prover] Key switching proof generation time: 31.9755ms

[Prover] Starting to generate proofs for sample extraction.
[Prover] Sample extraction proof generation time: 4.076375ms

--- Proofs generation done! ---

Proof generation time: 3.293485959s

[Verifier] Starting to check modulus switching.
[Verifier] Modulus switching verification time: 6.448792ms

[Verifier] Starting to check blind rotation.
[Verifier] Blind rotation verification time: 188.42475ms

[Verifier] Starting to check key switching.
[Verifier] Key switching verification time: 9.763834ms

[Verifier] Starting to check sample extraction.
[Verifier] Sample extraction verification time: 755.209µs

--- Proofs verification done! ---

Proof verification total time: 205.425875ms

--- SNARK Statistics Summary ---

Prover Total Time: 3.293485959s
Prover PCS Time (including commit and open): 566.734456ms, accounts for 17.21%
Prover PIOP Time: 2.726751503s

Verifier Total Time: 205.425875ms
Verifier PCS Time (including commit and open): 204.096709ms, accounts for 99.35%
Verifier PIOP Time: 1.329166ms

Proof Sizes: 135.45475006103516 MB total
PCS Proof Sizes: 135.27827739715576 MB, accounts for 99.87%
PIOP Proof Sizes: 0.17647266387939453 MB
```





### Batched VFHE Performance in Table 4






The crate dependency is organized as follows:
- TFHE: api of the TFHE binary operation (without snarks)
  - 
- VFHE: api of the TFHE binary operaion with snarks
  - trace: storing the FHE operation trace during executing TFHE bootstrapping
  - piop: PIOP protocol for relations
  - pcs: brakedown PCS implementation
  - helper: transcript and utility function
  - 