use algebra::AsInto;
use fhe_core::utils::*;
use rand::Rng;
use tfhe::{
    Decryptor, Encryptor, KeyGen,
    bfhe::{Evaluator, ZAMA_BINARY_128_BITS_PARAMETERS},
};

fn main() {
    // set random generator
    let mut rng = rand::rng();

    // set parameter
    let params = *ZAMA_BINARY_128_BITS_PARAMETERS;

    let noise_max = (params.lwe_cipher_modulus_value() as f64 / 16.0).as_into();

    let check_noise = |noise, op: &str| {
        assert!(
            noise < noise_max,
            "Type: {op}\nNoise: {noise} >= {noise_max}"
        );
        println!("{op:4.4} Noise: {noise:3} < {noise_max:3}");
    };

    // generate keys
    let sk = KeyGen::generate_secret_key(params);
    println!("Secret Key Generation done!\n");

    let enc = Encryptor::new(sk.clone());
    let eval = Evaluator::new(&sk);
    let dec = Decryptor::new(sk);
    println!("Evaluation Key Generation done!\n");

    let a: bool = rng.random();
    let b: bool = rng.random();

    let a = a.as_into();
    let b = b.as_into();

    let x = enc.encrypt(a);
    let y = enc.encrypt(b);

    // nand
    let time = std::time::Instant::now();
    let ct_nand = eval.nand(&x, &y);
    println!("NAND done in {:?}", time.elapsed());
    let (m, noise) = dec.decrypt_with_noise(&ct_nand);
    assert_eq!(m, nand(a, b), "Noise: {noise}");
    check_noise(noise, "nand");
}
