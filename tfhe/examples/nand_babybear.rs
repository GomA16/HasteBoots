use algebra::{AsInto, NTTField};
use fhe_core::{LWECiphertext, utils::*};
use rand::Rng;
use tfhe::{
    Decryptor, Encryptor, KeyGen,
    bfhe::{BABYBEAR_BINARY_128_BITS_PARAMETERS, Evaluator},
};

fn main() {
    // set random generator
    let mut rng = rand::rng();

    // set parameter
    let params = *BABYBEAR_BINARY_128_BITS_PARAMETERS;

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
    let c: bool = rng.random();

    let mut a = a.as_into();
    let mut b = b.as_into();
    let mut c = c.as_into();

    let mut x = enc.encrypt(a);
    let mut y = enc.encrypt(b);
    let mut z = enc.encrypt(c);

    // nand
    let time = std::time::Instant::now();
    let ct_nand = eval.nand(&x, &y);
    println!("NAND done in {:?}", time.elapsed());
    let (m, noise) = dec.decrypt_with_noise(&ct_nand);
    assert_eq!(m, nand(a, b), "Noise: {noise}");
    check_noise(noise, "nand");
    
}
