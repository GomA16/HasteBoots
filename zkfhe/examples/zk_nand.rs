use algebra::NTTField;
use fhe_core::{LWECiphertext, LWEModulusType, utils::*};
use rand::Rng;
use zkfhe::{
    Decryptor, Encryptor, KeyGen,
    bfhe_trace::{DEFAULT_TERNARY_128_BITS_PARAMETERS, Evaluator},
};

type M = bool;
type C = u16;

fn main() {
    // set random generator
    let mut rng = rand::rng();

    // set parameter
    let params = *DEFAULT_TERNARY_128_BITS_PARAMETERS;

    let noise_max = (params.lwe_cipher_modulus_value() as f64 / 16.0) as C;

    let check_noise = |noise: C, op: &str| {
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

    let a = rng.random();
    let b = rng.random();
    // let mut c = rng.random();

    let x = enc.encrypt(a);
    let y = enc.encrypt(b);
    // let mut z = enc.encrypt(c);

    for i in 0..1 {
        let start = std::time::Instant::now();
        let ct_nand = join_bit_operations(&eval, &x, &y);

        // nand
        let (m, noise) = dec.decrypt_with_noise::<M>(&ct_nand);
        assert_eq!(m, nand(a, b), "Noise: {noise}");
        check_noise(noise, "nand");

        println!("The {i} group test done!\n");
    }
}

#[allow(clippy::type_complexity)]
fn join_bit_operations<T: LWEModulusType, F: NTTField>(
    eval: &Evaluator<T, F>,
    x: &LWECiphertext<T>,
    y: &LWECiphertext<T>,
) -> LWECiphertext<T> {
    eval.nand(x, y)
}
