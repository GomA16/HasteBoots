use algebra::AsInto;
// cargo run --release --example count_ntt --features count_ntt
#[cfg(feature = "count_ntt")]
use algebra::transformation::count;

use fhe_core::utils::nand;
use rand::Rng;
use zkfhe::{
    Decryptor, Encryptor, KeyGen,
    bfhe::{DEFAULT_TERNARY_128_BITS_PARAMETERS, Evaluator},
};

fn main() {
    // set random generator
    let mut rng = rand::rng();

    // set parameter
    let params = *DEFAULT_TERNARY_128_BITS_PARAMETERS;

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

    let encryptor = Encryptor::new(sk.clone());
    let evaluator = Evaluator::new(&sk);
    let decryptor = Decryptor::new(sk);
    println!("Evaluation Key Generation done!\n");

    let a = rng.random();
    let b = rng.random();
    let c = rng.random();

    let x = encryptor.encrypt(a);
    let y = encryptor.encrypt(b);
    let z = encryptor.encrypt(c);

    #[cfg(feature = "count_ntt")]
    count::enable_count_ntt_and_intt();
    let ct = evaluator.nand(&x, &y);
    #[cfg(feature = "count_ntt")]
    {
        count::disable_count_ntt_and_intt();
        println!("ntt count: {}", count::get_ntt_count());
        println!("intt count: {}", count::get_intt_count());
    }

    let (m, noise) = decryptor.decrypt_with_noise(&ct);
    assert_eq!(m, nand(a, b), "Noise: {noise}");
    check_noise(noise, "nand");

    #[cfg(feature = "count_ntt")]
    {
        count::clear_ntt_count();
        count::clear_intt_count();
        count::enable_count_ntt_and_intt();
    }

    let ct = evaluator.mux(&x, &y, &z);

    #[cfg(feature = "count_ntt")]
    {
        count::disable_count_ntt_and_intt();
        println!("ntt count: {}", count::get_ntt_count());
        println!("intt count: {}", count::get_intt_count());
    }

    let (m, noise) = decryptor.decrypt_with_noise(&ct);
    assert_eq!(m, if a != 0 { b } else { c }, "Noise: {noise}");
    check_noise(noise, "mux");
}
