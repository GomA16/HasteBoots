use algebra::modulus::PowOf2Modulus;
use algebra::reduce::{AddReduce, SubReduce};
use fhe_core::{decode, encode};

type T = u32;

fn main() {
    let t: T = 4;
    let q: T = 512;

    // q/2t
    let noise_max = (q / (t * 2)) as T;

    let modulus = PowOf2Modulus::<T>::new(q as T);

    // check all message are encoded and decoded correctly, even with noise.
    for i in 0..t {
        let message: T = i.try_into().unwrap();

        let encoded: T = encode(message, t, q);

        let decoded: T = decode(encoded, t, q);
        assert_eq!(decoded, message);

        // add noise
        let decoded: T = decode(encoded.add_reduce(noise_max - 1, modulus), t, q);
        assert_eq!(decoded, message);

        // sub noise
        let decoded: T = decode(encoded.sub_reduce(noise_max - 1, modulus), t, q);
        assert_eq!(decoded, message);
    }
}
