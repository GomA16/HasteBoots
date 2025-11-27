//! PIOP for NTT evaluation

use algebra::Field;

pub struct NTTEvalInstance<F: Field> {
    /// number of polynomials denoted as log_m
    pub log_num_polys: u32,
    /// polynomial degree (assumed to be power of two) denoted as log_n
    pub log_poly_degree: u32,
    /// number of variables denoted as log_n + log_m
    
}

