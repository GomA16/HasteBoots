use algebra::Field;

use crate::{
    BlindRotationTrace, basic_ops::RowPermTrace, key_switching_trace::KeySwitchingTrace,
    modulus_switching_trace::ModulusSwitchingTrace,
};

pub struct PBSTrace<F: Field> {
    pub modulus_switching_trace: ModulusSwitchingTrace<F>,
    pub blind_rotation_trace: BlindRotationTrace<F>,
    pub key_switching_trace: KeySwitchingTrace<F>,
    pub sample_extraction_trace: RowPermTrace<F>,
}
