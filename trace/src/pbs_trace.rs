use algebra::Field;

use crate::{BlindRotationTrace, basic_ops::RowPermTrace, key_switching_trace::KeySwitchingTrace};

pub struct PBSTrace<F: Field> {
    pub blind_rotation_trace: BlindRotationTrace<F>,
    pub key_switching_trace: KeySwitchingTrace<F>,
    pub sample_extraction_trace: RowPermTrace<F>,
}
