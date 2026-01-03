use algebra::Field;

use crate::{BlindRotationTrace, key_switching_trace::KeySwitchingTrace};

pub struct PBSTrace<F: Field> {
    pub blind_rotation_trace: BlindRotationTrace<F>,
    pub key_switching_trace: KeySwitchingTrace<F>,
}
