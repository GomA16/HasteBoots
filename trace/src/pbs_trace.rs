use algebra::{DecomposableField, Field};

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


impl<F: DecomposableField> PBSTrace<F> {
    pub fn generate_batched_trace(self, log_batch_size: usize) -> Self {
        let batch_size = 1 << log_batch_size;

        let modulus_switching_trace_num = self.modulus_switching_trace.input.len();
        let modulus_switching_traces =
            vec![self.modulus_switching_trace; batch_size];
        let key_switching_traces =
            vec![self.key_switching_trace; batch_size];
        let sample_extraction_traces =
            vec![self.sample_extraction_trace; batch_size];

        PBSTrace {
            modulus_switching_trace: ModulusSwitchingTrace::from_batch_trace(
                modulus_switching_traces,
                modulus_switching_trace_num,
            ),
            blind_rotation_trace: self.blind_rotation_trace,
            key_switching_trace: KeySwitchingTrace::from_batch_trace(
                key_switching_traces,
            ),
            sample_extraction_trace: RowPermTrace::from_batch_trace(
                sample_extraction_traces,
            ),
        }
    }
}