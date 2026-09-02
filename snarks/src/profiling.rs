//! Feature-gated, semantics-neutral wall-clock instrumentation for the blind-rotation prover.

use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum BrPhase {
    Total,
    InputPreparation,
    BitOracle,
    TraceMle,
    MainCommit,
    Hadamard,
    HadamardToEf,
    HadamardInstances,
    HadamardBatching,
    HadamardSumcheck,
    HadamardFastEvaluate,
    HadamardTerminal,
    Ntt,
    NttSparseInstance,
    NttBitPackEval,
    NttKeyPackEval,
    NttSumcheck,
    TraceOpening,
    KeyOpening,
    Sparse,
    SparseInstance,
    SparseLookup,
    SparseLookupWitnessCommit,
    SparseLookupHelperCommit,
    SparseLookupSumcheck,
    SparseProof,
    Accumulator,
    AccumulatorTrace,
    AccumulatorLookup,
    AccumulatorLookupWitnessCommit,
    AccumulatorLookupHelperCommit,
    AccumulatorLookupSumcheck,
    AccumulatorPermutation,
    Decomposition,
    DecompositionTrace,
    DecompositionParams,
    DecompositionInputMaterialize,
    DecompositionInputCommit,
    DecompositionLookupTrace,
    DecompositionLookupParams,
    DecompositionLookupWitnessMaterialize,
    DecompositionLookupBaseCommit,
    DecompositionLookupHelperCompute,
    DecompositionLookupHelperMaterialize,
    DecompositionLookupHelperCommit,
    DecompositionLookupSumcheck,
    DecompositionLookupEvalPack,
    DecompositionLookupBaseOpen,
    DecompositionLookupHelperOpen,
    DecompositionInputEvals,
    DecompositionInputOpen,
    Finalize,
}

pub const BR_PHASE_COUNT: usize = 52;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct BrPhaseDefinition {
    pub id: &'static str,
    pub parent_id: Option<&'static str>,
}

pub const BR_PHASE_DEFINITIONS: [BrPhaseDefinition; BR_PHASE_COUNT] = [
    definition("br.total", None),
    definition("br.input-preparation", Some("br.total")),
    definition("br.input-preparation.bit-oracle", Some("br.input-preparation")),
    definition("br.input-preparation.trace-mle", Some("br.input-preparation")),
    definition("br.main-commit", Some("br.total")),
    definition("br.hadamard", Some("br.total")),
    definition("br.hadamard.base-to-extension", Some("br.hadamard")),
    definition("br.hadamard.instances", Some("br.hadamard")),
    definition("br.hadamard.batch-coefficients-kernel", Some("br.hadamard")),
    definition("br.hadamard.sumcheck", Some("br.hadamard")),
    definition("br.hadamard.fast-evaluate", Some("br.hadamard")),
    definition("br.hadamard.terminal-evaluations", Some("br.hadamard")),
    definition("br.ntt-equality", Some("br.total")),
    definition("br.ntt-equality.sparse-instance", Some("br.ntt-equality")),
    definition("br.ntt-equality.bit-pack-evaluate", Some("br.ntt-equality")),
    definition("br.ntt-equality.key-pack-evaluate", Some("br.ntt-equality")),
    definition("br.ntt-equality.batched-sumcheck-terminal", Some("br.ntt-equality")),
    definition("br.open.trace", Some("br.total")),
    definition("br.open.preprocessed-bsk", Some("br.total")),
    definition("br.sparse", Some("br.total")),
    definition("br.sparse.instance", Some("br.sparse")),
    definition("br.sparse.lookup", Some("br.sparse")),
    definition("br.sparse.lookup.witness-commit", Some("br.sparse.lookup")),
    definition("br.sparse.lookup.helper-commit", Some("br.sparse.lookup")),
    definition("br.sparse.lookup.sumcheck", Some("br.sparse.lookup")),
    definition("br.sparse.sumcheck", Some("br.sparse")),
    definition("br.accumulator", Some("br.total")),
    definition("br.accumulator.trace", Some("br.accumulator")),
    definition("br.accumulator.lookup", Some("br.accumulator")),
    definition("br.accumulator.lookup.witness-commit", Some("br.accumulator.lookup")),
    definition("br.accumulator.lookup.helper-commit", Some("br.accumulator.lookup")),
    definition("br.accumulator.lookup.sumcheck", Some("br.accumulator.lookup")),
    definition("br.accumulator.permutation-sumcheck", Some("br.accumulator")),
    definition("br.decomposition", Some("br.total")),
    definition("br.decomposition.trace", Some("br.decomposition")),
    definition("br.decomposition.params", Some("br.decomposition")),
    definition("br.decomposition.input-materialize", Some("br.decomposition")),
    definition("br.decomposition.input-recommit", Some("br.decomposition")),
    definition("br.decomposition.lookup-trace", Some("br.decomposition")),
    definition("br.decomposition.lookup-params", Some("br.decomposition")),
    definition("br.decomposition.lookup.witness-materialize", Some("br.decomposition")),
    definition("br.decomposition.lookup.base-commit", Some("br.decomposition")),
    definition("br.decomposition.lookup.helper-compute", Some("br.decomposition")),
    definition("br.decomposition.lookup.helper-materialize", Some("br.decomposition")),
    definition("br.decomposition.lookup.helper-commit", Some("br.decomposition")),
    definition("br.decomposition.lookup.sumcheck", Some("br.decomposition")),
    definition("br.decomposition.lookup.eval-pack", Some("br.decomposition")),
    definition("br.decomposition.lookup.base-open", Some("br.decomposition")),
    definition("br.decomposition.lookup.helper-open", Some("br.decomposition")),
    definition("br.decomposition.input-evaluations", Some("br.decomposition")),
    definition("br.decomposition.input-open", Some("br.decomposition")),
    definition("br.finalize", Some("br.total")),
];

const fn definition(id: &'static str, parent_id: Option<&'static str>) -> BrPhaseDefinition {
    BrPhaseDefinition { id, parent_id }
}

#[derive(Clone, Debug, Serialize)]
pub struct BrPhaseMeasurement {
    pub phase_id: &'static str,
    pub parent_id: Option<&'static str>,
    pub calls: u64,
    pub inclusive_ns: u128,
    pub exclusive_ns: u128,
    pub work: BTreeMap<&'static str, u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct BrProfile {
    pub enabled: bool,
    pub phases: Vec<BrPhaseMeasurement>,
}

#[derive(Default)]
struct Aggregate {
    calls: u64,
    inclusive_ns: u128,
    exclusive_ns: u128,
    work: BTreeMap<&'static str, u64>,
}

struct Frame {
    phase: BrPhase,
    started: Instant,
    child_ns: u128,
}

struct State {
    enabled: bool,
    stack: Vec<Frame>,
    aggregates: [Aggregate; BR_PHASE_COUNT],
    active_mask: u64,
}

impl Default for State {
    fn default() -> Self {
        Self {
            enabled: false,
            stack: Vec::with_capacity(12),
            aggregates: std::array::from_fn(|_| Aggregate::default()),
            active_mask: 0,
        }
    }
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(State::default()))
}

pub fn begin(enabled: bool) {
    let mut state = state().lock().expect("BR profiler mutex poisoned");
    *state = State::default();
    state.enabled = enabled;
}

pub fn finish() -> BrProfile {
    let mut state = state().lock().expect("BR profiler mutex poisoned");
    assert!(state.stack.is_empty(), "BR profiler has unfinished scopes");
    let enabled = state.enabled;
    state.enabled = false;
    state.active_mask = 0;
    let phases = state
        .aggregates
        .iter()
        .enumerate()
        .filter(|(_, aggregate)| aggregate.calls != 0 || !aggregate.work.is_empty())
        .map(|(index, aggregate)| BrPhaseMeasurement {
            phase_id: BR_PHASE_DEFINITIONS[index].id,
            parent_id: BR_PHASE_DEFINITIONS[index].parent_id,
            calls: aggregate.calls,
            inclusive_ns: aggregate.inclusive_ns,
            exclusive_ns: aggregate.exclusive_ns,
            work: aggregate.work.clone(),
        })
        .collect();
    BrProfile { enabled, phases }
}

pub fn active_phase_mask() -> u64 {
    state()
        .lock()
        .expect("BR profiler mutex poisoned")
        .active_mask
}

pub fn scope(phase: BrPhase) -> BrScope {
    let mut state = state().lock().expect("BR profiler mutex poisoned");
    if !state.enabled {
        return BrScope { active: false };
    }
    state.stack.push(Frame {
        phase,
        started: Instant::now(),
        child_ns: 0,
    });
    state.active_mask |= 1u64 << phase as u8;
    BrScope { active: true }
}

pub fn lookup_scope(sparse: BrPhase, accumulator: BrPhase) -> BrScope {
    let state = state().lock().expect("BR profiler mutex poisoned");
    if !state.enabled {
        return BrScope { active: false };
    }
    let phase = if state.stack.iter().any(|frame| frame.phase == BrPhase::SparseLookup) {
        sparse
    } else if state
        .stack
        .iter()
        .any(|frame| frame.phase == BrPhase::AccumulatorLookup)
    {
        accumulator
    } else {
        return BrScope { active: false };
    };
    drop(state);
    scope(phase)
}

pub fn add_lookup_work(
    sparse: BrPhase,
    accumulator: BrPhase,
    key: &'static str,
    value: u64,
) {
    let state = state().lock().expect("BR profiler mutex poisoned");
    if !state.enabled {
        return;
    }
    let phase = if state.stack.iter().any(|frame| frame.phase == BrPhase::SparseLookup) {
        sparse
    } else if state
        .stack
        .iter()
        .any(|frame| frame.phase == BrPhase::AccumulatorLookup)
    {
        accumulator
    } else {
        return;
    };
    drop(state);
    add_work(phase, key, value);
}

pub fn add_work(phase: BrPhase, key: &'static str, value: u64) {
    let mut state = state().lock().expect("BR profiler mutex poisoned");
    if state.enabled {
        *state.aggregates[phase as usize].work.entry(key).or_default() += value;
    }
}

pub struct BrScope {
    active: bool,
}

impl Drop for BrScope {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut state = state().lock().expect("BR profiler mutex poisoned");
        let frame = state.stack.pop().expect("BR profiler scope underflow");
        let elapsed_ns = frame.started.elapsed().as_nanos();
        let exclusive_ns = elapsed_ns.saturating_sub(frame.child_ns);
        let aggregate = &mut state.aggregates[frame.phase as usize];
        aggregate.calls += 1;
        aggregate.inclusive_ns += elapsed_ns;
        aggregate.exclusive_ns += exclusive_ns;
        if let Some(parent) = state.stack.last_mut() {
            parent.child_ns += elapsed_ns;
        }
        state.active_mask = state
            .stack
            .iter()
            .fold(0u64, |mask, frame| mask | (1u64 << frame.phase as u8));
    }
}
