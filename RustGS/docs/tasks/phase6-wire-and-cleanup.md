# Phase 6: Wire + Delete Old Code

**Status**: Ready for Implementation  
**Dependencies**: Phase 5 (Training Pipeline) must be complete  
**Estimated Complexity**: Medium

## Overview

Connect the new burn+wgpu backend to the public API, remove all Metal-specific code, and clean up dependencies. This phase completes the migration by making wgpu the default and only GPU backend.

## Architecture

```
Public API (unchanged signatures)
    ↓
orchestrator.rs (route to wgpu)
    ↓
training/wgpu/mod.rs (train_splats entry point)
    ↓
WgpuTrainer (Phase 5 implementation)
```

## Reference Implementation

- **Migration Plan**: `RustGS/docs/plans/2026-04-12-burn-wgpu-migration-plan.md` Section 10 (Phase 6-7)
- **Existing orchestrator**: `RustGS/src/training/pipeline/orchestrator.rs`
- **Existing config**: `RustGS/src/training/config.rs`

## Files to Modify

### Part A: Update orchestrator.rs

**File**: `RustGS/src/training/pipeline/orchestrator.rs`

Replace Metal routing with wgpu routing.

**Current structure** (lines 1-100):
```rust
use super::metal::entry as metal_entry;

pub fn train_splats_with_events<F>(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    mut on_event: F,
) -> Result<TrainingRun, TrainingError>
where
    F: FnMut(TrainingEvent),
{
    let run = match config.training_profile {
        TrainingProfile::LegacyMetal => train_legacy_metal(dataset, config, &mut on_event),
        TrainingProfile::LiteGsMacV1 => train_litegs_mac_v1(dataset, config, &mut on_event),
    }?;
    // ...
}

fn train_legacy_metal(...) -> Result<TrainingRun, TrainingError> {
    metal_entry::train_splats_with_report(dataset, config)
}

fn train_litegs_mac_v1(...) -> Result<TrainingRun, TrainingError> {
    // validation + metal_entry call
}
```

**New structure**:
```rust
use super::wgpu::train_splats_with_report as wgpu_train_splats_with_report;

pub fn train_splats_with_events<F>(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    mut on_event: F,
) -> Result<TrainingRun, TrainingError>
where
    F: FnMut(TrainingEvent),
{
    store_last_training_telemetry(None);
    emit_training_event(
        &mut on_event,
        TrainingEvent::RunStarted(TrainingRunStarted {
            profile: config.training_profile,
            iterations: config.iterations,
            frame_count: dataset.poses.len(),
            input_point_count: dataset.initial_points.len(),
        }),
    );

    // Route all profiles to wgpu backend
    let run = wgpu_train_splats_with_report(dataset, config)?;

    store_last_training_telemetry(run.report.telemetry.clone());
    emit_training_event(
        &mut on_event,
        TrainingEvent::RunCompleted(TrainingRunCompleted {
            report: run.report.clone(),
        }),
    );
    Ok(run)
}
```

**Key changes**:
- Remove `train_legacy_metal` and `train_litegs_mac_v1` functions
- Remove profile-based routing (wgpu handles all profiles)
- Update imports: `use super::wgpu::train_splats_with_report`
- Rename `store_last_metal_training_telemetry` → `store_last_training_telemetry` (or keep name, update impl)

---

### Part B: Update config.rs

**File**: `RustGS/src/training/config.rs`

Update TrainingBackend enum and remove Metal-specific config fields.

**Changes**:

1. **TrainingBackend enum**:
```rust
// Old:
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingBackend {
    Metal,
}

// New:
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingBackend {
    Wgpu,
}

impl Default for TrainingBackend {
    fn default() -> Self {
        Self::Wgpu
    }
}
```

2. **TrainingProfile enum** (optional - can keep for backward compat):
```rust
// Keep existing:
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingProfile {
    LegacyMetal,  // Keep name for serde compat, maps to wgpu internally
    LiteGsMacV1,  // Keep name for serde compat, maps to wgpu internally
}

// Or rename:
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingProfile {
    #[serde(alias = "legacy_metal")]
    Default,
    #[serde(alias = "litegs_mac_v1")]
    LiteGs,
}
```

**Recommendation**: Keep old names with serde aliases for config file compatibility.

3. **Remove Metal-specific fields** from `TrainingConfig`:
```rust
// Remove these fields:
pub metal_render_scale: f32,
pub metal_gaussian_batch_size: usize,
pub metal_profile_steps: bool,
pub metal_profile_interval: usize,
pub metal_use_native_forward: bool,

// Replace with (if needed):
pub render_scale: f32,
pub gpu_batch_size: usize,
pub profile_steps: bool,
pub profile_interval: usize,
```

4. **Update Default impl** to remove metal_ prefixed fields.

---

### Part C: Update training/mod.rs

**File**: `RustGS/src/training/mod.rs`

Remove metal module, export wgpu instead.

**Current** (lines 19-89):
```rust
#[cfg(feature = "gpu")]
mod metal;

#[cfg(feature = "gpu")]
pub use metal::trainer::MetalTrainer;

#[cfg(feature = "gpu-wgpu")]
pub mod wgpu;
```

**New**:
```rust
#[cfg(feature = "gpu")]
pub mod wgpu;

#[cfg(feature = "gpu")]
pub use wgpu::WgpuTrainer;
```

**Key changes**:
- Remove `mod metal;` line
- Remove `pub use metal::trainer::MetalTrainer;`
- Change `#[cfg(feature = "gpu-wgpu")]` → `#[cfg(feature = "gpu")]` for wgpu module
- Export `WgpuTrainer` (if public API needs it)

---

### Part D: Update lib.rs

**File**: `RustGS/src/lib.rs`

Remove Metal device detection, update to wgpu.

**Find and replace**:
```rust
// Old:
pub fn metal_available() -> bool {
    // Metal detection logic
}

pub(crate) fn preferred_device() -> candle_core::Device {
    // Metal device selection
}

// New:
pub fn gpu_available() -> bool {
    // wgpu is always available (software fallback)
    true
}

pub(crate) fn preferred_wgpu_device() -> crate::training::wgpu::GsDevice {
    crate::training::wgpu::GsDevice::default()
}
```

**Note**: Search for all references to `metal_available()` and `preferred_device()` and update call sites.

---

### Part E: Update wgpu/mod.rs exports

**File**: `RustGS/src/training/wgpu/mod.rs`

Ensure public API functions are exported.

**Add** (if not already present from Phase 5):
```rust
//! Burn+wgpu training backend

pub mod backend;
pub mod gpu_primitives;
pub mod render;
pub mod render_bwd;
pub mod splats;
pub mod optimizer;
pub mod loss;
pub mod topology_bridge;
pub mod topology_apply;
pub mod trainer;

pub use backend::{GsBackend, GsBackendBase, GsDevice, GsDiffBackend};
pub use splats::{device_splats_to_host, host_splats_to_device, DeviceSplats};
pub use trainer::WgpuTrainer;

// Public entry points matching orchestrator expectations
pub use trainer::{train_splats, train_splats_with_report, train_splats_with_events};
```

**Note**: Ensure Phase 5 implemented these entry points in `trainer.rs` or create a separate `entry.rs`.

---

## Files to Delete

### Part F: Remove Metal backend

**Delete entire directories**:
```bash
rm -rf RustGS/src/training/metal/
rm -rf RustGS/src/training/shaders/*.metal
rm -rf RustGS/src/diff/
```

**Specific files to delete**:

1. **Metal trainer** (19 files):
   - `src/training/metal/mod.rs`
   - `src/training/metal/trainer/mod.rs`
   - `src/training/metal/trainer/session.rs`
   - `src/training/metal/trainer/step.rs`
   - `src/training/metal/trainer/support.rs`
   - `src/training/metal/trainer/topology_impl.rs`
   - `src/training/metal/trainer/tests.rs`
   - `src/training/metal/entry.rs`
   - `src/training/metal/kernels.rs`
   - `src/training/metal/projection.rs`
   - `src/training/metal/pipelines.rs`
   - `src/training/metal/loss.rs`
   - `src/training/metal/optimizer.rs`
   - `src/training/metal/dispatch.rs`
   - `src/training/metal/forward.rs`
   - `src/training/metal/memory.rs`
   - `src/training/metal/resources.rs`
   - `src/training/metal/raster.rs`
   - `src/training/metal/runtime.rs`
   - `src/training/metal/backward.rs`

2. **Metal shaders** (7 files):
   - `src/training/shaders/fill_u32.metal`
   - `src/training/shaders/forward_raster.metal`
   - `src/training/shaders/projection.metal`
   - `src/training/shaders/tile_binning.metal`
   - `src/training/shaders/gradients.metal`
   - `src/training/shaders/adam.metal`
   - `src/training/shaders/backward_raster.metal`

3. **Diff module** (4 files):
   - `src/diff/mod.rs`
   - `src/diff/diff_renderer.rs`
   - `src/diff/analytical_backward.rs`
   - `src/diff/diff_splat.rs`

**Total**: 30 files to delete

---

### Part G: Update Cargo.toml

**File**: `RustGS/Cargo.toml`

Remove candle dependencies, update features.

**Remove dependencies**:
```toml
# Delete these lines:
candle-core = { version = "0.9.2", features = ["metal"], optional = true }
candle-metal = { version = "0.27.1", features = ["mps"], optional = true }
candle-metal-kernels = { version = "0.9.2", optional = true }
objc2-foundation = { version = "0.3.2", optional = true }
objc2-metal = { version = "0.3.2", optional = true }
```

**Update features**:
```toml
[features]
default = ["gpu", "cli"]

# Old:
gpu = [
    "dep:candle-core",
    "dep:candle-metal",
    "dep:candle-metal-kernels",
    "dep:objc2-foundation",
    "dep:objc2-metal",
]

# New:
gpu = [
    "dep:burn",
    "dep:burn-wgpu",
    "dep:burn-cubecl",
    "dep:wgpu",
    "dep:naga_oil",
    "dep:bytemuck",
    "dep:tokio",
]

# Remove gpu-wgpu feature (merge into gpu):
# gpu-wgpu = [...] # DELETE THIS
```

**Verify burn dependencies** are marked `optional = true`:
```toml
burn = { git = "https://github.com/tracel-ai/burn", default-features = false, features = ["autodiff", "wgpu"], optional = true }
burn-wgpu = { git = "https://github.com/tracel-ai/burn", default-features = false, features = ["exclusive-memory-only", "template", "cubecl-wgsl"], optional = true }
burn-cubecl = { git = "https://github.com/tracel-ai/burn", optional = true }
wgpu = { version = "29", default-features = false, features = ["naga-ir"], optional = true }
naga_oil = { version = "0.22", optional = true }
bytemuck = { version = "1.20", features = ["derive"], optional = true }
tokio = { version = "1", features = ["rt"], optional = true }
```

---

## Integration Points

### With Phase 5 (Training Pipeline)
- Phase 5 must export `train_splats`, `train_splats_with_report`, `train_splats_with_events`
- These functions must match the signatures expected by orchestrator.rs
- WgpuTrainer must handle both LegacyMetal and LiteGsMacV1 profiles

### With Public API
- `train_splats()` signature unchanged: `(dataset, config) -> Result<HostSplats, TrainingError>`
- `train_splats_with_report()` signature unchanged: `(dataset, config) -> Result<TrainingRun, TrainingError>`
- `train_splats_with_events()` signature unchanged: `(dataset, config, callback) -> Result<TrainingRun, TrainingError>`

### With Config System
- Existing TOML config files should continue to work
- `training_backend = "metal"` → map to wgpu internally or error with migration message
- `training_profile = "legacy_metal"` → map to wgpu default profile
- `training_profile = "litegs_mac_v1"` → map to wgpu litegs profile

---

## Testing Strategy

1. **Compilation**:
   - `cargo check --features gpu` (no errors)
   - `cargo build --features gpu` (successful build)
   - `cargo build --release --features gpu` (optimized build)

2. **API compatibility**:
   - Verify public API signatures unchanged
   - Test with existing config files
   - Ensure TrainingRun/TrainingEvent structures compatible

3. **Functional**:
   - Run small training test (10 iterations)
   - Verify PLY output format unchanged
   - Check TrainingRun report structure

4. **Cleanup verification**:
   - `git status` shows all Metal files deleted
   - No references to `candle-core` in codebase: `rg "candle_core|candle::"`
   - No references to Metal: `rg "metal::" --type rust`
   - No `.metal` files: `find . -name "*.metal"`

---

## Acceptance Criteria

- [ ] orchestrator.rs routes to wgpu backend
- [ ] config.rs updated (TrainingBackend::Wgpu, metal_ fields removed)
- [ ] training/mod.rs exports wgpu, removes metal
- [ ] lib.rs updated (gpu_available, preferred_wgpu_device)
- [ ] wgpu/mod.rs exports public API functions
- [ ] All 30 Metal/diff files deleted
- [ ] Cargo.toml: candle deps removed, gpu feature updated
- [ ] `cargo check --features gpu` passes
- [ ] `cargo build --features gpu` succeeds
- [ ] No references to candle/metal in codebase
- [ ] Public API signatures unchanged

---

## Migration Notes

### Backward Compatibility

**Config files**: Users with existing `rustscan.toml` files may have:
```toml
training_backend = "metal"
training_profile = "legacy_metal"
```

**Options**:
1. **Error with migration message**:
   ```rust
   if config.training_backend == TrainingBackend::Metal {
       return Err(TrainingError::Config(
           "Metal backend removed in v0.x. Please update config to use 'wgpu' backend.".into()
       ));
   }
   ```

2. **Auto-migrate** (recommended):
   ```rust
   // In config.rs Default impl or deserialization
   #[serde(default = "default_backend")]
   pub training_backend: TrainingBackend,
   
   fn default_backend() -> TrainingBackend {
       TrainingBackend::Wgpu
   }
   ```

### Profile Mapping

Both `LegacyMetal` and `LiteGsMacV1` profiles should work with wgpu backend:
- LegacyMetal → use default wgpu training parameters
- LiteGsMacV1 → use LiteGS-specific parameters (sh_degree, cluster_size, etc.)

WgpuTrainer should read `config.litegs` fields and apply them accordingly.

---

## Codex Prompt

When ready to implement, use:

```
Implement Phase 6 (Wire + Delete Old Code) for RustGS burn+wgpu migration.

Reference:
- Task doc: RustGS/docs/tasks/phase6-wire-and-cleanup.md
- Migration plan: RustGS/docs/plans/2026-04-12-burn-wgpu-migration-plan.md

Tasks:
1. Update RustGS/src/training/pipeline/orchestrator.rs - route to wgpu backend
2. Update RustGS/src/training/config.rs - TrainingBackend::Wgpu, remove metal_ fields
3. Update RustGS/src/training/mod.rs - export wgpu, remove metal
4. Update RustGS/src/lib.rs - gpu_available(), preferred_wgpu_device()
5. Update RustGS/src/training/wgpu/mod.rs - export public API functions
6. Delete 30 files: src/training/metal/, src/training/shaders/*.metal, src/diff/
7. Update RustGS/Cargo.toml - remove candle deps, update gpu feature

Key requirements:
- Public API signatures unchanged
- Config backward compatibility (auto-migrate or error with message)
- Both LegacyMetal and LiteGsMacV1 profiles work with wgpu
- Clean deletion: no candle/metal references remain

Verify with:
- cargo check --features gpu
- cargo build --features gpu
- rg "candle_core|candle::" (should be empty)
- rg "metal::" --type rust (should be empty)
- find . -name "*.metal" (should be empty)
```
