# Metal Backend Next Phase Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the remaining correctness and performance gaps so RustGS can treat the Metal trainer as the real production path, not just the default entrypoint.

**Architecture:** Keep `MetalTrainer` as the single top-level training path, but stop pretending the current simplified forward model is already full 3DGS. First make the current Metal path explicit about frozen rotations, then add true rotation-aware projection, then promote the native Metal forward path to the default and move the remaining CPU hot spots off the training loop while preserving small-scene parity tests.

**Tech Stack:** Rust, Candle, Metal, `objc2-metal`, `cargo test`, existing `metal_runtime`, `metal_trainer`, `metal_loss`, `analytical_backward`

---

### Task 1: Freeze Rotations In The Current Metal Path

**Files:**
- Modify: `RustGS/src/training/metal_trainer.rs`
- Modify: `RustGS/docs/plans/2026-03-23-metal-backend-next-phase.md`
- Test: `RustGS/src/training/metal_trainer.rs`

**Step 1: Write the failing config test**

Add a test that makes the frozen-rotation behavior explicit:

```rust
#[test]
fn metal_config_freezes_rotation_learning_until_projection_supports_it() {
    let effective = effective_metal_config(&TrainingConfig {
        lr_rotation: 0.25,
        ..TrainingConfig::default()
    });
    assert_eq!(effective.lr_rotation, 0.0);
}
```

**Step 2: Run the focused test to verify it fails**

Run: `cargo test -p rustgs metal_config_freezes_rotation_learning_until_projection_supports_it -- --nocapture`

Expected: FAIL because Metal config does not yet explicitly freeze rotation learning.

**Step 3: Freeze rotation learning in the effective Metal config**

Make the behavior explicit and logged:

```rust
if effective.lr_rotation != 0.0 {
    log::warn!("Metal backend currently freezes Gaussian rotations...");
    effective.lr_rotation = 0.0;
}
```

**Step 4: Remove dead rotation optimizer state**

Delete unused `lr_rot`, `m_rot`, `v_rot`, and any normalization/update code that suggests Metal is currently optimizing quaternions.

**Step 5: Run the focused tests**

Run: `cargo test -p rustgs metal_config_uses_safer_default_budget metal_config_freezes_rotation_learning_until_projection_supports_it -- --nocapture`

Expected: PASS with the new freeze behavior visible in tests.

**Step 6: Run the existing Metal regression set**

Run: `cargo test -p rustgs metal_trainer -- --nocapture`

Expected: PASS with all current Metal trainer tests still green.

**Step 7: Commit**

```bash
git add RustGS/src/training/metal_trainer.rs RustGS/docs/plans/2026-03-23-metal-backend-next-phase.md
git commit -m "refactor: freeze rotations in simplified metal trainer"
```

### Task 2: Implement Rotation-Aware Projection

**Files:**
- Modify: `RustGS/src/diff/diff_splat.rs`
- Modify: `RustGS/src/training/metal_trainer.rs`
- Modify: `RustGS/src/diff/analytical_backward.rs`
- Modify: `RustGS/src/training/metal_runtime.rs`
- Test: `RustGS/src/diff/diff_splat.rs`
- Test: `RustGS/src/training/metal_trainer.rs`

**Step 1: Write the failing projection test**

Add a test that proves changing rotation changes the projected footprint:

```rust
#[test]
fn projected_footprint_changes_with_rotation() {
    assert!((sigma_x_a - sigma_x_b).abs() > 1e-4 || (sigma_y_a - sigma_y_b).abs() > 1e-4);
}
```

**Step 2: Run the focused test to verify it fails**

Run: `cargo test -p rustgs projected_footprint_changes_with_rotation -- --nocapture`

Expected: FAIL because the current Metal path ignores rotations.

**Step 3: Replace the simplified `sigma_x/sigma_y` projection**

Compute a rotation-aware 3D covariance from quaternion + scale, then project it into a 2D ellipse instead of using independent axis-aligned scales.

**Step 4: Thread the new projected representation through the trainer**

Update `ProjectedGaussians`, tile data packing, and forward-intermediate capture so the projected footprint matches the new covariance math.

**Step 5: Extend analytical backward for the new projection**

Do not pretend rotation is learnable until this chain rule exists. Add the missing derivatives for rotation-aware projection and keep small-scene finite-difference parity tests.

**Step 6: Add trainer-side rotation sensitivity coverage**

Add a Metal trainer test that proves two scenes differing only by Gaussian rotation no longer produce identical projected footprints.

**Step 7: Run the focused projection tests**

Run: `cargo test -p rustgs rotation projected_footprint_changes_with_rotation -- --nocapture`

Expected: PASS for the new projection and trainer coverage.

**Step 8: Run the broader Metal suite**

Run: `cargo test -p rustgs metal_trainer -- --nocapture`

Expected: PASS with parity tests updated for the new projection model.

**Step 9: Commit**

```bash
git add RustGS/src/diff/diff_splat.rs RustGS/src/training/metal_trainer.rs RustGS/src/diff/analytical_backward.rs RustGS/src/training/metal_runtime.rs
git commit -m "feat: add rotation-aware metal projection"
```

### Task 3: Make Native Metal Forward The Default Path

**Files:**
- Modify: `RustGS/src/training/metal_trainer.rs`
- Modify: `RustGS/src/training/mod.rs`
- Modify: `RustGS/src/bin/rustgs.rs`
- Test: `RustGS/src/training/metal_trainer.rs`

**Step 1: Write the failing config/default-path test**

Add a test that makes the intended default explicit:

```rust
#[test]
fn metal_config_defaults_to_native_forward() {
    let config = TrainingConfig::default();
    assert!(config.metal_use_native_forward);
}
```

**Step 2: Run the focused test to verify it fails**

Run: `cargo test -p rustgs metal_config_defaults_to_native_forward -- --nocapture`

Expected: FAIL because the config flag does not yet exist.

**Step 3: Add an explicit forward-path switch**

Introduce a config knob with a safe fallback:

```rust
pub metal_use_native_forward: bool,
```

Default it to `true`, but allow CLI override for debugging and bisecting regressions.

**Step 4: Route normal Metal rendering through the native kernel**

Change `render()` so Metal devices use `runtime.rasterize_forward(...)` as the main path, while CPU or explicit fallback settings still call the current tensor path.

**Step 5: Keep parity verification as a separate debug-only path**

Retain the current `profile_native_forward()` comparison, but only for profiling or assertions. Do not run both paths every step once native forward is the default.

**Step 6: Add the fallback smoke test**

Add a test that proves CPU still uses the tensor fallback and Metal uses the native path when enabled.

**Step 7: Run the forward-path tests**

Run: `cargo test -p rustgs native_forward -- --nocapture`

Expected: PASS for default-path and fallback tests.

**Step 8: Run the broader Metal suite**

Run: `cargo test -p rustgs metal_trainer -- --nocapture`

Expected: PASS with `native_forward_matches_baseline_on_tiny_scene` still green.

**Step 9: Commit**

```bash
git add RustGS/src/training/metal_trainer.rs RustGS/src/training/mod.rs RustGS/src/bin/rustgs.rs
git commit -m "feat: default metal trainer to native forward"
```

### Task 4: Move Visible-Set Sorting And Tile Binning Off CPU

**Files:**
- Modify: `RustGS/src/training/metal_runtime.rs`
- Modify: `RustGS/src/training/metal_trainer.rs`
- Test: `RustGS/src/training/metal_trainer.rs`

**Step 1: Write the failing no-roundtrip test**

Add a profiling-oriented test that asserts the visible-set builder does not call `to_vec1()` in the normal Metal path. The simplest way is to extract CPU helpers first, then assert the new path bypasses them.

**Step 2: Run the focused test to verify it fails**

Run: `cargo test -p rustgs metal_visible_set_stays_on_device -- --nocapture`

Expected: FAIL because visible sort and tile binning still pull tensors to CPU.

**Step 3: Add a device-side visible index / tile-list staging API**

Extend `metal_runtime.rs` with persistent buffers for visible indices and tile assignments:

```rust
enum MetalBufferSlot {
    VisibleIndices,
    TileCounts,
    TileOffsets,
    TileIndices,
    // existing slots...
}
```

**Step 4: Move visible sorting to a dedicated runtime stage**

Replace the current CPU `visible.to_vec1()` and `depth.to_vec1()` path with a Metal-owned visible-set stage. If a fully sorted GPU path is too large for one patch, first move mask compaction and leave only a tiny CPU fallback behind a debug switch.

**Step 5: Move tile binning to runtime-managed buffers**

Replace `build_tile_bins()` CPU vectors with runtime-managed tile metadata and packed tile-index buffers so the normal forward path stays on device.

**Step 6: Update profiling output**

Make the step log clearly show visible count, tile count, and tile references from the new runtime path so perf deltas remain measurable.

**Step 7: Run the targeted regression tests**

Run: `cargo test -p rustgs tile_bins_only_include_overlapping_gaussians -- --nocapture`

Expected: PASS with matching tile coverage semantics.

**Step 8: Run the full Metal suite**

Run: `cargo test -p rustgs metal_trainer -- --nocapture`

Expected: PASS with parity tests still green.

**Step 9: Commit**

```bash
git add RustGS/src/training/metal_runtime.rs RustGS/src/training/metal_trainer.rs
git commit -m "feat: move metal visible set and tile binning off cpu"
```

### Task 5: Introduce GPU Backward Kernels And Remove CPU ForwardIntermediate

**Files:**
- Create: `RustGS/src/training/metal_backward.rs`
- Modify: `RustGS/src/training/mod.rs`
- Modify: `RustGS/src/training/metal_runtime.rs`
- Modify: `RustGS/src/training/metal_trainer.rs`
- Test: `RustGS/src/training/metal_trainer.rs`

**Step 1: Write the failing gradient-parity test**

Add a tiny-scene test that compares GPU backward outputs against the current analytical backward within tolerance:

```rust
#[test]
fn metal_backward_matches_cpu_reference_on_tiny_scene() {
    assert!(max_abs_diff < 1e-4);
}
```

**Step 2: Run the focused test to verify it fails**

Run: `cargo test -p rustgs metal_backward_matches_cpu_reference_on_tiny_scene -- --nocapture`

Expected: FAIL because there is no GPU backward module yet.

**Step 3: Create a dedicated Metal backward module**

Move kernel-specific gradient code out of `metal_trainer.rs` into `metal_backward.rs` so trainer orchestration stays readable:

```rust
pub(crate) struct MetalBackwardGrads {
    pub positions: Tensor,
    pub log_scales: Tensor,
    pub rotations: Tensor,
    pub opacity_logits: Tensor,
    pub colors: Tensor,
}
```

**Step 4: Implement position/scale/opacity/color kernels first**

Keep scope tight: ship the existing gradient set on GPU before adding more ambitious refinements.

**Step 5: Swap trainer step to GPU backward**

Replace:

```rust
let intermediate = self.build_forward_intermediate(&projected, &rendered)?;
let analytical_grads = analytical_backward::backward_weighted_l1(...);
```

with a runtime/kernel call that returns device tensors directly.

**Step 6: Delete or isolate the CPU-only intermediate builder**

Once parity is established, remove `build_forward_intermediate()` from the normal path. Keep it only behind a debug reference path if it still helps for diagnostics.

**Step 7: Run the focused backward tests**

Run: `cargo test -p rustgs metal_backward -- --nocapture`

Expected: PASS for gradient-parity tests.

**Step 8: Run the full Metal suite**

Run: `cargo test -p rustgs metal_trainer metal_loss -- --nocapture`

Expected: PASS with no regression in loss or trainer tests.

**Step 9: Commit**

```bash
git add RustGS/src/training/metal_backward.rs RustGS/src/training/mod.rs RustGS/src/training/metal_runtime.rs RustGS/src/training/metal_trainer.rs
git commit -m "feat: add gpu backward path for metal trainer"
```

### Task 6: Tighten Topology Scheduling, Throughput Guardrails, And Quality Gates

**Files:**
- Modify: `RustGS/src/training/mod.rs`
- Modify: `RustGS/src/training/metal_trainer.rs`
- Modify: `RustGS/src/bin/rustgs.rs`
- Modify: `RustGS/docs/plans/2026-03-22-metal-performance-roadmap.md`
- Test: `RustGS/src/training/metal_trainer.rs`

**Step 1: Write the failing scheduling test**

Add a test that proves prune cadence is independent from densify cadence:

```rust
#[test]
fn prune_interval_is_independent_from_densify_interval() {
    assert_eq!(trainer.should_prune_at(200), true);
    assert_eq!(trainer.should_densify_at(200), false);
}
```

**Step 2: Run the focused test to verify it fails**

Run: `cargo test -p rustgs prune_interval_is_independent_from_densify_interval -- --nocapture`

Expected: FAIL because prune is currently tied to `densify_interval`.

**Step 3: Add explicit topology controls**

Introduce only the config we need:

```rust
pub prune_interval: usize,
pub topology_warmup: usize,
pub topology_log_interval: usize,
```

Use them to avoid early clone/prune churn and to make logs readable on long runs.

**Step 4: Add throughput guardrails to topology updates**

Log per-update cost, cap changes per pass, and surface whether topology work is eating too much step time.

**Step 5: Add one benchmark/smoke command to the docs**

Document the minimum smoke run we use after performance changes:

```bash
cargo test -p rustgs metal_trainer -- --nocapture
```

and one real-scene profiling run through the CLI with `--metal-profile-steps`.

**Step 6: Update the roadmap status notes**

Mark which roadmap items are done, which are partially done, and which remain blocked by GPU backward work.

**Step 7: Run the topology and trainer suite**

Run: `cargo test -p rustgs metal_trainer -- --nocapture`

Expected: PASS with existing topology regression tests still green.

**Step 8: Commit**

```bash
git add RustGS/src/training/mod.rs RustGS/src/training/metal_trainer.rs RustGS/src/bin/rustgs.rs RustGS/docs/plans/2026-03-22-metal-performance-roadmap.md
git commit -m "feat: tighten metal topology scheduling and guardrails"
```

## Recommended Execution Order

1. Task 1: Freeze rotations in the current Metal path
2. Task 2: Implement rotation-aware projection
3. Task 3: Make native Metal forward the default path
4. Task 4: Move visible-set sorting and tile binning off CPU
5. Task 5: Introduce GPU backward kernels and remove CPU `ForwardIntermediate`
6. Task 6: Tighten topology scheduling, throughput guardrails, and quality gates

## Why This Order

- Task 1 removes a misleading intermediate state where Metal appears to learn rotations but does not use them.
- Task 2 adds the missing geometric capability needed before rotation can become a real trainable signal.
- Task 3 converts shipped Metal forward work into real default behavior.
- Task 4 removes the most obvious remaining CPU forward hot spots before backward work grows.
- Task 5 is the largest change, so it should land after the forward path is already stable.
- Task 6 prevents topology and benchmarking from becoming an afterthought after the big performance work lands.

## Definition Of Done

- The current simplified Metal path explicitly freezes rotations until rotation-aware projection lands.
- Rotation-aware projection is implemented before any Metal rotation updates are reintroduced.
- Normal Metal training uses the native forward kernel by default.
- Visible-set creation and tile binning no longer require large CPU round-trips.
- The default training step no longer builds a CPU `ForwardIntermediate`.
- Densify/prune has explicit scheduling and logging that keeps throughput measurable.
- `cargo test -p rustgs metal_trainer -- --nocapture` and `cargo test -p rustgs metal_loss -- --nocapture` stay green.
