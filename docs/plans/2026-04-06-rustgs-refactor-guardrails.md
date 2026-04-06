# RustGS Refactor Guardrails

Related plan: `docs/plans/2026-04-06-rustgs-brush-refactor-review-and-epics.md`
Implements: Epic 1 Story 1.1, Story 1.2, Story 1.3
Date: 2026-04-06

## Purpose

This document records the current RustGS training compatibility surface and the guardrails that must stay green while the Brush-inspired refactor lands.

It exists to answer two questions:

1. What behavior is a public contract and must be preserved?
2. What commands and tests define the minimum regression baseline for refactor work?

## Public Compatibility Surface

The following APIs and flows are treated as public contract for the refactor:

### Library Entry Points

- `rustgs::train_from_slam`
- `rustgs::train_from_path`
- `rustgs::training::train`
- `rustgs::evaluate_scene`
- `rustgs::save_scene_ply`
- `rustgs::load_scene_ply`
- `rustgs::metal_available`
- `rustgs::run_metal_training_benchmark`

### CLI Surface

The primary user-facing CLI contract is `rustgs train`, including:

- dataset input resolution from TUM, COLMAP, `SlamOutput` JSON, or `TrainingDataset` JSON
- `TrainingProfile` selection
- chunked training flags
- post-train evaluation flags
- LiteGS parity report emission for `litegs-mac-v1`

### Scene Export and Evaluation Surface

The refactor must preserve:

- PLY scene export via `save_scene_ply`
- PLY scene load via `load_scene_ply`
- evaluation summary fields produced by `evaluate_scene`
- parity-report sidecar behavior for LiteGS runs

### Chunked Training Surface

The refactor must preserve current route semantics for:

- standard training
- chunked single-chunk pass-through
- sequential chunked execution
- chunk artifact directory and report behavior

## Internal Implementation Details

The following are internal implementation details and may change as long as the public compatibility surface above remains stable:

- `metal_trainer.rs` layout and helper structure
- `metal_runtime.rs` decomposition
- `data_loading.rs`, `frame_loader.rs`, and `init_map.rs` internals
- `topology.rs` and `density_controller.rs` ownership boundaries
- optimizer state storage layout
- forward and backward module boundaries

## Current Training Flow Contract

The current top-level training contract is:

1. CLI or library entry resolves a `TrainingDataset`
2. `TrainingConfig` selects profile and chunking behavior
3. `training::train()` routes to the Metal-backed trainer path
4. the trainer returns a `GaussianMap`
5. the caller exports a scene PLY and optional parity sidecar
6. optional post-train evaluation produces a structured summary

Refactor work may reorganize the internals behind each step, but it must not silently change the shape or ownership of this flow.

## Refactor Baseline Commands

These commands are the minimum runnable baseline for Epic 1.

### 1. TUM Dataset Load and Smoke Integration Tests

```bash
cargo test --manifest-path RustGS/Cargo.toml --features gpu --test tum_training -- --nocapture
```

This covers:

- TUM dataset discovery
- evaluation-frame selection stability
- direct train-from-path smoke
- post-train evaluation-summary smoke

### 2. Execution-Plan and Chunk-Route Regression

```bash
cargo test --manifest-path RustGS/Cargo.toml --features gpu execution_plan -- --nocapture
```

This covers:

- standard route selection
- chunked route selection
- adaptive chunk-budget behavior

### 3. Optimizer and Topology Regression Coverage

```bash
cargo test --manifest-path RustGS/Cargo.toml --features gpu project_gaussians_handles_zero_visible_without_index_select -- --nocapture
cargo test --manifest-path RustGS/Cargo.toml --features gpu apply_backward_grads_sparse_grad_preserves_invisible_rows_and_moments -- --nocapture
cargo test --manifest-path RustGS/Cargo.toml --features gpu rebuild_adam_state_preserves_reordered_rows -- --nocapture
cargo test --manifest-path RustGS/Cargo.toml --features gpu topology_update_densifies_and_prunes_with_matching_adam_state -- --nocapture
```

These cover the critical trainer-path invariants used as refactor guardrails.

### 4. Benchmark Harness Baseline

```bash
cargo run --manifest-path RustGS/Cargo.toml --features gpu --example training_benchmark -- --profile litegs-mac-v1 --json
```

This provides a structured timing baseline for:

- average forward time
- average loss time
- average backward time
- average optimizer time
- average end-to-end step time
- smoke training wall-clock timing

### 5. LiteGS Parity Report Baseline

```bash
cargo run --manifest-path RustGS/Cargo.toml --features gpu --bin rustgs -- train \
  --input test_data/tum \
  --output /tmp/rustgs-epic1-smoke.ply \
  --training-profile litegs-mac-v1 \
  --iterations 1 \
  --max-frames 90 \
  --frame-stride 30 \
  --eval-after-train \
  --eval-render-scale 0.25 \
  --eval-max-frames 90 \
  --eval-frame-stride 30
```

Expected outputs:

- scene PLY
- optional post-train evaluation summary
- `.parity.json` sidecar for the LiteGS profile

## Regression Coverage Map

The following tests define the current trainer-path guardrails.

### Forward Output and Visibility

- `project_gaussians_handles_zero_visible_without_index_select`
- `project_gaussians_keeps_distinct_visible_indices_on_metal`
- `project_gaussians_applies_cluster_visible_mask_on_metal`

### Backward Gradient Plumbing

- `apply_backward_grads_updates_rotations_when_rotation_grad_is_present`
- `apply_backward_grads_sparse_grad_preserves_invisible_rows_and_moments`
- `apply_backward_grads_sparse_grad_noops_when_no_gaussians_are_visible`
- `apply_backward_grads_dense_updates_metal_params`

### Optimizer State Rebuild and Row Remap

- `rebuild_adam_state_preserves_reordered_rows`
- `topology_update_densifies_and_prunes_with_matching_adam_state`

### Topology-Triggered Parameter Mutation

- `topology_updates_can_grow_beyond_initial_gaussian_count_limit`
- `topology_updates_preserve_sh_representation_for_litegs_trainables`
- `topology_update_densifies_and_prunes_with_matching_adam_state`

### Execution-Plan Route Guardrails

- `non_chunked_execution_plan_uses_standard_route`
- `chunked_execution_plan_selects_single_chunk_route_when_affordable`
- `chunked_execution_plan_uses_sequential_route_when_subdivision_is_required`
- `adaptive_chunk_config_lowers_gaussian_cap_before_render_scale`
- `adaptive_chunk_config_reduces_render_scale_when_gaussian_cap_is_not_enough`

### TUM Smoke and Evaluation Guardrails

- `loads_workspace_tum_directory_as_training_dataset`
- `selects_stable_tum_eval_subset_with_stride`
- `trains_directly_from_workspace_tum_directory`
- `tum_training_smoke_produces_post_train_evaluation_summary`

### Benchmark Harness Guardrails

- `benchmark_config_disables_topology_for_guardrail_runs`
- `synthetic_loaded_training_data_matches_requested_fixture_shape`
- `benchmark_rejects_invalid_spec_before_requesting_metal`

## Refactor Rules

During the refactor:

- do not remove or rename a public entry point without an explicit migration note
- do not change scene export or evaluation summary shape silently
- do not merge or delete topology helpers until the guardrail tests above still pass
- do not claim a benchmark baseline exists unless the benchmark example compiles and runs
- do not treat parity reporting as optional for LiteGS profile validation

## Notes

- `metal_available()` is intentionally public because smoke tests and examples need a stable capability probe.
- `run_metal_training_benchmark()` is intentionally public because the benchmark example is part of the Epic 1 baseline.
- The existence of this document does not replace tests; it identifies which tests are the current refactor gate.
