# RustScan Architecture

**Updated:** 2026-04-08

## Overview

RustScan is a Rust workspace for 3D reconstruction tooling. The repository is no longer just `RustMesh + RustSLAM`; the current workspace contains six crates with a shared-documentation surface and a RustGS-centered active development track.

## Workspace Crates

- `rustscan-types`: shared data structures used across crates.
- `RustSLAM`: visual SLAM, sparse mapping, loop closing, dataset/video ingestion, and SLAM-side outputs.
- `RustGS`: Gaussian splatting training, evaluation, parity reporting, and chunked training orchestration.
- `RustMesh`: mesh connectivity, IO, and OpenMesh-aligned processing algorithms.
- `RustViewer`: interactive inspection UI for RustScan scene data.
- `RustFF`: feed-forward inference utilities for pose/depth-related reconstruction work.

## Cross-Crate Flow

1. `RustSLAM` or an external dataset source provides images, poses, and optional sparse geometry.
2. `RustGS` loads TUM, COLMAP, SLAM JSON, or `TrainingDataset` inputs and trains a Gaussian scene.
3. `RustGS` exports scene PLY files plus optional evaluation summaries and LiteGS parity sidecars.
4. `RustMesh` handles mesh-side processing and OpenMesh parity work when surface extraction or mesh cleanup is needed.
5. `RustViewer` loads scene/checkpoint outputs for inspection.

## Current RustGS Training Architecture

### Public Entry Surface

- `rustgs::train_from_path`
- `rustgs::train_from_slam`
- `rustgs::training::train`
- `rustgs::evaluate_scene`
- `rustgs::save_scene_ply`
- `rustgs::load_scene_ply`

These are the compatibility boundaries that current refactor work is preserving.

### Execution Planning

`RustGS/src/training/mod.rs` now acts as a thin public training assembly layer, while active orchestration and configuration have been split into narrower modules:

- `config.rs`: training enums, nested LiteGS config, public `TrainingConfig`, and `TrainingResult`
- `orchestrator.rs`: profile-aware `training::train()` entry and top-level route handoff
- `execution_plan.rs`: `Standard` / `ChunkedSingleChunk` / `ChunkedSequential` selection
- `chunk_training.rs`: sequential chunk execution, adaptive per-chunk config, scene merge
- `export.rs`: chunk artifact/report persistence

There is no active compiled `train_stream.rs` orchestration layer anymore.

### Data and Initialization

- `data_loading.rs`: dataset-to-training-data conversion
- `frame_loader.rs`: bounded frame decode/cache behavior
- `init_map.rs`: sparse-point or frame-based initialization
- `chunk_planner.rs`: chunk planning and per-chunk dataset materialization
- `splats.rs`: internal unified training-state representation plus conversion bridge for `GaussianMap <-> TrainableGaussians` and export-scene conversion
- `LoadedTrainingData` now carries `initial_splats` into the production and benchmark training paths instead of exposing a loader-owned `initial_map`

### Canonical State Roles

- `GaussianMap` remains the public scene IO and final return boundary.
- `Splats` is the canonical internal snapshot and exchange type for initialization, scene/export conversion, and explicit `GaussianMap <-> TrainableGaussians` boundary crossings.
- `TrainableGaussians` remains the canonical mutable step-loop state consumed by forward, backward, optimizer, and Metal runtime code.

### Step Execution

The production Metal path is split into explicit subsystems:

- `metal_forward.rs`
- `metal_loss.rs`
- `metal_backward.rs`
- `metal_optimizer.rs`
- `metal_trainer.rs`

`metal_trainer.rs` remains the lifecycle coordinator for the step loop, but it no longer owns the full forward/backward/runtime implementation inline.

### Metal Runtime Layer

The old monolithic runtime has been split into concrete modules:

- `metal_kernels.rs`
- `metal_pipelines.rs`
- `metal_resources.rs`
- `metal_dispatch.rs`
- `metal_projection.rs`
- `metal_raster.rs`
- `metal_runtime.rs` as the shared facade and compatibility layer

Shader source now lives in `RustGS/src/training/shaders/*.metal`.

### Topology and Parity

- `topology.rs` owns schedule calculation, execution planning, and snapshot mutation helpers for densify/prune/reset behavior.
- `density_controller.rs` remains as reference/parity-sensitive logic and is now adapted back into `topology.rs` through an explicit reference adapter for LiteGS telemetry and regression work.
- `parity_harness.rs` owns LiteGS comparison reports and parity gating utilities.
- `eval.rs` owns evaluation summaries and post-train metrics.

### Legacy Boundary

`training_pipeline.rs` remains in-tree only as a legacy/reference helper surface. It is not the landing zone for new production behavior, and its helpers are no longer re-exported from `training/mod.rs`; callers must import it explicitly when they really need the legacy surface.

## Current Architectural Constraints

- Public training, evaluation, scene IO, and chunked-routing behavior must remain stable while internals move.
- The production backend is still Metal-specific; no new multi-backend abstraction is planned unless the extracted code shape proves a concrete need.
- Topology-side side effects are partially migrated: scheduling and snapshot mutation moved into `topology.rs`, while some trainer-coupled state updates still live in `metal_trainer.rs`.
- The remaining Epic 6.5 debt is not raw conversion duplication anymore; it is reducing how often the trainer rebuilds `Splats` snapshots around topology/export checkpoints.

## Canonical Companion Docs

- `docs/current-project-status.md`
- `docs/plans/2026-04-06-rustgs-training-execution-plan.md`
- `docs/plans/2026-04-06-rustgs-brush-refactor-review-and-epics.md`
- `docs/plans/2026-04-06-rustgs-refactor-guardrails.md`
