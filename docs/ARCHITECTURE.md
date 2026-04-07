# RustScan Architecture

**Updated:** 2026-04-07

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

`RustGS/src/training/mod.rs` now owns training entry and route selection directly.

- `Standard`
- `ChunkedSingleChunk`
- `ChunkedSequential`

There is no active compiled `train_stream.rs` orchestration layer anymore; execution planning now lives with the public training module.

### Data and Initialization

- `data_loading.rs`: dataset-to-training-data conversion
- `frame_loader.rs`: bounded frame decode/cache behavior
- `init_map.rs`: sparse-point or frame-based initialization
- `chunk_planner.rs`: chunk planning and per-chunk dataset materialization
- `splats.rs`: internal unified training-state representation used across active training internals

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
- `density_controller.rs` remains as reference/parity-sensitive logic and is not being deleted prematurely.
- `parity_harness.rs` owns LiteGS comparison reports and parity gating utilities.
- `eval.rs` owns evaluation summaries and post-train metrics.

### Legacy Boundary

`training_pipeline.rs` remains in-tree only as a legacy/reference helper surface. It is not the landing zone for new production behavior.

## Current Architectural Constraints

- Public training, evaluation, scene IO, and chunked-routing behavior must remain stable while internals move.
- The production backend is still Metal-specific; no new multi-backend abstraction is planned unless the extracted code shape proves a concrete need.
- Topology-side side effects are partially migrated: scheduling and snapshot mutation moved into `topology.rs`, while some trainer-coupled state updates still live in `metal_trainer.rs`.

## Canonical Companion Docs

- `docs/current-project-status.md`
- `docs/plans/2026-04-06-rustgs-training-execution-plan.md`
- `docs/plans/2026-04-06-rustgs-brush-refactor-review-and-epics.md`
- `docs/plans/2026-04-06-rustgs-refactor-guardrails.md`
