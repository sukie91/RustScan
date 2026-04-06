---
stepsCompleted: [1, 2, 3, 4]
inputDocuments:
  - /Users/tfjiang/Projects/brush/crates/brush-process/src/train_stream.rs
  - /Users/tfjiang/Projects/brush/crates/brush-dataset/src/scene_loader.rs
  - /Users/tfjiang/Projects/brush/crates/brush-train/src/train.rs
  - /Users/tfjiang/Projects/brush/crates/brush-serde/src/import.rs
  - /Users/tfjiang/Projects/brush/crates/brush-serde/src/export.rs
  - /Users/tfjiang/Projects/RustScan/RustGS/src/training/mod.rs
  - /Users/tfjiang/Projects/RustScan/RustGS/src/training/metal_trainer.rs
  - /Users/tfjiang/Projects/RustScan/RustGS/src/training/data_loading.rs
  - /Users/tfjiang/Projects/RustScan/RustGS/src/training/training_pipeline.rs
  - /Users/tfjiang/Projects/RustScan/RustGS/src/io/scene_io.rs
---

# RustScan - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for RustScan, decomposing the Brush-inspired RustGS migration plan into implementable stories. The purpose of this plan is not to port Brush into RustGS, but to migrate the parts of Brush that improve RustGS engineering quality: training orchestration, data loading, dataset onboarding, topology state management, persistence hardening, and validation infrastructure.

## Requirements Inventory

### Functional Requirements

FR1: RustGS routes training execution through a dedicated orchestration module that owns initialization, loop scheduling, evaluation, export, and telemetry.
FR2: RustGS separates step-level Metal execution from lifecycle management so `metal_trainer` focuses on forward, backward, and parameter update work.
FR3: RustGS provides a high-throughput frame loader with asynchronous prefetch, bounded caching, and randomized batch delivery for color and depth data.
FR4: RustGS preserves existing synthetic-depth fallback and current frame preprocessing semantics when moving frame ingestion out of `data_loading.rs`.
FR5: RustGS can load COLMAP datasets directly, including camera and image parsing, sparse-point initialization, and eval split support.
FR6: RustGS can load Nerfstudio datasets directly, including `transforms*.json`, optional init splats, and eval or test split support.
FR7: RustGS supports dataset-local training configuration overlays that can be overridden from the CLI without code changes.
FR8: RustGS extracts initial-map construction, topology decisions, and optimizer-state reshaping into focused modules that preserve current LiteGS and Legacy semantics.
FR9: RustGS provides richer scene persistence with versioned metadata, dynamic SH-aware import and export, and round-trip safety.
FR10: RustGS adds training, loader, persistence, and topology regression coverage together with benchmark harnesses for forward, backward, and training smoke runs.
FR11: RustGS preserves current non-chunked, chunked, `LegacyMetal`, and `LiteGsMacV1` entry behavior while the migration lands incrementally.
FR12: Each migration increment is shippable as a standalone story that a single dev agent can complete without waiting for future stories in the same epic.

### NonFunctional Requirements

NFR1: Keep Metal and Candle as the runtime backbone; do not port RustGS to Burn, WGPU, or WGSL as part of this migration.
NFR2: Bound new loader memory usage and avoid unbounded CPU image accumulation during training.
NFR3: Preserve compatibility of existing CLI commands and scene outputs unless an explicit versioned format change is introduced.
NFR4: New modules must reduce trainer cognitive load by making ownership boundaries explicit and testable.
NFR5: Persistence changes must ship with round-trip and backward-compatibility regression coverage.
NFR6: The final architecture must support reproducible benchmarking and developer handoff across long-running workstreams.

### Additional Requirements

- Do not migrate Brush UI, mobile, browser, or viewer subsystems into RustGS.
- Do not replace `chunk_planner`, Metal kernels, or the analytical backward path in phase one; wrap them behind clearer orchestration boundaries instead.
- Keep `training_pipeline.rs` as legacy or reference utilities during migration instead of adding new production responsibilities.
- Prefer facade-first refactors that preserve public APIs while internal modules are split.
- Capture module ownership, rollout toggles, and migration notes in docs as part of the rollout epic.
- Use maintainer, researcher, and training-operator outcomes as the user-value lens for epics and stories.

### FR Coverage Map

FR1: Epic 1 - Predictable Training Lifecycle
FR2: Epic 1 - Predictable Training Lifecycle
FR3: Epic 2 - Fast Frame Input Pipeline
FR4: Epic 2 - Fast Frame Input Pipeline
FR5: Epic 3 - Direct Dataset Onboarding
FR6: Epic 3 - Direct Dataset Onboarding
FR7: Epic 3 - Direct Dataset Onboarding
FR8: Epic 4 - Safe Gaussian Topology Evolution
FR9: Epic 5 - Reliable Scene Interchange
FR10: Epic 6 - Regression-Safe Rollout
FR11: Epic 6 - Regression-Safe Rollout
FR12: Epic 6 - Regression-Safe Rollout

## Epic List

### Epic 1: Predictable Training Lifecycle
RustGS maintainers can reason about and extend training runs through a dedicated orchestration layer instead of a monolithic trainer file.
**FRs covered:** FR1, FR2

### Epic 2: Fast Frame Input Pipeline
Training operators can feed large RGB and depth datasets through bounded, asynchronous, reusable frame-loading infrastructure without degrading current output semantics.
**FRs covered:** FR3, FR4

### Epic 3: Direct Dataset Onboarding
Researchers can train RustGS directly from COLMAP and Nerfstudio datasets and apply per-dataset configuration without custom one-off conversion scripts.
**FRs covered:** FR5, FR6, FR7

### Epic 4: Safe Gaussian Topology Evolution
Maintainers can change densify, prune, and opacity-reset behavior and associated optimizer state without expanding `metal_trainer.rs` into another monolith.
**FRs covered:** FR8

### Epic 5: Reliable Scene Interchange
Users can save, load, and round-trip richer RustGS scene data safely across experiments and future SH-capable formats.
**FRs covered:** FR9

### Epic 6: Regression-Safe Rollout
The team can land the migration incrementally with confidence through smoke tests, benchmarks, docs, and compatibility gates.
**FRs covered:** FR10, FR11, FR12

## Epic 1: Predictable Training Lifecycle

Create a dedicated training orchestration layer that owns run lifecycle concerns, leaving `metal_trainer` responsible for step-level execution only.

### Story 1.1: Introduce `train_stream.rs` as the Training Lifecycle Facade

As a RustGS maintainer,
I want `training::train()` to delegate to a dedicated orchestration module,
So that lifecycle flow is explicit and `metal_trainer` no longer acts as the only control surface.

**Implements:** FR1, FR2, FR11.

**Acceptance Criteria:**

**Given** an existing `TrainingDataset` and `TrainingConfig`
**When** `training::train()` is called
**Then** execution is routed through `training/train_stream.rs`
**And** the public return type and caller-facing API remain unchanged.

**Given** `chunked_training` is enabled or disabled
**When** route selection runs
**Then** the existing standard and chunked execution paths are preserved
**And** no caller changes are required to select them.

### Story 1.2: Extract Evaluation Scheduling into `training/eval.rs`

As a RustGS maintainer,
I want evaluation scheduling and metric calculation to live in a dedicated module,
So that training-loop control and metric logic can evolve independently.

**Implements:** FR1, FR2, FR11.

**Acceptance Criteria:**

**Given** a training run with evaluation enabled
**When** the configured evaluation interval is reached
**Then** `training/eval.rs` owns evaluation dispatch and metric aggregation
**And** `metal_trainer` is no longer responsible for orchestration-level eval timing.

**Given** an evaluation failure occurs
**When** the failure is surfaced to the caller
**Then** the error message identifies the evaluation phase clearly
**And** the surrounding training state remains diagnosable.

### Story 1.3: Extract Export and Checkpoint Scheduling into `training/export.rs`

As a training operator,
I want export scheduling to be controlled by a dedicated module,
So that checkpoint behavior is consistent across training routes and easier to test.

**Implements:** FR1, FR11.

**Acceptance Criteria:**

**Given** a training run with periodic export enabled
**When** an export boundary is reached
**Then** `training/export.rs` decides the output path, naming, and persistence action
**And** the orchestration layer can call it without embedding export rules inline.

**Given** export is disabled or unchanged from current defaults
**When** training runs normally
**Then** no extra export side effects occur
**And** current output behavior remains compatible.

### Story 1.4: Extract Training Telemetry Assembly into `training/telemetry.rs`

As a RustGS maintainer,
I want telemetry assembly and last-step reporting to be centralized,
So that training metrics can be extended without adding more lifecycle code to `metal_trainer`.

**Implements:** FR1, FR2, FR11.

**Acceptance Criteria:**

**Given** per-step timing, loss, and topology data are produced during training
**When** telemetry is updated
**Then** `training/telemetry.rs` owns snapshot assembly and persistence hooks
**And** `metal_trainer` only returns raw step-level signals.

**Given** `LiteGsMacV1` telemetry fields already exist
**When** the new telemetry module is introduced
**Then** those fields remain available to existing callers
**And** no profile-specific data is dropped during migration.

## Epic 2: Fast Frame Input Pipeline

Introduce reusable frame-loading infrastructure with bounded memory, deterministic prefetch, and randomized batch delivery for RGB and depth supervision.

### Story 2.1: Extract Frame Decode Helpers from `data_loading.rs`

As a RustGS maintainer,
I want image and depth decoding logic moved into a dedicated frame-loader module,
So that raw IO is separated from training-state assembly.

**Implements:** FR3, FR4.

**Acceptance Criteria:**

**Given** the current color and depth paths in `data_loading.rs`
**When** decode helpers are moved into `training/frame_loader.rs`
**Then** the decoded tensor-ready outputs match the previous implementation
**And** existing unit tests for image and depth loading continue to pass.

**Given** synthetic depth fallback is enabled
**When** a frame lacks a depth map
**Then** the extracted loader still produces the same synthetic-depth behavior
**And** no caller needs to know whether the depth was real or synthetic.

### Story 2.2: Add a Bounded Prefetch Cache for Training Frames

As a training operator,
I want RustGS to prefetch and cache a bounded set of frames,
So that the training loop is less likely to stall on synchronous disk IO.

**Implements:** FR3, NFR2.

**Acceptance Criteria:**

**Given** a dataset larger than the immediate batch window
**When** training starts
**Then** the frame loader starts background prefetch for upcoming samples
**And** cache growth is bounded by an explicit capacity policy.

**Given** the cache reaches its configured limit
**When** more frames are requested
**Then** RustGS evicts or skips caching according to the defined policy
**And** memory usage does not grow without bound.

### Story 2.3: Add a Deterministic Randomized Batch Iterator

As a RustGS researcher,
I want frame batches to be shuffled deterministically from a seed,
So that training remains reproducible while still avoiding fixed-frame ordering.

**Implements:** FR3, FR12.

**Acceptance Criteria:**

**Given** a fixed seed and the same dataset
**When** the batch iterator is created twice
**Then** it emits the same shuffled frame order
**And** reproducibility can be asserted in tests.

**Given** a different seed is supplied
**When** batch iteration begins
**Then** the frame order changes
**And** the iterator still respects dataset bounds and cache rules.

### Story 2.4: Integrate the Frame Loader into `train_stream.rs`

As a RustGS maintainer,
I want the new frame loader to be the only training-loop source of frame batches,
So that orchestration, batching, and preprocessing are controlled from one place.

**Implements:** FR3, FR4, FR11.

**Acceptance Criteria:**

**Given** the new frame loader exists
**When** `train_stream.rs` runs a training session
**Then** batch acquisition happens through the loader abstraction
**And** direct synchronous per-step file reads are removed from orchestration code.

**Given** existing training datasets with and without depth maps
**When** they are trained through the integrated loader
**Then** current synthetic-depth and preprocessing semantics are preserved
**And** training outputs remain behaviorally compatible at smoke-test scope.

## Epic 3: Direct Dataset Onboarding

Let RustGS consume common reconstruction datasets and per-dataset configuration directly, removing the need for one-off conversion steps for core training workflows.

### Story 3.1: Introduce a Unified Dataset Loader Facade

As a RustGS integrator,
I want a single dataset-loading facade for supported dataset types,
So that training entrypoints can consume multiple dataset formats through one normalized contract.

**Implements:** FR5, FR6, FR7.

**Acceptance Criteria:**

**Given** a RustGS training input path
**When** dataset discovery runs
**Then** a unified dataset facade selects the appropriate loader implementation
**And** the resulting dataset descriptor is normalized for downstream training code.

**Given** an unsupported dataset layout
**When** loader discovery fails
**Then** RustGS returns a clear format-specific error
**And** the caller can see which dataset types were expected.

### Story 3.2: Add Direct COLMAP Dataset Support

As a RustGS researcher,
I want to train directly from a COLMAP reconstruction,
So that I can use standard camera, image, and sparse-point outputs without writing a converter first.

**Implements:** FR5.

**Acceptance Criteria:**

**Given** a COLMAP dataset with camera and image metadata
**When** the COLMAP loader runs
**Then** RustGS produces normalized poses, intrinsics, and image references
**And** world-to-camera data is converted into the camera convention expected by RustGS.

**Given** COLMAP sparse points are present
**When** initialization is prepared
**Then** the loader exposes sparse-point initialization data to RustGS
**And** eval split support can be applied without breaking dataset loading.

### Story 3.3: Add Direct Nerfstudio Dataset Support

As a RustGS researcher,
I want to train directly from a Nerfstudio dataset,
So that I can use `transforms*.json` scenes and optional init splats without manual restructuring.

**Implements:** FR6.

**Acceptance Criteria:**

**Given** a Nerfstudio dataset with `transforms.json` or train and val variants
**When** the Nerfstudio loader runs
**Then** RustGS loads train views and optional eval views correctly
**And** pose and intrinsics data are normalized for the RustGS trainer.

**Given** the dataset references an initialization PLY
**When** that PLY is accessible
**Then** the loader surfaces it as initialization input
**And** missing optional init splats degrade gracefully rather than failing the whole load.

### Story 3.4: Add Dataset-Local Config Overlay with CLI Override Merge

As a training operator,
I want dataset-local settings to live next to the dataset and still be overridable from the CLI,
So that experiments are reproducible without hard-coding per-dataset parameters.

**Implements:** FR7, FR12.

**Acceptance Criteria:**

**Given** a dataset-local config file is present
**When** RustGS starts training from that dataset
**Then** the config overlay is loaded automatically into the training configuration
**And** the effective config is visible to the caller or logs.

**Given** a CLI flag overrides the same field
**When** the final config is assembled
**Then** the CLI value wins
**And** the precedence rule is documented and test-covered.

### Story 3.5: Move Initial Map Construction into `training/init_map.rs`

As a RustGS maintainer,
I want initial-map construction to be its own module,
So that sparse-point initialization and frame-based fallback are no longer coupled to raw frame IO code.

**Implements:** FR5, FR6, FR8.

**Acceptance Criteria:**

**Given** a dataset with sparse initialization points
**When** `training/init_map.rs` builds the initial map
**Then** it uses sparse-point initialization through a dedicated path
**And** downstream training code receives the same `GaussianMap` contract as before.

**Given** sparse initialization is absent
**When** initial-map construction runs
**Then** frame-based fallback is used through the same module
**And** the caller does not need separate initialization logic.

## Epic 4: Safe Gaussian Topology Evolution

Make Gaussian topology edits explicit and testable by separating parameter views, refine scheduling, and optimizer-state reshaping from step-level Metal execution.

### Story 4.1: Introduce a Compact Splat Parameter View and Validation Helpers

As a RustGS maintainer,
I want a consistent trainable splat parameter view with shape validation,
So that topology edits can operate on one clear representation instead of ad hoc tensor slices.

**Implements:** FR8, NFR4.

**Acceptance Criteria:**

**Given** current trainable Gaussian parameters in RustGS
**When** the compact parameter view is introduced
**Then** means, rotations, log-scales, opacity, and color or SH state are addressable through one normalized abstraction
**And** no runtime backend migration is introduced.

**Given** malformed dimensions or incompatible parameter lengths
**When** validation helpers run
**Then** RustGS reports invariant violations clearly
**And** the failures are test-covered before topology edits execute.

### Story 4.2: Extract Topology Decision Logic into `training/topology.rs`

As a RustGS maintainer,
I want densify, prune, and opacity-reset decisions moved into a dedicated topology module,
So that behavior changes can be made without rewriting training-loop control flow.

**Implements:** FR8, FR11.

**Acceptance Criteria:**

**Given** the current topology heuristics in `metal_trainer.rs`
**When** they are extracted into `training/topology.rs`
**Then** scheduling and candidate selection logic live outside the step-level trainer
**And** current profile semantics are preserved at regression-test scope.

**Given** a future heuristic change is needed
**When** a maintainer updates topology policy
**Then** the change can be localized to topology modules
**And** the orchestration and Metal execution layers do not need unrelated edits.

### Story 4.3: Extract Optimizer-State Reshape Helpers into `training/optimizer_state.rs`

As a RustGS maintainer,
I want prune, clone, and split operations to update optimizer state through dedicated helpers,
So that parameter resizing remains correct when topology changes occur.

**Implements:** FR8, NFR4.

**Acceptance Criteria:**

**Given** a topology edit removes or adds Gaussians
**When** optimizer-state reshape helpers run
**Then** all affected optimizer tensors are resized consistently with the parameter tensors
**And** the operation does not rely on scattered one-off logic inside `metal_trainer.rs`.

**Given** a topology edit does not change tensor count
**When** optimizer-state helpers are invoked
**Then** they preserve existing state without introducing unnecessary copies
**And** tests cover prune and split paths separately.

### Story 4.4: Route Topology Scheduling Through `train_stream.rs`

As a RustGS maintainer,
I want topology scheduling to be called explicitly from the orchestration layer,
So that lifecycle control determines when refine work happens and the trainer only performs step-level computation.

**Implements:** FR1, FR8, FR11.

**Acceptance Criteria:**

**Given** a training run reaches a topology boundary
**When** the orchestration layer evaluates scheduling rules
**Then** `train_stream.rs` calls topology modules explicitly
**And** `metal_trainer` no longer embeds orchestration-level refine timing.

**Given** topology work is disabled by configuration or schedule
**When** the same training run executes
**Then** the orchestration layer skips topology actions cleanly
**And** step-level training continues without behavior regressions.

## Epic 5: Reliable Scene Interchange

Strengthen RustGS persistence so scene files can evolve safely, preserve richer metadata, and survive round-trip tests as SH-aware formats are introduced.

### Story 5.1: Introduce Versioned Scene Metadata

As a RustGS user,
I want scene files to carry explicit format metadata,
So that future import and export changes can remain backward compatible.

**Implements:** FR9, NFR3, NFR5.

**Acceptance Criteria:**

**Given** a RustGS scene is exported
**When** metadata is written
**Then** the file includes an explicit format version and relevant scene metadata fields
**And** existing metadata such as iteration and loss remain available.

**Given** an older scene file is loaded
**When** the version field is absent or older
**Then** RustGS falls back to compatible parsing behavior
**And** the caller receives a warning only when compatibility assumptions matter.

### Story 5.2: Split PLY Import and Export Behind the Existing `scene_io` Facade

As a RustGS maintainer,
I want import and export code split into focused modules behind the current facade,
So that persistence logic can grow without turning `scene_io.rs` into another monolith.

**Implements:** FR9, NFR4.

**Acceptance Criteria:**

**Given** the current `save_scene_ply` and `load_scene_ply` API
**When** import and export logic are split into dedicated modules
**Then** existing callers continue to use the facade unchanged
**And** the new internal module boundaries are explicit and documented.

**Given** a persistence bug is fixed in import or export logic
**When** the fix is made
**Then** it can be localized to the relevant module
**And** no unrelated caller-facing API churn is introduced.

### Story 5.3: Add Dynamic SH-Aware Scene Serialization

As a RustGS researcher,
I want scene serialization to preserve active SH degree and related parameter state,
So that richer color representations can survive save and load boundaries.

**Implements:** FR9, NFR3, NFR5.

**Acceptance Criteria:**

**Given** a scene uses RGB-only or SH-based color state
**When** it is exported
**Then** the persistence layer records the active color representation and SH degree explicitly
**And** import reconstructs the same representation or fails clearly if unsupported.

**Given** a scene does not use richer SH data
**When** it is saved and loaded
**Then** current RGB-only compatibility is preserved
**And** no legacy caller must supply extra metadata manually.

### Story 5.4: Add Round-Trip and Compatibility Regression Coverage for Scene IO

As a RustGS maintainer,
I want persistence changes guarded by regression tests,
So that scene interchange can evolve without silent corruption.

**Implements:** FR9, FR10, NFR5.

**Acceptance Criteria:**

**Given** representative RustGS scene fixtures
**When** round-trip tests run
**Then** exported scenes can be re-imported with matching metadata and parameter counts
**And** failures identify the field or representation that drifted.

**Given** compatibility fixtures from earlier scene versions
**When** import regression tests run
**Then** supported legacy files still load or fail with explicit versioned errors
**And** the expected compatibility matrix is documented.

## Epic 6: Regression-Safe Rollout

Land the migration incrementally and safely through route-level smoke tests, benchmark harnesses, ownership docs, and compatibility gates.

### Story 6.1: Add End-to-End Training Route Smoke Tests

As a RustGS maintainer,
I want smoke tests that exercise the main training routes,
So that orchestration refactors do not silently break standard, chunked, or profile-specific entry behavior.

**Implements:** FR10, FR11.

**Acceptance Criteria:**

**Given** representative tiny training fixtures
**When** the smoke suite runs
**Then** it exercises standard and chunked route selection through the new orchestration layer
**And** verifies that `LegacyMetal` and `LiteGsMacV1` still reach their expected execution paths.

**Given** a route regression is introduced
**When** smoke tests fail
**Then** the failure identifies the broken route clearly
**And** maintainers can see whether the breakage was in orchestration, trainer selection, or config assembly.

### Story 6.2: Add Loader, Topology, and Persistence Regression Suites

As a RustGS maintainer,
I want focused regression suites for the new modules,
So that internal refactors remain safe even when route-level smoke tests are too coarse.

**Implements:** FR10, FR11.

**Acceptance Criteria:**

**Given** the new loader, topology, and persistence modules
**When** targeted regression tests run
**Then** each module has fixtures that validate its core invariants independently
**And** failures are localized to one subsystem instead of one giant training test.

**Given** a dataset-specific or topology-specific bug is fixed
**When** a regression is added
**Then** that regression lands in the focused module test suite
**And** future maintainers can reproduce the failure without running the full pipeline.

### Story 6.3: Add Forward, Backward, and Training Benchmark Harnesses

As a RustGS performance owner,
I want benchmark harnesses for the core execution paths,
So that the migration can be measured instead of judged only by code structure.

**Implements:** FR10, NFR6.

**Acceptance Criteria:**

**Given** representative benchmark fixtures
**When** the benchmark harness runs
**Then** it reports forward, backward, and training-step timing separately
**And** the harness can be reused after later optimizer or topology changes.

**Given** a performance regression is introduced
**When** benchmark results are compared across revisions
**Then** the affected phase can be identified quickly
**And** the results are reproducible enough for developer handoff.

### Story 6.4: Publish Migration Ownership and Rollout Documentation

As a RustGS maintainer,
I want the migrated architecture and rollout rules documented,
So that future contributors can extend the system without rediscovering module boundaries.

**Implements:** FR10, FR12, NFR4, NFR6.

**Acceptance Criteria:**

**Given** the new module layout lands
**When** documentation is updated
**Then** it explains module ownership, call flow, rollout toggles, and compatibility constraints
**And** it explicitly states which Brush ideas were adopted and which were intentionally rejected.

**Given** a new contributor joins the project later
**When** they read the migration documentation
**Then** they can identify where orchestration, loading, topology, and persistence changes belong
**And** they do not need to infer ownership from a single giant trainer file.

### Story 6.5: Demote `training_pipeline.rs` to a Legacy and Reference Role After Validation Gates Pass

As a RustGS maintainer,
I want legacy pipeline utilities clearly separated from production training ownership,
So that new work stops accumulating in the wrong module once the migrated path is validated.

**Implements:** FR11, FR12, NFR4.

**Acceptance Criteria:**

**Given** orchestration, loading, topology, and persistence replacements are validated
**When** the migration cleanup story runs
**Then** `training_pipeline.rs` is explicitly marked and documented as legacy or reference-only
**And** new production responsibilities are routed to the new modules instead.

**Given** cleanup would break current callers or tests
**When** the story is executed
**Then** the migration stops short of unsafe deletion
**And** the file is only reduced to the extent supported by passing compatibility gates.
