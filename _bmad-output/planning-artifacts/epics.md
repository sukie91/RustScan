---
stepsCompleted:
  - step-01-validate-prerequisites
  - step-02-design-epics
  - step-03-create-stories
  - step-04-final-validation
inputDocuments:
  - /Users/tfjiang/Projects/RustScan/RustGS/docs/plans/2026-04-09-rustgs-soa-splat-architecture-proposal.md
---

# RustScan - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for RustScan, decomposing the requirements from the selected splat architecture proposal into implementable stories.

## Implementation Status

Status snapshot as of 2026-04-09:

- `Epic 1` is partially complete.
  - Runtime `Splats` rename landed.
  - Host `HostSplats` rename landed.
  - Explicit upload/snapshot APIs landed.
  - Host-side SH payload unification landed.
  - Most internal GPU training code now uses the canonical `Splats` name directly.
  - Runtime-side SH/color split is still pending cleanup.
- `Epic 2` is partially complete.
  - Runtime `Splats` is now the explicit training owner in the main non-chunked Metal path.
  - Existing optimizer/topology sidecars remain external.
  - Training data loading and initialization now build `HostSplats` directly.
  - Some compatibility paths still bounce through `GaussianMap`.
- `Epic 3` is partially complete.
  - `SplatView` landed.
  - `save_splats_ply` / `load_splats_ply` landed.
  - `save_splats_ply` / `load_splats_ply` now run directly on `HostSplats` SoA storage.
  - `evaluate_splats` / `trainable_from_splats` landed.
  - `evaluate_gaussians` / `trainable_from_gaussians` now provide splat-first canonical naming for Gaussian-array compatibility inputs.
  - Import and some compatibility reporting paths still use scene-oriented helpers.
- `Epic 4` is not complete.
  - AoS and scene/map owners still exist in compatibility paths.
  - SLAM/path adapters still exist at crate root.
  - Legacy CLI/evaluation/reporting terminology still exposes `scene` names in public compatibility surfaces.

## Requirements Inventory

### Functional Requirements

FR1: RustGS shall expose `Splats` as the only long-lived runtime splat owner for training.

FR2: RustGS shall expose `HostSplats` as the host-side boundary type for initialization, import, export, and checkpointing.

FR3: RustGS shall expose `SplatView` as the shared borrowed read-only interface for export, evaluation, and non-owning helpers.

FR4: RustGS shall represent all owning splat parameter types using structure-of-arrays layout.

FR5: RustGS runtime `Splats` shall own differentiable device-side fields for `positions`, `log_scales`, `rotations`, `opacity_logits`, `sh_coeffs`, `n`, and `sh_degree`.

FR6: RustGS `HostSplats` shall own host-side vector fields for `positions`, `log_scales`, `rotations`, `opacity_logits`, `sh_coeffs`, and `sh_degree`.

FR7: RustGS shall restrict owning conversion paths to `TrainingDataset -> HostSplats`, `HostSplats -> Splats`, and `Splats -> HostSplats`.

FR8: RustGS shall treat `HostSplats` as a boundary artifact, not as an eagerly synchronized runtime mirror of `Splats`.

FR9: RustGS `Splats` shall provide a `snapshot()` path that materializes `HostSplats` for host-facing workflows.

FR10: RustGS shall keep optimizer state, topology state, and densify/prune accumulators outside of the core splat owners.

FR11: RustGS shall unify SH storage so that `sh_degree = 0` represents the RGB-only case and `sh_coeffs` is always stored in coefficient-major RGB-triplet layout.

FR12: RustGS shall remove owning AoS and scene-oriented core types including `Gaussian3D`, `GaussianMap`, `render::tiled_renderer::Gaussian`, and `training_pipeline::TrainableGaussian`.

FR13: RustGS shall rename current `diff::TrainableGaussians` to `Splats`.

FR14: RustGS shall rename current host-side `training::splats::Splats` to `HostSplats`.

FR15: RustGS shall rename borrowed helper interfaces to `SplatView`.

FR16: RustGS core APIs shall be shaped around `TrainingDataset`, not around `SlamOutput` or other SLAM-derived input types.

FR17: RustGS shall demote `train_from_slam`, SLAM-centric entrypoints, and dataset autodetection to adapter or CLI layers outside the core splat architecture.

FR18: RustGS shall rename scene-oriented IO APIs such as `save_scene_ply` and `load_scene_ply` to splat-oriented names.

### NonFunctional Requirements

NFR1: The core splat architecture shall avoid representational drift by limiting the system to one long-lived runtime splat owner and one host boundary splat owner.

NFR2: SH information shall round-trip between `HostSplats` and `Splats` without semantic loss.

NFR3: Conversion boundaries shall remain minimal and explicit.

NFR4: The architecture shall preserve a logical distinction between host ownership and device ownership even on unified-memory Apple Silicon hardware.

NFR5: The core architecture shall avoid public owning AoS Gaussian types.

NFR6: The public naming model shall be simple and training-centric.

NFR7: Rendering, evaluation, and export shall prefer borrowed views rather than additional owning splat types.

NFR8: The migration shall be executable in phases without requiring a single destructive rewrite.

### Additional Requirements

- Introduce shared SH helper functions and remove duplicate color representation enums.

- Keep runtime sidecars such as optimizer state and topology bookkeeping separate from `Splats`.

- Ensure `HostSplats` can validate shape invariants before upload and after snapshot.

- Ensure `Splats` validates tensor shapes and degree-dependent SH coefficient layout.

- Keep `HostSplats` suitable for PLY import/export and checkpoint serialization.

- Ensure the runtime design moves RustGS closer to Brush's dominant-owner model without copying Brush's packed `[n, 10]` transform layout.

### FR Coverage Map

FR1: Epic 1 - Establish `Splats` as the only long-lived runtime splat owner.

FR2: Epic 1 - Establish `HostSplats` as the host-side boundary type.

FR3: Epic 3 - Introduce `SplatView` as the shared borrowed read-only interface.

FR4: Epic 1 - Make all owning splat parameter types SoA.

FR5: Epic 1 - Define the runtime `Splats` device-side field model.

FR6: Epic 1 - Define the `HostSplats` host-side field model.

FR7: Epic 1 - Restrict owning conversion paths to initialization, upload, and snapshot.

FR8: Epic 2 - Ensure `HostSplats` is boundary-only and not an eagerly synchronized runtime mirror.

FR9: Epic 3 - Provide `snapshot()` from runtime `Splats` into `HostSplats` for host-facing workflows.

FR10: Epic 2 - Move optimizer, topology, and densify/prune accumulators into runtime sidecars.

FR11: Epic 1 - Unify SH storage semantics and treat `sh_degree = 0` as RGB-only.

FR12: Epic 4 - Remove AoS and scene-oriented core types from the main architecture.

FR13: Epic 1 - Rename current `diff::TrainableGaussians` to `Splats`.

FR14: Epic 1 - Rename current host-side `training::splats::Splats` to `HostSplats`.

FR15: Epic 3 - Rename borrowed helper interfaces to `SplatView`.

FR16: Epic 4 - Shape the core API around `TrainingDataset` rather than SLAM-derived types.

FR17: Epic 4 - Demote SLAM-centric entrypoints and dataset autodetection to adapter or CLI layers.

FR18: Epic 3 - Rename scene-oriented IO APIs to splat-oriented names.

## Epic List

### Epic 1: Canonical Splat Model
Enable RustGS maintainers to work against one coherent SoA splat model by defining `Splats`, `HostSplats`, shared SH storage rules, and explicit ownership conversions.
**FRs covered:** FR1, FR2, FR4, FR5, FR6, FR7, FR11, FR13, FR14

### Epic 2: Runtime Ownership And Sidecars
Enable training-flow developers to run the system with `Splats` as the only long-lived runtime owner while keeping optimizer and topology process state outside the core model.
**FRs covered:** FR8, FR10

### Epic 3: Artifact And Read-Only Interfaces
Enable engineers to initialize, snapshot, evaluate, export, and checkpoint the model through `HostSplats` and `SplatView` without introducing more owning Gaussian types.
**FRs covered:** FR3, FR9, FR15, FR18

### Epic 4: Pure-Training API Cleanup
Enable RustGS to behave like a pure 3DGS training library by removing AoS and scene-oriented core types and pushing SLAM-shaped entrypoints out of the core architecture.
**FRs covered:** FR12, FR16, FR17

## Epic 1: Canonical Splat Model

Define one coherent SoA splat model centered on runtime `Splats`, boundary `HostSplats`, and one shared SH storage contract.

### Story 1.1: Shared SH Storage Contract

As a RustGS maintainer,
I want one shared SH storage contract and helper surface,
So that all splat code paths encode color with the same semantics.

**Implements:** FR11

**Acceptance Criteria:**

**Given** the current split between RGB and SH representations
**When** the shared SH helper module is introduced
**Then** degree-based coefficient counting is defined in one place
**And** `sh_degree = 0` is treated as the RGB-only case.

**Given** host and runtime splat types
**When** they store color coefficients
**Then** both use the same coefficient-major RGB-triplet layout
**And** there is no new path that stores separate owned `color`, `sh_dc`, and `sh_rest` fields.

### Story 1.2: Runtime `Splats` Rename And Field Alignment

As a RustGS maintainer,
I want the device-side runtime owner renamed to `Splats`,
So that the main training model has one simple, stable name.

**Implements:** FR1, FR5, FR13

**Acceptance Criteria:**

**Given** the current `diff::TrainableGaussians` runtime owner
**When** it is refactored
**Then** the public runtime type is named `Splats`
**And** its fields are aligned to `positions`, `log_scales`, `rotations`, `opacity_logits`, `sh_coeffs`, `n`, and `sh_degree`.

**Given** runtime tensor storage
**When** `Splats` is validated
**Then** tensor shapes are checked against the declared `n` and `sh_degree`
**And** shape validation fails on incompatible coefficient layout.

### Story 1.3: `HostSplats` Rename And Boundary Field Alignment

As a RustGS maintainer,
I want the host-side splat buffer renamed to `HostSplats`,
So that initialization and artifact workflows use a clearly named boundary type.

**Implements:** FR2, FR6, FR14

**Acceptance Criteria:**

**Given** the current host-side `training::splats::Splats`
**When** it is refactored
**Then** the public host boundary type is named `HostSplats`
**And** its fields mirror the runtime model semantically using `Vec<f32>` storage.

**Given** a `HostSplats` instance
**When** it is validated
**Then** vector lengths are checked against `n` and `sh_degree`
**And** invalid field lengths produce deterministic validation failure.

### Story 1.4: Explicit Upload And Snapshot Conversions

As a RustGS maintainer,
I want ownership transitions limited to explicit upload and snapshot paths,
So that host/device boundaries remain obvious and testable.

**Implements:** FR4, FR7

**Acceptance Criteria:**

**Given** the canonical type model
**When** owning conversions are implemented
**Then** the only supported owning paths are `TrainingDataset -> HostSplats`, `HostSplats -> Splats`, and `Splats -> HostSplats`
**And** no new direct owning conversion path is introduced through AoS Gaussian types.

**Given** runtime-to-host snapshot behavior
**When** `Splats::snapshot()` is called
**Then** it produces a `HostSplats` artifact with the same semantic content
**And** SH coefficient data round-trips without semantic loss.

## Epic 2: Runtime Ownership And Sidecars

Make `Splats` the only long-lived runtime owner and push optimizer and topology process state into explicit sidecars.

### Story 2.1: Extract Runtime Sidecars

As a training-flow developer,
I want optimizer and topology process state moved into explicit sidecar types,
So that the core splat model only owns model parameters.

**Implements:** FR10

**Acceptance Criteria:**

**Given** current runtime process data such as optimizer state, gradient accumulators, radii, and age counters
**When** sidecar types are introduced
**Then** those values are stored outside `Splats`
**And** `Splats` only contains model parameter ownership.

**Given** sidecar-managed runtime state
**When** training steps execute
**Then** sidecars stay index-aligned with the active splat count
**And** sidecar updates do not require adding non-parameter fields back into `Splats`.

### Story 2.2: Runtime Training Loop Uses `Splats` As Sole Owner

As a training-flow developer,
I want the trainer to hold `Splats` as the sole long-lived runtime owner,
So that the training loop has one authoritative model state.

**Implements:** FR8

**Acceptance Criteria:**

**Given** the runtime training path
**When** the trainer is refactored
**Then** `Splats` is the only long-lived model owner during training
**And** `HostSplats` is not kept as an eagerly synchronized runtime mirror.

**Given** runtime ownership
**When** forward, backward, and optimizer steps execute
**Then** they operate on `Splats` plus sidecars
**And** they do not depend on a parallel host-side owner staying live.

### Story 2.3: Topology Mutations Operate On `Splats` Plus Sidecars

As a training-flow developer,
I want densify and prune logic to mutate `Splats` and sidecars together,
So that runtime topology changes preserve model and process-state consistency.

**Implements:** FR10

**Acceptance Criteria:**

**Given** densify or prune operations
**When** the active splat count changes
**Then** `Splats` tensors and all required sidecars are updated in the same logical operation
**And** index alignment remains valid after insertions, removals, and reorders.

**Given** topology updates that affect SH-carrying splats
**When** those splats are cloned, split, or removed
**Then** SH coefficient storage remains well-formed
**And** no path silently drops SH data.

## Epic 3: Artifact And Read-Only Interfaces

Use `HostSplats` and `SplatView` to support initialization, snapshots, evaluation, export, and checkpointing without introducing extra owning Gaussian models.

### Story 3.1: Shared Borrowed `SplatView`

As a RustGS maintainer,
I want a single borrowed `SplatView` interface,
So that read-only host-facing workflows do not require additional owning splat types.

**Implements:** FR3, FR15

**Acceptance Criteria:**

**Given** the new type model
**When** borrowed read-only access is needed
**Then** `HostSplats` exposes `as_view()`
**And** the shared borrowed interface is named `SplatView`.

**Given** host-facing helper code
**When** evaluation or export needs read-only access
**Then** it accepts `SplatView` or `HostSplats`
**And** it does not require a public owning render Gaussian type.

### Story 3.2: PLY Import And Export Use `HostSplats`

As an engineer producing model artifacts,
I want PLY import and export to operate on `HostSplats`,
So that artifact workflows use the host boundary type directly.

**Implements:** FR2, FR11

**Acceptance Criteria:**

**Given** PLY import behavior
**When** splat artifacts are loaded
**Then** the import path produces `HostSplats`
**And** RGB compatibility inputs are converted to the unified SH storage contract at the boundary.

**Given** PLY export behavior
**When** splat artifacts are written
**Then** export accepts `HostSplats` or `SplatView`
**And** exported SH-aware artifacts preserve coefficient data and declared degree.

### Story 3.3: Snapshot And Checkpoint Artifacts

As an engineer operating the training pipeline,
I want snapshots and checkpoints to store `HostSplats`,
So that artifacts are host-readable without keeping a second live runtime owner.

**Implements:** FR2, FR9

**Acceptance Criteria:**

**Given** a running training session
**When** a snapshot or checkpoint is requested
**Then** runtime `Splats` materializes `HostSplats` through `snapshot()`
**And** artifact serialization stores the `HostSplats` payload rather than AoS scene objects.

**Given** a checkpoint restore path
**When** a checkpoint is loaded
**Then** it reconstructs `HostSplats`
**And** upload into runtime `Splats` follows the standard boundary conversion path.

### Story 3.4: Splat-Oriented IO And Evaluation Interfaces

As a RustGS maintainer,
I want IO and evaluation interfaces named around splats instead of scenes,
So that the public API matches the pure training architecture.

**Implements:** FR18

**Acceptance Criteria:**

**Given** current scene-oriented helper names
**When** the interface cleanup is applied
**Then** APIs are renamed to splat-oriented names such as `save_splats_ply` and `load_splats_ply`
**And** metadata naming is updated away from scene vocabulary.

**Given** evaluation and render helper code
**When** host-facing paths are refactored
**Then** they consume `SplatView` or `HostSplats`
**And** they do not reintroduce a public owning render Gaussian model.

## Epic 4: Pure-Training API Cleanup

Remove AoS and scene-oriented core types from the main architecture and push SLAM-shaped entrypoints out of the core training API.

### Story 4.1: Isolate Legacy AoS And Scene Owners

As a RustGS maintainer,
I want legacy AoS and scene owners isolated behind compatibility boundaries,
So that new core code no longer depends on them.

**Implements:** FR12

**Acceptance Criteria:**

**Given** current uses of `Gaussian3D`, `GaussianMap`, and related scene vocabulary
**When** compatibility boundaries are introduced
**Then** new core splat code stops depending on those types
**And** any remaining references are isolated behind migration adapters.

**Given** the updated architecture
**When** core modules are inspected
**Then** the primary ownership model is `TrainingDataset`, `HostSplats`, `Splats`, and sidecars
**And** scene-oriented types are no longer the default code path.

### Story 4.2: Remove Legacy Render And Training Gaussian Owners

As a RustGS maintainer,
I want legacy render and training Gaussian owners removed,
So that the system no longer duplicates parameter ownership through AoS wrappers.

**Implements:** FR12

**Acceptance Criteria:**

**Given** `render::tiled_renderer::Gaussian` and `training_pipeline::TrainableGaussian`
**When** cleanup is completed
**Then** these types are removed from the main architecture
**And** any required compatibility logic is rewritten to use `HostSplats`, `Splats`, or `SplatView`.

**Given** render and training call sites
**When** they are migrated
**Then** no public API depends on those legacy owners
**And** render-facing code no longer needs a second owning Gaussian representation.

### Story 4.3: Demote SLAM-Centric Entrypoints To Adapter Layers

As a RustGS integrator,
I want SLAM-shaped entrypoints moved out of the core training architecture,
So that the RustGS core stays focused on `TrainingDataset`-based training.

**Implements:** FR16, FR17

**Acceptance Criteria:**

**Given** current root-level helpers such as `train_from_slam` and dataset autodetection
**When** adapter cleanup is applied
**Then** those entrypoints are moved to adapter or CLI layers
**And** the core training API is shaped around `TrainingDataset`.

**Given** crate root exports
**When** the public API is reviewed
**Then** SLAM-derived types are not part of the core splat architecture surface
**And** training-centric types are the primary public path.

### Story 4.4: Final Core Naming And Documentation Cleanup

As a RustGS maintainer,
I want the final public naming and documentation aligned to the pure-training model,
So that the architecture is understandable without scene or SLAM mental overhead.

**Implements:** FR16

**Acceptance Criteria:**

**Given** the renamed type model
**When** documentation and public exports are updated
**Then** `Splats`, `HostSplats`, and `SplatView` are the canonical terms
**And** outdated references to trainable, scene, or map ownership are removed from primary docs.

**Given** the final cleaned API surface
**When** a maintainer reads the public architecture documentation
**Then** the core model is described as pure 3DGS training
**And** the migration path away from legacy AoS and SLAM-shaped concepts is explicit.
