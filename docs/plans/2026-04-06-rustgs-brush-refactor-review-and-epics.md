# RustGS Brush Training Refactor Review and Epic Breakdown

Review date: 2026-04-06
Status: Recommended with revisions

This document is the retained, current refactor plan. The earlier draft proposal has been removed from `docs/` because it no longer matched the active codebase closely enough to serve as a reliable execution document.

## Purpose

This document captures:

1. A codebase-grounded review of the proposed RustGS Brush-style training refactor
2. The architectural corrections required to make the refactor safe
3. A revised epic and story breakdown that can be executed incrementally

The goal is not to reject the original direction. The goal is to turn it into a migration plan that matches the current RustGS codebase and does not break the active training path.

## Executive Verdict

The proposal is directionally correct and worth pursuing.

The main diagnosis is accurate:

- `RustGS/src/training/metal_trainer.rs` is still a God Object
- training state is split across incompatible representations
- `metal_runtime.rs` mixes shader source, pipeline management, buffer layout, and dispatch
- topology logic is spread across overlapping modules

However, the original proposal is too aggressive in three places:

1. It promotes a new `Splats` type to the public core and IO boundary too early
2. It introduces a `MetalKernel` trait before the current Metal runtime has been factored into smaller concrete responsibilities
3. It treats the refactor like a trainer-file cleanup, while the real public contract also includes `training::train()`, chunked training, evaluation, parity reporting, CLI behavior, and scene IO

The refactor should proceed as an internal, compatibility-preserving migration first. Public types and entry points should remain stable until the new internal boundaries are proven.

## Codebase Facts That Drive the Review

### 1. The production training path is already broader than `metal_trainer.rs`

Today, the active path is not just a single trainer file. It spans:

- `RustGS/src/lib.rs`
- `RustGS/src/training/mod.rs`
- `RustGS/src/training/data_loading.rs`
- `RustGS/src/training/metal_trainer.rs`
- `RustGS/src/training/topology.rs`
- `RustGS/src/training/eval.rs`
- `RustGS/src/training/chunk_planner.rs`

That means the refactor must preserve more than step execution. It must preserve orchestration, data loading, evaluation, chunk routing, and external entry behavior.

### 2. The current data-model split is real

RustGS currently moves between three different representations:

- `Gaussian3D` / `GaussianMap` for scene storage and public scene IO
- `TrainableGaussians` for GPU training state
- internal bridges such as `Splats`

This validates the plan's core concern, but it also shows why public `core/` should not be rewritten first. Right now the split is bridged by conversion code in `training/data_loading.rs`, not by a stable shared model.

### 3. `render/` is not the same thing as training forward

The current top-level `render` module is a CPU and basic rendering surface:

- `RustGS/src/render/renderer.rs`
- `RustGS/src/render/tiled_renderer.rs`

It is not the same abstraction boundary as the Metal training forward path. Reusing the same namespace without a compatibility plan would blur responsibilities.

### 4. `density_controller.rs` and `topology.rs` overlap, but they are not interchangeable yet

`density_controller.rs` acts as a LiteGS-compatible reference implementation.
`topology.rs` already contains scheduling and candidate analysis logic for the active Metal path.

That overlap supports a strategy-based design, but it does not justify deleting either one in the first wave.

### 5. `metal_runtime.rs` has a concrete decomposition problem

The main problem is not lack of traits. The problem is that shader source, pipeline lookup, buffer slots, runtime caches, and dispatch functions are packed into one module. The first useful move is concrete separation, not polymorphism.

## What Is Reasonable in the Original Plan

The following ideas should be kept:

- Split trainer orchestration away from step-level execution details
- Create a unified internal training-state representation
- Separate forward rendering, backward computation, optimizer updates, and topology work
- Move toward strategy-driven topology and loss composition
- Add stronger validation through parity, smoke, and performance checks

## What Must Change Before Execution

### 1. Keep public scene and training entry boundaries stable

For the first phase, do not replace these as public contracts:

- `Gaussian3D`
- `GaussianMap`
- `training::train()`
- CLI `rustgs train`
- current scene import/export behavior

The migration should happen behind those interfaces.

### 2. Make `Splats` an internal training representation first

The new unified structure is useful, but it should start inside `training/` as an internal model for:

- trainable parameter storage
- forward/backward exchange
- optimizer updates
- topology mutations

Only after the model is proven should it be considered for promotion outside training internals.

### 3. Defer `MetalKernel` trait introduction

Before adding a trait abstraction, first split `metal_runtime.rs` into concrete sub-responsibilities:

- shader source loading
- pipeline cache
- resource and buffer layout
- dispatch wrappers

If a trait is still useful after that split, add it later.

### 4. Do not delete `density_controller.rs` in the first strategy pass

First convert it into a reference adapter or strategy implementation.
Delete or merge only after parity and ownership boundaries are clear.

### 5. Treat file-size reduction as an outcome, not a hard requirement

The target should be:

- orchestration no longer owns kernel details
- topology no longer lives inside trainer control flow
- optimizer and backward logic are independently testable

Whether the resulting orchestrator is 500 lines or 900 lines is secondary.

## Revised Architecture Principles

1. Preserve public behavior first
2. Unify internal training state second
3. Separate runtime layers concretely before adding abstraction traits
4. Keep legacy/reference paths intact until parity gates pass
5. Make every refactor step benchmarkable and reversible

## Non-Goals

This refactor should not do the following in the initial execution wave:

- replace `GaussianMap` as the public scene type
- create a second production training entry path
- redesign CLI behavior
- rewrite chunked training semantics
- delete parity infrastructure
- introduce multi-backend runtime abstraction just for architecture purity

## Epic Overview

### Epic 1: Refactor Guardrails and Compatibility Baseline
Create the regression and contract guardrails needed to refactor safely without breaking the active RustGS training surface.

### Epic 2: Internal Training-State Unification
Introduce a single internal training representation that can bridge scene IO, GPU trainables, forward outputs, optimizer updates, and topology mutations.

### Epic 3: Forward and Metal Runtime Extraction
Separate Metal shader/runtime concerns and define a clean forward-render boundary that trainer orchestration can call without owning dispatch details.

### Epic 4: Backward and Optimizer Extraction
Move gradient computation and parameter update behavior into explicit modules with stable interfaces and regression coverage.

### Epic 5: Topology Strategy Migration
Move densify, prune, opacity reset, and topology statistics into strategy-oriented components while preserving current LiteGS and legacy semantics.

### Epic 6: Orchestration Convergence and Cleanup
Converge on a thinner orchestration layer while preserving `training::train()`, chunked execution, evaluation, and CLI compatibility.

## Epic 1: Refactor Guardrails and Compatibility Baseline

Goal: Make the current RustGS training surface measurable and regression-safe before structural changes begin.

Exit criteria:

- public entry behavior is explicitly documented
- smoke, parity, and benchmark baselines exist
- critical trainer-path invariants have tests

### Story 1.1: Capture the Current Public Contract

As a RustGS maintainer,
I want the current training and export surface documented,
So that refactors do not accidentally break behavior outside the trainer file.

Acceptance criteria:

- `training::train()`, CLI train flow, scene export, eval flow, and chunked training behavior are documented in one place
- the document identifies which modules are public contracts and which are internal implementation details
- the document is stored with the refactor plan artifacts

### Story 1.2: Establish Smoke and Parity Baselines

As a RustGS maintainer,
I want baseline smoke and parity runs captured,
So that each refactor step can be checked against known-good behavior.

Acceptance criteria:

- at least one TUM smoke training path is runnable and documented
- the parity harness remains part of the validation path
- a benchmark or timing baseline exists for forward, backward, and end-to-end step timing

### Story 1.3: Add Critical Regression Coverage Around the Current Trainer Path

As a RustGS maintainer,
I want minimum regression tests for the current step pipeline,
So that extraction work does not silently regress numerical behavior.

Acceptance criteria:

- forward output shape and visibility behavior have regression coverage
- backward gradient plumbing has regression coverage
- optimizer state rebuild or row-remap behavior has regression coverage
- topology-triggered parameter mutation paths have regression coverage

## Epic 2: Internal Training-State Unification

Goal: Replace the current representation split with one internal training-state model while preserving `GaussianMap` as the external scene boundary.

Implementation note as of 2026-04-07:

- a new internal `training::splats::Splats` model now exists and captures trainable parameter storage for positions, log-scales, rotations, opacity logits, base color parameters, and SH-rest coefficients
- `training/data_loading.rs` now routes `GaussianMap <-> TrainableGaussians` conversion through `Splats`, so the canonical conversion logic is no longer duplicated there
- `metal_trainer.rs` snapshot export/rebuild flow now also uses `Splats`, removing the active trainer path's duplicated parameter snapshot type
- `training/eval.rs` scene-to-trainable conversion now also routes through `Splats`
- `topology.rs` now consumes `Splats` directly and is wired into `metal_trainer.rs` for schedule, candidate analysis, and snapshot mutation helpers
- stale compatibility/orchestration leftovers such as `SplatParameterView` and `train_stream.rs` have been removed from the compiled path
- Epic 2 exit criteria are now met locally; the remaining architecture work has shifted to Epic 3 and Epic 5 extraction rather than further training-state unification

Exit criteria:

- one internal splat model is used across training internals
- conversion code is centralized
- RGB and SH semantics both round-trip

### Story 2.1: Define Internal `Splats` for Training-Only Use

As a RustGS maintainer,
I want a single internal splat representation,
So that training modules stop exchanging partially overlapping parameter structures.

Acceptance criteria:

- `Splats` captures position, rotation, log-scale, opacity parameters, color parameters, and SH-rest coefficients
- the type lives under training internals or another explicitly internal namespace
- the first version does not replace `GaussianMap` as a public scene type

### Story 2.2: Centralize Scene-to-Trainable Conversion

As a RustGS maintainer,
I want `GaussianMap`, `Splats`, and `TrainableGaussians` conversion paths centralized,
So that data-model translation is explicit and testable.

Acceptance criteria:

- conversion code is moved out of ad hoc spread across the trainer path
- RGB and SH cases are both supported
- conversion tests cover round-trip correctness where applicable

### Story 2.3: Replace Ad Hoc Parameter Views with the Unified Model

As a RustGS maintainer,
I want parameter-view helper code to rely on the internal unified model,
So that topology, backward, and optimizer layers work from the same state assumptions.

Acceptance criteria:

- `SplatParameterView` or equivalent helpers are either removed or reduced to a narrow compatibility adapter
- topology and optimizer extraction can consume the same canonical internal representation
- row-count, color-layout, and SH-layout invariants are validated centrally

## Epic 3: Forward and Metal Runtime Extraction

Goal: Make the Metal forward path modular enough that orchestration code can call it without embedding runtime details.

Implementation note as of 2026-04-07:

- `training::metal_forward` now owns the forward-pass data contracts (`ProjectedGaussians`, `RenderedFrame`, forward profiling structs), projection helpers, and the concrete forward executor helpers for projection, tile binning, rasterization, and native-forward parity checks
- `metal_trainer.rs` now prepares camera-specific SH colors and then calls the `metal_forward` API instead of embedding the forward execution path inline
- trainer-only compatibility wrappers for direct forward-path tests remain, but the production training loop no longer owns the forward dispatch implementation details
- `stage_projected_records_from_tensors()` now reuses the extracted forward row-conversion helper instead of maintaining a second tensor-to-projection-row path
- Metal shader source now lives under `training/shaders/*.metal`, with `training::metal_kernels` owning kernel metadata and shader-source selection
- `training::metal_pipelines` now owns the pipeline-cache responsibility previously inlined inside `metal_runtime.rs`
- `training::metal_resources` now owns persistent buffer allocation/reuse, tensor binding, and Metal buffer read/write helpers, reducing `metal_runtime.rs` to dispatch/orchestration-heavy code instead of mixed ownership
- `training::metal_dispatch` now owns the simple compute launch wrappers for fill, gradient-magnitude, projected-gradient-magnitude, and fused-Adam kernels
- `training::metal_projection` now owns Gaussian projection kernel dispatch, projection-record staging, and CPU/GPU tile-bin construction
- `training::metal_raster` now owns forward/backward raster buffer preparation, target/SSIM staging, and native Metal raster dispatch
- `metal_runtime.rs` has been reduced to the shared runtime facade, Metal-specific type definitions, camera staging, and compatibility delegates, so the heavy execution paths are no longer concentrated there even though Story 3.2 is still not fully closed

Exit criteria:

- shader source is externalized
- runtime sub-responsibilities are separated
- trainer calls a forward-facing API rather than internal dispatch internals

### Story 3.1: Move MSL Source Out of `metal_runtime.rs`

As a RustGS maintainer,
I want Metal shader source files separated from runtime Rust code,
So that kernel logic and runtime control flow can evolve independently.

Acceptance criteria:

- MSL kernels are moved into dedicated shader files
- runtime compilation still succeeds without changing functional behavior
- build or load failures surface clear errors

Status as of 2026-04-07:

- implemented via `training/shaders/*.metal` plus `training::metal_kernels`
- `metal_runtime.rs` no longer embeds the MSL source strings directly
- `cargo check -p rustgs --all-features` and targeted GPU regression tests still pass after the move

### Story 3.2: Split the Metal Runtime into Concrete Subsystems

As a RustGS maintainer,
I want pipeline cache, buffer layout, and dispatch responsibilities separated,
So that runtime changes do not require editing one monolithic file.

Acceptance criteria:

- runtime code is divided into concrete modules for shader loading, pipeline lookup, resource layout, and dispatch helpers
- the split preserves the current Metal-only behavior
- no new multi-backend trait is introduced unless the split demonstrates a real need

Status as of 2026-04-07:

- partially implemented: pipeline-cache ownership now lives in `training::metal_pipelines`
- shader loading/kernel metadata now lives in `training::metal_kernels`
- persistent buffer ownership, tensor binding, and Metal read/write helpers now live in `training::metal_resources`
- simple compute dispatch wrappers now live in `training::metal_dispatch`
- projection dispatch and tile-bin construction now live in `training::metal_projection`
- forward/backward raster launch setup now lives in `training::metal_raster`
- `metal_runtime.rs` still centralizes shared Metal-side data contracts (`MetalProjectionRecord`, `MetalTileBins`, buffer-slot enums, runtime stats) plus compatibility delegates, so Story 3.2 remains active but is no longer the top architecture risk
- `cargo check -p rustgs --all-features`
- `cargo test -p rustgs --features gpu render_ -- --nocapture`
- `cargo test -p rustgs --features gpu training_step_ -- --nocapture`
- `cargo test -p rustgs --features gpu adam_step_var_ -- --nocapture`

### Story 3.3: Introduce a Stable Forward Boundary

As a RustGS maintainer,
I want a dedicated forward API,
So that trainer orchestration can request projection and rasterization without owning their implementation details.

Acceptance criteria:

- there is an explicit forward-facing API returning projected and rendered outputs
- forward outputs include the auxiliary state needed by later backward work
- trainer orchestration no longer directly manipulates low-level forward dispatch details

Status as of 2026-04-07:

- implemented via `training::metal_forward::{execute_forward_pass, project_gaussians, rasterize, build_tile_bins, profile_native_forward}`
- `MetalTrainer::render()` now calls that boundary after camera/color preparation
- low-level runtime split is still deferred to Story 3.2

### Story 3.4: Preserve Existing Rendering Namespace Clarity

As a RustGS maintainer,
I want the new forward-training modules named clearly,
So that CPU rendering and training forward rendering are not confused.

Acceptance criteria:

- naming avoids collapsing current top-level `render/` semantics into unrelated training-forward code
- module ownership between CPU render code and Metal training-forward code is documented
- the migration does not break existing public render exports

## Epic 4: Backward and Optimizer Extraction

Goal: Separate gradient computation and parameter updates from trainer orchestration so they can be reasoned about and tested independently.

Implementation note as of 2026-04-07:

- `training::metal_backward` now owns the explicit `MetalParameterGrads` container, backward loss scales, and parameter-gradient assembly for SH/color and optional rotation terms
- `training::metal_backward` also now owns a concrete backward request/execution boundary for SSIM staging, target-buffer staging, and raster backward dispatch
- `training::metal_loss` now owns structured step-loss evaluation, SSIM gradient generation, depth-validity scaling, loss telemetry packing, and backward loss-scale calculation
- a new internal `training::metal_optimizer` module now owns `MetalAdamState`, `MetalOptimizerConfig`, optimizer-step dispatch, Adam tensor helpers, and Adam-state rebuild logic
- `metal_trainer.rs` now asks `metal_loss` for a step-loss context, asks the backward layer for a full parameter-gradient container, and then delegates parameter updates to `metal_optimizer::apply_optimizer_step(...)`
- sparse and dense Adam paths are now isolated from the training loop, and the reorder/rebuild path no longer reconstructs Adam state inline inside the trainer
- the low-level raster backward still begins with `MetalBackwardGrads`, and a small amount of optimizer-adjacent regularization / pose plumbing still lives in the trainer, so Epic 4 is materially closer but not fully finished overall

Exit criteria:

- backward returns a stable gradient structure
- optimizer updates are isolated
- trainer composes forward, loss, backward, and optimize explicitly

### Story 4.1: Introduce a Stable Gradient Container

As a RustGS maintainer,
I want backward output represented by an explicit gradient structure,
So that trainer logic does not manually assemble or reinterpret gradient tensors.

Acceptance criteria:

- a single gradient container covers position, rotation, scale, opacity, color, and SH-rest gradients
- backward code returns that structure consistently
- regression tests validate shape and parameter-group alignment

Status as of 2026-04-07:

- largely implemented via `training::metal_backward::{MetalParameterGrads, assemble_parameter_grads}`
- the production trainer path now consumes a full parameter-gradient container before the optimizer step
- the remaining gap is mostly cleanup: the raw raster-backward container is still separate from the final parameter-gradient container, but the production step path no longer assembles parameter-group gradients inline

### Story 4.2: Extract Optimizer Step Logic and State Management

As a RustGS maintainer,
I want optimizer state and update logic isolated from trainer orchestration,
So that topology changes and parameter updates can be validated independently.

Acceptance criteria:

- Adam state initialization and rebuild logic live outside the main trainer control flow
- sparse-gradient and dense-gradient cases are both supported
- optimizer tests cover reorder, prune, and split/clone state behavior

Status as of 2026-04-07:

- largely implemented via `training::metal_optimizer::{MetalAdamState, rebuild_adam_state, apply_optimizer_step}`
- sparse and dense update paths are both routed through the extracted optimizer module
- reorder/rebuild regression coverage still passes through the trainer test suite

### Story 4.3: Make Trainer Orchestration Explicit

As a RustGS maintainer,
I want trainer code to read as forward, loss, backward, optimize, and optional refine,
So that the control flow is comprehensible without stepping through implementation detail modules.

Acceptance criteria:

- trainer orchestration calls explicit module boundaries for forward, loss, backward, and optimizer work
- step-level profiling and telemetry still work
- external training behavior remains unchanged

Status as of 2026-04-07:

- largely implemented: the production step path now reads as forward -> loss -> backward -> parameter-gradient assembly -> optimize -> topology
- `metal_loss::evaluate_training_step_loss(...)` now owns loss-term calculation, loss telemetry packing, SSIM gradient generation, and backward loss-scale preparation
- `metal_backward::execute_backward_pass(...)` now owns target/SSIM staging plus raster backward dispatch
- `MetalTrainer::training_step()` now reads as forward -> loss -> backward -> parameter-gradient assembly -> optimize -> topology on the production path
- the remaining cleanup is mostly secondary thinning around pose-gradient handling and the raw-raster-grad to parameter-grad handoff rather than the core step sequence itself

## Epic 5: Topology Strategy Migration

Goal: Separate topology policies from trainer control flow while preserving current LiteGS and legacy semantics.

Implementation note as of 2026-04-07:

- `training::topology` now owns the explicit topology schedule contract (`TopologySchedule`, `TopologyStepContext`) plus an extracted execution-planning layer (`TopologyExecutionPlan`, `TopologyExecutionDisposition`)
- snapshot mutation orchestration for densify/prune and LiteGS Morton reorder now lives in `training::topology::apply_snapshot_mutations(...)` instead of being assembled inline inside `metal_trainer.rs`
- `metal_trainer.rs` still owns topology-side effects that are tightly coupled to trainer state: opacity reset application, Adam-state rebuild, cluster reassignment, runtime buffer reservation, topology telemetry mutation, and topology logging
- Epic 5 is now in active implementation rather than just planned, but the trainer still owns enough topology lifecycle state that Story 5.1 and Story 5.2 remain only partially complete

Exit criteria:

- topology schedule and statistics contracts are explicit
- a baseline-compatible strategy exists
- LiteGS reference behavior is preserved during migration

### Story 5.1: Define the Topology Contract

As a RustGS maintainer,
I want topology scheduling, statistics input, and mutation results explicitly typed,
So that densify and prune behavior is no longer embedded in trainer internals.

Acceptance criteria:

- topology schedule inputs and outputs are explicitly modeled
- topology mutation results include enough information for telemetry and optimizer-state rebuild
- schedule logic remains compatible with current iteration and epoch semantics

Status as of 2026-04-07:

- partially implemented via `TopologySchedule`, `TopologyExecutionPlan`, `TopologyExecutionDisposition`, and `TopologyMutationResult`
- the topology module now returns typed schedule/decision/mutation information instead of leaving those branches embedded entirely in `metal_trainer.rs`
- trainer-owned telemetry mutation and optimizer/cluster side effects still sit outside the topology contract, so the story remains active

### Story 5.2: Introduce a Baseline-Compatible Topology Strategy

As a RustGS maintainer,
I want a strategy wrapper for the current production topology semantics,
So that the first strategy migration preserves behavior instead of replacing it.

Acceptance criteria:

- the first strategy implementation matches current active semantics
- the strategy can be exercised without introducing a second production training path
- the strategy boundary is testable in isolation

Status as of 2026-04-07:

- partially implemented: the active production semantics are now routed through `training::topology` execution helpers rather than trainer-local branch assembly
- topology-focused regression coverage still passes through the existing GPU trainer tests and `training::topology` unit tests
- the code is still concrete rather than trait-based, which remains intentional until the extraction settles

### Story 5.3: Adapt `density_controller.rs` as a Reference Strategy or Adapter

As a RustGS maintainer,
I want the LiteGS-compatible density controller preserved behind a strategy boundary,
So that parity work is not lost during the refactor.

Acceptance criteria:

- `density_controller.rs` is not deleted in the first migration wave
- its behavior is either wrapped or adapted behind the new topology contract
- parity-sensitive logic remains available for comparison and regression work

### Story 5.4: Add Topology Regression and Telemetry Coverage

As a RustGS maintainer,
I want densify, prune, and opacity-reset outcomes verified,
So that strategy extraction does not break primitive counts, visibility behavior, or optimizer rebuilds.

Acceptance criteria:

- tests cover clone, split, prune, and reset cases
- telemetry and reporting continue to expose topology metrics
- primitive-count evolution remains observable during training runs

## Epic 6: Orchestration Convergence and Cleanup

Goal: Land the refactor as a stable orchestration model while preserving the current external training surface.

Exit criteria:

- orchestration is thinner and clearer
- external behavior remains stable
- old monolithic files are reduced to necessary compatibility layers or removed

### Story 6.1: Converge on a Thin Training Orchestrator

As a RustGS maintainer,
I want one orchestration layer that coordinates loading, stepping, topology, telemetry, and finalization,
So that the active training flow is discoverable and maintainable.

Acceptance criteria:

- orchestration owns lifecycle flow instead of step-detail implementation
- chunked and non-chunked flows both remain supported
- profile-based routing remains intact

### Story 6.2: Preserve External Entry and Evaluation Behavior

As a RustGS user,
I want training, evaluation, and scene export to behave the same way during refactor rollout,
So that migration work does not force downstream changes.

Acceptance criteria:

- `rustgs::train()`, CLI train, scene export, and evaluation entry behavior remain compatible
- chunked training still routes and merges as before unless explicitly changed in a later plan
- any intentional behavior changes are documented and gated

### Story 6.3: Reduce Legacy File Ownership Safely

As a RustGS maintainer,
I want old monolithic files reduced only after replacement paths are proven,
So that cleanup does not outrun validation.

Acceptance criteria:

- `metal_trainer.rs` and `metal_runtime.rs` lose extracted responsibilities incrementally
- compatibility shims remain only where still required
- dead code is removed only after regression gates pass

### Story 6.4: Reassess Trait Abstractions After Concrete Extraction

As a RustGS maintainer,
I want abstraction layers introduced only when justified by the extracted code shape,
So that the final architecture stays pragmatic instead of ceremonial.

Acceptance criteria:

- `MetalKernel` or similar traits are introduced only if post-extraction duplication justifies them
- if no strong benefit exists, the runtime remains concrete and well-factored instead
- the decision is documented as an explicit architectural outcome

## Recommended Execution Order

Execute the epics in this order:

1. Epic 1
2. Epic 2
3. Epic 3
4. Epic 4
5. Epic 5
6. Epic 6

This order is intentional:

- Epic 1 prevents blind refactoring
- Epic 2 creates the internal model needed by later module splits
- Epic 3 and Epic 4 peel forward and backward responsibilities away from the trainer
- Epic 5 moves topology into a strategy boundary without deleting reference logic too early
- Epic 6 consolidates and cleans up only after the extracted pieces are proven

## Immediate Planning Guidance

If implementation starts now, the first execution slice should be:

1. capture compatibility and regression guardrails
2. define internal `Splats` without changing public scene types
3. externalize MSL shader source and split `metal_runtime.rs`

Do not begin with:

- rewriting public `core/` scene types
- renaming the top-level `render/` module around the new training-forward boundary
- deleting `density_controller.rs`
- introducing a generic kernel trait before runtime decomposition

## Final Recommendation

Proceed with the refactor, but execute it as an internal compatibility-preserving migration rather than an immediate top-level package rewrite.

The original proposal is good as a diagnosis and good as a direction.
This revised plan is the version that is safe to ship.
