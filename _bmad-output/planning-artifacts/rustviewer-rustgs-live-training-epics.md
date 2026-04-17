---
stepsCompleted:
  - step-01-requirements-extracted
  - step-02-epics-approved
  - step-03-stories-generated
  - step-04-final-validation
inputDocuments:
  - _bmad-output/planning-artifacts/prd.md
  - _bmad-output/planning-artifacts/architecture.md
  - _bmad-output/planning-artifacts/rustviewer-gui-design.md
  - _bmad-output/planning-artifacts/rustviewer-gui-design-refined.md
  - user-request: "RustViewer loads COLMAP datasets, invokes RustGS 3DGS training, and shows live training previews that continue updating for the current dragged viewpoint."
---

# RustScan - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown draft for RustScan, focused on integrating COLMAP dataset loading and live RustGS training visualization into RustViewer without regressing the existing viewer interaction model.

## Requirements Inventory

### Functional Requirements

FR1: RustViewer shall let the user select and load a COLMAP dataset directory that contains sparse reconstruction files and referenced images.

FR2: RustViewer shall use RustGS dataset-loading APIs to convert the selected COLMAP dataset into a `TrainingDataset` suitable for RustGS training.

FR3: After loading a COLMAP dataset, RustViewer shall visualize the dataset trajectory and sparse points in the existing 3D viewport.

FR4: RustViewer shall provide UI controls to configure and start a RustGS 3DGS training session from the loaded dataset.

FR5: RustViewer shall run RustGS training asynchronously so the UI remains interactive during training.

FR6: RustViewer shall display training lifecycle state including idle, loading, training, stopping, completed, and failed states.

FR7: RustViewer shall display per-training progress information including at least current iteration, latest loss, elapsed time, and current Gaussian count.

FR8: RustGS shall expose intermediate training progress events that can be consumed by RustViewer during an active training run.

FR9: RustGS shall expose intermediate host-side Gaussian snapshots during training at a configurable cadence so RustViewer can visualize evolving results.

FR10: RustViewer shall update its Gaussian visualization from the latest training snapshot while training is still running.

FR11: RustViewer shall provide a rendered training preview for the current Gaussian state rather than only a 3D world-space splat display.

FR12: The rendered preview shall support a user-controlled viewpoint that can be changed interactively while training is running.

FR13: When the user drags or otherwise changes the preview viewpoint, RustViewer shall re-render the latest available training snapshot from the new viewpoint.

FR14: As training iterations continue after a viewpoint change, RustViewer shall refresh the preview using the current viewpoint and newer training snapshots.

FR15: RustViewer shall keep existing manual camera controls for the 3D world viewport while adding the live training preview workflow.

FR16: RustViewer shall provide a way to stop or cancel an in-progress training session without killing the application.

FR17: RustViewer shall surface clear validation and runtime errors for invalid COLMAP datasets, missing images, training failures, and preview-render failures.

FR18: RustViewer shall preserve or restore the latest successful training result after training completes so the user can continue inspecting the scene.

### NonFunctional Requirements

NFR1: The RustViewer UI shall remain responsive during dataset loading, training, and preview refresh operations.

NFR2: Training-state sharing between RustGS and RustViewer shall use explicit thread-safe message passing or synchronization primitives rather than ad hoc shared mutable state.

NFR3: Intermediate snapshot generation shall be cadence-controlled to avoid unacceptable training slowdowns from per-iteration device-to-host readback.

NFR4: Preview rendering shall prefer existing RustGS rendering/evaluation primitives over duplicate renderer implementations.

NFR5: The integration shall reuse RustGS COLMAP loading and `TrainingDataset` types rather than introducing a parallel dataset representation.

NFR6: The feature shall preserve existing RustViewer viewport navigation behavior and current static file-loading workflows.

NFR7: The new UI controls and panels shall follow the existing RustViewer layout direction and refined design constraints from the viewer GUI design documents.

NFR8: Error reporting shall remain actionable and specific enough for users to recover from invalid dataset paths, unsupported camera/image layouts, and training/runtime failures.

NFR9: The implementation shall be incremental and testable in phases: dataset load, background training orchestration, snapshot streaming, and live preview.

NFR10: The feature shall target the existing macOS and wgpu-based RustViewer environment already established by the project architecture.

### Additional Requirements

- The implementation should not replace RustViewer's current `Scene`-based static visualization path; training-state management should be introduced as a separate layer that feeds the viewer.

- RustViewer should add a dedicated COLMAP loader path rather than forcing users to pre-export intermediate JSON artifacts.

- RustGS training events currently only expose run start and completion; intermediate iteration metrics and snapshot events need to be added to support the viewer workflow.

- RustViewer currently renders 3D scene content from CPU scene data every frame; live training preview should be introduced as a distinct preview surface or panel rather than conflating it with the world viewport.

- The 2D training preview should be driven by the latest `HostSplats` snapshot and a camera bridge that derives a RustGS render camera from the viewer-controlled preview camera and dataset intrinsics.

- The viewer should support both "dataset-derived scene context" and "free inspection camera" behavior without blocking ongoing training.

- Existing side-panel concepts from the RustViewer GUI design remain relevant: file operations, layer controls, rendering/training settings, statistics, and camera guidance.

- The refined RustViewer design guidance adds constraints for spacing, state transitions, visual hierarchy, and accessibility that should apply to any new training controls or preview panels.

### FR Coverage Map

FR1: Epic 17 - Load COLMAP dataset directories from RustViewer.
FR2: Epic 17 - Convert COLMAP data into RustGS `TrainingDataset`.
FR3: Epic 17 - Visualize trajectory and sparse points before training.
FR4: Epic 18 - Provide training configuration and start controls.
FR5: Epic 18 - Run training asynchronously without blocking the UI.
FR6: Epic 18 - Show training lifecycle state transitions.
FR7: Epic 18 - Show iteration, loss, elapsed time, and Gaussian-count progress.
FR8: Epic 18 - Expose intermediate RustGS progress events for viewer consumption.
FR9: Epic 19 - Expose intermediate host-side Gaussian snapshots at a controlled cadence.
FR10: Epic 19 - Update viewer Gaussian content from the latest training snapshot.
FR11: Epic 19 - Render a dedicated training preview from the current Gaussian state.
FR12: Epic 20 - Support a user-controlled preview viewpoint during training.
FR13: Epic 20 - Re-render the latest snapshot from the new viewpoint after user interaction.
FR14: Epic 20 - Keep refreshing the preview from the current viewpoint as newer snapshots arrive.
FR15: Epic 20 - Preserve existing world-viewport camera controls while adding preview controls.
FR16: Epic 18 - Stop or cancel an in-progress training session safely.
FR17: Epic 17 / Epic 18 - Surface actionable validation and runtime errors.
FR18: Epic 19 - Preserve the latest successful training result for post-training inspection.

## Epic List

### Epic 17: Load And Inspect COLMAP Training Datasets
Users can load a COLMAP dataset into RustViewer, validate it for RustGS use, and inspect trajectory plus sparse scene context before training starts.
**FRs covered:** FR1, FR2, FR3, FR17

### Epic 18: Run And Monitor RustGS Training From RustViewer
Users can configure, start, stop, and monitor a RustGS training session inside RustViewer while keeping the UI responsive and understanding current training state.
**FRs covered:** FR4, FR5, FR6, FR7, FR8, FR16, FR17

### Epic 19: View Evolving Gaussian Results During Training
Users can see the latest Gaussian state while training is still running, both as updated scene content and as a dedicated training-result visualization driven by streamed RustGS snapshots.
**FRs covered:** FR9, FR10, FR11, FR18

### Epic 20: Inspect Live Training Previews From User-Controlled Viewpoints
Users can interactively move the preview viewpoint during training and continue seeing newer training iterations rendered from that currently selected viewpoint without losing existing world-viewport controls.
**FRs covered:** FR12, FR13, FR14, FR15

## Epic 17: Load And Inspect COLMAP Training Datasets

RustViewer can load a COLMAP directory through RustGS-native dataset loading, validate required files, and show the resulting trajectory and sparse points in the existing world viewport.

### Story 17.1: RustViewer Can Resolve A COLMAP Dataset Into A TrainingDataset

As a RustViewer user,
I want the viewer to load a COLMAP directory directly into a RustGS-compatible dataset,
So that I can start from reconstruction outputs without creating extra intermediate files.

**Acceptance Criteria:**

**Given** a directory that contains a valid COLMAP sparse reconstruction and image folder
**When** RustViewer invokes the COLMAP loading path
**Then** the directory is converted through RustGS into a `TrainingDataset`
**And** the resulting dataset retains image paths, poses, intrinsics, and sparse initialization points.

**Given** a directory that is missing sparse files, images, or required COLMAP content
**When** the loader validates the directory
**Then** the load fails with a specific validation error
**And** the error distinguishes missing reconstruction structure from missing referenced images.

### Story 17.2: Loaded COLMAP Datasets Populate The Existing Viewer Scene

As a RustViewer user,
I want loaded COLMAP data to appear in the existing world viewport,
So that I can inspect camera trajectory and sparse scene structure before training.

**Acceptance Criteria:**

**Given** a successfully loaded `TrainingDataset`
**When** RustViewer maps dataset content into its scene model
**Then** the trajectory is populated from dataset poses
**And** sparse initialization points are populated as map points with bounds updated accordingly.

**Given** a dataset that contains valid poses and sparse points
**When** the scene is refreshed after loading
**Then** the existing viewport can render the trajectory and sparse points using current layer controls
**And** auto-fit places the camera around the loaded scene bounds.

### Story 17.3: RustViewer Exposes A COLMAP Load Flow And Dataset Load Status

As a RustViewer user,
I want a clear COLMAP load action and load-status feedback in the side panel,
So that I can understand whether the dataset is ready for training.

**Acceptance Criteria:**

**Given** the RustViewer side panel
**When** I choose the COLMAP load action and select a directory
**Then** RustViewer starts the load on a background thread
**And** the UI shows loading, success, or failure state without freezing the viewport.

**Given** a load success or failure
**When** the operation completes
**Then** the side panel shows dataset readiness details such as frame count, point count, and image resolution on success
**And** shows a clear actionable error on failure.

## Epic 18: Run And Monitor RustGS Training From RustViewer

RustViewer can launch a background RustGS training session from the loaded dataset, surface live progress, and stop training safely while preserving UI responsiveness.

### Story 18.1: RustViewer Manages Background RustGS Training Sessions

As a RustViewer user,
I want training to run in the background with explicit session state,
So that I can keep using the viewer while a model is being optimized.

**Acceptance Criteria:**

**Given** a loaded dataset and a valid training configuration
**When** I start a training run from RustViewer
**Then** the training work starts on a background session rather than the UI thread
**And** the viewer tracks session state including idle, starting, training, stopping, completed, and failed.

**Given** an active training session
**When** the viewport camera moves or the side panel is used
**Then** the UI remains responsive
**And** the application does not block waiting for the training loop.

### Story 18.2: RustGS Emits Intermediate Training Progress Events

As a RustViewer-integrated training client,
I want RustGS to emit intermediate progress events during training,
So that the viewer can show live iteration-level status instead of only start and finish.

**Acceptance Criteria:**

**Given** an active RustGS training run with an event sink attached
**When** training iterations advance
**Then** RustGS emits intermediate progress events containing at least iteration index, latest loss, elapsed timing, and current Gaussian count
**And** the existing run-start and run-complete events remain available.

**Given** a training consumer that does not need intermediate progress
**When** the normal RustGS API is used without a viewer session
**Then** existing training entrypoints continue to work
**And** the new progress-event path remains additive rather than breaking existing callers.

### Story 18.3: RustViewer Shows Training Controls And Progress Metrics

As a RustViewer user,
I want to configure and monitor training from the side panel,
So that I can run RustGS from the viewer and understand its current progress.

**Acceptance Criteria:**

**Given** a loaded COLMAP dataset
**When** I open the training controls in RustViewer
**Then** I can edit the supported training parameters needed for the first integrated workflow
**And** I can start training only when the dataset is in a valid ready state.

**Given** an active training session
**When** progress events arrive from RustGS
**Then** the UI updates the visible iteration, loss, elapsed time, and Gaussian count
**And** the displayed state transitions match the underlying session lifecycle.

### Story 18.4: RustViewer Can Stop Training And Surface Runtime Failures

As a RustViewer user,
I want to stop a training run or understand why it failed,
So that I stay in control of long-running work and can recover from errors.

**Acceptance Criteria:**

**Given** an active training session
**When** I choose the stop or cancel action
**Then** RustViewer sends a stop request through the training-session control path
**And** the session transitions to stopping and then to a terminal completed-or-cancelled state without killing the application.

**Given** a load, training, or runtime failure
**When** the failure is reported back to RustViewer
**Then** the side panel shows a specific actionable error message
**And** the session state is reset so the user can retry after correcting the issue.

## Epic 19: View Evolving Gaussian Results During Training

RustViewer can receive streamed Gaussian snapshots from RustGS, update its scene representation from those snapshots, render a dedicated training preview, and retain the latest completed result for continued inspection.

### Story 19.1: RustGS Streams Host-Side Gaussian Snapshots At A Configurable Cadence

As a RustViewer-integrated training client,
I want RustGS to produce host-side Gaussian snapshots at a controlled cadence,
So that the viewer can visualize evolving results without forcing full readback every iteration.

**Acceptance Criteria:**

**Given** a RustGS training run configured with snapshot streaming enabled
**When** the configured snapshot cadence is reached
**Then** RustGS emits a host-side `HostSplats` snapshot event
**And** the event includes the iteration number and associated progress metadata.

**Given** snapshot streaming is enabled
**When** cadence is increased or reduced
**Then** snapshot emission frequency follows the configured cadence
**And** the feature does not require per-iteration host readback to function.

### Story 19.2: RustViewer Applies Snapshot Updates To Live Gaussian Scene State

As a RustViewer user,
I want the latest training snapshot to update visible Gaussian content while training runs,
So that I can observe the scene converging in the viewer.

**Acceptance Criteria:**

**Given** a snapshot event received from RustGS
**When** RustViewer ingests the snapshot
**Then** it updates a dedicated training-result state object with the latest `HostSplats`
**And** the scene-facing Gaussian visualization can refresh from that state without corrupting static loaded content.

**Given** multiple snapshot events arrive in sequence
**When** RustViewer processes them
**Then** only the latest snapshot becomes the active visualization source
**And** the update path remains safe under concurrent training and UI repaint activity.

### Story 19.3: RustViewer Renders A Dedicated Training Preview From The Latest Snapshot

As a RustViewer user,
I want a dedicated rendered preview of the current training result,
So that I can evaluate image-space output rather than only world-space Gaussian positions.

**Acceptance Criteria:**

**Given** a latest available `HostSplats` snapshot and dataset intrinsics
**When** RustViewer requests a training preview render
**Then** the preview is rendered through RustGS rendering or evaluation primitives
**And** the preview is shown in a dedicated viewer surface or panel separate from the world viewport.

**Given** no training snapshot is yet available
**When** the preview panel is visible
**Then** the UI shows a clear waiting or empty state
**And** the world viewport continues to function normally.

### Story 19.4: RustViewer Retains The Latest Successful Training Result For Inspection

As a RustViewer user,
I want the latest successful training result to remain inspectable after training ends,
So that I can continue exploring the final output without rerunning training.

**Acceptance Criteria:**

**Given** a training session completes successfully
**When** the final snapshot or final splat result is available
**Then** RustViewer preserves it as the active completed training result
**And** both the scene visualization and preview can continue using it after the training thread exits.

**Given** a completed training result is being inspected
**When** no new training session has started
**Then** the final result remains available in the UI
**And** completion status is distinguishable from in-progress preview updates.

## Epic 20: Inspect Live Training Previews From User-Controlled Viewpoints

RustViewer lets the user drive the preview camera independently from the world viewport and keeps the live training preview updating from that currently selected viewpoint as newer snapshots arrive.

### Story 20.1: RustViewer Bridges Viewer Camera State Into A RustGS Preview Camera

As a RustViewer user,
I want my preview viewpoint to map correctly into RustGS rendering space,
So that the preview I see matches the camera pose I am controlling.

**Acceptance Criteria:**

**Given** a preview-camera state in RustViewer and dataset intrinsics
**When** RustViewer constructs a RustGS preview camera
**Then** the resulting render camera preserves the current viewpoint orientation and position semantics
**And** intrinsics are scaled correctly to the preview panel resolution.

**Given** the preview panel is resized
**When** the preview camera is rebuilt for rendering
**Then** the rendered frame uses the updated panel dimensions
**And** the camera-to-image mapping remains consistent.

### Story 20.2: Users Can Interactively Change The Training Preview Viewpoint

As a RustViewer user,
I want to drag, pan, and zoom the training preview viewpoint while training is active,
So that I can inspect emerging geometry and appearance from arbitrary viewpoints.

**Acceptance Criteria:**

**Given** the training preview is visible
**When** I perform the supported preview camera interactions
**Then** RustViewer updates the preview-camera state independently of the world viewport camera
**And** the latest available snapshot is re-rendered from the new viewpoint.

**Given** the world viewport remains visible
**When** I interact with the preview controls
**Then** the existing world viewport camera controls still behave as before
**And** the two camera states do not overwrite each other.

### Story 20.3: The Preview Continues Refreshing From The Current Viewpoint As Training Advances

As a RustViewer user,
I want newer training iterations to keep updating the preview from the viewpoint I last selected,
So that I can hold a viewpoint constant and watch convergence over time.

**Acceptance Criteria:**

**Given** I have changed the preview viewpoint during an active training run
**When** newer snapshot events arrive
**Then** RustViewer renders the new snapshots using the currently selected preview-camera state
**And** it does not revert to a default or dataset camera unless I explicitly request that.

**Given** snapshot updates and camera interactions can occur close together
**When** RustViewer schedules preview rerenders
**Then** it uses the latest camera state and latest snapshot available
**And** outdated preview work can be dropped or replaced to preserve responsiveness.
