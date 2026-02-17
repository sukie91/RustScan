---
stepsCompleted: [1, 2, 3, 4]
inputDocuments:
  - _bmad-output/planning-artifacts/prd.md
  - _bmad-output/planning-artifacts/architecture.md
---

# RustScan - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for RustScan, decomposing the requirements from the PRD and Architecture into implementable stories.

**Epic Completion Status:**
- ✅ Epic 1: CLI Infrastructure & Configuration (6/6 stories complete)
- ⏳ Epic 2: Video Input & Decoding (0/3 stories)
- ⏳ Epic 3: SLAM Processing Pipeline (0/6 stories)
- ⏳ Epic 4: 3DGS Training & Scene Generation (0/6 stories)
- ⏳ Epic 5: Mesh Extraction & Export (0/5 stories)
- ⏳ Epic 6: End-to-End Pipeline Integration (0/5 stories)

**Overall Progress:** 6/31 stories (19.4%)

## Requirements Inventory

### Functional Requirements

**Video Input Processing:**
- FR1: Users can input iPhone video files (MP4/MOV/HEVC)
- FR2: System validates video format and reports errors

**SLAM Processing:**
- FR3: System extracts features (ORB/Harris/FAST)
- FR4: System performs feature matching between frames
- FR5: System estimates camera poses
- FR6: System executes bundle adjustment
- FR7: System detects and closes loops

**3DGS Training:**
- FR8: System performs 3DGS training with depth constraints
- FR9: System utilizes GPU acceleration (Metal/MPS)
- FR10: System outputs trained 3DGS scene files

**Mesh Generation:**
- FR11: System fuses depth maps into TSDF volume
- FR12: System extracts mesh via Marching Cubes
- FR13: System outputs exportable mesh files (OBJ/PLY)

**CLI Interface:**
- FR14: Users execute complete pipeline via command line
- FR15: System runs in non-interactive mode
- FR16: System outputs structured data (JSON)
- FR17: System reads configuration files (YAML/TOML)
- FR18: Command-line arguments override config settings

**Logging & Diagnostics:**
- FR19: System outputs configurable log levels
- FR20: System provides clear error messages with recovery suggestions
- FR21: System provides diagnostic information on failure

### Non-Functional Requirements

**Performance:**
- NFR1: Processing Time ≤ 30 minutes (2-3 minute video)
- NFR2: 3DGS Rendering PSNR > 28 dB
- NFR3: SLAM Tracking Success Rate > 95%
- NFR4: Mesh Quality < 1% isolated triangles

**Integration:**
- NFR5: Output formats: OBJ, PLY mesh files
- NFR6: Compatibility: Blender and Unity importable

**Scriptability:**
- NFR7: Non-interactive execution (automation-friendly)
- NFR8: No prompts during execution
- NFR9: Structured output (JSON format)

### Additional Requirements

**From Architecture Decisions:**

1. **CLI Framework (ADR-001):**
   - Use clap with derive macros for type-safe argument parsing
   - Support configuration files (YAML/TOML) via clap-serde
   - Auto-generate help documentation

2. **Video Decoding (ADR-002):**
   - Use ffmpeg-next for hardware-accelerated decoding
   - Implement on-demand decoding with LRU cache
   - Support VideoToolbox hardware acceleration on macOS

3. **Pipeline Architecture (ADR-003):**
   - Sequential execution with checkpoint mechanism
   - Support recovery from failures
   - Save intermediate results for debugging

4. **Math Library (ADR-004):**
   - Use glam exclusively for all 3D math operations
   - Remove nalgebra dependency (dead code)
   - Ensure SIMD optimization throughout

5. **Output Management (ADR-005):**
   - Support multiple output formats (OBJ + PLY)
   - Generate structured metadata (JSON)
   - Organize output directory with clear structure

6. **Logging System (ADR-006):**
   - Use log + env_logger
   - Support configurable log levels
   - Optional JSON output format for automation

7. **Configuration Management (ADR-007):**
   - TOML configuration file format
   - Serde-based validation
   - Default values with override capability

8. **Implementation Rules:**
   - Follow 85 rules in project-context.md
   - Maintain type safety (u32 for handles, Result/Option for errors)
   - Use SoA memory layout for RustMesh
   - Preserve Half-edge invariants

**Project Context:**
- Brownfield project (~85% complete)
- Phase 1 core pipeline already connected
- ~98 files, ~27K lines of code, 245+ tests
- Rust Edition 2021, Apple Silicon target

### FR Coverage Map

| FR | Epic 1 | Epic 2 | Epic 3 | Epic 4 | Epic 5 | Epic 6 |
|----|--------|--------|--------|--------|--------|--------|
| FR1: Video input (MP4/MOV/HEVC) | | ✓ | | | | ✓ |
| FR2: Video format validation | | ✓ | | | | ✓ |
| FR3: Feature extraction | | | ✓ | | | ✓ |
| FR4: Feature matching | | | ✓ | | | ✓ |
| FR5: Camera pose estimation | | | ✓ | | | ✓ |
| FR6: Bundle adjustment | | | ✓ | | | ✓ |
| FR7: Loop detection & closing | | | ✓ | | | ✓ |
| FR8: 3DGS training with depth | | | | ✓ | | ✓ |
| FR9: GPU acceleration (Metal/MPS) | | | | ✓ | | ✓ |
| FR10: 3DGS scene output | | | | ✓ | | ✓ |
| FR11: TSDF volume fusion | | | | | ✓ | ✓ |
| FR12: Marching Cubes extraction | | | | | ✓ | ✓ |
| FR13: Mesh export (OBJ/PLY) | | | | | ✓ | ✓ |
| FR14: CLI execution | ✓ | | | | | ✓ |
| FR15: Non-interactive mode | ✓ | | | | | ✓ |
| FR16: Structured output (JSON) | ✓ | | | | ✓ | ✓ |
| FR17: Config file support | ✓ | | | | | ✓ |
| FR18: CLI argument override | ✓ | | | | | ✓ |
| FR19: Configurable log levels | ✓ | | | | | ✓ |
| FR20: Clear error messages | ✓ | | | | | ✓ |
| FR21: Diagnostic information | ✓ | | | | | ✓ |

**Coverage Summary:**
- Epic 1 (CLI Infrastructure): FR14-21 (8 FRs)
- Epic 2 (Video Input): FR1-2 (2 FRs)
- Epic 3 (SLAM Pipeline): FR3-7 (5 FRs)
- Epic 4 (3DGS Training): FR8-10 (3 FRs)
- Epic 5 (Mesh Extraction): FR11-13, FR16 (4 FRs)
- Epic 6 (E2E Integration): All FRs (21 FRs)

## Epic List

### Epic 1: CLI Infrastructure & Configuration ✅

**Status:** ✅ COMPLETE (Verified 2026-02-17)

**Description:**
Establish the command-line interface foundation that enables users to execute the RustScan pipeline with flexible configuration options, comprehensive logging, and clear error reporting.

**Functional Requirements:**
- FR14: CLI execution of complete pipeline ✅
- FR15: Non-interactive mode support ✅
- FR16: Structured JSON output ✅
- FR17: Configuration file support (TOML) ✅
- FR18: CLI argument override capability ✅
- FR19: Configurable log levels ✅
- FR20: Clear error messages with recovery suggestions ✅
- FR21: Diagnostic information on failure ✅

**Non-Functional Requirements:**
- NFR7: Non-interactive execution (automation-friendly) ✅
- NFR8: No prompts during execution ✅
- NFR9: Structured output (JSON format) ✅

**Architecture Decisions:**
- ADR-001: clap with derive macros for CLI ✅
- ADR-006: log + env_logger for logging ✅
- ADR-007: TOML configuration management ✅

**Success Criteria:**
- Users can run `rustscan --input video.mp4 --output ./results` successfully ✅
- Configuration files override defaults, CLI args override config ✅
- Log levels (trace/debug/info/warn/error) work correctly ✅
- JSON output contains all pipeline metadata ✅
- Error messages include actionable recovery suggestions ✅

**Dependencies:**
- None (foundational epic)

**Implementation:**
- Location: `RustSLAM/src/cli/mod.rs` (606 lines)
- Entry point: `RustSLAM/src/main.rs`
- All 6 stories fully implemented
- Includes comprehensive error handling with structured diagnostics
- Supports both JSON and text output formats
- Exit codes: 0 (success), 1 (user error), 2 (system error)

**Notes:**
- Implementation exceeds acceptance criteria
- Unit tests not yet added (recommended for future work)
- Video decoder integration already present

---

### Epic 2: Video Input & Decoding

**Description:**
Implement robust video input handling with hardware-accelerated decoding for iPhone video formats, supporting efficient frame extraction with validation and error handling.

**Functional Requirements:**
- FR1: Input iPhone video files (MP4/MOV/HEVC)
- FR2: Video format validation and error reporting

**Non-Functional Requirements:**
- NFR1: Processing time ≤ 30 minutes (2-3 minute video)

**Architecture Decisions:**
- ADR-002: ffmpeg-next with hardware acceleration
- ADR-003: On-demand decoding with LRU cache

**Success Criteria:**
- Decode MP4/MOV/HEVC formats from iPhone
- Hardware acceleration (VideoToolbox) works on macOS
- Invalid formats produce clear error messages
- Frame extraction performance meets NFR1 targets
- LRU cache reduces memory footprint

**Dependencies:**
- Epic 1 (CLI for input path handling)

---

### Epic 3: SLAM Processing Pipeline

**Description:**
Implement the complete Visual SLAM pipeline including feature extraction, matching, pose estimation, bundle adjustment, and loop closure detection to generate accurate camera trajectories and sparse 3D maps.

**Functional Requirements:**
- FR3: Feature extraction (ORB/Harris/FAST)
- FR4: Feature matching between frames
- FR5: Camera pose estimation
- FR6: Bundle adjustment optimization
- FR7: Loop detection and closing

**Non-Functional Requirements:**
- NFR3: SLAM tracking success rate > 95%

**Architecture Decisions:**
- ADR-003: Sequential pipeline with checkpoints
- ADR-004: glam for all 3D math operations

**Success Criteria:**
- Feature extraction produces stable keypoints
- Matching achieves > 95% tracking success rate
- Pose estimation converges reliably
- Bundle adjustment reduces reprojection error
- Loop closure improves trajectory consistency
- Checkpoint recovery works after failures

**Dependencies:**
- Epic 2 (video frames as input)

---

### Epic 4: 3DGS Training & Scene Generation

**Description:**
Implement 3D Gaussian Splatting training with depth constraints from SLAM, utilizing GPU acceleration to generate high-quality scene representations suitable for mesh extraction.

**Functional Requirements:**
- FR8: 3DGS training with depth constraints
- FR9: GPU acceleration (Metal/MPS)
- FR10: Trained 3DGS scene file output

**Non-Functional Requirements:**
- NFR2: 3DGS rendering PSNR > 28 dB
- NFR1: Processing time ≤ 30 minutes

**Architecture Decisions:**
- ADR-003: Checkpoint mechanism for training
- ADR-004: glam for Gaussian math operations

**Success Criteria:**
- Training converges with PSNR > 28 dB
- Metal/MPS GPU acceleration works on Apple Silicon
- Depth constraints improve geometric accuracy
- Scene files save/load correctly
- Training completes within time budget

**Dependencies:**
- Epic 3 (camera poses and sparse map)

---

### Epic 5: Mesh Extraction & Export

**Description:**
Implement TSDF volume fusion and Marching Cubes mesh extraction to convert 3DGS scenes into exportable mesh formats with high quality and minimal artifacts.

**Functional Requirements:**
- FR11: TSDF volume fusion from depth maps
- FR12: Marching Cubes mesh extraction
- FR13: Mesh export (OBJ/PLY formats)
- FR16: Structured JSON metadata output

**Non-Functional Requirements:**
- NFR4: Mesh quality < 1% isolated triangles
- NFR5: Output formats OBJ, PLY
- NFR6: Blender and Unity compatibility

**Architecture Decisions:**
- ADR-005: Multiple output formats with metadata
- ADR-004: glam for mesh vertex operations

**Success Criteria:**
- TSDF fusion produces clean volumes
- Marching Cubes generates watertight meshes
- < 1% isolated triangles in output
- OBJ/PLY files import correctly in Blender/Unity
- JSON metadata includes mesh statistics

**Dependencies:**
- Epic 4 (3DGS scene for depth rendering)

---

### Epic 6: End-to-End Pipeline Integration

**Description:**
Integrate all pipeline stages into a cohesive end-to-end workflow with checkpoint management, progress reporting, and comprehensive validation to ensure reliable execution from video input to mesh output.

**Functional Requirements:**
- All FRs (FR1-FR21)

**Non-Functional Requirements:**
- All NFRs (NFR1-NFR9)

**Architecture Decisions:**
- ADR-003: Sequential pipeline with checkpoints
- All ADRs apply

**Success Criteria:**
- Complete pipeline runs video → mesh successfully
- Checkpoint recovery works at each stage
- Progress reporting shows current stage
- All NFR targets met (time, quality, success rate)
- Integration tests pass for full pipeline
- Example videos process correctly

**Dependencies:**
- Epic 1 (CLI infrastructure)
- Epic 2 (video input)
- Epic 3 (SLAM pipeline)
- Epic 4 (3DGS training)
- Epic 5 (mesh extraction)

---

# Epic Stories

## Epic 1: CLI Infrastructure & Configuration

### Story 1.1: Basic CLI Argument Parsing

As a developer,
I want to execute RustScan with input/output arguments,
So that I can specify video files and output directories from the command line.

**Acceptance Criteria:**

**Given** the RustScan CLI is installed
**When** I run `rustscan --input video.mp4 --output ./results`
**Then** the CLI parses arguments correctly
**And** validates that input file exists
**And** creates output directory if it doesn't exist
**And** displays clear error if input file is missing

**Requirements:** FR14 (CLI execution)

---

### Story 1.2: Configuration File Support

As a developer,
I want to load pipeline settings from a TOML config file,
So that I can reuse configurations across multiple runs without repeating CLI arguments.

**Acceptance Criteria:**

**Given** a valid `rustscan.toml` config file exists
**When** I run `rustscan --config rustscan.toml`
**Then** the CLI loads all settings from the config file
**And** validates the TOML structure using serde
**And** displays clear error messages for invalid config syntax
**And** uses default values for missing optional fields

**Requirements:** FR17 (config file support), ADR-007 (TOML format)

---

### Story 1.3: CLI Argument Override

As a developer,
I want CLI arguments to override config file settings,
So that I can quickly test variations without editing the config file.

**Acceptance Criteria:**

**Given** a config file specifies `output = "./default_output"`
**When** I run `rustscan --config rustscan.toml --output ./custom_output`
**Then** the CLI uses `./custom_output` instead of the config value
**And** all other config settings remain unchanged
**And** the override is logged at debug level

**Requirements:** FR18 (CLI override), ADR-001 (clap framework)

---

### Story 1.4: Configurable Logging System

As a developer,
I want to control log verbosity via CLI or config,
So that I can see detailed diagnostics during debugging or minimal output in production.

**Acceptance Criteria:**

**Given** the RustScan CLI is running
**When** I set `--log-level debug` or `RUST_LOG=debug`
**Then** the system outputs debug-level logs
**And** supports all levels: trace, debug, info, warn, error
**And** logs include timestamps and module names
**And** log output goes to stderr (not stdout)

**Requirements:** FR19 (configurable log levels), ADR-006 (log + env_logger)

---

### Story 1.5: Structured JSON Output

As a developer,
I want pipeline results exported as JSON,
So that I can integrate RustScan into automated workflows and parse results programmatically.

**Acceptance Criteria:**

**Given** the pipeline completes successfully
**When** I run with `--output-format json`
**Then** the system generates a `results.json` file
**And** JSON includes: input video path, processing time, camera count, mesh statistics
**And** JSON is valid and parseable
**And** JSON includes error information if pipeline fails

**Requirements:** FR16 (structured JSON output), NFR9 (structured output)

---

### Story 1.6: Error Handling with Recovery Suggestions

As a developer,
I want clear error messages with actionable recovery suggestions,
So that I can quickly diagnose and fix issues without reading documentation.

**Acceptance Criteria:**

**Given** an error occurs during execution
**When** the error is displayed to the user
**Then** the message includes: error type, root cause, affected component
**And** provides specific recovery suggestions (e.g., "Install ffmpeg: brew install ffmpeg")
**And** includes relevant diagnostic information (file paths, system info)
**And** exits with appropriate error codes (0=success, 1=user error, 2=system error)

**Requirements:** FR20 (clear error messages), FR21 (diagnostic information)

---

## Epic 2: Video Input & Decoding

### Story 2.1: Video Format Detection and Validation

As a developer,
I want the system to detect and validate video formats,
So that I receive clear errors for unsupported formats before processing begins.

**Acceptance Criteria:**

**Given** a video file path is provided
**When** the system validates the input
**Then** it detects MP4, MOV, and HEVC formats correctly
**And** reports codec information (H.264, H.265, etc.)
**And** displays clear error for unsupported formats
**And** validates that the file is readable and not corrupted

**Requirements:** FR1, FR2 (video input and validation)

---

### Story 2.2: Hardware-Accelerated Video Decoding

As a developer,
I want video decoding to use hardware acceleration,
So that frame extraction is fast and doesn't consume excessive CPU resources.

**Acceptance Criteria:**

**Given** a valid iPhone video file (MP4/MOV/HEVC)
**When** the system decodes frames
**Then** it uses VideoToolbox hardware acceleration on macOS
**And** falls back to software decoding if hardware is unavailable
**And** logs which decoder is being used (hardware/software)
**And** decodes frames at native resolution without quality loss

**Requirements:** FR1, ADR-002 (ffmpeg-next with hardware acceleration)

---

### Story 2.3: On-Demand Frame Extraction with LRU Cache

As a developer,
I want frames to be decoded on-demand with caching,
So that memory usage stays reasonable for long videos.

**Acceptance Criteria:**

**Given** a video with 1000+ frames
**When** the SLAM pipeline requests frames
**Then** frames are decoded on-demand (not all at once)
**And** an LRU cache stores recently accessed frames
**And** cache size is configurable (default: 100 frames)
**And** memory usage stays below 2GB for typical videos

**Requirements:** FR1, ADR-002 (on-demand decoding with LRU cache), NFR1 (performance)

---

## Epic 3: SLAM Processing Pipeline

### Story 3.1: Feature Extraction (ORB/Harris/FAST)

As a developer,
I want the system to extract stable keypoints from video frames,
So that feature matching can establish frame-to-frame correspondences.

**Acceptance Criteria:**

**Given** a decoded video frame
**When** feature extraction runs
**Then** it extracts 500-2000 keypoints per frame using ORB
**And** supports Harris and FAST detectors as alternatives
**And** computes ORB descriptors for each keypoint
**And** filters keypoints by response threshold
**And** distributes keypoints across the image (not clustered)

**Requirements:** FR3 (feature extraction)

---

### Story 3.2: Feature Matching Between Frames

As a developer,
I want the system to match features between consecutive frames,
So that camera motion can be estimated.

**Acceptance Criteria:**

**Given** two consecutive frames with extracted features
**When** feature matching runs
**Then** it matches features using descriptor distance (Hamming for ORB)
**And** applies ratio test (Lowe's ratio) to filter ambiguous matches
**And** achieves > 95% inlier rate after RANSAC
**And** handles low-texture scenes gracefully (minimum 50 matches)

**Requirements:** FR4 (feature matching), NFR3 (tracking success rate > 95%)

---

### Story 3.3: Camera Pose Estimation

As a developer,
I want the system to estimate camera poses from feature matches,
So that the camera trajectory can be reconstructed.

**Acceptance Criteria:**

**Given** feature matches between frames
**When** pose estimation runs
**Then** it computes relative pose using Essential matrix decomposition
**And** uses RANSAC to reject outliers
**And** triangulates 3D points from inlier matches
**And** initializes map with first two keyframes
**And** tracks pose for subsequent frames using PnP

**Requirements:** FR5 (camera pose estimation), ADR-004 (glam for math)

---

### Story 3.4: Bundle Adjustment Optimization

As a developer,
I want the system to refine camera poses and 3D points,
So that accumulated drift is minimized and reconstruction accuracy improves.

**Acceptance Criteria:**

**Given** a set of keyframes with 3D map points
**When** bundle adjustment runs
**Then** it optimizes camera poses and 3D point positions jointly
**And** minimizes reprojection error across all observations
**And** runs local BA every 5 keyframes
**And** runs global BA after loop closure
**And** reduces mean reprojection error below 1.0 pixel

**Requirements:** FR6 (bundle adjustment)

---

### Story 3.5: Loop Detection and Closing

As a developer,
I want the system to detect when the camera revisits a location,
So that trajectory drift can be corrected through loop closure.

**Acceptance Criteria:**

**Given** a sequence of keyframes with BoW descriptors
**When** loop detection runs
**Then** it detects loops using BoW similarity scoring
**And** verifies loop candidates with geometric consistency check
**And** computes loop closure constraint (relative pose)
**And** triggers global bundle adjustment after loop closure
**And** corrects accumulated drift in the trajectory

**Requirements:** FR7 (loop detection and closing)

---

### Story 3.6: Pipeline Checkpoint and Recovery

As a developer,
I want the SLAM pipeline to save checkpoints,
So that processing can resume after failures without starting over.

**Acceptance Criteria:**

**Given** the SLAM pipeline is running
**When** a checkpoint is triggered (every 50 frames)
**Then** it saves: keyframes, map points, camera poses, BoW database
**And** checkpoint files are written to `<output>/checkpoints/slam_*.ckpt`
**And** pipeline can resume from latest checkpoint on restart
**And** logs checkpoint save/load operations

**Requirements:** ADR-003 (sequential pipeline with checkpoints)

---

## Epic 4: 3DGS Training & Scene Generation

### Story 4.1: Gaussian Initialization from SLAM

As a developer,
I want 3D Gaussians initialized from SLAM sparse points,
So that training starts with reasonable geometry.

**Acceptance Criteria:**

**Given** SLAM has produced sparse 3D map points
**When** Gaussian initialization runs
**Then** it creates one Gaussian per map point
**And** initializes position from 3D point coordinates
**And** initializes scale from nearest neighbor distance
**And** initializes rotation as identity
**And** initializes opacity to 0.5
**And** initializes SH coefficients from point color

**Requirements:** FR8 (3DGS training with depth constraints)

---

### Story 4.2: Differentiable Gaussian Rasterization (GPU)

As a developer,
I want Gaussians rendered using GPU-accelerated differentiable rasterization,
So that training is fast enough for practical use.

**Acceptance Criteria:**

**Given** a set of 3D Gaussians and a camera pose
**When** rendering runs
**Then** it uses Metal/MPS GPU acceleration on Apple Silicon
**And** implements tiled rasterization for efficiency
**And** performs depth sorting per tile
**And** computes alpha blending in front-to-back order
**And** renders at camera resolution (e.g., 1920x1080)
**And** rendering takes < 50ms per frame on M1/M2

**Requirements:** FR9 (GPU acceleration Metal/MPS), ADR-004 (glam for math)

---

### Story 4.3: 3DGS Training with Depth Loss

As a developer,
I want 3DGS training to use depth constraints from SLAM,
So that geometric accuracy is enforced during optimization.

**Acceptance Criteria:**

**Given** initialized Gaussians and training views with depth maps
**When** training runs
**Then** it optimizes Gaussian parameters (position, scale, rotation, opacity, SH)
**And** uses combined loss: RGB + depth + SSIM
**And** depth loss enforces consistency with SLAM depth
**And** training runs for configurable iterations (default: 3000)
**And** achieves PSNR > 28 dB on training views

**Requirements:** FR8 (depth constraints), NFR2 (PSNR > 28 dB)

---

### Story 4.4: Gaussian Densification and Pruning

As a developer,
I want the system to densify under-reconstructed regions and prune redundant Gaussians,
So that scene representation is both complete and efficient.

**Acceptance Criteria:**

**Given** Gaussians being trained
**When** densification runs (every 100 iterations)
**Then** it clones Gaussians in under-reconstructed regions (high gradient)
**And** splits large Gaussians covering multiple features
**And** prunes Gaussians with low opacity (< 0.05)
**And** prunes Gaussians with excessive scale
**And** maintains reasonable Gaussian count (< 1M for typical scenes)

**Requirements:** FR8 (3DGS training)

---

### Story 4.5: 3DGS Scene File Export

As a developer,
I want trained Gaussian scenes saved to disk,
So that they can be loaded for mesh extraction or visualization.

**Acceptance Criteria:**

**Given** a trained 3DGS scene
**When** export runs
**Then** it saves Gaussians to `<output>/scene.ply` format
**And** includes all parameters: position, scale, rotation, opacity, SH coefficients
**And** file format is compatible with standard 3DGS viewers
**And** includes metadata: training iterations, final loss, Gaussian count
**And** scene can be reloaded for further processing

**Requirements:** FR10 (3DGS scene output)

---

### Story 4.6: Training Checkpoint and Resume

As a developer,
I want 3DGS training to save checkpoints,
So that training can resume after interruption without losing progress.

**Acceptance Criteria:**

**Given** 3DGS training is running
**When** a checkpoint is triggered (every 500 iterations)
**Then** it saves: Gaussian parameters, optimizer state, iteration count
**And** checkpoint files are written to `<output>/checkpoints/3dgs_*.ckpt`
**And** training can resume from latest checkpoint
**And** logs checkpoint save/load operations

**Requirements:** ADR-003 (checkpoint mechanism), NFR1 (processing time)

---

## Epic 5: Mesh Extraction & Export

### Story 5.1: TSDF Volume Fusion from Depth Maps

As a developer,
I want depth maps from Gaussian rendering fused into a TSDF volume,
So that a volumetric representation is created for mesh extraction.

**Acceptance Criteria:**

**Given** a trained 3DGS scene and camera poses
**When** TSDF fusion runs
**Then** it renders depth maps from multiple viewpoints
**And** integrates depth into TSDF volume with configurable voxel size (default: 0.01m)
**And** uses truncation distance of 3x voxel size
**And** accumulates weights for each voxel
**And** handles occlusions correctly (front-to-back integration)

**Requirements:** FR11 (TSDF volume fusion)

---

### Story 5.2: Marching Cubes Mesh Extraction

As a developer,
I want the system to extract a triangle mesh from the TSDF volume,
So that a surface representation is generated.

**Acceptance Criteria:**

**Given** a fused TSDF volume
**When** Marching Cubes runs
**Then** it extracts mesh vertices and triangles
**And** uses full 256-case lookup table for correctness
**And** interpolates vertex positions for smooth surfaces
**And** interpolates vertex colors from TSDF
**And** generates watertight mesh (no holes)

**Requirements:** FR12 (Marching Cubes extraction)

---

### Story 5.3: Mesh Post-Processing

As a developer,
I want extracted meshes cleaned of artifacts,
So that output quality meets the < 1% isolated triangles requirement.

**Acceptance Criteria:**

**Given** a raw mesh from Marching Cubes
**When** post-processing runs
**Then** it removes isolated triangle clusters (< 100 triangles)
**And** smooths vertex normals for better shading
**And** validates mesh topology (manifold, watertight)
**And** achieves < 1% isolated triangles in final output
**And** logs mesh statistics (vertex count, triangle count, removed clusters)

**Requirements:** NFR4 (mesh quality < 1% isolated triangles)

---

### Story 5.4: Mesh Export (OBJ and PLY)

As a developer,
I want meshes exported in standard formats,
So that they can be imported into Blender, Unity, and other 3D tools.

**Acceptance Criteria:**

**Given** a post-processed mesh
**When** export runs
**Then** it saves mesh as `<output>/mesh.obj` and `<output>/mesh.ply`
**And** OBJ includes vertex positions, normals, and colors
**And** PLY includes vertex positions, normals, colors, and faces
**And** files are valid and importable in Blender 3.x
**And** files are valid and importable in Unity 2022+
**And** coordinate system matches industry standards (Y-up for OBJ)

**Requirements:** FR13 (mesh export OBJ/PLY), NFR5, NFR6 (Blender/Unity compatibility)

---

### Story 5.5: Mesh Metadata JSON Export

As a developer,
I want mesh statistics and metadata exported as JSON,
So that I can programmatically analyze results and track quality metrics.

**Acceptance Criteria:**

**Given** a completed mesh extraction
**When** metadata export runs
**Then** it generates `<output>/mesh_metadata.json`
**And** includes: vertex count, triangle count, bounding box, isolated triangle percentage
**And** includes: TSDF voxel size, truncation distance, viewpoint count
**And** includes: processing time for each stage
**And** JSON is valid and parseable

**Requirements:** FR16 (structured JSON output), NFR9 (structured output)

---

## Epic 6: End-to-End Pipeline Integration

### Story 6.1: Sequential Pipeline Orchestration

As a developer,
I want all pipeline stages executed in sequence,
So that video input flows through to mesh output automatically.

**Acceptance Criteria:**

**Given** a video file and output directory
**When** the pipeline runs
**Then** it executes stages in order: video decode → SLAM → 3DGS → mesh extraction
**And** each stage receives output from the previous stage
**And** pipeline stops on first error with clear diagnostics
**And** logs progress for each stage (e.g., "SLAM: 45/100 frames processed")

**Requirements:** ADR-003 (sequential pipeline), all FRs

---

### Story 6.2: Cross-Stage Checkpoint Management

As a developer,
I want the pipeline to resume from the last completed stage,
So that failures don't require reprocessing everything.

**Acceptance Criteria:**

**Given** a pipeline that failed at stage 3 (3DGS training)
**When** the pipeline restarts
**Then** it detects existing checkpoints for stages 1-2
**And** skips completed stages (video decode, SLAM)
**And** resumes from stage 3 checkpoint
**And** logs which stages are being skipped vs. resumed
**And** validates checkpoint integrity before resuming

**Requirements:** ADR-003 (checkpoint mechanism), NFR1 (processing time)

---

### Story 6.3: Progress Reporting and Logging

As a developer,
I want real-time progress updates during pipeline execution,
So that I can monitor processing status and estimate completion time.

**Acceptance Criteria:**

**Given** the pipeline is running
**When** progress updates are generated
**Then** it logs current stage and progress percentage
**And** logs estimated time remaining (based on current stage speed)
**And** logs memory usage and GPU utilization
**And** progress goes to stderr (not stdout)
**And** supports JSON log format for automation (`--log-format json`)

**Requirements:** FR19 (configurable logging), NFR7 (automation-friendly)

---

### Story 6.4: End-to-End Integration Tests

As a developer,
I want integration tests that validate the complete pipeline,
So that regressions are caught before release.

**Acceptance Criteria:**

**Given** a test video dataset
**When** integration tests run
**Then** they execute the full pipeline on test videos
**And** validate output files exist (mesh.obj, mesh.ply, results.json)
**And** validate quality metrics (PSNR > 28 dB, tracking > 95%, isolated triangles < 1%)
**And** validate processing time < 30 minutes for 2-3 minute video
**And** tests run in CI/CD pipeline

**Requirements:** All NFRs (NFR1-4)

---

### Story 6.5: Example Video Processing

As a developer,
I want example videos and expected outputs included,
So that I can verify the pipeline works correctly after installation.

**Acceptance Criteria:**

**Given** the RustScan repository
**When** I run the example workflow
**Then** it includes 2-3 sample iPhone videos (< 100MB each)
**And** includes expected output meshes for comparison
**And** includes a script to run examples: `./run_examples.sh`
**And** examples complete successfully on Apple Silicon Macs
**And** README documents how to run examples

**Requirements:** All FRs (end-to-end validation)

---
