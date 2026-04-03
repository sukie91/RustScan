---
stepsCompleted: [1, 2, 3, 4]
inputDocuments:
  - _bmad-output/planning-artifacts/prd.md
  - _bmad-output/planning-artifacts/architecture.md
  - _bmad-output/implementation-artifacts/tech-spec-rust-viewer-3d-gui.md
---

# RustScan - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for RustScan, decomposing the requirements from the PRD, UX Design if it exists, and Architecture requirements into implementable stories.

## Requirements Inventory

### Functional Requirements

FR1: Users can input iPhone video files (MP4/MOV/HEVC)
FR2: System validates video format and reports errors
FR3: System extracts features (ORB/Harris/FAST)
FR4: System performs feature matching between frames
FR5: System estimates camera poses
FR6: System executes bundle adjustment
FR7: System detects and closes loops
FR8: System performs 3DGS training with depth constraints
FR9: System utilizes GPU acceleration (Metal/MPS)
FR10: System outputs trained 3DGS scene files
FR11: System fuses depth maps into TSDF volume
FR12: System extracts mesh via Marching Cubes
FR13: System outputs exportable mesh files (OBJ/PLY)
FR14: Users execute complete pipeline via command line
FR15: System runs in non-interactive mode
FR16: System outputs structured data (JSON)
FR17: System reads configuration files (YAML/TOML)
FR18: Command-line arguments override config settings
FR19: System outputs configurable log levels
FR20: System provides clear error messages with recovery suggestions
FR21: System provides diagnostic information on failure

### NonFunctional Requirements

NFR1: Processing Time: ≤30 minutes (2-3 minute video)
NFR2: 3DGS Rendering Quality: PSNR > 28 dB
NFR3: SLAM Tracking Success Rate: >95%
NFR4: Mesh Quality: <1% isolated triangles
NFR5: Output Formats: OBJ, PLY mesh files
NFR6: Compatibility: Blender and Unity importable

### Additional Requirements

**From Architecture Document:**
- Use glam as the only math library, remove nalgebra dependency
- Metal/MPS backend for Apple Silicon GPU acceleration
- Parallel processing using rayon for data parallelism, crossbeam-channel for thread communication
- Error handling: library code returns Result/Option, use thiserror for custom error types
- 3DGS → Mesh extraction: TSDF volume + Marching Cubes with post-processing (cluster filtering, normal smoothing)
- Configuration management: TOML config files + serde validation
- Logging system: log + env_logger + structured output options
- Performance optimization: Use default config for MVP, defer optimization to Phase 2
- Pipeline integration: Sequential execution + checkpoint mechanism for reliability and debugging

**From RustViewer Tech Spec (tech-spec-rust-viewer-3d-gui.md):**
- New `RustViewer/` crate: Interactive 3D visualization GUI for SLAM results
- Offline file loading: `slam_checkpoint.json` (camera trajectory + map points), `scene.ply` (Gaussian point cloud), `mesh.obj/ply` (extracted mesh)
- 3D rendering layer (wgpu): Camera trajectory polylines, sparse point clouds, Gaussian point clouds, Mesh wireframe and solid faces
- egui control panel: File selection, layer visibility toggles, rendering mode switching
- 3D camera control: Mouse drag rotation, wheel zoom, right-click panning (arcball camera)
- Cargo workspace configuration: Add RustViewer as workspace member alongside RustMesh and RustSLAM
- RustSLAM feature gating: Add `viewer-types` feature to gate heavy dependencies (ffmpeg-next, candle-core, candle-metal)
- GPU rendering pipelines: Point cloud/Gaussian pipeline, trajectory polyline pipeline, Mesh pipeline
- Unit testing for data loading and camera calculations
- 11 acceptance criteria for GUI functionality and user interaction

### FR Coverage Map

FR1, FR2 → Epic 1: 视频输入处理
FR3, FR4, FR5, FR6, FR7 → Epic 2: SLAM处理管道
FR8, FR9, FR10 → Epic 3: 3DGS训练和场景生成 → **迁移至 Epic 9: RustGS**
FR11, FR12, FR13 → Epic 4: 网格提取和导出 → **迁移至 Epic 10: RustMesh网格提取**
FR14, FR15, FR16, FR17, FR18, FR19, FR20, FR21 → Epic 5: CLI接口和配置管理
NFR1, NFR2, NFR3, NFR4 → Epic 6: 端到端管道集成 → **重构为 Epic 11: 管道集成更新**
架构技术需求 → Epic 7: 基础设施质量保证
RustViewer需求 → Epic 8: RustViewer 3D可视化GUI

## Epic List

### Epic 1: 视频输入处理
用户可以输入iPhone视频文件，系统验证格式并提供清晰的错误反馈
**FRs covered:** FR1, FR2

### Epic 2: SLAM处理管道
系统可以提取特征、匹配帧、估计相机位姿、优化地图并检测回环
**FRs covered:** FR3, FR4, FR5, FR6, FR7

### Epic 3: 3DGS训练和场景生成
系统可以使用GPU加速训练3D高斯体，并输出可渲染的场景文件
**FRs covered:** FR8, FR9, FR10

### Epic 4: 网格提取和导出
系统可以将深度图融合成TSDF体积，提取网格并导出为行业标准格式
**FRs covered:** FR11, FR12, FR13

### Epic 5: CLI接口和配置管理
用户可以通过命令行运行完整管道，使用配置文件自定义参数，并获得结构化输出
**FRs covered:** FR14, FR15, FR16, FR17, FR18, FR19, FR20, FR21

### Epic 6: 端到端管道集成
系统可以顺序执行所有阶段，支持检查点和恢复，提供进度反馈
**FRs covered:** NFR1, NFR2, NFR3, NFR4

### Epic 7: 基础设施质量保证
系统具有线程安全的架构、统一配置、参数验证和错误处理
**FRs covered:** 架构需求（数学库统一、GPU加速、并行处理、错误处理、配置管理等）

### Epic 8: RustViewer 3D可视化GUI
用户可以交互式查看SLAM重建结果，包括相机轨迹、稀疏点云、高斯体和网格
**FRs covered:** RustViewer技术规范中的所有需求

### Epic 9: RustGS Crate 提取
将3DGS训练代码从RustSLAM分离到独立的RustGS crate，实现离线3DGS训练
**FRs covered:** FR8, FR9, FR10 (从 Epic 3 迁移)
**Stories:**
- 9-1: 创建 rustscan-types 共享 crate
- 9-2: 创建 RustGS crate 结构
- 9-3: 迁移核心 Gaussian 文件
- 9-4: 迁移渲染文件
- 9-5: 迁移可微渲染
- 9-6: 迁移训练文件
- 9-7: 迁移 IO 和初始化
- 9-8: 创建 RustGS CLI
- 9-9: 更新 RustSLAM 依赖

### Epic 10: RustMesh 网格提取集成
将网格提取功能从RustSLAM迁移到RustMesh，RustGS提供深度渲染API
**FRs covered:** FR11, FR12, FR13 (从 Epic 4 迁移)
**Stories:**
- 10-1: 迁移 TSDF 到 RustMesh
- 10-2: 迁移 Marching Cubes
- 10-3: 迁移网格提取器
- 10-4: 集成 RustGS 深度渲染
- 10-5: 创建网格提取 CLI

### Epic 11: 管道集成更新
更新端到端管道以支持新的架构：RustSLAM → RustGS → RustMesh
**FRs covered:** NFR1, NFR2, NFR3, NFR4 (重构)
**Stories:**
- 11-1: 更新 RustSLAM 输出格式
- 11-2: 创建端到端管道脚本
- 11-3: 更新文档

### Epic 12: 分块训练入口与预算控制
用户可以显式启用分块训练，并用内存预算驱动训练规模，使 RustGS 在 16GB 机器上避免直接 OOM
**FRs covered:** FR1, FR2, FR5, FR6, FR7, FR11, FR12

### Epic 13: 空间分块与相机子集构建
系统可以从 SlamOutput/TrainingDataset 自动生成空间块、overlap 和块级相机子集，为逐块训练准备稳定输入
**FRs covered:** FR3, FR4, FR5

### Epic 14: 逐块训练执行与块级产物导出
系统可以顺序训练每个 chunk，自动应用块级参数覆盖，并输出可合并的块级场景结果
**FRs covered:** FR6, FR7, FR8, FR11

### Epic 15: 场景合并与核心区保留
系统可以将多个 chunk 的训练结果合并为单个场景文件，并通过 merge-core-only 控制 overlap 区域的保留策略
**FRs covered:** FR8, FR9, FR10

### Epic 16: 状态跟踪、恢复上下文与长期开发可维护性
开发者可以基于文档 frontmatter 和每条 Story 状态字段持续推进长期开发，并在更换电脑或中断后快速恢复上下文
**FRs covered:** FR13, NFR4, NFR8

### Epic 17: LiteGS 对齐分析与验收 Harness
为 LiteGS-on-RustGS Mac 方案建立权威差异矩阵、固定 fixtures 和统一验收阈值，确保后续开发围绕同一套 parity 标准推进
**FRs covered:** FR8, FR9, FR10, NFR1, NFR2

### Epic 18: LiteGS Mac MVP 训练路径
在保留现有 LegacyMetal 行为的前提下，为 Apple Silicon 引入 LiteGS 兼容训练 profile、配置面和 Metal 训练入口
**FRs covered:** FR8, FR9, FR10, NFR1, NFR2

### Epic 19: LiteGS Cluster 与 Sparse-Gradient 路径
将 LiteGS 的 cluster、稀疏梯度和空间重排机制引入 RustGS Metal 运行时，实现 clustered parity
**FRs covered:** FR8, FR9, FR10, NFR1, NFR2

### Epic 20: LiteGS Densify/Prune/Opacity Reset 对齐
实现 LiteGS/TamingGS 的统计、densify、prune 与优化器状态变更语义，使拓扑编辑与上游保持一致
**FRs covered:** FR8, FR9, FR10, NFR1, NFR2

### Epic 21: 运行时对齐、导出恢复与默认切换
补齐 LiteGS 兼容路径的评估、checkpoint、PLY 往返、Mac 操作文档，并在验收通过后升级默认 profile
**FRs covered:** FR8, FR9, FR10, NFR1, NFR2, FR20, FR21

## Epic 12: 分块训练入口与预算控制

让 RustGS 具备正式的 chunked training 入口、预算模型和模式路由，使 16GB 机器能够以可预期方式启动分块训练。

### Story 12.1: Chunked Training CLI and Config Entry

Status: done

As a RustGS CLI user,
I want to enable chunked training and provide chunk-related options from the command line,
So that I can run training under a fixed memory budget on a 16GB machine without changing code.

**Acceptance Criteria:**

**Given** the existing `rustgs train` command
**When** I pass chunked-training flags
**Then** the CLI parses them successfully into `rustgs::TrainingConfig`

**Given** no chunk-related flags are passed
**When** training starts
**Then** the current non-chunked path behaves exactly as before

**Given** invalid chunk parameters are passed
**When** argument validation runs
**Then** the CLI returns a clear error message describing the invalid value and expected range

### Story 12.2: Budget-Driven Chunk Capacity Estimation

Status: done

As a RustGS training operator,
I want chunk capacity to be derived from a memory budget,
So that chunk sizing is predictable and safe on a 16GB machine.

**Acceptance Criteria:**

**Given** a chunk memory budget in GiB
**When** the estimator runs
**Then** it computes a maximum allowable per-chunk training scale using the existing Metal memory estimate model

**Given** a training input and budget
**When** the estimator predicts a chunk would exceed budget
**Then** the system marks the chunk for further subdivision or degradation rather than attempting unsafe training

### Story 12.3: Chunked Path Selection and Early Guardrails

Status: done

As a RustGS user,
I want the training entrypoint to route into chunked orchestration only when requested,
So that normal training and chunked training remain clearly separated.

**Acceptance Criteria:**

**Given** `chunked_training == true`
**When** training begins
**Then** the system enters the chunked training orchestration path

**Given** `chunked_training == false`
**When** training begins
**Then** the system uses the existing normal training path

### Story 12.4: Non-Chunked Compatibility and Entry Regression

Status: done

As a maintainer,
I want regression coverage around the new chunked entrypoint,
So that adding chunked mode does not break the existing trainer.

**Acceptance Criteria:**

**Given** existing non-chunked test inputs
**When** tests run
**Then** existing training behavior still passes unchanged

**Given** chunked mode is enabled on a small fixture
**When** integration tests run
**Then** the entrypoint reaches the chunk planner and completes without changing tracked file formats

## Epic 13: 空间分块与相机子集构建

让 RustGS 能从弱几何输入中自动切分空间块并构建块级相机子集，为逐块训练提供稳定数据准备能力。

### Story 13.1: Spatial Chunk Generation from Sparse Geometry or Camera Trajectory

Status: done

As a chunk planner,
I want to generate spatial chunks from sparse points or fallback camera trajectory bounds,
So that chunking works for SlamOutput-style inputs without per-frame depth.

**Acceptance Criteria:**

**Given** `initial_points` is present
**When** chunk planning starts
**Then** chunk bounds are derived from the point-cloud bounding box

**Given** `initial_points` is empty
**When** chunk planning starts
**Then** chunk bounds fall back to camera-trajectory bounds

### Story 13.2: Camera Assignment and Weak-Chunk Filtering

Status: done

As a chunk planner,
I want to assign cameras to chunks and filter weak chunks,
So that each chunk has enough observations to be trainable.

**Acceptance Criteria:**

**Given** a set of chunks and camera poses
**When** assignment runs
**Then** a camera is included in a chunk if its center or relevant sparse points intersect the chunk overlap region

**Given** a chunk has fewer than the configured minimum cameras
**When** chunk validation runs
**Then** the chunk is merged, skipped, or degraded according to a deterministic rule

### Story 13.3: Per-Chunk Dataset Materialization

Status: done

As a chunked trainer,
I want per-chunk datasets to be materialized from the global input,
So that each chunk can train independently on local views and local initialization.

**Acceptance Criteria:**

**Given** a validated chunk
**When** dataset materialization runs
**Then** the chunk dataset contains only the assigned poses and only the local initial points relevant to that chunk

**Given** a chunk has no usable local points
**When** dataset materialization runs
**Then** the system falls back to frame-based initialization or fails with a clear chunk-scoped diagnostic

### Story 13.4: Chunk Planning and Boundary Regression Tests

Status: done

As a maintainer,
I want deterministic tests for chunk planning and assignment,
So that planner behavior stays stable as training evolves.

**Acceptance Criteria:**

**Given** synthetic bounding boxes, points, and camera poses
**When** unit tests run
**Then** chunk count, overlap expansion, and camera assignment are deterministic

**Given** boundary cameras and overlap regions
**When** tests run
**Then** edge assignment cases are explicitly validated

## Epic 14: 逐块训练执行与块级产物导出

让 RustGS 能顺序训练每个 chunk、根据预算应用块级覆盖参数，并输出可合并的块级结果和执行报告。

### Story 14.1: Sequential Chunk Training Orchestrator

Status: done

As a RustGS user on a 16GB machine,
I want chunks to train one by one,
So that peak memory remains bounded.

**Acceptance Criteria:**

**Given** a list of trainable chunks
**When** chunked training starts
**Then** chunks are processed strictly sequentially rather than in parallel

**Given** one chunk finishes
**When** the next chunk begins
**Then** the previous chunk's heavy training state is released before continuing

### Story 14.2: Per-Chunk Adaptive Parameter Overrides

Status: done

As a chunked trainer,
I want to override training parameters per chunk when needed,
So that over-budget chunks can still finish under 16GB constraints.

**Acceptance Criteria:**

**Given** a chunk exceeds budget at default settings
**When** adaptive planning runs
**Then** the system first tries spatial subdivision, then chunk-local gaussian limits, then render-scale reduction

**Given** a chunk still cannot be made safe
**When** the planner exhausts allowed degradation
**Then** the chunk is marked failed or skipped with a clear reason

### Story 14.3: Chunk Artifact and Report Persistence

Status: done

As a maintainer,
I want chunk training to emit explicit intermediate outputs,
So that merge and debugging have stable inputs.

**Acceptance Criteria:**

**Given** a chunk finishes training successfully
**When** the trainer persists outputs
**Then** it produces a chunk scene artifact and a machine-readable chunk report entry

**Given** chunked training completes
**When** the overall report is written
**Then** it contains one entry per chunk with final status and effective parameters

### Story 14.4: Lifecycle Cleanup and Memory-Bounded Regression Validation

Status: done

As a maintainer,
I want proof that chunked execution actually limits memory pressure,
So that the feature meets its core goal.

**Acceptance Criteria:**

**Given** chunk-local resources are released
**When** moving from one chunk to the next
**Then** tests or diagnostics confirm stale trainer state is not retained across chunks

**Given** chunked training on a bounded fixture
**When** profiling or estimator-based regression tests run
**Then** per-chunk memory remains within the configured budget envelope

## Epic 15: 场景合并与核心区保留

让 RustGS 能把多个 chunk 的训练结果稳定合并为单场景输出，同时控制 overlap 区的高斯保留策略。

### Story 15.1: Core-Only Merge Filtering

Status: done

As a chunked training pipeline,
I want to keep only core-region gaussians by default,
So that overlap regions do not create duplicated seam geometry.

**Acceptance Criteria:**

**Given** a trained chunk with core and overlap AABBs
**When** merge-core-only is enabled
**Then** only gaussians whose centers fall inside the core AABB are retained for merge

### Story 15.2: Final Merged Scene Output

Status: done

As a RustGS user,
I want chunked training to still produce one final scene file,
So that downstream tooling does not need a new consumption model.

**Acceptance Criteria:**

**Given** multiple successful chunk outputs
**When** merge completes
**Then** the system produces one merged `GaussianMap` and one final PLY scene file

### Story 15.3: Merge Correctness and Seam Regression Tests

Status: done

As a maintainer,
I want tests around chunk merge correctness,
So that seam handling stays stable as training code evolves.

**Acceptance Criteria:**

**Given** synthetic chunk outputs with overlapping gaussians
**When** merge-core-only tests run
**Then** only expected core-region gaussians remain

## Epic 16: 状态跟踪、恢复上下文与长期开发可维护性

让长期开发和跨机器恢复有统一的状态字段、handoff 约定和工作流程，而不是依赖口头同步。

### Story 16.1: Story Status Field Convention

Status: done

As a project maintainer,
I want every story to carry an explicit status field,
So that progress is visible and durable across long-running development.

**Acceptance Criteria:**

**Given** the chunked-training epic/story document
**When** story sections are finalized
**Then** each story includes a required status field initialized to `todo`

### Story 16.2: Handoff Note Convention for Interrupted Work

Status: done

As a developer who may switch computers,
I want a lightweight handoff note per story,
So that I can resume work without reconstructing context from scratch.

**Acceptance Criteria:**

**Given** a story is in `in_progress` or `blocked`
**When** the developer pauses work
**Then** the story includes a short handoff note describing current state, next step, and known blocker

### Story 16.3: Chunked Training Development Workflow Documentation

Status: done

As a maintainer,
I want the chunked-training development workflow documented,
So that implementation, review, and status maintenance stay consistent over a long project duration.

**Acceptance Criteria:**

**Given** the chunked-training epic section
**When** documentation is finalized
**Then** it defines the required story lifecycle, status values, and completion/update rules

## Epic 17: LiteGS 对齐分析与验收 Harness

让 RustGS 的 LiteGS-on-Mac 开发以统一的差异矩阵、固定 fixture 和可复用验收阈值为依据，而不是在实现过程中临时解释“什么叫对齐”。

### Story 17.1: Full LiteGS-vs-RustGS Parity Matrix

Status: done

As a RustGS maintainer,
I want one authoritative subsystem-by-subsystem parity matrix,
So that every later LiteGS story has an explicit source of truth, mismatch statement, migration decision, acceptance metric, and owner.

**Acceptance Criteria:**

**Given** the LiteGS mirror and current RustGS implementation
**When** the parity document is written
**Then** it contains sections for training loop, dataset/camera model, Gaussian parameterization, render preprocess, loss, optimizer, densify/prune/reset, clustering/sparse-grad, evaluation/export, and Mac-specific constraints

**Given** a mismatch row in the matrix
**When** a maintainer reads it
**Then** the row points to both LiteGS and RustGS source files and names the owning story

### Story 17.2: Fixed Reference Fixtures for Parity Work

Status: done

As a parity harness owner,
I want two named reference fixtures with stable metadata,
So that correctness and convergence checks do not drift as implementation changes.

**Acceptance Criteria:**

**Given** the parity harness registry
**When** fixtures are enumerated
**Then** it defines one tiny correctness fixture and one Apple Silicon convergence fixture with stable IDs and notes

**Given** the canonical small COLMAP fixture is not yet checked in
**When** the harness is bootstrapped locally
**Then** the registry still records the intended COLMAP path and the temporary bootstrap dataset used meanwhile

### Story 17.3: Parity Harness Metrics Schema

Status: in_progress

As a LiteGS parity developer,
I want a reusable harness report format,
So that later training runs can record the same initialization, loss, topology, export, and timing metrics across every story.

**Acceptance Criteria:**

**Given** a parity run
**When** metrics are persisted
**Then** the schema can record initialization counts, active SH degree, loss terms, PSNR, Gaussian counts, densify/prune events, export outputs, checkpoint/export round-trips, and wall-clock timing

**Given** a stored parity report
**When** it is loaded back
**Then** the JSON round-trip is deterministic

### Story 17.4: Shared Acceptance Thresholds

Status: done

As a project maintainer,
I want one shared set of pass/fail thresholds,
So that every later LiteGS story validates against the same rules.

**Acceptance Criteria:**

**Given** the parity harness defaults
**When** thresholds are read
**Then** they require no NaNs, no OOM on Apple Silicon, non-clustered PSNR delta ≤ 0.5 dB, non-clustered Gaussian-count delta ≤ 10%, clustered PSNR delta ≤ 0.7 dB, and deterministic export/load round-trip

## Epic 18: LiteGS Mac MVP 训练路径

让 RustGS 在 Apple Silicon 上拥有一条显式的 LiteGS 兼容训练入口，先实现可运行、可验证的 non-clustered MVP，再继续追 cluster/sparse-grad 与 densify parity。

### Story 18.1: Public LiteGS Mac Training Profile and CLI Surface

Status: done

As a RustGS user,
I want a new LiteGS-compatible training profile and nested config surface,
So that I can opt into parity work without silently changing existing LegacyMetal behavior.

**Acceptance Criteria:**

**Given** the RustGS public training API
**When** I inspect `TrainingConfig`
**Then** it includes `training_profile` plus nested `litegs` config rather than scattering LiteGS-only knobs across the top level

**Given** the `rustgs train` CLI
**When** I pass `--training-profile` and `--litegs-*` options
**Then** the values parse and map into the nested config surface

**Given** I do not opt into the new profile
**When** I train with default config
**Then** the existing LegacyMetal path remains the default behavior

### Story 18.2: LiteGS-Compatible Initialization Defaults

Status: done

As a LiteGS parity developer,
I want initialization to match LiteGS defaults,
So that RustGS starts from comparable xyz/scale/opacity/SH state on Mac.

**Acceptance Criteria:**

**Given** COLMAP sparse points are available
**When** LiteGsMacV1 initialization runs
**Then** it prefers sparse-point initialization over frame-sampling fallback

**Given** a point-initialized Gaussian
**When** scale and opacity are initialized
**Then** scale uses distance-based log scale and opacity uses inverse-sigmoid(0.1)

### Story 18.3: LiteGS Activation and SH-Based Rendering Inputs

Status: in_progress

As a LiteGS parity developer,
I want Metal-side activation to consume LiteGS-style trainable tensors,
So that rendering derives RGB from SH instead of storing trained RGB directly.

**Acceptance Criteria:**

**Given** LiteGsMacV1 trainable Gaussians
**When** a render step begins
**Then** rotations are normalized, scales are exponentiated, opacities are sigmoided, and per-view color is derived from `sh_0/sh_rest`

### Story 18.4: LiteGS Loss Semantics for Mac V1

Status: todo

As a LiteGS parity developer,
I want Mac V1 to use LiteGS-style objective terms,
So that optimization pressure matches LiteGS more closely.

**Acceptance Criteria:**

**Given** LiteGsMacV1 default training
**When** loss is computed
**Then** the primary objective is L1+SSIM, scale regularization is optional, transmittance penalty is optional, and depth is disabled by default

### Story 18.5: LiteGS Parameter Groups and XYZ LR Decay

Status: todo

As a LiteGS parity developer,
I want optimizer groups to match LiteGS naming and learning-rate behavior,
So that parameter updates follow the same coarse schedule as the reference.

**Acceptance Criteria:**

**Given** LiteGsMacV1 optimizer setup
**When** parameter groups are built
**Then** groups exist for `xyz`, `sh_0`, `sh_rest`, `opacity`, `scale`, and `rot`

**Given** learning-rate scheduling runs
**When** the schedule advances
**Then** only xyz learning rate decays exponentially while other groups retain their initial learning rates

### Story 18.6: Rotation-Learning Guardrail

Status: todo

As a RustGS maintainer,
I want rotation learning disabled unless the backward path is truly correct,
So that the LiteGS profile never fakes rotation updates.

**Acceptance Criteria:**

**Given** rotation backward parity is incomplete
**When** LiteGsMacV1 trains
**Then** rotation learning remains explicitly disabled or inert rather than pretending to update correctly

## Epic 19: LiteGS Cluster 与 Sparse-Gradient 路径

让 RustGS 的 Metal runtime 具备 LiteGS clustered training 所需的数据表示、可见性压缩和稀疏优化语义。

### Story 19.1: Cluster Representation and AABB Frustum Culling

Status: todo

As a LiteGS parity developer,
I want clustered Gaussian representation and cluster-level frustum culling,
So that Apple Silicon can follow LiteGS visibility behavior without depending on RustGS chunk orchestration.

### Story 19.2: Visible-Chunk and Visible-Primitive Compaction

Status: todo

As a LiteGS parity developer,
I want compacted visibility bookkeeping,
So that sparse-grad mode can update only visible data.

### Story 19.3: Morton-Order Spatial Refine

Status: todo

As a LiteGS parity developer,
I want Morton-order reordering and cluster-bound refresh scheduling,
So that clustered topology and cache locality stay aligned with LiteGS expectations.

### Story 19.4: Sparse-Adam Parity on Metal

Status: todo

As a LiteGS parity developer,
I want sparse-Adam semantics equivalent to LiteGS for clustered and non-clustered paths,
So that visibility-gated updates match the reference optimizer.

### Story 19.5: Clustered Apple Silicon Parity Validation

Status: todo

As a project maintainer,
I want clustered parity runs validated with harness thresholds,
So that clustered PSNR and topology behavior stay within agreed tolerances on Apple Silicon.

## Epic 20: LiteGS Densify/Prune/Opacity Reset 对齐

让 RustGS 在拓扑编辑时收集与 LiteGS/TamingGS 对应的统计信息，并在 append/prune/reorder/recluster 过程中正确维护优化器状态。

### Story 20.1: Statistics Helper Equivalent

Status: todo

As a LiteGS parity developer,
I want Rust-side statistic accumulation equivalent to LiteGS,
So that densify and prune logic can consume the same high-level signals.

### Story 20.2: Official Density Controller Behavior

Status: todo

As a LiteGS parity developer,
I want clone/split/prune/reset behavior matching the official controller,
So that non-clustered topology work behaves like LiteGS before TamingGS extensions.

### Story 20.3: TamingGS Target Primitive and Weighted Prune Behavior

Status: todo

As a LiteGS parity developer,
I want TamingGS-specific target scheduling and weighted pruning,
So that RustGS can follow the full LiteGS densification strategy rather than only the official subset.

### Story 20.4: Optimizer-State Mutation Parity

Status: todo

As a LiteGS parity developer,
I want optimizer state to survive append/replace/prune/reorder/recluster edits,
So that topology work remains numerically stable across long training runs.

### Story 20.5: Mac-Safe Topology Guardrails

Status: todo

As a RustGS maintainer,
I want deterministic and explicitly timed topology mutations on Mac,
So that parity work remains debuggable on 16 GB Apple Silicon hardware.

## Epic 21: 运行时对齐、导出恢复与默认切换

让 LiteGS 兼容训练路径具备可持续运维能力，包括评估、checkpoint、PLY 往返、Mac 操作文档，以及验收通过后的默认 profile 升级。

### Story 21.1: LiteGS-Style Evaluation and Progress Reporting

Status: todo

As a RustGS operator,
I want stable PSNR evaluation and progress reporting from LiteGsMacV1,
So that fixture runs produce comparable outputs to the LiteGS reference.

### Story 21.2: Checkpoint Save/Resume Parity

Status: todo

As a LiteGS parity developer,
I want checkpoint save and resume to preserve SH tensors and optimizer state,
So that long Mac runs can be resumed without losing LiteGS-compatible state.

### Story 21.3: PLY Export/Import Parity

Status: todo

As a RustGS maintainer,
I want LiteGS-compatible PLY export/import to round-trip without data loss,
So that RustGS IO remains stable as the trainable parameter set expands.

### Story 21.4: Mac Operator Workflow Documentation

Status: todo

As a Mac operator,
I want one document that explains supported hardware, memory envelope, known gaps, and recommended flags,
So that I can run LiteGsMacV1 predictably on Apple Silicon.

### Story 21.5: Default Promotion Gate

Status: todo

As a project maintainer,
I want LiteGsMacV1 promoted only after the fixture suite passes the Epic 17 thresholds,
So that the default switch is backed by measured parity rather than optimism.

### Story 21.6: Simplified Training Pipeline Retirement

Status: todo

As a RustGS maintainer,
I want `training_pipeline` removed from production-path ownership,
So that the intended algorithm is defined by the LiteGS-compatible Metal path instead of the simplified reference implementation.
