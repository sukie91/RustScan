---
stepsCompleted: [1, 2]
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

<!-- Repeat for each epic in epics_list (N = 1, 2, 3...) -->

## Epic {{N}}: {{epic_title_N}}

{{epic_goal_N}}

<!-- Repeat for each story (M = 1, 2, 3...) within epic N -->

### Story {{N}}.{{M}}: {{story_title_N_M}}

As a {{user_type}},
I want {{capability}},
So that {{value_benefit}}.

**Acceptance Criteria:**

<!-- for each AC on this story -->

**Given** {{precondition}}
**When** {{action}}
**Then** {{expected_outcome}}
**And** {{additional_criteria}}

<!-- End story repeat -->