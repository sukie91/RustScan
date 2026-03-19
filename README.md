# RustScan

<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.75+-dea584?style=for-the-badge&logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge" alt="Status">
</p>

A complete 3D scanning and reconstruction technology stack implemented in pure Rust, featuring Visual SLAM, 3D Gaussian Splatting, mesh extraction, and an interactive 3D visualization GUI.

## Project Goals

Build a pure Rust implementation of 3D scanning and reconstruction technology, covering the complete pipeline from data acquisition to mesh processing and visualization.

```
Pipeline: Video Input → RustSLAM → RustGS → RustMesh → Export
                                        ↓
                                  RustViewer (3D Visualization)
```

---

## Core Modules

### 🟢 RustMesh (Mesh Processing)

**Core mesh representation and geometric processing library**

- Mesh data structures (Half-edge, SoA layout)
- IO format support (OBJ, OFF, PLY, STL, OM)
- Mesh algorithms
  - Subdivision (Loop, Catmull-Clark, Sqrt3)
  - Simplification (Decimation + Quadric error)
  - Smoothing (Laplace, Tangential)
  - Hole filling
  - Mesh repair
  - Dualization
  - Progressive mesh (VDPM)
- Smart Handle navigation system
- Attribute system
- **TSDF Volume Fusion + Marching Cubes mesh extraction**

**Progress: 100%** | [Details](./RustMesh/README.md)

---

### 🟢 RustSLAM (Visual SLAM)

**Pure Rust implementation of Visual SLAM library**

- Feature extraction (ORB, Harris, FAST with BRIEF descriptors)
- Feature matching (Hamming distance for binary descriptors, KNN matching)
- Visual Odometry (VO + DLT-PnP + RANSAC)
- Local mapping (Triangulation + Bundle Adjustment)
- Loop closing (BoW + Sim3 pose optimization)
- Relocalization support
- Pipeline checkpoint and recovery
- Video decoding with hardware acceleration (VideoToolbox on macOS)

**Tech Stack**:
- glam: SIMD math library
- rayon: Data parallelism
- crossbeam-channel: Thread communication
- ffmpeg-next: Video decoding
- thiserror: Error handling

**Progress: 100%** | [Details](./RustSLAM/README.md)

---

### 🟢 RustGS (3D Gaussian Splatting)

**Standalone 3DGS training library**

- Gaussian data structures (position, covariance, color, opacity)
- Differentiable rendering (CPU + GPU via Candle + Metal)
- Tiled rasterization with depth sorting
- Alpha blending (front-to-back compositing)
- Training pipeline with SSIM + L1 loss
- Densification and pruning
- PLY scene file export/import
- Training checkpoint and resume

**Tech Stack**:
- candle-core + candle-metal: GPU acceleration (Apple MPS)
- glam: 3D math

**Progress: 100%**

---

### 🟢 RustViewer (3D Visualization GUI)

**Interactive 3D visualization for SLAM results**

- Offline file loading:
  - `slam_checkpoint.json` (camera trajectory + map points)
  - `scene.ply` (Gaussian point cloud)
  - `mesh.obj/ply` (extracted mesh)
- 3D rendering with wgpu:
  - Camera trajectory polylines
  - Sparse point clouds
  - Gaussian point clouds
  - Mesh wireframe and solid faces
- egui control panel:
  - File selection dialog
  - Layer visibility toggles
  - Scene statistics display
- Arcball camera control (orbit, pan, zoom)
- Apple HIG-compliant UI design

**Tech Stack**:
- eframe 0.31 + egui 0.31: GUI framework
- wgpu: GPU rendering
- glam: 3D math
- bytemuck: Zero-copy GPU buffer uploads

**Progress: 100%**

---

### 🟢 rustscan-types

**Shared type definitions across crates**

- Common data types for inter-crate communication
- Feature-gated to minimize dependency footprint

**Progress: 100%**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RustScan Workspace                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  rustscan-types ──┬── RustSLAM ──→ RustGS ──→ RustMesh         │
│                   │       │           │           │              │
│                   │       └───────────┴───────────┘              │
│                   │                    │                         │
│                   └────────────→ RustViewer                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
RustScan/
├── rustscan-types/     # Shared types (100%)
├── RustMesh/           # Mesh processing library (100%)
│   ├── Core/           # Data structures (handles, connectivity, SoA)
│   ├── Tools/          # Algorithms (decimation, subdivision, smoothing)
│   └── Utils/          # Utilities (circulators, quadric)
│
├── RustSLAM/           # Visual SLAM (100%)
│   ├── core/           # Frame, KeyFrame, MapPoint, Camera, SE3
│   ├── features/       # ORB, Harris, FAST, matching
│   ├── tracker/        # Visual Odometry
│   ├── optimizer/      # Bundle Adjustment
│   ├── loop_closing/   # BoW vocabulary, loop detector
│   ├── io/             # Video decoder (ffmpeg + VideoToolbox)
│   └── cli/            # Pipeline orchestration
│
├── RustGS/             # 3DGS training (100%)
│   ├── core/           # Gaussian data structures
│   ├── render/         # Differentiable rendering
│   ├── train/          # Training pipeline
│   └── io/             # PLY import/export
│
└── RustViewer/         # 3D GUI (100%)
    ├── loader/         # Checkpoint, Gaussian, Mesh loaders
    ├── renderer/       # wgpu pipelines, scene graph
    └── ui/             # egui panels and controls
```

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Rust 2021 Edition |
| **Math** | glam 0.25 (SIMD accelerated) |
| **GPU** | wgpu, candle-core + candle-metal (Apple MPS) |
| **GUI** | eframe 0.31, egui 0.31 |
| **Video** | ffmpeg-next 8.0 (VideoToolbox HW accel) |
| **Parallelism** | rayon 1.8, crossbeam-channel 0.5 |
| **Optimization** | apex-solver 1.0 (Bundle Adjustment) |
| **CLI** | clap 4.5 |

---

## Quick Start

### Build All Crates

```bash
cargo build --release
```

### RustViewer (3D GUI)

```bash
cargo run -p rust-viewer
```

### RustSLAM

```bash
cd RustSLAM
cargo build --release
cargo run --example run_vo
cargo test
```

### RustMesh

```bash
cd RustMesh
cargo build
cargo test
cargo run --example smart_handles_demo
```

---

## Examples

Run the end-to-end sample pipeline on three short iPhone clips:

```bash
./run_examples.sh
```

Outputs are written to `output/examples` and compared against `test_data/expected`.

Environment overrides:
- `RUSTSCAN_PROFILE` default `release`
- `RUSTSCAN_MAX_FRAMES` default `12`
- `RUSTSCAN_FRAME_STRIDE` default `2`
- `RUSTSCAN_MESH_VOXEL_SIZE` default `0.05`
- `RUSTSCAN_PREFER_HW` default `false`

---

## Progress Overview

| Module | Completion | Status |
|--------|------------|--------|
| **RustSLAM** | 100% | ✅ Complete |
| **RustGS** | 100% | ✅ Complete |
| **RustMesh** | 100% | ✅ Complete |
| **RustViewer** | 100% | ✅ Complete |
| **rustscan-types** | 100% | ✅ Complete |

### All Features Implemented

- [x] CLI Infrastructure & Configuration
- [x] Video Input & Hardware Decoding
- [x] SLAM Processing Pipeline (Feature extraction, VO, BA, Loop closing)
- [x] 3DGS Training & Scene Generation
- [x] Mesh Extraction & Export (TSDF + Marching Cubes)
- [x] End-to-End Pipeline Integration
- [x] Cross-Cutting Infrastructure (Thread safety, Config validation)
- [x] RustViewer 3D Visualization GUI
- [x] RustGS Crate Extraction
- [x] RustMesh Integration
- [x] Pipeline Documentation

---

## Output Formats

| Type | Formats | Description |
|------|---------|-------------|
| **3DGS Scene** | `.ply` | Gaussian point cloud with metadata |
| **Mesh** | `.obj`, `.ply` | Triangle mesh with vertex colors |
| **Checkpoint** | `.json` | SLAM trajectory and map points |
| **Metadata** | `.json` | Processing metrics and configuration |

All outputs are compatible with Blender and Unity.

---

## References

- [OpenMesh](https://www.openmesh.org/) - C++ mesh processing library
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) - Visual SLAM
- [Open3D](http://www.open3d.org/) - 3D reconstruction library
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - SIGGRAPH 2023
- [SplaTAM](https://github.com/spla-tam/SplaTAM) - 3DGS SLAM (CVPR 2024)
- [RTG-SLAM](https://github.com/MisEty/RTG-SLAM) - Real-time 3DGS

---

## License

MIT License - see LICENSE file for details.

---

<p align="center">
Built with ❤️ in Rust
</p>