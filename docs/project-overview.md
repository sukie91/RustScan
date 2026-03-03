# RustScan Project Overview

**Version:** 0.1.0
**Language:** Rust (Edition 2021)
**License:** MIT
**Last Updated:** 2026-03-01

## Project Summary

RustScan is a **pure Rust implementation of 3D scanning algorithms**, comprising two main libraries designed for high-performance 3D reconstruction from video input.

### Key Components

| Component | Description | Status |
|-----------|-------------|--------|
| **RustMesh** | SIMD-accelerated mesh processing library (Rust port of OpenMesh) | ~60% complete |
| **RustSLAM** | Visual SLAM library with 3D Gaussian Splatting support | ~85% complete |

### Primary Use Case

End-to-end 3D scanning pipeline:
```
iPhone Video (MP4/MOV/HEVC)
    → SLAM Processing (Sparse Reconstruction)
    → 3DGS Training (Dense Reconstruction)
    → Mesh Extraction (TSDF + Marching Cubes)
    → Export (OBJ/PLY + Metadata JSON)
```

## Project Statistics

| Metric | Value |
|--------|-------|
| Total Rust Files | 187 |
| Total Lines of Code | ~27,000 |
| RustMesh Source Files | 25 |
| RustSLAM Source Files | 83 |
| Examples | 32 (27 RustMesh + 5 RustSLAM) |
| Test Coverage | 287+ passing tests |

## Technology Stack

### Core Dependencies

| Category | Libraries |
|----------|-----------|
| Math/Geometry | `glam` (SIMD), `nalgebra` |
| GPU Acceleration | `candle-core`, `candle-metal` (Apple Metal/MPS) |
| Parallelism | `rayon`, `crossbeam-channel` |
| Serialization | `serde`, `serde_json`, `serde_yaml`, `toml` |
| CLI | `clap` (derive macros) |
| Video I/O | `ffmpeg-next` (VideoToolbox HW accel) |
| Optimization | `apex-solver` (Bundle Adjustment) |

### Optional Dependencies

| Feature | Dependencies |
|---------|--------------|
| Computer Vision | `opencv` (optional) |
| Deep Learning | `tch` (LibTorch bindings, optional) |
| Image Processing | `image` (optional) |

## Key Features

### RustMesh

- **Half-Edge Data Structure**: Efficient mesh representation with O(1) adjacency queries
- **SoA Memory Layout**: Structure of Arrays for cache-friendly SIMD operations
- **Mesh I/O**: Full support for OFF, OBJ, PLY, STL formats
- **Decimation**: Quadric-based mesh simplification with error metrics
- **Subdivision**: Loop, Catmull-Clark, and √3 subdivision algorithms
- **Mesh Repair**: Hole filling, smoothing, and topology repair

### RustSLAM

- **Visual Odometry**: Feature-based camera tracking (ORB/Harris/FAST)
- **Bundle Adjustment**: Sparse optimization of poses and landmarks
- **Loop Closing**: Bag-of-Words place recognition + pose graph optimization
- **3D Gaussian Splatting**: Real-time differentiable rendering
- **GPU Training**: Apple Metal acceleration via Candle
- **Mesh Extraction**: TSDF volume fusion + Marching Cubes

## Project Maturity

| Component | Maturity | Notes |
|-----------|----------|-------|
| CLI Infrastructure | ✅ Production-ready | Complete with config, logging, JSON output |
| Video Input | ✅ Production-ready | HW-accelerated decoding, LRU cache |
| SLAM Pipeline | ⚠️ Functional with caveats | Core algorithms work, some quality issues |
| 3DGS Training | ✅ Production-ready | GPU acceleration, analytical backward pass |
| Mesh Extraction | ✅ Production-ready | TSDF + Marching Cubes, OBJ/PLY export |
| RustMesh Core | ✅ Production-ready | Half-edge, I/O, circulators |
| RustMesh Tools | ⚠️ Basic implementation | Decimation/subdivision functional but limited |

## Architecture Highlights

1. **Monorepo Structure**: Two independent crates with clear separation
2. **SIMD Optimization**: glam provides SIMD-accelerated Vec3/Mat4 operations
3. **GPU Acceleration**: Candle + Metal for differentiable rendering on Apple Silicon
4. **Hardware Video Decoding**: ffmpeg-next with VideoToolbox support
5. **Pipeline Checkpoints**: Cross-stage state persistence for long-running jobs

## Build & Test

```bash
# Build both libraries
cd RustMesh && cargo build --release
cd RustSLAM && cargo build --release

# Run tests
cd RustSLAM && cargo test --lib  # 287 tests
cd RustMesh && cargo test

# Run CLI
cd RustSLAM && cargo run --release -- --input video.mp4 --output ./results
```

## Quick Start

```rust
// RustMesh example
use rustmesh::{RustMesh, Vec3};

let mut mesh = RustMesh::new();
let v0 = mesh.add_vertex(Vec3::new(0.0, 0.0, 0.0));
let v1 = mesh.add_vertex(Vec3::new(1.0, 0.0, 0.0));
let v2 = mesh.add_vertex(Vec3::new(0.0, 1.0, 0.0));
mesh.add_face(&[v0, v1, v2]);

// RustSLAM example
use rustslam::{SE3, Frame, VisualOdometry};

let vo = VisualOdometry::new();
let pose = SE3::identity();
```

## Target Platform

- **Primary**: macOS (Apple Silicon)
- **GPU**: Apple Metal Performance Shaders (MPS)
- **Video**: VideoToolbox hardware decoding

## Documentation Structure

```
docs/
├── index.md              # Master index (this overview)
├── project-overview.md   # Project summary
├── ARCHITECTURE.md       # System architecture
├── API.md               # API reference
├── DEVELOPMENT.md       # Development guide
├── RustMesh-README.md   # RustMesh-specific docs
├── RustSLAM-README.md   # RustSLAM-specific docs
├── RustSLAM-DESIGN.md   # RustSLAM design decisions
└── RustSLAM-ToDo.md     # Development roadmap
```

## References

- [OpenMesh Documentation](https://www.graphics.rwth-aachen.de/software/openmesh/)
- [ORB-SLAM Paper](https://arxiv.org/abs/1502.00956)
- [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [PGSR Project](https://github.com/pgsr-project)