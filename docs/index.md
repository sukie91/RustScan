# RustScan Documentation Index

**Version:** 0.1.0 | **Updated:** 2026-04-04 | **Status:** Active Development

---

## Quick Navigation

| For New Users | For Contributors | For Researchers |
|---------------|------------------|-----------------|
| [Project Overview](project-overview.md) | [Development Guide](DEVELOPMENT.md) | [Architecture](ARCHITECTURE.md) |
| [Getting Started](#getting-started) | [API Reference](API.md) | [RustSLAM Design](RustSLAM-DESIGN.md) |

---

## Project Overview

RustScan is a **pure Rust implementation of 3D scanning algorithms** for generating 3D meshes from video input.

### Key Components

| Library | Description | Status |
|---------|-------------|--------|
| [**RustMesh**](RustMesh-README.md) | SIMD-accelerated mesh processing (OpenMesh port) | ~60% complete |
| [**RustSLAM**](RustSLAM-README.md) | Visual SLAM with 3D Gaussian Splatting | ~85% complete |

### Pipeline Overview

```
iPhone Video (MP4/MOV/HEVC)
    ↓ Hardware Decoding (VideoToolbox)
Frame Extraction
    ↓ Feature Extraction (ORB/Harris/FAST)
Visual Odometry
    ↓ Bundle Adjustment
Sparse Map + Camera Poses
    ↓ 3D Gaussian Splatting Training
Dense 3DGS Scene
    ↓ TSDF Volume + Marching Cubes
Mesh Export (OBJ/PLY + JSON Metadata)
```

---

## Documentation Structure

### Core Documentation

| Document | Description |
|----------|-------------|
| [Project Overview](project-overview.md) | Project summary, statistics, technology stack |
| [Source Tree Analysis](source-tree-analysis.md) | Complete source code structure and organization |
| [Architecture](ARCHITECTURE.md) | System architecture and component overview |
| [API Reference](API.md) | Complete API documentation for both libraries |
| [Development Guide](DEVELOPMENT.md) | Building, testing, and contributing |

### Component Documentation

| Document | Description |
|----------|-------------|
| [RustMesh README](RustMesh-README.md) | RustMesh-specific documentation |
| [RustMesh OpenMesh Progress 2026-04-04](RustMesh-OpenMesh-Progress-2026-04-04.md) | RustMesh 对齐 OpenMesh 的最新进展、benchmark 与 trace 结论 |
| [RustSLAM README](RustSLAM-README.md) | RustSLAM-specific documentation |
| [RustSLAM Design](RustSLAM-DESIGN.md) | Detailed design decisions |
| [RustSLAM ToDo](RustSLAM-ToDo.md) | Development roadmap |
| [RustGS LiteGS Parity Progress 2026-04-05](RustGS-LiteGS-Parity-Progress-2026-04-05.md) | RustGS 相对 LiteGS 的当前对齐进展、验证结果与后续任务 |

### Project Planning

| Document | Description |
|----------|-------------|
| [ROADMAP](../ROADMAP.md) | Project roadmap and future plans |
| [CLAUDE](../CLAUDE.md) | Claude Code integration guide |

---

## Getting Started

### Prerequisites

- **Rust**: Edition 2021 (install via [rustup](https://rustup.rs/))
- **Platform**: macOS (Apple Silicon recommended)
- **FFmpeg**: For video decoding

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd RustScan

# Build RustMesh
cd RustMesh && cargo build --release

# Build RustSLAM
cd ../RustSLAM && cargo build --release
```

### Quick Test

```bash
# Run RustSLAM tests (287 tests)
cd RustSLAM && cargo test --lib

# Run CLI
cargo run --release -- --input video.mp4 --output ./results

# Run example
cargo run --release --example e2e_slam_to_mesh
```

### Usage Examples

```rust
// RustMesh: Create and manipulate meshes
use rustmesh::{RustMesh, Vec3};

let mut mesh = RustMesh::new();
let v0 = mesh.add_vertex(Vec3::new(0.0, 0.0, 0.0));
let v1 = mesh.add_vertex(Vec3::new(1.0, 0.0, 0.0));
let v2 = mesh.add_vertex(Vec3::new(0.0, 1.0, 0.0));
mesh.add_face(&[v0, v1, v2]);

// RustSLAM: Use SE3 poses
use rustslam::{SE3, VisualOdometry};
let pose = SE3::identity();
```

---

## Technology Stack

### Core Dependencies

| Category | Library | Purpose |
|----------|---------|---------|
| Math | `glam` | SIMD-accelerated 3D math |
| GPU | `candle-metal` | Apple Metal acceleration |
| Video | `ffmpeg-next` | Hardware video decoding |
| Parallelism | `rayon` | Data parallelism |
| CLI | `clap` | Command-line interface |
| BA | `apex-solver` | Bundle adjustment |

### Build System

- **Build Tool**: Cargo
- **Test Framework**: `cargo test`
- **Benchmark**: Criterion
- **Profile**: Release with LTO, opt-level 3

---

## Project Statistics

| Metric | Value |
|--------|-------|
| Total Rust Files | 187 |
| Total Lines of Code | ~27,000 |
| Examples | 32 |
| Tests | 287+ (all passing) |
| Documentation Files | 12 |

---

## Key Concepts

### RustMesh Concepts

- **Half-Edge Data Structure**: Efficient mesh representation with O(1) adjacency queries
- **SoA Layout**: Structure of Arrays for SIMD cache efficiency
- **Smart Handles**: Type-safe mesh element references
- **Circulators**: Efficient mesh traversal patterns

### RustSLAM Concepts

- **SE(3) Pose**: Special Euclidean group for 3D transformations
- **Visual Odometry**: Camera motion estimation from image sequences
- **Bundle Adjustment**: Joint optimization of poses and 3D points
- **3D Gaussian Splatting**: Differentiable rendering for dense reconstruction
- **TSDF Fusion**: Truncated Signed Distance Field for volumetric integration
- **Marching Cubes**: Iso-surface extraction from volumetric data

---

## Documentation by Role

### For New Users

1. Read [Project Overview](project-overview.md) for project summary
2. Follow [Getting Started](#getting-started) to build the project
3. Explore [API Reference](API.md) for code examples
4. Check [Architecture](ARCHITECTURE.md) for system design

### For Contributors

1. Read [Development Guide](DEVELOPMENT.md) for workflow and guidelines
2. Check [ROADMAP](../ROADMAP.md) for planned features
3. Review [RustSLAM ToDo](RustSLAM-ToDo.md) for specific tasks
4. Follow code style guidelines in [DEVELOPMENT.md](DEVELOPMENT.md)

### For Researchers

1. Read [Architecture](ARCHITECTURE.md) for system overview
2. Study [RustSLAM Design](RustSLAM-DESIGN.md) for algorithm details
3. Review [API Reference](API.md) for implementation details
4. Check [ROADMAP](../ROADMAP.md) for research directions

---

## External Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [OpenMesh Documentation](https://www.graphics.rwth-aachen.de/software/openmesh/)
- [ORB-SLAM Paper](https://arxiv.org/abs/1502.00956)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

---

## Support

- **Issues**: Report bugs via GitHub Issues
- **Contributing**: See [Development Guide](DEVELOPMENT.md)
- **Code Style**: Rust standard style (enforced by `cargo fmt`)
- **Testing**: All contributions must include tests

---

## License

MIT License - See LICENSE file for details.

---

**Last Updated:** 2026-03-01
**Documentation Version:** 2.0
**Project Status:** Active Development
