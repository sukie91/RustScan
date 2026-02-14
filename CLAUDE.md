# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RustScan is a pure Rust implementation of 3D scanning algorithms, comprising two main libraries:

- **RustMesh**: A mesh processing library (Rust port of OpenMesh)
- **RustSLAM**: A Visual SLAM library with 3D Gaussian Splatting support

## Common Commands

### Build
```bash
# Build RustMesh
cd RustMesh && cargo build

# Build RustSLAM
cd RustSLAM && cargo build --release
```

### Test
```bash
# Test RustMesh
cd RustMesh && cargo test

# Test RustSLAM
cd RustSLAM && cargo test
```

### Run Examples
```bash
# Run VO example
cd RustSLAM && cargo run --example run_vo

# Run RustMesh demo
cd RustMesh && cargo run --example smart_handles_demo
```

### Benchmark
```bash
cd RustMesh && cargo bench
```

## Architecture

### RustMesh (`RustMesh/`)

Mesh data structure library implementing half-edge data structure with SoA (Structure of Arrays) memory layout.

```
RustMesh/src/
├── Core/               # Core data structures
│   ├── handles.rs      # Handle types (VertexHandle, EdgeHandle, HalfedgeHandle, FaceHandle)
│   ├── connectivity.rs # Half-edge connectivity relations
│   ├── soa_kernel.rs   # SoA storage layer
│   ├── kernel.rs       # ArrayKernel base
│   ├── attrib_kernel.rs # Attribute management
│   ├── geometry.rs     # Geometric operations
│   └── io.rs           # File I/O (OFF, OBJ, PLY, STL)
├── Tools/              # Mesh algorithms
│   ├── decimation.rs   # Quadric-based mesh simplification
│   ├── subdivision.rs  # Loop, Catmull-Clark, Sqrt3 subdivision
│   ├── smoother.rs     # Laplace, Tangential smoothing
│   ├── hole_filling.rs # Hole filling
│   └── mesh_repair.rs # Mesh repair utilities
└── Utils/
    ├── circulators.rs  # Vertex/Edge/Face circulators
    └── quadric.rs      # Quadric error computation
```

### RustSLAM (`RustSLAM/`)

Visual SLAM library with sparse feature-based VO and dense 3D Gaussian Splatting reconstruction.

```
RustSLAM/src/
├── core/              # Core data structures
│   ├── frame.rs       # Frame
│   ├── keyframe.rs   # KeyFrame
│   ├── map_point.rs  # MapPoint
│   ├── map.rs        # Map management
│   ├── camera.rs     # Camera model
│   └── pose.rs       # SE3 Pose
├── features/          # Feature extraction
│   ├── orb.rs        # ORB extractor
│   ├── pure_rust.rs  # Harris/FAST corner detection
│   ├── matcher.rs    # Feature matching
│   └── knn_matcher.rs
├── tracker/           # Visual Odometry
│   └── vo.rs         # Main VO pipeline
├── optimizer/        # Bundle Adjustment
│   └── ba.rs
├── loop_closing/     # Loop Detection
│   ├── vocabulary.rs # BoW Vocabulary
│   ├── detector.rs   # Loop Detector
│   └── relocalization.rs
└── fusion/           # 3D Gaussian Splatting
    ├── gaussian.rs   # Gaussian data structures
    ├── renderer.rs   # Forward renderer
    ├── diff_renderer.rs    # Differentiable renderer (CPU)
    ├── diff_splat.rs       # GPU differentiable splatting
    ├── autodiff.rs         # True autodiff with backward
    ├── tiled_renderer.rs   # Tiled rasterization + densify + prune
    ├── training_pipeline.rs # Training with SSIM loss
    ├── complete_trainer.rs  # Complete trainer with LR scheduler
    ├── autodiff_trainer.rs  # GPU trainer
    ├── tracker.rs     # Gaussian tracking (ICP)
    ├── mapper.rs      # Incremental Gaussian mapping
    ├── slam_integrator.rs  # Sparse + Dense SLAM integration
    ├── tsdf_volume.rs     # TSDF volume (NEW!)
    ├── marching_cubes.rs  # Marching Cubes (NEW!)
    └── mesh_extractor.rs  # High-level mesh extraction (NEW!)
```

## Module Status

### RustSLAM Progress (~85%)

| Feature | Status |
|---------|--------|
| SE3 Pose | ✅ |
| ORB/Harris/FAST | ✅ |
| Feature Matching | ✅ |
| Visual Odometry | ✅ |
| Bundle Adjustment | ✅ |
| Loop Closing | ✅ |
| 3DGS Data Structures | ✅ |
| Tiled Rasterization | ✅ |
| Depth Sorting | ✅ |
| Alpha Blending | ✅ |
| Gaussian Tracking (ICP) | ✅ |
| Densification | ✅ |
| Pruning | ✅ |
| Candle + Metal GPU | ✅ |
| True Backward Propagation | ✅ |
| SLAM Integration | ✅ |
| **3DGS → Mesh Extraction** | ✅ NEW |
| IMU Integration | ⏳ |
| Multi-map SLAM | ⏳ |

### RustMesh Progress (~50-60%)

| Feature | Status |
|---------|--------|
| Handle System | ✅ |
| Half-edge + SoA | ✅ |
| OFF/OBJ/PLY/STL IO | ✅ |
| Smart Handles | ✅ |
| Decimation | ⚠️ Basic |
| Subdivision | ⚠️ Loop/CC/√3 |
| Hole Filling | ✅ |
| AttribKernel Integration | ⏳ |
| **3DGS → Mesh** | ✅ (via RustSLAM) |

### 3DGS → Mesh Extraction (IMPLEMENTED)

New files in `RustSLAM/src/fusion/`:

1. **tsdf_volume.rs** - Pure Rust TSDF volume implementation
   - `TsdfVolume` - volumetric fusion
   - `TsdfConfig` - configurable voxel size, truncation distance
   - Supports depth map integration from Gaussian rendering

2. **marching_cubes.rs** - Marching Cubes algorithm
   - Full 256-case lookup table
   - Mesh vertex and triangle generation
   - Color interpolation

3. **mesh_extractor.rs** - High-level API
   - `MeshExtractor` - easy-to-use interface
   - `MeshExtractionConfig` - post-processing options
   - Cluster filtering (remove floaters)
   - Normal smoothing

**Usage**:
```rust
use rustslam::fusion::{MeshExtractor, MeshExtractionConfig};
use glam::Vec3;
use glam::Mat4;

// Create extractor
let mut extractor = MeshExtractor::centered(Vec3::ZERO, 2.0, 0.01);

// Integrate depth frames from Gaussian rendering
extractor.integrate_from_gaussians(
    |idx| depth[idx],
    width, height,
    [fx, fy, cx, cy],
    &camera_pose,
);

// Extract mesh with post-processing
let mesh = extractor.extract_with_postprocessing();
```

## Key Dependencies

- **glam**: SIMD-accelerated math library (Vec3, Mat4, Quat)
- **nalgebra**: Linear algebra (for BA solver)
- **rayon**: Data parallelism
- **opencv** (optional): Image processing
- **candle-metal**: GPU acceleration via Apple MPS
- **apex-solver**: Bundle adjustment

## Pipeline

```
Camera Input → RustSLAM (VO + Mapping) → 3DGS Fusion → Mesh Extraction → RustMesh Post-processing → Export
                              ↓                              ↓
                     RustSLAM/fusion/              tsdf_volume.rs
                                          + marching_cubes.rs
                                          + mesh_extractor.rs
```

**Complete**: 3DGS → Mesh extraction is now implemented (Pure Rust, PGSR-style).
