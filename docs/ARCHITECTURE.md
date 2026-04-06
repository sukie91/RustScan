# RustScan Architecture

## Overview

RustScan is a Rust workspace for 3D reconstruction. The main algorithm crates are RustMesh and RustSLAM, with additional crates for Gaussian training, visualization, and shared types.

## Project Structure

```text
RustScan/
├── RustMesh/          # Mesh processing library
├── RustSLAM/          # Visual SLAM library
├── RustGS/            # Gaussian splatting crate
├── RustViewer/        # Visualization crate
├── rustscan-types/    # Shared types
├── docs/              # Documentation
├── test_data/         # Test datasets
└── README.md          # Project overview
```

## Core Components

### 1. RustMesh - Mesh Processing Library

**Purpose**: A Rust port of OpenMesh providing versatile geometric data structures and mesh processing algorithms.

**Architecture**:
- **Core Layer**: Half-edge data structure with SoA (Structure of Arrays) memory layout
- **Tools Layer**: Mesh algorithms (decimation, subdivision, smoothing, hole filling)
- **Utils Layer**: Helper utilities (circulators, quadric error metrics, performance tools)

**Key Modules**:
- `Core/handles.rs` - Handle types (VertexHandle, EdgeHandle, HalfedgeHandle, FaceHandle)
- `Core/connectivity.rs` - Half-edge connectivity relations
- `Core/soa_kernel.rs` - SoA storage layer
- `Core/geometry.rs` - Geometric operations
- `Core/io/` - File I/O (OFF, OBJ, PLY, STL formats)
- `Tools/decimation.rs` - Quadric-based mesh simplification
- `Tools/subdivision.rs` - Loop, Catmull-Clark, Sqrt3 subdivision
- `Tools/smoother.rs` - Laplace, Tangential smoothing
- `Tools/hole_filling.rs` - Hole filling algorithms
- `Utils/circulators.rs` - Vertex/Edge/Face circulators
- `Utils/quadric.rs` - Quadric error computation

**Dependencies**:
- `glam` - SIMD-accelerated math library
- `nalgebra` - Linear algebra
- `serde` - Serialization
- `criterion` - Benchmarking

### 2. RustSLAM - Visual SLAM Library

**Purpose**: A pure Rust Visual SLAM library with sparse feature-based VO and dense 3D Gaussian Splatting reconstruction.

**Architecture**:
- **Core Layer**: Fundamental data structures (Frame, KeyFrame, MapPoint, Map, Camera, Pose)
- **Features Layer**: Feature extraction and matching (ORB, Harris, FAST)
- **Tracker Layer**: Visual Odometry pipeline
- **Optimizer Layer**: Bundle Adjustment
- **Loop Closing Layer**: Loop detection and relocalization
- **Fusion Layer**: mesh extraction and related fusion support
- **Pipeline Layer**: Real-time SLAM pipeline
- **I/O Layer**: Dataset loading and video processing

**Key Modules**:

#### Core (`core/`)
- `frame.rs` - Frame data structure
- `keyframe.rs` - KeyFrame management
- `map_point.rs` - 3D point representation
- `map.rs` - Map management
- `camera.rs` - Camera model (pinhole)
- `pose.rs` - SE3 Pose representation

#### Features (`features/`)
- `orb.rs` - ORB feature extractor
- `pure_rust.rs` - Pure Rust Harris/FAST corner detection
- `knn_matcher.rs` - KNN-based feature matching

#### Tracker (`tracker/`)
- `vo.rs` - Visual Odometry pipeline
- `solver.rs` - PnP solver

#### Optimizer (`optimizer/`)
- `ba.rs` - Bundle Adjustment using apex-solver

#### Loop Closing (`loop_closing/`)
- `vocabulary.rs` - BoW vocabulary
- `detector.rs` - Loop detector
- `relocalization.rs` - Relocalization module

#### Fusion (`fusion/`)
- `gaussian.rs` - Gaussian data structures
- `renderer.rs` - Rendering support
- `tiled_renderer.rs` - Tiled rasterization
- `tracker.rs` - Gaussian tracking support
- `mapper.rs` - Incremental mapping support
- `slam_integrator.rs` - sparse/dense integration
- `tsdf_volume.rs` - TSDF volume fusion
- `marching_cubes.rs` - Marching Cubes mesh extraction
- `mesh_extractor.rs` - high-level mesh extraction API
- `mesh_io.rs`, `mesh_metadata.rs`, `scene_io.rs` - mesh and scene IO helpers

#### Pipeline (`pipeline/`)
- `realtime.rs` - Real-time SLAM pipeline

#### I/O (`io/`)
- `dataset.rs` - Dataset loader
- `video_loader.rs` - Video processing

**Dependencies**:
- `glam` - SIMD-accelerated math
- `nalgebra` - Linear algebra
- `apex-solver` - Bundle adjustment solver
- `rayon` - Data parallelism
- `crossbeam-channel` - Thread communication
- `candle-core`, `candle-metal` - GPU acceleration (Apple MPS)
- `kiddo` - KD-Tree for KNN matching
- `opencv` (optional) - Image processing
- `tch` (optional) - Deep learning

## Data Flow

```
Camera Input → Feature Extraction → Feature Matching → Visual Odometry
                                                              ↓
                                                         Tracking
                                                              ↓
                                                    Local Mapping
                                                              ↓
                                                      Loop Closing
                                                              ↓
                                                   3DGS Fusion
                                                              ↓
                                                   TSDF Volume
                                                              ↓
                                                  Marching Cubes
                                                              ↓
                                                   Mesh (RustMesh)
                                                              ↓
                                                  Post-processing
                                                              ↓
                                                      Export
```

## Integration Points

### RustSLAM → RustMesh

The fusion layer in RustSLAM generates mesh data that can be processed by RustMesh:

1. **3DGS → TSDF**: Gaussian splatting results are converted to TSDF volume
2. **TSDF → Mesh**: Marching Cubes extracts triangle mesh
3. **Mesh → RustMesh**: Mesh data can be imported into RustMesh for:
   - Decimation (mesh simplification)
   - Smoothing
   - Hole filling
   - Subdivision
   - Export to various formats (OFF, OBJ, PLY, STL)

## Memory Layout

### RustMesh SoA (Structure of Arrays)

RustMesh uses SoA memory layout for cache efficiency:

```rust
// Instead of Array of Structures (AoS):
struct Vertex { position: Vec3, normal: Vec3, ... }
vertices: Vec<Vertex>

// Uses Structure of Arrays (SoA):
positions: Vec<Vec3>
normals: Vec<Vec3>
```

Benefits:
- Better cache locality for operations on single attributes
- SIMD-friendly memory access patterns
- Reduced memory bandwidth

### RustSLAM Frame Management

- Frames are managed in a sliding window
- KeyFrames are selected based on motion and feature distribution
- MapPoints maintain references to observing KeyFrames

## Parallelism

### RustMesh
- Uses `rayon` for parallel mesh operations
- Parallel feature: Optional `parallel` feature flag

### RustSLAM
- Multi-threaded pipeline:
  - Tracking thread
  - Local mapping thread
  - Loop closing thread
  - Gaussian fusion thread
- Uses `rayon` for parallel feature extraction
- Uses `crossbeam-channel` for thread communication

## GPU Acceleration

RustSLAM uses Apple Metal Performance Shaders (MPS) via `candle-metal`:
- Gaussian splatting rasterization
- Differentiable rendering
- Backward propagation for training
- Densification and pruning operations

## Testing Strategy

### RustMesh
- Unit tests in `#[cfg(test)]` modules
- Integration tests in `examples/`
- Benchmarks using `criterion`

### RustSLAM
- Unit tests in `#[cfg(test)]` modules
- Integration-style coverage in `examples/`
- Current worktree note: the RustSLAM library suite is not fully green, so architectural presence should not be read as release readiness

## Build Configuration

### Release Profile (Aggressive Optimization)
```toml
[profile.release]
lto = true              # Link-time optimization
codegen-units = 1       # Single codegen unit for better optimization
opt-level = 3           # Maximum optimization
strip = true            # Strip symbols
panic = "abort"         # Abort on panic (smaller binary)
```

### Development Profile (Fast Compilation)
```toml
[profile.dev]
opt-level = 1           # Basic optimization
debug = "line-tables-only"  # Minimal debug info
```

## Future Directions

### Planned Features
- IMU integration for RustSLAM
- Multi-map SLAM
- Enhanced RustMesh-RustSLAM integration
- Additional mesh processing algorithms

### Performance Optimization
- Further GPU acceleration
- SIMD optimization for critical paths
- Memory pool allocators

## References

- OpenMesh: https://www.graphics.rwth-aachen.de/software/openmesh/
- ORB-SLAM: https://github.com/raulmur/ORB_SLAM2
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
