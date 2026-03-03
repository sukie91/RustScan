# RustScan Source Tree Analysis

**Generated:** 2026-03-01
**Scan Level:** Exhaustive

## Root Directory Structure

```
RustScan/
├── RustMesh/              # Mesh processing library (25 source files)
│   ├── src/
│   ├── examples/          # 27 example programs
│   ├── benches/           # Criterion benchmarks
│   └── Cargo.toml
├── RustSLAM/              # Visual SLAM library (83 source files)
│   ├── src/
│   ├── examples/          # 5 example programs
│   └── Cargo.toml
├── docs/                  # Documentation
├── test_data/             # Test datasets
├── _bmad/                 # BMAD workflow system
├── _bmad-output/          # BMAD generated artifacts
└── Mirror/                # Reference implementations
```

---

## RustMesh Source Tree

### Overview

| Directory | Files | Purpose |
|-----------|-------|---------|
| `src/Core/` | 10 | Core data structures (half-edge, handles, kernel) |
| `src/Tools/` | 7 | Mesh algorithms (decimation, subdivision, repair) |
| `src/Utils/` | 6 | Utilities (circulators, quadric, status) |
| `examples/` | 27 | Demo programs and benchmarks |

### Core Module (`src/Core/`)

```
Core/
├── handles.rs          # Handle types (VertexHandle, EdgeHandle, HalfedgeHandle, FaceHandle)
├── items.rs            # Mesh element structs (Vertex, Halfedge, Edge, Face)
├── soa_kernel.rs       # SoA storage kernel (SIMD-friendly)
├── attrib_soa_kernel.rs # Attribute-aware kernel
├── connectivity.rs     # PolyConnectivity implementation (main mesh type)
├── geometry.rs         # Geometric operations (normals, bounding boxes)
└── io/                 # File I/O module
    ├── mod.rs          # I/O dispatcher
    ├── obj.rs          # Wavefront OBJ format
    ├── ply.rs          # Stanford PLY format (binary + ASCII)
    ├── off.rs          # Object File Format
    └── stl.rs          # Stereolithography format (binary + ASCII)
```

**Key Exports:**
- `RustMesh` - Main mesh type (half-edge connectivity + SoA storage)
- `VertexHandle`, `HalfedgeHandle`, `EdgeHandle`, `FaceHandle` - Type-safe handles
- `load_obj`, `load_ply`, `load_off`, `load_stl` - Import functions
- `save_obj`, `save_ply`, `save_off`, `save_stl` - Export functions

### Tools Module (`src/Tools/`)

```
Tools/
├── decimation.rs       # Quadric-based mesh simplification
├── subdivision.rs      # Loop, Catmull-Clark, √3 subdivision
├── smoother.rs         # Laplace and tangential smoothing
├── hole_filling.rs     # Hole detection and filling
├── mesh_repair.rs      # Topology repair utilities
├── dualizer.rs         # Dual mesh construction
└── vdpm.rs            # View-dependent progressive meshes
```

**Key Exports:**
- `Decimater`, `decimate_mesh`, `DecimationConfig` - Mesh simplification
- `loop_subdivide`, `catmull_clark_subdivide` - Subdivision
- `laplace_smooth`, `tangential_smooth` - Smoothing
- `fill_hole` - Hole filling

### Utils Module (`src/Utils/`)

```
Utils/
├── status.rs           # Status flags for mesh elements
├── circulators.rs      # Vertex/Edge/Face circulators
├── quadric.rs          # Quadric error metrics
├── smart_ranges.rs     # Range-based iteration
├── test_data.rs        # Test mesh generators
└── performance.rs      # Performance utilities
```

### Examples (`examples/`)

| Example | Purpose |
|---------|---------|
| `bench.rs` | Performance benchmarking |
| `quadric_mesh_test.rs` | Quadric decimation test |
| `test_circulator.rs` | Circulator validation |
| `test_smart.rs` | Smart handle demo |
| `test_smooth.rs` | Smoothing demo |
| `test_subdivision.rs` | Subdivision demo |
| `convert_obj_to_off.rs` | Format conversion |
| `e2e_export.rs` | End-to-end export test |

---

## RustSLAM Source Tree

### Overview

| Directory | Files | Purpose |
|-----------|-------|---------|
| `src/core/` | 9 | Core data structures (SE3, Frame, Map, Camera) |
| `src/features/` | 6 | Feature extraction and matching |
| `src/tracker/` | 3 | Visual Odometry pipeline |
| `src/optimizer/` | 2 | Bundle Adjustment |
| `src/loop_closing/` | 6 | Loop detection and closing |
| `src/fusion/` | 20 | 3D Gaussian Splatting + Mesh Extraction |
| `src/cli/` | 4 | CLI implementation |
| `src/config/` | 3 | Configuration management |
| `src/io/` | 4 | Video and dataset I/O |
| `src/pipeline/` | 3 | Checkpoint and realtime pipeline |
| `src/mapping/` | 2 | Local mapping |
| `src/depth/` | 3 | Depth estimation |
| `src/viewer/` | 1 | Visualization (optional) |

### Core Module (`src/core/`)

```
core/
├── mod.rs              # Module exports
├── pose.rs             # SE3 pose representation
├── frame.rs            # Frame (image + features)
├── keyframe.rs         # KeyFrame (selected frames for mapping)
├── map_point.rs        # MapPoint (3D landmark)
├── map.rs              # Map (collection of KeyFrames + MapPoints)
├── camera.rs           # Camera model (intrinsics + distortion)
└── keyframe_selector.rs # KeyFrame selection logic
```

**Key Exports:**
- `SE3` - SE(3) pose with rotation + translation
- `Frame`, `FrameFeatures` - Frame data
- `KeyFrame` - Keyframe for mapping
- `MapPoint`, `Map` - Sparse map
- `Camera` - Pinhole camera model

### Features Module (`src/features/`)

```
features/
├── mod.rs              # Module exports
├── base.rs             # Trait definitions (FeatureExtractor, FeatureMatcher)
├── orb.rs              # ORB feature extractor
├── pure_rust.rs        # Harris/FAST corner detectors
├── knn_matcher.rs      # KNN feature matcher (kiddo KD-Tree)
├── hamming_matcher.rs  # Hamming distance matcher (for binary descriptors)
└── utils.rs            # Feature utilities
```

**Key Exports:**
- `FeatureExtractor`, `FeatureMatcher` - Traits
- `KeyPoint`, `Descriptors`, `Match` - Data types
- `OrbExtractor` - ORB implementation
- `KnnMatcher`, `DistanceMetric` - KNN matching
- `HammingMatcher` - Binary descriptor matching
- `HarrisDetector`, `FastDetector` - Corner detectors

### Tracker Module (`src/tracker/`)

```
tracker/
├── mod.rs              # Module exports
├── vo.rs               # Visual Odometry pipeline
└── solver.rs           # PnP, Essential, Sim3 solvers
```

**Key Exports:**
- `VisualOdometry`, `VOState`, `VOResult` - VO pipeline
- `PnPSolver` - Perspective-n-Point solver
- `EssentialSolver` - Essential matrix estimation
- `Triangulator` - Point triangulation
- `Sim3Solver` - Sim(3) alignment for loop closure

### Optimizer Module (`src/optimizer/`)

```
optimizer/
├── mod.rs              # Module exports
└── ba.rs               # Bundle Adjustment (apex-solver based)
```

**Key Exports:**
- `BundleAdjuster` - BA solver
- `BACamera`, `BALandmark`, `BAObservation` - BA data types

### Loop Closing Module (`src/loop_closing/`)

```
loop_closing/
├── mod.rs              # Module exports
├── vocabulary.rs       # Bag-of-Words vocabulary
├── database.rs         # KeyFrame database
├── detector.rs         # Loop detector
├── optimized_detector.rs # Optimized detector with inverted index
├── relocalization.rs   # Relocalization module
└── closing.rs          # Loop closing optimization
```

**Key Exports:**
- `Vocabulary` - BoW vocabulary
- `LoopDetector`, `LoopCandidate` - Loop detection
- `Relocalizer` - Camera relocalization
- `LoopClosing` - Loop closure optimization

### Fusion Module (`src/fusion/`) - 3D Gaussian Splatting

```
fusion/
├── mod.rs              # Module exports
├── gaussian.rs         # Gaussian3D data structure
├── gaussian_init.rs    # Initialization from SLAM map
├── scene_io.rs         # PLY scene save/load
├── training_checkpoint.rs # Training state persistence
├── renderer.rs         # Basic forward renderer
├── diff_renderer.rs    # Differentiable renderer (CPU)
├── diff_splat.rs       # Differentiable splatting (Candle + Metal)
├── analytical_backward.rs # Analytical backward pass
├── autodiff.rs         # True autodiff wrapper
├── tiled_renderer.rs   # Tiled rasterization
├── training_pipeline.rs # Training with SSIM loss
├── complete_trainer.rs # Complete trainer with LR scheduler
├── gpu_trainer.rs      # GPU-accelerated trainer
├── slam_integrator.rs  # Sparse + Dense SLAM integration
├── tracker.rs          # Gaussian tracking (ICP)
├── mapper.rs           # Incremental Gaussian mapping
├── trainer.rs          # Basic trainer
├── tsdf_volume.rs      # TSDF volume fusion
├── marching_cubes.rs   # Marching Cubes (256 cases)
├── mesh_extractor.rs   # High-level mesh extraction API
├── mesh_io.rs          # OBJ/PLY mesh export
└── mesh_metadata.rs    # JSON metadata export
```

**Key Exports:**
- `Gaussian3D`, `GaussianMap`, `GaussianCamera` - 3DGS data
- `DiffSplatRenderer`, `TrainableGaussians` - Differentiable rendering
- `TiledRenderer`, `densify`, `prune` - Tiled rasterization
- `CompleteTrainer`, `LrScheduler` - Training pipeline
- `TsdfVolume`, `MarchingCubes`, `MeshExtractor` - Mesh extraction
- `save_mesh_obj`, `save_mesh_ply`, `export_mesh_metadata` - Export

### CLI Module (`src/cli/`)

```
cli/
├── mod.rs              # CLI implementation (clap-based)
├── pipeline_checkpoint.rs # Cross-stage checkpoint management
├── slam_pipeline.rs    # SLAM pipeline (unused)
└── integration_tests.rs # End-to-end tests
```

**CLI Features:**
- Argument parsing with `clap` derive macros
- TOML configuration file support
- JSON/text output formats
- Configurable logging (trace/debug/info/warn/error)
- Video cache configuration
- Mesh voxel size override

### Config Module (`src/config/`)

```
config/
├── mod.rs              # Module exports
├── config.rs           # SlamConfig and config loader
└── params.rs           # Component parameters
```

**Key Exports:**
- `SlamConfig`, `CameraConfig` - Main configuration
- `TrackerParams`, `MapperParams`, `OptimizerParams` - Component params
- `GaussianSplattingParams`, `TsdfParams` - 3DGS params
- `ConfigLoader` - TOML file loader

### I/O Module (`src/io/`)

```
io/
├── mod.rs              # Module exports
├── video_decoder.rs    # ffmpeg-next decoder with VideoToolbox
├── video_loader.rs     # Video abstraction
└── dataset.rs          # Dataset loaders (TUM, etc.)
```

**Key Features:**
- Hardware-accelerated decoding (VideoToolbox on macOS)
- LRU frame cache (configurable capacity)
- Format support: MP4, MOV, HEVC

### Pipeline Module (`src/pipeline/`)

```
pipeline/
├── mod.rs              # Module exports
├── checkpoint.rs       # SLAM checkpoint save/load
└── realtime.rs         # Multi-threaded realtime pipeline
```

---

## Integration Points

### RustMesh → RustSLAM Integration

```
RustSLAM/fusion/mesh_extractor.rs
    → Uses glam::Vec3 (shared math library)
    → Outputs mesh compatible with RustMesh I/O formats
```

### External Dependencies Flow

```
Video Input → ffmpeg-next (VideoToolbox) → Frame
    → ORB/Harris/FAST → Features
    → VisualOdometry → Poses
    → BundleAdjuster → Optimized Poses
    → GaussianMap → 3DGS Training
    → TsdfVolume → MarchingCubes
    → OBJ/PLY Export
```

---

## Test Coverage

### RustSLAM Tests (287 passing)

| Module | Test Count | Coverage |
|--------|------------|----------|
| core | 30+ | SE3, Frame, Map, Camera |
| features | 20+ | ORB, Matching, KNN |
| tracker | 15+ | VO, PnP, Triangulation |
| loop_closing | 10+ | Vocabulary, Detection |
| fusion | 40+ | Gaussian, Rendering, Mesh |
| cli | 5+ | Integration tests |

### RustMesh Tests

| Module | Test Coverage |
|--------|---------------|
| Core | Handles, Connectivity, I/O |
| Tools | Decimation, Subdivision |
| Utils | Circulators, Quadric |

---

## Entry Points

| Component | Entry Point | Purpose |
|-----------|-------------|---------|
| RustMesh | `src/lib.rs` | Library entry point |
| RustSLAM | `src/lib.rs` | Library entry point |
| RustSLAM CLI | `src/main.rs` | CLI binary entry point |

---

## Key Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| `RustSLAM/src/cli/mod.rs` | 606 | CLI implementation |
| `RustSLAM/src/fusion/tiled_renderer.rs` | 500+ | Tiled rasterization |
| `RustSLAM/src/fusion/gaussian.rs` | 400+ | Gaussian data structures |
| `RustSLAM/src/fusion/tsdf_volume.rs` | 350+ | TSDF volume |
| `RustSLAM/src/tracker/vo.rs` | 300+ | Visual Odometry |
| `RustMesh/src/Core/connectivity.rs` | 600+ | Half-edge connectivity |

---

## Build Profiles

### Release (Optimized)
```toml
[profile.release]
lto = true
codegen-units = 1
opt-level = 3
strip = true
panic = "abort"
```

### Development (Fast Compilation)
```toml
[profile.dev]
opt-level = 1
debug = "line-tables-only"
```