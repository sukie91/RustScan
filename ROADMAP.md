# RustScan Project Roadmap

> Last updated: 2026-02-15 (✅ IO module complete - Pipeline connected!)

## Project Overview

RustScan is a pure Rust implementation of a 3D scanning and reconstruction technology stack, covering the complete pipeline from camera input to mesh processing.

```
Pipeline: Camera Input → RustSLAM → 3DGS Fusion → Mesh Extraction → RustMesh Post-processing → Export ✅
```

**🎉 Major Milestone**: Phase 1 core pipeline is fully connected!

---

## I. Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      RustScan Overview                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ Camera  │ →  │ RustSLAM│ →  │ 3DGS    │ →  │ RustMesh│  │
│  │ Input   │    │ (SLAM)  │    │ (Recon) │    │ (Post)  │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│       │              │              │              │            │
│       ▼              ▼              ▼              ▼            │
│   Image/Depth    Pose Estimation  Real-time      Export ✅   │
│                  + Trajectory     Reconstruction  OBJ/PLY     │
│                                   + Rendering                   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    RustGUI (Planned)                      │   │
│  │              Real-time Visualization + GUI                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## II. Module Progress

### 2.1 RustSLAM (Visual SLAM + 3DGS)

**Progress: ~85%** ✅ Core complete + Mesh extraction

| Feature | Status | Notes |
|------|------|------|
| **Basic SLAM** |
| SE3 Pose | ✅ | Complete Lie group/algebra |
| ORB Features | ✅ | Feature extraction |
| Harris/FAST | ✅ | Corner detection |
| Feature Matching | ✅ | BFMatcher, KNN, Lowe |
| Visual Odometry | ✅ | Monocular/Stereo/RGB-D |
| BA Optimization | ✅ | Gauss-Newton |
| Loop Closing | ✅ | BoW + Database |
| Relocalization | ✅ | Recovery from loss |
| **3D Gaussian** |
| Gaussian Structure | ✅ | Gaussian3D |
| Renderer | ✅ | Tiled Rasterization |
| Depth Sorting | ✅ | Depth Sorting |
| Alpha Blending | ✅ | Alpha Blending |
| Gaussian Tracking | ✅ | ICP |
| Incremental Mapping | ✅ | Incremental Mapping |
| Densification | ✅ | Gaussian splitting |
| Pruning | ✅ | Opacity-based pruning |
| Differentiable Renderer | ✅ | Candle + Metal |
| Training Pipeline | ✅ | Trainer + Adam |
| SLAM Integration | ✅ | Sparse + Dense |
| **Mesh Extraction** |
| TSDF Volume | ✅ | Pure Rust implementation |
| Marching Cubes | ✅ | 256-case lookup table |
| Mesh Extractor | ✅ | Post-processing (cluster filtering) |
| **To Do** |
| IMU Integration | ⏳ | - |
| Multi-map SLAM | ⏳ | - |
| Semantic Mapping | ⏳ | - |
| Offline 3DGS Optimization | ⏳ | - |

**Tests:** 116/116 passing ✅

---

### 2.2 RustMesh (Mesh Processing)

**Progress: ~85%** ✅ IO module complete, core functionality ready

#### Completed

| Feature | Status |
|------|------|
| **Data Structures** |
| Handle System | ✅ |
| Half-edge | ✅ |
| SoA Layout | ✅ (Unique SIMD optimization) |
| RustMesh (Unified interface) | ✅ |
| Smart Handles | ✅ |
| **IO Formats** |
| OBJ Read/Write | ✅ (Full support) |
| PLY Export | ✅ (ASCII/Binary) |
| Conversion API | ✅ (from_triangle_mesh) |
| STL/OFF | ⏳ (Placeholders created) |
| **Circulators** |
| Vertex-* | ✅ |
| Face-* | ✅ |
| EdgeFace | ✅ |
| **Algorithms** |
| Decimation | ✅ |
| Smoother | ✅ |
| Subdivision | ✅ (Loop/CC/√3) |
| Hole Filling | ✅ |
| Mesh Repair | ✅ |
| Dualizer | ✅ |
| VDPM | ⚠️ Basic |

**Tests:** 129/129 passing ✅

#### To Do

| Priority | Feature | Notes |
|--------|------|------|
| **P1** |
| PLY Import | Complete PLY import functionality |
| STL Format | For 3D printing applications |
| MeshChecker | Mesh validation |
| **P2** |
| Advanced Decimation | Hausdorff, NormalDeviation |
| Modified Butterfly | Interpolating subdivision |
| VTK Writer | Scientific visualization |

---

### 2.3 RustGUI (GUI + 3D Rendering)

**Progress: 0%** ⬜ To be started

| Feature | Technology Choice |
|------|----------|
| 3D Rendering | egui + wgpu (recommended) |
| Camera Control | or three-d |
| UI Framework | egui / iced |

---

## III. Key Milestones

### Phase 1: Core Connection ✅ **Complete!**

```
Goal: Implement complete 3D scanning → export pipeline
```

- [x] **3DGS → Mesh Extraction** ✅
- [x] **RustMesh IO Module** ✅
- [x] **Connect SLAM → 3DGS → Mesh → Export** ✅

**Completion Date: 2026-02-15**

**Key Achievements:**
- TSDF Volume + Marching Cubes mesh extraction
- OBJ/PLY format export
- `RustMesh::from_triangle_mesh()` conversion API
- End-to-end example `e2e_export.rs`

---

### Phase 2: Feature Enhancement (Current Stage)

```
Goal: Improve algorithm toolchain
```

- [ ] Complete PLY read/write support
- [ ] STL format implementation
- [ ] MeshChecker validation
- [ ] Advanced Decimation module
- [ ] Modified Butterfly subdivision
- [ ] Offline 3DGS global optimization
- [ ] Texture mapping

**Expected Completion: TBD**

---

### Phase 3: User Experience

```
Goal: Provide visualization interface
```

- [ ] Create RustGUI project
- [ ] Real-time 3D visualization
- [ ] GUI control panel
- [ ] Multi-camera support

**Expected Completion: TBD**

---

## IV. Tech Stack

| Component | Technology |
|------|------|
| Language | Rust 2021 |
| Math Library | glam (SIMD) |
| GPU | wgpu, candle-metal |
| Image | opencv-rust, image |
| Optimization | apex-solver, g2o-rs |
| Concurrency | rayon |
| Testing | criterion |

---

## V. Comparison with Existing Open Source Projects

| Feature | ORB-SLAM3 | Open3D | RustScan |
|------|-----------|--------|----------|
| **SLAM** | ✅ | ❌ | ✅ |
| **3DGS** | ❌ | ❌ | ✅ |
| **Mesh Processing** | ❌ | ✅ | ✅ |
| **End-to-end Pipeline** | ❌ | ⚠️ Partial | ✅ |
| **Pure Rust** | ❌ | ❌ | ✅ |
| **GPU Rendering** | ❌ | ✅ | ✅ (wgpu) |

---

## VI. Code Statistics

| Module | Source Files | Lines | Tests |
|------|--------|------|------|
| RustSLAM | 48 | ~15K | 116 ✅ |
| RustMesh | ~50 | ~12K | 129 ✅ |
| **Total** | **~98** | **~27K** | **245+** |

---

## VII. Task Board

### ✅ P0 (Complete - Phase 1)
- [x] **3DGS → Mesh Extraction** - TSDF + Marching Cubes
- [x] **IO Module Implementation** - OBJ/PLY export
- [x] **Pipeline Connection** - End-to-end usable

### 🚧 P1 (Current Priority)
- [ ] Complete PLY import support
- [ ] STL format implementation
- [ ] MeshChecker validation tool
- [ ] End-to-end real data example

### ⏳ P2 (Enhancement Features)
- [ ] Advanced Decimation
- [ ] Modified Butterfly subdivision
- [ ] Offline 3DGS optimization
- [ ] VTK Writer

### 📅 P3 (User Experience)
- [ ] RustGUI project launch
- [ ] Real-time visualization
- [ ] Multi-camera support

---

## VIII. Usage Example

### Complete End-to-End Flow

```rust
// 1. RustSLAM: Extract mesh from 3DGS
use rustslam::fusion::MeshExtractor;

let mut extractor = MeshExtractor::centered(Vec3::ZERO, 2.0, 0.01);
extractor.integrate_from_gaussians(|idx| depth[idx], ...);
let slam_mesh = extractor.extract_with_postprocessing();

// 2. Convert to RustMesh
let vertices: Vec<Vec3> = slam_mesh.vertices.iter()
    .map(|v| v.position).collect();
let triangles: Vec<[usize; 3]> = slam_mesh.triangles.iter()
    .map(|t| t.indices).collect();
let normals: Vec<Vec3> = slam_mesh.vertices.iter()
    .map(|v| v.normal).collect();
let colors: Vec<[f32; 3]> = slam_mesh.vertices.iter()
    .map(|v| v.color).collect();

let mesh = RustMesh::from_triangle_mesh(
    &vertices,
    &triangles,
    Some(&normals),
    Some(&colors),
);

// 3. Export
rustmesh::io::write_obj(&mesh, "output.obj")?;
rustmesh::io::write_ply(&mesh, "output.ply", PlyFormat::Ascii)?;
```

---

## IX. Contribution Guidelines

### Code Style
- Follow Rust standards (`rustfmt`)
- Add unit tests
- Documentation comments

### Commit Convention
- Use conventional commits
- Link related modules

---

## X. References

### SLAM Related
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [SplaTAM](https://github.com/spla-tam/SplaTAM)
- [RTG-SLAM](https://github.com/MisEty/RTG-SLAM)

### Mesh Processing
- [OpenMesh](https://www.openmesh.org/)
- [Open3D](http://www.open3d.org/)

### 3DGS
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [PGSR](https://github.com/zju3dv/PGSR)
