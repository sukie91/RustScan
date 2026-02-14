# RustSLAM

<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.75+-dea584?style=for-the-badge&logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License">
</p>

A pure Rust implementation of Visual SLAM (Simultaneous Localization and Mapping) supporting monocular, stereo, and RGB-D cameras.

## 📋 Features

### Core SLAM
- ✅ **Visual Odometry** - Monocular/Stereo/RGB-D
- ✅ **Bundle Adjustment** - Gauss-Newton optimization
- ✅ **Loop Closing** - BoW-based detection
- ✅ **Relocalization** - Recover from tracking loss

### Features
- ✅ **ORB Feature Extraction**
- ✅ **Harris/FAST Corner Detection**
- ✅ **Feature Matching** - BFMatcher, KNN, Lowe Ratio Test

### Map Representations (Switchable)
- 🏗️ **Sparse Map** - Traditional feature point SLAM (done)
- 🔮 **Dense Map** - 3D Gaussian Splatting (Phase 1 ✅)

## 📁 Project Structure

```
RustSLAM/
├── src/
│   ├── core/              # Core data structures
│   │   ├── frame.rs       # Frame
│   │   ├── keyframe.rs    # KeyFrame
│   │   ├── map_point.rs   # MapPoint
│   │   ├── map.rs         # Map
│   │   ├── camera.rs      # Camera model
│   │   └── pose.rs        # SE3 Pose
│   │
│   ├── features/          # Feature extraction
│   │   ├── orb.rs         # ORB extractor
│   │   ├── pure_rust.rs   # Harris/FAST
│   │   ├── matcher.rs     # Feature matching
│   │   └── knn_matcher.rs # KNN matching
│   │
│   ├── tracker/           # Visual Odometry
│   │   ├── vo.rs          # Main VO pipeline
│   │   └── solver.rs      # PnP, Essential Matrix, Triangulation
│   │
│   ├── mapping/           # Local Mapping
│   │   └── local_mapping.rs
│   │
│   ├── optimizer/         # Bundle Adjustment
│   │   └── ba.rs
│   │
│   ├── loop_closing/      # Loop Detection
│   │   ├── vocabulary.rs  # BoW Vocabulary
│   │   ├── database.rs    # KeyFrame Database
│   │   ├── detector.rs    # Loop Detector
│   │   └── relocalization.rs
│   │
│   ├── fusion/            # Dense Fusion (Coming Soon)
│   │   └── gaussian.rs    # 3D Gaussian
│   │
│   └── viewer/            # Visualization
│       └── mod.rs
│
├── examples/              # Examples
│   └── run_vo.rs
│
├── Cargo.toml
└── DESIGN.md             # Design document
```

## 🚀 Quick Start

### Prerequisites

- Rust 1.75+
- (Optional) OpenCV 4.x for enhanced features

### Build

```bash
cd RustSLAM
cargo build --release
```

### Run Visual Odometry

```bash
cargo run --example run_vo
```

### Tests

```bash
cargo test
```

## 📊 Test Results

```
test result: ok. 77 passed, 0 failed
```

## 🗺️ Roadmap

### Phase 1: Core SLAM ✅
- [x] SE3 Pose
- [x] ORB Feature Extraction
- [x] Feature Matching
- [x] Visual Odometry
- [x] Bundle Adjustment
- [x] Loop Closing
- [x] Relocalization

### Phase 2: Dense Reconstruction ✅ COMPLETE
- [x] 3D Gaussian data structures
- [x] Gaussian Renderer (color + depth)
- [x] **Tiled Rasterization** (完整光栅化!)
- [x] **Depth Sorting** (深度排序)
- [x] **Alpha Blending** (alpha 混合)
- [x] Gaussian Tracking (ICP)
- [x] Incremental Gaussian Mapping
- [x] **Densification** (高斯分裂)
- [x] **Pruning** (透明度裁剪)
- [x] Differentiable Renderer (Candle + Metal MPS)
- [x] Training Pipeline (Trainer + Adam optimizer)
- [x] TRUE Backward Propagation (Var + backward() + gradients.get())
- [x] **SLAM Integration** (Sparse + Dense 融合!)

### Phase 3: Advanced Features
- [ ] IMU Integration
- [ ] Multi-map SLAM
- [ ] Semantic Mapping

## 🔬 Comparison with pySLAM

| Feature | pySLAM | RustSLAM |
|---------|--------|-----------|
| Visual Odometry | ✅ | ✅ |
| Bundle Adjustment | ✅ | ✅ |
| BoW Vocabulary | ✅ | ✅ |
| KeyFrame Database | ✅ | ✅ |
| Loop Closing | ✅ | ✅ |
| Relocalization | ✅ | ✅ |
| 3D Gaussian | ✅ | 🔄 Coming |
| Volumetric | ✅ | ❌ |
| Depth Prediction | ✅ | ❌ |

## 📖 References

- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [pySLAM](https://github.com/luigifreda/pyslam)
- [RTG-SLAM](https://github.com/MisEty/RTG-SLAM) - Real-time 3DGS
- [SplaTAM](https://github.com/spla-tam/SplaTAM) - CVPR 2024

## 📄 License

MIT License - see LICENSE file for details.

---

<p align="center">
Built with ❤️ in Rust
</p>
