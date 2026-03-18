# Story 9-1: 创建 rustscan-types 共享 crate

**Status:** done
**Epic:** 9 - RustGS Crate 提取
**Created:** 2026-03-18

## User Story

As a RustScan developer,
I want a shared types crate for common data structures,
So that RustSLAM, RustGS, RustMesh, and RustViewer can share types without circular dependencies.

## Acceptance Criteria

- [x] Create `rustscan-types` crate with Cargo.toml
- [x] Implement `SE3` pose type (from RustSLAM/src/core/pose.rs)
- [x] Implement `Intrinsics` camera parameters type
- [x] Implement `ScenePose` type for training dataset
- [x] Implement `TrainingDataset` type for 3DGS offline training
- [x] Implement `MapPointData` type for SLAM point cloud
- [x] Implement `SlamOutput` type for SLAM → GS data transfer
- [x] Add serde serialization support for all types
- [x] Add unit tests for all types
- [x] Update workspace Cargo.toml to include rustscan-types

## Implementation Notes

### Crate Structure
```
rustscan-types/
├── Cargo.toml
└── src/
    ├── lib.rs      # Public API + re-exports
    ├── pose.rs     # SE3 pose type
    ├── camera.rs   # Intrinsics, ScenePose, TrainingDataset
    └── scene.rs    # MapPointData, SlamOutput
```

### Key Types

1. **SE3** - 3D pose representation (rotation + translation)
   - Uses glam::Quat for rotation, glam::Vec3 for translation
   - Supports serde serialization with glam's serde feature
   - Provides exp/log for Lie algebra operations

2. **Intrinsics** - Camera intrinsic parameters
   - fx, fy, cx, cy, width, height
   - Helper methods for coordinate conversion

3. **TrainingDataset** - Input for 3DGS training
   - Camera intrinsics (shared)
   - Per-frame poses with image paths
   - Optional initial point cloud

4. **SlamOutput** - Output from SLAM
   - Can convert to/from TrainingDataset
   - Contains map points for initialization

### Dependencies

- `glam` (with serde feature) - Math types
- `serde` + `serde_json` - Serialization

## Verification

- [x] `cargo check -p rustscan-types` passes
- [x] `cargo test -p rustscan-types` passes (14 tests)

## Next Steps

- Story 9-2: Create RustGS crate structure
- Update RustSLAM to use rustscan-types instead of internal SE3