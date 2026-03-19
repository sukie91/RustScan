# Story 9-3: 迁移核心 Gaussian 文件

**Status:** done
**Epic:** 9 - RustGS Crate 提取
**Created:** 2026-03-19

## User Story

As a RustScan developer,
I want to migrate core Gaussian files from RustSLAM to RustGS,
So that RustGS has the core 3DGS data structures and rendering capabilities.

## Acceptance Criteria

- [x] Migrate `gaussian.rs` to `RustGS/src/core/gaussian.rs`
  - [x] Gaussian3D with position, scale, rotation, opacity, color
  - [x] GaussianMap collection with add/remove operations
  - [x] GaussianState enum (New, Unstable, Stable)
  - [x] Serde support for serialization
  - [x] Unit tests pass
- [x] Migrate `tiled_renderer.rs` to `RustGS/src/render/tiled_renderer.rs`
  - [x] Gaussian struct for rendering (array-based)
  - [x] ProjectedGaussian for 2D projection
  - [x] TiledRenderer with tile-based rasterization
  - [x] RenderBuffer output
  - [x] densify() and prune() functions
  - [x] Unit tests pass
- [x] Update module exports in `lib.rs`
- [x] All tests pass

## Implementation Notes

### Files Modified

1. **RustGS/src/core/gaussian.rs** - Migrated from RustSLAM/src/fusion/gaussian.rs
   - Added serde support for serialization
   - Added bounding_box to GaussianMap
   - Added from_gaussians() constructor
   - Kept all original functionality (from_depth_point, project, etc.)

2. **RustGS/src/render/tiled_renderer.rs** - Migrated from RustSLAM/src/fusion/tiled_renderer.rs
   - Gaussian struct (array-based for rendering efficiency)
   - Conversion between Gaussian and Gaussian3D
   - TiledRenderer with 16x16 tiles
   - Alpha blending with front-to-back compositing

3. **RustGS/src/render/mod.rs** - Updated exports
4. **RustGS/src/lib.rs** - Updated public API exports

### Key Differences from RustSLAM

1. **GaussianState**: RustGS uses enum (New, Unstable, Stable) while placeholder had struct
2. **Serialization**: Added serde support for scene save/load
3. **Type conversion**: Added `from_gaussian3d()` and `to_gaussian3d()` for rendering

## Verification

```bash
cargo check -p rustgs  # Passes
cargo test -p rustgs   # 12 tests pass
```

## Next Steps

- Story 9-4: Migrate renderer.rs (basic forward renderer with depth rendering)
- Story 9-5: Migrate differentiable rendering files
- Story 9-6: Migrate training files