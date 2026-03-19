# Story 9-4: 迁移渲染文件

**Status:** done
**Epic:** 9 - RustGS Crate 提取
**Created:** 2026-03-19

## User Story

As a RustScan developer,
I want to migrate the Gaussian renderer from RustSLAM to RustGS,
So that RustGS can render 3DGS scenes with depth output for mesh extraction.

## Acceptance Criteria

- [x] Migrate `renderer.rs` to `RustGS/src/render/renderer.rs`
  - [x] GaussianRenderer with render(), render_depth(), render_depth_and_color()
  - [x] RenderOutput struct
  - [x] Adapted to use new GaussianCamera with Intrinsics/SE3
  - [x] Unit tests pass
- [x] Update module exports
- [x] All tests pass

## Implementation Notes

### Key Changes from RustSLAM

1. **GaussianCamera**: Uses new types from rustscan-types
   - `intrinsics: Intrinsics` (fx, fy, cx, cy, width, height)
   - `extrinsics: SE3` (rotation + translation)

2. **SE3 Integration**: The renderer now extracts rotation matrix and translation from SE3:
   ```rust
   let rot_mat = camera.extrinsics.rotation_matrix();
   let t_arr = camera.extrinsics.translation();
   ```

3. **Rendering Methods**:
   - `render()` - Full color + depth render
   - `render_depth()` - Depth only (for TSDF integration in RustMesh)
   - `render_depth_and_color()` - Depth + color (for colored mesh extraction)

### Files Modified

- [RustGS/src/render/renderer.rs](RustGS/src/render/renderer.rs) - Migrated from RustSLAM/src/fusion/renderer.rs

## Verification

```bash
cargo check -p rustgs  # Passes
cargo test -p rustgs   # 16 tests pass
```

## Next Steps

- Story 9-5: Migrate differentiable rendering files
- Story 9-6: Migrate training files