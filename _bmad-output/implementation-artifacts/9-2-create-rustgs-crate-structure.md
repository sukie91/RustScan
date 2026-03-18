# Story 9-2: 创建 RustGS crate 结构

**Status:** done
**Epic:** 9 - RustGS Crate 提取
**Created:** 2026-03-19

## User Story

As a RustScan developer,
I want a RustGS crate structure for 3DGS training,
So that I can migrate 3DGS code from RustSLAM to a dedicated crate.

## Acceptance Criteria

- [x] Create `RustGS/` directory structure
- [x] Create `Cargo.toml` with dependencies (candle, rayon, kiddo)
- [x] Create `src/lib.rs` with public API and re-exports
- [x] Create module structure: core/, render/, diff/, training/, io/, init/
- [x] Create placeholder types for each module
- [x] Create CLI binary `rustgs` with train/render commands
- [x] Update workspace Cargo.toml to include RustGS
- [x] All tests pass

## Implementation Notes

### Crate Structure
```
RustGS/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API
│   ├── bin/
│   │   └── rustgs.rs       # CLI entry point
│   ├── core/
│   │   ├── mod.rs
│   │   ├── gaussian.rs     # Gaussian3D, GaussianMap, GaussianState
│   │   └── camera.rs       # GaussianCamera
│   ├── render/
│   │   ├── mod.rs
│   │   └── renderer.rs     # GaussianRenderer, RenderOutput
│   ├── diff/
│   │   ├── mod.rs
│   │   └── diff_splat.rs   # DiffSplatRenderer (placeholder)
│   ├── training/
│   │   └── mod.rs          # TrainingConfig, train()
│   ├── io/
│   │   └── mod.rs          # save_scene_ply, load_scene_ply
│   └── init/
│       └── mod.rs          # initialize_gaussians_from_points
```

### Dependencies

- `rustscan-types` - Shared types
- `glam` - Math (workspace)
- `rayon` - Parallelism
- `serde` - Serialization
- `thiserror` - Error handling
- `kiddo` - KD-tree for initialization
- `candle-core` + `candle-metal` - GPU (optional, default)

### CLI Features

- `cli` feature (default) - Enables CLI binary
- `gpu` feature (default) - Enables Candle GPU support

### CLI Commands

```bash
# Train 3DGS scene
rustgs train --input slam_output.json --output scene.ply --iterations 30000

# Render scene (not yet implemented)
rustgs render --input scene.ply --camera pose.json --output image.png
```

## Verification

- [x] `cargo check -p rustgs` passes
- [x] `cargo test -p rustgs` passes (4 tests)

## Next Steps

- Story 9-3: Migrate core Gaussian files (gaussian.rs, tiled_renderer.rs)
- Story 9-4: Migrate rendering files (renderer.rs)
- Story 9-5: Migrate differentiable rendering
- Story 9-6: Migrate training files
- Story 9-7: Migrate IO and initialization