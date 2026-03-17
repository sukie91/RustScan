# Story 8.3: Add cfg Gates for Heavy Dependencies

Status: done

## Story

As a developer building RustViewer,
I want RustSLAM's heavy dependencies (ffmpeg-next, candle-core, candle-metal) to be gated behind feature flags,
so that RustViewer can depend on RustSLAM without pulling in unnecessary build-time-heavy crates.

## Acceptance Criteria

1. **Given** RustSLAM's Cargo.toml has `ffmpeg-next`, `candle-core`, `candle-metal` as dependencies
   **When** building with `--no-default-features --features viewer-types`
   **Then** none of these heavy dependencies are compiled
   **And** `cargo tree` shows no ffmpeg/candle/lru/sysinfo entries

2. **Given** RustSLAM modules that use ffmpeg (io/, cli/) or candle (fusion GPU modules)
   **When** the corresponding features are disabled
   **Then** those modules are excluded via `#[cfg(feature = "...")]` and compilation succeeds

3. **Given** RustViewer depends on `rustslam = { default-features = false, features = ["viewer-types"] }`
   **When** building RustViewer with `cargo check -p rust-viewer`
   **Then** compilation succeeds without errors
   **And** no heavy dependencies are transitively included

4. **Given** the full RustSLAM pipeline with default features
   **When** building with `cargo check -p rustslam`
   **Then** all modules compile correctly (no regressions from cfg gating)

## Tasks / Subtasks

- [x] Task 1: Verify existing cfg gates are complete (AC: #1, #2)
  - [x] 1.1: Confirm `slam-pipeline` feature gates `io` and `cli` modules in lib.rs
  - [x] 1.2: Confirm `gpu` feature gates candle-dependent fusion modules in fusion/mod.rs
  - [x] 1.3: Confirm `ffmpeg-next`, `candle-core`, `candle-metal` are `optional = true` in Cargo.toml
  - [x] 1.4: Verify `viewer-types` feature exists (currently empty, which is correct)
- [x] Task 2: Validate RustViewer builds cleanly (AC: #3)
  - [x] 2.1: Run `cargo check -p rust-viewer` — must pass
  - [x] 2.2: Run `cargo tree -p rust-viewer` — must not contain ffmpeg/candle/lru/sysinfo
- [x] Task 3: Validate full pipeline still works (AC: #4)
  - [x] 3.1: Run `cargo check -p rustslam` (default features) — must pass
  - [x] 3.2: Run `cargo test --lib -p rustslam` — 228 tests pass (viewer-types); pre-existing E0308 in cli/integration_tests.rs unrelated to cfg gates
- [x] Task 4: Add any missing cfg gates if found during validation
  - [x] 4.1: Check if any non-gated code paths import from gated modules — found `pipeline/additional_tests.rs`
  - [x] 4.2: Fix compilation error: gated `pipeline/additional_tests.rs` behind `slam-pipeline` feature

## Dev Notes

### Current State Analysis (Pre-Implementation)

**This story may already be complete.** Analysis of the current codebase shows:

1. **Cargo.toml features already configured:**
   ```toml
   [features]
   default = ["slam-pipeline", "gpu"]
   slam-pipeline = ["dep:ffmpeg-next", "dep:lru", "dep:sysinfo"]
   gpu = ["dep:candle-core", "dep:candle-metal"]
   viewer-types = []
   ```

2. **lib.rs cfg gates already in place:**
   - `#[cfg(feature = "slam-pipeline")] pub mod io;`
   - `#[cfg(feature = "slam-pipeline")] pub mod cli;`

3. **fusion/mod.rs cfg gates already in place:**
   - All GPU-dependent modules gated behind `#[cfg(feature = "gpu")]`
   - Non-GPU modules (scene_io, marching_cubes, mesh_io, etc.) are unconditionally available

4. **Validation results (pre-story):**
   - `cargo check -p rust-viewer` ✅ passes
   - `cargo tree -p rust-viewer | grep ffmpeg/candle` → empty (no heavy deps)
   - `cargo check -p rustslam --no-default-features --features viewer-types` ✅ passes
   - `cargo check -p rustslam` (default features) ✅ passes

5. **RustViewer only uses:**
   - `rustslam::fusion::load_scene_ply()` — from `scene_io` module (not gated, correct)
   - No other rustslam imports

### Architecture Compliance

- [Source: RustSLAM/Cargo.toml] Feature flags follow Rust convention: `dep:crate_name` syntax
- [Source: tech-spec-rust-viewer-3d-gui.md] Tech spec requires `viewer-types` feature to gate heavy deps
- [Source: RustSLAM/src/lib.rs] Module-level cfg gates prevent compilation of unused code paths
- [Source: RustSLAM/src/fusion/mod.rs] GPU modules properly gated, non-GPU modules available for viewer

### Key Files

| File | Role |
|------|------|
| `RustSLAM/Cargo.toml` | Feature definitions and optional deps |
| `RustSLAM/src/lib.rs` | Module-level cfg gates |
| `RustSLAM/src/fusion/mod.rs` | GPU module cfg gates |
| `RustViewer/Cargo.toml` | Consumer of `viewer-types` feature |

### Testing Standards

- Run `cargo check` with various feature combinations
- Run `cargo tree` to verify dependency exclusion
- Run `cargo test --lib -p rustslam` to verify no regressions

### Project Structure Notes

- Workspace root: `Cargo.toml` with members `[RustMesh, RustSLAM, RustViewer]`
- RustViewer depends on RustSLAM with `default-features = false, features = ["viewer-types"]`
- This is the standard Rust pattern for lightweight feature-gated dependencies

### References

- [Source: _bmad-output/implementation-artifacts/tech-spec-rust-viewer-3d-gui.md#Technical Decisions §3]
- [Source: _bmad-output/planning-artifacts/architecture.md#Optional Features]
- [Source: _bmad-output/project-context.md#Optional Features]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

- `cargo check -p rust-viewer` → pass (23 warnings, 0 errors)
- `cargo tree -p rust-viewer | grep ffmpeg/candle/lru/sysinfo` → empty
- `cargo check -p rustslam` → pass (54 warnings, 0 errors)
- `cargo test --lib -p rustslam --no-default-features --features viewer-types` → 228 passed, 0 failed
- Pre-existing E0308 in `cli/integration_tests.rs:140` (type mismatch from uncommitted `cli/mod.rs` changes, unrelated to cfg gates)

### Completion Notes List

- ✅ All cfg gates verified: `slam-pipeline` gates io/cli, `gpu` gates 9 candle-dependent fusion modules
- ✅ All 5 heavy deps confirmed `optional = true`: ffmpeg-next, candle-core, candle-metal, lru, sysinfo
- ✅ `viewer-types` feature exists and is empty (correct — no deps needed for type re-exports)
- ✅ RustViewer compiles cleanly with zero heavy transitive deps
- ✅ Fixed missing cfg gate: `pipeline/additional_tests.rs` now gated behind `slam-pipeline` (was causing E0432 when building tests without slam-pipeline feature)
- ⚠️ Pre-existing test compile error in `cli/integration_tests.rs:140` — NOT introduced by this story, caused by uncommitted changes to `cli/mod.rs` API signature

### File List

- `RustSLAM/src/pipeline/mod.rs` — Changed `#[cfg(test)]` to `#[cfg(all(test, feature = "slam-pipeline"))]` for `additional_tests` module
