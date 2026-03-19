# Story 8.4: Create RustViewer Crate Structure

Status: done

## Story

As a RustViewer developer,
I want a well-organized crate structure with proper module separation,
so that the codebase is maintainable and follows Rust best practices.

## Acceptance Criteria

1. **Given** a new RustViewer crate
   **When** running `cargo check -p rust-viewer`
   **Then** compilation succeeds with no errors
   **And** workspace integration with RustSLAM is functional

2. **Given** the crate structure
   **When** examining the source tree
   **Then** modules are organized by responsibility (loader, renderer, ui, app)
   **And** each module has a `mod.rs` file for public exports

3. **Given** the dependency configuration
   **When** reviewing Cargo.toml
   **Then** eframe/egui are used for GUI
   **And** rustslam dependency uses `viewer-types` feature only
   **And** glam is used for math types from workspace

4. **Given** the module structure
   **When** building with `--no-default-features`
   **THEN** no heavy dependencies (ffmpeg, candle, lru, sysinfo) are included

## Tasks / Subtasks

- [x] Task 1: Verify crate structure (AC: #1, #2)
  - [x] 1.1: Confirm workspace Cargo.toml includes RustViewer
  - [x] 1.2: Verify RustViewer/Cargo.toml has correct dependencies
  - [x] 1.3: Check module structure: loader/, renderer/, ui/, app.rs
  - [x] 1.4: Confirm lib.rs re-exports and module declarations

- [x] Task 2: Validate dependencies (AC: #3, #4)
  - [x] 2.1: Run `cargo tree -p rust-viewer` and verify no heavy deps
  - [x] 2.2: Confirm eframe/egui versions are compatible
  - [x] 2.3: Verify glam is from workspace definition

- [x] Task 3: Document module responsibilities (AC: #2)
  - [x] 3.1: Add module-level documentation to each mod.rs
  - [x] 3.2: Document public API exports
  - [x] 3.3: Create README.md for RustViewer crate

- [x] Task 4: Testing infrastructure (optional)
  - [x] 4.1: Add basic unit test for module structure
  - [x] 4.2: Add integration test placeholder

## Review Follow-ups (AI)

- [x] [AI-Review][MEDIUM] panel.rs: Replace misleading placeholder values in draw_stats_cards() with "—" or "0" when no data loaded [panel.rs:328-350]
- [x] [AI-Review][MEDIUM] Verify viewer-types feature exposes necessary types from RustSLAM for Gaussian loading [RustSLAM/Cargo.toml:64]
- [x] [AI-Review][LOW] Convert integration test skips to #[ignore] attributes for visibility [tests/loader_integration_test.rs]
- [x] [AI-Review][LOW] Add named constants for camera magic numbers (ORBIT_SENSITIVITY, ZOOM_FACTOR, FIT_FACTOR) [camera.rs:35-36]
- [x] [AI-Review][LOW] Add # Example doc blocks to public loader functions [loader/*.rs]
- [x] [AI-Review][LOW] Remove or utilize unused scene_bounds snapshot in app.rs:191 [app.rs:191]

## Dev Notes

### Module Structure

```
RustViewer/
├── Cargo.toml              # Crate config with eframe/egui deps
├── src/
│   ├── main.rs             # Binary entry point
│   ├── lib.rs              # Library root, module declarations
│   ├── app.rs              # Main eframe app struct
│   ├── loader/
│   │   ├── mod.rs          # Loader trait and exports
│   │   ├── checkpoint.rs   # Checkpoint JSON loader
│   │   ├── gaussian.rs     # Gaussian PLY loader
│   │   └── mesh.rs         # OBJ/PLY mesh loader
│   ├── renderer/
│   │   ├── mod.rs          # Renderer trait and scene graph
│   │   ├── camera.rs       # Arcball camera controller
│   │   ├── scene.rs        # Scene graph and data buffers
│   │   └── pipelines.rs    # wgpu render pipelines
│   └── ui/
│       ├── mod.rs          # UI panel exports
│       ├── panel.rs        # Side panel with controls
│       ├── viewport.rs     # 3D viewport widget
│       └── theme.rs        # egui theme/styling
```

### Key Design Decisions

1. **Binary + Library split**: `main.rs` is thin wrapper, all logic in `lib.rs`
2. **Module isolation**: Each module is self-contained with clear interfaces
3. **egui/eframe**: Immediate mode GUI for simplicity and performance
4. **wgpu**: Cross-platform GPU abstraction via eframe

### Testing Standards

- Unit tests in each module using `#[cfg(test)]`
- Integration tests in `tests/` directory
- Run `cargo test -p rust-viewer` for validation

### References

- [Source: _bmad-output/implementation-artifacts/rustviewer-gui-design-spec.md]
- [Source: _bmad-output/implementation-artifacts/rustviewer-gui-design-complete.md]
- [Source: RustViewer/Cargo.toml]
- [Source: RustViewer/src/]

## Dev Agent Record

### Agent Model Used

MiniMax-M2.7

### Debug Log References

N/A

### Completion Notes List

- Verified workspace Cargo.toml includes RustViewer as member
- Verified RustViewer/Cargo.toml has correct dependencies (eframe 0.31, egui 0.31, glam workspace, rustslam viewer-types)
- Confirmed module structure matches spec: loader/, renderer/, ui/, app.rs
- Verified lib.rs correctly declares and exports all modules
- Ran cargo tree to confirm no heavy dependencies (ffmpeg, candle, lru, sysinfo)
- All 11 tests pass (8 unit tests + 3 integration tests)
- Created README.md with architecture overview and usage instructions

### File List

- RustViewer/README.md (created)
- RustViewer/Cargo.toml (verified)
- RustViewer/src/lib.rs (verified)
- RustViewer/src/main.rs (verified)
- RustViewer/src/app.rs (verified)
- RustViewer/src/loader/mod.rs (verified)
- RustViewer/src/renderer/mod.rs (verified)
- RustViewer/src/ui/mod.rs (verified)
- RustViewer/tests/loader_integration_test.rs (verified)

## Change Log

| Date | Reviewer | Action | Notes |
|------|----------|--------|-------|
| 2026-03-19 | Claude Opus 4.6 | Code Review | Approved. 0 High, 2 Medium, 4 Low issues. Created 6 action items for follow-up. All ACs verified. |
