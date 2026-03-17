# Story 8.4: Create RustViewer Crate Structure

Status: ready-for-dev

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

- [ ] Task 1: Verify crate structure (AC: #1, #2)
  - [ ] 1.1: Confirm workspace Cargo.toml includes RustViewer
  - [ ] 1.2: Verify RustViewer/Cargo.toml has correct dependencies
  - [ ] 1.3: Check module structure: loader/, renderer/, ui/, app.rs
  - [ ] 1.4: Confirm lib.rs re-exports and module declarations

- [ ] Task 2: Validate dependencies (AC: #3, #4)
  - [ ] 2.1: Run `cargo tree -p rust-viewer` and verify no heavy deps
  - [ ] 2.2: Confirm eframe/egui versions are compatible
  - [ ] 2.3: Verify glam is from workspace definition

- [ ] Task 3: Document module responsibilities (AC: #2)
  - [ ] 3.1: Add module-level documentation to each mod.rs
  - [ ] 3.2: Document public API exports
  - [ ] 3.3: Create README.md for RustViewer crate

- [ ] Task 4: Testing infrastructure (optional)
  - [ ] 4.1: Add basic unit test for module structure
  - [ ] 4.2: Add integration test placeholder

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

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
