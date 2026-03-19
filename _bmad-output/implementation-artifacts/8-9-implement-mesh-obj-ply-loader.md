---
title: "Story 8.9: Implement Mesh OBJ/PLY Loader"
status: review
created: 2026-03-19
---

# Story 8.9: Implement Mesh OBJ/PLY Loader

Status: review

## Story

As a **RustViewer developer**,
I want **mesh loaders that parse OBJ and PLY files produced by RustSLAM and populate the Scene with mesh vertices and indices**,
so that **users can visualize extracted 3D meshes with both solid and wireframe rendering**.

## Acceptance Criteria

1. [x] AC1: `load_obj(path)` function exists in `RustViewer/src/loader/mesh.rs` and returns `(Vec<MeshGpuVertex>, Vec<u32>, Vec<u32>)`
2. [x] AC2: `load_ply(path)` function exists in `RustViewer/src/loader/mesh.rs` and returns `(Vec<MeshGpuVertex>, Vec<u32>, Vec<u32>)`
3. [x] AC3: OBJ loader parses RustSLAM format: `v x y z r g b`, `vn nx ny nz`, `f A//A B//B C//C`
4. [x] AC4: PLY loader parses ASCII format with vertex data `x y z nx ny nz r g b` and faces `3 i j k`
5. [x] AC5: Both loaders compute edge indices for wireframe rendering
6. [x] AC6: `load_mesh(path, &mut Scene)` dispatcher function handles format selection by extension
7. [x] AC7: Updates `Scene::mesh_vertices`, `Scene::mesh_indices`, `Scene::mesh_edge_indices`
8. [x] AC8: Updates `Scene::bounds` from all loaded positions
9. [x] AC9: Returns `Result<(), LoadError>` with proper error handling
10. [x] AC10: Unit tests cover: OBJ with color, PLY with color normalization
11. [x] AC11: All existing unit tests pass (`cargo test -p rust-viewer`)

## Tasks / Subtasks

**NOTE:** This story is largely implemented already. The task breakdown documents the existing implementation and identifies any gaps against the tech spec.

- [x] Task 1: Verify load_obj function (AC: #1, #3)
  - [x] Confirm function signature returns `(Vec<MeshGpuVertex>, Vec<u32>, Vec<u32>)`
  - [x] Confirm parsing of `v x y z r g b` vertex lines with color
  - [x] Confirm parsing of `vn nx ny nz` normal lines
  - [x] Confirm parsing of `f A//A B//B C//C` face lines (vertex//normal format)
  - [x] Confirm support for alternative formats (v/vt/vn, simple v)

- [x] Task 2: Verify load_ply function (AC: #2, #4)
  - [x] Confirm function signature returns `(Vec<MeshGpuVertex>, Vec<u32>, Vec<u32>)`
  - [x] Confirm header parsing for vertex_count and face_count
  - [x] Confirm vertex data parsing `x y z nx ny nz r g b` with u8 color normalization
  - [x] Confirm face data parsing `3 i j k` (skip count, read indices)

- [x] Task 3: Verify edge index extraction (AC: #5)
  - [x] Confirm OBJ loader extracts unique edges from faces
  - [x] Confirm PLY loader extracts unique edges from faces
  - [x] Confirm edge indices are stored as pairs for LineList rendering

- [x] Task 4: Verify load_mesh dispatcher (AC: #6, #7, #8, #9)
  - [x] Confirm format selection by file extension (.obj, .ply)
  - [x] Confirm Scene fields are populated correctly
  - [x] Confirm bounds are extended from vertex positions
  - [x] Confirm error handling for unsupported formats

- [x] Task 5: Run all tests (AC: #10, #11)
  - [x] Run `cargo test -p rust-viewer` and verify all pass
  - [x] Confirm mesh loader tests cover required scenarios

## Dev Notes

### Existing Implementation (Already Complete)

The mesh loaders were implemented during Stories 8.4-8.5 as part of the crate structure work. The following code exists in `RustViewer/src/loader/mesh.rs`:

**load_obj function:**
- Parses RustSLAM OBJ format with embedded color
- Supports multiple face formats: `v//vn`, `v/vt/vn`, simple `v`
- Computes face normals from vertex positions using cross product
- Extracts unique edges for wireframe rendering
- Returns `(vertices, indices, edge_indices)`

**load_ply function:**
- Parses ASCII PLY header to get vertex and face counts
- Reads vertex data: `x y z nx ny nz r g b`
- Normalizes u8 colors (0-255) to f32 (0.0-1.0)
- Reads face data: `3 i j k` (count + 3 indices)
- Extracts unique edges for wireframe rendering
- Returns `(vertices, indices, edge_indices)`

**load_mesh dispatcher:**
- Selects loader based on file extension
- Populates `Scene::mesh_vertices`, `Scene::mesh_indices`, `Scene::mesh_edge_indices`
- Extends `Scene::bounds` from vertex positions

### Technical Notes

- OBJ uses 1-based indices (converted to 0-based internally)
- PLY uses 0-based indices directly
- Edge indices are stored as consecutive pairs for `wgpu::PrimitiveTopology::LineList`
- Normals in OBJ are computed from face geometry, not read from file (ensures consistency with winding order)
- Only ASCII PLY is supported (RustSLAM only outputs ASCII)

### Testing Standards

- Unit tests in `#[cfg(test)]` module within mesh.rs
- Tests use tempfile for temporary file creation
- Tests cover: OBJ triangle with color, PLY triangle with color normalization
- All tests are pure CPU, no GPU dependency

### Project Structure Notes

- Mesh loader lives at `RustViewer/src/loader/mesh.rs`
- Re-exported via `RustViewer/src/loader/mod.rs`
- The module path is `crate::loader::mesh::load_mesh`

## References

- [Source: tech-spec-rust-viewer-3d-gui.md#Task-9] - Mesh OBJ/PLY loader specification
- [Source: RustSLAM/src/fusion/mesh_io.rs] - Reference for output format (save functions)
- [Source: RustViewer/src/loader/mesh.rs] - Existing implementation

---

## Change Log

| Date | Change |
|------|--------|
| 2026-03-19 | Story verification complete - all 11 ACs verified, 8 tests pass |

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

N/A - This was a verification story for already-implemented code.

### Completion Notes List

**2026-03-19:**
- Verified all 11 Acceptance Criteria pass for the Mesh OBJ/PLY Loader
- `load_obj(path)` at line 14: returns `(Vec<MeshGpuVertex>, Vec<u32>, Vec<u32>)`
- `load_ply(path)` at line 193: returns `(Vec<MeshGpuVertex>, Vec<u32>, Vec<u32>)`
- OBJ parsing: supports `v x y z r g b` (lines 33-45), `vn nx ny nz` (lines 47-54), `f A//A B//B C//C` (lines 56-103)
- OBJ also supports alternative formats: `v/vt/vn` (lines 79-91) and simple `v` (lines 92-98)
- PLY parsing: header for counts (lines 203-223), vertex data `x y z nx ny nz r g b` (lines 227-246), faces `3 i j k` (lines 249-272)
- PLY color normalization: `vals[6] / 255.0` etc (line 244)
- Edge extraction: OBJ uses HashSet of unique edges (lines 161-186), PLY similar (lines 250-277)
- `load_mesh(path, &mut Scene)` dispatcher at line 283: selects by extension, populates Scene fields, extends bounds
- Returns `Result<(), LoadError>` with UnsupportedFormat for unknown extensions
- 2 unit tests: `test_load_obj_triangle` (lines 315-336), `test_load_ply_ascii` (lines 338-372)
- All 8 tests pass: 3 camera + 3 checkpoint + 2 mesh

### File List

Files verified (no changes required - verification only):
- `RustViewer/src/loader/mesh.rs` - Mesh OBJ/PLY loader implementation with load_obj, load_ply, load_mesh functions
- `RustViewer/src/loader/mod.rs` - Module re-exports