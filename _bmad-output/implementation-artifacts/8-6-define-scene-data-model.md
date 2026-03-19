---
title: "Story 8.6: Define Scene Data Model"
status: ready-for-dev
created: 2026-03-19
---

# Story 8.6: Define Scene Data Model

Status: done

## Story

As a **RustViewer developer**,
I want **a well-defined Scene data model that holds all loaded 3D data (camera trajectories, point clouds, gaussians, meshes) with layer visibility flags and bounding box computation**,
so that **all loaders can populate a unified structure and the renderer can consume it consistently**.

## Acceptance Criteria

1. [x] AC1: `Scene` struct exists in `RustViewer/src/renderer/scene.rs` and holds all data categories: trajectory, map points, gaussians, mesh vertices/indices
2. [x] AC2: `LayerVisibility` struct provides per-layer toggle flags (trajectory, map_points, gaussians, mesh_wireframe, mesh_solid) with sensible defaults
3. [x] AC3: `SceneBounds` struct computes axis-aligned bounding box via `extend()` method and provides `center()`, `diagonal()`, `is_valid()` helpers
4. [x] AC4: `GaussianPoint` struct holds position and display color (pre-multiplied by opacity)
5. [x] AC5: `MeshGpuVertex` struct is `#[repr(C)]` + `bytemuck::Pod + Zeroable` with position, normal, color fields (36 bytes)
6. [x] AC6: `Scene::has_data()` correctly reports whether any layer contains data
7. [x] AC7: `Scene` provides count accessor methods: `keyframe_count()`, `map_point_count()`, `gaussian_count()`, `mesh_vertex_count()`
8. [x] AC8: `Scene` has `mesh_edge_indices: Vec<u32>` field for wireframe rendering support
9. [x] AC9: All existing loaders (checkpoint, gaussian, mesh) compile and function correctly against the Scene data model
10. [x] AC10: All existing unit tests pass (`cargo test -p rust-viewer`)

## Tasks / Subtasks

**NOTE:** This story is largely implemented already. The task breakdown documents the existing implementation and identifies any gaps against the tech spec.

- [x] Task 1: Verify Scene struct completeness (AC: #1, #6, #7, #8)
  - [x] Confirm `Scene` struct fields match tech spec Task 6 definition
  - [x] Verify `has_data()` checks all data categories
  - [x] Verify count accessors return correct values
  - [x] Verify `mesh_edge_indices` field exists for wireframe support

- [x] Task 2: Verify LayerVisibility defaults (AC: #2)
  - [x] Confirm defaults: trajectory=true, map_points=true, gaussians=true, mesh_wireframe=false, mesh_solid=true
  - [x] Verify defaults match panel UI checkbox initial states

- [x] Task 3: Verify SceneBounds computation (AC: #3)
  - [x] Confirm `extend()` correctly updates min/max for all 3 axes
  - [x] Confirm `center()` returns midpoint of bounds
  - [x] Confirm `diagonal()` returns Euclidean diagonal length
  - [x] Confirm `is_valid()` returns false for default (uninitialized) bounds

- [x] Task 4: Verify GaussianPoint and MeshGpuVertex (AC: #4, #5)
  - [x] Confirm `GaussianPoint` has position and color fields
  - [x] Confirm `MeshGpuVertex` derives `Pod + Zeroable` and is `#[repr(C)]`
  - [x] Confirm field layout matches WGSL vertex shader attribute expectations

- [x] Task 5: Verify loader integration (AC: #9)
  - [x] Run `cargo check -p rust-viewer` to confirm compilation
  - [x] Verify checkpoint loader populates trajectory, map_points, map_point_colors, bounds
  - [x] Verify gaussian loader populates gaussians, bounds
  - [x] Verify mesh loader populates mesh_vertices, mesh_indices, mesh_edge_indices, bounds

- [x] Task 6: Run all tests (AC: #10)
  - [x] Run `cargo test -p rust-viewer` and verify all pass
  - [x] Confirm checkpoint, OBJ, and PLY loader tests cover the data model

## Dev Notes

### Existing Implementation (Already Complete)

The scene data model was implemented during Stories 8.4-8.5 as part of the crate structure and loader work. The following structures already exist in `RustViewer/src/renderer/scene.rs`:

**Scene struct:**
```rust
#[derive(Debug, Default)]
pub struct Scene {
    pub trajectory: Vec<[f32; 3]>,         // Camera position sequence (world space)
    pub map_points: Vec<[f32; 3]>,         // Map point positions (world space)
    pub map_point_colors: Vec<[f32; 3]>,   // Map point display colors (depth-shaded)
    pub gaussians: Vec<GaussianPoint>,     // Gaussian point cloud
    pub mesh_vertices: Vec<MeshGpuVertex>, // Mesh vertices (GPU ready)
    pub mesh_indices: Vec<u32>,            // Mesh triangle indices
    pub mesh_edge_indices: Vec<u32>,       // Mesh edge indices for wireframe rendering
    pub layers: LayerVisibility,           // Layer visibility flags
    pub bounds: SceneBounds,               // Axis-aligned bounding box
}
```

**Key methods:**
- `has_data()` - checks if any data category is non-empty
- `keyframe_count()`, `map_point_count()`, `gaussian_count()`, `mesh_vertex_count()` - count accessors

**LayerVisibility:**
```rust
pub struct LayerVisibility {
    pub trajectory: bool,      // default: true
    pub map_points: bool,      // default: true
    pub gaussians: bool,       // default: true
    pub mesh_wireframe: bool,  // default: false
    pub mesh_solid: bool,      // default: true
}
```

**SceneBounds:**
```rust
pub struct SceneBounds {
    pub min: [f32; 3],  // default: [f32::MAX; 3]
    pub max: [f32; 3],  // default: [f32::MIN; 3]
}
// Methods: is_valid(), extend(p), center(), diagonal()
```

**GaussianPoint:**
```rust
pub struct GaussianPoint {
    pub position: [f32; 3],
    pub color: [f32; 3],  // Pre-multiplied by opacity in loader
}
```

**MeshGpuVertex:**
```rust
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MeshGpuVertex {
    pub position: [f32; 3],  // offset 0
    pub normal: [f32; 3],    // offset 12
    pub color: [f32; 3],     // offset 24
}  // total: 36 bytes, matches WGSL vertex layout
```

### Tech Spec Deviations

The actual implementation differs from the tech spec Task 6 in one notable way:

1. **No `camera_orientations` field**: The tech spec listed `camera_orientations: Vec<[f32; 4]>` for quaternion data. The actual implementation omits this because trajectory rendering only needs positions (polyline). Quaternion data would be needed only for camera frustum visualization, which is out of scope for the first phase.

2. **`mesh_edge_indices` added**: The implementation includes `mesh_edge_indices: Vec<u32>` for wireframe rendering via a separate LineList pipeline, instead of using `POLYGON_MODE_LINE` as originally spec'd. This avoids requiring the `POLYGON_MODE_LINE` feature and provides more reliable wireframe rendering.

3. **`MeshGpuVertex` layout**: Uses position + normal + color (36 bytes) rather than the tech spec's position + color (24 bytes) for points. The mesh pipeline needs normals for Lambert shading.

### Architecture Context

- The Scene struct is the central data contract between loaders (`RustViewer/src/loader/`) and the renderer (`RustViewer/src/renderer/`)
- Loaders take `&mut Scene` and populate specific fields, extending bounds as they go
- The renderer reads Scene fields during `update_buffers()` and caches layer flags for `render_with_layers()`
- Scene is wrapped in `Arc<Mutex<Scene>>` in `ViewerApp` for thread-safe access from file loading threads and the render callback
- `bytemuck::Pod + Zeroable` derives on `MeshGpuVertex` enable zero-copy GPU buffer uploads

### Testing Standards

- Unit tests for loaders verify data model population (checkpoint, OBJ, PLY tests in `loader/checkpoint.rs` and `loader/mesh.rs`)
- Camera tests verify bounds integration (`fit_scene()` uses `SceneBounds`)
- All tests are pure CPU, no GPU dependency
- Run: `cargo test -p rust-viewer`

### Project Structure Notes

- Scene data model lives at `RustViewer/src/renderer/scene.rs`
- This file is imported by all loaders and the renderer module
- The module path is `crate::renderer::scene::Scene`

---

## Change Log

| Date | Change |
|------|--------|
| 2026-03-19 | Story verification complete - all 10 ACs verified, 8 tests pass |

## References

- [Source: tech-spec-rust-viewer-3d-gui.md#Task-6] - Scene data model definition and field specifications
- [Source: tech-spec-rust-viewer-3d-gui.md#Task-7] - Checkpoint loader using Scene
- [Source: tech-spec-rust-viewer-3d-gui.md#Task-8] - Gaussian loader using Scene
- [Source: tech-spec-rust-viewer-3d-gui.md#Task-9] - Mesh loader using Scene
- [Source: RustViewer/src/renderer/scene.rs] - Existing implementation
- [Source: RustViewer/src/loader/checkpoint.rs] - Checkpoint loader consuming Scene
- [Source: RustViewer/src/loader/gaussian.rs] - Gaussian loader consuming Scene
- [Source: RustViewer/src/loader/mesh.rs] - Mesh loader consuming Scene
- [Source: RustViewer/src/renderer/mod.rs] - SceneRenderer consuming Scene

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

N/A - This was a verification story for already-implemented code.

### Completion Notes List

**2026-03-19:**
- Verified all 10 Acceptance Criteria pass for the Scene data model
- Scene struct contains all required fields: trajectory, map_points, map_point_colors, gaussians, mesh_vertices, mesh_indices, mesh_edge_indices, layers, bounds
- LayerVisibility defaults match spec: trajectory=true, map_points=true, gaussians=true, mesh_wireframe=false, mesh_solid=true
- SceneBounds implements all required methods: extend(), center(), diagonal(), is_valid()
- GaussianPoint has position and color fields (color pre-multiplied by opacity in loader)
- MeshGpuVertex is #[repr(C)] with Pod + Zeroable, 36 bytes total (position + normal + color)
- has_data() correctly checks all 4 data categories (trajectory, map_points, gaussians, mesh_vertices)
- Count accessors implemented: keyframe_count(), map_point_count(), gaussian_count(), mesh_vertex_count()
- mesh_edge_indices field exists for wireframe rendering support
- All 3 loaders (checkpoint, gaussian, mesh) correctly populate Scene fields and extend bounds
- All 8 unit tests pass: 3 camera tests, 3 checkpoint tests, 2 mesh tests

### File List

Files verified (no changes required - verification only):
- `RustViewer/src/renderer/scene.rs` - Scene, LayerVisibility, SceneBounds, GaussianPoint, MeshGpuVertex structs
- `RustViewer/src/loader/checkpoint.rs` - Checkpoint loader (populates trajectory, map_points, map_point_colors, bounds)
- `RustViewer/src/loader/gaussian.rs` - Gaussian loader (populates gaussians, bounds)
- `RustViewer/src/loader/mesh.rs` - Mesh loader (populates mesh_vertices, mesh_indices, mesh_edge_indices, bounds)
- `RustViewer/src/renderer/mod.rs` - SceneRenderer (consumes Scene for GPU buffer updates)
- `RustViewer/src/app.rs` - ViewerApp (owns Scene via Arc<Mutex<Scene>>)
