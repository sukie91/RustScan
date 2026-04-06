# RustMesh

RustMesh is a mesh processing crate in pure Rust, inspired by OpenMesh and built around a half-edge connectivity core plus SoA-oriented storage.

This README is the canonical RustMesh overview for the current workspace. Branch-specific parity details live in [`../docs/RustMesh-OpenMesh-Progress-2026-04-05.md`](../docs/RustMesh-OpenMesh-Progress-2026-04-05.md).

## Verified Status

Validated on 2026-04-06 in the `rm-opt` worktree:

- `cargo test --manifest-path RustMesh/Cargo.toml --lib --quiet`: `250 passed; 0 failed`
- `cargo test --manifest-path RustMesh/Cargo.toml --lib tools::decimation::tests --quiet`: `12 passed; 0 failed`
- `cargo test --manifest-path RustMesh/Cargo.toml --lib tools::remeshing::tests --quiet`: `8 passed; 0 failed`
- `cargo test --manifest-path RustMesh/Cargo.toml --lib tools::vdpm::tests --quiet`: `16 passed; 0 failed`
- `cargo test --manifest-path RustMesh/Cargo.toml --example openmesh_compare_decimation_trace --quiet`: example builds cleanly
- `env RUSTFLAGS=-Awarnings cargo run --manifest-path RustMesh/Cargo.toml --release --example openmesh_compare_decimation_trace --quiet -- 10`: RustMesh matches OpenMesh for the first 10 traced decimation steps under the default `OpenMeshParity` import mode

## Core Capabilities

### Data Structures

- Half-edge mesh connectivity
- Smart typed handles for vertices, halfedges, edges, and faces
- SoA and attribute-aware kernels
- Status flags and smart range iteration helpers

### IO

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| OBJ | Yes | Yes | normals, texcoords, colors |
| PLY | Yes | Yes | ASCII and binary |
| STL | Yes | Yes | ASCII and binary |
| OFF | Yes | Yes | includes `read_off_openmesh_parity()` helper |

### Algorithms

| Area | Status | Notes |
|------|--------|-------|
| Decimation | Implemented | quadric decimation plus OpenMesh comparison tooling |
| Decimation modules | Implemented | modular constraints including quadric, normal, aspect ratio, boundary |
| Smoothing | Implemented | uniform and tangential paths |
| Subdivision | Implemented | Loop, Catmull-Clark, sqrt3, midpoint, butterfly |
| Hole filling | Implemented | mesh repair support |
| Mesh repair | Implemented | topology cleanup utilities |
| Dualization | Implemented | includes boundary-aware dualization |
| Analysis | Implemented | curvature, quality, area, volume, edge-length stats |
| Circulators | Implemented | vertex/face/edge plus HH and EE circulators |
| Remeshing | Implemented and regression-covered on the shared primitive path | split/collapse/flip/valence/isotropic remesh are present; representative acceptance tests now validate topology and long-edge threshold behavior |
| Progressive mesh | Partial | simplify, exact-record refine, reset, `simplification_progress()`, vertex split, and `get_lod(level)` exist; `get_lod(level)` still resets to `original` instead of navigating incrementally from current state |

## OpenMesh Comparison

RustMesh is not globally "feature parity complete" with OpenMesh, but the current branch has a solid verified baseline:

- The default decimation trace example now uses `OpenMeshParity` import mode.
- On that baseline, the first 10 traced decimation steps match OpenMesh exactly.
- Result-level summary also matches on the comparison mesh: `collapsed=61, boundary=31, interior=30, final V=60, final F=109`.
- A dedicated `openmesh_compare_normals` example now measures `update_face_normals`, `update_vertex_normals`, and `update_normals` against a local OpenMesh driver on the same deterministic OFF input.
- The legacy `standard` import mode remains available for debugging and still diverges earlier, so it is not treated as the default parity baseline.

Use these examples for comparison work:

```bash
cargo run --release --example openmesh_compare_decimation
cargo run --release --example openmesh_compare_decimation_trace -- 10
env RUSTFLAGS=-Awarnings cargo run --manifest-path RustMesh/Cargo.toml --release --example openmesh_compare_normals --quiet
cargo run --release --example openmesh_compare_smoothing
cargo run --release --example openmesh_compare_io
```

## Current Gaps

- The detailed RustMesh-vs-OpenMesh comparison matrix lives in [`../docs/RustMesh-OpenMesh-Progress-2026-04-05.md`](../docs/RustMesh-OpenMesh-Progress-2026-04-05.md).
- The current performance-gap execution notes live in [`../docs/RustMesh-OpenMesh-Gap-Analysis-2026-04-06.md`](../docs/RustMesh-OpenMesh-Gap-Analysis-2026-04-06.md).
- The authoritative implementation backlog now lives in [`../docs/RustMesh-OpenMesh-Parity-Roadmap.md`](../docs/RustMesh-OpenMesh-Parity-Roadmap.md) and is organized as epics/stories instead of another flat checklist.
- `AttribSoAKernel` dynamic properties now have typed per-entity handles, automatic resize, supported PLY round-trips for `f32`, `i32`, and `Vec3`, and deterministic propagation on the maintained `collapse` / `split_edge` / triangle `split_face` path; the remaining scope decision is whether rebuild-backed n-gon fallbacks should gain the same propagation contract, while `Vec2` / `Vec4` persistence still fails explicitly.
- `split_edge()` and triangle `split_face()` now use local half-edge surgery on the maintained topology path, while `triangulate_face()` and non-triangle `split_face()` still use controlled rebuild-backed baselines.
- Remeshing acceptance on the shared split/collapse/flip path is now regression-covered; the remaining topology gap is whether non-triangle `split_face()` / `triangulate_face()` should stay rebuild-backed or gain deeper local surgery.
- Vertex-normal semantics and refresh policy are now explicit: RustMesh defaults to area-weighted accumulation, `VertexNormalWeighting::FaceAverage` provides an OpenMesh-compatible equal-face-weight path, maintained topology edits do not auto-refresh normals, and rebuild-backed topology paths drop face-normal storage until explicit refresh; the remaining normals gap is durable comparison coverage rather than raw speed.
- Progressive mesh now exposes exact refine / `vertex_split()` replay records plus monotonic LOD regression coverage, but `get_lod(level)` still resets to `original` because incremental current-state navigation is not wired yet.
- OpenMesh verification is strongest around decimation; broader algorithm-by-algorithm comparison coverage is still selective.
- Some helper/test-data paths still contain older TODO markers that do not affect the verified library surface.

## Key Commands

```bash
# Build
cargo build --manifest-path RustMesh/Cargo.toml --release

# Full library test suite
cargo test --manifest-path RustMesh/Cargo.toml --lib

# Focused RustMesh areas
cargo test --manifest-path RustMesh/Cargo.toml --lib tools::decimation::tests
cargo test --manifest-path RustMesh/Cargo.toml --lib tools::remeshing::tests
cargo test --manifest-path RustMesh/Cargo.toml --lib tools::vdpm::tests

# Reproducible normals parity check
env RUSTFLAGS=-Awarnings cargo run --manifest-path RustMesh/Cargo.toml --release --example openmesh_compare_normals --quiet
```

## Related Docs

- Workspace overview: [`../README.md`](../README.md)
- RustMesh `rm-opt` status: [`../docs/RustMesh-OpenMesh-Progress-2026-04-05.md`](../docs/RustMesh-OpenMesh-Progress-2026-04-05.md)
- RustMesh epic/story backlog: [`../docs/RustMesh-OpenMesh-Parity-Roadmap.md`](../docs/RustMesh-OpenMesh-Parity-Roadmap.md)
