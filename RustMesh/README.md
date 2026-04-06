# RustMesh

RustMesh is a mesh processing crate in pure Rust, inspired by OpenMesh and built around a half-edge connectivity core plus SoA-oriented storage.

This README is the canonical RustMesh overview for the current workspace. Branch-specific parity details live in [`../docs/RustMesh-OpenMesh-Progress-2026-04-05.md`](../docs/RustMesh-OpenMesh-Progress-2026-04-05.md).

## Verified Status

Validated on 2026-04-06 in the `rm-opt` worktree:

- `cargo test --manifest-path RustMesh/Cargo.toml --lib --quiet`: `221 passed; 0 failed`
- `cargo test --manifest-path RustMesh/Cargo.toml --lib tools::decimation::tests --quiet`: `12 passed; 0 failed`
- `cargo test --manifest-path RustMesh/Cargo.toml --lib tools::remeshing::tests --quiet`: `7 passed; 0 failed`
- `cargo test --manifest-path RustMesh/Cargo.toml --lib tools::vdpm::tests --quiet`: `9 passed; 0 failed`
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
| Remeshing | Functional but still being hardened | split/collapse/flip/valence/isotropic remesh are present; split path now routes through public mesh primitives |
| Progressive mesh | Partial | simplify, refine, reset, `simplification_progress()`, vertex split, and `get_lod(level)` exist; upward LOD still replays from the original mesh |

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
cargo run --release --example openmesh_compare_normals
cargo run --release --example openmesh_compare_smoothing
cargo run --release --example openmesh_compare_io
```

## Current Gaps

- The detailed RustMesh-vs-OpenMesh comparison matrix lives in [`../docs/RustMesh-OpenMesh-Progress-2026-04-05.md`](../docs/RustMesh-OpenMesh-Progress-2026-04-05.md).
- The current performance-gap execution notes live in [`../docs/RustMesh-OpenMesh-Gap-Analysis-2026-04-06.md`](../docs/RustMesh-OpenMesh-Gap-Analysis-2026-04-06.md).
- Public `split_edge()` / `split_face()` now exist, but the split primitive still uses a controlled rebuild internally rather than local half-edge surgery.
- The normals benchmark now exists and currently shows RustMesh behind OpenMesh on face / vertex / combined normal recomputation.
- Progressive mesh now exposes `get_lod(level)`, but upward LOD changes still replay from the original mesh because incremental refinement remains approximate.
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
```

## Related Docs

- Workspace overview: [`../README.md`](../README.md)
- RustMesh `rm-opt` status: [`../docs/RustMesh-OpenMesh-Progress-2026-04-05.md`](../docs/RustMesh-OpenMesh-Progress-2026-04-05.md)
- RustMesh backlog: [`../docs/RustMesh-OpenMesh-Parity-Roadmap.md`](../docs/RustMesh-OpenMesh-Parity-Roadmap.md)
