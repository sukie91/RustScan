# RustMesh OpenMesh Progress

**Date:** 2026-04-05
**Worktree:** `rm-opt`
**Purpose:** Canonical branch-status document for current RustMesh/OpenMesh comparison work

This document records verified facts for the current worktree. It replaces older progress reports and absorbs the stale checklist expectations that were still describing already-fixed parity failures.

## Verified Baseline

### Tests

| Command | Result |
|---------|--------|
| `cargo test --manifest-path RustMesh/Cargo.toml --lib --quiet` | `221 passed; 0 failed` |
| `cargo test --manifest-path RustMesh/Cargo.toml --lib tools::decimation::tests --quiet` | `12 passed; 0 failed` |
| `cargo test --manifest-path RustMesh/Cargo.toml --lib tools::remeshing::tests --quiet` | `7 passed; 0 failed` |
| `cargo test --manifest-path RustMesh/Cargo.toml --lib tools::vdpm::tests --quiet` | `9 passed; 0 failed` |
| `cargo test --manifest-path RustMesh/Cargo.toml --example openmesh_compare_decimation_trace --quiet` | example builds cleanly |

### Performance Snapshot

| Command | Verified result |
|---------|-----------------|
| `cargo run --manifest-path RustMesh/Cargo.toml --release --example openmesh_compare_benchmark --quiet` | `Add 1000 triangles x 1000`: RustMesh `99.7 ms`, OpenMesh `115.9 ms` after the `SoAKernel` edge-lookup rewrite |
| `cargo run --manifest-path RustMesh/Cargo.toml --release --example openmesh_compare_vector_benchmark --quiet` | `Vec4f_add_compare` remains an isolated benchmark anomaly; broader Vec3/Vec4 arithmetic is still near parity |
| `cargo run --manifest-path RustMesh/Cargo.toml --example openmesh_compare_normals --quiet` | Deterministic `64x64` sphere benchmark now exists; current run showed RustMesh `update_face_normals=511.130 ms`, `update_vertex_normals=747.596 ms`, `update_normals=1248.581 ms` vs OpenMesh `138.725 ms`, `133.423 ms`, `262.480 ms` |

### Decimation Trace Comparison

Command:

```bash
env RUSTFLAGS=-Awarnings \
  cargo run --manifest-path RustMesh/Cargo.toml --release \
  --example openmesh_compare_decimation_trace --quiet -- 10
```

Verified result:

- Default import mode: `OpenMeshParity`
- Matching prefix on removed/kept/boundary/faces_removed: `10 steps`
- Matching prefix on undirected edge + faces_removed: `10 steps`
- Result summary matches OpenMesh:
  - RustMesh: `collapsed=61, boundary=31, interior=30, final V=60, final F=109`
  - OpenMesh: `collapsed=61, boundary=31, interior=30, final V=60, final F=109`

The debugging-only `standard` import mode still diverges earlier and should not be treated as the parity baseline.

## Current Code Status

### Areas that are implemented and verified

- OBJ, PLY, STL, and OFF read/write paths are present in the library.
- HH and EE circulators exist in `RustMesh/src/Utils/circulators.rs`.
- Decimation parity closure for the default OpenMesh comparison baseline is complete.
- Remeshing tests are stable again after the split-path rework.
- VDPM create/simplify/refine/reset/`simplification_progress()` and `vertex_split()` are implemented and covered by the focused test module.
- `ProgressiveMesh::get_lod(level)` now exists as a normalized LOD API with clamp and midpoint regression coverage.
- `SoAKernel` edge lookup no longer uses a global HashMap on the active mesh path; triangle insertion now beats OpenMesh on the current benchmark harness.
- `calc_face_normal`, `calc_vertex_normal`, `update_face_normals`, `update_vertex_normals`, and `update_normals` now exist in `connectivity.rs` and are covered by focused regression tests.
- `openmesh_compare_normals.rs` now benchmarks RustMesh normal recomputation directly against an OpenMesh driver compiled from the local source mirror.
- `RustMesh::split_edge()` and `RustMesh::split_face()` now exist as public topology-edit APIs, and `remeshing::split_long_edges()` routes through `mesh.split_edge()` instead of maintaining its own duplicate rebuild path.

### Areas that are implemented but still incomplete

- Remeshing is still on a hardening track:
  - `split_long_edges()` now routes through `mesh.split_edge()`, but the split primitive still uses a controlled mesh rebuild internally rather than local half-edge surgery.
  - edge-length statistics exist, but the older roadmap's proposed histogram-style API does not.
  - current tests prove the feature is runnable and regression-protected, not that it has full OpenMesh-grade acceptance coverage.
- Normal recomputation now has a direct OpenMesh benchmark, and the current `64x64` sphere workload shows a material performance gap in RustMesh despite API parity.
- Progressive mesh LOD changes still replay from `original` because the current `refine()` / `vertex_split()` path is approximate rather than exact.
- OpenMesh comparison depth is strongest for decimation; broader algorithm-level parity coverage remains selective.

## RustMesh vs OpenMesh Comparison Matrix

| Capability area | RustMesh current state | Gap relative to OpenMesh | Gap type |
|-----------------|------------------------|---------------------------|----------|
| Core connectivity / half-edge mesh | Implemented and verified in the library test suite | No obvious missing top-level capability in the current branch | Baseline complete |
| IO (OBJ / PLY / STL / OFF) | Implemented for read/write | Core format support is present; `OpenMeshParity` import is still a parity/debug path rather than a second general-purpose API surface | Mostly complete |
| Circulators | Vertex/face/edge plus HH and EE are present | No clear feature hole in the current parity scope | Baseline complete |
| Decimation core | Quadric decimation and modular constraints are implemented | The implementation exists, but OpenMesh-grade parity is only strongly verified on the default comparison baseline | Verification gap |
| Decimation parity regression | Default `OpenMeshParity` baseline matches for the first 10 traced steps and on the current summary mesh | Regression coverage is still thinner than the underlying feature deserves; `standard` import remains a debug-only contrast path | Verification gap |
| Remeshing | Split / collapse / flip / valence / isotropic remeshing are runnable and covered by focused tests | Core topology steps still rely on shortcut behavior such as rebuild-style splitting instead of fully general low-level primitives | Feature hardening gap |
| Progressive mesh / VDPM | `create`, `simplify`, `refine`, `reset`, `simplification_progress()`, `get_lod(level)`, and `vertex_split()` are present | Upward LOD currently replays from the original mesh because incremental `refine()` / `vertex_split()` behavior is still approximate | Feature hardening gap |
| Smoothing | Uniform and tangential smoothing are implemented | Comparison depth against OpenMesh remains selective rather than comprehensive | Verification gap |
| Subdivision | Loop, Catmull-Clark, sqrt3, midpoint, and butterfly are implemented | Functionality is present, but there is not a full OpenMesh-style parity closure story around it | Verification gap |
| Dualization / hole filling / mesh repair / analysis | Implemented in the library surface | These areas are closer to "implemented" than "fully parity-validated against OpenMesh behavior" | Verification gap |
| OpenMesh comparison tooling | Dedicated examples and trace tooling exist | Coverage is still concentrated around decimation; broader algorithm-by-algorithm comparison remains selective | Coverage gap |
| Helper / test-data paths | Non-blocking helper paths exist | Some older TODO markers still remain outside the verified core library surface | Minor cleanup gap |

## Remaining Gaps Relative to OpenMesh

1. Keep decimation parity stable with a stronger automated regression than the current face-bit and example checks.
2. Replace the current rebuild-backed split primitive with local topology editing where warranted.
3. Harden incremental progressive-mesh refinement so `get_lod(level)` can move upward without replaying from the original mesh.
4. Optimize normal recomputation now that the direct OpenMesh benchmark shows a clear gap.
5. Decide whether normal recomputation should stay explicit or be auto-triggered by selected topology-changing algorithms.
6. Redesign the Vec4 comparison benchmark before treating it as a library-performance gap.
7. Expand comparison coverage only where the maintenance cost is justified by real workflow value.

## Documentation Responsibilities

To keep status reporting consistent:

- [`../RustMesh/README.md`](../RustMesh/README.md) is the crate overview.
- [`RustMesh-OpenMesh-Parity-Roadmap.md`](RustMesh-OpenMesh-Parity-Roadmap.md) is the forward backlog.
- This file is the current-branch factual snapshot.
- Compatibility redirects and older planning files should not carry their own independent status tables.
