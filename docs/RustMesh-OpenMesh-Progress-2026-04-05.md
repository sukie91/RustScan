# RustMesh OpenMesh Progress

**Updated:** 2026-04-06
**Worktree:** `rm-opt`
**Purpose:** Canonical branch-status document for current RustMesh/OpenMesh comparison work

This document records verified facts for the current worktree. It replaces older progress reports and absorbs the stale checklist expectations that were still describing already-fixed parity failures.

For a concise "current progress + next execution plan" view, see [`plans/2026-04-06-rustmesh-next-phase-plan.md`](plans/2026-04-06-rustmesh-next-phase-plan.md).

## Verified Baseline

### Tests

| Command | Result |
|---------|--------|
| `cargo test --manifest-path RustMesh/Cargo.toml --lib --quiet` | `250 passed; 0 failed` |
| `cargo test --manifest-path RustMesh/Cargo.toml --lib tools::decimation::tests --quiet` | `12 passed; 0 failed` |
| `cargo test --manifest-path RustMesh/Cargo.toml --lib tools::remeshing::tests --quiet` | `8 passed; 0 failed` |
| `cargo test --manifest-path RustMesh/Cargo.toml --lib tools::vdpm::tests --quiet` | `16 passed; 0 failed` |
| `cargo test --manifest-path RustMesh/Cargo.toml --example openmesh_compare_decimation_trace --quiet` | example builds cleanly |

### Performance Snapshot

| Command | Verified result |
|---------|-----------------|
| `cargo run --manifest-path RustMesh/Cargo.toml --release --example openmesh_compare_benchmark --quiet` | `Add 1000 triangles x 1000`: RustMesh `99.7 ms`, OpenMesh `115.9 ms` after the `SoAKernel` edge-lookup rewrite |
| `cargo run --manifest-path RustMesh/Cargo.toml --release --example openmesh_compare_vector_benchmark --quiet` | `Vec4f_add_compare` remains an isolated benchmark anomaly; broader Vec3/Vec4 arithmetic is still near parity |
| `env RUSTFLAGS=-Awarnings cargo run --manifest-path RustMesh/Cargo.toml --release --example openmesh_compare_normals --quiet` | Deterministic `64x64` sphere benchmark now exists; current release run showed RustMesh `update_face_normals=13.719 ms`, `update_vertex_normals=13.647 ms`, `update_normals=16.926 ms` vs OpenMesh `132.596 ms`, `130.897 ms`, `260.475 ms` |

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
- Remeshing acceptance now covers the shared split/collapse/flip path with topology validation and long-edge threshold checks on representative patch and sphere flows.
- VDPM create/simplify/refine/reset/`simplification_progress()` and `vertex_split()` are implemented and covered by the focused test module.
- Progressive-mesh refine records are now exact enough to restore deterministic pre-collapse snapshots without replaying simplification from `original`, and direct monotonic/bidirectional LOD regression tests now protect that behavior; the remaining VDPM gap is incremental `get_lod(level)` navigation rather than approximate replay state.
- `ProgressiveMesh::get_lod(level)` now exists as a normalized LOD API with clamp and midpoint regression coverage.
- `SoAKernel` edge lookup no longer uses a global HashMap on the active mesh path; triangle insertion now beats OpenMesh on the current benchmark harness.
- `calc_face_normal`, `calc_vertex_normal`, `update_face_normals`, `update_vertex_normals`, and `update_normals` now exist in `connectivity.rs` and are covered by focused regression tests.
- Vertex-normal semantics are now explicit in the public API: RustMesh keeps area-weighted normals as the default contract, while `VertexNormalWeighting::FaceAverage` plus `*_with_mode` recomputation APIs expose an OpenMesh-compatible equal-face-weight path.
- Topology-edit normal refresh policy is now explicit and regression-covered: maintained local edits keep requested normal arrays but do not recompute them automatically, while rebuild-backed `triangulate_face()` / n-gon `split_face()` preserve requested vertex normals but drop face-normal storage until explicit refresh.
- `openmesh_compare_normals.rs` now benchmarks RustMesh normal recomputation directly against an OpenMesh driver compiled from the local source mirror, and the maintained reproducible command is `env RUSTFLAGS=-Awarnings cargo run --manifest-path RustMesh/Cargo.toml --release --example openmesh_compare_normals --quiet` rather than any debug-build shortcut.
- `RustMesh::split_edge()` now exists as a local half-edge edit for the maintained boundary/interior triangle path, `RustMesh::split_face()` now uses local face/halfedge rewiring for valid triangle faces, and `remeshing::split_long_edges()` routes through `mesh.split_edge()` instead of maintaining its own duplicate split path.
- `RustMesh::triangulate_face()` now exists as a public rebuild-backed baseline for n-gon fan triangulation, and non-triangle `split_face()` still falls back to that controlled rebuild path; the triangle local-edit path is covered by focused regression tests.
- `AttribSoAKernel` dynamic properties now use typed per-entity handles (`VPropHandle`, `EPropHandle`, `FPropHandle`, `HPropHandle`) with typed get/set for `f32`, `Vec2`, `Vec3`, `Vec4`, and `i32`, and the stores now auto-resize on `add_vertex`, `add_edge`, and `add_face`.
- `AttribSoAKernel` PLY persistence now exists through `read_attrib_ply()` / `write_attrib_ply()`, with round-trip coverage for supported custom vertex properties `f32`, `i32`, and `Vec3`; unsupported `Vec2` / `Vec4` properties fail explicitly.
- Dynamic properties now propagate deterministically through the maintained `collapse`, `split_edge`, and triangle `split_face` paths, with focused regression coverage for vertex, edge, and face values plus a green full library suite.

### Areas that are implemented but still incomplete

- Dynamic-property persistence is still intentionally limited to supported custom vertex properties (`f32`, `i32`, `Vec3`), and rebuild-backed n-gon topology fallbacks do not yet carry the same documented propagation contract as the maintained local-edit path.
- Remeshing is still on a hardening track:
  - `split_long_edges()` now routes through `mesh.split_edge()`, and the maintained `split_edge()` plus triangle `split_face()` paths are now local.
  - representative acceptance tests now validate topology and long-edge threshold behavior on the shared remeshing primitive path.
  - the remaining gap is the rebuild-backed n-gon split/triangulation fallback plus broader OpenMesh comparison depth, not the main shared primitive route itself.
  - edge-length statistics exist, but the older roadmap's proposed histogram-style API does not.
  - current tests prove the feature is runnable and regression-protected on maintained flows, not that it has full OpenMesh-grade acceptance coverage across every algorithmic variant.
- Normal recomputation now has a direct OpenMesh benchmark, an explicit compatibility path, and a documented topology-edit refresh policy; after the latest hot-path work, the `64x64` sphere release workload shows RustMesh ahead of OpenMesh, while the remaining work is durable comparison coverage rather than raw speed.
- Progressive mesh LOD changes still replay from `original`, but the blocking gap has narrowed to incremental current-state navigation rather than inexact `refine()` / `vertex_split()` replay.
- OpenMesh comparison depth is strongest for decimation; broader algorithm-level parity coverage remains selective.

## RustMesh vs OpenMesh Comparison Matrix

| Capability area | RustMesh current state | Gap relative to OpenMesh | Gap type |
|-----------------|------------------------|---------------------------|----------|
| Core connectivity / half-edge mesh | Implemented and verified in the library test suite | No obvious missing top-level capability in the current branch | Baseline complete |
| IO (OBJ / PLY / STL / OFF) | Implemented for read/write | Core format support is present; `OpenMeshParity` import is still a parity/debug path rather than a second general-purpose API surface | Mostly complete |
| Dynamic property system | Implemented on the maintained topology path | Typed per-entity handles, typed access, automatic resize, supported PLY persistence for `f32` / `i32` / `Vec3`, and deterministic propagation through maintained `collapse` / `split_edge` / triangle `split_face` now exist; rebuild-backed n-gon fallback semantics and broader persistence coverage remain deliberate scope decisions | Scope gap |
| Circulators | Vertex/face/edge plus HH and EE are present | No clear feature hole in the current parity scope | Baseline complete |
| Decimation core | Quadric decimation and modular constraints are implemented | The implementation exists, but OpenMesh-grade parity is only strongly verified on the default comparison baseline | Verification gap |
| Decimation parity regression | Default `OpenMeshParity` baseline matches for the first 10 traced steps and on the current summary mesh | Regression coverage is still thinner than the underlying feature deserves; `standard` import remains a debug-only contrast path | Verification gap |
| Remeshing | Split / collapse / flip / valence / isotropic remeshing are runnable and acceptance-covered on the shared primitive path | The maintained path is now regression-protected, but non-triangle topology depth and broader OpenMesh comparison coverage remain thinner than the decimation baseline | Verification gap |
| Progressive mesh / VDPM | `create`, `simplify`, exact-record `refine`, `reset`, `simplification_progress()`, `get_lod(level)`, and `vertex_split()` are present | Upward LOD currently replays from the original mesh because incremental current-state navigation is still missing even though replay records are now exact | Feature hardening gap |
| Normals | API, explicit default contract, OpenMesh-compatible face-average mode, explicit topology-edit refresh policy, and release benchmark are in place | Remaining gap is durable comparison coverage, not raw performance | Coverage gap |
| Smoothing | Uniform and tangential smoothing are implemented | Comparison depth against OpenMesh remains selective rather than comprehensive | Verification gap |
| Subdivision | Loop, Catmull-Clark, sqrt3, midpoint, and butterfly are implemented | Functionality is present, but there is not a full OpenMesh-style parity closure story around it | Verification gap |
| Dualization / hole filling / mesh repair / analysis | Implemented in the library surface | These areas are closer to "implemented" than "fully parity-validated against OpenMesh behavior" | Verification gap |
| OpenMesh comparison tooling | Dedicated examples and trace tooling exist | Coverage is still concentrated around decimation; broader algorithm-by-algorithm comparison remains selective | Coverage gap |
| Helper / test-data paths | Non-blocking helper paths exist | Some older TODO markers still remain outside the verified core library surface | Minor cleanup gap |

## Remaining Gaps Relative to OpenMesh

1. Keep the normals comparison harness durable now that the default, compatibility mode, and refresh policy are explicit.
2. Wire the new exact refine records into incremental progressive-mesh navigation so `get_lod(level)` can move upward without replaying from the original mesh.
3. Keep decimation parity stable with stronger automated regression and expand comparison coverage only where it protects real workflow value.
4. Decide later whether non-triangle `split_face()` should remain a controlled rebuild fallback or gain deeper local surgery, including whether those fallback paths need the same property-propagation contract as the maintained local edits.
5. Redesign the Vec4 comparison benchmark before treating it as a library-performance gap.

## Documentation Responsibilities

To keep status reporting consistent:

- [`../RustMesh/README.md`](../RustMesh/README.md) is the crate overview.
- [`RustMesh-OpenMesh-Parity-Roadmap.md`](RustMesh-OpenMesh-Parity-Roadmap.md) is the authoritative epic/story backlog.
- This file is the current-branch factual snapshot.
- Compatibility redirects and older planning files should not carry their own independent status tables.
