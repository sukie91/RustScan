# RustMesh vs OpenMesh Gap Analysis

**Date:** 2026-04-06
**Branch:** `worktree-rm-opt`
**Goal:** Track the real remaining gaps versus OpenMesh without mixing completed work, outdated benchmark conclusions, and future planning.

This document is diagnostic, not a task list. The authoritative execution backlog now lives in [`RustMesh-OpenMesh-Parity-Roadmap.md`](RustMesh-OpenMesh-Parity-Roadmap.md).

---

## Re-Verified Benchmark Results (release mode, 2026-04-06)

Benchmarks referenced here were re-run from the current worktree during this doc refresh.

| Test | RustMesh | OpenMesh | Interpretation |
|------|----------|----------|----------------|
| Build 1000 tetrahedra | `755 us` | `1.78 ms` | RustMesh is ahead on the current harness |
| Add 1000 triangles x 1000 | `99.7 ms` | `115.9 ms` | `SoAKernel` edge lookup hot path is no longer a live parity gap |
| Triangle area x 2M | `16.7 ms` | `126.0 ms` | RustMesh is ahead on the current harness |
| Vec3f arithmetic | `~0.7-3.1 ns` | `~0.7-2.4 ns` | Near parity / benchmark noise |
| Vec4f_add_compare | `~14-15 ns` | `~0.8 ns` | Still anomalous, but benchmark-scoped rather than a confirmed library gap |
| update_face_normals | `13.719 ms` | `132.596 ms` | RustMesh release path is ahead on the current sphere workload |
| update_vertex_normals | `13.647 ms` | `130.897 ms` | RustMesh release path is ahead on the current sphere workload |
| update_normals | `16.926 ms` | `260.475 ms` | RustMesh release path is ahead on the current sphere workload |

Key correction: earlier normals conclusions based on Rust debug builds versus OpenMesh `-O3` were misleading. For the maintained comparison, release-mode numbers are the relevant data.

---

## What Is No Longer a Meaningful Gap

### 1. `add_face` hot-path lookup cost

The active `SoAKernel` path no longer depends on the old global edge `HashMap` lookup strategy. The per-vertex sorted adjacency rewrite moved `Add 1000 triangles x 1000` from roughly `156 ms` to `99.7 ms`, which now beats the current OpenMesh comparison result.

### 2. Normals raw performance

The normals API family now exists:

- `calc_face_normal`
- `calc_vertex_normal`
- `update_face_normals`
- `update_vertex_normals`
- `update_normals`

The current remaining normals gap is not raw release performance. It is behavior definition and integration policy.

### 3. Public split API surface

RustMesh already exposes public `split_edge(eh, pt)` and `split_face(fh, pt)` entry points, and remeshing now routes through `mesh.split_edge()` rather than maintaining a second split path.

### 4. Missing face triangulation API

RustMesh now exposes `triangulate_face(fh)` as a public baseline API for fan-triangulating n-gons. Like the current public split primitives, it is rebuild-backed for now, but the API surface is no longer missing.

---

## Actual Remaining Gaps

### G1 — Dynamic Property Core Gap Is Closed on the Maintained Path

**Primary files:** `RustMesh/src/Core/attrib_soa_kernel.rs`, `RustMesh/src/Core/soa_kernel.rs`, `RustMesh/src/Core/connectivity.rs`

Current state:

- typed per-entity handles and per-domain stores now exist
- typed get/set now works for the supported numeric/vector property kinds
- stores now auto-resize on vertex, edge, halfedge, and face growth
- supported `AttribSoAKernel` PLY round-trips now exist for vertex `f32`, `i32`, and `Vec3` properties
- unsupported `Vec2` / `Vec4` vertex-property persistence fails explicitly instead of silently degrading
- deterministic propagation now exists on the maintained `collapse`, `split_edge`, and triangle `split_face` paths
- the remaining scope question is whether rebuild-backed n-gon topology fallbacks should inherit the same propagation contract or stay explicitly out of parity scope

Why this matters:

- OpenMesh-grade mesh processing relies heavily on per-entity extensibility
- RustMesh now has the right storage shape, a bounded IO path, and maintained-path propagation semantics
- the active remaining work has moved to normals, incremental LOD, parity breadth, and the still-mixed non-triangle topology surface rather than core property propagation

### G2 — Normals Gap Is Semantic, Not Performance-Based

**Primary file:** `RustMesh/src/Core/connectivity.rs`

Current state:

- RustMesh now explicitly documents area-weighted vertex normals as the default contract
- OpenMesh default `update_vertex_normals()` uses `calc_vertex_normal_fast()`, which averages adjacent face normals and normalizes
- RustMesh now exposes `VertexNormalWeighting::FaceAverage` plus `*_with_mode` APIs for an OpenMesh-compatible equal-face-weight path
- topology-changing operations now explicitly follow an explicit-refresh policy on the maintained path, while rebuild-backed triangulation/split fallbacks drop face-normal storage until refresh
- current checksum deltas in the normals benchmark are therefore explainable semantic differences rather than automatic proof of a bug

Remaining work:

- keep the normals comparison harness and regression coverage aligned with the now-explicit default-vs-compatible contract

### G3 — Topology Editing Still Needs Deeper Local Surgery

**Primary files:** `RustMesh/src/Core/connectivity.rs`, `RustMesh/src/Tools/remeshing.rs`

Current state:

- public split APIs exist
- remeshing uses the shared split API
- `split_edge()` now uses local half-edge surgery on the maintained boundary/interior triangle path
- triangle `split_face()` now uses local face/halfedge rewiring on the maintained path
- `triangulate_face()` and non-triangle `split_face()` still fall back to controlled rebuild behavior internally

Why this is still a gap:

- the shared remeshing path is now acceptance-covered, but non-triangle topology depth is still mixed local/rebuild behavior
- non-triangle face splitting still uses a controlled rebuild fallback, so topology depth is not yet uniformly local
- rebuild-backed non-triangle topology still does not share the same documented local property-propagation contract as the maintained triangle/local-edit path

### G4 — Progressive-Mesh Upward LOD Still Replays from `original`

**Primary file:** `RustMesh/src/Tools/vdpm.rs`

Current state:

- `get_lod(level)` now exists and is regression-covered
- upward movement still re-runs simplification from `original` to the requested face budget

Why this is still a gap:

- the API shape suggests practical normalized LOD navigation
- replay-from-original is acceptable as a baseline but not OpenMesh-grade incremental behavior

### G5 — Parity Verification Is Uneven Outside Decimation

Current state:

- decimation has the strongest OpenMesh comparison story
- smoothing, subdivision, dualization, hole filling, analysis, and IO are implemented
- those areas do not yet have the same parity-protection depth as decimation

Why this matters:

- today RustMesh is closer to "implemented with selective parity evidence" than "broadly behavior-locked to OpenMesh"

### G6 — `Vec4f_add_compare` Is Still a Benchmark Investigation, Not a Closed Issue

The current anomaly remains useful as a benchmark note, but it should not be treated as a confirmed RustMesh library deficit until the harness is redesigned to rule out benchmark-shape and codegen artifacts.

---

## Priority Order

If the goal is practical OpenMesh parity instead of feature-count inflation, the current priority order is:

1. Durable normals comparison coverage
2. Incremental progressive-mesh LOD
3. Selective parity regression outside decimation
4. Any remaining non-triangle topology depth decisions, including whether rebuild-backed fallbacks need the maintained property contract
5. Benchmark-only cleanup such as `Vec4f_add_compare`

---

## Mapping to the Roadmap

The active execution backlog is maintained in [`RustMesh-OpenMesh-Parity-Roadmap.md`](RustMesh-OpenMesh-Parity-Roadmap.md).

| Gap | Roadmap Epic |
|-----|--------------|
| `G1` dynamic properties | Epic 1 core maintained-path work complete |
| `G2` normals semantics and refresh policy | Epic 4 |
| `G3` local split internals and remeshing hardening | Epic 2 |
| `G4` incremental progressive-mesh LOD | Epic 3 |
| `G5` parity verification breadth | Epic 5 |
| `G6` benchmark-scoped anomaly cleanup | Epic 5 |

---

## Working Notes

- Build command: `cargo build --manifest-path RustMesh/Cargo.toml --release`
- Test command: `cargo test --manifest-path RustMesh/Cargo.toml --lib --quiet`
- Decimation parity check: `env RUSTFLAGS=-Awarnings cargo run --manifest-path RustMesh/Cargo.toml --release --example openmesh_compare_decimation_trace --quiet -- 10`
- Normals comparison: `env RUSTFLAGS=-Awarnings cargo run --manifest-path RustMesh/Cargo.toml --release --example openmesh_compare_normals --quiet`
- OpenMesh mirror remains the API/behavior reference at `Mirror/OpenMesh-11.0.0/`
- The SoA layout is intentional; parity work should not collapse it into an AoS shortcut
