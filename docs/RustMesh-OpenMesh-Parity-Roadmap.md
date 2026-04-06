# RustMesh OpenMesh Parity Roadmap

**Updated:** 2026-04-06
**Scope:** Authoritative epic/story backlog for the remaining RustMesh vs OpenMesh work in `rm-opt`

This file is the only execution backlog for the current parity effort. Verified facts belong in [`RustMesh-OpenMesh-Progress-2026-04-05.md`](RustMesh-OpenMesh-Progress-2026-04-05.md); detailed diagnosis and benchmark rationale belong in [`RustMesh-OpenMesh-Gap-Analysis-2026-04-06.md`](RustMesh-OpenMesh-Gap-Analysis-2026-04-06.md).

For the condensed "where we are now + what we do next" snapshot, see [`plans/2026-04-06-rustmesh-next-phase-plan.md`](plans/2026-04-06-rustmesh-next-phase-plan.md).

## Planning Inputs

- Current verified status: [`RustMesh-OpenMesh-Progress-2026-04-05.md`](RustMesh-OpenMesh-Progress-2026-04-05.md)
- Detailed remaining-gap analysis: [`RustMesh-OpenMesh-Gap-Analysis-2026-04-06.md`](RustMesh-OpenMesh-Gap-Analysis-2026-04-06.md)
- User-facing crate overview: [`../RustMesh/README.md`](../RustMesh/README.md)
- OpenMesh reference source: `Mirror/OpenMesh-11.0.0/`

Latest re-verified checks for this planning pass:

- `cargo test --manifest-path RustMesh/Cargo.toml --lib --quiet`: `250 passed; 0 failed`
- `env RUSTFLAGS=-Awarnings cargo run --manifest-path RustMesh/Cargo.toml --release --example openmesh_compare_normals --quiet`:
  - `update_face_normals`: RustMesh `13.719 ms`, OpenMesh `132.596 ms`
  - `update_vertex_normals`: RustMesh `13.647 ms`, OpenMesh `130.897 ms`
  - `update_normals`: RustMesh `16.926 ms`, OpenMesh `260.475 ms`

## Remaining Requirement Inventory

### Functional Requirements

- `FR1`: Dynamic properties must propagate predictably through topology edits such as `collapse`, `split_edge`, and `split_face`.
- `FR2`: Supported vertex dynamic properties must survive PLY write/read round-trips.
- `FR3`: `split_edge()` and the maintained triangle `split_face()` path must stop relying on rebuild-backed shortcuts and become local topology edits.
- `FR4`: Remeshing must continue to route through the shared split/collapse primitives without behavior regressions.
- `FR5`: Progressive mesh LOD movement must support upward navigation without replaying from the original mesh on every request.
- `FR6`: Normal recomputation behavior must have an explicit contract relative to OpenMesh, especially for vertex-normal semantics and refresh timing.
- `FR7`: OpenMesh parity coverage must be strong enough to protect decimation and other high-value workflows from silent drift.

### Non-Functional Requirements

- `NFR1`: `cargo test --manifest-path RustMesh/Cargo.toml --lib --quiet` must remain green throughout parity work.
- `NFR2`: New parity work must not materially regress the current release-mode performance baselines without explicit documentation.
- `NFR3`: Planning status must stay centralized across the RustMesh README, progress doc, and this roadmap.
- `NFR4`: The active SoA storage model is intentional and must not be replaced with an AoS layout as a shortcut.
- `NFR5`: Comparison tooling should be added selectively, only where it protects behavior that matters to real RustMesh workflows.

### Additional Constraints

- OpenMesh decimation parity is defined against the default `OpenMeshParity` import path, not the debugging-only `standard` path.
- `AttribSoAKernel` already has typed per-entity property handles, automatic store resize, and supported PLY persistence for `f32`, `i32`, and `Vec3`; deterministic propagation is now implemented on the maintained `collapse` / `split_edge` / triangle `split_face` path, so any remaining property work is deliberate persistence-scope expansion or non-triangle fallback semantics rather than core propagation absence.
- `RustMesh::triangulate_face()` already exists as a rebuild-backed baseline API, and non-triangle `split_face()` currently reuses that controlled fallback, so the remaining topology backlog is about maintained-path local editing depth rather than API absence.
- RustMesh currently computes area-weighted vertex normals, while OpenMesh default `update_vertex_normals()` averages adjacent face normals.
- The current release-mode normals benchmark is already favorable to RustMesh; remaining normals work is semantic and integration focused, not raw-speed driven.
- Public `split_edge()` and `split_face()` already exist, so follow-up work should harden internals rather than introduce new surface APIs.

## Epic Summary

| Epic | Priority | Goal | Depends on |
|------|----------|------|------------|
| Epic 2 | Critical | Replace rebuild-backed topology edits with local half-edge surgery and finish remeshing hardening | None |
| Epic 1 | High | Finish the remaining dynamic property work on top of the stabilized topology path | Epic 2 |
| Epic 4 | High | Lock normals semantics and refresh policy relative to OpenMesh | Epic 2 helpful, especially for refresh-policy decisions |
| Epic 3 | Medium | Make progressive-mesh LOD movement incremental and exact enough for repeated navigation | None, but isolated from the main parity-critical path |
| Epic 5 | Low | Promote selective parity harnesses and regression coverage into durable protection | Best after the corresponding feature semantics are stable |

## Epic 1: Dynamic Property System Parity

Finish the remaining property-system gaps now that typed per-entity handles, typed access, automatic resize, and the first supported PLY persistence slice are already in place.

### Story 1.1: Property Propagation Through Topology Edits

As a mesh-processing algorithm author,
I want dynamic properties to propagate through `collapse`, `split_edge`, and `split_face`,
So that topology edits do not destroy semantic mesh data.

**Status:** Implemented on the maintained local-edit topology path. Vertex properties now interpolate deterministically on `collapse`, `split_edge`, and triangle `split_face`; derived faces copy parent face properties; split-derived edge copies are preserved on `split_edge`; and focused regression coverage plus the full RustMesh library suite are green (`cargo test --manifest-path RustMesh/Cargo.toml propagates --quiet`: `3 passed; 0 failed`, `cargo test --manifest-path RustMesh/Cargo.toml --lib --quiet`: `250 passed; 0 failed`).

**Acceptance Criteria:**

**Given** scalar and vector properties on vertices, edges, and faces
**When** `collapse`, `split_edge`, or `split_face` is executed
**Then** surviving and new elements receive documented propagated values
**And** collapse uses deterministic interpolation where interpolation is required.

### Story 1.2: Vertex Property Persistence in PLY

As a RustMesh user exporting analysis data,
I want supported vertex properties serialized to and from PLY,
So that custom numeric attributes survive interchange.

**Status:** Implemented for `AttribSoAKernel` through `read_attrib_ply()` / `write_attrib_ply()` with supported vertex property types `f32`, `i32`, and `Vec3`. `Vec2` and `Vec4` currently return explicit write errors.

**Acceptance Criteria:**

**Given** a mesh with supported custom vertex properties
**When** it is written to PLY and read back
**Then** the property schema and values are preserved for the supported types
**And** unsupported property kinds fail with explicit documentation or error behavior.

## Epic 2: Local Topology Editing and Remeshing Hardening

Replace rebuild-backed split internals with local topology surgery so RustMesh behaves like a real half-edge editing library instead of a controlled rebuild wrapper.

### Story 2.1: Local `split_edge()` Implementation

As a topology-edit caller,
I want `split_edge()` implemented with local half-edge surgery,
So that edge splits preserve topology and scale without rebuilding the whole mesh.

**Status:** Implemented for the maintained boundary/interior triangle path, with direct half-edge rewiring and regression coverage. `tools::remeshing::tests` and the full RustMesh library suite are green on top of this path.

**Acceptance Criteria:**

**Given** interior and boundary edges on valid meshes
**When** `split_edge()` is called
**Then** connectivity, handles, and face counts match the documented local-edit semantics
**And** the implementation no longer routes through a whole-mesh rebuild helper.

### Story 2.2: Local `split_face()` Implementation

As a topology-edit caller,
I want `split_face()` implemented with local face/halfedge rewiring,
So that face insertion semantics are stable and property propagation hooks stay local.

**Status:** Implemented for valid triangle faces with direct local face/halfedge rewiring, plus focused `split_face` regression coverage and a green full RustMesh library suite (`250 passed; 0 failed`).

**Acceptance Criteria:**

**Given** a valid triangle face
**When** `split_face()` is called
**Then** a new vertex and the expected replacement faces are created
**And** local connectivity remains valid without rebuilding the mesh.

### Story 2.3: Remeshing Acceptance on Shared Primitives

As a remeshing maintainer,
I want `split_long_edges()` and related flows verified against the shared primitives,
So that remeshing stays stable after the topology-internal rewrite.

**Status:** Implemented with stronger remeshing acceptance coverage on representative patch and sphere flows, including topology validation and long-edge threshold assertions on the shared primitive path. Verified with `cargo test --manifest-path RustMesh/Cargo.toml --lib tools::remeshing::tests --quiet` (`8 passed; 0 failed`) and the full RustMesh library suite (`250 passed; 0 failed`).

**Acceptance Criteria:**

**Given** the existing remeshing test suite and representative meshes
**When** remeshing operations run through the local split/collapse path
**Then** topology validity and edge-length improvement expectations remain green
**And** no rebuild-only helper is required on the main remeshing path.

## Epic 3: Incremental Progressive-Mesh LOD

Harden VDPM/refine behavior so normalized LOD queries can move in both directions without replaying the original mesh every time.

### Story 3.1: Exact Refine and Vertex-Split Records

As a progressive-mesh maintainer,
I want refine metadata to be exact enough for repeated forward/backward navigation,
So that LOD changes are not approximated through full resimplification.

**Status:** Implemented. `CollapseRecord` now stores exact pre-collapse mesh snapshots, `vertex_split()` restores those snapshots deterministically, and `refine()` replays them without depending on re-running simplification from `original`. Verified with `cargo test --manifest-path RustMesh/Cargo.toml --lib tools::vdpm::tests --quiet` (`16 passed; 0 failed`) and the full RustMesh library suite (`250 passed; 0 failed`).

**Acceptance Criteria:**

**Given** a simplification sequence
**When** split/refine records are stored and replayed
**Then** upward refinement reconstructs the intended topology deterministically
**And** the replay does not depend on re-running simplification from `original`.

### Story 3.2: Incremental `get_lod(level)` Navigation

As a caller using normalized LOD controls,
I want `get_lod(level)` to move incrementally from the current state,
So that repeated LOD scrubbing is practical and semantics match the API shape.

**Acceptance Criteria:**

**Given** successive `get_lod()` requests that move both down and up
**When** the caller navigates across multiple target levels
**Then** RustMesh adjusts from the current LOD state when possible
**And** the implementation documents any remaining fallback cases explicitly.

### Story 3.3: Monotonic LOD Regression Coverage

As a progressive-mesh maintainer,
I want direct tests around face-count monotonicity and state reuse,
So that incremental LOD behavior is protected from regressions.

**Status:** Implemented. The VDPM test module now covers monotonic face-count behavior for increasing normalized LOD requests, ordered downward/upward scrubbing across repeated `get_lod()` calls, deterministic repeated same-level requests after scrubbing, and exact LIFO replay of stored refine records. Verified with `cargo test --manifest-path RustMesh/Cargo.toml --lib tools::vdpm::tests --quiet` (`16 passed; 0 failed`) and the full RustMesh library suite (`250 passed; 0 failed`).

**Acceptance Criteria:**

**Given** a fixed input mesh and a sequence of normalized level requests
**When** the LOD API is exercised
**Then** face counts change monotonically with the target level
**And** tests verify both downward and upward navigation behavior.

## Epic 4: Normals Semantics and Refresh Policy

Lock the remaining normals gap around behavior, parity expectations, and topology-edit integration rather than chasing raw release-mode speed.

### Story 4.1: Explicit Vertex-Normal Semantics Contract

As a RustMesh/OpenMesh comparison user,
I want vertex-normal behavior documented and optionally exposed in an OpenMesh-compatible mode when needed,
So that checksum mismatches are explainable instead of ambiguous.

**Status:** Implemented. RustMesh now documents and preserves area-weighted vertex normals as the default contract, while `VertexNormalWeighting::FaceAverage` plus `calc_vertex_normal_with_mode()`, `update_vertex_normals_with_mode()`, and `update_normals_with_mode()` expose an OpenMesh-compatible equal-face-weight path. Verified with `cargo test --manifest-path RustMesh/Cargo.toml normals --quiet` (`11 passed; 0 failed`) and the full RustMesh library suite (`250 passed; 0 failed`).

**Acceptance Criteria:**

**Given** the current area-weighted RustMesh implementation and OpenMesh default behavior
**When** the normals API is documented and finalized
**Then** the default contract is explicit
**And** any OpenMesh-compatible compatibility path is either implemented or explicitly deferred with rationale.

### Story 4.2: Topology-Edit Normal Refresh Policy

As an algorithm author,
I want a documented policy for when topology-changing operations refresh normals automatically,
So that callers know whether normals are explicit or maintained eagerly.

**Status:** Implemented. Maintained local edits (`collapse`, `split_edge`, triangle `split_face`, `flip_edge`) now explicitly document an explicit-refresh policy: requested normal arrays are preserved/resized but never recomputed automatically. Rebuild-backed `triangulate_face()` and n-gon `split_face()` preserve requested vertex normals but drop face-normal storage until an explicit refresh. Verified with `cargo test --manifest-path RustMesh/Cargo.toml normals --quiet` (`11 passed; 0 failed`) and the full RustMesh library suite (`250 passed; 0 failed`).

**Acceptance Criteria:**

**Given** topology-changing operations such as split, collapse, remesh, and triangulate
**When** their normal behavior is reviewed
**Then** each operation is classified as explicit-refresh or auto-refresh
**And** the resulting behavior is covered by tests and documentation.

### Story 4.3: Durable Normal Comparison Coverage

As a parity maintainer,
I want the normals comparison harness and fixture coverage to stay trustworthy,
So that future changes do not reintroduce debug/release confusion or semantic drift.

**Acceptance Criteria:**

**Given** the `openmesh_compare_normals` example
**When** it is run in release mode on the fixed sphere workload
**Then** the benchmark output remains documented with semantic notes
**And** regression checks cover the intended face and vertex normal behavior.

## Epic 5: Selective Parity Regression and Comparison Coverage

Convert the highest-value comparison paths into durable regression protection while resisting low-value benchmark sprawl.

### Story 5.1: Promote Decimation Parity into Automated Regression

As a parity maintainer,
I want the current decimation parity baseline protected by direct regression checks,
So that the verified OpenMesh-aligned behavior cannot drift silently.

**Acceptance Criteria:**

**Given** the current `OpenMeshParity` baseline and traced-step expectations
**When** regression checks run in CI or local verification
**Then** the known-good prefix and result summary are validated automatically
**And** the `standard` import mode remains clearly labeled as debug-only contrast.

### Story 5.2: Add High-Value Comparison Coverage Beyond Decimation

As a maintainer deciding where parity work matters,
I want selective comparison harnesses for the next most valuable modules,
So that RustMesh gains coverage where behavior differences would actually hurt users.

**Acceptance Criteria:**

**Given** smoothing, subdivision, dualization, and IO as candidate areas
**When** parity coverage is expanded
**Then** only the modules with clear workflow value gain maintained harnesses
**And** each new harness documents what behavior it protects.

### Story 5.3: Benchmark and Documentation Cleanup

As a maintainer reading parity reports,
I want benchmark anomalies and stale helper notes cleaned up,
So that documentation does not overstate gaps or send future work in the wrong direction.

**Acceptance Criteria:**

**Given** the current `Vec4f_add_compare` anomaly and older stale TODO markers
**When** benchmark/docs cleanup is completed
**Then** benchmark-scoped anomalies are labeled accurately
**And** obsolete status claims are removed from maintained docs.

## Recommended Execution Order

1. Story 3.2: Incremental `get_lod(level)` navigation.
2. Story 5.1: Promote decimation parity into automated regression.
3. Story 5.2: Add high-value comparison coverage beyond decimation.
4. Story 5.3: Benchmark and documentation cleanup.

Why this order:

- Stories 1.1 through 4.2 are now complete on the maintained parity-critical topology path, so the next blocker is durable normals comparison coverage rather than further semantic or topology plumbing
- normals are already fast enough in release mode, and both the default-vs-compatible semantics and refresh policy are now explicit, so the next normals work is durable regression protection
- progressive-mesh LOD is still a real gap, but the current replay-from-original fallback is functional and isolated, so it is less urgent than topology/edit semantics
- broader parity regression is valuable, but it should follow the stabilization of the behavior it is meant to protect

## Exit Criteria

This roadmap is materially complete when:

- dynamic properties are typed, resizable, propagated, and persistable for the supported vertex-property set,
- split primitives on the maintained parity-critical path are local topology edits rather than rebuild-backed fallbacks,
- normalized progressive-mesh LOD navigation no longer depends on replay-from-original for routine upward movement,
- normals behavior is explicit relative to OpenMesh and covered by durable tests,
- and the highest-value OpenMesh parity paths are protected by maintained regression coverage.
