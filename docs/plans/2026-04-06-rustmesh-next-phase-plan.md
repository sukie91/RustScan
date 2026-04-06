# RustMesh Next-Phase Plan (rm-opt)

**Updated:** 2026-04-06  
**Worktree:** `rm-opt`  
**Goal:** Consolidate current delivery status and the immediate execution plan.

## Current Progress Snapshot

### Completed Stories

- `Epic 1 / Story 1.1`: Dynamic-property propagation is implemented on the maintained local-edit path (`collapse`, `split_edge`, triangle `split_face`) with deterministic behavior and regression coverage.
- `Epic 2 / Story 2.1`: `split_edge()` is local on the maintained boundary/interior triangle path.
- `Epic 2 / Story 2.2`: Triangle `split_face()` is local with stable connectivity semantics.
- `Epic 2 / Story 2.3`: Remeshing acceptance is hardened on shared split/collapse/flip primitives.
- `Epic 4 / Story 4.1`: Vertex-normal semantics are explicit, with default area-weighted mode and OpenMesh-compatible face-average mode.
- `Epic 4 / Story 4.2`: Topology-edit normal refresh policy is explicit and regression-covered.
- `Epic 4 / Story 4.3`: Normals comparison coverage is durable and reproducible, with explicit benchmark contract metadata plus regression checks for default and compatible paths.
- `Epic 3 / Story 3.1`: Refine and vertex-split replay records are exact enough to restore deterministic pre-collapse states without re-running simplification from `original`.
- `Epic 3 / Story 3.3`: Monotonic and bidirectional LOD regression coverage now protects repeated `get_lod()` sweeps plus exact refine-record state reuse.

### Verified Baseline

- `cargo test --manifest-path RustMesh/Cargo.toml --lib --quiet`: `250 passed; 0 failed`
- `cargo test --manifest-path RustMesh/Cargo.toml normals --quiet`: `11 passed; 0 failed`
- `cargo test --manifest-path RustMesh/Cargo.toml --lib tools::vdpm::tests --quiet`: `16 passed; 0 failed`
- `cargo test --manifest-path RustMesh/Cargo.toml propagates --quiet`: `3 passed; 0 failed`

## Next Development Plan

## Phase A: Epic 3 / Story 3.2 (Highest Priority)

**Objective:** Remove replay-from-original behavior for routine upward LOD navigation.

Planned work:

1. Track current LOD state explicitly enough to decide whether to simplify further or replay exact records upward.
2. Move `get_lod(level)` from unconditional `reset(self)` to incremental navigation from current state where possible.
3. Document any remaining fallback paths explicitly if exact incremental motion is still not available for some edge cases.

Exit criteria:

- Upward/downward LOD changes are incremental on maintained paths.
- Regression suite validates monotonic and bidirectional behavior.
- Callers no longer pay a full replay-from-original cost for routine scrubbing.

## Phase B: Epic 5 (Selective Parity Hardening)

**Objective:** Expand parity protection where it delivers workflow value.

Planned sequence:

1. Story 5.1: Promote decimation parity checks into durable automated regression.
2. Story 5.2: Add selective comparison coverage beyond decimation.
3. Story 5.3: Benchmark/documentation cleanup (`Vec4f_add_compare` and stale notes).

## Guardrails for All Remaining Work

- Keep `cargo test --manifest-path RustMesh/Cargo.toml --lib --quiet` green after each story.
- Preserve SoA-oriented storage model and maintained local-edit topology behavior.
- Treat non-triangle rebuild-backed fallback behavior as explicit scope, not accidental parity.
