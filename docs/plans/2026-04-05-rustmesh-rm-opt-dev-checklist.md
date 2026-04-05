# RustMesh rm-opt Development Checklist Implementation Plan

**Goal:** Close the last OpenMesh decimation parity gap in `rm-opt`, then finish the highest-value follow-up work that is still missing or only partially hardened.

**Architecture:** Treat `docs/RustMesh-OpenMesh-Progress-2026-04-05.md` as the source of truth for the current `rm-opt` branch. First eliminate the remaining step-10 trace divergence by making per-face quadric construction match OpenMesh bitwise on the known bad faces. After parity is closed, harden the existing remeshing and circulator implementations, finish the missing VDPM LOD API, and sync the roadmap docs to actual code status.

**Tech Stack:** Rust, Cargo, RustMesh half-edge core, OpenMesh parity example tooling, Markdown docs

---

## Current Status

- Completed: decimation parity closure
  - `openmesh_compare_decimation_trace` now defaults to `OpenMeshParity`
  - default parity trace matches OpenMesh through the first 10 steps
  - face-quadric regressions for faces `192` and `196` are covered by unit tests
- Completed: remeshing stabilization first pass
  - `split_long_edges()` no longer performs in-place placeholder face deletion/recreation
  - `test_isotropic_remesh_sphere` no longer hangs
  - remeshing test suite is passing again
- Remaining high-priority work:
  - add stronger decimation trace-prefix regression coverage
  - continue remeshing hardening beyond the split-path fix
  - finish the VDPM LOD API

---

### Task 1: Reproduce And Pin The Remaining Step-10 Parity Gap

**Files:**
- Reference: `docs/RustMesh-OpenMesh-Progress-2026-04-05.md`
- Modify: `RustMesh/examples/openmesh_compare_decimation_trace.rs`
- Modify: `RustMesh/src/Tools/decimation.rs`
- Test: `cargo run --release --example openmesh_compare_decimation_trace --quiet -- 10`

**Step 1: Re-run the known failing baseline**

Run:

```bash
cargo run --release --example openmesh_compare_decimation_trace --quiet -- 10
```

Expected: output still says step 10 diverges with RustMesh `115->116` vs OpenMesh `116->115`.

**Step 2: Add focused operand-level debug output for face 192 and face 196**

Add a gated debug helper that can print the intermediate values used by `face_quadric_from_points()`:

```rust
if debug_face_operands_enabled(face_idx) {
    println!(
        "FACE_OPS face={} e1=({:.17e},{:.17e},{:.17e}) e2=({:.17e},{:.17e},{:.17e}) n=({:.17e},{:.17e},{:.17e}) area={:.17e} d={:.17e}",
        face_idx, e1x, e1y, e1z, e2x, e2y, e2z, nx, ny, nz, area, d
    );
}
```

**Step 3: Re-run exact dumps with the new focused instrumentation**

Run:

```bash
env RUSTMESH_TRACE_DUMP_VERTICES=115,116,117,118,119,120 \
    RUSTMESH_TRACE_DUMP_EXACT=1 \
    RUSTMESH_TRACE_DUMP_FACE_QUADRICS=1 \
  cargo run --release --example openmesh_compare_decimation_trace --quiet -- 0
```

Expected: output includes stable operand traces for face `192` and `196`.

**Step 4: Record the exact mismatch before changing math**

Copy the observed face ids, operand ordering, and current bit patterns into a nearby code comment or a temporary test fixture block so the next step has a fixed target.

**Step 5: Commit**

```bash
git add RustMesh/examples/openmesh_compare_decimation_trace.rs RustMesh/src/Tools/decimation.rs
git commit -m "Add focused quadric parity instrumentation"
```

### Task 2: Make Per-Face Quadric Construction Match OpenMesh

**Files:**
- Modify: `RustMesh/src/Tools/decimation.rs`
- Modify: `RustMesh/examples/openmesh_compare_decimation_trace.rs`
- Test: `cargo test --lib tools::decimation::tests --quiet`
- Test: `cargo run --release --example openmesh_compare_decimation_trace --quiet -- 10`

**Step 1: Write a failing regression test for the known bad face construction path**

Add a focused unit test around the face-quadric helper using captured points or captured expected bit patterns:

```rust
#[test]
fn test_face_quadric_matches_openmesh_bits_for_known_face() {
    let (p0, p1, p2) = known_face_192_points();
    let q = face_quadric_from_points(p0, p1, p2).unwrap();
    assert_eq!(quadric_bits(q), KNOWN_FACE_192_OPENMESH_BITS);
}
```

**Step 2: Run the new test to confirm it fails**

Run:

```bash
cargo test --lib tools::decimation::tests::test_face_quadric_matches_openmesh_bits_for_known_face -- --exact
```

Expected: FAIL because current order-of-operations still differs from OpenMesh.

**Step 3: Change only the numeric construction path**

Adjust the exact order of operations in `face_quadric_from_points()`:
- edge subtraction order
- cross product component order
- normalization and `sqrt()` timing
- `d = -(p0 · n)` accumulation order
- the point at which `area` scales the quadric coefficients

Keep the public decimation behavior unchanged outside this helper.

**Step 4: Re-run the focused test and the trace baseline**

Run:

```bash
cargo test --lib tools::decimation::tests::test_face_quadric_matches_openmesh_bits_for_known_face -- --exact
cargo run --release --example openmesh_compare_decimation_trace --quiet -- 10
```

Expected:
- focused face-quadric regression passes
- trace reports no remaining divergence through step 10

**Step 5: Commit**

```bash
git add RustMesh/src/Tools/decimation.rs RustMesh/examples/openmesh_compare_decimation_trace.rs
git commit -m "Align face quadric construction with OpenMesh"
```

### Task 3: Lock The Parity Result With Regression Coverage

**Files:**
- Modify: `RustMesh/src/Tools/decimation.rs`
- Modify: `RustMesh/examples/openmesh_compare_decimation_trace.rs`
- Test: `cargo test --lib tools::decimation::tests --quiet`

**Step 1: Add a regression test for the step-10 prefix**

Use the same synthetic trace input path already built by the example and assert the prefix length:

```rust
#[test]
fn test_decimation_trace_matches_openmesh_prefix_10() {
    let (rust_trace, openmesh_trace) = run_known_trace_pair(10);
    assert_eq!(matching_prefix(&rust_trace, &openmesh_trace), 10);
}
```

**Step 2: Run the focused regression**

Run:

```bash
cargo test --lib tools::decimation::tests::test_decimation_trace_matches_openmesh_prefix_10 -- --exact
```

Expected: PASS.

**Step 3: Run the full decimation suite**

Run:

```bash
cargo test --lib tools::decimation::tests --quiet
```

Expected: PASS with no new failures.

**Step 4: Remove temporary debug noise if it is no longer needed**

Keep only reusable debug flags that are still valuable for future parity work.

**Step 5: Commit**

```bash
git add RustMesh/src/Tools/decimation.rs RustMesh/examples/openmesh_compare_decimation_trace.rs
git commit -m "Add decimation parity regression coverage"
```

### Task 4: Harden Remeshing From Prototype To Verified Feature

**Files:**
- Modify: `RustMesh/src/Tools/remeshing.rs`
- Modify: `RustMesh/src/Core/connectivity.rs`
- Modify: `RustMesh/src/lib.rs`
- Test: `cargo test --lib tools::remeshing::tests --quiet`

**Step 1: Write failing tests for the current weak spots**

Add tests that cover:
- interior edge split preserves manifold topology
- boundary edge split preserves boundary shape
- `edge_length_histogram()` returns stable buckets
- `isotropic_remesh()` improves edge-length spread without invalid topology

Example test shape:

```rust
#[test]
fn test_split_long_edges_preserves_active_faces() {
    let mut mesh = make_two_triangle_patch();
    let before = active_face_count(&mesh);
    let split = split_long_edges(&mut mesh, 0.5);
    assert_eq!(split, 1);
    assert!(active_face_count(&mesh) >= before);
    assert!(mesh.check_topology().is_ok());
}
```

**Step 2: Run the new remeshing tests to confirm current failures**

Run:

```bash
cargo test --lib tools::remeshing::tests --quiet
```

Expected: at least one test fails around the placeholder split/collapse behavior.

**Step 3: Replace placeholder topology edits with proper half-edge operations**

Specifically:
- stop relying on the current "delete adjacent faces and recreate triangles" shortcut in `split_long_edges()`
- prefer a real split primitive or add one in `connectivity.rs`
- add the missing `edge_length_histogram()` API
- keep `flip_edges_for_valence()` and `isotropic_remesh()` working on active elements only

**Step 4: Re-run the full remeshing suite**

Run:

```bash
cargo test --lib tools::remeshing::tests --quiet
```

Expected: PASS.

**Step 5: Commit**

```bash
git add RustMesh/src/Tools/remeshing.rs RustMesh/src/Core/connectivity.rs RustMesh/src/lib.rs
git commit -m "Harden remeshing topology operations"
```

### Task 5: Finish The Missing VDPM LOD API

**Files:**
- Modify: `RustMesh/src/Tools/vdpm.rs`
- Modify: `RustMesh/src/lib.rs`
- Test: `cargo test --lib tools::vdpm::tests --quiet`

**Step 1: Write a failing API-level test for `get_lod(level)`**

Add a focused test that simplifies a progressive mesh and then requests multiple LOD levels:

```rust
#[test]
fn test_get_lod_returns_monotonic_face_counts() {
    let mesh = generate_sphere(1.0, 12, 12);
    let mut pm = create_progressive_mesh(&mesh);
    simplify(&mut pm, mesh.n_faces() / 2);

    let lod0 = get_lod(&pm, 0.0);
    let lod1 = get_lod(&pm, 0.5);
    let lod2 = get_lod(&pm, 1.0);

    assert!(lod0.n_faces() >= lod1.n_faces());
    assert!(lod1.n_faces() >= lod2.n_faces());
}
```

**Step 2: Run the focused test to confirm it fails**

Run:

```bash
cargo test --lib tools::vdpm::tests::test_get_lod_returns_monotonic_face_counts -- --exact
```

Expected: FAIL because `get_lod` does not exist yet.

**Step 3: Implement the smallest usable `get_lod(level)` API**

Requirements:
- accept a normalized `level` in `[0.0, 1.0]`
- map it to a collapse/refine target
- return a mesh view or mesh clone without corrupting `ProgressiveMesh`
- keep simplify/refine bookkeeping consistent

**Step 4: Run the VDPM suite**

Run:

```bash
cargo test --lib tools::vdpm::tests --quiet
```

Expected: PASS.

**Step 5: Commit**

```bash
git add RustMesh/src/Tools/vdpm.rs RustMesh/src/lib.rs
git commit -m "Add progressive mesh LOD API"
```

### Task 6: Sync Roadmap Docs To Actual Code State

**Files:**
- Modify: `docs/RustMesh-OpenMesh-Parity-Roadmap.md`
- Modify: `docs/RustMesh-OpenMesh-Progress-2026-04-05.md`
- Modify: `docs/index.md`
- Modify: `docs/project-overview.md`

**Step 1: Update story status markers that are now stale**

At minimum, reconcile:
- remeshing items that now exist in `RustMesh/src/Tools/remeshing.rs`
- HH/EE circulator items that now exist in `RustMesh/src/Utils/circulators.rs`
- the latest decimation parity status after Tasks 1-3

**Step 2: Clearly mark what is still incomplete**

Keep the backlog honest:
- remeshing hardening or acceptance gaps
- `get_lod(level)` if Task 5 is not finished yet
- any remaining OpenMesh comparison work

**Step 3: Run a quick documentation consistency pass**

Check that `docs/index.md` and `docs/project-overview.md` agree on RustMesh tool maturity and active work.

**Step 4: Commit**

```bash
git add docs/RustMesh-OpenMesh-Parity-Roadmap.md docs/RustMesh-OpenMesh-Progress-2026-04-05.md docs/index.md docs/project-overview.md
git commit -m "Sync RustMesh roadmap with rm-opt status"
```

### Priority Order

1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 5
6. Task 6

### Exit Criteria

- `openmesh_compare_decimation_trace` has no divergence through step 10
- `cargo test --lib tools::decimation::tests --quiet` passes
- remeshing tests cover split, collapse, flip, histogram, and isotropic remesh behavior
- `get_lod(level)` exists and is covered by tests
- roadmap docs reflect actual code state instead of stale `TODO` markers
