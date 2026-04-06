# RustMesh vs OpenMesh Gap Analysis

**Date:** 2026-04-06  
**Branch:** `worktree-rm-opt`  
**Goal:** Achieve full OpenMesh feature parity and surpass it in runtime performance.

---

## Benchmark Results (release mode, 2026-04-06)

Benchmarks run via `cargo run --release --example openmesh_compare_benchmark` and `openmesh_compare_vector_benchmark` after the Sprint 1 kernel update.

| Test | RustMesh | OpenMesh | Ratio |
|------|----------|----------|-------|
| Build 1000 tetrahedra | 755 µs | 1.78 ms | **RustMesh 2.35× faster** ✅ |
| Add 1000 triangles ×1000 | 99.7 ms | 115.9 ms | **RustMesh 1.16× faster** ✅ |
| Triangle area ×2M | 16.7 ms | 126.0 ms | **RustMesh 7.5× faster** ✅ |
| Vec3f arithmetic (add/dot/cross/norm) | ~0.7–3.1 ns | ~0.7–2.4 ns | **Near parity / benchmark noise** ✅ |
| Vec4f_add_compare | ~14–15 ns | ~0.8 ns | **Still anomalous, but benchmark-scoped** ⚠️ |

---

## Performance Gaps

### P1 — `add_face` hot path lookup cost (DONE 2026-04-06)

**File:** `RustMesh/src/Core/soa_kernel.rs`, method `add_edge`

Baseline: each `add_face(a, b, c)` call triggered 3 global HashMap lookups of `(min_v, max_v) → HalfedgeHandle`. The active `RustMesh` mesh type uses `SoAKernel`, not `AttribSoAKernel`, so the original document target file was wrong.

Implemented fix:

- replaced the global `HashMap<(u32, u32), HalfedgeHandle>` with a per-vertex sorted adjacency index in `SoAKernel`
- `add_edge`, `find_halfedge`, `edge_exists`, and `delete_edge` now use bucketed binary search instead of hashing
- `delete_edge` no longer does a whole-map `retain`

Verified result:

- previous benchmark: `RustMesh 156 ms` vs `OpenMesh 111 ms`
- current benchmark: `RustMesh 99.7 ms` vs `OpenMesh 115.9 ms`

### P2 — Vec4f_add_compare anomaly is not a confirmed RustMesh library gap

**File:** `RustMesh/examples/openmesh_compare_vector_benchmark.rs`

Investigation result:

- the benchmark does not exercise RustMesh library code; it exercises `glam::Vec4`
- the original document target `geometry.rs` was therefore misleading
- replacing `glam::Vec4 ==` with an explicit four-component comparison in the benchmark did not materially change the result; both paths stay around `14–15 ns`
- other Vec4 operations remain close to OpenMesh, so this is isolated to the current compare benchmark shape / compiler codegen interaction

Current disposition:

- treat this as a benchmark-design / codegen investigation, not a RustMesh feature-parity blocker
- if this benchmark matters, redesign the harness to prevent compiler-specific constant-folding / loop-shape artifacts before drawing library conclusions

---

## Functional Gaps

### G1 — Dynamic property system is broken (HIGH)

**Files:** `RustMesh/src/Core/attrib_soa_kernel.rs`

**Current state:**
- Single global `HashMap<u32, DynamicProperty>` — vertex, edge, face, halfedge properties share one store with no entity-type distinction.
- `get_property<T>()` only works for `f32`; the generic parameter is silently ignored for other types.
- Properties do not auto-resize when vertices/edges/faces are added.
- Properties are not propagated during topology changes (`collapse`, `split`, `flip`).
- Dynamic properties are not serialized to PLY/OBJ.

**Expected OpenMesh API:**
```rust
// Per-entity type-safe property handles
let vprop: VPropHandle<f32> = mesh.add_vertex_property("quality", 0.0_f32);
let eprop: EPropHandle<Vec3> = mesh.add_edge_property("flow", Vec3::ZERO);

mesh.set_vertex_property(vprop, vh, 0.95);
let v = mesh.vertex_property(vprop, vh);   // → 0.95_f32

// Automatic propagation on collapse: interpolate between v0 and v1
mesh.collapse(heh);  // surviving vertex gets lerp of quality values
```

**Required changes:**
1. Split `dynamic_props` into four typed stores: `vertex_props`, `edge_props`, `face_props`, `halfedge_props`.
2. Add typed handles: `VPropHandle<T>`, `EPropHandle<T>`, `FPropHandle<T>`, `HPropHandle<T>`.
3. Fix `get_property<T>` to dispatch on `T` for all `PropValue` variants.
4. Auto-resize all per-entity stores in `add_vertex`, `add_edge`, `add_face`.
5. Add `copy_property` and interpolated propagation hooks to `collapse` and `split`.
6. Serialize dynamic `Vec3` / `f32` / `i32` vertex properties to PLY.

---

### G2 — Normal update API baseline is now implemented; remaining work is optimization/integration

**Files:** `RustMesh/src/Core/connectivity.rs`

Current state after 2026-04-06 implementation:

- `calc_face_normal(fh) -> Vec3`
- `calc_vertex_normal(vh) -> Vec3`
- `update_face_normals()`
- `update_vertex_normals()`
- `update_normals()`

The API family now exists and is covered by focused correctness tests. The remaining gap versus the original plan is that the implementation is still scalar and topology-changing algorithms do not yet consistently refresh normals automatically.

**Expected API:**
```rust
mesh.update_face_normals();       // compute cross-product per face, store in face normal array
mesh.update_vertex_normals();     // area-weighted average from incident face normals
mesh.update_normals();            // both of the above
mesh.calc_face_normal(fh) -> Vec3;
mesh.calc_vertex_normal(vh) -> Vec3;
```

**Remaining work:**
1. Batch/SIMD-optimize face normal updates on the SoA storage layout.
2. Add a broader reference/benchmark example for `update_normals`.
3. Decide which topology-changing algorithms should call `update_normals()` automatically versus leaving refresh explicit.

---

### G3 — Public split primitives now exist; remaining gap is implementation depth

**Files:** `RustMesh/src/Core/connectivity.rs`, `RustMesh/src/Tools/remeshing.rs`

Current state after 2026-04-06 implementation:

- `RustMesh::split_edge(eh, pt)` now exists
- `RustMesh::split_face(fh, pt)` now exists
- the baseline API is covered by regression tests for boundary-edge and triangle-face cases

Remaining gap:

- the current implementation uses a controlled rebuild via `rebuild_preserving_vertex_indices()` rather than true local half-edge surgery
- this is enough to unify call sites and stabilize semantics, but it is not yet the final OpenMesh-grade topology-edit path

**Expected API on `RustMesh`:**
```rust
// Split edge at midpoint, return new vertex handle
mesh.split_edge(eh, midpoint) -> VertexHandle;

// Split face at centroid (or given point), return new vertex handle
mesh.split_face(fh, point) -> VertexHandle;
```

These primitives now exist and unblock higher-level call-site cleanup.

---

### G4 — Remeshing split path now uses public primitives; underlying split implementation still rebuilds internally

**File:** `RustMesh/src/Tools/remeshing.rs`, `split_long_edges()` (line 171)

Current state after 2026-04-06 implementation:

- `split_long_edges()` now calls `mesh.split_edge()` instead of maintaining its own whole-mesh rebuild path
- this removes duplicate split logic from remeshing and centralizes behavior in the core mesh API

Remaining gap:

- `mesh.split_edge()` itself still uses controlled rebuild semantics internally, so the performance/property-propagation concerns are reduced but not fully eliminated

**Next fix:** replace the internal rebuild-based split primitive with local topology editing once the public API shape is settled.

---

### G5 — Progressive Mesh normalized LOD API added; upward replay is still not incremental (MEDIUM)

**File:** `RustMesh/src/Tools/vdpm.rs`

Current state after 2026-04-06 implementation:

- `ProgressiveMesh::get_lod(&mut self, level: f32) -> &RustMesh` now exists
- `level` is clamped to `[0.0, 1.0]`
- `0.0` requests the maximally simplified mesh reachable by current legality checks
- `0.5` maps to an approximately half-face budget
- `1.0` restores the original mesh
- focused tests cover below-range, midpoint, and above-range behavior

**Expected API:**
```rust
pm.get_lod(0.0)  // fully simplified (minimum triangles)
pm.get_lod(0.5)  // half-way
pm.get_lod(1.0)  // original mesh
```

Remaining gap:

- `get_lod()` currently replays from `original` and re-runs simplification to the target face budget
- this is intentional because `refine()` / `vertex_split()` are still approximate and do not yet provide exact bidirectional LOD navigation

**Next fix:** harden incremental `refine()` / `vertex_split()` so upward LOD changes can avoid a full replay from the original mesh.

---

### G6 — Dynamic properties not persisted in IO (MEDIUM)

**Files:** `RustMesh/src/Core/io/ply.rs`, `RustMesh/src/Core/io/obj.rs`

Only fixed attributes (positions, built-in normals, colors, texcoords) are written. User-added properties via `add_vertex_property` are silently dropped on save.

**Fix:** In `write_ply`, enumerate `vertex_props` and emit each as a PLY property element. Map `f32` → `float`, `Vec3` → `float[3]`, `i32` → `int`.

---

### G7 — `triangulate_face()` missing (LOW)

OpenMesh exposes a fan-triangulation of n-gons as a public method. RustMesh only supports triangle faces at add time. No n-gon → triangle conversion exists.

---

## Prioritized Task List for Codex

Each task is self-contained. Tackle in order; G3 unblocks G4.

### Sprint 1 — Performance critical path

| ID | Task | File(s) | Acceptance |
|----|------|---------|------------|
| T1 | Replace HashMap edge lookup with sorted adjacency | `soa_kernel.rs` | Completed: Case 3 improved from `156 ms` to `99.7 ms` and now beats OpenMesh on the current harness |
| T2 | Investigate Vec4f_add_compare regression | benchmark only | Completed investigation: not currently attributable to RustMesh library code; benchmark redesign is the next step |

### Sprint 2 — Normal update API

| ID | Task | File(s) | Acceptance |
|----|------|---------|------------|
| T3 | Implement `update_face_normals()` batch API | `connectivity.rs` | Completed: API and correctness tests exist; SIMD specialization still pending |
| T4 | Implement `update_vertex_normals()` area-weighted | `connectivity.rs` | Completed: area-weighted API and regression tests exist; broader reference benchmark still pending |
| T5 | Add `update_normals()` convenience wrapper | `connectivity.rs` | Completed: wrapper exists; automatic algorithm refresh policy still pending |
| T6 | Benchmark `update_normals` vs OpenMesh equivalent | new example | Completed baseline: `openmesh_compare_normals.rs` exists; on a deterministic `64x64` sphere, RustMesh is currently slower than OpenMesh on face / vertex / combined normal recomputation |

### Sprint 3 — Split/collapse primitives + fix remeshing

| ID | Task | File(s) | Acceptance |
|----|------|---------|------------|
| T7 | Expose `split_edge(eh, pt) -> VertexHandle` as public | `connectivity.rs` | Completed baseline: public API exists and boundary-edge regression test verifies `V+1, E+2, F+1` |
| T8 | Expose `split_face(fh, pt) -> VertexHandle` as public | `connectivity.rs` | Completed baseline: public API exists and triangle-face regression test verifies `V+1, F+2` |
| T9 | Rewrite `split_long_edges` using `split_edge` | `remeshing.rs` | Completed baseline: remeshing tests pass through the public split primitive |
| T10 | Rewrite `collapse_short_edges` using `collapse` | `remeshing.rs` | Already aligned: current implementation already calls `collapse()` directly |

### Sprint 4 — Property system overhaul

| ID | Task | File(s) | Acceptance |
|----|------|---------|------------|
| T11 | Split into four typed prop stores (vertex/edge/face/halfedge) | `attrib_soa_kernel.rs` | Existing tests still pass |
| T12 | Add typed handles `VPropHandle<T>`, etc. | new `Core/properties.rs` | Type-safe get/set for f32, Vec3, i32 |
| T13 | Fix `get_property<T>` dispatch for all `PropValue` | `attrib_soa_kernel.rs` | Test: get Vec3 property returns correct value |
| T14 | Auto-resize props on `add_vertex` / `add_edge` / `add_face` | `attrib_soa_kernel.rs` | No out-of-bounds after adding entities |
| T15 | Propagate properties on `collapse` (linear interp) | `connectivity.rs` | Collapse test: quality prop interpolated correctly |
| T16 | Propagate properties on `split_edge` | `connectivity.rs` | Split test: new vertex inherits edge midpoint props |
| T17 | Serialize vertex props to PLY | `io/ply.rs` | Round-trip: write + read preserves custom f32 prop |

### Sprint 5 — Remaining gaps

| ID | Task | File(s) | Acceptance |
|----|------|---------|------------|
| T18 | Add `get_lod(level: f32)` to ProgressiveMesh | `vdpm.rs` | Completed baseline: normalized API exists and focused tests verify clamp + midpoint semantics |
| T19 | Add `triangulate_face(fh)` for n-gon support | `connectivity.rs` | Quad splits into 2 triangles correctly |

---

## Notes for Codex

- **Build command:** `cd RustMesh && cargo build --release`
- **Test command:** `cd RustMesh && cargo test`
- **Benchmark:** `cargo run --release --example openmesh_compare_benchmark`
- **Vector benchmark:** `cargo run --release --example openmesh_compare_vector_benchmark`
- **Decimation parity check:** `cargo run --release --example openmesh_compare_decimation_trace -- 10`
- OpenMesh source mirror is at `Mirror/OpenMesh-11.0.0/` in the repo root — use it as ground truth for API shape and algorithm behavior.
- All changes must keep `cargo test` green (221 tests currently passing).
- The SoA layout (`x[]`, `y[]`, `z[]` separate arrays) is intentional for SIMD — do not consolidate into `Vec<Vec3>`.
