# RustMesh OpenMesh Parity Roadmap

**Updated:** 2026-04-05
**Scope:** Forward-looking RustMesh backlog after the current `rm-opt` verification pass

This roadmap is deliberately shorter than the older version. The previous document mixed historical design sketches, completed tasks, and future work in one place. The goal now is to keep a single accurate backlog that matches the current code.

## Current Baseline

Verified separately in [`RustMesh-OpenMesh-Progress-2026-04-05.md`](RustMesh-OpenMesh-Progress-2026-04-05.md):

- RustMesh library tests are green.
- The default OpenMesh decimation trace baseline matches for the first 10 traced steps.
- Remeshing is runnable and regression-covered.
- HH and EE circulators exist.
- VDPM simplify/refine is present, but normalized LOD selection is still missing.

## Status by Area

| Area | Status | Notes |
|------|--------|-------|
| Core connectivity | Done | half-edge mesh, handles, SoA kernels |
| IO | Done | OBJ, OFF, PLY, STL read/write paths implemented |
| Circulators | Done | vertex/face/edge plus HH and EE circulators |
| Decimation core | Done | quadric decimation and modular constraints implemented |
| Decimation parity | Partial | default `OpenMeshParity` baseline verified; broader regression coverage still thin |
| Smoothing | Done | uniform/tangential paths implemented |
| Subdivision | Done | Loop, Catmull-Clark, sqrt3, midpoint, butterfly |
| Dualization | Done | includes boundary-aware path |
| Analysis | Done | area, volume, curvature, quality, edge-length stats |
| Remeshing | Partial | split/collapse/flip/valence/isotropic remesh present, but topology hardening is still active |
| Progressive mesh / VDPM | Partial | simplify/refine/reset/progress/vertex split present, no normalized `get_lod(level)` |

## Next Priorities

### 1. Strengthen Decimation Parity Regression

- Move from ad hoc example validation toward a repeatable regression that protects the verified 10-step prefix.
- Keep the default parity baseline on `OpenMeshParity`.
- Treat `standard` import mode as a debug contrast path, not a release-quality parity contract.

### 2. Finish Remeshing Hardening

- Replace rebuild-style shortcuts with lower-level topology operations where they materially improve correctness.
- Add a stronger acceptance story around edge-length improvement and topology validity.
- Keep scope disciplined: stabilize the implemented feature before chasing more variants.

### 3. Complete Progressive Mesh LOD

- Add a normalized `get_lod(level)` style API.
- Define behavior for `0.0`, `1.0`, and intermediate requests.
- Add tests that verify monotonic face-count behavior and non-destructive access.

### 4. Expand Comparison Coverage Selectively

- Only add new OpenMesh comparison work where it protects a real workflow:
  - decimation regressions
  - smoothing or IO round-trips if they are part of active interoperability work
  - focused module parity when a behavior difference is blocking users

## Acceptance Targets

The roadmap is considered materially complete when:

- the decimation parity baseline is protected by direct regression coverage,
- remeshing no longer depends on fragile shortcut behavior for its core topology steps,
- progressive mesh exposes a normalized LOD selection API,
- and status remains centralized across `RustMesh/README.md`, the progress doc, and this roadmap.
