# RustScan Source Tree Analysis

**Updated:** 2026-04-06
**Purpose:** Structural snapshot of the current workspace

This document describes the current tree layout. It is not the authoritative status document for feature completeness.

## Workspace Snapshot

| Path | Role | Rust Source Files | Examples |
|------|------|-------------------|----------|
| `RustMesh/` | mesh processing crate | `28` | `36` |
| `RustSLAM/` | visual SLAM crate | `70` | `5` |
| `RustGS/` | Gaussian splatting crate | `26` | `1` |
| `RustViewer/` | visualization crate | `15` | `0` |
| `rustscan-types/` | shared types crate | `4` | `0` |

Total Rust files in this snapshot: `195`

## RustMesh Layout

```text
RustMesh/
  src/
    Core/
      attrib_soa_kernel.rs
      connectivity.rs
      geometry.rs
      handles.rs
      io.rs
      io/
        obj.rs
        off.rs
        ply.rs
        stl.rs
      items.rs
      soa_kernel.rs
    Tools/
      analysis.rs
      decimation.rs
      decimation_modules.rs
      dualizer.rs
      hole_filling.rs
      mesh_repair.rs
      remeshing.rs
      smoother.rs
      subdivision.rs
      vdpm.rs
    Utils/
      circulators.rs
      performance.rs
      quadric.rs
      smart_ranges.rs
      status.rs
      test_data.rs
    lib.rs
  examples/
  benches/
```

### RustMesh Notes

- IO lives under `src/Core/io/` and exports read/write helpers for OBJ, OFF, PLY, and STL.
- Comparison tooling for OpenMesh lives primarily in `examples/openmesh_compare_*`.
- Remeshing and progressive mesh work are both present in the library surface, but their remaining backlog is tracked separately in the RustMesh roadmap.

## RustSLAM Layout

```text
RustSLAM/
  src/
    cli/
    config/
    core/
    depth/
    features/
    fusion/
    io/
    loop_closing/
    mapping/
    optimizer/
    pipeline/
    tracker/
    viewer/
```

### RustSLAM Notes

- The crate spans SLAM, video IO, dense reconstruction support, and mesh extraction.
- This document is structural only; current RustSLAM test status is intentionally deferred to crate-specific verification.

## Documentation Notes

- Canonical status docs: `README.md`, `docs/index.md`, `RustMesh/README.md`, `docs/RustMesh-OpenMesh-Progress-2026-04-05.md`, `ROADMAP.md`
- Compatibility redirects: `docs/README.md`, `docs/RustMesh-README.md`, `docs/ROADMAP.md`
