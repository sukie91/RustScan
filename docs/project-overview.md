# RustScan Project Overview

**Updated:** 2026-04-05
**Scope:** Workspace summary for the current repository state

## Summary

RustScan is a Rust workspace for end-to-end 3D reconstruction workflows. The crates are intentionally split so SLAM, Gaussian training, mesh processing, and visualization can evolve independently while still sharing a common data path.

## Crates

| Crate | Role | Current Status |
|-------|------|----------------|
| `rustscan-types` | Shared data model | Small, stable support crate |
| `RustSLAM` | Visual SLAM, video IO, sparse mapping, loop closing, mesh extraction | Active development; current `rm-opt` worktree still has 2 failing lib tests |
| `RustGS` | Gaussian splatting training and rendering | Integrated workspace component |
| `RustMesh` | Mesh IO, connectivity, processing algorithms, OpenMesh comparison tooling | Advanced; active OpenMesh parity and topology hardening work |
| `RustViewer` | Visualization and inspection UI | Integrated workspace component |

## Pipeline

```text
Video / dataset input
  -> RustSLAM
  -> RustGS
  -> RustMesh
  -> export / visualization
```

Not every workflow uses every crate, but this is the primary intended data flow.

## Current Verification Snapshot

These checks were run against the current workspace on 2026-04-05:

| Check | Result |
|-------|--------|
| `cargo test --manifest-path RustMesh/Cargo.toml --lib --quiet` | `214 passed; 0 failed` |
| `cargo test --manifest-path RustMesh/Cargo.toml --lib tools::remeshing::tests --quiet` | `7 passed; 0 failed` |
| `cargo test --manifest-path RustMesh/Cargo.toml --lib tools::vdpm::tests --quiet` | `7 passed; 0 failed` |
| `env RUSTFLAGS=-Awarnings cargo run --manifest-path RustMesh/Cargo.toml --release --example openmesh_compare_decimation_trace --quiet -- 10` | RustMesh matches OpenMesh for the first 10 traced decimation steps under `OpenMeshParity` |
| `cargo test --manifest-path RustSLAM/Cargo.toml --lib --quiet` | `261 passed; 2 failed` |

## RustMesh Status

RustMesh is the area with the most actively maintained documentation in this worktree.

- OBJ, OFF, PLY, and STL read/write paths are implemented.
- Core half-edge connectivity, smart handles, and circulators are implemented.
- Decimation has an OpenMesh comparison harness with a verified 10-step parity prefix on the default baseline.
- Remeshing is implemented and testable, but still undergoing topology hardening.
- Progressive mesh simplify/refine is implemented, but a normalized `get_lod(level)` API is still missing.

Use these documents for RustMesh work:

- [RustMesh README](../RustMesh/README.md)
- [RustMesh OpenMesh Progress](RustMesh-OpenMesh-Progress-2026-04-05.md)
- [RustMesh OpenMesh Roadmap](RustMesh-OpenMesh-Parity-Roadmap.md)

## Documentation Layout

| Document | Use |
|----------|-----|
| [Repository README](../README.md) | Short repository entry point |
| [Documentation Index](index.md) | Canonical doc map |
| [Development Guide](DEVELOPMENT.md) | Build and test workflow |
| [Architecture](ARCHITECTURE.md) | System design and integration |
| [API Reference](API.md) | Public API surface |
| [Roadmap](../ROADMAP.md) | Workspace-level future work |

## Guidance

- Treat `docs/index.md` as the navigation root.
- Treat `README.md`, `RustMesh/README.md`, `docs/RustMesh-OpenMesh-Progress-2026-04-05.md`, and `ROADMAP.md` as the maintained status documents.
- Treat scan artifacts and redirect files as secondary references only.
