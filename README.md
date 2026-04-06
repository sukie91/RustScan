# RustScan

<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.75+-dea584?style=for-the-badge&logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Status-Active%20Development-blue?style=for-the-badge" alt="Status">
</p>

RustScan is a Rust workspace for 3D reconstruction tooling: visual SLAM, Gaussian splatting, mesh extraction, mesh processing, and visualization.

This README is intentionally brief. Current status lives in a small set of canonical documents so the repo does not keep drifting copies of the same state.

## Workspace

- `rustscan-types`: shared data structures used across crates.
- `RustSLAM`: visual SLAM, sparse mapping, loop closing, video IO, and mesh extraction.
- `RustGS`: Gaussian splatting training and rendering.
- `RustMesh`: mesh connectivity, IO, processing algorithms, OpenMesh comparison tooling.
- `RustViewer`: visualization and inspection UI.

## Verified Snapshot

As verified in the `rm-opt` worktree on 2026-04-05:

- RustMesh library tests: `214 passed; 0 failed`
- RustMesh decimation tests: `12 passed; 0 failed`
- RustMesh remeshing tests: `7 passed; 0 failed`
- RustMesh VDPM tests: `7 passed; 0 failed`
- `openmesh_compare_decimation_trace` matches OpenMesh for the first 10 traced steps under the default `OpenMeshParity` import mode
- RustSLAM library tests are not fully green in this worktree: `261 passed; 2 failed`

## Documentation

- Workspace overview: [`docs/index.md`](./docs/index.md)
- Project summary: [`docs/project-overview.md`](./docs/project-overview.md)
- RustMesh crate overview: [`RustMesh/README.md`](./RustMesh/README.md)
- RustMesh `rm-opt` status: [`docs/RustMesh-OpenMesh-Progress-2026-04-05.md`](./docs/RustMesh-OpenMesh-Progress-2026-04-05.md)
- Forward roadmap: [`ROADMAP.md`](./ROADMAP.md)

## Getting Started

```bash
# Build the workspace
cargo build --release

# RustMesh
cargo test --manifest-path RustMesh/Cargo.toml --lib

# RustSLAM
cargo test --manifest-path RustSLAM/Cargo.toml --lib
```

## Notes

- The compatibility entry points under `docs/README.md`, `docs/RustMesh-README.md`, and `docs/ROADMAP.md` are intentionally thin wrappers around the canonical docs above.
- For branch-specific OpenMesh parity work, use the progress and roadmap docs instead of older planning artifacts.
