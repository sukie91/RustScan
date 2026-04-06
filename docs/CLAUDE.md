# CLAUDE.md

This file provides operational guidance for AI coding sessions in this repository.

## Canonical Status Docs

Do not infer current project status from this file. Use these documents instead:

- [`../README.md`](../README.md)
- [`index.md`](index.md)
- [`../RustMesh/README.md`](../RustMesh/README.md)
- [`RustMesh-OpenMesh-Progress-2026-04-05.md`](RustMesh-OpenMesh-Progress-2026-04-05.md)
- [`../ROADMAP.md`](../ROADMAP.md)

## Project Overview

RustScan is a Rust workspace with multiple crates:

- `RustMesh`: mesh processing and OpenMesh comparison tooling
- `RustSLAM`: visual SLAM, sparse mapping, mesh extraction, and related IO
- `RustGS`: Gaussian splatting training
- `RustViewer`: visualization
- `rustscan-types`: shared types

## Common Commands

### Build

```bash
# Build workspace
cargo build --release

# Build RustMesh only
cargo build --manifest-path RustMesh/Cargo.toml --release

# Build RustSLAM only
cargo build --manifest-path RustSLAM/Cargo.toml --release
```

### Test

```bash
# RustMesh library tests
cargo test --manifest-path RustMesh/Cargo.toml --lib

# RustSLAM library tests
cargo test --manifest-path RustSLAM/Cargo.toml --lib
```

### Run Examples

```bash
# RustMesh
cargo run --manifest-path RustMesh/Cargo.toml --example test_smart
cargo run --manifest-path RustMesh/Cargo.toml --release --example openmesh_compare_decimation_trace -- 10

# RustSLAM
cargo run --manifest-path RustSLAM/Cargo.toml --example load_tum_dataset -- --dataset path/to/dataset
cargo run --manifest-path RustSLAM/Cargo.toml --release --example e2e_slam_to_mesh
```

### Benchmarks

```bash
cargo bench --manifest-path RustMesh/Cargo.toml
```

## Structural Notes

### RustMesh

- `src/Core/`: connectivity, kernels, handles, geometry, IO
- `src/Tools/`: decimation, remeshing, subdivision, smoothing, repair, dualization, analysis, VDPM
- `src/Utils/`: circulators, quadric helpers, smart ranges, status, performance
- `examples/openmesh_compare_*`: OpenMesh comparison harnesses

### RustSLAM

- `src/core/`: frame, pose, map, camera primitives
- `src/features/`: feature extraction and matching
- `src/tracker/`: visual odometry and geometry solvers
- `src/optimizer/`: bundle adjustment
- `src/loop_closing/`: loop detection and relocalization
- `src/fusion/`: Gaussian data structures, mesh extraction, and related IO helpers
- `src/io/`: dataset and video loading

## Guidance

- Prefer verification-backed statements over legacy percent-complete summaries.
- If you need current RustMesh parity state, read the progress doc before touching comparison code.
- Treat `docs/README.md`, `docs/RustMesh-README.md`, and `docs/ROADMAP.md` as compatibility redirects only.
