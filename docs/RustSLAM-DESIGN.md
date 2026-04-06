# RustSLAM Design

**Updated:** 2026-04-05

This document records the current design boundary for RustSLAM. It is not intended to be a line-by-line file inventory.

## Scope

RustSLAM is responsible for:

- frame, pose, camera, keyframe, and sparse-map primitives
- feature extraction and matching
- visual odometry and geometry solvers
- bundle adjustment
- loop detection and relocalization
- dataset and video IO
- mesh extraction support inside the SLAM pipeline

RustSLAM is not the workspace home for every dense-reconstruction concern:

- `RustGS` owns Gaussian training and rendering work that has moved out of RustSLAM
- `RustMesh` owns mesh post-processing after extraction

## Module Layout

Current high-level module layout:

```text
RustSLAM/src/
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

## Design Principles

### 1. Sparse SLAM First

The core pipeline is still sparse-feature based:

- features are extracted and matched frame to frame,
- visual odometry estimates pose,
- mapping and optimization refine the sparse state,
- loop closing and relocalization recover longer-horizon consistency.

### 2. Mesh Extraction Is Downstream

RustSLAM can produce mesh-oriented outputs through its fusion and extraction code, but mesh quality and post-processing are deliberately downstream concerns.

### 3. Worktree Reality Beats Historical Plans

In the current `rm-opt` worktree, the RustSLAM library suite is not fully green. That means design discussions should not be mistaken for a readiness claim.

## Related Docs

- Current repository state: [../README.md](../README.md)
- Documentation index: [index.md](index.md)
- RustSLAM overview: [RustSLAM-README.md](RustSLAM-README.md)
- RustSLAM experiment notes: [RustSLAM-Experiment-2026-03-28.md](RustSLAM-Experiment-2026-03-28.md)
- Current RustSLAM backlog: [RustSLAM-ToDo.md](RustSLAM-ToDo.md)
