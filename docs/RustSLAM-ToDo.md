# RustSLAM Backlog

**Updated:** 2026-04-05
**Purpose:** Current RustSLAM backlog for the active workspace state

This file replaces older milestone-style TODO content that referenced modules and training paths no longer aligned with the current repository layout.

## Current Verification Snapshot

Latest check run in this worktree:

```bash
cargo test --manifest-path RustSLAM/Cargo.toml --lib --quiet
```

Result:

- `261 passed; 2 failed`
- failing tests:
  - `optimizer::ba::tests::test_optimize_updates_camera_pose`
  - `tracker::vo::tests::test_relocalize_failure_does_not_report_success`

## Near-Term Priorities

### 1. Fix Current Failing Tests

- restore the bundle-adjustment cost-reduction expectation
- fix relocalization failure-state handling so the VO state stays coherent after an unsuccessful relocalization attempt

### 2. Tighten Real Video and Dataset Workflows

- keep dataset and video-loading paths aligned with the examples that actually exist
- verify that the end-to-end mesh extraction path remains usable on representative inputs
- avoid documenting examples or binaries that are not present in the tree

### 3. Keep Ownership Boundaries Clear

- RustSLAM should focus on SLAM, sparse mapping, loop closing, IO, and extraction-side support
- RustGS should remain the home for Gaussian training/rendering concerns
- RustMesh should remain the home for mesh post-processing and OpenMesh comparison work

## Medium-Term Work

- improve reliability on real-world video inputs
- continue calibration and quality evaluation work
- strengthen end-to-end regression coverage where it protects real workflows

## Related Docs

- Workspace overview: [../README.md](../README.md)
- Documentation index: [index.md](index.md)
- RustSLAM design boundary: [RustSLAM-DESIGN.md](RustSLAM-DESIGN.md)
- RustSLAM experiment notes: [RustSLAM-Experiment-2026-03-28.md](RustSLAM-Experiment-2026-03-28.md)
