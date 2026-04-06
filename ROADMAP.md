# RustScan Roadmap

**Updated:** 2026-04-05

This roadmap is intentionally high level. Current branch facts live in `README.md` and `docs/RustMesh-OpenMesh-Progress-2026-04-05.md`; this file only tracks forward work.

## Current State

- The workspace is functional enough to build and iterate on RustMesh, RustSLAM, RustGS, and RustViewer in one repository.
- RustMesh is the most actively validated crate in the current `rm-opt` worktree.
- RustSLAM remains active development and is not fully green in the current worktree.

## Near-Term Priorities

### 1. RustMesh OpenMesh Follow-Through

- Keep decimation parity stable with stronger automated regression coverage.
- Replace remeshing shortcuts with more robust topology operations where needed.
- Add the missing normalized progressive-mesh LOD API.
- Expand parity coverage beyond the current decimation trace baseline where it is worth the maintenance cost.

### 2. RustSLAM Quality and Reliability

- Fix the current failing library tests in bundle adjustment and relocalization paths.
- Continue tightening video-input and real-data pipeline behavior.
- Improve correctness before adding more user-facing claims about readiness.

### 3. Documentation Discipline

- Keep only one maintained status source per topic.
- Use redirect documents for backwards compatibility instead of duplicating state.
- Avoid percent-complete tables unless they are backed by a reproducible checklist.

## RustMesh-Specific Backlog

For the detailed RustMesh backlog, use:

- [`docs/RustMesh-OpenMesh-Parity-Roadmap.md`](./docs/RustMesh-OpenMesh-Parity-Roadmap.md)
- [`docs/RustMesh-OpenMesh-Progress-2026-04-05.md`](./docs/RustMesh-OpenMesh-Progress-2026-04-05.md)

## Workspace Direction

- Keep the crates loosely coupled and testable in isolation.
- Prefer verification-backed claims over marketing-style completeness claims.
- Reduce documentation churn by centralizing status reporting in a few maintained entry points.
