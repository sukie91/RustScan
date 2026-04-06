# RustMesh rm-opt Checklist Archive

This file is intentionally reduced to an archive note.

The original step-by-step checklist mixed completed parity-debugging tasks with still-open follow-up work, which made it a stale second source of truth.

Use these documents instead:

- Current verified state: [`../RustMesh-OpenMesh-Progress-2026-04-05.md`](../RustMesh-OpenMesh-Progress-2026-04-05.md)
- Remaining backlog: [`../RustMesh-OpenMesh-Parity-Roadmap.md`](../RustMesh-OpenMesh-Parity-Roadmap.md)
- Current execution snapshot: [`2026-04-06-rustmesh-next-phase-plan.md`](2026-04-06-rustmesh-next-phase-plan.md)

The remaining actionable work is now tracked in the roadmap as epics/stories. At the top level, it is:

1. Keep normals comparison coverage durable now that the default, compatibility-mode, and refresh-policy contracts are explicit.
2. Harden incremental progressive-mesh LOD so `get_lod(level)` can move upward without replaying from `original`.
3. Expand parity regression only where it protects real workflow behavior.
4. Decide separately whether deeper n-gon topology surgery is worth the complexity, including whether rebuild-backed fallbacks need the same property-propagation contract as the maintained local path.

Dynamic-property propagation on the maintained topology path and the explicit normals contract are now complete and covered by focused regression tests plus the full RustMesh library suite.
