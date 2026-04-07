# LiteGS Parity Remaining Work Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the remaining gap between `RustGS` `LiteGsMacV1` and the current LiteGS reference behavior by grounding parity on a real fixture, encoding report-level tolerances, and then finishing the still-approximate performance and semantic paths.

**Architecture:** Use the parity harness and CLI parity report as the contract layer. First lock a canonical reference fixture and thresholded side-by-side comparisons. Then tighten depth and topology convergence parity. After parity is measurable, optimize the two remaining approximation-heavy areas: `sparse/clustered` updates and `learnable_viewproj`.

**Tech Stack:** Rust, Candle, Metal, serde/JSON, `cargo test`, fixture-based parity reports

---

## Re-baselined Status

The 2026-04-04 roadmap is now partially stale. These items are already implemented or materially advanced in the current `main` line and should not be treated as the critical path:

- COLMAP input loading and CLI source detection
- LiteGS scale regularization semantics
- LiteGS transmittance option wiring
- Morton reorder hook on densify
- LiteGS target primitive budget plumbing
- Opacity reset controls and telemetry
- Progressive SH activation in Metal training
- Position LR decay and runtime LR telemetry
- Visible-only sparse optimizer first pass
- Cluster assignment refresh on topology / position updates
- Learnable pose update first pass
- Depth supervision enablement plus internal telemetry

What is still missing is not “support exists or not”, but “is RustGS demonstrably close enough to LiteGS on a real fixture, and is the remaining perf/semantic delta worth closing”.

## Phase 1: Canonical Fixture Parity Gate

**Priority:** P0
**Outcome:** `LiteGsMacV1` parity is measured against a checked-in LiteGS reference report instead of only internal telemetry.

### Task 1: Check in the canonical reference fixture

**Files:**
- Create: `test_data/fixtures/litegs/colmap-small/`
- Create: `test_data/fixtures/litegs/colmap-small/parity-reference.json`
- Modify: `RustGS/src/training/parity_harness.rs`
- Modify: `docs/index.md`

**Work:**
- Add the real Apple Silicon convergence fixture or the minimum repo-safe subset needed for parity.
- Check in a real LiteGS-produced `ParityHarnessReport` JSON at the default path already wired in the fixture spec.
- Remove ambiguity between canonical fixture and bootstrap fallback in docs.

**Validation:**
- `cargo test -p rustgs resolve_litegs_reference_report_path_returns_existing_reference_only -- --nocapture`
- `cargo test -p rustgs litegs_parity_report_populates_reference_comparison_from_workspace_fixture -- --nocapture`

**Exit Criteria:**
- Default fixture resolution finds a real `parity-reference.json` in the repo.
- A local LiteGS parity run no longer depends on a synthetic temp workspace for reference comparison coverage.

### Task 2: Turn comparison data into a parity gate

**Files:**
- Modify: `RustGS/src/training/parity_harness.rs`
- Modify: `RustGS/src/bin/rustgs.rs`
- Modify: `RustGS/src/lib.rs`
- Modify: `RustGS/src/training/mod.rs`

**Work:**
- Extend the parity report from “record deltas” to “evaluate deltas”.
- Add threshold fields for depth curve, total loss curve, final PSNR, and gaussian-count evolution.
- Add explicit pass/fail summaries so parity runs can be triaged without manually reading raw deltas.

**Validation:**
- Add unit tests for threshold evaluation and missing-reference behavior.
- Re-run existing parity report tests.

**Exit Criteria:**
- A parity report can answer “did this run stay within tolerance?” without manual inspection.
- Thresholds are versioned in code and described in docs.

## Phase 2: Depth Loss Side-by-Side Calibration

**Priority:** P0
**Outcome:** depth loss is calibrated against LiteGS using real fixture data rather than only internal consistency checks.

### Task 3: Lock depth reference tolerances

**Files:**
- Modify: `RustGS/src/training/metal_trainer.rs`
- Modify: `RustGS/src/training/metal_loss.rs`
- Modify: `RustGS/src/training/parity_harness.rs`
- Modify: `RustGS/src/bin/rustgs.rs`

**Work:**
- Compare RustGS and LiteGS `depth` loss samples over the convergence fixture.
- Calibrate tolerances for:
  - `depth_mean_abs_delta`
  - `depth_max_abs_delta`
  - `depth_valid_pixels`
  - `depth_grad_scale`
- Decide whether `depth_valid_pixels` and `depth_grad_scale` are strict parity metrics or diagnostics-only metrics.

**Validation:**
- `cargo test -p rustgs depth_backward_scale_uses_only_valid_depth_samples -- --nocapture`
- `cargo test -p rustgs training_step_records_depth_telemetry_with_clustered_sparse_grad -- --nocapture`
- Add a new test that asserts threshold evaluation for depth deltas.

**Exit Criteria:**
- Depth parity has a documented tolerance band.
- Reports distinguish true depth regressions from harmless numeric drift.

### Task 4: Extend parity from point samples to convergence shape

**Files:**
- Modify: `RustGS/src/training/parity_harness.rs`
- Modify: `RustGS/src/bin/rustgs.rs`
- Modify: `RustGS/src/training/metal_trainer.rs`

**Work:**
- Decide whether the current sampling schedule (`iter < 5`, every `25`, final iter) is sufficient.
- If not, add denser sampling around bootstrap, first densify window, opacity reset boundaries, and final stabilization.
- Compare not only scalar deltas but also event-aligned milestones.

**Validation:**
- Add tests for sample alignment when the reference and current reports have sparse but overlapping iteration sets.

**Exit Criteria:**
- The parity harness can explain where the curves diverge, not only that they diverged.

## Phase 3: Topology and Convergence Parity

**Priority:** P0
**Outcome:** parity covers topology evolution, not only losses.

### Task 5: Compare densify / prune / opacity-reset behavior

**Files:**
- Modify: `RustGS/src/training/metal_trainer.rs`
- Modify: `RustGS/src/training/density_controller.rs`
- Modify: `RustGS/src/training/parity_harness.rs`

**Work:**
- Add report-level comparison for:
  - `densify_events`
  - `densify_added`
  - `prune_events`
  - `prune_removed`
  - `opacity_reset_events`
  - `final_gaussians`
- Decide which of these require hard thresholds and which are trend-only diagnostics.

**Validation:**
- Extend report round-trip tests.
- Add unit tests for topology comparison summaries.

**Exit Criteria:**
- A run can fail parity even if losses look acceptable but topology evolution is obviously wrong.

### Task 6: Add a real fixture regression command path

**Files:**
- Modify: `RustGS/src/bin/rustgs.rs`
- Modify: `docs/index.md`
- Modify: `docs/current-project-status.md`

**Work:**
- Write down the exact one-command recipe for:
  - running the canonical parity fixture
  - emitting a parity report
  - reading the pass/fail fields
- Keep the GPU test skip behavior for non-Metal environments, but define the Apple Silicon regression path as the primary acceptance flow.

**Validation:**
- Dry-run the documented command path in a Metal-capable environment.

**Exit Criteria:**
- There is one stable documented way to rerun parity after every meaningful training change.

## Phase 4: Sparse / Clustered Finalization

**Priority:** P1
**Outcome:** RustGS keeps the current LiteGS-like semantics but reduces the remaining performance and data-layout gap.

### Task 7: Decide the final clustered update unit

**Files:**
- Modify: `RustGS/src/training/clustering.rs`
- Modify: `RustGS/src/training/metal_trainer.rs`
- Modify: `RustGS/src/training/morton.rs`

**Work:**
- Decide whether clustered sparse updates should stay primitive-based with cluster-derived visibility, or move to a more explicit cluster-native compacted layout.
- Measure visibility-set size, gather/scatter overhead, and reorder churn after densify.

**Validation:**
- Add profiling counters or telemetry for visible primitive count and sparse-update time.

**Exit Criteria:**
- The update model is explicit and documented.
- The team can justify keeping the current layout or replacing it.

### Task 8: If profiling justifies it, implement compacted sparse state

**Files:**
- Modify: `RustGS/src/training/metal_trainer.rs`
- Modify: `RustGS/src/training/clustering.rs`
- Modify: `RustGS/src/training/metal_runtime.rs`
- Modify: `RustGS/src/training/mod.rs`

**Work:**
- Replace generic `index_select/index_add`-style sparse updates with a compacted visible-state path if the current overhead is material.
- Keep parity report semantics unchanged while improving execution cost.

**Validation:**
- Existing sparse regression tests
- New perf comparison logs on the canonical fixture

**Exit Criteria:**
- Either the compacted path lands, or profiling proves the current path is good enough and the task closes with evidence.

## Phase 5: Learnable ViewProj Performance Pass

**Priority:** P1
**Outcome:** pose learning remains available but stops being dominated by finite-difference re-renders.

### Task 9: Instrument pose-learning cost and convergence impact

**Files:**
- Modify: `RustGS/src/training/metal_trainer.rs`
- Modify: `RustGS/src/training/pose_embedding.rs`
- Modify: `RustGS/src/bin/rustgs.rs`

**Work:**
- Add telemetry for pose-gradient wall time, extra render count, and per-step pose update cost.
- Measure whether pose learning materially changes parity on the canonical fixture.

**Validation:**
- Add unit coverage for telemetry plumbing.
- Run one Apple Silicon comparison with pose learning on and off.

**Exit Criteria:**
- The cost of `learnable_viewproj` is visible in logs and reports.

### Task 10: Replace or narrow the finite-difference fallback

**Files:**
- Modify: `RustGS/src/training/metal_trainer.rs`
- Modify: `RustGS/src/training/pose_embedding.rs`
- Modify: `RustGS/src/diff/analytical_backward.rs`
- Modify: `RustGS/src/training/mod.rs`

**Work:**
- Prefer analytical or lower-cost approximate pose gradients where possible.
- If full analytical pose gradients are too invasive, reduce FD scope to a smaller parameter subset or lower-frequency update schedule.

**Validation:**
- Pose embedding tests
- One convergence comparison on the canonical fixture

**Exit Criteria:**
- Pose learning is either materially faster or explicitly demoted to an experimental path with measured tradeoffs.

## Phase 6: Peripheral Cleanup

**Priority:** P2
**Outcome:** docs and surrounding tooling match the implemented training parity story.

### Task 11: Refresh the retained parity documentation set

**Files:**
- Modify: `docs/index.md`
- Modify: `docs/current-project-status.md`

**Work:**
- Keep `docs/index.md` and `docs/current-project-status.md` aligned with the retained parity plan and guardrail docs.
- Avoid re-introducing background-only roadmap copies into `docs/`.
- Make the retained docs point to one current parity gap list and one current acceptance path.

**Exit Criteria:**
- The retained docs no longer imply that removed or superseded parity docs still exist.

### Task 12: Revisit render-side parity only after training parity stabilizes

**Files:**
- Modify: `RustGS/src/bin/rustgs.rs`
- Modify: `RustGS/src/render/mod.rs`
- Modify: `RustGS/src/render/tiled_renderer.rs`
- Modify: `docs/README.md`

**Work:**
- Fill render CLI and viewer-adjacent gaps only after the training parity gate is trustworthy.
- Avoid spending time on user-facing polish before the training semantics are locked.

**Exit Criteria:**
- Render work is explicitly downstream of training parity, not competing with it.

## Recommended Execution Order

1. Task 1: Check in the real LiteGS reference report
2. Task 2: Convert delta recording into parity gating
3. Task 3: Lock depth tolerances
4. Task 5: Add topology comparison
5. Task 6: Finalize the real regression command path
6. Task 7: Measure sparse / clustered overhead
7. Task 8: Optimize sparse state only if profiling proves it matters
8. Task 9: Instrument pose-learning cost
9. Task 10: Optimize or explicitly demote finite-difference pose learning
10. Task 11: Refresh the old roadmap docs
11. Task 12: Only then revisit render / CLI polish

## Milestones

- **M1: Measurable Parity**
  - Real reference fixture checked in
  - Report-level pass/fail thresholds defined
  - Depth parity calibrated

- **M2: Stable Regression Loop**
  - Topology comparison added
  - Apple Silicon fixture command documented
  - Every major training change can be replayed against the same gate

- **M3: Performance Closure**
  - Sparse / clustered cost characterized and either optimized or accepted
  - Pose-learning path instrumented and narrowed

- **M4: Documentation Closure**
  - Old roadmap updated
  - Training parity story becomes the canonical team workflow

## Immediate Recommendation

The next engineering move should still be **Task 1 + Task 2 together**:

- without a real checked-in LiteGS reference report, parity is still self-referential
- without thresholded pass/fail logic, the new comparison plumbing is informative but not actionable

That pair is the shortest path from “we record parity data” to “we can actually reject regressions”.
