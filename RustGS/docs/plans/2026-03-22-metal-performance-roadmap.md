# Metal Performance Roadmap

## Status Snapshot: 2026-03-23

- Done:
  - step-level Metal profiling with visible/tile/native-forward breakdowns
  - calibrated Metal memory guardrails, safer default budget handling, and preflight auto-clamping for oversized initial Gaussian budgets
  - rotation-aware projection in both the Metal trainer and `diff_splat`
  - native Metal forward enabled by default with tensor fallback switch
  - runtime-managed packed tile metadata/index assembly
  - independent densify/prune scheduling with topology warmup/logging guardrails
- Partially done:
  - visible-set handling now prefers runtime buffer reads and avoids repeated Candle `to_vec*` round-trips in the normal Metal path, but full device-side sort/compaction is still an optimization target
  - backward is now isolated in `metal_backward.rs`, stages gradients directly onto device tensors, and removes trainer-side `ForwardIntermediate` construction from the hot path; fully dedicated Metal backward kernels remain a future optimization
- Remaining long-range roadmap work:
  - fully kernelized Metal backward math
  - GPU-side visible sort/compaction
  - higher-quality loss terms such as SSIM on the native path

## Smoke Commands

Minimum regression smoke test after Metal changes:

```bash
cargo test -p rustgs metal_trainer -- --nocapture
```

Full package regression pass used for this phase:

```bash
cargo test -p rustgs -- --nocapture
```

Recommended real-scene profiling command:

```bash
cargo run -p rustgs --features gpu -- train \
  --input <tum_dataset_dir> \
  --output /tmp/rustgs-metal-smoke.ply \
  --iterations 50 \
  --max-initial-gaussians 4096 \
  --metal-render-scale 0.25 \
  --metal-profile-steps \
  --metal-profile-interval 10
```

## Goal

Turn the current Metal training backend from a "can run on 16 GB Macs" baseline
into a high-performance 3DGS training path with predictable memory usage,
faster step times, and a clear route toward native Metal kernels.

## Recommended Delivery Strategy

Three possible strategies were considered:

1. Kernel-first rewrite
   Build the final Metal tile rasterizer and backward kernels immediately.
   Highest upside, but the riskiest path and hardest to debug.

2. Milestone-based optimization
   Add profiling, culling, sparse/tile structure, and only then replace the
   remaining dense tensor path with native kernels.
   This is the recommended path because each step is measurable and leaves the
   system runnable.

3. Parameter-tuning only
   Keep the current dense tensor renderer and mostly tune settings.
   Lowest effort, but it will not reach the target performance envelope.

This roadmap follows strategy 2.

## Epic 1: Baseline, Guardrails, and Observability

### Objective

Make performance visible and reproducible before deeper optimization work.

### Story 1.1: Add step-level profiling breakdown

- Goal: report time spent in projection, sorting, rasterization, loss,
  backward, and optimizer.
- Deliverables:
  - structured timing logs in `metal_trainer`
  - optional benchmark mode for short training runs
- Acceptance criteria:
  - a single training step prints a stable per-phase timing breakdown
  - timing can be compared across commits

### Story 1.2: Fix frame selection semantics and CLI clarity

- Goal: make `max_frames` and `frame_stride` behave the way users expect.
- Deliverables:
  - revised frame selection logic or better-documented semantics
  - updated CLI help text
- Acceptance criteria:
  - `max_frames=25, frame_stride=5` clearly results in 5 usable frames
  - tests cover the frame selection behavior

### Story 1.3: Calibrate Metal memory estimator

- Goal: replace the rough OOM guard with a more accurate estimator.
- Deliverables:
  - empirical memory model based on gaussian count, pixels, and chunk size
  - warnings that recommend safe parameters
- Acceptance criteria:
  - guard no longer significantly underestimates memory
  - logs explain why a run is blocked or allowed

## Epic 2: Sparse Forward Pass

### Objective

Stop paying dense `gaussians x pixels` cost for every training step.

### Story 2.1: Add frustum and screen-space culling

- Goal: drop gaussians that cannot contribute to the current frame.
- Deliverables:
  - visibility mask after projection
  - clipping against screen bounds and depth validity
- Acceptance criteria:
  - visible gaussian count is logged
  - step time decreases on typical TUM scenes

### Story 2.2: Add per-gaussian screen-space bounds

- Goal: compute 2D splat extents once and reuse them downstream.
- Deliverables:
  - min/max x/y bounds per visible gaussian
  - tests validating bounds against rendered support
- Acceptance criteria:
  - downstream stages consume bounds instead of full-image dense access

### Story 2.3: Add tile binning

- Goal: assign gaussians only to tiles they overlap.
- Deliverables:
  - tile metadata structure
  - tile-to-gaussian index lists
- Acceptance criteria:
  - each tile processes only overlapping gaussians
  - dense full-image traversal is removed from the forward path

## Epic 3: Native Metal Forward Renderer

### Objective

Replace the current Candle dense compositing path with a true tile-based Metal
renderer.

### Story 3.1: Build a Metal buffer/pipeline abstraction

- Goal: establish reusable device buffers and pipeline setup for custom kernels.
- Deliverables:
  - reusable Metal helper module
  - persistent buffers for camera, gaussians, and tile lists
- Acceptance criteria:
  - no per-step allocation churn for core buffers
  - kernels can be launched through a stable abstraction

### Story 3.2: Implement tile forward rasterization kernel

- Goal: render color/depth/alpha in Metal without dense Candle graph ops.
- Deliverables:
  - forward tile rasterization kernel
  - parity test against the current baseline on tiny scenes
- Acceptance criteria:
  - kernel output numerically matches baseline within defined tolerance
  - step time improves materially versus dense tensor compositing

### Story 3.3: Remove dense fallback from normal Metal training path

- Goal: make the tile renderer the default forward path for `backend=metal`.
- Deliverables:
  - runtime switch or permanent replacement in `metal_trainer`
- Acceptance criteria:
  - normal Metal training no longer depends on dense per-pixel broadcasts

## Epic 4: Native Backward and Loss Path

### Objective

Remove general-purpose autograd from the hot path.

### Story 4.1: Move loss accumulation to dedicated GPU kernels

- Goal: compute color/depth loss without building a large generic backward graph.
- Deliverables:
  - dedicated loss kernel or tightly scoped custom op
- Acceptance criteria:
  - loss no longer requires large intermediate autograd state

### Story 4.2: Implement analytic/specialized backward on GPU

- Goal: compute gradients for position, scale, opacity, rotation, and color in
  dedicated kernels.
- Deliverables:
  - backward kernel set
  - validation against the current training step on tiny scenes
- Acceptance criteria:
  - GPU gradients are numerically stable
  - generic `loss.backward()` is removed from the hot path

### Story 4.3: Optimize optimizer state updates

- Goal: keep Adam state and parameter updates fully on persistent GPU buffers.
- Deliverables:
  - buffer-based optimizer updates
  - reduced temporary tensor creation
- Acceptance criteria:
  - optimizer no longer dominates non-render time

## Epic 5: Quality Features and Scale-Up

### Objective

Recover the features expected from a production 3DGS trainer after the fast path
is in place.

### Story 5.1: Multi-frame scheduling and training curriculum

- Goal: improve frame sampling and training stability across scenes.
- Deliverables:
  - deterministic frame schedule
  - optional curriculum or visibility-aware frame sampling
- Acceptance criteria:
  - multi-frame training behaves predictably

### Story 5.2: Densify/prune for Metal backend

- Goal: support controlled gaussian growth and pruning without losing Metal
  execution benefits.
- Deliverables:
  - metadata sync points and resize strategy
  - bounded memory growth policy
- Acceptance criteria:
  - densify/prune works without destroying throughput

### Story 5.3: SSIM and full-quality loss path

- Goal: reintroduce higher-quality loss terms once the native renderer is fast.
- Deliverables:
  - GPU SSIM or equivalent perceptual loss path
- Acceptance criteria:
  - full-quality mode is available and benchmarked

## Development Order

Recommended execution order:

1. Story 1.1
2. Story 1.2
3. Story 2.1
4. Story 2.2
5. Story 2.3
6. Story 3.1
7. Story 3.2
8. Story 4.1
9. Story 4.2
10. Story 4.3
11. Story 5.1
12. Story 5.2
13. Story 5.3

## Why This Order

- Epic 1 makes performance work measurable and avoids tuning blind.
- Epic 2 cuts the biggest algorithmic waste before kernel work starts.
- Epic 3 replaces the hottest forward path once sparse structures exist.
- Epic 4 removes the remaining generic autograd overhead.
- Epic 5 restores quality and scalability features on top of the faster core.

## First Story Recommendation

Start with Story 1.1: add a step-level profiling breakdown.

Reason:

- it is low risk
- it gives us hard numbers for each phase
- it makes every later performance story easier to validate
- it helps confirm whether `0.4s` is dominated by projection, sorting,
  dense alpha generation, or backward
