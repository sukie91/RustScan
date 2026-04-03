# LiteGS vs RustGS Mac Parity Analysis

## Scope

- Goal: evolve `RustGS` into a Mac-runnable Rust implementation of LiteGS by extending the Metal runtime path rather than promoting the simplified [`training_pipeline.rs`](../../RustGS/src/training/training_pipeline.rs).
- Source-of-truth rule: LiteGS parity is defined against the mirrored LiteGS Python/CUDA implementation and the current RustGS Metal path.
- Compatibility rule: `LegacyMetal` remains available until Epic 21 acceptance passes; chunked training remains a separate compatibility surface and is not the parity target for the first Mac MVP.

## Fixture and Harness Baseline

- Tiny correctness fixture: `litegs-tiny-synthetic-v1`
- Apple Silicon convergence fixture: `litegs-apple-silicon-convergence-v1`
- Harness code: [`parity_harness.rs`](../../RustGS/src/training/parity_harness.rs)
- Shared thresholds: no NaNs, no OOM on Apple Silicon, non-clustered PSNR delta `<= 0.5 dB`, non-clustered Gaussian-count delta `<= 10%`, clustered PSNR delta `<= 0.7 dB`, deterministic export/load round-trip

## Notes

- LiteGS references:
  - [`trainer.py`](../../Mirror/LiteGS/litegs/training/trainer.py)
  - [`optimizer.py`](../../Mirror/LiteGS/litegs/training/optimizer.py)
  - [`densify.py`](../../Mirror/LiteGS/litegs/training/densify.py)
  - [`render/__init__.py`](../../Mirror/LiteGS/litegs/render/__init__.py)
  - [`scene/point.py`](../../Mirror/LiteGS/litegs/scene/point.py)
  - [`data.py`](../../Mirror/LiteGS/litegs/data.py)
  - [`io_manager/colmap.py`](../../Mirror/LiteGS/litegs/io_manager/colmap.py)
  - [`arguments.py`](../../Mirror/LiteGS/litegs/arguments.py)
- RustGS references:
  - [`training/mod.rs`](../../RustGS/src/training/mod.rs)
  - [`metal_trainer.rs`](../../RustGS/src/training/metal_trainer.rs)
  - [`metal_loss.rs`](../../RustGS/src/training/metal_loss.rs)
  - [`data_loading.rs`](../../RustGS/src/training/data_loading.rs)
  - [`initialization.rs`](../../RustGS/src/init/initialization.rs)
  - [`training_pipeline.rs`](../../RustGS/src/training/training_pipeline.rs)
  - [`rustgs.rs`](../../RustGS/src/bin/rustgs.rs)

## Training Loop

| LiteGS source of truth | Current RustGS behavior | Mismatch | Migration decision | Acceptance metric | Owning story |
| --- | --- | --- | --- | --- | --- |
| [`trainer.start`](../../Mirror/LiteGS/litegs/training/trainer.py) loads COLMAP frames, builds DataLoaders, advances `actived_sh_degree`, runs densify per epoch, and saves PLY/checkpoints on schedule. | [`training::train`](../../RustGS/src/training/mod.rs) routes directly into [`metal_trainer::train`](../../RustGS/src/training/metal_trainer.rs) or chunked orchestration; there is no LiteGS-specific profile boundary before this change. | RustGS had one public Metal path and one chunked compatibility path, but no explicit LiteGS parity lane. | Add `TrainingProfile::{LegacyMetal, LiteGsMacV1}` and route the public entrypoint through a dedicated LiteGS Mac function, even before full semantic parity lands. | `LiteGsMacV1` is selectable from API and CLI while `LegacyMetal` remains default and behavior-preserving. | 18.1 |
| LiteGS treats clustered training as part of its main runtime loop. | RustGS chunked training is an external orchestration layer over the current Metal trainer. | RustGS chunking and LiteGS clustering solve different problems and should not be conflated. | Keep chunked training as compatibility work; build LiteGS clustering later on the Metal runtime rather than on the chunk planner. | Non-clustered LiteGS Mac runs do not depend on chunked mode for correctness. | 19.1 |
| LiteGS advances SH degree during training (`actived_sh_degree=min(epoch/5, sh_degree)`). | RustGS Metal trainer currently has no public LiteGS SH schedule contract. | No stable parity contract for progressive SH activation yet. | Capture active SH degree in the parity harness now; wire the real schedule in Epic 18.3. | Harness schema records `active_sh_degree` for every run. | 17.3, 18.3 |

## Dataset / Camera Model

| LiteGS source of truth | Current RustGS behavior | Mismatch | Migration decision | Acceptance metric | Owning story |
| --- | --- | --- | --- | --- | --- |
| [`io_manager.load_colmap_result`](../../Mirror/LiteGS/litegs/training/trainer.py) and [`colmap.py`](../../Mirror/LiteGS/litegs/io_manager/colmap.py) treat COLMAP as the preferred source for cameras, frames, sparse points, and train/test splits. | [`load_training_dataset`](../../RustGS/src/lib.rs) currently resolves TUM RGB-D directories, `SlamOutput` JSON, or serialized `TrainingDataset` JSON. | RustGS does not yet have a first-class COLMAP training-input path, which blocks literal parity with LiteGS initialization and dataset semantics. | Treat COLMAP ingestion as the preferred LiteGS target, but keep the current dataset loader operational and use a bootstrap convergence fixture until COLMAP fixture ingestion is added. | Tiny fixture and convergence fixture are named and stable; convergence fixture registry records the intended COLMAP path. | 17.2, 18.2 |
| [`data.CameraFrameDataset`](../../Mirror/LiteGS/litegs/data.py) computes normalization radius and supplies view/projection/frustum tensors per frame. | [`data_loading.rs`](../../RustGS/src/training/data_loading.rs) converts `ScenePose` to [`DiffCamera`](../../RustGS/src/diff/diff_splat.rs), loads RGB/depth images, and can synthesize depth. | RustGS already has camera conversion and image loading, but not the same normalization/reporting surface as LiteGS. | Reuse the Metal data-loading path as the Mac V1 base; add parity reporting for fixture metadata and camera-derived normalization rather than reintroducing a Python-style loader. | Harness captures fixture ID, initialization counts, and wall-clock timing across fixed fixtures. | 17.3 |
| LiteGS has optional learnable camera optimization (`learnable_viewproj`). | RustGS has no equivalent public feature today. | Camera optimization surface is absent and should not be partially emulated. | Reserve config surface if needed, but keep camera optimization disabled and unsupported for Mac V1. | `learnable_viewproj` remains false and LiteGsMacV1 rejects attempts to enable it. | 18.1 |

## Gaussian Parameterization

| LiteGS source of truth | Current RustGS behavior | Mismatch | Migration decision | Acceptance metric | Owning story |
| --- | --- | --- | --- | --- | --- |
| [`scene.create_gaussians`](../../Mirror/LiteGS/litegs/scene/point.py) creates `xyz`, `scale`, `rot`, `sh_0`, `sh_rest`, `opacity`; scale is stored in log space, opacity in inverse-sigmoid space, SH degree defaults to 3. | [`TrainableGaussians`](../../RustGS/src/diff/diff_splat.rs) and [`trainable_from_map`](../../RustGS/src/training/data_loading.rs) currently model `positions`, `scales`, `rotations`, `opacities`, and RGB-like `colors`. | RustGS still trains RGB directly and has no full SH split (`sh_0` + `sh_rest`) in its public trainable representation. | Introduce the public LiteGS config/profile first, then evolve the trainable representation from RGB to SH-backed parameters in Epic 18.3. | Public config exposes `sh_degree`; later acceptance requires SH-backed rendering inputs instead of trained RGB. | 18.1, 18.3 |
| LiteGS initializes opacity to inverse-sigmoid of `0.1`. | [`GaussianInitConfig`](../../RustGS/src/init/initialization.rs) defaults opacity to `0.5`; helper converts that to logit for trainable tensors. | RustGS opacity bootstrap is materially different. | Move LiteGsMacV1 initialization to LiteGS opacity defaults while preserving LegacyMetal behavior. | Unit tests cover inverse-sigmoid(0.1) initialization on the LiteGS path. | 18.2 |
| LiteGS computes scale from nearest-neighbor distance and stores log-scales. | RustGS also stores log-scales in trainable tensors, but the public initializer clamps with `min_scale/max_scale` and uses `scale_factor=0.5`. | Similar mechanism, different defaults and clamping policy. | Reuse RustGS distance-based initialization machinery, but change LiteGsMacV1 defaults to LiteGS-compatible scale policy. | Fixture initialization counts and resulting scale distributions match reference expectations. | 18.2 |

## Render Preprocess

| LiteGS source of truth | Current RustGS behavior | Mismatch | Migration decision | Acceptance metric | Owning story |
| --- | --- | --- | --- | --- | --- |
| [`render_preprocess`](../../Mirror/LiteGS/litegs/render/__init__.py) normalizes quaternion rotation, exponentiates log scale, sigmoids opacity, derives per-view RGB from SH, and optionally performs cluster culling/compaction. | [`metal_trainer.rs`](../../RustGS/src/training/metal_trainer.rs) already keeps log-scales and opacity logits and projects/rasterizes on Metal, but still consumes stored RGB-like colors rather than SH-derived view-dependent color. | Activation math is partly aligned, but color representation and clustered preprocess are not. | Preserve the current Metal activation backbone, then replace RGB training inputs with SH evaluation and cluster-aware preprocess in separate stories. | LiteGsMacV1 can report active SH degree and eventually render from SH without regressing the Metal path. | 18.3, 19.1 |
| LiteGS preprocess integrates cluster AABB visibility and sparse-grad compaction in the same call chain. | RustGS preprocess is centered on per-Gaussian projection and the current Metal tile binning path; chunk planning lives outside the runtime. | No in-runtime cluster representation yet. | Add clustered preprocess on top of the current Metal runtime, not through chunk orchestration. | Clustered parity runs stay within `0.7 dB` PSNR delta and record visible-chunk bookkeeping in the harness. | 19.1, 19.2, 19.5 |

## Loss

| LiteGS source of truth | Current RustGS behavior | Mismatch | Migration decision | Acceptance metric | Owning story |
| --- | --- | --- | --- | --- | --- |
| [`trainer.py`](../../Mirror/LiteGS/litegs/training/trainer.py) uses fused L1+SSIM loss, optional scale regularization, optional transmittance penalty, and depth disabled by default. | [`metal_loss.rs`](../../RustGS/src/training/metal_loss.rs) provides mean absolute difference and SSIM gradient helpers; the simplified [`training_pipeline.rs`](../../RustGS/src/training/training_pipeline.rs) is not the intended production path. | RustGS lacks a public LiteGS loss contract on the Metal trainer and should not inherit semantics from the simplified pipeline. | Keep `training_pipeline.rs` out of parity ownership; implement LiteGS loss semantics directly in the Metal trainer path. | Harness report records loss terms (`l1`, `ssim`, `scale_regularization`, `transmittance`, `depth`, `total`) for every parity run. | 17.3, 18.4, 21.6 |
| LiteGS allows optional transmittance and depth terms from pipeline flags. | RustGS depth usage is currently coupled to data availability and existing training logic rather than explicit LiteGS switches. | The config surface for these terms was missing. | Add nested `LiteGsConfig` switches now, but keep non-default overrides rejected until the Metal loss path actually supports them. | LiteGsMacV1 rejects unsupported loss overrides before silent mis-training can occur. | 18.1, 18.4 |

## Optimizer

| LiteGS source of truth | Current RustGS behavior | Mismatch | Migration decision | Acceptance metric | Owning story |
| --- | --- | --- | --- | --- | --- |
| [`get_optimizer`](../../Mirror/LiteGS/litegs/training/optimizer.py) builds groups for `xyz`, `sh_0`, `sh_rest`, `opacity`, `scale`, `rot`; xyz decays exponentially while other groups stay fixed. | [`metal_trainer.rs`](../../RustGS/src/training/metal_trainer.rs) maintains a custom Adam state over positions, scales, opacities, and colors, but not the same public grouping or SH split. | Parameter groups and LR schedule semantics are not yet LiteGS-shaped. | Align the public profile/config first, then refactor Metal optimizer internals around LiteGS grouping and xyz-only decay. | Unit tests verify grouped optimizer defaults and xyz-only decay schedule. | 18.5 |
| LiteGS optionally uses sparse Adam when sparse-grad is enabled. | RustGS has no LiteGS-style sparse Adam path on Metal today. | Sparse-gradient parity is missing. | Implement sparse Adam only after cluster/visibility bookkeeping exists. | Sparse Adam semantics validated only in Epic 19. | 19.2, 19.4 |

## Densify / Prune / Opacity Reset

| LiteGS source of truth | Current RustGS behavior | Mismatch | Migration decision | Acceptance metric | Owning story |
| --- | --- | --- | --- | --- | --- |
| [`DensityControllerOfficial`](../../Mirror/LiteGS/litegs/training/densify.py) and [`DensityControllerTamingGS`](../../Mirror/LiteGS/litegs/training/densify.py) consume `mean2d_grad`, visibility, fragment weight/error, clone/split masks, prune masks, and opacity reset modes. | RustGS currently has simplified densify/prune/reset utilities in [`training_pipeline.rs`](../../RustGS/src/training/training_pipeline.rs) and some topology logic in the Metal trainer, but no LiteGS-equivalent statistics helper. | Topology heuristics and optimizer-state mutation semantics are not yet parity-correct. | Add a Rust parity harness/report schema now, then implement statistics accumulation and official/TamingGS density-controller behavior on the Metal path. | Harness schema includes densify/prune/reset counters; later unit tests cover masks and optimizer-state rebuild after topology edits. | 17.3, 20.1, 20.2, 20.4 |
| LiteGS supports `opacity_reset_mode` (`decay` / `reset`) and `prune_mode` (`weight` / `threshold`). | RustGS had no nested public config surface for these concepts. | There was no stable way to express the intended LiteGS controller modes. | Add the config surface immediately and reject unsupported non-default overrides until the controller lands. | CLI/API parse `opacity_reset_mode` and `prune_mode`; LiteGsMacV1 surfaces clear errors for premature overrides. | 18.1, 20.2, 20.3 |

## Clustering / Sparse-Grad

| LiteGS source of truth | Current RustGS behavior | Mismatch | Migration decision | Acceptance metric | Owning story |
| --- | --- | --- | --- | --- | --- |
| LiteGS clusters points, computes cluster AABBs, does frustum culling on clusters, and uses compacted visible-chunk / visible-primitive updates. | RustGS has budget-based chunked training and current Metal tile binning, but no in-runtime cluster AABB representation tied to optimizer updates. | RustGS chunking is not a drop-in replacement for LiteGS clustering. | Build clustering directly on the Metal runtime and keep chunked training as a separate operational path. | Clustered parity harness captures visible chunks/primitives and stays within the agreed clustered PSNR tolerance. | 19.1, 19.2, 19.5 |
| LiteGS reorders clustered state with Morton-order `spatial_refine`. | RustGS has no equivalent public runtime guarantee today. | No stable spatial-refine hook or cluster-bound refresh schedule. | Implement Morton-order reorder and optimizer-state refresh as explicit Metal runtime functionality. | Deterministic reorder and optimizer-state preservation tests pass on fixtures. | 19.3, 20.4 |

## Evaluation / Export

| LiteGS source of truth | Current RustGS behavior | Mismatch | Migration decision | Acceptance metric | Owning story |
| --- | --- | --- | --- | --- | --- |
| LiteGS periodically evaluates PSNR, saves PLY, and saves checkpoints with full tensor state. | RustGS saves final PLY and already has scene IO/checkpoint infrastructure, but no LiteGS-specific evaluation/reporting contract or SH-aware checkpoint/export path yet. | Export format and reporting do not yet cover LiteGS SH/state semantics. | Extend evaluation/reporting, checkpointing, and PLY IO only after the LiteGS-compatible parameterization is in place. | Harness records PSNR, export round-trip, and checkpoint round-trip deterministically. | 17.3, 21.1, 21.2, 21.3 |
| LiteGS final save path exports unclustered tensors if clustered mode was used. | RustGS chunk merge/export is based on current `GaussianMap` representation. | Future clustered LiteGS export must uncluster or serialize cluster-aware state without loss. | Keep current export stable for LegacyMetal; add LiteGS-compatible SH export/import when the representation changes. | Deterministic export/load round-trip passes fixture thresholds. | 21.3 |

## Mac-Specific Constraints

| LiteGS source of truth | Current RustGS behavior | Mismatch | Migration decision | Acceptance metric | Owning story |
| --- | --- | --- | --- | --- | --- |
| LiteGS assumes CUDA fused kernels and SparseAdam paths. | RustGS targets Apple Silicon via Metal/Candle and already contains memory-budget and chunked orchestration work to survive 16 GB machines. | CUDA-first LiteGS features cannot be copied directly onto Mac. | Phase parity: first ship non-clustered LiteGsMacV1 on Metal, then add clustered/sparse-grad behavior where operationally safe on Mac. | Tiny fixture finishes without OOM on a 16 GB Apple Silicon machine; no NaNs in parity runs. | 17.4, 18.1, 19.5 |
| LiteGS defaults favor clustered + sparse training. | Mac V1 parity target intentionally starts non-clustered with sparse-grad off. | Upstream defaults are not yet Mac-safe in RustGS. | Use Mac-safe bootstrap defaults in `LiteGsConfig` (`cluster_size=0`, `sparse_grad=false`) and promote upstream-like behavior only after Epic 19/20 acceptance. | `LiteGsMacV1` rejects unsupported overrides rather than silently ignoring them. | 18.1, 19.1 |
| LiteGS has no notion of RustGS `LegacyMetal` compatibility fallback. | RustGS must preserve current production behavior while parity work lands. | Default switch cannot happen until parity is measured. | Keep `LegacyMetal` as default until Epic 21 gates pass, then promote `LiteGsMacV1` explicitly. | Promotion happens only after fixture suite passes Epic 17 thresholds. | 21.5 |
