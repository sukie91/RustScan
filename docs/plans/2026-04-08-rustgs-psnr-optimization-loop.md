# RustGS 500-Iter PSNR Optimization Loop on `tum_freiburg1_xyz_colmap`

## Goal

Use the held-out `tum_freiburg1_xyz_colmap` benchmark split as the stop condition for iterative RustGS optimization.

Stop only when:

- RustGS `500`-iteration PSNR exceeds Brush `500`-iteration PSNR on the same benchmark.

## Benchmark Standard

- Source dataset: `/Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap`
- Benchmark split:
  - train: `/tmp/tum_freiburg1_xyz_colmap_split/train`
  - eval: `/tmp/tum_freiburg1_xyz_colmap_split/eval`
- Unified evaluator:
  - `target/debug/examples/evaluate_psnr`
- RustGS training command family:
  - `cargo run -p rustgs -- train ... --iterations 500`
- Brush reference artifact:
  - `/tmp/tum_brush_500_fullsplit_exports/export_500_rustgs_ascii.ply`

## Reference Target

- Brush 500-iter PSNR: `7.889576 dB`
- Reference artifact:
  - scene: `/tmp/tum_brush_500_fullsplit_exports/export_500_rustgs_ascii.ply`
  - evaluator: `target/debug/examples/evaluate_psnr --scene /tmp/tum_brush_500_fullsplit_exports/export_500_rustgs_ascii.ply --dataset /tmp/tum_freiburg1_xyz_colmap_split/eval --render-scale 1.0 --frame-stride 1 --max-frames 160 --device cpu --json`

## Attempt Log

| Attempt | Date | Change | RustGS 500-iter PSNR | Delta vs Prev | Gap vs Brush | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| ref | 2026-04-08 | Pre-loop inherited RustGS reference from the previous full-split run before the memory-guard refactor. | `4.262130 dB` | n/a | `-3.627446 dB` | Historical baseline from `/tmp/tum_rustgs_500_fullsplit.ply` on the same eval split. |
| 1 | 2026-04-08 | Removed the fake `gaussians * pixels` Metal guard model and switched the main Metal training path to build one frame at a time instead of preloading all GPU targets. | `5.685438 dB` | `+1.423308 dB` | `-2.204138 dB` | Artifact: `/tmp/tum_rustgs_500_optloop_attempt1.ply`. Training no longer downsampled the init to `398`; it trained all `8986` gaussians. However, topology still did not densify at all by 500 iter: `densify_events=0`, `final_gaussians=8986`. |
| 2 | 2026-04-08 | Compressed the LiteGS topology epoch window for short benchmark runs so sub-epoch 500-iter training still activates the default refine cadence, instead of waiting for an impossible `epoch >= 3`. | `7.644472 dB` | `+1.959034 dB` | `-0.245104 dB` | Artifact: `/tmp/tum_rustgs_500_optloop_attempt2.ply`. Parity: `/tmp/tum_rustgs_500_optloop_attempt2.parity.json`. Topology now activated twice inside 500 iter: `densify_events=2`, `densify_added=1154`, `prune_events=1`, `final_gaussians=10137`. The remaining gap to Brush is now small enough to target with short-run refine-quality tuning instead of another structural memory/topology fix. |
| 3 | 2026-04-08 | Matched Brush's full-resolution training regime after the short-run topology fix by raising RustGS Metal training from `320x240` to `640x480` (`--metal-render-scale 1.0`) while keeping the same 500-iter schedule and split. | `7.945506 dB` | `+0.301034 dB` | `+0.055930 dB` | Artifact: `/tmp/tum_rustgs_500_optloop_attempt3_fullres.ply`. Parity: `/tmp/tum_rustgs_500_optloop_attempt3_fullres.parity.json`. Topology still activated twice (`densify_events=2`, `densify_added=1193`, `prune_events=1`, `final_gaussians=10177`), and the stop condition was met because RustGS now exceeds Brush's `7.889576 dB` baseline on the held-out eval split. |
| 4 | 2026-04-09 | Revalidated against the new pulled `main` after the 2026-04-09 fast-forward. Re-ran the same `500`-iter full-resolution RustGS benchmark on the held-out split before attempting any new optimization. | `8.709845 dB` | `+0.764339 dB` | `+0.820269 dB` | Artifact: `/tmp/tum_rustgs_500_revalidate_2026_04_09_fullres.ply`. Parity: `/tmp/tum_rustgs_500_revalidate_2026_04_09_fullres.parity.json`. Current `main` already stays well above Brush without additional changes: `densify_events=2`, `added≈1151 net` (`8986 -> 10127`), `prune_events=1`. No extra optimization pass was needed in this session. |

## Notes

- Every attempt must update this document before moving to the next one.
- All PSNR values in this log must come from the unified RustGS evaluator on the held-out eval split.
- Stop condition reached on 2026-04-08 at attempt `3`.
- Revalidated on 2026-04-09 after pulling new `main`; stop condition still holds at attempt `4`.

## Release Rebaseline

On 2026-04-10 the benchmark loop was re-pinned to the current release evaluator and a stable Brush export that is known to round-trip correctly through the RustGS evaluator.

- Unified evaluator:
  - `target/release/examples/evaluate_psnr`
- RustGS training command family:
  - `target/release/rustgs train ... --iterations 500`
- Stable Brush reference artifact:
  - `/tmp/tum_brush_500_fullsplit_exports/export_500_rustgs_ascii.ply`
- Stable Brush 500-iter PSNR:
  - `7.4447455 dB`
- Updated stop target:
  - RustGS `500`-iter PSNR must reach at least `8.9336946 dB` to exceed Brush by `20%`.

### Release Attempt Log

| Attempt | Date | Change | RustGS 500-iter PSNR | Delta vs Prev | Gap vs 20% Target | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 2026-04-10 | Re-ran the release baseline on the split with the current refactor branch before changing LiteGS defaults. | `8.333964 dB` | n/a | `-0.599731 dB` | Artifact: `/tmp/tum_rustgs_500_release_20260409_232434_fullres.ply`. This is already above the stable Brush reference, but still below the new `+20%` stop target. |
| 6 | 2026-04-10 | Tuned short-run LiteGS refine cadence from `200` to `160` iterations. | `8.654089 dB` | `+0.320125 dB` | `-0.279606 dB` | Artifact: `/tmp/tum_rustgs_500_release_20260410_refine160.ply`. This confirmed the release branch still benefits from earlier topology activation inside a 500-iter budget. |
| 7 | 2026-04-10 | Kept `refine_every=160` and raised `growth_select_fraction` from `0.20` to `0.25`. | `8.789234 dB` | `+0.135145 dB` | `-0.144461 dB` | Artifact: `/tmp/tum_rustgs_500_release_20260410_refine160_grow025.ply`. Parity: `/tmp/tum_rustgs_500_release_20260410_refine160_grow025.parity.json`. This became the best clean tuning result before updating defaults. |
| 8 | 2026-04-10 | Kept `refine_every=160`, `growth_select_fraction=0.25`, and lowered `growth_grad_threshold` from `0.00015` to `0.00014`. | `8.830818 dB` | `+0.041584 dB` | `-0.102877 dB` | Artifact: `/tmp/tum_rustgs_500_release_20260410_refine160_grow025_grad014.ply`. Parity: `/tmp/tum_rustgs_500_release_20260410_refine160_grow025_grad014.parity.json`. This is the best release result of the session so far, but it still does not clear the `+20%` target. |
| 9 | 2026-04-10 | Lowered `growth_grad_threshold` further to `0.00013` with the same short-run settings. | `8.724605 dB` | `-0.106213 dB` | `-0.209090 dB` | Artifact: `/tmp/tum_rustgs_500_release_20260410_refine160_grow025_grad013.ply`. Parity: `/tmp/tum_rustgs_500_release_20260410_refine160_grow025_grad013.parity.json`. This overshot the useful growth window and was rejected. |
| 10 | 2026-04-10 | Promoted the best short-run tuning found so far into the LiteGS defaults (`refine_every=160`, `growth_select_fraction=0.25`, `growth_grad_threshold=0.00014`) and re-ran the default release profile with no extra LiteGS CLI overrides. | `8.737700 dB` | `+0.013095 dB` | `-0.195995 dB` | Artifact: `/tmp/tum_rustgs_500_release_20260410_default_profile_v2.ply`. Parity: `/tmp/tum_rustgs_500_release_20260410_default_profile_v2.parity.json`. This improved the baked-in default profile versus the old release baseline, but it did not reproduce the best hand-tuned run exactly. |
| 11 | 2026-04-10 | Added a small Brush-inspired post-refine `opacity/scale` decay pass after LiteGS topology updates, while keeping the same SoA owner model and default short-run config. | `8.810633 dB` | `+0.072933 dB` | `-0.123062 dB` | Artifact: `/tmp/tum_rustgs_500_release_20260410_refine_decay_v1.ply`. Parity: `/tmp/tum_rustgs_500_release_20260410_refine_decay_v1.parity.json`. The housekeeping pass recovered most of the default-profile regression, but it still stayed below the best observed manual sweep (`8.830818 dB`). |

### Rejected Structural Experiment

- A Brush-style densify semantic rewrite in `RustGS/src/training/topology.rs` was benchmarked and then reverted.
  - Result: `7.231229 dB`
  - Artifact: `/tmp/tum_rustgs_500_release_20260410_refine_semantics.ply`
  - Conclusion: matching Brush's split/opacity rewrite more literally hurt RustGS on this 500-iter TUM case, so the current branch keeps the simpler baseline LiteGS mutation semantics and only tunes short-run scheduling defaults.

### Current Best

- Best observed release result on this benchmark split remains attempt `8`:
  - `8.830818 dB`
  - artifact: `/tmp/tum_rustgs_500_release_20260410_refine160_grow025_grad014.ply`
- The current gap to the `+20%` stop target is:
  - `8.9336946 - 8.830818 = 0.1028766 dB`
