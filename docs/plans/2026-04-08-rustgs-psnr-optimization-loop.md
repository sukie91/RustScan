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

## Notes

- Every attempt must update this document before moving to the next one.
- All PSNR values in this log must come from the unified RustGS evaluator on the held-out eval split.
- Stop condition reached on 2026-04-08 at attempt `3`.
