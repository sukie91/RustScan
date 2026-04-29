# TUM Freiburg1 XYZ RustGS profiles

Dataset:

```sh
/Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap
```

## Train profiles

Quality profile:

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- \
  train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_profiles/tum-prefix-quality.ply \
  --train-preset tum-prefix-quality
```

Compact profile:

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- \
  train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_profiles/tum-prefix-compact.ply \
  --train-preset tum-prefix-compact
```

Efficient profile:

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- \
  train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_profiles/tum-prefix-efficient.ply \
  --train-preset tum-prefix-efficient
```

Full trajectory baseline:

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- \
  train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_profiles/tum-full-798-baseline.ply \
  --train-preset tum-full-798-baseline
```

## Evaluation suite

```sh
scripts/rustgs_eval_suite.sh \
  --scene RustGS/output/experiments/tum_profiles/tum-prefix-compact.ply \
  --dataset /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --out RustGS/output/experiments/tum_profiles/tum-prefix-compact_eval \
  --profile-hint tum-prefix-compact \
  --gate-profile tum-prefix-compact
```

The suite evaluates:

- `post_train_6_frame`: `--max-frames 180 --frame-stride 30`
- `full_180`: `--max-frames 180 --frame-stride 1`
- `static_162`: `--max-frames 180 --frame-stride 1 --exclude-frame-ranges 76-93`
- `full_trajectory_stride_4`: `--max-frames 0 --frame-stride 4`

Outputs:

- `summary.json`
- `summary.md`
- `crops/<case>/rank_*_strip.png`

## Residual heatmap diagnostics

```sh
scripts/rustgs_residual_heatmap.sh \
  --scene RustGS/output/experiments/tum_profiles/tum-prefix-compact.ply \
  --dataset /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --out RustGS/output/experiments/tum_profiles/tum-prefix-compact_residuals \
  --frames 0,30,60,76-93,120,150 \
  --render-scale 0.25 \
  --raster-cov-blur 0.2 \
  --residual-threshold 0.12
```

This exports per-frame residual heatmaps and target/render/heatmap strips, plus connected-component stats for deciding whether dynamic/occlusion masking is worth enabling.

## Experimental dynamic mask

Dynamic residual masking is implemented as an experimental, default-off training control:

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- \
  train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_dynamic_mask_20260428/prefix180_compact_dynmask.ply \
  --train-preset tum-prefix-compact \
  --loss-dynamic-mask-threshold-low 0.35 \
  --loss-dynamic-mask-threshold-high 0.60 \
  --loss-dynamic-mask-min-weight 0.70
```

Current TUM results are negative. Do not use this as a default profile unless a later experiment beats `tum-prefix-compact`.

## Visibility prune dry-run diagnostics

Visibility / age prune is still diagnostic-only. This command records low-visibility splat counters in the parity report and logs, but it does not enable a new prune mode:

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- \
  train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_visibility_prune_20260428/prefix180_compact_visdry.ply \
  --train-preset tum-prefix-compact \
  --litegs-prune-visibility-dry-run \
  --litegs-prune-visibility-threshold 0.05 \
  --litegs-prune-high-opacity-threshold 0.80
```

Inspect `<scene>.litegs_parity.json` under `topology.topology_step_samples` for:

- `low_visibility_splats`
- `near_low_visibility_splats`
- `high_opacity_low_visibility_splats`
- `visibility_prune_dry_run_candidates`
- `actual_visibility_ratio`
