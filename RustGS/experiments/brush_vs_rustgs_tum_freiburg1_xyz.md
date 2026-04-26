# Brush vs RustGS TUM Freiburg1 XYZ Training-View Comparison

Date: 2026-04-26

Dataset:

```text
/Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap
```

## Models

RustGS baseline scene:

```text
RustGS/output/experiments/tum_psnr/final_30000_freeze_epoch4.ply
```

RustGS stabilized scene:

```text
RustGS/output/experiments/tum_visual/stabilized_lr_freeze4_30000.ply
```

Brush final scene:

```text
RustGS/output/experiments/brush_tum_compare/brush_30000.ply
```

## Brush Training Command

```sh
/Users/tfjiang/Projects/brush/target/release/brush \
  /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --total-train-iters 30000 \
  --export-every 30000 \
  --export-path /Users/tfjiang/Projects/RustScan/RustGS/output/experiments/brush_tum_compare \
  --export-name brush_30000.ply \
  --eval-every 30000
```

Brush exported 148876 splats.

## Training-View Render Comparison

Six input training views were rendered at scale 0.25:

- frame_0000.png
- frame_0030.png
- frame_0060.png
- frame_0090.png
- frame_0120.png
- frame_0150.png

RustGS was rendered with the RustGS evaluation renderer. Brush was rendered with Brush's own renderer through a local helper example in `/Users/tfjiang/Projects/brush/crates/brush-process/examples/render_training_views.rs`.

Combined visual comparison output:

```text
RustGS/output/experiments/brush_tum_compare/combined_training_view_comparison/
```

Each combined PNG is laid out as:

```text
GT | RustGS render | RustGS diff | brush render | brush diff
```

## Results

| Frame | RustGS PSNR | Brush PSNR | Brush - RustGS |
| --- | ---: | ---: | ---: |
| 0000 | 19.0903 | 20.9778 | +1.8875 |
| 0030 | 21.9620 | 21.6478 | -0.3142 |
| 0060 | 23.2353 | 23.6701 | +0.4348 |
| 0090 | 17.2860 | 17.6469 | +0.3609 |
| 0120 | 21.4723 | 20.2543 | -1.2180 |
| 0150 | 22.7668 | 23.5380 | +0.7712 |
| Mean | 20.9688 | 21.2892 | +0.3204 |

Important caveat:

Rendering the Brush `.ply` through the RustGS evaluator produced a misleading mean PSNR of 11.9891 dB. That path should not be used for the visual conclusion, because the two projects use different PLY metadata/property conventions. The valid Brush visual result above uses Brush's own renderer.

## Visual Conclusion

Brush is only modestly better on these six training views when rendered with its own renderer. It is not a night-and-day quality gap: both systems show similar artifacts around high-frequency object edges, the monitor silhouette, keyboard, and the moving/person-background region. Brush is visibly better on frame 0000 and slightly better on frames 0060, 0090, and 0150. RustGS is better on frames 0030 and 0120.

The user's complaint that the RustGS result "looks wrong" is still valid from the rendered strips: the error maps are heavy even where mean PSNR is above 20 dB. The next useful optimization target is not simply raising aggregate PSNR; it should focus on visual fidelity on training views, especially edge sharpness, exposure/brightness consistency, and dynamic/outlier regions such as frame 0090.

## Experiment: RustGS Default Topology Without Freeze

Command:

```sh
cargo run --release -p rustgs --bin rustgs -- train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_visual/default_topology_30000.ply \
  --iterations 30000 \
  --eval-after-train \
  --eval-json \
  --log-level info
```

Result:

| Metric | Value |
| --- | ---: |
| Final splats | 202002 |
| Final loss | 0.066538 |
| Mean PSNR, 6 training views at 0.25 scale | 16.2746 dB |
| Worst frame | frame 0120, 7.4384 dB |
| Second worst frame | frame 0030, 8.6531 dB |

Review output:

```text
RustGS/output/experiments/tum_visual/default_topology_30000_review/
```

Visual conclusion:

Removing the epoch-4 topology freeze is not an improvement. It grows to many more splats than the frozen run, but the rendered strips show large colored fog/splat pollution on the worst training views. This explains the PSNR collapse and confirms that the issue is not just insufficient Gaussian count. The next optimization target is to make continuing topology refinement safer.

Code hypothesis:

Brush applies an opacity and scale decay after every refine step (`opac_decay=0.004`, `scale_decay=0.002`). RustGS exposed an `opacity_reset_mode=decay` setting but did not actually apply this Brush-style post-refine decay. A follow-up patch adds `litegs.opacity_decay` and `litegs.scale_decay`, applies them to all splats after scheduled refine steps, and normalizes the refine quaternion before offsetting split splats.

## Optimization Experiment Log

All RustGS experiments below used release builds and the same COLMAP TUM dataset. Unless noted, PSNR is measured with the RustGS training-view evaluator at render scale 0.25 on frames 0, 30, 60, 90, 120, and 150.

| Experiment | Iterations | Key settings | Splats | Mean PSNR | Worst PSNR | Visual conclusion |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| Brush-style decay sweep | 1000 | `opacity_decay=0.004`, `scale_decay=0.002` | 19903 | 20.7506 | 13.1499 | Early signal acceptable, but frame 90 already weak. |
| Brush-style decay sweep | 3000 | same decay | 67412 | 21.3543 | 16.1314 | Best short-run decay result. |
| Brush-style decay sweep | 5000 | same decay | 107359 | 16.0379 | 6.4288 | Failed: heavy colored fog and severe frame 120 collapse. |
| Brush-ish defaults | 1000 | lower growth defaults | 9118 | 20.6079 | 12.8470 | Stable but under-densified. |
| Brush-ish defaults | 3000 | lower growth defaults | 9402 | 20.8016 | 14.8073 | Stable but not enough detail. |
| Threshold sweep | 3000 | `growth_grad_threshold=0.001` | 13100 | 21.0864 | 15.4780 | Stable but sparse. |
| Threshold sweep | 3000 | `growth_grad_threshold=0.0003` | 27086 | 21.1690 | 15.5357 | Stable, still below the best short run. |
| Threshold sweep | 5000 | `growth_grad_threshold=0.0003` | 39020 | 19.7428 | 15.8636 | Did not clear the 20 dB target. |
| Aggressive freeze + decay | 5000 | freeze epoch 4, decay on | 75318 | 15.7598 | 10.2884 | Failed: Brush-style global decay was too destructive in RustGS. |
| Aggressive freeze, no decay | 5000 | freeze epoch 4, decay off | 75858 | 21.1996 | 15.7690 | Good short-run result. |
| Aggressive freeze, no decay | 10000 | freeze epoch 4, decay off | 75304 | 16.5266 | 7.2229 | Failed: whole-frame purple wash appeared by 10k. |
| Restored defaults + freeze | 10000 | default growth restored, freeze epoch 4 | 75113 | 21.3430 | 16.4550 | Good and visually clean at 10k. |
| Restored defaults + freeze | 30000 | same as 10k | 74208 | 18.3567 | 14.4062 | Failed late: washed/ghosted overlays appeared by 30k. |
| Stabilized LR + freeze | 10000 | freeze epoch 4, final LR scale/rot/opacity/color = 10% at 10k | 76762 | 21.7477 | 16.3050 | Best 10k signal. No full-frame color pollution. |
| Stabilized LR + freeze | 30000 | same LR schedule, continue to 30k | 76981 | 21.0972 | 16.9482 | Passes the 30k target and keeps visuals aligned. |

The final stabilized command was:

```sh
cargo run --release -p rustgs --bin rustgs -- train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output output/experiments/tum_visual/stabilized_lr_freeze4_30000.ply \
  --iterations 30000 \
  --litegs-topology-freeze-after-epoch 4 \
  --lr-decay-iterations 10000 \
  --lr-scale-final 0.0005 \
  --lr-rotation-final 0.0001 \
  --lr-opacity-final 0.005 \
  --lr-color-final 0.00025 \
  --eval-after-train --eval-json --log-level info
```

The stabilized 30000-step model was also evaluated on all 180 training views at scale 0.25:

| Metric | Value |
| --- | ---: |
| Mean PSNR | 21.1622 dB |
| Median PSNR | 21.5816 dB |
| Worst PSNR | 15.2855 dB |
| Worst frame cluster | frames 76-88 |

Full-view review output:

```text
RustGS/output/experiments/tum_visual/stabilized_lr_freeze4_30000_review/
```

Shared six-frame review output:

```text
RustGS/output/experiments/tum_visual/stabilized_lr_freeze4_30000_sampled_review/
```

## Final Visual Conclusion

The final RustGS result is no longer qualitatively broken. On training views it matches the input camera viewpoints and preserves the major scene structure, monitor silhouette, desk layout, and person/background placement. The remaining visible gap versus Brush is mostly sharpness: Brush's own renderer is still a bit cleaner on keyboard keys, object edges, and the foreground monitor boundary, while RustGS has stronger blur and slight ghosting around high-frequency desktop clutter and the moving/person region.

On the shared six frames, the stabilized RustGS run is close to the earlier Brush 30000 render:

| Frame | Stabilized RustGS PSNR | Brush PSNR | Brush - RustGS |
| --- | ---: | ---: | ---: |
| 0000 | 19.6859 | 20.9778 | +1.2919 |
| 0030 | 22.4676 | 21.6478 | -0.8198 |
| 0060 | 22.2953 | 23.6701 | +1.3748 |
| 0090 | 16.9482 | 17.6469 | +0.6987 |
| 0120 | 21.6111 | 20.2543 | -1.3568 |
| 0150 | 23.5751 | 23.5380 | -0.0371 |
| Mean | 21.0972 | 21.2892 | +0.1920 |

Conclusion: the 30000-step RustGS target is met. Brush remains slightly sharper visually, but the final RustGS/Brush gap on the shared training views is now small enough that the next improvements should focus on renderer/detail fidelity rather than gross training instability.
