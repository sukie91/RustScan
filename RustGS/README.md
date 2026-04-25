# RustGS

RustGS is the RustScan 3D Gaussian Splatting training crate. It provides a wgpu-based
training path, host-side PLY scene I/O, dataset loaders for COLMAP/TUM/serialized
`TrainingDataset` inputs, and simple evaluation/rendering utilities.

## Features

- `default = ["gpu", "cli"]`: builds the library and `rustgs` CLI.
- `gpu`: enables Burn/wgpu training, PLY scene load/save, and evaluation rendering.
- `cli`: enables the `rustgs` binary and CLI-only dependencies.
- `gpu-wgpu`: compatibility alias for `gpu`.

The library can be checked without optional features:

```sh
cargo check --no-default-features --all-targets
```

Examples and tests that require GPU APIs are gated through Cargo features.

## Training

Training currently requires sparse initialization points. COLMAP inputs must include
`points3D.bin` or `points3D.txt`; serialized `TrainingDataset` JSON inputs must include
`initial_points`.

```sh
cargo run --bin rustgs -- train \
  --input /path/to/colmap-or-training-dataset \
  --output output/scene.ply \
  --iterations 30000
```

Useful flags:

- `--max-frames` and `--frame-stride` select directory-backed dataset frames.
- `--render-scale` controls training target resolution and is validated in `[0.0625, 1.0]`.
- `--eval-after-train` runs a post-training PSNR evaluation pass.
- `--eval-json` prints the evaluation summary as JSON.

## Rendering

Render a saved scene with a camera JSON:

```sh
cargo run --bin rustgs -- render \
  --input output/scene.ply \
  --camera camera.json \
  --output output/render.png
```

`camera.json` format:

```json
{
  "intrinsics": {
    "fx": 500.0,
    "fy": 500.0,
    "cx": 320.0,
    "cy": 240.0,
    "width": 640,
    "height": 480
  },
  "pose": {
    "rotation": [0.0, 0.0, 0.0, 1.0],
    "translation": [0.0, 0.0, 0.0]
  },
  "pose_is_world_to_camera": false
}
```

By default, `pose` is interpreted as camera-to-world, matching `ScenePose`. Set
`pose_is_world_to_camera` to `true` when the pose is already a view transform.

## Verification

```sh
cargo fmt --check
cargo check --all-targets
cargo check --no-default-features --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo test --test integration_test -- --ignored
```

The ignored integration tests run tiny wgpu training jobs and are useful smoke tests before
changing the training loop.
