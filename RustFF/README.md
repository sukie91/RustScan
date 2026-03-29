# RustFF - FeedForward 3D Reconstruction

FeedForward neural network inference for 3D reconstruction, providing per-frame camera pose estimation and dense depth prediction without traditional feature matching or bundle adjustment.

## Overview

RustFF is a Rust library that runs inference of feedforward 3D reconstruction models (starting with Spann3R) via ONNX, producing globally-consistent camera poses and dense pointmaps from image sequences.

```
RustScan workspace
├── RustMesh   - Mesh processing (half-edge data structure)
├── RustSLAM   - Traditional Visual SLAM (features + optimization)
├── RustGS     - 3D Gaussian Splatting training
├── RustFF     - FeedForward inference (this crate) ← NEW
└── RustViewer - 3D visualization
```

## Architecture

```
Video Frame Sequence
    ↓
ONNX Encoder (CroCo ViT-Large)
    ↓ feature tokens [B, T, D]
Spatial Memory Bank (explicit, bounded)
    ↓ concatenate current + memory features
ONNX Decoder (DUSt3R-style pointmap head)
    ↓ pointmap [B, H*W, 3] + confidence [B, H*W]
Pose Extraction (Procrustes alignment)
    ↓
Output: SE3 Pose + Dense Depth Map
```

## Status

| Component | Status |
|-----------|--------|
| Crate structure | ✅ |
| Data types (InferenceConfig, PointmapResult) | ✅ |
| Spann3RInference skeleton | ✅ |
| ONNX encoder export script | ✅ |
| ONNX decoder export script | ⚠️ placeholder |
| ORT inference backend | ⚠️ skeleton |
| Procrustes pose extraction | ⏳ |
| RustSLAM pipeline integration | ⏳ |

## Dependencies

- [Spann3R](https://github.com/HengyiWang/spann3r) — Python model + ONNX export
- [ONNX Runtime](https://onnxruntime.ai/) — Inference runtime (`ort` crate)
- [DUSt3R](https://github.com/naver/dust3r) — Base model weights

## Quick Start

### 1. Export Models (Python)

```bash
# Setup Spann3R environment
cd spann3r
pip install -r requirements.txt onnx onnxruntime

# Download weights
mkdir checkpoints && cd checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
# Download Spann3R v1.01 from Google Drive link in README

# Export to ONNX
python ../RustFF/scripts/export_onnx.py \
    --checkpoint checkpoints/spann3r_v1.01.pth \
    --output ../RustFF/models/
```

### 2. Run Inference (Rust)

```rust,no_run
use rustff::Spann3RInference;
use image::open;

fn main() -> anyhow::Result<()> {
    let mut model = Spann3RInference::from_onnx(
        "models/encoder.onnx",
        "models/decoder.onnx",
    )?;

    for i in 0..100 {
        let frame = open(format!("frames/frame_{:04}.png"))?;
        let result = model.process_frame(&frame)?;
        println!("Frame {}: pose={:?}", i, result.pose);
        println!("  depth range: [{:.2}, {:.2}]",
            result.depth_map().iter().cloned().fold(f32::INFINITY, f32::min),
            result.depth_map().iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        );
    }

    Ok(())
}
```

## Model Details

### Spann3R

- **Backbone**: CroCo ViT-Large (~300M params)
- **Memory**: Explicit spatial memory bank (bounded, sliding window)
- **Processing**: Online / sequential (no batch global alignment)
- **Output**: Per-frame camera pose [4x4] + dense pointmap [H×W×3]
- **Speed**: ~4-8 fps on V100 for 512px input

### Why Spann3R?

| Method | Memory | Speed (200 imgs) | Online | Code |
|--------|--------|-------------------|--------|------|
| VGGT | ~48GB | slow | ❌ | ✅ |
| Light3R-SfM | ~16GB | 33s | ❌ | ❌ |
| MASt3R-SfM | ~32GB | 27min | ❌ | ✅ |
| **Spann3R** | ~8GB | ~30s | ✅ | ✅ |

## License

MIT
