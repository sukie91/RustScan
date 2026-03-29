# COLMAP + RustGS Pipeline

本文档描述如何使用 COLMAP 作为 RustSLAM 的备选方案，直接用 COLMAP 的位姿和稀疏点训练 RustGS。

## 背景

RustSLAM 当前在真实 iPhone 视频上跟踪不稳定：
- 跟踪成功率低（11% vs COLMAP 100%）
- 轨迹跳变严重（max 17m vs COLMAP 1.07m）
- 稀疏点不足（3.7K vs COLMAP 100K）

COLMAP 作为成熟的 SfM 工具，可以提供可靠的位姿和稀疏点，用于：
1. 验证后续 RustGS 训练和 mesh 提取 pipeline
2. 作为 ground truth 诊断 RustSLAM 问题
3. 直接生产可用的高质量 3DGS 重建

## Pipeline

```
iPhone 视频 (sofa.MOV)
    ↓ ffmpeg (提取帧 @ 6fps)
COLMAP SfM
    ↓ (位姿 + 稀疏点)
colmap_output.json
    ↓ colmap_to_rustgs.py
training_dataset.json
    ↓ RustGS Metal 训练
scene.ply (3DGS 模型)
    ↓ TSDF + Marching Cubes
mesh.obj + mesh.ply
```

## 使用方法

### 1. 从视频提取帧

```bash
# 6fps = 每5帧提取1帧（适合室内扫描）
ffmpeg -i test_data/video/sofa.MOV -vf "fps=6" -q:v 2 output/colmap_sofa/images/frame_%06d.jpg
```

### 2. 运行 COLMAP

```bash
scripts/run_colmap.sh
```

这会自动：
- 提取 SIFT 特征（GPU 加速）
- 顺序匹配（适合视频序列）
- 稀疏重建
- 导出 `colmap_output.json`

### 3. 转换为 RustGS 格式

```bash
python3 scripts/colmap_to_rustgs.py \
  --colmap output/colmap_sofa/colmap_output.json \
  --images output/colmap_sofa/images \
  --output output/colmap_sofa/training_dataset.json \
  --max-frames 50  # 限制帧数进行快速测试
```

### 4. 训练 RustGS

```bash
RUSTGS_SKIP_METAL_MEMORY_GUARD=1 \
target/release/rustgs train \
  --input output/colmap_sofa/training_dataset.json \
  --output output/colmap_sofa/scene.ply \
  --iterations 3000 \
  --max-initial-gaussians 50000 \
  --metal-render-scale 0.5
```

参数说明：
- `--iterations`: 训练迭代次数（典型值 3000-30000）
- `--max-initial-gaussians`: 初始高斯数量上限
- `--metal-render-scale`: 渲染分辨率缩放（0.5 = 960x540）
- `RUSTGS_SKIP_METAL_MEMORY_GUARD=1`: 跳过内存限制检查

### 5. 导出 Mesh（可选）

训练完成后，可以用 RustViewer 或独立的 mesh 提取工具将 3DGS 转换为 mesh。

## COLMAP vs RustSLAM 对比

使用 `scripts/compare_colmap_rustslam.py` 快速对比：

```bash
python3 scripts/compare_colmap_rustslam.py \
  --colmap output/colmap_sofa/colmap_output.json \
  --rustslam output/sofa_balanced_sanity/slam_output.json
```

典型输出：

```
COLMAP:
  Registered poses: 346
  Median inter-frame jump: 0.12m
  Max jump: 1.07m

RustSLAM:
  Keyframes: 8
  Median inter-keyframe jump: 1.41m
  Max jump: 17.0m
```

## 文件格式

### colmap_output.json

```json
{
  "poses": [
    {
      "frame_id": 5,
      "image_name": "frame_000001.jpg",
      "center": [0.0, 0.0, 0.0],
      "qvec": [qx, qy, qz, qw],
      "tvec": [tx, ty, tz]
    }
  ],
  "map_points": [
    {
      "position": [x, y, z],
      "color": [r, g, b]
    }
  ],
  "cameras": {
    "1": {
      "model": "PINHOLE",
      "width": 1920,
      "height": 1080,
      "params": [fx, fy, cx, cy]
    }
  }
}
```

### training_dataset.json

```json
{
  "intrinsics": {
    "fx": 2304.0,
    "fy": 2304.0,
    "cx": 960.0,
    "cy": 540.0,
    "width": 1920,
    "height": 1080
  },
  "poses": [
    {
      "frame_id": 5,
      "image_path": "output/colmap_sofa/images/frame_000001.jpg",
      "pose": {
        "rotation": [x, y, z, w],
        "translation": [x, y, z]
      },
      "timestamp": 0.167
    }
  ],
  "initial_points": [
    [[x, y, z], [r, g, b]]
  ]
}
```

## 已知问题

1. **内存限制** - Metal 训练器会限制高斯数量以避免 OOM
   - 解决：设置 `RUSTGS_SKIP_METAL_MEMORY_GUARD=1` 或降低 `--max-initial-gaussians`

2. **COLMAP 优化警告** - "Matrix not positive definite"
   - 这是正常的，COLMAP 仍然可以成功重建

3. **旋转冻结** - Metal 后端当前不支持旋转优化
   - 仅影响最终质量，不影响验证 pipeline

## 下一步

1. 完成完整视频训练（3000-30000 迭代）
2. 用 RustViewer 可视化 3DGS 场景
3. 导出 mesh 并用 RustMesh 处理
4. 对比 RustSLAM 位姿与 COLMAP ground truth
5. 修复 RustSLAM 的 VO 问题

## 参考

- COLMAP 文档: https://colmap.github.io/
- RustGS 训练参数: `rustgs train --help`
- RustSLAM 设计: `docs/RustSLAM-DESIGN.md`