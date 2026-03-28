# RustSLAM

<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.75+-dea584?style=for-the-badge&logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License">
</p>

RustSLAM 是 RustScan 里的 SLAM 阶段。自 2026-03-28 起，它只负责视觉 SLAM，不再承担 3D Gaussian Splatting 的训练、渲染或网格生成。

## 当前职责

- 从视频或数据集读取图像帧
- 执行稀疏特征跟踪、PnP、重定位和局部建图
- 导出相机位姿、关键帧 RGB 和稀疏点到 `slam_output.json`
- 作为 RustGS 的前置阶段，为后续高斯训练提供输入

## 不再负责的内容

- 高斯初始化
- 3DGS 训练
- 高斯渲染
- Mesh 提取

这些能力现在分别由 `RustGS` 和 `RustMesh` 负责。

## 当前事实来源

- 最新真实视频实验结论见 [RustSLAM-Experiment-2026-03-28.md](./RustSLAM-Experiment-2026-03-28.md)
- 历史设计思路见 [RustSLAM-DESIGN.md](./RustSLAM-DESIGN.md)

如果两者冲突，以实验纪要和当前代码行为为准。

## 关键变化

- `RustSLAM/src/tracker/solver.rs`
  - PnP 在 RANSAC 之后会继续评估完整 DLT 候选，避免更好的绝对位姿被较差的随机样本遮蔽。
  - 三角化阈值放宽到更适合手持 iPhone 视频的小基线场景，并修正了相机中心/可见性判断。
- `RustSLAM/src/tracker/vo.rs`
  - 新增轻量级 anchor keyframe 缓存和重定位统计。
  - 将重定位最小内点阈值与常规 tracking 阈值解耦。
  - 单目恢复阶段也允许持续补充 anchor，避免 anchor 池很快枯竭。
- `RustSLAM/src/cli/mod.rs`
  - CLI 切换为 SLAM-only 输出。
  - `slam_output.json` 和 checkpoint 中都包含真实稀疏点。
  - 运行日志会输出 `sparse_points` 和 `VO relocalization` 统计，便于评估真实视频效果。

## 快速开始

在仓库根目录执行：

```bash
cargo build -p rustslam --release
target/release/rustslam \
  --input test_data/video/sofa.MOV \
  --output output/sofa_full_anchor_diag6 \
  --output-format json
```

主要输出：

- `output/<run>/results.json`
- `output/<run>/slam_output.json`
- `output/<run>/checkpoints/pipeline.json`

交给 RustGS 的核心文件是 `slam_output.json`。

## 当前结论

- RustSLAM 现在已经能在真实 iPhone 视频上导出非空、有限值的稀疏点和相机位姿。
- `sofa.MOV` 全量运行的当前最好结果为 `1731` 帧、`100` 个位姿、`43004` 个稀疏点。
- 但整段视频的 tracking success 仍只有 `50.9%`，说明这版已经明显改善，却还没有达到“直接放心交给 RustGS 做最终训练”的理想状态。

## 已知限制

- 目前仍是单目 RGB 稀疏 SLAM，不包含基于输入视频的稠密深度重建。
- 对纹理弱、视差小、长时间遮挡的视频，重定位仍不够稳定。
- 更高的稀疏点数量伴随更长的全量运行时间，这是当前鲁棒性换来的代价。

## 参考

- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [pySLAM](https://github.com/luigifreda/pyslam)

## License

MIT License
