# RustSLAM 真实视频实验纪要（2026-03-28）

## 目的

验证当前 RustSLAM 是否已经满足下面这个前置条件：

> 用 iPhone 拍摄的视频作为输入，在 16GB Mac mini 上完成 SLAM，产出足够合理的稀疏点和相机位姿，再交给 RustGS 训练。

本次重点不是 3DGS 训练本身，而是确认 RustSLAM 的导出结果是否可信。

## 本轮代码结论

### 1. 稀疏点导出问题已经修复

当前 `slam_output.json` 不再只是关键帧壳子，而是会包含真实 `map_points`。这意味着 RustGS 至少能拿到真实的位姿和稀疏几何，而不是空场景。

### 2. 主要瓶颈不是“完全没有重定位”，而是三角化过于保守

在真实手持视频里，很多可用视角对的基线和夹角都偏小。旧阈值过于保守，导致：

- 可复用的 3D 点太少
- PnP 可用约束不足
- 一旦丢跟踪，只能频繁退回单目重启

这也是为什么之前看起来像“重定位不可靠”，但更底层的问题其实是 anchor keyframe 缺少足够的稀疏几何。

### 3. 本轮有效改动

- `RustSLAM/src/tracker/solver.rs`
  - PnP 会在 RANSAC 之外继续评估完整 DLT 候选。
  - 三角化阈值从 `min_angle=3.0 deg` 放宽到 `1.5 deg`。
  - 最小基线从 `0.1` 放宽到 `0.02`。
  - 修正三角化时相机中心和可见性判断。
- `RustSLAM/src/tracker/vo.rs`
  - 引入 anchor keyframe 缓存。
  - 重定位最小内点阈值与 tracking 阈值解耦。
  - 单目恢复阶段也允许继续补 anchor。
  - 增加 `RelocalizationStats`，把真实序列上的恢复路径跑数输出出来。
- `RustSLAM/src/cli/mod.rs`
  - CLI 已切换为 SLAM-only。
  - `slam_output.json` 与 checkpoint 会写出稀疏点。
  - 运行日志输出 `sparse_points` 和 `VO relocalization` 统计。

## 实验环境

- 机器：16GB Mac mini
- 构建：`release`
- 输入视频：`test_data/video/sofa.MOV`
- 视频信息：1920x1080，30 FPS，H.265/HEVC

代表性命令：

```bash
target/release/rustslam \
  --input test_data/video/sofa.MOV \
  --output output/sofa_smoke_anchor_diag5 \
  --output-format json \
  --max-frames 300

target/release/rustslam \
  --input test_data/video/sofa.MOV \
  --output output/sofa_full_anchor_diag6 \
  --output-format json
```

## 导出物核验

当前最佳 full run 导出物：

- `output/sofa_full_anchor_diag6/slam_output.json`
- `output/sofa_full_anchor_diag6/results.json`
- `output/sofa_full_anchor_diag6/checkpoints/pipeline.json`

已核验：

- `poses = 100`
- `map_points = 43004`
- `frame_count = 1731`
- `null position = 0`
- `non-numeric position = 0`
- `abs(position) > 50 = 0`

这说明导出的稀疏点至少在格式和数值范围上是自洽的，没有出现明显坏数据。

## 结果对比

### Smoke Run（300 帧）

| Run | Frames | Keyframes | Sparse Points | Tracking Success | Time |
|-----|--------|-----------|---------------|------------------|------|
| `output/sofa_smoke_fix3` | 300 | 31 | 996 | 50.3% | 37.97s |
| `output/sofa_smoke_anchor_diag5` | 300 | 31 | 4193 | 50.3% | 41.21s |

提升：

- 稀疏点增加约 `4.2x`
- tracking success 基本持平，但重定位开始真正发生，而不只是单目恢复

当前 improved smoke run 的重定位统计：

- `lost_events=148`
- `direct_retrack=2/147`
- `anchor_store=11`
- `anchor_success=5/146`
- `anchor_candidates_tested=709`
- `monocular_reinit=141/141`
- `cached_anchors=11`

### Full Run（1731 帧）

| Run | Frames | Keyframes | Sparse Points | Tracking Success | Time |
|-----|--------|-----------|---------------|------------------|------|
| `output/sofa_full_release_fix` | 1731 | 100 | 8946 | 50.0% | 314.71s |
| `output/sofa_full_anchor_diag6` | 1731 | 100 | 43004 | 50.9% | 688.84s |

提升：

- 稀疏点增加约 `4.8x`
- tracking success 从 `50.0%` 提升到 `50.9%`
- 当前版本已经能在整段视频中多次成功走到 anchor-based relocalization，而不只是频繁退回纯单目重启

当前 improved full run 的重定位统计：

- `lost_events=850`
- `direct_retrack=12/846`
- `anchor_store=67`
- `anchor_success=19/838`
- `anchor_candidates_tested=15944`
- `monocular_reinit=819/819`
- `cached_anchors=24`

## 如何判断输出“合理”

### 相机位姿

可以先看这几个硬指标：

- 位姿数量是否与关键帧数量一致
- 位姿时间顺序是否与视频帧顺序一致
- 相邻位姿不应频繁出现非常大的跳变
- 轨迹整体方向应和拍摄路径一致，而不是突然折返或远跳

### 稀疏点

至少应满足：

- 没有 `NaN`、`null` 或明显爆炸的坐标
- 点云大致围绕被拍摄物体和相机轨迹分布
- 不应大面积漂到非常远的空间位置
- 从早期关键帧到后期关键帧，点云应逐步变密，而不是长期几乎为空

### 联合判断

最重要的是一起看：

- 如果位姿连续，但稀疏点始终很少，RustGS 初始几何会太弱
- 如果稀疏点很多，但轨迹跳变大，RustGS 会学到错误相机
- 只有“轨迹基本连续 + 稀疏点足够多 + 导出值稳定”时，才适合进入高斯训练

## 当前判断

当前结论是：

1. RustSLAM 已经从“导出结果明显不对”进入到“可以产出真实稀疏点和位姿”的阶段。
2. 这轮改动已经证明 anchor-based relocalization 在真实 iPhone 视频上开始生效，不再只是理论路径。
3. 但 `tracking_success=50.9%` 仍然偏低，说明长序列中还有大量片段无法稳定连续跟踪。
4. 因此，这一版更适合作为“继续改进 RustSLAM 稳定性”的里程碑，而不是直接宣称已经完全满足 RustGS 训练前置条件。

## 对 RustGS 的意义

正面影响：

- `slam_output.json` 现在有真实 `poses`
- `slam_output.json` 现在有真实 `map_points`
- RustGS 不再需要面对一个几乎空的 SLAM 初始化

仍然存在的风险：

- 长序列跟踪中断仍然过多
- 单目恢复片段仍然很多，几何一致性可能不足
- 对最终高斯训练质量的上限仍有约束

## 下一步建议

优先级建议如下：

1. 继续提升长序列 tracking 和 relocalization 成功率，目标不是多一点点稀疏点，而是减少中后段频繁丢跟踪。
2. 把当前 `VO relocalization` 统计固化进结果文件，而不只打印日志，方便后续自动回归比较。
3. 在 RustViewer 或独立脚本里增加轨迹和稀疏点快速可视化，降低人工判断成本。
4. 等 tracking success 明显高于当前水平后，再把这套 `slam_output.json` 作为 RustGS 默认训练入口。
