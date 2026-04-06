# RustGS 对标 Brush 的 3DGS 训练重构计划

**日期**: 2026-04-06
**参考实现**: `~/Projects/Brush` (Rust/WGPU 3DGS)

## Context

RustGS 当前训练无法收敛（PSNR不涨），尝试对标 LiteGS 改写也走入死胡同。根本原因是**训练管线逻辑有多处关键偏差**，而非数学基础（投影、SH、alpha blending 等底层数学是正确的）。

本计划参照 `~/Projects/Brush`（一个成熟的 Rust/WGPU 3DGS 实现），将 RustGS 的训练管线重构为标准 3DGS 流程。

---

## 诊断：RustGS vs Brush 关键差异

### 1. Densification 严重有误（training_pipeline.rs:170-220）

**当前 RustGS**:
- Clone: `cloned.x += 0.001` — 固定偏移，无方向感知
- Split: 只修改 `scale_log[0]`（X轴）— 应该缩小所有维度
- 选择标准: `max_scale > 0.3` 就split — 缺少梯度门槛

**Brush 正确实现**:
- Clone: 对位置添加 covariance-scaled 的随机偏移 `offset = R(q) * (N(0,1) * scales)`
- Split: 将最大维度 scale 乘 0.5, opacity 用 `1 - sqrt(1-old_opac)` 分配
- 选择: 基于 xy-plane 梯度 norm > `growth_grad_threshold(0.003)` + 随机选 20%

### 2. SSIM 实现错误（training_pipeline.rs:261-309）

**当前 RustGS**: 计算全图全局 mean/variance — 完全丧失局部结构比较能力
**Brush**: 11x11 Gaussian window(σ=1.5) 的**逐像素局部** SSIM

### 3. 缺少关键训练技巧

| 技巧 | Brush | RustGS |
|------|-------|--------|
| 随机背景色噪声 | base + U(-0.1, 0.1) | 无 |
| 低opacity Gaussian位置噪声 | inv_opacity^150 * noise | 无 |
| 连续 opacity decay | 每refine步 `opac -= 0.004 * (1-t)` | 周期性 reset 到 0.5 |
| 连续 scale decay | 每refine步 `scale *= (1 - 0.002*(1-t))` | 无 |
| Position LR × scene_scale | `lr_mean * median_scene_scale` | 固定 lr |
| SH rest LR = DC/20 | `lr_coeffs_sh / 20` | 无分级 |
| Adam epsilon | 1e-15 | 未确认 |

### 4. 超参差异

| 参数 | Brush | RustGS |
|------|-------|--------|
| lr_mean | 2e-5 (× scene_scale) | 1.6e-4 |
| lr_scale | 7e-3 → 5e-3 | 5e-3 |
| lr_rotation | 2e-3 | 1e-3 |
| lr_opacity | 0.012 | 0.05 (4x!) |
| lr_sh_dc | 2e-3 | 2.5e-3 |
| refine_every | 200 iters | 100 iters |
| growth_stop | 15000 | 无 |
| total_iters | 30000 | 3000 |
| prune_threshold | 1/255 ≈ 0.004 | 0.05 (12x!) |
| densify_grad_threshold | 0.003 | 0.0002 |

### 5. 两条并行代码路径

- `training_pipeline.rs`: CPU路径，densification/SSIM都是错的
- `metal_trainer.rs`: GPU Metal路径，实现了LiteGS变体

需要统一到一条**对标Brush的正确路径**。

---

## 实施计划

### Phase 1: 新建 Brush-compatible 训练配置

**文件**: `RustGS/src/training/mod.rs`, `RustGS/src/training/metal_trainer.rs`

- 新增 `TrainingProfile::BrushV1` 配置 profile
- 对齐 Brush 默认超参:
  ```
  lr_mean: 2e-5, lr_scale: 7e-3, lr_rotation: 2e-3
  lr_opacity: 0.012, lr_sh_dc: 2e-3, lr_sh_rest: 1e-4
  total_iters: 30000, refine_every: 200, growth_stop: 15000
  growth_grad_threshold: 0.003, prune_opacity: 1/255
  opac_decay: 0.004, scale_decay: 0.002
  ```
- Position LR 乘以 scene median scale（从初始点云计算）

### Phase 2: 修复 Densification（最关键）

**文件**: `RustGS/src/training/metal_trainer.rs` (densify 相关方法)

新增 Brush-style densification:
1. **梯度追踪**: 累积每个 Gaussian 在 xy 平面的 gradient norm（已有 `update_gaussian_stats`）
2. **Refine 触发**: 每 200 iter 执行一次（growth_stop=15000 后停止）
3. **Prune**: opacity < 1/255、scale < 1e-10、scale > 100x median、position out of bounds
4. **Clone dead**: 随机采样高可见度 Gaussian 替补被 prune 的
5. **Split/Grow**: 
   - 选择 xy_grad_norm > 0.003 的 top 20%
   - 缩小最大维度 scale × 0.5
   - Opacity: `new_opac = 1 - sqrt(1 - old_opac)`
   - Position offset: `R(q) * (N(0,1) * scales)` 
   - 原 Gaussian position -= offset, 新 clone position += offset

### Phase 3: 修复 SSIM 计算

**文件**: `RustGS/src/training/metal_loss.rs`

Metal trainer 路径中的 `ssim_gradient()` 可能已经是局部窗口版本（需确认）。
CPU 路径 `training_pipeline.rs` 的 `compute_ssim_loss` 是全局版本，需替换为 11×11 Gaussian window 的 separable convolution 版本。

### Phase 4: 添加训练稳定技巧

**文件**: `RustGS/src/training/metal_trainer.rs`

1. **随机背景色**: 渲染时 background = base(0,0,0) + U(-0.1, 0.1), clamp [0,1]
   - GT 图片也需在相同随机背景上合成（如有 alpha）
2. **连续 opacity/scale decay**: 在每次 refine 时:
   ```
   t_shrink = 1 - iter/total_iters
   new_opac = sigmoid(raw_opac) - opac_decay * t_shrink
   raw_opac = inv_sigmoid(clamp(new_opac))
   scales *= (1 - scale_decay * t_shrink)
   ```
3. **位置噪声注入**: 对低 opacity Gaussians 添加 noise
   ```
   noise_weight = inv_opacity^150 * visible * lr_mean * noise_scale
   position += clamp(N(0,1) * noise_weight, -median_scale, median_scale)
   ```

### Phase 5: Scene Scale 感知

**文件**: `RustGS/src/training/data_loading.rs`, `RustGS/src/training/metal_trainer.rs`

1. 从初始点云计算 `median_scene_scale` = 3 × average nearest-neighbor camera distance（至少 1m）
2. `lr_mean_effective = lr_mean * median_scene_scale`
3. Scale pruning bound = 100 × median_scene_scale

### Phase 6: 超参调优与验证

- 用 COLMAP 数据集（而非 TUM/SLAM）做纯 3DGS 训练测试
- 对比 Brush 在相同数据集上的 PSNR 曲线
- 目标: 30k iter 后 PSNR ≥ 25 dB（典型场景）

---

## 实施优先级

1. **P0 - Phase 2 (Densification)** — 这是 PSNR 不涨的最可能根因
2. **P0 - Phase 1 (超参)** — 错误的 LR 会导致训练不稳定
3. **P1 - Phase 4 (decay + noise)** — 防止 floater, 帮助收敛
4. **P1 - Phase 5 (scene scale)** — 场景尺度不同时保持训练稳定
5. **P2 - Phase 3 (SSIM)** — Metal 路径可能已正确，优先级次之
6. **P2 - Phase 6 (验证)** — 持续对比

## 关键文件列表

| 文件 | 作用 |
|------|------|
| `RustGS/src/training/metal_trainer.rs` | 主训练循环、densify/prune、optimizer |
| `RustGS/src/training/mod.rs` | TrainingConfig、Profile 定义 |
| `RustGS/src/training/training_pipeline.rs` | CPU 训练路径（需修复或废弃） |
| `RustGS/src/training/metal_loss.rs` | SSIM gradient、L1 loss |
| `RustGS/src/training/metal_runtime.rs` | Metal GPU kernels (forward/backward/adam) |
| `RustGS/src/training/data_loading.rs` | 数据加载、Gaussian 初始化 |
| `RustGS/src/training/density_controller.rs` | LiteGS stats（可能重构） |
| `RustGS/src/diff/diff_splat.rs` | TrainableGaussians 结构 |

## 验证方案

1. `cargo build --release --features gpu` 编译通过
2. 在 COLMAP 格式数据集上运行 30k iter 训练
3. 每 1000 iter 输出 PSNR, 观察收敛曲线
4. 与 Brush 在同一数据集上的 PSNR 做对比（目标差距 < 2dB）
5. 检查 Gaussian 数量增长曲线是否合理（应先增后稳）
