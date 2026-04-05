# RustGS LiteGS Parity Progress 2026-04-05

> Updated: 2026-04-05
> Scope: `rustgs-opt` worktree
> Status: In Progress

本文档记录 RustGS 相对 Mirror/LiteGS 的当前对齐进展、已完成改动、待开发任务和建议优先级。

## 当前结论

RustGS 的 LiteGS 对齐工作已经从“规划阶段”进入“实现阶段”。

截至当前，以下几类能力已经补齐或明显推进：

- COLMAP 数据输入与 CLI 自动识别已打通
- scale regularization 已对齐到 LiteGS 的 activated scale 语义
- clustered 路径的 `cluster_assignment` 过期问题已修复
- `learnable_viewproj` 不再只是占位结构，训练步中已经可以执行 pose finite-difference 梯度与 pose Adam 更新

仍然阻塞“更高程度训练语义一致性”的核心缺口主要是：

- `sparse_grad=true` 仍未落地
- `enable_depth=true` 仍被配置层拦截
- clustered 路径仍不是 LiteGS 那种 cluster + sparse optimizer 的完整组合
- `learnable_viewproj` 当前是 finite-difference fallback，功能可用但性能不够接近 LiteGS

## 已完成项

### 1. COLMAP 输入支持

已完成的对齐点：

- 支持从 COLMAP 目录直接加载训练数据
- CLI/库层自动识别输入来源：COLMAP 目录、TUM 目录或训练 JSON
- 修正了 COLMAP 外参语义：将 world-to-camera 变换反转为 RustGS 训练所需的 camera-to-world pose

结果：

- RustGS 不再局限于 TUM / 手工 JSON 输入
- COLMAP 训练路径与 LiteGS 常见使用方式更接近

涉及文件：

- `RustGS/src/io/colmap_dataset.rs`
- `RustGS/src/lib.rs`
- `RustGS/src/bin/rustgs.rs`

### 2. Scale Regularization 对齐 LiteGS

LiteGS 正则化的是 activated scale，而不是 log-scale 参数本身。

当前已完成：

- loss term 改为对 `exp(log_scale)^2` 求均值
- 额外梯度改为与该 loss 语义一致的解析梯度
- 补充单测验证数值正确性

结果：

- RustGS 这一项 loss 的数值语义与 LiteGS 一致
- 后续做 loss parity 或训练曲线比对时，这一项不再是显著偏差源

涉及文件：

- `RustGS/src/training/metal_trainer.rs`

### 3. Cluster Assignment 同步修复

此前 clustered 路径存在一个训练期行为问题：

- `cluster_assignment` 只在初始化时创建
- Gaussian 位置更新后 AABB 不刷新
- densify / prune / Morton reorder 后 cluster 索引也不重建

当前已完成：

- 每个训练 step 会基于当前位置刷新 cluster AABB
- topology 发生变化时会重建 cluster assignment
- 增加了位置更新与 topology 变更两类测试

结果：

- frustum culling 不再持续使用过期 cluster 数据
- clustered 路径的行为更加稳定

涉及文件：

- `RustGS/src/training/metal_trainer.rs`
- `RustGS/src/training/clustering.rs`

### 4. Learnable ViewProj 从“占位”推进到“可训练”

此前这条能力只完成了部分结构：

- `PoseEmbeddings` 已存在
- render 前向可以切到 embedding 相机
- 但 CLI 里被硬编码为 `false`
- 训练步未执行 pose 更新
- forward / backward 使用的相机对象也不一致

当前已完成：

- CLI 暴露 `--litegs-learnable-viewproj`
- CLI 暴露 `--litegs-lr-pose`
- `training_step` 的 forward / backward / rotation grad / SH color grad 全部统一使用 `render_camera`
- 增加基于 loss 重算的 pose finite-difference 梯度
- 将 pose gradient 接入 `PoseEmbeddings::adam_step`
- 修复 `PoseEmbeddings::adam_step` 中原有的 Candle 标量广播 / 归一化问题

结果：

- `learnable_viewproj` 不再只是初始化 pose embedding
- 显式开启后，训练步会真实更新当前 frame 的 pose 参数

限制：

- 当前仍是 finite-difference fallback
- 功能上可用，但速度和 LiteGS 的原生 autograd / sparse pose update 仍有明显差距

涉及文件：

- `RustGS/src/bin/rustgs.rs`
- `RustGS/src/training/mod.rs`
- `RustGS/src/training/metal_trainer.rs`
- `RustGS/src/training/pose_embedding.rs`

## 已验证内容

当前已跑通或新增的定向验证包括：

- `cargo fmt -p rustgs`
- `cargo test -p rustgs test_load_colmap_dataset_text_format`
- `cargo test -p rustgs test_load_training_dataset_with_source_detects_colmap_directory`
- `cargo test -p rustgs test_load_colmap_with_stride`
- `cargo test -p rustgs litegs_scale_regularization_uses_activated_scales`
- `cargo test -p rustgs sync_cluster_assignment_`
- `cargo test -p rustgs topology_update_densifies_and_prunes_with_matching_adam_state`
- `cargo test -p rustgs test_pose_embeddings_adam_step_updates_selected_frame`
- `cargo test -p rustgs pose_parameter_grads_returns_tensor_pair`
- `cargo test -p rustgs train_command_parses_litegs_flags_and_builds_nested_config`

说明：

- 当前验证以定向单测和局部回归为主
- 还没有做完整的 LiteGS parity fixture 级别回归

## 待开发任务

### P0: Sparse Gradient Optimizer 语义落地

当前状态：

- `sparse_grad=true` 仍被配置校验拦截
- RustGS 现有 Adam / fused Adam 还是全量参数更新
- 与 LiteGS 的 visible-only sparse update 语义不一致

需要完成的事情：

- 让 optimizer 接收 visible primitive / visible chunk 索引
- 只对可见参数执行 moment 更新与参数更新
- clustered 路径和非 clustered 路径都明确 sparse update 行为
- 补充 parity 测试与性能对比

这是当前最重要的剩余缺口。

### P0: Depth Loss Parity

当前状态：

- `enable_depth=true` 仍被 `validate_litegs_mac_v1_config()` 拦截
- 训练主循环里已有部分 depth loss 路径，但没有被正式放开

需要完成的事情：

- 核对 RustGS 当前 depth loss 的权重、mask 和 LiteGS 是否一致
- 放开配置校验
- 补充开启 depth supervision 的训练回归

### P1: Learnable ViewProj 性能版本

当前状态：

- 功能已接通
- 但当前实现使用 finite-difference loss 重算，额外渲染次数多

需要完成的事情：

- 评估 analytical pose gradient 是否可接入
- 或改成更低开销的近似更新方案
- 如果继续保留 FD 路径，至少要在配置和日志中明确其性能成本

### P1: Clustered Path 进一步接近 LiteGS

当前状态：

- cluster assignment 同步问题已经修复
- 但仍然是 RustGS 自己的 CPU mask + Metal render 路径

需要完成的事情：

- 评估 cluster culling 与 sparse optimizer 的配合方式
- 明确 clustered path 的更新单位是 cluster 还是 primitive
- 决定是否需要更接近 LiteGS 的 compacted tensor / visible chunk 结构

### P2: Render 子命令 / 周边能力补齐

当前状态：

- 核心训练 parity 优先级高于 CLI 周边功能
- 一些周边能力仍不完整，例如 render 子命令仍未完全形成 LiteGS 对应体验

建议：

- 在训练语义核心差距补齐后再回头处理

## 建议开发顺序

推荐按下面顺序继续推进：

1. `sparse_grad` 真正落地
2. `enable_depth` 放开并校准 loss parity
3. clustered 路径与 sparse update 的组合语义
4. `learnable_viewproj` 从 finite-difference 过渡到更高效实现
5. 周边 CLI / render 能力补齐

原因：

- 前两项直接影响 LiteGS 训练行为的一致性
- 后三项更偏性能、规模化使用和体验完善

## 风险与注意事项

- 当前 worktree 处于脏状态，存在多处并行中的本地改动
- 后续继续做 parity 时，应避免覆盖与本次无关的用户改动
- `learnable_viewproj` 现阶段虽然可训练，但不应默认开启，除非明确接受额外训练开销
- clustered 路径已经修掉明显 bug，但不代表已经达到 LiteGS 的最终语义

## 相关文档

- `docs/RustGS-LiteGS-Parity-Roadmap.md`
- `docs/RustGS-Training-Report.md`

