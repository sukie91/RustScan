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
- `sparse_grad=true` 已具备 visible-only sparse optimizer 的第一版语义
- `enable_depth=true` 已允许通过 LiteGS profile 正式开启
- `learnable_viewproj` 不再只是占位结构，训练步中已经可以执行 pose finite-difference 梯度与 pose Adam 更新

仍然阻塞“更高程度训练语义一致性”的核心缺口主要是：

- 还没有做完整的 LiteGS parity fixture / 训练曲线级别回归
- depth loss 目前已放开，但权重、mask 和 LiteGS 之间还缺显式标定
- clustered 路径虽然已经能和 sparse update 组合，但仍不是 LiteGS 那种 cluster + sparse optimizer 的最终形态
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

### 5. Sparse Gradient Optimizer 第一版落地

此前这条能力还停留在“配置预留”阶段：

- `sparse_grad=true` 会被 LiteGS profile 校验拦截
- optimizer 仍然对全部参数执行 Adam / fused Adam 更新
- clustered 路径和非 clustered 路径不会共享同一套 visible-only update 语义

当前已完成：

- 放开 `sparse_grad=true` 的 LiteGS profile 校验
- `apply_backward_grads()` 在 LiteGS + sparse_grad 下改为 visible-only row update
- moment 和参数只对 `projected.source_indices` 对应的可见 primitive 执行更新
- 新增 2D / 3D 参数张量与 zero-visible 两类稀疏 Adam 回归测试
- Metal clustered 路径的投影结果也会在 CPU 侧套用 cluster mask，保证 clustered render 与 sparse update 使用同一批 visible primitive

结果：

- `sparse_grad` 不再只是 CLI / config 层的开关
- RustGS 现在已经具备 LiteGS 风格 visible-only sparse optimizer 的第一版训练语义
- 后续剩余工作主要转向 parity 校准、性能对比和数据结构优化，而不是“是否支持”本身

限制：

- 当前 sparse update 仍然通过通用张量 `index_select/index_add` 路径完成，不是 LiteGS 那种更紧凑的 compacted tensor / chunk-native 实现
- 这意味着语义已接近，但性能路径还未接近 LiteGS 的最终形态

涉及文件：

- `RustGS/src/training/mod.rs`
- `RustGS/src/training/metal_trainer.rs`

### 6. Depth Supervision 开关放开

此前这条能力的状态是：

- 训练主循环中已有 depth loss 路径
- 但 `enable_depth=true` 仍会被 `validate_litegs_mac_v1_config()` 拦截

当前已完成：

- 放开 `enable_depth=true` 的 LiteGS profile 校验
- 增加定向测试，确认 LiteGS loss weight 只会在显式开启时启用 depth term
- 增加 parity report 回归测试，确认 depth telemetry 与 `sparse_grad/cluster_size/enable_depth` 配置都会进入报告

结果：

- `enable_depth` 已经从“代码存在但不可用”推进到“可正式开启”
- 后续重点从“放开开关”转为“校准 depth parity”

限制：

- 当前只证明了功能已接通与报告已记录
- 还没有完成对 LiteGS depth loss 权重、mask 和训练曲线的显式 side-by-side 校准

涉及文件：

- `RustGS/src/training/mod.rs`
- `RustGS/src/training/metal_trainer.rs`
- `RustGS/src/bin/rustgs.rs`

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
- `cargo test -p rustgs apply_backward_grads_sparse_grad`
- `cargo test -p rustgs adam_step_var_sparse_preserves_invisible_rows_for_tensor3_params`
- `cargo test -p rustgs project_gaussians_applies_cluster_visible_mask_on_metal`
- `cargo test -p rustgs litegs_mac_v1_accepts_sparse_grad_override`
- `cargo test -p rustgs litegs_mac_v1_accepts_enable_depth_override`
- `cargo test -p rustgs litegs_loss_weights_only_enable_depth_when_requested`
- `cargo test -p rustgs training_step_records_depth_telemetry_with_clustered_sparse_grad`
- `cargo test -p rustgs litegs_parity_report_persists_depth_and_sparse_cluster_config`

说明：

- 当前验证以定向单测和局部回归为主
- 还没有做完整的 LiteGS parity fixture 级别回归

## 待开发任务

### P0: Fixture 级 LiteGS Parity 回归

当前状态：

- `sparse_grad` 与 `enable_depth` 都已接入训练路径
- parity report 与运行时 telemetry 也已经能记录这些能力
- 但当前仍以定向单测和组件级回归为主

需要完成的事情：

- 建立 fixture 级别的训练回归
- 对关键 loss term、gaussian 数量演化和训练曲线做 LiteGS side-by-side 对比
- 明确 sparse / depth / clustered 组合下的容忍阈值

这是当前最重要的剩余缺口。

### P0: Depth Loss Parity

当前状态：

- `enable_depth=true` 已允许开启
- 训练主循环中的 depth loss 已接通
- 但目前仍缺 LiteGS 对照标定

需要完成的事情：

- 核对 RustGS 当前 depth loss 的权重、mask 和 LiteGS 是否一致
- 补充开启 depth supervision 的训练回归与曲线对照

### P1: Sparse / Clustered 数据路径性能版

当前状态：

- sparse optimizer 第一版已经落地
- clustered 路径也已经会把 cluster mask 同步到 sparse update 可见集合
- 但当前实现仍然依赖通用张量 gather/scatter 语义

需要完成的事情：

- 评估是否引入更接近 LiteGS 的 compacted tensor / visible chunk 结构
- 明确 clustered path 的更新单位是 cluster 还是 primitive
- 补充性能对比

### P1: Learnable ViewProj 性能版本

当前状态：

- 功能已接通
- 但当前实现使用 finite-difference loss 重算，额外渲染次数多

需要完成的事情：

- 评估 analytical pose gradient 是否可接入
- 或改成更低开销的近似更新方案
- 如果继续保留 FD 路径，至少要在配置和日志中明确其性能成本

### P2: Render 子命令 / 周边能力补齐

当前状态：

- 核心训练 parity 优先级高于 CLI 周边功能
- 一些周边能力仍不完整，例如 render 子命令仍未完全形成 LiteGS 对应体验

建议：

- 在训练语义核心差距补齐后再回头处理

## 建议开发顺序

推荐按下面顺序继续推进：

1. fixture 级 LiteGS parity 回归
2. depth loss parity 标定
3. sparse / clustered 数据路径性能化
4. `learnable_viewproj` 从 finite-difference 过渡到更高效实现
5. 周边 CLI / render 能力补齐

原因：

- 前两项直接决定“现在的实现是否真的和 LiteGS 行为接近”
- 后三项更偏性能、规模化使用和体验完善

## 风险与注意事项

- 当前 worktree 在最近几次提交后已经回到干净状态
- 后续继续做 parity 时，应避免覆盖与本次无关的用户改动
- `learnable_viewproj` 现阶段虽然可训练，但不应默认开启，除非明确接受额外训练开销
- sparse / depth 已经接通，不代表 parity 标定已经结束
- clustered 路径已经修掉明显 bug，但不代表已经达到 LiteGS 的最终语义

## 相关文档

- `docs/RustGS-LiteGS-Parity-Roadmap.md`
- `docs/RustGS-Training-Report.md`
