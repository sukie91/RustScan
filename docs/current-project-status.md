# RustScan Current Project Status

**Updated:** 2026-04-07  
**Branch:** `main`

## Overall

当前 `main` 的活跃工程主线是 RustGS 训练重构与验证闭环，不再是 `rm-opt` 上的 RustMesh LOD/OpenMesh 专项工作。仓库仍然是多 crate workspace，但 `docs/` 目录现在只保留和当前主线一致的文档。

## Verified Snapshot

本次状态清理时，本地已确认：

- `cargo check -p rustgs --all-features` 通过
- `cargo test -p rustgs --features gpu execution_plan -- --nocapture` 通过
- `cargo test -p rustgs --features gpu topology_update_densifies_and_prunes_with_matching_adam_state -- --nocapture` 通过
- `cargo test -p rustgs --features gpu pose_parameter_grads_returns_tensor_pair -- --nocapture` 通过
- `cargo test -p rustgs --features gpu scale_regularization_ -- --nocapture` 通过
- `cargo test -p rustgs --features gpu test_pose_embeddings_adam_step_updates_selected_frame -- --nocapture` 通过
- `cargo test -p rustgs --features gpu tile_bins_only_include_overlapping_gaussians -- --nocapture` 通过
- RustGS refactor guardrails 文档最后验证时间为 2026-04-07

## Current Progress

- Epic 2 已基本完成：内部统一训练态模型 `splats.rs` 已接入主路径，旧的 `train_stream.rs` 与 `splat_params.rs` 已退出编译主路径。
- Epic 3 已继续收口：前向执行与 Metal runtime 已拆分为 `metal_forward`、`metal_projection`、`metal_raster`、`metal_resources`、`metal_dispatch`、`metal_pipelines` 等模块，`ProjectedTileBins` 已成为前向返回的契约类型，trainer/backward 不再直接依赖 `MetalTileBins`，生产 trainer 主路径上的 tile-index slot 细节也已收口到 `MetalRuntime` helper。
- Epic 4 当前 active story 已落地：`metal_loss`、`metal_backward`、`metal_optimizer` 已抽离，`learnable_viewproj` 的 render-camera 解析、pose FD 梯度和 pose update 现在都在 `pose_embedding.rs` 边界后，LiteGS scale regularization 的 full-tensor scatter 也已移出 `MetalTrainer::training_step()`。
- Epic 5 的关键 aftermath 收口已完成：`TopologyMutationAftermath`/`TopologyMetricsDelta` 已把 densify/prune 之后的 rebuild、stats action、cluster resync、runtime reserve、opacity reset、telemetry 更新 contract 化，trainer 不再在 `apply_snapshot_mutations()` 后手工重建整套副作用序列。
- TUM 对照已经给出清晰结论：`litegs-mac-v1` 的 late-stage topology freeze 值得保留，`freeze80` 在当前配置下能减少训练开销且几乎不损失质量。

## Active Gaps

- Epic 5 仍在推进：`density_controller.rs` 还没有作为 reference strategy/adapter 显式挂到当前 topology contract 后面，拓扑遥测和回归覆盖也还能继续补强。
- Epic 3 仍有内部边界泄漏：生产调用方已经不再直接吃 `MetalTileBins`，但 `metal_forward` 内部仍要处理 runtime projection record / staging 细节，forward boundary 还没完全成为纯 DTO 内核。
- Epic 4 还剩最后一点结构债：底层 raster backward 仍先产出 `MetalBackwardGrads`，再被装配成最终的 `MetalParameterGrads`，不过这已经不再阻塞主训练路径的可读性。
- Epic 6 仍未收口：`training/mod.rs` 还同时承担入口、profile 校验、route selection、chunk execution 和 persistence。
- 内部 canonical training state 还没最终定案；当前仍在 `GaussianMap`、`TrainableGaussians`、`Splats` 之间切换。
- LiteGS parity 仍缺真实参考 fixture 与稳定阈值门禁。
- TUM 评估闭环仍需要继续固定输出格式与回归口径。
- scene-scale-aware normalization 仍未落地。

## Next Priorities

1. 继续 Epic 5：把 `density_controller.rs` 挂成显式 reference strategy/adapter，并补 topology telemetry / regression coverage，完成 5.3 和 5.4。
2. 继续 Epic 6：先拆薄 `training/mod.rs`，把 execution-plan 选择、chunk route、persistence 从模块根移走，完成 6.1/6.6 的主干工作。
3. 明确并收紧 canonical internal training state，完成 6.5，避免 `GaussianMap`、`TrainableGaussians`、`Splats` 在训练内部继续漂移。
4. 视需要继续补 Story 3.5 的内部收口，把 projection-record/staging 选择进一步压回 `metal_forward`/`metal_runtime` 内部。
5. 在上述边界稳定后，继续固化 TUM train/eval 摘要与 LiteGS parity gate，把对照验证升级成长期回归门禁。
