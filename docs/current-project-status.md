# RustScan Current Project Status

**Updated:** 2026-04-08  
**Branch:** `main`

## Overall

当前 `main` 的活跃工程主线是 RustGS 训练重构与验证闭环，不再是 `rm-opt` 上的 RustMesh LOD/OpenMesh 专项工作。仓库仍然是多 crate workspace，但 `docs/` 目录现在只保留和当前主线一致的文档。

## Verified Snapshot

本次状态清理时，本地已确认：

- `cargo check -p rustgs --all-features` 通过
- `cargo test -p rustgs --features gpu synthetic_loaded_training_data_matches_requested_fixture_shape -- --nocapture` 通过
- `cargo test -p rustgs --features gpu splats_ -- --nocapture` 通过
- `cargo test -p rustgs --features gpu execution_plan -- --nocapture` 通过
- `cargo test -p rustgs --features gpu sequential_chunk_executor_runs_chunks_one_at_a_time -- --nocapture` 通过
- `cargo test -p rustgs --features gpu adaptive_chunk_configs_keep_each_trainable_chunk_within_budget_envelope -- --nocapture` 通过
- `cargo test -p rustgs --features gpu chunk_persistence_writes_report_entries -- --nocapture` 通过
- `cargo test -p rustgs --features gpu persist_gaussian_map_scene_writes_chunk_ply -- --nocapture` 通过
- `cargo test -p rustgs --features gpu persist_gaussian_map_scene_preserves_spherical_harmonics_metadata -- --nocapture` 通过
- `cargo test -p rustgs --features gpu litegs_mac_v1_accepts_bootstrap_defaults -- --nocapture` 通过
- `cargo test -p rustgs --features gpu default_training_backend_is_metal -- --nocapture` 通过
- `cargo test -p rustgs --features gpu topology_update_densifies_and_prunes_with_matching_adam_state -- --nocapture` 通过
- `cargo test --manifest-path RustGS/Cargo.toml --features gpu density_controller_reference_summary_ -- --nocapture` 通过
- `cargo test -p rustgs --features gpu pose_parameter_grads_returns_tensor_pair -- --nocapture` 通过
- `cargo test -p rustgs --features gpu scale_regularization_ -- --nocapture` 通过
- `cargo test -p rustgs --features gpu test_pose_embeddings_adam_step_updates_selected_frame -- --nocapture` 通过
- `cargo test -p rustgs --features gpu tile_bins_only_include_overlapping_gaussians -- --nocapture` 通过
- RustGS refactor guardrails 文档最后验证时间为 2026-04-07

## Current Progress

- Epic 2 已基本完成：内部统一训练态模型 `splats.rs` 已接入主路径，旧的 `train_stream.rs` 与 `splat_params.rs` 已退出编译主路径。
- Epic 3 已继续收口：前向执行与 Metal runtime 已拆分为 `metal_forward`、`metal_projection`、`metal_raster`、`metal_resources`、`metal_dispatch`、`metal_pipelines` 等模块，`ProjectedTileBins` 已成为前向返回的契约类型，trainer/backward 不再直接依赖 `MetalTileBins`，生产 trainer 主路径上的 tile-index slot 细节也已收口到 `MetalRuntime` helper。
- Epic 4 当前 active story 已落地：`metal_loss`、`metal_backward`、`metal_optimizer` 已抽离，`learnable_viewproj` 的 render-camera 解析、pose FD 梯度和 pose update 现在都在 `pose_embedding.rs` 边界后，LiteGS scale regularization 的 full-tensor scatter 也已移出 `MetalTrainer::training_step()`。
- Epic 5 已继续推进：`TopologyMutationAftermath`/`TopologyMetricsDelta` 已把 densify/prune 之后的 rebuild、stats action、cluster resync、runtime reserve、opacity reset、telemetry 更新 contract 化；同时 `training::topology::{DensityControllerReferenceAdapter, density_controller_reference_summary}` 已把 `density_controller.rs` 接成显式 reference adapter，LiteGS 拓扑日志开始并列输出 reference clone/split/prune/budget 遥测。
- Epic 6 已继续收口：`training::orchestrator`、`training::execution_plan`、`training::chunk_training`、`training::export` 已接管 train route、execution-plan 选择、chunk 顺序执行和 chunk artifact/report 持久化；同时 `training::config` 已接管训练配置/枚举定义，`training_pipeline` 的根级 re-export 已移除，`training/mod.rs` 已缩成约 124 行的模块装配层，chunk scene export 已修正为保留 SH metadata 与 `sh_rest`，而 `training::splats` 现在还接管了 `GaussianMap <-> TrainableGaussians` 和 `GaussianMap -> scene Gaussian/metadata` 的核心桥接逻辑。
- Epic 6.5 又往前走了一步：`LoadedTrainingData` 已改成直接输出 `initial_splats`，production trainer 和 benchmark 都从 `Splats` 进入 step loop，原来的 `map_from_trainable(...)` 兼容 helper 已删除，训练内部的 scene/trainable 回转现在显式经过 `Splats::from_trainable(...).to_gaussian_map()`。
- TUM 对照已经给出清晰结论：`litegs-mac-v1` 的 late-stage topology freeze 值得保留，`freeze80` 在当前配置下能减少训练开销且几乎不损失质量。

## Active Gaps

- Epic 5 仍未完全收口：`density_controller.rs` 已经有显式 reference adapter，但生产 mutation path 还没有直接由这层 adapter/strategy 驱动；topology telemetry / regression coverage 也还可以继续补强。
- Epic 3 仍有内部边界泄漏：生产调用方已经不再直接吃 `MetalTileBins`，但 `metal_forward` 内部仍要处理 runtime projection record / staging 细节，forward boundary 还没完全成为纯 DTO 内核。
- Epic 4 还剩最后一点结构债：底层 raster backward 仍先产出 `MetalBackwardGrads`，再被装配成最终的 `MetalParameterGrads`，不过这已经不再阻塞主训练路径的可读性。
- Epic 6 仍有尾项，但 6.6/6.7 的主收口已经完成：当前剩余主要是 6.1/6.5 级别的问题。现在角色分工已经更清楚了: `GaussianMap` 负责公共 scene IO，`Splats` 负责内部 snapshot/exchange 边界，`TrainableGaussians` 负责 live step-loop mutation；剩下的债主要是减少 trainer 在 topology/export 检查点附近重建 `Splats` snapshot 的次数。
- LiteGS parity 仍缺真实参考 fixture 与稳定阈值门禁。
- TUM 评估闭环仍需要继续固定输出格式与回归口径。
- scene-scale-aware normalization 仍未落地。

## Next Priorities

1. 继续完成 Epic 6.5 的尾项：把 trainer/topology/export 附近仍然存在的 `Splats` snapshot churn 再压回更窄的边界，避免 canonical state 虽然已明确但调用点仍偏散。
2. 回到 Epic 5：决定 LiteGS production topology 是否直接转向新的 `density_controller` adapter；如果不转，就把 reference-only 边界和差异回归门禁写死。
3. 继续补 Story 3.5 的内部收口，把 projection-record/staging 选择进一步压回 `metal_forward`/`metal_runtime` 内部。
4. 在上述边界稳定后，继续固化 TUM train/eval 摘要与 LiteGS parity gate，把对照验证升级成长期回归门禁。
5. 继续推进 scene-scale-aware normalization，这项仍然缺实现。
