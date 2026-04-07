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
- RustGS refactor guardrails 文档最后验证时间为 2026-04-07

## Current Progress

- Epic 2 已基本完成：内部统一训练态模型 `splats.rs` 已接入主路径，旧的 `train_stream.rs` 与 `splat_params.rs` 已退出编译主路径。
- Epic 3 已大部分落地：前向执行与 Metal runtime 已拆分为 `metal_forward`、`metal_projection`、`metal_raster`、`metal_resources`、`metal_dispatch`、`metal_pipelines` 等模块，MSL shader 已外置。
- Epic 4 已大部分落地：`metal_loss`、`metal_backward`、`metal_optimizer` 已抽离，生产训练步骤已经接近显式的 `forward -> loss -> backward -> optimize -> topology` 流程。
- TUM 对照已经给出清晰结论：`litegs-mac-v1` 的 late-stage topology freeze 值得保留，`freeze80` 在当前配置下能减少训练开销且几乎不损失质量。

## Active Gaps

- Epic 5 仍在推进：拓扑副作用还没有完全从 `metal_trainer.rs` 收口到 `topology.rs` 契约内。
- LiteGS parity 仍缺真实参考 fixture 与稳定阈值门禁。
- TUM 评估闭环仍需要继续固定输出格式与回归口径。
- scene-scale-aware normalization 仍未落地。

## Next Priorities

1. 固化 TUM train/eval 口径和统一摘要输出。
2. 复验 `1200` iterations 下 `no-freeze` vs `freeze80`，决定是否把 freeze 收敛为默认 schedule。
3. 将真实 `parity-reference.json` 接入 canonical fixture，并把 comparison 升级成 pass/fail gate。
4. 继续完成 topology contract 收口，再考虑 sparse/clustered 与 pose-learning 的性能化。
