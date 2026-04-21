# RustScan Current Project Status

**Updated:** 2026-04-10  
**Branch:** `main`

## Overall

当前 `main` 的活跃主线仍然是 RustGS 训练架构收口与 TUM 质量闭环，但这轮状态已经向前推进了一步：RustGS 的 legacy API 和对应文件已经从源码主路径里清掉，文档也同步收口到 splat-first 架构。

## Verified Snapshot

本轮清理后的本地验证目标是：

- `cargo check -p rustgs`
- `cargo test -p rustgs`
- 已删除的 public legacy API 名称只会出现在“removed/deleted”说明里，不再作为当前 active surface 出现

本轮完成后，guardrail 文档和索引只保留当前代码事实对应的文档集合。

## Current Progress

- RustGS 的公开训练路径已经收口到 splat-first 入口：`train_splats`、`train_splats_from_path`、`evaluate_splats`、`save_splats_ply`、`load_splats_ply`。
- 旧兼容层已从源码主路径删除：`legacy/`、`training_pipeline.rs`、`io/dataset_loader.rs`、`io/scene_io/scene_import.rs`、`io/scene_io/scene_export.rs`。
- Canonical 表示已经明确：
  - `training::HostSplats` 负责 host 侧边界、checkpoint 与 PLY。
  - `diff::Splats` 负责 device/runtime 侧可微训练状态。
  - `training::SplatView` 负责 host 侧只读借用。
- 训练装配层已经缩到 `training/mod.rs`，活跃逻辑由 `orchestrator`、`execution_plan`、`chunk_training`、`export`、`metal_*` 子模块承担。
- Metal runtime 与 trainer 拆分保持成立，`topology.rs`、`density_controller.rs`、`parity_harness.rs`、`eval.rs` 的职责边界比前一轮更清楚。
- 文档层也已经同步：旧的 execution-plan/brush-epic 文档不再作为 active docs 保留。

## Active Gaps

- 纯 SoA 目标还没有完全走完：`render::Gaussian` 仍然作为 CPU renderer / 测试 / 局部兼容路径的 AoS 适配类型存在。
- 评估与导出命名仍有少量 scene-era 术语残留，例如 `SceneMetadata`、`SceneEvaluationConfig`、`SceneEvaluationError`。
- RustGS 训练主路径现在已经收口到 LiteGS 语义；legacy 残留点主要是已删除 flag 的 tombstone 测试与文档里的 removed-surface 说明，而不是活跃代码路径。
- LiteGS parity gate、TUM PSNR 回归口径、scene-scale-aware normalization 仍然属于待开发任务。

## Next Priorities

1. 继续收口 splat-first 命名，把评估/导出周边仍然残留的 scene-era 术语清理掉。
2. 评估是否要继续删除 `render::Gaussian` 这层 AoS 适配，推动 renderer/eval 端进一步直接吃 SoA 视图。
3. 固化 LiteGS parity gate 和 TUM PSNR 回归输出，避免后续质量工作缺统一验收口径。
4. 推进 scene-scale-aware normalization，这仍然是训练质量侧最直接的结构性待办。
