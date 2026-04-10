# RustScan Documentation Index

**Updated:** 2026-04-10

`docs/` 目录只列出当前仍作为执行依据、验证依据或状态依据的文档。已经被当前代码状态淘汰的 RustGS legacy/兼容层计划文档已移除；如需追溯，请直接查看 git 历史。

## Canonical Status Docs

| Document | Purpose |
|---|---|
| [current-project-status.md](current-project-status.md) | 当前仓库主线状态、已验证结果与下一步优先级 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 当前 workspace 结构与 RustGS 训练架构边界 |

## Active RustGS Docs

| Document | Purpose |
|---|---|
| [plans/2026-04-06-rustgs-refactor-guardrails.md](plans/2026-04-06-rustgs-refactor-guardrails.md) | 当前 public surface、回归基线与 guardrail 命令 |
| [../RustGS/docs/plans/2026-04-09-rustgs-soa-splat-architecture-proposal.md](../RustGS/docs/plans/2026-04-09-rustgs-soa-splat-architecture-proposal.md) | RustGS 当前唯一的 splat 表示设计文档与收口状态 |
| [plans/2026-04-05-litegs-parity-roadmap-refresh.md](plans/2026-04-05-litegs-parity-roadmap-refresh.md) | 当前剩余 LiteGS parity 工作与优先级 |
| [RustGS-TUM-Profile-Comparison-2026-04-06.md](RustGS-TUM-Profile-Comparison-2026-04-06.md) | 当前有效的 TUM 训练对照记录与 topology-freeze 决策依据 |

## Retention Rule

- 只保留和当前代码事实一致的文档。
- 已被删除 API、旧 scene/map ownership、`training_pipeline.rs`、`legacy/` 兼容层相关计划不再保留在 active docs 集合里。
