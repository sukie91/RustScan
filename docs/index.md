# RustScan Documentation Index

**Updated:** 2026-04-06

这份索引只列出当前仍与活跃代码路径或当前开发目标对齐的文档。若文档之间存在冲突，以 RustGS 训练专项文档为准。

## Current RustGS Training Docs

| Document | Purpose |
|---|---|
| [RustGS Training Alignment Implementation Plan](plans/2026-04-06-rustgs-training-execution-plan.md) | 当前有效的 RustGS 训练执行方案；基于现有代码 ownership 和真实 TUM 评估结论制定 |
| [RustGS TUM Profile Comparison 2026-04-06](RustGS-TUM-Profile-Comparison-2026-04-06.md) | 真实 `TUM RGB-D` 上 `legacy-metal` 与 `litegs-mac-v1` 的训练/评估基线 |
| [RustGS Brush Migration Architecture 2026-04-05](RustGS-Brush-Migration-Architecture-2026-04-05.md) | RustGS 当前训练模块边界、调用流与 rollout 约束 |

## Core Project Docs

| Document | Purpose |
|---|---|
| [README](README.md) | 仓库总览 |
| [Development Guide](DEVELOPMENT.md) | 构建、测试、开发约束 |
| [Architecture](ARCHITECTURE.md) | 系统级结构说明 |
| [Project Overview](project-overview.md) | 项目范围与组件概览 |
| [API Reference](API.md) | 公开 API 说明 |
| [COLMAP Pipeline](COLMAP-PIPELINE.md) | COLMAP 相关流程记录 |

## Component Docs

| Document | Purpose |
|---|---|
| [RustMesh README](RustMesh-README.md) | RustMesh 组件说明 |
| [RustMesh OpenMesh Progress 2026-04-04](RustMesh-OpenMesh-Progress-2026-04-04.md) | RustMesh/OpenMesh 对齐进展 |
| [RustMesh OpenMesh Test Report 2026-04-04](RustMesh-OpenMesh-Test-Report-2026-04-04.md) | RustMesh/OpenMesh 测试记录 |
| [RustMesh OpenMesh Parity Roadmap](RustMesh-OpenMesh-Parity-Roadmap.md) | RustMesh 后续路线 |
| [RustSLAM README](RustSLAM-README.md) | RustSLAM 组件说明 |
| [RustSLAM Design](RustSLAM-DESIGN.md) | RustSLAM 设计记录 |
| [RustSLAM ToDo](RustSLAM-ToDo.md) | RustSLAM backlog |
| [RustSLAM Experiment 2026-03-28](RustSLAM-Experiment-2026-03-28.md) | RustSLAM 实验记录 |

## Reference Artifacts

| Document | Purpose |
|---|---|
| [PROJECT-SCAN-SUMMARY](PROJECT-SCAN-SUMMARY.md) | 项目扫描摘要 |
| [source-tree-analysis](source-tree-analysis.md) | 源码树分析 |
| [project-scan-report.json](project-scan-report.json) | 项目扫描原始数据 |
| [CLAUDE](CLAUDE.md) | Claude 协作约束 |

## Superseded On 2026-04-06

以下 RustGS 训练文档已删除，不再维护：

- `RustGS-LiteGS-Parity-Status-2026-04-04.md`
- `RustGS-Smoke-Benchmark-Report-2026-04-06.md`
- `RustGS-Training-Report.md`
