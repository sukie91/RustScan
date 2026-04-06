# RustScan 开发计划

**更新时间：** 2026-04-06  
**工作树：** `rm-opt`

## 目标

以最小风险完成 RustMesh 与 OpenMesh 的下一阶段对齐：优先收敛 LOD 增量导航语义，再做选择性 parity 覆盖扩展。

## 阶段 A（最高优先级）

### Epic 3 / Story 3.2：LOD 增量导航

目标：去掉常规上行 LOD 的“从 original 全量回放”路径，让 `get_lod(level)` 在维护路径上增量切换。

执行项：

1. 显式维护当前 LOD 状态，区分“继续 simplify”与“向上 replay exact records”的决策路径。
2. 将 `get_lod(level)` 从无条件 reset/replay 改为优先基于当前状态增量迁移。
3. 对仍需 fallback 的边界场景保留明确文档化行为，不做隐式语义。

完成标准：

- 常规上下切换为增量行为。
- 单调与双向切换回归通过。
- 常规 scrub 不再付出 full replay 成本。

## 阶段 B（次优先级）

### Epic 5：选择性 Parity Hardening

目标：在高价值路径补强 OpenMesh 对齐验证，而不是无差别扩量。

执行顺序：

1. Story 5.1：将 decimation parity 检查提升为稳定自动回归。
2. Story 5.2：补充 decimation 之外的选择性算法对比覆盖。
3. Story 5.3：清理基准与文档噪声（含 `Vec4f_add_compare` 相关说明）。

## 全程约束

- 每个 story 结束后保持 `cargo test --manifest-path RustMesh/Cargo.toml --lib --quiet` 通过。
- 保持 SoA 导向和维护路径局部编辑语义稳定。
- 明确区分“维护路径语义承诺”和“非三角面 fallback 受控范围”。

## 里程碑出口

- M1：LOD 增量导航可用且有回归保护。
- M2：Decimation parity 自动回归稳定。
- M3：关键非 decimation 模块具备选择性对比覆盖并完成文档收敛。
