# RustMesh 对齐 OpenMesh 进展报告

**日期**: 2026-04-04  
**工作分支**: `rustmesh-opt`  
**工作目录**: `.claude/worktrees/rustmesh-opt`  
**目标**: 参考 `Mirror/OpenMesh-11.0.0`，提升 `RustMesh` 在功能行为、示例覆盖和 decimation 效率上的一致性。

## 1. 当前结论

截至 2026-04-04，本轮工作已经完成了一批基础对齐和验证设施，尤其是：

- 稳定了 `RustMesh` 的局部 collapse / deleted-state 语义，避免明显的拓扑损坏。
- 增加了和 OpenMesh 逐 case 对比的 example。
- 增加了 decimation trace 能力，可以逐步对比 RustMesh 与 OpenMesh 的 collapse 序列。
- 已确认部分 OpenMesh Example 可以达到结果一致。
- 已确认 decimation 仍然存在和 OpenMesh 的行为偏差，且该偏差已经能被精确定位。

当前最重要的结论不是“RustMesh 不能 decimate”，而是：

- `RustMesh` 现在能稳定 decimate，并保持结果 mesh 没有退化面或非流形边。
- 但它和 OpenMesh 的 collapse 选择顺序、collapse 方向、boundary/interior mix 仍不一致。
- 这种不一致不是随机噪声，已经在 trace 中表现为从第 1 步开始偏离，并在第 31 步后造成面数差异扩大。

## 2. 本轮已完成事项

### 2.1 Decimation 基础语义修正

以下基础实现已经在当前工作树中建立并保留：

- `RustMesh/src/Core/soa_kernel.rs`
- `RustMesh/src/Core/connectivity.rs`
- `RustMesh/src/Tools/decimation.rs`

这些改动的作用是：

- 让 collapse 后的 deleted 状态更稳定。
- 让 decimation 结果可重复。
- 避免在当前 benchmark 下出现明显的 degenerate face 或 non-manifold edge。

### 2.2 新增 OpenMesh 对比 example

已新增或扩展以下 example：

- `RustMesh/examples/openmesh_compare_decimation.rs`
- `RustMesh/examples/openmesh_compare_examples.rs`
- `RustMesh/examples/openmesh_compare_decimation_trace.rs`

用途分别是：

- `openmesh_compare_decimation`
  - 对比 RustMesh 与 OpenMesh 的 decimation 结果和主流程耗时。
- `openmesh_compare_examples`
  - 对比 OpenMesh 教程示例的结果和局部性能。
- `openmesh_compare_decimation_trace`
  - 逐步对比 RustMesh 与 OpenMesh 的 collapse trace，定位第一处分歧及其后续影响。

### 2.3 新增 decimation trace API

`RustMesh/src/Tools/decimation.rs` 和 `RustMesh/src/lib.rs` 已新增并导出：

- `DecimationTraceStep`
- `DecimationTrace`
- `decimate_with_trace(...)`
- `decimate_to_with_trace(...)`

这些 API 的作用是：

- 保持原有 decimation 行为不变。
- 在不破坏现有使用方式的前提下输出逐步 collapse 轨迹。
- 为 OpenMesh 对照提供精确证据，而不是只看最终 mesh。

### 2.4 补充测试

当前 `cargo test --lib --quiet` 结果：

- `133 passed`
- `0 failed`

新增 trace 相关测试已经通过，用于确保 trace 输出在 collapse limit 下行为正确。

## 3. 已验证的结果

### 3.1 OpenMesh Example 对齐现状

命令：

```bash
cargo run --release --example openmesh_compare_examples --quiet
```

当前已验证结果：

- `Tutorial01: Cube Build + OFF Roundtrip`
  - RustMesh 结果与 OpenMesh 预期一致。
  - `V=8 F=6`
- `Iterators`
  - RustMesh 的 handle/index 两条遍历路径结果一致。
  - 在该微基准下，index path 明显快于 OpenMesh 风格 handle 遍历。
- `Circulators`
  - RustMesh 的邻接顶点结果与 OpenMesh `vv_iter` 等价。
- `Mesh I/O`
  - OBJ 读取到 OFF 写出链路已可运行。
- `Tutorial08: Delete Geometry + Garbage Collection`
  - 已达到 OpenMesh 示例结果一致。
  - RustMesh 结果：
    - `active V=4`
    - `active F=1`
    - `centroid=(0.0000, 0.0000, -1.0000)`
  - 与 OpenMesh 预期 `V=4, F=1, centroid=(0.0, 0.0, -1.0)` 一致。

### 3.2 Decimation 主流程对比

命令：

```bash
cargo run --release --example openmesh_compare_decimation --quiet
```

当前固定输入为球面网格：

- 输入 mesh:
  - `V=121`
  - `F=200`
- target vertices:
  - `60`

当前结果：

| 项目 | RustMesh | OpenMesh |
|------|----------|----------|
| 最终顶点数 | 60 | 60 |
| 最终面数 | 101 | 109 |
| collapse 数 | 61 | 61 |
| boundary collapse | 23 | 31 |
| interior collapse | 38 | 30 |
| 退化面 | 0 | 参考 OFF 输出 |
| 非流形边 | 0 | 参考 OFF 输出 |
| 主流程耗时 | 14.675 ms | 8.791 ms |

当前可以确认：

- RustMesh 的结果 mesh 在该 case 下是稳定的。
- 但和 OpenMesh 相比，RustMesh 删掉了更多面。
- RustMesh 当前 boundary collapse 偏少，interior collapse 偏多。
- RustMesh 当前主流程耗时仍慢于 OpenMesh。

### 3.3 Decimation Trace 对比

命令：

```bash
cargo run --release --example openmesh_compare_decimation_trace --quiet -- 40
```

当前 trace 结果：

| 项目 | RustMesh | OpenMesh |
|------|----------|----------|
| collapsed | 61 | 61 |
| boundary | 23 | 31 |
| interior | 38 | 30 |
| final V | 60 | 60 |
| final F | 101 | 109 |

关键 trace 结论：

- `Matching prefix on removed/kept/boundary/faces_removed: 0`
- `Matching prefix on undirected edge + faces_removed: 1`
- 第一个分歧出现在 step 1：
  - RustMesh: `2->0 B faces=1 prio=0.000000 active=200=>199`
  - OpenMesh: `0->2 B faces=1 prio=0.000000 active=200=>199`
- 前 40 步中的 boundary collapse 数：
  - RustMesh: `20`
  - OpenMesh: `29`
- 第一个累计 boundary/interior mix 差异：
  - `step 31`
- 第一个 active-face 差异：
  - `step 31`
  - RustMesh: `158`
  - OpenMesh: `159`

## 4. “第一步就不同” 的判断

### 4.1 什么是合理的

如果只从“这一步是不是一个合法的 collapse”看，当前 RustMesh 的第 1 步是合理的：

- 选中的是和 OpenMesh 同一条无向边。
- `faces_removed` 一样，都是 `1`。
- 拓扑上没有立刻出错。

也就是说，这不是明显错误的随机坏边，也不是立即破坏 mesh 的非法 collapse。

### 4.2 什么是不合理的

如果从“是否实现了 OpenMesh 等价语义”看，这个差异是不合理的：

- RustMesh 第 1 步和 OpenMesh collapse 的方向相反。
- 被删除顶点和保留顶点身份不同。
- 这会改变后续 quadric 合并、one-ring 更新和候选集。
- 最终不是只差一个 trace 文本，而是会放大成后续 boundary/interior mix 和面数差异。

因此，这一问题的准确定性是：

- 它不是 topology correctness bug。
- 它是 OpenMesh parity bug。

## 5. 已确认的根因方向

### 5.1 RustMesh 当前选择策略

`RustMesh/src/Tools/decimation.rs` 当前策略是：

- 全局扫描 halfedge。
- 根据以下规则比较候选：
  - `priority`
  - `is_boundary`
  - `faces_removed`
  - `halfedge idx`

这意味着：

- 它更像“全局最优 halfedge 选择器”。
- 它不是 OpenMesh 的“每顶点选一个本地最优 candidate，再走 vertex heap”。

### 5.2 OpenMesh 当前选择策略

OpenMesh `DecimaterT` 的关键行为位于：

- `Mirror/OpenMesh-11.0.0/src/OpenMesh/Tools/Decimater/DecimaterT_impl.hh`
- `Mirror/OpenMesh-11.0.0/src/OpenMesh/Tools/Decimater/BaseDecimaterT_impl.hh`

OpenMesh 的关键语义：

- 对每个顶点，只在它自己的 outgoing halfedge 中选一个最优 collapse target。
- tie-break 不是“全局 halfedge idx 最小”。
- 在 `heap_vertex(...)` 内，只有 `prio < best_prio` 才更新，不是 `<=`。
- 主循环是 vertex heap，而不是每轮全局重扫所有 halfedge。
- legality 检查除了 `mesh_.is_collapse_ok(...)` 之外，还有额外边界和局部结构约束。

这解释了为什么：

- RustMesh 第 1 步虽然合法，但方向和 OpenMesh 不同。
- 方向一旦不同，后续 trace 就会快速漂移。

## 6. 尝试过但未保留的方案

本轮曾尝试过更激进地改写为“每顶点 heap”版本，目标是直接贴近 `OpenMesh::DecimaterT`。

该尝试的结果不好，已撤回，原因包括：

- 最终面数回退到约 `93-95`，比当前基线更差。
- boundary collapse 数进一步下降。
- 性能也变差。

因此当前采取的策略是：

- 保留稳定基线。
- 先做 trace 和证据收集。
- 再做受控的小改动，而不是一次性重写调度逻辑。

## 7. 当前代码状态

### 7.1 关键已修改文件

- `RustMesh/src/Core/soa_kernel.rs`
- `RustMesh/src/Core/connectivity.rs`
- `RustMesh/src/Tools/decimation.rs`
- `RustMesh/src/lib.rs`
- `RustMesh/examples/openmesh_compare_decimation.rs`
- `RustMesh/examples/openmesh_compare_examples.rs`
- `RustMesh/examples/openmesh_compare_decimation_trace.rs`

### 7.2 已存在的历史提交

当前分支之前已经有以下相关提交可作为基线参考：

- `fb15b06` `Improve RustMesh OpenMesh parity and decimation checks`
- `a2c93b1` `Align RustMesh decimation collapse semantics with OpenMesh`

### 7.3 当前工作树状态

截至本报告生成时，本轮修改尚未形成新提交，仍处于工作树阶段。

## 8. 接下来要做的事

优先级建议如下。

### 8.1 P0: 修正 tied boundary case 的方向和 tie-break 语义

目标：

- 先缩小第 1 步就反向 collapse 的问题。
- 尽量让零代价 boundary candidate 的方向选择更接近 OpenMesh。

建议做法：

- 不先整体重写 heap。
- 先把 candidate 选择逻辑改到更接近：
  - “按顶点挑 outgoing halfedge”
  - “严格使用 `<` 保留第一个 best candidate”

### 8.2 P0: 补齐 OpenMesh legality 层

目标：

- 缩小“RustMesh 合法但 OpenMesh 不会选”的候选空间。

重点检查：

- boundary vertex 不能向 inner vertex collapse。
- 特定局部 valence / ring 结构约束。
- `BaseDecimaterT::is_collapse_legal(...)` 中的额外限制。

### 8.3 P1: 扩展 Example 覆盖

目标：

- 继续仿照 `Mirror/OpenMesh-11.0.0/src/OpenMesh/Examples` 增补 RustMesh example。
- 每个 example 同时输出：
  - 结果对比
  - 耗时对比

建议优先级：

- 继续补 Tutorial 系列。
- 优先挑对 connectivity、status、circulator、IO 和 decimation 行为敏感的 case。

### 8.4 P1: 建立更公平的性能对比

注意：

- `openmesh_compare_decimation_trace` 的总耗时不能直接当算法性能结论。
- 该 example 会临时编译 OpenMesh C++ driver，编译时间远大于算法本身。

需要补的内容：

- 固定编译产物缓存。
- 将“trace 生成耗时”和“编译 driver 耗时”分开统计。
- 对比算法净耗时，而不是对比一次性实验流水线耗时。

## 9. 复现命令

```bash
# RustMesh 单元测试
cargo test --lib --quiet

# OpenMesh example 对齐验证
cargo run --release --example openmesh_compare_examples --quiet

# 主 decimation 对比
cargo run --release --example openmesh_compare_decimation --quiet

# 逐步 trace 对比
cargo run --release --example openmesh_compare_decimation_trace --quiet -- 40
```

## 10. 结论摘要

当前阶段最重要的成果不是“已经完全追平 OpenMesh”，而是：

- RustMesh 已经有了稳定的 decimation 基线。
- OpenMesh example 对齐已经开始落地，至少 `Tutorial08` 结果已一致。
- trace 基础设施已经把“哪里不同、从什么时候开始不同、为什么最终面数不同”说清楚了。

当前阶段最重要的未完成项也很明确：

- collapse 方向与 tie-break 还没有贴近 OpenMesh。
- legality 过滤仍不完整。
- decimation 的结果和速度都还没有达到 OpenMesh 水平。

后续优化不应该盲目大改，而应该围绕 trace 逐步逼近 OpenMesh 的 candidate 选择语义。
