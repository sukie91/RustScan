# RustMesh 对齐 OpenMesh 进展报告

**日期**: 2026-04-05  
**Worktree**: `rm-opt`  
**工作目录**: `.claude/worktrees/rm-opt`  
**当前唯一状态文档**: 本文档  
**已取代并清理**:

- `docs/RustMesh-OpenMesh-Execution-Plan-2026-04-05.md`
- `docs/RustMesh-OpenMesh-Gap-Assessment-2026-04-05.md`
- `docs/RustMesh-OpenMesh-Progress-2026-04-04.md`
- `docs/RustMesh-OpenMesh-Test-Report-2026-04-04.md`

## 1. 当前状态快照

截至 2026-04-05 当前 worktree 的最新事实是：

- RustMesh 与 OpenMesh 的 `decimation trace` 已对齐到前 9 步完整签名一致。
- 第 10 步仍有唯一剩余差异：
  - RustMesh: `115->116`
  - OpenMesh: `116->115`
- 前 10 步无向边序列已经对齐：
  - `Matching prefix on undirected edge + faces_removed: 10 steps`
- 结果级 parity 已稳定：
  - RustMesh: `collapsed=61, boundary=31, interior=30, final V=60, final F=109`
  - OpenMesh: `collapsed=61, boundary=31, interior=30, final V=60, final F=109`
- `cargo test --lib tools::decimation::tests --quiet` 通过。

这意味着当前问题已经不是“大范围 decimater 行为不一致”，而是一个被压缩到非常窄的数值/初始化差异。

## 2. 当前已验证基线

### 2.1 Trace 基线

命令：

```bash
cargo run --release --example openmesh_compare_decimation_trace --quiet -- 10
```

当前输出要点：

- `Matching prefix on removed/kept/boundary/faces_removed: 9 steps`
- `Matching prefix on undirected edge + faces_removed: 10 steps`
- `First divergence at step 10`
  - RustMesh: `115->116`
  - OpenMesh: `116->115`
- traced prefix 内没有 active-face gap

### 2.2 测试基线

命令：

```bash
cargo test --lib tools::decimation::tests --quiet
```

结果：

- `10 passed`
- `0 failed`

## 3. 当前最重要的定位结论

### 3.1 已排除 point 坐标差异

通过 `point_bits` 精确 dump 已确认：

- RustMesh 与 OpenMesh 在顶点 `115..120` 的点坐标位级一致。
- 因此剩余分歧不来自输入点位误差，也不来自 collapse 时目标点不同。

### 3.2 顶点 quadric 差异已收敛到低位 `f64` 系数

通过 `quadric_bits` 精确 dump 已确认：

- RustMesh 与 OpenMesh 的顶点 quadric 在目标区域不是“数量级错误”。
- 差异是若干系数的 `1-2 ULP` 级别偏移。
- 这些微小偏移足以把 `116->115` / `116->117` 一类候选从 `0` 或极小正值翻成极小负值，从而让 RustMesh 把候选踢出 heap。

当前最典型的现象是：

- RustMesh live state:
  - `116->117 raw_error=-6.938893903907e-18`
  - `116->115 raw_error=-6.938893903907e-18`
- OpenMesh live state:
  - `116->117 raw_error=6.938893903907e-18`
  - `116->115 raw_error=0`

### 3.3 根因不再是 outgoing halfedge 顺序

此前已确认：

- `adjust_outgoing_halfedge()` 的局部 ring-walk 修正是有效的。
- 目标区域的 outgoing 顺序已经和 OpenMesh 对齐。

因此当前剩余差异不再主要归因于：

- boundary anchor 选择
- outgoing halfedge 枚举顺序
- 主循环的 heap pop 顺序本身

### 3.4 当前根因已进一步收敛到 face quadric 初始化

通过 face-level `face_quadric_bits` dump 已确认：

- RustMesh 与 OpenMesh 在 `115..120` 附近的 incident faces 上，face 顶点循环普遍只差一个 cyclic rotation。
- 对大部分 face，只要把 Rust 的顶点顺序做旋转，OpenMesh 的 face quadric 位模式就能匹配上。
- 但至少两个 face 目前仍然不只是“旋转问题”：
  - `face 192`
  - `face 196`
- 这两个 face 在尝试 `rot0 / rot1 / rot2` 后，仍然没有得到 OpenMesh 的位模式。

结论：

- 当前问题已经缩小到 per-face quadric 构造路径。
- 既有 face vertex cycle 差异，也有更细的数值构造差异。
- 下一步应该直接对 `face 192` / `face 196` 做逐操作对照，而不是继续在 heap tie-break 或 legality 上盲调。

## 4. 本轮已尝试且不应重复的方案

以下尝试已经验证无效或副作用过大：

- 对 tiny negative quadric error 做 clamping
  - 会破坏更长前缀，已回退
- 把 face quadric 初始化直接改成 `from_vertex` 顺序
  - 会把 trace 从第 3 步就打崩，已回退
- 在主循环层继续做大范围 tie-break 试探
  - 已不在主要怀疑路径上

这些实验说明当前不该继续做“表层排序修补”，而要做“face quadric bitwise 对照”。

## 5. 当前保留的调试能力

### 5.1 Heap / support / candidate dump

`RustMesh/src/Tools/decimation.rs` 与  
`RustMesh/examples/openmesh_compare_decimation_trace.rs` 当前支持：

- `RUSTMESH_TRACE_DEBUG_STEPS`
- `RUSTMESH_TRACE_DEBUG_TOP`
- `RUSTMESH_TRACE_DUMP_VERTICES`
- `RUSTMESH_TRACE_DUMP_AFTER_COLLAPSES`
- `RUSTMESH_TRACE_DUMP_EXACT`
- `RUSTMESH_TRACE_DUMP_FACE_QUADRICS`

可输出：

- `RUST_SUPPORT`
- `RUST_HEAP before_updates / after_updates`
- `OPENMESH_SUPPORT`
- `OPENMESH_HEAP before_updates / after_updates`
- `point_bits`
- `quadric_bits`
- `face_quadric_bits`
- `face_quadric_bits_rot1`
- `face_quadric_bits_rot2`

### 5.2 常用命令

基线 trace：

```bash
cargo run --release --example openmesh_compare_decimation_trace --quiet -- 10
```

顶点 live state：

```bash
env RUSTMESH_TRACE_DUMP_VERTICES=115,116,117,118,119,120 \
  cargo run --release --example openmesh_compare_decimation_trace --quiet -- 0
```

位级 dump：

```bash
env RUSTMESH_TRACE_DUMP_VERTICES=115,116,117,118,119,120 \
    RUSTMESH_TRACE_DUMP_EXACT=1 \
  cargo run --release --example openmesh_compare_decimation_trace --quiet -- 0
```

face quadric dump：

```bash
env RUSTMESH_TRACE_DUMP_VERTICES=115,116,117,118,119,120 \
    RUSTMESH_TRACE_DUMP_FACE_QUADRICS=1 \
  cargo run --release --example openmesh_compare_decimation_trace --quiet -- 0
```

## 6. 当前涉及文件

当前这轮 trace 对齐和精确定位直接相关的文件：

- `RustMesh/examples/openmesh_compare_decimation_trace.rs`
- `RustMesh/src/Core/connectivity.rs`
- `RustMesh/src/Core/io/off.rs`
- `RustMesh/src/Tools/decimation.rs`
- `RustMesh/src/Utils/circulators.rs`

## 7. 下一步计划

下一步计划已经非常明确，不再建议扩散排查范围：

1. 对 `face 192` 和 `face 196` 做逐操作对照。
   - 重点核查：
   - cross product
   - `norm()` / `sqrt()`
   - `d = -(p0·n)` 的求值顺序
   - quadric 每个系数乘以 `area` 的时机
2. 把 Rust 的 per-face quadric 构造调整到和 OpenMesh 在这两个 face 上位级一致。
3. 重跑以下验证：
   - `RUSTMESH_TRACE_DUMP_EXACT=1`
   - `RUSTMESH_TRACE_DUMP_FACE_QUADRICS=1`
   - `cargo run --release --example openmesh_compare_decimation_trace --quiet -- 10`
   - `cargo test --lib tools::decimation::tests --quiet`
4. 目标是把当前唯一剩余差异从：
   - RustMesh `115->116`
   - OpenMesh `116->115`
   收敛到第 10 步也完全一致。

## 8. 结论摘要

当前最准确的结论可以压缩为三句话：

- RustMesh 与 OpenMesh 的 decimation trace 现在已经对齐到前 9 步完整签名、前 10 步无向边一致。
- 剩余差异已被压缩到 face quadric 初始化的低位 `f64` 数值构造，不再是 heap 或 outgoing halfedge 的大问题。
- 下一阶段不该再做主循环盲调，而应该直接把 `face 192` / `face 196` 的 per-face quadric 构造做成 OpenMesh bitwise parity。
