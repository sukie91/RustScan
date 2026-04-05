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

- `openmesh_compare_decimation_trace` 现在默认走 `OpenMeshParity` 导入模式。
- 在默认模式下，RustMesh 与 OpenMesh 的 `decimation trace` 前 10 步已经完整对齐：
  - `Matching prefix on removed/kept/boundary/faces_removed: 10 steps`
  - `Matching prefix on undirected edge + faces_removed: 10 steps`
- 第 10 步历史残差已经消除：
  - RustMesh: `116->115`
  - OpenMesh: `116->115`
- 结果级 parity 已稳定：
  - RustMesh: `collapsed=61, boundary=31, interior=30, final V=60, final F=109`
  - OpenMesh: `collapsed=61, boundary=31, interior=30, final V=60, final F=109`
- `cargo test --lib tools::decimation::tests --quiet` 通过，当前为 `12 passed`。
- `remeshing` 已完成第一轮加固：
  - `split_long_edges()` 改为批量重建三角面，不再使用原先会破坏流程稳定性的占位式删面重建
  - `test_isotropic_remesh_sphere` 不再卡死
  - `cargo test --lib tools::remeshing::tests --quiet` 通过，当前为 `7 passed`
- `RUSTMESH_TRACE_IMPORT_MODE=standard` 仍然保留为调试入口，但当前只对齐到前 6 步，不再作为 parity example 的默认基线。
- 当前 worktree 已清空，`git status --short` 无未提交修改。

这意味着当前这轮工作已经从“定位最后一步 decimation 分歧”推进到“parity 闭环完成，并开始清理后续 remeshing 稳定性问题”。

## 2. 当前已验证基线

### 2.1 Trace 基线

命令：

```bash
env RUSTFLAGS=-Awarnings cargo run --release --example openmesh_compare_decimation_trace --quiet -- 10
```

当前输出要点：

- `RustMesh import mode: OpenMeshParity`
- `Matching prefix on removed/kept/boundary/faces_removed: 10 steps`
- `Matching prefix on undirected edge + faces_removed: 10 steps`
- traced prefix 内无 divergence
- traced prefix 内没有 active-face gap
- 第 10 步输出：
  - RustMesh: `116->115`
  - OpenMesh: `116->115`

对照命令：

```bash
env RUSTFLAGS=-Awarnings RUSTMESH_TRACE_IMPORT_MODE=standard \
  cargo run --release --example openmesh_compare_decimation_trace --quiet -- 10
```

对照输出要点：

- `Matching prefix on removed/kept/boundary/faces_removed: 6 steps`
- `Matching prefix on undirected edge + faces_removed: 6 steps`
- `First divergence at step 7`
  - RustMesh: `2->6`
  - OpenMesh: `113->114`

### 2.2 测试基线

命令：

```bash
cargo test --lib tools::decimation::tests --quiet
```

结果：

- `12 passed`
- `0 failed`

附加验证：

```bash
cargo test --example openmesh_compare_decimation_trace --quiet
```

结果：

- example 编译通过
- `0 failed`

### 2.3 Remeshing 基线

命令：

```bash
cargo test --lib tools::remeshing::tests --quiet
```

结果：

- `7 passed`
- `0 failed`

关键现象：

- `test_isotropic_remesh_sphere -- --exact --nocapture` 已能正常结束
- 当前一次样例输出为：
  - `split=587`
  - `collapse=609`
  - `flip=106`
  - `Final edge lengths: min=0.1194512, max=1.1829344, mean=0.44982117`

新增回归：

- `test_split_long_edges_splits_shared_interior_edge_into_four_triangles`

## 3. 当前最重要的定位结论

### 3.1 face quadric 数值路径已经对齐到 OpenMesh

通过 `face_quadric_bits` 与 operand dump 已确认：

- `face 192`
- `face 196`

这两个历史 bad faces 现在都能在 `openmesh_parity` face order 下达到位级一致。

当前对应的回归测试为：

- `test_face_192_quadric_bits_match_openmesh_parity`
- `test_face_196_quadric_bits_match_openmesh_parity`

### 3.2 根因定位完成：关键不在 heap，而在 per-face quadric 构造路径

本轮最终确认的关键点是：

- 旧残差不是输入点误差，也不是 heap tie-break 的主问题。
- 关键差异确实来自 `face_quadric_from_points()` 的数值求值顺序。
- 将 cross product、`sqrnorm` 与 `plane_dot` 的求值路径对齐后，step-10 trace divergence 消失。
- parity example 默认切换到 `OpenMeshParity` 后，RustMesh 侧使用和 OpenMesh 对齐的 face 插入路径作为标准调试入口。

结论：

- 这一轮“最后一步 decimation trace gap”已经闭合。
- 当前更值得继续开发的方向不再是 trace 盲调，而是把后续 remeshing / VDPM / 文档清单继续往前推。

### 3.3 remeshing 当前已从“会挂起”恢复到“可验证”

本轮对 `remeshing` 的最新结论是：

- 旧 `split_long_edges()` 在迭代过程中直接删面重建，容易把 `isotropic_remesh()` 推进到不稳定甚至卡住的状态。
- 当前实现改成：
  - 先收集要 split 的长边
  - 为 midpoint 统一建点
  - 再按每个三角面的 split mask 批量重建面片
- 这不是最终版 half-edge split primitive，但已经足以：
  - 让 remeshing 测试恢复可跑
  - 给下一阶段更深入的拓扑加固留出稳定基线

## 4. 本轮已尝试且不应重复的方案

以下尝试已经验证无效或副作用过大：

- 对 tiny negative quadric error 做 clamping
  - 会破坏更长前缀，已回退
- 把 face quadric 初始化直接改成 `from_vertex` 顺序
  - 会把 trace 从第 3 步就打崩，已回退
- 在主循环层继续做大范围 tie-break 试探
  - 已不在主要怀疑路径上
- 把 `standard` 导入模式当成默认 parity 基线
  - 在当前数学路径下只对齐到前 6 步，应该仅保留为对照调试入口

这些实验说明，本轮正确路径是“对齐 per-face quadric 构造 + 使用 OpenMeshParity 导入基线”，而不是继续做表层排序修补。

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
- `RUSTMESH_TRACE_DUMP_FACE_OPS`

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
env RUSTFLAGS=-Awarnings cargo run --release --example openmesh_compare_decimation_trace --quiet -- 10
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

face operand dump：

```bash
env RUSTFLAGS=-Awarnings \
    RUSTMESH_TRACE_DUMP_VERTICES=115,116,117,118,119,120 \
    RUSTMESH_TRACE_DUMP_FACE_QUADRICS=1 \
    RUSTMESH_TRACE_DUMP_FACE_OPS=192,196 \
  cargo run --release --example openmesh_compare_decimation_trace --quiet -- 0
```

## 6. 当前涉及文件

当前这轮 trace 对齐和精确定位直接相关的文件：

- `RustMesh/examples/openmesh_compare_decimation_trace.rs`
- `RustMesh/src/Core/connectivity.rs`
- `RustMesh/src/Core/io/off.rs`
- `RustMesh/src/Tools/decimation.rs`
- `RustMesh/src/Tools/remeshing.rs`
- `RustMesh/src/Utils/circulators.rs`
- `docs/plans/2026-04-05-rustmesh-rm-opt-dev-checklist.md`

## 7. 当前提交状态

截至本文档更新时，本轮已落地的最新提交为：

- `d1fa637` `Add rm-opt development checklist`
- `3709e53` `Stabilize remeshing edge splitting`
- `92c4f04` `Close OpenMesh decimation parity gap`

当前工作树状态：

- `git status --short` 为空
- 可以直接在干净基线上继续后续任务

## 8. 下一步计划

当前 parity 闭环已经完成，后续优先级建议如下：

1. 如果要继续加强 decimation parity，可补一条“trace prefix 10”自动回归，而不只是 face-bit regression。
2. 进入 `docs/plans/2026-04-05-rustmesh-rm-opt-dev-checklist.md` 的下一阶段任务：
   - remeshing 拓扑操作继续加固，尤其是 collapse 路径和真实 half-edge split primitive
   - VDPM LOD API 补齐
   - roadmap / progress 文档继续同步
3. 保留 `standard` 模式作为调试对照，但不要再把它作为默认期望结果。

## 9. 结论摘要

当前最准确的结论可以压缩为三句话：

- 默认 `OpenMeshParity` 模式下，RustMesh 与 OpenMesh 的 decimation trace 前 10 步现在已经完整对齐。
- `face 192` / `face 196` 的 per-face quadric 构造已做到位级 parity，并由回归测试覆盖。
- `remeshing` 已经从“测试会挂起”恢复到“测试可稳定验证”，下一阶段应继续推进更完整的拓扑实现、VDPM 和文档收口。
