# RustGS Brush Migration Architecture（2026-04-05）

## 目标

这份文档记录 RustGS 在吸收 Brush 架构思路之后的当前生产布局。目标不是把 Brush 直接移植进 RustGS，而是把那些已经证明有工程价值的部分迁进来：

- 训练 orchestration 拆分
- 有界 frame loading
- 直接数据集接入
- topology / optimizer state 拆分
- 版本化 scene interchange
- regression + benchmark + rollout 文档

不在本次迁移范围内的内容：

- Burn / WGPU / WGSL 运行时替换
- Brush UI / viewer / browser / mobile 子系统
- 推翻现有 Metal kernel 与 analytical backward
- 把新的生产责任继续堆回 `training_pipeline.rs`

## 当前 Ownership

| 模块 | 生产责任 | 说明 |
| --- | --- | --- |
| `RustGS/src/training/train_stream.rs` | 顶层训练 orchestration | 负责 profile 校验、route 规划、chunked/standard 分流、训练 loop 调度 |
| `RustGS/src/training/frame_loader.rs` | frame decode / prefetch / shuffle | 负责颜色图、深度图、synthetic depth fallback、有界 cache、确定性 batch 顺序 |
| `RustGS/src/training/data_loading.rs` | 训练态数据装配 | 负责把 dataset/frame 变成 `LoadedTrainingData` 与 `TrainableGaussians` |
| `RustGS/src/training/init_map.rs` | 初始高斯构造 | 负责 sparse-point / frame-based 初始化分流 |
| `RustGS/src/training/metal_trainer.rs` | step-level Metal 执行 | 负责 forward、loss、backward、optimizer update、step profiling |
| `RustGS/src/training/topology.rs` | densify / prune / opacity reset 调度与分析 | 负责 topology policy、schedule 和 snapshot 变换 |
| `RustGS/src/training/optimizer_state.rs` | Adam state reshape | 负责 prune / reorder / densify 后的 optimizer state 重建 |
| `RustGS/src/training/splat_params.rs` | trainable splat 参数视图 | 负责 topology 操作期间的参数 snapshot / rebuild |
| `RustGS/src/training/eval.rs` | eval scheduling | 负责训练过程中的评估触发与汇总 |
| `RustGS/src/training/export.rs` | scene / chunk export scheduling | 负责 checkpoint / chunk artifact 写出 |
| `RustGS/src/training/telemetry.rs` | 训练遥测聚合 | 负责最后一步 loss/topology/profile 汇总 |
| `RustGS/src/io/dataset_loader.rs` | 训练输入统一发现入口 | 负责 TUM / COLMAP / Nerfstudio / JSON 入口分流与 overlay 合并 |
| `RustGS/src/io/scene_io.rs` | scene IO facade | 对外保留 `save_scene_ply` / `load_scene_ply` facade |
| `RustGS/src/io/scene_io/scene_export.rs` | scene export | 负责版本化 metadata 与 SH-aware PLY 写出 |
| `RustGS/src/io/scene_io/scene_import.rs` | scene import | 负责 legacy fallback、显式 metadata 解析、SH 重建 |
| `RustGS/src/training/benchmark.rs` | benchmark harness | 负责 forward/backward/training smoke 基准输出 |
| `RustGS/src/training/training_pipeline.rs` | legacy/reference only | 保留旧 helper、utility loss、兼容 surface；不是新功能落点 |

## 训练调用流

1. `rustgs::train()` / `rustgs::train_from_path()` 进入 `training::train()`
2. `training::train()` 立即委托给 `train_stream::train()`
3. `train_stream` 做 profile 校验和 route 选择：
   - `Standard`
   - `ChunkedSingleChunk`
   - `ChunkedSequential`
4. `data_loading` 用 `frame_loader` 解码数据并构建 `LoadedTrainingData`
5. `metal_trainer` 执行 step-level forward / backward / optimizer
6. `topology` 在训练 loop 中按 schedule 驱动 densify / prune / opacity reset
7. `export` / `telemetry` / `scene_io` 负责产物和最终遥测

核心边界：

- `train_stream` 决定“什么时候做什么”
- `metal_trainer` 只负责“这一 step 怎么算”
- `topology` 决定“高斯如何演化”
- `scene_io` 决定“结果如何安全落盘和再读取”

## 兼容与 Rollout 约束

当前必须保持兼容的入口：

- `LegacyMetal`
- `LiteGsMacV1`
- non-chunked standard route
- chunked single-chunk pass-through route
- chunked sequential route

当前 scene compatibility 规则：

- 新格式写出 `format_version = 2`
- scene metadata 显式记录：
  - `iterations`
  - `final_loss`
  - `gaussian_count`
  - `color_representation`
  - `sh_degree`
- 旧文件没有 `format_version` 时，import 走 fallback
- 旧文件如果只有 RGB property，则回落到 `Rgb`
- 旧文件如果包含 `sh0_*` / `sh_rest_*` property，则推断 SH degree
- metadata 声明 SH 但 property 缺失时，显式报错而不是静默降级

## 已采用的 Brush 思路

- 用独立 orchestration 模块管理训练生命周期，而不是把所有控制流堆进 trainer
- dataset onboarding 走 facade + format-specific loader
- topology、optimizer state、parameter snapshot 拆模块
- persistence 走 facade + import/export 内部分拆
- regression 测试和 benchmark harness 跟架构迁移一起落地

## 明确拒绝的 Brush 思路

- 不迁移 Brush 的运行时栈，不把 RustGS 改成 Burn/WGPU/WGSL
- 不迁移 Brush 的 UI / viewer / app 侧模块
- 不在第一阶段替换已有 Metal kernels
- 不把 analytical backward 改成完全不同的反传框架
- 不再把新增生产逻辑堆回 `training_pipeline.rs`

## Regression / Benchmark 入口

推荐验证命令：

```bash
cargo test --manifest-path RustGS/Cargo.toml
cargo test --manifest-path RustGS/Cargo.toml train_stream::tests -- --nocapture
cargo test --manifest-path RustGS/Cargo.toml io::scene_io::tests -- --nocapture
```

benchmark harness：

```bash
cargo run --manifest-path RustGS/Cargo.toml --example training_benchmark -- \
  --profile legacy-metal \
  --width 64 \
  --height 64 \
  --frames 3 \
  --gaussians 128 \
  --warmup 2 \
  --measure 5 \
  --smoke-iters 8
```

如果需要便于保存结果，可以加 `--json`。

## 对后续开发的约束

- 新的训练控制流改动应优先放进 `train_stream.rs`
- 新的 frame ingestion / batching 逻辑应优先放进 `frame_loader.rs`
- 新的 densify/prune/controller 规则应优先放进 `topology.rs`
- scene format 变化必须同时改 `scene_export.rs`、`scene_import.rs` 和回归测试
- 新功能如果只能通过修改 `training_pipeline.rs` 才能实现，默认说明模块边界又退化了，应该先重审设计
