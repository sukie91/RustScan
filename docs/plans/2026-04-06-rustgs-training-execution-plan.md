# RustGS Training Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让 RustGS 的训练开发回到一条和当前代码事实一致、和当前开发目标一致的主线上：基于已经能在真实 `TUM RGB-D` 上工作的 `litegs-mac-v1`，优先建立稳定评估闭环并提升中后期质量收益，而不是重启一条 `BrushV1` / CPU 重构路线。

**Architecture:** 生产训练继续以 `rustgs::train()` -> `training::train()` -> `training/mod.rs` 内 execution-plan 路由 -> `metal_trainer::train()` / `train_chunked_sequentially()` 为主干，`topology.rs` 负责 densify / prune / opacity reset 调度，`eval.rs` 和 `examples/evaluate_psnr.rs` 负责质量汇总。`training_pipeline.rs` 继续只保留 legacy/reference 责任；Brush 只做 selective migration，不新增并行训练主线。

**Tech Stack:** Rust, Cargo, Candle Metal, RustGS training modules, TUM RGB-D, scene PLY export, PSNR evaluation.

---

## Current Facts

1. 当前生产训练入口已经不是 `training_pipeline.rs`。
   - 顶层入口在 `RustGS/src/lib.rs`
   - 训练 orchestration 和 route 选择现在直接在 `RustGS/src/training/mod.rs`
   - step-level Metal 执行在 `RustGS/src/training/metal_trainer.rs`
   - topology 调度在 `RustGS/src/training/topology.rs`
   - `RustGS/src/training/mod.rs` 已明确把 `training_pipeline.rs` 定义为 legacy/reference only

2. `litegs-mac-v1` 已经证明“可以训练”，而不是“完全不收敛”。
   - `docs/RustGS-TUM-Profile-Comparison-2026-04-06.md` 已证明 `LiteGS` 在真实 `TUM RGB-D` 子集上相对 `legacy-metal` 有稳定优势
   - `360` iterations 时 `PSNR mean = 6.9847 dB`
   - `5000` iterations 时 `PSNR mean = 7.1194 dB`
   - `360 -> 5000` 只提升 `0.1347 dB`，但额外耗时约 `40.88` 分钟

3. 当前最真实的问题不是“完全训不起来”，而是：
   - 评估闭环还不够硬，导致拓扑/超参改动缺乏统一验收口径
   - LiteGS 后期 topology churn 偏重，`360` 之后已明显进入收益递减区
   - 生产模块边界虽然已经成型，但仍有局部遗留耦合，例如 topology side effect 和 trainer lifecycle state 还没有完全分离

4. 当前开发目标应收敛到真实 `TUM RGB-D` 分析，不再扩展 `Nerfstudio` 方向。

## Superseded Assumptions

以下判断已经被最新代码和最新 TUM 结果淘汰，不应再作为执行依据：

- “RustGS 当前训练无法收敛，PSNR 不涨”
- “下一步应该先重建一条 Brush 风格 CPU/通用训练管线”
- “应该新开一个 `BrushV1` profile，再决定生产路径”
- “现在的主要 blocker 仍然是能不能把训练跑起来”
- “smoke benchmark / parity 状态记录足以指导下一阶段开发”

## Non-Goals

本轮计划明确不做以下事情：

- 不新建 `BrushV1` 训练 profile
- 不把新的生产逻辑继续堆回 `training_pipeline.rs`
- 不优先做 clustered parity、sparse-grad parity、learnable viewproj
- 不补 Nerfstudio 无深度训练方案
- 不把 `gsplat-mlx-main` 对照放在当前实现收敛之前
- 不把 Burn/WGPU/WGSL runtime 迁移当作当前主线

### Task 1: 固化 TUM 评估闭环

**Files:**
- Modify: `RustGS/examples/evaluate_psnr.rs`
- Modify: `RustGS/src/training/eval.rs`
- Modify: `RustGS/src/bin/rustgs.rs`
- Modify: `RustGS/tests/tum_training.rs`
- Modify: `docs/RustGS-TUM-Profile-Comparison-2026-04-06.md`

**Step 1: 固定 TUM train/eval 切分和指标输出格式**

- 统一当前 TUM 分析口径，至少固定：训练帧选择、评估帧选择、`render_scale`、输出 JSON schema
- 明确后续所有 profile / schedule 调整都必须落在同一组评估帧上

**Step 2: 让训练过程能稳定产出最终评估摘要**

- 把 `final_loss`、`final_step_loss`、`PSNR mean/median/min/max`、worst-frame 信息收束到统一汇总结构
- CLI 和 example 输出同一套字段，避免手工拼接结论

**Step 3: 补 TUM regression smoke**

Run: `cargo test --manifest-path RustGS/Cargo.toml --features gpu tum_training -- --nocapture`

Expected: 至少覆盖 TUM 数据发现、训练 smoke、评估字段存在性，不要求长程跑满

**Step 4: 回写基线结果**

- 用当前 `legacy-metal` / `litegs-mac-v1` 的 `120 / 240 / 360` 结果作为后续比较基线
- `5000` 结果保留为“收益递减存在”的证据，而不是常规训练预算

**Step 5: Commit**

```bash
git add RustGS/examples/evaluate_psnr.rs RustGS/src/training/eval.rs RustGS/src/bin/rustgs.rs RustGS/tests/tum_training.rs docs/RustGS-TUM-Profile-Comparison-2026-04-06.md
git commit -m "docs: lock TUM evaluation gate for RustGS training"
```

**Exit Criteria:** 之后每个训练相关改动，都能在固定 TUM 评估口径下给出结构化结果。

### Task 2: 收口生产路径对 legacy helper 的剩余依赖

**Files:**
- Modify: `RustGS/src/training/topology.rs`
- Modify: `RustGS/src/training/mod.rs`
- Modify: `RustGS/src/training/metal_trainer.rs`
- Modify: `RustGS/src/training/training_pipeline.rs`
- Test: `RustGS/src/training/mod.rs`

**Step 1: 把 topology 与 route 逻辑的参数来源完全收口到当前生产配置面**

- 移除或验证 `topology.rs` / `metal_trainer.rs` 对 legacy helper 默认值的剩余借用
- 只允许 `TrainingConfig` / `LiteGsConfig` 作为生产路径的参数来源

**Step 2: 明确 legacy 和 litegs 的责任边界**

- `LegacyMetal` 仍保留现状行为
- `LiteGsMacV1` 的 topology 语义只通过当前 `mod.rs` 配置面暴露，不再从 legacy helper 间接继承行为

**Step 3: 跑 route/config regression**

Run: `cargo test --manifest-path RustGS/Cargo.toml --features gpu execution_plan -- --nocapture`
Run: `cargo test --manifest-path RustGS/Cargo.toml --features gpu litegs_mac_v1_accepts_ -- --nocapture`

Expected: route 规划和 LiteGS config validation 结果保持稳定

**Step 4: Commit**

```bash
git add RustGS/src/training/topology.rs RustGS/src/training/mod.rs RustGS/src/training/metal_trainer.rs RustGS/src/training/training_pipeline.rs
git commit -m "refactor: remove legacy default leakage from topology path"
```

**Exit Criteria:** 新的生产行为不再依赖 `training_pipeline.rs` 的默认配置。

### Task 3: 优先解决 LiteGS 后期收益递减，而不是继续拉长训练时长

**Files:**
- Modify: `RustGS/src/training/topology.rs`
- Modify: `RustGS/src/training/metal_trainer.rs`
- Modify: `RustGS/src/training/telemetry.rs`
- Modify: `docs/RustGS-TUM-Profile-Comparison-2026-04-06.md`

**Step 1: 增加 late-stage topology telemetry**

- 把 `densify`、`prune`、`opacity reset` 的触发区间、累计次数、后期节奏记录清楚
- 重点观察 `240 -> 360 -> 720 -> 1200` 是否仍在重复 churn

**Step 2: 为后期阶段引入可验证的收敛策略**

优先尝试：
- topology freeze / cooldown
- 更低频的 opacity reset
- densify / prune 预算上限
- 在 topology 已稳定后只做参数优化

**Step 3: 用固定 TUM 口径验证而不是凭感觉判断**

Run: `cargo run --manifest-path RustGS/Cargo.toml --example evaluate_psnr -- --scene <scene> --dataset test_data/tum/rgbd_dataset_freiburg1_xyz --render-scale 0.25 --max-frames 180 --frame-stride 30 --device metal`

Expected: 目标不是绝对 PSNR 数字立刻大涨，而是证明同等或略高训练时长下，`PSNR/time` 和最差帧表现不再继续恶化

**Step 4: Commit**

```bash
git add RustGS/src/training/topology.rs RustGS/src/training/metal_trainer.rs RustGS/src/training/telemetry.rs docs/RustGS-TUM-Profile-Comparison-2026-04-06.md
git commit -m "tune: reduce late-stage LiteGS topology churn"
```

**Exit Criteria:** `360` 之后的训练预算要么被明确砍掉，要么被证明还能带来稳定收益。

### Task 4: 引入 scene-scale-aware normalization

**Files:**
- Modify: `RustGS/src/training/data_loading.rs`
- Modify: `RustGS/src/training/init_map.rs`
- Modify: `RustGS/src/training/mod.rs`
- Modify: `RustGS/src/training/metal_trainer.rs`
- Modify: `RustGS/src/training/topology.rs`

**Step 1: 定义当前 RustGS 可接受的 scene scale 估计方式**

优先从当前已有数据里估计：
- camera baseline / pose spacing
- 初始高斯或点云分布范围
- 不依赖额外外部格式或新 loader

**Step 2: 把 scene scale 只用于当前已观察到的敏感项**

先收敛到这几项：
- `lr_position`
- clone/split 阈值
- prune bound / scene bound

**Step 3: 给 telemetry 和配置面补可见性**

- 训练日志里输出实际使用的 scene scale
- 避免未来再次出现“同一套超参跨场景完全失控，但原因看不出来”

**Step 4: 在 TUM 上跑回归**

Run: `cargo test --manifest-path RustGS/Cargo.toml --features gpu`

Expected: 单测保持通过；TUM smoke 不出现更差的稳定性退化

**Step 5: Commit**

```bash
git add RustGS/src/training/data_loading.rs RustGS/src/training/init_map.rs RustGS/src/training/mod.rs RustGS/src/training/metal_trainer.rs RustGS/src/training/topology.rs
git commit -m "feat: add scene-scale-aware training normalization"
```

**Exit Criteria:** 位置学习率和拓扑阈值开始具备跨场景可解释性。

### Task 5: 只迁移 Brush 中对当前问题真正有帮助的 topology 语义

**Files:**
- Modify: `RustGS/src/training/topology.rs`
- Modify: `RustGS/src/training/splats.rs`
- Modify: `RustGS/src/training/metal_optimizer.rs`
- Modify: `RustGS/src/training/metal_trainer.rs`
- Test: `RustGS/tests/tum_training.rs`

**Step 1: 限定迁移范围**

只优先考虑这些已知与当前问题直接相关的 Brush 语义：
- 基于 `xy-plane grad` 的 split / grow 排序
- scale-aware position offset
- split 后 opacity 重新分配
- invisible-age-aware prune 保护

明确不迁：
- Brush runtime
- Brush UI / viewer
- clustered training
- sparse-grad optimizer

**Step 2: 保持与当前模块边界一致**

- 调度留在 `topology.rs`
- 参数重建留在 `splats.rs` / `metal_optimizer.rs`
- step-level 执行仍留在 `metal_trainer.rs`

**Step 3: 以 TUM 指标判定迁移值不值得保留**

Expected: 迁移后的版本至少在以下一项上显著更好，才允许继续留下：
- `PSNR mean`
- `worst-frame`
- `PSNR/time`
- topology churn 下降

**Step 4: Commit**

```bash
git add RustGS/src/training/topology.rs RustGS/src/training/splats.rs RustGS/src/training/metal_optimizer.rs RustGS/src/training/metal_trainer.rs RustGS/tests/tum_training.rs
git commit -m "feat: selectively migrate Brush topology semantics"
```

**Exit Criteria:** Brush 迁移从“架构理想”变成“经过 TUM 验证的增量收益”。

### Task 6: 内部评估稳定后，再做 `gsplat-mlx-main` 对照

**Files:**
- Create: `docs/RustGS-vs-gsplat-mlx-TUM-Comparison-<date>.md`
- Modify: `docs/plans/2026-04-06-rustgs-training-execution-plan.md`

**Step 1: 锁定统一对照口径**

- 同一组 TUM 数据
- 同一组 train/eval 帧
- 同一 `render_scale`
- 同一套最终渲染评估指标

**Step 2: 先把 RustGS 自己的收敛策略跑稳，再去比**

- 先完成 Task 1-5
- 再用 `gsplat-mlx-main` 做同口径训练/评估

**Step 3: 把外部对照用作验收，不用作架构指挥棒**

- 如果 `gsplat-mlx-main` 更好，优先拆解“哪条机制有效”
- 不因为对照结果就立即推翻当前 Metal 训练主线

**Step 4: Commit**

```bash
git add docs/RustGS-vs-gsplat-mlx-TUM-Comparison-<date>.md docs/plans/2026-04-06-rustgs-training-execution-plan.md
git commit -m "docs: record RustGS vs gsplat-mlx TUM comparison"
```

**Exit Criteria:** 外部对照成为后验验证工具，而不是当前实现方向的前置 blocker。

## Recommended Order

1. Task 1: 固化 TUM 评估闭环
2. Task 2: 去掉生产路径对 legacy defaults 的隐式依赖
3. Task 3: 优先解决 LiteGS 后期收益递减
4. Task 4: 引入 scene-scale-aware normalization
5. Task 5: 选择性迁移 Brush topology 语义
6. Task 6: 做 `gsplat-mlx-main` 对照

## Acceptance Criteria

- 所有训练相关改动都能在固定 TUM 评估口径下复现
- `training_pipeline.rs` 不再成为新生产逻辑的落点
- `360` 之后的训练预算有明确去留依据，而不是继续盲目加迭代
- scene-scale 进入训练配置和 telemetry，可解释跨场景行为
- Brush 迁移只能以 TUM 实测收益为依据保留
- `gsplat-mlx-main` 对照放在内部闭环稳定之后执行

## Canonical Docs For This Track

- `docs/current-project-status.md`
- `docs/ARCHITECTURE.md`
- `docs/RustGS-TUM-Profile-Comparison-2026-04-06.md`
- `docs/plans/2026-04-06-rustgs-training-execution-plan.md`
- `docs/plans/2026-04-06-rustgs-brush-refactor-review-and-epics.md`
- `docs/plans/2026-04-06-rustgs-refactor-guardrails.md`
- `docs/plans/2026-04-05-litegs-parity-roadmap-refresh.md`
