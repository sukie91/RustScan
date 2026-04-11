# Brush 架构分析与 RustGS 重构建议

日期: 2026-04-11

## 目的

这份文档基于两个代码库的实际代码结构做对照分析:

- 参考项目: `/Users/tfjiang/Projects/brush`
- 当前项目: `/Users/tfjiang/Projects/RustScan/RustGS`

目标不是复述目录，而是回答两个问题:

1. Brush 为什么没有演化出一个和 `metal_trainer.rs` 等价的巨型文件。
2. RustGS 应该如何沿着现有代码继续重构，尤其是 `RustGS/src/training/metal_trainer.rs` 应该怎么拆。

说明:

- 这份文档里的 `execution_plan.rs`、`chunk_training.rs`、`ChunkCapacityEstimate` 等内容属于当时的重构设想。
- 在后续代码整理中，RustGS 已明确回到单一路径训练，这些 chunked-training 模块和公开 API 已删除；阅读本文时请以当前代码为准。

## 结论先行

Brush 的核心优势不是“算法更少”，而是“边界更稳定”:

- 应用入口、训练流程编排、训练算法、渲染前向、渲染反向、数据集、UI、序列化、GPU 基础设施都被分在不同 crate。
- `SplatTrainer` 只负责“训练 step 和 refine”，不负责数据加载、预检、导出、UI 消息、执行计划、文件系统、设备启动。
- 配置、消息、统计、初始化、优化器、SSIM、LOD 都被拆成单独文件，训练主文件虽然是核心，但不是整个系统的交通枢纽。

RustGS 已经做对了一半:

- `metal_forward.rs`、`metal_backward.rs`、`metal_runtime.rs`、`metal_loss.rs`、`metal_optimizer.rs`、`topology.rs` 已经被抽出来了。
- 顶层训练编排已经从巨型 trainer 中抽离；其中早期探索过的 `execution_plan.rs` / `chunk_training.rs` 后来因放弃分块训练而被删除，当前保留的是标准训练路径的 `orchestrator.rs`。

但 RustGS 还没有完成第二步:

- `metal_trainer.rs` 仍然同时承担“训练入口 + 预检 + session 初始化 + frame 准备 + step 执行 + topology 集成 + cluster/pose 特性拼装 + 内存预算 + chunk capacity + 遥测 + 测试”。
- 它已经不是单纯的 trainer，而是整个 Metal 训练栈的集成中枢。

最有价值的下一步不是重写算法，而是把 `metal_trainer.rs` 改造成一个薄门面模块，把职责拆成 5 到 7 个聚焦文件。

## 一、Brush 的架构是什么样的

### 1.1 Workspace 分层很明确

Brush 在 workspace 根的 `Cargo.toml` 中直接按子域拆成多个 crate:

- 应用壳层:
  - `brush-app`
  - `brush-ui`
  - `brush-cli`
- 流程编排层:
  - `brush-process`
- 核心领域层:
  - `brush-train`
  - `brush-render`
  - `brush-render-bwd`
  - `brush-dataset`
  - `brush-serde`
  - `brush-vfs`
  - `brush-rerun`
- GPU 基础设施层:
  - `brush-kernel`
  - `brush-sort`
  - `brush-prefix-sum`
  - `brush-wgsl`
- 支撑 crate:
  - `colmap-reader`
  - `lpips`
  - `rrfd`

这意味着 Brush 的“模块边界”不是只靠文件名约定，而是已经上升到 crate 边界。这样做的直接收益是:

- 训练算法不需要依赖 UI。
- 渲染前向与反向不需要知道训练流程。
- 流程编排可以调用训练，但训练不反向依赖流程层。

### 1.2 Brush 的调用链是分层的，不是集中到 trainer

典型路径如下:

1. `brush-app` / `brush-cli` 负责启动。
2. `brush-process` 创建流程流和设备上下文。
3. `brush-process::train_stream` 负责:
   - 加载数据集
   - 初始化 splats
   - 构建 dataloader
   - 驱动 eval / export / LOD
   - 实例化 `brush_train::train::SplatTrainer`
4. `SplatTrainer` 只负责:
   - `step`
   - `refine`
5. `brush-render` 提供前向渲染。
6. `brush-render-bwd` 提供可微反向渲染。
7. UI 只消费消息和共享的 splat slot。

也就是说，Brush 里真正接近“trainer”的对象，只拥有训练语义本身，而不是整个训练产品流程。

### 1.3 Brush 的关键架构模式

#### 模式 A: 薄编排层 + 厚能力层

`brush-process` 是个流程壳。它持有:

- 设备初始化
- 进程 stream
- 消息流
- `Slot<Splats<...>>`

但不会把训练算法塞进去。

`brush-train` 则只负责训练相关算法:

- `train.rs`
- `adam_scaled.rs`
- `ssim.rs`
- `stats.rs`
- `splat_init.rs`
- `lod.rs`
- `config.rs`
- `msg.rs`

#### 模式 B: 配置、消息、算法、统计拆文件

`brush-train` 里面并没有一个大文件包办所有东西:

- `config.rs` 放训练配置
- `msg.rs` 放训练消息和统计结构
- `adam_scaled.rs` 放优化器实现
- `ssim.rs` 放损失组件
- `stats.rs` 放 refine 统计
- `splat_init.rs` 放初始化逻辑
- `lod.rs` 放 decimation / LOD
- `train.rs` 留给核心训练语义

这就是为什么 `brush-train/src/train.rs` 只有 684 行，而整个 crate 仍然完整。

#### 模式 C: 前向和反向渲染独立

Brush 没把“训练反向逻辑”塞进 trainer:

- `brush-render` 暴露 `SplatOps`
- `brush-render-bwd` 暴露 `SplatBwdOps`

trainer 只依赖这些抽象和入口，不负责保存着所有 GPU 细节。

#### 模式 D: trainer 不拥有产品流程

Brush 的 trainer 不负责:

- 数据集加载
- 训练流消息
- UI 对接
- export 路径拼接
- LOD 阶段切换
- rerun 可视化

这些都在 `brush-process::train_stream`。

这点对 RustGS 非常重要，因为 RustGS 的 `metal_trainer.rs` 目前还把一部分“产品流程”级别的职责一起扛着。

### 1.4 Brush 的量化信号

从文件尺度看，Brush 明显在主动压制 God file:

- `brush-train` 整个 crate 约 1917 行
- `brush-train/src/train.rs` 684 行
- `brush-render` + `brush-render-bwd` 合计约 2006 行
- 最大文件也只是:
  - `brush-train/src/train.rs` 684 行
  - `brush-render-bwd/src/burn_glue.rs` 495 行

这说明 Brush 并不是“简单所以短”，而是“复杂度被分摊到了多个稳定边界里”。

## 二、Brush 哪些设计最值得 RustGS 借鉴

### 2.1 把“训练产品流程”和“训练算法内核”分开

Brush 里:

- 训练流程在 `brush-process`
- 训练算法在 `brush-train`

RustGS 里更理想的对应关系应该是:

- 流程编排:
  - `orchestrator.rs`
  - `execution_plan.rs`
  - `chunk_training.rs`
- Metal 训练内核:
  - `metal_trainer/*`

目前 RustGS 虽然已经有了 `orchestrator.rs`，但 `metal_trainer.rs` 仍然保留了很多流程层职责。

### 2.2 不让 trainer 直接成为所有模块的依赖汇聚点

Brush 的 `SplatTrainer` 不是 workspace 的“中心节点”。

RustGS 的 `metal_trainer.rs` 现在直接依赖:

- `clustering`
- `data_loading`
- `eval`
- `events`
- `frame_targets`
- `metal_backward`
- `metal_forward`
- `metal_loss`
- `metal_optimizer`
- `metal_runtime`
- `parity_harness`
- `pose_embedding`
- `runtime_splats`
- `splats`
- `telemetry`
- `topology`

这个 fan-in 已经说明它承担的是“集成器”角色，而不是纯 trainer 角色。

### 2.3 用 dedicated module 承载 cross-cutting state

Brush 里配置、消息、统计、初始化都单独建模。

RustGS 现在的 `MetalTrainer` 字段很多，但大部分字段其实属于不同生命周期:

- 固定配置
- 可选特性状态
- step 运行时状态
- topology 状态
- telemetry 状态

如果这些不先分组，文件拆开后仍然会因为共享太多字段而重新耦合回去。

## 三、RustGS 当前训练架构的现状

### 3.1 现有优点

RustGS 其实已经不是“完全没拆”的状态了，做得好的地方包括:

- 顶层路由已经从 `metal_trainer.rs` 抽走:
  - `orchestrator.rs`
  - `execution_plan.rs`
  - `chunk_training.rs`
- Metal 子系统已经拆开:
  - `metal_runtime.rs`
  - `metal_forward.rs`
  - `metal_backward.rs`
  - `metal_loss.rs`
  - `metal_optimizer.rs`
  - `topology.rs`
- 数据和评估相关也有独立模块:
  - `data_loading.rs`
  - `frame_loader.rs`
  - `frame_targets.rs`
  - `eval.rs`

这说明 RustGS 不需要“推倒重来”，只需要继续沿着已经出现的模块边界推进。

### 3.2 当前最大的问题不是算法，而是集中度

`RustGS/src/training/metal_trainer.rs` 当前:

- 总行数: 5895
- 测试起点: 第 2769 行
- 测试行数: 3127
- 非测试代码仍有: 2768 行

非测试部分里还同时放着:

- `MetalTrainer` 主状态
- frame 准备
- session 初始化
- 训练主循环
- 单步 step 逻辑
- topology 调度与 mutation aftermath
- pose / cluster 可选特性拼装
- 内存预算估算
- chunk capacity 预估
- top-level `train_splats_with_report`

所以它现在不是“trainer implementation file”，而是“Metal training integration module”。

### 3.3 `metal_trainer.rs` 当前承担的职责

按职责看，大致是下面这些块:

1. 数据与状态定义
   - `MetalTrainingFrame`
   - `MetalTrainer`
   - `MetalTrainingStats`
   - `MetalStepOutcome`

2. telemetry / profiling / debug helpers
   - loss curve sample
   - abs stats
   - debug probe
   - `MetalStepProfile`

3. LiteGS 和 topology 窗口控制
   - SH 激活策略
   - refine window reset
   - gaussian stats 更新

4. topology integration
   - `maybe_apply_topology_updates`
   - mutation aftermath
   - cluster resync
   - runtime/adam rebuild

5. frame/session lifecycle
   - `prepare_frame`
   - `prepare_frames`
   - `initialize_training_session`

6. 训练执行
   - `train`
   - `train_loaded`
   - `training_step`

7. forward / loss / backward / optimizer glue
   - `render`
   - `loss_for_camera`
   - `apply_parameter_grads`
   - `forward_settings`

8. 顶层入口和预检
   - `train_splats_with_report`
   - pose init
   - cluster init
   - memory guard

9. chunk capacity / memory estimation
   - `estimate_chunk_capacity`
   - `training_memory_budget`
   - `estimate_peak_memory_*`

10. 大量测试

换句话说，RustGS 当前的问题不是“没模块”，而是“模块边界没有继续推进到 trainer 这一层”。

## 四、Brush 和 RustGS 的关键差异

| 维度 | Brush | RustGS 当前 |
|---|---|---|
| 顶层组织 | workspace 多 crate | 单 crate，多模块 |
| 流程编排 | `brush-process` 独立 | `orchestrator` 已抽出，但 trainer 仍保留部分流程职责 |
| trainer 职责 | 只做 `step/refine` | 做入口、预检、frame、step、topology glue、capacity 估算 |
| forward/backward | `brush-render` / `brush-render-bwd` 独立 | 已独立 |
| config/message/stats | 明确独立文件 | 有部分独立，但 trainer 自带大量 state glue |
| 大文件风险 | 被 crate/file 边界控制 | `metal_trainer.rs` 仍是中心大文件 |
| 测试布局 | 分散到 crate 与模块 | 大量内联在 `metal_trainer.rs` |

## 五、RustGS 不应该“照搬 Brush”的地方

虽然 Brush 的架构方向值得借鉴，但 RustGS 现在不适合直接照搬成多 crate:

- RustGS 还处在训练后端快速演化阶段，内部 API 不够稳定。
- `training` 目录里已经有不少相互协作的 Candle/Metal 专用类型，过早 crate 化会引入大量公开 API 设计和循环依赖治理成本。
- 当前最大收益来自“模块级重构”，不是“包级重构”。

因此推荐路线是:

1. 先在 `training/` 内部把 `metal_trainer.rs` 拆成目录模块。
2. 等边界稳定后，再决定是否把 `training/metal/*` 提升成独立 crate。

## 六、RustGS 的目标结构

### 6.1 推荐的近期目标: 把 `metal_trainer.rs` 改造成目录模块

建议从单文件:

- `src/training/metal_trainer.rs`

重构为目录模块:

```text
src/training/
  metal_trainer/
    mod.rs
    entry.rs
    trainer.rs
    session.rs
    step.rs
    topology_integration.rs
    memory.rs
    telemetry.rs
    tests/
      memory_tests.rs
      step_tests.rs
      topology_tests.rs
      integration_tests.rs
```

### 6.2 每个文件的职责建议

#### `metal_trainer/mod.rs`

只做门面和重导出:

- `pub use trainer::MetalTrainer`
- `pub use memory::{estimate_chunk_capacity, ChunkCapacityEstimate, ChunkCapacityDisposition}`
- `pub(crate) use entry::train_splats_with_report`

目标:

- 外部模块继续只依赖 `metal_trainer`
- 内部实现可以自由拆

#### `metal_trainer/trainer.rs`

只放核心状态和少量构造逻辑:

- `MetalTrainer`
- `MetalTrainingStats`
- 必要的 `impl MetalTrainer::new`

同时把字段分组为子状态结构，避免一个 struct 装下所有概念:

建议拆成:

- `MetalTrainerConfig`
- `MetalTrainerState`
- `MetalTrainerTelemetryState`
- `MetalTrainerFeatures`

这样拆之后，`step.rs`、`topology_integration.rs`、`telemetry.rs` 可以更清楚地只依赖自己该操作的那部分状态。

#### `metal_trainer/session.rs`

负责训练 session 的准备与 frame 生命周期:

- `MetalTrainingFrame`
- `prepare_frame`
- `prepare_frames`
- `initialize_training_session`

这个模块的职责是“把 dataset / loaded frames 变成 step 可消费的状态”，不要掺杂优化逻辑。

#### `metal_trainer/step.rs`

负责单步训练执行:

- `training_step`
- `render`
- `forward_settings`
- `loss_weights`
- `loss_config`
- `total_loss_for_render_result`
- `loss_for_camera`
- `apply_parameter_grads`
- `render_colors_for_camera`
- `compute_lr_pos`
- `synchronize_if_needed`

这是未来最像 Brush 里 `brush-train/src/train.rs` 的文件。

目标是让它回答的唯一问题变成:

“给我一个 frame 和 trainer 当前状态，我如何完成一次 forward/loss/backward/update？”

#### `metal_trainer/topology_integration.rs`

负责把 `topology.rs` 和 trainer 状态连接起来:

- `maybe_apply_topology_updates`
- `apply_topology_mutation_aftermath`
- `rebuild_adam_state`
- `max_topology_gaussians`
- `sync_cluster_assignment`
- `cluster_visible_mask_for_camera`
- `clustering_positions`
- LiteGS topology window / refine window helpers
- `update_gaussian_stats`
- opacity reset / refine decay

注意这里叫 `topology_integration`，而不是再起一个 `topology.rs`，避免和现有 `training/topology.rs` 概念混淆。

这里的核心原则是:

- `training/topology.rs` 负责“策略与计划”
- `metal_trainer/topology_integration.rs` 负责“把计划落到 runtime splats / adam / cluster / telemetry 上”

#### `metal_trainer/memory.rs`

负责预检与容量估算:

- `ChunkCapacityEstimate`
- `MetalMemoryEstimate`
- `MetalMemoryBudget`
- `estimate_chunk_capacity`
- `training_memory_budget`
- `affordable_initial_gaussian_cap`
- `preflight_initial_gaussian_cap`
- `estimate_peak_memory_*`
- `assess_memory_estimate`
- `detect_metal_memory_budget`
- `format_memory` / `bytes_to_gib` / `gib_to_bytes`

这个模块非常适合从 trainer 核心中独立出来，因为:

- 它本质上不依赖 step 逻辑
- `execution_plan.rs` 与 `chunk_training.rs` 其实需要的是 capacity estimator，而不是 trainer 本身

一旦抽出来，`execution_plan.rs` 和 `chunk_training.rs` 就不必再“为了算容量而依赖 trainer 模块”。

#### `metal_trainer/telemetry.rs`

负责:

- `MetalStepProfile`
- `MetalStepOutcome`
- `should_record_loss_curve_sample`
- `record_loss_curve_sample`
- `current_telemetry`
- `duration_ms`
- `should_profile_iteration`
- debug stats helpers

目标是把“训练行为”与“训练观测”分开。

#### `metal_trainer/entry.rs`

负责顶层 Metal 训练入口:

- `train_splats`
- `train_splats_with_report`
- `effective_metal_config`
- dataset load
- preflight
- pose init
- cluster init
- final report assembly

这个模块本质上对应 Brush 的 `brush-process::train_stream` 中与训练内核耦合最紧的那一层。

它不应该再和 `training_step` 混在一个文件里。

## 七、`metal_trainer.rs` 的具体拆分映射

建议按下面顺序迁移，不要一次性大搬家。

### 第 1 步: 先把测试全部挪走

当前收益最大、风险最低。

把 `#[cfg(test)] mod tests` 从 `metal_trainer.rs` 移到:

- `metal_trainer/tests/memory_tests.rs`
- `metal_trainer/tests/step_tests.rs`
- `metal_trainer/tests/topology_tests.rs`
- `metal_trainer/tests/render_tests.rs`

先做这一步就能把文件从 5895 行降到约 2768 行。

### 第 2 步: 抽出 `memory.rs`

把这些函数整体迁走:

- `estimate_chunk_capacity`
- `training_memory_budget`
- `affordable_initial_gaussian_cap`
- `preflight_initial_gaussian_cap`
- `estimate_peak_memory`
- `estimate_peak_memory_with_source_pixels`
- `assess_memory_estimate`
- `detect_metal_memory_budget`
- `resolve_chunk_memory_budget`
- `detect_physical_memory_bytes`
- `apply_ratio`
- `bytes_to_gib`
- `gib_to_bytes`
- `format_memory`

这一步做完后:

- `execution_plan.rs` 依赖 `memory` 而不是 `metal_trainer`
- `chunk_training.rs` 依赖 `memory` + `entry`

这是降低耦合的关键一步。

### 第 3 步: 抽出 `entry.rs`

把顶层训练入口迁走:

- `train_splats`
- `train_splats_with_report`
- `effective_metal_config`

外加:

- pose embeddings 初始化
- cluster assignment 初始化
- preflight guard
- final `TrainingRunReport` 拼装

做完这一步后，`MetalTrainer` 就不再需要承担 dataset load / report assembly / config normalization。

### 第 4 步: 抽出 `session.rs`

把这些放到 session:

- `MetalTrainingFrame`
- `prepare_frame`
- `prepare_frames`
- `initialize_training_session`

这样 `train` / `train_loaded` 就会更接近:

- session setup
- loop
- step call
- topology call

### 第 5 步: 抽出 `step.rs`

把 `training_step` 及其直接依赖整体迁走。

这一步是最重要的，因为它决定以后谁才是真正的“trainer 核心文件”。

迁完后 `step.rs` 应该只关心:

- 当前一帧怎么渲染
- 怎么算 loss
- 怎么反向
- 怎么更新参数

### 第 6 步: 抽出 `topology_integration.rs`

这是第二大块。

迁移:

- `maybe_apply_topology_updates`
- runtime rebuild / adam rebuild / cluster resync
- gaussian stats update
- LiteGS reset / decay

最终形成一个清晰的边界:

- `step.rs` 只完成一步优化
- `topology_integration.rs` 在步之后决定是否改变点集拓扑

### 第 7 步: 收敛 `MetalTrainer` 的字段

文件拆开后，再做一次结构收敛:

- 把 immutable config 聚合
- 把 telemetry 聚合
- 把 runtime mutable state 聚合
- 把 optional feature state 聚合

如果不做这一步，虽然文件拆了，但每个模块还会因为访问太多字段而维持高耦合。

## 八、推荐的 `MetalTrainer` 新结构

建议把当前大 flat struct 调整成下面的样子:

```rust
pub struct MetalTrainer {
    config: MetalTrainerConfig,
    state: MetalTrainerState,
    telemetry: MetalTrainerTelemetryState,
    features: MetalTrainerFeatures,
    runtime: MetalRuntime,
}
```

### `MetalTrainerConfig`

放基本不随训练过程变化的内容:

- `training_profile`
- `litegs`
- render dimensions
- pixel counts
- chunk size
- 各类 learning rate / beta / eps
- topology thresholds / intervals
- `max_iterations`
- `max_gaussian_budget`
- `topology_memory_budget`

### `MetalTrainerState`

放训练过程中的运行时状态:

- `adam`
- `gaussian_stats`
- `iteration`
- `last_step_duration`
- `cached_target_frame_idx`
- `rotation_frozen`
- `active_sh_degree`

### `MetalTrainerTelemetryState`

放观测面状态:

- `last_loss_terms`
- `topology_metrics`
- `last_learning_rates`
- `last_depth_valid_pixels`
- `last_depth_grad_scale`
- `loss_curve_samples`
- `loss_history`

### `MetalTrainerFeatures`

放可选特性:

- `pose_embeddings`
- `cluster_assignment`
- 未来如果有 frustum cache / dynamic masks 也放这里

这样做的好处是:

- `step.rs` 大多只需要 `config + state + runtime + telemetry`
- `topology_integration.rs` 主要操作 `state + features + telemetry`
- `entry.rs` 主要操作 `config + features`

## 九、对 `orchestrator` / `execution_plan` / `chunk_training` 的配套调整

为了真正学到 Brush 的边界设计，不应该只拆文件名，还应该顺手收紧调用关系。

### 9.1 `execution_plan.rs`

当前它直接依赖 `metal_trainer::estimate_chunk_capacity`。

建议改为依赖:

- `training::metal_trainer::memory::estimate_chunk_capacity`

或者如果想把 API 稳定暴露出来:

- `training::estimate_chunk_capacity`

原则是:

- execution plan 依赖 capacity estimator
- 不依赖 trainer 内核本身

### 9.2 `chunk_training.rs`

建议只依赖两个公开接口:

- `entry::train_splats_with_report`
- `memory::ChunkCapacityEstimate`

不要把 `chunk_training` 和 `MetalTrainer` struct 耦合起来。

### 9.3 `orchestrator.rs`

`orchestrator` 继续保留顶层职责是对的。

但未来它调用的应该是:

- 标准训练入口
- chunked 训练入口
- execution plan

而不是再穿透到 trainer 内部的 estimator / config glue。

## 十、RustGS 从 Brush 真正应该学的不是“多 crate”，而是“边界纪律”

对 RustGS 来说，最该吸收的是下面四条纪律:

1. trainer 只做训练语义，不做产品流程。
2. 配置、消息、统计、capacity 估算不要混在 trainer 文件里。
3. topology 策略和 topology 落地 glue 分开。
4. 测试按关注点拆，不把所有测试压在主实现文件末尾。

只要把这四条执行好，即使 RustGS 仍然保持单 crate，也能把当前训练代码从“大文件可工作”演化到“模块边界稳定、可继续增长”。

## 十一、推荐的实际实施顺序

### 阶段 A: 低风险，优先做

1. 挪走 `metal_trainer.rs` 内联测试。
2. 抽出 `memory.rs`。
3. 抽出 `entry.rs`。

这三步做完，文件体积和耦合会立刻下降，而且对训练数值行为几乎没有影响。

### 阶段 B: 中风险，核心收益最大

4. 抽出 `session.rs`。
5. 抽出 `step.rs`。
6. 抽出 `topology_integration.rs`。

这一步完成后，`metal_trainer` 才会真正从“集成中心”变回“trainer 门面”。

### 阶段 C: 结构收敛

7. 重组 `MetalTrainer` 字段为子状态结构。
8. 清理模块间 helper 的可见性。
9. 让 `execution_plan.rs` / `chunk_training.rs` 不再依赖 trainer 核心。

### 阶段 D: 可选的后续升级

10. 如果 `metal_*` 文件继续增长，再考虑把 `training/metal/` 升级为目录命名空间。
11. 只有在 API 稳定后，再评估是否 crate 化。

## 十二、一个务实判断

如果只允许做一件事，我建议做这件事:

把 `metal_trainer.rs` 先拆成:

- `entry.rs`
- `memory.rs`
- `step.rs`
- `topology_integration.rs`
- `tests/*`

原因很简单:

- 这是对当前痛点命中率最高的一刀。
- 它直接对应 Brush 最成功的边界设计。
- 它不会要求你现在就重写 `metal_forward`、`metal_backward`、`topology` 等已经拆出来的核心模块。

## 附录 A: 关键量化对比

### Brush

- workspace 成员: 21 个 crate
- `brush-train` 总行数: 1917
- `brush-train/src/train.rs`: 684
- `brush-render` + `brush-render-bwd`: 2006

### RustGS

- `training` 目录总行数: 23734
- `metal_trainer.rs`: 5895
- `topology.rs`: 2355
- `metal_forward.rs`: 1388
- `metal_backward.rs`: 1100
- `metal_runtime.rs`: 905
- `metal_trainer.rs` 内测试: 3127
- `metal_trainer.rs` 非测试代码: 2768

## 附录 B: 一句总结

Brush 的成功做法是: 让 trainer 成为“训练算法核心”，而不是“整个训练系统的装配车间”。

RustGS 下一阶段最值得做的，就是把 `metal_trainer.rs` 从装配车间拆回训练核心。
