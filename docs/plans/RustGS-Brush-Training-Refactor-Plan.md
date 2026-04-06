# RustGS 重构方案

## Context

RustGS 当前架构存在严重的 "God Object" 问题：`metal_trainer.rs` 有 8696 行代码，职责混杂（训练循环、渲染、拓扑、优化器、内存管理全部混在一起）。对比 Brush 的 `train.rs` 仅 673 行，渲染和反向传播分离到独立模块。

本次重构参考 Brush 架构，将 RustGS 拆分为清晰的模块边界，但保持 Metal 专用（不追求跨平台）。

## 当前问题

### 1. God Object: `metal_trainer.rs` (8696 行)
```
RustGS/src/training/metal_trainer.rs:
├── 训练循环 (step 函数)
├── 渲染逻辑 (project_gaussians, rasterize)
├── 反向传播 (backward pass)
├── 拓扑控制 (densify, prune, split, clone)
├── 优化器 (Adam state)
├── 内存管理 (chunk planning)
├── SH 计算 (球谐函数)
└── 聚类 (cluster assignment)
```

### 2. 数据结构分裂
```rust
// 两套不兼容的结构
Gaussian3D          // 存储/序列化
TrainableGaussians  // 训练
GaussianMap          // 高斯集合
```

### 3. Kernel 与业务逻辑耦合
- Metal shader 代码内嵌在 `metal_runtime.rs` 中
- 无法独立测试 kernel

### 4. 模块边界模糊
```
training/
├── metal_trainer.rs    # 万物皆在其中
├── metal_runtime.rs    # 3320 行，混合了 kernel + buffer 管理
├── topology.rs         # 拓扑逻辑但 trainer 也有
└── density_controller.rs # 新增但与 topology.rs 重叠
```

---

## 目标架构

参考 Brush，RustGS 重构为以下模块结构：

```
RustGS/src/
├── core/
│   ├── splats.rs           # 统一的 Gaussian 数据结构 (新)
│   ├── camera.rs           # 相机模型
│   └── bounds.rs           # 包围盒
│
├── kernel/                 # Metal Kernel 抽象层 (新)
│   ├── mod.rs
│   ├── shaders/            # MSL shader 文件
│   │   ├── project.msl
│   │   ├── rasterize.msl
│   │   └── backward.msl
│   └── dispatch.rs         # Kernel dispatch 逻辑
│
├── render/                 # 前向渲染 (拆分)
│   ├── mod.rs
│   ├── projector.rs        # 高斯投影
│   ├── rasterizer.rs       # 光栅化
│   └── output.rs           # RenderOutput, RenderAux
│
├── backward/               # 反向传播 (新)
│   ├── mod.rs
│   ├── rasterize_bwd.rs    # 光栅化反向
│   └── projector_bwd.rs    # 投影反向
│
├── train/                  # 训练循环 (策略模式)
│   ├── mod.rs
│   ├── trainer.rs          # 主训练器 (~500 行)
│   ├── strategy.rs         # TrainStrategy trait
│   ├── optimizer.rs        # Adam + 学习率调度
│   ├── loss.rs             # L1, SSIM, LPIPS (LossStrategy)
│   └── stats.rs            # 训练统计
│
├── topology/               # 拓扑控制 (策略模式)
│   ├── mod.rs              # TopologyStrategy trait
│   ├── strategy.rs         # 策略定义
│   ├── brush.rs            # Brush 风格策略 (初始实现)
│   └── litegs.rs           # LiteGS 策略 (后续添加)
│
├── io/                     # 数据加载
│   └── ... (保持不变)
│
└── lib.rs
```

---

## 核心变更

### 1. 统一数据结构: `Splats`

```rust
// core/splats.rs
/// 统一的高斯数据结构，用于存储、训练、渲染
pub struct Splats {
    /// [N, 10] = position(3) + rotation(4) + log_scale(3)
    pub transforms: Tensor,  // Metal buffer
    /// [N, coeffs, 3] SH 系数
    pub sh_coeffs: Tensor,
    /// [N] 原始 opacity (sigmoid 前的值)
    pub raw_opacities: Tensor,
}

impl Splats {
    pub fn means(&self) -> Tensor { self.transforms.slice(.., 0..3) }
    pub fn rotations(&self) -> Tensor { self.transforms.slice(.., 3..7) }
    pub fn log_scales(&self) -> Tensor { self.transforms.slice(.., 7..10) }
    pub fn opacities(&self) -> Tensor { sigmoid(self.raw_opacities) }
    pub fn scales(&self) -> Tensor { self.log_scales().exp() }

    pub fn len(&self) -> usize { self.transforms.dim(0) }
    pub fn to_gaussian3d(&self) -> Vec<Gaussian3D> { ... }
    pub fn from_gaussian3d(gaussians: &[Gaussian3D], device: &Device) -> Self { ... }
}
```

### 2. Kernel 抽象层

```rust
// kernel/mod.rs
pub trait MetalKernel {
    fn dispatch(&self, command_buffer: &MetalCommandBuffer, args: KernelArgs);
}

// kernel/shaders/project.msl
// 分离的 Metal shader 文件

// kernel/dispatch.rs
pub fn project_gaussians(
    splats: &Splats,
    camera: &Camera,
    config: &ProjectConfig,
) -> ProjectOutput { ... }

pub fn rasterize(
    projected: &ProjectedSplats,
    config: &RasterConfig,
) -> RenderOutput { ... }
```

### 3. 渲染模块拆分

```rust
// render/mod.rs
pub struct RenderOutput {
    pub color: Tensor,           // [H, W, 3]
    pub depth: Tensor,           // [H, W]
    pub aux: RenderAux,          // 辅助信息
    pub backward_state: BackwardState,  // 反向传播所需状态
}

pub struct RenderAux {
    pub num_visible: u32,
    pub visible_mask: Tensor,
    pub tile_offsets: Tensor,
}

// render/projector.rs
pub fn project(splats: &Splats, camera: &Camera) -> ProjectedSplats;

// render/rasterizer.rs
pub fn rasterize(projected: &ProjectedSplats, config: &Config) -> RenderOutput;
```

### 4. 反向传播独立

```rust
// backward/mod.rs
pub struct SplatGrads {
    pub v_positions: Tensor,
    pub v_rotations: Tensor,
    pub v_scales: Tensor,
    pub v_opacities: Tensor,
    pub v_sh_coeffs: Tensor,
}

pub fn backward(
    render_output: &RenderOutput,
    grad_output: &Tensor,
) -> SplatGrads;
```

### 5. 训练器简化

```rust
// train/trainer.rs (~500 行)
pub struct Trainer {
    config: TrainConfig,
    splats: Splats,
    optimizer: AdamOptimizer,
    lr_scheduler: LrScheduler,
    refine_stats: RefineStats,
    ssim: Option<Ssim>,
    
    // 组合策略
    topology: Box<dyn TopologyStrategy>,
    loss: Box<dyn LossStrategy>,
}

impl Trainer {
    pub fn new(config: TrainConfig) -> Self {
        let topology = config.topology.create_strategy();
        let loss = config.loss.create_strategy();
        // ...
    }
    
    pub fn step(&mut self, batch: &FrameBatch) -> StepStats {
        // 1. Forward
        let output = render(&self.splats, &batch.camera, &self.config.render)?;

        // 2. Loss
        let loss = self.loss.compute(&output, &batch)?;

        // 3. Backward
        let grads = backward(&output, &loss.grad)?;

        // 4. Optimizer step
        self.optimizer.step(&mut self.splats, &grads, &self.lr_scheduler)?;

        // 5. Refine (if needed)
        if self.topology.should_refine(&self.stats) {
            self.topology.refine(&mut self.splats, &grads);
        }

        Ok(stats)
    }
}
```

### 6. 策略模式设计

支持多套训练/拓扑策略，通过 TOML 配置文件切换：

```rust
// topology/strategy.rs
pub trait TopologyStrategy {
    fn name(&self) -> &'static str;
    
    fn should_refine(&self, stats: &RefineStats) -> bool;
    fn refine(&mut self, splats: &mut Splats, grads: &SplatGrads);
}

// topology/brush.rs (初始实现)
pub struct BrushTopology {
    config: BrushConfig,
}

impl TopologyStrategy for BrushTopology {
    fn name(&self) -> &'static str { "brush" }
    
    fn should_refine(&self, stats: &RefineStats) -> bool {
        stats.iter % self.config.refine_interval == 0
    }
    
    fn refine(&mut self, splats: &mut Splats, grads: &SplatGrads) {
        // Brush 风格的 densify + prune + reset_opacity
    }
}

// topology/litegs.rs (后续添加)
pub struct LiteGsTopology { ... }
```

```rust
// train/loss.rs
pub trait LossStrategy {
    fn name(&self) -> &'static str;
    fn compute(&self, output: &RenderOutput, target: &TargetImage) -> LossOutput;
}

pub struct DefaultLoss;       // L1 + SSIM
pub struct LpipsLoss;         // L1 + SSIM + LPIPS
```

```rust
// config.rs
impl TopologyConfig {
    pub fn create_strategy(&self) -> Box<dyn TopologyStrategy> {
        match self.strategy.as_str() {
            "brush" => Box::new(BrushTopology::new(self)),
            // "litegs" => Box::new(LiteGsTopology::new(self)),  // 后续添加
            _ => panic!("Unknown strategy: {}", self.strategy),
        }
    }
}
```

**TOML 配置切换示例：**

```toml
# experiments/brush.toml
[topology]
strategy = "brush"
refine_interval = 100
freeze_iter = 80

[loss]
type = "default"  # L1 + SSIM

# experiments/litegs.toml (后续)
[topology]
strategy = "litegs"
refine_interval = 100

[loss]
type = "lpips"
```

**运行：**
```bash
cargo run --config experiments/brush.toml
```
```

---

## 实施步骤

### Phase 1: 统一数据结构 (Week 1)

1. 创建 `core/splats.rs`，实现 `Splats` 结构
2. 添加 `Gaussian3D` ↔ `Splats` 转换
3. 修改 `io/scene_io.rs` 直接导出 `Splats`
4. 更新测试确保兼容性

### Phase 2: 拆分渲染器 (Week 2)

1. 从 `metal_trainer.rs` 提取投影逻辑 → `render/projector.rs`
2. 提取光栅化逻辑 → `render/rasterizer.rs`
3. 提取 `RenderOutput` → `render/output.rs`
4. 创建 `render/mod.rs` 整合

### Phase 3: 拆分反向传播 (Week 3)

1. 创建 `backward/` 目录
2. 从 `metal_trainer.rs` 提取反向逻辑
3. 实现 `backward()` 函数
4. 添加反向传播测试

### Phase 4: 拓扑策略模式 (Week 4)

1. 定义 `TopologyStrategy` trait → `topology/strategy.rs`
2. 实现 `BrushTopology` 策略 → `topology/brush.rs` (初始唯一策略)
3. 从现有代码迁移 densify/prune/reset 逻辑到 BrushTopology
4. 移除 `density_controller.rs` 重叠代码
5. 配置驱动策略选择 → `config.rs`

### Phase 5: 训练策略 + 简化训练器 (Week 5)

1. 定义 `LossStrategy` trait → `train/loss.rs`
2. 实现 `DefaultLoss` (L1 + SSIM)
3. 重写 `trainer.rs` (~500 行)，组合策略
4. 提取 optimizer → `train/optimizer.rs`
5. 清理 `training_pipeline.rs`（删除或保留为兼容层）

### Phase 6: Kernel 抽象 (Week 6)

1. 创建 `kernel/` 目录
2. 分离 MSL shader 到独立文件
3. 实现 `MetalKernel` trait
4. 添加 kernel 单元测试

### Phase 7: 新增策略 (后续)

待 Brush 策略验证通过后：

1. 实现 `LiteGsTopology` → `topology/litegs.rs`
2. 实现 `LpipsLoss` → `train/loss.rs`
3. 更新配置 match 分支

---

## 文件大小目标

| 文件 | 当前行数 | 目标行数 |
|------|----------|----------|
| `trainer.rs` | 8696 | ~500 |
| `metal_runtime.rs` | 3320 | ~1000 |
| `mod.rs` (training) | 1738 | ~500 |
| `topology.rs` | 826 | ~400 (每个子模块) |

---

## 策略演进计划

| 状态 | 策略 | 说明 |
|------|------|------|
| **Phase 4-6** | `BrushTopology` | 初始唯一拓扑策略，验证 Brush 风格训练 |
| **Phase 4-6** | `DefaultLoss` | 初始唯一 loss 策略 (L1 + SSIM) |
| **Phase 7+** | `LiteGsTopology` | 待验证通过后添加 |
| **Phase 7+** | `LpipsLoss` | 待验证通过后添加 |

### 策略模式好处

| 方面 | 优势 |
|------|------|
| **切换简单** | 只改 TOML 配置文件，无需改代码 |
| **并行实验** | 多个配置文件可同时跑不同实验 |
| **参数独立** | 每个策略有自己参数，不会互相污染 |
| **扩展方便** | 新策略只需实现 Trait，加一个 match 分支 |
| **测试独立** | 每个策略可单独单元测试 |

---

## 验证方案

1. **单元测试**
   - 每个 `Splats` 方法有测试
   - 渲染器 forward/backward 一致性测试
   - 拓扑操作的 ID 保持测试

2. **集成测试**
   - 完整训练流程测试（TUM 数据集）
   - PSNR 不退化

3. **回归测试**
   - 使用 `parity_harness.rs` 验证输出一致性

---

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 重构过程中功能退化 | 每个 Phase 后运行完整测试套件 |
| 性能下降 | 基准测试对比重构前后 |
| API 不兼容 | 保持 `train()` 入口不变，内部重构 |