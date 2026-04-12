# RustGS: candle-metal → burn+wgpu 迁移方案

日期: 2026-04-12

## 目的

本文档规划 RustGS 从 `candle-core` + `candle-metal` + 手写 `.metal` 着色器迁移到 `burn` + `wgpu` 架构的完整技术路线。

参考项目: `/Users/tfjiang/Projects/brush`（仅参考架构思路，不引入 brush crate 依赖）。

## 一、动机

### 1.1 当前问题

- **平台限制**: candle-metal 仅支持 Apple Silicon Metal，无法在 Linux/Windows/WebGPU 上运行
- **shader 维护负担**: 7 个手写 `.metal` 着色器，每次修改需要同时维护 Rust 绑定 + Metal kernel
- **autodiff 缺失**: candle 没有完整的自动微分图，RustGS 手写了 `analytical_backward.rs`，每新增一个可微参数就要手写 backward kernel
- **生态孤立**: candle-metal 生态较小，GPU primitives (sort, prefix sum) 需要自己实现或用 CPU fallback

### 1.2 目标架构

- **burn + wgpu**: 跨平台 GPU compute（Metal / Vulkan / DX12 / WebGPU）
- **WGSL 着色器**: 单一 shader 语言覆盖所有平台
- **burn autodiff**: 自动微分图管理，自定义 op 只需注册 forward/backward
- **Fusion backend**: 运算融合减少 GPU dispatch 开销

### 1.3 参考 brush 的什么

| 参考内容 | 具体来源 |
|---|---|
| Backend 类型定义 | `brush-render/src/lib.rs` 的 `MainBackendBase` / `MainBackend` |
| Splats 打包布局 | `brush-render/src/gaussian_splats.rs` 的 `[N,10]` transforms |
| 5 阶段渲染管线 | `brush-render/src/render.rs` 的 `project → sort → project_visible → tile_map → rasterize` |
| WGSL shader 结构 | `brush-render/src/shaders/` 的 5 个着色器 |
| Backward autodiff 集成 | `brush-render-bwd/src/burn_glue.rs` 的 `Backward` trait 模式 |
| Per-column Adam LR | `brush-train/src/adam_scaled.rs` 的 `AdamScaled` |
| GPU radix sort | `brush-sort/src/lib.rs` |
| GPU prefix sum | `brush-prefix-sum/src/lib.rs` |

**不依赖 brush crate**，全部自行实现。

## 二、依赖变更

### 2.1 移除

```toml
# GPU dependencies (删除)
candle-core = { version = "0.9.2", features = ["metal"] }
candle-metal = { version = "0.27.1", features = ["mps"] }
candle-metal-kernels = { version = "0.9.2" }
objc2-foundation = { version = "0.3.2" }
objc2-metal = { version = "0.3.2" }
```

### 2.2 新增

```toml
# GPU dependencies (新增)
burn = { git = "https://github.com/tracel-ai/burn", features = ["autodiff", "wgpu"] }
burn-wgpu = { git = "https://github.com/tracel-ai/burn", features = ["exclusive-memory-only"] }
burn-cubecl = { git = "https://github.com/tracel-ai/burn" }
burn-fusion = { git = "https://github.com/tracel-ai/burn" }
burn-ir = { git = "https://github.com/tracel-ai/burn" }
wgpu = { version = "29", default-features = false, features = ["naga-ir"] }
naga_oil = "0.22"
bytemuck = { version = "1.20", features = ["derive"] }
tokio = { version = "1", features = ["rt"] }
```

### 2.3 保留不变

```toml
rustscan-types = { path = "../rustscan-types" }
glam = { workspace = true }
rayon = "1.8"
rand = "0.8"
serde = { workspace = true }
serde_json = "1.0"
thiserror = { workspace = true }
kiddo = "5.2.1"
image = "0.25"
log = "0.4"
```

### 2.4 Features 更新

```toml
[features]
default = ["gpu", "cli"]
gpu = [
    "dep:burn",
    "dep:burn-wgpu",
    "dep:burn-cubecl",
    "dep:burn-fusion",
    "dep:burn-ir",
    "dep:wgpu",
    "dep:naga_oil",
    "dep:bytemuck",
    "dep:tokio",
]
cli = ["dep:clap", "dep:env_logger", "dep:anyhow"]
```

`libc` 依赖保留与否取决于是否还有非 GPU 代码使用它。

## 三、新模块结构

### 3.1 新建: `src/training/wgpu/`

```text
src/training/wgpu/
├── mod.rs                      # 模块导出
├── backend.rs                  # Backend 类型别名
├── splats.rs                   # DeviceSplats<B> 及 HostSplats 桥接
├── adam_scaled.rs              # Per-column LR scaling 的 Adam 优化器
├── loss.rs                     # L1 + SSIM 损失计算
├── topology_bridge.rs          # GPU → CPU 参数快照（喂给 topology 分析）
├── topology_apply.rs           # topology mutation → burn tensor ops
├── trainer.rs                  # WgpuTrainer 主训练循环
├── entry.rs                    # 同步入口 (tokio::runtime::block_on wrapper)
│
├── render/                     # Forward 渲染管线
│   ├── mod.rs                  # 渲染管线编排
│   ├── projection.rs           # Stage 1: 3D→2D 投影 + 视锥裁剪
│   ├── sorting.rs              # Stage 2: GPU depth radix sort
│   ├── project_visible.rs      # Stage 3: 2D 协方差 + SH 求值
│   ├── tile_mapping.rs         # Stage 4: Gaussian→Tile 映射 + prefix sum
│   └── rasterize.rs            # Stage 5: Tile-based alpha compositing
│
├── render_bwd/                 # Backward 渲染管线
│   ├── mod.rs                  # Backward 管线编排
│   ├── rasterize_bwd.rs        # 反向光栅化 kernel
│   ├── project_bwd.rs          # 反向投影 kernel
│   └── autodiff.rs             # Burn autodiff 自定义 op 注册
│
├── shaders/                    # WGSL 计算着色器
│   ├── helpers.wgsl            # 共享函数 (quat→mat3, SH 基函数, sigmoid 等)
│   ├── project_forward.wgsl    # 投影 + 裁剪
│   ├── project_visible.wgsl    # 可见 splat 2D 协方差 + SH color
│   ├── map_gaussian_to_intersects.wgsl  # Tile binning
│   ├── rasterize.wgsl          # Forward 光栅化
│   ├── rasterize_backwards.wgsl # Backward 光栅化
│   ├── project_backwards.wgsl  # Backward 投影
│   ├── radix_sort.wgsl         # GPU radix sort kernel
│   └── prefix_sum.wgsl         # GPU prefix sum kernel
│
└── gpu_primitives/             # GPU 基础算法
    ├── mod.rs
    ├── radix_sort.rs           # GPU radix sort 封装
    └── prefix_sum.rs           # GPU prefix sum 封装
```

### 3.2 删除

| 目标 | 文件数 | 说明 |
|---|---|---|
| `src/training/metal/` | ~15 文件 | 完整 Metal 训练后端 |
| `src/training/shaders/*.metal` | 7 文件 | Metal 计算着色器 |
| `src/diff/diff_splat.rs` | 1 | candle Var-based Splats |
| `src/diff/analytical_backward.rs` | 1 | 手写 backward pass |
| `src/diff/diff_renderer.rs` | 1 | candle differentiable renderer |

### 3.3 保留不变

| 模块 | 行数 | 说明 |
|---|---|---|
| `src/training/topology/mod.rs` | ~2243 | 核心 IP: 拓扑策略与分析 |
| `src/training/topology/density_controller.rs` | - | 密度控制逻辑 |
| `src/training/state/splats.rs` | - | HostSplats CPU 存储 |
| `src/training/state/runtime_splats.rs` | - | TopologySplatMetrics |
| `src/training/data/` | - | 数据加载 |
| `src/training/pipeline/` | - | 训练编排（orchestrator 需小改） |
| `src/training/eval.rs` | - | PSNR 评估 |
| `src/training/parity_harness.rs` | - | 训练对比测试 |
| `src/io/` | - | PLY/TUM/COLMAP/NeRFStudio IO |
| `src/init/` | - | 点云初始化 |
| `src/core/` | - | GaussianCamera 等 |
| `src/render/` | - | CPU forward renderer (调试) |

### 3.4 修改

| 文件 | 修改内容 |
|---|---|
| `Cargo.toml` | 替换依赖 (§2) |
| `src/lib.rs` | `candle_core::Device` → `WgpuDevice`; `metal_available()` → `gpu_available()` |
| `src/training/mod.rs` | `mod metal` → `mod wgpu`; 更新 re-exports |
| `src/training/config.rs` | `TrainingBackend::Metal` → `Wgpu`; 简化 `TrainingProfile` |
| `src/training/state/splats.rs` | 新增 `from_device_splats<B>()` async 方法 |
| `src/training/pipeline/orchestrator.rs` | `metal_entry` → `wgpu_entry` 调度 |

## 四、核心类型设计

### 4.1 Backend 类型别名 (`backend.rs`)

```rust
use burn_cubecl::CubeBackend;
use burn_fusion::Fusion;
use burn_wgpu::WgpuRuntime;
use burn::backend::Autodiff;

/// 基础 wgpu compute backend
pub type GsBackendBase = CubeBackend<WgpuRuntime, f32, i32, u32>;

/// 加 fusion 优化的 backend（合并连续 GPU dispatch）
pub type GsBackend = Fusion<GsBackendBase>;

/// 可微分 backend（训练用）
pub type GsDiffBackend = Autodiff<GsBackend>;

/// Device 类型
pub type GsDevice = <GsBackend as burn::prelude::Backend>::Device;
```

### 4.2 DeviceSplats (`splats.rs`)

参考 brush 的打包布局，将 position/rotation/scale 合并为一个 `[N, 10]` tensor:

```rust
use burn::prelude::*;
use burn::module::Param;

/// GPU-resident differentiable Gaussian splat set
pub struct DeviceSplats<B: Backend> {
    /// Packed transforms: means(3) + quats(4) + log_scales(3) = [N, 10]
    pub transforms: Param<Tensor<B, 2>>,

    /// Spherical harmonics coefficients: [N, (degree+1)², 3]
    pub sh_coeffs: Param<Tensor<B, 3>>,

    /// Raw opacities (sigmoid activation in shader): [N]
    pub raw_opacities: Param<Tensor<B, 1>>,

    /// Whether to apply mip-splatting anti-aliasing
    pub render_mip: bool,
}
```

与旧结构的对应关系:

| 旧 (candle `Var`) | 新 (burn `Param<Tensor>`) |
|---|---|
| `positions: Var` [N,3] | `transforms[:,0..3]` |
| `rotations: Var` [N,4] | `transforms[:,3..7]` |
| `scales: Var` [N,3] (log-space) | `transforms[:,7..10]` |
| `opacities: Var` [N] | `raw_opacities: Param<Tensor<B,1>>` |
| `colors: Var` [N,3] + `sh_rest: Var` [N,K,3] | `sh_coeffs: Param<Tensor<B,3>>` [N,K+1,3] |

### 4.3 HostSplats ↔ DeviceSplats 桥接

```rust
/// CPU → GPU: 将 HostSplats 上传为 DeviceSplats
pub fn host_splats_to_device<B: Backend>(
    hs: &HostSplats,
    device: &B::Device,
) -> DeviceSplats<B>;

/// GPU → CPU: 将 DeviceSplats 下载为 HostSplats (async for GPU readback)
pub async fn device_splats_to_host<B: Backend>(
    splats: &DeviceSplats<B>,
) -> HostSplats;
```

`HostSplats` 本身不变 — 它仍然是 `Vec<f32>` 的平坦布局，负责 PLY 序列化、checkpoint、公开 API 返回值。

## 五、渲染管线设计

### 5.1 五阶段 Forward Pipeline

参考 brush 的五阶段设计:

```text
Stage 1: ProjectForward
  Input:  DeviceSplats (transforms [N,10]), camera uniforms
  Output: depths [N], intersect_counts [N], visibility mask
  Kernel: project_forward.wgsl
  Dispatch: 1 thread per Gaussian

Stage 2: DepthSort
  Input:  depths [num_visible]
  Output: sorted indices → global_from_compact_gid [num_visible]
  Method: GPU radix argsort (32-bit depth keys)
  Kernel: radix_sort.wgsl

Stage 3: ProjectVisible
  Input:  sorted visible Gaussians, camera uniforms
  Output: ProjectedSplat [num_visible] = {xy, conic(3), color(4)}
  Kernel: project_visible.wgsl
  Computation: 2D covariance from 3D, SH color evaluation

Stage 4: TileMapping
  Input:  ProjectedSplat bounding boxes
  Output: tile_id×compact_gid pairs, tile_offsets [tile_h, tile_w, 2]
  Steps:  map_gaussian_to_intersects → tile radix sort → prefix_sum
  Kernels: map_gaussian_to_intersects.wgsl, radix_sort.wgsl, prefix_sum.wgsl

Stage 5: Rasterize
  Input:  ProjectedSplats, tile_offsets, sorted indices
  Output: RGBA image [H, W, 4]
  Kernel: rasterize.wgsl
  Method: Per-tile workgroup, per-pixel front-to-back alpha compositing
```

### 5.2 GPU 数据结构 (bytemuck `#[repr(C)]`)

```rust
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ProjectUniforms {
    pub fx: f32, pub fy: f32, pub cx: f32, pub cy: f32,
    pub width: u32, pub height: u32,
    pub tile_size: u32, pub num_tiles_x: u32, pub num_tiles_y: u32,
    pub near: f32, pub far: f32,
    pub viewmat: [f32; 16],  // 4x4 view matrix, column-major
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ProjectedSplat {
    pub xy: [f32; 2],
    pub conic: [f32; 3],    // 2D inverse covariance (upper triangle)
    pub color: [f32; 4],    // RGBA from SH evaluation
}
```

### 5.3 Kernel 注册方式

使用 `wgpu::ComputePipeline` + `naga_oil` 运行时编译 WGSL:

```rust
/// 加载 WGSL 着色器并创建 compute pipeline
fn create_compute_pipeline(
    device: &wgpu::Device,
    shader_source: &str,
    entry_point: &str,
) -> wgpu::ComputePipeline;
```

通过 `burn_cubecl` 的 `CubeTask` 接口将 compute shader dispatch 集成到 burn 的执行图中。

### 5.4 Tile 参数

- **TILE_SIZE**: 16×16 (256 pixels per tile)，与 brush 保持一致
- **Workgroup size**: 每个 tile 一个 workgroup（256 threads）
- **Dispatch 限制**: WebGPU 65535 workgroup limit → 2D tiling 处理

## 六、Backward Pass + Autodiff 集成

### 6.1 Backward Kernels

两个 WGSL backward kernel:

1. **`rasterize_backwards.wgsl`** — 反向光栅化
   - Input: `dL/d(output_image) [H,W,4]` + forward 保存的中间值
   - Output: `dL/d(projected_splats) [num_visible, 10]` (sparse)
   - Method: per-tile reverse alpha compositing

2. **`project_backwards.wgsl`** — 反向投影
   - Input: `dL/d(projected_splats)` (sparse)
   - Output: `dL/d(transforms) [N,10]`, `dL/d(sh_coeffs) [N,K,3]`, `dL/d(opacities) [N]`
   - Method: chain rule through projection + SH evaluation

### 6.2 Burn Autodiff 注册 (`autodiff.rs`)

参考 brush 的 `burn_glue.rs` 模式:

```rust
/// 可微分渲染入口
pub fn render_splats<B: AutodiffBackend>(
    splats: DeviceSplats<B>,
    camera: &GaussianCamera,
    img_size: (usize, usize),
    background: [f32; 3],
) -> DiffRenderOutput<B>;

/// 内部: 注册自定义 backward op
struct RenderBackward;

impl<B: Backend> Backward<B, 3> for RenderBackward {
    type State = RenderCheckpoint<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
    ) {
        // 1. rasterize_bwd: dL/d(output) → dL/d(projected)
        // 2. project_bwd: dL/d(projected) → dL/d(transforms, sh, opacity)
    }
}
```

`RenderCheckpoint<B>` 保存 forward 中间值（projected splats, tile offsets, sorted indices），用于 backward 重用。

## 七、训练管线

### 7.1 AdamScaled (`adam_scaled.rs`)

参考 brush 的 `AdamScaled` 实现:

```rust
pub struct AdamScaled;

pub struct AdamScaledConfig {
    pub beta_1: f32,      // default: 0.9
    pub beta_2: f32,      // default: 0.999
    pub epsilon: f32,     // default: 1e-15
}

pub struct AdamScaledState<B: Backend, const D: usize> {
    pub moment_1: Tensor<B, D>,
    pub moment_2: Tensor<B, D>,
    pub scaling: Option<Tensor<B, D>>,  // per-column LR scaling
    pub time: usize,
}
```

核心设计:
- `scaling` tensor shape `[1, 10]` 用于 transforms 参数：`[lr_pos, lr_pos, lr_pos, lr_rot, lr_rot, lr_rot, lr_rot, lr_scale, lr_scale, lr_scale]`
- 每列乘以对应学习率再执行 Adam update
- 可选 `reduce_moment_2`: 对高维 SH 参数存标量方差（省显存）

### 7.2 Loss 计算 (`loss.rs`)

```rust
/// L1 + SSIM 光度损失
pub fn photometric_loss<B: AutodiffBackend>(
    pred: Tensor<B, 3>,     // [H, W, 3]
    gt: Tensor<B, 3>,       // [H, W, 3]
    ssim_weight: f32,       // default: 0.2
) -> Tensor<B, 1>;

/// SSIM 结构相似性 (11×11 patch)
pub fn ssim_loss<B: AutodiffBackend>(
    pred: Tensor<B, 3>,
    gt: Tensor<B, 3>,
) -> Tensor<B, 1>;

/// 可选: 深度损失
pub fn depth_loss<B: AutodiffBackend>(
    pred_depth: Tensor<B, 2>,   // [H, W]
    gt_depth: Tensor<B, 2>,
    mask: Tensor<B, 2>,         // valid pixels
) -> Tensor<B, 1>;
```

### 7.3 WgpuTrainer (`trainer.rs`)

```rust
pub struct WgpuTrainer {
    config: TrainingConfig,
    device: GsDevice,

    // 渲染尺寸
    render_width: usize,
    render_height: usize,

    // 优化器
    optim: OptimizerAdaptor<AdamScaled, DeviceSplats<GsDiffBackend>, GsDiffBackend>,

    // LR 调度
    sched_mean: ExponentialLrScheduler,
    sched_scale: ExponentialLrScheduler,

    // Topology 状态 (CPU-side, 不变)
    gaussian_stats: Vec<MetalGaussianStats>,
    topology_policy: TopologyPolicy,

    // 训练进度
    iteration: usize,
    scene_extent: f32,
}
```

主训练循环 (`train_loaded`):

```rust
pub async fn train_loaded(
    &mut self,
    splats: DeviceSplats<GsDiffBackend>,
    dataset: &TrainingDataset,
) -> Result<TrainingRun, TrainingError> {
    let mut splats = splats;

    for iter in 0..self.config.iterations {
        // 1. 采样训练帧
        let (camera, gt_image) = self.sample_frame(dataset, iter);

        // 2. 可微分渲染
        let diff_out = render_splats(
            splats.clone(), &camera, self.img_size(), self.background
        ).await;

        // 3. 计算损失
        let loss = photometric_loss(diff_out.image, gt_tensor, 0.2);

        // 4. 反向传播
        let grads = loss.backward();

        // 5. 参数更新
        splats = self.optim.step(lr, splats, grads);

        // 6. Topology 更新 (每 N 步)
        if self.topology_policy.should_update(iter) {
            let metrics = snapshot_topology_metrics(&splats).await;
            let plan = self.topology_policy.analyze(&metrics);
            splats = apply_topology_plan(splats, &plan, &mut self.optim);
        }
    }

    Ok(self.build_report(splats))
}
```

### 7.4 同步入口 (`entry.rs`)

```rust
pub fn train_splats_with_report(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<TrainingRun, TrainingError> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| TrainingError::Runtime(e.to_string()))?;

    rt.block_on(async {
        let device = GsDevice::default();
        let host_splats = initialize_host_splats(dataset, config)?;
        let device_splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);

        let mut trainer = WgpuTrainer::new(config, device)?;
        trainer.train_loaded(device_splats, dataset).await
    })
}
```

### 7.5 Topology 桥接

**不变部分:**
- `topology/mod.rs` (2243 行) — `TopologyPolicy`, `MetalGaussianStats`, `TopologyAnalysis`, `TopologyMutationPlan` 等全部保留
- `topology/density_controller.rs` — 克隆/分裂/剪枝候选分析

**新增桥接层:**

`topology_bridge.rs` — 从 GPU 读回参数快照:

```rust
pub async fn snapshot_topology_metrics<B: Backend>(
    splats: &DeviceSplats<B>,
) -> TopologySplatMetrics {
    let transforms_data = splats.transforms.val().into_data_async().await;
    let opacities_data = splats.raw_opacities.val().into_data_async().await;
    // 拆解 [N,10] → positions, rotations, log_scales
    // 构建 TopologySplatMetrics
}
```

`topology_apply.rs` — 将 mutation plan 应用到 burn tensors:

```rust
pub fn apply_topology_plan<B: AutodiffBackend>(
    splats: DeviceSplats<B>,
    plan: &TopologyMutationPlan,
    optim: &mut OptimizerAdaptor<...>,
) -> DeviceSplats<B> {
    // clone: Tensor::cat 追加新行
    // split: Tensor::cat 追加 + 原行 scale 减半
    // prune: Tensor::select 保留未剪枝行
    // 同步 Adam moment 状态的行数
}
```

## 八、Device Bootstrap

### 8.1 替换 `src/lib.rs` 中的 Metal 设备引导

```rust
// 旧:
pub fn metal_available() -> bool;
pub(crate) fn preferred_device() -> candle_core::Device;

// 新:
pub fn gpu_available() -> bool {
    // wgpu 总是可用 (软件 fallback)
    true
}

pub(crate) fn preferred_wgpu_device() -> GsDevice {
    GsDevice::default()  // 自动选择最佳 GPU
}
```

### 8.2 公开 API 签名

```rust
// 不变 — 公开 API 维持同步签名
pub fn train_splats(dataset: &TrainingDataset, config: &TrainingConfig) -> Result<HostSplats, TrainingError>;
pub fn train_splats_with_report(dataset: &TrainingDataset, config: &TrainingConfig) -> Result<TrainingRun, TrainingError>;
pub fn train_splats_with_events<F: FnMut(TrainingEvent)>(...) -> Result<TrainingRun, TrainingError>;
```

## 九、Config 更新

### 9.1 `TrainingBackend` 枚举

```rust
// 旧:
pub enum TrainingBackend { Metal }

// 新:
pub enum TrainingBackend { Wgpu }
```

### 9.2 `TrainingProfile` 简化

```rust
// 旧:
pub enum TrainingProfile { LegacyMetal, LiteGsMacV1 }

// 新:
pub enum TrainingProfile { Default, LiteGs }
// 或保留旧名做 serde alias 兼容
```

### 9.3 Config 字段清理

移除 `metal_*` 前缀的字段:

```rust
// 移除:
pub metal_render_scale: f32,
pub metal_gaussian_batch_size: usize,
pub metal_profile_steps: bool,
pub metal_profile_interval: usize,
pub metal_use_native_forward: bool,

// 替换为 (如需保留):
pub render_scale: f32,
pub gpu_batch_size: usize,
pub profile_steps: bool,
pub profile_interval: usize,
```

## 十、迁移策略

### 硬切换 (Hard Cut)

**原因:**
1. candle `Var` 和 burn `Param<Tensor>` 类型不兼容，无法在同一编译目标中并存
2. `.metal` 着色器和 `.wgsl` 着色器管线完全不同
3. Metal 后端代码渗透到 `diff_splat.rs`、backward、optimizer、topology snapshot 各处
4. wgpu 的 Metal backend 天然覆盖 Apple Silicon，不需要保留 candle-metal 做回退

### 实施顺序

```text
Phase 1: 脚手架 (不删旧代码)
  ├── Cargo.toml 添加 burn/wgpu deps (暂保留 candle)
  ├── 创建 src/training/wgpu/ 目录结构
  ├── 实现 backend.rs
  └── cargo check --features gpu 确认双模块共存编译

Phase 2: GPU 基础设施
  ├── gpu_primitives/radix_sort.rs + radix_sort.wgsl
  ├── gpu_primitives/prefix_sum.rs + prefix_sum.wgsl
  └── 单元测试验证 GPU sort/prefix sum 正确性

Phase 3: Forward 渲染管线
  ├── shaders/helpers.wgsl
  ├── shaders/project_forward.wgsl + render/projection.rs
  ├── shaders/project_visible.wgsl + render/project_visible.rs
  ├── shaders/map_gaussian_to_intersects.wgsl + render/tile_mapping.rs
  ├── shaders/rasterize.wgsl + render/rasterize.rs
  ├── render/sorting.rs (集成 radix sort)
  ├── render/mod.rs (管线编排)
  └── splats.rs (DeviceSplats 定义 + HostSplats 桥接)

Phase 4: Backward + Autodiff
  ├── shaders/rasterize_backwards.wgsl + render_bwd/rasterize_bwd.rs
  ├── shaders/project_backwards.wgsl + render_bwd/project_bwd.rs
  └── render_bwd/autodiff.rs (burn Backward trait 注册)

Phase 5: 训练管线
  ├── adam_scaled.rs
  ├── loss.rs
  ├── topology_bridge.rs
  ├── topology_apply.rs
  ├── trainer.rs (WgpuTrainer)
  └── entry.rs (同步入口)

Phase 6: 接线 + 删旧
  ├── 修改 orchestrator.rs → wgpu_entry
  ├── 修改 config.rs → TrainingBackend::Wgpu
  ├── 修改 lib.rs → WgpuDevice
  ├── 修改 training/mod.rs → mod wgpu
  ├── 删除 src/training/metal/ (全部)
  ├── 删除 src/training/shaders/*.metal
  ├── 删除 src/diff/ (diff_splat, analytical_backward, diff_renderer)
  └── 移除 Cargo.toml 中 candle 依赖

Phase 7: 验证
  ├── cargo build --features gpu
  ├── cargo test (topology 单元测试)
  ├── 小数据集完整训练 → PLY 输出
  └── PSNR parity 对比 (参考旧 Metal 结果)
```

## 十一、Codex CLI 执行规划

每个 Phase 作为一个或多个 Codex 任务下发。Codex 负责代码实现，每步完成后验证编译。

### Phase 1 — 脚手架

```
Codex Task 1: Update Cargo.toml with burn/wgpu deps (keep candle for now).
              Create src/training/wgpu/ with mod.rs and backend.rs.
              Add #[cfg(feature = "gpu")] mod wgpu to src/training/mod.rs.
              Verify: cargo check --features gpu
```

### Phase 2 — GPU Primitives

```
Codex Task 2: Implement GPU radix sort in src/training/wgpu/gpu_primitives/radix_sort.rs
              with WGSL shader src/training/wgpu/shaders/radix_sort.wgsl.
              Reference: brush-sort implementation at ~/Projects/brush/crates/brush-sort/.
              Include unit test.

Codex Task 3: Implement GPU prefix sum in src/training/wgpu/gpu_primitives/prefix_sum.rs
              with WGSL shader src/training/wgpu/shaders/prefix_sum.wgsl.
              Reference: brush-prefix-sum at ~/Projects/brush/crates/brush-prefix-sum/.
              Include unit test.
```

### Phase 3 — Forward Pipeline

```
Codex Task 4: Implement WGSL shaders and Rust render modules for the 5-stage forward pipeline.
              Reference: brush-render at ~/Projects/brush/crates/brush-render/.
              Write DeviceSplats<B> and HostSplats bridge in splats.rs.
```

### Phase 4 — Backward + Autodiff

```
Codex Task 5: Implement backward WGSL shaders and burn autodiff integration.
              Reference: brush-render-bwd at ~/Projects/brush/crates/brush-render-bwd/.
```

### Phase 5 — Training

```
Codex Task 6: Implement AdamScaled optimizer, loss functions, topology bridge/apply,
              WgpuTrainer, and sync entry point.
              Reference: brush-train at ~/Projects/brush/crates/brush-train/.
```

### Phase 6-7 — Wire + Delete + Verify

```
Codex Task 7: Rewire orchestrator/config/lib.rs to wgpu backend.
              Delete Metal code and candle deps.
              Fix compilation errors and run tests.
```

## 十二、风险与缓解

| 风险 | 缓解措施 |
|---|---|
| burn API 不稳定 (git dep) | 锁定 commit hash；等 burn 发稳定版后切换 |
| WGSL shader 调试困难 | 分阶段验证，每个 kernel 写 unit test；用 CPU reference 对比 |
| wgpu 在某些平台性能不如 native Metal | 关注 wgpu 的 Metal backend 优化进展；必要时添加 benchmark |
| topology 代码假设特定 tensor 布局 | topology 只操作 `Vec<f32>` 快照，与 tensor layout 解耦 |
| 训练数值不一致 | 用 parity harness 对比旧 Metal 结果，控制差异在可接受范围 |

## 十三、预期收益

| 维度 | 当前 | 迁移后 |
|---|---|---|
| 平台支持 | Apple Silicon only | Metal + Vulkan + DX12 + WebGPU |
| Shader 语言 | Metal Shading Language | WGSL (跨平台) |
| 自动微分 | 手写 analytical backward | burn autodiff + 自定义 op |
| 运算融合 | 无 | Fusion backend |
| 代码量 (GPU 后端) | ~15 Rust + 7 Metal | ~15 Rust + 9 WGSL |
| 依赖 | candle-core/metal/kernels + objc2-* | burn + wgpu |
