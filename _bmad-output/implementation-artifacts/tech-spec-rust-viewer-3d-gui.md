---
title: 'RustViewer — RustScan 实时 3D 可视化 GUI'
slug: 'rust-viewer-3d-gui'
created: '2026-03-03'
status: 'completed'
stepsCompleted: [1, 2, 3, 4]
tech_stack: ['eframe-0.31', 'egui-0.31', 'wgpu-via-eframe', 'glam-0.25', 'rfd-0.15', 'bytemuck-1', 'serde_json-1.0', 'rustslam-feature-viewer-types']
files_to_modify:
  - 'Cargo.toml (root, new workspace)'
  - 'RustSLAM/Cargo.toml (add viewer-types feature, gate ffmpeg/candle)'
  - 'RustSLAM/src/lib.rs (cfg-gate heavy modules)'
files_to_create:
  - 'RustViewer/Cargo.toml'
  - 'RustViewer/src/main.rs'
  - 'RustViewer/src/app.rs'
  - 'RustViewer/src/renderer/mod.rs'
  - 'RustViewer/src/renderer/scene.rs'
  - 'RustViewer/src/renderer/camera.rs'
  - 'RustViewer/src/renderer/pipelines.rs'
  - 'RustViewer/src/loader/mod.rs'
  - 'RustViewer/src/loader/checkpoint.rs'
  - 'RustViewer/src/loader/gaussian.rs'
  - 'RustViewer/src/loader/mesh.rs'
  - 'RustViewer/src/ui/mod.rs'
  - 'RustViewer/src/ui/panel.rs'
  - 'RustViewer/src/ui/viewport.rs'
code_patterns:
  - 'thiserror for all error types'
  - 'glam::Vec3/Mat4 for 3D math'
  - 'feature-gated deps in RustSLAM'
  - 'arcball camera: (target, distance, yaw, pitch)'
  - 'wgpu point/line/triangle primitives via eframe wgpu callback'
test_patterns:
  - 'unit tests in #[cfg(test)] modules'
  - 'integration tests load from test_data/ samples'
---

# Tech-Spec: RustViewer — RustScan 实时 3D 可视化 GUI

**Created:** 2026-03-03

## Overview

### Problem Statement

RustScan 目前只有 CLI 输出和静态图片导出（`viewer/mod.rs` 生成 PNG），缺乏交互式 3D 可视化界面。用户无法直观地实时查看 SLAM 重建过程中的相机轨迹、稀疏点云、高斯体（3DGS）和提取的 Mesh，只能在 pipeline 结束后查看静态结果文件，调试和分析体验极差。

### Solution

新建独立 Cargo crate `RustViewer/`，使用 **egui + wgpu**（通过 eframe 框架）构建真正的 3D 交互式 GUI 应用。

- **第一阶段（离线回放）**：加载 pipeline 输出文件（`slam_checkpoint.json`、`scene.ply` Gaussian、`mesh.obj/ply`），在 3D 视口中交互式浏览重建结果。
- **第二阶段（实时模式）**：通过 crossbeam-channel 接收 RustSLAM pipeline 实时数据帧，边扫描边可视化。

### Scope

**In Scope（第一阶段）：**
- 新建 `RustViewer/` crate，独立于 RustSLAM
- 离线文件加载：`SlamCheckpoint` JSON（相机轨迹 + 地图点）、`scene.ply`（Gaussian 高斯体，点云表示）、`mesh.obj/ply`（提取的 Mesh）
- 3D 渲染层（wgpu）：相机轨迹折线、稀疏点云、Gaussian 点云、Mesh 线框和实体面
- egui 控制面板：文件选择、图层显示开关、渲染模式切换
- 3D 相机控制：鼠标拖拽旋转、滚轮缩放、右键平移（arcball camera）
- 独立可执行文件：`cargo run` 直接启动 GUI

**Out of Scope（第一阶段不包含）：**
- 实时 channel 通信（第二阶段）
- 修改现有 RustSLAM pipeline 逻辑
- Gaussian Splatting GPU 渲染（只做点云代表，不做真正的 splat 渲染）
- 视频播放/回放
- 网络传输/远程可视化

## Context for Development

### Codebase Patterns

- 项目使用 Rust Edition 2021，所有 crate 遵循 snake_case 文件名、PascalCase 类型名
- 错误处理使用 `thiserror` 派生 `Error` trait，库代码不 panic，返回 `Result`
- 数学库：3D 操作用 `glam`（Vec3、Mat4、Quat），线性代数用 `nalgebra`
- 并发：线程间通信用 `crossbeam-channel`，数据并行用 `rayon`
- 已有 `viewer/mod.rs`：正交投影软件渲染器，生成静态 PNG，不做修改
- 现有数据结构：`SlamCheckpoint`（JSON 序列化）、`Gaussian`（PLY）、mesh OBJ/PLY

### Files to Reference

| File | Purpose | 关键发现 |
| ---- | ------- | -------- |
| `RustSLAM/src/viewer/mod.rs` | 现有 MapViewer，**不修改** | 软件渲染 2D 正交投影，生成静态 PNG；定义了 `Point3D`、`Color=[u8;4]` |
| `RustSLAM/src/pipeline/checkpoint.rs` | 离线数据主来源 | `load_checkpoint(path) -> SlamCheckpoint`（pub）；`SlamCheckpoint` 含 `keyframes: Vec<CheckpointKeyFrame>`（含 `pose: CheckpointPose{translation:[f32;3], quaternion:[f32;4]}`）和 `map_points: Vec<CheckpointMapPoint>` |
| `RustSLAM/src/fusion/scene_io.rs` | Gaussian PLY 加载 | `load_scene_ply(path) -> (Vec<Gaussian>, SceneMetadata)`（pub）；Gaussian 来自 `tiled_renderer.rs` |
| `RustSLAM/src/fusion/tiled_renderer.rs` | `Gaussian` 结构体 | `pub struct Gaussian { position: [f32;3], color: [f32;3], opacity: f32, ... }`（所有字段 pub） |
| `RustSLAM/src/fusion/marching_cubes.rs` | Mesh 数据结构 | `pub struct Mesh { vertices: Vec<MeshVertex>, triangles: Vec<MeshTriangle> }`；`MeshVertex { position: Vec3, normal: Vec3, color: [f32;3] }`；`MeshTriangle { indices: [usize;3] }` |
| `RustSLAM/src/fusion/mesh_io.rs` | Mesh 导出（无加载 API） | 只有 `save_mesh_obj`、`save_mesh_ply`、`export_mesh`，**无 load API**，需在 RustViewer 中自行实现简单 OBJ/PLY 解析 |
| `RustSLAM/src/fusion/mod.rs` | fusion re-exports | `pub use marching_cubes::{Mesh, MeshVertex, MeshTriangle}`；`pub use tiled_renderer::{Gaussian, ...}`；`pub use scene_io::{load_scene_ply, ...}` |
| `RustSLAM/Cargo.toml` | 依赖参考 | `ffmpeg-next = "8.0"`、`candle-core`、`candle-metal` 目前是必选依赖，需改为 optional |

### Technical Decisions

1. **crate 位置**：新建 `RustViewer/` 独立 crate。根目录新建 `Cargo.toml` workspace，members: `["RustMesh", "RustSLAM", "RustViewer"]`，resolver = "2"，共享 `glam = "0.25"` 等依赖版本。

2. **GUI 框架**：`eframe 0.31`（内置 wgpu 后端）。**注意**：不单独引入 `wgpu` 依赖，由 eframe 管理版本，避免类型不兼容。3D 渲染通过 `egui::PaintCallback` + `eframe::wgpu` 的 `CallbackFn` 注入到 egui 渲染管线。

3. **RustSLAM 依赖方式**：在 `RustSLAM/Cargo.toml` 新增 `viewer-types` feature，将 `ffmpeg-next`、`candle-core`、`candle-metal` 改为 optional，并在对应模块加 `#[cfg(feature = "...")]`。`RustViewer` 依赖：`rustslam = { path = "../RustSLAM", default-features = false, features = ["viewer-types"] }`。

4. **arcball 相机**：存储 `{ target: Vec3, distance: f32, yaw: f32, pitch: f32 }`。**左键拖拽** → 更新 yaw/pitch；**右键拖拽** → 平移 target；**滚轮** → 缩放 distance。view matrix = `Mat4::look_at_rh(eye, target, up)`，eye 由 spherical coords 计算。

5. **Mesh 加载**：`mesh_io.rs` 无 load API，需在 `RustViewer/src/loader/mesh.rs` 实现简单 OBJ 解析（逐行解析 `v`/`f` 指令）和 PLY 解析（读取 ASCII PLY header + data）。

6. **Gaussian 点云渲染**：每个 `Gaussian.position` 作为顶点，`color * opacity` 混合为顶点色，wgpu `point-list` topology，point_size 通过 WGSL `@builtin(position)` 的片元着色器模拟（画小方块）。

7. **空状态 UX**：初始界面居中显示「📂 打开 SLAM 结果目录」按钮（rfd::FileDialog），无数据时不显示黑色空视口。图层面板有颜色图例：蓝=地图点、金=相机轨迹、橙=Gaussian、灰=Mesh。

## Implementation Plan

### Tasks

**阶段 0：基础设施准备**

- [x] Task 1: 创建根目录 workspace Cargo.toml
  - File: `Cargo.toml`（根目录，新建）
  - Action: 创建 workspace 配置，包含 `RustMesh`、`RustSLAM`、`RustViewer` 三个成员，`resolver = "2"`
  - Content:
    ```toml
    [workspace]
    members = ["RustMesh", "RustSLAM", "RustViewer"]
    resolver = "2"

    [workspace.dependencies]
    glam = "0.25"
    serde = { version = "1.0", features = ["derive"] }
    thiserror = "1.0"
    ```

- [x] Task 2: 在 RustSLAM 中添加 `viewer-types` feature，将重型依赖改为 optional
  - File: `RustSLAM/Cargo.toml`
  - Action:
    1. 将 `ffmpeg-next`、`lru`、`sysinfo` 改为 `optional = true`，归入新 feature `slam-pipeline`
    2. 将 `candle-core`、`candle-metal` 改为 `optional = true`，归入新 feature `gpu`
    3. 新增 `viewer-types = []` feature（空 feature，标记只编译数据结构）
    4. 更新 `[features]` 块：`default = ["slam-pipeline", "gpu"]`，确保现有 CLI 构建不受影响
  - Notes: `apex-solver`、`rayon`、`kiddo`、`nalgebra`、`glam` 保持必选（数据结构需要）

- [ ] Task 3: 在 RustSLAM src/ 中添加 cfg 门控，隔离 candle 和 ffmpeg 依赖
  - Files: `RustSLAM/src/lib.rs`、`RustSLAM/src/io/video_decoder.rs`、`RustSLAM/src/fusion/mod.rs`
  - Action:
    1. **`RustSLAM/src/lib.rs`**：对 `pub mod io` 加 `#[cfg(feature = "slam-pipeline")]`；对 `pub mod pipeline::realtime` 或 `pub use pipeline::realtime` 加同样门控
    2. **`RustSLAM/src/io/video_decoder.rs`**：顶部加 `#![cfg(feature = "slam-pipeline")]`
    3. **`RustSLAM/src/fusion/mod.rs`**（关键）：将使用 candle 的子模块 re-export 包入 `#[cfg(feature = "gpu")]`：
       ```rust
       #[cfg(feature = "gpu")]
       pub use diff_renderer::{...};
       #[cfg(feature = "gpu")]
       pub use diff_splat::{...};
       #[cfg(feature = "gpu")]
       pub use complete_trainer::{...};
       #[cfg(feature = "gpu")]
       pub use autodiff::{...};
       #[cfg(feature = "gpu")]
       pub use gpu_trainer::{...};
       #[cfg(feature = "gpu")]
       pub use autodiff_trainer::{...};
       #[cfg(feature = "gpu")]
       pub use training_checkpoint::{...};
       #[cfg(feature = "gpu")]
       pub use gaussian_init::{...};  // uses candle_core::Device
       // 以下保持无 gate（不使用 candle）：
       pub use tiled_renderer::{Gaussian, ...};
       pub use scene_io::{load_scene_ply, ...};
       pub use marching_cubes::{Mesh, MeshVertex, MeshTriangle, ...};
       pub use mesh_extractor::{...};
       pub use mesh_io::{...};
       pub use tsdf_volume::{...};
       pub use slam_integrator::{...};  // 验证：slam_integrator 是否用 candle？需检查
       ```
    4. 同理对 `fusion/` 中每个 candle-using 文件顶部加 `#![cfg(feature = "gpu")]` 模块级守卫：`diff_splat.rs`、`complete_trainer.rs`、`autodiff.rs`、`diff_renderer.rs`、`trainer.rs`、`training_checkpoint.rs`、`gpu_trainer.rs`、`autodiff_trainer.rs`、`gaussian_init.rs`
    5. 对 `pipeline/realtime.rs` 加 `#![cfg(feature = "slam-pipeline")]`（它 `use candle_core::Device`）
  - Notes:
    - **验证方法**：完成后运行 `cd RustSLAM && cargo check --no-default-features --features viewer-types` 确认无 candle/ffmpeg 依赖
    - `tiled_renderer.rs`、`scene_io.rs`、`marching_cubes.rs`、`checkpoint.rs` 本身不 use candle，可保持无 gate
    - `slam_integrator.rs` 需要在实现时检查其 imports 是否涉及 candle，若有则同样加 gate

**阶段 1：RustViewer crate 骨架**

- [ ] Task 4: 创建 RustViewer crate 目录结构和 Cargo.toml
  - File: `RustViewer/Cargo.toml`（新建）
  - Action: 创建 binary crate 配置
    ```toml
    [package]
    name = "rust-viewer"
    version = "0.1.0"
    edition = "2021"
    description = "Interactive 3D viewer for RustScan SLAM results"

    [[bin]]
    name = "rust-viewer"
    path = "src/main.rs"

    [dependencies]
    eframe = { version = "0.31", default-features = false, features = ["wgpu", "default_fonts", "accesskit"] }
    egui = "0.31"
    glam = "0.25"
    bytemuck = { version = "1", features = ["derive"] }
    rfd = "0.14"
    thiserror = "1.0"
    serde = { version = "1.0", features = ["derive"] }
    serde_json = "1.0"
    rustslam = { path = "../RustSLAM", default-features = false, features = ["viewer-types"] }
    ```
  - Notes:
    - **不**单独引入 `wgpu` 直接依赖；在 `pipelines.rs` 等文件中通过 `eframe::wgpu::` 命名空间访问所有 wgpu 类型（eframe 的 wgpu feature 会 re-export wgpu）
    - `rfd = "0.14"` 使用同步 `FileDialog::pick_file()`；在 egui update 循环中通过 `std::thread::spawn` 异步调用，避免冻结 GUI 主线程。具体模式：spawn 线程 → 线程内调用 rfd → 结果通过 `std::sync::mpsc::channel` 发回 App
    - eframe 0.31 窗口大小通过 `NativeOptions { viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]), ..Default::default() }` 设置

- [ ] Task 5: 创建 main.rs 入口和 app.rs 应用骨架
  - Files: `RustViewer/src/main.rs`（新建）、`RustViewer/src/app.rs`（新建）
  - Action:
    1. `main.rs`: 调用 `eframe::run_native`，配置 `NativeOptions`（窗口尺寸 1280×800，标题 "RustViewer"，wgpu backend），启动 `ViewerApp`
    2. `app.rs`: 定义 `ViewerApp` struct，实现 `eframe::App` trait 的 `update()` 方法；持有字段：`scene: Scene`、`camera: ArcballCamera`、`ui_state: UiState`、`renderer: Option<SceneRenderer>`
    3. `update()` 中：左侧 `SidePanel` 渲染控制 UI，右侧 `CentralPanel` 渲染 3D 视口
  - Notes: `SceneRenderer` 用 `Option` 包裹，因为 wgpu 资源在首帧才能初始化

**阶段 2：数据模型与加载器**

- [ ] Task 6: 定义 scene 数据模型
  - File: `RustViewer/src/renderer/scene.rs`（新建）
  - Action: 定义以下结构体（纯数据，无 GPU 资源）：
    ```rust
    pub struct Scene {
        pub trajectory: Vec<[f32; 3]>,       // 相机位置序列
        pub camera_orientations: Vec<[f32; 4]>, // 对应四元数
        pub map_points: Vec<[f32; 3]>,        // 稀疏地图点位置
        pub map_point_colors: Vec<[f32; 3]>,  // 地图点颜色（深度着色）
        pub gaussians: Vec<GaussianPoint>,    // Gaussian 点云
        pub mesh_vertices: Vec<MeshGpuVertex>, // Mesh 顶点
        pub mesh_indices: Vec<u32>,           // Mesh 三角形索引
        pub layers: LayerVisibility,          // 图层开关
        pub bounds: SceneBounds,              // 场景包围盒（用于自动对焦）
    }
    pub struct GaussianPoint { pub position: [f32;3], pub color: [f32;3] }
    pub struct MeshGpuVertex { pub position: [f32;3], pub normal: [f32;3], pub color: [f32;3] }
    pub struct LayerVisibility {
        pub trajectory: bool,
        pub map_points: bool,
        pub gaussians: bool,
        pub mesh_wireframe: bool,
        pub mesh_solid: bool,
    }
    pub struct SceneBounds { pub min: [f32;3], pub max: [f32;3] }
    ```
  - Notes: `MeshGpuVertex` 需 `#[repr(C)]` + `bytemuck::Pod + bytemuck::Zeroable`

- [ ] Task 7: 实现 checkpoint 加载器
  - File: `RustViewer/src/loader/checkpoint.rs`（新建）
  - Action:
    1. 调用 `rustslam::pipeline::checkpoint::load_checkpoint(path)` 读取 JSON
    2. 从 `SlamCheckpoint.keyframes` 提取 `pose.translation` 填入 `Scene::trajectory`
    3. 从 `SlamCheckpoint.map_points` 提取位置填入 `Scene::map_points`，按 Y 深度计算绿→红渐变色填入 `map_point_colors`
    4. 计算 `SceneBounds`（trajectory + map_points 的 min/max）
    5. 返回 `Result<Scene, LoadError>`

- [ ] Task 8: 实现 Gaussian PLY 加载器
  - File: `RustViewer/src/loader/gaussian.rs`（新建）
  - Action:
    1. 调用 `rustslam::fusion::load_scene_ply(path)` 读取 `Vec<Gaussian>`
    2. 将每个 `Gaussian` 的 `position` 和 `color * opacity.clamp(0,1)` 转为 `GaussianPoint`
    3. 更新 `Scene::gaussians` 和 `SceneBounds`
    4. 返回 `Result<Vec<GaussianPoint>, LoadError>`

- [ ] Task 9: 实现 Mesh OBJ/PLY 加载器
  - File: `RustViewer/src/loader/mesh.rs`（新建）
  - Action: 实现两个函数，**严格匹配 RustSLAM 实际输出格式**：

    **`load_obj(path) -> Result<(Vec<MeshGpuVertex>, Vec<u32>), LoadError>`**
    - RustSLAM 的 `save_mesh_obj` 输出格式：
      - 顶点行：`v x y z r g b`（6 个浮点数，含嵌入式 RGB 颜色，范围 0-1）
      - 法线行：`vn nx ny nz`（独立法线列表，与顶点一一对应）
      - 面行：`f A//A B//B C//C`（vertex//normal 格式，1-based 索引）
      - 注释行以 `#` 开头，跳过
    - 解析策略：两遍扫描。第一遍收集所有 `v` 和 `vn` 行（按顺序，索引对应）；第二遍解析 `f` 行，分割 `//` 取第一个整数（vertex index，转 0-based）
    - 顶点颜色：从 `v` 行的第 4-6 个浮点数读取，已在 0-1 范围内（RustSLAM 写入前已 clamp+round/255）
    - 法线：从 `vn` 行，按面索引的第二个整数（normal index，转 0-based）取对应法线

    **`load_ply(path) -> Result<(Vec<MeshGpuVertex>, Vec<u32>), LoadError>`**
    - RustSLAM 的 `save_mesh_ply` 输出格式（ASCII PLY）：
      - Header：`element vertex N`、`property float x/y/z/nx/ny/nz`、`property uchar red/green/blue`、`element face N`、`property list uchar int vertex_indices`、`end_header`
      - 顶点数据行：`x y z nx ny nz r g b`（6 个 float + 3 个 u8 整数）
      - 面数据行：`3 i j k`（先是顶点数 `3`，再是 0-based 整数索引）
    - 解析策略：逐行读取 header 获取 vertex_count 和 face_count；然后按序读取 vertex_count 行（split 空格，取 9 个值），再读 face_count 行（split 空格，跳过第一个值 `3`，取后 3 个）
    - 颜色：u8 整数 → [0.0, 1.0] = value as f32 / 255.0

  - Notes: 只支持 ASCII PLY（RustSLAM 只输出 ASCII）；不支持 Binary PLY 和 OBJ mtl 材质文件

**阶段 3：3D 相机控制**

- [ ] Task 10: 实现 ArcballCamera
  - File: `RustViewer/src/renderer/camera.rs`（新建）
  - Action:
    ```rust
    pub struct ArcballCamera {
        pub target: glam::Vec3,
        pub distance: f32,
        pub yaw: f32,   // radians
        pub pitch: f32, // radians, clamped [-PI/2+eps, PI/2-eps]
        pub fov_y: f32, // radians, default PI/4
    }
    impl ArcballCamera {
        pub fn eye(&self) -> Vec3 { ... }  // spherical to cartesian
        pub fn view_matrix(&self) -> Mat4 { Mat4::look_at_rh(eye, target, Vec3::Y) }
        pub fn proj_matrix(&self, aspect: f32) -> Mat4 { Mat4::perspective_rh(fov_y, aspect, 0.01, 1000.0) }
        pub fn orbit(&mut self, delta_x: f32, delta_y: f32) { ... } // 左键拖拽
        pub fn pan(&mut self, delta_x: f32, delta_y: f32) { ... }   // 右键拖拽
        pub fn zoom(&mut self, delta: f32) { ... }                   // 滚轮
        pub fn fit_scene(&mut self, bounds: &SceneBounds) { ... }    // 自动对焦场景
    }
    ```
  - Notes: `orbit` 灵敏度系数 0.005 rad/px；`zoom` 乘以 1.1 或 0.9；`pan` 按 distance 比例缩放平移量

**阶段 4：wgpu 渲染管线**

- [ ] Task 11: 实现 GPU buffer 管理和着色器
  - File: `RustViewer/src/renderer/pipelines.rs`（新建）
  - Action: 使用 `eframe::wgpu::` 命名空间（**不**需要单独引入 `wgpu` crate）实现三条 render pipeline：

    **所有 pipeline 共用：**
    - Uniform buffer（64 bytes）：传递 `view_proj: [[f32;4];4]`（Mat4），通过 `eframe::wgpu::Queue::write_buffer` 更新
    - BindGroupLayout：`binding=0, visibility=VERTEX, ty=Uniform, has_dynamic_offset=false, min_binding_size=NonZeroU64(64)`
    - **注意**：uniform buffer 无需 256-byte padding，因为是非 dynamic offset 用法（static binding）
    - Depth texture 和 DepthStencilState：**必须**在 `SceneRenderer` 中创建深度纹理（`format: Depth32Float`），并在每条 pipeline 的 `RenderPipelineDescriptor` 中设置 `depth_stencil: Some(DepthStencilState { format: Depth32Float, depth_write_enabled: true, depth_compare: Less, ... })`；render pass 中必须附加 depth attachment

    **1. 点云/Gaussian pipeline**（`PointList` topology）：
    - 顶点格式：`[f32;3]` position（offset 0）+ `[f32;3]` color（offset 12）= stride 24 bytes，`#[repr(C)] + bytemuck::Pod`
    - WGSL 顶点着色器：`@builtin(position)` 计算 clip position；**不使用** `@builtin(point_size)`（Metal/wgpu 不支持通过 WGSL 内置变量设置点大小）
    - 替代方案：将每个点展开为 2 个三角形（4 个顶点，共 6 个索引）的小正方形，在 CPU 端扩展顶点缓冲，点大小固定 4px（通过 `ndc_size = 4.0 / viewport_size` 计算偏移）。使用 `TriangleList` topology
    - 片元着色器：输出顶点色

    **2. 轨迹折线 pipeline**（`LineList` topology）：
    - 顶点格式同上（position + color），轨迹颜色 `[0.2, 0.6, 1.0]` 写入 color 字段
    - 每对相邻轨迹点（i, i+1）构成一条线段：CPU 端生成 line list 顶点
    - 注意：空轨迹（0 或 1 个点）时 vertex buffer 为空，`draw(0..0)` 正常，不崩溃

    **3. Mesh pipeline**（`TriangleList` topology）：
    - 顶点格式：`[f32;3]` position + `[f32;3]` normal + `[f32;3]` color = stride 36 bytes
    - Solid 模式：`polygon_mode: Fill`，片元着色器：`Lambert = max(dot(normalize(normal), vec3(0.5, 1.0, 0.5)), 0.1)`，输出 `vertex_color * Lambert`
    - Wireframe 模式：需要 `eframe::wgpu::Features::POLYGON_MODE_LINE`；在创建 wgpu device 时通过 `eframe::egui_wgpu::WgpuConfiguration` 的 `device_descriptor` 字段请求该 feature：
      ```rust
      wgpu_options.device_descriptor = Arc::new(|_adapter| wgpu::DeviceDescriptor {
          required_features: wgpu::Features::POLYGON_MODE_LINE,
          ..Default::default()
      });
      ```
      若 Metal adapter 不支持（实际上 Metal 支持），则 wireframe pipeline 创建失败时回退到 Fill 模式并在 UI 显示提示

  - Notes:
    - 点云通过 CPU 扩展为小方块三角形，避免依赖 `point_size`（跨平台兼容）
    - `POLYGON_MODE_LINE` 在 Metal 后端受支持，在 WebGL 后端不受支持

- [ ] Task 12: 实现 SceneRenderer 和 egui PaintCallback 集成
  - File: `RustViewer/src/renderer/mod.rs`（新建）
  - Action:
    1. 定义 `SceneRenderer`，持有：三条 pipeline（点云/轨迹/mesh）、uniform buffer、各图层 vertex buffer、**depth texture**（`Texture` + `TextureView`）
    2. 实现 `SceneRenderer::new(device: &eframe::wgpu::Device, queue: &eframe::wgpu::Queue, surface_format: eframe::wgpu::TextureFormat) -> Self`
       - 在此处创建 depth texture：`device.create_texture(&TextureDescriptor { format: Depth32Float, usage: RENDER_ATTACHMENT, ... })`
       - 创建三条 pipeline（含 DepthStencilState）
       - 创建 uniform buffer（64 bytes，`UNIFORM | COPY_DST`）
    3. 实现 `SceneRenderer::update_buffers(device, queue, scene)` — 每帧将 Scene 数据写入 GPU vertex buffer（点云、轨迹、mesh 各自独立 buffer）
    4. 实现 `SceneRenderer::render(render_pass: &mut RenderPass, camera: &ArcballCamera, viewport_size: [f32;2], scene: &Scene)`
       - 首先更新 uniform buffer（view_proj matrix）
       - 按图层可见性 + Scene 有效数据发出 draw call
    5. **eframe wgpu callback 集成**（在 `app.rs` 的 `CentralPanel` 中）：
       ```rust
       // SceneRenderer 存储在 eframe 的 paint_callback_resources（类型 TypeMap）
       // 初始化：在首帧的 prepare callback 中检查 resources 是否有 SceneRenderer，无则创建
       let cb = egui_wgpu::Callback::new_paint_callback(
           viewport_rect,
           ViewerCallback { scene: self.scene.clone(), camera: self.camera.clone() },
       );
       ui.painter().add(cb);
       ```
       - `ViewerCallback` 实现 `egui_wgpu::CallbackTrait`：
         - `prepare()` 方法：获取/初始化 SceneRenderer，调用 `update_buffers`，更新 uniform buffer
         - `paint()` 方法：调用 `renderer.render(render_pass, ...)`，**必须**传入 depth attachment
    6. **Depth texture resize**：当 viewport 大小改变时（`viewport_rect.size() != last_size`），重建 depth texture 以匹配新尺寸。在 `prepare()` 中检测大小变化。
  - Notes:
    - `egui_wgpu::CallbackTrait` 是 eframe 0.31 中的接口（替代旧版 `CallbackFn`），参考 eframe 仓库 `examples/custom3d_wgpu/` 示例
    - `paint()` callback 接收的 `RenderPass` 已绑定 color attachment，depth attachment 需在 `prepare()` 中通过 `egui_wgpu::CallbackResources` 传递 TextureView，在 `paint()` 中手动设置（或提前配置好的 depth-aware RenderPass）
    - 实际上 eframe 0.31 的 `paint()` 不直接支持附加 depth attachment 到传入的 RenderPass，**解决方案**：在 `prepare()` 中手动创建一个新的 `CommandEncoder` + `RenderPass`（含 depth），提交后让 eframe 叠加（或使用 `egui_wgpu::Painter` 扩展点）。**备选方案**：放弃 depth buffer，改用画家算法（按深度排序绘制顺序）——对于点云/轨迹/mesh 分层场景可接受
    - **推荐实现备选**：先不用深度缓冲（备选方案），顺序：先画 mesh（实体），再画轨迹，再画点云，再画 Gaussian。这样避免 eframe depth attachment 的集成复杂性，第一阶段 MVP 可接受

**阶段 5：egui UI 面板**

- [ ] Task 13: 实现左侧控制面板
  - File: `RustViewer/src/ui/panel.rs`（新建）
  - Action: 实现 `fn draw_side_panel(ui: &mut egui::Ui, state: &mut UiState, scene: &mut Scene, camera: &mut ArcballCamera)` 包含：
    1. **文件加载区**：三个按钮「📂 加载 Checkpoint」、「✨ 加载 Gaussians」、「🔷 加载 Mesh」，点击调用 `rfd::FileDialog::new().pick_file()`，根据文件扩展名分发到对应 loader
    2. **图层控制区**（`egui::CollapsingHeader`）：五个 checkbox（相机轨迹🔵、地图点🔵、Gaussian✨、Mesh 线框🔲、Mesh 实体🔷），带颜色图例小方块（`ui.colored_label`）
    3. **场景信息区**：显示当前加载的统计信息（keyframe 数、地图点数、Gaussian 数、顶点数）
    4. **相机控制区**：「🎯 自动对焦」按钮调用 `camera.fit_scene(&scene.bounds)`

- [ ] Task 14: 实现空状态引导 UI 和 viewport 覆盖层
  - File: `RustViewer/src/ui/viewport.rs`（新建）
  - Action:
    1. `fn draw_empty_state(ui: &mut egui::Ui)` — 场景无数据时居中显示大号引导按钮和说明文字
    2. `fn draw_viewport_overlay(ui: &mut egui::Ui, camera: &ArcballCamera)` — 右下角覆盖显示当前相机参数（可选，调试用）
    3. 在 `app.rs` 的 `CentralPanel` 中：若 `scene` 无数据，显示 empty_state；否则触发 PaintCallback 渲染 + 处理鼠标输入事件更新相机

- [ ] Task 15: 实现鼠标输入到相机控制的映射
  - File: `RustViewer/src/app.rs`（修改）
  - Action: 在 `CentralPanel` 的 `response` 上读取 egui 输入：
    ```rust
    if response.dragged_by(egui::PointerButton::Primary) {
        let delta = response.drag_delta();
        camera.orbit(delta.x, delta.y);
    }
    if response.dragged_by(egui::PointerButton::Secondary) {
        let delta = response.drag_delta();
        camera.pan(delta.x, delta.y);
    }
    if let Some(scroll) = ui.input(|i| i.smooth_scroll_delta.y) {
        camera.zoom(scroll);
    }
    ```
  - Notes: 在 `response.hovered()` 时才处理 scroll，避免与面板 scroll 冲突

**阶段 6：测试**

- [ ] Task 16: 编写数据加载单元测试
  - File: `RustViewer/src/loader/checkpoint.rs`、`RustViewer/src/loader/mesh.rs`（内嵌 `#[cfg(test)]` 模块）
  - Action:
    1. `test_load_checkpoint_empty` — 构造最小合法 JSON（`{"version":1,"frame_index":0,"keyframes":[],"map_points":[]}`），通过 `tempfile::NamedTempFile` 写入磁盘后调用 `load_checkpoint`，验证返回 Ok，且 trajectory 为空、bounds 为默认值
    2. `test_load_obj_triangle` — 内联 OBJ 字符串（匹配 RustSLAM 实际格式）：
       ```
       # rustslam mesh
       v 0.0 0.0 0.0 0.8 0.2 0.2
       v 1.0 0.0 0.0 0.2 0.8 0.2
       v 0.0 1.0 0.0 0.2 0.2 0.8
       vn 0.0 0.0 1.0
       vn 0.0 0.0 1.0
       vn 0.0 0.0 1.0
       f 1//1 2//2 3//3
       ```
       验证：3 顶点，1 个三角形（indices=[0,1,2]），第一顶点颜色 ≈ [0.8, 0.2, 0.2]
    3. `test_load_ply_ascii` — 内联 ASCII PLY（匹配 RustSLAM 实际格式）：
       ```
       ply
       format ascii 1.0
       comment rustslam_mesh
       element vertex 3
       property float x
       property float y
       property float z
       property float nx
       property float ny
       property float nz
       property uchar red
       property uchar green
       property uchar blue
       element face 1
       property list uchar int vertex_indices
       end_header
       0.0 0.0 0.0 0.0 0.0 1.0 204 51 51
       1.0 0.0 0.0 0.0 0.0 1.0 51 204 51
       0.0 1.0 0.0 0.0 0.0 1.0 51 51 204
       3 0 1 2
       ```
       验证：3 顶点，1 个三角形，颜色正确归一化（204/255 ≈ 0.8）
  - Notes: 测试不依赖 GPU，纯数据解析；需在 `[dev-dependencies]` 中添加 `tempfile = "3.3"`

- [ ] Task 17: 编写相机矩阵单元测试
  - File: `RustViewer/src/renderer/camera.rs`（内嵌 `#[cfg(test)]` 模块）
  - Action:
    1. `test_eye_position` — `yaw=0, pitch=0, distance=5` 时 eye 应在 `[0,0,5]`（相对 target）
    2. `test_pitch_clamp` — pitch 超过 PI/2 时被 clamp，不出现相机翻转
    3. `test_zoom_clamp` — distance 不能 <= 0.1（防止穿入场景）

### Acceptance Criteria

- [ ] AC 1: Given 用户启动 RustViewer，when 没有加载任何文件，then 窗口居中显示「📂 打开 SLAM 结果目录」引导界面，不显示黑色空视口

- [ ] AC 2: Given 用户点击「加载 Checkpoint」并选择有效的 `slam_checkpoint.json`，when 文件加载成功，then 3D 视口显示蓝色相机轨迹折线和彩色稀疏点云，左侧面板显示正确的 keyframe 数和地图点数

- [ ] AC 3: Given 用户点击「加载 Gaussians」并选择有效的 `scene.ply`，when 加载成功，then 视口显示橙色 Gaussian 点云，点的位置与相机轨迹空间坐标一致

- [ ] AC 4: Given 用户点击「加载 Mesh」并选择 `.obj` 或 `.ply` 文件，when 加载成功，then 视口显示灰色 Mesh（默认线框模式），可切换为实体面模式

- [ ] AC 5: Given 3D 视口中有数据，when 用户按住左键并拖拽，then 场景绕目标点旋转（arcball），松开鼠标后旋转停止

- [ ] AC 6: Given 3D 视口中有数据，when 用户滚动鼠标滚轮，then 相机距目标点的距离增大或减小（zoom），距离最小值不低于 0.1

- [ ] AC 7: Given 3D 视口中有数据，when 用户按住右键并拖拽，then 场景目标点随鼠标平移（pan）

- [ ] AC 8: Given 图层控制面板中的「相机轨迹」checkbox，when 用户取消勾选，then 视口中轨迹折线立即消失，重新勾选后恢复显示

- [ ] AC 9: Given 用户点击「🎯 自动对焦」按钮，when 场景有数据，then 相机自动调整位置和缩放以显示整个场景的包围盒

- [ ] AC 10: Given 用户选择了一个格式错误的文件，when 加载失败，then 左侧面板显示红色错误提示信息，不 panic，现有场景数据不丢失

- [ ] AC 11: Given 在 macOS 上构建，when 执行 `cd RustViewer && cargo build --release`，then 编译成功，不拉取 ffmpeg-next 或 candle 依赖

## Additional Context

### Dependencies

**新增（`RustViewer/Cargo.toml`）：**
- `eframe = "0.31"` — egui + wgpu 后端（**不单独引入 wgpu**，避免版本冲突）
- `egui = "0.31"` — 即时模式 GUI
- `glam = "0.25"` — 与 RustSLAM 共享版本（workspace 统一）
- `bytemuck = { version = "1", features = ["derive"] }` — GPU buffer 数据转换
- `rfd = "0.15"` — native 文件选择对话框
- `serde = { version = "1.0", features = ["derive"] }` — OBJ/PLY 解析辅助
- `serde_json = "1.0"` — checkpoint JSON 解析（也可通过 rustslam re-export）
- `thiserror = "1.0"` — 错误类型
- `rustslam = { path = "../RustSLAM", default-features = false, features = ["viewer-types"] }`

**修改（`RustSLAM/Cargo.toml`）：**
- `ffmpeg-next` → `optional = true`，归入 `slam-pipeline` feature
- `candle-core`、`candle-metal` → `optional = true`，归入 `gpu` feature
- 新增 `viewer-types = []` feature（只编译数据结构模块，无重依赖）

### Testing Strategy

- **单元测试（`#[cfg(test)]` 内嵌模块）：**
  - `camera.rs`：eye 位置计算、pitch clamp、zoom clamp（Task 17）
  - `loader/checkpoint.rs`：空 checkpoint JSON 不 panic（Task 16）
  - `loader/mesh.rs`：最小 OBJ 解析、最小 ASCII PLY 解析（Task 16）

- **手动集成测试：**
  - 使用 `test_data/` 目录下样本或 pipeline 运行生成的输出文件验证全流程加载渲染
  - 验证三种文件格式（JSON checkpoint、scene.ply、mesh.obj/ply）均可正常加载显示
  - 验证 `cargo build -p rust-viewer --release` 在不启用 slam-pipeline/gpu feature 时编译通过

### Notes

- **关于 eframe wgpu callback API**（高风险）：eframe 0.31 使用 `egui_wgpu::CallbackTrait` trait（非旧版 `CallbackFn`），实现前**必须**参考 eframe 官方示例 `examples/custom3d_wgpu/src/main.rs`。`prepare()` 返回 `Vec<CommandBuffer>`，`paint()` 接收已配置 color attachment 的 `&RenderPass`。Depth buffer 集成见 Task 12 Notes 中的备选方案（推荐画家算法避免复杂度）。
- **关于 wgpu 命名空间**：所有 wgpu 类型通过 `eframe::wgpu::` 访问，无需单独 `wgpu` crate 依赖。
- **关于四元数约定**（F17 修复）：`CheckpointPose.quaternion: [f32;4]` 为 `[w, x, y, z]`，glam 转换：`Quat::from_xyzw(q[1], q[2], q[3], q[0])`。轨迹显示只用 translation，不受影响。
- **关于 `rfd` 文件对话框**：使用 `rfd::FileDialog::pick_file()` 同步 API，在 `std::thread::spawn` 中调用，通过 `std::sync::mpsc::channel` 返回结果给 App，避免冻结 GUI。
- **已知限制**：只支持 ASCII PLY；OBJ 不支持 `.mtl`；点云用 CPU 扩展小方块而非 GPU point_size（跨平台兼容）。
- **第二阶段预留**：`Scene` 字段全部用 `Vec`；`ViewerApp` 预留 `live_rx: Option<crossbeam_channel::Receiver<SceneUpdate>>`。
- **`test_data/` 说明**：Task 16 单元测试用内联字符串（不依赖 test_data/）；手动集成测试可用 `test_data/expected/sofa_sample_01/mesh.obj` 验证 Mesh 加载；checkpoint/gaussian 文件需先运行一次 pipeline 生成。
- **macOS Metal**：eframe 自动选择 Metal 后端，`POLYGON_MODE_LINE` 在 Metal 受支持，需在 device creation 时显式请求（见 Task 11）。
