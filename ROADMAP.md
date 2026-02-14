# RustScan 项目路线图

> 最后更新: 2026-02-14 (完成 3DGS → Mesh 抽取)

## 项目概述

RustScan 是一个纯 Rust 实现的 3D 扫描重建技术栈，涵盖从相机输入到网格处理的完整流程。

```
Pipeline: 相机输入 → RustSLAM → 3DGS 融合 → 网格抽取 → RustMesh 后处理 → 导出
```

---

## 一、项目架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      RustScan 全景                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ 相机输入 │ →  │ RustSLAM│ →  │ 3DGS    │ →  │ RustMesh│  │
│  │         │    │ (SLAM)  │    │ (重建)   │    │ (后处理)│  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│       │              │              │              │            │
│       ▼              ▼              ▼              ▼            │
│   图像/深度      位姿估计       实时重建        导出         │
│                  + 轨迹         + 渲染        OBJ/STL       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    RustGUI (计划中)                       │   │
│  │              实时可视化 + GUI 界面                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、各模块进度

### 2.1 RustSLAM (视觉 SLAM + 3DGS)

**进度: ~85%** ✅ 核心完备 + Mesh 抽取

| 功能 | 状态 | 说明 |
|------|------|------|
| **基础 SLAM** |
| SE3 Pose | ✅ | 完整的李群/李代数 |
| ORB 特征 | ✅ | 特征提取 |
| Harris/FAST | ✅ | 角点检测 |
| 特征匹配 | ✅ | BFMatcher, KNN, Lowe |
| 视觉里程计 | ✅ | Monocular/Stereo/RGB-D |
| BA 优化 | ✅ | Gauss-Newton |
| 回环检测 | ✅ | BoW + Database |
| 重定位 | ✅ | 丢失恢复 |
| **3D Gaussian** |
| 高斯结构 | ✅ | Gaussian3D |
| 渲染器 | ✅ | Tiled Rasterization |
| 深度排序 | ✅ | Depth Sorting |
| Alpha 混合 | ✅ | Alpha Blending |
| 高斯追踪 | ✅ | ICP |
| 增量建图 | ✅ | Incremental Mapping |
| Densification | ✅ | 高斯分裂 |
| Pruning | ✅ | 透明度裁剪 |
| 可微渲染 | ✅ | Candle + Metal |
| 训练管道 | ✅ | Trainer + Adam |
| SLAM 集成 | ✅ | Sparse + Dense |
| **Mesh 抽取** |
| TSDF Volume | ✅ | 纯 Rust 实现 |
| Marching Cubes | ✅ | 256 案例查找表 |
| Mesh Extractor | ✅ | 后处理 (聚类过滤) |
| **待完成** |
| IMU 集成 | ⏳ | - |
| 多地图 SLAM | ⏳ | - |
| 语义建图 | ⏳ | - |
| 离线 3DGS 优化 | ⏳ | - |

**测试:** 77 个测试通过

---

### 2.2 RustMesh (网格处理)

**进度: ~50-60%** ⚠️ 基础扎实，需完善

#### 已完成

| 功能 | 状态 |
|------|------|
| **数据结构** |
| Handle 系统 | ✅ |
| Half-edge | ✅ |
| SoA 布局 | ✅ (独有) |
| ArrayKernel | ✅ |
| PolyConnectivity | ✅ |
| TriConnectivity | ✅ |
| Smart Handles | ✅ (新增) |
| **IO 格式** |
| OFF 读写 | ✅ |
| OBJ + MTL | ✅ |
| PLY 读写 | ✅ |
| STL (ASCII + Binary) | ✅ |
| OM 原生格式 | ⚠️ 基础 |
| **循环器** |
| Vertex-* | ✅ |
| Face-* | ✅ |
| EdgeFace | ✅ (新增) |
| **算法** |
| Decimation | ⚠️ 基础 |
| Smoother | ⚠️ 基础 |
| Subdivision | ⚠️ Loop/CC/√3 |
| Hole Filling | ✅ |
| Mesh Repair | ✅ |
| Dualizer | ✅ |
| VDPM | ⚠️ 基础 |

#### 待完成

| 优先级 | 功能 | 说明 |
|--------|------|------|
| **P0** |
| 属性系统集成 | AttribKernel 与 SoAKernel 合并 |
| 3DGS → Mesh | 从 Splatting 抽取网格 |
| **P1** |
| MeshChecker | 网格验证 |
| 高级 Decimation | Hausdorff, NormalDeviation |
| Modified Butterfly | 插值细分 |
| **P2** |
| 自适应细分 | Composite/RulesT |
| Stripifier | 三角形条带 |
| VTK Writer | - |

---

### 2.3 RustGUI (GUI + 3D 渲染)

**进度: 0%** ⬜ 待启动

| 功能 | 技术选型 |
|------|----------|
| 3D 渲染 | egui + wgpu (推荐) |
| 相机控制 | 或 three-d |
| 界面框架 | egui / iced |

---

## 三、关键里程碑

### Phase 1: 核心连通 (当前)

```
目标: 实现完整的 3D 扫描 → 导出 流水线
```

- [ ] 实现 3DGS → Mesh 抽取 (关键！)
- [ ] RustMesh 属性系统集成
- [ ] 打通 SLAM → 3DGS → Mesh → 导出

**预计完成: 待定**

---

### Phase 2: 功能增强

```
目标: 完善算法工具链
```

- [ ] MeshChecker 网格验证
- [ ] 高级 Decimation 模块
- [ ] Modified Butterfly 细分
- [ ] 离线 3DGS 全局优化
- [ ] 纹理映射

**预计完成: 待定**

---

### Phase 3: 用户体验

```
目标: 提供可视化界面
```

- [ ] 创建 RustGUI 项目
- [ ] 实时 3D 可视化
- [ ] GUI 控制面板
- [ ] 多相机支持

**预计完成: 待定**

---

## 四、技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Rust 2021 |
| 数学库 | glam (SIMD) |
| GPU | wgpu, candle-metal |
| 图像 | opencv-rust, image |
| 优化 | apex-solver, g2o-rs |
| 并发 | rayon |
| 测试 | criterion |

---

## 五、与现有开源项目对比

| 特性 | ORB-SLAM3 | Open3D | RustScan |
|------|-----------|--------|----------|
| **SLAM** | ✅ | ❌ | ✅ (Phase 1) |
| **3DGS** | ❌ | ❌ | ✅ (Phase 1) |
| **网格处理** | ❌ | ✅ | ✅ (Phase 1) |
| **纯 Rust** | ❌ | ❌ | ✅ |
| **GPU 渲染** | ❌ | ✅ | ✅ (wgpu) |

---

## 六、代码统计

| 模块 | 源文件 | 测试 |
|------|--------|------|
| RustSLAM | 48 | 77 |
| RustMesh | ~45 | - |

---

## 七、任务看板

### P0 (阻塞流水线)
- [x] **3DGS → Mesh 抽取** - 已完成 (纯 Rust 实现)
- [ ] 属性系统集成 - 完善 OM 格式

### P1 (重要)
- [ ] MeshChecker 验证工具
- [ ] 高级 Decimation
- [ ] Modified Butterfly 细分

### P2 (增强)
- [ ] 自适应细分
- [ ] 离线 3DGS 优化

### P3 (用户体验)
- [ ] RustGUI 项目启动
- [ ] 实时可视化

---

## 八、贡献指南

### 代码风格
- 遵循 Rust 标准 (`rustfmt`)
- 添加单元测试
- 文档注释

### 提交规范
- 使用 conventional commits
- 关联相关模块

---

## 九、参考

- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [OpenMesh](https://www.openmesh.org/)
- [Open3D](http://www.open3d.org/)
- [SplaTAM](https://github.com/spla-tam/SplaTAM)
- [RTG-SLAM](https://github.com/MisEty/RTG-SLAM)
- [PensieveRust](https://github.com/sukie91/PensieveRust)
