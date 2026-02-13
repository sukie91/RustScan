# RustScanner

用 Rust 语言实现的 3D Scanner 全套算法库。

## 项目目标

打造一个纯 Rust 实现的 3D 扫描与重建技术栈，涵盖从数据获取到网格处理的完整流程。

## 核心模块

### RustSLAM (视觉 SLAM)

**纯 Rust 实现的视觉 SLAM 库**

- 特征提取 (ORB, AKAZE, SuperPoint)
- 视觉里程计 (VO + PnP)
- 局部建图 (三角化 + BA)
- 回环检测 (BoW)
- 可选的 TSDF 稠密融合

**技术栈**:
- opencv-rust: 图像处理
- glam: SIMD 数学库
- tch-rs: PyTorch 绑定
- g2o-rs: 图优化

### RustMesh

**核心网格表示与几何处理算法库**

- 网格数据结构 (Half-edge, SoA 布局)
- IO 格式支持 (OBJ, OFF, PLY, STL)
- 网格算法
  - 细分 (Loop, Catmull-Clark, Sqrt3)
  - 简化 (Decimation + Quadric 误差)
  - 光滑 (Laplace, Tangential)
  - 孔洞填充
  - 网格修复
  - 对偶变换
  - 渐进网格 (VDPM)

---

## 完整流水线

```
┌─────────────────────────────────────────────────────────────────┐
│                    3D Scanning Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [数据获取] → [SLAM] → [配准] → [融合] → [重建] → [后处理] → [导出] │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 模块设计

```
RustScanner/
├── RustMesh/           # 核心网格库 (已完成)
│   ├── Core/           # 基础数据结构
│   ├── Tools/          # 网格算法
│   └── Utils/          # 工具
│
├── slam/               # SLAM (待开发)
│   ├── frontend/       # 里程计 (视觉/激光/IMU)
│   ├── backend/       # 图优化
│   ├── loop_closure/  # 回环检测
│   └── mapping/       # 建图
│
├── registration/       # 配准 (待开发)
│   └── icp.rs
│
├── fusion/             # 融合 (待开发)
│   └── tsdf_fusion.rs
│
├── reconstruction/     # 表面重建 (待开发)
│   ├── poisson.rs
│   ├── ball_pivoting.rs
│   └── delaunay.rs
│
├── preprocessing/     # 预处理 (待开发)
│   ├── noise_filter.rs
│   └── downsampling.rs
│
└── io/                # IO (待扩展)
    └── (pcd, e57, las)
```

---

## 技术栈

- **语言**: Rust
- **数学库**: glam (SIMD 加速)
- **多线程**: rayon
- **GPU**: wgpu (compute shader)
- **对标**: OpenMesh, Open3D

---

## 优先级

| 优先级 | 模块 | 说明 |
|--------|------|------|
| P0 | SLAM | 核心，同时定位与建图 |
| P1 | 表面重建 | Poisson、Ball-Pivoting |
| P2 | 纹理映射 | UV 展开 + 贴图 |
| P3 | 传感器支持 | LiDAR、双目、结构光 |

---

## 参考

- [OpenMesh](https://www.openmesh.org/) - C++ 网格处理库
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) - 视觉 SLAM
- [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM) - 激光 SLAM
- [Open3D](http://www.open3d.org/) - 3D 重建库
- [PensieveRust](https://github.com/sukie91/PensieveRust) - 3D Gaussian Splatting
