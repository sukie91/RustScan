# RustMesh OpenMesh Parity Roadmap

## Overview

RustMesh 与 OpenMesh 功能对齐的开发路线图。

**总估时**: ~121h (~15 工作日)

---

## Epic 1: IO 格式完整性

**目标**: 补齐常用 3D 文件格式的读写支持
**估时**: 21h
**优先级**: P0 (Phase 1)
**依赖**: 无

### Stories

| ID | Story | 描述 | 估时 | 状态 | 依赖 |
|----|-------|------|------|------|------|
| E1-S1 | PLY ASCII 读取 | 解析 header + ASCII 数据体，支持 vertex/face/normal/color | 4h | ✅ DONE | - |
| E1-S2 | PLY Binary 读取 | 支持 BinaryLittleEndian/BinaryBigEndian，字节序处理 | 3h | ✅ DONE | E1-S1 |
| E1-S3 | PLY 属性扩展 | 处理非标准 property 顺序，支持自定义属性映射 | 2h | ✅ DONE | E1-S2 |
| E1-S4 | STL ASCII 写入 | 输出 solid/facet/vertex 结构 | 2h | ✅ DONE | - |
| E1-S5 | STL Binary 写入 | 80-byte header + triangle data 二进制输出 | 2h | ✅ DONE | E1-S4 |
| E1-S6 | STL ASCII 读取 | 解析 solid/facet 结构，处理顶点重复 | 3h | ✅ DONE | - |
| E1-S7 | STL Binary 读取 | 读取二进制 triangle data，顶点合并去重 | 2h | ✅ DONE | E1-S6 |
| E1-S8 | IO 测试覆盖 | 添加 PLY/STL 读写单元测试 + 与 OpenMesh 输出对比 | 3h | ✅ DONE | E1-S1~S7 |

### 技术方案

#### E1-S1: PLY ASCII 读取

```rust
// io/ply.rs
pub fn read_ply(path: impl AsRef<Path>) -> io::Result<RustMesh> {
    // 1. 解析 header
    let header = parse_ply_header(&mut reader)?;

    // 2. 根据 property 定义映射属性索引
    let vertex_schema = build_vertex_schema(&header);
    let face_schema = build_face_schema(&header);

    // 3. 读取数据体
    for vertex in header.vertex_count {
        let line = reader.read_line()?;
        let values = parse_ascii_values(&line);
        let (pos, normal, color) = extract_vertex_data(&values, &vertex_schema);
        mesh.add_vertex_with_attrs(pos, normal, color);
    }

    for face in header.face_count {
        let line = reader.read_line()?;
        let indices = parse_face_indices(&line, &face_schema);
        mesh.add_face(&indices);
    }
}
```

#### E1-S6: STL ASCII 读取

```rust
// io/stl.rs
pub fn read_stl(path: impl AsRef<Path>) -> io::Result<RustMesh> {
    // STL 是 triangle-only，需要顶点合并
    let mut vertex_map: HashMap<Vec3, VertexHandle> = HashMap::new();

    for triangle in parse_stl_triangles(&reader)? {
        let mut handles = vec![];
        for v in [triangle.v0, triangle.v1, triangle.v2] {
            let vh = vertex_map.entry(v).or_insert_with(|| mesh.add_vertex(v));
            handles.push(*vh);
        }
        mesh.add_face(&handles);
    }
}
```

---

## Epic 2: 网格平滑增强

**目标**: 提升平滑算法精度，支持高级平滑模式
**估时**: 11h
**优先级**: P1 (Phase 2)
**依赖**: 无

### Stories

| ID | Story | 描述 | 估时 | 状态 | 依赖 |
|----|-------|------|------|------|------|
| E2-S1 | 三角形角度计算 | 新增 `triangle_angle_at_vertex()` 几何函数 | 1h | ✅ DONE | - |
| E2-S2 | 余切权重计算 | 实现 `compute_cotangent_weight()` per-vertex | 3h | ✅ DONE | E2-S1 |
| E2-S3 | 余切权重 Laplacian | 完善 `cotangent_weight_laplacian()` 平滑 | 2h | ✅ DONE | E2-S2 |
| E2-S4 | 自适应平滑 | 基于曲率自适应调整平滑强度 | 3h | ✅ DONE | E2-S3 |
| E2-S5 | 平滑测试覆盖 | 添加余切权重平滑测试 + 收敛性验证 | 2h | ✅ DONE | E2-S3 |

### 技术方案

#### E2-S1: 三角形角度计算

```rust
// Core/geometry.rs
/// 计算三角形在指定顶点处的内角
pub fn triangle_angle_at_vertex(p0: Vec3, p1: Vec3, p2: Vec3, at_vertex: usize) -> f32 {
    let (a, b, c) = match at_vertex {
        0 => (p0, p1, p2),
        1 => (p1, p2, p0),
        2 => (p2, p0, p1),
        _ => panic!("Invalid vertex index"),
    };

    let v1 = (b - a).normalize();
    let v2 = (c - a).normalize();
    v1.dot(v2).acos()
}
```

#### E2-S2: 余切权重计算

```rust
// Tools/smoother.rs
/// 计算顶点 vi 与邻居 vj 之间的余切权重
/// 公式: w_ij = (cot(α) + cot(β)) / 2
/// α, β 是边 (vi, vj) 对面的两个三角形角度
pub fn compute_cotangent_weights(mesh: &RustMesh, vh: VertexHandle) -> Vec<(VertexHandle, f32)> {
    let mut weights = vec![];

    for heh in mesh.vertex_halfedges(vh)? {
        let vj = mesh.to_vertex_handle(heh);

        // 获取相邻两个三角形
        let tri_left = get_adjacent_triangle(mesh, heh);
        let tri_right = get_adjacent_triangle(mesh, heh.opposite());

        // 计算对角 α 和 β
        let alpha = angle_at_opposite_vertex(mesh, tri_left, vh, vj);
        let beta = angle_at_opposite_vertex(mesh, tri_right, vh, vj);

        // cot = 1/tan
        let weight = (1.0 / alpha.tan() + 1.0 / beta.tan()) / 2.0;
        weights.push((vj, weight));
    }

    weights
}
```

---

## Epic 3: 网格重构算法

**目标**: 实现 Isotropic Remeshing，支持高质量网格重建
**估时**: 19h
**优先级**: P2 (Phase 3)
**依赖**: E2-S3 (余切权重 Laplacian)

### Stories

| ID | Story | 描述 | 估时 | 状态 | 依赖 |
|----|-------|------|------|------|------|
| E3-S1 | 边长统计工具 | 新增 `edge_length_histogram()` 分析函数 | 1h | TODO | - |
| E3-S2 | 长边分裂 | 实现 `split_long_edges(threshold)` | 2h | TODO | - |
| E3-S3 | 端边折叠 | 实现 `collapse_short_edges(threshold)` | 2h | TODO | - |
| E3-S4 | 边翻转操作 | 实现 `flip_edge()` 半边拓扑翻转 | 3h | TODO | - |
| E3-S5 | Valence 计算 | 实现 `compute_valence()` + valence 平衡目标计算 | 1h | TODO | - |
| E3-S6 | Valence 驱动边翻转 | 实现 `flip_edges_for_valence(target)` | 3h | TODO | E3-S4, E3-S5 |
| E3-S7 | Isotropic Remeshing 主体 | 组合上述操作，实现迭代 remesh 管道 | 4h | TODO | E2-S3, E3-S2~S6 |
| E3-S8 | Remesh 测试覆盖 | 添加 remesh 算法测试 + 网格质量指标验证 | 3h | TODO | E3-S7 |

### 技术方案

#### E3-S7: Isotropic Remeshing

```rust
// Tools/remeshing.rs (新文件)
/// Isotropic Remeshing (Botsch & Kobbelt 2004)
/// 目标: 均匀边长 + 保持形状
pub fn isotropic_remesh(mesh: &mut RustMesh, target_edge_length: f32, iterations: usize) {
    for _ in 0..iterations {
        // 1. Split long edges (L > 4/3 * target)
        split_long_edges(mesh, target_edge_length * 4.0 / 3.0);

        // 2. Collapse short edges (L < 4/5 * target)
        collapse_short_edges(mesh, target_edge_length * 4.0 / 5.0);

        // 3. Equalize valences (flip edges to balance degree)
        flip_edges_for_valence(mesh, target_valence: 6);

        // 4. Tangential relaxation
        tangential_smooth(mesh, config);
    }
}

fn flip_edge(mesh: &mut RustMesh, eh: EdgeHandle) -> Result<()> {
    // 边翻转拓扑操作:
    // 翻转边连接的两个三角形，改变边的连接顶点
    // 检查翻转合法性 (不产生非流形)
}
```

---

## Epic 4: 渐进网格完善

**目标**: 完善 VDPM refine 功能，支持真正的 vertex split
**估时**: 15h
**优先级**: P2 (Phase 3)
**依赖**: 无

### Stories

| ID | Story | 描述 | 估时 | 状态 | 依赖 |
|----|-------|------|------|------|------|
| E4-S1 | CollapseRecord 扩展 | 添加被删除拓扑信息记录 | 2h | ✅ DONE | - |
| E4-S2 | SplitTopologyInfo 设计 | 设计 vertex split 拓扑恢复数据结构 | 2h | ✅ DONE | E4-S1 |
| E4-S3 | Vertex Split 实现 | 实现 `vertex_split()` 逆操作 | 4h | ✅ DONE | E4-S2 |
| E4-S4 | VDPM Refine 重构 | 替换 clone-from-original 回退实现 | 3h | ✅ DONE | E4-S3 |
| E4-S5 | Progressive Mesh LOD | 实现 LOD 选择接口 `get_lod(level)` | 2h | ⚠️ PARTIAL | E4-S4 |
| E4-S6 | VDPM 测试覆盖 | 添加 simplify-refine 循环测试 + LOD 测试 | 2h | ✅ DONE | E4-S4 |

### 技术方案

#### E4-S1: CollapseRecord 扩展

```rust
// Tools/vdpm.rs
pub struct CollapseRecord {
    halfedge: HalfedgeHandle,
    v_removed: VertexHandle,
    v_kept: VertexHandle,
    original_v_kept_position: Vec3,
    original_v_removed_position: Vec3,  // 新增
    new_position: Vec3,
    error: f32,

    // 新增: 被删除的拓扑信息
    removed_faces: [FaceHandle; 2],     // 折叠删除的两个面
    removed_halfedges: Vec<HalfedgeHandle>, // 被删除的半边

    // 新增: 邻居信息 (用于 split 恢复)
    left_neighbor_face: Option<FaceHandle>,
    right_neighbor_face: Option<FaceHandle>,
    neighbor_vertices_before: Vec<VertexHandle>,
}
```

#### E4-S3: Vertex Split 实现

```rust
/// Vertex Split 是 Edge Collapse 的逆操作
pub fn vertex_split(mesh: &mut RustMesh, record: &CollapseRecord) -> Result<()> {
    // 1. 恢复 v_removed 顶点
    let v_restored = mesh.add_vertex(record.original_v_removed_position);

    // 2. 恢复被删除的半边和面
    // 需要重新连接拓扑:
    // - 创建新的半边 pair
    // - 创建新的两个三角形面
    // - 更新邻居面的连接

    // 3. 恢复 v_kept 的原始位置
    mesh.set_point(record.v_kept, record.original_v_kept_position);

    Ok(())
}
```

---

## Epic 5: 简化约束系统

**目标**: 实现模块化简化约束，支持 OpenMesh 风格的约束组合
**估时**: 18h
**优先级**: P2 (Phase 3)
**依赖**: 无

### Stories

| ID | Story | 描述 | 估时 | 状态 | 依赖 |
|----|-------|------|------|------|------|
| E5-S1 | DecimationModule Trait | 设计约束模块 trait 接口 | 2h | ✅ DONE | - |
| E5-S2 | ModQuadric 约束 | 实现 quadric 误差约束模块 | 2h | ✅ DONE | E5-S1 |
| E5-S3 | ModNormal 约束 | 实现法线偏差约束模块 | 2h | ✅ DONE | E5-S1 |
| E5-S4 | ModAspectRatio 约束 | 实现面长宽比约束模块 | 2h | ✅ DONE | E5-S1 |
| E5-S5 | ModBoundary 约束 | 实现边界保护约束模块 | 1h | ✅ DONE | E5-S1 |
| E5-S6 | Decimater 重构 | 重构 Decimater 支持模块组合 | 4h | ✅ DONE | E5-S1~S5 |
| E5-S7 | 约束优先级融合 | 实现多约束优先级聚合策略 | 2h | ✅ DONE | E5-S6 |
| E5-S8 | 约束测试覆盖 | 添加各约束模块独立测试 + 组合测试 | 3h | ✅ DONE | E5-S6 |

### 技术方案

#### E5-S1: DecimationModule Trait

```rust
// Tools/decimation/modules.rs (新文件)

/// 简化约束模块接口
pub trait DecimationModule {
    /// 模块名称
    fn name(&self) -> &str;

    /// 检查折叠是否合法
    fn is_collapse_legal(&self, mesh: &RustMesh, info: &CollapseInfo) -> bool;

    /// 计算折叠优先级 (返回 None 表示禁止)
    fn compute_priority(&self, mesh: &RustMesh, info: &CollapseInfo) -> Option<f32>;

    /// 折叠后处理 (更新模块内部状态)
    fn post_collapse(&self, mesh: &mut RustMesh, record: &CollapseRecord);
}

pub struct CollapseInfo {
    halfedge: HalfedgeHandle,
    v_from: VertexHandle,
    v_to: VertexHandle,
    target_position: Vec3,
}
```

#### E5-S2~S5: 约束模块实现

```rust
// Quadric 误差约束
pub struct ModQuadric {
    max_error: f32,
    boundary_factor: f32,
}

impl DecimationModule for ModQuadric {
    fn compute_priority(&self, mesh: &RustMesh, info: &CollapseInfo) -> Option<f32> {
        let error = compute_quadric_error(mesh, info);
        if error > self.max_error { None } else { Some(error) }
    }
}

// 法线偏差约束
pub struct ModNormal {
    max_normal_deviation: f32, // radians
}

impl DecimationModule for ModNormal {
    fn is_collapse_legal(&self, mesh: &RustMesh, info: &CollapseInfo) -> bool {
        let deviation = compute_normal_deviation(mesh, info);
        deviation < self.max_normal_deviation
    }
}
```

---

## Epic 6: 遍历器完善

**目标**: 补齐缺失的遍历器，完善遍历 API
**估时**: 8h
**优先级**: P1 (Phase 2)
**依赖**: 无

### Stories

| ID | Story | 描述 | 估时 | 状态 | 依赖 |
|----|-------|------|------|------|------|
| E6-S1 | HalfedgeHalfedge 遍历器 | 实现 HH circulator (共享顶点/面相邻) | 2h | TODO | - |
| E6-S2 | EdgeEdge 遍历器 | 实现 EE circulator (共享顶点相邻) | 2h | TODO | - |
| E6-S3 | 遍历器性能优化 | 添加遍历器缓存优化 + batch 模式 | 2h | TODO | - |
| E6-S4 | 遍历器测试覆盖 | 添加新遍历器测试 + 遍历完整性验证 | 2h | TODO | E6-S1, E6-S2 |

### 技术方案

#### E6-S1: HalfedgeHalfedge 遍历器

```rust
// Utils/circulators.rs

/// Halfedge-Halfedge Circulator
/// 遍历与给定半边相邻的所有半边
pub struct HalfedgeHalfedgeCirculator<'a> {
    mesh: &'a RustMesh,
    center_heh: HalfedgeHandle,
    phase: HHPhase,
    current_heh: HalfedgeHandle,
}

enum HHPhase {
    /// 遍历从 to_vertex 出发的半边
    ToVertexOutgoing,
    /// 遍历从 from_vertex 出发的半边
    FromVertexOutgoing,
    /// 遍历同一面内的相邻半边
    FaceAdjacent,
    Done,
}

impl Iterator for HalfedgeHalfedgeCirculator<'_> {
    type Item = HalfedgeHandle;
    fn next(&mut self) -> Option<Self::Item> { ... }
}
```

---

## Epic 7: 细分算法扩展

**目标**: 补充中点和插值细分方案
**估时**: 9h
**优先级**: P1 (Phase 2)
**依赖**: E2-S2 (余切权重，用于 Butterfly)

### Stories

| ID | Story | 描述 | 估时 | 状态 | 依赖 |
|----|-------|------|------|------|------|
| E7-S1 | 中点细分 | 实现 `midpoint_subdivide()` | 2h | ✅ DONE | - |
| E7-S2 | Butterfly 细分 | 实现 Butterfly 插值细分 | 3h | ✅ DONE | E2-S2 |
| E7-S3 | 插值细分框架 | 设计通用插值细分 trait | 2h | ✅ DONE | E7-S2 |
| E7-S4 | 细分测试覆盖 | 添加中点/Butterfly 测试 + 层数对比 | 2h | ✅ DONE | E7-S1, E7-S2 |

### 技术方案

#### E7-S1: 中点细分

```rust
// Tools/subdivision.rs

/// 中点细分: 每条边在中点分裂
pub fn midpoint_subdivide(mesh: &mut RustMesh) -> SubdivisionStats {
    let edges: Vec<_> = mesh.edges().collect();

    for eh in edges {
        let midpoint = edge_midpoint(mesh, eh);
        split_edge(mesh, eh, midpoint);
    }

    // 连接新顶点形成四边形 (三角网格需特殊处理)
}
```

#### E7-S2: Butterfly 细分

```rust
/// Butterfly 细分 (插值细分，保持原始顶点位置)
/// 新顶点位置由邻居顶点加权计算
pub fn butterfly_subdivide(mesh: &mut RustMesh) -> SubdivisionStats {
    // Butterfly stencil 权重:
    // 对于边 (v0, v1)，新顶点位置:
    // p_new = 0.5*(v0+v1) + 0.125*(v2+v3) - 0.0625*(v4+v5+v6+v7)
    // 其中 v2,v3 是相邻三角形对角顶点
    // v4,v5,v6,v7 是更远的邻居
}
```

---

## Epic 8: 几何分析工具

**目标**: 补充几何分析功能
**估时**: 13h
**优先级**: P3 (Phase 4)
**依赖**: E2-S2 (余切权重), E1-S1 (PLY 写入)

### Stories

| ID | Story | 描述 | 估时 | 状态 | 依赖 |
|----|-------|------|------|------|------|
| E8-S1 | 最小包围球 | 实现 `minimum_enclosing_sphere()` | 2h | ✅ DONE | - |
| E8-S2 | Voronoi 面积计算 | 实现 `voronoi_area()` for cotangent weights | 2h | ✅ DONE | E2-S2 |
| E8-S3 | Gauss 曲率估计 | 实现 `gaussian_curvature()` | 3h | ✅ DONE | E8-S2 |
| E8-S4 | Mean 曲率估计 | 实现 `mean_curvature()` | 2h | ✅ DONE | E8-S2 |
| E8-S5 | 曲率可视化导出 | 导出曲率到 PLY scalar 属性 | 2h | ✅ DONE | E1-S1, E8-S3, E8-S4 |
| E8-S6 | 分析测试覆盖 | 添加曲率/包围球测试 + 与 MeshLab 对比 | 2h | ✅ DONE | E8-S1~S5 |

### 技术方案

#### E8-S3: Gauss 曲率

```rust
// Utils/analysis.rs (新文件)

/// Gauss 曲率: K = (2π - Σθi) / Ai
/// 其中 θi 是顶点处各三角形的角度
/// Ai 是 Voronoi 面积
pub fn gaussian_curvature(mesh: &RustMesh, vh: VertexHandle) -> f32 {
    let mut angle_sum = 0.0;
    let voronoi_area = voronoi_area(mesh, vh);

    for fh in mesh.vertex_faces(vh)? {
        let angle = vertex_angle_in_face(mesh, vh, fh);
        angle_sum += angle;
    }

    (2.0 * PI - angle_sum) / voronoi_area
}
```

---

## Epic 9: 性能优化

**目标**: 完善 SIMD 实现，优化性能瓶颈
**估时**: 11h
**优先级**: P3 (Phase 4)
**依赖**: 无

### Stories

| ID | Story | 描述 | 估时 | 状态 | 依赖 |
|----|-------|------|------|------|------|
| E9-S1 | SIMD 质心修复 | 修复 `compute_centroid_simd()` 作用域 bug | 1h | ✅ DONE | - |
| E9-S2 | SIMD 边界框完善 | 完善 `bounding_box_simd()` NEON 实现 | 2h | ✅ DONE | - |
| E9-S3 | 并行遍历器 | 实现 rayon 并行 circulator | 3h | ✅ DONE | - |
| E9-S4 | 批量操作优化 | 实现 batch collapse/split 减少拓扑更新开销 | 3h | ✅ DONE | - |
| E9-S5 | 性能基准测试 | 添加 cargo bench 测试 + 与 OpenMesh 对比 | 2h | ✅ DONE | E9-S1~S4 |

---

## Epic 10: 对偶化扩展

**目标**: 支持带边界网格的对偶化
**估时**: 6h
**优先级**: P3 (Phase 4)
**依赖**: 无

### Stories

| ID | Story | 描述 | 估时 | 状态 | 依赖 |
|----|-------|------|------|------|------|
| E10-S1 | 边界对偶处理设计 | 设计边界边对偶策略 | 2h | ✅ DONE | - |
| E10-S2 | 带边界对偶实现 | 实现 `dualize_with_boundary()` | 3h | ✅ DONE | E10-S1 |
| E10-S3 | 对偶化测试覆盖 | 添加边界网格对偶测试 | 1h | ✅ DONE | E10-S2 |

---

## 开发阶段规划

### Phase 1: IO 基础 (Week 1)

```
E1: IO 格式完整性
├─ E1-S1~S3: PLY 读取 (9h)
├─ E1-S4~S7: STL 写入/读取 (9h)
└─ E1-S8: IO 测试 (3h)
```

### Phase 2: 核心算法 (Week 2)

```
E2: 网格平滑增强 (11h)
E6: 遍历器完善 (8h)
E7: 细分算法扩展 (9h)
```

### Phase 3: 高级算法 (Week 3)

```
E3: 网格重构算法 (19h)
E4: 渐进网格完善 (15h)
E5: 简化约束系统 (18h)
```

### Phase 4: 分析优化 (Week 4)

```
E8: 几何分析工具 (13h)
E9: 性能优化 (11h)
E10: 对偶化扩展 (6h)
```

---

## 依赖关系图

```
Phase 1                          Phase 2                          Phase 3                          Phase 4
┌─────────────┐                 ┌─────────────┐                 ┌─────────────┐                 ┌─────────────┐
│ E1: IO      │                 │ E2: Smooth  │                 │ E3: Remesh  │                 │ E8: Analysis│
│ (无依赖)     │                 │ (无依赖)     │ ──need──▶      │ (need E2-S3)│ ──need──▶      │ (need E2, E1)│
└─────────────┘                 └─────────────┘                 └─────────────┘                 └─────────────┘
                                       │                              │                              │
                                       │                              │                              │
                                       ▼                              ▼                              ▼
                                ┌─────────────┐                 ┌─────────────┐                 ┌─────────────┐
                                │ E6: Circ    │                 │ E4: VDPM    │                 │ E9: Perf    │
                                │ (无依赖)     │                 │ (无依赖)     │                 │ (无依赖)     │
                                └─────────────┘                 └─────────────┘                 └─────────────┘
                                       │                              │
                                       │                              │
                                       ▼                              ▼
                                ┌─────────────┐                 ┌─────────────┐                 ┌─────────────┐
                                │ E7: Subdiv  │                 │ E5: Constr  │                 │ E10: Dual   │
                                │ (need E2-S2)│                 │ (无依赖)     │                 │ (无依赖)     │
                                └─────────────┘                 └─────────────┘                 └─────────────┘
```

---

## 验收标准

### E1: IO 格式完整性
- [ ] PLY ASCII/Binary 读写正确
- [ ] STL ASCII/Binary 读写正确
- [ ] 与 OpenMesh/MeshLab 输出对比一致
- [ ] 支持自定义属性映射

### E2: 网格平滑增强
- [ ] 余切权重 Laplacian 收敛正确
- [ ] 平滑后网格无退化
- [ ] 与 OpenMesh smooth 结果对比

### E3: 网格重构算法
- [ ] Isotropic remesh 后边长分布均匀
- [ ] 保持原始形状特征
- [ ] Valence 分布接近理想值 6

### E4: 渐进网格完善
- [ ] Simplify-refine 循环可逆
- [ ] LOD 选择正确工作
- [ ] 无拓扑错误

### E5: 简化约束系统
- [ ] 各约束模块独立工作正确
- [ ] 模块组合正确融合优先级
- [ ] 与 OpenMesh Decimater 对比

---

## 测试策略

1. **单元测试**: 每个 Story 完成后添加单元测试
2. **对比测试**: 与 OpenMesh C++ 实现输出对比
3. **集成测试**: Epic 完成后添加端到端测试
4. **性能测试**: Phase 4 完成 cargo bench

---

## 参考文档

- OpenMesh Documentation: https://www.openmesh.org/
- Botsch & Kobbelt 2004: "A Remeshing Approach to Multiresolution Modeling"
- Hoppe 1996: "Progressive Meshes"
- Sorkine et al. 2004: "Laplacian Surface Editing"