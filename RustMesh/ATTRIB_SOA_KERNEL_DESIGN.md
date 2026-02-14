# AttribSoAKernel 设计文档

## 概述

统一 `ArrayKernel` 的属性接口与 `SoAKernel` 的 SoA 优势为单一的 `AttribSoAKernel`，保留 SoA 内存布局以获得 SIMD 性能，同时支持动态属性系统。

## 现有代码分析

### SoAKernel (已实现)
- ✅ SoA 位置数据 (x, y, z slices)
- ✅ 连接数据 (halfedges, edges, faces)
- ✅ 预设属性 (vertex_normals, colors, texcoords)
- ✅ request_*/has_*/get_*/set_* 接口

### ArrayKernel + Attributes (已完成)
- ✅ 属性接口已合并进 `ArrayKernel` (AoS 布局)
- ✅ 预设属性 (normals/colors/texcoords)
- ✅ request_*/has_*/get_*/set_* 接口

## 设计目标

1. **保留 SoA 布局** - 获得 SIMD 性能
2. **统一接口** - request_*/has_*/get_*/set_*
3. **动态属性支持** - 类似 OpenMesh 的 property system
4. **OM 格式兼容** - 完善原生格式支持

## 架构设计

```rust
/// 属性类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttributeType {
    VertexNormal,
    VertexColor,
    VertexTexCoord,
    HalfedgeNormal,
    HalfedgeColor,
    HalfedgeTexCoord,
    EdgeColor,
    FaceNormal,
    FaceColor,
    Custom(u32),  // 动态属性
}

/// 属性句柄
#[derive(Debug, Clone, Copy)]
pub struct PropHandle {
    id: u32,
    attr_type: AttributeType,
}

/// AttribSoAKernel - 统一的 Kernel
pub struct AttribSoAKernel {
    // === SoA 基础数据 ===
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    halfedge_handles: Vec<Option<HalfedgeHandle>>,

    // === 连接数据 ===
    halfedges: Vec<Halfedge>,
    edges: Vec<Edge>,
    faces: Vec<Face>,
    edge_map: HashMap<(u32, u32), HalfedgeHandle>,

    // === 预设属性 (SoA 布局) ===
    vertex_normals: Option<Vec<Vec3>>,
    vertex_colors: Option<Vec<Vec4>>,
    vertex_texcoords: Option<Vec<Vec2>>,
    halfedge_normals: Option<Vec<Vec3>>,
    halfedge_colors: Option<Vec<Vec4>>,
    halfedge_texcoords: Option<Vec<Vec2>>,
    edge_colors: Option<Vec<Vec4>>,
    face_normals: Option<Vec<Vec3>>,
    face_colors: Option<Vec<Vec4>>,

    // === 动态属性 ===
    dynamic_props: HashMap<u32, DynamicProperty>,
}

/// 动态属性存储
enum DynamicProperty {
    Float(Vec<f32>),
    Vec2(Vec<Vec2>),
    Vec3(Vec<Vec3>),
    Vec4(Vec<Vec4>),
    Int(Vec<i32>),
}
```

## 接口设计

```rust
impl AttribSoAKernel {
    // --- 基础操作 (来自 SoAKernel) ---

    /// 添加顶点
    fn add_vertex(&mut self, point: Vec3) -> VertexHandle;

    /// 获取顶点位置
    fn point(&self, vh: VertexHandle) -> Option<Vec3>;

    /// 获取顶点数量
    fn n_vertices(&self) -> usize;

    // --- 预设属性 (统一接口) ---

    /// 请求顶点法 request_vertex_normals线
    fn(&mut self);

    /// 是否有顶点法线
    fn has_vertex_normals(&self) -> bool;

    /// 获取顶点法线
    fn vertex_normal(&self, vh: VertexHandle) -> Option<Vec3>;

    /// 设置顶点法线
    fn set_vertex_normal(&mut self, vh: VertexHandle, n: Vec3);

    /// 请求顶点颜色
    fn request_vertex_colors(&mut self);

    /// 是否有顶点颜色
    fn has_vertex_colors(&self) -> bool;

    /// 获取顶点颜色
    fn vertex_color(&self, vh: VertexHandle) -> Option<Vec4>;

    // ... 其他属性类似

    // --- 动态属性 ---

    /// 注册动态属性
    fn add_property<T: PropType>(&mut self, name: &str) -> PropHandle;

    /// 获取动态属性值
    fn get_property<T: PropType>(&self, handle: PropHandle, idx: usize) -> Option<&T>;

    /// 设置动态属性值
    fn set_property<T: PropType>(&mut self, handle: PropHandle, idx: usize, value: &T);
}
```

## 实现计划

### Phase 1: 创建基础结构
1. 创建 `src/Core/attrib_soa_kernel.rs`
2. 从 SoAKernel 复制基础功能
3. 添加属性类型枚举

### Phase 2: 添加动态属性
1. 实现 DynamicProperty 存储
2. 实现 add_property/get_property/set_property

### Phase 3: 集成和测试
1. 更新 RustMesh 使用新 kernel
2. 运行测试确保兼容性
3. 添加 OM 格式支持
