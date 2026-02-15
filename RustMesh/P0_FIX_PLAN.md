# RustMesh P0 问题修复方案

## P0-5: 细分算法不删除原始面

### 问题分析
Loop/Catmull-Clark/Sqrt3 细分添加新面后，没有正确删除原始面。

### 原因
1. `delete_face()` 已在 P0-4 修复，现在能正确清理 halfedge
2. 细分代码调用了 `delete_face()`，但可能没有收集所有原始 face handle
3. 需要确保在创建新面之前保存原始 face handles

### 实现方案
```rust
// 1. 在细分开始时收集所有原始 face handles
let original_faces: Vec<FaceHandle> = mesh.faces().collect();

// 2. 细分操作（添加新顶点、新面）...

// 3. 删除原始面
for fh in &original_faces {
    mesh.delete_face(*fh);
}

// 4. 在所有细分操作完成后调用 mesh.compact() 清理
```

### ⚠️ 注意事项
1. **Handle 有效期**：收集的 `FaceHandle` 在细分过程中必须保持有效
2. **compact() 时机**：在所有细分操作完成后再调用 `compact()`，而非每次删除后立即调用

---

## P0-6: split_edge 只添加顶点不修改拓扑

### 问题分析
`split_edge()` 只调用 `add_vertex()` 添加孤立点，没有修改 half-edge 连通性。

### 实现方案
```rust
pub fn split_edge(mesh: &mut RustMesh, v0: VertexHandle, v1: VertexHandle) -> Result<VertexHandle> {
    // 1. 找到对应的 halfedge
    let heh = find_halfedge(mesh, v0, v1)?;
    
    // 2. 在边中点创建新顶点
    let mid_point = (mesh.point(v0) + mesh.point(v1)) / 2.0;
    let new_v = mesh.add_vertex(mid_point);
    
    // 3. 关键：创建两个新 halfedge 而不是只修改 to_vertex
    // 原 halfedge: v0 -> v1 变成 v0 -> new_v
    // 新 halfedge: new_v -> v1
    
    // 3.1 获取相关 halfedges
    let heh_prev = mesh.prev_halfedge_handle(heh);
    let heh_opp = mesh.opposite_halfedge_handle(heh);
    let heh_opp_next = mesh.next_halfedge_handle(heh_opp);
    let fh_left = mesh.face_handle(heh);
    let fh_right = mesh.face_handle(heh_opp);
    
    // 3.2 更新原 halfedge 的 to_vertex
    mesh.kernel.set_halfedge_to_vertex(heh, new_v);
    
    // 3.3 创建新 halfedge: new_v -> v1
    let new_heh = mesh.kernel.add_halfedge();
    mesh.kernel.set_halfedge_to_vertex(new_heh, v1);
    mesh.kernel.set_halfedge_next(new_heh, heh_opp_next);
    mesh.kernel.set_halfedge_face(new_heh, fh_right);
    
    // 3.4 更新 next/prev 指针形成环
    if let Some(prev) = heh_prev {
        mesh.kernel.set_halfedge_next(prev, heh);
    }
    if let Some(next) = heh_opp_next {
        mesh.kernel.set_halfedge_prev(next, new_heh);
    }
    mesh.kernel.set_halfedge_prev(heh, new_heh);
    mesh.kernel.set_halfedge_next(new_heh, heh_opp_next);
    
    // 3.5 如果面存在，更新面的 halfedge 指向
    if let Some(fh) = fh_right {
        mesh.kernel.set_face_halfedge(fh, new_heh);
    }
    
    Ok(new_v)
}
```

### ⚠️ 注意事项
- 需要处理 boundary edge 的特殊情况
- 需要更新所有受影响面的 halfedge 指针
- 需要确保 halfedge ring 的完整性

---

## P0-10: OBJ 读取不支持独立 normal/texcoord 索引

### 问题分析
OBJ 格式允许 `f v1/vt1/vn1`，v、vt、vn 可以用不同索引。当前实现忽略了 vn、vt 索引。

### 实现方案
```rust
// 解析 face 时存储独立索引
struct FaceIndex {
    v: usize,  // vertex index
    vt: Option<usize>,  // texcoord index  
    vn: Option<usize>, // normal index
}

fn parse_face_indices(face_str: &str) -> Vec<FaceIndex> {
    // "v1/vt1/vn1" -> FaceIndex { v: 0, vt: Some(0), vn: Some(0) }
    // "v1/vt1" -> FaceIndex { v: 0, vt: Some(0), vn: None }
    // "v1//vn1" -> FaceIndex { v: 0, vt: None, vn: Some(0) }
    // "v1" -> FaceIndex { v: 0, vt: None, vn: None }
}

// 存储时需要分别建立顶点、法线、UV 的数组
// 并在创建面时正确映射

### ⚠️ 注意事项
- RustMesh 需要扩展以支持独立的 normal/texcoord 属性
- 可能需要添加新的顶点属性系统
```

---

## P0-13: decimate() O(n²) 暴力搜索

### 问题分析
每次迭代遍历所有 halfedge 找最小 error。`CollapseCandidate` 和 `BinaryHeap` 已定义但未使用。

### 实现方案
```rust
pub fn decimate(&mut self, max_collapses: usize) -> usize {
    // 1. 初始化 BinaryHeap
    let mut heap: BinaryHeap<CollapseCandidate> = BinaryHeap::new();
    
    // 2. 初始时遍历所有 halfedge，计算 error 并加入堆
    for heh_idx in 0..self.mesh.n_halfedges() {
        let heh = HalfedgeHandle::new(heh_idx as u32);
        if let Some(error) = self.compute_collapse_error(heh) {
            heap.push(CollapseCandidate {
                priority: error,
                halfedge: heh,
            });
        }
    }
    
    let mut collapses = 0;
    let mut retry_count = 0;
    let max_retries = 1000;  // 防止无限循环
    
    // 3. 迭代：弹出最小 error 的候选
    while collapses < max_collapses && retry_count < max_retries {
        match heap.pop() {
            Some(candidate) => {
                // 检查是否仍然有效（可能因之前的 collapse 失效）
                if !is_collapse_still_valid(candidate.halfedge) {
                    retry_count += 1;
                    continue;
                }
                
                // 执行 collapse
                match self.mesh.collapse(candidate.halfedge) {
                    Ok(_) => {
                        collapses += 1;
                        retry_count = 0;  // 重置重试计数
                        
                        // 4. 更新受影响的邻居 halfedges 的 error
                        for neighbor_heh in get_affected_neighbors(candidate.halfedge) {
                            if let Some(error) = self.compute_collapse_error(neighbor_heh) {
                                heap.push(CollapseCandidate {
                                    priority: error,
                                    halfedge: neighbor_heh,
                                });
                            }
                        }
                    }
                    Err(_) => {
                        retry_count += 1;
                    }
                }
            }
            None => break,
        }
    }
    
    collapses
}
```

### ⚠️ 注意事项
- 添加 `max_retries` 防止无限循环
- 每次失败时增加 retry_count
- 成功时重置 retry_count

---

## 总结

| P0 | 难度 | 预计工作量 |
|----|------|----------|
| P0-5 细分删除原始面 | 中 | 1-2 小时 |
| P0-13 BinaryHeap | 中 | 1-2 小时 |
| P0-10 OBJ 独立索引 | 中 | 2 小时 |
| P0-6 split_edge 拓扑 | 高 | 2-3 小时 |

### 建议优先级
1. **P0-5** - 最简单，先做
2. **P0-13** - BinaryHeap 框架已有
3. **P0-10** - 独立索引解析
4. **P0-6** - 最复杂，最后做
