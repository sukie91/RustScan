# RustGS LiteGS Parity Roadmap

> Generated: 2026-04-04
> Status: Planning

本文档记录 RustGS 与 Mirror/LiteGS 功能对齐的开发计划。

## Feature Gap Summary

基于对 LiteGS (Python+CUDA)、Gausplat (Rust+WGPU)、RustGS (Rust+Metal) 三个项目的完整对比分析，RustGS 目前缺失以下关键功能。

### 高优先级缺失 (P0)

| 功能 | LiteGS | Gausplat | RustGS | 影响 |
|------|--------|----------|--------|------|
| Clustered Gaussian | ✅ chunk_size=128 | ❌ | ❌ | 内存访问效率、cluster-level剔除 |
| Sparse梯度优化 | ✅ 只更新可见 | ❌ | ⚠️ 部分 | 训练速度 |
| Morton空间重排序 | ✅ densify时 | ❌ | ❌ | 内存coherence |
| 可学习相机外参 | ✅ pose embedding | ❌ | ❌ | 大场景pose drift |
| 渐进SH激活 | ✅ 每5 epoch | ✅ iter 1000-4000 | ❌ | 训练稳定性 |
| 低通滤波 | ⚠️ | ✅ diag(0.3) | ❌ | Aliasing artifacts |
| 学习率衰减 | ✅ position decay | ✅ position decay | ⚠️ 固定LR | 收敛质量 |
| TamingGS预算控制 | ✅ target_primitives | ❌ | ⚠️ 基础 | Gaussian数量控制 |
| 透明度重置 | ✅ 每3000 iter | ❌ | ⚠️ 配置可选 | Densify效果 |

### 中优先级缺失 (P1)

| 功能 | LiteGS | Gausplat | RustGS | 影响 |
|------|--------|----------|--------|------|
| COLMAP数据加载 | ✅ | ✅ | ❌ (只有TUM) | 数据集兼容性 |
| Scale正则化 | ✅ optional | ❌ | ❌ | 过大Gaussian |
| Transmitance正则化 | ✅ | ❌ | ❌ | 背景区域 |
| Early exit阈值 | ✅ | ✅ trans<(1/255)² | ⚠️ | 渲染性能 |
| Full fused CUDA ops | ✅ 多个 | ⚠️ 分散kernel | ⚠️ 分散kernel | Kernel launch开销 |

---

## Epic & Story Breakdown

### Epic 1: Training Stability Improvements (训练稳定性增强)

**Priority**: P0 - Highest
**Goal**: Improve training convergence quality and stability

#### Story 1.1: Progressive SH Activation (渐进SH激活)

**Description**: Implement progressive SH degree activation, starting from DC-only, incrementing one degree every N iterations.

**Reference**:
- LiteGS: Every 5 epochs
- Gausplat: Iterations 1000-4000, every 1000

**Acceptance Criteria**:
- [ ] Initial training uses only DC (degree 0)
- [ ] Configurable activation interval and max degree
- [ ] Gradients propagate correctly when degree increases
- [ ] Training stability improves (measured by loss curve smoothness)

**Implementation Notes**:
```rust
// In training config
pub struct ShProgressionConfig {
    pub start_iteration: usize,      // Default: 0
    pub interval: usize,             // Default: 1000
    pub max_degree: usize,           // Default: 3
}

// During training
let active_sh_degree = min(
    (iteration - config.start_iteration) / config.interval,
    config.max_degree
);
```

**Estimate**: 2-3 days
**Dependencies**: None

---

#### Story 1.2: Learning Rate Decay (学习率衰减)

**Description**: Implement exponential decay for position learning rate.

**Reference**:
- LiteGS: `lr_init=0.00016 → lr_final=0.0000016` over 30000 steps
- Gausplat: Same schedule

**Acceptance Criteria**:
- [ ] Position LR decays from `0.00016` to `0.0000016`
- [ ] Configurable decay start/end iterations
- [ ] Optional decay for other parameters (scale, rotation, color)
- [ ] LR schedule logged in training output

**Implementation Notes**:
```rust
pub fn compute_position_lr(iteration: usize, config: &TrainingConfig) -> f32 {
    if iteration < config.lr_decay_start {
        return config.lr_position_init;
    }
    if iteration > config.lr_decay_end {
        return config.lr_position_final;
    }
    let progress = (iteration - config.lr_decay_start) as f32
                 / (config.lr_decay_end - config.lr_decay_start) as f32;
    config.lr_position_init * (config.lr_position_final / config.lr_position_init).powf(progress)
}
```

**Estimate**: 1-2 days
**Dependencies**: None

---

#### Story 1.3: Opacity Reset (透明度重置)

**Description**: Periodically reset opacity values to avoid over-lock-in.

**Reference**: LiteGS resets every 3000 iterations:
- Mode A: Decay by 0.5
- Mode B: Clamp to 0.005

**Acceptance Criteria**:
- [ ] Configurable reset interval (default 3000)
- [ ] Support decay mode: `opacity *= 0.5`
- [ ] Support clamp mode: `opacity = min(opacity, 0.005)`
- [ ] Coordinates with densification schedule
- [ ] Option to reset after densify only

**Implementation Notes**:
```rust
pub enum OpacityResetMode {
    Decay(f32),   // opacity *= factor (default 0.5)
    Clamp(f32),   // opacity = min(opacity, threshold)
}

if iteration % config.opacity_reset_interval == 0 {
    for g in &mut gaussians {
        match config.opacity_reset_mode {
            OpacityResetMode::Decay(f) => g.opacity *= f,
            OpacityResetMode::Clamp(t) => g.opacity = g.opacity.min(t),
        }
    }
}
```

**Estimate**: 1 day
**Dependencies**: None

---

### Epic 2: Memory & Compute Efficiency (内存与计算效率)

**Priority**: P0 - Highest
**Goal**: Optimize memory access patterns and compute efficiency

#### Story 2.1: Morton Code Spatial Reordering (Morton空间重排序)

**Description**: Reorder Gaussians by Morton code during densification for better memory coherence.

**Reference**: LiteGS `_gen_morton_code()` - 21-bit Z-order curve

**Acceptance Criteria**:
- [ ] Implement Morton code calculation (interleave x/y/z bits)
- [ ] Automatic reordering after densification
- [ ] Configurable bits count (default 21)
- [ ] Parity tests do not regress
- [ ] Memory access pattern verified via profiling

**Implementation Notes**:
```rust
pub fn morton_encode_21bit(x: u32, y: u32, z: u32) -> u64 {
    // Interleave bits: x at positions 0,3,6,...; y at 1,4,7,...; z at 2,5,8,...
    fn spread_bits(v: u32) -> u64 {
        let mut v = v as u64;
        v = (v | (v << 16)) & 0x030000FF;
        v = (v | (v << 8))  & 0x0300F00F;
        v = (v | (v << 4))  & 0x030C30C3;
        v = (v | (v << 2))  & 0x09249249;
        v
    }
    spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)
}

pub fn reorder_by_morton(gaussians: &mut Vec<Gaussian>, bounds: &BoundingBox) {
    // Normalize positions to [0, 2^21-1]
    // Compute Morton codes
    // Sort by code
    // Reorder Gaussian array
}
```

**Estimate**: 3-4 days
**Dependencies**: Densification module

---

#### Story 2.2: Sparse Gradient Optimization (稀疏梯度优化)

**Description**: Only update visible Gaussians' parameters, skip invisible regions.

**Reference**: LiteGS `SparseGaussianAdam` + `CompactedTensor`

**Acceptance Criteria**:
- [ ] Track `visible_chunk` and `primitive_visible` mask
- [ ] Adam step only processes visible indices
- [ ] Support compacted gradient tensor storage
- [ ] Training speed improvement >20%
- [ ] Memory usage reduced for sparse updates

**Implementation Notes**:
```rust
pub struct SparseAdam {
    // First moment: only stored for active indices
    m: HashMap<usize, Vec3>,  // or compacted array
    // Second moment
    v: HashMap<usize, Vec3>,
}

pub fn sparse_adam_step(
    params: &mut [f32],
    grads: &[f32],
    visible_indices: &[usize],
    m: &mut [f32],
    v: &mut [f32],
    config: &AdamConfig,
) {
    for idx in visible_indices {
        let g = grads[idx];
        m[idx] = config.beta1 * m[idx] + (1 - config.beta1) * g;
        v[idx] = config.beta2 * v[idx] + (1 - config.beta2) * g * g;
        // Bias-corrected update
        params[idx] -= config.lr * m_hat / (sqrt(v_hat) + eps);
    }
}
```

**Estimate**: 4-5 days
**Dependencies**: Story 2.3 (Clustered Gaussian)

---

#### Story 2.3: Clustered Gaussian Representation (聚类Gaussian表示)

**Description**: Organize Gaussians into chunks of 128 for efficient memory access.

**Reference**: LiteGS `[3, N] → [3, chunks_num, chunk_size]` layout

**Acceptance Criteria**:
- [ ] Data structure supports cluster layout
- [ ] Cluster-level frustum culling implemented
- [ ] Compatible with existing render pipeline
- [ ] Memory access pattern verified
- [ ] Performance improvement measured

**Implementation Notes**:
```rust
pub struct ClusteredGaussians {
    pub positions: Array3<f32>,     // [3, chunks, 128]
    pub scales: Array3<f32>,        // [3, chunks, 128]
    pub rotations: Array3<f32>,     // [4, chunks, 128]
    pub opacities: Array2<f32>,     // [chunks, 128]
    pub colors_sh: Array3<f32>,     // [coeffs, chunks, 128]
    pub chunk_size: usize,          // Default 128
    pub num_chunks: usize,
}

pub fn frustum_cull_clusters(
    clusters: &ClusteredGaussians,
    camera: &GaussianCamera,
) -> Vec<usize> {
    // Compute cluster AABB from positions + scales
    // Frustum test each cluster AABB
    // Return visible cluster indices
}
```

**Estimate**: 5-7 days
**Dependencies**: None

---

### Epic 3: Quality Enhancement (质量增强)

**Priority**: P1 - High
**Goal**: Improve rendering quality and reconstruction accuracy

#### Story 3.1: Anti-aliasing Low-pass Filter (低通滤波)

**Description**: Add low-pass filter to 2D covariance to prevent aliasing.

**Reference**: Gausplat adds `diag(0.3, 0.3)` to covariance

**Acceptance Criteria**:
- [ ] 2D covariance receives additive `[[0.3, 0], [0, 0.3]]`
- [ ] Configurable filter strength
- [ ] Far-distance Gaussians quality improved
- [ ] No visible artifacts in rendered images

**Implementation Notes**:
```rust
// In covariance projection
let cov_2d = compute_2d_covariance(...);
// Add low-pass filter
cov_2d[0][0] += config.lowpass_filter;  // 0.3
cov_2d[1][1] += config.lowpass_filter;
// Recompute eigenvalues/inverse
```

**Estimate**: 1 day
**Dependencies**: Render pipeline

---

#### Story 3.2: Learnable Camera Extrinsics (可学习相机外参)

**Description**: Support end-to-end optimization of camera poses.

**Reference**: LiteGS `Embedding(num_frames, 7)` + `CreateViewProj.apply`

**Acceptance Criteria**:
- [ ] Support quaternion + translation embedding per frame
- [ ] Sparse Adam optimizer for pose parameters
- [ ] Configurable pose LR (default 1e-4)
- [ ] Reconstruction accuracy improvement verified
- [ ] Pose drift correction in large scenes

**Implementation Notes**:
```rust
pub struct PoseEmbedding {
    pub quaternions: Vec<Quat>,  // [num_frames, 4]
    pub translations: Vec<Vec3>, // [num_frames, 3]
}

pub fn create_view_matrix(embedding: &PoseEmbedding, frame_idx: usize) -> Mat4 {
    let q = embedding.quaternions[frame_idx];
    let t = embedding.translations[frame_idx];
    // q.to_matrix() + t
}

// During training
let pose_grad = loss.backward().pose_gradients;
sparse_adam_step(&mut embedding, pose_grad, visible_frames, pose_lr);
```

**Estimate**: 4-5 days
**Dependencies**: None

---

#### Story 3.3: TamingGS Budget-controlled Densify (预算控制Densify)

**Description**: Implement score-based densification with target primitive count.

**Reference**: LiteGS `DensityControllerTamingGS` + multinomial sampling

**Acceptance Criteria**:
- [ ] Compute score = variance × fragment_count × opacity²
- [ ] Multinomial sampling selects densify candidates
- [ ] Target Gaussian count approaches `target_primitives`
- [ ] Avoid unbounded Gaussian growth
- [ ] Quality maintained with controlled count

**Implementation Notes**:
```rust
pub fn compute_densify_scores(
    gaussians: &[Gaussian],
    stats: &StatisticsHelper,
) -> Vec<f32> {
    gaussians.iter().map(|g| {
        let variance = stats.get_variance(g.id);
        let fragments = stats.get_fragment_count(g.id);
        let opacity = g.opacity;
        variance * fragments * opacity * opacity
    }).collect()
}

pub fn sample_densify_candidates(scores: &[f32], budget: usize) -> Vec<usize> {
    // Normalize scores to probabilities
    // Multinomial sample budget indices
    // Return candidate Gaussian IDs
}
```

**Estimate**: 4-5 days
**Dependencies**: Statistics helper (existing foundation)

---

### Epic 4: Regularization & Refinement (正则化与细化)

**Priority**: P2 - Medium
**Goal**: Add regularization constraints to prevent anomalous Gaussians

#### Story 4.1: Scale Regularization (缩放正则化)

**Description**: Add L2 regularization on scale to prevent oversized Gaussians.

**Reference**: LiteGS `scale.square().mean() * reg_weight`

**Acceptance Criteria**:
- [ ] Configurable `reg_weight` (default 0 or 0.01)
- [ ] Gradients propagate to log_scale
- [ ] Large scale Gaussian count reduced
- [ ] Loss formula: `loss += scale.powf(2).mean() * weight`

**Implementation Notes**:
```rust
let scale_reg_loss = gaussians.iter()
    .map(|g| g.scale.length_squared())
    .sum::<f32>() / gaussians.len() as f32;

loss += scale_reg_loss * config.scale_reg_weight;
```

**Estimate**: 1 day
**Dependencies**: None

---

#### Story 4.2: Transmitance/Background Regularization (背景正则化)

**Description**: Add transmitance regularization for transparent background regions.

**Reference**: LiteGS `(1 - transmitance).abs().mean()`

**Acceptance Criteria**:
- [ ] Configurable weight
- [ ] Render outputs transmitance map
- [ ] Background regions alpha approaches 0
- [ ] Loss formula: `loss += (1 - transmitance).abs().mean()`

**Implementation Notes**:
```rust
// Render must output transmitance
let transmitance_map = render_with_transmitance(...);
let bg_reg_loss = transmitance_map.iter()
    .map(|t| (1.0 - t).abs())
    .sum::<f32>() / transmitance_map.len() as f32;

loss += bg_reg_loss * config.transmitance_reg_weight;
```

**Estimate**: 1-2 days
**Dependencies**: Render pipeline transmitance output

---

#### Story 4.3: Prune Over-scaled Gaussians (剔除过大Gaussian)

**Description**: Prune Gaussians with excessive scale during prune phase.

**Reference**: Gausplat prunes `scale > 0.5` (10x threshold)

**Acceptance Criteria**:
- [ ] Configurable scale prune threshold
- [ ] Combined with opacity prune
- [ ] Floating Gaussians/flyers reduced
- [ ] Prune formula: `scale.max() > threshold`

**Implementation Notes**:
```rust
pub fn prune_over_scaled(gaussians: &mut Vec<Gaussian>, threshold: f32) {
    gaussians.retain(|g| {
        g.opacity > config.opacity_threshold
        && g.scale.max_element() < threshold  // 0.5 default
    });
}
```

**Estimate**: 0.5 days
**Dependencies**: Prune module

---

### Epic 5: Rendering Performance (渲染性能优化)

**Priority**: P2 - Medium
**Goal**: Improve rendering speed

#### Story 5.1: Early Exit Threshold Optimization (Early exit优化)

**Description**: Optimize alpha blending early exit condition.

**Reference**: Gausplat `transmittance < (1/255)² ≈ 0.0001`

**Acceptance Criteria**:
- [ ] Configurable exit threshold
- [ ] GPU kernel handles early exit correctly
- [ ] Rendering speed improvement >10%
- [ ] No quality degradation

**Implementation Notes**:
```metal
// In Metal kernel
constant float EARLY_EXIT_THRESHOLD = 0.0001;

if (transmittance < EARLY_EXIT_THRESHOLD) {
    // Early exit from blending loop
    break;
}
```

**Estimate**: 1 day
**Dependencies**: Metal kernel

---

#### Story 5.2: Fused Transform Kernel (融合Transform kernel)

**Description**: Merge MVP transform + covariance + SH→RGB into single kernel.

**Reference**: LiteGS/Gausplat fused ops

**Acceptance Criteria**:
- [ ] Single Metal kernel completes preprocessing
- [ ] Reduced kernel launch overhead
- [ ] Intermediate results in shared memory
- [ ] Performance improvement measured

**Implementation Notes**:
```metal
kernel void fused_transform(
    constant CameraUniform& camera,
    device const float* positions,
    device const float* log_scales,
    device const float* rotations,
    device const float* sh_coeffs,
    device float* out_projected,
    device float* out_cov_2d,
    device float* out_rgb,
    uint gid [[thread_position_in_grid]]
) {
    // All in one kernel:
    // 1. Transform to camera space
    // 2. Compute 3D covariance
    // 3. Project to 2D covariance
    // 4. Evaluate SH → RGB
}
```

**Estimate**: 3-4 days
**Dependencies**: Metal kernel

---

### Epic 6: Dataset Compatibility (数据集兼容)

**Priority**: P3 - Low
**Goal**: Extend dataset support

#### Story 6.1: COLMAP Dataset Loader (COLMAP数据加载)

**Description**: Support COLMAP format cameras/images/points3D.bin.

**Reference**: LiteGS `io_manager/colmap.py`

**Acceptance Criteria**:
- [ ] Parse binary and text formats
- [ ] Support PINHOLE camera model
- [ ] Generate TrainableGaussians
- [ ] Integrate with existing pipeline
- [ ] Test with standard COLMAP datasets

**Implementation Notes**:
```rust
pub fn load_colmap_dataset(path: &Path) -> Result<TrainingDataset, Error> {
    // Parse cameras.bin (or .txt)
    let cameras = parse_colmap_cameras(path.join("cameras.bin"))?;

    // Parse images.bin
    let images = parse_colmap_images(path.join("images.bin"))?;

    // Parse points3D.bin
    let points = parse_colmap_points3d(path.join("points3D.bin"))?;

    // Convert to TrainingDataset
}
```

**Estimate**: 3-4 days
**Dependencies**: None

---

#### Story 6.2: PLY Format Alignment (PLY格式对齐)

**Description**: Align PLY output format with LiteGS standard for viewer compatibility.

**Reference**: LiteGS PLY header with `f_dc_*, f_rest_*, nx, ny, nz`

**Acceptance Criteria**:
- [ ] Output includes `nx, ny, nz` (zeros)
- [ ] `f_dc_0, f_dc_1, f_dc_2` correct order
- [ ] `f_rest_0...f_rest_44` correctly mapped
- [ ] Compatible with 3DGS viewer tools (e.g., 3DGS viewer, Luma AI)
- [ ] Rotation quaternion order matches standard (w-first)

**Implementation Notes**:
```rust
// PLY header
ply
format binary_little_endian 1.0
element vertex {count}
property float x
property float y
property float z
property float nx  // 0.0
property float ny  // 0.0
property float nz  // 0.0
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float f_rest_0
...
property float f_rest_44
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0  // w
property float rot_1  // x
property float rot_2  // y
property float rot_3  // z
end_header
```

**Estimate**: 1-2 days
**Dependencies**: SH module

---

## Development Schedule

```
Phase 1 (Week 1-2): Training Stability
├── Story 1.1: Progressive SH Activation     [2-3d]
├── Story 1.2: Learning Rate Decay           [1-2d]
└── Story 1.3: Opacity Reset                 [1d]

Phase 2 (Week 2-4): Memory Efficiency
├── Story 2.1: Morton Reordering             [3-4d]
├── Story 2.3: Clustered Gaussian            [5-7d]
└── Story 2.2: Sparse Gradient               [4-5d] ← depends on 2.3

Phase 3 (Week 4-5): Quality Enhancement
├── Story 3.1: Low-pass Filter               [1d]
├── Story 3.2: Learnable Extrinsics          [4-5d]
└── Story 3.3: TamingGS Budget               [4-5d]

Phase 4 (Week 5-6): Regularization
├── Story 4.1: Scale Regularization          [1d]
├── Story 4.2: Transmitance Regularization   [1-2d]
└── Story 4.3: Prune Over-scaled             [0.5d]

Phase 5 (Week 6-7): Performance & Compatibility
├── Story 5.1: Early Exit                    [1d]
├── Story 5.2: Fused Kernel                  [3-4d]
├── Story 6.1: COLMAP Loader                 [3-4d]
└── Story 6.2: PLY Format                    [1-2d]
```

---

## Milestones

| Milestone | Stories | Expected Outcome |
|-----------|---------|------------------|
| **M1: Training Convergence** | 1.1, 1.2, 1.3 | PSNR improvement 1-2 dB |
| **M2: Memory Efficiency** | 2.1, 2.2, 2.3 | Training speed +30-50% |
| **M3: LiteGS Parity** | 3.1, 3.2, 3.3, 4.x | PSNR error <0.5 dB vs LiteGS |
| **M4: Full Compatibility** | 5.x, 6.x | COLMAP + 3DGS viewer support |

---

## Technical Debt Notes

### Existing RustGS Advantages (Keep)

- **SH Degree 4**: RustGS supports up to degree 4 (25 coeffs), higher than LiteGS/Gausplat
- **Analytical Backward**: Hand-written chain rule, ~100x faster than finite-diff
- **Adam Fused Kernel**: Metal kernel for optimizer step
- **Chunked Training**: Spatial subdivision for memory-constrained large scenes
- **GaussianState Tracking**: New/Unstable/Stable states for topology control

### Rotation Quaternion Convention

- **LiteGS/RustGS**: w-first `(w, x, y, z)`
- **Gausplat**: w-last `(x, y, z, w)` ⚠️ different

Ensure consistent convention when importing/exporting.

---

## References

- LiteGS Source: `/Users/tfjiang/Projects/RustScan/Mirror/LiteGS/litegs/`
- Gausplat Source: `/Users/tfjiang/Projects/RustScan/Mirror/Gausplat/crates/`
- RustGS Source: `RustGS/src/`
- Original 3DGS Paper: "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)
- Taming 3DGS Paper: "Taming 3DGS: High-Quality Radiance Fields with Limited Resources"