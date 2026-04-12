# Phase 3: Forward Rendering Pipeline

## Context

Phase 1 (scaffolding) and Phase 2 (GPU primitives: radix_sort_u32, prefix_sum_u32) are complete.

This phase implements:
1. `DeviceSplats<B>` GPU tensor struct and `HostSplats ↔ DeviceSplats` bridge (`splats.rs`)
2. The 5-stage forward rendering pipeline (`render/`)
3. All 8 WGSL compute shaders (`shaders/`)

Working directory: `/Users/tfjiang/Projects/RustScan/RustGS`

Reference: `/Users/tfjiang/Projects/brush/crates/brush-render/`  
**Do NOT add brush as a dependency. Understand the algorithm, implement it yourself.**

---

## Part A: DeviceSplats and Splat Bridge (`src/training/wgpu/splats.rs`)

### A.1 DeviceSplats<B>

```rust
use burn::prelude::*;
use burn::module::Param;

/// GPU-resident differentiable Gaussian splat set.
///
/// Packed layout matching brush:
///   transforms [N, 10]: means(0..3) + quats(3..7) + log_scales(7..10)
///   sh_coeffs  [N, K, 3]: K = (sh_degree+1)^2 coefficients per splat
///   raw_opacities [N]: pre-sigmoid opacity
pub struct DeviceSplats<B: Backend> {
    pub transforms:      Param<Tensor<B, 2>>,   // [N, 10]
    pub sh_coeffs:       Param<Tensor<B, 3>>,   // [N, K, 3]
    pub raw_opacities:   Param<Tensor<B, 1>>,   // [N]
    pub sh_degree:       u32,
}

impl<B: Backend> DeviceSplats<B> {
    pub fn num_splats(&self) -> usize {
        self.transforms.val().dims()[0]
    }
}
```

### A.2 HostSplats → DeviceSplats upload

```rust
/// Upload HostSplats (Vec<f32> flat layout) to GPU tensors.
///
/// HostSplats layout (existing code in src/training/state/splats.rs):
///   positions:  [N, 3]
///   rotations:  [N, 4]  (quaternions, wxyz order)
///   log_scales: [N, 3]
///   sh_dc:      [N, 3]  (SH degree-0 coefficient)
///   sh_rest:    [N, (K-1), 3]
///   opacities:  [N]     (raw logit)
pub fn host_splats_to_device<B: Backend>(
    hs: &HostSplats,
    device: &B::Device,
) -> DeviceSplats<B>;
```

Pack positions/rotations/log_scales into a single `[N, 10]` tensor:
- `transforms[:, 0:3]` ← positions
- `transforms[:, 3:7]` ← rotations (quat wxyz)
- `transforms[:, 7:10]` ← log_scales

Concatenate sh_dc and sh_rest into `[N, K, 3]` where K = total SH coeffs.

### A.3 DeviceSplats → HostSplats download

```rust
/// Download GPU tensors back to HostSplats (for checkpointing / export).
pub async fn device_splats_to_host<B: Backend>(
    splats: &DeviceSplats<B>,
) -> HostSplats;
```

Split transforms back into positions/rotations/log_scales.  
Split sh_coeffs into sh_dc (index 0) and sh_rest (indices 1..).

### A.4 Module export

`src/training/wgpu/splats.rs` — no module subdirectory needed, just this one file.

Add to `src/training/wgpu/mod.rs`:
```rust
pub mod splats;
pub use splats::{DeviceSplats, device_splats_to_host, host_splats_to_device};
```

---

## Part B: Shared WGSL Helpers (`src/training/wgpu/shaders/helpers.wgsl`)

This file is `#include`-d by all other shaders via naga_oil `#import`.

### Constants
```wgsl
const TILE_WIDTH: u32 = 16u;
const TILE_SIZE: u32  = 256u;   // TILE_WIDTH * TILE_WIDTH
const SH_C0: f32 = 0.28209479177387814;
```

### Structs

```wgsl
struct ProjectUniforms {
    viewmat:         mat4x4<f32>,   // world→camera (column-major, 64 bytes)
    focal:           vec2<f32>,     // (fx, fy)
    img_size:        vec2<u32>,     // (width, height)
    tile_bounds:     vec2<u32>,     // (nx_tiles, ny_tiles)
    pixel_center:    vec2<f32>,     // (cx, cy) in pixels
    camera_position: vec4<f32>,     // (x, y, z, pad)
    sh_degree:       u32,
    total_splats:    u32,
    pad_a:           u32,
    pad_b:           u32,
}

// 9 f32 per visible splat
struct ProjectedSplat {
    xy_x:    f32,   // projected center x
    xy_y:    f32,   // projected center y
    conic_x: f32,   // inv-cov2d [0][0]
    conic_y: f32,   // inv-cov2d [0][1]
    conic_z: f32,   // inv-cov2d [1][1]
    color_r: f32,   // SH-evaluated R + 0.5
    color_g: f32,   // SH-evaluated G + 0.5
    color_b: f32,   // SH-evaluated B + 0.5
    color_a: f32,   // opacity after compensation
}
```

### Key helper functions
```wgsl
fn sigmoid(x: f32) -> f32
fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32>
fn calc_cov3d(scale: vec3<f32>, quat: vec4<f32>) -> mat3x3<f32>
fn calc_cov2d(cov3d: mat3x3<f32>, mean_c: vec3<f32>,
              focal: vec2<f32>, img_size: vec2<u32>) -> mat2x2<f32>
fn compensate_cov2d(cov2d: ptr<function, mat2x2<f32>>) -> f32  // mip-splatting
fn inverse2x2(m: mat2x2<f32>) -> mat2x2<f32>
fn get_tile_bbox(center: vec2<f32>, extent: vec2<f32>,
                 tile_bounds: vec2<u32>) -> vec4<u32>  // (min_tx, min_ty, max_tx, max_ty)
fn will_primitive_contribute(rect: vec4<u32>, mean2d: vec2<f32>,
                              conic: vec3<f32>, threshold: f32) -> bool
fn num_sh_coeffs(degree: u32) -> u32   // = (degree+1)*(degree+1)
fn sh_coeffs_to_color(degree: u32, viewdir: vec3<f32>, ...) -> vec3<f32>
```

---

## Part C: Stage 1 — ProjectForward

**Files:**
- `src/training/wgpu/shaders/project_forward.wgsl`
- `src/training/wgpu/render/projection.rs`

### C.1 Shader bindings (`@group(0)`)

```wgsl
@binding(0) var<storage, read>       transforms:               array<f32>;         // [N*10]
@binding(1) var<storage, read>       raw_opacities:            array<f32>;         // [N]
@binding(2) var<storage, read_write> global_from_presort_gid:  array<u32>;         // [N]
@binding(3) var<storage, read_write> depths:                   array<f32>;         // [N]
@binding(4) var<storage, read_write> num_visible:              atomic<u32>;
@binding(5) var<storage, read_write> intersect_counts:         array<u32>;         // [N]
@binding(6) var<storage, read_write> num_intersections:        atomic<u32>;
@binding(7) var<uniform>             uniforms:                  ProjectUniforms;
```

### C.2 Workgroup size and dispatch
```wgsl
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>)
```
Dispatch: `ceil(total_splats / 256)` workgroups.

### C.3 Per-splat algorithm
1. Read `transforms[gid*10 .. gid*10+10]` → mean (xyz), quat (wxyz), scale (log).
2. Apply `viewmat` → camera-space mean_c. If `mean_c.z < 0.01`, return.
3. Read `raw_opacities[gid]`, compute `opac = sigmoid(raw_opacities[gid])`.
4. `scale = exp(log_scale)`; compute `cov3d = calc_cov3d(scale, quat)`.
5. `cov2d = calc_cov2d(cov3d, mean_c, focal, img_size)`.
6. `opac *= compensate_cov2d(&cov2d)`. If `opac < 1/255`, return.
7. Project `mean2d = focal * mean_c.xy / mean_c.z + pixel_center`.
8. `conic = inverse2x2(cov2d)`.
9. `power_threshold = log(opac * 255.0)`.
10. `extent = compute_bbox_extent(cov2d, power_threshold)` (how far the gaussian extends in pixels).
11. Frustum check: if bounding box lies entirely outside image, return.
12. `tile_bbox = get_tile_bbox(mean2d, extent, tile_bounds)`.
13. Count how many tiles will_primitive_contribute → `num_tiles_hit`.
14. `atomicAdd(&intersect_counts[gid], num_tiles_hit)`.
15. `atomicAdd(&num_intersections, num_tiles_hit)`.
16. `write_id = atomicAdd(&num_visible, 1u)`.
17. `global_from_presort_gid[write_id] = gid`.
18. `depths[write_id] = mean_c.z`.

### C.4 Rust wrapper (`render/projection.rs`)

```rust
pub struct ProjectForwardOutput<B: Backend> {
    pub global_from_presort_gid: Tensor<B, 1>,  // [N] u32
    pub depths:                  Tensor<B, 1>,  // [N] f32, only [0..num_visible] valid
    pub intersect_counts:        Tensor<B, 1>,  // [N] u32
    pub num_visible_buf:         Tensor<B, 1>,  // [1] u32, atomic result
    pub num_intersections_buf:   Tensor<B, 1>,  // [1] u32, atomic result
}

pub fn project_forward<B: Backend>(
    splats: &DeviceSplats<B>,
    camera: &GaussianCamera,
    img_size: (u32, u32),
    device: &B::Device,
) -> ProjectForwardOutput<B>;
```

---

## Part D: Stage 2 — DepthSort (uses Phase 2)

**File:** `src/training/wgpu/render/sorting.rs`

```rust
/// Sort visible splats by depth (ascending = front-to-back).
/// Returns global_from_compact_gid: the sorted global indices.
pub fn sort_by_depth<B: Backend>(
    depths: Tensor<B, 1>,                  // [num_visible] f32
    global_from_presort_gid: Tensor<B, 1>, // [num_visible] u32
    num_visible: usize,
    device: &B::Device,
) -> Tensor<B, 1>;  // [num_visible] u32, sorted indices
```

Slice both inputs to `[0..num_visible]`, then call `radix_sort_u32` from Phase 2 on the depth keys (reinterpret f32 bits as u32 for radix sort — valid for positive depths). Return the sorted index permutation.

---

## Part E: Stage 3 — ProjectVisible

**Files:**
- `src/training/wgpu/shaders/project_visible.wgsl`
- `src/training/wgpu/render/project_visible.rs`

### E.1 Shader bindings

```wgsl
@binding(0) var<storage, read>       transforms:              array<f32>;             // [N*10]
@binding(1) var<storage, read>       sh_coeffs:               array<f32>;             // [N*K*3]
@binding(2) var<storage, read>       raw_opacities:           array<f32>;             // [N]
@binding(3) var<storage, read>       global_from_compact_gid: array<u32>;             // [num_visible]
@binding(4) var<storage, read_write> projected:               array<ProjectedSplat>;  // [num_visible]
@binding(5) var<uniform>             uniforms:                 ProjectUniforms;
```

### E.2 Workgroup size and dispatch
```wgsl
@compute @workgroup_size(256, 1, 1)
```
Dispatch: `ceil(num_visible / 256)` workgroups.

### E.3 Per-splat algorithm (compact_gid = thread index)
1. Look up `global_gid = global_from_compact_gid[compact_gid]`.
2. Re-read transforms[global_gid] → mean, quat, scale.
3. Re-project mean_c, mean2d, cov2d (same as stage 1).
4. Evaluate SH: `viewdir = normalize(mean - camera_position.xyz)`.
5. Load `num_coeffs = num_sh_coeffs(sh_degree)` coefficients from `sh_coeffs[global_gid * num_coeffs * 3 ...]`.
6. `color = sh_coeffs_to_color(sh_degree, viewdir, coeffs) + vec3(0.5)`.
7. `opac = sigmoid(raw_opacities[global_gid]) * compensate_cov2d(&cov2d)`.
8. `conic = inverse2x2(cov2d)`.
9. Write `ProjectedSplat` to `projected[compact_gid]`.

### E.4 Rust wrapper

```rust
pub fn project_visible<B: Backend>(
    splats: &DeviceSplats<B>,
    global_from_compact_gid: Tensor<B, 1>,   // [num_visible] u32
    num_visible: usize,
    camera: &GaussianCamera,
    img_size: (u32, u32),
    device: &B::Device,
) -> Tensor<B, 2>;   // [num_visible, 9] f32 (ProjectedSplat array)
```

---

## Part F: Stage 4 — TileMapping

**Files:**
- `src/training/wgpu/shaders/map_gaussian_to_intersects.wgsl`
- `src/training/wgpu/render/tile_mapping.rs`

### F.1 Prefix sum step

```rust
// Compute cumulative tile counts per visible splat (for index base calculation)
let cum_tiles_hit = prefix_sum_u32(intersect_counts_compact, device);
// cum_tiles_hit[i] = sum of intersect_counts[0..=i]
```

`intersect_counts_compact` = `intersect_counts` indexed by `global_from_compact_gid`, sliced to `[0..num_visible]`.

### F.2 MapGaussiansToIntersects shader bindings

```wgsl
struct MGIUniforms {
    tile_bounds: vec2<u32>,
    num_visible: u32,
    pad:         u32,
}

@binding(0) var<storage, read>       projected:              array<ProjectedSplat>;  // [num_visible]
@binding(1) var<storage, read>       cum_tiles_hit:          array<u32>;             // [num_visible]
@binding(2) var<storage, read_write> tile_id_from_isect:     array<u32>;             // [num_intersections]
@binding(3) var<storage, read_write> compact_gid_from_isect: array<u32>;             // [num_intersections]
@binding(4) var<uniform>             uniforms:               MGIUniforms;
```

### F.3 Workgroup size and dispatch
```wgsl
@compute @workgroup_size(256, 1, 1)
```
Dispatch: `ceil(num_visible / 256)` workgroups.

### F.4 Per-splat algorithm (compact_gid = thread index)
1. Read `projected[compact_gid]` → mean2d, conic, opac.
2. `power_threshold = log(opac * 255.0)`.
3. Compute 2D extent from conic and power_threshold.
4. `tile_bbox = get_tile_bbox(mean2d, extent, tile_bounds)` → (min_tx, min_ty, max_tx, max_ty).
5. `base_isect_id = cum_tiles_hit[compact_gid - 1]` (0 if compact_gid == 0).
6. Iterate over tiles in bbox. For each tile, call `will_primitive_contribute(...)`.
7. For contributing tiles: write `tile_id` and `compact_gid` into output arrays at `base_isect_id + offset`.

### F.5 Tile sort step (reuse radix_sort_u32 from Phase 2)

```rust
// Sort (tile_id, compact_gid) pairs by tile_id
let bits = u32::BITS - num_tiles.leading_zeros();
let (tile_id_sorted, compact_gid_sorted) = radix_sort_by_key_u32(
    tile_id_from_isect,
    compact_gid_from_isect,
    bits,
);
```

Add `radix_sort_by_key_u32` to `gpu_primitives/radix_sort.rs`: same as `radix_sort_u32` but carries a companion values array through the sort.

### F.6 Rust wrapper

```rust
pub struct TileMappingOutput<B: Backend> {
    pub tile_id_from_isect:     Tensor<B, 1>,   // [num_intersections] u32, sorted
    pub compact_gid_from_isect: Tensor<B, 1>,   // [num_intersections] u32
}

pub fn tile_mapping<B: Backend>(
    projected_splats:        Tensor<B, 2>,   // [num_visible, 9]
    intersect_counts:        Tensor<B, 1>,   // [num_visible] u32
    num_intersections:       usize,
    num_tiles:               u32,
    tile_bounds:             (u32, u32),
    device:                  &B::Device,
) -> TileMappingOutput<B>;
```

---

## Part G: Stage 5 — GetTileOffsets

**Files:**
- `src/training/wgpu/shaders/get_tile_offsets.wgsl`
- `src/training/wgpu/render/tile_mapping.rs` (add function here)

### G.1 Shader bindings

```wgsl
@binding(0) var<storage, read>       tile_id_from_isect: array<u32>;   // [num_intersections]
@binding(1) var<storage, read_write> tile_offsets:       array<u32>;   // [ny*nx*2]

struct Uniforms { num_intersections: u32, num_tiles: u32, pad: vec2<u32> }
@binding(2) var<uniform> uniforms: Uniforms;
```

### G.2 Algorithm
For each intersection `i`:
- If `i == 0` or `tile_id[i] != tile_id[i-1]`: write `tile_offsets[tile_id[i]*2] = i` (start of this tile).
- If `i == num_intersections-1` or `tile_id[i] != tile_id[i+1]`: write `tile_offsets[tile_id[i]*2+1] = i+1` (end).

### G.3 Output
```rust
pub fn get_tile_offsets<B: Backend>(
    tile_id_from_isect: Tensor<B, 1>,   // [num_intersections] u32, sorted
    num_intersections: usize,
    tile_bounds: (u32, u32),
    device: &B::Device,
) -> Tensor<B, 3>;   // [ny_tiles, nx_tiles, 2] u32: (start, end) per tile
```

---

## Part H: Stage 6 (final) — Rasterize

**Files:**
- `src/training/wgpu/shaders/rasterize.wgsl`
- `src/training/wgpu/render/rasterize.rs`

### H.1 Shader bindings

```wgsl
@binding(0) var<storage, read>       compact_gid_from_isect:   array<u32>;            // [num_intersections]
@binding(1) var<storage, read>       tile_offsets:             array<u32>;            // [ny*nx*2]
@binding(2) var<storage, read>       projected:                array<ProjectedSplat>; // [num_visible]
@binding(3) var<storage, read_write> out_img:                  array<f32>;            // [H*W*4]
@binding(4) var<storage, read>       global_from_compact_gid:  array<u32>;            // [num_visible]
@binding(5) var<storage, read_write> visible:                  array<f32>;            // [total_splats]
@binding(6) var<uniform>             uniforms:                  RasterizeUniforms;

struct RasterizeUniforms {
    tile_bounds: vec2<u32>,
    img_size:    vec2<u32>,
    background:  vec4<f32>,
}
```

### H.2 Workgroup config
```wgsl
var<workgroup> range_uniform:  vec2<u32>;
var<workgroup> local_batch:    array<ProjectedSplat, 256>;
var<workgroup> load_gid:       array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
```
One workgroup = one 16×16 tile.  
Dispatch: `num_tiles` workgroups.

### H.3 Per-pixel alpha blending algorithm
1. Determine which tile this workgroup handles: `tile_id = workgroup_id`.
2. Map local thread index to pixel: `pix = (tile_x*16 + local%16, tile_y*16 + local/16)`.
3. Load `range = tile_offsets[tile_id*2 .. tile_id*2+2]`.
4. Loop over intersections `[range.x .. range.y]` in batches of 256:
   - Load batch of `ProjectedSplat` into workgroup shared memory.
   - `workgroupBarrier()`.
   - For each splat in batch (inner loop over 256 threads):
     - `delta = splat.xy - pixel_coord`.
     - `sigma = 0.5*(conic.x*delta.x² + 2*conic.y*delta.x*delta.y + conic.z*delta.y²)`.
     - `alpha = min(0.999, splat.color_a * exp(-sigma))`.
     - If `sigma >= 0 && alpha >= 1/255`:
       - `pix_out += alpha * T * max(splat.color_rgb, 0)`.
       - `T *= (1 - alpha)`.
       - If `T < 1e-4`: stop (fully opaque).
       - Mark `visible[global_from_compact_gid[compact_gid]] = 1.0`.
5. Final pixel color: `pix_out + T * background.rgb`, alpha = `1 - T`.
6. Write `out_img[(pix.y * img_size.x + pix.x) * 4 .. +4]`.

### H.4 Rust wrapper

```rust
pub struct RasterizeOutput<B: Backend> {
    pub out_img: Tensor<B, 3>,   // [H, W, 4] f32 RGBA
    pub visible: Tensor<B, 1>,   // [total_splats] f32 (1.0 if contributed)
}

pub fn rasterize<B: Backend>(
    compact_gid_from_isect:  Tensor<B, 1>,   // [num_intersections] u32
    tile_offsets:            Tensor<B, 3>,   // [ny, nx, 2] u32
    projected_splats:        Tensor<B, 2>,   // [num_visible, 9] f32
    global_from_compact_gid: Tensor<B, 1>,   // [num_visible] u32
    total_splats:            usize,
    img_size:                (u32, u32),
    tile_bounds:             (u32, u32),
    background:              [f32; 3],
    device:                  &B::Device,
) -> RasterizeOutput<B>;
```

---

## Part I: Pipeline Orchestration (`src/training/wgpu/render/mod.rs`)

### I.1 RenderOutput

```rust
pub struct RenderOutput<B: Backend> {
    pub out_img:                 Tensor<B, 3>,   // [H, W, 4] f32
    pub visible:                 Tensor<B, 1>,   // [total_splats] f32
    pub projected_splats:        Tensor<B, 2>,   // [num_visible, 9]
    pub global_from_compact_gid: Tensor<B, 1>,   // [num_visible] u32
    pub compact_gid_from_isect:  Tensor<B, 1>,   // [num_intersections] u32
    pub tile_offsets:            Tensor<B, 3>,   // [ny, nx, 2] u32
    pub num_visible:             usize,
    pub num_intersections:       usize,
}
```

### I.2 Main render function

```rust
pub async fn render_forward<B: Backend>(
    splats: &DeviceSplats<B>,
    camera: &GaussianCamera,
    img_size: (u32, u32),      // (width, height)
    background: [f32; 3],
    device: &B::Device,
) -> RenderOutput<B> {
    // Stage 1: project all splats, get visibility + depth
    let proj_out = project_forward(splats, camera, img_size, device);

    // Read atomic counters from GPU
    let num_visible = read_u32_async(&proj_out.num_visible_buf).await;
    let num_intersections = read_u32_async(&proj_out.num_intersections_buf).await;

    // Stage 2: sort visible splats by depth
    let global_from_compact_gid = sort_by_depth(
        proj_out.depths.clone(),
        proj_out.global_from_presort_gid,
        num_visible,
        device,
    );

    // Stage 3: project visible splats (full SH + covariance)
    let projected_splats = project_visible(
        splats,
        global_from_compact_gid.clone(),
        num_visible,
        camera,
        img_size,
        device,
    );

    // Stage 4: tile mapping (prefix sum + map + tile sort)
    let tile_bounds = (
        (img_size.0 + 15) / 16,
        (img_size.1 + 15) / 16,
    );
    let num_tiles = tile_bounds.0 * tile_bounds.1;
    let tile_out = tile_mapping(
        projected_splats.clone(),
        proj_out.intersect_counts,
        num_intersections,
        num_tiles,
        tile_bounds,
        device,
    );

    // Stage 5: get tile offsets
    let tile_offsets = get_tile_offsets(
        tile_out.tile_id_from_isect,
        num_intersections,
        tile_bounds,
        device,
    );

    // Stage 6: rasterize
    let raster_out = rasterize(
        tile_out.compact_gid_from_isect.clone(),
        tile_offsets.clone(),
        projected_splats.clone(),
        global_from_compact_gid.clone(),
        splats.num_splats(),
        img_size,
        tile_bounds,
        background,
        device,
    );

    RenderOutput {
        out_img: raster_out.out_img,
        visible: raster_out.visible,
        projected_splats,
        global_from_compact_gid,
        compact_gid_from_isect: tile_out.compact_gid_from_isect,
        tile_offsets,
        num_visible,
        num_intersections,
    }
}
```

---

## Part J: GaussianCamera type (`src/core/camera.rs` — check if already exists)

Check if `GaussianCamera` already exists in `src/core/`. If yes, reuse it.  
If not, define:

```rust
pub struct GaussianCamera {
    pub width:       u32,
    pub height:      u32,
    pub fx:          f32,
    pub fy:          f32,
    pub cx:          f32,
    pub cy:          f32,
    pub view_matrix: glam::Mat4,  // world-to-camera, column-major
    pub position:    glam::Vec3,  // world-space camera position
}
```

Helper: `viewmat_from_pose(rotation: glam::Quat, translation: glam::Vec3) -> glam::Mat4`

---

## Part K: WGSL Shader Loading

All WGSL shaders should be embedded at compile time:

```rust
// In each Rust module:
const SHADER_SRC: &str = include_str!("../shaders/project_forward.wgsl");
```

Use `naga_oil`'s `Composer` to resolve `#import helpers` between shaders:
```rust
use naga_oil::compose::{Composer, ComposableModuleDescriptor, NagaModuleDescriptor};

fn build_shader_module(composer: &mut Composer, source: &str) -> naga::Module { ... }
```

---

## Files to Create

```
src/training/wgpu/splats.rs
src/training/wgpu/render/mod.rs           (orchestration + RenderOutput)
src/training/wgpu/render/projection.rs
src/training/wgpu/render/sorting.rs
src/training/wgpu/render/project_visible.rs
src/training/wgpu/render/tile_mapping.rs  (stages 4 + 5)
src/training/wgpu/render/rasterize.rs
src/training/wgpu/shaders/helpers.wgsl
src/training/wgpu/shaders/project_forward.wgsl
src/training/wgpu/shaders/project_visible.wgsl
src/training/wgpu/shaders/map_gaussian_to_intersects.wgsl
src/training/wgpu/shaders/get_tile_offsets.wgsl
src/training/wgpu/shaders/rasterize.wgsl
```

Also update:
- `src/training/wgpu/mod.rs`: add `pub mod splats;`
- `src/training/wgpu/gpu_primitives/radix_sort.rs`: add `radix_sort_by_key_u32`

---

## Verification

```bash
cd /Users/tfjiang/Projects/RustScan/RustGS
cargo check --features gpu-wgpu
```

No unit tests required for this phase — correctness is verified in Phase 7 via end-to-end PSNR comparison.
