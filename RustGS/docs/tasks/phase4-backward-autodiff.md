# Phase 4: Backward Pass + Burn Autodiff Integration

## Context

Phases 1-3 are complete. We now have:
- `DeviceSplats<B>` and forward render pipeline in `src/training/wgpu/render/`
- `RenderOutput<B>` with: `out_img`, `visible`, `projected_splats`,
  `global_from_compact_gid`, `compact_gid_from_isect`, `tile_offsets`,
  `num_visible`, `num_intersections`

This phase implements:
1. Two backward WGSL compute shaders (`rasterize_backwards.wgsl`, `project_backwards.wgsl`)
2. Rust wrappers for each backward kernel (`render_bwd/`)
3. Burn autodiff custom op registration (`render_bwd/autodiff.rs`)
4. A `render_splats()` entry point that returns a differentiable image tensor

Working directory: `/Users/tfjiang/Projects/RustScan/RustGS`

Reference: `/Users/tfjiang/Projects/brush/crates/brush-render-bwd/`
**Do NOT add brush as a dependency.**

---

## Key Data Structures (from Phase 3)

```rust
// In src/training/wgpu/splats.rs
pub struct DeviceSplats<B: Backend> {
    pub transforms:    Param<Tensor<B, 2>>,  // [N, 10]: means(3)+quats(4)+log_scales(3)
    pub sh_coeffs:     Param<Tensor<B, 3>>,  // [N, K, 3]
    pub raw_opacities: Param<Tensor<B, 1>>,  // [N]
    pub sh_degree:     u32,
}

// In src/training/wgpu/render/mod.rs
pub struct RenderOutput<B: Backend> {
    pub out_img:                 Tensor<B, 3>,  // [H, W, 4] f32
    pub visible:                 Tensor<B, 1>,  // [total_splats] f32
    pub projected_splats:        Tensor<B, 2>,  // [num_visible, 9] f32
    pub global_from_compact_gid: Tensor<B, 1>,  // [num_visible] u32
    pub compact_gid_from_isect:  Tensor<B, 1>,  // [num_intersections] u32
    pub tile_offsets:            Tensor<B, 3>,  // [ny_tiles, nx_tiles, 2] u32
    pub num_visible:             usize,
    pub num_intersections:       usize,
}
```

---

## Part A: Backward Shader 1 — Rasterize Backwards

**Files:**
- `src/training/wgpu/shaders/rasterize_backwards.wgsl`
- `src/training/wgpu/render_bwd/rasterize_bwd.rs`

### A.1 WGSL Bindings

```wgsl
// imports helpers from helpers.wgsl via naga_oil #import
@group(0) @binding(0) var<storage, read>       compact_gid_from_isect: array<u32>;
@group(0) @binding(1) var<storage, read>       tile_offsets:           array<u32>;  // [ny*nx*2]
@group(0) @binding(2) var<storage, read>       projected:              array<ProjectedSplat>;
@group(0) @binding(3) var<storage, read>       out_img:                array<f32>;  // [H*W*4]
@group(0) @binding(4) var<storage, read>       v_output:               array<f32>;  // [H*W*4]
@group(0) @binding(5) var<storage, read_write> v_splats:               array<atomic<u32>>;  // [num_visible*10]
@group(0) @binding(6) var<uniform>             uniforms:               RasterizeBwdUniforms;

struct RasterizeBwdUniforms {
    tile_bounds: vec2<u32>,  // (nx_tiles, ny_tiles)
    img_size:    vec2<u32>,  // (width, height)
    background:  vec4<f32>,
}
```

Note: `v_splats` uses `atomic<u32>` (bit-cast trick for float atomics on WebGPU/WGSL).

### A.2 Workgroup and Dispatch

```wgsl
const THREADS_PER_WG: u32 = 256u;

@compute @workgroup_size(THREADS_PER_WG, 1, 1)
fn main(
    @builtin(workgroup_id)          wg_id:    vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
)
```

Dispatch: one workgroup per tile.  
`dispatch_count = num_tiles = tile_bounds.x * tile_bounds.y`

Each thread handles exactly one pixel within its 16×16 tile.  
`local_idx → pixel_x = local_idx % 16, pixel_y = local_idx / 16`

### A.3 Algorithm

**Per-tile (workgroup) setup:**
```wgsl
let tile_id = wg_id.x + wg_id.y * num_workgroups.x;
let tile_x  = tile_id % uniforms.tile_bounds.x;
let tile_y  = tile_id / uniforms.tile_bounds.x;
let pix_x   = tile_x * 16u + local_idx % 16u;
let pix_y   = tile_y * 16u + local_idx / 16u;
let inside  = pix_x < uniforms.img_size.x && pix_y < uniforms.img_size.y;
let pix_id  = pix_x + pix_y * uniforms.img_size.x;

let range_start = tile_offsets[tile_id * 2u];
let range_end   = tile_offsets[tile_id * 2u + 1u];
```

**Initialize pixel state from forward output:**
```wgsl
var T = 1.0;          // accumulated transmittance (initialized to 1.0)
var v_rgb_acc = vec3(0.0);
var v_alpha_local: f32;

// Load dL/d(out_img) for this pixel
let v_out_rgba = vec4(v_output[pix_id*4], v_output[pix_id*4+1],
                      v_output[pix_id*4+2], v_output[pix_id*4+3]);

// Forward output tells us final alpha = 1 - T_final
let final_alpha  = out_img[pix_id*4 + 3];
let T_final      = 1.0 - final_alpha;
```

**Reverse traversal over intersections (back to front):**
```wgsl
// IMPORTANT: traverse in REVERSE order (back-to-front → contributions)
var done = !inside;

// We need to recover T at each step. Two approaches:
// Option A (brush approach): do forward pass first to get T_final,
//   then reverse pass reads T from forward stored state.
// Option B (simpler): store T per-pixel in forward in a buffer.
//
// Use Option A: store nothing extra. In reverse pass, iterate forward
//   first in shared memory to reconstruct T at the stopping point.
//
// Simpler implementation: traverse the range forward collecting
// the list, then process in reverse. For a single tile this is fine.
```

**Core per-Gaussian backward computation:**
For each Gaussian `g` in the tile range (traversed in the same order as forward but accumulating backwards):

```wgsl
let splat = projected[compact_gid];
let delta = vec2(splat.xy_x, splat.xy_y) - pixel_coord;
let conic = vec3(splat.conic_x, splat.conic_y, splat.conic_z);

let sigma = 0.5 * (conic.x * delta.x * delta.x
           + 2.0 * conic.y * delta.x * delta.y
           + conic.z * delta.y * delta.y);
let gaussian = exp(-sigma);
let alpha = min(0.999, splat.color_a * gaussian);

if sigma < 0.0 || alpha < 1.0 / 255.0 { continue; }

let next_T = T * (1.0 - alpha);
let vis    = alpha * T;

// Chain rule:
// C_out = C_prev + vis * color_rgb
// A_out = 1 - T = A_prev + vis (where vis = alpha * T)

// dL/d(color_rgb) = vis * dL/d(C_out_rgb) (only where color > 0)
let v_rgb_local = vis * v_out_rgba.rgb;  // clamped in forward

// dL/d(alpha):
// Forward: C_prev_new = C_prev + vis * color
//          T_new = T * (1 - alpha)
// Backward: dL/d(alpha) = dot(color_rgb, dL/d(C_out)) * T
//                       - dL/d(T) ... (chain through T accumulator)
// Simplified (matching brush):
let ra = 1.0 / (1.0 - alpha);
// We need current accumulated color from forward. Use the fact that
// rgb_pixel_finals = out_img.rgb - T_final * background.rgb
// and at this gaussian: rgb_accumulated_after = running_acc
let v_alpha = dot(vec3(splat.color_r, splat.color_g, splat.color_b),
                  v_out_rgba.rgb) * T
            + v_out_rgba.a * ra;

// dL/d(sigma) = -alpha * dL/d(alpha)
let v_sigma = -alpha * v_alpha;

// dL/d(xy):
// sigma = 0.5*(cx*dx^2 + 2*cy*dx*dy + cz*dy^2)
// d(sigma)/d(xy) = [cx*dx + cy*dy, cy*dx + cz*dy]
let v_xy = v_sigma * vec2(conic.x * delta.x + conic.y * delta.y,
                          conic.y * delta.x + conic.z * delta.y);

// dL/d(conic):
// sigma = 0.5*(c00*dx^2 + 2*c01*dx*dy + c11*dy^2)
let v_conic = vec3(
    0.5 * v_sigma * delta.x * delta.x,  // d/d(conic.x)
    v_sigma * delta.x * delta.y,         // d/d(conic.y)
    0.5 * v_sigma * delta.y * delta.y,  // d/d(conic.z)
);

// dL/d(opacity_after_activation):
// alpha = opac * gaussian  →  d(alpha)/d(opac) = gaussian
let v_color_a = gaussian * v_alpha;

// Accumulate T for next (previous) step
T = next_T;

// === Atomic-add to v_splats[compact_gid, :] ===
let base = compact_gid * 10u;
atomic_add_float(&v_splats[base + 0u], v_xy.x);
atomic_add_float(&v_splats[base + 1u], v_xy.y);
atomic_add_float(&v_splats[base + 2u], v_conic.x);
atomic_add_float(&v_splats[base + 3u], v_conic.y);
atomic_add_float(&v_splats[base + 4u], v_conic.z);
atomic_add_float(&v_splats[base + 5u], v_rgb_local.x);
atomic_add_float(&v_splats[base + 6u], v_rgb_local.y);
atomic_add_float(&v_splats[base + 7u], v_rgb_local.z);
atomic_add_float(&v_splats[base + 8u], v_color_a);
```

**Float atomic helper (WebGPU CAS loop):**
```wgsl
fn atomic_add_float(ptr: ptr<storage, atomic<u32>, read_write>, val: f32) {
    var old = atomicLoad(ptr);
    loop {
        let new_val = bitcast<u32>(bitcast<f32>(old) + val);
        let result = atomicCompareExchangeWeak(ptr, old, new_val);
        if result.exchanged { break; }
        old = result.old_value;
    }
}
```

### A.4 Output

`v_splats: [num_visible, 10]` where each row stores:
```
[0]: dL/d(xy.x)
[1]: dL/d(xy.y)
[2]: dL/d(conic.x)
[3]: dL/d(conic.y)
[4]: dL/d(conic.z)
[5]: dL/d(color_r)
[6]: dL/d(color_g)
[7]: dL/d(color_b)
[8]: dL/d(color_a)   ← opacity-after-activation gradient
[9]: 0.0             ← unused (reserved for refine_weight)
```

### A.5 Rust wrapper (`render_bwd/rasterize_bwd.rs`)

```rust
/// Run rasterize backward kernel.
///
/// Returns v_splats: [num_visible, 10] f32 sparse gradient buffer.
pub fn rasterize_bwd<B: Backend>(
    compact_gid_from_isect: Tensor<B, 1>,  // [num_intersections] u32
    tile_offsets:           Tensor<B, 3>,  // [ny, nx, 2] u32
    projected_splats:       Tensor<B, 2>,  // [num_visible, 9] f32
    out_img:                Tensor<B, 3>,  // [H, W, 4] f32 (forward output)
    v_output:               Tensor<B, 3>,  // [H, W, 4] f32 (dL/d(out_img))
    num_visible:            usize,
    img_size:               (u32, u32),
    tile_bounds:            (u32, u32),
    background:             [f32; 3],
    device:                 &B::Device,
) -> Tensor<B, 2>;  // [num_visible, 10] f32
```

---

## Part B: Backward Shader 2 — Project Backwards

**Files:**
- `src/training/wgpu/shaders/project_backwards.wgsl`
- `src/training/wgpu/render_bwd/project_bwd.rs`

### B.1 WGSL Bindings

```wgsl
@group(0) @binding(0) var<storage, read>       transforms:              array<f32>;  // [N*10]
@group(0) @binding(1) var<storage, read>       sh_coeffs:               array<f32>;  // [N*K*3]
@group(0) @binding(2) var<storage, read>       raw_opacities:           array<f32>;  // [N]
@group(0) @binding(3) var<storage, read>       global_from_compact_gid: array<u32>;  // [num_visible]
@group(0) @binding(4) var<storage, read>       v_splats:                array<f32>;  // [num_visible*10]
@group(0) @binding(5) var<storage, read_write> v_transforms:            array<f32>;  // [N*10]
@group(0) @binding(6) var<storage, read_write> v_sh_coeffs:             array<f32>;  // [N*K*3]
@group(0) @binding(7) var<storage, read_write> v_raw_opacities:         array<f32>;  // [N]
@group(0) @binding(8) var<uniform>             uniforms:                ProjectUniforms;  // reuse from Phase 3
```

### B.2 Workgroup and Dispatch

```wgsl
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>)
```

Dispatch: `ceil(num_visible / 256)` workgroups.  
Each thread handles one `compact_gid` → looks up `global_gid`.

### B.3 Algorithm

**Step 1: Load sparse gradients**
```wgsl
let compact_gid = gid.x;
if compact_gid >= uniforms.num_visible { return; }

let global_gid = global_from_compact_gid[compact_gid];
let rg_base = compact_gid * 10u;

let v_xy     = vec2(v_splats[rg_base + 0u], v_splats[rg_base + 1u]);
let v_conic  = vec3(v_splats[rg_base + 2u], v_splats[rg_base + 3u], v_splats[rg_base + 4u]);
let v_color  = vec3(v_splats[rg_base + 5u], v_splats[rg_base + 6u], v_splats[rg_base + 7u]);
let v_color_a = v_splats[rg_base + 8u];
```

**Step 2: Re-compute forward quantities (same as project_visible)**
```wgsl
// Re-read transforms
let t_base  = global_gid * 10u;
let mean    = vec3(transforms[t_base], transforms[t_base+1u], transforms[t_base+2u]);
let quat    = normalize(vec4(transforms[t_base+3u], transforms[t_base+4u],
                             transforms[t_base+5u], transforms[t_base+6u]));
let log_s   = vec3(transforms[t_base+7u], transforms[t_base+8u], transforms[t_base+9u]);
let scale   = exp(log_s);

let viewmat = uniforms.viewmat;
let R       = mat3x3(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);
let mean_c  = R * mean + viewmat[3].xyz;

let rotmat  = quat_to_mat3(quat);
let M       = rotmat * scale_to_mat3(scale);
let cov3d   = M * transpose(M);
let cov_cam = R * cov3d * transpose(R);
let J       = calc_cam_J(mean_c, uniforms.focal, uniforms.img_size, uniforms.pixel_center);
let cov2d   = mat2x2(
    J[0].xy,
    J[1].xy,
) * cov_cam.xyz.xy... // full 2D projection (see Phase 3 helpers)
```

**Step 3: dL/d(cov2d) via conic inverse**
```wgsl
// conic = inverse(cov2d)
// d(inverse(A))/dA = -A^{-T} * dL/dConic * A^{-T}
let conic_mat = mat2x2(
    vec2(v_conic.x, v_conic.y * 0.5),
    vec2(v_conic.y * 0.5, v_conic.z),
);
let cov2d_inv = inverse2x2(cov2d);
let v_cov2d = -(transpose(cov2d_inv) * conic_mat * transpose(cov2d_inv));
// v_cov2d is mat2x2
```

**Step 4: dL/d(mean_c) via perspective projection**
```wgsl
// mean2d = focal * mean_c.xy / mean_c.z + pixel_center
// d(mean2d)/d(mean_c) forms the Jacobian J
let rz  = 1.0 / mean_c.z;
let rz2 = rz * rz;

// dL/d(mean_c) from dL/d(mean2d):
var v_mean_c = vec3(
    uniforms.focal.x * rz * v_xy.x,
    uniforms.focal.y * rz * v_xy.y,
    -(uniforms.focal.x * mean_c.x * v_xy.x
    + uniforms.focal.y * mean_c.y * v_xy.y) * rz2,
);

// dL/d(mean_c) from dL/d(cov2d) via J:
// cov2d = J * cov_cam * J^T  →  dL/dJ = dL/dcov2d * J * cov_cam^T + ...
// dL/d(mean_c) via d(J)/d(mean_c) (perspective Jacobian chain rule)
// See brush project_backwards.wgsl persp_proj_vjp for full derivation.
// Key terms:
let v_J_from_cov2d = v_cov2d * (mat3x2(J) * cov_cam) + transpose(v_cov2d) * (mat3x2(J) * cov_cam);
// propagate v_J through the perspective formula to get more v_mean_c contributions
// (omitting clamping details — see brush reference lines ~216-285)
```

**Step 5: dL/d(cov3d) via camera rotation**
```wgsl
// cov_cam = R * cov3d * R^T
// dL/d(cov3d) = R^T * dL/d(cov_cam) * R
let v_cov_cam = transpose(mat3x2(J)) * v_cov2d * mat3x2(J);
let v_cov3d = transpose(R) * mat3x3(v_cov_cam) * R;
```

**Step 6: dL/d(M) via cov3d = M * M^T**
```wgsl
// cov3d = M * M^T
// dL/dM = (dL/d(cov3d) + dL/d(cov3d)^T) * M
let v_M = (v_cov3d + transpose(v_cov3d)) * M;
```

**Step 7: dL/d(scale) and dL/d(quat)**
```wgsl
// M = rotmat * diag(scale)
// dL/d(scale) = diag(rotmat^T * v_M)
let v_scale = vec3(
    dot(rotmat[0], v_M[0]),
    dot(rotmat[1], v_M[1]),
    dot(rotmat[2], v_M[2]),
);
// Log-space: dL/d(log_scale) = scale * dL/d(scale)
let v_log_scale = v_scale * scale;

// dL/d(rotmat) = v_M * diag(scale)^T = v_M * diag(scale)
// dL/d(quat) = quat_to_mat_vjp(quat, v_M * scale_to_mat3(scale))
let v_rotmat = v_M * scale_to_mat3(scale);
let v_quat_unnorm = quat_to_mat3_vjp(quat, v_rotmat);
// Account for normalization: d(normalize(q))/dq
let v_quat = normalize_vjp(quat) * v_quat_unnorm;
```

**Step 8: dL/d(mean_world)**
```wgsl
// mean_c = R * mean + t  →  dL/d(mean) = R^T * dL/d(mean_c)
let v_mean = transpose(R) * v_mean_c;
```

**Step 9: dL/d(sh_coeffs)**
```wgsl
let viewdir = normalize(mean - uniforms.camera_position.xyz);
let num_coeffs = num_sh_coeffs(uniforms.sh_degree);

// VJP of SH evaluation: apply chain rule for each basis function
// sh_coeffs_to_color_vjp(degree, viewdir, v_color) → v_coeffs per basis
let v_coeff = sh_to_color_vjp(uniforms.sh_degree, viewdir, v_color);

// Write to dense v_sh_coeffs
let sh_base = global_gid * num_coeffs * 3u;
// ... write v_coeff elements for each SH basis
```

**Step 10: dL/d(raw_opacities)**
```wgsl
// Forward: opac_activated = sigmoid(raw_opac) * compensate_cov2d(&cov2d)
// Backward:
let filter_comp = compensate_cov2d(&cov2d);
let opac = sigmoid(raw_opacities[global_gid]);
// d(sigmoid)/d(x) = sigmoid * (1 - sigmoid)
v_raw_opacities[global_gid] = filter_comp * v_color_a * opac * (1.0 - opac);
```

**Step 11: Write dense gradients**
```wgsl
let vt_base = global_gid * 10u;
v_transforms[vt_base + 0u] = v_mean.x;
v_transforms[vt_base + 1u] = v_mean.y;
v_transforms[vt_base + 2u] = v_mean.z;
v_transforms[vt_base + 3u] = v_quat.x;
v_transforms[vt_base + 4u] = v_quat.y;
v_transforms[vt_base + 5u] = v_quat.z;
v_transforms[vt_base + 6u] = v_quat.w;
v_transforms[vt_base + 7u] = v_log_scale.x;
v_transforms[vt_base + 8u] = v_log_scale.y;
v_transforms[vt_base + 9u] = v_log_scale.z;
```

### B.4 Rust wrapper (`render_bwd/project_bwd.rs`)

```rust
pub struct ProjectBwdOutput<B: Backend> {
    pub v_transforms:    Tensor<B, 2>,  // [N, 10] f32
    pub v_sh_coeffs:     Tensor<B, 3>,  // [N, K, 3] f32
    pub v_raw_opacities: Tensor<B, 1>,  // [N] f32
}

pub fn project_bwd<B: Backend>(
    splats:                  &DeviceSplats<B>,
    global_from_compact_gid: Tensor<B, 1>,  // [num_visible] u32
    v_splats:                Tensor<B, 2>,  // [num_visible, 10] f32
    camera:                  &GaussianCamera,
    img_size:                (u32, u32),
    num_visible:             usize,
    device:                  &B::Device,
) -> ProjectBwdOutput<B>;
```

---

## Part C: Burn Autodiff Integration (`src/training/wgpu/render_bwd/autodiff.rs`)

### C.1 Checkpoint State

```rust
/// State saved during forward pass, used in backward.
pub struct RenderCheckpoint<B: Backend> {
    // Inputs (needed to re-derive quantities)
    pub transforms:              Tensor<B, 2>,  // [N, 10]
    pub sh_coeffs:               Tensor<B, 3>,  // [N, K, 3]
    pub raw_opacities:           Tensor<B, 1>,  // [N]
    pub sh_degree:               u32,
    pub camera:                  GaussianCamera,
    pub img_size:                (u32, u32),
    pub background:              [f32; 3],

    // Forward intermediates
    pub out_img:                 Tensor<B, 3>,  // [H, W, 4]
    pub projected_splats:        Tensor<B, 2>,  // [num_visible, 9]
    pub global_from_compact_gid: Tensor<B, 1>,  // [num_visible]
    pub compact_gid_from_isect:  Tensor<B, 1>,  // [num_intersections]
    pub tile_offsets:            Tensor<B, 3>,  // [ny, nx, 2]
    pub num_visible:             usize,
    pub num_intersections:       usize,
}
```

### C.2 Backward Op Registration

```rust
use burn::backend::autodiff::{
    checkpoint::strategy::CheckpointStrategy,
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};

#[derive(Debug)]
struct RenderBackward;

// 3 differentiable inputs:
//   [0] transforms
//   [1] sh_coeffs
//   [2] raw_opacities
impl<B: Backend> Backward<B, 3> for RenderBackward {
    type State = RenderCheckpoint<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
        _checkpointer: &mut burn::backend::autodiff::checkpoint::Checkpointer,
    ) {
        let state = ops.state;
        let [transforms_node, sh_coeffs_node, raw_opacities_node] = ops.parents;

        // Consume upstream gradient dL/d(out_img)
        let v_output = grads.consume::<B, 3>(&ops.node);

        // Stage 1: rasterize backward
        let v_splats = rasterize_bwd::<B>(
            state.compact_gid_from_isect,
            state.tile_offsets,
            state.projected_splats.clone(),
            state.out_img,
            v_output,
            state.num_visible,
            state.img_size,
            tile_bounds(state.img_size),
            state.background,
            &device,
        );

        // Stage 2: project backward
        let splats_inner = DeviceSplats::<B> {
            transforms:    Param::uninitialized(state.transforms),
            sh_coeffs:     Param::uninitialized(state.sh_coeffs),
            raw_opacities: Param::uninitialized(state.raw_opacities),
            sh_degree:     state.sh_degree,
        };

        let bwd = project_bwd::<B>(
            &splats_inner,
            state.global_from_compact_gid,
            v_splats,
            &state.camera,
            state.img_size,
            state.num_visible,
            &device,
        );

        // Stage 3: register gradients into autodiff graph
        if let Some(node) = transforms_node {
            grads.register::<B, 2>(node.id, bwd.v_transforms);
        }
        if let Some(node) = sh_coeffs_node {
            grads.register::<B, 3>(node.id, bwd.v_sh_coeffs);
        }
        if let Some(node) = raw_opacities_node {
            grads.register::<B, 1>(node.id, bwd.v_raw_opacities);
        }
    }
}
```

### C.3 Differentiable Render Entry Point

```rust
/// Render with autodiff tracking. Returns a differentiable image tensor.
///
/// Call `.backward()` on the loss to propagate gradients to
/// splats.transforms, splats.sh_coeffs, and splats.raw_opacities.
pub async fn render_splats<B: AutodiffBackend>(
    splats:     &DeviceSplats<B>,
    camera:     &GaussianCamera,
    img_size:   (u32, u32),
    background: [f32; 3],
) -> Tensor<B, 3>  // [H, W, 4] differentiable RGBA
{
    // Extract inner backend tensors
    let transforms_inner    = splats.transforms.val().inner();
    let sh_coeffs_inner     = splats.sh_coeffs.val().inner();
    let raw_opacities_inner = splats.raw_opacities.val().inner();
    let device              = splats.transforms.val().device();

    // Forward pass on inner backend (no autodiff overhead)
    let fwd_out = render_forward::<B::InnerBackend>(
        // construct inner DeviceSplats from inner tensors
        &inner_splats,
        camera,
        img_size,
        background,
        &device.inner(),
    ).await;

    // Prepare autodiff nodes for the 3 differentiable inputs
    let prep = RenderBackward
        .prepare::<B::CheckpointStrategy>([
            splats.transforms.val().into_primitive().node,
            splats.sh_coeffs.val().into_primitive().node,
            splats.raw_opacities.val().into_primitive().node,
        ])
        .compute_bound()
        .stateful();

    match prep {
        OpsKind::Tracked(mut prep) => {
            let state = RenderCheckpoint {
                transforms:              transforms_inner,
                sh_coeffs:               sh_coeffs_inner,
                raw_opacities:           raw_opacities_inner,
                sh_degree:               splats.sh_degree,
                camera:                  camera.clone(),
                img_size,
                background,
                out_img:                 fwd_out.out_img.clone(),
                projected_splats:        fwd_out.projected_splats,
                global_from_compact_gid: fwd_out.global_from_compact_gid,
                compact_gid_from_isect:  fwd_out.compact_gid_from_isect,
                tile_offsets:            fwd_out.tile_offsets,
                num_visible:             fwd_out.num_visible,
                num_intersections:       fwd_out.num_intersections,
            };

            // Wrap forward output with autodiff tracking
            let out_img_diff = Tensor::from_inner(
                prep.finish(state, fwd_out.out_img.into_primitive())
            );

            out_img_diff
        }
        OpsKind::UnTracked(prep) => {
            // No gradient needed — return plain forward output
            Tensor::from_inner(prep.finish(fwd_out.out_img.into_primitive()))
        }
    }
}
```

### C.4 Module structure (`render_bwd/mod.rs`)

```rust
//! Backward rendering pipeline

pub mod autodiff;
pub mod project_bwd;
pub mod rasterize_bwd;

pub use autodiff::render_splats;
pub use project_bwd::{project_bwd, ProjectBwdOutput};
pub use rasterize_bwd::rasterize_bwd;
```

Update `src/training/wgpu/mod.rs`:
```rust
pub use render_bwd::render_splats;
```

---

## Part D: Helper Functions in `helpers.wgsl` (additions from Phase 3)

Ensure the following are present (add to Phase 3's helpers.wgsl if missing):

```wgsl
// Quaternion to rotation matrix
fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32>

// VJP of quat_to_mat3
fn quat_to_mat3_vjp(q: vec4<f32>, v_R: mat3x3<f32>) -> vec4<f32>

// VJP of normalize
fn normalize_vjp(v: vec4<f32>) -> mat4x4<f32>

// 2x2 matrix inverse
fn inverse2x2(m: mat2x2<f32>) -> mat2x2<f32>

// VJP of 2x2 matrix inverse: given Minv and dL/d(Minv), returns dL/dM
fn inverse2x2_vjp(m_inv: mat2x2<f32>, v_m_inv: mat2x2<f32>) -> mat2x2<f32>

// diagonal matrix from vec3
fn scale_to_mat3(s: vec3<f32>) -> mat3x3<f32>

// SH color → VJP
fn sh_to_color_vjp(degree: u32, viewdir: vec3<f32>, v_color: vec3<f32>) -> array<vec3<f32>, 25>

// Perspective projection Jacobian
fn calc_cam_J(mean_c: vec3<f32>, focal: vec2<f32>,
              img_size: vec2<u32>, pixel_center: vec2<f32>) -> mat3x2<f32>
```

---

## Files to Create

```
src/training/wgpu/render_bwd/mod.rs
src/training/wgpu/render_bwd/rasterize_bwd.rs
src/training/wgpu/render_bwd/project_bwd.rs
src/training/wgpu/render_bwd/autodiff.rs
src/training/wgpu/shaders/rasterize_backwards.wgsl
src/training/wgpu/shaders/project_backwards.wgsl
```

Also update:
- `src/training/wgpu/shaders/helpers.wgsl` — add VJP helper functions listed in Part D
- `src/training/wgpu/mod.rs` — add `pub use render_bwd::render_splats;`

---

## Verification

```bash
cd /Users/tfjiang/Projects/RustScan/RustGS
cargo check --features gpu-wgpu
```

After Phase 5 is complete, end-to-end gradient correctness is verified by:
1. Rendering a single frame
2. Computing L1 loss against a target image
3. Calling `.backward()` and checking that `splats.transforms.grad()` is `Some`

---

## Reference (study, do not copy)

- `~/Projects/brush/crates/brush-render-bwd/src/burn_glue.rs` — autodiff integration pattern
- `~/Projects/brush/crates/brush-render-bwd/src/shaders/rasterize_backwards.wgsl` — backward rasterize
- `~/Projects/brush/crates/brush-render-bwd/src/shaders/project_backwards.wgsl` — backward projection
- `~/Projects/brush/crates/brush-render-bwd/src/render_bwd.rs` — kernel dispatch
