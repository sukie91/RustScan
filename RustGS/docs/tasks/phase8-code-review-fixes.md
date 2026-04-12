# Phase 8: Code Review Fixes — Numerical Safety & Performance

**Status**: Ready for Implementation
**Dependencies**: Phase 7 complete — 122 unit tests + 2 integration tests passing
**Estimated Complexity**: Medium

## Overview

Post-migration code review from HPC + visual algorithm expert perspective identified 4
correctness bugs, 3 high-priority performance issues, and 2 medium-priority improvements.
This phase addresses them in priority order. The migration is functionally correct; these
fixes improve numerical robustness, training convergence quality, and GPU utilization.

---

## Part A: Tier 1 — Critical Correctness Fixes

### A1. Missing Near-Plane Guard in Backward Shader

**Files**:
- `src/training/wgpu/shaders/project_backwards.wgsl`
- `src/training/wgpu/shaders/helpers.wgsl`

**Problem**: `project_backwards.wgsl:70` computes `mean_c` in camera space but has no
near-plane guard. `helpers.wgsl` `calc_cam_j` then divides by `mean_c.z` without
protection. `project_forward.wgsl:34` has this guard; the backward pass does not. When
`mean_c.z ≈ 0`, the Jacobian becomes ±Inf, all gradients for that Gaussian are NaN, and
Adam propagates NaN through its moments permanently.

**Fix 1 — `project_backwards.wgsl`**: add guard immediately after computing `mean_c` (line 70):

```wgsl
let mean_c = rotation * mean + uniforms.viewmat[3].xyz;
// Guard: skip Gaussians behind or too close to camera (mirrors project_forward.wgsl)
if mean_c.z < 0.01 {
    return;
}
```

**Fix 2 — `helpers.wgsl`, function `calc_cam_j`** (line ~141):

```wgsl
// Before:
let rz = 1.0 / mean_c.z;

// After:
let rz = 1.0 / max(mean_c.z, 0.01);
```

**Fix 3 — `helpers.wgsl`, function `persp_proj_vjp`** (line ~388):

```wgsl
// Before:
let rz = 1.0 / z;

// After:
let rz = 1.0 / max(abs(z), 0.01);
```

**Acceptance Criteria**:
- [ ] `project_backwards.wgsl` has `if mean_c.z < 0.01 { return; }` after `mean_c` is computed
- [ ] `calc_cam_j` uses `max(mean_c.z, 0.01)` in denominator
- [ ] `persp_proj_vjp` uses `max(abs(z), 0.01)` in denominator
- [ ] `cargo test --features gpu --lib` passes

---

### A2. `inverse2x2` Silently Blocks Gradient Flow for Degenerate Gaussians

**File**: `src/training/wgpu/shaders/helpers.wgsl` (lines 180–189)

**Problem**: When `det <= 0.0`, `inverse2x2` returns a zero matrix. In forward rendering
this correctly skips degenerate Gaussians. In the backward pass
(`project_backwards.wgsl:88`), the zero matrix propagates through `inverse2x2_vjp`,
setting all gradients for that Gaussian to zero permanently — it can never recover.

**Fix**: Apply Tikhonov regularization — add a small value to the diagonal before inverting:

```wgsl
fn inverse2x2(m: mat2x2<f32>) -> mat2x2<f32> {
    // Tikhonov regularization: backward pass always produces finite (non-zero) gradients
    var m_reg = m;
    m_reg[0][0] += 1e-6;
    m_reg[1][1] += 1e-6;
    let det = determinant(m_reg);
    if det <= 1e-10 {
        return mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));
    }
    let inv_det = 1.0 / det;
    return mat2x2<f32>(
        vec2<f32>(m_reg[1][1] * inv_det, -m_reg[0][1] * inv_det),
        vec2<f32>(-m_reg[0][1] * inv_det, m_reg[0][0] * inv_det),
    );
}
```

**Note**: `project_visible.wgsl` also calls `helpers::inverse2x2`; the 1e-6 diagonal
perturbation on a real cov2d is negligible for the forward pass.

**Acceptance Criteria**:
- [ ] `inverse2x2` adds `1e-6` to both diagonal elements before computing determinant
- [ ] Guard threshold changed to `det <= 1e-10` to reflect the regularized determinant
- [ ] `cargo test --features gpu --lib` passes

---

### A3. NaN/Inf Loss Silently Corrupts Adam Moments

**File**: `src/training/wgpu/trainer.rs` (line 94)

**Problem**: `loss_value` is extracted but never validated. If NaN reaches the loss
(from degenerate geometry or SSIM instability), `loss.backward()` injects NaN into Adam's
`moment_1` and `moment_2`. Once corrupted, Adam produces NaN parameters forever.

**Fix**: Check before calling `.backward()`:

```rust
// trainer.rs — after line 94:
let loss_value = loss.clone().into_scalar_async().await.expect("loss scalar");

// Guard: skip backward on NaN/Inf to protect Adam optimizer state
if !loss_value.is_finite() {
    log::warn!(
        "Non-finite loss ({loss_value:.6}) at iteration {iteration}; skipping gradient update"
    );
    return loss_value;
}

let mut grads = loss.backward();
```

**Acceptance Criteria**:
- [ ] `train_step` returns early (without calling `.backward()`) when `loss_value` is not finite
- [ ] Warning is logged with iteration number and loss value
- [ ] `cargo test --features gpu --lib` passes

---

### A4. SSIM Denominator Can Reach Near-Zero

**File**: `src/training/wgpu/loss.rs` (line 51)

**Problem**: The `1e-6` epsilon is added only to the variance term. For flat/uniform image
regions, `mu_x_sq + mu_y_sq ≈ 0` and `c1 = 0.0001`, so the mean-term factor is also small.
The product of two near-zero values can produce extreme gradient magnitudes.

**Fix**: Apply `clamp_min` to both factors:

```rust
// loss.rs — replace line 51:
// Before:
let denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2 + 1e-6);

// After:
let denominator = (mu_x_sq + mu_y_sq + c1).clamp_min(1e-8_f32)
    * (sigma_x_sq + sigma_y_sq + c2).clamp_min(1e-8_f32);
let ssim_map = numerator / denominator;
```

**Acceptance Criteria**:
- [ ] Both denominator factors use `.clamp_min(1e-8_f32)`
- [ ] SSIM loss on all-zeros input pair is finite (not NaN/Inf)
- [ ] SSIM loss on identical images remains near zero (existing test passes)
- [ ] `cargo test --features gpu --lib` passes

---

## Part B: Tier 2 — High-Priority Performance

### B1. Batch GPU→CPU Readback (Two Sync Points → One)

**File**: `src/training/wgpu/render/mod.rs` (lines 150–151)

**Problem**: `num_visible` and `num_intersections` are read with two sequential
`read_u32_async` calls, each forcing a full GPU flush and CPU sync. This is two round-trips
per render forward pass.

**Current code**:
```rust
let num_visible = read_u32_async(&proj_out.num_visible_buf).await as usize;
let num_intersections = read_u32_async(&proj_out.num_intersections_buf).await as usize;
```

**Fix**: Concatenate both scalars into a single 2-element tensor and read once:

```rust
// Concatenate both count tensors along dim 0 → [2] tensor, then read in one round-trip
let counts_tensor = Tensor::<B, 1, Int>::cat(
    vec![
        proj_out.num_visible_buf.clone().reshape([1]),
        proj_out.num_intersections_buf.clone().reshape([1]),
    ],
    0,
);
let counts_data = counts_tensor.into_data_async().await.expect("projection count readback");
let num_visible = if let Ok(v) = counts_data.as_slice::<i32>() {
    v[0].max(0) as usize
} else {
    counts_data.as_slice::<u32>().expect("u32 counts")[0] as usize
};
let num_intersections = if let Ok(v) = counts_data.as_slice::<i32>() {
    v[1].max(0) as usize
} else {
    counts_data.as_slice::<u32>().expect("u32 counts")[1] as usize
};
```

**Note**: The exact approach depends on the tensor shape returned by `proj_out`. Read the
`ProjectionOutput` struct in `projection.rs` before implementing.

**Acceptance Criteria**:
- [ ] At most one GPU→CPU sync for projection counts in `render_forward`
- [ ] `cargo test --features gpu --lib` passes

---

### B2. Eliminate Redundant Tensor Clones in `render_forward`

**File**: `src/training/wgpu/render/mod.rs` (lines 183–265)

**Problem**: Several tensors are cloned when they could be moved:
- `proj_out.depths.clone()` — passed to `sort_by_depth` and never used again
- `proj_out.intersect_counts.clone()` — used once for `gather` then discarded
- `global_from_compact_gid.clone()` — cloned 3 times; only 1 is needed (for return value)

**Fix**: Destructure `proj_out` after the readback to take ownership of its fields:

```rust
// After reading num_visible / num_intersections, destructure proj_out:
let depths = proj_out.depths;
let global_from_presort_gid = proj_out.global_from_presort_gid;
let intersect_counts = proj_out.intersect_counts;

// Then pass by move instead of clone:
let global_from_compact_gid = sort_by_depth(depths, global_from_presort_gid, num_visible, device);
//                                           ^^^^^  no .clone() needed

let compact_intersect_counts = intersect_counts   // no .clone() needed
    .gather(0, global_from_compact_gid.clone());  // clone here for rasterize + return
```

Read the actual field names from `ProjectionOutput` in `projection.rs` before implementing.

**Acceptance Criteria**:
- [ ] `depths` and `intersect_counts` passed by move (no `.clone()`)
- [ ] `global_from_compact_gid` cloned at most once
- [ ] `cargo test --features gpu --lib` passes

---

### B3. Position Learning Rate Exponential Decay

**Files**:
- `src/training/wgpu/trainer.rs`
- `src/training/wgpu/optimizer.rs`

**Problem**: `TrainingConfig` has `lr_pos_final` (default `1.6e-6`, 100× smaller than
`lr_position = 1.6e-4`) but the trainer never uses it. Standard 3DGS uses exponential
decay from `lr_position` → `lr_pos_final` over the full training run. Without decay,
Gaussians keep shifting aggressively in late iterations, degrading final quality.

**Fix — `trainer.rs`**: Add `position_lr_at` and call it each step.

```rust
impl WgpuTrainer {
    /// Exponential decay: lr_position → lr_pos_final over config.iterations steps.
    fn position_lr_at(&self, iteration: usize) -> f32 {
        let lr_init = self.config.lr_position;
        let lr_final = self.config.lr_pos_final;
        if lr_final <= 0.0 || lr_final >= lr_init || self.config.iterations == 0 {
            return lr_init;
        }
        let t = (iteration as f32) / (self.config.iterations as f32);
        lr_init * ((lr_final / lr_init).ln() * t).exp()
    }

    fn update_position_lr(&mut self, lr: f32) {
        use burn::tensor::TensorData;
        let pos_cols = Tensor::<GsBackendBase, 2>::from_data(
            TensorData::from([[lr, lr, lr]]),
            &self.device,
        );
        // Rebuild transform_scales: [lr,lr,lr, rot×4, scale×3]
        // Read the current scaling to preserve non-position columns
        let current = self.optimizer.transform_scaling();
        let rest = current.slice(s![.., 3..10]);
        self.optimizer.set_transform_scaling(Tensor::cat(vec![pos_cols, rest], 1));
    }
}
```

In `train_step`, after the non-finite loss guard and before the optimizer step:

```rust
let pos_lr = self.position_lr_at(iteration);
self.update_position_lr(pos_lr);
```

**Fix — `optimizer.rs`**: Add a `transform_scaling` getter to `AdamScaled`:

```rust
pub fn transform_scaling(&self) -> Tensor<GsBackendBase, 2> {
    self.transforms_state
        .as_ref()
        .expect("transform scaling not set")
        .scaling
        .clone()
}
```

Also update `reset_accumulators` (trainer.rs) to call `update_position_lr` with the correct
LR for the current iteration rather than always using `self.config.lr_position`.

**Add unit test**:

```rust
#[test]
fn test_position_lr_decay() {
    let lr_init = 1.6e-4_f32;
    let lr_final = 1.6e-6_f32;
    let iterations = 1000_usize;

    // Simulate position_lr_at by inlining the formula
    let lr_at = |iter: usize| -> f32 {
        let t = iter as f32 / iterations as f32;
        lr_init * ((lr_final / lr_init).ln() * t).exp()
    };

    let at_0 = lr_at(0);
    let at_end = lr_at(iterations);

    assert!((at_0 - lr_init).abs() < 1e-8, "initial LR should equal lr_position");
    assert!((at_end - lr_final).abs() < lr_final * 0.01, "final LR should ≈ lr_pos_final");
    assert!(lr_at(500) < lr_init && lr_at(500) > lr_final, "mid LR should be between bounds");
}
```

**Acceptance Criteria**:
- [ ] `position_lr_at(0) == config.lr_position`
- [ ] `position_lr_at(config.iterations) ≈ config.lr_pos_final`
- [ ] Transform position columns updated every step
- [ ] Unit test passes
- [ ] `cargo test --features gpu --lib` passes

---

## Part C: Tier 3 — Medium Priority

### C1. Fix Int Type Try Order in `topology_bridge.rs`

**File**: `src/training/wgpu/topology_bridge.rs` (around lines 38–48)

**Problem**: `GsBackendBase::IntElement` is `i32`, but the code tries `u32` first then
falls back to `i32`. The `u32` branch always fails silently.

**Fix**: Swap the try order, add a comment:

```rust
// GsBackendBase::IntElement = i32 (wgpu backend default); try i32 first
let num_obs_val = if let Ok(values) = num_obs_data.as_slice::<i32>() {
    values[0].max(0) as usize
} else if let Ok(values) = num_obs_data.as_slice::<u32>() {
    values[0] as usize
} else {
    panic!("num_observations: expected i32 or u32 scalar");
};
```

**Acceptance Criteria**:
- [ ] `i32` branch is tried before `u32`
- [ ] Comment explains why (`GsBackendBase::IntElement = i32`)
- [ ] `cargo test --features gpu --lib` passes

---

### C2. SSIM Separable Blur: Loop → `conv2d`

**File**: `src/training/wgpu/loss.rs` (lines 92–157)

**Problem**: `blur_width` and `blur_height` iterate over 11 kernel positions, each doing
`slice + clone + mul + add` — approximately 22 GPU dispatch cycles per SSIM call, and SSIM
is called multiple times per training step. Burn's `conv2d` collapses this to 2 dispatches.

**Fix**: Replace `separable_blur` with `burn::tensor::module::conv2d` (depthwise):

```rust
use burn::tensor::module::conv2d;

fn separable_blur<B: Backend>(tensor: Tensor<B, 4>, kernel: Tensor<B, 1>) -> Tensor<B, 4> {
    let k = kernel.dims()[0];
    let pad = k / 2;
    let [_n, c, _h, _w] = tensor.dims();

    // Horizontal pass: kernel shape [C, 1, 1, K], padding [0, pad]
    let k_h = kernel.clone()
        .reshape([1, 1, 1, k])
        .repeat_dim(0, c);
    let horiz = conv2d(tensor, k_h, None, [1, 1], [[0, 0], [pad, pad]], [1, 1], c);

    // Vertical pass: kernel shape [C, 1, K, 1], padding [pad, 0]
    let k_v = kernel
        .reshape([1, 1, k, 1])
        .repeat_dim(0, c);
    conv2d(horiz, k_v, None, [1, 1], [[pad, pad], [0, 0]], [1, 1], c)
}
```

**Note**: The exact `conv2d` API signature (especially padding format) may differ between
burn versions. Check `burn::tensor::module::conv2d` in the installed burn version before
implementing. If the module API doesn't support groups/depthwise, use `burn::nn::Conv2d`
with `groups = c` in `Conv2dConfig`.

Verify numerical equivalence with the old loop implementation by running:
```rust
// In tests:
let old_result = blur_width_loop(tensor.clone(), kernel.clone());
let new_result = separable_blur_conv2d(tensor, kernel);
// assert element-wise difference < 1e-4
```

**Acceptance Criteria**:
- [ ] `separable_blur` no longer contains a for-loop over kernel indices
- [ ] SSIM output is numerically equivalent to old implementation (tolerance 1e-4)
- [ ] `cargo test --features gpu --lib` passes

---

## Summary Checklist

### Tier 1 (Correctness — must fix first)
- [ ] **A1**: Near-plane guard in `project_backwards.wgsl` + `calc_cam_j` / `persp_proj_vjp` in `helpers.wgsl`
- [ ] **A2**: `inverse2x2` diagonal regularization (1e-6) to prevent zero gradient flow
- [ ] **A3**: NaN/Inf loss guard before `loss.backward()` in `trainer.rs`
- [ ] **A4**: SSIM denominator double `clamp_min(1e-8_f32)` in `loss.rs`

### Tier 2 (Performance — high value)
- [ ] **B1**: Single GPU→CPU readback for `num_visible` + `num_intersections`
- [ ] **B2**: Eliminate redundant tensor clones in `render_forward`
- [ ] **B3**: Position LR exponential decay from `lr_position` → `lr_pos_final`

### Tier 3 (Medium priority)
- [ ] **C1**: Fix Int type try order in `topology_bridge.rs` (i32 before u32)
- [ ] **C2**: Replace loop-based separable blur with `conv2d`

---

## Verification

After completing each tier, run:

```bash
cargo test --features gpu --lib
cargo test --features gpu --test integration_test -- --ignored
cargo clippy --features gpu -- -D warnings
```

All 122 unit tests + 2 integration tests must pass with zero new clippy warnings.
