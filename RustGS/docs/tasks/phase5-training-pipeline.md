# Phase 5: Training Pipeline

**Status**: Ready for Implementation  
**Dependencies**: Phase 4 (Backward/Autodiff) must be complete  
**Estimated Complexity**: High

## Overview

Implement the complete training pipeline for Gaussian Splatting using burn+wgpu backend. This phase bridges the GPU rendering/autodiff system with the CPU-side topology mutations and training orchestration.

## Architecture

```
WgpuTrainer (async)
    ├─> AdamScaled optimizer (per-column LR scaling)
    ├─> L1 + SSIM loss computation
    ├─> Training loop (7 steps per iteration)
    ├─> topology_bridge.rs (GPU → CPU snapshot)
    └─> topology_apply.rs (mutation plan → GPU tensor ops)

train_splats() (sync wrapper)
    └─> tokio::runtime → WgpuTrainer
```

## Reference Implementation

- **brush**: `crates/brush-train/src/train.rs` - AdamScaled, SSIM, training loop
- **brush**: `crates/brush-train/src/scene.rs` - Topology operations
- **RustGS**: `src/training/metal/trainer.rs` - MetalTrainer interface (reference only)

## Files to Create

### Part A: AdamScaled Optimizer

**File**: `RustGS/src/training/wgpu/optimizer.rs`

Implement custom Adam optimizer with per-column learning rate scaling.

```rust
use burn::optim::{Optimizer, SimpleOptimizer};
use burn::tensor::{backend::Backend, Tensor};

pub struct AdamScaledConfig {
    pub lr: f64,
    pub betas: (f64, f64),
    pub eps: f64,
    pub weight_decay: f64,
    pub lr_scale_factors: Vec<f64>,  // Per-column LR multipliers
}

pub struct AdamScaled<B: Backend> {
    config: AdamScaledConfig,
    moment1: Option<Tensor<B, 2>>,
    moment2: Option<Tensor<B, 2>>,
    step: usize,
}

impl<B: Backend> SimpleOptimizer<B> for AdamScaled<B> {
    type Config = AdamScaledConfig;
    
    fn step(&mut self, lr: f64, param: Tensor<B, 2>, grad: Tensor<B, 2>) -> Tensor<B, 2> {
        // 1. Initialize moments if first step
        // 2. Update biased moments: m_t = β1*m_{t-1} + (1-β1)*g_t
        // 3. Update biased second moments: v_t = β2*v_{t-1} + (1-β2)*g_t²
        // 4. Compute bias correction: m_hat = m_t / (1 - β1^t)
        // 5. Compute bias correction: v_hat = v_t / (1 - β2^t)
        // 6. Apply per-column LR scaling
        // 7. Update: param -= lr * scale * m_hat / (sqrt(v_hat) + eps)
        // 8. Apply weight decay if configured
        todo!()
    }
}
```

**Key Requirements**:
- Per-column LR scaling via `lr_scale_factors` vector
- Standard Adam with bias correction
- Optional weight decay (AdamW style)
- Efficient tensor operations (avoid loops)

**brush Reference**: `crates/brush-train/src/train.rs:AdamScaled`

---

### Part B: SSIM Loss

**File**: `RustGS/src/training/wgpu/loss.rs`

Implement Structural Similarity Index (SSIM) loss with separable Gaussian convolution.

```rust
use burn::prelude::*;

pub struct SsimConfig {
    pub window_size: usize,  // Default: 11
    pub sigma: f64,          // Default: 1.5
    pub k1: f64,             // Default: 0.01
    pub k2: f64,             // Default: 0.03
    pub data_range: f64,     // Default: 1.0
}

pub fn ssim_loss<B: Backend>(
    pred: Tensor<B, 3>,      // [H, W, C]
    target: Tensor<B, 3>,    // [H, W, C]
    config: &SsimConfig,
    device: &B::Device,
) -> Tensor<B, 1> {
    // 1. Create 1D Gaussian kernel
    // 2. Apply separable convolution (horizontal then vertical)
    // 3. Compute local means: μ_x, μ_y
    // 4. Compute local variances: σ_x², σ_y²
    // 5. Compute local covariance: σ_xy
    // 6. Compute SSIM map: (2μ_xμ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
    // 7. Return 1 - mean(SSIM)
    todo!()
}

pub fn combined_loss<B: Backend>(
    pred: Tensor<B, 3>,
    target: Tensor<B, 3>,
    l1_weight: f64,
    ssim_weight: f64,
    ssim_config: &SsimConfig,
    device: &B::Device,
) -> Tensor<B, 1> {
    let l1 = (pred.clone() - target.clone()).abs().mean();
    let ssim = ssim_loss(pred, target, ssim_config, device);
    l1 * l1_weight + ssim * ssim_weight
}
```

**Key Requirements**:
- 11x11 Gaussian window (configurable)
- Separable convolution for efficiency
- Constants C1 = (k1 * data_range)², C2 = (k2 * data_range)²
- Return 1 - SSIM (loss, not similarity)
- Combined L1 + SSIM loss function

**brush Reference**: `crates/brush-train/src/ssim.rs`

---

### Part C: Topology Bridge (GPU → CPU)

**File**: `RustGS/src/training/wgpu/topology_bridge.rs`

Extract GPU tensor data to CPU for topology mutation planning.

```rust
use burn::prelude::*;
use crate::training::wgpu::splats::DeviceSplats;
use crate::training::topology::TopologyMutationPlan;

pub struct TopologySnapshot {
    pub means: Vec<[f32; 3]>,
    pub log_scales: Vec<[f32; 3]>,
    pub quats: Vec<[f32; 4]>,
    pub opacities: Vec<f32>,
    pub sh_dc: Vec<[f32; 3]>,
    pub sh_rest: Vec<Vec<[f32; 3]>>,
    pub grad_2d_accum: Vec<f32>,
    pub grad_color_accum: Vec<f32>,
    pub num_observations: Vec<u32>,
}

pub async fn snapshot_for_topology<B: Backend>(
    splats: &DeviceSplats<B>,
    grad_2d_accum: &Tensor<B, 1>,
    grad_color_accum: &Tensor<B, 1>,
    num_observations: &Tensor<B, 1, Int>,
) -> TopologySnapshot {
    // 1. Unpack transforms tensor [N, 10] → means, quats, log_scales
    // 2. Read all tensors to CPU via into_data_async().await
    // 3. Convert to Vec<[f32; D]> format
    // 4. Return snapshot
    todo!()
}

pub fn plan_mutations(
    snapshot: &TopologySnapshot,
    config: &crate::training::TrainingConfig,
    iteration: usize,
) -> TopologyMutationPlan {
    // Delegate to existing RustGS topology planning logic
    crate::training::topology::plan_topology_mutations(
        snapshot,
        config,
        iteration,
    )
}
```

**Key Requirements**:
- Async tensor reads (into_data_async)
- Unpack transforms [N,10] → means(3) + quats(4) + log_scales(3)
- Convert burn Data<f32, 1> to Vec<[f32; D]>
- Reuse existing `crate::training::topology::plan_topology_mutations`

---

### Part D: Topology Apply (CPU Plan → GPU Tensors)

**File**: `RustGS/src/training/wgpu/topology_apply.rs`

Apply topology mutation plan to GPU tensors.

```rust
use burn::prelude::*;
use crate::training::wgpu::splats::DeviceSplats;
use crate::training::topology::TopologyMutationPlan;

pub fn apply_clone<B: Backend>(
    splats: &mut DeviceSplats<B>,
    indices: &[usize],
    device: &B::Device,
) {
    // 1. Create index tensor from indices
    // 2. Gather rows: new_rows = splats.transforms.gather(0, index_tensor)
    // 3. Concatenate: splats.transforms = cat([splats.transforms, new_rows], dim=0)
    // 4. Repeat for all other tensors (opacities, sh_dc, sh_rest)
    todo!()
}

pub fn apply_split<B: Backend>(
    splats: &mut DeviceSplats<B>,
    indices: &[usize],
    scale_factor: f32,
    device: &B::Device,
) {
    // 1. Gather parent rows
    // 2. Unpack transforms → means, quats, log_scales
    // 3. Scale down: log_scales_new = log_scales - log(scale_factor)
    // 4. Create 2 children per parent with small position offset
    // 5. Remove parents (see apply_prune)
    // 6. Concatenate children
    todo!()
}

pub fn apply_prune<B: Backend>(
    splats: &mut DeviceSplats<B>,
    keep_mask: &[bool],
    device: &B::Device,
) {
    // 1. Create boolean mask tensor
    // 2. Use mask_where or gather with filtered indices
    // 3. Update all tensors (transforms, opacities, sh_dc, sh_rest)
    // 4. Update splats.num_splats
    todo!()
}

pub fn apply_mutations<B: Backend>(
    splats: &mut DeviceSplats<B>,
    plan: &TopologyMutationPlan,
    device: &B::Device,
) {
    if !plan.clone_indices.is_empty() {
        apply_clone(splats, &plan.clone_indices, device);
    }
    if !plan.split_indices.is_empty() {
        apply_split(splats, &plan.split_indices, plan.split_scale_factor, device);
    }
    if !plan.prune_mask.is_empty() {
        apply_prune(splats, &plan.prune_mask, device);
    }
}
```

**Key Requirements**:
- Clone: gather + concatenate
- Split: gather + unpack + scale + offset + remove parents + concat
- Prune: boolean mask filtering
- Update DeviceSplats.num_splats after mutations
- All operations use burn tensor ops (no manual loops)

**brush Reference**: `crates/brush-train/src/scene.rs:clone_splats`, `split_splats`, `cull_splats`

---

### Part E: WgpuTrainer

**File**: `RustGS/src/training/wgpu/trainer.rs`

Main training loop orchestration.

```rust
use burn::prelude::*;
use crate::core::GaussianCamera;
use crate::training::wgpu::{DeviceSplats, GsDiffBackend, GsDevice};
use crate::training::wgpu::render::render_forward;
use crate::training::wgpu::loss::combined_loss;
use crate::training::wgpu::optimizer::AdamScaled;
use crate::training::{TrainingConfig, TrainingReport, TrainingEvent};

pub struct WgpuTrainer<B: Backend> {
    config: TrainingConfig,
    optimizer: AdamScaled<B>,
    device: B::Device,
    grad_2d_accum: Tensor<B, 1>,
    grad_color_accum: Tensor<B, 1>,
    num_observations: Tensor<B, 1, Int>,
}

impl<B: GsDiffBackend> WgpuTrainer<B> {
    pub fn new(config: TrainingConfig, device: B::Device) -> Self {
        // Initialize optimizer with per-column LR scaling
        // means: 1.6e-4, quats: 1e-3, log_scales: 5e-3, opacities: 5e-2, sh: 2.5e-3
        todo!()
    }
    
    pub async fn train_step(
        &mut self,
        splats: &mut DeviceSplats<B>,
        camera: &GaussianCamera,
        target_img: &Tensor<B, 3>,
        iteration: usize,
    ) -> f32 {
        // Step 1: Prepare data
        let img_size = (target_img.dims()[1] as u32, target_img.dims()[0] as u32);
        let background = [0.0, 0.0, 0.0];
        
        // Step 2: Forward render (with autodiff)
        let render_out = render_forward(splats, camera, img_size, background, &self.device).await;
        
        // Step 3: Compute loss
        let loss = combined_loss(
            render_out.out_img.clone(),
            target_img.clone(),
            0.8,  // L1 weight
            0.2,  // SSIM weight
            &Default::default(),
            &self.device,
        );
        
        // Step 4: Backward pass
        let grads = loss.backward();
        
        // Step 5: Accumulate gradients for topology
        self.accumulate_gradients(&render_out, &grads);
        
        // Step 6: Optimizer step
        self.optimizer.step(/* params and grads */);
        
        // Step 7: Topology mutations (every N iterations)
        if iteration % self.config.densify_interval == 0 {
            self.apply_topology_mutations(splats, iteration).await;
        }
        
        loss.into_scalar_async().await
    }
    
    async fn accumulate_gradients(&mut self, render_out: &RenderOutput<B>, grads: &Gradients) {
        // Extract 2D position gradients and color gradients
        // Accumulate into grad_2d_accum and grad_color_accum
        todo!()
    }
    
    async fn apply_topology_mutations(&mut self, splats: &mut DeviceSplats<B>, iteration: usize) {
        // 1. Snapshot GPU → CPU
        let snapshot = topology_bridge::snapshot_for_topology(
            splats,
            &self.grad_2d_accum,
            &self.grad_color_accum,
            &self.num_observations,
        ).await;
        
        // 2. Plan mutations (CPU)
        let plan = topology_bridge::plan_mutations(&snapshot, &self.config, iteration);
        
        // 3. Apply mutations (GPU)
        topology_apply::apply_mutations(splats, &plan, &self.device);
        
        // 4. Reset accumulators
        self.reset_accumulators(splats.num_splats());
    }
    
    fn reset_accumulators(&mut self, num_splats: usize) {
        self.grad_2d_accum = Tensor::zeros([num_splats], &self.device);
        self.grad_color_accum = Tensor::zeros([num_splats], &self.device);
        self.num_observations = Tensor::zeros([num_splats], &self.device);
    }
    
    pub async fn train(
        &mut self,
        splats: &mut DeviceSplats<B>,
        cameras: &[GaussianCamera],
        target_images: &[Tensor<B, 3>],
        num_iterations: usize,
    ) -> TrainingReport {
        let mut report = TrainingReport::default();
        
        for iter in 0..num_iterations {
            // Sample random camera
            let idx = rand::random::<usize>() % cameras.len();
            let camera = &cameras[idx];
            let target = &target_images[idx];
            
            // Train step
            let loss = self.train_step(splats, camera, target, iter).await;
            
            // Update report
            report.losses.push(loss);
            report.num_splats.push(splats.num_splats());
            
            // Logging
            if iter % 100 == 0 {
                println!("Iteration {}: loss = {:.6}, splats = {}", iter, loss, splats.num_splats());
            }
        }
        
        report
    }
}
```

**Key Requirements**:
- 7-step training loop per iteration
- Autodiff integration via burn's Backward trait
- Gradient accumulation for topology decisions
- Periodic topology mutations (clone/split/prune)
- Per-column LR scaling in optimizer
- Random camera sampling
- Progress logging

**brush Reference**: `crates/brush-train/src/train.rs:train_step`

---

### Part F: Synchronous Entry Point

**File**: `RustGS/src/training/wgpu/mod.rs` (add to existing)

Wrap async trainer in synchronous API matching existing RustGS interface.

```rust
use crate::training::{HostSplats, TrainingConfig, TrainingReport};
use crate::core::GaussianCamera;

pub fn train_splats(
    splats: &mut HostSplats,
    cameras: &[GaussianCamera],
    target_images: &[Vec<u8>],  // RGB8 images
    config: &TrainingConfig,
) -> TrainingReport {
    // 1. Create tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    rt.block_on(async {
        // 2. Initialize device
        let device = GsDevice::default();
        
        // 3. Convert HostSplats → DeviceSplats
        let mut device_splats = host_splats_to_device(splats, &device);
        
        // 4. Convert target images to tensors
        let target_tensors = convert_images_to_tensors(target_images, &device);
        
        // 5. Create trainer
        let mut trainer = WgpuTrainer::new(config.clone(), device.clone());
        
        // 6. Train
        let report = trainer.train(
            &mut device_splats,
            cameras,
            &target_tensors,
            config.num_iterations,
        ).await;
        
        // 7. Convert DeviceSplats → HostSplats
        *splats = device_splats_to_host(&device_splats).await;
        
        report
    })
}

fn convert_images_to_tensors<B: Backend>(
    images: &[Vec<u8>],
    device: &B::Device,
) -> Vec<Tensor<B, 3>> {
    // Convert RGB8 → f32 [H, W, 3] tensors
    todo!()
}
```

**Key Requirements**:
- Match existing `train_splats` signature
- Tokio runtime for async execution
- HostSplats ↔ DeviceSplats conversion
- Image format conversion (RGB8 → f32 tensors)

---

## Integration Points

### With Phase 4 (Backward)
- Use `render_forward` with autodiff enabled
- Extract gradients via `loss.backward()`
- Gradients flow through custom backward ops

### With Existing RustGS
- Reuse `TrainingConfig` struct
- Reuse `TopologyMutationPlan` and planning logic
- Reuse `HostSplats` data structure
- Match `train_splats` API signature

### With Phase 3 (Forward)
- Call `render_forward` from trainer
- Use `RenderOutput` for gradient accumulation

---

## Testing Strategy

1. **Unit Tests**:
   - AdamScaled optimizer (verify bias correction, LR scaling)
   - SSIM loss (compare against reference implementation)
   - Topology operations (clone/split/prune tensor shapes)

2. **Integration Tests**:
   - Single training step (forward + backward + optimizer)
   - Topology mutation cycle (snapshot → plan → apply)
   - Full training loop (10 iterations, verify loss decreases)

3. **Validation**:
   - Compare loss values with MetalTrainer
   - Verify splat count evolution matches expected pattern
   - Visual inspection of rendered outputs

---

## Acceptance Criteria

- [ ] AdamScaled optimizer implemented with per-column LR scaling
- [ ] SSIM loss with 11x11 Gaussian window
- [ ] Combined L1 + SSIM loss function
- [ ] Topology bridge (GPU → CPU snapshot)
- [ ] Topology apply (clone/split/prune on GPU tensors)
- [ ] WgpuTrainer with 7-step training loop
- [ ] Gradient accumulation for topology decisions
- [ ] Synchronous `train_splats` entry point
- [ ] `cargo check --features gpu-wgpu` passes
- [ ] Unit tests pass
- [ ] Integration test: 10 iterations, loss decreases

---

## Notes

- **LR Scaling**: means(1.6e-4), quats(1e-3), log_scales(5e-3), opacities(5e-2), sh(2.5e-3)
- **SSIM Window**: 11x11 Gaussian, σ=1.5
- **Loss Weights**: L1(0.8) + SSIM(0.2)
- **Topology Interval**: Every 100 iterations (configurable)
- **Prune Threshold**: opacity < 0.005, scale > scene_extent
- **Clone Threshold**: grad_2d > threshold
- **Split Threshold**: grad_2d > threshold AND scale > threshold

---

## Codex Prompt

When ready to implement, use:

```
Implement Phase 5 (Training Pipeline) for RustGS burn+wgpu migration.

Reference:
- Task doc: RustGS/docs/tasks/phase5-training-pipeline.md
- brush implementation: ~/Projects/brush/crates/brush-train/src/train.rs
- Existing RustGS: RustGS/src/training/topology.rs

Create these files:
1. RustGS/src/training/wgpu/optimizer.rs - AdamScaled with per-column LR
2. RustGS/src/training/wgpu/loss.rs - SSIM + combined loss
3. RustGS/src/training/wgpu/topology_bridge.rs - GPU → CPU snapshot
4. RustGS/src/training/wgpu/topology_apply.rs - Mutation plan → GPU tensors
5. RustGS/src/training/wgpu/trainer.rs - WgpuTrainer main loop
6. Update RustGS/src/training/wgpu/mod.rs - Add train_splats entry point

Key requirements:
- AdamScaled: per-column LR scaling, bias correction
- SSIM: 11x11 Gaussian window, separable convolution
- Topology: clone/split/prune via tensor ops (no loops)
- Training loop: 7 steps (data → render → loss → backward → optimizer → accumulate → topology)
- Match existing train_splats API signature

Verify with: cargo check --features gpu-wgpu
```
