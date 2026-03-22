//! Complete Training Loop with Backward Propagation
//!
//! This module implements the full backward propagation for 3DGS:
//! - Forward pass: rendering
//! - Loss computation
//! - Backward pass: gradient computation
//! - Parameter update with Adam (GPU-accelerated)

#[cfg(feature = "gpu")]
use candle_core::{DType, Device, Tensor, Var};
#[cfg(feature = "gpu")]
use rayon::prelude::*;
#[cfg(feature = "gpu")]
use std::time::Instant;

#[cfg(feature = "gpu")]
use crate::diff::diff_splat::{
    DiffCamera, DiffSplatRenderer, SurrogateGradients, TrainableGaussians,
};

#[cfg(feature = "gpu")]
use crate::diff::analytical_backward;

use serde::{Deserialize, Serialize};

/// Learning rate scheduler
#[derive(Debug, Clone)]
pub struct LrScheduler {
    /// Base learning rate
    base_lr: f32,
    /// Warmup iterations
    warmup_iters: usize,
    /// Total iterations
    total_iters: usize,
}

impl LrScheduler {
    pub fn new(base_lr: f32, warmup_iters: usize, total_iters: usize) -> Self {
        Self {
            base_lr,
            warmup_iters,
            total_iters,
        }
    }

    /// Get learning rate for current iteration
    pub fn get_lr(&self, iteration: usize) -> f32 {
        if iteration < self.warmup_iters {
            // Linear warmup
            if self.warmup_iters == 0 {
                return self.base_lr;
            }
            self.base_lr * (iteration as f32 / self.warmup_iters as f32)
        } else {
            // Cosine decay
            let decay_span = self.total_iters.saturating_sub(self.warmup_iters).max(1);
            let progress = (iteration.saturating_sub(self.warmup_iters)) as f32 / decay_span as f32;
            self.base_lr * 0.5 * (1.0 + (progress * std::f32::consts::PI).cos())
        }
    }
}

/// Training result
#[derive(Debug)]
pub struct TrainingResult {
    pub final_loss: f32,
    pub iterations: usize,
    pub num_gaussians: usize,
    pub loss_history: Vec<f32>,
}

/// Adam optimizer state for checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerAdamState {
    pub m_pos: Vec<f32>,
    pub m_scale: Vec<f32>,
    pub m_rot: Vec<f32>,
    pub m_op: Vec<f32>,
    pub m_color: Vec<f32>,
    pub v_pos: Vec<f32>,
    pub v_scale: Vec<f32>,
    pub v_rot: Vec<f32>,
    pub v_op: Vec<f32>,
    pub v_color: Vec<f32>,
}

/// Complete trainer with backward propagation (GPU only)
///
/// Uses GPU-side Adam optimizer to avoid CPU↔GPU data transfers each step.
#[cfg(feature = "gpu")]
pub struct CompleteTrainer {
    renderer: DiffSplatRenderer,
    device: Device,
    width: usize,
    height: usize,
    /// Learning rate for positions
    lr_pos: f32,
    /// Learning rate for scales
    lr_scale: f32,
    /// Learning rate for rotations
    lr_rot: f32,
    /// Learning rate for opacities
    lr_op: f32,
    /// Learning rate for colors
    lr_color: f32,
    /// Beta1 for Adam
    beta1: f32,
    /// Beta2 for Adam
    beta2: f32,
    /// GPU Adam first moment (lazily initialized)
    m_pos: Option<Tensor>,
    m_scale: Option<Tensor>,
    m_rot: Option<Tensor>,
    m_op: Option<Tensor>,
    m_color: Option<Tensor>,
    /// GPU Adam second moment (lazily initialized)
    v_pos: Option<Tensor>,
    v_scale: Option<Tensor>,
    v_rot: Option<Tensor>,
    v_op: Option<Tensor>,
    v_color: Option<Tensor>,
    /// Current iteration
    iteration: usize,
    /// Loss history
    loss_history: Vec<f32>,
    /// Learning-rate scheduler multiplier
    lr_scheduler: LrScheduler,
    /// Use analytical backward pass instead of finite differences (default: true)
    pub use_analytical_backward: bool,
    /// Compute surrogate gradients every N iterations (default: 500)
    surrogate_freq: usize,
    /// Cached surrogate gradients from last computation
    cached_surrogate: Option<SurrogateGradients>,
}

#[cfg(feature = "gpu")]
impl CompleteTrainer {
    /// Create a new trainer
    pub fn new(
        width: usize,
        height: usize,
        lr_pos: f32,
        lr_scale: f32,
        lr_rot: f32,
        lr_op: f32,
        lr_color: f32,
    ) -> Self {
        Self::with_device(
            width,
            height,
            lr_pos,
            lr_scale,
            lr_rot,
            lr_op,
            lr_color,
            crate::preferred_device(),
        )
    }

    pub fn with_device(
        width: usize,
        height: usize,
        lr_pos: f32,
        lr_scale: f32,
        lr_rot: f32,
        lr_op: f32,
        lr_color: f32,
        device: Device,
    ) -> Self {
        let renderer = DiffSplatRenderer::with_device(width, height, device.clone());

        log::info!("CompleteTrainer using device: {:?}", device);

        Self {
            renderer,
            device,
            width,
            height,
            lr_pos,
            lr_scale,
            lr_rot,
            lr_op,
            lr_color,
            beta1: 0.9,
            beta2: 0.999,
            m_pos: None,
            m_scale: None,
            m_rot: None,
            m_op: None,
            m_color: None,
            v_pos: None,
            v_scale: None,
            v_rot: None,
            v_op: None,
            v_color: None,
            iteration: 0,
            loss_history: Vec::new(),
            lr_scheduler: LrScheduler::new(1.0, 100, 3000),
            use_analytical_backward: true,
            surrogate_freq: 500,
            cached_surrogate: None,
        }
    }

    /// Get training device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Initialize GPU Adam optimizer state
    fn init_adam_state(&mut self, n: usize) {
        let d = &self.device;
        let dt = DType::F32;
        // First moments: zero-initialized
        self.m_pos = Some(Tensor::zeros((n, 3), dt, d).unwrap());
        self.m_scale = Some(Tensor::zeros((n, 3), dt, d).unwrap());
        self.m_rot = Some(Tensor::zeros((n, 4), dt, d).unwrap());
        self.m_op = Some(Tensor::zeros((n,), dt, d).unwrap());
        self.m_color = Some(Tensor::zeros((n, 3), dt, d).unwrap());
        // Second moments: initialized to eps for numerical stability
        self.v_pos = Some(
            Tensor::zeros((n, 3), dt, d)
                .unwrap()
                .affine(0.0, 1e-8)
                .unwrap(),
        );
        self.v_scale = Some(
            Tensor::zeros((n, 3), dt, d)
                .unwrap()
                .affine(0.0, 1e-8)
                .unwrap(),
        );
        self.v_rot = Some(
            Tensor::zeros((n, 4), dt, d)
                .unwrap()
                .affine(0.0, 1e-8)
                .unwrap(),
        );
        self.v_op = Some(
            Tensor::zeros((n,), dt, d)
                .unwrap()
                .affine(0.0, 1e-8)
                .unwrap(),
        );
        self.v_color = Some(
            Tensor::zeros((n, 3), dt, d)
                .unwrap()
                .affine(0.0, 1e-8)
                .unwrap(),
        );
    }

    /// Training step with backward propagation
    ///
    /// When `use_analytical_backward` is true (default), uses exact analytical
    /// gradients from a single forward+backward pass. All parameter updates
    /// happen on GPU via Tensor ops — no CPU roundtrip.
    ///
    /// When false, falls back to the legacy finite-difference approach with
    /// CPU-side Adam.
    pub fn training_step(
        &mut self,
        gaussians: &mut TrainableGaussians,
        camera: &DiffCamera,
        target_color: &[f32],
        target_depth: &[f32],
    ) -> candle_core::Result<f32> {
        let n = gaussians.len();

        if self.m_pos.is_none() {
            self.init_adam_state(n);
        }

        // Get gradients (either analytical or finite_diff)
        let (loss_value, pos_grad, scale_grad, rot_grad, op_grad, color_grad) =
            if self.use_analytical_backward {
                self.analytical_training_step(gaussians, camera, target_color, target_depth, n)?
            } else {
                // Finite diff needs CPU params for perturbation
                let mut pos_data = flatten_2d(&gaussians.positions().to_vec2::<f32>()?);
                let mut scale_data = flatten_2d(&gaussians.scales.as_tensor().to_vec2::<f32>()?);
                let mut rot_data = flatten_2d(&gaussians.rotations.as_tensor().to_vec2::<f32>()?);
                let mut op_data = gaussians.opacities.as_tensor().to_vec1::<f32>()?;
                let mut color_data = flatten_2d(&gaussians.colors().to_vec2::<f32>()?);
                self.finite_diff_training_step(
                    gaussians,
                    camera,
                    target_color,
                    target_depth,
                    n,
                    &mut pos_data,
                    &mut scale_data,
                    &mut rot_data,
                    &mut op_data,
                    &mut color_data,
                )?
            };

        self.loss_history.push(loss_value);
        self.iteration += 1;

        let lr_factor = self.lr_scheduler.get_lr(self.iteration).max(1e-4);

        // GPU Adam update — gradient vecs are uploaded once, all math on GPU
        gpu_adam_step(
            &gaussians.positions,
            &pos_grad,
            self.m_pos.as_mut().unwrap(),
            self.v_pos.as_mut().unwrap(),
            self.lr_pos * lr_factor,
            self.beta1,
            self.beta2,
            self.iteration,
            &self.device,
        )?;
        gpu_adam_step(
            &gaussians.scales,
            &scale_grad,
            self.m_scale.as_mut().unwrap(),
            self.v_scale.as_mut().unwrap(),
            self.lr_scale * lr_factor,
            self.beta1,
            self.beta2,
            self.iteration,
            &self.device,
        )?;
        gpu_adam_step(
            &gaussians.rotations,
            &rot_grad,
            self.m_rot.as_mut().unwrap(),
            self.v_rot.as_mut().unwrap(),
            self.lr_rot * lr_factor,
            self.beta1,
            self.beta2,
            self.iteration,
            &self.device,
        )?;
        gpu_adam_step(
            &gaussians.opacities,
            &op_grad,
            self.m_op.as_mut().unwrap(),
            self.v_op.as_mut().unwrap(),
            self.lr_op * lr_factor,
            self.beta1,
            self.beta2,
            self.iteration,
            &self.device,
        )?;
        gpu_adam_step(
            &gaussians.colors,
            &color_grad,
            self.m_color.as_mut().unwrap(),
            self.v_color.as_mut().unwrap(),
            self.lr_color * lr_factor,
            self.beta1,
            self.beta2,
            self.iteration,
            &self.device,
        )?;

        // GPU rotation normalization
        let rot = gaussians.rotations.as_tensor();
        let norm = rot
            .sqr()?
            .sum(1)?
            .sqrt()?
            .clamp(1e-6, f32::MAX)?
            .unsqueeze(1)?;
        gaussians.rotations.set(&rot.broadcast_div(&norm)?)?;

        Ok(loss_value)
    }

    /// Analytical backward: 1 forward + 1 backward pass.
    fn analytical_training_step(
        &mut self,
        gaussians: &TrainableGaussians,
        camera: &DiffCamera,
        target_color: &[f32],
        target_depth: &[f32],
        n: usize,
    ) -> candle_core::Result<(f32, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        // Forward pass with intermediates (GPU projection + CPU rendering)
        let (_output, intermediate) = self.renderer.render_with_intermediates(gaussians, camera)?;

        // Use CPU data directly — avoids GPU→CPU tensor copy from compute_loss
        let loss_value = self.renderer.compute_loss_from_cpu(
            &intermediate.rendered_color,
            &intermediate.rendered_depth,
            target_color,
            target_depth,
        );

        // Analytical backward
        let grads = analytical_backward::backward(
            &intermediate,
            target_color,
            n,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
        );

        // Blend with surrogate gradients (10% weight for regularization)
        // Only recompute every surrogate_freq iterations to save GPU backward passes
        if self.cached_surrogate.is_none() || self.iteration % self.surrogate_freq == 0 {
            self.cached_surrogate = Some(self.renderer.compute_surrogate_gradients(gaussians)?);
        }
        let surrogate = self.cached_surrogate.as_ref().unwrap();
        let mut pos_grad = grads.positions;
        let mut scale_grad = grads.log_scales;
        let mut op_grad = grads.opacity_logits;
        let mut color_grad = grads.colors;
        blend_gradients(&mut pos_grad, &surrogate.positions, 0.1);
        blend_gradients(&mut scale_grad, &surrogate.scales, 0.1);
        blend_gradients(&mut op_grad, &surrogate.opacities, 0.1);
        blend_gradients(&mut color_grad, &surrogate.colors, 0.1);

        let mut rot_grad = surrogate.rotations.clone();
        if rot_grad.len() != n * 4 {
            rot_grad = vec![0.0; n * 4];
        }

        Ok((
            loss_value, pos_grad, scale_grad, rot_grad, op_grad, color_grad,
        ))
    }

    /// Legacy finite-difference gradient estimation.
    fn finite_diff_training_step(
        &mut self,
        gaussians: &TrainableGaussians,
        camera: &DiffCamera,
        target_color: &[f32],
        target_depth: &[f32],
        n: usize,
        pos_data: &mut Vec<f32>,
        scale_data: &mut Vec<f32>,
        rot_data: &mut Vec<f32>,
        op_data: &mut Vec<f32>,
        color_data: &mut Vec<f32>,
    ) -> candle_core::Result<(f32, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let output = self.renderer.render(gaussians, camera)?;
        let loss = self
            .renderer
            .compute_loss(&output, target_color, target_depth)?;
        let loss_value = loss.total.to_vec0::<f32>()?;

        let mut pos_grad = vec![0.0f32; n * 3];
        let mut scale_grad = vec![0.0f32; n * 3];
        let mut op_grad = vec![0.0f32; n];
        let mut color_grad = vec![0.0f32; n * 3];
        let eps = 1e-3f32;
        let sample_count = n.min(8).max(1);
        let stride = n.div_ceil(sample_count);

        for s in 0..sample_count {
            let i = s * stride;
            if i >= n {
                break;
            }

            for d in 0..3 {
                let idx = i * 3 + d;
                pos_data[idx] += eps;
                let plus = self.evaluate_loss(
                    pos_data,
                    scale_data,
                    rot_data,
                    op_data,
                    color_data,
                    n,
                    camera,
                    target_color,
                    target_depth,
                )?;
                pos_data[idx] -= 2.0 * eps;
                let minus = self.evaluate_loss(
                    pos_data,
                    scale_data,
                    rot_data,
                    op_data,
                    color_data,
                    n,
                    camera,
                    target_color,
                    target_depth,
                )?;
                pos_data[idx] += eps;
                pos_grad[idx] = (plus - minus) / (2.0 * eps);
            }

            for d in 0..3 {
                let idx = i * 3 + d;
                scale_data[idx] += eps;
                let plus = self.evaluate_loss(
                    pos_data,
                    scale_data,
                    rot_data,
                    op_data,
                    color_data,
                    n,
                    camera,
                    target_color,
                    target_depth,
                )?;
                scale_data[idx] -= 2.0 * eps;
                let minus = self.evaluate_loss(
                    pos_data,
                    scale_data,
                    rot_data,
                    op_data,
                    color_data,
                    n,
                    camera,
                    target_color,
                    target_depth,
                )?;
                scale_data[idx] += eps;
                scale_grad[idx] = (plus - minus) / (2.0 * eps);
            }

            let op_idx = i;
            op_data[op_idx] += eps;
            let plus = self.evaluate_loss(
                pos_data,
                scale_data,
                rot_data,
                op_data,
                color_data,
                n,
                camera,
                target_color,
                target_depth,
            )?;
            op_data[op_idx] -= 2.0 * eps;
            let minus = self.evaluate_loss(
                pos_data,
                scale_data,
                rot_data,
                op_data,
                color_data,
                n,
                camera,
                target_color,
                target_depth,
            )?;
            op_data[op_idx] += eps;
            op_grad[op_idx] = (plus - minus) / (2.0 * eps);

            for d in 0..3 {
                let idx = i * 3 + d;
                color_data[idx] += eps;
                let plus = self.evaluate_loss(
                    pos_data,
                    scale_data,
                    rot_data,
                    op_data,
                    color_data,
                    n,
                    camera,
                    target_color,
                    target_depth,
                )?;
                color_data[idx] -= 2.0 * eps;
                let minus = self.evaluate_loss(
                    pos_data,
                    scale_data,
                    rot_data,
                    op_data,
                    color_data,
                    n,
                    camera,
                    target_color,
                    target_depth,
                )?;
                color_data[idx] += eps;
                color_grad[idx] = (plus - minus) / (2.0 * eps);
            }
        }

        let surrogate_grads = self.renderer.compute_surrogate_gradients(gaussians)?;
        blend_gradients(&mut pos_grad, &surrogate_grads.positions, 0.2);
        blend_gradients(&mut scale_grad, &surrogate_grads.scales, 0.2);
        blend_gradients(&mut op_grad, &surrogate_grads.opacities, 0.2);
        blend_gradients(&mut color_grad, &surrogate_grads.colors, 0.2);
        let mut rot_grad = surrogate_grads.rotations;
        if rot_grad.len() != n * 4 {
            rot_grad = vec![0.0; n * 4];
        }

        Ok((
            loss_value, pos_grad, scale_grad, rot_grad, op_grad, color_grad,
        ))
    }

    fn evaluate_loss(
        &mut self,
        positions: &[f32],
        scales: &[f32],
        rotations: &[f32],
        opacities: &[f32],
        colors: &[f32],
        n: usize,
        camera: &DiffCamera,
        target_color: &[f32],
        target_depth: &[f32],
    ) -> candle_core::Result<f32> {
        let temp = TrainableGaussians::new(
            positions,
            scales,
            rotations,
            opacities,
            colors,
            &self.device,
        )?;
        let output = self.renderer.render(&temp, camera)?;
        let loss = self
            .renderer
            .compute_loss(&output, target_color, target_depth)?;
        let value = loss.total.to_vec0::<f32>()?;
        if !value.is_finite() {
            return Ok(0.0);
        }
        if temp.len() != n {
            return Ok(0.0);
        }
        Ok(value)
    }

    /// Train on multiple frames
    pub fn train(
        &mut self,
        gaussians: &mut TrainableGaussians,
        cameras: &[DiffCamera],
        colors: &[&[f32]],
        depths: &[&[f32]],
        max_iterations: usize,
    ) -> candle_core::Result<TrainingResult> {
        let num_frames = cameras.len();
        let train_start = Instant::now();

        log::info!(
            "Training on {} frames, max {} iterations, initial gaussians={}",
            num_frames,
            max_iterations,
            gaussians.len()
        );

        for iter in 0..max_iterations {
            // Sample a frame
            let frame_idx = iter % num_frames;
            let should_log = iter < 5 || iter % 100 == 0;

            if should_log {
                log::info!(
                    "Starting iter {:5}/{:5} | frame {:3}/{:3} | gaussians={}",
                    iter,
                    max_iterations,
                    frame_idx + 1,
                    num_frames,
                    gaussians.len()
                );
            }
            let step_start = Instant::now();

            // Training step
            let loss = self.training_step(
                gaussians,
                &cameras[frame_idx],
                colors[frame_idx],
                depths[frame_idx],
            )?;

            // Print progress
            if should_log {
                log::info!(
                    "Iter {:5} done | Loss: {:.6} | step_time={:.2}s | elapsed={:.2}s",
                    iter,
                    loss,
                    step_start.elapsed().as_secs_f64(),
                    train_start.elapsed().as_secs_f64()
                );
            }
        }

        let final_loss = self.loss_history.last().copied().unwrap_or(0.0);

        log::info!("Training complete! Final loss: {:.6}", final_loss);

        Ok(TrainingResult {
            final_loss,
            iterations: max_iterations,
            num_gaussians: gaussians.len(),
            loss_history: self.loss_history.clone(),
        })
    }

    /// Train with optional checkpoint manager.
    pub fn train_with_checkpoints<P: AsRef<std::path::Path>>(
        &mut self,
        gaussians: &mut TrainableGaussians,
        cameras: &[DiffCamera],
        colors: &[&[f32]],
        depths: &[&[f32]],
        max_iterations: usize,
        mut checkpoint_manager: Option<&mut crate::io::TrainingCheckpoint>,
        checkpoint_interval: usize,
        checkpoint_path: P,
    ) -> candle_core::Result<TrainingResult> {
        let num_frames = cameras.len();
        let train_start = Instant::now();

        for iter in 0..max_iterations {
            let frame_idx = iter % num_frames;
            let should_log = iter < 5 || iter % 100 == 0;

            if should_log {
                log::info!(
                    "Starting iter {:5}/{:5} | frame {:3}/{:3} | gaussians={}",
                    iter,
                    max_iterations,
                    frame_idx + 1,
                    num_frames,
                    gaussians.len()
                );
            }
            let step_start = Instant::now();
            let loss = self.training_step(
                gaussians,
                &cameras[frame_idx],
                colors[frame_idx],
                depths[frame_idx],
            )?;

            if should_log {
                log::info!(
                    "Iter {:5} done | Loss: {:.6} | step_time={:.2}s | elapsed={:.2}s",
                    iter,
                    loss,
                    step_start.elapsed().as_secs_f64(),
                    train_start.elapsed().as_secs_f64()
                );
            }

            if let Some(manager) = checkpoint_manager.as_mut() {
                if iter % checkpoint_interval == 0 && iter > 0 {
                    manager.iteration = self.iteration;
                    manager.loss = loss;
                    if let Err(e) = manager.save(checkpoint_path.as_ref()) {
                        log::warn!("Failed to save training checkpoint: {}", e);
                    }
                }
            }
        }

        let final_loss = self.loss_history.last().copied().unwrap_or(0.0);
        Ok(TrainingResult {
            final_loss,
            iterations: max_iterations,
            num_gaussians: gaussians.len(),
            loss_history: self.loss_history.clone(),
        })
    }

    pub fn export_adam_state(&self) -> TrainerAdamState {
        let tensor_to_vec = |t: &Option<Tensor>| -> Vec<f32> {
            match t {
                Some(tensor) => tensor
                    .flatten_all()
                    .and_then(|t| t.to_vec1::<f32>())
                    .unwrap_or_default(),
                None => Vec::new(),
            }
        };
        TrainerAdamState {
            m_pos: tensor_to_vec(&self.m_pos),
            m_scale: tensor_to_vec(&self.m_scale),
            m_rot: tensor_to_vec(&self.m_rot),
            m_op: tensor_to_vec(&self.m_op),
            m_color: tensor_to_vec(&self.m_color),
            v_pos: tensor_to_vec(&self.v_pos),
            v_scale: tensor_to_vec(&self.v_scale),
            v_rot: tensor_to_vec(&self.v_rot),
            v_op: tensor_to_vec(&self.v_op),
            v_color: tensor_to_vec(&self.v_color),
        }
    }

    pub fn import_adam_state(&mut self, state: TrainerAdamState) {
        let n = state.m_op.len();
        if n == 0 {
            return;
        }
        let d = &self.device;
        let vec_to_tensor = |data: &[f32], shape: &[usize]| -> Option<Tensor> {
            if data.is_empty() {
                return None;
            }
            Tensor::from_slice(data, shape, d).ok()
        };
        self.m_pos = vec_to_tensor(&state.m_pos, &[n, 3]);
        self.m_scale = vec_to_tensor(&state.m_scale, &[n, 3]);
        self.m_rot = vec_to_tensor(&state.m_rot, &[n, 4]);
        self.m_op = vec_to_tensor(&state.m_op, &[n]);
        self.m_color = vec_to_tensor(&state.m_color, &[n, 3]);
        self.v_pos = vec_to_tensor(&state.v_pos, &[n, 3]);
        self.v_scale = vec_to_tensor(&state.v_scale, &[n, 3]);
        self.v_rot = vec_to_tensor(&state.v_rot, &[n, 4]);
        self.v_op = vec_to_tensor(&state.v_op, &[n]);
        self.v_color = vec_to_tensor(&state.v_color, &[n, 3]);
    }

    pub fn iteration(&self) -> usize {
        self.iteration
    }

    pub fn set_iteration(&mut self, iteration: usize) {
        self.iteration = iteration;
    }

    pub fn loss_history(&self) -> &[f32] {
        &self.loss_history
    }

    pub fn set_loss_history(&mut self, history: Vec<f32>) {
        self.loss_history = history;
    }
}

/// GPU Adam update step: uploads gradient once, does all math on GPU, updates Var in-place.
#[cfg(feature = "gpu")]
fn gpu_adam_step(
    var: &Var,
    grad_data: &[f32],
    m: &mut Tensor,
    v: &mut Tensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    step: usize,
    device: &Device,
) -> candle_core::Result<()> {
    let dims: Vec<usize> = var.as_tensor().dims().to_vec();
    let grad = Tensor::from_slice(grad_data, dims.as_slice(), device)?;

    // m = beta1 * m + (1 - beta1) * grad
    *m = m
        .affine(beta1 as f64, 0.0)?
        .broadcast_add(&grad.affine((1.0 - beta1) as f64, 0.0)?)?;

    // v = beta2 * v + (1 - beta2) * grad^2
    *v = v
        .affine(beta2 as f64, 0.0)?
        .broadcast_add(&grad.sqr()?.affine((1.0 - beta2) as f64, 0.0)?)?;

    // Bias correction
    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);

    let m_hat = m.affine(1.0 / bc1 as f64, 0.0)?;
    let v_hat = v.affine(1.0 / bc2 as f64, 0.0)?;

    // update = lr * m_hat / (sqrt(v_hat) + eps)
    let eps = Tensor::new(1e-8f32, device)?;
    let denom = v_hat.sqrt()?.broadcast_add(&eps)?;
    let update = m_hat.broadcast_div(&denom)?.affine(lr as f64, 0.0)?;

    // param -= update
    let new_val = var.as_tensor().sub(&update)?;
    var.set(&new_val)?;

    Ok(())
}

/// Adam update step (standalone CPU function, kept for external use)
pub fn adam_update(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    iteration: usize,
) {
    let eps = 1e-8;
    let bias_correction1 = 1.0 - beta1.powi(iteration as i32);
    let bias_correction2 = 1.0 - beta2.powi(iteration as i32);

    (
        param.par_iter_mut(),
        grad.par_iter(),
        m.par_iter_mut(),
        v.par_iter_mut(),
    )
        .into_par_iter()
        .for_each(|(p, &g, m_i, v_i)| {
            // Update biased first moment estimate
            let old_m = *m_i;
            *m_i = beta1 * old_m + (1.0 - beta1) * g;

            // Update biased second raw moment estimate
            let old_v = *v_i;
            *v_i = beta2 * old_v + (1.0 - beta2) * g * g;

            // Compute bias-corrected moment estimates
            let m_hat = *m_i / bias_correction1;
            let v_hat = *v_i / bias_correction2;

            // Update parameters
            *p -= lr * m_hat / (v_hat.sqrt() + eps);
        });
}

fn flatten_2d(data: &[Vec<f32>]) -> Vec<f32> {
    data.iter().flatten().copied().collect()
}

fn blend_gradients(dst: &mut [f32], src: &[f32], weight: f32) {
    if dst.len() != src.len() {
        return;
    }
    let w = weight.clamp(0.0, 1.0);
    for i in 0..dst.len() {
        dst[i] = (1.0 - w) * dst[i] + w * src[i];
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_creation() {
        let trainer = CompleteTrainer::new(640, 480, 0.00016, 0.005, 0.001, 0.05, 0.0025);

        assert_eq!(trainer.width, 640);
        assert_eq!(trainer.height, 480);
    }

    #[test]
    fn test_lr_scheduler() {
        let scheduler = LrScheduler::new(0.001, 100, 1000);

        // Warmup
        assert!(scheduler.get_lr(0) < scheduler.get_lr(50));
        assert!(scheduler.get_lr(50) < scheduler.get_lr(100));

        // After warmup (should decay)
        assert!(scheduler.get_lr(100) > scheduler.get_lr(500));
    }

    #[test]
    fn test_adam_update() {
        let mut param = vec![0.0, 0.0, 0.0];
        let grad = vec![0.1, -0.2, 0.15];
        let mut m = vec![0.0, 0.0, 0.0];
        let mut v = vec![1e-8, 1e-8, 1e-8];

        adam_update(&mut param, &grad, &mut m, &mut v, 0.001, 0.9, 0.999, 1);

        // Parameters should have moved
        assert!(param[0] != 0.0 || param[1] != 0.0 || param[2] != 0.0);
    }
}
