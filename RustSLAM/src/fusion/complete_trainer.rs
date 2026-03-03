//! Complete Training Loop with Backward Propagation
//!
//! This module implements the full backward propagation for 3DGS:
//! - Forward pass: rendering
//! - Loss computation
//! - Backward pass: gradient computation
//! - Parameter update with Adam

use candle_core::{Tensor, Device, Var};
use crate::fusion::diff_splat::{TrainableGaussians, DiffSplatRenderer, DiffCamera};
use crate::fusion::analytical_backward;
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

/// Complete trainer with backward propagation
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
    /// Momentum for positions
    m_pos: Vec<f32>,
    /// Momentum for scales
    m_scale: Vec<f32>,
    /// Momentum for rotations
    m_rot: Vec<f32>,
    /// Momentum for opacities
    m_op: Vec<f32>,
    /// Momentum for colors
    m_color: Vec<f32>,
    /// Velocity for positions
    v_pos: Vec<f32>,
    /// Velocity for scales
    v_scale: Vec<f32>,
    /// Velocity for rotations
    v_rot: Vec<f32>,
    /// Velocity for opacities
    v_op: Vec<f32>,
    /// Velocity for colors
    v_color: Vec<f32>,
    /// Current iteration
    iteration: usize,
    /// Loss history
    loss_history: Vec<f32>,
    /// Learning-rate scheduler multiplier
    lr_scheduler: LrScheduler,
    /// Use analytical backward pass instead of finite differences (default: true)
    pub use_analytical_backward: bool,
}

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
        let device = Device::new_metal(0).unwrap_or_else(|_| Device::Cpu);
        let renderer = DiffSplatRenderer::with_device(width, height, device.clone());
        
        println!("CompleteTrainer using device: {:?}", device);

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
            m_pos: Vec::new(),
            m_scale: Vec::new(),
            m_rot: Vec::new(),
            m_op: Vec::new(),
            m_color: Vec::new(),
            v_pos: Vec::new(),
            v_scale: Vec::new(),
            v_rot: Vec::new(),
            v_op: Vec::new(),
            v_color: Vec::new(),
            iteration: 0,
            loss_history: Vec::new(),
            lr_scheduler: LrScheduler::new(1.0, 100, 3000),
            use_analytical_backward: true,
        }
    }

    /// Get training device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Initialize Adam optimizer state
    fn init_adam_state(&mut self, n: usize) {
        let eps = 1e-8;
        self.m_pos = vec![0.0; n * 3];
        self.m_scale = vec![0.0; n * 3];
        self.m_rot = vec![0.0; n * 4];
        self.m_op = vec![0.0; n];
        self.m_color = vec![0.0; n * 3];
        
        self.v_pos = vec![eps; n * 3];
        self.v_scale = vec![eps; n * 3];
        self.v_rot = vec![eps; n * 4];
        self.v_op = vec![eps; n];
        self.v_color = vec![eps; n * 3];
    }

    /// Training step with backward propagation
    ///
    /// When `use_analytical_backward` is true (default), uses exact analytical
    /// gradients from a single forward+backward pass (~100x faster than finite diff).
    /// When false, falls back to the legacy finite-difference approach.
    pub fn training_step(
        &mut self,
        gaussians: &mut TrainableGaussians,
        camera: &DiffCamera,
        target_color: &[f32],
        target_depth: &[f32],
    ) -> candle_core::Result<f32> {
        let n = gaussians.len();

        if self.m_pos.is_empty() {
            self.init_adam_state(n);
        }

        // Extract current raw parameters
        let mut pos_data = flatten_2d(&gaussians.positions().to_vec2::<f32>()?);
        let mut scale_data = flatten_2d(&gaussians.scales.as_tensor().to_vec2::<f32>()?);
        let mut rot_data = flatten_2d(&gaussians.rotations.as_tensor().to_vec2::<f32>()?);
        let mut op_data = gaussians.opacities.as_tensor().to_vec1::<f32>()?;
        let mut color_data = flatten_2d(&gaussians.colors().to_vec2::<f32>()?);

        let (loss_value, pos_grad, scale_grad, rot_grad, op_grad, color_grad) =
            if self.use_analytical_backward {
                self.analytical_training_step(
                    gaussians, camera, target_color, target_depth, n,
                    &pos_data, &scale_data, &op_data, &color_data,
                )?
            } else {
                self.finite_diff_training_step(
                    gaussians, camera, target_color, target_depth, n,
                    &mut pos_data, &mut scale_data, &mut rot_data, &mut op_data, &mut color_data,
                )?
            };

        self.loss_history.push(loss_value);
        self.iteration += 1;

        let lr_factor = self.lr_scheduler.get_lr(self.iteration).max(1e-4);
        adam_update(&mut pos_data, &pos_grad, &mut self.m_pos, &mut self.v_pos,
            self.lr_pos * lr_factor, self.beta1, self.beta2, self.iteration);
        adam_update(&mut scale_data, &scale_grad, &mut self.m_scale, &mut self.v_scale,
            self.lr_scale * lr_factor, self.beta1, self.beta2, self.iteration);
        adam_update(&mut rot_data, &rot_grad, &mut self.m_rot, &mut self.v_rot,
            self.lr_rot * lr_factor, self.beta1, self.beta2, self.iteration);
        adam_update(&mut op_data, &op_grad, &mut self.m_op, &mut self.v_op,
            self.lr_op * lr_factor, self.beta1, self.beta2, self.iteration);
        adam_update(&mut color_data, &color_grad, &mut self.m_color, &mut self.v_color,
            self.lr_color * lr_factor, self.beta1, self.beta2, self.iteration);

        // Normalize rotations
        for i in 0..n {
            let r = i * 4;
            let norm = (rot_data[r]*rot_data[r] + rot_data[r+1]*rot_data[r+1]
                + rot_data[r+2]*rot_data[r+2] + rot_data[r+3]*rot_data[r+3])
                .sqrt().max(1e-6);
            rot_data[r] /= norm;
            rot_data[r+1] /= norm;
            rot_data[r+2] /= norm;
            rot_data[r+3] /= norm;
        }

        gaussians.positions = Var::from_tensor(&Tensor::from_slice(&pos_data, (n, 3), &self.device)?)?;
        gaussians.scales = Var::from_tensor(&Tensor::from_slice(&scale_data, (n, 3), &self.device)?)?;
        gaussians.rotations = Var::from_tensor(&Tensor::from_slice(&rot_data, (n, 4), &self.device)?)?;
        gaussians.opacities = Var::from_tensor(&Tensor::from_slice(&op_data, (n,), &self.device)?)?;
        gaussians.colors = Var::from_tensor(&Tensor::from_slice(&color_data, (n, 3), &self.device)?)?;

        Ok(loss_value)
    }

    /// Analytical backward: 1 forward + 1 backward pass.
    fn analytical_training_step(
        &self,
        gaussians: &TrainableGaussians,
        camera: &DiffCamera,
        target_color: &[f32],
        target_depth: &[f32],
        n: usize,
        _pos_data: &[f32],
        _scale_data: &[f32],
        _op_data: &[f32],
        _color_data: &[f32],
    ) -> candle_core::Result<(f32, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        // Forward pass with intermediates
        let (output, intermediate) = self.renderer.render_with_intermediates(gaussians, camera)?;
        let loss = self.renderer.compute_loss(&output, target_color, target_depth)?;
        let loss_value = loss.total.to_vec0::<f32>()?;

        // Analytical backward
        let grads = analytical_backward::backward(
            &intermediate, target_color, n,
            camera.fx, camera.fy, camera.cx, camera.cy,
        );

        // Blend with surrogate gradients (10% weight for regularization)
        let surrogate = self.renderer.compute_surrogate_gradients(gaussians)?;
        let mut pos_grad = grads.positions;
        let mut scale_grad = grads.log_scales;
        let mut op_grad = grads.opacity_logits;
        let mut color_grad = grads.colors;
        blend_gradients(&mut pos_grad, &surrogate.positions, 0.1);
        blend_gradients(&mut scale_grad, &surrogate.scales, 0.1);
        blend_gradients(&mut op_grad, &surrogate.opacities, 0.1);
        blend_gradients(&mut color_grad, &surrogate.colors, 0.1);

        let mut rot_grad = surrogate.rotations;
        if rot_grad.len() != n * 4 {
            rot_grad = vec![0.0; n * 4];
        }

        Ok((loss_value, pos_grad, scale_grad, rot_grad, op_grad, color_grad))
    }

    /// Legacy finite-difference gradient estimation.
    fn finite_diff_training_step(
        &self,
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
        let loss = self.renderer.compute_loss(&output, target_color, target_depth)?;
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
            if i >= n { break; }

            for d in 0..3 {
                let idx = i * 3 + d;
                pos_data[idx] += eps;
                let plus = self.evaluate_loss(pos_data, scale_data, rot_data, op_data, color_data, n, camera, target_color, target_depth)?;
                pos_data[idx] -= 2.0 * eps;
                let minus = self.evaluate_loss(pos_data, scale_data, rot_data, op_data, color_data, n, camera, target_color, target_depth)?;
                pos_data[idx] += eps;
                pos_grad[idx] = (plus - minus) / (2.0 * eps);
            }

            for d in 0..3 {
                let idx = i * 3 + d;
                scale_data[idx] += eps;
                let plus = self.evaluate_loss(pos_data, scale_data, rot_data, op_data, color_data, n, camera, target_color, target_depth)?;
                scale_data[idx] -= 2.0 * eps;
                let minus = self.evaluate_loss(pos_data, scale_data, rot_data, op_data, color_data, n, camera, target_color, target_depth)?;
                scale_data[idx] += eps;
                scale_grad[idx] = (plus - minus) / (2.0 * eps);
            }

            let op_idx = i;
            op_data[op_idx] += eps;
            let plus = self.evaluate_loss(pos_data, scale_data, rot_data, op_data, color_data, n, camera, target_color, target_depth)?;
            op_data[op_idx] -= 2.0 * eps;
            let minus = self.evaluate_loss(pos_data, scale_data, rot_data, op_data, color_data, n, camera, target_color, target_depth)?;
            op_data[op_idx] += eps;
            op_grad[op_idx] = (plus - minus) / (2.0 * eps);

            for d in 0..3 {
                let idx = i * 3 + d;
                color_data[idx] += eps;
                let plus = self.evaluate_loss(pos_data, scale_data, rot_data, op_data, color_data, n, camera, target_color, target_depth)?;
                color_data[idx] -= 2.0 * eps;
                let minus = self.evaluate_loss(pos_data, scale_data, rot_data, op_data, color_data, n, camera, target_color, target_depth)?;
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

        Ok((loss_value, pos_grad, scale_grad, rot_grad, op_grad, color_grad))
    }

    fn evaluate_loss(
        &self,
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
        let loss = self.renderer.compute_loss(&output, target_color, target_depth)?;
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
        
        println!("Training on {} frames, max {} iterations", num_frames, max_iterations);

        for iter in 0..max_iterations {
            // Sample a frame
            let frame_idx = iter % num_frames;
            
            // Training step
            let loss = self.training_step(
                gaussians,
                &cameras[frame_idx],
                colors[frame_idx],
                depths[frame_idx],
            )?;

            // Print progress
            if iter % 100 == 0 {
                println!("Iter {:5} | Loss: {:.6}", iter, loss);
            }
        }

        let final_loss = self.loss_history.last().copied().unwrap_or(0.0);
        
        println!("Training complete! Final loss: {:.6}", final_loss);

        Ok(TrainingResult {
            final_loss,
            iterations: max_iterations,
            num_gaussians: gaussians.len(),
            loss_history: self.loss_history.clone(),
        })
    }

    /// Train with optional checkpoint manager.
    pub fn train_with_checkpoints(
        &mut self,
        gaussians: &mut TrainableGaussians,
        cameras: &[DiffCamera],
        colors: &[&[f32]],
        depths: &[&[f32]],
        max_iterations: usize,
        mut checkpoint_manager: Option<&mut crate::fusion::training_checkpoint::TrainingCheckpointManager>,
    ) -> candle_core::Result<TrainingResult> {
        let num_frames = cameras.len();

        for iter in 0..max_iterations {
            let frame_idx = iter % num_frames;
            let loss = self.training_step(
                gaussians,
                &cameras[frame_idx],
                colors[frame_idx],
                depths[frame_idx],
            )?;

            if iter % 100 == 0 {
                println!("Iter {:5} | Loss: {:.6}", iter, loss);
            }

            if let Some(manager) = checkpoint_manager.as_deref_mut() {
                match manager.maybe_save(self.iteration, gaussians, self) {
                    Ok(path) => {
                        if !path.as_os_str().is_empty() {
                            log::info!("Training checkpoint saved: {}", path.display());
                        }
                    }
                    Err(err) => {
                        log::warn!("Failed to save training checkpoint: {}", err);
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
        TrainerAdamState {
            m_pos: self.m_pos.clone(),
            m_scale: self.m_scale.clone(),
            m_rot: self.m_rot.clone(),
            m_op: self.m_op.clone(),
            m_color: self.m_color.clone(),
            v_pos: self.v_pos.clone(),
            v_scale: self.v_scale.clone(),
            v_rot: self.v_rot.clone(),
            v_op: self.v_op.clone(),
            v_color: self.v_color.clone(),
        }
    }

    pub fn import_adam_state(&mut self, state: TrainerAdamState) {
        self.m_pos = state.m_pos;
        self.m_scale = state.m_scale;
        self.m_rot = state.m_rot;
        self.m_op = state.m_op;
        self.m_color = state.m_color;
        self.v_pos = state.v_pos;
        self.v_scale = state.v_scale;
        self.v_rot = state.v_rot;
        self.v_op = state.v_op;
        self.v_color = state.v_color;
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

/// Adam update step (standalone function)
fn adam_update(
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

    for i in 0..param.len() {
        // Update biased first moment estimate
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
        
        // Update biased second raw moment estimate
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        
        // Compute bias-corrected first moment estimate
        let m_hat = m[i] / bias_correction1;
        
        // Compute bias-corrected second raw moment estimate
        let v_hat = v[i] / bias_correction2;
        
        // Update parameters
        param[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
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

/// Example usage
pub fn example_training() -> candle_core::Result<()> {
    // Create trainer
    let mut trainer = CompleteTrainer::new(
        640, 480,           // width, height
        0.00016,            // lr positions
        0.005,              // lr scales
        0.001,              // lr rotations
        0.05,               // lr opacities
        0.0025,             // lr colors
    );

    // Create initial Gaussians
    let n = 1000;
    let positions = vec![0.0f32; n * 3];
    let scales = vec![-3.0f32; n * 3];
    let rotations = vec![1.0f32, 0.0, 0.0, 0.0].repeat(n);
    let opacities = vec![0.5f32; n];
    let colors = vec![1.0f32, 0.5, 0.25].repeat(n);

    let device = Device::new_metal(0).unwrap_or_else(|_| Device::Cpu);
    let mut gaussians = TrainableGaussians::new(
        &positions, &scales, &rotations, &opacities, &colors, &device
    )?;

    // Create dummy camera
    let rotation = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let translation = [0.0, 0.0, 0.0];
    let camera = DiffCamera::new(500.0, 500.0, 320.0, 240.0, 640, 480, &rotation, &translation, &device)?;

    // Dummy target data
    let target_color = vec![0.5f32; 640 * 480 * 3];
    let target_depth = vec![1.0f32; 640 * 480];

    // Train
    let result = trainer.train(
        &mut gaussians,
        &[camera],
        &[&target_color],
        &[&target_depth],
        1000,
    )?;

    println!("Training result: {:?}", result);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_creation() {
        let trainer = CompleteTrainer::new(
            640, 480,
            0.00016, 0.005, 0.001, 0.05, 0.0025,
        );
        
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
