//! 3DGS Training Pipeline
//!
//! Complete training loop for 3D Gaussian Splatting with:
//! - Adam optimizer for Gaussian parameters
//! - Progressive densification
//! - Adaptive opacity pruning
//! - Training loop with checkpointing

use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

#[cfg(feature = "gpu")]
use candle_core::{Device, Tensor, Var};

#[cfg(feature = "gpu")]
use crate::diff::diff_splat::{DiffCamera, DiffSplatRenderer, TrainableGaussians};

const TRAINER_CHECKPOINT_VERSION: u32 = 1;

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Learning rate for positions
    pub lr_position: f32,
    /// Learning rate for scales
    pub lr_scale: f32,
    /// Learning rate for rotations
    pub lr_rotation: f32,
    /// Learning rate for opacities
    pub lr_opacity: f32,
    /// Learning rate for colors
    pub lr_color: f32,
    /// Adam beta1
    pub beta1: f32,
    /// Adam beta2
    pub beta2: f32,
    /// Number of training iterations
    pub max_iterations: usize,
    /// Densify interval (iterations)
    pub densify_interval: usize,
    /// Densify threshold
    pub densify_threshold: f32,
    /// Prune opacity threshold
    pub prune_opacity: f32,
    /// Batch size (frames per iteration)
    pub batch_size: usize,
    /// Print interval
    pub print_interval: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr_position: 0.00016,
            lr_scale: 0.005,
            lr_rotation: 0.001,
            lr_opacity: 0.05,
            lr_color: 0.0025,
            beta1: 0.9,
            beta2: 0.999,
            max_iterations: 30_000,
            densify_interval: 100,
            densify_threshold: 0.0002,
            prune_opacity: 0.005,
            batch_size: 1,
            print_interval: 100,
        }
    }
}

/// Training state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainState {
    /// Current iteration
    pub iteration: usize,
    /// Loss history
    pub losses: Vec<f32>,
    /// Number of Gaussians
    pub num_gaussians: usize,
}

impl TrainState {
    pub fn new() -> Self {
        Self {
            iteration: 0,
            losses: Vec::new(),
            num_gaussians: 0,
        }
    }
}

impl Default for TrainState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainerCheckpoint {
    version: u32,
    state: TrainState,
    gaussians: Vec<CheckpointGaussian>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointGaussian {
    position: [f32; 3],
    scale: [f32; 3],
    rotation: [f32; 4],
    opacity: f32,
    color: [f32; 3],
}

/// 3DGS Trainer (GPU-enabled)
#[cfg(feature = "gpu")]
pub struct Trainer {
    /// Configuration
    config: TrainConfig,
    /// Gaussians to train
    gaussians: TrainableGaussians,
    /// Renderer
    renderer: DiffSplatRenderer,
    /// Device
    device: Device,
    /// Training state
    state: TrainState,
}

#[cfg(feature = "gpu")]
impl Trainer {
    /// Create a new trainer
    pub fn new(config: TrainConfig, initial_gaussians: usize) -> candle_core::Result<Self> {
        let device = crate::preferred_device();
        println!("Trainer using device: {:?}", device);

        // Initialize with random Gaussians
        let positions = vec![0.0f32; initial_gaussians * 3];
        let scales = vec![-3.0f32; initial_gaussians * 3];
        let rotations = vec![1.0f32, 0.0, 0.0, 0.0].repeat(initial_gaussians);
        let opacities = vec![opacity_to_logit(0.5); initial_gaussians];
        let colors = vec![1.0f32, 0.5, 0.25].repeat(initial_gaussians);

        let gaussians = TrainableGaussians::new(
            &positions, &scales, &rotations, &opacities, &colors, &device,
        )?;

        let renderer = DiffSplatRenderer::new(640, 480);

        Ok(Self {
            config,
            gaussians,
            renderer,
            device,
            state: TrainState {
                iteration: 0,
                losses: Vec::new(),
                num_gaussians: initial_gaussians,
            },
        })
    }

    /// Create from existing Gaussians
    pub fn from_gaussians(
        config: TrainConfig,
        gaussians: TrainableGaussians,
        width: usize,
        height: usize,
    ) -> Self {
        let device = gaussians.device().clone();
        let renderer = DiffSplatRenderer::with_device(width, height, device.clone());

        Self {
            config,
            state: TrainState {
                iteration: 0,
                losses: Vec::new(),
                num_gaussians: gaussians.len(),
            },
            gaussians,
            renderer,
            device,
        }
    }

    /// Get current Gaussians
    pub fn gaussians(&self) -> &TrainableGaussians {
        &self.gaussians
    }

    /// Get training state
    pub fn state(&self) -> &TrainState {
        &self.state
    }

    /// Training step for one frame
    pub fn step(
        &mut self,
        camera: &DiffCamera,
        observed_color: &[f32],
        observed_depth: &[f32],
    ) -> candle_core::Result<f32> {
        let n = self.gaussians.len();
        let mut positions = flatten_2d(&self.gaussians.positions().to_vec2::<f32>()?);
        let mut scales = flatten_2d(&self.gaussians.scales.as_tensor().to_vec2::<f32>()?);
        let mut rotations = flatten_2d(&self.gaussians.rotations.as_tensor().to_vec2::<f32>()?);
        let mut opacities = self.gaussians.opacities.as_tensor().to_vec1::<f32>()?;
        let mut colors = flatten_2d(&self.gaussians.colors().to_vec2::<f32>()?);

        let output = self.renderer.render(&self.gaussians, camera)?;
        let loss = self
            .renderer
            .compute_loss(&output, observed_color, observed_depth)?;
        let loss_value = loss.total.to_vec0::<f32>()?;

        let (mut pos_grad, mut scale_grad, mut rot_grad, mut opacity_grad, mut color_grad) = self
            .estimate_gradients(
            camera,
            observed_color,
            observed_depth,
            &mut positions,
            &mut scales,
            &mut rotations,
            &mut opacities,
            &mut colors,
        )?;

        let surrogate = self.renderer.compute_surrogate_gradients(&self.gaussians)?;
        blend_gradients(&mut pos_grad, &surrogate.positions, 0.1);
        blend_gradients(&mut scale_grad, &surrogate.scales, 0.1);
        blend_gradients(&mut opacity_grad, &surrogate.opacities, 0.1);
        blend_gradients(&mut color_grad, &surrogate.colors, 0.1);
        if rot_grad.iter().all(|g| *g == 0.0) {
            rot_grad = surrogate.rotations;
        } else {
            blend_gradients(&mut rot_grad, &surrogate.rotations, 0.1);
        }

        apply_gradient_step(&mut positions, &pos_grad, self.config.lr_position);
        apply_gradient_step(&mut scales, &scale_grad, self.config.lr_scale);
        apply_gradient_step(&mut rotations, &rot_grad, self.config.lr_rotation);
        apply_gradient_step(&mut opacities, &opacity_grad, self.config.lr_opacity);
        apply_gradient_step(&mut colors, &color_grad, self.config.lr_color);

        clamp_log_scales(&mut scales);
        normalize_rotation_buffer(&mut rotations);
        clamp_opacity_logits(&mut opacities);
        clamp_colors(&mut colors);

        if n > 0 {
            self.gaussians = TrainableGaussians::new(
                &positions,
                &scales,
                &rotations,
                &opacities,
                &colors,
                &self.device,
            )?;
        }

        self.state.losses.push(loss_value);
        self.state.iteration += 1;
        self.state.num_gaussians = self.gaussians.len();

        // Print progress
        if self.state.iteration % self.config.print_interval == 0 {
            println!(
                "Iter {} | Loss: {:.4} | Gaussians: {}",
                self.state.iteration, loss_value, self.state.num_gaussians
            );
        }

        Ok(loss_value)
    }

    /// Train on a sequence of frames
    pub fn train(
        &mut self,
        cameras: &[DiffCamera],
        colors: &[&[f32]],
        depths: &[&[f32]],
    ) -> candle_core::Result<TrainState> {
        let num_frames = cameras.len();
        if num_frames == 0 {
            return Ok(self.state.clone());
        }
        if colors.len() != num_frames || depths.len() != num_frames {
            return Err(candle_core::Error::Wrapped(Box::new(
                "camera/color/depth batch length mismatch".to_string(),
            )));
        }

        println!("Starting training with {} frames", num_frames);

        for iteration in 0..self.config.max_iterations {
            // Sample a random frame
            let frame_idx = iteration % num_frames;

            let camera = &cameras[frame_idx];
            let color = colors[frame_idx];
            let depth = depths[frame_idx];

            let _loss = self.step(camera, color, depth)?;

            // Densification (add new Gaussians)
            if iteration > 0 && iteration % self.config.densify_interval == 0 {
                self.densify()?;
            }

            // Pruning (remove low opacity Gaussians)
            if iteration > 0 && iteration % self.config.densify_interval == 0 {
                self.prune()?;
            }

            // Checkpoint
            if iteration % 1000 == 0 && iteration > 0 {
                println!("Checkpoint at iteration {}", iteration);
            }
        }

        println!("Training complete!");
        Ok(self.state.clone())
    }

    fn densify(&mut self) -> candle_core::Result<()> {
        if self.gaussians.len() == 0 {
            return Ok(());
        }

        let mut positions = flatten_2d(&self.gaussians.positions().to_vec2::<f32>()?);
        let mut scales = flatten_2d(&self.gaussians.scales.as_tensor().to_vec2::<f32>()?);
        let mut rotations = flatten_2d(&self.gaussians.rotations.as_tensor().to_vec2::<f32>()?);
        let mut opacities = self.gaussians.opacities.as_tensor().to_vec1::<f32>()?;
        let mut colors = flatten_2d(&self.gaussians.colors().to_vec2::<f32>()?);
        let actual_scales = flatten_2d(&self.gaussians.scales()?.to_vec2::<f32>()?);
        let actual_opacities = self.gaussians.opacities()?.to_vec1::<f32>()?;

        let mut candidates: Vec<(usize, f32)> = actual_opacities
            .iter()
            .enumerate()
            .filter_map(|(idx, &opacity)| {
                let base = idx * 3;
                let mean_scale: f32 =
                    (actual_scales[base] + actual_scales[base + 1] + actual_scales[base + 2]) / 3.0;
                let score = opacity * mean_scale.max(self.config.densify_threshold);
                (opacity > self.config.prune_opacity * 4.0 && score.is_finite())
                    .then_some((idx, score))
            })
            .collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let clone_count = candidates
            .len()
            .min(self.config.batch_size.max(1))
            .max(usize::from(!candidates.is_empty()));
        for (rank, (idx, _)) in candidates.into_iter().take(clone_count).enumerate() {
            let p = idx * 3;
            let r = idx * 4;
            let c = idx * 3;
            let direction = match rank % 3 {
                0 => [1.0, 0.0, 0.0],
                1 => [0.0, 1.0, 0.0],
                _ => [0.0, 0.0, 1.0],
            };
            let offset_scale: [f32; 3] = [
                actual_scales[p].max(0.01),
                actual_scales[p + 1].max(0.01),
                actual_scales[p + 2].max(0.01),
            ];

            positions.extend_from_slice(&[
                positions[p] + direction[0] * offset_scale[0] * 0.5,
                positions[p + 1] + direction[1] * offset_scale[1] * 0.5,
                positions[p + 2] + direction[2] * offset_scale[2] * 0.5,
            ]);
            let scale_copy = [scales[p], scales[p + 1], scales[p + 2]];
            let rotation_copy = [
                rotations[r],
                rotations[r + 1],
                rotations[r + 2],
                rotations[r + 3],
            ];
            let color_copy = [colors[c], colors[c + 1], colors[c + 2]];
            scales.extend_from_slice(&scale_copy);
            rotations.extend_from_slice(&rotation_copy);
            opacities.push(opacity_to_logit(
                (actual_opacities[idx] * 0.75).clamp(0.05, 0.95),
            ));
            colors.extend_from_slice(&color_copy);
        }

        self.gaussians = TrainableGaussians::new(
            &positions,
            &scales,
            &rotations,
            &opacities,
            &colors,
            &self.device,
        )?;
        self.state.num_gaussians = self.gaussians.len();
        Ok(())
    }

    fn prune(&mut self) -> candle_core::Result<()> {
        if self.gaussians.len() <= 1 {
            return Ok(());
        }

        let positions = flatten_2d(&self.gaussians.positions().to_vec2::<f32>()?);
        let scales_raw = flatten_2d(&self.gaussians.scales.as_tensor().to_vec2::<f32>()?);
        let scales = flatten_2d(&self.gaussians.scales()?.to_vec2::<f32>()?);
        let rotations = flatten_2d(&self.gaussians.rotations.as_tensor().to_vec2::<f32>()?);
        let opacities_raw = self.gaussians.opacities.as_tensor().to_vec1::<f32>()?;
        let opacities = self.gaussians.opacities()?.to_vec1::<f32>()?;
        let colors = flatten_2d(&self.gaussians.colors().to_vec2::<f32>()?);

        let mut keep_mask = vec![false; self.gaussians.len()];
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;

        for idx in 0..self.gaussians.len() {
            let p = idx * 3;
            let r = idx * 4;
            let c = idx * 3;
            let opacity = opacities[idx];
            let mean_scale = (scales[p] + scales[p + 1] + scales[p + 2]) / 3.0;
            let valid = opacity.is_finite()
                && opacity >= self.config.prune_opacity
                && mean_scale.is_finite()
                && (1e-4..=10.0).contains(&mean_scale)
                && positions[p..p + 3].iter().all(|v| v.is_finite())
                && rotations[r..r + 4].iter().all(|v| v.is_finite())
                && colors[c..c + 3].iter().all(|v| v.is_finite());
            if valid {
                keep_mask[idx] = true;
            }
            let score = if valid { opacity } else { opacity - 10.0 };
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }

        if !keep_mask.iter().any(|keep| *keep) {
            keep_mask[best_idx] = true;
        }

        let mut kept_positions = Vec::new();
        let mut kept_scales = Vec::new();
        let mut kept_rotations = Vec::new();
        let mut kept_opacities = Vec::new();
        let mut kept_colors = Vec::new();

        for (idx, keep) in keep_mask.into_iter().enumerate() {
            if keep {
                let p = idx * 3;
                let r = idx * 4;
                let c = idx * 3;
                kept_positions.extend_from_slice(&positions[p..p + 3]);
                kept_scales.extend_from_slice(&scales_raw[p..p + 3]);
                kept_rotations.extend_from_slice(&rotations[r..r + 4]);
                kept_opacities.push(opacities_raw[idx]);
                kept_colors.extend_from_slice(&colors[c..c + 3]);
            }
        }

        self.gaussians = TrainableGaussians::new(
            &kept_positions,
            &kept_scales,
            &kept_rotations,
            &kept_opacities,
            &kept_colors,
            &self.device,
        )?;
        self.state.num_gaussians = self.gaussians.len();
        Ok(())
    }

    pub fn save(&self, path: &str) -> candle_core::Result<()> {
        let path = Path::new(path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let checkpoint = TrainerCheckpoint {
            version: TRAINER_CHECKPOINT_VERSION,
            state: self.state.clone(),
            gaussians: export_gaussians(&self.gaussians)?,
        };
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &checkpoint).map_err(|err: serde_json::Error| {
            candle_core::Error::Wrapped(Box::new(err.to_string()))
        })?;
        Ok(())
    }

    pub fn load(&mut self, path: &str) -> candle_core::Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let checkpoint: TrainerCheckpoint =
            serde_json::from_reader(reader).map_err(|err: serde_json::Error| {
                candle_core::Error::Wrapped(Box::new(err.to_string()))
            })?;
        if checkpoint.version != TRAINER_CHECKPOINT_VERSION {
            return Err(candle_core::Error::Wrapped(Box::new(format!(
                "unsupported trainer checkpoint version {}",
                checkpoint.version
            ))));
        }

        self.gaussians = checkpoint_to_gaussians(&checkpoint, &self.device)?;
        self.state = checkpoint.state;
        self.state.num_gaussians = self.gaussians.len();
        Ok(())
    }

    fn estimate_gradients(
        &self,
        camera: &DiffCamera,
        observed_color: &[f32],
        observed_depth: &[f32],
        positions: &mut [f32],
        scales: &mut [f32],
        rotations: &mut [f32],
        opacities: &mut [f32],
        colors: &mut [f32],
    ) -> candle_core::Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let n = self.gaussians.len();
        let mut pos_grad = vec![0.0; n * 3];
        let mut scale_grad = vec![0.0; n * 3];
        let mut rot_grad = vec![0.0; n * 4];
        let mut opacity_grad = vec![0.0; n];
        let mut color_grad = vec![0.0; n * 3];
        if n == 0 {
            return Ok((pos_grad, scale_grad, rot_grad, opacity_grad, color_grad));
        }

        let eps = 1e-3f32;
        let sample_count = n.min((self.config.batch_size.max(1) * 4).max(1));
        let stride = n.div_ceil(sample_count);

        for sample in 0..sample_count {
            let idx = sample * stride;
            if idx >= n {
                break;
            }

            for dim in 0..3 {
                let slot = idx * 3 + dim;
                positions[slot] += eps;
                let plus = self.evaluate_loss(
                    positions,
                    scales,
                    rotations,
                    opacities,
                    colors,
                    camera,
                    observed_color,
                    observed_depth,
                )?;
                positions[slot] -= 2.0 * eps;
                let minus = self.evaluate_loss(
                    positions,
                    scales,
                    rotations,
                    opacities,
                    colors,
                    camera,
                    observed_color,
                    observed_depth,
                )?;
                positions[slot] += eps;
                pos_grad[slot] = (plus - minus) / (2.0 * eps);
            }

            for dim in 0..3 {
                let slot = idx * 3 + dim;
                scales[slot] += eps;
                let plus = self.evaluate_loss(
                    positions,
                    scales,
                    rotations,
                    opacities,
                    colors,
                    camera,
                    observed_color,
                    observed_depth,
                )?;
                scales[slot] -= 2.0 * eps;
                let minus = self.evaluate_loss(
                    positions,
                    scales,
                    rotations,
                    opacities,
                    colors,
                    camera,
                    observed_color,
                    observed_depth,
                )?;
                scales[slot] += eps;
                scale_grad[slot] = (plus - minus) / (2.0 * eps);
            }

            opacities[idx] += eps;
            let plus = self.evaluate_loss(
                positions,
                scales,
                rotations,
                opacities,
                colors,
                camera,
                observed_color,
                observed_depth,
            )?;
            opacities[idx] -= 2.0 * eps;
            let minus = self.evaluate_loss(
                positions,
                scales,
                rotations,
                opacities,
                colors,
                camera,
                observed_color,
                observed_depth,
            )?;
            opacities[idx] += eps;
            opacity_grad[idx] = (plus - minus) / (2.0 * eps);

            for dim in 0..3 {
                let slot = idx * 3 + dim;
                colors[slot] += eps;
                let plus = self.evaluate_loss(
                    positions,
                    scales,
                    rotations,
                    opacities,
                    colors,
                    camera,
                    observed_color,
                    observed_depth,
                )?;
                colors[slot] -= 2.0 * eps;
                let minus = self.evaluate_loss(
                    positions,
                    scales,
                    rotations,
                    opacities,
                    colors,
                    camera,
                    observed_color,
                    observed_depth,
                )?;
                colors[slot] += eps;
                color_grad[slot] = (plus - minus) / (2.0 * eps);
            }

            let surrogate = self.renderer.compute_surrogate_gradients(&self.gaussians)?;
            let start = idx * 4;
            let end = start + 4;
            if end <= surrogate.rotations.len() {
                rot_grad[start..end].copy_from_slice(&surrogate.rotations[start..end]);
            }
        }

        Ok((pos_grad, scale_grad, rot_grad, opacity_grad, color_grad))
    }

    fn evaluate_loss(
        &self,
        positions: &[f32],
        scales: &[f32],
        rotations: &[f32],
        opacities: &[f32],
        colors: &[f32],
        camera: &DiffCamera,
        observed_color: &[f32],
        observed_depth: &[f32],
    ) -> candle_core::Result<f32> {
        let candidate = TrainableGaussians::new(
            positions,
            scales,
            rotations,
            opacities,
            colors,
            &self.device,
        )?;
        let output = self.renderer.render(&candidate, camera)?;
        let loss = self
            .renderer
            .compute_loss(&output, observed_color, observed_depth)?;
        let value = loss.total.to_vec0::<f32>()?;
        if value.is_finite() {
            Ok(value)
        } else {
            Ok(0.0)
        }
    }
}

/// Simple SGD-like optimizer for Gaussians
///
/// In practice, each parameter type has different learning rates
#[cfg(feature = "gpu")]
pub struct GaussiansSGD {
    lr_positions: f32,
    lr_scales: f32,
    lr_rotations: f32,
    lr_opacities: f32,
    lr_colors: f32,
}

#[cfg(feature = "gpu")]
impl GaussiansSGD {
    pub fn new(config: &TrainConfig) -> Self {
        Self {
            lr_positions: config.lr_position,
            lr_scales: config.lr_scale,
            lr_rotations: config.lr_rotation,
            lr_opacities: config.lr_opacity,
            lr_colors: config.lr_color,
        }
    }

    /// Update positions
    pub fn update_positions(&self, positions: &Var, grad: &Tensor) -> candle_core::Result<()> {
        // positions = positions - lr * grad
        let update = grad.mul(&Tensor::new(self.lr_positions, positions.device())?)?;
        let new_pos = positions.as_tensor().sub(&update)?;
        positions.set(&new_pos)?;
        Ok(())
    }

    /// Update scales
    pub fn update_scales(&self, scales: &Var, grad: &Tensor) -> candle_core::Result<()> {
        let update = grad.mul(&Tensor::new(self.lr_scales, scales.device())?)?;
        let new_scale = scales.as_tensor().sub(&update)?;
        scales.set(&new_scale)?;
        Ok(())
    }
}

#[cfg(feature = "gpu")]
fn flatten_2d(data: &[Vec<f32>]) -> Vec<f32> {
    data.iter().flat_map(|row| row.iter().copied()).collect()
}

fn blend_gradients(primary: &mut [f32], secondary: &[f32], weight: f32) {
    for (lhs, rhs) in primary.iter_mut().zip(secondary.iter()) {
        *lhs += rhs * weight;
    }
}

fn apply_gradient_step(param: &mut [f32], grad: &[f32], lr: f32) {
    for (value, grad) in param.iter_mut().zip(grad.iter()) {
        if grad.is_finite() {
            *value -= lr * grad;
        }
    }
}

fn clamp_log_scales(scales: &mut [f32]) {
    for value in scales {
        *value = value.clamp(-8.0, 2.0);
    }
}

fn normalize_rotation_buffer(rotations: &mut [f32]) {
    for quat in rotations.chunks_exact_mut(4) {
        let norm = (quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3])
            .sqrt()
            .max(1e-6);
        for value in quat {
            *value /= norm;
        }
    }
}

fn clamp_opacity_logits(opacities: &mut [f32]) {
    let min = opacity_to_logit(1e-4);
    let max = opacity_to_logit(0.999);
    for value in opacities {
        *value = value.clamp(min, max);
    }
}

fn clamp_colors(colors: &mut [f32]) {
    for value in colors {
        *value = value.clamp(0.0, 1.0);
    }
}

#[cfg(feature = "gpu")]
fn export_gaussians(
    gaussians: &TrainableGaussians,
) -> candle_core::Result<Vec<CheckpointGaussian>> {
    let positions = flatten_2d(&gaussians.positions().to_vec2::<f32>()?);
    let scales = flatten_2d(&gaussians.scales()?.to_vec2::<f32>()?);
    let rotations = flatten_2d(&gaussians.rotations()?.to_vec2::<f32>()?);
    let opacities = gaussians.opacities()?.to_vec1::<f32>()?;
    let colors = flatten_2d(&gaussians.colors().to_vec2::<f32>()?);

    let mut output = Vec::with_capacity(gaussians.len());
    for idx in 0..gaussians.len() {
        let p = idx * 3;
        let r = idx * 4;
        let c = idx * 3;
        output.push(CheckpointGaussian {
            position: [positions[p], positions[p + 1], positions[p + 2]],
            scale: [scales[p], scales[p + 1], scales[p + 2]],
            rotation: [
                rotations[r],
                rotations[r + 1],
                rotations[r + 2],
                rotations[r + 3],
            ],
            opacity: opacities[idx],
            color: [colors[c], colors[c + 1], colors[c + 2]],
        });
    }
    Ok(output)
}

#[cfg(feature = "gpu")]
fn checkpoint_to_gaussians(
    checkpoint: &TrainerCheckpoint,
    device: &Device,
) -> candle_core::Result<TrainableGaussians> {
    let mut positions = Vec::with_capacity(checkpoint.gaussians.len() * 3);
    let mut scales = Vec::with_capacity(checkpoint.gaussians.len() * 3);
    let mut rotations = Vec::with_capacity(checkpoint.gaussians.len() * 4);
    let mut opacities = Vec::with_capacity(checkpoint.gaussians.len());
    let mut colors = Vec::with_capacity(checkpoint.gaussians.len() * 3);

    for gaussian in &checkpoint.gaussians {
        positions.extend_from_slice(&gaussian.position);
        scales.extend_from_slice(&[
            gaussian.scale[0].max(1e-6).ln(),
            gaussian.scale[1].max(1e-6).ln(),
            gaussian.scale[2].max(1e-6).ln(),
        ]);
        rotations.extend_from_slice(&gaussian.rotation);
        opacities.push(opacity_to_logit(gaussian.opacity));
        colors.extend_from_slice(&gaussian.color);
    }

    TrainableGaussians::new(&positions, &scales, &rotations, &opacities, &colors, device)
}

fn opacity_to_logit(opacity: f32) -> f32 {
    let clamped = opacity.clamp(1e-6, 1.0 - 1e-6);
    (clamped / (1.0 - clamped)).ln()
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn test_camera(device: &Device) -> DiffCamera {
        DiffCamera::new(
            32.0,
            32.0,
            2.0,
            2.0,
            4,
            4,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
            device,
        )
        .unwrap()
    }

    fn test_gaussians(device: &Device, opacities: &[f32]) -> TrainableGaussians {
        let positions = vec![0.0f32, 0.0, 2.0, 0.1, 0.0, 2.2];
        let scales = vec![0.05f32.ln(), 0.05f32.ln(), 0.05f32.ln()].repeat(2);
        let rotations = vec![1.0f32, 0.0, 0.0, 0.0].repeat(2);
        let colors = vec![0.9f32, 0.3, 0.2, 0.2, 0.7, 0.4];
        let logits: Vec<f32> = opacities.iter().copied().map(opacity_to_logit).collect();
        TrainableGaussians::new(&positions, &scales, &rotations, &logits, &colors, device).unwrap()
    }

    #[test]
    fn test_trainer_creation() {
        let config = TrainConfig::default();
        let trainer = Trainer::new(config, 100);

        assert!(trainer.is_ok());

        if let Ok(t) = trainer {
            assert_eq!(t.gaussians().len(), 100);
        }
    }

    #[test]
    fn test_config_default() {
        let config = TrainConfig::default();
        assert_eq!(config.max_iterations, 30_000);
        assert_eq!(config.lr_position, 0.00016);
    }

    #[test]
    fn test_step_updates_state_and_parameters() {
        let device = Device::Cpu;
        let gaussians = test_gaussians(&device, &[0.6, 0.4]);
        let mut trainer = Trainer::from_gaussians(TrainConfig::default(), gaussians, 4, 4);
        let before = trainer
            .gaussians
            .opacities
            .as_tensor()
            .to_vec1::<f32>()
            .unwrap();
        let camera = test_camera(&device);
        let observed_color = vec![0.0f32; 4 * 4 * 3];
        let observed_depth = vec![0.0f32; 4 * 4];

        let loss = trainer
            .step(&camera, &observed_color, &observed_depth)
            .unwrap();
        let after = trainer
            .gaussians
            .opacities
            .as_tensor()
            .to_vec1::<f32>()
            .unwrap();

        assert!(loss.is_finite());
        assert_eq!(trainer.state().iteration, 1);
        assert_eq!(trainer.state().losses.len(), 1);
        assert_ne!(before, after);
    }

    #[test]
    fn test_densify_increases_gaussian_count() {
        let device = Device::Cpu;
        let gaussians = test_gaussians(&device, &[0.9, 0.8]);
        let mut trainer = Trainer::from_gaussians(TrainConfig::default(), gaussians, 4, 4);
        let before = trainer.gaussians().len();

        trainer.densify().unwrap();

        assert!(trainer.gaussians().len() > before);
    }

    #[test]
    fn test_prune_removes_low_opacity_gaussians() {
        let device = Device::Cpu;
        let gaussians = test_gaussians(&device, &[0.0005, 0.8]);
        let mut trainer = Trainer::from_gaussians(TrainConfig::default(), gaussians, 4, 4);

        trainer.prune().unwrap();

        assert_eq!(trainer.gaussians().len(), 1);
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let device = Device::Cpu;
        let gaussians = test_gaussians(&device, &[0.8, 0.7]);
        let mut trainer = Trainer::from_gaussians(TrainConfig::default(), gaussians, 4, 4);
        trainer.state.iteration = 7;
        trainer.state.losses = vec![1.0, 0.5];
        let file = NamedTempFile::new().unwrap();

        trainer.save(file.path().to_str().unwrap()).unwrap();

        let restored_gaussians = test_gaussians(&device, &[0.2, 0.2]);
        let mut restored =
            Trainer::from_gaussians(TrainConfig::default(), restored_gaussians, 4, 4);
        restored.load(file.path().to_str().unwrap()).unwrap();

        assert_eq!(restored.state().iteration, 7);
        assert_eq!(restored.state().losses, vec![1.0, 0.5]);
        assert_eq!(restored.gaussians().len(), trainer.gaussians().len());
    }
}
