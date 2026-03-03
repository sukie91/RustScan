//! Differentiable Gaussian Splatting with Candle autograd (candle-core 0.9.x)
//!
//! Implements automatic differentiation for 3DGS training.
//! Based on: "3D Gaussian Splatting for Real-Time Radiance Field Rendering"

use candle_core::{Tensor, Device, DType, Var};

/// Trainable Gaussian parameters with gradient tracking via `Var`.
pub struct DiffGaussian {
    /// Position parameters [N, 3]
    pos: Var,
    /// Scale parameters (log scale) [N, 3]
    scale: Var,
    /// Rotation parameters (quaternion) [N, 4]
    rot: Var,
    /// Opacity parameters (logit) [N]
    opacity: Var,
    /// Color parameters [N, 3]
    color: Var,
    /// Number of Gaussians
    n: usize,
    device: Device,
}

impl DiffGaussian {
    /// Create new differentiable Gaussians from flat f32 slices.
    pub fn new(
        positions: &[f32],
        scales: &[f32],
        rotations: &[f32],
        opacities: &[f32],
        colors: &[f32],
        device: &Device,
    ) -> candle_core::Result<Self> {
        let n = positions.len() / 3;

        Ok(Self {
            pos: Var::from_tensor(&Tensor::from_slice(positions, (n, 3), device)?)?,
            scale: Var::from_tensor(&Tensor::from_slice(scales, (n, 3), device)?)?,
            rot: Var::from_tensor(&Tensor::from_slice(rotations, (n, 4), device)?)?,
            opacity: Var::from_tensor(&Tensor::from_slice(opacities, (n,), device)?)?,
            color: Var::from_tensor(&Tensor::from_slice(colors, (n, 3), device)?)?,
            n,
            device: device.clone(),
        })
    }

    /// Position tensor (world space).
    pub fn position(&self) -> &Tensor {
        self.pos.as_tensor()
    }

    /// Scale (exp of stored log-scale).
    pub fn scale(&self) -> candle_core::Result<Tensor> {
        self.scale.as_tensor().exp()
    }

    /// Rotation (normalized quaternion).
    pub fn rotation(&self) -> candle_core::Result<Tensor> {
        let q = self.rot.as_tensor();
        let norm = q.mul(q)?.sum(1)?.sqrt()?;
        let norm = norm.unsqueeze(1)?;
        q.broadcast_div(&norm)
    }

    /// Opacity (sigmoid of stored logit).
    pub fn opacity(&self) -> candle_core::Result<Tensor> {
        sigmoid(self.opacity.as_tensor())
    }

    /// Color tensor.
    pub fn color(&self) -> &Tensor {
        self.color.as_tensor()
    }

    /// Number of Gaussians.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Device this lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// All trainable `Var`s (for manual SGD updates).
    pub fn vars(&self) -> [&Var; 5] {
        [&self.pos, &self.scale, &self.rot, &self.opacity, &self.color]
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn sigmoid(x: &Tensor) -> candle_core::Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg_x = neg_x.exp()?;
    let denom = exp_neg_x.affine(1.0, 1.0)?; // 1 + exp(-x)
    denom.recip()
}

// ── Camera ───────────────────────────────────────────────────────────────────

/// Pinhole camera for differentiable rendering.
pub struct DiffRenderCamera {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: usize,
    pub height: usize,
}

impl DiffRenderCamera {
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32, width: usize, height: usize) -> Self {
        Self { fx, fy, cx, cy, width, height }
    }
}

// ── Renderer ─────────────────────────────────────────────────────────────────

/// Differentiable Gaussian Splatting Renderer.
pub struct DiffSplat {
    device: Device,
    width: usize,
    height: usize,
}

impl DiffSplat {
    pub fn new(width: usize, height: usize) -> Self {
        let device = Device::new_metal(0).unwrap_or(Device::Cpu);
        Self { device, width, height }
    }

    /// Forward pass: render Gaussians to color + depth images.
    pub fn render(
        &self,
        gaussians: &DiffGaussian,
        camera: &DiffRenderCamera,
    ) -> candle_core::Result<DiffRendered> {
        let pos = gaussians.position();
        let scale = gaussians.scale()?;
        let _rot = gaussians.rotation()?;
        let opacity = gaussians.opacity()?;
        let color = gaussians.color();

        let (u, v, depth) = self.project(pos, camera)?;
        let weights = self.compute_gaussian_weights(&u, &v, &scale, &depth)?;
        let rendered_color = self.alpha_blend(color, &opacity, &weights)?;
        let rendered_depth = self.depth_blend(&depth, &opacity, &weights)?;

        Ok(DiffRendered { color: rendered_color, depth: rendered_depth })
    }

    fn project(
        &self,
        pos: &Tensor,
        camera: &DiffRenderCamera,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let x = pos.narrow(1, 0, 1)?.squeeze(1)?;
        let y = pos.narrow(1, 1, 1)?.squeeze(1)?;
        let z = pos.narrow(1, 2, 1)?.squeeze(1)?;

        let fx = Tensor::new(camera.fx, &self.device)?;
        let fy = Tensor::new(camera.fy, &self.device)?;
        let cx = Tensor::new(camera.cx, &self.device)?;
        let cy = Tensor::new(camera.cy, &self.device)?;

        let z_safe = z.clamp(1e-6f64, f64::MAX)?;
        let u = x.broadcast_mul(&fx)?.broadcast_div(&z_safe)?.broadcast_add(&cx)?;
        let v = y.broadcast_mul(&fy)?.broadcast_div(&z_safe)?.broadcast_add(&cy)?;

        Ok((u, v, z))
    }

    fn compute_gaussian_weights(
        &self,
        _u: &Tensor,
        _v: &Tensor,
        _scale: &Tensor,
        _depth: &Tensor,
    ) -> candle_core::Result<Tensor> {
        // Simplified uniform weights — full 2D Gaussian kernel is in tiled_renderer.
        let n = _u.dim(0)?;
        let weights = Tensor::ones((n,), DType::F32, &self.device)?;
        let sum = weights.sum(0)?;
        weights.broadcast_div(&sum)
    }

    fn alpha_blend(
        &self,
        color: &Tensor,
        opacity: &Tensor,
        weights: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let n = color.dim(0)?;
        let w = opacity.reshape((n, 1))?.broadcast_mul(weights)?;
        color.mul(&w)?.sum(0)?.clamp(0.0f64, 1.0f64)
    }

    fn depth_blend(
        &self,
        depth: &Tensor,
        opacity: &Tensor,
        weights: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let w = opacity.broadcast_mul(weights)?;
        depth.broadcast_mul(&w)?.sum(0)
    }

    /// Compute L1 color + depth loss.
    pub fn compute_loss(
        &self,
        rendered: &DiffRendered,
        target_color: &[f32],
        target_depth: &[f32],
    ) -> candle_core::Result<DiffLoss> {
        let target_c = Tensor::from_slice(target_color, (self.height, self.width, 3), &self.device)?;
        let target_d = Tensor::from_slice(target_depth, (self.height, self.width), &self.device)?;

        let color_loss = rendered.color.sub(&target_c)?.abs()?.sum(0)?;
        let depth_loss = rendered.depth.sub(&target_d)?.abs()?.sum(0)?;

        let color_weight = Tensor::new(1.0f32, &self.device)?;
        let depth_weight = Tensor::new(1.0f32, &self.device)?;

        let total = color_loss
            .broadcast_mul(&color_weight)?
            .add(&depth_loss.broadcast_mul(&depth_weight)?)?;

        Ok(DiffLoss { total, color_loss, depth_loss })
    }
}

/// Rendered output tensors.
pub struct DiffRendered {
    pub color: Tensor,
    pub depth: Tensor,
}

/// Loss tensors.
pub struct DiffLoss {
    pub total: Tensor,
    pub color_loss: Tensor,
    pub depth_loss: Tensor,
}

// ── Trainer ──────────────────────────────────────────────────────────────────

/// Trainer using Candle autograd (candle-core 0.9.x API).
pub struct AutodiffTrainer {
    renderer: DiffSplat,
    device: Device,
}

impl AutodiffTrainer {
    pub fn new(width: usize, height: usize) -> Self {
        let device = Device::new_metal(0).unwrap_or(Device::Cpu);
        Self {
            renderer: DiffSplat::new(width, height),
            device,
        }
    }

    /// Single training step: forward → backward → SGD update.
    ///
    /// `GradStore::get(&Tensor) -> Option<&Tensor>` is the 0.9.x API.
    pub fn training_step(
        &self,
        gaussians: &DiffGaussian,
        camera: &DiffRenderCamera,
        target_color: &[f32],
        target_depth: &[f32],
        lr_pos: f64,
        lr_scale: f64,
    ) -> candle_core::Result<f32> {
        // Forward
        let rendered = self.renderer.render(gaussians, camera)?;
        let loss = self.renderer.compute_loss(&rendered, target_color, target_depth)?;

        // Backward — returns GradStore (candle 0.9.x)
        let grads = loss.total.backward()?;

        // SGD: param = param - lr * grad  (using Var::set)
        let lrs = [lr_pos, lr_scale, 1e-3, 5e-2, 2.5e-3];
        for (var, lr) in gaussians.vars().iter().zip(lrs.iter()) {
            let t = var.as_tensor();
            if let Some(g) = grads.get(t) {
                let update = g.affine(-lr, 0.0)?;
                var.set(&t.add(&update)?)?;
            }
        }

        loss.total.to_vec0::<f32>()
    }

    /// Run training for `iterations` steps cycling through provided frames.
    pub fn train(
        &self,
        gaussians: &DiffGaussian,
        cameras: &[DiffRenderCamera],
        colors: &[&[f32]],
        depths: &[&[f32]],
        iterations: usize,
    ) -> candle_core::Result<()> {
        let num_frames = cameras.len();
        println!("AutodiffTrainer: {} iterations over {} frames", iterations, num_frames);

        for iter in 0..iterations {
            let f = iter % num_frames;
            let loss = self.training_step(
                gaussians,
                &cameras[f],
                colors[f],
                depths[f],
                1.6e-4,
                5e-3,
            )?;
            if iter % 10 == 0 {
                println!("  iter {:5} | loss {:.6}", iter, loss);
            }
        }

        println!("AutodiffTrainer: done.");
        Ok(())
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_creation() {
        let renderer = DiffSplat::new(640, 480);
        assert_eq!(renderer.width, 640);
        assert_eq!(renderer.height, 480);
    }

    #[test]
    fn test_diff_gaussian_creation() {
        let device = Device::new_metal(0).unwrap_or(Device::Cpu);

        let g = DiffGaussian::new(
            &[0.0, 0.0, 0.0],
            &[-2.0, -2.0, -2.0],
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0],
            &[1.0, 0.5, 0.25],
            &device,
        );

        assert!(g.is_ok());
        assert_eq!(g.unwrap().len(), 1);
    }

    #[test]
    fn test_autodiff_trainer_creation() {
        let trainer = AutodiffTrainer::new(64, 64);
        // Just verify it constructs without panicking.
        let _ = trainer.device();
    }
}
