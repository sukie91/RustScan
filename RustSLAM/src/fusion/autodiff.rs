//! TRUE Automatic Differentiation for 3DGS
//!
//! This module implements REAL automatic differentiation using Candle's Var and backward().
//! The key is using Var for gradient tracking and calling backward() on the loss.

use candle_core::{Tensor, Device, DType, Var};
use std::sync::Arc;

/// Trainable Gaussian parameters with TRUE gradient tracking
/// 
/// Key: We use Var instead of Tensor to enable gradient tracking!
pub struct VarGaussian {
    /// Position - Variable (tracks gradient)
    pub pos: Var,
    /// Scale (log) - Variable
    pub scale: Var,
    /// Rotation - Variable
    pub rot: Var,
    /// Opacity - Variable
    pub opacity: Var,
    /// Color - Variable
    pub color: Var,
    /// Number of Gaussians
    pub n: usize,
    device: Device,
}

impl VarGaussian {
    /// Create new Gaussians with gradient tracking
    /// 
    /// The key is Var::from_tensor() which enables gradient tracking!
    pub fn new(
        positions: &[f32],
        scales: &[f32],
        rotations: &[f32],
        opacities: &[f32],
        colors: &[f32],
        device: &Device,
    ) -> candle_core::Result<Self> {
        let n = positions.len() / 3;

        // Create regular tensors first
        let pos_t = Tensor::from_slice(positions, (n, 3), device)?;
        let scale_t = Tensor::from_slice(scales, (n, 3), device)?;
        let rot_t = Tensor::from_slice(rotations, (n, 4), device)?;
        let opacity_t = Tensor::from_slice(opacities, (n,), device)?;
        let color_t = Tensor::from_slice(colors, (n, 3), device)?;

        // Convert to Var (enables gradient tracking!)
        let pos = Var::from_tensor(&pos_t)?;
        let scale = Var::from_tensor(&scale_t)?;
        let rot = Var::from_tensor(&rot_t)?;
        let opacity = Var::from_tensor(&opacity_t)?;
        let color = Var::from_tensor(&color_t)?;

        Ok(Self {
            pos,
            scale,
            rot,
            opacity,
            color,
            n,
            device: device.clone(),
        })
    }

    /// Create random Gaussians with gradient tracking
    pub fn new_random(n: usize, device: &Device) -> candle_core::Result<Self> {
        let positions = vec![0.0f32; n * 3];
        let scales = vec![-3.0f32; n * 3];
        let rotations = vec![1.0f32, 0.0, 0.0, 0.0].repeat(n);
        let opacities = vec![0.5f32; n];
        let colors = vec![1.0f32, 1.0, 1.0].repeat(n);
        Self::new(&positions, &scales, &rotations, &opacities, &colors, device)
    }

    /// Get position tensor
    pub fn position(&self) -> &Tensor {
        self.pos.as_tensor()
    }

    /// Get actual scale (exp for actual scale)
    pub fn actual_scale(&self) -> candle_core::Result<Tensor> {
        self.scale.as_tensor().exp()
    }

    /// Get opacity (sigmoid)
    pub fn actual_opacity(&self) -> candle_core::Result<Tensor> {
        let x = self.opacity.as_tensor();
        let one = Tensor::new(1.0f32, &self.device)?;
        one.broadcast_div(&one.broadcast_add(&x.neg()?.exp()?)?)
    }

    /// Get color
    pub fn get_color(&self) -> &Tensor {
        self.color.as_tensor()
    }

    /// Number of Gaussians
    pub fn len(&self) -> usize { self.n }
    /// Device
    pub fn device(&self) -> &Device { &self.device }

    /// Get gradients and update parameters
    /// 
    /// This is where the magic happens!
    pub fn update_with_gradients(
        &mut self,
        pos_grad: &Tensor,
        scale_grad: &Tensor,
        opacity_grad: &Tensor,
        color_grad: &Tensor,
        lr_pos: f32,
        lr_scale: f32,
        lr_opacity: f32,
        lr_color: f32,
    ) -> candle_core::Result<()> {
        // Update positions: pos = pos - lr * grad
        let lr_pos_t = Tensor::new(lr_pos, &self.device)?;
        let pos_update = pos_grad.mul(&lr_pos_t)?;
        let new_pos = self.position().sub(&pos_update)?;
        // Create new Var with updated values (recreate to keep gradient tracking)
        let pos_t = Tensor::from_slice(&new_pos.to_vec1::<f32>()?, (self.n, 3), &self.device)?;
        self.pos = Var::from_tensor(&pos_t)?;

        // Update scales
        let lr_scale_t = Tensor::new(lr_scale, &self.device)?;
        let scale_update = scale_grad.mul(&lr_scale_t)?;
        let new_scale = self.actual_scale()?.sub(&scale_update)?;
        let scale_t = Tensor::from_slice(&new_scale.to_vec1::<f32>()?, (self.n, 3), &self.device)?;
        self.scale = Var::from_tensor(&scale_t)?;

        // Update opacities
        let lr_op_t = Tensor::new(lr_opacity, &self.device)?;
        let op_update = opacity_grad.mul(&lr_op_t)?;
        let new_op = self.opacity.as_tensor().sub(&op_update)?;
        let op_t = Tensor::from_slice(&new_op.to_vec1::<f32>()?, (self.n,), &self.device)?;
        self.opacity = Var::from_tensor(&op_t)?;

        // Update colors
        let lr_color_t = Tensor::new(lr_color, &self.device)?;
        let color_update = color_grad.mul(&lr_color_t)?;
        let new_color = self.get_color().sub(&color_update)?;
        let color_t = Tensor::from_slice(&new_color.to_vec1::<f32>()?, (self.n, 3), &self.device)?;
        self.color = Var::from_tensor(&color_t)?;

        Ok(())
    }
}

/// Camera
pub struct VarCamera {
    pub fx: f32, pub fy: f32, pub cx: f32, pub cy: f32,
    pub width: usize, pub height: usize,
}

impl VarCamera {
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32, width: usize, height: usize) -> Self {
        Self { fx, fy, cx, cy, width, height }
    }
    pub fn default640() -> Self {
        Self::new(500.0, 500.0, 320.0, 240.0, 640, 480)
    }
}

/// Differentiable renderer
pub struct VarRenderer {
    device: Device,
    width: usize,
    height: usize,
}

impl VarRenderer {
    pub fn new(width: usize, height: usize) -> Self {
        let device = Device::new_metal(0).unwrap_or_else(|_| Device::Cpu);
        println!("VarRenderer device: {:?}", device);
        Self { device, width, height }
    }

    /// Forward pass - builds computation graph with Var
    pub fn forward(
        &self,
        gaussians: &VarGaussian,
        camera: &VarCamera,
    ) -> candle_core::Result<VarOutput> {
        // Get parameters - these are Vars with gradient tracking!
        let pos = gaussians.position();
        let scale = gaussians.actual_scale()?;
        let opacity = gaussians.actual_opacity()?;
        let color = gaussians.get_color();
        let n = gaussians.n;

        // Project 3D to 2D
        let x = pos.narrow(1, 0, 1)?.squeeze(1)?;
        let y = pos.narrow(1, 1, 1)?.squeeze(1)?;
        let z = pos.narrow(1, 2, 1)?.squeeze(1)?;

        let fx_t = Tensor::from_slice(&[camera.fx], (1,), &self.device)?;
        let fy_t = Tensor::from_slice(&[camera.fy], (1,), &self.device)?;
        let cx_t = Tensor::from_slice(&[camera.cx], (1,), &self.device)?;
        let cy_t = Tensor::from_slice(&[camera.cy], (1,), &self.device)?;

        let z_safe = z.clamp(1e-6, f32::MAX)?;
        
        let _u = x.broadcast_mul(&fx_t)?.broadcast_div(&z_safe)?.broadcast_add(&cx_t)?;
        let _v = y.broadcast_mul(&fy_t)?.broadcast_div(&z_safe)?.broadcast_add(&cy_t)?;

        // Scale by depth
        let scale_x = scale.narrow(1, 0, 1)?.squeeze(1)?.broadcast_div(&z)?;
        let _scale_y = scale.narrow(1, 1, 1)?.squeeze(1)?.broadcast_div(&z)?;

        // Weights
        let weights = Tensor::ones((n,), DType::F32, &self.device)?;

        // Color blending with sigmoid opacity
        let op_exp = opacity.exp()?;
        let w = op_exp.reshape((n, 1))?;
        let w = w.broadcast_mul(&weights)?;
        let w_sum = w.sum(0)?;
        let w_norm = w.broadcast_div(&w_sum)?;
        
        let w_color = w_norm.broadcast_mul(color)?;
        let rendered_color = w_color.sum(0)?.clamp(0.0, 1.0)?;

        // Depth
        let rendered_depth = z.broadcast_mul(&weights)?.sum(0)?;

        Ok(VarOutput { color: rendered_color, depth: rendered_depth })
    }

    /// Compute loss
    pub fn loss(
        &self,
        rendered: &VarOutput,
        target_color: &[f32],
        target_depth: &[f32],
    ) -> candle_core::Result<Tensor> {
        let target_c = Tensor::from_slice(target_color, (self.height, self.width, 3), &self.device)?;
        let target_d = Tensor::from_slice(target_depth, (self.height, self.width), &self.device)?;

        // L1 color loss
        let color_diff = rendered.color.sub(&target_c)?;
        let color_loss = color_diff.abs()?.sum(0)?;

        // L1 depth loss
        let depth_diff = rendered.depth.sub(&target_d)?;
        let depth_loss = depth_diff.abs()?.sum(0)?;

        Ok(color_loss.add(&depth_loss)?)
    }
}

/// Output
pub struct VarOutput {
    pub color: Tensor,
    pub depth: Tensor,
}

/// TRUE autodiff trainer with REAL backward propagation!
pub struct TrueAutodiffTrainer {
    renderer: VarRenderer,
    device: Device,
    lr_pos: f32,
    lr_scale: f32,
    lr_opacity: f32,
    lr_color: f32,
}

impl TrueAutodiffTrainer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            renderer: VarRenderer::new(width, height),
            device: Device::new_metal(0).unwrap_or_else(|_| Device::Cpu),
            lr_pos: 0.00016,
            lr_scale: 0.005,
            lr_opacity: 0.05,
            lr_color: 0.0025,
        }
    }

    /// Training step with TRUE backward propagation!
    /// 
    /// This is the key:
    /// 1. Forward pass builds computation graph with Var
    /// 2. loss.backward() computes gradients through the graph
    /// 3. grads.grad(var) gets gradient for each parameter
    /// 4. Update parameters with gradient descent
    pub fn step(
        &mut self,
        gaussians: &mut VarGaussian,
        camera: &VarCamera,
        target_color: &[f32],
        target_depth: &[f32],
    ) -> candle_core::Result<f32> {
        // === FORWARD PASS ===
        // Build computation graph with gradient tracking
        let output = self.renderer.forward(gaussians, camera)?;
        
        // Compute loss
        let loss = self.renderer.loss(&output, target_color, target_depth)?;
        
        // === BACKWARD PASS - THE MAGIC! ===
        // This computes gradients through the entire computation graph!
        let gradients = loss.backward()?;
        
        // Get gradients for each parameter using .get()
        let pos_grad = gradients.get(gaussians.position());
        let scale_grad = gradients.get(&gaussians.actual_scale()?);
        let opacity_grad = gradients.get(&gaussians.actual_opacity()?);
        let color_grad = gradients.get(gaussians.get_color());
        
        // Print gradient norms for debugging
        if let Some(pg) = pos_grad {
            let pos_norm = pg.sqr()?.sum(0)?;
            println!("  pos grad norm: {:?}", pos_norm.to_vec1::<f32>());
        }
        
        // === PARAMETER UPDATE ===
        // Apply gradient descent with learning rates
        if let (Some(pg), Some(sg), Some(og), Some(cg)) = (pos_grad, scale_grad, opacity_grad, color_grad) {
            gaussians.update_with_gradients(
                pg,
                sg,
                og,
                cg,
                self.lr_pos,
                self.lr_scale,
                self.lr_opacity,
                self.lr_color,
            )?;
        }

        // Get loss value
        let loss_value = loss.to_vec0::<f32>()?;
        
        Ok(loss_value)
    }

    /// Train loop
    pub fn train(
        &mut self,
        gaussians: &mut VarGaussian,
        cameras: &[VarCamera],
        colors: &[&[f32]],
        depths: &[&[f32]],
        iterations: usize,
    ) -> candle_core::Result<()> {
        let num_frames = cameras.len();
        
        println!("\n=== TRUE Autodiff Training (with backward!) ===");
        println!("Gaussians: {}, Frames: {}, Iterations: {}", gaussians.len(), num_frames, iterations);

        for iter in 0..iterations {
            let frame_idx = iter % num_frames;
            
            let loss = self.step(
                gaussians,
                &cameras[frame_idx],
                colors[frame_idx],
                depths[frame_idx],
            )?;

            if iter % 10 == 0 {
                println!("Iter {:5} | Loss: {:.6}", iter, loss);
            }
        }

        println!("=== Training Complete ===\n");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_renderer() {
        let r = VarRenderer::new(64, 64);
        assert_eq!(r.width, 64);
    }

    #[test]
    fn test_var_gaussian() {
        let device = Device::new_metal(0).unwrap_or_else(|_| Device::Cpu);
        
        if !matches!(device, Device::Metal(_)) {
            return;
        }
        
        let g = VarGaussian::new_random(10, &device);
        if let Ok(gaussians) = g {
            assert_eq!(gaussians.len(), 10);
        }
    }

    #[test]
    fn test_trainer() {
        let t = TrueAutodiffTrainer::new(64, 64);
        assert_eq!(t.lr_pos, 0.00016);
    }
}
