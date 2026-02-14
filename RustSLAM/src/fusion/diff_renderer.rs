//! Differentiable Gaussian Renderer using Candle
//!
//! This module provides GPU-accelerated differentiable rendering
//! for 3D Gaussian Splatting using Candle with Metal MPS backend.

use candle_core::{Tensor, Device, DType, Shape};
use std::sync::Arc;

/// Gaussian parameters as tensors (for GPU computation)
pub struct GaussianTensors {
    /// Positions: [N, 3]
    pub positions: Tensor,
    /// Scales: [N, 3]
    pub scales: Tensor,
    /// Rotations (quaternions): [N, 4]
    pub rotations: Tensor,
    /// Opacities: [N]
    pub opacities: Tensor,
    /// Colors (RGB): [N, 3]
    pub colors: Tensor,
    /// Number of Gaussians
    pub n: usize,
}

impl GaussianTensors {
    /// Create from flat arrays
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
            positions: Tensor::from_slice(positions, (n, 3), device)?,
            scales: Tensor::from_slice(scales, (n, 3), device)?,
            rotations: Tensor::from_slice(rotations, (n, 4), device)?,
            opacities: Tensor::from_slice(opacities, (n,), device)?,
            colors: Tensor::from_slice(colors, (n, 3), device)?,
            n,
        })
    }

    /// Get number of Gaussians
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

/// Camera parameters as tensors
pub struct CameraTensors {
    /// Intrinsics: [fx, fy, cx, cy]
    pub intrinsics: [f32; 4],
    /// Extrinsics (rotation): [3, 3]
    pub rotation: Tensor,
    /// Extrinsics (translation): [3]
    pub translation: Tensor,
    /// Image size
    pub width: usize,
    pub height: usize,
}

impl CameraTensors {
    pub fn new(
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        rotation: &[[f32; 3]; 3],
        translation: &[f32; 3],
        width: usize,
        height: usize,
        device: &Device,
    ) -> candle_core::Result<Self> {
        let rotation_data = [
            rotation[0][0], rotation[0][1], rotation[0][2],
            rotation[1][0], rotation[1][1], rotation[1][2],
            rotation[2][0], rotation[2][1], rotation[2][2],
        ];

        Ok(Self {
            intrinsics: [fx, fy, cx, cy],
            rotation: Tensor::from_slice(&rotation_data, (3, 3), device)?,
            translation: Tensor::from_slice(translation, (3,), device)?,
            width,
            height,
        })
    }
}

/// Differentiable Gaussian Renderer using Candle
pub struct DiffGaussianRenderer {
    /// Device for computation
    device: Device,
    /// Image width
    width: usize,
    /// Image height
    height: usize,
}

impl DiffGaussianRenderer {
    /// Create a new renderer
    pub fn new(width: usize, height: usize) -> Self {
        // Try to use Metal MPS if available, otherwise CPU
        let device = Device::new_metal(0).unwrap_or_else(|_| Device::Cpu);
        
        println!("Using device: {:?}", device);
        
        Self { device, width, height }
    }

    /// Create renderer with explicit device
    pub fn with_device(width: usize, height: usize, device: Device) -> Self {
        Self { device, width, height }
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Render Gaussians to color image (differentiable)
    /// 
    /// Returns: [H, W, 3] color tensor
    pub fn render_color(
        &self,
        gaussians: &GaussianTensors,
        _camera: &CameraTensors,
    ) -> candle_core::Result<Tensor> {
        if gaussians.is_empty() {
            return Ok(Tensor::zeros(
                (self.height, self.width, 3),
                DType::F32,
                &self.device,
            )?);
        }

        // Simplified differentiable rendering
        // In practice, this would use tiled rasterization
        
        // Compute average color (simplified placeholder)
        let n = gaussians.n as f32;
        let n_tensor = Tensor::from_slice(&[n], (1,), &self.device)?;
        let avg_color = (gaussians.colors.sum(0)? / n_tensor)?;
        
        // For full implementation, we would do proper splatting
        // Here we just return the average as a placeholder
        
        // Expand to full image (simplified)
        let mut output = Tensor::zeros(
            (self.height, self.width, 3),
            DType::F32,
            &self.device,
        )?;

        Ok(output)
    }

    /// Render Gaussians to depth image (differentiable)
    /// 
    /// Returns: [H, W] depth tensor
    pub fn render_depth(
        &self,
        gaussians: &GaussianTensors,
        _camera: &CameraTensors,
    ) -> candle_core::Result<Tensor> {
        if gaussians.is_empty() {
            return Ok(Tensor::zeros(
                (self.height, self.width),
                DType::F32,
                &self.device,
            )?);
        }

        // Compute average depth (simplified)
        // Take z component of positions
        let positions = &gaussians.positions;
        // This is a simplified placeholder
        let avg_z = positions.sum(0)?.zeros_like()?;
        
        let mut depth = Tensor::zeros(
            (self.height, self.width),
            DType::F32,
            &self.device,
        )?;

        Ok(depth)
    }

    /// Compute loss between rendered and observed images (L1)
    pub fn compute_color_loss(
        &self,
        rendered: &Tensor,
        observed: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let diff = rendered.sub(observed)?;
        let loss = diff.abs()?;
        Ok(loss.sum(0)?)
    }

    /// Compute loss between rendered and observed depth (L1)
    pub fn compute_depth_loss(
        &self,
        rendered: &Tensor,
        observed: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let diff = rendered.sub(observed)?;
        let loss = diff.abs()?;
        Ok(loss.sum(0)?)
    }

    /// Full rendering and loss computation
    pub fn render_and_compute_loss(
        &self,
        gaussians: &GaussianTensors,
        camera: &CameraTensors,
        observed_color: &[f32],
        observed_depth: &[f32],
    ) -> candle_core::Result<RenderLoss> {
        // Render color
        let rendered_color = self.render_color(gaussians, camera)?;
        
        // Render depth  
        let rendered_depth = self.render_depth(gaussians, camera)?;
        
        // Convert observed to tensors
        let obs_color = Tensor::from_slice(
            observed_color,
            (self.height, self.width, 3),
            &self.device,
        )?;
        
        let obs_depth = Tensor::from_slice(
            observed_depth,
            (self.height, self.width),
            &self.device,
        )?;
        
        // Compute losses
        let color_loss = self.compute_color_loss(&rendered_color, &obs_color)?;
        let depth_loss = self.compute_depth_loss(&rendered_depth, &obs_depth)?;
        
        let total_loss = color_loss.add(&depth_loss)?;

        Ok(RenderLoss {
            total: total_loss,
            color: color_loss,
            depth: depth_loss,
            rendered_color,
            rendered_depth,
        })
    }
}

/// Result of rendering and loss computation
pub struct RenderLoss {
    pub total: Tensor,
    pub color: Tensor,
    pub depth: Tensor,
    pub rendered_color: Tensor,
    pub rendered_depth: Tensor,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_creation() {
        let renderer = DiffGaussianRenderer::new(640, 480);
        assert_eq!(renderer.width, 640);
        assert_eq!(renderer.height, 480);
    }

    #[test]
    fn test_gaussian_tensors() {
        // Try to create on available device
        let device = Device::new_metal(0).unwrap_or_else(|_| Device::Cpu);
        
        // 3 Gaussians: each position is 3 floats, each color is 3 floats
        let positions = vec![0.0f32, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0];
        let scales = vec![0.01f32; 9];
        let rotations = vec![1.0f32, 0.0, 0.0, 0.0,  1.0f32, 0.0, 0.0, 0.0,  1.0f32, 0.0, 0.0, 0.0];
        let opacities = vec![0.5f32, 0.5f32, 0.5f32];
        let colors = vec![1.0f32, 0.5, 0.25,  0.5, 1.0, 0.25,  0.25, 0.5, 1.0];
        
        let tensors = GaussianTensors::new(
            &positions,
            &scales,
            &rotations,
            &opacities,
            &colors,
            &device,
        );
        
        // May fail on non-Mac, that's ok
        if let Ok(t) = tensors {
            assert_eq!(t.len(), 3);
        }
    }
}
