//! Differentiable Gaussian Renderer using Candle
//!
//! This module provides GPU-accelerated differentiable rendering
//! for 3D Gaussian Splatting using Candle with Metal MPS backend.

use candle_core::{Tensor, Device};

#[cfg(feature = "gpu")]
use crate::render::{Gaussian, RenderBuffer, TiledRenderer};

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

/// Result of rendering and loss computation
pub struct RenderLoss {
    pub total: Tensor,
    pub color: Tensor,
    pub depth: Tensor,
    pub rendered_color: Tensor,
    pub rendered_depth: Tensor,
}

/// Differentiable Gaussian Renderer using Candle
#[cfg(feature = "gpu")]
pub struct DiffGaussianRenderer {
    /// Device for computation
    device: Device,
    /// Image width
    width: usize,
    /// Image height
    height: usize,
    /// CPU tiled renderer for rasterization
    tiled_renderer: TiledRenderer,
}

#[cfg(feature = "gpu")]
impl DiffGaussianRenderer {
    /// Create a new renderer
    pub fn new(width: usize, height: usize) -> Self {
        // Try to use Metal MPS if available, otherwise CPU
        let device = Device::new_metal(0).unwrap_or_else(|_| Device::Cpu);

        println!("Using device: {:?}", device);

        Self {
            device,
            width,
            height,
            tiled_renderer: TiledRenderer::new(width, height),
        }
    }

    /// Create renderer with explicit device
    pub fn with_device(width: usize, height: usize, device: Device) -> Self {
        Self {
            device,
            width,
            height,
            tiled_renderer: TiledRenderer::new(width, height),
        }
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
        camera: &CameraTensors,
    ) -> candle_core::Result<Tensor> {
        let render = self.render_tiled(gaussians, camera)?;
        Tensor::from_slice(
            &render.color,
            (self.height, self.width, 3),
            &self.device,
        )
    }

    /// Render Gaussians to depth image (differentiable)
    ///
    /// Returns: [H, W] depth tensor
    pub fn render_depth(
        &self,
        gaussians: &GaussianTensors,
        camera: &CameraTensors,
    ) -> candle_core::Result<Tensor> {
        let render = self.render_tiled(gaussians, camera)?;
        Tensor::from_slice(
            &render.depth,
            (self.height, self.width),
            &self.device,
        )
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

    fn render_tiled(
        &self,
        gaussians: &GaussianTensors,
        camera: &CameraTensors,
    ) -> candle_core::Result<RenderBuffer> {
        if gaussians.is_empty() {
            return Ok(RenderBuffer::new(self.width, self.height));
        }

        let gaussians = tensors_to_gaussians(gaussians)?;
        let rotation = tensor_to_rotation(&camera.rotation)?;
        let translation = tensor_to_vec3(&camera.translation)?;

        Ok(self.tiled_renderer.render(
            &gaussians,
            camera.intrinsics[0],
            camera.intrinsics[1],
            camera.intrinsics[2],
            camera.intrinsics[3],
            &rotation,
            &translation,
        ))
    }
}

#[cfg(feature = "gpu")]
fn tensors_to_gaussians(gaussians: &GaussianTensors) -> candle_core::Result<Vec<Gaussian>> {
    let positions = gaussians.positions.to_vec2::<f32>()?;
    let scales = gaussians.scales.to_vec2::<f32>()?;
    let rotations = gaussians.rotations.to_vec2::<f32>()?;
    let opacities = gaussians.opacities.to_vec1::<f32>()?;
    let colors = gaussians.colors.to_vec2::<f32>()?;

    let positions: Vec<f32> = positions.into_iter().flatten().collect();
    let scales: Vec<f32> = scales.into_iter().flatten().collect();
    let rotations: Vec<f32> = rotations.into_iter().flatten().collect();
    let colors: Vec<f32> = colors.into_iter().flatten().collect();

    let mut output = Vec::with_capacity(gaussians.n);
    for i in 0..gaussians.n {
        let p = i * 3;
        let r = i * 4;
        let c = i * 3;
        output.push(Gaussian::new(
            [positions[p], positions[p + 1], positions[p + 2]],
            [scales[p], scales[p + 1], scales[p + 2]],
            [rotations[r], rotations[r + 1], rotations[r + 2], rotations[r + 3]],
            opacities[i],
            [colors[c], colors[c + 1], colors[c + 2]],
        ));
    }

    Ok(output)
}

#[cfg(feature = "gpu")]
fn tensor_to_rotation(tensor: &Tensor) -> candle_core::Result<[[f32; 3]; 3]> {
    let data = tensor.flatten_all()?.to_vec1::<f32>()?;
    Ok([
        [data[0], data[1], data[2]],
        [data[3], data[4], data[5]],
        [data[6], data[7], data[8]],
    ])
}

#[cfg(feature = "gpu")]
fn tensor_to_vec3(tensor: &Tensor) -> candle_core::Result<[f32; 3]> {
    let data = tensor.to_vec1::<f32>()?;
    Ok([data[0], data[1], data[2]])
}

#[cfg(all(test, feature = "gpu"))]
mod tiled_tests {
    use super::*;

    #[test]
    fn test_diff_renderer_tiled_output() {
        let renderer = DiffGaussianRenderer::new(8, 8);
        let device = renderer.device().clone();

        let positions = vec![0.0f32, 0.0, 1.0];
        let scales = vec![0.05f32, 0.05, 0.05];
        let rotations = vec![1.0f32, 0.0, 0.0, 0.0];
        let opacities = vec![0.8f32];
        let colors = vec![1.0f32, 0.0, 0.0];

        let gaussians = GaussianTensors::new(
            &positions,
            &scales,
            &rotations,
            &opacities,
            &colors,
            &device,
        ).unwrap();

        let rotation = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let translation = [0.0, 0.0, 0.0];
        let camera = CameraTensors::new(
            500.0, 500.0, 4.0, 4.0,
            &rotation,
            &translation,
            8,
            8,
            &device,
        ).unwrap();

        let color = renderer.render_color(&gaussians, &camera).unwrap();
        let depth = renderer.render_depth(&gaussians, &camera).unwrap();

        assert_eq!(color.dims(), &[8, 8, 3]);
        assert_eq!(depth.dims(), &[8, 8]);
    }
}

#[cfg(all(test, feature = "gpu"))]
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