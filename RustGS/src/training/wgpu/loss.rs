use burn::prelude::*;
use burn::tensor::s;

#[derive(Debug, Clone)]
pub struct SsimConfig {
    pub window_size: usize,
    pub sigma: f64,
    pub k1: f64,
    pub k2: f64,
    pub data_range: f64,
}

impl Default for SsimConfig {
    fn default() -> Self {
        Self {
            window_size: 11,
            sigma: 1.5,
            k1: 0.01,
            k2: 0.03,
            data_range: 1.0,
        }
    }
}

pub fn ssim_loss<B: Backend>(
    pred: Tensor<B, 3>,
    target: Tensor<B, 3>,
    config: &SsimConfig,
    device: &B::Device,
) -> Tensor<B, 1> {
    let pred = to_nchw(pred);
    let target = to_nchw(target);
    let kernel = gaussian_kernel_1d::<B>(config, device);

    let mu_x = separable_blur(pred.clone(), kernel.clone());
    let mu_y = separable_blur(target.clone(), kernel.clone());

    let mu_x_sq = mu_x.clone().powi_scalar(2);
    let mu_y_sq = mu_y.clone().powi_scalar(2);
    let mu_xy = mu_x.clone() * mu_y.clone();

    let sigma_x_sq = separable_blur(pred.clone().powi_scalar(2), kernel.clone()) - mu_x_sq.clone();
    let sigma_y_sq =
        separable_blur(target.clone().powi_scalar(2), kernel.clone()) - mu_y_sq.clone();
    let sigma_xy = separable_blur(pred * target, kernel) - mu_xy.clone();

    let c1 = ((config.k1 * config.data_range).powi(2)) as f32;
    let c2 = ((config.k2 * config.data_range).powi(2)) as f32;

    let numerator = (mu_xy.mul_scalar(2.0) + c1) * (sigma_xy.mul_scalar(2.0) + c2);
    let denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2 + 1e-6);
    let ssim_map = numerator / denominator;

    ssim_map.mean().mul_scalar(-1.0).add_scalar(1.0).reshape([1])
}

pub fn combined_loss<B: Backend>(
    pred: Tensor<B, 3>,
    target: Tensor<B, 3>,
    l1_weight: f64,
    ssim_weight: f64,
    ssim_config: &SsimConfig,
    device: &B::Device,
) -> Tensor<B, 1> {
    let l1 = (pred.clone() - target.clone()).abs().mean().reshape([1]);
    let ssim = ssim_loss(pred, target, ssim_config, device);
    l1.mul_scalar(l1_weight as f32) + ssim.mul_scalar(ssim_weight as f32)
}

fn to_nchw<B: Backend>(tensor: Tensor<B, 3>) -> Tensor<B, 4> {
    tensor.unsqueeze_dim(0).swap_dims(1, 3).swap_dims(2, 3)
}

fn gaussian_kernel_1d<B: Backend>(config: &SsimConfig, device: &B::Device) -> Tensor<B, 1> {
    let radius = (config.window_size / 2) as isize;
    let mut values = Vec::with_capacity(config.window_size);
    let mut sum = 0.0f32;

    for offset in -radius..=radius {
        let value = (-((offset * offset) as f64) / (2.0 * config.sigma * config.sigma)).exp() as f32;
        values.push(value);
        sum += value;
    }

    for value in &mut values {
        *value /= sum.max(1e-8);
    }

    Tensor::<B, 1>::from_floats(values.as_slice(), device)
}

fn separable_blur<B: Backend>(tensor: Tensor<B, 4>, kernel: Tensor<B, 1>) -> Tensor<B, 4> {
    let horizontal = blur_width(tensor, kernel.clone());
    blur_height(horizontal, kernel)
}

fn blur_width<B: Backend>(tensor: Tensor<B, 4>, kernel: Tensor<B, 1>) -> Tensor<B, 4> {
    let [_, _, _, width] = tensor.dims();
    let pad = kernel.dims()[0] / 2;
    let padded = pad_width(tensor, pad);
    let mut accum = padded
        .clone()
        .slice(s![.., .., .., 0..width])
        .mul(kernel.clone().slice(s![0]).unsqueeze());

    for index in 1..kernel.dims()[0] {
        let weight = kernel.clone().slice(s![index]).unsqueeze();
        let window = padded.clone().slice(s![.., .., .., index..index + width]);
        accum = accum + window.mul(weight);
    }

    accum
}

fn blur_height<B: Backend>(tensor: Tensor<B, 4>, kernel: Tensor<B, 1>) -> Tensor<B, 4> {
    let [_, _, height, _] = tensor.dims();
    let pad = kernel.dims()[0] / 2;
    let padded = pad_height(tensor, pad);
    let mut accum = padded
        .clone()
        .slice(s![.., .., 0..height, ..])
        .mul(kernel.clone().slice(s![0]).unsqueeze());

    for index in 1..kernel.dims()[0] {
        let weight = kernel.clone().slice(s![index]).unsqueeze();
        let window = padded.clone().slice(s![.., .., index..index + height, ..]);
        accum = accum + window.mul(weight);
    }

    accum
}

fn pad_width<B: Backend>(tensor: Tensor<B, 4>, pad: usize) -> Tensor<B, 4> {
    if pad == 0 {
        return tensor;
    }
    let width = tensor.dims()[3];
    let left = tensor.clone().slice(s![.., .., .., 0..1]).repeat_dim(3, pad);
    let right = tensor
        .clone()
        .slice(s![.., .., .., width - 1..width])
        .repeat_dim(3, pad);
    Tensor::cat(vec![left, tensor, right], 3)
}

fn pad_height<B: Backend>(tensor: Tensor<B, 4>, pad: usize) -> Tensor<B, 4> {
    if pad == 0 {
        return tensor;
    }
    let height = tensor.dims()[2];
    let top = tensor.clone().slice(s![.., .., 0..1, ..]).repeat_dim(2, pad);
    let bottom = tensor
        .clone()
        .slice(s![.., .., height - 1..height, ..])
        .repeat_dim(2, pad);
    Tensor::cat(vec![top, tensor, bottom], 2)
}

#[cfg(test)]
mod tests {
    use super::{combined_loss, ssim_loss, SsimConfig};
    use crate::training::wgpu::backend::GsBackendBase;
    use burn::tensor::Tensor;

    fn device() -> <GsBackendBase as burn::tensor::backend::Backend>::Device {
        Default::default()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_ssim_loss_identical_images() {
        let device = device();
        let image = Tensor::<GsBackendBase, 3>::ones([16, 16, 3], &device);

        let loss = ssim_loss(image.clone(), image, &SsimConfig::default(), &device)
            .into_scalar_async()
            .await
            .expect("ssim scalar");

        assert!(loss.abs() < 0.01, "expected near-zero SSIM loss, got {loss}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_combined_loss() {
        let device = device();
        let pred = Tensor::<GsBackendBase, 3>::zeros([16, 16, 3], &device);
        let target = Tensor::<GsBackendBase, 3>::ones([16, 16, 3], &device);

        let loss = combined_loss(pred, target, 1.0, 1.0, &SsimConfig::default(), &device)
            .into_scalar_async()
            .await
            .expect("combined loss scalar");

        assert!(loss > 0.5, "expected positive image mismatch loss, got {loss}");
    }
}
