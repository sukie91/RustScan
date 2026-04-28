use burn::prelude::*;
use burn::tensor::{
    module::conv2d,
    ops::{ConvOptions, PadMode},
    s,
};

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

#[cfg(test)]
pub fn ssim_loss<B: Backend>(
    pred: Tensor<B, 3>,
    target: Tensor<B, 3>,
    config: &SsimConfig,
    device: &B::Device,
) -> Tensor<B, 1> {
    let pred = to_nchw(pred);
    let target = to_nchw(target);
    let kernel = gaussian_kernel_1d::<B>(config, device);
    ssim_loss_with_kernel(pred, target, kernel, config)
}

pub fn ssim_loss_with_kernel<B: Backend>(
    pred: Tensor<B, 4>,
    target: Tensor<B, 4>,
    kernel: Tensor<B, 1>,
    config: &SsimConfig,
) -> Tensor<B, 1> {
    let pred = pred;
    let target = target;

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
    let denominator = (mu_x_sq + mu_y_sq + c1).clamp_min(1e-8_f32)
        * (sigma_x_sq + sigma_y_sq + c2).clamp_min(1e-8_f32);
    let ssim_map = numerator / denominator;

    ssim_map
        .mean()
        .mul_scalar(-1.0)
        .add_scalar(1.0)
        .reshape([1])
}

#[cfg(test)]
pub fn combined_loss<B: Backend>(
    pred: Tensor<B, 3>,
    target: Tensor<B, 3>,
    l1_weight: f64,
    ssim_weight: f64,
    gradient_weight: f64,
    robust_delta: f64,
    outlier_threshold: f64,
    outlier_weight: f64,
    ssim_config: &SsimConfig,
    device: &B::Device,
) -> Tensor<B, 1> {
    let kernel = gaussian_kernel_1d::<B>(ssim_config, device);
    combined_loss_with_kernel(
        pred,
        target,
        l1_weight,
        ssim_weight,
        gradient_weight,
        robust_delta,
        outlier_threshold,
        outlier_weight,
        ssim_config,
        kernel,
    )
}

pub fn combined_loss_with_kernel<B: Backend>(
    pred: Tensor<B, 3>,
    target: Tensor<B, 3>,
    l1_weight: f64,
    ssim_weight: f64,
    gradient_weight: f64,
    robust_delta: f64,
    outlier_threshold: f64,
    outlier_weight: f64,
    ssim_config: &SsimConfig,
    ssim_kernel: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let l1 = reconstruction_residual_loss(
        pred.clone(),
        target.clone(),
        robust_delta as f32,
        outlier_threshold as f32,
        outlier_weight as f32,
    );
    let gradient = gradient_difference_loss(pred.clone(), target.clone());
    let ssim = ssim_loss_with_kernel(to_nchw(pred), to_nchw(target), ssim_kernel, ssim_config);
    l1.mul_scalar(l1_weight as f32)
        + ssim.mul_scalar(ssim_weight as f32)
        + gradient.mul_scalar(gradient_weight as f32)
}

fn reconstruction_residual_loss<B: Backend>(
    pred: Tensor<B, 3>,
    target: Tensor<B, 3>,
    robust_delta: f32,
    outlier_threshold: f32,
    outlier_weight: f32,
) -> Tensor<B, 1> {
    let abs_residual = (pred - target).abs();
    let loss = if robust_delta.is_finite() && robust_delta > 0.0 {
        // Saturating L1: behaves like L1 near zero but reduces the influence
        // of large residuals from dynamic objects, occlusion changes, or bad pixels.
        let delta = robust_delta.max(1e-6);
        abs_residual.clone().mul_scalar(delta) / (abs_residual + delta)
    } else if outlier_threshold.is_finite()
        && outlier_threshold > 0.0
        && outlier_weight.is_finite()
        && outlier_weight < 1.0
    {
        // Soft outlier weighting: preserves near-L1 behavior for small residuals
        // while retaining a configurable gradient floor for high-residual pixels.
        let threshold = outlier_threshold.max(1e-6);
        let floor = outlier_weight.clamp(0.0, 1.0);
        let adaptive_weight = abs_residual.clone().mul_scalar(0.0).add_scalar(floor)
            + (abs_residual
                .clone()
                .mul_scalar(0.0)
                .add_scalar(1.0 - floor)
                .mul_scalar(threshold)
                / (abs_residual.clone() + threshold));
        abs_residual * adaptive_weight
    } else {
        abs_residual
    };
    loss.mean().reshape([1])
}

fn gradient_difference_loss<B: Backend>(pred: Tensor<B, 3>, target: Tensor<B, 3>) -> Tensor<B, 1> {
    let [height, width, _channels] = pred.dims();
    debug_assert_eq!(pred.dims(), target.dims());
    let dx_pred =
        pred.clone().slice(s![.., 1..width, ..]) - pred.clone().slice(s![.., 0..width - 1, ..]);
    let dx_target =
        target.clone().slice(s![.., 1..width, ..]) - target.clone().slice(s![.., 0..width - 1, ..]);
    let dy_pred = pred.clone().slice(s![1..height, .., ..]) - pred.slice(s![0..height - 1, .., ..]);
    let dy_target =
        target.clone().slice(s![1..height, .., ..]) - target.slice(s![0..height - 1, .., ..]);
    let dx = (dx_pred - dx_target).abs().mean();
    let dy = (dy_pred - dy_target).abs().mean();
    (dx + dy).mul_scalar(0.5).reshape([1])
}

fn to_nchw<B: Backend>(tensor: Tensor<B, 3>) -> Tensor<B, 4> {
    tensor.unsqueeze_dim(0).swap_dims(1, 3).swap_dims(2, 3)
}

pub fn gaussian_kernel_1d<B: Backend>(config: &SsimConfig, device: &B::Device) -> Tensor<B, 1> {
    let radius = (config.window_size / 2) as isize;
    let mut values = Vec::with_capacity(config.window_size);
    let mut sum = 0.0f32;

    for offset in -radius..=radius {
        let value =
            (-((offset * offset) as f64) / (2.0 * config.sigma * config.sigma)).exp() as f32;
        values.push(value);
        sum += value;
    }

    for value in &mut values {
        *value /= sum.max(1e-8);
    }

    Tensor::<B, 1>::from_floats(values.as_slice(), device)
}

fn separable_blur<B: Backend>(tensor: Tensor<B, 4>, kernel: Tensor<B, 1>) -> Tensor<B, 4> {
    let kernel_size = kernel.dims()[0];
    let pad = kernel_size / 2;
    let [_n, channels, _height, _width] = tensor.dims();
    let horizontal_kernel = kernel
        .clone()
        .reshape([1, 1, 1, kernel_size])
        .repeat_dim(0, channels);
    let horizontal = conv2d(
        tensor.pad([(0, 0), (pad, pad)], PadMode::Edge),
        horizontal_kernel,
        None,
        ConvOptions::new([1, 1], [0, 0], [1, 1], channels),
    );
    let vertical_kernel = kernel
        .reshape([1, 1, kernel_size, 1])
        .repeat_dim(0, channels);

    conv2d(
        horizontal.pad([(pad, pad), (0, 0)], PadMode::Edge),
        vertical_kernel,
        None,
        ConvOptions::new([1, 1], [0, 0], [1, 1], channels),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        combined_loss, gradient_difference_loss, reconstruction_residual_loss, ssim_loss,
        SsimConfig,
    };
    use crate::training::engine::GsBackendBase;
    use burn::prelude::Backend;
    use burn::tensor::{ops::PadMode, s, Tensor, TensorData};

    fn device() -> <GsBackendBase as burn::tensor::backend::Backend>::Device {
        Default::default()
    }

    fn separable_blur_reference<B: Backend>(
        tensor: Tensor<B, 4>,
        kernel: Tensor<B, 1>,
    ) -> Tensor<B, 4> {
        let horizontal = blur_width_reference(tensor, kernel.clone());
        blur_height_reference(horizontal, kernel)
    }

    fn blur_width_reference<B: Backend>(
        tensor: Tensor<B, 4>,
        kernel: Tensor<B, 1>,
    ) -> Tensor<B, 4> {
        let [_, _, _, width] = tensor.dims();
        let pad = kernel.dims()[0] / 2;
        let padded = tensor.pad([(0, 0), (pad, pad)], PadMode::Edge);
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

    fn blur_height_reference<B: Backend>(
        tensor: Tensor<B, 4>,
        kernel: Tensor<B, 1>,
    ) -> Tensor<B, 4> {
        let [_, _, height, _] = tensor.dims();
        let pad = kernel.dims()[0] / 2;
        let padded = tensor.pad([(pad, pad), (0, 0)], PadMode::Edge);
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

    #[tokio::test(flavor = "current_thread")]
    async fn test_ssim_loss_identical_images() {
        let device = device();
        let image = Tensor::<GsBackendBase, 3>::ones([16, 16, 3], &device);

        let loss = ssim_loss(image.clone(), image, &SsimConfig::default(), &device)
            .into_scalar_async()
            .await
            .expect("ssim scalar");

        assert!(
            loss.abs() < 0.01,
            "expected near-zero SSIM loss, got {loss}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_ssim_loss_zero_images_is_finite() {
        let device = device();
        let image = Tensor::<GsBackendBase, 3>::zeros([16, 16, 3], &device);

        let loss = ssim_loss(image.clone(), image, &SsimConfig::default(), &device)
            .into_scalar_async()
            .await
            .expect("ssim scalar");

        assert!(loss.is_finite(), "expected finite SSIM loss, got {loss}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_combined_loss() {
        let device = device();
        let pred = Tensor::<GsBackendBase, 3>::zeros([16, 16, 3], &device);
        let target = Tensor::<GsBackendBase, 3>::ones([16, 16, 3], &device);

        let loss = combined_loss(
            pred,
            target,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            &SsimConfig::default(),
            &device,
        )
        .into_scalar_async()
        .await
        .expect("combined loss scalar");

        assert!(
            loss > 0.5,
            "expected positive image mismatch loss, got {loss}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_robust_residual_loss_downweights_large_errors() {
        let device = device();
        let pred = Tensor::<GsBackendBase, 3>::zeros([1, 2, 3], &device);
        let target = Tensor::<GsBackendBase, 3>::from_data(
            TensorData::new(vec![0.1, 0.1, 0.1, 1.0, 1.0, 1.0], [1, 2, 3]),
            &device,
        );

        let exact = reconstruction_residual_loss(pred.clone(), target.clone(), 0.0, 0.0, 1.0)
            .into_scalar_async()
            .await
            .expect("exact residual loss");
        let robust = reconstruction_residual_loss(pred, target, 0.1, 0.0, 1.0)
            .into_scalar_async()
            .await
            .expect("robust residual loss");

        assert!(robust < exact, "robust loss should downweight outliers");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_soft_outlier_loss_preserves_gradient_floor() {
        let device = device();
        let pred = Tensor::<GsBackendBase, 3>::zeros([1, 2, 3], &device);
        let target = Tensor::<GsBackendBase, 3>::from_data(
            TensorData::new(vec![0.1, 0.1, 0.1, 1.0, 1.0, 1.0], [1, 2, 3]),
            &device,
        );

        let exact = reconstruction_residual_loss(pred.clone(), target.clone(), 0.0, 0.0, 1.0)
            .into_scalar_async()
            .await
            .expect("exact residual loss");
        let weighted = reconstruction_residual_loss(pred, target, 0.0, 0.25, 0.25)
            .into_scalar_async()
            .await
            .expect("soft outlier residual loss");

        assert!(
            weighted < exact,
            "soft outlier loss should downweight large residuals"
        );
        assert!(
            weighted > 0.0,
            "soft outlier loss should retain a gradient floor"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_gradient_difference_loss_detects_edge_mismatch() {
        let device = device();
        let pred = Tensor::<GsBackendBase, 3>::zeros([4, 4, 3], &device);
        let mut target_values = vec![0.0f32; 4 * 4 * 3];
        for y in 0..4 {
            for x in 2..4 {
                let base = (y * 4 + x) * 3;
                target_values[base] = 1.0;
                target_values[base + 1] = 1.0;
                target_values[base + 2] = 1.0;
            }
        }
        let target = Tensor::<GsBackendBase, 3>::from_data(
            TensorData::new(target_values, [4, 4, 3]),
            &device,
        );

        let loss = gradient_difference_loss(pred, target)
            .into_scalar_async()
            .await
            .expect("gradient loss scalar");

        assert!(loss > 0.0, "expected edge mismatch gradient loss");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_separable_blur_matches_reference() {
        let device = device();
        let tensor = Tensor::<GsBackendBase, 4>::from_data(
            TensorData::new((0..120).map(|v| v as f32 / 120.0).collect(), [1, 3, 5, 8]),
            &device,
        );
        let kernel = Tensor::<GsBackendBase, 1>::from_floats([0.25, 0.5, 0.25], &device);

        let reference = separable_blur_reference(tensor.clone(), kernel.clone())
            .into_data_async()
            .await
            .expect("reference blur");
        let optimized = super::separable_blur(tensor, kernel)
            .into_data_async()
            .await
            .expect("optimized blur");

        let reference = reference.as_slice::<f32>().expect("reference f32");
        let optimized = optimized.as_slice::<f32>().expect("optimized f32");

        for (idx, (lhs, rhs)) in reference.iter().zip(optimized.iter()).enumerate() {
            let delta = (lhs - rhs).abs();
            assert!(
                delta < 1e-4,
                "blur mismatch at index {idx}: {lhs} vs {rhs} (delta={delta})"
            );
        }
    }
}
