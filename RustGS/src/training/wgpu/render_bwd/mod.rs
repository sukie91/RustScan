//! Backward rendering pipeline

pub mod autodiff;
pub mod project_bwd;
pub mod rasterize_bwd;

pub use autodiff::render_splats;

#[cfg(test)]
mod tests {
    use super::render_splats;
    use crate::core::GaussianCamera;
    use crate::sh::rgb_to_sh0_value;
    use crate::training::wgpu::backend::GsDiffBackend;
    use crate::training::wgpu::splats::host_splats_to_device;
    use crate::training::HostSplats;
    use crate::{Intrinsics, SE3};

    fn test_camera() -> GaussianCamera {
        GaussianCamera::new(Intrinsics::from_focal(500.0, 64, 64), SE3::identity())
    }

    fn test_splats() -> HostSplats {
        HostSplats::from_raw_parts(
            vec![0.0, 0.0, 2.0],
            vec![0.2f32.ln(), 0.2f32.ln(), 0.2f32.ln()],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0],
            [1.0, 0.5, 0.25].map(rgb_to_sh0_value).into(),
            0,
        )
        .expect("valid host splats")
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_render_splats_autodiff() {
        let device = <GsDiffBackend as burn::tensor::backend::Backend>::Device::default();
        let splats = host_splats_to_device::<GsDiffBackend>(&test_splats(), &device);
        let image = render_splats(&splats, &test_camera(), (64, 64), [0.0, 0.0, 0.0]).await;
        let loss = image.mean();
        let mut grads = loss.backward();

        assert!(splats.transforms.grad_remove(&mut grads).is_some());
        assert!(splats.sh_coeffs.grad_remove(&mut grads).is_some());
        assert!(splats.raw_opacities.grad_remove(&mut grads).is_some());
    }
}
