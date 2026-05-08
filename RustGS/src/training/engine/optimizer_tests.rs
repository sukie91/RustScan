use super::{AdamScaled, AdamScaledConfig, AdamState};
use crate::training::engine::GsBackendBase;
use burn::tensor::Tensor;

fn device() -> <GsBackendBase as burn::tensor::backend::Backend>::Device {
    Default::default()
}

#[tokio::test(flavor = "current_thread")]
async fn test_adam_scaled_step() {
    let device = device();
    let config = AdamScaledConfig {
        lr: 0.1,
        ..AdamScaledConfig::default()
    };
    let param = Tensor::<GsBackendBase, 1>::from_floats([1.0, 2.0, 3.0], &device);
    let grad = Tensor::<GsBackendBase, 1>::ones([3], &device);
    let mut state = AdamState::default();

    let updated =
        AdamScaled::<GsBackendBase>::step_tensor(&config, param.clone(), grad, &mut state);

    let before = param.into_data_async().await.expect("param readback");
    let after = updated.into_data_async().await.expect("updated readback");
    let before = before.as_slice::<f32>().expect("f32 params");
    let after = after.as_slice::<f32>().expect("f32 params");

    assert!(after
        .iter()
        .zip(before.iter())
        .all(|(after, before)| after < before));
}
