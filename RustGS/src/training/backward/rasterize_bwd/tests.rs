use super::rasterize_bwd;
use crate::training::engine::GsBackendBase;
use burn::prelude::*;
use burn::tensor::{Int, Tensor, TensorData};

#[test]
fn test_rasterize_bwd_kernel_writes_output_buffer() {
    let device = <GsBackendBase as Backend>::Device::default();
    let compact_gid_from_isect =
        Tensor::<GsBackendBase, 1, Int>::from_data(TensorData::from([0i32]), &device);
    let tile_offsets =
        Tensor::<GsBackendBase, 1, Int>::from_data(TensorData::from([0i32, 1i32]), &device);
    let projected_splats = Tensor::<GsBackendBase, 2>::from_data(
        TensorData::new(vec![8.5f32, 8.5, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.9], [1, 9]),
        &device,
    );
    let out_img = Tensor::<GsBackendBase, 3>::ones([16, 16, 4], &device);
    let v_output = Tensor::<GsBackendBase, 3>::ones([16, 16, 4], &device);

    let bwd = rasterize_bwd::<GsBackendBase>(
        compact_gid_from_isect,
        tile_offsets,
        projected_splats,
        out_img,
        v_output,
        1,
        (16, 16),
        (1, 1),
        [0.0, 0.0, 0.0],
        &device,
    );

    let data = bwd.v_splats.into_data();
    let values = data.as_slice::<f32>().expect("v_splats should be f32");
    let mean_abs = values.iter().map(|v| v.abs()).sum::<f32>() / values.len() as f32;
    assert!(
        mean_abs > 1e-6,
        "expected rasterize_bwd kernel to produce non-zero grads, mean_abs={mean_abs:e}"
    );

    let stats = bwd.screen_grad_splats.into_data();
    let values = stats
        .as_slice::<f32>()
        .expect("screen_grad_splats should be f32");
    assert_eq!(values.len(), 5);
    assert!(
        values[2].abs() + values[3].abs() >= values[0].abs() + values[1].abs(),
        "absolute screen-space stats should dominate signed stats: {values:?}"
    );
    assert!(
        values[4] > 0.0,
        "expected per-pixel coverage count to be positive: {values:?}"
    );
}
