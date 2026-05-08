use super::PrefixSumBackend;
use crate::training::engine::GsBackendBase;
use burn::tensor::{Int, Shape, Tensor, TensorData};

fn device() -> <GsBackendBase as burn::tensor::backend::Backend>::Device {
    Default::default()
}

fn run_async<T>(future: impl core::future::Future<Output = T>) -> T {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
        .block_on(future)
}

async fn assert_prefix(input: &[u32]) {
    let device = device();
    let tensor = Tensor::<GsBackendBase, 1, Int>::from_data(
        TensorData::new(input.to_vec(), Shape::new([input.len()])),
        &device,
    );
    let primitive =
        <GsBackendBase as PrefixSumBackend>::prefix_sum_u32_primitive(tensor.into_primitive())
            .expect("prefix sum");
    let result = Tensor::<GsBackendBase, 1, Int>::from_primitive(primitive);
    let data = result.into_data_async().await.expect("readback");
    let values = data.as_slice::<i32>().expect("i32 data");

    let expected: Vec<i32> = input
        .iter()
        .scan(0i32, |acc, value| {
            *acc += *value as i32;
            Some(*acc)
        })
        .collect();

    assert_eq!(values, expected.as_slice());
}

#[test]
fn test_prefix_sum_empty() {
    run_async(assert_prefix(&[]));
}

#[test]
fn test_prefix_sum_single() {
    run_async(assert_prefix(&[7]));
}

#[test]
fn test_prefix_sum_power_of_two() {
    run_async(assert_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]));
}

#[test]
fn test_prefix_sum_non_power_of_two() {
    run_async(assert_prefix(&[4, 1, 0, 9, 2, 8, 3]));
}
