use super::RadixSortBackend;
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

async fn assert_argsort(input: &[u32]) {
    let device = device();
    let keys = Tensor::<GsBackendBase, 1, Int>::from_data(
        TensorData::new(input.to_vec(), Shape::new([input.len()])),
        &device,
    );
    let indices = Tensor::<GsBackendBase, 1, Int>::from_data(
        TensorData::new(
            (0..input.len() as i32).collect::<Vec<_>>(),
            Shape::new([input.len()]),
        ),
        &device,
    );
    let (_, result) = <GsBackendBase as RadixSortBackend>::radix_sort_by_key_u32_primitive(
        keys.into_primitive(),
        indices.into_primitive(),
    )
    .expect("argsort");
    let result = Tensor::<GsBackendBase, 1, Int>::from_primitive(result);
    let data = result.into_data_async().await.expect("readback");
    let values = data.as_slice::<i32>().expect("i32 data");

    let mut expected: Vec<i32> = (0..input.len() as i32).collect();
    expected.sort_by_key(|index| (input[*index as usize], *index));

    assert_eq!(values, expected.as_slice());
}

async fn assert_sort_by_key(input_keys: &[u32], input_values: &[u32]) {
    let device = device();
    let keys = Tensor::<GsBackendBase, 1, Int>::from_data(
        TensorData::new(input_keys.to_vec(), Shape::new([input_keys.len()])),
        &device,
    );
    let values = Tensor::<GsBackendBase, 1, Int>::from_data(
        TensorData::new(input_values.to_vec(), Shape::new([input_values.len()])),
        &device,
    );

    let (sorted_keys, sorted_values) =
        <GsBackendBase as RadixSortBackend>::radix_sort_by_key_u32_primitive(
            keys.into_primitive(),
            values.into_primitive(),
        )
        .expect("sort by key");
    let sorted_keys = Tensor::<GsBackendBase, 1, Int>::from_primitive(sorted_keys);
    let sorted_values = Tensor::<GsBackendBase, 1, Int>::from_primitive(sorted_values);
    let sorted_keys = sorted_keys.into_data_async().await.expect("keys readback");
    let sorted_values = sorted_values
        .into_data_async()
        .await
        .expect("values readback");
    let sorted_keys = sorted_keys.as_slice::<i32>().expect("i32 keys");
    let sorted_values = sorted_values.as_slice::<i32>().expect("i32 values");

    let mut expected: Vec<(u32, u32, usize)> = input_keys
        .iter()
        .copied()
        .zip(input_values.iter().copied())
        .enumerate()
        .map(|(index, (key, value))| (key, value, index))
        .collect();
    expected.sort_by_key(|(key, _, index)| (*key, *index));

    let expected_keys: Vec<i32> = expected.iter().map(|(key, _, _)| *key as i32).collect();
    let expected_values: Vec<i32> =
        expected.iter().map(|(_, value, _)| *value as i32).collect();

    assert_eq!(sorted_keys, expected_keys.as_slice());
    assert_eq!(sorted_values, expected_values.as_slice());
}

#[test]
fn test_radix_sort_empty() {
    run_async(assert_argsort(&[]));
}

#[test]
fn test_radix_sort_single() {
    run_async(assert_argsort(&[42]));
}

#[test]
fn test_radix_sort_small() {
    run_async(assert_argsort(&[9, 4, 1, 7, 4, 3, 2]));
}

#[test]
fn test_radix_sort_power_of_two() {
    run_async(assert_argsort(&[10, 2, 8, 6, 4, 0, 12, 14]));
}

#[test]
fn test_radix_sort_large() {
    let input: Vec<u32> = (0..513).map(|index| ((index * 37) % 101) as u32).collect();
    run_async(assert_argsort(&input));
}

#[test]
fn test_radix_sort_by_key_small() {
    run_async(assert_sort_by_key(
        &[9, 4, 1, 7, 4, 3, 2],
        &[90, 40, 10, 70, 41, 30, 20],
    ));
}
