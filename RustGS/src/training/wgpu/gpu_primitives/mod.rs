//! GPU primitives (radix sort, prefix sum)

pub mod prefix_sum;
pub mod radix_sort;

pub use prefix_sum::prefix_sum_u32;
pub use radix_sort::{radix_sort_by_key_u32, radix_sort_u32};

#[cfg(test)]
mod tests {
    use super::{prefix_sum_u32, radix_sort_u32};
    use crate::training::wgpu::backend::GsBackendBase;
    use burn::prelude::Backend;
    use burn::tensor::{Int, Shape, Tensor, TensorData};

    fn device() -> <GsBackendBase as Backend>::Device {
        Default::default()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_radix_sort_u32() {
        let device = device();
        let input = [5u32, 2, 8, 1, 9, 3];
        let tensor = Tensor::<GsBackendBase, 1, Int>::from_data(
            TensorData::new(input.to_vec(), Shape::new([input.len()])),
            &device,
        );

        let sorted_indices = radix_sort_u32::<GsBackendBase>(tensor, &device)
            .await
            .expect("radix sort indices");
        let data = sorted_indices
            .into_data_async()
            .await
            .expect("indices readback");
        let indices = data.as_slice::<i32>().expect("i32 indices");
        let sorted_values: Vec<u32> = indices.iter().map(|index| input[*index as usize]).collect();

        assert_eq!(sorted_values, vec![1, 2, 3, 5, 8, 9]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_prefix_sum_u32() {
        let device = device();
        let tensor = Tensor::<GsBackendBase, 1, Int>::from_data(
            TensorData::new(vec![1u32, 2, 3, 4], Shape::new([4])),
            &device,
        );

        let prefix = prefix_sum_u32::<GsBackendBase>(tensor, &device)
            .await
            .expect("prefix sum");
        let data = prefix.into_data_async().await.expect("prefix sum readback");
        let values = data.as_slice::<i32>().expect("i32 prefix sum");

        assert_eq!(values, &[1, 3, 6, 10]);
    }
}
