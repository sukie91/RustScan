use burn::tensor::{DType, Int, Shape, Tensor, TensorData, TensorMetadata};
use burn_cubecl::cubecl::{prelude::KernelId, server::KernelArguments, CubeCount};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::{CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};
use bytemuck::{Pod, Zeroable};

const WORKGROUP_SIZE: u32 = 256;
const MODE_INIT: u32 = 0;
const MODE_COMPARE: u32 = 1;
const MODE_COMPACT: u32 = 2;

struct RadixSortRaw;

impl RadixSortRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("../shaders/radix_sort.wgsl"))
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SortParams {
    mode: u32,
    len: u32,
    padded_len: u32,
    stage_k: u32,
    stage_j: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(Debug)]
struct RadixSortKernel;

impl KernelSource for RadixSortKernel {
    fn source(&self) -> SourceTemplate {
        RadixSortRaw
            .source()
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

pub trait RadixSortBackend: burn::tensor::backend::Backend {
    fn radix_sort_u32_primitive(
        keys: Self::IntTensorPrimitive,
    ) -> Result<Self::IntTensorPrimitive, String>;

    fn radix_sort_by_key_u32_primitive(
        keys: Self::IntTensorPrimitive,
        values: Self::IntTensorPrimitive,
    ) -> Result<(Self::IntTensorPrimitive, Self::IntTensorPrimitive), String>;
}

impl<F, I, BT> RadixSortBackend for CubeBackend<WgpuRuntime, F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn radix_sort_u32_primitive(
        keys: Self::IntTensorPrimitive,
    ) -> Result<Self::IntTensorPrimitive, String> {
        let len = keys.shape()[0];
        let device = keys.device.clone();
        let indices = Tensor::<Self, 1, Int>::from_data(
            TensorData::new((0..len as i32).collect::<Vec<_>>(), Shape::new([len])),
            &device,
        )
        .into_primitive();
        let (_, sorted_indices) = Self::radix_sort_by_key_u32_primitive(keys, indices)?;
        Ok(sorted_indices)
    }

    fn radix_sort_by_key_u32_primitive(
        keys: Self::IntTensorPrimitive,
        values: Self::IntTensorPrimitive,
    ) -> Result<(Self::IntTensorPrimitive, Self::IntTensorPrimitive), String> {
        let keys = into_contiguous(keys);
        let values = into_contiguous(values);

        if keys.dtype() != DType::U32 && keys.dtype() != DType::I32 {
            return Err(format!(
                "radix_sort_by_key_u32 expects 32-bit integer keys, got {:?}",
                keys.dtype()
            ));
        }
        if values.dtype() != DType::U32 && values.dtype() != DType::I32 {
            return Err(format!(
                "radix_sort_by_key_u32 expects 32-bit integer values, got {:?}",
                values.dtype()
            ));
        }
        if keys.shape()[0] != values.shape()[0] {
            return Err(format!(
                "radix_sort_by_key_u32 expects matching lengths, got {} keys and {} values",
                keys.shape()[0],
                values.shape()[0]
            ));
        }
        if keys.device != values.device {
            return Err("radix_sort_by_key_u32 expects keys and values on the same device".into());
        }

        let len = keys.shape()[0];
        if len == 0 {
            return Ok((keys, values));
        }
        if len == 1 {
            return Ok((keys, values));
        }

        let padded_len = len.next_power_of_two();
        let client = keys.client.clone();
        let device = keys.device.clone();
        let keys_dtype = keys.dtype();
        let values_dtype = values.dtype();
        let padded_shape = Shape::new([padded_len]);
        let output_shape = Shape::new([len]);

        let work_keys = CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            padded_shape.clone(),
            client.empty(padded_shape.num_elements() * core::mem::size_of::<u32>()),
            keys_dtype,
        );
        let work_values = CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            padded_shape.clone(),
            client.empty(padded_shape.num_elements() * core::mem::size_of::<u32>()),
            values_dtype,
        );
        let work_indices = CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            padded_shape,
            client.empty(padded_len * core::mem::size_of::<u32>()),
            DType::U32,
        );
        let output_keys = CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            output_shape.clone(),
            client.empty(output_shape.num_elements() * core::mem::size_of::<u32>()),
            keys_dtype,
        );
        let output_values = CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            output_shape,
            client.empty(len * core::mem::size_of::<u32>()),
            values_dtype,
        );

        let cube_dim = CubeDim::new_1d(WORKGROUP_SIZE);
        let init_count = CubeCount::Static((padded_len as u32).div_ceil(WORKGROUP_SIZE), 1, 1);

        let init_params = SortParams {
            mode: MODE_INIT,
            len: len as u32,
            padded_len: padded_len as u32,
            stage_k: 0,
            stage_j: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let init_handle = client.create_from_slice(bytemuck::bytes_of(&init_params));
        client.launch(
            Box::new(SourceKernel::new(RadixSortKernel, cube_dim)),
            init_count.clone(),
            KernelArguments::new().with_buffers(vec![
                keys.handle.clone().binding(),
                values.handle.clone().binding(),
                work_keys.handle.clone().binding(),
                work_values.handle.clone().binding(),
                work_indices.handle.clone().binding(),
                output_keys.handle.clone().binding(),
                output_values.handle.clone().binding(),
                init_handle.binding(),
            ]),
        );

        let mut k = 2usize;
        while k <= padded_len {
            let mut j = k / 2;
            while j > 0 {
                let params = SortParams {
                    mode: MODE_COMPARE,
                    len: len as u32,
                    padded_len: padded_len as u32,
                    stage_k: k as u32,
                    stage_j: j as u32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                };
                let params_handle = client.create_from_slice(bytemuck::bytes_of(&params));
                client.launch(
                    Box::new(SourceKernel::new(RadixSortKernel, cube_dim)),
                    init_count.clone(),
                    KernelArguments::new().with_buffers(vec![
                        keys.handle.clone().binding(),
                        values.handle.clone().binding(),
                        work_keys.handle.clone().binding(),
                        work_values.handle.clone().binding(),
                        work_indices.handle.clone().binding(),
                        output_keys.handle.clone().binding(),
                        output_values.handle.clone().binding(),
                        params_handle.binding(),
                    ]),
                );
                j >>= 1;
            }
            k <<= 1;
        }

        let compact_params = SortParams {
            mode: MODE_COMPACT,
            len: len as u32,
            padded_len: padded_len as u32,
            stage_k: 0,
            stage_j: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let compact_handle = client.create_from_slice(bytemuck::bytes_of(&compact_params));
        let compact_count = CubeCount::Static((len as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        client.launch(
            Box::new(SourceKernel::new(RadixSortKernel, cube_dim)),
            compact_count,
            KernelArguments::new().with_buffers(vec![
                keys.handle.clone().binding(),
                values.handle.clone().binding(),
                work_keys.handle.clone().binding(),
                work_values.handle.clone().binding(),
                work_indices.handle.clone().binding(),
                output_keys.handle.clone().binding(),
                output_values.handle.clone().binding(),
                compact_handle.binding(),
            ]),
        );

        Ok((output_keys, output_values))
    }
}

/// GPU sort for 32-bit keys, returns sorted indices.
///
/// The current implementation uses a GPU compare-exchange network internally to
/// provide the required argsort primitive on burn+wgpu.
pub async fn radix_sort_u32<B: RadixSortBackend>(
    keys: Tensor<B, 1, Int>,
    _device: &B::Device,
) -> Result<Tensor<B, 1, Int>, String> {
    let primitive = B::radix_sort_u32_primitive(keys.into_primitive())?;
    Ok(Tensor::from_primitive(primitive))
}

/// GPU sort for 32-bit keys while carrying a companion values array through the sort.
pub async fn radix_sort_by_key_u32<B: RadixSortBackend>(
    keys: Tensor<B, 1, Int>,
    values: Tensor<B, 1, Int>,
    bits: u32,
    _device: &B::Device,
) -> Result<(Tensor<B, 1, Int>, Tensor<B, 1, Int>), String> {
    debug_assert!(bits <= u32::BITS);
    let (sorted_keys, sorted_values) =
        B::radix_sort_by_key_u32_primitive(keys.into_primitive(), values.into_primitive())?;
    Ok((
        Tensor::from_primitive(sorted_keys),
        Tensor::from_primitive(sorted_values),
    ))
}

#[cfg(test)]
mod tests {
    use super::{radix_sort_by_key_u32, radix_sort_u32};
    use crate::training::wgpu::backend::GsBackendBase;
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
        let tensor = Tensor::<GsBackendBase, 1, Int>::from_data(
            TensorData::new(input.to_vec(), Shape::new([input.len()])),
            &device,
        );
        let result = radix_sort_u32::<GsBackendBase>(tensor, &device)
            .await
            .expect("argsort");
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
            radix_sort_by_key_u32::<GsBackendBase>(keys, values, u32::BITS, &device)
                .await
                .expect("sort by key");
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
}
