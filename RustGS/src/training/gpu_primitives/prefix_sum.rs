use burn::tensor::{DType, Shape, TensorMetadata};
use burn_cubecl::cubecl::{prelude::KernelId, server::KernelArguments, CubeCount};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::{CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};
use bytemuck::{Pod, Zeroable};

const WORKGROUP_SIZE: u32 = 256;

struct PrefixSumRaw;

impl PrefixSumRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("../shaders/prefix_sum.wgsl"))
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrefixSumParams {
    len: u32,
    offset: u32,
    _pad0: u32,
    _pad1: u32,
}

#[derive(Debug)]
struct PrefixSumKernel;

impl KernelSource for PrefixSumKernel {
    fn source(&self) -> SourceTemplate {
        PrefixSumRaw
            .source()
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

pub trait PrefixSumBackend: burn::tensor::backend::Backend {
    fn prefix_sum_u32_primitive(
        input: Self::IntTensorPrimitive,
    ) -> Result<Self::IntTensorPrimitive, String>;
}

impl<F, I, BT> PrefixSumBackend for CubeBackend<WgpuRuntime, F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn prefix_sum_u32_primitive(
        input: Self::IntTensorPrimitive,
    ) -> Result<Self::IntTensorPrimitive, String> {
        let input = into_contiguous(input);
        if input.dtype() != DType::U32 && input.dtype() != DType::I32 {
            return Err(format!(
                "prefix_sum_u32 expects a 32-bit integer tensor, got {:?}",
                input.dtype()
            ));
        }

        let len = input.shape()[0];
        if len <= 1 {
            return Ok(input);
        }

        let client = input.client.clone();
        let device = input.device.clone();
        let dtype = input.dtype();
        let shape = Shape::new([len]);

        let lhs_buffer = client.empty(shape.num_elements() * core::mem::size_of::<u32>());
        let rhs_buffer = client.empty(shape.num_elements() * core::mem::size_of::<u32>());

        let mut src = CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            shape.clone(),
            lhs_buffer,
            dtype,
        );
        let mut dst =
            CubeTensor::new_contiguous(client.clone(), device.clone(), shape, rhs_buffer, dtype);

        let init_params = PrefixSumParams {
            len: len as u32,
            offset: 0,
            _pad0: 0,
            _pad1: 0,
        };
        let init_handle = client.create_from_slice(bytemuck::bytes_of(&init_params));
        let init_kernel = PrefixSumKernel;
        let init_count = CubeCount::Static((len as u32).div_ceil(WORKGROUP_SIZE), 1, 1);

        client.launch(
            Box::new(SourceKernel::new(
                init_kernel,
                CubeDim::new_1d(WORKGROUP_SIZE),
            )),
            init_count.clone(),
            KernelArguments::new().with_buffers(vec![
                input.handle.clone().binding(),
                src.handle.clone().binding(),
                init_handle.binding(),
            ]),
        );

        let mut offset = 1usize;
        while offset < len {
            let params = PrefixSumParams {
                len: len as u32,
                offset: offset as u32,
                _pad0: 0,
                _pad1: 0,
            };
            let params_handle = client.create_from_slice(bytemuck::bytes_of(&params));
            client.launch(
                Box::new(SourceKernel::new(
                    PrefixSumKernel,
                    CubeDim::new_1d(WORKGROUP_SIZE),
                )),
                init_count.clone(),
                KernelArguments::new().with_buffers(vec![
                    src.handle.clone().binding(),
                    dst.handle.clone().binding(),
                    params_handle.binding(),
                ]),
            );
            core::mem::swap(&mut src, &mut dst);
            offset <<= 1;
        }

        Ok(src)
    }
}

#[cfg(test)]
mod tests {
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
}
