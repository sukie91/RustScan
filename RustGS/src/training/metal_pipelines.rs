use std::collections::HashMap;

use candle_core::Device;
use candle_metal_kernels::metal::ComputePipeline;

use super::metal_kernels::MetalKernel;

#[derive(Default)]
pub(super) struct MetalPipelineCache {
    pipelines: HashMap<MetalKernel, ComputePipeline>,
}

impl MetalPipelineCache {
    pub(super) fn ensure(
        &mut self,
        device: &Device,
        kernel: MetalKernel,
    ) -> candle_core::Result<bool> {
        if self.pipelines.contains_key(&kernel) {
            return Ok(false);
        }

        let metal = device.as_metal_device()?;
        let library = metal
            .new_library_with_source(kernel.source(), None)
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
        let function = library
            .get_function(kernel.function_name(), None)
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
        let pipeline = metal
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
        self.pipelines.insert(kernel, pipeline);
        Ok(true)
    }

    pub(super) fn get(&self, kernel: MetalKernel) -> Option<&ComputePipeline> {
        self.pipelines.get(&kernel)
    }
}
