use std::collections::HashMap;
use std::mem::{size_of, size_of_val};
use std::sync::{Arc, RwLockReadGuard};

use candle_core::op::BackpropOp;
use candle_core::{DType, Device, MetalStorage, Shape, Storage, Tensor};
use candle_metal_kernels::metal::Buffer;
use objc2_foundation::NSRange;

use crate::diff::diff_splat::Splats;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MetalBufferSlot {
    CameraUniforms,
    VisibleIndices,
    TileCounts,
    TileOffsets,
    TileMetadata,
    TileIndices,
    ProjectedGaussians,
    ProjectionRecords,
    VisibleSourceIndices,
    GradPositions,
    GradProjectedPositions,
    GradRotations,
    GradScales,
    GradOpacity,
    GradColors,
    GradRefineWeight,
    OutputColor,
    OutputDepth,
    OutputAlpha,
    TargetColor,
    TargetDepth,
    LossScalars,
    GradMagnitudes,
    VisibleCount,
    TotalAssignments,
    AdamGradPos,
    AdamMPos,
    AdamVPos,
    AdamParamPos,
    AdamGradScale,
    AdamMScale,
    AdamVScale,
    AdamParamScale,
    AdamGradOpacity,
    AdamMOpacity,
    AdamVOpacity,
    AdamParamOpacity,
    AdamGradColor,
    AdamMColor,
    AdamVColor,
    AdamParamColor,
    AdamHyperparams,
    SsimColorGrad,
}

impl MetalBufferSlot {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::CameraUniforms => "camera_uniforms",
            Self::VisibleIndices => "visible_indices",
            Self::TileCounts => "tile_counts",
            Self::TileOffsets => "tile_offsets",
            Self::TileMetadata => "tile_metadata",
            Self::TileIndices => "tile_indices",
            Self::ProjectedGaussians => "projected_gaussians",
            Self::ProjectionRecords => "projection_records",
            Self::VisibleSourceIndices => "visible_source_indices",
            Self::GradPositions => "grad_positions",
            Self::GradProjectedPositions => "grad_projected_positions",
            Self::GradRotations => "grad_rotations",
            Self::GradScales => "grad_scales",
            Self::GradOpacity => "grad_opacity",
            Self::GradColors => "grad_colors",
            Self::GradRefineWeight => "grad_refine_weight",
            Self::OutputColor => "output_color",
            Self::OutputDepth => "output_depth",
            Self::OutputAlpha => "output_alpha",
            Self::TargetColor => "target_color",
            Self::TargetDepth => "target_depth",
            Self::LossScalars => "loss_scalars",
            Self::GradMagnitudes => "grad_magnitudes",
            Self::VisibleCount => "visible_count",
            Self::TotalAssignments => "total_assignments",
            Self::AdamGradPos => "adam_grad_pos",
            Self::AdamMPos => "adam_m_pos",
            Self::AdamVPos => "adam_v_pos",
            Self::AdamParamPos => "adam_param_pos",
            Self::AdamGradScale => "adam_grad_scale",
            Self::AdamMScale => "adam_m_scale",
            Self::AdamVScale => "adam_v_scale",
            Self::AdamParamScale => "adam_param_scale",
            Self::AdamGradOpacity => "adam_grad_opacity",
            Self::AdamMOpacity => "adam_m_opacity",
            Self::AdamVOpacity => "adam_v_opacity",
            Self::AdamParamOpacity => "adam_param_opacity",
            Self::AdamGradColor => "adam_grad_color",
            Self::AdamMColor => "adam_m_color",
            Self::AdamVColor => "adam_v_color",
            Self::AdamParamColor => "adam_param_color",
            Self::AdamHyperparams => "adam_hyperparams",
            Self::SsimColorGrad => "ssim_color_grad",
        }
    }
}

struct PersistentMetalBuffer {
    byte_capacity: usize,
    backing: Option<Arc<Buffer>>,
}

impl PersistentMetalBuffer {
    fn new(byte_capacity: usize, backing: Option<Arc<Buffer>>) -> Self {
        Self {
            byte_capacity,
            backing,
        }
    }
}

pub(crate) struct MetalTensorView<'a> {
    storage: RwLockReadGuard<'a, Storage>,
    byte_offset: usize,
    element_count: usize,
    dtype: DType,
}

impl<'a> MetalTensorView<'a> {
    pub(crate) fn buffer(&self) -> candle_core::Result<&Buffer> {
        match &*self.storage {
            Storage::Metal(storage) => Ok(storage.buffer()),
            _ => candle_core::bail!("tensor is not backed by Metal storage"),
        }
    }

    pub(crate) fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    pub(crate) fn element_count(&self) -> usize {
        self.element_count
    }

    pub(crate) fn dtype(&self) -> DType {
        self.dtype
    }
}

pub(crate) struct MetalGaussianBindings<'a> {
    pub positions: MetalTensorView<'a>,
    pub scales: MetalTensorView<'a>,
    pub rotations: MetalTensorView<'a>,
    pub opacities: MetalTensorView<'a>,
    pub colors: MetalTensorView<'a>,
}

#[derive(Default)]
pub(crate) struct MetalResources {
    buffers: HashMap<MetalBufferSlot, PersistentMetalBuffer>,
    buffer_allocations: usize,
    buffer_reuses: usize,
}

impl MetalResources {
    pub(crate) fn buffer_allocations(&self) -> usize {
        self.buffer_allocations
    }

    pub(crate) fn buffer_reuses(&self) -> usize {
        self.buffer_reuses
    }

    pub(crate) fn buffer_capacity(&self, slot: MetalBufferSlot) -> usize {
        self.buffers
            .get(&slot)
            .map(|buffer| buffer.byte_capacity)
            .unwrap_or(0)
    }

    pub(crate) fn ensure_buffer(
        &mut self,
        device: &Device,
        slot: MetalBufferSlot,
        byte_capacity: usize,
    ) -> candle_core::Result<()> {
        let required = byte_capacity.max(1);
        if let Some(existing) = self.buffers.get(&slot) {
            if existing.byte_capacity >= required {
                self.buffer_reuses += 1;
                return Ok(());
            }
        }

        let backing = if device.is_metal() {
            let metal = device.as_metal_device()?;
            Some(metal.new_buffer(required, DType::U8, slot.label())?)
        } else {
            None
        };
        self.buffers
            .insert(slot, PersistentMetalBuffer::new(required, backing));
        self.buffer_allocations += 1;
        Ok(())
    }

    pub(crate) fn bind_tensor<'a>(
        &self,
        tensor: &'a Tensor,
    ) -> candle_core::Result<MetalTensorView<'a>> {
        let (storage, layout) = tensor.storage_and_layout();
        if !layout.is_contiguous() {
            candle_core::bail!("metal tensor binding requires contiguous layout");
        }
        let byte_offset = layout.start_offset() * tensor.dtype().size_in_bytes();
        let element_count = layout.shape().elem_count();
        match &*storage {
            Storage::Metal(_) => Ok(MetalTensorView {
                storage,
                byte_offset,
                element_count,
                dtype: tensor.dtype(),
            }),
            _ => candle_core::bail!("tensor is not backed by Metal storage"),
        }
    }

    pub(crate) fn bind_gaussians<'a>(
        &self,
        gaussians: &'a Splats,
        render_colors: &'a Tensor,
    ) -> candle_core::Result<MetalGaussianBindings<'a>> {
        Ok(MetalGaussianBindings {
            positions: self.bind_tensor(gaussians.positions())?,
            scales: self.bind_tensor(gaussians.scales.as_tensor())?,
            rotations: self.bind_tensor(gaussians.rotations.as_tensor())?,
            opacities: self.bind_tensor(gaussians.opacities.as_tensor())?,
            colors: self.bind_tensor(render_colors)?,
        })
    }

    pub(crate) fn buffer_handle(
        &self,
        slot: MetalBufferSlot,
    ) -> candle_core::Result<Option<&Arc<Buffer>>> {
        Ok(self
            .buffers
            .get(&slot)
            .and_then(|buffer| buffer.backing.as_ref()))
    }

    pub(crate) fn tensor_from_buffer<S: Into<Shape>>(
        &self,
        device: &Device,
        slot: MetalBufferSlot,
        element_count: usize,
        dtype: DType,
        shape: S,
    ) -> candle_core::Result<Tensor> {
        let Some(buffer) = self
            .buffers
            .get(&slot)
            .and_then(|buffer| buffer.backing.as_ref().cloned())
        else {
            candle_core::bail!("buffer {:?} is not backed by Metal", slot);
        };
        let metal = device.as_metal_device()?.clone();
        let storage = Storage::Metal(MetalStorage::new(buffer, metal, element_count, dtype));
        Ok(Tensor::from_storage(
            storage,
            shape,
            BackpropOp::none(),
            false,
        ))
    }

    #[cfg(test)]
    pub(crate) fn stage_tensor_from_slice<T: candle_core::WithDType + Copy, S: Into<Shape>>(
        &mut self,
        device: &Device,
        slot: MetalBufferSlot,
        values: &[T],
        shape: S,
    ) -> candle_core::Result<Tensor> {
        let shape = shape.into();
        if !device.is_metal() {
            return Tensor::from_slice(values, shape, device);
        }
        self.write_slice(device, slot, values)?;
        self.tensor_from_buffer(device, slot, values.len(), T::DTYPE, shape)
    }

    pub(crate) fn read_tensor_flat<T: candle_core::WithDType + Copy>(
        &self,
        device: &Device,
        tensor: &Tensor,
    ) -> candle_core::Result<Vec<T>> {
        let element_count = tensor.elem_count();
        if !device.is_metal() {
            return tensor.flatten_all()?.to_vec1::<T>();
        }

        let view = self.bind_tensor(tensor)?;
        if view.dtype() != T::DTYPE {
            candle_core::bail!(
                "tensor dtype mismatch for runtime read: expected {:?}, got {:?}",
                T::DTYPE,
                view.dtype()
            );
        }
        if view.element_count() != element_count {
            candle_core::bail!(
                "tensor element count mismatch for runtime read: expected {element_count}, got {}",
                view.element_count()
            );
        }

        device.synchronize()?;
        let buffer = view.buffer()?;
        let byte_offset = view.byte_offset();
        let ptr = unsafe { (buffer.contents() as *const u8).add(byte_offset) }.cast::<T>();
        let values = unsafe { std::slice::from_raw_parts(ptr, element_count) };
        Ok(values.to_vec())
    }

    pub(crate) fn read_buffer_structs<T: Copy>(
        &self,
        device: &Device,
        slot: MetalBufferSlot,
        len: usize,
    ) -> candle_core::Result<Vec<T>> {
        let Some(buffer) = self
            .buffers
            .get(&slot)
            .and_then(|buffer| buffer.backing.as_ref())
        else {
            candle_core::bail!("buffer {:?} is not backed by Metal", slot);
        };
        device.synchronize()?;
        let values = unsafe { std::slice::from_raw_parts(buffer.contents() as *const T, len) };
        Ok(values.to_vec())
    }

    #[cfg(test)]
    pub(crate) fn read_u32_buffer(
        &self,
        device: &Device,
        slot: MetalBufferSlot,
        len: usize,
    ) -> candle_core::Result<Vec<u32>> {
        let Some(buffer) = self
            .buffers
            .get(&slot)
            .and_then(|buffer| buffer.backing.as_ref())
        else {
            candle_core::bail!("buffer {:?} is not backed by Metal", slot);
        };
        device.synchronize()?;
        let values = unsafe { std::slice::from_raw_parts(buffer.contents() as *const u32, len) };
        Ok(values.to_vec())
    }

    pub(crate) fn write_struct<T: Copy>(
        &mut self,
        device: &Device,
        slot: MetalBufferSlot,
        value: &T,
    ) -> candle_core::Result<()> {
        let data =
            unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) };
        self.write_bytes(device, slot, data)
    }

    pub(crate) fn write_slice<T: Copy>(
        &mut self,
        device: &Device,
        slot: MetalBufferSlot,
        values: &[T],
    ) -> candle_core::Result<()> {
        let data = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values))
        };
        self.write_bytes(device, slot, data)
    }

    fn write_bytes(
        &mut self,
        device: &Device,
        slot: MetalBufferSlot,
        data: &[u8],
    ) -> candle_core::Result<()> {
        self.ensure_buffer(device, slot, data.len())?;
        let Some(buffer) = self
            .buffers
            .get(&slot)
            .and_then(|buffer| buffer.backing.as_ref())
        else {
            return Ok(());
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                buffer.contents().cast::<u8>(),
                data.len(),
            );
        }
        buffer.did_modify_range(NSRange::new(0, data.len()));
        Ok(())
    }
}
