use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Arc, RwLockReadGuard};

use candle_core::{DType, Device, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputePipeline};
use objc2_foundation::NSRange;
use objc2_metal::MTLSize;

use crate::diff::diff_splat::{DiffCamera, TrainableGaussians};

pub(crate) const METAL_TILE_SIZE: usize = 16;

const METAL_FILL_U32_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fill_u32(
    device uint* dst [[buffer(0)]],
    constant uint& value [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    dst[gid] = value;
}
"#;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ScreenRect {
    pub min_x: usize,
    pub max_x: usize,
    pub min_y: usize,
    pub max_y: usize,
}

pub(crate) struct ChunkPixelWindow {
    pub pixel_x: Tensor,
    pub pixel_y: Tensor,
    pub indices: Tensor,
    pub pixel_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MetalCameraUniform {
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    width: u32,
    height: u32,
    tile_size: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MetalTileRecord {
    start: u32,
    count: u32,
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MetalBufferSlot {
    CameraUniforms,
    TileMetadata,
    TileIndices,
}

impl MetalBufferSlot {
    fn label(self) -> &'static str {
        match self {
            Self::CameraUniforms => "camera_uniforms",
            Self::TileMetadata => "tile_metadata",
            Self::TileIndices => "tile_indices",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MetalKernel {
    FillU32,
}

impl MetalKernel {
    fn function_name(self) -> &'static str {
        match self {
            Self::FillU32 => "fill_u32",
        }
    }

    fn source(self) -> &'static str {
        match self {
            Self::FillU32 => METAL_FILL_U32_KERNEL,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct MetalRuntimeStats {
    pub tile_windows: usize,
    pub buffer_allocations: usize,
    pub buffer_reuses: usize,
    pub pipeline_compilations: usize,
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

pub(crate) struct MetalRuntime {
    device: Device,
    tile_windows: Vec<ChunkPixelWindow>,
    num_tiles_x: usize,
    num_tiles_y: usize,
    buffers: HashMap<MetalBufferSlot, PersistentMetalBuffer>,
    pipelines: HashMap<MetalKernel, ComputePipeline>,
    stats: MetalRuntimeStats,
}

impl MetalRuntime {
    pub(crate) fn new(
        render_width: usize,
        render_height: usize,
        device: Device,
    ) -> candle_core::Result<Self> {
        let num_tiles_x = render_width.div_ceil(METAL_TILE_SIZE);
        let num_tiles_y = render_height.div_ceil(METAL_TILE_SIZE);
        let mut tile_windows = Vec::with_capacity(num_tiles_x * num_tiles_y);

        for tile_y in 0..num_tiles_y {
            for tile_x in 0..num_tiles_x {
                let min_x = tile_x * METAL_TILE_SIZE;
                let min_y = tile_y * METAL_TILE_SIZE;
                let max_x = (min_x + METAL_TILE_SIZE)
                    .min(render_width)
                    .saturating_sub(1);
                let max_y = (min_y + METAL_TILE_SIZE)
                    .min(render_height)
                    .saturating_sub(1);
                tile_windows.push(build_chunk_pixel_window(
                    &device,
                    render_width,
                    ScreenRect {
                        min_x,
                        max_x,
                        min_y,
                        max_y,
                    },
                )?);
            }
        }

        Ok(Self {
            device,
            tile_windows,
            num_tiles_x,
            num_tiles_y,
            buffers: HashMap::new(),
            pipelines: HashMap::new(),
            stats: MetalRuntimeStats {
                tile_windows: num_tiles_x * num_tiles_y,
                ..Default::default()
            },
        })
    }

    pub(crate) fn tile_window(&self, tile_idx: usize) -> candle_core::Result<&ChunkPixelWindow> {
        self.tile_windows
            .get(tile_idx)
            .ok_or_else(|| candle_core::Error::Msg(format!("invalid tile index {tile_idx}")))
    }

    pub(crate) fn tile_grid(&self) -> (usize, usize) {
        (self.num_tiles_x, self.num_tiles_y)
    }

    pub(crate) fn stats(&self) -> MetalRuntimeStats {
        self.stats
    }

    pub(crate) fn buffer_capacity(&self, slot: MetalBufferSlot) -> usize {
        self.buffers
            .get(&slot)
            .map(|buffer| buffer.byte_capacity)
            .unwrap_or(0)
    }

    pub(crate) fn reserve_core_buffers(
        &mut self,
        gaussian_capacity: usize,
    ) -> candle_core::Result<()> {
        self.ensure_buffer(
            MetalBufferSlot::CameraUniforms,
            size_of::<MetalCameraUniform>(),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::TileMetadata,
            self.tile_windows
                .len()
                .saturating_mul(size_of::<MetalTileRecord>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::TileIndices,
            gaussian_capacity
                .saturating_mul(4)
                .saturating_mul(size_of::<u32>()),
        )?;
        Ok(())
    }

    pub(crate) fn reserve_tile_index_capacity(
        &mut self,
        tile_refs: usize,
    ) -> candle_core::Result<()> {
        self.ensure_buffer(
            MetalBufferSlot::TileIndices,
            tile_refs.saturating_mul(size_of::<u32>()),
        )
    }

    pub(crate) fn stage_camera(&mut self, camera: &DiffCamera) -> candle_core::Result<()> {
        let camera_uniform = MetalCameraUniform {
            fx: camera.fx,
            fy: camera.fy,
            cx: camera.cx,
            cy: camera.cy,
            width: camera.width as u32,
            height: camera.height as u32,
            tile_size: METAL_TILE_SIZE as u32,
            _padding: 0,
        };
        self.write_struct(MetalBufferSlot::CameraUniforms, &camera_uniform)
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
        gaussians: &'a TrainableGaussians,
    ) -> candle_core::Result<MetalGaussianBindings<'a>> {
        Ok(MetalGaussianBindings {
            positions: self.bind_tensor(gaussians.positions())?,
            scales: self.bind_tensor(gaussians.scales.as_tensor())?,
            rotations: self.bind_tensor(gaussians.rotations.as_tensor())?,
            opacities: self.bind_tensor(gaussians.opacities.as_tensor())?,
            colors: self.bind_tensor(gaussians.colors())?,
        })
    }

    pub(crate) fn dispatch_fill_u32(
        &mut self,
        slot: MetalBufferSlot,
        value: u32,
        len: usize,
    ) -> candle_core::Result<()> {
        if len == 0 {
            return Ok(());
        }
        self.ensure_buffer(slot, len.saturating_mul(size_of::<u32>()))?;
        let pipeline = self.ensure_pipeline(MetalKernel::FillU32)?.clone();
        let Some(buffer) = self
            .buffers
            .get(&slot)
            .and_then(|buffer| buffer.backing.as_ref().cloned())
        else {
            candle_core::bail!("fill_u32 requires a Metal device");
        };
        let metal = self.device.as_metal_device()?;
        let encoder = metal.command_encoder()?;
        encoder.set_label(MetalKernel::FillU32.function_name());
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(buffer.as_ref()), 0);
        encoder.set_bytes(1, &value);
        let count = len as u32;
        encoder.set_bytes(2, &count);
        let threads_per_group = pipeline.max_total_threads_per_threadgroup().min(len).max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: len,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        self.device.synchronize()?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn read_u32_buffer(
        &self,
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
        self.device.synchronize()?;
        let values = unsafe { std::slice::from_raw_parts(buffer.contents() as *const u32, len) };
        Ok(values.to_vec())
    }

    fn ensure_pipeline(&mut self, kernel: MetalKernel) -> candle_core::Result<&ComputePipeline> {
        if !self.pipelines.contains_key(&kernel) {
            let metal = self.device.as_metal_device()?;
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
            self.stats.pipeline_compilations += 1;
        }
        Ok(self
            .pipelines
            .get(&kernel)
            .expect("pipeline inserted before lookup"))
    }

    fn ensure_buffer(
        &mut self,
        slot: MetalBufferSlot,
        byte_capacity: usize,
    ) -> candle_core::Result<()> {
        let required = byte_capacity.max(1);
        if let Some(existing) = self.buffers.get(&slot) {
            if existing.byte_capacity >= required {
                self.stats.buffer_reuses += 1;
                return Ok(());
            }
        }

        let backing = if self.device.is_metal() {
            let metal = self.device.as_metal_device()?;
            Some(metal.new_buffer(required, DType::U8, slot.label())?)
        } else {
            None
        };
        self.buffers
            .insert(slot, PersistentMetalBuffer::new(required, backing));
        self.stats.buffer_allocations += 1;
        Ok(())
    }

    fn write_struct<T: Copy>(
        &mut self,
        slot: MetalBufferSlot,
        value: &T,
    ) -> candle_core::Result<()> {
        let data =
            unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) };
        self.write_bytes(slot, data)
    }

    fn write_bytes(&mut self, slot: MetalBufferSlot, data: &[u8]) -> candle_core::Result<()> {
        self.ensure_buffer(slot, data.len())?;
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

fn build_chunk_pixel_window(
    device: &Device,
    render_width: usize,
    rect: ScreenRect,
) -> candle_core::Result<ChunkPixelWindow> {
    let width = rect.max_x - rect.min_x + 1;
    let height = rect.max_y - rect.min_y + 1;
    let mut xs = Vec::with_capacity(width * height);
    let mut ys = Vec::with_capacity(width * height);
    let mut indices = Vec::with_capacity(width * height);

    for y in rect.min_y..=rect.max_y {
        for x in rect.min_x..=rect.max_x {
            xs.push(x as f32 + 0.5);
            ys.push(y as f32 + 0.5);
            indices.push((y * render_width + x) as u32);
        }
    }

    let pixel_count = indices.len();
    Ok(ChunkPixelWindow {
        pixel_x: Tensor::from_slice(&xs, (1, pixel_count), device)?,
        pixel_y: Tensor::from_slice(&ys, (1, pixel_count), device)?,
        indices: Tensor::from_slice(&indices, pixel_count, device)?,
        pixel_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_windows_cover_entire_surface() {
        let runtime = MetalRuntime::new(32, 17, Device::Cpu).unwrap();
        assert_eq!(runtime.tile_grid(), (2, 2));
        assert_eq!(runtime.stats().tile_windows, 4);
        assert_eq!(runtime.tile_window(0).unwrap().pixel_count, 16 * 16);
        assert_eq!(runtime.tile_window(3).unwrap().pixel_count, 16);
    }

    #[test]
    fn core_buffer_reservations_reuse_capacity() {
        let mut runtime = MetalRuntime::new(32, 16, Device::Cpu).unwrap();
        runtime.reserve_core_buffers(64).unwrap();
        let initial = runtime.stats();
        assert_eq!(initial.buffer_allocations, 3);

        runtime.reserve_core_buffers(32).unwrap();
        let reused = runtime.stats();
        assert_eq!(reused.buffer_allocations, initial.buffer_allocations);
        assert!(reused.buffer_reuses >= 3);
    }

    #[test]
    fn fill_kernel_writes_shared_buffer() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        };

        let mut runtime = MetalRuntime::new(16, 16, device).unwrap();
        runtime
            .dispatch_fill_u32(MetalBufferSlot::TileIndices, 7, 8)
            .unwrap();
        let values = runtime
            .read_u32_buffer(MetalBufferSlot::TileIndices, 8)
            .unwrap();
        assert_eq!(values, vec![7; 8]);
    }
}
