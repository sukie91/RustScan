//! Forward rendering pipeline.

use burn::prelude::*;
use burn::tensor::{Int, Tensor, TensorData};
use bytemuck::{Pod, Zeroable};
use naga_oil::compose::{ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderType};
use wgpu::naga;

use crate::core::GaussianCamera;
use crate::training::engine::DeviceSplats;
use crate::training::gpu_primitives::{prefix_sum::PrefixSumBackend, radix_sort::RadixSortBackend};

pub mod project_visible;
pub mod projection;
pub mod rasterize;
pub mod sorting;
pub mod tile_mapping;

pub(crate) use project_visible::project_visible;
pub(crate) use projection::project_forward;
pub(crate) use rasterize::rasterize;
pub(crate) use sorting::sort_by_depth;
pub(crate) use tile_mapping::{get_tile_offsets, tile_mapping};

pub(crate) const TILE_WIDTH: u32 = 16;
pub(crate) const TILE_SIZE: u32 = TILE_WIDTH * TILE_WIDTH;
pub(crate) const HELPERS_SRC: &str = include_str!("../shaders/helpers.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ProjectUniforms {
    pub viewmat: [[f32; 4]; 4],
    pub focal: [f32; 2],
    pub img_size: [u32; 2],
    pub tile_bounds: [u32; 2],
    pub pixel_center: [f32; 2],
    pub camera_position: [f32; 4],
    pub sh_degree: u32,
    pub total_splats: u32,
    pub num_visible: u32,
    pub pad_a: u32,
}

impl ProjectUniforms {
    pub(crate) fn new(
        camera: &GaussianCamera,
        img_size: (u32, u32),
        tile_bounds: (u32, u32),
        sh_degree: u32,
        total_splats: u32,
        num_visible: u32,
    ) -> Self {
        let camera_position = camera.position();
        Self {
            viewmat: camera.view_matrix().to_cols_array_2d(),
            focal: [camera.intrinsics.fx, camera.intrinsics.fy],
            img_size: [img_size.0, img_size.1],
            tile_bounds: [tile_bounds.0, tile_bounds.1],
            pixel_center: [camera.intrinsics.cx, camera.intrinsics.cy],
            camera_position: [camera_position.x, camera_position.y, camera_position.z, 0.0],
            sh_degree,
            total_splats,
            num_visible,
            pad_a: 0,
        }
    }
}

pub(crate) fn calc_tile_bounds(img_size: (u32, u32)) -> (u32, u32) {
    (
        img_size.0.div_ceil(TILE_WIDTH),
        img_size.1.div_ceil(TILE_WIDTH),
    )
}

pub(crate) fn compose_shader(file_path: &str, source: &str) -> String {
    let mut composer = Composer::default();
    composer.capabilities = naga::valid::Capabilities::all();
    composer
        .add_composable_module(ComposableModuleDescriptor {
            source: HELPERS_SRC,
            file_path: "helpers.wgsl",
            ..Default::default()
        })
        .expect("helpers shader module");

    let module = composer
        .make_naga_module(NagaModuleDescriptor {
            source,
            file_path,
            shader_type: ShaderType::Wgsl,
            ..Default::default()
        })
        .expect("compose shader");

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("validate shader");

    naga::back::wgsl::write_string(
        &module,
        &info,
        naga::back::wgsl::WriterFlags::EXPLICIT_TYPES,
    )
    .expect("serialize shader")
}

fn usize_from_int_data(data: &TensorData, index: usize, label: &str) -> usize {
    if let Ok(values) = data.as_slice::<i32>() {
        values[index].max(0) as usize
    } else if let Ok(values) = data.as_slice::<u32>() {
        values[index] as usize
    } else {
        panic!("{label}: expected i32/u32 tensor");
    }
}

async fn read_projection_counts_async<B: Backend>(
    num_visible_buf: Tensor<B, 1, Int>,
    num_intersections_buf: Tensor<B, 1, Int>,
) -> (usize, usize) {
    let counts = Tensor::<B, 1, Int>::cat(
        vec![
            num_visible_buf.reshape([1]),
            num_intersections_buf.reshape([1]),
        ],
        0,
    );
    let counts_data = counts
        .into_data_async()
        .await
        .expect("projection count readback");

    (
        usize_from_int_data(&counts_data, 0, "num_visible"),
        usize_from_int_data(&counts_data, 1, "num_intersections"),
    )
}

#[allow(dead_code)]
pub(crate) struct RenderOutput<B: Backend> {
    pub out_img: Tensor<B, 3>,
    pub visible: Tensor<B, 1>,
    pub projected_splats: Tensor<B, 2>,
    pub global_from_compact_gid: Tensor<B, 1, Int>,
    pub compact_gid_from_isect: Tensor<B, 1, Int>,
    pub tile_offsets: Tensor<B, 1, Int>,
    pub num_visible: usize,
    pub num_intersections: usize,
}

pub(crate) async fn render_forward<B>(
    splats: &DeviceSplats<B>,
    camera: &GaussianCamera,
    img_size: (u32, u32),
    background: [f32; 3],
    device: &B::Device,
) -> RenderOutput<B>
where
    B: projection::ProjectionBackend
        + sorting::SortingBackend
        + project_visible::ProjectVisibleBackend
        + tile_mapping::TileMappingBackend
        + rasterize::RasterizeBackend
        + PrefixSumBackend
        + RadixSortBackend,
{
    let proj_out = project_forward(splats, camera, img_size, device);
    let projection::ProjectForwardOutput {
        global_from_presort_gid,
        depths,
        intersect_counts,
        num_visible_buf,
        num_intersections_buf,
    } = proj_out;
    let (num_visible, num_intersections) =
        read_projection_counts_async(num_visible_buf, num_intersections_buf).await;
    let tile_bounds = calc_tile_bounds(img_size);
    let num_tiles = tile_bounds.0 * tile_bounds.1;

    if num_visible == 0 {
        let empty_indices = Tensor::<B, 1, Int>::zeros([0], device);
        let projected_splats = Tensor::<B, 2>::zeros([0, 9], device);
        let tile_offsets = Tensor::<B, 1, Int>::zeros([2 * num_tiles as usize], device);
        let raster_out = rasterize(
            &empty_indices,
            &tile_offsets,
            &projected_splats,
            &empty_indices,
            splats.num_splats(),
            img_size,
            tile_bounds,
            background,
            device,
        );

        return RenderOutput {
            out_img: raster_out.out_img,
            visible: raster_out.visible,
            projected_splats,
            global_from_compact_gid: empty_indices.clone(),
            compact_gid_from_isect: empty_indices,
            tile_offsets,
            num_visible,
            num_intersections,
        };
    }

    let global_from_compact_gid =
        sort_by_depth(depths, global_from_presort_gid, num_visible, device);

    let compact_intersect_counts = intersect_counts.gather(0, global_from_compact_gid.clone());

    let projected_splats = project_visible(
        splats,
        &global_from_compact_gid,
        num_visible,
        camera,
        img_size,
        device,
    );

    if num_intersections == 0 {
        let compact_gid_from_isect = Tensor::<B, 1, Int>::zeros([0], device);
        let tile_offsets = Tensor::<B, 1, Int>::zeros([2 * num_tiles as usize], device);
        let raster_out = rasterize(
            &compact_gid_from_isect,
            &tile_offsets,
            &projected_splats,
            &global_from_compact_gid,
            splats.num_splats(),
            img_size,
            tile_bounds,
            background,
            device,
        );

        return RenderOutput {
            out_img: raster_out.out_img,
            visible: raster_out.visible,
            projected_splats,
            global_from_compact_gid,
            compact_gid_from_isect,
            tile_offsets,
            num_visible,
            num_intersections,
        };
    }

    let tile_out = tile_mapping(
        &projected_splats,
        compact_intersect_counts,
        num_intersections,
        num_tiles,
        tile_bounds,
        device,
    );

    let tile_offsets = get_tile_offsets(
        tile_out.tile_id_from_isect.clone(),
        num_intersections,
        tile_bounds,
        device,
    );

    let raster_out = rasterize(
        &tile_out.compact_gid_from_isect,
        &tile_offsets,
        &projected_splats,
        &global_from_compact_gid,
        splats.num_splats(),
        img_size,
        tile_bounds,
        background,
        device,
    );

    RenderOutput {
        out_img: raster_out.out_img,
        visible: raster_out.visible,
        projected_splats,
        global_from_compact_gid,
        compact_gid_from_isect: tile_out.compact_gid_from_isect,
        tile_offsets,
        num_visible,
        num_intersections,
    }
}

#[cfg(test)]
mod tests {
    use super::render_forward;
    use crate::core::GaussianCamera;
    use crate::core::HostSplats;
    use crate::sh::rgb_to_sh0_value;
    use crate::training::engine::{host_splats_to_device, GsBackendBase};
    use crate::{Intrinsics, SE3};

    fn test_camera() -> GaussianCamera {
        GaussianCamera::new(Intrinsics::from_focal(500.0, 64, 64), SE3::identity())
    }

    fn test_splats() -> HostSplats {
        HostSplats::from_raw_parts(
            vec![0.0, 0.0, 2.0],
            vec![0.2f32.ln(), 0.2f32.ln(), 0.2f32.ln()],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0],
            [1.0, 0.5, 0.25].map(rgb_to_sh0_value).into(),
            0,
        )
        .expect("valid host splats")
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_render_forward_single_gaussian() {
        let device = <GsBackendBase as burn::tensor::backend::Backend>::Device::default();
        let splats = host_splats_to_device::<GsBackendBase>(&test_splats(), &device);
        let output = render_forward::<GsBackendBase>(
            &splats,
            &test_camera(),
            (64, 64),
            [0.0, 0.0, 0.0],
            &device,
        )
        .await;

        assert_eq!(output.out_img.dims(), [64, 64, 4]);
    }
}
