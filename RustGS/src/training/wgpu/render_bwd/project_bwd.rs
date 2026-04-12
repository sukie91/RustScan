use burn::module::Param;
use burn::prelude::*;
use burn::tensor::{Int, Tensor, TensorMetadata, TensorPrimitive};
use burn_cubecl::cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};
use burn_cubecl::{BoolElement, CubeBackend, FloatElement, IntElement, kernel::into_contiguous};
use burn_wgpu::{CubeDim, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};

use crate::core::GaussianCamera;
use crate::training::wgpu::render::{ProjectUniforms, calc_tile_bounds, compose_shader};
use crate::training::wgpu::splats::DeviceSplats;

const WORKGROUP_SIZE: u32 = 256;
const SHADER_SRC: &str = include_str!("../shaders/project_backwards.wgsl");

struct ProjectBwdRaw;

impl ProjectBwdRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(compose_shader("project_backwards.wgsl", SHADER_SRC))
    }
}

#[derive(Debug)]
struct ProjectBwdKernel;

impl KernelSource for ProjectBwdKernel {
    fn source(&self) -> SourceTemplate {
        ProjectBwdRaw.source()
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

pub(crate) struct ProjectBwdOutput<B: Backend> {
    pub v_transforms: Tensor<B, 2>,
    pub v_sh_coeffs: Tensor<B, 3>,
    pub v_raw_opacities: Tensor<B, 1>,
}

pub(crate) trait ProjectBwdBackend: Backend {
    fn project_bwd_primitive(
        transforms: Self::FloatTensorPrimitive,
        sh_coeffs: Self::FloatTensorPrimitive,
        raw_opacities: Self::FloatTensorPrimitive,
        global_from_compact_gid: Self::IntTensorPrimitive,
        v_splats: Self::FloatTensorPrimitive,
        uniforms: ProjectUniforms,
    ) -> (
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
    );
}

impl<F, I, BT> ProjectBwdBackend for CubeBackend<WgpuRuntime, F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn project_bwd_primitive(
        transforms: Self::FloatTensorPrimitive,
        sh_coeffs: Self::FloatTensorPrimitive,
        raw_opacities: Self::FloatTensorPrimitive,
        global_from_compact_gid: Self::IntTensorPrimitive,
        v_splats: Self::FloatTensorPrimitive,
        uniforms: ProjectUniforms,
    ) -> (
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
    ) {
        let transforms = into_contiguous(transforms);
        let sh_coeffs = into_contiguous(sh_coeffs);
        let raw_opacities = into_contiguous(raw_opacities);
        let global_from_compact_gid = into_contiguous(global_from_compact_gid);
        let v_splats = into_contiguous(v_splats);
        let device = transforms.device.clone();
        let client = transforms.client.clone();

        let num_splats = transforms.shape()[0];
        let num_coeffs = sh_coeffs.shape()[1];

        let v_transforms = Tensor::<Self, 2>::zeros([num_splats, 10], &device);
        let v_sh_coeffs = Tensor::<Self, 3>::zeros([num_splats, num_coeffs, 3], &device);
        let v_raw_opacities = Tensor::<Self, 1>::zeros([num_splats], &device);

        if uniforms.num_visible > 0 {
            let uniforms_handle = client.create_from_slice(bytemuck::bytes_of(&uniforms));
            client.launch(
                Box::new(SourceKernel::new(
                    ProjectBwdKernel,
                    CubeDim::new_1d(WORKGROUP_SIZE),
                )),
                CubeCount::Static(uniforms.num_visible.div_ceil(WORKGROUP_SIZE), 1, 1),
                KernelArguments::new().with_buffers(vec![
                    transforms.handle.binding(),
                    sh_coeffs.handle.binding(),
                    raw_opacities.handle.binding(),
                    global_from_compact_gid.handle.binding(),
                    v_splats.handle.binding(),
                    v_transforms.clone().into_primitive().tensor().handle.binding(),
                    v_sh_coeffs.clone().into_primitive().tensor().handle.binding(),
                    v_raw_opacities.clone().into_primitive().tensor().handle.binding(),
                    uniforms_handle.binding(),
                ]),
            );
        }

        (
            v_transforms.into_primitive().tensor(),
            v_sh_coeffs.into_primitive().tensor(),
            v_raw_opacities.into_primitive().tensor(),
        )
    }
}

pub(crate) fn project_bwd<B: ProjectBwdBackend>(
    splats: &DeviceSplats<B>,
    global_from_compact_gid: Tensor<B, 1, Int>,
    v_splats: Tensor<B, 2>,
    camera: &GaussianCamera,
    img_size: (u32, u32),
    num_visible: usize,
    _device: &B::Device,
) -> ProjectBwdOutput<B> {
    let uniforms = ProjectUniforms::new(
        camera,
        img_size,
        calc_tile_bounds(img_size),
        splats.sh_degree,
        splats.num_splats() as u32,
        num_visible as u32,
    );

    let (v_transforms, v_sh_coeffs, v_raw_opacities) = B::project_bwd_primitive(
        splats.transforms.val().into_primitive().tensor(),
        splats.sh_coeffs.val().into_primitive().tensor(),
        splats.raw_opacities.val().into_primitive().tensor(),
        global_from_compact_gid.into_primitive(),
        v_splats.into_primitive().tensor(),
        uniforms,
    );

    ProjectBwdOutput {
        v_transforms: Tensor::from_primitive(TensorPrimitive::Float(v_transforms)),
        v_sh_coeffs: Tensor::from_primitive(TensorPrimitive::Float(v_sh_coeffs)),
        v_raw_opacities: Tensor::from_primitive(TensorPrimitive::Float(v_raw_opacities)),
    }
}

pub(crate) fn from_inner_splats<B: Backend>(
    transforms: Tensor<B, 2>,
    sh_coeffs: Tensor<B, 3>,
    raw_opacities: Tensor<B, 1>,
    sh_degree: u32,
) -> DeviceSplats<B> {
    DeviceSplats {
        transforms: Param::from_tensor(transforms),
        sh_coeffs: Param::from_tensor(sh_coeffs),
        raw_opacities: Param::from_tensor(raw_opacities),
        sh_degree,
    }
}
