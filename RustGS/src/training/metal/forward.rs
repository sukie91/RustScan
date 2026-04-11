use std::cmp::Ordering;
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor};
use glam::{Mat3, Quat, Vec3};

use crate::diff::diff_splat::{DiffCamera, Splats};

use super::runtime::{
    ChunkPixelWindow, MetalBufferSlot, MetalProjectionRecord, MetalRuntime, MetalTileBins,
    NativeForwardProfile,
};
use super::splats::row_slice;

pub(crate) const SH_C1: f32 = 0.488_602_52;
const SH_C2: [f32; 5] = [
    1.092_548_5,
    -1.092_548_5,
    0.315_391_57,
    -1.092_548_5,
    0.546_274_24,
];
const SH_C3: [f32; 7] = [
    -0.590_043_6,
    2.890_611_4,
    -0.457_045_8,
    0.373_176_34,
    -0.457_045_8,
    1.445_305_7,
    -0.590_043_6,
];
const SH_C4: [f32; 9] = [
    2.503_343,
    -1.770_130_8,
    0.946_174_7,
    -0.669_046_5,
    0.105_785_55,
    -0.669_046_5,
    0.473_087_34,
    -1.770_130_8,
    0.625_835_7,
];

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct TileBinningStats {
    pub(super) active_tiles: usize,
    pub(super) tile_gaussian_refs: usize,
    pub(super) max_gaussians_per_tile: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct NativeParityProfile {
    pub(super) setup: Duration,
    pub(super) staging: Duration,
    pub(super) kernel: Duration,
    pub(super) total: Duration,
    pub(super) color_max_abs: f32,
    pub(super) depth_max_abs: f32,
    pub(super) alpha_max_abs: f32,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct MetalRenderProfile {
    pub(super) projection: Duration,
    pub(super) sorting: Duration,
    pub(super) rasterization: Duration,
    pub(super) native_forward: Option<NativeParityProfile>,
    pub(super) visible_gaussians: usize,
    pub(super) total_gaussians: usize,
    pub(super) active_tiles: usize,
    pub(super) tile_gaussian_refs: usize,
    pub(super) max_gaussians_per_tile: usize,
}

pub(crate) struct ProjectedGaussians {
    pub(super) source_indices: Tensor,
    pub(super) u: Tensor,
    pub(super) v: Tensor,
    pub(super) sigma_x: Tensor,
    pub(super) sigma_y: Tensor,
    pub(super) raw_sigma_x: Tensor,
    pub(super) raw_sigma_y: Tensor,
    pub(super) depth: Tensor,
    pub(super) opacity: Tensor,
    pub(super) opacity_logits: Tensor,
    pub(super) scale3d: Tensor,
    pub(super) colors: Tensor,
    pub(super) min_x: Tensor,
    pub(super) max_x: Tensor,
    pub(super) min_y: Tensor,
    pub(super) max_y: Tensor,
    pub(super) visible_source_indices: Vec<u32>,
    pub(super) visible_count: usize,
    pub(super) tile_bins: ProjectedTileBins,
    pub(super) staging_source: ProjectionStagingSource,
}

impl ProjectedGaussians {
    pub(super) fn empty(device: &Device) -> candle_core::Result<Self> {
        Ok(Self {
            source_indices: Tensor::zeros((0,), DType::U32, device)?,
            u: Tensor::zeros((0,), DType::F32, device)?,
            v: Tensor::zeros((0,), DType::F32, device)?,
            sigma_x: Tensor::zeros((0,), DType::F32, device)?,
            sigma_y: Tensor::zeros((0,), DType::F32, device)?,
            raw_sigma_x: Tensor::zeros((0,), DType::F32, device)?,
            raw_sigma_y: Tensor::zeros((0,), DType::F32, device)?,
            depth: Tensor::zeros((0,), DType::F32, device)?,
            opacity: Tensor::zeros((0,), DType::F32, device)?,
            opacity_logits: Tensor::zeros((0,), DType::F32, device)?,
            scale3d: Tensor::zeros((0, 3), DType::F32, device)?,
            colors: Tensor::zeros((0, 3), DType::F32, device)?,
            min_x: Tensor::zeros((0,), DType::F32, device)?,
            max_x: Tensor::zeros((0,), DType::F32, device)?,
            min_y: Tensor::zeros((0,), DType::F32, device)?,
            max_y: Tensor::zeros((0,), DType::F32, device)?,
            visible_source_indices: Vec::new(),
            visible_count: 0,
            tile_bins: ProjectedTileBins::default(),
            staging_source: ProjectionStagingSource::TensorReadback,
        })
    }

    pub(super) fn visible_source_indices(&self) -> &[u32] {
        &self.visible_source_indices
    }
}

pub(crate) struct RenderedFrame {
    pub(crate) color: Tensor,
    pub(crate) depth: Tensor,
    pub(crate) alpha: Tensor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ProjectedTileRecord {
    start: usize,
    count: usize,
}

impl ProjectedTileRecord {
    pub(super) fn start(&self) -> usize {
        self.start
    }

    pub(super) fn count(&self) -> usize {
        self.count
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ProjectedTileBins {
    runtime: MetalTileBins,
}

impl ProjectedTileBins {
    pub(super) fn from_runtime(runtime: MetalTileBins) -> Self {
        Self { runtime }
    }

    pub(super) fn active_tiles(&self) -> &[usize] {
        self.runtime.active_tiles()
    }

    pub(super) fn active_tile_count(&self) -> usize {
        self.runtime.active_tile_count()
    }

    pub(super) fn total_assignments(&self) -> usize {
        self.runtime.total_assignments()
    }

    pub(super) fn max_gaussians_per_tile(&self) -> usize {
        self.runtime.max_gaussians_per_tile()
    }

    pub(super) fn packed_indices(&self) -> &[u32] {
        self.runtime.packed_indices()
    }

    pub(super) fn record(&self, tile_idx: usize) -> Option<ProjectedTileRecord> {
        self.runtime
            .record(tile_idx)
            .map(|record| ProjectedTileRecord {
                start: record.start(),
                count: record.count(),
            })
    }

    pub(super) fn as_runtime(&self) -> &MetalTileBins {
        &self.runtime
    }

    #[cfg(test)]
    pub(super) fn indices_for_tile(&self, tile_idx: usize) -> &[u32] {
        self.runtime.indices_for_tile(tile_idx)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalForwardSettings {
    pub(crate) pixel_count: usize,
    pub(crate) render_width: usize,
    pub(crate) render_height: usize,
    pub(crate) chunk_size: usize,
    pub(crate) use_native_forward: bool,
    pub(crate) litegs_mode: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct MetalForwardInputs<'a> {
    pub(crate) gaussians: &'a Splats,
    pub(crate) positions: &'a Tensor,
    pub(crate) colors: &'a Tensor,
    pub(crate) camera: &'a DiffCamera,
    pub(crate) should_profile: bool,
    pub(crate) collect_visible_indices: bool,
    pub(crate) cluster_visible_mask: Option<&'a [bool]>,
}

pub(super) struct MetalForwardContext<'a> {
    pub(super) runtime: &'a mut MetalRuntime,
    pub(super) device: &'a Device,
    pub(super) settings: MetalForwardSettings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProjectionStagingSource {
    TensorReadback,
    RuntimeBufferRead,
}

#[derive(Debug, Clone)]
pub(super) struct CpuProjectedGaussian {
    pub(super) source_idx: u32,
    pub(super) u: f32,
    pub(super) v: f32,
    pub(super) sigma_x: f32,
    pub(super) sigma_y: f32,
    pub(super) raw_sigma_x: f32,
    pub(super) raw_sigma_y: f32,
    pub(super) depth: f32,
    pub(super) opacity: f32,
    pub(super) opacity_logit: f32,
    pub(super) scale3d: [f32; 3],
    pub(super) color: [f32; 3],
    pub(super) min_x: f32,
    pub(super) max_x: f32,
    pub(super) min_y: f32,
    pub(super) max_y: f32,
}

pub(super) fn cpu_projected_from_record(record: MetalProjectionRecord) -> CpuProjectedGaussian {
    CpuProjectedGaussian {
        source_idx: record.source_idx,
        u: record.u,
        v: record.v,
        sigma_x: record.sigma_x,
        sigma_y: record.sigma_y,
        raw_sigma_x: record.raw_sigma_x,
        raw_sigma_y: record.raw_sigma_y,
        depth: record.depth,
        opacity: record.opacity,
        opacity_logit: record.opacity_logit,
        scale3d: [record.scale_x, record.scale_y, record.scale_z],
        color: [record.color_r, record.color_g, record.color_b],
        min_x: record.min_x,
        max_x: record.max_x,
        min_y: record.min_y,
        max_y: record.max_y,
    }
}

pub(super) fn projected_rows_to_cpu(
    projected: &ProjectedGaussians,
) -> candle_core::Result<Vec<CpuProjectedGaussian>> {
    let source_indices = projected.source_indices.to_vec1::<u32>()?;
    let u = projected.u.to_vec1::<f32>()?;
    let v = projected.v.to_vec1::<f32>()?;
    let sigma_x = projected.sigma_x.to_vec1::<f32>()?;
    let sigma_y = projected.sigma_y.to_vec1::<f32>()?;
    let raw_sigma_x = projected.raw_sigma_x.to_vec1::<f32>()?;
    let raw_sigma_y = projected.raw_sigma_y.to_vec1::<f32>()?;
    let depth = projected.depth.to_vec1::<f32>()?;
    let opacity = projected.opacity.to_vec1::<f32>()?;
    let opacity_logits = projected.opacity_logits.to_vec1::<f32>()?;
    let scale3d = projected.scale3d.to_vec2::<f32>()?;
    let colors = projected.colors.to_vec2::<f32>()?;
    let min_x = projected.min_x.to_vec1::<f32>()?;
    let max_x = projected.max_x.to_vec1::<f32>()?;
    let min_y = projected.min_y.to_vec1::<f32>()?;
    let max_y = projected.max_y.to_vec1::<f32>()?;
    let mut projected_cpu = Vec::with_capacity(source_indices.len());

    for idx in 0..source_indices.len() {
        projected_cpu.push(CpuProjectedGaussian {
            source_idx: source_indices[idx],
            u: u.get(idx).copied().unwrap_or(0.0),
            v: v.get(idx).copied().unwrap_or(0.0),
            sigma_x: sigma_x.get(idx).copied().unwrap_or(0.0),
            sigma_y: sigma_y.get(idx).copied().unwrap_or(0.0),
            raw_sigma_x: raw_sigma_x.get(idx).copied().unwrap_or(0.0),
            raw_sigma_y: raw_sigma_y.get(idx).copied().unwrap_or(0.0),
            depth: depth.get(idx).copied().unwrap_or(0.0),
            opacity: opacity.get(idx).copied().unwrap_or(0.0),
            opacity_logit: opacity_logits.get(idx).copied().unwrap_or(0.0),
            scale3d: row_to_vec3(scale3d.get(idx).map(Vec::as_slice).unwrap_or(&[])),
            color: row_to_vec3(colors.get(idx).map(Vec::as_slice).unwrap_or(&[])),
            min_x: min_x.get(idx).copied().unwrap_or(0.0),
            max_x: max_x.get(idx).copied().unwrap_or(0.0),
            min_y: min_y.get(idx).copied().unwrap_or(0.0),
            max_y: max_y.get(idx).copied().unwrap_or(0.0),
        });
    }

    Ok(projected_cpu)
}

pub(super) fn row_to_vec3(row: &[f32]) -> [f32; 3] {
    [
        row.first().copied().unwrap_or(0.0),
        row.get(1).copied().unwrap_or(0.0),
        row.get(2).copied().unwrap_or(0.0),
    ]
}

pub(super) fn row_to_quaternion(row: &[f32]) -> [f32; 4] {
    [
        row.first().copied().unwrap_or(1.0),
        row.get(1).copied().unwrap_or(0.0),
        row.get(2).copied().unwrap_or(0.0),
        row.get(3).copied().unwrap_or(0.0),
    ]
}

fn mat3_from_row_major(rotation: &[[f32; 3]; 3]) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(rotation[0][0], rotation[1][0], rotation[2][0]),
        Vec3::new(rotation[0][1], rotation[1][1], rotation[2][1]),
        Vec3::new(rotation[0][2], rotation[1][2], rotation[2][2]),
    )
}

fn quat_from_wxyz(rotation: [f32; 4]) -> Quat {
    let length_sq = rotation.iter().map(|value| value * value).sum::<f32>();
    if !length_sq.is_finite() || length_sq <= 1e-12 {
        return Quat::IDENTITY;
    }
    Quat::from_xyzw(rotation[1], rotation[2], rotation[3], rotation[0]).normalize()
}

pub(super) fn projected_axis_covariance_terms(
    x: f32,
    y: f32,
    z: f32,
    scale: [f32; 3],
    rotation: [f32; 4],
    camera_rotation: &[[f32; 3]; 3],
    fx: f32,
    fy: f32,
) -> (f32, f32, [f32; 3], [f32; 3]) {
    let inv_z = 1.0 / z.max(1e-4);
    let object_rotation = Mat3::from_quat(quat_from_wxyz(rotation));
    let camera_rotation = mat3_from_row_major(camera_rotation);
    let total_rotation = camera_rotation * object_rotation;

    let projection_row_x = Vec3::new(fx * inv_z, 0.0, -fx * x * inv_z * inv_z);
    let projection_row_y = Vec3::new(0.0, fy * inv_z, -fy * y * inv_z * inv_z);

    let proj_axis_x = [
        projection_row_x.dot(total_rotation.col(0)),
        projection_row_x.dot(total_rotation.col(1)),
        projection_row_x.dot(total_rotation.col(2)),
    ];
    let proj_axis_y = [
        projection_row_y.dot(total_rotation.col(0)),
        projection_row_y.dot(total_rotation.col(1)),
        projection_row_y.dot(total_rotation.col(2)),
    ];

    let covariance_x = scale[0] * scale[0] * proj_axis_x[0] * proj_axis_x[0]
        + scale[1] * scale[1] * proj_axis_x[1] * proj_axis_x[1]
        + scale[2] * scale[2] * proj_axis_x[2] * proj_axis_x[2];
    let covariance_y = scale[0] * scale[0] * proj_axis_y[0] * proj_axis_y[0]
        + scale[1] * scale[1] * proj_axis_y[1] * proj_axis_y[1]
        + scale[2] * scale[2] * proj_axis_y[2] * proj_axis_y[2];

    const LOWPASS_FILTER: f32 = 0.3;
    const LOWPASS_VAR: f32 = LOWPASS_FILTER * LOWPASS_FILTER;

    (
        (covariance_x + LOWPASS_VAR).max(1e-6).sqrt(),
        (covariance_y + LOWPASS_VAR).max(1e-6).sqrt(),
        proj_axis_x,
        proj_axis_y,
    )
}

pub(super) fn projected_axis_aligned_sigmas(
    x: f32,
    y: f32,
    z: f32,
    scale: [f32; 3],
    rotation: [f32; 4],
    camera_rotation: &[[f32; 3]; 3],
    fx: f32,
    fy: f32,
) -> (f32, f32) {
    let (sigma_x, sigma_y, _, _) =
        projected_axis_covariance_terms(x, y, z, scale, rotation, camera_rotation, fx, fy);
    (sigma_x, sigma_y)
}

pub(super) fn finite_difference_sigma_wrt_rotation_component(
    x: f32,
    y: f32,
    z: f32,
    scale: [f32; 3],
    raw_rotation: [f32; 4],
    component: usize,
    camera: &DiffCamera,
) -> (f32, f32) {
    const ROTATION_FD_EPS: f32 = 1e-3;
    if component >= 4 {
        return (0.0, 0.0);
    }

    let mut plus = raw_rotation;
    let mut minus = raw_rotation;
    plus[component] += ROTATION_FD_EPS;
    minus[component] -= ROTATION_FD_EPS;

    let (plus_sigma_x, plus_sigma_y) =
        projected_axis_aligned_sigmas(x, y, z, scale, plus, &camera.rotation, camera.fx, camera.fy);
    let (minus_sigma_x, minus_sigma_y) = projected_axis_aligned_sigmas(
        x,
        y,
        z,
        scale,
        minus,
        &camera.rotation,
        camera.fx,
        camera.fy,
    );
    (
        (plus_sigma_x.clamp(0.5, 256.0) - minus_sigma_x.clamp(0.5, 256.0))
            / (2.0 * ROTATION_FD_EPS),
        (plus_sigma_y.clamp(0.5, 256.0) - minus_sigma_y.clamp(0.5, 256.0))
            / (2.0 * ROTATION_FD_EPS),
    )
}

pub(super) fn scale_camera(
    src: &DiffCamera,
    width: usize,
    height: usize,
    device: &Device,
) -> candle_core::Result<DiffCamera> {
    let sx = width as f32 / src.width as f32;
    let sy = height as f32 / src.height as f32;
    DiffCamera::new(
        src.fx * sx,
        src.fy * sy,
        src.cx * sx,
        src.cy * sy,
        width,
        height,
        &src.rotation,
        &src.translation,
        device,
    )
}

fn camera_center_world(camera: &DiffCamera) -> [f32; 3] {
    [
        -(camera.rotation[0][0] * camera.translation[0]
            + camera.rotation[1][0] * camera.translation[1]
            + camera.rotation[2][0] * camera.translation[2]),
        -(camera.rotation[0][1] * camera.translation[0]
            + camera.rotation[1][1] * camera.translation[1]
            + camera.rotation[2][1] * camera.translation[2]),
        -(camera.rotation[0][2] * camera.translation[0]
            + camera.rotation[1][2] * camera.translation[1]
            + camera.rotation[2][2] * camera.translation[2]),
    ]
}

fn view_directions_for_camera(
    positions: &Tensor,
    camera: &DiffCamera,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let camera_center = Tensor::from_slice(&camera_center_world(camera), (1, 3), device)?;
    let dirs = positions.broadcast_sub(&camera_center)?;
    let norms = dirs
        .sqr()?
        .sum(1)?
        .sqrt()?
        .clamp(1e-6, f32::MAX)?
        .reshape((positions.dim(0)?, 1))?;
    dirs.broadcast_div(&norms)
}

pub(crate) fn render_colors_for_camera(
    gaussians: &Splats,
    positions: &Tensor,
    camera: &DiffCamera,
    device: &Device,
    active_sh_degree: usize,
) -> candle_core::Result<Tensor> {
    if !gaussians.uses_spherical_harmonics() {
        return Ok(gaussians.render_colors()?.detach());
    }

    let active_degree = active_sh_degree.min(gaussians.sh_degree());
    let row_count = gaussians.len();
    if row_count == 0 {
        return Tensor::zeros((0, 3), DType::F32, device);
    }

    let dirs = view_directions_for_camera(positions, camera, device)?;
    let x = dirs.narrow(1, 0, 1)?.squeeze(1)?;
    let y = dirs.narrow(1, 1, 1)?.squeeze(1)?;
    let z = dirs.narrow(1, 2, 1)?.squeeze(1)?;
    let xx = x.sqr()?;
    let yy = y.sqr()?;
    let zz = z.sqr()?;
    let xy = x.broadcast_mul(&y)?;
    let yz = y.broadcast_mul(&z)?;
    let xz = x.broadcast_mul(&z)?;

    let sh_0 = gaussians.sh_0().detach();
    let sh_rest = gaussians.sh_rest().detach();
    let mut color = sh_0.affine(crate::diff::diff_splat::SH_C0 as f64, 0.5)?;

    macro_rules! add_sh_term {
        ($coeff_idx:expr, $basis:expr) => {{
            let coeff = sh_rest.narrow(1, $coeff_idx, 1)?.squeeze(1)?;
            let basis = ($basis)?.reshape((row_count, 1))?;
            color = color.broadcast_add(&coeff.broadcast_mul(&basis)?)?;
        }};
    }

    if active_degree > 0 {
        add_sh_term!(0, y.affine((-SH_C1) as f64, 0.0));
        add_sh_term!(1, z.affine(SH_C1 as f64, 0.0));
        add_sh_term!(2, x.affine((-SH_C1) as f64, 0.0));
    }

    if active_degree > 1 {
        add_sh_term!(3, xy.affine(SH_C2[0] as f64, 0.0));
        add_sh_term!(4, yz.affine(SH_C2[1] as f64, 0.0));
        add_sh_term!(
            5,
            zz.affine((2.0 * SH_C2[2]) as f64, 0.0)?
                .broadcast_sub(&xx.affine(SH_C2[2] as f64, 0.0)?)?
                .broadcast_sub(&yy.affine(SH_C2[2] as f64, 0.0)?)
        );
        add_sh_term!(6, xz.affine(SH_C2[3] as f64, 0.0));
        add_sh_term!(
            7,
            xx.affine(SH_C2[4] as f64, 0.0)?
                .broadcast_sub(&yy.affine(SH_C2[4] as f64, 0.0)?)
        );
    }

    if active_degree > 2 {
        add_sh_term!(
            8,
            y.broadcast_mul(&xx.affine(3.0, 0.0)?.broadcast_sub(&yy)?)?
                .affine(SH_C3[0] as f64, 0.0)
        );
        add_sh_term!(9, xy.broadcast_mul(&z)?.affine(SH_C3[1] as f64, 0.0));
        add_sh_term!(
            10,
            y.broadcast_mul(
                &zz.affine(4.0, 0.0)?
                    .broadcast_sub(&xx)?
                    .broadcast_sub(&yy)?,
            )?
            .affine(SH_C3[2] as f64, 0.0)
        );
        add_sh_term!(
            11,
            z.broadcast_mul(
                &zz.affine(2.0, 0.0)?
                    .broadcast_sub(&xx.affine(3.0, 0.0)?)?
                    .broadcast_sub(&yy.affine(3.0, 0.0)?)?,
            )?
            .affine(SH_C3[3] as f64, 0.0)
        );
        add_sh_term!(
            12,
            x.broadcast_mul(
                &zz.affine(4.0, 0.0)?
                    .broadcast_sub(&xx)?
                    .broadcast_sub(&yy)?,
            )?
            .affine(SH_C3[4] as f64, 0.0)
        );
        add_sh_term!(
            13,
            z.broadcast_mul(&xx.broadcast_sub(&yy)?)?
                .affine(SH_C3[5] as f64, 0.0)
        );
        add_sh_term!(
            14,
            x.broadcast_mul(&xx.broadcast_sub(&yy.affine(3.0, 0.0)?)?)?
                .affine(SH_C3[6] as f64, 0.0)
        );
    }

    if active_degree > 3 {
        let zz7_minus_1 = zz.affine(7.0, -1.0)?;
        let zz7_minus_3 = zz.affine(7.0, -3.0)?;
        add_sh_term!(
            15,
            xy.broadcast_mul(&xx.broadcast_sub(&yy)?)?
                .affine(SH_C4[0] as f64, 0.0)
        );
        add_sh_term!(
            16,
            yz.broadcast_mul(&xx.affine(3.0, 0.0)?.broadcast_sub(&yy)?)?
                .affine(SH_C4[1] as f64, 0.0)
        );
        add_sh_term!(
            17,
            xy.broadcast_mul(&zz.affine(7.0, -1.0)?)?
                .affine(SH_C4[2] as f64, 0.0)
        );
        add_sh_term!(
            18,
            yz.broadcast_mul(&zz7_minus_3)?.affine(SH_C4[3] as f64, 0.0)
        );
        add_sh_term!(
            19,
            zz.broadcast_mul(&zz.affine(35.0, -30.0)?)?
                .affine(SH_C4[4] as f64, 3.0 * SH_C4[4] as f64)
        );
        add_sh_term!(
            20,
            xz.broadcast_mul(&zz7_minus_3)?.affine(SH_C4[5] as f64, 0.0)
        );
        add_sh_term!(
            21,
            xx.broadcast_sub(&yy)?
                .broadcast_mul(&zz7_minus_1)?
                .affine(SH_C4[6] as f64, 0.0)
        );
        add_sh_term!(
            22,
            xz.broadcast_mul(&xx.broadcast_sub(&yy.affine(3.0, 0.0)?)?)?
                .affine(SH_C4[7] as f64, 0.0)
        );
        add_sh_term!(
            23,
            xx.broadcast_mul(&xx.broadcast_sub(&yy.affine(3.0, 0.0)?)?)?
                .broadcast_sub(&yy.broadcast_mul(&xx.affine(3.0, 0.0)?.broadcast_sub(&yy)?)?)?
                .affine(SH_C4[8] as f64, 0.0)
        );
    }

    color.clamp(0.0, f32::MAX)
}

pub(super) fn filter_projected_gaussians_by_cluster_visibility(
    projected_cpu: &mut Vec<CpuProjectedGaussian>,
    cluster_visible_mask: Option<&[bool]>,
) {
    let Some(mask) = cluster_visible_mask else {
        return;
    };

    projected_cpu.retain(|gaussian| {
        mask.get(gaussian.source_idx as usize)
            .copied()
            .unwrap_or(true)
    });
}

pub(super) fn execute_forward_pass(
    ctx: &mut MetalForwardContext<'_>,
    inputs: MetalForwardInputs<'_>,
) -> candle_core::Result<(RenderedFrame, ProjectedGaussians, MetalRenderProfile)> {
    if inputs.gaussians.len() == 0 {
        return Ok((
            RenderedFrame {
                color: Tensor::zeros((ctx.settings.pixel_count, 3), DType::F32, ctx.device)?,
                depth: Tensor::zeros((ctx.settings.pixel_count,), DType::F32, ctx.device)?,
                alpha: Tensor::zeros((ctx.settings.pixel_count,), DType::F32, ctx.device)?,
            },
            ProjectedGaussians::empty(ctx.device)?,
            MetalRenderProfile::default(),
        ));
    }

    let (projected, mut profile) = project_gaussians(ctx, inputs)?;
    let raster_start = Instant::now();
    let tile_bins = build_tile_bins(ctx.runtime, ctx.device, &projected)?;
    let (rendered, tile_stats, native_profile) = if ctx.settings.use_native_forward {
        let (rendered, tile_stats, native_profile) = rasterize_native(
            ctx.runtime,
            &projected,
            &tile_bins,
            ctx.settings.render_width,
            ctx.settings.render_height,
        )?;
        (rendered, tile_stats, Some(native_profile))
    } else {
        let (rendered, tile_stats) = rasterize(
            ctx.runtime,
            ctx.device,
            ctx.settings.pixel_count,
            ctx.settings.chunk_size,
            &projected,
            &tile_bins,
        )?;
        (rendered, tile_stats, None)
    };
    synchronize_if_needed(ctx.device, inputs.should_profile)?;

    let mut projected = projected;
    projected.tile_bins = tile_bins;

    profile.rasterization = raster_start.elapsed();
    profile.active_tiles = tile_stats.active_tiles;
    profile.tile_gaussian_refs = tile_stats.tile_gaussian_refs;
    profile.max_gaussians_per_tile = tile_stats.max_gaussians_per_tile;
    if inputs.should_profile && ctx.device.is_metal() {
        profile.native_forward = if let Some(native_profile) = native_profile {
            let (baseline, _) = rasterize(
                ctx.runtime,
                ctx.device,
                ctx.settings.pixel_count,
                ctx.settings.chunk_size,
                &projected,
                &projected.tile_bins,
            )?;
            Some(build_native_parity_profile(
                &baseline,
                &rendered,
                native_profile,
            )?)
        } else {
            Some(profile_native_forward(
                ctx.runtime,
                ctx.settings.render_width,
                ctx.settings.render_height,
                &projected,
                &projected.tile_bins,
                &rendered,
            )?)
        };
    }

    Ok((rendered, projected, profile))
}

pub(crate) fn execute_forward_pass_on_runtime(
    runtime: &mut MetalRuntime,
    device: &Device,
    settings: MetalForwardSettings,
    inputs: MetalForwardInputs<'_>,
) -> candle_core::Result<(RenderedFrame, ProjectedGaussians, MetalRenderProfile)> {
    let mut ctx = MetalForwardContext {
        runtime,
        device,
        settings,
    };
    execute_forward_pass(&mut ctx, inputs)
}

pub(super) fn project_gaussians(
    ctx: &mut MetalForwardContext<'_>,
    inputs: MetalForwardInputs<'_>,
) -> candle_core::Result<(ProjectedGaussians, MetalRenderProfile)> {
    ctx.runtime.stage_camera(inputs.camera)?;
    let mut profile = MetalRenderProfile::default();
    profile.total_gaussians = inputs.gaussians.len();
    let projection_start = Instant::now();
    let scales = inputs.gaussians.scales.as_tensor().detach().exp()?;
    let opacity_logits = inputs.gaussians.opacities.as_tensor().detach();
    let colors = inputs.colors.detach();
    let rotations = inputs.gaussians.rotations()?.detach();

    let px = inputs.positions.narrow(1, 0, 1)?.squeeze(1)?;
    let py = inputs.positions.narrow(1, 1, 1)?.squeeze(1)?;
    let pz = inputs.positions.narrow(1, 2, 1)?.squeeze(1)?;

    let x = px
        .affine(
            inputs.camera.rotation[0][0] as f64,
            inputs.camera.translation[0] as f64,
        )?
        .broadcast_add(&py.affine(inputs.camera.rotation[0][1] as f64, 0.0)?)?
        .broadcast_add(&pz.affine(inputs.camera.rotation[0][2] as f64, 0.0)?)?;
    let y = px
        .affine(
            inputs.camera.rotation[1][0] as f64,
            inputs.camera.translation[1] as f64,
        )?
        .broadcast_add(&py.affine(inputs.camera.rotation[1][1] as f64, 0.0)?)?
        .broadcast_add(&pz.affine(inputs.camera.rotation[1][2] as f64, 0.0)?)?;
    let z = px
        .affine(
            inputs.camera.rotation[2][0] as f64,
            inputs.camera.translation[2] as f64,
        )?
        .broadcast_add(&py.affine(inputs.camera.rotation[2][1] as f64, 0.0)?)?
        .broadcast_add(&pz.affine(inputs.camera.rotation[2][2] as f64, 0.0)?)?;

    let staging_source = if ctx.device.is_metal() {
        ProjectionStagingSource::RuntimeBufferRead
    } else {
        ProjectionStagingSource::TensorReadback
    };
    let max_x_bound = inputs.camera.width.saturating_sub(1) as f32;
    let max_y_bound = inputs.camera.height.saturating_sub(1) as f32;

    let mut projected_cpu = Vec::with_capacity(inputs.gaussians.len());
    let mut visible_source_indices = Vec::new();
    if ctx.device.is_metal() {
        let gpu_batch = ctx.runtime.project_gaussians(
            inputs.gaussians,
            &colors,
            inputs.collect_visible_indices,
        )?;
        visible_source_indices = gpu_batch.visible_source_indices;
        profile.visible_gaussians = gpu_batch.visible_count;
        if !ctx.settings.use_native_forward || inputs.should_profile || ctx.settings.litegs_mode {
            let records = ctx.runtime.read_buffer_structs::<MetalProjectionRecord>(
                MetalBufferSlot::ProjectionRecords,
                gpu_batch.visible_count,
            )?;
            projected_cpu.extend(records.into_iter().map(cpu_projected_from_record));
        }
    } else {
        let x_values = ctx.runtime.read_tensor_flat::<f32>(&x)?;
        let y_values = ctx.runtime.read_tensor_flat::<f32>(&y)?;
        let z_values = ctx.runtime.read_tensor_flat::<f32>(&z)?;
        let scale_values = ctx.runtime.read_tensor_flat::<f32>(&scales)?;
        let rotation_values = ctx.runtime.read_tensor_flat::<f32>(&rotations)?;
        let opacity_logit_values = ctx.runtime.read_tensor_flat::<f32>(&opacity_logits)?;
        let color_values = ctx.runtime.read_tensor_flat::<f32>(&colors)?;
        for idx in 0..inputs.gaussians.len() {
            if let Some(mask) = inputs.cluster_visible_mask {
                if !mask.get(idx).copied().unwrap_or(true) {
                    continue;
                }
            }

            let z_value = x_values
                .get(idx)
                .zip(y_values.get(idx))
                .zip(z_values.get(idx))
                .map(|((_, _), z)| *z)
                .unwrap_or(0.0);
            if !z_value.is_finite() || z_value < 1e-4 {
                continue;
            }
            let x_value = x_values[idx];
            let y_value = y_values[idx];
            let scale3d = row_to_vec3(row_slice(&scale_values, 3, idx));
            let rotation = row_to_quaternion(row_slice(&rotation_values, 4, idx));
            let u_value = inputs.camera.fx * x_value / z_value + inputs.camera.cx;
            let v_value = inputs.camera.fy * y_value / z_value + inputs.camera.cy;
            let (raw_sigma_x, raw_sigma_y) = projected_axis_aligned_sigmas(
                x_value,
                y_value,
                z_value,
                scale3d,
                rotation,
                &inputs.camera.rotation,
                inputs.camera.fx,
                inputs.camera.fy,
            );
            if !raw_sigma_x.is_finite() || !raw_sigma_y.is_finite() {
                continue;
            }
            let sigma_x = raw_sigma_x.clamp(0.5, 256.0);
            let sigma_y = raw_sigma_y.clamp(0.5, 256.0);
            let support_x = sigma_x * 3.0;
            let support_y = sigma_y * 3.0;
            if u_value + support_x < 0.0
                || u_value - support_x > inputs.camera.width as f32
                || v_value + support_y < 0.0
                || v_value - support_y > inputs.camera.height as f32
            {
                continue;
            }

            let opacity_logit = opacity_logit_values.get(idx).copied().unwrap_or(0.0);
            let color = row_to_vec3(row_slice(&color_values, 3, idx));
            projected_cpu.push(CpuProjectedGaussian {
                source_idx: idx as u32,
                u: u_value,
                v: v_value,
                sigma_x,
                sigma_y,
                raw_sigma_x,
                raw_sigma_y,
                depth: z_value,
                opacity: sigmoid_scalar(opacity_logit),
                opacity_logit,
                scale3d,
                color,
                min_x: (u_value - support_x).clamp(0.0, max_x_bound),
                max_x: (u_value + support_x).clamp(0.0, max_x_bound),
                min_y: (v_value - support_y).clamp(0.0, max_y_bound),
                max_y: (v_value + support_y).clamp(0.0, max_y_bound),
            });
        }
    }

    let had_projected_cpu_rows = !projected_cpu.is_empty();
    if had_projected_cpu_rows {
        filter_projected_gaussians_by_cluster_visibility(
            &mut projected_cpu,
            inputs.cluster_visible_mask,
        );
    }

    if !ctx.device.is_metal() || had_projected_cpu_rows {
        visible_source_indices = projected_cpu.iter().map(|g| g.source_idx).collect();
        profile.visible_gaussians = projected_cpu.len();
    }
    synchronize_if_needed(ctx.device, inputs.should_profile)?;
    profile.projection = projection_start.elapsed();

    if profile.visible_gaussians == 0 {
        let mut empty = ProjectedGaussians::empty(ctx.device)?;
        empty.visible_source_indices = visible_source_indices;
        empty.visible_count = profile.visible_gaussians;
        empty.staging_source = staging_source;
        return Ok((empty, profile));
    }

    let sort_start = Instant::now();
    if !ctx.device.is_metal() {
        projected_cpu.sort_unstable_by(|lhs, rhs| {
            lhs.depth.partial_cmp(&rhs.depth).unwrap_or(Ordering::Equal)
        });
        visible_source_indices = projected_cpu.iter().map(|g| g.source_idx).collect();
    }
    let source_indices: Vec<u32> = if ctx.device.is_metal() && projected_cpu.is_empty() {
        visible_source_indices.clone()
    } else {
        projected_cpu.iter().map(|g| g.source_idx).collect()
    };

    let effective_count = if matches!(staging_source, ProjectionStagingSource::RuntimeBufferRead) {
        profile.visible_gaussians
    } else {
        source_indices.len()
    };

    let u: Vec<f32> = projected_cpu.iter().map(|g| g.u).collect();
    let v: Vec<f32> = projected_cpu.iter().map(|g| g.v).collect();
    let sigma_x: Vec<f32> = projected_cpu.iter().map(|g| g.sigma_x).collect();
    let sigma_y: Vec<f32> = projected_cpu.iter().map(|g| g.sigma_y).collect();
    let raw_sigma_x: Vec<f32> = projected_cpu.iter().map(|g| g.raw_sigma_x).collect();
    let raw_sigma_y: Vec<f32> = projected_cpu.iter().map(|g| g.raw_sigma_y).collect();
    let depth: Vec<f32> = projected_cpu.iter().map(|g| g.depth).collect();
    let opacity: Vec<f32> = projected_cpu.iter().map(|g| g.opacity).collect();
    let opacity_logits: Vec<f32> = projected_cpu.iter().map(|g| g.opacity_logit).collect();
    let scale3d: Vec<f32> = projected_cpu.iter().flat_map(|g| g.scale3d).collect();
    let color_values: Vec<f32> = projected_cpu.iter().flat_map(|g| g.color).collect();
    let min_x: Vec<f32> = projected_cpu.iter().map(|g| g.min_x).collect();
    let max_x: Vec<f32> = projected_cpu.iter().map(|g| g.max_x).collect();
    let min_y: Vec<f32> = projected_cpu.iter().map(|g| g.min_y).collect();
    let max_y: Vec<f32> = projected_cpu.iter().map(|g| g.max_y).collect();
    let projected = ProjectedGaussians {
        source_indices: if effective_count == 0 {
            Tensor::zeros((0,), DType::U32, ctx.device)?
        } else if source_indices.is_empty() {
            Tensor::zeros((effective_count,), DType::U32, ctx.device)?
        } else {
            Tensor::from_slice(&source_indices, source_indices.len(), ctx.device)?
        },
        u: if u.is_empty() {
            Tensor::zeros((effective_count,), DType::F32, ctx.device)?
        } else {
            Tensor::from_slice(&u, u.len(), ctx.device)?
        },
        v: if v.is_empty() {
            Tensor::zeros((effective_count,), DType::F32, ctx.device)?
        } else {
            Tensor::from_slice(&v, v.len(), ctx.device)?
        },
        sigma_x: if sigma_x.is_empty() {
            Tensor::zeros((effective_count,), DType::F32, ctx.device)?
        } else {
            Tensor::from_slice(&sigma_x, sigma_x.len(), ctx.device)?
        },
        sigma_y: if sigma_y.is_empty() {
            Tensor::zeros((effective_count,), DType::F32, ctx.device)?
        } else {
            Tensor::from_slice(&sigma_y, sigma_y.len(), ctx.device)?
        },
        raw_sigma_x: if raw_sigma_x.is_empty() {
            Tensor::zeros((effective_count,), DType::F32, ctx.device)?
        } else {
            Tensor::from_slice(&raw_sigma_x, raw_sigma_x.len(), ctx.device)?
        },
        raw_sigma_y: if raw_sigma_y.is_empty() {
            Tensor::zeros((effective_count,), DType::F32, ctx.device)?
        } else {
            Tensor::from_slice(&raw_sigma_y, raw_sigma_y.len(), ctx.device)?
        },
        depth: if depth.is_empty() {
            Tensor::zeros((effective_count,), DType::F32, ctx.device)?
        } else {
            Tensor::from_slice(&depth, depth.len(), ctx.device)?
        },
        opacity: if opacity.is_empty() {
            Tensor::zeros((effective_count,), DType::F32, ctx.device)?
        } else {
            Tensor::from_slice(&opacity, opacity.len(), ctx.device)?
        },
        opacity_logits: if opacity_logits.is_empty() {
            Tensor::zeros((effective_count,), DType::F32, ctx.device)?
        } else {
            Tensor::from_slice(&opacity_logits, opacity_logits.len(), ctx.device)?
        },
        scale3d: if scale3d.is_empty() {
            Tensor::zeros((effective_count, 3), DType::F32, ctx.device)?
        } else {
            Tensor::from_slice(&scale3d, (effective_count, 3), ctx.device)?
        },
        colors: if color_values.is_empty() {
            Tensor::zeros((effective_count, 3), DType::F32, ctx.device)?
        } else {
            Tensor::from_slice(&color_values, (effective_count, 3), ctx.device)?
        },
        min_x: if min_x.is_empty() {
            Tensor::zeros((effective_count,), DType::F32, ctx.device)?
        } else {
            Tensor::from_slice(&min_x, min_x.len(), ctx.device)?
        },
        max_x: if max_x.is_empty() {
            Tensor::zeros((effective_count,), DType::F32, ctx.device)?
        } else {
            Tensor::from_slice(&max_x, max_x.len(), ctx.device)?
        },
        min_y: if min_y.is_empty() {
            Tensor::zeros((effective_count,), DType::F32, ctx.device)?
        } else {
            Tensor::from_slice(&min_y, min_y.len(), ctx.device)?
        },
        max_y: if max_y.is_empty() {
            Tensor::zeros((effective_count,), DType::F32, ctx.device)?
        } else {
            Tensor::from_slice(&max_y, max_y.len(), ctx.device)?
        },
        visible_source_indices,
        visible_count: profile.visible_gaussians,
        tile_bins: ProjectedTileBins::default(),
        staging_source,
    };
    synchronize_if_needed(ctx.device, inputs.should_profile)?;
    profile.sorting = sort_start.elapsed();

    Ok((projected, profile))
}

#[cfg(test)]
#[allow(dead_code)]
pub(super) fn project_gaussians_on_runtime(
    runtime: &mut MetalRuntime,
    device: &Device,
    settings: MetalForwardSettings,
    inputs: MetalForwardInputs<'_>,
) -> candle_core::Result<(ProjectedGaussians, MetalRenderProfile)> {
    let mut ctx = MetalForwardContext {
        runtime,
        device,
        settings,
    };
    project_gaussians(&mut ctx, inputs)
}

pub(super) fn rasterize(
    runtime: &mut MetalRuntime,
    device: &Device,
    pixel_count: usize,
    chunk_size: usize,
    projected: &ProjectedGaussians,
    tile_bins: &ProjectedTileBins,
) -> candle_core::Result<(RenderedFrame, TileBinningStats)> {
    let mut color_acc = Tensor::zeros((pixel_count, 3), DType::F32, device)?;
    let mut depth_acc = Tensor::zeros((pixel_count,), DType::F32, device)?;
    let mut alpha_acc = Tensor::zeros((pixel_count,), DType::F32, device)?;
    let tile_stats = tile_binning_stats(tile_bins);
    let tile_index_tensor = if tile_bins.total_assignments() == 0 {
        Tensor::zeros((0,), DType::U32, device)?
    } else {
        Tensor::from_slice(
            tile_bins.packed_indices(),
            tile_bins.total_assignments(),
            device,
        )?
    };

    for &tile_idx in tile_bins.active_tiles() {
        let Some(record) = tile_bins.record(tile_idx) else {
            continue;
        };
        if record.count() == 0 {
            continue;
        }

        let window = runtime.tile_window(tile_idx)?;
        let mut tile_color_acc = Tensor::zeros((window.pixel_count, 3), DType::F32, device)?;
        let mut tile_depth_acc = Tensor::zeros((window.pixel_count,), DType::F32, device)?;
        let mut tile_alpha_acc = Tensor::zeros((window.pixel_count,), DType::F32, device)?;
        let mut tile_trans = Tensor::ones((window.pixel_count,), DType::F32, device)?;

        for start in (0..record.count()).step_by(chunk_size) {
            let len = (record.count() - start).min(chunk_size);
            let chunk_indices = tile_index_tensor.narrow(0, record.start() + start, len)?;
            let alpha = chunk_alpha(
                &window,
                &projected.u.index_select(&chunk_indices, 0)?,
                &projected.v.index_select(&chunk_indices, 0)?,
                &projected.sigma_x.index_select(&chunk_indices, 0)?,
                &projected.sigma_y.index_select(&chunk_indices, 0)?,
                &projected.opacity.index_select(&chunk_indices, 0)?,
            )?;
            let (chunk_color, chunk_depth, chunk_alpha, tail_trans) = integrate_chunk(
                device,
                &alpha,
                &projected.colors.index_select(&chunk_indices, 0)?,
                &projected.depth.index_select(&chunk_indices, 0)?,
            )?;
            let tile_trans_col = tile_trans.reshape((window.pixel_count, 1))?;
            tile_color_acc =
                tile_color_acc.broadcast_add(&chunk_color.broadcast_mul(&tile_trans_col)?)?;
            tile_depth_acc =
                tile_depth_acc.broadcast_add(&chunk_depth.broadcast_mul(&tile_trans)?)?;
            tile_alpha_acc =
                tile_alpha_acc.broadcast_add(&chunk_alpha.broadcast_mul(&tile_trans)?)?;
            tile_trans = tile_trans.broadcast_mul(&tail_trans)?;
        }

        color_acc = color_acc.index_add(&window.indices, &tile_color_acc, 0)?;
        depth_acc = depth_acc.index_add(&window.indices, &tile_depth_acc, 0)?;
        alpha_acc = alpha_acc.index_add(&window.indices, &tile_alpha_acc, 0)?;
    }

    let denom = alpha_acc.broadcast_add(&Tensor::new(1e-6f32, device)?)?;
    Ok((
        RenderedFrame {
            color: color_acc.clamp(0.0, 1.0)?,
            depth: depth_acc.broadcast_div(&denom)?,
            alpha: alpha_acc,
        },
        tile_stats,
    ))
}

pub(super) fn rasterize_native(
    runtime: &mut MetalRuntime,
    projected: &ProjectedGaussians,
    tile_bins: &ProjectedTileBins,
    render_width: usize,
    render_height: usize,
) -> candle_core::Result<(RenderedFrame, TileBinningStats, NativeForwardProfile)> {
    if !matches!(
        projected.staging_source,
        ProjectionStagingSource::RuntimeBufferRead
    ) {
        stage_projected_records_from_tensors(runtime, projected)?;
    }
    let (native_frame, native_profile) = runtime.rasterize_forward(
        projected.visible_count,
        tile_bins.as_runtime(),
        render_width,
        render_height,
    )?;
    let tile_stats = tile_binning_stats(tile_bins);
    Ok((
        RenderedFrame {
            color: native_frame.color,
            depth: native_frame.depth,
            alpha: native_frame.alpha,
        },
        tile_stats,
        native_profile,
    ))
}

pub(super) fn build_tile_bins(
    runtime: &mut MetalRuntime,
    device: &Device,
    projected: &ProjectedGaussians,
) -> candle_core::Result<ProjectedTileBins> {
    if device.is_metal() {
        if !matches!(
            projected.staging_source,
            ProjectionStagingSource::RuntimeBufferRead
        ) {
            stage_projected_records_from_tensors(runtime, projected)?;
        }
        return runtime
            .build_tile_bins_gpu(projected.visible_count)
            .map(ProjectedTileBins::from_runtime);
    }
    let min_x_values = projected.min_x.to_vec1::<f32>()?;
    let max_x_values = projected.max_x.to_vec1::<f32>()?;
    let min_y_values = projected.min_y.to_vec1::<f32>()?;
    let max_y_values = projected.max_y.to_vec1::<f32>()?;
    runtime
        .build_tile_bins(&min_x_values, &max_x_values, &min_y_values, &max_y_values)
        .map(ProjectedTileBins::from_runtime)
}

pub(super) fn profile_native_forward(
    runtime: &mut MetalRuntime,
    render_width: usize,
    render_height: usize,
    projected: &ProjectedGaussians,
    tile_bins: &ProjectedTileBins,
    baseline: &RenderedFrame,
) -> candle_core::Result<NativeParityProfile> {
    if !matches!(
        projected.staging_source,
        ProjectionStagingSource::RuntimeBufferRead
    ) {
        stage_projected_records_from_tensors(runtime, projected)?;
    }
    let (native_frame, native_profile) = runtime.rasterize_forward(
        projected.visible_count,
        tile_bins.as_runtime(),
        render_width,
        render_height,
    )?;
    build_native_parity_profile(
        baseline,
        &RenderedFrame {
            color: native_frame.color,
            depth: native_frame.depth,
            alpha: native_frame.alpha,
        },
        native_profile,
    )
}

pub(super) fn build_native_parity_profile(
    baseline: &RenderedFrame,
    native: &RenderedFrame,
    native_profile: NativeForwardProfile,
) -> candle_core::Result<NativeParityProfile> {
    let color_max_abs = baseline
        .color
        .sub(&native.color)?
        .abs()?
        .max_all()?
        .to_vec0::<f32>()?;
    let depth_max_abs = baseline
        .depth
        .sub(&native.depth)?
        .abs()?
        .max_all()?
        .to_vec0::<f32>()?;
    let alpha_max_abs = baseline
        .alpha
        .sub(&native.alpha)?
        .abs()?
        .max_all()?
        .to_vec0::<f32>()?;
    Ok(NativeParityProfile {
        setup: native_profile.setup,
        staging: native_profile.staging,
        kernel: native_profile.kernel,
        total: native_profile.total,
        color_max_abs,
        depth_max_abs,
        alpha_max_abs,
    })
}

pub(super) fn stage_projected_records_from_tensors(
    runtime: &mut MetalRuntime,
    projected: &ProjectedGaussians,
) -> candle_core::Result<()> {
    let projected_cpu = projected_rows_to_cpu(projected)?;
    let mut records = Vec::with_capacity(projected_cpu.len());
    for gaussian in projected_cpu {
        records.push(MetalProjectionRecord {
            source_idx: gaussian.source_idx,
            visible: 1,
            u: gaussian.u,
            v: gaussian.v,
            sigma_x: gaussian.sigma_x,
            sigma_y: gaussian.sigma_y,
            raw_sigma_x: gaussian.raw_sigma_x,
            raw_sigma_y: gaussian.raw_sigma_y,
            depth: gaussian.depth,
            opacity: gaussian.opacity,
            opacity_logit: gaussian.opacity_logit,
            scale_x: gaussian.scale3d[0],
            scale_y: gaussian.scale3d[1],
            scale_z: gaussian.scale3d[2],
            color_r: gaussian.color[0],
            color_g: gaussian.color[1],
            color_b: gaussian.color[2],
            min_x: gaussian.min_x,
            max_x: gaussian.max_x,
            min_y: gaussian.min_y,
            max_y: gaussian.max_y,
        });
    }
    runtime.ensure_projection_record_buffer(records.len())?;
    runtime.write_projection_records(&records)
}

pub(super) fn tile_binning_stats(tile_bins: &ProjectedTileBins) -> TileBinningStats {
    TileBinningStats {
        active_tiles: tile_bins.active_tile_count(),
        tile_gaussian_refs: tile_bins.total_assignments(),
        max_gaussians_per_tile: tile_bins.max_gaussians_per_tile(),
    }
}

fn chunk_alpha(
    window: &ChunkPixelWindow,
    u: &Tensor,
    v: &Tensor,
    sigma_x: &Tensor,
    sigma_y: &Tensor,
    opacity: &Tensor,
) -> candle_core::Result<Tensor> {
    let len = u.dim(0)?;
    let dx = window
        .pixel_x
        .broadcast_sub(&u.reshape((len, 1))?)?
        .broadcast_div(&sigma_x.reshape((len, 1))?)?;
    let dy = window
        .pixel_y
        .broadcast_sub(&v.reshape((len, 1))?)?
        .broadcast_div(&sigma_y.reshape((len, 1))?)?;
    let exponent = dx.sqr()?.broadcast_add(&dy.sqr()?)?.affine(-0.5, 0.0)?;
    exponent
        .exp()?
        .broadcast_mul(&opacity.reshape((len, 1))?)?
        .clamp(0.0, 0.99)
}

fn integrate_chunk(
    device: &Device,
    alpha: &Tensor,
    colors: &Tensor,
    depth: &Tensor,
) -> candle_core::Result<(Tensor, Tensor, Tensor, Tensor)> {
    let len = alpha.dim(0)?;
    let pixel_count = alpha.dim(1)?;
    let zero_row = Tensor::zeros((1, pixel_count), DType::F32, device)?;
    let inclusive = Tensor::ones_like(alpha)?
        .broadcast_sub(alpha)?
        .clamp(1e-4, 1.0)?
        .log()?
        .cumsum(0)?;
    let exclusive = if len == 1 {
        zero_row
    } else {
        Tensor::cat(&[&zero_row, &inclusive.narrow(0, 0, len - 1)?], 0)?
    };
    let local_contrib = alpha.broadcast_mul(&exclusive.exp()?)?;
    let local_contrib_t = local_contrib.t()?;
    let chunk_color = local_contrib_t.matmul(colors)?;
    let chunk_depth = local_contrib_t
        .matmul(&depth.reshape((len, 1))?)?
        .squeeze(1)?;
    let chunk_alpha = local_contrib.sum(0)?;
    let tail_trans = inclusive.get_on_dim(0, len - 1)?.exp()?;
    Ok((chunk_color, chunk_depth, chunk_alpha, tail_trans))
}

fn synchronize_if_needed(device: &Device, should_profile: bool) -> candle_core::Result<()> {
    if should_profile {
        device.synchronize()?;
    }
    Ok(())
}

fn sigmoid_scalar(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}
