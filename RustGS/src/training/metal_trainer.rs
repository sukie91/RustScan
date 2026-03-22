//! Metal-native 3DGS training backend.
//!
//! This path keeps projection, rasterization, loss computation, backward, and
//! optimizer updates on the Metal device by expressing the renderer as Candle
//! tensor ops. It is intentionally lower resolution than the legacy hybrid
//! path so we can keep the whole step on GPU without CPU tile rasterization.

use std::time::{Duration, Instant};

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};

use crate::diff::diff_splat::{DiffCamera, TrainableGaussians};
use crate::{GaussianMap, TrainingDataset, TrainingError};

use super::data_loading::{
    load_training_data, map_from_trainable, trainable_from_map, LoadedTrainingData,
};
use super::TrainingConfig;

const DEFAULT_METAL_MAX_INITIAL_GAUSSIANS: usize = 4_096;
const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * 1024 * 1024;
const DEFAULT_METAL_MEMORY_BUDGET_BYTES: u64 = 24 * GIB;
const METAL_SYSTEM_MEMORY_BUDGET_NUMERATOR: u64 = 13;
const METAL_SYSTEM_MEMORY_BUDGET_DENOMINATOR: u64 = 20;
const METAL_WARN_BUDGET_NUMERATOR: u64 = 85;
const METAL_WARN_BUDGET_DENOMINATOR: u64 = 100;
const METAL_ESTIMATE_SAFETY_NUMERATOR: u64 = 15;
const METAL_ESTIMATE_SAFETY_DENOMINATOR: u64 = 100;
const METAL_ESTIMATE_MIN_SAFETY_BYTES: u64 = 256 * MIB;
const METAL_FRAME_BYTES_PER_PIXEL: u64 = 16;
const METAL_PIXEL_STATE_BYTES_PER_PIXEL: u64 = 40;
const METAL_GAUSSIAN_STATE_BYTES: u64 = 168;
const METAL_PROJECTED_BYTES_PER_GAUSSIAN: u64 = 64;
const METAL_CHUNK_WORKSPACE_BYTES_PER_GAUSSIAN_PIXEL: u64 = 64;
const METAL_RETAINED_GRAPH_BYTES_PER_GAUSSIAN_PIXEL: u64 = 224;

struct MetalTrainingFrame {
    camera: DiffCamera,
    target_color: Tensor,
    target_depth: Tensor,
}

struct ProjectedGaussians {
    u: Tensor,
    v: Tensor,
    sigma_x: Tensor,
    sigma_y: Tensor,
    depth: Tensor,
    opacity: Tensor,
    colors: Tensor,
}

struct RenderedFrame {
    color: Tensor,
    depth: Tensor,
}

struct MetalAdamState {
    m_pos: Tensor,
    v_pos: Tensor,
    m_scale: Tensor,
    v_scale: Tensor,
    m_rot: Tensor,
    v_rot: Tensor,
    m_op: Tensor,
    v_op: Tensor,
    m_color: Tensor,
    v_color: Tensor,
}

impl MetalAdamState {
    fn new(gaussians: &TrainableGaussians) -> candle_core::Result<Self> {
        Ok(Self {
            m_pos: gaussians.positions().zeros_like()?,
            v_pos: gaussians.positions().zeros_like()?,
            m_scale: gaussians.scales.as_tensor().zeros_like()?,
            v_scale: gaussians.scales.as_tensor().zeros_like()?,
            m_rot: gaussians.rotations.as_tensor().zeros_like()?,
            v_rot: gaussians.rotations.as_tensor().zeros_like()?,
            m_op: gaussians.opacities.as_tensor().zeros_like()?,
            v_op: gaussians.opacities.as_tensor().zeros_like()?,
            m_color: gaussians.colors().zeros_like()?,
            v_color: gaussians.colors().zeros_like()?,
        })
    }
}

pub struct MetalTrainer {
    device: Device,
    render_width: usize,
    render_height: usize,
    pixel_count: usize,
    pixel_x: Tensor,
    pixel_y: Tensor,
    chunk_size: usize,
    lr_pos: f32,
    lr_scale: f32,
    lr_rot: f32,
    lr_opacity: f32,
    lr_color: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    profile_steps: bool,
    profile_interval: usize,
    adam: Option<MetalAdamState>,
    iteration: usize,
    loss_history: Vec<f32>,
}

struct MetalTrainingStats {
    final_loss: f32,
}

#[derive(Debug, Clone, Copy)]
struct MetalMemoryEstimate {
    gaussian_state_bytes: u64,
    frame_bytes: u64,
    pixel_state_bytes: u64,
    projection_bytes: u64,
    chunk_workspace_bytes: u64,
    retained_graph_bytes: u64,
    safety_margin_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
struct MetalMemoryBudget {
    safe_bytes: u64,
    physical_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetalMemoryDecision {
    Allow,
    Warn,
    Block,
}

impl MetalMemoryEstimate {
    fn subtotal_bytes(&self) -> u64 {
        self.gaussian_state_bytes
            .saturating_add(self.frame_bytes)
            .saturating_add(self.pixel_state_bytes)
            .saturating_add(self.projection_bytes)
            .saturating_add(self.chunk_workspace_bytes)
            .saturating_add(self.retained_graph_bytes)
    }

    fn total_bytes(&self) -> u64 {
        self.subtotal_bytes()
            .saturating_add(self.safety_margin_bytes)
    }

    fn persistent_bytes(&self) -> u64 {
        self.gaussian_state_bytes
            .saturating_add(self.pixel_state_bytes)
            .saturating_add(self.projection_bytes)
    }

    fn top_components_summary(&self, count: usize) -> String {
        let mut components = vec![
            ("graph", self.retained_graph_bytes),
            ("chunk", self.chunk_workspace_bytes),
            ("frames", self.frame_bytes),
            ("persistent", self.persistent_bytes()),
            ("safety", self.safety_margin_bytes),
        ];
        components.retain(|(_, bytes)| *bytes > 0);
        components.sort_by(|lhs, rhs| rhs.1.cmp(&lhs.1));
        components
            .into_iter()
            .take(count)
            .map(|(label, bytes)| format!("{label}≈{}", format_memory(bytes)))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn recommendations(&self) -> Vec<&'static str> {
        let total = self.total_bytes().max(1);
        let mut recommendations = Vec::new();
        if self.retained_graph_bytes.saturating_mul(100) >= total.saturating_mul(45) {
            recommendations.push("lower --max-initial-gaussians or increase --sampling-step");
        }
        if self
            .frame_bytes
            .saturating_add(self.pixel_state_bytes)
            .saturating_mul(100)
            >= total.saturating_mul(20)
        {
            recommendations.push("lower --metal-render-scale");
        }
        if self.chunk_workspace_bytes.saturating_mul(100) >= total.saturating_mul(10) {
            recommendations.push("lower --metal-gaussian-chunk-size");
        }
        if recommendations.is_empty() {
            recommendations.push("lower --max-initial-gaussians or --metal-render-scale");
        }
        recommendations
    }
}

impl MetalMemoryBudget {
    fn describe(&self) -> String {
        match self.physical_bytes {
            Some(physical_bytes) => format!(
                "{:.1} GiB safe budget on {:.1} GiB system memory",
                bytes_to_gib(self.safe_bytes),
                bytes_to_gib(physical_bytes)
            ),
            None => format!(
                "{:.1} GiB safe budget (system memory unavailable)",
                bytes_to_gib(self.safe_bytes)
            ),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct MetalRenderProfile {
    projection: Duration,
    sorting: Duration,
    rasterization: Duration,
}

#[derive(Debug, Default, Clone, Copy)]
struct MetalStepProfile {
    projection: Duration,
    sorting: Duration,
    rasterization: Duration,
    loss: Duration,
    backward: Duration,
    optimizer: Duration,
    total: Duration,
}

impl MetalStepProfile {
    fn from_render(render: MetalRenderProfile) -> Self {
        Self {
            projection: render.projection,
            sorting: render.sorting,
            rasterization: render.rasterization,
            ..Default::default()
        }
    }

    fn log(&self, iter: usize, max_iterations: usize) {
        log::info!(
            "Metal profile iter {:5}/{:5} | total={:.2}ms | project={:.2}ms | sort={:.2}ms | raster={:.2}ms | loss={:.2}ms | backward={:.2}ms | optimizer={:.2}ms",
            iter,
            max_iterations,
            duration_ms(self.total),
            duration_ms(self.projection),
            duration_ms(self.sorting),
            duration_ms(self.rasterization),
            duration_ms(self.loss),
            duration_ms(self.backward),
            duration_ms(self.optimizer),
        );
    }
}

impl MetalTrainer {
    pub fn new(
        input_width: usize,
        input_height: usize,
        config: &TrainingConfig,
        device: Device,
    ) -> candle_core::Result<Self> {
        let (render_width, render_height) =
            scaled_dimensions(input_width, input_height, config.metal_render_scale);
        let pixel_count = render_width * render_height;
        let (pixel_x, pixel_y) = build_pixel_grids(render_width, render_height, &device)?;

        Ok(Self {
            device,
            render_width,
            render_height,
            pixel_count,
            pixel_x,
            pixel_y,
            chunk_size: config.metal_gaussian_chunk_size.max(1),
            lr_pos: config.lr_position,
            lr_scale: config.lr_scale,
            lr_rot: config.lr_rotation,
            lr_opacity: config.lr_opacity,
            lr_color: config.lr_color,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            profile_steps: config.metal_profile_steps,
            profile_interval: config.metal_profile_interval.max(1),
            adam: None,
            iteration: 0,
            loss_history: Vec::new(),
        })
    }

    fn prepare_frames(
        &self,
        loaded: &LoadedTrainingData,
    ) -> Result<Vec<MetalTrainingFrame>, TrainingError> {
        let mut frames = Vec::with_capacity(loaded.cameras.len());
        for idx in 0..loaded.cameras.len() {
            let src_camera = &loaded.cameras[idx];
            let target_color_cpu = resize_rgb(
                &loaded.colors[idx],
                src_camera.width,
                src_camera.height,
                self.render_width,
                self.render_height,
            );
            let target_depth_cpu = resize_depth(
                &loaded.depths[idx],
                src_camera.width,
                src_camera.height,
                self.render_width,
                self.render_height,
            );
            let scaled_camera = scale_camera(
                src_camera,
                self.render_width,
                self.render_height,
                &self.device,
            )?;
            frames.push(MetalTrainingFrame {
                camera: scaled_camera,
                target_color: Tensor::from_slice(
                    &target_color_cpu,
                    (self.pixel_count, 3),
                    &self.device,
                )?,
                target_depth: Tensor::from_slice(
                    &target_depth_cpu,
                    (self.pixel_count,),
                    &self.device,
                )?,
            });
        }
        Ok(frames)
    }

    fn train(
        &mut self,
        gaussians: &mut TrainableGaussians,
        frames: &[MetalTrainingFrame],
        max_iterations: usize,
    ) -> candle_core::Result<MetalTrainingStats> {
        if frames.is_empty() {
            candle_core::bail!("metal backend received zero training frames");
        }
        self.adam = Some(MetalAdamState::new(gaussians)?);

        log::info!(
            "MetalTrainer running at {}x{} | chunk_size={} | frames={} | initial_gaussians={}",
            self.render_width,
            self.render_height,
            self.chunk_size,
            frames.len(),
            gaussians.len()
        );

        let train_start = Instant::now();
        for iter in 0..max_iterations {
            let frame_idx = iter % frames.len();
            let should_log = iter < 5 || iter % 25 == 0;
            let should_profile =
                should_profile_iteration(self.profile_steps, self.profile_interval, iter);
            let step_start = Instant::now();
            let (loss, profile) =
                self.training_step(gaussians, &frames[frame_idx], should_profile)?;
            if should_log {
                log::info!(
                    "Metal iter {:5}/{:5} | frame {:3}/{:3} | loss {:.6} | step_time={:.2}s | elapsed={:.2}s",
                    iter,
                    max_iterations,
                    frame_idx + 1,
                    frames.len(),
                    loss,
                    step_start.elapsed().as_secs_f64(),
                    train_start.elapsed().as_secs_f64()
                );
            }
            if let Some(profile) = profile {
                profile.log(iter, max_iterations);
            }
        }

        Ok(MetalTrainingStats {
            final_loss: self.loss_history.last().copied().unwrap_or(0.0),
        })
    }

    fn training_step(
        &mut self,
        gaussians: &mut TrainableGaussians,
        frame: &MetalTrainingFrame,
        should_profile: bool,
    ) -> candle_core::Result<(f32, Option<MetalStepProfile>)> {
        self.iteration += 1;
        let total_start = Instant::now();
        let (rendered, render_profile) = self.render(gaussians, &frame.camera, should_profile)?;
        let mut profile = MetalStepProfile::from_render(render_profile);

        let loss_start = Instant::now();
        let color_loss = rendered.color.sub(&frame.target_color)?.abs()?.mean_all()?;
        let depth_loss = rendered.depth.sub(&frame.target_depth)?.abs()?.mean_all()?;
        let total = color_loss.broadcast_add(&depth_loss.affine(0.1, 0.0)?)?;
        let loss_value = total.to_vec0::<f32>()?;
        self.synchronize_if_needed(should_profile)?;
        profile.loss = loss_start.elapsed();

        let backward_start = Instant::now();
        let grads = total.backward()?;
        self.synchronize_if_needed(should_profile)?;
        profile.backward = backward_start.elapsed();

        let optimizer_start = Instant::now();
        self.apply_gradients(gaussians, &grads)?;
        self.synchronize_if_needed(should_profile)?;
        profile.optimizer = optimizer_start.elapsed();
        profile.total = total_start.elapsed();
        self.loss_history.push(loss_value);

        Ok((
            loss_value,
            if should_profile { Some(profile) } else { None },
        ))
    }

    fn apply_gradients(
        &mut self,
        gaussians: &mut TrainableGaussians,
        grads: &GradStore,
    ) -> candle_core::Result<()> {
        let adam = self
            .adam
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("adam state not initialized".into()))?;

        if let Some(grad) = grads.get(gaussians.positions()) {
            adam_step_var(
                &gaussians.positions,
                grad,
                &mut adam.m_pos,
                &mut adam.v_pos,
                self.lr_pos,
                self.beta1,
                self.beta2,
                self.eps,
                self.iteration,
            )?;
        }
        if let Some(grad) = grads.get(gaussians.scales.as_tensor()) {
            adam_step_var(
                &gaussians.scales,
                grad,
                &mut adam.m_scale,
                &mut adam.v_scale,
                self.lr_scale,
                self.beta1,
                self.beta2,
                self.eps,
                self.iteration,
            )?;
        }
        if let Some(grad) = grads.get(gaussians.rotations.as_tensor()) {
            adam_step_var(
                &gaussians.rotations,
                grad,
                &mut adam.m_rot,
                &mut adam.v_rot,
                self.lr_rot,
                self.beta1,
                self.beta2,
                self.eps,
                self.iteration,
            )?;
        }
        if let Some(grad) = grads.get(gaussians.opacities.as_tensor()) {
            adam_step_var(
                &gaussians.opacities,
                grad,
                &mut adam.m_op,
                &mut adam.v_op,
                self.lr_opacity,
                self.beta1,
                self.beta2,
                self.eps,
                self.iteration,
            )?;
        }
        if let Some(grad) = grads.get(gaussians.colors()) {
            adam_step_var(
                &gaussians.colors,
                grad,
                &mut adam.m_color,
                &mut adam.v_color,
                self.lr_color,
                self.beta1,
                self.beta2,
                self.eps,
                self.iteration,
            )?;
        }

        normalize_rotations(&gaussians.rotations)?;
        Ok(())
    }

    fn render(
        &self,
        gaussians: &TrainableGaussians,
        camera: &DiffCamera,
        should_profile: bool,
    ) -> candle_core::Result<(RenderedFrame, MetalRenderProfile)> {
        if gaussians.len() == 0 {
            return Ok((
                RenderedFrame {
                    color: Tensor::zeros((self.pixel_count, 3), DType::F32, &self.device)?,
                    depth: Tensor::zeros((self.pixel_count,), DType::F32, &self.device)?,
                },
                MetalRenderProfile::default(),
            ));
        }

        let (projected, mut profile) = self.project_gaussians(gaussians, camera, should_profile)?;
        let raster_start = Instant::now();
        let rendered = self.rasterize(&projected)?;
        self.synchronize_if_needed(should_profile)?;
        profile.rasterization = raster_start.elapsed();
        Ok((rendered, profile))
    }

    fn project_gaussians(
        &self,
        gaussians: &TrainableGaussians,
        camera: &DiffCamera,
        should_profile: bool,
    ) -> candle_core::Result<(ProjectedGaussians, MetalRenderProfile)> {
        let mut profile = MetalRenderProfile::default();
        let projection_start = Instant::now();
        let pos = gaussians.positions();
        let scales = gaussians.scales()?;
        let opacity = gaussians.opacities()?;

        let px = pos.narrow(1, 0, 1)?.squeeze(1)?;
        let py = pos.narrow(1, 1, 1)?.squeeze(1)?;
        let pz = pos.narrow(1, 2, 1)?.squeeze(1)?;

        let x = px
            .affine(camera.rotation[0][0] as f64, camera.translation[0] as f64)?
            .broadcast_add(&py.affine(camera.rotation[0][1] as f64, 0.0)?)?
            .broadcast_add(&pz.affine(camera.rotation[0][2] as f64, 0.0)?)?;
        let y = px
            .affine(camera.rotation[1][0] as f64, camera.translation[1] as f64)?
            .broadcast_add(&py.affine(camera.rotation[1][1] as f64, 0.0)?)?
            .broadcast_add(&pz.affine(camera.rotation[1][2] as f64, 0.0)?)?;
        let z = px
            .affine(camera.rotation[2][0] as f64, camera.translation[2] as f64)?
            .broadcast_add(&py.affine(camera.rotation[2][1] as f64, 0.0)?)?
            .broadcast_add(&pz.affine(camera.rotation[2][2] as f64, 0.0)?)?;

        let z_safe = z.clamp(1e-4, f32::MAX)?;
        let fx = Tensor::new(camera.fx, &self.device)?;
        let fy = Tensor::new(camera.fy, &self.device)?;
        let cx = Tensor::new(camera.cx, &self.device)?;
        let cy = Tensor::new(camera.cy, &self.device)?;

        let u = x
            .broadcast_mul(&fx)?
            .broadcast_div(&z_safe)?
            .broadcast_add(&cx)?;
        let v = y
            .broadcast_mul(&fy)?
            .broadcast_div(&z_safe)?
            .broadcast_add(&cy)?;

        let sx = scales
            .narrow(1, 0, 1)?
            .squeeze(1)?
            .broadcast_mul(&fx)?
            .broadcast_div(&z_safe)?
            .clamp(0.5, 256.0)?;
        let sy = scales
            .narrow(1, 1, 1)?
            .squeeze(1)?
            .broadcast_mul(&fy)?
            .broadcast_div(&z_safe)?
            .clamp(0.5, 256.0)?;

        let valid_threshold = Tensor::full(1e-4f32, (gaussians.len(),), &self.device)?;
        let valid = z.ge(&valid_threshold)?.to_dtype(DType::F32)?;
        let opacity = opacity.broadcast_mul(&valid)?;
        self.synchronize_if_needed(should_profile)?;
        profile.projection = projection_start.elapsed();

        let sort_start = Instant::now();
        let sort_idx = z
            .reshape((1, gaussians.len()))?
            .arg_sort_last_dim(true)?
            .squeeze(0)?;
        let projected = ProjectedGaussians {
            u: u.index_select(&sort_idx, 0)?,
            v: v.index_select(&sort_idx, 0)?,
            sigma_x: sx.index_select(&sort_idx, 0)?,
            sigma_y: sy.index_select(&sort_idx, 0)?,
            depth: z.index_select(&sort_idx, 0)?,
            opacity: opacity.index_select(&sort_idx, 0)?,
            colors: gaussians.colors().index_select(&sort_idx, 0)?,
        };
        self.synchronize_if_needed(should_profile)?;
        profile.sorting = sort_start.elapsed();

        Ok((projected, profile))
    }

    fn rasterize(&self, projected: &ProjectedGaussians) -> candle_core::Result<RenderedFrame> {
        let mut color_acc = Tensor::zeros((self.pixel_count, 3), DType::F32, &self.device)?;
        let mut depth_acc = Tensor::zeros((self.pixel_count,), DType::F32, &self.device)?;
        let mut alpha_acc = Tensor::zeros((self.pixel_count,), DType::F32, &self.device)?;
        let mut trans = Tensor::ones((self.pixel_count,), DType::F32, &self.device)?;
        let trans_col = |t: &Tensor| t.reshape((self.pixel_count, 1));

        let total = projected.depth.dim(0)?;
        for start in (0..total).step_by(self.chunk_size) {
            let len = (total - start).min(self.chunk_size);
            let alpha = self.chunk_alpha(
                &projected.u.narrow(0, start, len)?,
                &projected.v.narrow(0, start, len)?,
                &projected.sigma_x.narrow(0, start, len)?,
                &projected.sigma_y.narrow(0, start, len)?,
                &projected.opacity.narrow(0, start, len)?,
            )?;
            let (chunk_color, chunk_depth, chunk_alpha, tail_trans) = self.integrate_chunk(
                &alpha,
                &projected.colors.narrow(0, start, len)?,
                &projected.depth.narrow(0, start, len)?,
            )?;

            color_acc =
                color_acc.broadcast_add(&chunk_color.broadcast_mul(&trans_col(&trans)?)?)?;
            depth_acc = depth_acc.broadcast_add(&chunk_depth.broadcast_mul(&trans)?)?;
            alpha_acc = alpha_acc.broadcast_add(&chunk_alpha.broadcast_mul(&trans)?)?;
            trans = trans.broadcast_mul(&tail_trans)?;
        }

        let denom = alpha_acc.broadcast_add(&Tensor::new(1e-6f32, &self.device)?)?;
        Ok(RenderedFrame {
            color: color_acc.clamp(0.0, 1.0)?,
            depth: depth_acc.broadcast_div(&denom)?,
        })
    }

    fn chunk_alpha(
        &self,
        u: &Tensor,
        v: &Tensor,
        sigma_x: &Tensor,
        sigma_y: &Tensor,
        opacity: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let len = u.dim(0)?;
        let dx = self
            .pixel_x
            .broadcast_sub(&u.reshape((len, 1))?)?
            .broadcast_div(&sigma_x.reshape((len, 1))?)?;
        let dy = self
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
        &self,
        alpha: &Tensor,
        colors: &Tensor,
        depth: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor, Tensor)> {
        let len = alpha.dim(0)?;
        let pixel_count = alpha.dim(1)?;
        let zero_row = Tensor::zeros((1, pixel_count), DType::F32, &self.device)?;
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

    fn synchronize_if_needed(&self, should_profile: bool) -> candle_core::Result<()> {
        if should_profile {
            self.device.synchronize()?;
        }
        Ok(())
    }
}

pub fn train(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<GaussianMap, TrainingError> {
    let start = Instant::now();
    let device = crate::require_metal_device()?;
    let effective_config = effective_metal_config(config);
    let loaded = load_training_data(dataset, &effective_config, &device)?;
    let mut gaussians = trainable_from_map(&loaded.initial_map, &device)?;

    if gaussians.len() == 0 {
        return Err(TrainingError::InvalidInput(
            "training initialization produced zero Gaussians".to_string(),
        ));
    }

    let mut trainer = MetalTrainer::new(
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        config,
        device,
    )?;
    let memory_budget = detect_metal_memory_budget();
    let estimated_peak = estimate_peak_memory(
        gaussians.len(),
        trainer.pixel_count,
        loaded.cameras.len(),
        trainer.chunk_size,
    );
    log::info!(
        "MetalTrainer preflight | gaussians={} | frames={} | pixels={} | chunk_size={} | estimated_peak≈{:.1} GiB | budget={} | dominant={}",
        gaussians.len(),
        loaded.cameras.len(),
        trainer.pixel_count,
        trainer.chunk_size,
        bytes_to_gib(estimated_peak.total_bytes()),
        memory_budget.describe(),
        estimated_peak.top_components_summary(3),
    );
    let skip_memory_guard = std::env::var_os("RUSTGS_SKIP_METAL_MEMORY_GUARD").is_some();
    match assess_memory_estimate(&estimated_peak, &memory_budget) {
        MetalMemoryDecision::Allow => {
            let headroom = memory_budget
                .safe_bytes
                .saturating_sub(estimated_peak.total_bytes());
            log::info!(
                "MetalTrainer preflight passed with {:.1} GiB headroom",
                bytes_to_gib(headroom)
            );
        }
        MetalMemoryDecision::Warn => {
            log::warn!(
                "MetalTrainer preflight is close to the safe memory budget; recommendations: {}",
                estimated_peak.recommendations().join("; ")
            );
        }
        MetalMemoryDecision::Block if skip_memory_guard => {
            log::warn!(
                "MetalTrainer preflight exceeds the safe memory budget but RUSTGS_SKIP_METAL_MEMORY_GUARD=1 is set; continuing anyway. Recommendations: {}",
                estimated_peak.recommendations().join("; ")
            );
        }
        MetalMemoryDecision::Block => {
            return Err(TrainingError::TrainingFailed(format!(
                "metal backend is estimated to need about {:.1} GiB for a single training step, above the safe budget of {}. Dominant terms: {}. Recommendations: {}. Set RUSTGS_SKIP_METAL_MEMORY_GUARD=1 to bypass this guard.",
                bytes_to_gib(estimated_peak.total_bytes()),
                memory_budget.describe(),
                estimated_peak.top_components_summary(3),
                estimated_peak.recommendations().join("; ")
            )));
        }
    }
    let frames = trainer.prepare_frames(&loaded)?;
    let stats = trainer.train(&mut gaussians, &frames, config.iterations)?;
    let trained_map = map_from_trainable(&gaussians)?;

    log::info!(
        "Metal backend complete in {:.2}s | frames={} | render={}x{} | initial_gaussians={} | final_gaussians={} | final_loss={:.6}",
        start.elapsed().as_secs_f64(),
        dataset.poses.len(),
        trainer.render_width,
        trainer.render_height,
        loaded.initial_map.len(),
        trained_map.len(),
        stats.final_loss,
    );

    Ok(trained_map)
}

fn effective_metal_config(config: &TrainingConfig) -> TrainingConfig {
    let mut effective = config.clone();
    let defaults = TrainingConfig::default();
    if effective.max_initial_gaussians == defaults.max_initial_gaussians {
        effective.max_initial_gaussians = DEFAULT_METAL_MAX_INITIAL_GAUSSIANS;
        log::warn!(
            "Metal backend lowered max_initial_gaussians from {} to {} to avoid OOM. Override with --max-initial-gaussians if you want a different budget.",
            defaults.max_initial_gaussians,
            effective.max_initial_gaussians
        );
    }
    effective
}

fn estimate_peak_memory(
    num_gaussians: usize,
    pixel_count: usize,
    frame_count: usize,
    chunk_size: usize,
) -> MetalMemoryEstimate {
    let num_gaussians = num_gaussians as u64;
    let pixel_count = pixel_count as u64;
    let frame_count = frame_count as u64;
    let chunk_size = chunk_size.max(1) as u64;
    let gaussian_state_bytes = num_gaussians.saturating_mul(METAL_GAUSSIAN_STATE_BYTES);
    let frame_bytes = frame_count
        .saturating_mul(pixel_count)
        .saturating_mul(METAL_FRAME_BYTES_PER_PIXEL);
    let pixel_state_bytes = pixel_count.saturating_mul(METAL_PIXEL_STATE_BYTES_PER_PIXEL);
    let projection_bytes = num_gaussians.saturating_mul(METAL_PROJECTED_BYTES_PER_GAUSSIAN);
    let chunk_workspace_bytes = chunk_size
        .saturating_mul(pixel_count)
        .saturating_mul(METAL_CHUNK_WORKSPACE_BYTES_PER_GAUSSIAN_PIXEL);
    let retained_graph_bytes = num_gaussians
        .saturating_mul(pixel_count)
        .saturating_mul(METAL_RETAINED_GRAPH_BYTES_PER_GAUSSIAN_PIXEL);
    let subtotal = gaussian_state_bytes
        .saturating_add(frame_bytes)
        .saturating_add(pixel_state_bytes)
        .saturating_add(projection_bytes)
        .saturating_add(chunk_workspace_bytes)
        .saturating_add(retained_graph_bytes);
    let safety_margin_bytes = apply_ratio(
        subtotal,
        METAL_ESTIMATE_SAFETY_NUMERATOR,
        METAL_ESTIMATE_SAFETY_DENOMINATOR,
    )
    .max(METAL_ESTIMATE_MIN_SAFETY_BYTES);

    MetalMemoryEstimate {
        gaussian_state_bytes,
        frame_bytes,
        pixel_state_bytes,
        projection_bytes,
        chunk_workspace_bytes,
        retained_graph_bytes,
        safety_margin_bytes,
    }
}

fn assess_memory_estimate(
    estimate: &MetalMemoryEstimate,
    budget: &MetalMemoryBudget,
) -> MetalMemoryDecision {
    let total = estimate.total_bytes();
    if total > budget.safe_bytes {
        return MetalMemoryDecision::Block;
    }
    if total.saturating_mul(METAL_WARN_BUDGET_DENOMINATOR)
        >= budget
            .safe_bytes
            .saturating_mul(METAL_WARN_BUDGET_NUMERATOR)
    {
        return MetalMemoryDecision::Warn;
    }
    MetalMemoryDecision::Allow
}

fn detect_metal_memory_budget() -> MetalMemoryBudget {
    let physical_bytes = detect_physical_memory_bytes();
    let safe_bytes = physical_bytes
        .map(|bytes| {
            apply_ratio(
                bytes,
                METAL_SYSTEM_MEMORY_BUDGET_NUMERATOR,
                METAL_SYSTEM_MEMORY_BUDGET_DENOMINATOR,
            )
        })
        .map(|bytes| bytes.min(DEFAULT_METAL_MEMORY_BUDGET_BYTES))
        .unwrap_or(DEFAULT_METAL_MEMORY_BUDGET_BYTES);

    MetalMemoryBudget {
        safe_bytes,
        physical_bytes,
    }
}

fn detect_physical_memory_bytes() -> Option<u64> {
    #[allow(unsafe_code)]
    unsafe {
        let pages = libc::sysconf(libc::_SC_PHYS_PAGES);
        let page_size = libc::sysconf(libc::_SC_PAGESIZE);
        if pages <= 0 || page_size <= 0 {
            return None;
        }
        Some(((pages as u128).saturating_mul(page_size as u128)).min(u64::MAX as u128) as u64)
    }
}

fn apply_ratio(bytes: u64, numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        return bytes;
    }
    ((bytes as u128)
        .saturating_mul(numerator as u128)
        .checked_div(denominator as u128)
        .unwrap_or(u128::MAX))
    .min(u64::MAX as u128) as u64
}

fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / 1024f64 / 1024f64 / 1024f64
}

fn format_memory(bytes: u64) -> String {
    if bytes >= GIB {
        format!("{:.1} GiB", bytes_to_gib(bytes))
    } else {
        format!("{:.0} MiB", bytes as f64 / 1024f64 / 1024f64)
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn should_profile_iteration(profile_steps: bool, profile_interval: usize, iter: usize) -> bool {
    profile_steps && (iter < 5 || iter % profile_interval.max(1) == 0)
}

fn adam_step_var(
    var: &Var,
    grad: &Tensor,
    m: &mut Tensor,
    v: &mut Tensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: usize,
) -> candle_core::Result<()> {
    *m = m
        .affine(beta1 as f64, 0.0)?
        .broadcast_add(&grad.affine((1.0 - beta1) as f64, 0.0)?)?;
    *v = v
        .affine(beta2 as f64, 0.0)?
        .broadcast_add(&grad.sqr()?.affine((1.0 - beta2) as f64, 0.0)?)?;

    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);
    let m_hat = m.affine(1.0 / bc1 as f64, 0.0)?;
    let v_hat = v.affine(1.0 / bc2 as f64, 0.0)?;
    let denom = v_hat
        .sqrt()?
        .broadcast_add(&Tensor::new(eps, var.as_tensor().device())?)?;
    let update = m_hat.broadcast_div(&denom)?.affine(lr as f64, 0.0)?;
    var.set(&var.as_tensor().sub(&update)?)?;
    Ok(())
}

fn normalize_rotations(rotations: &Var) -> candle_core::Result<()> {
    let rot = rotations.as_tensor();
    let norm = rot
        .sqr()?
        .sum(1)?
        .sqrt()?
        .clamp(1e-6, f32::MAX)?
        .unsqueeze(1)?;
    rotations.set(&rot.broadcast_div(&norm)?)?;
    Ok(())
}

fn build_pixel_grids(
    width: usize,
    height: usize,
    device: &Device,
) -> candle_core::Result<(Tensor, Tensor)> {
    let mut xs = Vec::with_capacity(width * height);
    let mut ys = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            xs.push(x as f32 + 0.5);
            ys.push(y as f32 + 0.5);
        }
    }
    Ok((
        Tensor::from_slice(&xs, (1, width * height), device)?,
        Tensor::from_slice(&ys, (1, width * height), device)?,
    ))
}

fn scale_camera(
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

fn scaled_dimensions(width: usize, height: usize, render_scale: f32) -> (usize, usize) {
    let scale = render_scale.clamp(0.0625, 1.0);
    let scaled_width = ((width as f32) * scale).round().max(1.0) as usize;
    let scaled_height = ((height as f32) * scale).round().max(1.0) as usize;
    (scaled_width, scaled_height)
}

fn resize_rgb(
    src: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0f32; dst_width * dst_height * 3];
    for dy in 0..dst_height {
        let sy0 = dy * src_height / dst_height;
        let sy1 = ((dy + 1) * src_height / dst_height)
            .max(sy0 + 1)
            .min(src_height);
        for dx in 0..dst_width {
            let sx0 = dx * src_width / dst_width;
            let sx1 = ((dx + 1) * src_width / dst_width)
                .max(sx0 + 1)
                .min(src_width);
            let mut acc = [0.0f32; 3];
            let mut count = 0usize;
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    let src_idx = (sy * src_width + sx) * 3;
                    acc[0] += src[src_idx];
                    acc[1] += src[src_idx + 1];
                    acc[2] += src[src_idx + 2];
                    count += 1;
                }
            }
            let dst_idx = (dy * dst_width + dx) * 3;
            let inv = 1.0 / count.max(1) as f32;
            dst[dst_idx] = acc[0] * inv;
            dst[dst_idx + 1] = acc[1] * inv;
            dst[dst_idx + 2] = acc[2] * inv;
        }
    }
    dst
}

fn resize_depth(
    src: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0f32; dst_width * dst_height];
    for dy in 0..dst_height {
        let sy0 = dy * src_height / dst_height;
        let sy1 = ((dy + 1) * src_height / dst_height)
            .max(sy0 + 1)
            .min(src_height);
        for dx in 0..dst_width {
            let sx0 = dx * src_width / dst_width;
            let sx1 = ((dx + 1) * src_width / dst_width)
                .max(sx0 + 1)
                .min(src_width);
            let mut acc = 0.0f32;
            let mut count = 0usize;
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    let depth = src[sy * src_width + sx];
                    if depth > 0.0 {
                        acc += depth;
                        count += 1;
                    }
                }
            }
            dst[dy * dst_width + dx] = if count == 0 { 0.0 } else { acc / count as f32 };
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::TrainingBackend;

    #[test]
    fn scaled_dimensions_keep_minimum_size() {
        assert_eq!(scaled_dimensions(640, 480, 0.25), (160, 120));
        assert_eq!(scaled_dimensions(1, 1, 0.0), (1, 1));
    }

    #[test]
    fn resize_depth_ignores_invalid_values() {
        let src = vec![1.0, 0.0, 3.0, 5.0];
        let resized = resize_depth(&src, 2, 2, 1, 1);
        assert_eq!(resized.len(), 1);
        assert!((resized[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn metal_config_uses_safer_default_budget() {
        let mut cfg = TrainingConfig::default();
        cfg.backend = TrainingBackend::Metal;
        let effective = effective_metal_config(&cfg);
        assert_eq!(
            effective.max_initial_gaussians,
            DEFAULT_METAL_MAX_INITIAL_GAUSSIANS
        );
    }

    #[test]
    fn peak_estimate_scales_with_problem_size() {
        let small = estimate_peak_memory(4_096, 4_800, 5, 32);
        let large = estimate_peak_memory(57_474, 4_800, 5, 32);
        assert!(large.total_bytes() > small.total_bytes());
        assert!(bytes_to_gib(large.total_bytes()) > 10.0);
    }

    #[test]
    fn peak_estimate_accounts_for_frames_and_chunk_size() {
        let baseline = estimate_peak_memory(4_096, 4_800, 5, 32);
        let more_frames = estimate_peak_memory(4_096, 4_800, 25, 32);
        let larger_chunk = estimate_peak_memory(4_096, 4_800, 5, 128);
        assert!(more_frames.total_bytes() > baseline.total_bytes());
        assert!(larger_chunk.total_bytes() > baseline.total_bytes());
    }

    #[test]
    fn detected_budget_prefers_fraction_of_system_memory() {
        let physical = 16 * GIB;
        let safe = apply_ratio(
            physical,
            METAL_SYSTEM_MEMORY_BUDGET_NUMERATOR,
            METAL_SYSTEM_MEMORY_BUDGET_DENOMINATOR,
        )
        .min(DEFAULT_METAL_MEMORY_BUDGET_BYTES);
        assert!((bytes_to_gib(safe) - 10.4).abs() < 0.05);
    }

    #[test]
    fn preflight_warns_when_close_to_budget() {
        let estimate = estimate_peak_memory(8_000, 4_800, 5, 32);
        let budget = MetalMemoryBudget {
            safe_bytes: estimate.total_bytes().saturating_add(512 * MIB),
            physical_bytes: Some(16 * GIB),
        };
        assert_eq!(
            assess_memory_estimate(&estimate, &budget),
            MetalMemoryDecision::Warn
        );
    }

    #[test]
    fn preflight_blocks_when_estimate_exceeds_budget() {
        let estimate = estimate_peak_memory(57_474, 4_800, 1, 32);
        let budget = MetalMemoryBudget {
            safe_bytes: 16 * GIB,
            physical_bytes: Some(24 * GIB),
        };
        assert_eq!(
            assess_memory_estimate(&estimate, &budget),
            MetalMemoryDecision::Block
        );
    }

    #[test]
    fn profile_schedule_honors_interval() {
        assert!(should_profile_iteration(true, 25, 0));
        assert!(should_profile_iteration(true, 25, 4));
        assert!(!should_profile_iteration(true, 25, 5));
        assert!(should_profile_iteration(true, 25, 25));
        assert!(!should_profile_iteration(false, 25, 25));
    }
}
