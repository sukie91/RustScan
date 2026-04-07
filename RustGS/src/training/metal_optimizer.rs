use candle_core::{Device, Tensor, Var};

use crate::diff::diff_splat::TrainableGaussians;

use super::metal_backward::MetalParameterGrads;
use super::metal_runtime::{MetalBufferSlot, MetalRuntime};

pub(crate) struct MetalAdamState {
    pub(crate) m_pos: Tensor,
    pub(crate) v_pos: Tensor,
    pub(crate) m_scale: Tensor,
    pub(crate) v_scale: Tensor,
    pub(crate) m_rot: Tensor,
    pub(crate) v_rot: Tensor,
    pub(crate) m_op: Tensor,
    pub(crate) v_op: Tensor,
    pub(crate) m_color: Tensor,
    pub(crate) v_color: Tensor,
    pub(crate) m_sh_rest: Tensor,
    pub(crate) v_sh_rest: Tensor,
}

impl MetalAdamState {
    pub(crate) fn new(gaussians: &TrainableGaussians) -> candle_core::Result<Self> {
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
            m_sh_rest: gaussians.sh_rest().zeros_like()?,
            v_sh_rest: gaussians.sh_rest().zeros_like()?,
        })
    }

    pub(crate) fn reset_moments(&mut self, only_opacity: bool) -> candle_core::Result<()> {
        self.m_op = Tensor::zeros_like(&self.m_op)?;
        self.v_op = Tensor::zeros_like(&self.v_op)?;
        if only_opacity {
            return Ok(());
        }

        self.m_pos = Tensor::zeros_like(&self.m_pos)?;
        self.v_pos = Tensor::zeros_like(&self.v_pos)?;
        self.m_scale = Tensor::zeros_like(&self.m_scale)?;
        self.v_scale = Tensor::zeros_like(&self.v_scale)?;
        self.m_rot = Tensor::zeros_like(&self.m_rot)?;
        self.v_rot = Tensor::zeros_like(&self.v_rot)?;
        self.m_color = Tensor::zeros_like(&self.m_color)?;
        self.v_color = Tensor::zeros_like(&self.v_color)?;
        self.m_sh_rest = Tensor::zeros_like(&self.m_sh_rest)?;
        self.v_sh_rest = Tensor::zeros_like(&self.v_sh_rest)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalOptimizerConfig {
    pub(crate) effective_lr_pos: f32,
    pub(crate) lr_scale: f32,
    pub(crate) lr_rotation: f32,
    pub(crate) lr_opacity: f32,
    pub(crate) lr_color: f32,
    pub(crate) lr_sh_rest: f32,
    pub(crate) beta1: f32,
    pub(crate) beta2: f32,
    pub(crate) eps: f32,
    pub(crate) step: usize,
    pub(crate) use_sparse_updates: bool,
}

pub(crate) fn apply_optimizer_step(
    gaussians: &mut TrainableGaussians,
    runtime: &mut MetalRuntime,
    adam: &mut MetalAdamState,
    grads: &MetalParameterGrads,
    visible_row_indices: Option<&Tensor>,
    config: MetalOptimizerConfig,
) -> candle_core::Result<()> {
    let sparse_row_indices = if config.use_sparse_updates {
        Some(visible_row_indices.ok_or_else(|| {
            candle_core::Error::Msg("sparse optimizer step missing visible row indices".into())
        })?)
    } else {
        None
    };
    if let Some(row_indices) = sparse_row_indices {
        if row_indices.dim(0)? == 0 {
            return Ok(());
        }
    }

    if let Some(row_indices) = sparse_row_indices {
        adam_step_var_sparse(
            &gaussians.positions,
            &grads.positions,
            &mut adam.m_pos,
            &mut adam.v_pos,
            row_indices,
            config.effective_lr_pos,
            config.beta1,
            config.beta2,
            config.eps,
            config.step,
        )?;
        adam_step_var_sparse(
            &gaussians.scales,
            &grads.log_scales,
            &mut adam.m_scale,
            &mut adam.v_scale,
            row_indices,
            config.lr_scale,
            config.beta1,
            config.beta2,
            config.eps,
            config.step,
        )?;
    } else {
        adam_step_var_fused(
            &gaussians.positions,
            &grads.positions,
            &mut adam.m_pos,
            &mut adam.v_pos,
            runtime,
            config.effective_lr_pos,
            config.beta1,
            config.beta2,
            config.eps,
            config.step,
            MetalBufferSlot::AdamGradPos,
            MetalBufferSlot::AdamMPos,
            MetalBufferSlot::AdamVPos,
            MetalBufferSlot::AdamParamPos,
        )?;
        adam_step_var_fused(
            &gaussians.scales,
            &grads.log_scales,
            &mut adam.m_scale,
            &mut adam.v_scale,
            runtime,
            config.lr_scale,
            config.beta1,
            config.beta2,
            config.eps,
            config.step,
            MetalBufferSlot::AdamGradScale,
            MetalBufferSlot::AdamMScale,
            MetalBufferSlot::AdamVScale,
            MetalBufferSlot::AdamParamScale,
        )?;
    }

    if config.lr_rotation > 0.0 {
        let Some(rotation_grads) = grads.rotations.as_ref() else {
            return apply_non_rotation_step(
                gaussians,
                runtime,
                adam,
                grads,
                sparse_row_indices,
                config,
            );
        };
        if let Some(row_indices) = sparse_row_indices {
            adam_step_var_sparse(
                &gaussians.rotations,
                rotation_grads,
                &mut adam.m_rot,
                &mut adam.v_rot,
                row_indices,
                config.lr_rotation,
                config.beta1,
                config.beta2,
                config.eps,
                config.step,
            )?;
        } else {
            adam_step_var(
                &gaussians.rotations,
                rotation_grads,
                &mut adam.m_rot,
                &mut adam.v_rot,
                config.lr_rotation,
                config.beta1,
                config.beta2,
                config.eps,
                config.step,
            )?;
        }
    }

    apply_non_rotation_step(gaussians, runtime, adam, grads, sparse_row_indices, config)
}

fn apply_non_rotation_step(
    gaussians: &mut TrainableGaussians,
    runtime: &mut MetalRuntime,
    adam: &mut MetalAdamState,
    grads: &MetalParameterGrads,
    sparse_row_indices: Option<&Tensor>,
    config: MetalOptimizerConfig,
) -> candle_core::Result<()> {
    if let Some(row_indices) = sparse_row_indices {
        adam_step_var_sparse(
            &gaussians.opacities,
            &grads.opacity_logits,
            &mut adam.m_op,
            &mut adam.v_op,
            row_indices,
            config.lr_opacity,
            config.beta1,
            config.beta2,
            config.eps,
            config.step,
        )?;
        adam_step_var_sparse(
            &gaussians.colors,
            &grads.colors,
            &mut adam.m_color,
            &mut adam.v_color,
            row_indices,
            config.lr_color,
            config.beta1,
            config.beta2,
            config.eps,
            config.step,
        )?;
    } else {
        adam_step_var_fused(
            &gaussians.opacities,
            &grads.opacity_logits,
            &mut adam.m_op,
            &mut adam.v_op,
            runtime,
            config.lr_opacity,
            config.beta1,
            config.beta2,
            config.eps,
            config.step,
            MetalBufferSlot::AdamGradOpacity,
            MetalBufferSlot::AdamMOpacity,
            MetalBufferSlot::AdamVOpacity,
            MetalBufferSlot::AdamParamOpacity,
        )?;
        adam_step_var_fused(
            &gaussians.colors,
            &grads.colors,
            &mut adam.m_color,
            &mut adam.v_color,
            runtime,
            config.lr_color,
            config.beta1,
            config.beta2,
            config.eps,
            config.step,
            MetalBufferSlot::AdamGradColor,
            MetalBufferSlot::AdamMColor,
            MetalBufferSlot::AdamVColor,
            MetalBufferSlot::AdamParamColor,
        )?;
    }

    if gaussians.uses_spherical_harmonics() {
        if let Some(row_indices) = sparse_row_indices {
            adam_step_var_sparse(
                &gaussians.sh_rest,
                &grads.sh_rest,
                &mut adam.m_sh_rest,
                &mut adam.v_sh_rest,
                row_indices,
                config.lr_sh_rest,
                config.beta1,
                config.beta2,
                config.eps,
                config.step,
            )?;
        } else {
            adam_step_var_fused(
                &gaussians.sh_rest,
                &grads.sh_rest,
                &mut adam.m_sh_rest,
                &mut adam.v_sh_rest,
                runtime,
                config.lr_sh_rest,
                config.beta1,
                config.beta2,
                config.eps,
                config.step,
                MetalBufferSlot::AdamGradColor,
                MetalBufferSlot::AdamMColor,
                MetalBufferSlot::AdamVColor,
                MetalBufferSlot::AdamParamColor,
            )?;
        }
    }

    Ok(())
}

pub(crate) fn rebuild_adam_state(
    device: &Device,
    old_state: &MetalAdamState,
    origins: &[Option<usize>],
) -> candle_core::Result<MetalAdamState> {
    let row_count = origins.len();
    let m_pos = Tensor::from_slice(
        &gather_rows(&flatten_rows(old_state.m_pos.to_vec2::<f32>()?), 3, origins),
        (row_count, 3),
        device,
    )?;
    let v_pos = Tensor::from_slice(
        &gather_rows(&flatten_rows(old_state.v_pos.to_vec2::<f32>()?), 3, origins),
        (row_count, 3),
        device,
    )?;
    let m_scale = Tensor::from_slice(
        &gather_rows(
            &flatten_rows(old_state.m_scale.to_vec2::<f32>()?),
            3,
            origins,
        ),
        (row_count, 3),
        device,
    )?;
    let v_scale = Tensor::from_slice(
        &gather_rows(
            &flatten_rows(old_state.v_scale.to_vec2::<f32>()?),
            3,
            origins,
        ),
        (row_count, 3),
        device,
    )?;
    let m_rot = Tensor::from_slice(
        &gather_rows(&flatten_rows(old_state.m_rot.to_vec2::<f32>()?), 4, origins),
        (row_count, 4),
        device,
    )?;
    let v_rot = Tensor::from_slice(
        &gather_rows(&flatten_rows(old_state.v_rot.to_vec2::<f32>()?), 4, origins),
        (row_count, 4),
        device,
    )?;
    let m_op = Tensor::from_slice(
        &gather_rows(&old_state.m_op.to_vec1::<f32>()?, 1, origins),
        row_count,
        device,
    )?;
    let v_op = Tensor::from_slice(
        &gather_rows(&old_state.v_op.to_vec1::<f32>()?, 1, origins),
        row_count,
        device,
    )?;
    let m_color = Tensor::from_slice(
        &gather_rows(
            &flatten_rows(old_state.m_color.to_vec2::<f32>()?),
            3,
            origins,
        ),
        (row_count, 3),
        device,
    )?;
    let v_color = Tensor::from_slice(
        &gather_rows(
            &flatten_rows(old_state.v_color.to_vec2::<f32>()?),
            3,
            origins,
        ),
        (row_count, 3),
        device,
    )?;
    let sh_rest_coeff_count = old_state.m_sh_rest.dims().get(1).copied().unwrap_or(0);
    let sh_rest_shape = (row_count, sh_rest_coeff_count, 3usize);
    let m_sh_rest = Tensor::from_slice(
        &gather_rows(
            &flatten_3d(old_state.m_sh_rest.to_vec3::<f32>()?),
            sh_rest_coeff_count.saturating_mul(3),
            origins,
        ),
        sh_rest_shape,
        device,
    )?;
    let v_sh_rest = Tensor::from_slice(
        &gather_rows(
            &flatten_3d(old_state.v_sh_rest.to_vec3::<f32>()?),
            sh_rest_coeff_count.saturating_mul(3),
            origins,
        ),
        sh_rest_shape,
        device,
    )?;

    Ok(MetalAdamState {
        m_pos,
        v_pos,
        m_scale,
        v_scale,
        m_rot,
        v_rot,
        m_op,
        v_op,
        m_color,
        v_color,
        m_sh_rest,
        v_sh_rest,
    })
}

pub(crate) fn adam_step_var_fused(
    var: &Var,
    grad: &Tensor,
    m: &mut Tensor,
    v: &mut Tensor,
    runtime: &mut MetalRuntime,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: usize,
    grad_slot: MetalBufferSlot,
    m_slot: MetalBufferSlot,
    v_slot: MetalBufferSlot,
    param_slot: MetalBufferSlot,
) -> candle_core::Result<()> {
    if !var.as_tensor().device().is_metal() {
        return adam_step_var(var, grad, m, v, lr, beta1, beta2, eps, step);
    }

    let element_count = grad.elem_count();
    let grad_flat = runtime.read_tensor_flat::<f32>(grad)?;
    let m_flat = runtime.read_tensor_flat::<f32>(m)?;
    let v_flat = runtime.read_tensor_flat::<f32>(v)?;
    let param_flat = runtime.read_tensor_flat::<f32>(var.as_tensor())?;

    let shape = grad.shape().clone();
    runtime.ensure_buffer(grad_slot, element_count * std::mem::size_of::<f32>())?;
    runtime.ensure_buffer(m_slot, element_count * std::mem::size_of::<f32>())?;
    runtime.ensure_buffer(v_slot, element_count * std::mem::size_of::<f32>())?;
    runtime.ensure_buffer(param_slot, element_count * std::mem::size_of::<f32>())?;
    runtime.write_slice(grad_slot, &grad_flat)?;
    runtime.write_slice(m_slot, &m_flat)?;
    runtime.write_slice(v_slot, &v_flat)?;
    runtime.write_slice(param_slot, &param_flat)?;

    runtime.adam_step_fused(
        param_slot,
        grad_slot,
        m_slot,
        v_slot,
        element_count,
        lr,
        beta1,
        beta2,
        eps,
        step,
    )?;

    runtime.device.synchronize()?;
    let new_m_flat = runtime.read_buffer_structs::<f32>(m_slot, element_count)?;
    let new_v_flat = runtime.read_buffer_structs::<f32>(v_slot, element_count)?;
    let new_param_flat = runtime.read_buffer_structs::<f32>(param_slot, element_count)?;

    *m = Tensor::from_slice(&new_m_flat, shape.clone(), var.as_tensor().device())?;
    *v = Tensor::from_slice(&new_v_flat, shape.clone(), var.as_tensor().device())?;
    var.set(&Tensor::from_slice(
        &new_param_flat,
        shape,
        var.as_tensor().device(),
    )?)?;
    Ok(())
}

pub(crate) fn adam_step_var(
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
    let (new_param, new_m, new_v) =
        adam_updated_tensors(var.as_tensor(), grad, m, v, lr, beta1, beta2, eps, step)?;
    *m = new_m;
    *v = new_v;
    var.set(&new_param)?;
    Ok(())
}

pub(crate) fn adam_step_var_sparse(
    var: &Var,
    grad: &Tensor,
    m: &mut Tensor,
    v: &mut Tensor,
    row_indices: &Tensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: usize,
) -> candle_core::Result<()> {
    let grad_rows = grad.index_select(row_indices, 0)?;
    if grad_rows.dim(0)? == 0 {
        return Ok(());
    }

    let m_rows = m.index_select(row_indices, 0)?;
    let v_rows = v.index_select(row_indices, 0)?;
    let param_rows = var.as_tensor().index_select(row_indices, 0)?;
    let (new_param_rows, new_m_rows, new_v_rows) = adam_updated_tensors(
        &param_rows,
        &grad_rows,
        &m_rows,
        &v_rows,
        lr,
        beta1,
        beta2,
        eps,
        step,
    )?;

    *m = m.index_add(row_indices, &new_m_rows.broadcast_sub(&m_rows)?, 0)?;
    *v = v.index_add(row_indices, &new_v_rows.broadcast_sub(&v_rows)?, 0)?;
    let updated_params =
        var.as_tensor()
            .index_add(row_indices, &new_param_rows.broadcast_sub(&param_rows)?, 0)?;
    var.set(&updated_params)?;
    Ok(())
}

pub(crate) fn adam_updated_tensors(
    param: &Tensor,
    grad: &Tensor,
    m: &Tensor,
    v: &Tensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: usize,
) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
    let new_m = m
        .affine(beta1 as f64, 0.0)?
        .broadcast_add(&grad.affine((1.0 - beta1) as f64, 0.0)?)?;
    let new_v = v
        .affine(beta2 as f64, 0.0)?
        .broadcast_add(&grad.sqr()?.affine((1.0 - beta2) as f64, 0.0)?)?;

    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);
    let m_hat = new_m.affine(1.0 / bc1 as f64, 0.0)?;
    let v_hat = new_v.affine(1.0 / bc2 as f64, 0.0)?;
    let denom = v_hat
        .sqrt()?
        .broadcast_add(&Tensor::new(eps, param.device())?)?;
    let update = m_hat.broadcast_div(&denom)?.affine(lr as f64, 0.0)?;
    let new_param = param.sub(&update)?;
    Ok((new_param, new_m, new_v))
}

fn flatten_rows(rows: Vec<Vec<f32>>) -> Vec<f32> {
    rows.into_iter().flatten().collect()
}

fn flatten_3d(rows: Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    rows.into_iter().flatten().flatten().collect()
}

fn gather_rows(source: &[f32], row_width: usize, origins: &[Option<usize>]) -> Vec<f32> {
    let mut gathered = Vec::with_capacity(origins.len().saturating_mul(row_width));
    for origin in origins {
        match origin {
            Some(index) => {
                let start = index.saturating_mul(row_width);
                let end = start.saturating_add(row_width).min(source.len());
                gathered.extend_from_slice(&source[start..end]);
                if end.saturating_sub(start) < row_width {
                    gathered.resize(gathered.len() + (row_width - (end - start)), 0.0);
                }
            }
            None => gathered.resize(gathered.len() + row_width, 0.0),
        }
    }
    gathered
}
