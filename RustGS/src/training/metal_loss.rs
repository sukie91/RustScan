use candle_core::op::BackpropOp;
use candle_core::{
    CpuStorage, CustomOp2, Layout, MetalStorage, Shape, Storage, Tensor,
};

#[derive(Debug, Clone, Copy, Default)]
struct MeanAbsDiff;

pub(crate) fn mean_abs_diff(predicted: &Tensor, target: &Tensor) -> candle_core::Result<Tensor> {
    if predicted.shape() != target.shape() {
        candle_core::bail!(
            "mean_abs_diff shape mismatch: lhs={:?}, rhs={:?}",
            predicted.dims(),
            target.dims()
        );
    }
    predicted.contiguous()?.apply_op2(&target.contiguous()?, MeanAbsDiff)
}

impl CustomOp2 for MeanAbsDiff {
    fn name(&self) -> &'static str {
        "mean-abs-diff"
    }

    fn cpu_fwd(
        &self,
        lhs_storage: &CpuStorage,
        lhs_layout: &Layout,
        rhs_storage: &CpuStorage,
        rhs_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        validate_layouts(self.name(), lhs_layout, rhs_layout)?;
        let lhs = Tensor::from_storage(
            Storage::Cpu(lhs_storage.clone()),
            lhs_layout.shape().clone(),
            BackpropOp::none(),
            false,
        );
        let rhs = Tensor::from_storage(
            Storage::Cpu(rhs_storage.clone()),
            rhs_layout.shape().clone(),
            BackpropOp::none(),
            false,
        );
        extract_cpu_scalar(&forward_loss(lhs, rhs)?)
    }

    fn metal_fwd(
        &self,
        lhs_storage: &MetalStorage,
        lhs_layout: &Layout,
        rhs_storage: &MetalStorage,
        rhs_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        validate_layouts(self.name(), lhs_layout, rhs_layout)?;
        let lhs = Tensor::from_storage(
            Storage::Metal(lhs_storage.clone()),
            lhs_layout.shape().clone(),
            BackpropOp::none(),
            false,
        );
        let rhs = Tensor::from_storage(
            Storage::Metal(rhs_storage.clone()),
            rhs_layout.shape().clone(),
            BackpropOp::none(),
            false,
        );
        extract_metal_scalar(&forward_loss(lhs, rhs)?)
    }

    fn bwd(
        &self,
        arg: &Tensor,
        target: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>)> {
        let dtype = arg.dtype();
        let diff = arg.sub(target)?;
        let positive = diff.gt(0.0f64)?.to_dtype(dtype)?;
        let negative = diff.lt(0.0f64)?.to_dtype(dtype)?;
        let sign = positive.broadcast_sub(&negative)?;
        let grad = sign
            .broadcast_mul(&grad_res.to_dtype(dtype)?)?
            .affine(1.0 / arg.elem_count() as f64, 0.0)?;
        Ok((Some(grad), None))
    }
}

fn validate_layouts(name: &str, lhs: &Layout, rhs: &Layout) -> candle_core::Result<()> {
    if lhs.shape() != rhs.shape() {
        candle_core::bail!(
            "{name} shape mismatch: lhs={:?}, rhs={:?}",
            lhs.shape().dims(),
            rhs.shape().dims()
        );
    }
    ensure_simple_layout(name, lhs)?;
    ensure_simple_layout(name, rhs)?;
    Ok(())
}

fn ensure_simple_layout(name: &str, layout: &Layout) -> candle_core::Result<()> {
    if !layout.is_contiguous() || layout.start_offset() != 0 {
        candle_core::bail!(
            "{name} expects contiguous zero-offset tensors, got contiguous={} offset={}",
            layout.is_contiguous(),
            layout.start_offset()
        );
    }
    Ok(())
}

fn forward_loss(lhs: Tensor, rhs: Tensor) -> candle_core::Result<Tensor> {
    lhs.sub(&rhs)?.abs()?.mean_all()
}

fn extract_cpu_scalar(tensor: &Tensor) -> candle_core::Result<(CpuStorage, Shape)> {
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Cpu(storage) => Ok((storage.clone(), layout.shape().clone())),
        other => candle_core::bail!("expected cpu scalar storage, got {:?}", other.device()),
    }
}

fn extract_metal_scalar(tensor: &Tensor) -> candle_core::Result<(MetalStorage, Shape)> {
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Metal(storage) => Ok((storage.clone(), layout.shape().clone())),
        other => candle_core::bail!("expected metal scalar storage, got {:?}", other.device()),
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor, Var};

    use super::mean_abs_diff;

    fn assert_close_scalar(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}, tol={tolerance}"
        );
    }

    fn assert_close_vec(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "idx={idx} expected {expected}, got {actual}, tol={tolerance}"
            );
        }
    }

    fn run_loss_parity_case(device: &Device, tolerance: f32) -> candle_core::Result<()> {
        let target = Tensor::from_vec(vec![0.2f32, -0.4, 0.9, -1.3], (2, 2), device)?;

        let baseline_pred = Var::from_tensor(&Tensor::from_vec(
            vec![0.5f32, -0.8, 0.3, -0.7],
            (2, 2),
            device,
        )?)?;
        let custom_pred = Var::from_tensor(&Tensor::from_vec(
            vec![0.5f32, -0.8, 0.3, -0.7],
            (2, 2),
            device,
        )?)?;

        let baseline = baseline_pred.sub(&target)?.abs()?.mean_all()?;
        let baseline_value = baseline.to_vec0::<f32>()?;
        let baseline_grads = baseline.backward()?;
        let baseline_grad = baseline_grads
            .get(&baseline_pred)
            .expect("baseline grad available")
            .flatten_all()?
            .to_vec1::<f32>()?;

        let custom = mean_abs_diff(custom_pred.as_tensor(), &target)?;
        let custom_value = custom.to_vec0::<f32>()?;
        let custom_grads = custom.backward()?;
        let custom_grad = custom_grads
            .get(&custom_pred)
            .expect("custom grad available")
            .flatten_all()?
            .to_vec1::<f32>()?;

        assert_close_scalar(custom_value, baseline_value, tolerance);
        assert_close_vec(&custom_grad, &baseline_grad, tolerance);
        Ok(())
    }

    #[test]
    fn mean_abs_diff_matches_baseline_on_cpu() -> candle_core::Result<()> {
        run_loss_parity_case(&Device::Cpu, 1e-6)
    }

    #[test]
    fn mean_abs_diff_matches_baseline_on_metal() -> candle_core::Result<()> {
        if !candle_core::utils::metal_is_available() {
            return Ok(());
        }
        let device = Device::new_metal(0)?;
        run_loss_parity_case(&device, 1e-5)
    }
}
