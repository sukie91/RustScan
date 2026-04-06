use candle_core::{Device, Tensor};

use crate::diff::diff_splat::TrainableGaussians;

#[derive(Debug)]
pub(super) struct MetalAdamState {
    pub(super) m_pos: Tensor,
    pub(super) v_pos: Tensor,
    pub(super) m_scale: Tensor,
    pub(super) v_scale: Tensor,
    pub(super) m_rot: Tensor,
    pub(super) v_rot: Tensor,
    pub(super) m_op: Tensor,
    pub(super) v_op: Tensor,
    pub(super) m_color: Tensor,
    pub(super) v_color: Tensor,
    pub(super) m_sh_rest: Tensor,
    pub(super) v_sh_rest: Tensor,
}

impl MetalAdamState {
    pub(super) fn new(gaussians: &TrainableGaussians) -> candle_core::Result<Self> {
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

    pub(super) fn rebuild(
        &self,
        device: &Device,
        origins: &[Option<usize>],
    ) -> candle_core::Result<Self> {
        let row_count = origins.len();
        let m_pos = Tensor::from_slice(
            &gather_rows(&flatten_rows(self.m_pos.to_vec2::<f32>()?), 3, origins),
            (row_count, 3),
            device,
        )?;
        let v_pos = Tensor::from_slice(
            &gather_rows(&flatten_rows(self.v_pos.to_vec2::<f32>()?), 3, origins),
            (row_count, 3),
            device,
        )?;
        let m_scale = Tensor::from_slice(
            &gather_rows(&flatten_rows(self.m_scale.to_vec2::<f32>()?), 3, origins),
            (row_count, 3),
            device,
        )?;
        let v_scale = Tensor::from_slice(
            &gather_rows(&flatten_rows(self.v_scale.to_vec2::<f32>()?), 3, origins),
            (row_count, 3),
            device,
        )?;
        let m_rot = Tensor::from_slice(
            &gather_rows(&flatten_rows(self.m_rot.to_vec2::<f32>()?), 4, origins),
            (row_count, 4),
            device,
        )?;
        let v_rot = Tensor::from_slice(
            &gather_rows(&flatten_rows(self.v_rot.to_vec2::<f32>()?), 4, origins),
            (row_count, 4),
            device,
        )?;
        let m_op = Tensor::from_slice(
            &gather_rows(&self.m_op.to_vec1::<f32>()?, 1, origins),
            row_count,
            device,
        )?;
        let v_op = Tensor::from_slice(
            &gather_rows(&self.v_op.to_vec1::<f32>()?, 1, origins),
            row_count,
            device,
        )?;
        let m_color = Tensor::from_slice(
            &gather_rows(&flatten_rows(self.m_color.to_vec2::<f32>()?), 3, origins),
            (row_count, 3),
            device,
        )?;
        let v_color = Tensor::from_slice(
            &gather_rows(&flatten_rows(self.v_color.to_vec2::<f32>()?), 3, origins),
            (row_count, 3),
            device,
        )?;
        let sh_rest_dims = self.m_sh_rest.dims();
        let sh_rest_coeff_count = sh_rest_dims.get(1).copied().unwrap_or(0);
        let sh_rest_shape = (row_count, sh_rest_coeff_count, 3usize);
        let m_sh_rest = Tensor::from_slice(
            &gather_rows(
                &flatten_3d(self.m_sh_rest.to_vec3::<f32>()?),
                sh_rest_coeff_count.saturating_mul(3),
                origins,
            ),
            sh_rest_shape,
            device,
        )?;
        let v_sh_rest = Tensor::from_slice(
            &gather_rows(
                &flatten_3d(self.v_sh_rest.to_vec3::<f32>()?),
                sh_rest_coeff_count.saturating_mul(3),
                origins,
            ),
            sh_rest_shape,
            device,
        )?;

        Ok(Self {
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
}

fn flatten_rows(rows: Vec<Vec<f32>>) -> Vec<f32> {
    rows.into_iter().flatten().collect()
}

fn flatten_3d(rows: Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    rows.into_iter().flatten().flatten().collect()
}

fn gather_rows(source: &[f32], row_width: usize, origins: &[Option<usize>]) -> Vec<f32> {
    let mut gathered = Vec::with_capacity(origins.len() * row_width);
    for origin in origins {
        match origin {
            Some(idx) => {
                let start = idx.saturating_mul(row_width);
                let end = start.saturating_add(row_width).min(source.len());
                gathered.extend_from_slice(&source[start..end]);
                if end - start < row_width {
                    gathered.resize(gathered.len() + (row_width - (end - start)), 0.0);
                }
            }
            None => gathered.resize(gathered.len() + row_width, 0.0),
        }
    }
    gathered
}

#[cfg(test)]
mod tests {
    use super::MetalAdamState;
    use crate::diff::diff_splat::TrainableGaussians;
    use candle_core::{Device, Tensor};

    #[test]
    fn rebuild_preserves_pruned_and_reordered_rows() {
        let device = Device::Cpu;
        let gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 0.0, 3.0],
            &[0.0; 9],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &device,
        )
        .unwrap();
        let mut adam = MetalAdamState::new(&gaussians).unwrap();
        adam.m_pos = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            (3, 3),
            &device,
        )
        .unwrap();
        adam.v_pos = Tensor::from_slice(
            &[10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            (3, 3),
            &device,
        )
        .unwrap();

        let rebuilt = adam.rebuild(&device, &[Some(2), Some(0)]).unwrap();

        assert_eq!(
            rebuilt.m_pos.to_vec2::<f32>().unwrap(),
            vec![vec![7.0, 8.0, 9.0], vec![1.0, 2.0, 3.0]]
        );
        assert_eq!(
            rebuilt.v_pos.to_vec2::<f32>().unwrap(),
            vec![vec![16.0, 17.0, 18.0], vec![10.0, 11.0, 12.0]]
        );
    }

    #[test]
    fn rebuild_zero_initializes_split_rows() {
        let device = Device::Cpu;
        let gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 1.0],
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0],
            &[1.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        let mut adam = MetalAdamState::new(&gaussians).unwrap();
        adam.m_pos = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (1, 3), &device).unwrap();
        adam.v_pos = Tensor::from_slice(&[4.0f32, 5.0, 6.0], (1, 3), &device).unwrap();

        let rebuilt = adam.rebuild(&device, &[Some(0), None]).unwrap();

        assert_eq!(
            rebuilt.m_pos.to_vec2::<f32>().unwrap(),
            vec![vec![1.0, 2.0, 3.0], vec![0.0, 0.0, 0.0]]
        );
        assert_eq!(
            rebuilt.v_pos.to_vec2::<f32>().unwrap(),
            vec![vec![4.0, 5.0, 6.0], vec![0.0, 0.0, 0.0]]
        );
    }
}
