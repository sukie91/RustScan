use candle_core::Device;

use crate::diff::diff_splat::{TrainableColorRepresentation, TrainableGaussians};

#[derive(Debug, Clone)]
pub(super) struct SplatParameterView {
    pub(super) positions: Vec<f32>,
    pub(super) log_scales: Vec<f32>,
    pub(super) rotations: Vec<f32>,
    pub(super) opacity_logits: Vec<f32>,
    pub(super) colors: Vec<f32>,
    pub(super) sh_rest: Vec<f32>,
    pub(super) color_representation: TrainableColorRepresentation,
}

impl SplatParameterView {
    pub(super) fn from_trainable(gaussians: &TrainableGaussians) -> candle_core::Result<Self> {
        let view = Self {
            positions: flatten_rows(gaussians.positions().to_vec2::<f32>()?),
            log_scales: flatten_rows(gaussians.scales.as_tensor().to_vec2::<f32>()?),
            rotations: flatten_rows(gaussians.rotations.as_tensor().to_vec2::<f32>()?),
            opacity_logits: gaussians.opacities.as_tensor().to_vec1::<f32>()?,
            colors: flatten_rows(gaussians.colors().to_vec2::<f32>()?),
            sh_rest: flatten_3d(gaussians.sh_rest().to_vec3::<f32>()?),
            color_representation: gaussians.color_representation(),
        };
        view.validate()?;
        Ok(view)
    }

    pub(super) fn validate(&self) -> candle_core::Result<()> {
        let row_count = self.opacity_logits.len();
        validate_component_len("positions", self.positions.len(), row_count, 3)?;
        validate_component_len("log_scales", self.log_scales.len(), row_count, 3)?;
        validate_component_len("rotations", self.rotations.len(), row_count, 4)?;
        validate_component_len("colors", self.colors.len(), row_count, 3)?;
        validate_component_len(
            "sh_rest",
            self.sh_rest.len(),
            row_count,
            self.sh_rest_row_width(),
        )?;
        Ok(())
    }

    pub(super) fn len(&self) -> usize {
        self.opacity_logits.len()
    }

    pub(super) fn position(&self, idx: usize) -> [f32; 3] {
        let base = idx * 3;
        [
            self.positions[base],
            self.positions[base + 1],
            self.positions[base + 2],
        ]
    }

    pub(super) fn log_scale(&self, idx: usize) -> [f32; 3] {
        let base = idx * 3;
        [
            self.log_scales[base],
            self.log_scales[base + 1],
            self.log_scales[base + 2],
        ]
    }

    pub(super) fn rotation(&self, idx: usize) -> [f32; 4] {
        let base = idx * 4;
        [
            self.rotations[base],
            self.rotations[base + 1],
            self.rotations[base + 2],
            self.rotations[base + 3],
        ]
    }

    pub(super) fn color(&self, idx: usize) -> [f32; 3] {
        let base = idx * 3;
        [
            self.colors[base],
            self.colors[base + 1],
            self.colors[base + 2],
        ]
    }

    pub(super) fn sh_rest_row_width(&self) -> usize {
        self.color_representation.sh_rest_coeff_count() * 3
    }

    pub(super) fn sh_rest(&self, idx: usize) -> &[f32] {
        row_slice(&self.sh_rest, self.sh_rest_row_width(), idx)
    }

    pub(super) fn scale(&self, idx: usize) -> [f32; 3] {
        let log = self.log_scale(idx);
        [log[0].exp(), log[1].exp(), log[2].exp()]
    }

    pub(super) fn push(
        &mut self,
        position: [f32; 3],
        log_scale: [f32; 3],
        rotation: [f32; 4],
        opacity_logit: f32,
        color: [f32; 3],
        sh_rest: &[f32],
    ) {
        self.positions.extend_from_slice(&position);
        self.log_scales.extend_from_slice(&log_scale);
        self.rotations.extend_from_slice(&rotation);
        self.opacity_logits.push(opacity_logit);
        self.colors.extend_from_slice(&color);
        let sh_rest_row_width = self.sh_rest_row_width();
        if sh_rest_row_width > 0 {
            let copied = sh_rest.len().min(sh_rest_row_width);
            self.sh_rest.extend_from_slice(&sh_rest[..copied]);
            if copied < sh_rest_row_width {
                self.sh_rest
                    .resize(self.sh_rest.len() + (sh_rest_row_width - copied), 0.0);
            }
        }
    }

    pub(super) fn retained_view(&self, row_count: usize) -> Self {
        Self {
            positions: Vec::with_capacity(row_count * 3),
            log_scales: Vec::with_capacity(row_count * 3),
            rotations: Vec::with_capacity(row_count * 4),
            opacity_logits: Vec::with_capacity(row_count),
            colors: Vec::with_capacity(row_count * 3),
            sh_rest: Vec::with_capacity(row_count * self.sh_rest_row_width()),
            color_representation: self.color_representation,
        }
    }

    pub(super) fn to_trainable(&self, device: &Device) -> candle_core::Result<TrainableGaussians> {
        self.validate()?;
        match self.color_representation {
            TrainableColorRepresentation::Rgb => TrainableGaussians::new(
                &self.positions,
                &self.log_scales,
                &self.rotations,
                &self.opacity_logits,
                &self.colors,
                device,
            ),
            TrainableColorRepresentation::SphericalHarmonics { degree } => {
                TrainableGaussians::new_with_sh(
                    &self.positions,
                    &self.log_scales,
                    &self.rotations,
                    &self.opacity_logits,
                    &self.colors,
                    &self.sh_rest,
                    degree,
                    device,
                )
            }
        }
    }
}

pub(super) fn row_slice(values: &[f32], width: usize, idx: usize) -> &[f32] {
    let start = idx.saturating_mul(width);
    let end = start.saturating_add(width);
    values.get(start..end).unwrap_or(&[])
}

fn validate_component_len(
    name: &str,
    actual: usize,
    row_count: usize,
    row_width: usize,
) -> candle_core::Result<()> {
    let expected = row_count.saturating_mul(row_width);
    if actual != expected {
        candle_core::bail!(
            "splat parameter view invariant violated: {name} expected {expected} values for {row_count} gaussians, got {actual}"
        );
    }
    Ok(())
}

fn flatten_rows(rows: Vec<Vec<f32>>) -> Vec<f32> {
    rows.into_iter().flatten().collect()
}

fn flatten_3d(rows: Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    rows.into_iter().flatten().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::SplatParameterView;
    use crate::diff::diff_splat::{
        rgb_to_sh0_value, TrainableColorRepresentation, TrainableGaussians,
    };
    use candle_core::Device;

    #[test]
    fn validation_rejects_mismatched_component_lengths() {
        let invalid = SplatParameterView {
            positions: vec![0.0; 5],
            log_scales: vec![0.0; 6],
            rotations: vec![0.0; 8],
            opacity_logits: vec![0.0; 2],
            colors: vec![0.0; 6],
            sh_rest: Vec::new(),
            color_representation: TrainableColorRepresentation::Rgb,
        };

        let err = invalid.validate().unwrap_err().to_string();
        assert!(err.contains("positions expected 6 values"));
    }

    #[test]
    fn rgb_view_round_trips_trainables() {
        let device = Device::Cpu;
        let gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 1.0, 1.0, 0.5, 2.0],
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            &[1.0, 0.0, 0.0, 0.0, 0.707, 0.0, 0.707, 0.0],
            &[0.1, -0.2],
            &[1.0, 0.0, 0.0, 0.2, 0.4, 0.6],
            &device,
        )
        .unwrap();

        let view = SplatParameterView::from_trainable(&gaussians).unwrap();
        let rebuilt = view.to_trainable(&device).unwrap();

        assert_eq!(
            rebuilt.positions().to_vec2::<f32>().unwrap(),
            gaussians.positions().to_vec2::<f32>().unwrap()
        );
        assert_eq!(
            rebuilt.colors().to_vec2::<f32>().unwrap(),
            gaussians.colors().to_vec2::<f32>().unwrap()
        );
        assert_eq!(
            rebuilt.color_representation(),
            TrainableColorRepresentation::Rgb
        );
    }

    #[test]
    fn sh_view_round_trips_trainables() {
        let device = Device::Cpu;
        let gaussians = TrainableGaussians::new_with_sh(
            &[0.0, 0.0, 1.0],
            &[0.1, 0.2, 0.3],
            &[1.0, 0.0, 0.0, 0.0],
            &[0.1],
            &[
                rgb_to_sh0_value(0.2),
                rgb_to_sh0_value(0.4),
                rgb_to_sh0_value(0.6),
            ],
            &vec![0.5; 15 * 3],
            3,
            &device,
        )
        .unwrap();

        let view = SplatParameterView::from_trainable(&gaussians).unwrap();
        let rebuilt = view.to_trainable(&device).unwrap();

        assert!(rebuilt.uses_spherical_harmonics());
        assert_eq!(rebuilt.sh_degree(), 3);
        assert_eq!(
            rebuilt.sh_rest().to_vec3::<f32>().unwrap(),
            gaussians.sh_rest().to_vec3::<f32>().unwrap()
        );
    }
}
