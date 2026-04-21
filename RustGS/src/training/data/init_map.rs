use crate::core::HostSplats;
use crate::init::{initialize_host_splats_from_points, GaussianInitConfig};
use crate::{TrainingConfig, TrainingDataset, TrainingError};

pub(crate) fn build_initial_splats(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<HostSplats, TrainingError> {
    if dataset.initial_points.is_empty() {
        return Err(TrainingError::InvalidInput(
            "training now requires COLMAP sparse points for initialization; no initial_points were found in the dataset".to_string(),
        ));
    }

    let sh_degree = config.litegs.sh_degree;
    let init_config = gaussian_init_config_for_training();
    let mut splats =
        initialize_host_splats_from_points(&dataset.initial_points, &init_config, sh_degree)
            .map_err(TrainingError::from)?;
    let max_initial = config.max_initial_gaussians.max(1);
    if splats.len() > max_initial {
        log::warn!(
            "Truncating point-initialized chunk from {} to {} gaussians to respect max_initial_gaussians",
            splats.len(),
            max_initial,
        );
        splats.truncate_rows(max_initial);
    }

    splats
        .validate()
        .map_err(|err| TrainingError::TrainingFailed(err.to_string()))?;
    Ok(splats)
}

pub(super) fn gaussian_init_config_for_training() -> GaussianInitConfig {
    GaussianInitConfig::default()
}

#[cfg(test)]
mod tests {
    use super::{build_initial_splats, gaussian_init_config_for_training};
    use crate::{Intrinsics, TrainingDataset};

    #[test]
    fn training_profiles_share_brush_sparse_point_defaults() {
        let init = gaussian_init_config_for_training();
        assert_eq!(init.min_scale, 1e-3);
        assert_eq!(init.scale_factor, 0.5);
        assert_eq!(init.opacity, 0.5);
        assert_eq!(init.max_scale, f32::MAX);
    }

    #[test]
    fn build_initial_splats_requires_sparse_points() {
        let dataset = TrainingDataset::new(Intrinsics::new(1.0, 1.0, 0.0, 0.0, 2, 1));
        let err = build_initial_splats(&dataset, &crate::TrainingConfig::default()).unwrap_err();
        assert!(
            err.to_string()
                .contains("COLMAP sparse points for initialization"),
            "unexpected error: {err}"
        );
    }
}
