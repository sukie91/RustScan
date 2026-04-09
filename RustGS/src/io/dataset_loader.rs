use super::tum_dataset::TumRgbdConfig;
use super::{colmap_dataset, nerfstudio_dataset, tum_dataset};
use crate::{TrainingConfig, TrainingDataset, TrainingError};
use rustscan_types::SlamOutput;
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetFormat {
    TumRgbd,
    Colmap,
    Nerfstudio,
    SlamOutputJson,
    TrainingDatasetJson,
}

#[derive(Debug, Clone)]
pub struct ResolvedTrainingDataset {
    pub dataset: TrainingDataset,
    pub format: DatasetFormat,
    pub source_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ResolvedTrainingInput {
    pub dataset: TrainingDataset,
    pub format: DatasetFormat,
    pub source_path: PathBuf,
    pub effective_config: TrainingConfig,
    pub overlay_path: Option<PathBuf>,
}

pub fn load_training_dataset_resolved(
    input: &Path,
    tum_config: &TumRgbdConfig,
) -> Result<ResolvedTrainingDataset, TrainingError> {
    if input.is_dir() {
        if tum_dataset::looks_like_tum_dataset(input) {
            return Ok(ResolvedTrainingDataset {
                dataset: tum_dataset::load_tum_rgbd_dataset(input, tum_config)?,
                format: DatasetFormat::TumRgbd,
                source_path: input.to_path_buf(),
            });
        }
        if colmap_dataset::looks_like_colmap_dataset(input) {
            return Ok(ResolvedTrainingDataset {
                dataset: colmap_dataset::load_colmap_dataset(input)?,
                format: DatasetFormat::Colmap,
                source_path: input.to_path_buf(),
            });
        }
        if nerfstudio_dataset::looks_like_nerfstudio_dataset(input) {
            return Ok(ResolvedTrainingDataset {
                dataset: nerfstudio_dataset::load_nerfstudio_dataset(input)?,
                format: DatasetFormat::Nerfstudio,
                source_path: input.to_path_buf(),
            });
        }
    }

    let input_buf = input.to_path_buf();
    if let Ok(slam_output) = SlamOutput::load(&input_buf) {
        return Ok(ResolvedTrainingDataset {
            dataset: slam_output.to_dataset(),
            format: DatasetFormat::SlamOutputJson,
            source_path: input_buf,
        });
    }
    if let Ok(dataset) = TrainingDataset::load(&input_buf) {
        return Ok(ResolvedTrainingDataset {
            dataset,
            format: DatasetFormat::TrainingDatasetJson,
            source_path: input_buf,
        });
    }

    Err(TrainingError::InvalidInput(format!(
        "failed to recognize dataset format at {}. Expected one of: TUM RGB-D dataset, COLMAP reconstruction, Nerfstudio transforms*.json scene, SlamOutput JSON, or TrainingDataset JSON",
        input.display(),
    )))
}

pub fn resolve_training_input(
    input: &Path,
    tum_config: &TumRgbdConfig,
    cli_config: &TrainingConfig,
) -> Result<ResolvedTrainingInput, TrainingError> {
    let resolved = load_training_dataset_resolved(input, tum_config)?;
    let overlay_path = discover_overlay_path(&resolved.source_path);
    let effective_config = if let Some(path) = overlay_path.as_ref() {
        merge_dataset_overlay(cli_config, path)?
    } else {
        cli_config.clone()
    };

    Ok(ResolvedTrainingInput {
        dataset: resolved.dataset,
        format: resolved.format,
        source_path: resolved.source_path,
        effective_config,
        overlay_path,
    })
}

fn discover_overlay_path(source_path: &Path) -> Option<PathBuf> {
    let search_root = if source_path.is_dir() {
        source_path
    } else {
        source_path.parent()?
    };
    ["rustgs.config.json", "rustgs.dataset.json"]
        .into_iter()
        .map(|name| search_root.join(name))
        .find(|path| path.exists())
}

fn merge_dataset_overlay(
    cli_config: &TrainingConfig,
    overlay_path: &Path,
) -> Result<TrainingConfig, TrainingError> {
    let overlay: Value =
        serde_json::from_str(&std::fs::read_to_string(overlay_path)?).map_err(|err| {
            TrainingError::InvalidInput(format!(
                "failed to parse dataset config overlay {}: {}",
                overlay_path.display(),
                err
            ))
        })?;
    let base = serde_json::to_value(cli_config).map_err(|err| {
        TrainingError::TrainingFailed(format!(
            "failed to serialize CLI config for overlay merge: {}",
            err
        ))
    })?;
    let defaults = serde_json::to_value(TrainingConfig::default()).map_err(|err| {
        TrainingError::TrainingFailed(format!(
            "failed to serialize default config for overlay merge: {}",
            err
        ))
    })?;
    let mut merged = base.clone();
    merge_overlay_value(&mut merged, &base, &defaults, &overlay);
    serde_json::from_value(merged).map_err(|err| {
        TrainingError::InvalidInput(format!(
            "failed to materialize merged dataset config from {}: {}",
            overlay_path.display(),
            err
        ))
    })
}

fn merge_overlay_value(target: &mut Value, base: &Value, defaults: &Value, overlay: &Value) {
    match (target, base, defaults, overlay) {
        (
            Value::Object(target_map),
            Value::Object(base_map),
            Value::Object(default_map),
            Value::Object(overlay_map),
        ) => {
            for (key, overlay_value) in overlay_map {
                let base_value = base_map.get(key).unwrap_or(&Value::Null);
                let default_value = default_map.get(key).unwrap_or(&Value::Null);
                if let (Value::Object(_), Value::Object(_), Value::Object(_)) =
                    (overlay_value, base_value, default_value)
                {
                    let entry = target_map
                        .entry(key.clone())
                        .or_insert_with(|| base_value.clone());
                    merge_overlay_value(entry, base_value, default_value, overlay_value);
                } else if base_value == default_value {
                    target_map.insert(key.clone(), overlay_value.clone());
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::{load_training_dataset_resolved, resolve_training_input, DatasetFormat};
    use crate::io::tum_dataset::TumRgbdConfig;
    use crate::TrainingConfig;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn unsupported_layout_reports_expected_formats() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("README.txt"), "no dataset here").unwrap();

        let err =
            load_training_dataset_resolved(dir.path(), &TumRgbdConfig::default()).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("TUM RGB-D"));
        assert!(message.contains("COLMAP"));
        assert!(message.contains("Nerfstudio"));
    }

    #[test]
    fn json_training_dataset_uses_unified_facade() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("dataset.json");
        let dataset = crate::TrainingDataset::new(crate::Intrinsics::from_focal(500.0, 32, 32));
        dataset.save(&path).unwrap();

        let resolved = load_training_dataset_resolved(&path, &TumRgbdConfig::default()).unwrap();
        assert_eq!(resolved.format, DatasetFormat::TrainingDatasetJson);
    }

    #[test]
    fn dataset_overlay_applies_when_cli_keeps_defaults() {
        let dir = tempdir().unwrap();
        let dataset_path = dir.path().join("dataset.json");
        let dataset = crate::TrainingDataset::new(crate::Intrinsics::from_focal(500.0, 32, 32));
        dataset.save(&dataset_path).unwrap();
        fs::write(
            dir.path().join("rustgs.config.json"),
            r#"{"iterations":123,"frame_shuffle_seed":17}"#,
        )
        .unwrap();

        let resolved = resolve_training_input(
            &dataset_path,
            &TumRgbdConfig::default(),
            &TrainingConfig::default(),
        )
        .unwrap();
        assert_eq!(resolved.effective_config.iterations, 123);
        assert_eq!(resolved.effective_config.frame_shuffle_seed, 17);
        assert!(resolved.overlay_path.is_some());
    }

    #[test]
    fn cli_values_override_dataset_overlay_when_non_default() {
        let dir = tempdir().unwrap();
        let dataset_path = dir.path().join("dataset.json");
        let dataset = crate::TrainingDataset::new(crate::Intrinsics::from_focal(500.0, 32, 32));
        dataset.save(&dataset_path).unwrap();
        fs::write(
            dir.path().join("rustgs.config.json"),
            r#"{"iterations":123}"#,
        )
        .unwrap();

        let cli_config = TrainingConfig {
            iterations: 456,
            ..TrainingConfig::default()
        };
        let resolved =
            resolve_training_input(&dataset_path, &TumRgbdConfig::default(), &cli_config).unwrap();
        assert_eq!(resolved.effective_config.iterations, 456);
    }
}
