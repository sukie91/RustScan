use super::{LiteGsConfig, TrainingProfile};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub const DEFAULT_TINY_FIXTURE_ID: &str = "litegs-tiny-synthetic-v1";
pub const DEFAULT_CONVERGENCE_FIXTURE_ID: &str = "litegs-apple-silicon-convergence-v1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ParityFixtureKind {
    SyntheticTrainingDataset,
    TumRgbdDirectory,
    ColmapDirectory,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParityFixtureSpec {
    pub id: String,
    pub kind: ParityFixtureKind,
    pub description: String,
    pub input_path: Option<PathBuf>,
    pub bootstrap_input_path: Option<PathBuf>,
    pub reference_report_path: Option<PathBuf>,
    pub max_frames: Option<usize>,
    pub frame_stride: Option<usize>,
    pub notes: Vec<String>,
}

impl ParityFixtureSpec {
    pub fn resolve_input_path(&self, workspace_root: &Path) -> Option<PathBuf> {
        [self.input_path.as_ref(), self.bootstrap_input_path.as_ref()]
            .into_iter()
            .flatten()
            .map(|path| workspace_root.join(path))
            .find(|path| path.exists())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParityThresholds {
    pub require_no_nans: bool,
    pub require_no_oom: bool,
    pub require_deterministic_export_roundtrip: bool,
    pub psnr_delta_db_tolerance: Option<f32>,
    pub gaussian_count_delta_ratio_tolerance: Option<f32>,
    pub depth_mean_abs_delta_tolerance: Option<f32>,
    pub depth_max_abs_delta_tolerance: Option<f32>,
    pub total_mean_abs_delta_tolerance: Option<f32>,
    pub total_max_abs_delta_tolerance: Option<f32>,
}

impl Default for ParityThresholds {
    fn default() -> Self {
        Self {
            require_no_nans: true,
            require_no_oom: true,
            require_deterministic_export_roundtrip: true,
            psnr_delta_db_tolerance: None,
            gaussian_count_delta_ratio_tolerance: Some(0.10),
            depth_mean_abs_delta_tolerance: None,
            depth_max_abs_delta_tolerance: None,
            total_mean_abs_delta_tolerance: None,
            total_max_abs_delta_tolerance: None,
        }
    }
}

pub fn default_litegs_parity_fixtures() -> Vec<ParityFixtureSpec> {
    vec![
        ParityFixtureSpec {
            id: DEFAULT_TINY_FIXTURE_ID.to_string(),
            kind: ParityFixtureKind::SyntheticTrainingDataset,
            description: "Deterministic tiny synthetic scene for fast correctness and no-NaN checks.".to_string(),
            input_path: None,
            bootstrap_input_path: None,
            reference_report_path: None,
            max_frames: Some(3),
            frame_stride: Some(1),
            notes: vec![
                "Keeps parity checks independent from external datasets.".to_string(),
                "Used for initialization counts, active SH degree, export round-trip, and loss-term smoke checks.".to_string(),
            ],
        },
        ParityFixtureSpec {
            id: DEFAULT_CONVERGENCE_FIXTURE_ID.to_string(),
            kind: ParityFixtureKind::ColmapDirectory,
            description: "Apple Silicon convergence fixture reserved for LiteGS parity tracking.".to_string(),
            input_path: Some(PathBuf::from("test_data/fixtures/litegs/colmap-small")),
            bootstrap_input_path: Some(PathBuf::from("test_data/tum/rgbd_dataset_freiburg1_xyz")),
            reference_report_path: Some(PathBuf::from(
                "test_data/fixtures/litegs/colmap-small/parity-reference.json",
            )),
            max_frames: Some(90),
            frame_stride: Some(30),
            notes: vec![
                "The canonical parity target is a small COLMAP scene.".to_string(),
                "Until that fixture is checked into the workspace, the harness boots from the existing Freiburg1 XYZ TUM subset to keep Apple Silicon smoke coverage live.".to_string(),
            ],
        },
    ]
}

pub fn parity_fixture_id_for_input_path(input: &Path) -> String {
    let normalized_input = normalize_match_path(input);
    for fixture in default_litegs_parity_fixtures() {
        for candidate in [
            fixture.input_path.as_ref(),
            fixture.bootstrap_input_path.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if normalized_input.ends_with(&normalize_match_path(candidate)) {
                return fixture.id;
            }
        }
    }

    let stem = input
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("external");
    format!("external::{stem}")
}

pub fn resolve_litegs_parity_fixture_input_path(
    fixture_id: &str,
    workspace_root: &Path,
) -> Option<PathBuf> {
    default_litegs_parity_fixtures()
        .into_iter()
        .find(|fixture| fixture.id == fixture_id)
        .and_then(|fixture| fixture.resolve_input_path(workspace_root))
}

pub fn resolve_litegs_parity_reference_report_path(
    fixture_id: &str,
    workspace_root: &Path,
) -> Option<PathBuf> {
    default_litegs_parity_fixtures()
        .into_iter()
        .find(|fixture| fixture.id == fixture_id)
        .and_then(|fixture| fixture.reference_report_path)
        .map(|path| workspace_root.join(path))
        .filter(|path| path.exists())
}

pub fn default_parity_report_path(output_path: &Path) -> PathBuf {
    let parent = output_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = output_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("scene");
    parent.join(format!("{stem}.parity.json"))
}

fn normalize_match_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ParityLossTerms {
    pub l1: Option<f32>,
    pub ssim: Option<f32>,
    pub scale_regularization: Option<f32>,
    pub transmittance: Option<f32>,
    pub depth: Option<f32>,
    pub total: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ParityTopologyMetrics {
    pub initialization_gaussians: Option<usize>,
    pub final_gaussians: Option<usize>,
    pub total_epochs: Option<usize>,
    pub densify_until_epoch: Option<usize>,
    pub late_stage_start_epoch: Option<usize>,
    pub topology_freeze_epoch: Option<usize>,
    pub densify_events: usize,
    pub densify_added: usize,
    pub first_densify_epoch: Option<usize>,
    pub last_densify_epoch: Option<usize>,
    pub late_stage_densify_events: usize,
    pub late_stage_densify_added: usize,
    pub prune_events: usize,
    pub prune_removed: usize,
    pub first_prune_epoch: Option<usize>,
    pub last_prune_epoch: Option<usize>,
    pub late_stage_prune_events: usize,
    pub late_stage_prune_removed: usize,
    pub opacity_reset_events: usize,
    pub first_opacity_reset_epoch: Option<usize>,
    pub last_opacity_reset_epoch: Option<usize>,
    pub late_stage_opacity_reset_events: usize,
    pub export_outputs: usize,
    pub checkpoint_roundtrips: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ParityTimingMetrics {
    pub setup_ms: Option<u64>,
    pub training_ms: Option<u64>,
    pub export_ms: Option<u64>,
    pub total_wall_clock_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ParityLossCurveSample {
    pub iteration: usize,
    pub frame_idx: usize,
    pub l1: Option<f32>,
    pub ssim: Option<f32>,
    pub depth: Option<f32>,
    pub total: Option<f32>,
    pub depth_valid_pixels: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ParityReferenceComparison {
    pub compared_iterations: usize,
    pub compared_depth_samples: usize,
    pub compared_total_samples: usize,
    pub depth_mean_abs_delta: Option<f32>,
    pub depth_max_abs_delta: Option<f32>,
    pub total_mean_abs_delta: Option<f32>,
    pub total_max_abs_delta: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ParityGateStatus {
    Passed,
    Failed,
    MissingReference,
    Incomplete,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ParityCheckStatus {
    Passed,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParityCheckOutcome {
    pub name: String,
    pub status: ParityCheckStatus,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParityGateEvaluation {
    pub status: ParityGateStatus,
    pub checks: Vec<ParityCheckOutcome>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ParityMetricSnapshot {
    pub active_sh_degree: Option<usize>,
    pub final_psnr: Option<f32>,
    pub litegs_reference_psnr: Option<f32>,
    pub gaussian_count_delta_ratio: Option<f32>,
    pub depth_valid_pixels: Option<usize>,
    pub depth_grad_scale: Option<f32>,
    pub rotation_frozen: Option<bool>,
    pub had_nan: bool,
    pub had_oom: bool,
    pub export_roundtrip_ok: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParityHarnessReport {
    pub fixture_id: String,
    pub profile: TrainingProfile,
    pub litegs: LiteGsConfig,
    pub thresholds: ParityThresholds,
    pub loss_terms: ParityLossTerms,
    pub loss_curve_samples: Vec<ParityLossCurveSample>,
    pub reference_comparison: Option<ParityReferenceComparison>,
    pub gate: Option<ParityGateEvaluation>,
    pub topology: ParityTopologyMetrics,
    pub metrics: ParityMetricSnapshot,
    pub timing: ParityTimingMetrics,
    pub notes: Vec<String>,
}

impl ParityHarnessReport {
    pub fn new(
        fixture_id: impl Into<String>,
        profile: TrainingProfile,
        litegs: &LiteGsConfig,
    ) -> Self {
        Self {
            fixture_id: fixture_id.into(),
            profile,
            litegs: litegs.clone(),
            thresholds: ParityThresholds::default(),
            loss_terms: ParityLossTerms::default(),
            loss_curve_samples: Vec::new(),
            reference_comparison: None,
            gate: None,
            topology: ParityTopologyMetrics::default(),
            metrics: ParityMetricSnapshot::default(),
            timing: ParityTimingMetrics::default(),
            notes: Vec::new(),
        }
    }

    pub fn save_json(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_vec_pretty(self)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        fs::write(path, json)
    }

    pub fn load_json(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        serde_json::from_slice(&bytes)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
    }

    pub fn evaluate_gate(&self) -> ParityGateEvaluation {
        fn push_check(
            checks: &mut Vec<ParityCheckOutcome>,
            name: &str,
            status: ParityCheckStatus,
            detail: String,
        ) -> bool {
            let failed = status == ParityCheckStatus::Failed;
            checks.push(ParityCheckOutcome {
                name: name.to_string(),
                status,
                detail,
            });
            failed
        }

        fn compare_optional_delta(
            checks: &mut Vec<ParityCheckOutcome>,
            failed_checks: &mut usize,
            skipped_required_checks: &mut usize,
            reference_required: bool,
            reference_loaded: bool,
            name: &str,
            tolerance: Option<f32>,
            actual: Option<f32>,
        ) {
            let Some(tolerance) = tolerance else {
                return;
            };
            match actual {
                Some(actual) => {
                    if actual <= tolerance {
                        checks.push(ParityCheckOutcome {
                            name: name.to_string(),
                            status: ParityCheckStatus::Passed,
                            detail: format!("actual={actual:.6} tolerance={tolerance:.6}"),
                        });
                    } else {
                        *failed_checks += 1;
                        checks.push(ParityCheckOutcome {
                            name: name.to_string(),
                            status: ParityCheckStatus::Failed,
                            detail: format!("actual={actual:.6} tolerance={tolerance:.6}"),
                        });
                    }
                }
                None => {
                    checks.push(ParityCheckOutcome {
                        name: name.to_string(),
                        status: ParityCheckStatus::Skipped,
                        detail: "comparison value was unavailable".to_string(),
                    });
                    if reference_required && reference_loaded {
                        *skipped_required_checks += 1;
                    }
                }
            }
        }

        let mut checks = Vec::new();
        let mut failed_checks = 0usize;
        let mut skipped_required_checks = 0usize;
        let reference_required = self.fixture_id == DEFAULT_CONVERGENCE_FIXTURE_ID;
        let reference_loaded = self.reference_comparison.is_some()
            || self.metrics.litegs_reference_psnr.is_some()
            || self.metrics.gaussian_count_delta_ratio.is_some();

        if self.thresholds.require_no_nans {
            if push_check(
                &mut checks,
                "no_nans",
                if self.metrics.had_nan {
                    ParityCheckStatus::Failed
                } else {
                    ParityCheckStatus::Passed
                },
                if self.metrics.had_nan {
                    "report recorded non-finite values".to_string()
                } else {
                    "no non-finite values were recorded".to_string()
                },
            ) {
                failed_checks += 1;
            }
        }

        if self.thresholds.require_no_oom {
            if push_check(
                &mut checks,
                "no_oom",
                if self.metrics.had_oom {
                    ParityCheckStatus::Failed
                } else {
                    ParityCheckStatus::Passed
                },
                if self.metrics.had_oom {
                    "report recorded an out-of-memory event".to_string()
                } else {
                    "no out-of-memory event was recorded".to_string()
                },
            ) {
                failed_checks += 1;
            }
        }

        if self.thresholds.require_deterministic_export_roundtrip {
            if push_check(
                &mut checks,
                "export_roundtrip",
                if self.metrics.export_roundtrip_ok {
                    ParityCheckStatus::Passed
                } else {
                    ParityCheckStatus::Failed
                },
                if self.metrics.export_roundtrip_ok {
                    "export round-trip matched the trained scene".to_string()
                } else {
                    "export round-trip diverged from the trained scene".to_string()
                },
            ) {
                failed_checks += 1;
            }
        }

        if reference_required && !reference_loaded {
            checks.push(ParityCheckOutcome {
                name: "reference_report".to_string(),
                status: ParityCheckStatus::Skipped,
                detail: "no reference parity report was loaded for the convergence fixture"
                    .to_string(),
            });
        }

        let missing_required_data_counts = |reference_required: bool, reference_loaded: bool| {
            reference_required && reference_loaded
        };

        if let Some(tolerance) = self.thresholds.gaussian_count_delta_ratio_tolerance {
            match self.metrics.gaussian_count_delta_ratio {
                Some(actual) => {
                    if push_check(
                        &mut checks,
                        "gaussian_count_delta_ratio",
                        if actual <= tolerance {
                            ParityCheckStatus::Passed
                        } else {
                            ParityCheckStatus::Failed
                        },
                        format!("actual={actual:.6} tolerance={tolerance:.6}"),
                    ) {
                        failed_checks += 1;
                    }
                }
                None => {
                    checks.push(ParityCheckOutcome {
                        name: "gaussian_count_delta_ratio".to_string(),
                        status: ParityCheckStatus::Skipped,
                        detail: "gaussian count reference delta was unavailable".to_string(),
                    });
                    if missing_required_data_counts(reference_required, reference_loaded) {
                        skipped_required_checks += 1;
                    }
                }
            }
        }

        if let Some(tolerance) = self.thresholds.psnr_delta_db_tolerance {
            match (self.metrics.final_psnr, self.metrics.litegs_reference_psnr) {
                (Some(current), Some(reference)) => {
                    let actual = (current - reference).abs();
                    if push_check(
                        &mut checks,
                        "final_psnr_delta_db",
                        if actual <= tolerance {
                            ParityCheckStatus::Passed
                        } else {
                            ParityCheckStatus::Failed
                        },
                        format!(
                            "current={current:.6} reference={reference:.6} actual={actual:.6} tolerance={tolerance:.6}"
                        ),
                    ) {
                        failed_checks += 1;
                    }
                }
                _ => {
                    checks.push(ParityCheckOutcome {
                        name: "final_psnr_delta_db".to_string(),
                        status: ParityCheckStatus::Skipped,
                        detail: "final PSNR or reference PSNR was unavailable".to_string(),
                    });
                    if missing_required_data_counts(reference_required, reference_loaded) {
                        skipped_required_checks += 1;
                    }
                }
            }
        }

        let reference = self.reference_comparison.as_ref();
        compare_optional_delta(
            &mut checks,
            &mut failed_checks,
            &mut skipped_required_checks,
            reference_required,
            reference_loaded,
            "depth_mean_abs_delta",
            self.thresholds.depth_mean_abs_delta_tolerance,
            reference.and_then(|comparison| comparison.depth_mean_abs_delta),
        );
        compare_optional_delta(
            &mut checks,
            &mut failed_checks,
            &mut skipped_required_checks,
            reference_required,
            reference_loaded,
            "depth_max_abs_delta",
            self.thresholds.depth_max_abs_delta_tolerance,
            reference.and_then(|comparison| comparison.depth_max_abs_delta),
        );
        compare_optional_delta(
            &mut checks,
            &mut failed_checks,
            &mut skipped_required_checks,
            reference_required,
            reference_loaded,
            "total_mean_abs_delta",
            self.thresholds.total_mean_abs_delta_tolerance,
            reference.and_then(|comparison| comparison.total_mean_abs_delta),
        );
        compare_optional_delta(
            &mut checks,
            &mut failed_checks,
            &mut skipped_required_checks,
            reference_required,
            reference_loaded,
            "total_max_abs_delta",
            self.thresholds.total_max_abs_delta_tolerance,
            reference.and_then(|comparison| comparison.total_max_abs_delta),
        );

        let status = if failed_checks > 0 {
            ParityGateStatus::Failed
        } else if reference_required && !reference_loaded {
            ParityGateStatus::MissingReference
        } else if skipped_required_checks > 0 {
            ParityGateStatus::Incomplete
        } else {
            ParityGateStatus::Passed
        };

        ParityGateEvaluation { status, checks }
    }
}

pub fn compare_loss_curve_samples(
    current: &[ParityLossCurveSample],
    reference: &[ParityLossCurveSample],
) -> Option<ParityReferenceComparison> {
    use std::collections::BTreeMap;

    let mut reference_by_iter = BTreeMap::new();
    for sample in reference {
        reference_by_iter.insert(sample.iteration, sample);
    }

    let mut compared_iterations = 0usize;
    let mut depth_deltas = Vec::new();
    let mut total_deltas = Vec::new();
    for sample in current {
        let Some(reference_sample) = reference_by_iter.get(&sample.iteration) else {
            continue;
        };
        compared_iterations += 1;
        if let (Some(current_depth), Some(reference_depth)) = (sample.depth, reference_sample.depth)
        {
            depth_deltas.push((current_depth - reference_depth).abs());
        }
        if let (Some(current_total), Some(reference_total)) = (sample.total, reference_sample.total)
        {
            total_deltas.push((current_total - reference_total).abs());
        }
    }

    if compared_iterations == 0 {
        return None;
    }

    let summarize = |deltas: &[f32]| -> (Option<f32>, Option<f32>) {
        if deltas.is_empty() {
            (None, None)
        } else {
            let sum = deltas.iter().sum::<f32>();
            let max = deltas
                .iter()
                .copied()
                .fold(0.0f32, |acc, delta| acc.max(delta));
            (Some(sum / deltas.len() as f32), Some(max))
        }
    };
    let (depth_mean_abs_delta, depth_max_abs_delta) = summarize(&depth_deltas);
    let (total_mean_abs_delta, total_max_abs_delta) = summarize(&total_deltas);

    Some(ParityReferenceComparison {
        compared_iterations,
        compared_depth_samples: depth_deltas.len(),
        compared_total_samples: total_deltas.len(),
        depth_mean_abs_delta,
        depth_max_abs_delta,
        total_mean_abs_delta,
        total_max_abs_delta,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        compare_loss_curve_samples, default_litegs_parity_fixtures, default_parity_report_path,
        parity_fixture_id_for_input_path, resolve_litegs_parity_fixture_input_path,
        resolve_litegs_parity_reference_report_path, ParityGateStatus, ParityHarnessReport,
        ParityLossCurveSample, ParityReferenceComparison, ParityThresholds,
        DEFAULT_CONVERGENCE_FIXTURE_ID, DEFAULT_TINY_FIXTURE_ID,
    };
    use crate::{LiteGsConfig, TrainingProfile};
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    #[test]
    fn default_fixture_registry_contains_tiny_and_convergence_targets() {
        let fixtures = default_litegs_parity_fixtures();
        assert_eq!(fixtures.len(), 2);
        assert_eq!(fixtures[0].id, DEFAULT_TINY_FIXTURE_ID);
        assert_eq!(fixtures[1].id, DEFAULT_CONVERGENCE_FIXTURE_ID);
        assert_eq!(
            fixtures[1]
                .bootstrap_input_path
                .as_ref()
                .unwrap()
                .to_string_lossy(),
            "test_data/tum/rgbd_dataset_freiburg1_xyz"
        );
    }

    #[test]
    fn parity_thresholds_match_epic_17_targets() {
        let thresholds = ParityThresholds::default();
        assert!(thresholds.require_no_nans);
        assert!(thresholds.require_no_oom);
        assert!(thresholds.require_deterministic_export_roundtrip);
        assert_eq!(thresholds.psnr_delta_db_tolerance, None);
        assert_eq!(thresholds.gaussian_count_delta_ratio_tolerance, Some(0.10));
        assert_eq!(thresholds.depth_mean_abs_delta_tolerance, None);
        assert_eq!(thresholds.depth_max_abs_delta_tolerance, None);
        assert_eq!(thresholds.total_mean_abs_delta_tolerance, None);
        assert_eq!(thresholds.total_max_abs_delta_tolerance, None);
    }

    #[test]
    fn parity_report_round_trips_through_json() {
        let tempdir = tempdir().unwrap();
        let path = tempdir.path().join("parity/report.json");
        let mut report = ParityHarnessReport::new(
            DEFAULT_TINY_FIXTURE_ID,
            TrainingProfile::LiteGsMacV1,
            &LiteGsConfig::default(),
        );
        report.metrics.final_psnr = Some(28.4);
        report.loss_curve_samples.push(ParityLossCurveSample {
            iteration: 4,
            frame_idx: 1,
            depth: Some(0.2),
            total: Some(0.5),
            depth_valid_pixels: Some(64),
            ..Default::default()
        });
        report.topology.initialization_gaussians = Some(1024);
        report.notes.push("roundtrip".to_string());

        report.save_json(&path).unwrap();
        let loaded = ParityHarnessReport::load_json(&path).unwrap();

        assert_eq!(loaded, report);
    }

    #[test]
    fn parity_fixture_id_matches_known_bootstrap_fixture_and_external_fallback() {
        assert_eq!(
            parity_fixture_id_for_input_path(Path::new("test_data/tum/rgbd_dataset_freiburg1_xyz")),
            DEFAULT_CONVERGENCE_FIXTURE_ID
        );
        assert_eq!(
            parity_fixture_id_for_input_path(Path::new("/tmp/my_scene")),
            "external::my_scene"
        );
    }

    #[test]
    fn default_parity_report_path_uses_output_stem() {
        let path = default_parity_report_path(Path::new("/tmp/scene.ply"));
        assert_eq!(path, PathBuf::from("/tmp/scene.parity.json"));
    }

    #[test]
    fn resolve_litegs_fixture_input_prefers_canonical_fixture_and_falls_back_to_bootstrap() {
        let tempdir = tempdir().unwrap();
        let workspace_root = tempdir.path();
        let bootstrap = workspace_root.join("test_data/tum/rgbd_dataset_freiburg1_xyz");
        std::fs::create_dir_all(&bootstrap).unwrap();

        let resolved_bootstrap = resolve_litegs_parity_fixture_input_path(
            DEFAULT_CONVERGENCE_FIXTURE_ID,
            workspace_root,
        )
        .unwrap();
        assert_eq!(resolved_bootstrap, bootstrap);

        let canonical = workspace_root.join("test_data/fixtures/litegs/colmap-small");
        std::fs::create_dir_all(&canonical).unwrap();

        let resolved_canonical = resolve_litegs_parity_fixture_input_path(
            DEFAULT_CONVERGENCE_FIXTURE_ID,
            workspace_root,
        )
        .unwrap();
        assert_eq!(resolved_canonical, canonical);
    }

    #[test]
    fn resolve_litegs_reference_report_path_returns_existing_reference_only() {
        let tempdir = tempdir().unwrap();
        let workspace_root = tempdir.path();
        assert!(resolve_litegs_parity_reference_report_path(
            DEFAULT_CONVERGENCE_FIXTURE_ID,
            workspace_root,
        )
        .is_none());

        let reference =
            workspace_root.join("test_data/fixtures/litegs/colmap-small/parity-reference.json");
        std::fs::create_dir_all(reference.parent().unwrap()).unwrap();
        std::fs::write(&reference, "{}").unwrap();

        let resolved = resolve_litegs_parity_reference_report_path(
            DEFAULT_CONVERGENCE_FIXTURE_ID,
            workspace_root,
        )
        .unwrap();
        assert_eq!(resolved, reference);
    }

    #[test]
    fn compare_loss_curve_samples_matches_iterations_and_reports_deltas() {
        let current = vec![
            ParityLossCurveSample {
                iteration: 0,
                depth: Some(0.6),
                total: Some(1.0),
                ..Default::default()
            },
            ParityLossCurveSample {
                iteration: 25,
                depth: Some(0.3),
                total: Some(0.7),
                ..Default::default()
            },
        ];
        let reference = vec![
            ParityLossCurveSample {
                iteration: 0,
                depth: Some(0.5),
                total: Some(0.8),
                ..Default::default()
            },
            ParityLossCurveSample {
                iteration: 25,
                depth: Some(0.4),
                total: Some(0.6),
                ..Default::default()
            },
            ParityLossCurveSample {
                iteration: 50,
                depth: Some(0.2),
                total: Some(0.5),
                ..Default::default()
            },
        ];

        let comparison = compare_loss_curve_samples(&current, &reference).unwrap();
        assert_eq!(comparison.compared_iterations, 2);
        assert_eq!(comparison.compared_depth_samples, 2);
        assert_eq!(comparison.compared_total_samples, 2);
        assert!(
            (comparison.depth_mean_abs_delta.unwrap() - 0.1).abs() < 1e-6,
            "{comparison:?}"
        );
        assert!(
            (comparison.depth_max_abs_delta.unwrap() - 0.1).abs() < 1e-6,
            "{comparison:?}"
        );
        assert!(
            (comparison.total_mean_abs_delta.unwrap() - 0.15).abs() < 1e-6,
            "{comparison:?}"
        );
        assert!(
            (comparison.total_max_abs_delta.unwrap() - 0.2).abs() < 1e-6,
            "{comparison:?}"
        );
    }

    #[test]
    fn parity_gate_reports_missing_reference_for_convergence_fixture() {
        let mut report = ParityHarnessReport::new(
            DEFAULT_CONVERGENCE_FIXTURE_ID,
            TrainingProfile::LiteGsMacV1,
            &LiteGsConfig::default(),
        );
        report.metrics.export_roundtrip_ok = true;

        let evaluation = report.evaluate_gate();
        assert_eq!(evaluation.status, ParityGateStatus::MissingReference);
        assert!(evaluation
            .checks
            .iter()
            .any(|check| check.name == "reference_report"));
    }

    #[test]
    fn parity_gate_fails_when_gaussian_count_delta_exceeds_tolerance() {
        let mut report = ParityHarnessReport::new(
            DEFAULT_CONVERGENCE_FIXTURE_ID,
            TrainingProfile::LiteGsMacV1,
            &LiteGsConfig::default(),
        );
        report.metrics.export_roundtrip_ok = true;
        report.metrics.gaussian_count_delta_ratio = Some(0.25);

        let evaluation = report.evaluate_gate();
        assert_eq!(evaluation.status, ParityGateStatus::Failed);
        assert!(evaluation.checks.iter().any(|check| {
            check.name == "gaussian_count_delta_ratio"
                && check.status == super::ParityCheckStatus::Failed
        }));
    }

    #[test]
    fn parity_gate_passes_when_reference_backed_checks_are_within_tolerance() {
        let mut report = ParityHarnessReport::new(
            DEFAULT_CONVERGENCE_FIXTURE_ID,
            TrainingProfile::LiteGsMacV1,
            &LiteGsConfig::default(),
        );
        report.metrics.export_roundtrip_ok = true;
        report.metrics.gaussian_count_delta_ratio = Some(0.05);
        report.reference_comparison = Some(ParityReferenceComparison {
            compared_iterations: 2,
            compared_depth_samples: 2,
            compared_total_samples: 2,
            depth_mean_abs_delta: Some(0.02),
            depth_max_abs_delta: Some(0.05),
            total_mean_abs_delta: Some(0.03),
            total_max_abs_delta: Some(0.05),
        });
        report.thresholds.depth_mean_abs_delta_tolerance = Some(0.05);
        report.thresholds.depth_max_abs_delta_tolerance = Some(0.10);

        let evaluation = report.evaluate_gate();
        assert_eq!(evaluation.status, ParityGateStatus::Passed);
    }
}
