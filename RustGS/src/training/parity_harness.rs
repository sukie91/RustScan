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
    pub max_frames: Option<usize>,
    pub frame_stride: Option<usize>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParityThresholds {
    pub require_no_nans: bool,
    pub require_no_oom: bool,
    pub non_clustered_psnr_delta_db: f32,
    pub non_clustered_gaussian_count_tolerance: f32,
    pub clustered_psnr_delta_db: f32,
    pub require_deterministic_export_roundtrip: bool,
}

impl Default for ParityThresholds {
    fn default() -> Self {
        Self {
            require_no_nans: true,
            require_no_oom: true,
            non_clustered_psnr_delta_db: 0.5,
            non_clustered_gaussian_count_tolerance: 0.10,
            clustered_psnr_delta_db: 0.7,
            require_deterministic_export_roundtrip: true,
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
    pub densify_events: usize,
    pub prune_events: usize,
    pub opacity_reset_events: usize,
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
pub struct ParityMetricSnapshot {
    pub active_sh_degree: Option<usize>,
    pub final_psnr: Option<f32>,
    pub litegs_reference_psnr: Option<f32>,
    pub gaussian_count_delta_ratio: Option<f32>,
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
}

#[cfg(test)]
mod tests {
    use super::{
        default_litegs_parity_fixtures, default_parity_report_path,
        parity_fixture_id_for_input_path, ParityHarnessReport, ParityThresholds,
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
        assert_eq!(thresholds.non_clustered_psnr_delta_db, 0.5);
        assert_eq!(thresholds.non_clustered_gaussian_count_tolerance, 0.10);
        assert_eq!(thresholds.clustered_psnr_delta_db, 0.7);
        assert!(thresholds.require_deterministic_export_roundtrip);
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
}
