//! 3DGS scene export and import utilities (PLY).
//!
//! Stores Gaussian parameters in an ASCII PLY with versioned metadata comments.

mod scene_export;
mod scene_import;

use std::path::Path;

use thiserror::Error;

use crate::core::GaussianColorRepresentation;
use crate::render::tiled_renderer::Gaussian;

pub const CURRENT_SCENE_FORMAT_VERSION: u32 = 2;

#[derive(Debug, Clone)]
pub struct SceneMetadata {
    pub format_version: u32,
    pub iterations: usize,
    pub final_loss: f32,
    pub gaussian_count: usize,
    pub color_representation: GaussianColorRepresentation,
}

impl Default for SceneMetadata {
    fn default() -> Self {
        Self {
            format_version: CURRENT_SCENE_FORMAT_VERSION,
            iterations: 0,
            final_loss: 0.0,
            gaussian_count: 0,
            color_representation: GaussianColorRepresentation::Rgb,
        }
    }
}

#[derive(Debug, Error)]
pub enum SceneIoError {
    #[error("failed to write scene {path}: {source}")]
    Write {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to read scene {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("invalid scene format: {message}")]
    InvalidFormat { message: String },
    #[error("parse error: {0}")]
    Parse(String),
}

pub fn save_scene_ply(
    path: &Path,
    gaussians: &[Gaussian],
    metadata: &SceneMetadata,
) -> Result<(), SceneIoError> {
    scene_export::save_scene_ply(path, gaussians, metadata)
}

pub fn load_scene_ply(path: &Path) -> Result<(Vec<Gaussian>, SceneMetadata), SceneIoError> {
    scene_import::load_scene_ply(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_scene_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("scene.ply");

        let gaussians = vec![
            Gaussian::new(
                [0.0, 0.0, 1.0],
                [0.1, 0.1, 0.1],
                [1.0, 0.0, 0.0, 0.0],
                0.5,
                [0.2, 0.3, 0.4],
            ),
            Gaussian::new(
                [1.0, 0.0, 2.0],
                [0.2, 0.2, 0.2],
                [1.0, 0.0, 0.0, 0.0],
                0.6,
                [0.5, 0.4, 0.3],
            ),
        ];

        let metadata = SceneMetadata {
            iterations: 3000,
            final_loss: 0.42,
            gaussian_count: gaussians.len(),
            ..SceneMetadata::default()
        };

        save_scene_ply(&path, &gaussians, &metadata).unwrap();

        let (loaded, loaded_meta) = load_scene_ply(&path).unwrap();
        assert_eq!(loaded.len(), gaussians.len());
        assert_eq!(loaded_meta.format_version, CURRENT_SCENE_FORMAT_VERSION);
        assert_eq!(loaded_meta.iterations, 3000);
        assert!((loaded_meta.final_loss - 0.42).abs() < 1e-6);
        assert!(matches!(
            loaded_meta.color_representation,
            GaussianColorRepresentation::Rgb
        ));
    }

    #[test]
    fn sh_scene_roundtrip_preserves_metadata_and_coefficients() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("scene_sh.ply");

        let gaussians = vec![Gaussian::new(
            [0.0, 0.0, 1.0],
            [0.1, 0.1, 0.1],
            [1.0, 0.0, 0.0, 0.0],
            0.5,
            [0.2, 0.3, 0.4],
        )
        .with_color_state(
            GaussianColorRepresentation::SphericalHarmonics { degree: 1 },
            Some([0.8, 0.9, 1.0]),
            Some(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.1, -0.2, -0.3]),
        )];
        let metadata = SceneMetadata {
            iterations: 25,
            final_loss: 0.01,
            gaussian_count: gaussians.len(),
            color_representation: GaussianColorRepresentation::SphericalHarmonics { degree: 1 },
            ..SceneMetadata::default()
        };

        save_scene_ply(&path, &gaussians, &metadata).unwrap();
        let (loaded, loaded_meta) = load_scene_ply(&path).unwrap();

        assert_eq!(
            loaded_meta.color_representation,
            metadata.color_representation
        );
        assert_eq!(loaded[0].sh_dc, gaussians[0].sh_dc);
        assert_eq!(loaded[0].sh_rest, gaussians[0].sh_rest);
    }

    #[test]
    fn legacy_scene_without_version_comment_loads_with_rgb_fallback() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("legacy_scene.ply");
        std::fs::write(
            &path,
            concat!(
                "ply\n",
                "format ascii 1.0\n",
                "comment rustgs_scene\n",
                "comment iterations 12\n",
                "comment final_loss 0.5\n",
                "element vertex 1\n",
                "property float x\n",
                "property float y\n",
                "property float z\n",
                "property float scale_x\n",
                "property float scale_y\n",
                "property float scale_z\n",
                "property float rot_w\n",
                "property float rot_x\n",
                "property float rot_y\n",
                "property float rot_z\n",
                "property float opacity\n",
                "property float color_r\n",
                "property float color_g\n",
                "property float color_b\n",
                "end_header\n",
                "0 0 1 0.1 0.1 0.1 1 0 0 0 0.5 0.2 0.3 0.4\n"
            ),
        )
        .unwrap();

        let (loaded, metadata) = load_scene_ply(&path).unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(metadata.format_version, 0);
        assert!(matches!(
            metadata.color_representation,
            GaussianColorRepresentation::Rgb
        ));
    }

    #[test]
    fn legacy_scene_without_metadata_infers_sh_degree_from_vertex_properties() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("legacy_scene_sh.ply");
        std::fs::write(
            &path,
            concat!(
                "ply\n",
                "format ascii 1.0\n",
                "comment rustgs_scene\n",
                "element vertex 1\n",
                "property float x\n",
                "property float y\n",
                "property float z\n",
                "property float scale_x\n",
                "property float scale_y\n",
                "property float scale_z\n",
                "property float rot_w\n",
                "property float rot_x\n",
                "property float rot_y\n",
                "property float rot_z\n",
                "property float opacity\n",
                "property float color_r\n",
                "property float color_g\n",
                "property float color_b\n",
                "property float sh0_r\n",
                "property float sh0_g\n",
                "property float sh0_b\n",
                "property float sh_rest_0_r\n",
                "property float sh_rest_0_g\n",
                "property float sh_rest_0_b\n",
                "property float sh_rest_1_r\n",
                "property float sh_rest_1_g\n",
                "property float sh_rest_1_b\n",
                "property float sh_rest_2_r\n",
                "property float sh_rest_2_g\n",
                "property float sh_rest_2_b\n",
                "end_header\n",
                "0 0 1 0.1 0.1 0.1 1 0 0 0 0.5 0.2 0.3 0.4 0.8 0.9 1.0 0.1 0.2 0.3 0.4 0.5 0.6 -0.1 -0.2 -0.3\n"
            ),
        )
        .unwrap();

        let (loaded, metadata) = load_scene_ply(&path).unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(metadata.format_version, 0);
        assert_eq!(
            metadata.color_representation,
            GaussianColorRepresentation::SphericalHarmonics { degree: 1 }
        );
        assert_eq!(loaded[0].sh_dc, Some([0.8, 0.9, 1.0]));
        assert_eq!(
            loaded[0].sh_rest,
            Some(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.1, -0.2, -0.3])
        );
    }

    #[test]
    fn explicit_sh_metadata_without_sh_properties_fails_clearly() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("invalid_scene_sh.ply");
        std::fs::write(
            &path,
            concat!(
                "ply\n",
                "format ascii 1.0\n",
                "comment rustgs_scene\n",
                "comment format_version 2\n",
                "comment color_representation spherical_harmonics\n",
                "comment sh_degree 1\n",
                "element vertex 1\n",
                "property float x\n",
                "property float y\n",
                "property float z\n",
                "property float scale_x\n",
                "property float scale_y\n",
                "property float scale_z\n",
                "property float rot_w\n",
                "property float rot_x\n",
                "property float rot_y\n",
                "property float rot_z\n",
                "property float opacity\n",
                "property float color_r\n",
                "property float color_g\n",
                "property float color_b\n",
                "end_header\n",
                "0 0 1 0.1 0.1 0.1 1 0 0 0 0.5 0.2 0.3 0.4\n"
            ),
        )
        .unwrap();

        let err = load_scene_ply(&path).expect_err("scene should fail without SH properties");
        assert!(
            err.to_string()
                .contains("scene metadata declares spherical_harmonics"),
            "unexpected error: {err}"
        );
    }
}
