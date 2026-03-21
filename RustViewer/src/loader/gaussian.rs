//! Loader for Gaussian scene PLY files.

use std::path::Path;

use crate::loader::checkpoint::LoadError;
use crate::renderer::scene::{GaussianSplat, Scene};

/// Load a 3DGS scene.ply and add Gaussians to the scene.
///
/// # Example
/// ```no_run
/// use std::path::Path;
/// use rust_viewer::loader::gaussian::load_gaussians;
/// use rust_viewer::renderer::scene::Scene;
///
/// let mut scene = Scene::default();
/// load_gaussians(Path::new("scene.ply"), &mut scene).unwrap();
/// println!("Loaded {} gaussians", scene.gaussians.len());
/// ```
pub fn load_gaussians(path: &Path, scene: &mut Scene) -> Result<(), LoadError> {
    let (gaussians, _meta) = rustslam::fusion::load_scene_ply(path).map_err(|e| {
        LoadError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            e.to_string(),
        ))
    })?;

    for g in &gaussians {
        scene.gaussians.push(GaussianSplat {
            position: g.position,
            scale: g.scale,
            rotation: g.rotation,
            opacity: g.opacity,
            color: g.color,
        });
        scene.bounds.extend(g.position);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustslam::fusion::{save_scene_ply, Gaussian, SceneMetadata};

    #[test]
    fn loads_full_gaussian_parameters_from_ply() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let gaussians = vec![Gaussian::new(
            [1.0, 2.0, 3.0],
            [0.1, 0.2, 0.3],
            [0.707, 0.0, 0.707, 0.0],
            0.42,
            [0.9, 0.8, 0.7],
        )];
        save_scene_ply(
            tmp.path(),
            &gaussians,
            &SceneMetadata {
                iterations: 1,
                final_loss: 0.5,
                gaussian_count: 1,
            },
        )
        .unwrap();

        let mut scene = Scene::default();
        load_gaussians(tmp.path(), &mut scene).unwrap();

        assert_eq!(scene.gaussians.len(), 1);
        assert_eq!(scene.gaussians[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(scene.gaussians[0].scale, [0.1, 0.2, 0.3]);
        assert_eq!(scene.gaussians[0].rotation, [0.707, 0.0, 0.707, 0.0]);
        assert!((scene.gaussians[0].opacity - 0.42).abs() < 1e-6);
        assert_eq!(scene.gaussians[0].color, [0.9, 0.8, 0.7]);
    }
}
