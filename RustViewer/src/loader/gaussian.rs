//! Loader for Gaussian scene PLY files.

use std::path::Path;

use crate::loader::checkpoint::LoadError;
use crate::renderer::scene::{GaussianPoint, Scene};

/// Load a 3DGS scene.ply and add Gaussians to the scene.
pub fn load_gaussians(path: &Path, scene: &mut Scene) -> Result<(), LoadError> {
    let (gaussians, _meta) = rustslam::fusion::load_scene_ply(path)
        .map_err(|e| LoadError::Io(std::io::Error::other(e.to_string())))?;

    for g in &gaussians {
        let color = [
            (g.color[0] * g.opacity).clamp(0.0, 1.0),
            (g.color[1] * g.opacity).clamp(0.0, 1.0),
            (g.color[2] * g.opacity).clamp(0.0, 1.0),
        ];
        scene.gaussians.push(GaussianPoint {
            position: g.position,
            color,
        });
        scene.bounds.extend(g.position);
    }

    Ok(())
}
