//! Legacy AoS scene/model compatibility surface.

mod gaussian;

pub use gaussian::{Gaussian3D, GaussianColorRepresentation, GaussianMap, GaussianState};

#[allow(deprecated)]
pub use crate::io::scene_io::{load_scene_ply, save_scene_ply};

#[cfg(feature = "gpu")]
#[allow(deprecated)]
pub use crate::training::{
    evaluate_scene, runtime_from_scene, trainable_from_scene, SceneEvaluationResult,
    SceneEvaluationSummary,
};
