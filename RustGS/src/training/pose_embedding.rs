//! Learnable camera extrinsics (pose embedding) for end-to-end pose optimization.
//!
//! This module implements Story 3.2: Learnable Camera Extrinsics.
//! Each camera frame has a learnable quaternion + translation that can be
//! optimized during training to correct pose drift.

use candle_core::{DType, Device, Tensor, Var};

use crate::diff::diff_splat::DiffCamera;
use crate::SE3;

/// Learnable pose parameters for a single frame.
///
/// Stores quaternion (w, x, y, z) and translation (x, y, z) as
/// separate tensors for efficient optimization.
#[derive(Debug, Clone)]
pub struct PoseEmbedding {
    /// Quaternion (w, x, y, z) - shape [4]
    quaternion: Var,
    /// Translation (x, y, z) - shape [3]
    translation: Var,
    /// Frame index
    frame_idx: usize,
}

impl PoseEmbedding {
    /// Create a new pose embedding from an initial SE3 pose.
    ///
    /// The pose is expected to be in camera-to-world form (same as ScenePose).
    pub fn from_se3(pose: &SE3, frame_idx: usize, device: &Device) -> candle_core::Result<Self> {
        // SE3 stores quaternion as (qx, qy, qz, qw) - need to reorder to (w, x, y, z)
        let q = pose.quaternion();
        let quat_arr = [q[3], q[0], q[1], q[2]]; // Reorder to w, x, y, z

        let t = pose.translation();

        let quaternion = Var::from_slice(&quat_arr, (4,), device)?;
        let translation = Var::from_slice(&t, (3,), device)?;

        Ok(Self {
            quaternion,
            translation,
            frame_idx,
        })
    }

    /// Create an identity pose embedding.
    pub fn identity(frame_idx: usize, device: &Device) -> candle_core::Result<Self> {
        let quat_arr = [1.0f32, 0.0, 0.0, 0.0]; // Identity quaternion (w=1, x=y=z=0)
        let trans_arr = [0.0f32, 0.0, 0.0];

        let quaternion = Var::from_slice(&quat_arr, (4,), device)?;
        let translation = Var::from_slice(&trans_arr, (3,), device)?;

        Ok(Self {
            quaternion,
            translation,
            frame_idx,
        })
    }

    /// Get the frame index.
    pub fn frame_idx(&self) -> usize {
        self.frame_idx
    }

    /// Get the quaternion variable.
    pub fn quaternion(&self) -> &Var {
        &self.quaternion
    }

    /// Get the translation variable.
    pub fn translation(&self) -> &Var {
        &self.translation
    }

    /// Convert to SE3 pose (camera-to-world).
    ///
    /// Returns the current pose as an SE3 for use in non-differentiable contexts.
    pub fn to_se3(&self) -> candle_core::Result<SE3> {
        let q = self.quaternion.as_tensor().to_vec1::<f32>()?;
        let t = self.translation.as_tensor().to_vec1::<f32>()?;

        // Normalize quaternion
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        let q_normalized: [f32; 4] = if norm > 1e-8 {
            [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
        } else {
            [1.0, 0.0, 0.0, 0.0] // Identity fallback
        };

        let t_arr: [f32; 3] = [t[0], t[1], t[2]];

        // Convert back to SE3 convention (qx, qy, qz, qw)
        Ok(SE3::new(
            &[q_normalized[1], q_normalized[2], q_normalized[3], q_normalized[0]],
            &t_arr,
        ))
    }

    /// Build a DiffCamera from this pose embedding.
    ///
    /// The resulting DiffCamera has world-to-camera extrinsics.
    pub fn to_diff_camera(
        &self,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        width: usize,
        height: usize,
        device: &Device,
    ) -> candle_core::Result<DiffCamera> {
        // Get the camera-to-world pose
        let pose = self.to_se3()?;
        // Invert to get world-to-camera
        let view_pose = pose.inverse();

        let rotation = view_pose.rotation();
        let translation = view_pose.translation();

        DiffCamera::new(fx, fy, cx, cy, width, height, &rotation, &translation, device)
    }
}

/// Collection of pose embeddings for all frames in a dataset.
///
/// Manages learnable camera poses with sparse Adam optimization.
#[derive(Debug)]
pub struct PoseEmbeddings {
    /// Per-frame pose embeddings
    embeddings: Vec<PoseEmbedding>,
    /// Adam first moment for quaternions (per frame: [4])
    quaternion_m: Vec<Tensor>,
    /// Adam first moment for translations (per frame: [3])
    translation_m: Vec<Tensor>,
    /// Adam second moment for quaternions
    quaternion_v: Vec<Tensor>,
    /// Adam second moment for translations
    translation_v: Vec<Tensor>,
    /// Learning rate for pose optimization
    lr: f32,
    /// Adam beta1
    beta1: f32,
    /// Adam beta2
    beta2: f32,
    /// Adam epsilon
    eps: f32,
    /// Current iteration (for bias correction)
    iteration: usize,
}

impl PoseEmbeddings {
    /// Create pose embeddings from a training dataset.
    ///
    /// Initializes from the poses in the dataset (camera-to-world form).
    pub fn from_dataset(
        poses: &[crate::ScenePose],
        lr: f32,
        device: &Device,
    ) -> candle_core::Result<Self> {
        let n = poses.len();
        let mut embeddings = Vec::with_capacity(n);
        let zeros_4 = Tensor::zeros((4,), DType::F32, device)?;
        let zeros_3 = Tensor::zeros((3,), DType::F32, device)?;

        for (frame_idx, scene_pose) in poses.iter().enumerate() {
            let embedding = PoseEmbedding::from_se3(&scene_pose.pose, frame_idx, device)?;
            embeddings.push(embedding);
        }

        let quaternion_m = vec![zeros_4.clone(); n];
        let translation_m = vec![zeros_3.clone(); n];
        let quaternion_v = vec![zeros_4.clone(); n];
        let translation_v = vec![zeros_3; n];

        Ok(Self {
            embeddings,
            quaternion_m,
            translation_m,
            quaternion_v,
            translation_v,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            iteration: 0,
        })
    }

    /// Get the number of pose embeddings.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Get a pose embedding by frame index.
    pub fn get(&self, frame_idx: usize) -> Option<&PoseEmbedding> {
        self.embeddings.get(frame_idx)
    }

    /// Get a mutable pose embedding by frame index.
    pub fn get_mut(&mut self, frame_idx: usize) -> Option<&mut PoseEmbedding> {
        self.embeddings.get_mut(frame_idx)
    }

    /// Apply Adam step for pose parameters.
    ///
    /// This implements sparse Adam: only update poses that have gradients.
    /// The gradients are expected to be from backward pass through the camera.
    ///
    /// # Arguments
    /// * `frame_indices` - Indices of frames that were rendered this iteration
    /// * `quaternion_grads` - Quaternion gradients for those frames
    /// * `translation_grads` - Translation gradients for those frames
    pub fn adam_step(
        &mut self,
        frame_indices: &[usize],
        quaternion_grads: &[Tensor],
        translation_grads: &[Tensor],
    ) -> candle_core::Result<()> {
        if frame_indices.is_empty() {
            return Ok(());
        }

        self.iteration += 1;
        let t = self.iteration as f32;
        let bias_correction1 = 1.0 - self.beta1.powf(t);
        let bias_correction2 = 1.0 - self.beta2.powf(t);

        // Create scalar tensors for division
        let bc1_tensor = Tensor::new(bias_correction1, &Device::Cpu)?;
        let bc2_tensor = Tensor::new(bias_correction2, &Device::Cpu)?;
        let eps_tensor = Tensor::new(self.eps, &Device::Cpu)?;
        let lr_tensor = Tensor::new(self.lr, &Device::Cpu)?;

        for (i, &frame_idx) in frame_indices.iter().enumerate() {
            if frame_idx >= self.embeddings.len() {
                continue;
            }

            let embedding = &mut self.embeddings[frame_idx];

            // Update quaternion
            let q_grad = &quaternion_grads[i];
            let q_m = &self.quaternion_m[frame_idx];
            let q_v = &self.quaternion_v[frame_idx];

            let beta1_t = Tensor::new(self.beta1, q_grad.device())?;
            let one_minus_beta1 = Tensor::new(1.0 - self.beta1, q_grad.device())?;
            let beta2_t = Tensor::new(self.beta2, q_grad.device())?;
            let one_minus_beta2 = Tensor::new(1.0 - self.beta2, q_grad.device())?;

            let q_m_new = q_m.mul(&beta1_t)?.add(&q_grad.mul(&one_minus_beta1)?)?;
            let q_v_new = q_v.mul(&beta2_t)?.add(&q_grad.sqr()?.mul(&one_minus_beta2)?)?;

            let q_m_hat = q_m_new.div(&bc1_tensor)?;
            let q_v_hat = q_v_new.div(&bc2_tensor)?;

            let q_update = q_m_hat.div(&q_v_hat.sqrt()?.add(&eps_tensor)?)?.mul(&lr_tensor)?;
            let q_new = embedding.quaternion.as_tensor().sub(&q_update)?;

            // Normalize quaternion to maintain unit length
            let q_norm = q_new.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            let q_normalized = if q_norm > 1e-8 {
                q_new.div(&Tensor::new(q_norm, q_new.device())?)?
            } else {
                q_new
            };

            embedding.quaternion = Var::from_tensor(&q_normalized)?;
            self.quaternion_m[frame_idx] = q_m_new;
            self.quaternion_v[frame_idx] = q_v_new;

            // Update translation
            let t_grad = &translation_grads[i];
            let t_m = &self.translation_m[frame_idx];
            let t_v = &self.translation_v[frame_idx];

            let t_m_new = t_m.mul(&beta1_t)?.add(&t_grad.mul(&one_minus_beta1)?)?;
            let t_v_new = t_v.mul(&beta2_t)?.add(&t_grad.sqr()?.mul(&one_minus_beta2)?)?;

            let t_m_hat = t_m_new.div(&bc1_tensor)?;
            let t_v_hat = t_v_new.div(&bc2_tensor)?;

            let t_update = t_m_hat.div(&t_v_hat.sqrt()?.add(&eps_tensor)?)?.mul(&lr_tensor)?;
            let t_new = embedding.translation.as_tensor().sub(&t_update)?;

            embedding.translation = Var::from_tensor(&t_new)?;
            self.translation_m[frame_idx] = t_m_new;
            self.translation_v[frame_idx] = t_v_new;
        }

        Ok(())
    }

    /// Get all poses as SE3 (camera-to-world).
    pub fn to_se3_poses(&self) -> candle_core::Result<Vec<SE3>> {
        self.embeddings.iter().map(|e| e.to_se3()).collect()
    }
}

/// Compute pose gradients from loss via finite differences.
///
/// This is a fallback when analytical gradients are not available.
/// Uses central differences to approximate gradients for quaternion and translation.
#[cfg(feature = "gpu")]
pub fn compute_pose_gradients_fd<F>(
    embedding: &PoseEmbedding,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    width: usize,
    height: usize,
    loss_fn: F,
    device: &Device,
) -> candle_core::Result<(Tensor, Tensor)>
where
    F: Fn(&DiffCamera) -> candle_core::Result<f32>,
{
    const EPS: f32 = 1e-4;

    let q_vec = embedding.quaternion.as_tensor().to_vec1::<f32>()?;
    let t_vec = embedding.translation.as_tensor().to_vec1::<f32>()?;

    let q: [f32; 4] = [q_vec[0], q_vec[1], q_vec[2], q_vec[3]];
    let t: [f32; 3] = [t_vec[0], t_vec[1], t_vec[2]];

    let mut q_grads = [0.0f32; 4];
    let mut t_grads = [0.0f32; 3];

    // Compute quaternion gradients
    for i in 0..4 {
        let q_plus = {
            let mut q_new = q;
            q_new[i] += EPS;
            let embedding = PoseEmbedding::from_se3(
                &SE3::new(&[q_new[1], q_new[2], q_new[3], q_new[0]], &t),
                embedding.frame_idx(),
                device,
            )?;
            embedding.to_diff_camera(fx, fy, cx, cy, width, height, device)?
        };
        let q_minus = {
            let mut q_new = q;
            q_new[i] -= EPS;
            let embedding = PoseEmbedding::from_se3(
                &SE3::new(&[q_new[1], q_new[2], q_new[3], q_new[0]], &t),
                embedding.frame_idx(),
                device,
            )?;
            embedding.to_diff_camera(fx, fy, cx, cy, width, height, device)?
        };

        let loss_plus = loss_fn(&q_plus)?;
        let loss_minus = loss_fn(&q_minus)?;
        q_grads[i] = (loss_plus - loss_minus) / (2.0 * EPS);
    }

    // Compute translation gradients
    for i in 0..3 {
        let t_plus = {
            let mut t_new = t;
            t_new[i] += EPS;
            let embedding = PoseEmbedding::from_se3(
                &SE3::new(&[q[1], q[2], q[3], q[0]], &t_new),
                embedding.frame_idx(),
                device,
            )?;
            embedding.to_diff_camera(fx, fy, cx, cy, width, height, device)?
        };
        let t_minus = {
            let mut t_new = t;
            t_new[i] -= EPS;
            let embedding = PoseEmbedding::from_se3(
                &SE3::new(&[q[1], q[2], q[3], q[0]], &t_new),
                embedding.frame_idx(),
                device,
            )?;
            embedding.to_diff_camera(fx, fy, cx, cy, width, height, device)?
        };

        let loss_plus = loss_fn(&t_plus)?;
        let loss_minus = loss_fn(&t_minus)?;
        t_grads[i] = (loss_plus - loss_minus) / (2.0 * EPS);
    }

    Ok((
        Tensor::from_slice(&q_grads, (4,), device)?,
        Tensor::from_slice(&t_grads, (3,), device)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SE3;

    #[test]
    fn test_pose_embedding_identity() {
        let device = Device::Cpu;
        let embedding = PoseEmbedding::identity(0, &device).unwrap();

        let pose = embedding.to_se3().unwrap();
        let expected = SE3::identity();

        assert!((pose.quaternion()[0] - expected.quaternion()[0]).abs() < 1e-6);
        assert!((pose.quaternion()[1] - expected.quaternion()[1]).abs() < 1e-6);
        assert!((pose.quaternion()[2] - expected.quaternion()[2]).abs() < 1e-6);
        assert!((pose.quaternion()[3] - expected.quaternion()[3]).abs() < 1e-6);
    }

    #[test]
    fn test_pose_embedding_roundtrip() {
        let device = Device::Cpu;

        // Create a non-trivial pose
        let q = [0.0, 0.0, 0.0, 1.0]; // Identity quaternion (qx, qy, qz, qw)
        let t = [1.0, 2.0, 3.0]; // Translation
        let original = SE3::new(&q, &t);

        let embedding = PoseEmbedding::from_se3(&original, 0, &device).unwrap();
        let recovered = embedding.to_se3().unwrap();

        // Check quaternion (need to compare in same order)
        let orig_q = original.quaternion();
        let rec_q = recovered.quaternion();
        for i in 0..4 {
            assert!((orig_q[i] - rec_q[i]).abs() < 1e-5, "Quaternion mismatch at {}", i);
        }

        // Check translation
        let orig_t = original.translation();
        let rec_t = recovered.translation();
        for i in 0..3 {
            assert!((orig_t[i] - rec_t[i]).abs() < 1e-5, "Translation mismatch at {}", i);
        }
    }
}