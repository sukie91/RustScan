//! Arcball camera controller for 3D navigation.

use crate::renderer::scene::SceneBounds;
use glam::{Mat4, Vec3};

/// Arcball camera: orbits around a target point.
#[derive(Debug, Clone)]
pub struct ArcballCamera {
    /// World-space look-at target.
    pub target: Vec3,
    /// Distance from target to eye.
    pub distance: f32,
    /// Horizontal rotation angle in radians.
    pub yaw: f32,
    /// Vertical rotation angle in radians. Clamped to avoid gimbal.
    pub pitch: f32,
    /// Vertical field-of-view in radians.
    pub fov_y: f32,
}

impl Default for ArcballCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 5.0,
            yaw: 0.0,
            pitch: 0.3,
            fov_y: std::f32::consts::FRAC_PI_4,
        }
    }
}

impl ArcballCamera {
    const MIN_DISTANCE: f32 = 0.1;
    const ORBIT_SENSITIVITY: f32 = 0.005;
    const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
    const ZOOM_FACTOR: f32 = 1.1;
    const PAN_SCALE: f32 = 0.001;
    const FIT_MARGIN: f32 = 1.5;

    /// Eye position in world space.
    pub fn eye(&self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        let offset = Vec3::new(
            cos_pitch * self.yaw.sin(),
            self.pitch.sin(),
            cos_pitch * self.yaw.cos(),
        ) * self.distance;
        self.target + offset
    }

    /// View matrix (right-handed, looking from eye toward target).
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    /// Perspective projection matrix.
    pub fn proj_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect, 0.01, 1000.0)
    }

    /// Combined view-projection matrix.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj_matrix(aspect) * self.view_matrix()
    }

    /// Orbit around target (left-button drag).
    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        // Match common DCC/model-viewer interaction:
        // dragging the pointer should make the scene appear to move
        // in the same direction as the drag.
        self.yaw -= delta_x * Self::ORBIT_SENSITIVITY;
        self.pitch += delta_y * Self::ORBIT_SENSITIVITY;
        self.pitch = self.pitch.clamp(-Self::MAX_PITCH, Self::MAX_PITCH);
    }

    /// Pan the target point (right-button drag).
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let scale = self.distance * Self::PAN_SCALE;
        let right = self.view_matrix().col(0).truncate();
        let up = self.view_matrix().col(1).truncate();
        self.target -= right * delta_x * scale;
        self.target += up * delta_y * scale;
    }

    /// Zoom by scaling distance (scroll wheel).
    pub fn zoom(&mut self, delta: f32) {
        if delta > 0.0 {
            self.distance /= Self::ZOOM_FACTOR;
        } else if delta < 0.0 {
            self.distance *= Self::ZOOM_FACTOR;
        }
        self.distance = self.distance.max(Self::MIN_DISTANCE);
    }

    /// Fit the camera to display the full scene bounds.
    pub fn fit_scene(&mut self, bounds: &SceneBounds) {
        if !bounds.is_valid() {
            return;
        }
        let center = bounds.center();
        self.target = Vec3::new(center[0], center[1], center[2]);
        let diag = bounds.diagonal();
        self.distance = (diag * Self::FIT_MARGIN).max(Self::MIN_DISTANCE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eye_position() {
        let cam = ArcballCamera {
            target: Vec3::ZERO,
            distance: 5.0,
            yaw: 0.0,
            pitch: 0.0,
            fov_y: std::f32::consts::FRAC_PI_4,
        };
        let eye = cam.eye();
        assert!(
            (eye.z - 5.0).abs() < 1e-4,
            "eye.z should be ~5.0, got {}",
            eye.z
        );
        assert!(eye.x.abs() < 1e-4);
        assert!(eye.y.abs() < 1e-4);
    }

    #[test]
    fn test_pitch_clamp() {
        let mut cam = ArcballCamera::default();
        // Orbit far past PI/2
        cam.orbit(0.0, -100000.0);
        assert!(cam.pitch <= ArcballCamera::MAX_PITCH + 1e-4);
        assert!(cam.pitch >= -ArcballCamera::MAX_PITCH - 1e-4);
    }

    #[test]
    fn test_orbit_drag_direction_follows_pointer() {
        let mut cam = ArcballCamera::default();
        let initial_yaw = cam.yaw;
        let initial_pitch = cam.pitch;

        // Dragging right should rotate the view so the scene appears to move right.
        cam.orbit(100.0, 0.0);
        assert!(cam.yaw < initial_yaw);

        // Dragging up should rotate the view so the scene appears to move up.
        cam.orbit(0.0, -100.0);
        assert!(cam.pitch < initial_pitch);
    }

    #[test]
    fn test_zoom_clamp() {
        let mut cam = ArcballCamera::default();
        // Zoom in extremely
        for _ in 0..1000 {
            cam.zoom(1.0);
        }
        assert!(cam.distance >= ArcballCamera::MIN_DISTANCE);
    }
}
