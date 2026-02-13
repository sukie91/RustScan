//! ORB feature extractor
//! 
//! This module provides the ORB feature extractor interface.
//! The actual OpenCV implementation requires opencv-rust with system OpenCV.
//! For now, this provides a placeholder that can be replaced with actual implementation.

use crate::features::base::{FeatureExtractor, FeatureError, KeyPoint, Descriptors};

/// ORB (Oriented FAST and Rotated BRIEF) feature extractor
/// 
/// This is the interface for ORB feature extraction.
/// 
/// To enable OpenCV support, ensure OpenCV 4.x is installed and
/// compile with the `opencv` feature:
/// 
/// ```toml
/// rustrslam = { version = "0.1", features = ["opencv"] }
/// ```
pub struct OrbExtractor {
    /// Number of features to extract
    num_features: usize,
    /// Scale factor between pyramid levels
    scale_factor: f64,
    /// Number of pyramid levels
    num_levels: i32,
    /// Edge threshold
    edge_threshold: i32,
    /// First level
    first_level: i32,
    /// WTA_K (points for BRIEF)
    wta_k: i32,
    /// Patch size
    patch_size: i32,
}

impl OrbExtractor {
    /// Create a new ORB extractor with default parameters
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            scale_factor: 1.2,
            num_levels: 8,
            edge_threshold: 31,
            first_level: 0,
            wta_k: 2,  // 2 points for each BRIEF comparison
            patch_size: 31,
        }
    }

    /// Create with custom parameters
    pub fn with_params(
        num_features: usize,
        scale_factor: f64,
        num_levels: i32,
    ) -> Self {
        Self {
            num_features,
            scale_factor,
            num_levels,
            edge_threshold: 31,
            first_level: 0,
            wta_k: 2,
            patch_size: 31,
        }
    }

    /// Full configuration
    pub fn configure(
        num_features: usize,
        scale_factor: f64,
        num_levels: i32,
        edge_threshold: i32,
        first_level: i32,
        wta_k: i32,
        patch_size: i32,
    ) -> Self {
        Self {
            num_features,
            scale_factor,
            num_levels,
            edge_threshold,
            first_level,
            wta_k,
            patch_size,
        }
    }

    /// Get parameters as tuple for reference
    pub fn params(&self) -> (
        usize,    // num_features
        f64,      // scale_factor  
        i32,      // num_levels
        i32,      // edge_threshold
        i32,      // first_level
        i32,      // wta_k
        i32,      // patch_size
    ) {
        (
            self.num_features,
            self.scale_factor,
            self.num_levels,
            self.edge_threshold,
            self.first_level,
            self.wta_k,
            self.patch_size,
        )
    }
}

impl Default for OrbExtractor {
    fn default() -> Self {
        Self::new(2000)
    }
}

impl FeatureExtractor for OrbExtractor {
    fn detect_and_compute(
        &mut self,
        _image: &[u8],
        _width: u32,
        _height: u32,
    ) -> Result<(Vec<KeyPoint>, Descriptors), FeatureError> {
        // Without OpenCV, return empty results
        // In production, this would call opencv-rust
        Ok((
            Vec::new(),
            Descriptors::new(),
        ))
    }

    fn detect(
        &mut self,
        _image: &[u8],
        _width: u32,
        _height: u32,
    ) -> Result<Vec<KeyPoint>, FeatureError> {
        Ok(Vec::new())
    }

    fn num_features(&self) -> usize {
        self.num_features
    }

    fn set_num_features(&mut self, num: usize) {
        self.num_features = num;
    }
}

// OpenCV implementation - enabled when opencv feature is available
#[cfg(feature = "opencv")]
mod opencv_impl {
    use super::*;
    use opencv::core::{Mat, Vector, no_array};
    use opencv::features::ORB;

    impl OrbExtractor {
        /// Detect and compute with OpenCV
        pub fn detect_and_compute_opencv(
            &mut self,
            image: &[u8],
            width: i32,
            height: i32,
        ) -> Result<(Vec<KeyPoint>, Descriptors), FeatureError> {
            // Create Mat from image buffer (grayscale)
            let mat = unsafe {
                Mat::new_rows_cols_with_data(
                    height,
                    width,
                    opencv::core::CV_8UC1,
                    image.as_ptr() as *mut std::ffi::c_void,
                    opencv::core::Mat_AUTO_STEP,
                )
            }.map_err(|e| FeatureError::OpenCV(e.to_string()))?;

            // Create ORB detector
            let mut orb = ORB::create(
                self.num_features as i32,
                self.scale_factor,
                self.num_levels,
                self.edge_threshold,
                self.first_level,
                self.patch_size,
                self.wta_k,
                0,  // score_type (0 = HARRIS, 1 = FAST)
            ).map_err(|e| FeatureError::OpenCV(e.to_string()))?;

            // Detect keypoints and compute descriptors
            let mut keypoints = Vector::<opencv::core::KeyPoint>::new();
            let mut descriptors = Mat::new().map_err(|e| FeatureError::OpenCV(e.to_string()))?;

            orb.detect_and_compute(
                &mat,
                &no_array(),
                &mut keypoints,
                &mut descriptors,
            ).map_err(|e| FeatureError::OpenCV(e.to_string()))?;

            // Convert OpenCV keypoints to our format
            let mut kps = Vec::with_capacity(keypoints.len());
            for kp in keypoints.iter() {
                let pt = kp.pt();
                kps.push(KeyPoint {
                    pt: (pt.x, pt.y),
                    size: kp.size() as f32,
                    angle: kp.angle() as f32,
                    response: kp.response() as f32,
                    octave: kp.octave(),
                });
            }

            // Convert descriptors
            let mut des_data = Vec::new();
            let descriptor_size = if !descriptors.empty() {
                descriptors.elem_size() as usize
            } else {
                32  // ORB uses 32 bytes
            };

            if !descriptors.data().is_empty() {
                let rows = descriptors.rows();
                if rows > 0 {
                    for i in 0..rows {
                        let row = descriptors.row(i).map_err(|e| FeatureError::OpenCV(e.to_string()))?;
                        let slice = row.data().map_err(|e| FeatureError::OpenCV(e.to_string()))?;
                        let len = row.elem_size() as usize * row.cols();
                        des_data.extend_from_slice(unsafe {
                            std::slice::from_raw_parts(slice, len)
                        });
                    }
                }
            }

            let des = Descriptors {
                data: des_data,
                size: descriptor_size,
                count: kps.len(),
            };

            Ok((kps, des))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orb_creation() {
        let orb = OrbExtractor::new(1000);
        assert_eq!(orb.num_features(), 1000);
    }

    #[test]
    fn test_orb_params() {
        let orb = OrbExtractor::with_params(500, 1.5, 4);
        let (num, scale, levels, _, _, _, _) = orb.params();
        assert_eq!(num, 500);
        assert!((scale - 1.5).abs() < 1e-6);
        assert_eq!(levels, 4);
    }

    #[test]
    fn test_orb_default() {
        let orb = OrbExtractor::default();
        assert_eq!(orb.num_features(), 2000);
    }

    #[test]
    fn test_orb_configure() {
        let orb = OrbExtractor::configure(
            3000,  // num_features
            1.5,   // scale_factor
            6,     // num_levels
            20,    // edge_threshold
            1,     // first_level
            3,     // wta_k
            25,    // patch_size
        );
        let (num, scale, levels, edge, first, wta, patch) = orb.params();
        assert_eq!(num, 3000);
        assert!((scale - 1.5).abs() < 1e-6);
        assert_eq!(levels, 6);
        assert_eq!(edge, 20);
        assert_eq!(first, 1);
        assert_eq!(wta, 3);
        assert_eq!(patch, 25);
    }
}
