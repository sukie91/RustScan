//! Pure Rust feature detection algorithms
//! 
//! Implements Harris corner detection and FAST detector in pure Rust.

use glam::Vec3;

/// A detected keypoint
#[derive(Debug, Clone, Copy)]
pub struct KeyPoint {
    /// x coordinate (pixel)
    pub x: f32,
    /// y coordinate (pixel)
    pub y: f32,
    /// Response (corner strength for Harris)
    pub response: f32,
    /// Scale (for multi-scale detection)
    pub scale: f32,
    /// Orientation in radians
    pub angle: f32,
}

impl KeyPoint {
    /// Create a new keypoint
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            response: 0.0,
            scale: 1.0,
            angle: 0.0,
        }
    }

    /// Create with response
    pub fn with_response(x: f32, y: f32, response: f32) -> Self {
        Self {
            x,
            y,
            response,
            scale: 1.0,
            angle: 0.0,
        }
    }
}

/// Harris corner detector parameters
#[derive(Debug, Clone, Copy)]
pub struct HarrisParams {
    /// Sobel aperture size (must be 3, 5, or 7)
    pub aperture_size: i32,
    /// Free parameter of Harris detector
    pub k: f32,
    /// Threshold for corner detection
    pub threshold: f32,
    /// Non-maximum suppression radius
    pub nms_radius: i32,
}

impl Default for HarrisParams {
    fn default() -> Self {
        Self {
            aperture_size: 3,
            k: 0.04,
            threshold: 0.01,
            nms_radius: 3,
        }
    }
}

/// Harris corner detector
pub struct HarrisDetector {
    params: HarrisParams,
}

impl HarrisDetector {
    /// Create a new Harris detector
    pub fn new(params: HarrisParams) -> Self {
        Self { params }
    }

    /// Create with default parameters
    pub fn default() -> Self {
        Self::new(HarrisParams::default())
    }

    /// Detect corners in a grayscale image
    /// 
    /// Input image should be in row-major format (y fastest)
    pub fn detect(&self, image: &[u8], width: u32, height: u32) -> Vec<KeyPoint> {
        let w = width as usize;
        let h = height as usize;
        
        if image.len() != w * h {
            return Vec::new();
        }

        // Compute gradients
        let mut Ix = vec![0i32; w * h];
        let mut Iy = vec![0i32; w * h];
        
        self.compute_gradients(image, width, height, &mut Ix, &mut Iy);

        // Compute Ixx, Iyy, Ixy
        let mut Ixx = vec![0i64; w * h];
        let mut Iyy = vec![0i64; w * h];
        let mut Ixy = vec![0i64; w * h];
        
        for i in 1..(h - 1) {
            for j in 1..(w - 1) {
                let idx = i * w + j;
                let ix = Ix[idx] as i64;
                let iy = Iy[idx] as i64;
                Ixx[idx] = ix * ix;
                Iyy[idx] = iy * iy;
                Ixy[idx] = ix * iy;
            }
        }

        // Compute Harris response using Gaussian window
        let mut response = vec![0.0f32; w * h];
        let kernel_size = 5isize;
        let half_k = kernel_size / 2;
        
        for i in half_k..(h as isize - half_k) {
            for j in half_k..(w as isize - half_k) {
                let mut sum_ixx = 0i64;
                let mut sum_iyy = 0i64;
                let mut sum_ixy = 0i64;
                
                // Simple box filter (faster than Gaussian)
                for di in -half_k..=half_k {
                    for dj in -half_k..=half_k {
                        let idx = ((i + di) * w as isize + (j + dj)) as usize;
                        sum_ixx += Ixx[idx];
                        sum_iyy += Iyy[idx];
                        sum_ixy += Ixy[idx];
                    }
                }
                
                // Compute Harris response: det(M) - k * trace(M)^2
                let det = sum_ixx as f32 * sum_iyy as f32 - sum_ixy as f32 * sum_ixy as f32;
                let trace = sum_ixx as f32 + sum_iyy as f32;
                response[(i * w as isize + j) as usize] = det - self.params.k * trace * trace;
            }
        }

        // Non-maximum suppression and threshold
        self.nms_and_threshold(&response, width, height)
    }

    /// Compute image gradients using Sobel operator
    fn compute_gradients(&self, image: &[u8], width: u32, height: u32, Ix: &mut [i32], Iy: &mut [i32]) {
        let w = width as usize;
        let h = height as usize;
        
        for i in 1..(h - 1) {
            for j in 1..(w - 1) {
                // Sobel X
                let gx = 
                    -1 * image[(i-1)*w + (j-1)] as i32 +
                    -2 * image[i*w + (j-1)] as i32 +
                    -1 * image[(i+1)*w + (j-1)] as i32 +
                     1 * image[(i-1)*w + (j+1)] as i32 +
                     2 * image[i*w + (j+1)] as i32 +
                     1 * image[(i+1)*w + (j+1)] as i32;
                
                // Sobel Y
                let gy = 
                    -1 * image[(i-1)*w + (j-1)] as i32 +
                    -2 * image[(i-1)*w + j] as i32 +
                    -1 * image[(i-1)*w + (j+1)] as i32 +
                     1 * image[(i+1)*w + (j-1)] as i32 +
                     2 * image[(i+1)*w + j] as i32 +
                     1 * image[(i+1)*w + (j+1)] as i32;
                
                Ix[i * w + j] = gx;
                Iy[i * w + j] = gy;
            }
        }
    }

    /// Non-maximum suppression and thresholding
    fn nms_and_threshold(&self, response: &[f32], width: u32, height: u32) -> Vec<KeyPoint> {
        let w = width as usize;
        let h = height as usize;
        let r = self.params.nms_radius as isize;
        let thresh = self.params.threshold;
        
        let mut corners = Vec::new();
        
        for i in (r as usize + 1)..(h - r as usize - 1) {
            for j in (r as usize + 1)..(w - r as usize - 1) {
                let val = response[i * w + j];
                
                if val > thresh {
                    // Check if it's a local maximum
                    let mut is_max = true;
                    for di in -r..=r {
                        for dj in -r..=r {
                            if di == 0 && dj == 0 {
                                continue;
                            }
                            let ni = (i as isize + di) as usize;
                            let nj = (j as isize + dj) as usize;
                            if response[ni * w + nj] > val {
                                is_max = false;
                                break;
                            }
                        }
                        if !is_max {
                            break;
                        }
                    }
                    
                    if is_max {
                        corners.push(KeyPoint::with_response(j as f32, i as f32, val));
                    }
                }
            }
        }
        
        // Sort by response (strongest first)
        corners.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());
        
        corners
    }
}

/// FAST corner detector parameters
#[derive(Debug, Clone, Copy)]
pub struct FastParams {
    /// Threshold for pixel intensity difference
    pub threshold: u8,
    /// Minimum consecutive pixels for corner
    pub min_consecutive: usize,
    /// Non-maximum suppression
    pub nms: bool,
}

impl Default for FastParams {
    fn default() -> Self {
        Self {
            threshold: 10,
            min_consecutive: 9,
            nms: true,
        }
    }
}

/// FAST (Features from Accelerated Segment Test) detector
pub struct FastDetector {
    params: FastParams,
}

impl FastDetector {
    /// Create a new FAST detector
    pub fn new(params: FastParams) -> Self {
        Self { params }
    }

    /// Create with default parameters
    pub fn default() -> Self {
        Self::new(FastParams::default())
    }

    /// Detect corners in a grayscale image
    pub fn detect(&self, image: &[u8], width: u32, height: u32) -> Vec<KeyPoint> {
        let w = width as usize;
        let h = height as usize;
        
        if image.len() != w * h {
            return Vec::new();
        }

        let mut corners = Vec::new();
        let t = self.params.threshold as i32;
        
        // FAST corner detection (simplified)
        // Check 16 pixels in a circle around the center pixel
        let circle_16 = [
            (0isize, -3isize), (1isize, -3isize), (2isize, -2isize), (3isize, -1isize),
            (3isize, 0isize), (3isize, 1isize), (2isize, 2isize), (1isize, 3isize),
            (0isize, 3isize), (-1isize, 3isize), (-2isize, 2isize), (-3isize, 1isize),
            (-3isize, 0isize), (-3isize, -1isize), (-2isize, -2isize), (-1isize, -3isize),
        ];
        
        for i in 3isize..(h as isize - 3) {
            for j in 3isize..(w as isize - 3) {
                let center = image[(i * w as isize + j) as usize] as i32;
                let mut consecutive = 0;
                
                for (dx, dy) in &circle_16 {
                    let pixel = image[((i + dy) * w as isize + (j + dx)) as usize] as i32;
                    if (pixel - center).abs() > t {
                        consecutive += 1;
                        if consecutive >= self.params.min_consecutive {
                            corners.push(KeyPoint::new(j as f32, i as f32));
                            break;
                        }
                    } else {
                        consecutive = 0;
                    }
                }
            }
        }
        
        corners
    }
}

/// Simple image convolution kernels
pub struct Kernels;

impl Kernels {
    /// Gaussian kernel (simplified)
    pub fn gaussian(size: usize, sigma: f32) -> Vec<f32> {
        let mut kernel = Vec::with_capacity(size * size);
        let half = size / 2;
        
        for dy in 0..size {
            for dx in 0..size {
                let x = dx as f32 - half as f32;
                let y = dy as f32 - half as f32;
                let val = (-(x * x + y * y) / (2.0 * sigma * sigma)).exp();
                kernel.push(val);
            }
        }
        
        // Normalize
        let sum: f32 = kernel.iter().sum();
        for v in &mut kernel {
            *v /= sum;
        }
        
        kernel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harris_params() {
        let params = HarrisParams::default();
        assert_eq!(params.aperture_size, 3);
        assert_eq!(params.nms_radius, 3);
    }

    #[test]
    fn test_keypoint_creation() {
        let kp = KeyPoint::new(100.0, 200.0);
        assert_eq!(kp.x, 100.0);
        assert_eq!(kp.y, 200.0);
    }

    #[test]
    fn test_gaussian_kernel() {
        let kernel = Kernels::gaussian(5, 1.0);
        assert_eq!(kernel.len(), 25);
        
        // Check sum is approximately 1
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_harris_detector() {
        let detector = HarrisDetector::default();
        
        // Create a simple test image with corners
        let mut image = vec![0u8; 100 * 100];
        // Add a corner-like pattern
        for i in 40..60 {
            for j in 40..60 {
                image[i * 100 + j] = 255;
            }
        }
        
        let corners = detector.detect(&image, 100, 100);
        // Should detect at least some corners
        assert!(corners.len() > 0);
    }

    #[test]
    fn test_fast_detector() {
        let detector = FastDetector::default();
        
        // Create a simple test image
        let image = vec![128u8; 100 * 100];
        
        let corners = detector.detect(&image, 100, 100);
        // On uniform image, should find no corners
        assert_eq!(corners.len(), 0);
    }
}
