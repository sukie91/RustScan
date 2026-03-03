//! Real-time Multi-threaded SLAM Pipeline
//!
//! Implements a three-thread architecture for real-time processing:
//! - Tracking Thread: High priority, processes frames at 30-60 FPS
//! - Mapping Thread: Medium priority, processes keyframes at 5-10 FPS
//! - Optimization Thread: Low priority, runs BA and 3DGS training at 1-2 FPS
//!
//! Architecture:
//! ```text
//! ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
//! │ Tracking     │ ──> │ Mapping      │ ──> │ Optimization │
//! │ (30-60 FPS) │     │ (5-10 FPS)   │     │ (1-2 FPS)   │
//! └──────────────┘     └──────────────┘     └──────────────┘
//!    高优先级              中优先级              低优先级
//! ```

use crossbeam_channel::{bounded, Sender, Receiver};
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};
use std::path::PathBuf;

use crate::core::{Camera, Frame as CoreFrame, FrameFeatures as CoreFrameFeatures, KeyFrame, Map, MapPoint, SE3};
use crate::fusion::{
    CompleteTrainer,
    DiffCamera,
    GaussianInitConfig,
    GaussianMapper,
    TrainableGaussians,
    initialize_trainable_gaussians_from_map,
};
use crate::io::{Dataset, Frame};
use crate::mapping::local_mapping::{LocalMapping, LocalMappingConfig};
use crate::optimizer::ba::{BACamera, BAObservation, BALandmark, BundleAdjuster};
use crate::pipeline::checkpoint::{CheckpointConfig, CheckpointManager, load_latest_checkpoint};
use candle_core::Device;
#[cfg(feature = "opencv")]
use std::path::Path;

/// Message sent from Tracking thread to Mapping thread
#[derive(Debug, Clone)]
pub struct TrackingMessage {
    /// Frame data
    pub frame: Frame,
    /// Estimated pose
    pub pose: SE3,
    /// Number of inliers
    pub num_inliers: usize,
    /// Whether tracking is successful
    pub success: bool,
    /// Extracted features (for keyframes)
    pub features: Option<CoreFrameFeatures>,
}

/// Message sent from Mapping thread to Optimization thread
#[derive(Debug, Clone)]
pub struct MappingMessage {
    /// Keyframe index
    pub keyframe_index: usize,
    /// Keyframe pose
    pub pose: SE3,
    /// Keyframe data
    pub frame: Frame,
    /// Optional map points snapshot for Gaussian initialization
    pub map_points: Option<Vec<MapPoint>>,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum frames in flight for tracking -> mapping channel
    pub track_map_channel_size: usize,
    /// Maximum keyframes in mapping -> optimization channel
    pub map_opt_channel_size: usize,
    /// Keyframe insertion interval (frames)
    pub keyframe_interval: usize,
    /// Whether to enable optimization thread
    pub enable_optimization: bool,
    /// Optional checkpoint directory
    pub checkpoint_dir: Option<PathBuf>,
    /// Checkpoint interval (frames)
    pub checkpoint_interval: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            track_map_channel_size: 16,
            map_opt_channel_size: 8,
            keyframe_interval: 5,
            enable_optimization: true,
            checkpoint_dir: None,
            checkpoint_interval: 50,
        }
    }
}

/// Pipeline state
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineState {
    /// Pipeline is not started
    Stopped,
    /// Pipeline is initializing
    Initializing,
    /// Pipeline is running
    Running,
    /// Pipeline encountered an error
    Error(String),
}

/// Real-time SLAM Pipeline
pub struct RealtimePipeline {
    /// Handle for tracking thread
    tracking_thread: Option<thread::JoinHandle<()>>,
    /// Handle for mapping thread
    mapping_thread: Option<thread::JoinHandle<()>>,
    /// Handle for optimization thread
    optimization_thread: Option<thread::JoinHandle<()>>,
    /// Stop flag shared across threads
    stop_flag: Arc<AtomicBool>,
    /// Current pipeline state
    state: PipelineState,
    /// Configuration
    config: PipelineConfig,
}

impl RealtimePipeline {
    /// Create a new pipeline with default configuration
    pub fn new() -> Self {
        Self::with_config(PipelineConfig::default())
    }

    /// Create a new pipeline with custom configuration
    pub fn with_config(config: PipelineConfig) -> Self {
        Self {
            tracking_thread: None,
            mapping_thread: None,
            optimization_thread: None,
            stop_flag: Arc::new(AtomicBool::new(false)),
            state: PipelineState::Stopped,
            config,
        }
    }

    /// Start the pipeline with a dataset
    pub fn start<D: Dataset + Send + 'static>(&mut self, dataset: D)
    where
        D: Dataset,
    {
        self.stop_flag.store(false, Ordering::SeqCst);

        // Create channels
        let (track_tx, track_rx) = bounded::<TrackingMessage>(self.config.track_map_channel_size);
        let (map_tx, map_rx) = bounded::<MappingMessage>(self.config.map_opt_channel_size);

        let keyframe_interval = self.config.keyframe_interval;
        let enable_optimization = self.config.enable_optimization;
        let camera = dataset.camera();
        let checkpoint_dir = self.config.checkpoint_dir.clone();
        let checkpoint_interval = self.config.checkpoint_interval;

        let mut resume_frame: Option<usize> = None;
        let mut resume_map: Option<Arc<RwLock<Map>>> = None;
        if let Some(dir) = &checkpoint_dir {
            match load_latest_checkpoint(dir) {
                Ok(Some(checkpoint)) => {
                    resume_frame = Some(checkpoint.frame_index);
                    resume_map = Some(Arc::new(RwLock::new(checkpoint.to_map())));
                    log::info!(
                        "Resuming from checkpoint at frame {}",
                        checkpoint.frame_index
                    );
                }
                Ok(None) => {}
                Err(err) => {
                    log::warn!("Failed to load checkpoint: {}", err);
                }
            }
        }

        // Clone stop flag for each thread
        let stop_tracking = Arc::clone(&self.stop_flag);
        let stop_mapping = Arc::clone(&self.stop_flag);
        let stop_optimization = Arc::clone(&self.stop_flag);

        // Spawn tracking thread
        let tracking_thread = thread::spawn(move || {
            tracking_thread_main(dataset, track_tx, stop_tracking, resume_frame);
        });

        // Spawn mapping thread
        let mapping_thread = thread::spawn(move || {
            mapping_thread_main(
                camera.clone(),
                track_rx,
                map_tx,
                stop_mapping,
                keyframe_interval,
                checkpoint_dir,
                checkpoint_interval,
                resume_map,
            );
        });

        // Spawn optimization thread (optional)
        let optimization_thread = if enable_optimization {
            Some(thread::spawn(move || {
                optimization_thread_main(map_rx, stop_optimization);
            }))
        } else {
            None
        };

        self.tracking_thread = Some(tracking_thread);
        self.mapping_thread = Some(mapping_thread);
        self.optimization_thread = optimization_thread;
        self.state = PipelineState::Running;
    }

    /// Stop the pipeline
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);

        if let Some(handle) = self.tracking_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.mapping_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.optimization_thread.take() {
            let _ = handle.join();
        }

        self.state = PipelineState::Stopped;
    }

    /// Get current pipeline state
    pub fn state(&self) -> &PipelineState {
        &self.state
    }

    #[cfg(feature = "opencv")]
    pub fn start_video<P: AsRef<Path>>(&mut self, path: P) -> Result<(), crate::io::VideoError> {
        let loader = crate::io::VideoLoader::open(path)?;
        self.start(loader);
        Ok(())
    }
}

impl Default for RealtimePipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for RealtimePipeline {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Tracking thread main function
fn tracking_thread_main<D: Dataset>(
    dataset: D,
    track_tx: Sender<TrackingMessage>,
    stop_flag: Arc<AtomicBool>,
    resume_frame: Option<usize>,
) {
    let camera = dataset.camera();
    let mut vo = crate::tracker::VisualOdometry::new(camera);
    let skip_until = resume_frame.unwrap_or(0);

    for frame_result in dataset.frames() {
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        let frame = match frame_result {
            Ok(f) => f,
            Err(_) => continue,
        };

        if frame.index < skip_until {
            continue;
        }

        let gray = rgb_to_grayscale(&frame.color, frame.width as usize, frame.height as usize);
        let vo_result = vo.process_frame(&gray, frame.width, frame.height);
        let features = if vo_result.success { vo.last_features() } else { None };

        let msg = TrackingMessage {
            frame,
            pose: vo_result.pose,
            num_inliers: vo_result.num_inliers,
            success: vo_result.success,
            features,
        };

        // Best-effort send to keep tracking thread real-time
        let _ = track_tx.try_send(msg);
    }
}

/// Mapping thread main function
fn mapping_thread_main(
    camera: Camera,
    track_rx: Receiver<TrackingMessage>,
    map_tx: Sender<MappingMessage>,
    stop_flag: Arc<AtomicBool>,
    keyframe_interval: usize,
    checkpoint_dir: Option<PathBuf>,
    checkpoint_interval: usize,
    resume_map: Option<Arc<RwLock<Map>>>,
) {
    let mut mapper = GaussianMapper::new(
        camera.width as usize,
        camera.height as usize,
    );
    let mut local_mapping = LocalMapping::new(LocalMappingConfig::default());
    local_mapping.set_camera(camera.clone());

    if let Some(shared_map) = resume_map {
        local_mapping.set_shared_map_and_init(shared_map);
    } else {
        local_mapping.set_map(Map::new());
    }

    let mut checkpoint_manager = checkpoint_dir
        .map(|dir| CheckpointConfig { dir, interval: checkpoint_interval })
        .and_then(|config| CheckpointManager::new(config).ok());

    let mut sent_gaussian_init = false;

    loop {
        // Check for stop signal
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        // Try to receive tracking message
        match track_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(msg) => {
                if msg.success && msg.frame.index % keyframe_interval == 0 {
                    if let Some(features) = msg.features.clone() {
                        let mut core_frame = CoreFrame::new(
                            msg.frame.index as u64,
                            msg.frame.timestamp,
                            msg.frame.width,
                            msg.frame.height,
                        );
                        core_frame.set_pose(msg.pose);
                        core_frame.mark_as_keyframe();

                        let keyframe = KeyFrame::new(core_frame, features);
                        local_mapping.insert_keyframe(keyframe);
                    }

                    let frame_index = msg.frame.index;
                    let frame = msg.frame;
                    if let Some(depth) = &frame.depth {
                        let color: Vec<[u8; 3]> = frame.color
                            .chunks(3)
                            .map(|c| [c[0], c[1], c[2]])
                            .collect();
                        let rot = msg.pose.rotation();
                        let t = msg.pose.translation();

                        let _ = mapper.update(
                            depth,
                            &color,
                            frame.width as usize,
                            frame.height as usize,
                            camera.focal.x,
                            camera.focal.y,
                            camera.principal.x,
                            camera.principal.y,
                            &rot,
                            &t,
                        );
                    }

                    let _ = map_tx.send(MappingMessage {
                        keyframe_index: frame_index,
                        pose: msg.pose,
                        frame,
                        map_points: if !sent_gaussian_init {
                            if let Some(map) = local_mapping.map() {
                                let points: Vec<MapPoint> = map.valid_points().cloned().collect();
                                if !points.is_empty() {
                                    sent_gaussian_init = true;
                                    Some(points)
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        },
                    });

                    if let (Some(manager), Some(map)) = (checkpoint_manager.as_mut(), local_mapping.map()) {
                        match manager.maybe_save(frame_index, &*map) {
                            Ok(path) => {
                                if !path.as_os_str().is_empty() {
                                    log::info!("Checkpoint saved: {}", path.display());
                                }
                            }
                            Err(err) => {
                                log::warn!("Failed to save checkpoint: {}", err);
                            }
                        }
                    }
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // Continue waiting
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                break;
            }
        }
    }
}

/// Optimization thread main function
fn optimization_thread_main(
    map_rx: Receiver<MappingMessage>,
    stop_flag: Arc<AtomicBool>,
) {
    let mut trainer: Option<CompleteTrainer> = None;
    let mut trainer_dims: Option<(usize, usize)> = None;
    let mut last_train = Instant::now();
    let mut gaussians: Option<TrainableGaussians> = None;

    let mut ba = BundleAdjuster::new();
    let mut ba_cameras = 0usize;
    let mut ba_landmarks = 0usize;
    let mut ba_observations = 0usize;

    loop {
        // Check for stop signal
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        // Try to receive mapping message
        match map_rx.recv_timeout(Duration::from_secs(1)) {
            Ok(msg) => {
                let frame = msg.frame;
                let (width, height) = (frame.width as usize, frame.height as usize);

                // Run lightweight BA (best-effort) on sampled depth points
                let (c_added, l_added, o_added) = add_ba_observations(
                    &mut ba,
                    &msg.pose,
                    &frame,
                    20,
                    200,
                );
                ba_cameras += c_added;
                ba_landmarks += l_added;
                ba_observations += o_added;

                if ba_cameras >= 2 && ba_observations >= 100 {
                    let _ = ba.optimize(5);
                    ba = BundleAdjuster::new();
                    ba_cameras = 0;
                    ba_landmarks = 0;
                    ba_observations = 0;
                }

                // 3DGS training step (best-effort)
                if last_train.elapsed() >= Duration::from_millis(500) {
                    if trainer_dims != Some((width, height)) {
                        trainer = Some(CompleteTrainer::new(
                            width,
                            height,
                            0.00016,
                            0.005,
                            0.001,
                            0.05,
                            0.0025,
                        ));
                        trainer_dims = Some((width, height));
                    }

                    if let Some(trainer) = trainer.as_mut() {
                        let device = trainer.device().clone();
                        if gaussians.is_none() {
                            if let Some(points) = msg.map_points.as_ref() {
                                gaussians = initialize_gaussians_from_points(points, &device);
                            }
                        }

                        if let Some(gaussians) = gaussians.as_mut() {
                            if let Some((camera, target_color, target_depth)) =
                                build_training_targets(&frame, &device)
                            {
                                let _ = trainer.training_step(
                                    gaussians,
                                    &camera,
                                    &target_color,
                                    &target_depth,
                                );
                            }
                        } else if let Some((mut gaussians, camera, target_color, target_depth)) =
                            build_training_batch(&frame, &device, 4, 5000)
                        {
                            let _ = trainer.training_step(
                                &mut gaussians,
                                &camera,
                                &target_color,
                                &target_depth,
                            );
                        }
                    }

                    last_train = Instant::now();
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // Continue waiting
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                break;
            }
        }
    }
}

fn initialize_gaussians_from_points(
    points: &[MapPoint],
    device: &Device,
) -> Option<TrainableGaussians> {
    if points.is_empty() {
        return None;
    }

    let mut map = Map::new();
    for point in points {
        map.insert_point_with_id(point.id, point.clone());
    }

    let config = GaussianInitConfig {
        opacity: 0.5,
        ..GaussianInitConfig::default()
    };

    initialize_trainable_gaussians_from_map(&map, &config, device).ok()
}

fn build_training_targets(
    frame: &Frame,
    device: &Device,
) -> Option<(DiffCamera, Vec<f32>, Vec<f32>)> {
    let depth = frame.depth.as_ref()?;
    let width = frame.width as usize;
    let height = frame.height as usize;
    if depth.len() != width * height {
        return None;
    }

    let fx = frame.camera.focal.x;
    let fy = frame.camera.focal.y;
    let cx = frame.camera.principal.x;
    let cy = frame.camera.principal.y;

    let rotation = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let translation = [0.0, 0.0, 0.0];
    let camera = DiffCamera::new(
        fx, fy, cx, cy,
        width, height,
        &rotation,
        &translation,
        device,
    ).ok()?;

    let target_color: Vec<f32> = frame.color
        .iter()
        .map(|v| *v as f32 / 255.0)
        .collect();

    Some((camera, target_color, depth.clone()))
}

fn build_training_batch(
    frame: &Frame,
    device: &Device,
    sample_step: usize,
    max_gaussians: usize,
) -> Option<(TrainableGaussians, DiffCamera, Vec<f32>, Vec<f32>)> {
    let depth = frame.depth.as_ref()?;
    let width = frame.width as usize;
    let height = frame.height as usize;
    if depth.len() != width * height {
        return None;
    }

    let fx = frame.camera.focal.x;
    let fy = frame.camera.focal.y;
    let cx = frame.camera.principal.x;
    let cy = frame.camera.principal.y;

    let mut positions = Vec::new();
    let mut scales = Vec::new();
    let mut rotations = Vec::new();
    let mut opacities = Vec::new();
    let mut colors = Vec::new();

    for y in (0..height).step_by(sample_step) {
        for x in (0..width).step_by(sample_step) {
            let idx = y * width + x;
            let z = depth[idx];
            if z <= 0.0 {
                continue;
            }

            let x_cam = (x as f32 - cx) * z / fx;
            let y_cam = (y as f32 - cy) * z / fy;

            positions.extend_from_slice(&[x_cam, y_cam, z]);
            scales.extend_from_slice(&[-3.0, -3.0, -3.0]);
            rotations.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            opacities.push(0.5);

            let cidx = idx * 3;
            if cidx + 2 < frame.color.len() {
                colors.push(frame.color[cidx] as f32 / 255.0);
                colors.push(frame.color[cidx + 1] as f32 / 255.0);
                colors.push(frame.color[cidx + 2] as f32 / 255.0);
            } else {
                colors.extend_from_slice(&[0.5, 0.5, 0.5]);
            }

            if positions.len() / 3 >= max_gaussians {
                break;
            }
        }
        if positions.len() / 3 >= max_gaussians {
            break;
        }
    }

    if positions.is_empty() {
        return None;
    }

    let gaussians = TrainableGaussians::new(
        &positions,
        &scales,
        &rotations,
        &opacities,
        &colors,
        device,
    ).ok()?;

    let (camera, target_color, target_depth) = build_training_targets(frame, device)?;

    Some((gaussians, camera, target_color, target_depth))
}

fn add_ba_observations(
    ba: &mut BundleAdjuster,
    pose: &SE3,
    frame: &Frame,
    sample_step: usize,
    max_points: usize,
) -> (usize, usize, usize) {
    let depth = match &frame.depth {
        Some(d) => d,
        None => return (0, 0, 0),
    };

    let width = frame.width as usize;
    let height = frame.height as usize;
    if depth.len() != width * height {
        return (0, 0, 0);
    }

    let fx = frame.camera.focal.x;
    let fy = frame.camera.focal.y;
    let cx = frame.camera.principal.x;
    let cy = frame.camera.principal.y;

    let cam_idx = ba.add_camera(
        BACamera::new(fx as f64, fy as f64, cx as f64, cy as f64)
            .with_pose(*pose)
    );

    let rot = pose.rotation();
    let t = pose.translation();

    let mut added_landmarks = 0usize;
    let mut added_obs = 0usize;

    'outer: for y in (0..height).step_by(sample_step) {
        for x in (0..width).step_by(sample_step) {
            let idx = y * width + x;
            let z = depth[idx];
            if z <= 0.0 {
                continue;
            }

            let x_cam = (x as f32 - cx) * z / fx;
            let y_cam = (y as f32 - cy) * z / fy;

            let wx = rot[0][0] * x_cam + rot[0][1] * y_cam + rot[0][2] * z + t[0];
            let wy = rot[1][0] * x_cam + rot[1][1] * y_cam + rot[1][2] * z + t[1];
            let wz = rot[2][0] * x_cam + rot[2][1] * y_cam + rot[2][2] * z + t[2];

            let lm_idx = ba.add_landmark(BALandmark::new(wx as f64, wy as f64, wz as f64));
            ba.add_observation(cam_idx, lm_idx, BAObservation::new(x as f64, y as f64));
            added_landmarks += 1;
            added_obs += 1;

            if added_landmarks >= max_points {
                break 'outer;
            }
        }
    }

    (1, added_landmarks, added_obs)
}

fn rgb_to_grayscale(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    let expected = width.saturating_mul(height).saturating_mul(3);
    if rgb.len() != expected {
        return Vec::new();
    }

    let mut gray = Vec::with_capacity(width * height);
    for c in rgb.chunks(3) {
        let r = c[0] as u16;
        let g = c[1] as u16;
        let b = c[2] as u16;
        let y = (30 * r + 59 * g + 11 * b) / 100;
        gray.push(y as u8);
    }
    gray
}

/// Builder for creating RealtimePipeline with fluent API
pub struct RealtimePipelineBuilder {
    config: PipelineConfig,
}

impl RealtimePipelineBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
        }
    }

    /// Set track-map channel size
    pub fn track_map_channel_size(mut self, size: usize) -> Self {
        self.config.track_map_channel_size = size;
        self
    }

    /// Set map-opt channel size
    pub fn map_opt_channel_size(mut self, size: usize) -> Self {
        self.config.map_opt_channel_size = size;
        self
    }

    /// Set keyframe interval
    pub fn keyframe_interval(mut self, interval: usize) -> Self {
        self.config.keyframe_interval = interval;
        self
    }

    /// Enable or disable optimization thread
    pub fn enable_optimization(mut self, enable: bool) -> Self {
        self.config.enable_optimization = enable;
        self
    }

    /// Set checkpoint directory
    pub fn checkpoint_dir<P: Into<PathBuf>>(mut self, dir: P) -> Self {
        self.config.checkpoint_dir = Some(dir.into());
        self
    }

    /// Set checkpoint interval (frames)
    pub fn checkpoint_interval(mut self, interval: usize) -> Self {
        self.config.checkpoint_interval = interval;
        self
    }

    /// Build the pipeline
    pub fn build(self) -> RealtimePipeline {
        RealtimePipeline::with_config(self.config)
    }
}

impl Default for RealtimePipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.track_map_channel_size, 16);
        assert_eq!(config.map_opt_channel_size, 8);
        assert_eq!(config.keyframe_interval, 5);
        assert!(config.enable_optimization);
        assert!(config.checkpoint_dir.is_none());
        assert_eq!(config.checkpoint_interval, 50);
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = RealtimePipeline::new();
        assert_eq!(*pipeline.state(), PipelineState::Stopped);
    }

    #[test]
    fn test_pipeline_builder() {
        let pipeline = RealtimePipelineBuilder::new()
            .track_map_channel_size(32)
            .map_opt_channel_size(16)
            .keyframe_interval(10)
            .enable_optimization(false)
            .build();

        assert_eq!(*pipeline.state(), PipelineState::Stopped);
    }

    #[test]
    fn test_pipeline_state() {
        let mut pipeline = RealtimePipeline::new();
        assert_eq!(*pipeline.state(), PipelineState::Stopped);

        // Pipeline should be stoppable even if not started
        pipeline.stop();
        assert_eq!(*pipeline.state(), PipelineState::Stopped);
    }

    #[test]
    fn test_message_sending() {
        let (tx, rx) = bounded::<TrackingMessage>(1);

        let camera = Camera::new(100.0, 100.0, 1.0, 1.0, 2, 2);
        let frame = Frame::new(
            0,
            0.0,
            vec![0u8; 2 * 2 * 3],
            None,
            camera,
            None,
        );
        let msg = TrackingMessage {
            frame,
            pose: SE3::identity(),
            num_inliers: 100,
            success: true,
            features: None,
        };

        tx.send(msg).unwrap();

        let received = rx.recv().unwrap();
        assert!(received.success);
        assert_eq!(received.num_inliers, 100);
    }

    #[test]
    fn test_ba_threshold_logic() {
        // Test BA threshold logic (>= 2 cameras and >= 100 observations)
        let test_cases = vec![
            (0, 0, false),
            (1, 100, false),
            (2, 99, false),
            (2, 100, true),
            (3, 150, true),
            (10, 1000, true),
        ];

        for (cameras, observations, should_run) in test_cases {
            let result = cameras >= 2 && observations >= 100;
            assert_eq!(
                result, should_run,
                "BA with {} cameras and {} observations: expected {}, got {}",
                cameras, observations, should_run, result
            );
        }
    }

    #[test]
    fn test_training_throttle() {
        // Test 500ms training interval logic
        let mut last_train = Instant::now();

        // Immediately after, should not train
        assert!(
            last_train.elapsed() < Duration::from_millis(500),
            "Should not train immediately"
        );

        // Wait 600ms
        std::thread::sleep(Duration::from_millis(600));

        // Now should train
        assert!(
            last_train.elapsed() >= Duration::from_millis(500),
            "Should train after 500ms"
        );

        // Update last_train
        last_train = Instant::now();

        // Should not train again immediately
        assert!(
            last_train.elapsed() < Duration::from_millis(500),
            "Should not train immediately after update"
        );
    }

    #[test]
    fn test_mapping_message_creation() {
        let camera = Camera::new(100.0, 100.0, 1.0, 1.0, 2, 2);
        let frame = Frame::new(
            0,
            0.0,
            vec![0u8; 2 * 2 * 3],
            None,
            camera,
            None,
        );

        let msg = MappingMessage {
            keyframe_index: 5,
            pose: SE3::identity(),
            frame,
            map_points: None,
        };

        assert_eq!(msg.keyframe_index, 5);
        assert_eq!(msg.frame.index, 0);
    }

    #[test]
    fn test_ba_initialization() {
        // Test BundleAdjuster creation
        let ba = BundleAdjuster::new();

        // BA should be created successfully
        // (We can't test much without adding data, but we can verify it doesn't panic)
        drop(ba);
    }

    #[test]
    fn test_message_channel_capacity() {
        // Test channel capacity limits
        let (tx, rx) = bounded::<MappingMessage>(2);

        let camera = Camera::new(100.0, 100.0, 1.0, 1.0, 2, 2);

        // Should be able to send 2 messages
        for i in 0..2 {
            let frame = Frame::new(
                i,
                i as f64,
                vec![0u8; 2 * 2 * 3],
                None,
                camera.clone(),
                None,
            );
            let msg = MappingMessage {
                keyframe_index: i,
                pose: SE3::identity(),
                frame,
                map_points: None,
            };
            tx.send(msg).unwrap();
        }

        // Third message should block (we won't actually send it)
        // Just verify we can receive the first two
        let msg1 = rx.recv().unwrap();
        assert_eq!(msg1.keyframe_index, 0);

        let msg2 = rx.recv().unwrap();
        assert_eq!(msg2.keyframe_index, 1);
    }
}
