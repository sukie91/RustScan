use super::*;
use crate::fusion::{GaussianCamera, GaussianRenderer};
use crate::fusion::training_pipeline::compute_psnr;
use std::path::PathBuf;
use std::time::Instant;
use tempfile::tempdir;

fn e2e_enabled() -> bool {
    match std::env::var("RUSTSCAN_E2E") {
        Ok(value) => matches!(value.as_str(), "1" | "true" | "yes"),
        Err(_) => false,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(value.as_str(), "1" | "true" | "yes"),
        Err(_) => default,
    }
}

fn find_test_video() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let candidate = dir.join("test_data").join("video").join("sofa.MOV");
        if candidate.exists() {
            return Some(candidate);
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

fn test_resolved_config(input: PathBuf) -> ResolvedConfig {
    ResolvedConfig {
        input,
        output: tempdir().unwrap().keep(),
        output_format: OutputFormat::Json,
        slam: SlamConfig::default(),
        video: VideoConfig {
            cache_capacity: 4,
            prefer_hardware: false,
        },
        log_format: LogFormat::Text,
        mesh_voxel_size: None,
    }
}

fn average_psnr(gaussian: &GaussianStageOutput, samples: usize) -> f32 {
    if gaussian.keyframes.is_empty() || gaussian.map.is_empty() {
        return 0.0;
    }

    let width = gaussian.camera.width as usize;
    let height = gaussian.camera.height as usize;
    let renderer = GaussianRenderer::new(width, height);

    let total = gaussian.keyframes.len();
    let samples = samples.max(1).min(total);
    let step = (total / samples).max(1);

    let mut sum = 0.0f32;
    let mut count = 0usize;

    for keyframe in gaussian.keyframes.iter().step_by(step).take(samples) {
        let camera = GaussianCamera::new(
            gaussian.camera.focal.x,
            gaussian.camera.focal.y,
            gaussian.camera.principal.x,
            gaussian.camera.principal.y,
        )
        .with_pose(keyframe.pose.rotation(), keyframe.pose.translation());

        let rendered = renderer.render(&gaussian.map, &camera);
        let rendered_f32: Vec<f32> = rendered
            .color
            .iter()
            .map(|c| *c as f32 / 255.0)
            .collect();
        let psnr = compute_psnr(&rendered_f32, &keyframe.color, rendered.width, rendered.height);
        sum += psnr;
        count += 1;
    }

    if count == 0 { 0.0 } else { sum / count as f32 }
}

#[test]
fn test_load_input_source_detects_kitti_dataset() {
    let dir = tempdir().unwrap();
    let image_dir = dir.path().join("image_0");
    std::fs::create_dir_all(&image_dir).unwrap();
    std::fs::write(
        dir.path().join("calib.txt"),
        "P0: 718.856 0.0 607.1928 0.0 0.0 718.856 185.2157 0.0 0.0 0.0 1.0 0.0\n",
    )
    .unwrap();
    std::fs::write(image_dir.join("000000.png"), b"dummy").unwrap();

    let mut resolved = test_resolved_config(dir.path().to_path_buf());
    resolved.slam.dataset.dataset_type = "kitti".to_string();

    let input = load_input_source(&resolved).unwrap();
    assert!(matches!(input, InputSource::Dataset(_)));
}

#[test]
fn test_load_input_source_detects_euroc_dataset() {
    let dir = tempdir().unwrap();
    let cam0 = dir.path().join("mav0").join("cam0");
    let data_dir = cam0.join("data");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::write(
        cam0.join("data.csv"),
        "#timestamp,filename\n1403636579763555584,1403636579763555584.png\n",
    )
    .unwrap();
    std::fs::write(data_dir.join("1403636579763555584.png"), b"dummy").unwrap();

    let mut resolved = test_resolved_config(dir.path().to_path_buf());
    resolved.slam.dataset.dataset_type = "euroc".to_string();

    let input = load_input_source(&resolved).unwrap();
    assert!(matches!(input, InputSource::Dataset(_)));
}

#[test]
fn test_end_to_end_pipeline_video() {
    if !e2e_enabled() {
        eprintln!("Skipping E2E pipeline test (set RUSTSCAN_E2E=1 to enable)");
        return;
    }

    let video_path = match find_test_video() {
        Some(path) => path,
        None => {
            eprintln!("Skipping E2E pipeline test (missing test_data/video/sofa.MOV)");
            return;
        }
    };

    let output_dir = tempdir().expect("create temp output dir");

    let mut slam = SlamConfig::default();
    slam.dataset.max_frames = env_usize("RUSTSCAN_E2E_MAX_FRAMES", 0);
    slam.dataset.stride = env_usize("RUSTSCAN_E2E_STRIDE", 1).max(1);
    slam.mapper.max_keyframes = env_usize("RUSTSCAN_E2E_MAX_KEYFRAMES", slam.mapper.max_keyframes)
        .max(1);
    slam.mapper.keyframe_interval = env_usize(
        "RUSTSCAN_E2E_KEYFRAME_INTERVAL",
        slam.mapper.keyframe_interval,
    )
    .max(1);

    let resolved = ResolvedConfig {
        input: video_path,
        output: output_dir.path().to_path_buf(),
        output_format: OutputFormat::Json,
        slam,
        video: VideoConfig {
            cache_capacity: env_usize("RUSTSCAN_E2E_CACHE_CAPACITY", 50).max(1),
            prefer_hardware: env_bool("RUSTSCAN_E2E_PREFER_HW", false),
        },
        log_format: LogFormat::Text,
        mesh_voxel_size: None,
    };

    let start = Instant::now();

    let decoded = decode_video(&resolved).expect("decode video");
    let slam_output = run_slam_stage(InputSource::Video(decoded), &resolved).expect("run SLAM stage");
    let tracking_ratio = slam_output.tracking_success_ratio.unwrap_or(0.0);
    let camera_count = slam_output.keyframes.len();

    let gaussian = run_gaussian_stage(slam_output, &resolved).expect("run 3DGS stage");
    let psnr = average_psnr(&gaussian, env_usize("RUSTSCAN_E2E_PSNR_SAMPLES", 5));

    let mesh_stats = run_mesh_stage(gaussian, &resolved.output, resolved.mesh_voxel_size)
        .expect("run mesh stage");

    let results = ResultsJson {
        status: "success".to_string(),
        input: Some(resolved.input.display().to_string()),
        output: Some(resolved.output.display().to_string()),
        processing_time_ms: start.elapsed().as_millis(),
        camera_count,
        mesh: MeshStats {
            vertex_count: mesh_stats.vertex_count,
            triangle_count: mesh_stats.triangle_count,
            isolated_triangle_percent: mesh_stats.isolated_triangle_percent,
        },
        error: None,
        diagnostics: build_diagnostics(
            Some(&resolved.input),
            Some(&resolved.output),
            None,
        ),
    };

    write_results(&results, &resolved.output, resolved.output_format)
        .expect("write results.json");

    let obj_path = resolved.output.join("mesh.obj");
    let ply_path = resolved.output.join("mesh.ply");
    let results_path = resolved.output.join("results.json");

    assert!(obj_path.exists(), "mesh.obj not found at {}", obj_path.display());
    assert!(ply_path.exists(), "mesh.ply not found at {}", ply_path.display());
    assert!(
        results_path.exists(),
        "results.json not found at {}",
        results_path.display()
    );

    let psnr_threshold = env_f32("RUSTSCAN_E2E_PSNR", 28.0);
    let tracking_threshold = env_f32("RUSTSCAN_E2E_TRACKING", 0.95);
    let isolated_threshold = env_f32("RUSTSCAN_E2E_ISOLATED_TRI_PCT", 1.0);
    let max_seconds = env_u64("RUSTSCAN_E2E_MAX_SECONDS", 30 * 60);

    assert!(
        psnr >= psnr_threshold,
        "PSNR {:.2} dB below threshold {:.2} dB",
        psnr,
        psnr_threshold
    );
    assert!(
        tracking_ratio >= tracking_threshold,
        "Tracking success {:.2}% below threshold {:.2}%",
        tracking_ratio * 100.0,
        tracking_threshold * 100.0
    );
    assert!(
        mesh_stats.isolated_triangle_percent <= isolated_threshold,
        "Isolated triangles {:.2}% exceeds threshold {:.2}%",
        mesh_stats.isolated_triangle_percent,
        isolated_threshold
    );
    assert!(
        start.elapsed().as_secs() <= max_seconds,
        "Processing time {}s exceeds threshold {}s",
        start.elapsed().as_secs(),
        max_seconds
    );
}
