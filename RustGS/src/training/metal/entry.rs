use super::clustering::ClusterAssignment;
use super::data_loading::load_training_data;
use super::events::{TrainingRun, TrainingRunReport};
use super::memory::{
    affordable_initial_gaussian_cap, assess_memory_estimate, bytes_to_gib,
    estimate_peak_memory_with_source_pixels, preflight_initial_gaussian_cap,
    training_memory_budget, MetalMemoryDecision,
};
use super::pose_embedding::PoseEmbeddings;
use super::splats::HostSplats;
use super::trainer::MetalTrainer;
use super::{LiteGsConfig, TrainingConfig, TrainingProfile};
use crate::{TrainingDataset, TrainingError};
use std::time::Instant;

pub(crate) fn effective_metal_config(config: &TrainingConfig) -> TrainingConfig {
    let mut effective = config.clone();
    if effective.training_profile == TrainingProfile::LiteGsMacV1 {
        let defaults = TrainingConfig::default();
        let litegs_defaults = LiteGsConfig::default();
        let mut aligned = Vec::new();

        if (effective.lr_position - defaults.lr_position).abs() < f32::EPSILON {
            effective.lr_position = 2e-5;
            aligned.push("lr_position=2e-5");
        }
        if (effective.lr_pos_final - defaults.lr_pos_final).abs() < f32::EPSILON {
            effective.lr_pos_final = 2e-7;
            aligned.push("lr_pos_final=2e-7");
        }
        if (effective.lr_scale - defaults.lr_scale).abs() < f32::EPSILON {
            effective.lr_scale = 7e-3;
            aligned.push("lr_scale=7e-3");
        }
        if (effective.lr_rotation - defaults.lr_rotation).abs() < f32::EPSILON {
            effective.lr_rotation = 2e-3;
            aligned.push("lr_rotation=2e-3");
        }
        if (effective.lr_opacity - defaults.lr_opacity).abs() < f32::EPSILON {
            effective.lr_opacity = 0.012;
            aligned.push("lr_opacity=0.012");
        }
        if (effective.lr_color - defaults.lr_color).abs() < f32::EPSILON {
            effective.lr_color = 0.002;
            aligned.push("lr_color=0.002");
        }
        if effective.litegs.refine_every == litegs_defaults.refine_every {
            effective.litegs.refine_every = 200;
            aligned.push("litegs.refine_every=200");
        }
        if (effective.litegs.growth_grad_threshold - litegs_defaults.growth_grad_threshold).abs()
            < f32::EPSILON
        {
            effective.litegs.growth_grad_threshold = 0.003;
            aligned.push("litegs.growth_grad_threshold=0.003");
        }
        if (effective.litegs.growth_select_fraction - litegs_defaults.growth_select_fraction).abs()
            < f32::EPSILON
        {
            effective.litegs.growth_select_fraction = 0.2;
            aligned.push("litegs.growth_select_fraction=0.2");
        }

        if !aligned.is_empty() {
            log::info!(
                "LiteGS Mac V1 aligned effective defaults to Brush-compatible values: {}",
                aligned.join(", ")
            );
        }
    }
    if effective.training_profile != TrainingProfile::LiteGsMacV1 && effective.lr_rotation != 0.0 {
        log::warn!(
            "Legacy Metal training still freezes Gaussian rotations; overriding lr_rotation from {} to 0.0 for this run.",
            effective.lr_rotation
        );
        effective.lr_rotation = 0.0;
    }
    effective
}

pub(crate) fn train_splats_with_report(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<TrainingRun, TrainingError> {
    let start = Instant::now();
    let device = crate::require_metal_device()?;
    let mut effective_config = effective_metal_config(config);
    let mut loaded = load_training_data(dataset, &effective_config, &device)?;

    if loaded.initial_splats.is_empty() {
        return Err(TrainingError::InvalidInput(
            "training initialization produced zero Gaussians".to_string(),
        ));
    }

    let mut trainer = MetalTrainer::new(
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        &effective_config,
        device,
    )?;
    let memory_budget = training_memory_budget();
    let frame_count = loaded.cameras.len();
    log::info!(
        "MetalTrainer preflight | gaussians={} | frames={} | pixels={} | gaussian_batch_size={} | estimated_peak≈{:.1} GiB | budget={} | dominant={}",
        loaded.initial_splats.len(),
        frame_count,
        trainer.pixel_count(),
        trainer.batch_size(),
        bytes_to_gib(
            estimate_peak_memory_with_source_pixels(
                loaded.initial_splats.len(),
                trainer.pixel_count(),
                trainer.source_pixel_count(),
                frame_count,
                trainer.batch_size(),
            )
            .total_bytes()
        ),
        memory_budget.describe(),
        estimate_peak_memory_with_source_pixels(
            loaded.initial_splats.len(),
            trainer.pixel_count(),
            trainer.source_pixel_count(),
            frame_count,
            trainer.batch_size(),
        )
        .top_components_summary(3),
    );
    let skip_memory_guard = std::env::var_os("RUSTGS_SKIP_METAL_MEMORY_GUARD").is_some();
    trainer.set_topology_memory_budget(if skip_memory_guard {
        None
    } else {
        Some(memory_budget)
    });
    let affordable_cap = affordable_initial_gaussian_cap(
        effective_config
            .max_initial_gaussians
            .max(loaded.initial_splats.len()),
        trainer.pixel_count(),
        trainer.source_pixel_count(),
        frame_count,
        trainer.batch_size(),
        &memory_budget,
    );
    if !skip_memory_guard && affordable_cap > 0 && loaded.initial_splats.len() > affordable_cap {
        let initial_cap =
            preflight_initial_gaussian_cap(effective_config.training_profile, affordable_cap);
        log::warn!(
            "MetalTrainer preflight lowered initial_gaussians from {} to {} for this run to fit the safe memory budget using even coverage downsampling. Growth budget remains capped at {}. Set RUSTGS_SKIP_METAL_MEMORY_GUARD=1 to keep the larger initialization.",
            loaded.initial_splats.len(),
            initial_cap,
            affordable_cap,
        );
        loaded.initial_splats.downsample_evenly(initial_cap);
        effective_config.max_initial_gaussians = initial_cap;
    }
    trainer.set_max_gaussian_budget(if skip_memory_guard {
        if effective_config.training_profile == TrainingProfile::LiteGsMacV1 {
            effective_config
                .litegs
                .target_primitives
                .max(loaded.initial_splats.len())
        } else {
            effective_config
                .max_initial_gaussians
                .max(loaded.initial_splats.len())
        }
    } else {
        let profile_cap = if effective_config.training_profile == TrainingProfile::LiteGsMacV1 {
            effective_config
                .litegs
                .target_primitives
                .max(loaded.initial_splats.len())
        } else {
            affordable_cap.max(loaded.initial_splats.len())
        };
        profile_cap.min(affordable_cap.max(loaded.initial_splats.len()))
    });
    let estimated_peak = estimate_peak_memory_with_source_pixels(
        loaded.initial_splats.len(),
        trainer.pixel_count(),
        trainer.source_pixel_count(),
        frame_count,
        trainer.batch_size(),
    );
    match assess_memory_estimate(&estimated_peak, &memory_budget) {
        MetalMemoryDecision::Allow => {
            let headroom = memory_budget
                .safe_bytes
                .saturating_sub(estimated_peak.total_bytes());
            log::info!(
                "MetalTrainer preflight passed with {:.1} GiB headroom",
                bytes_to_gib(headroom)
            );
        }
        MetalMemoryDecision::Warn => {
            log::warn!(
                "MetalTrainer preflight is close to the safe memory budget; recommendations: {}",
                estimated_peak.recommendations().join("; ")
            );
        }
        MetalMemoryDecision::Block if skip_memory_guard => {
            log::warn!(
                "MetalTrainer preflight exceeds the safe memory budget but RUSTGS_SKIP_METAL_MEMORY_GUARD=1 is set; continuing anyway. Recommendations: {}",
                estimated_peak.recommendations().join("; ")
            );
        }
        MetalMemoryDecision::Block => {
            return Err(TrainingError::TrainingFailed(format!(
                "metal backend is estimated to need about {:.1} GiB for a single training step, above the safe budget of {}. Dominant terms: {}. Recommendations: {}. Set RUSTGS_SKIP_METAL_MEMORY_GUARD=1 to bypass this guard.",
                bytes_to_gib(estimated_peak.total_bytes()),
                memory_budget.describe(),
                estimated_peak.top_components_summary(3),
                estimated_peak.recommendations().join("; ")
            )));
        }
    }
    let mut gaussians = loaded.initial_splats.upload(trainer.device())?;
    trainer.set_scene_extent(loaded.initial_splats.scene_extent());

    if config.litegs.learnable_viewproj && config.litegs.lr_pose > 0.0 {
        let pose_embeddings =
            PoseEmbeddings::from_dataset(&dataset.poses, config.litegs.lr_pose, trainer.device())?;
        log::info!(
            "MetalTrainer initialized {} learnable camera poses with lr={}",
            pose_embeddings.len(),
            config.litegs.lr_pose
        );
        trainer.set_pose_embeddings(pose_embeddings);
    }

    if config.litegs.cluster_size > 0 {
        let positions = loaded.initial_splats.positions_vec3();
        let assignment = ClusterAssignment::assign_spatial_hash(
            &positions,
            config.litegs.cluster_size,
            loaded.initial_splats.scene_extent(),
        );
        log::info!(
            "MetalTrainer initialized {} spatial clusters with target size {}",
            assignment.num_clusters,
            config.litegs.cluster_size
        );
        trainer.set_cluster_assignment(assignment);
    }

    if gaussians.len() == 0 {
        return Err(TrainingError::InvalidInput(
            "training initialization produced zero Gaussians".to_string(),
        ));
    }
    let stats = trainer.train_loaded(&mut gaussians, &loaded, config.iterations)?;
    let trained_splats = HostSplats::from_runtime(&gaussians)?;
    let training_elapsed = start.elapsed();
    let (render_width, render_height) = trainer.render_dimensions();

    log::info!(
        "Metal backend complete in {:.2}s | frames={} | render={}x{} | initial_gaussians={} | final_gaussians={} | final_loss_mean={:.6} | last_step_loss={:.6}",
        training_elapsed.as_secs_f64(),
        dataset.poses.len(),
        render_width,
        render_height,
        loaded.initial_splats.len(),
        trained_splats.len(),
        stats.final_loss,
        stats.final_step_loss,
    );

    Ok(TrainingRun {
        report: TrainingRunReport {
            elapsed: training_elapsed,
            final_loss: Some(stats.final_loss),
            final_step_loss: Some(stats.final_step_loss),
            gaussian_count: trained_splats.len(),
            sh_degree: trained_splats.sh_degree(),
            telemetry: Some(stats.telemetry),
        },
        splats: trained_splats,
    })
}
