//! Main application struct implementing eframe::App.

use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};

use eframe::egui::{self, Color32, Rect, Sense, TextureHandle, TextureOptions, Vec2};
use eframe::egui_wgpu;
use eframe::wgpu;
use rustgs::{ColmapConfig, HostSplats, TrainingConfig};

use crate::loader::checkpoint::LoadError;
use crate::loader::{
    load_colmap_training_dataset, map_training_dataset_to_scene, LoadedColmapDataset,
};
use crate::renderer::camera::ArcballCamera;
use crate::renderer::scene::{GaussianSplat, Scene};
use crate::renderer::ViewerCallback;
use crate::training::preview::{LivePreviewBridge, PreviewRenderStatus};
use crate::training::{TrainingControlOptions, TrainingManager, TrainingSessionEvent};
use crate::ui::panel::{draw_side_panel, DatasetUiSummary, PanelAction, UiState};
use crate::ui::theme::{
    overlay_bg, PANEL_BG, TEXT_PRIMARY, TEXT_SECONDARY, VIEWPORT_BG, WINDOW_BG,
};
use crate::ui::viewport::{draw_empty_state, draw_viewport_overlay};

#[derive(Debug, Clone, Copy)]
enum AssetLoadKind {
    Checkpoint,
    Gaussian,
    Mesh,
}

enum AppCommand {
    LoadAsset { kind: AssetLoadKind, path: PathBuf },
    ColmapLoaded(Result<LoadedColmapDataset, String>),
}

pub struct ViewerApp {
    scene: Arc<Mutex<Scene>>,
    camera: ArcballCamera,
    preview_camera: ArcballCamera,
    ui_state: UiState,
    loaded_colmap: Option<LoadedColmapDataset>,
    command_rx: mpsc::Receiver<AppCommand>,
    command_tx: mpsc::Sender<AppCommand>,
    training_manager: TrainingManager,
    preview_bridge: LivePreviewBridge,
    preview_texture: Option<TextureHandle>,
    preview_texture_size: Option<[usize; 2]>,
    preview_dirty: bool,
    /// Actual wgpu surface format read from eframe at startup.
    surface_format: wgpu::TextureFormat,
}

impl ViewerApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (command_tx, command_rx) = mpsc::channel();
        let surface_format = cc
            .wgpu_render_state
            .as_ref()
            .map(|rs| rs.target_format)
            .unwrap_or(wgpu::TextureFormat::Bgra8Unorm);
        Self {
            scene: Arc::new(Mutex::new(Scene::default())),
            camera: ArcballCamera::default(),
            preview_camera: ArcballCamera::default(),
            ui_state: UiState::default(),
            loaded_colmap: None,
            command_rx,
            command_tx,
            training_manager: TrainingManager::new(),
            preview_bridge: LivePreviewBridge::default(),
            preview_texture: None,
            preview_texture_size: None,
            preview_dirty: true,
            surface_format,
        }
    }

    fn poll_commands(&mut self) {
        while let Ok(command) = self.command_rx.try_recv() {
            match command {
                AppCommand::LoadAsset { kind, path } => self.handle_asset_load(kind, path),
                AppCommand::ColmapLoaded(result) => self.handle_colmap_loaded(result),
            }
        }
    }

    fn handle_asset_load(&mut self, kind: AssetLoadKind, path: PathBuf) {
        if let Ok(mut scene) = self.scene.lock() {
            let result: Result<(), LoadError> = match kind {
                AssetLoadKind::Checkpoint => {
                    crate::loader::checkpoint::load_checkpoint(&path, &mut scene)
                }
                AssetLoadKind::Gaussian => {
                    crate::loader::gaussian::load_gaussians(&path, &mut scene)
                }
                AssetLoadKind::Mesh => crate::loader::mesh::load_mesh(&path, &mut scene),
            };

            match result {
                Ok(()) => {
                    self.ui_state.load_error = None;
                    if scene.has_data() {
                        self.camera.fit_scene(&scene.bounds);
                    }
                }
                Err(err) => {
                    self.ui_state.load_error = Some(err.to_string());
                }
            }
        }
    }

    fn handle_colmap_loaded(&mut self, result: Result<LoadedColmapDataset, String>) {
        self.ui_state.is_loading = false;
        self.ui_state.loading_message = None;

        match result {
            Ok(loaded) => {
                if let Ok(mut scene) = self.scene.lock() {
                    let layers = scene.layers.clone();
                    *scene = Scene::default();
                    scene.layers = layers;
                    map_training_dataset_to_scene(&loaded.dataset, &mut scene);
                    if scene.has_data() {
                        self.camera.fit_scene(&scene.bounds);
                        self.preview_camera = self.camera.clone();
                    }
                }

                self.ui_state.load_error = None;
                self.ui_state.dataset_summary = Some(DatasetUiSummary {
                    root_path: loaded.summary.input_dir.display().to_string(),
                    frame_count: loaded.summary.frame_count,
                    sparse_point_count: loaded.summary.sparse_point_count,
                    width: loaded.summary.intrinsics.width,
                    height: loaded.summary.intrinsics.height,
                });
                self.ui_state.training_error = None;
                self.ui_state.preview_error = None;
                self.loaded_colmap = Some(loaded);
                self.preview_dirty = true;
            }
            Err(err) => {
                self.ui_state.load_error = Some(err);
            }
        }
    }

    fn process_panel_actions(&mut self, actions: Vec<PanelAction>) {
        for action in actions {
            match action {
                PanelAction::OpenCheckpoint => self.spawn_file_dialog(AssetLoadKind::Checkpoint),
                PanelAction::OpenGaussian => self.spawn_file_dialog(AssetLoadKind::Gaussian),
                PanelAction::OpenMesh => self.spawn_file_dialog(AssetLoadKind::Mesh),
                PanelAction::OpenColmap => self.spawn_colmap_load(),
                PanelAction::StartTraining => self.start_training(),
                PanelAction::StopTraining => self.stop_training(),
                PanelAction::AutoFitScene => {}
            }
        }
    }

    fn spawn_file_dialog(&self, kind: AssetLoadKind) {
        let tx = self.command_tx.clone();
        std::thread::spawn(move || {
            let dialog = match kind {
                AssetLoadKind::Checkpoint => rfd::FileDialog::new().add_filter("JSON", &["json"]),
                AssetLoadKind::Gaussian => rfd::FileDialog::new().add_filter("PLY", &["ply"]),
                AssetLoadKind::Mesh => rfd::FileDialog::new().add_filter("Mesh", &["obj", "ply"]),
            };

            if let Some(path) = dialog.pick_file() {
                let _ = tx.send(AppCommand::LoadAsset { kind, path });
            }
        });
    }

    fn spawn_colmap_load(&mut self) {
        self.ui_state.is_loading = true;
        self.ui_state.loading_message = Some("Loading COLMAP dataset…".to_string());
        self.ui_state.load_error = None;

        let tx = self.command_tx.clone();
        std::thread::spawn(move || {
            let Some(path) = rfd::FileDialog::new().pick_folder() else {
                return;
            };

            let result = load_colmap_training_dataset(&path, &ColmapConfig::default())
                .map_err(|err| err.to_string());
            let _ = tx.send(AppCommand::ColmapLoaded(result));
        });
    }

    fn start_training(&mut self) {
        let Some(loaded) = self.loaded_colmap.as_ref() else {
            self.ui_state.training_error =
                Some("Load a COLMAP dataset before training.".to_string());
            return;
        };

        let mut config = TrainingConfig::default();
        config.iterations = self.ui_state.training_controls.iterations;
        config.render_scale = self.ui_state.training_controls.render_scale;
        config.litegs_mode = self.ui_state.training_controls.litegs_mode;

        let options = TrainingControlOptions {
            progress_every: self.ui_state.training_controls.progress_every,
            snapshot_every: Some(self.ui_state.training_controls.snapshot_every),
            retain_snapshot_on_cancel: true,
        };

        match self
            .training_manager
            .start(loaded.dataset.clone(), config, options)
        {
            Ok(()) => {
                self.ui_state.training_error = None;
                self.ui_state.preview_error = None;
                self.preview_dirty = true;
            }
            Err(err) => {
                self.ui_state.training_error = Some(err.to_string());
            }
        }
    }

    fn stop_training(&mut self) {
        if let Err(err) = self.training_manager.stop() {
            self.ui_state.training_error = Some(err.to_string());
        }
    }

    fn poll_training_events(&mut self, ctx: &egui::Context) {
        for event in self.training_manager.poll_events() {
            match event {
                TrainingSessionEvent::StateChanged { to, .. } => {
                    self.ui_state.training_state = to;
                }
                TrainingSessionEvent::ProgressUpdated(progress) => {
                    self.ui_state.training_progress = progress;
                }
                TrainingSessionEvent::SnapshotUpdated { .. } => {
                    self.sync_latest_snapshot_to_scene();
                    self.preview_dirty = true;
                }
                TrainingSessionEvent::Completed(report) => {
                    self.ui_state.training_progress.latest_loss = report.final_loss;
                    self.ui_state.training_progress.gaussian_count = Some(report.gaussian_count);
                    self.sync_latest_snapshot_to_scene();
                    self.preview_dirty = true;
                }
                TrainingSessionEvent::Failed(error) => {
                    self.ui_state.training_error = Some(error);
                }
                TrainingSessionEvent::Cancelled => {
                    self.preview_dirty = true;
                }
                TrainingSessionEvent::BackendEvent(_) => {}
            }
        }

        self.ui_state.training_state = self.training_manager.state();
        self.ui_state.training_progress = self.training_manager.progress();
        if let Some(error) = self.training_manager.latest_error() {
            self.ui_state.training_error = Some(error);
        }

        if matches!(
            self.ui_state.training_state,
            crate::training::TrainingSessionState::Starting
                | crate::training::TrainingSessionState::Training
                | crate::training::TrainingSessionState::Stopping
        ) {
            ctx.request_repaint();
        }
    }

    fn sync_latest_snapshot_to_scene(&mut self) {
        let Some(snapshot) = self.training_manager.latest_snapshot() else {
            return;
        };

        if let Ok(mut scene) = self.scene.lock() {
            scene.gaussians = host_splats_to_scene_gaussians(&snapshot);
            scene.recompute_bounds();
        }
    }

    fn refresh_preview_texture(&mut self, ctx: &egui::Context, size: Vec2) {
        let Some(loaded) = self.loaded_colmap.as_ref() else {
            self.preview_texture = None;
            self.preview_texture_size = None;
            return;
        };

        let rounded_size = if size.x >= 1.0 && size.y >= 1.0 {
            Some([size.x.floor() as usize, size.y.floor() as usize])
        } else {
            None
        };
        let needs_refresh = self.preview_dirty
            || self.preview_texture.is_none()
            || self.preview_texture_size != rounded_size;
        if !needs_refresh {
            return;
        }

        let snapshot = self.training_manager.latest_snapshot();
        match self.preview_bridge.render_from_arcball(
            snapshot.as_deref(),
            &self.preview_camera,
            loaded.dataset.intrinsics,
            size,
        ) {
            Ok(PreviewRenderStatus::Frame(image)) => {
                if let Some(texture) = self.preview_texture.as_mut() {
                    texture.set(image, TextureOptions::LINEAR);
                } else {
                    self.preview_texture =
                        Some(ctx.load_texture("training-preview", image, TextureOptions::LINEAR));
                }
                self.preview_texture_size = rounded_size;
                self.ui_state.preview_error = None;
                self.preview_dirty = false;
            }
            Ok(PreviewRenderStatus::EmptySnapshot) => {
                self.preview_texture = None;
                self.preview_texture_size = None;
                self.ui_state.preview_error = None;
                self.preview_dirty = false;
            }
            Ok(PreviewRenderStatus::InvalidViewport) => {}
            Err(err) => {
                self.ui_state.preview_error = Some(err.to_string());
                self.preview_dirty = false;
            }
        }
    }

    fn draw_preview_panel(&mut self, ctx: &egui::Context) {
        if self.loaded_colmap.is_none() {
            return;
        }

        egui::SidePanel::right("training_preview")
            .min_width(360.0)
            .default_width(420.0)
            .frame(
                egui::Frame::new()
                    .fill(PANEL_BG)
                    .inner_margin(egui::Margin::same(20)),
            )
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new("Live Preview")
                            .size(13.0)
                            .strong()
                            .color(TEXT_PRIMARY),
                    );
                    ui.label(
                        egui::RichText::new(
                            "Drag left to orbit, drag right to pan, scroll to zoom.",
                        )
                        .size(11.0)
                        .color(TEXT_SECONDARY),
                    );
                    ui.add_space(12.0);

                    let desired_size =
                        Vec2::new(ui.available_width(), ui.available_height().max(220.0));
                    let (rect, response) = ui.allocate_exact_size(desired_size, Sense::drag());

                    let mut preview_moved = false;
                    if response.dragged_by(egui::PointerButton::Primary) {
                        let delta = response.drag_delta();
                        self.preview_camera.orbit(delta.x, delta.y);
                        preview_moved = true;
                    }
                    if response.dragged_by(egui::PointerButton::Secondary) {
                        let delta = response.drag_delta();
                        self.preview_camera.pan(delta.x, delta.y);
                        preview_moved = true;
                    }
                    if response.hovered() {
                        let scroll = ui.input(|input| input.smooth_scroll_delta.y);
                        if scroll != 0.0 {
                            self.preview_camera.zoom(scroll);
                            preview_moved = true;
                        }
                    }
                    if preview_moved {
                        self.preview_dirty = true;
                        ctx.request_repaint();
                    }

                    self.refresh_preview_texture(ctx, rect.size());
                    ui.painter().rect_filled(rect, 10.0, overlay_bg());

                    if let Some(texture) = &self.preview_texture {
                        ui.painter().image(
                            texture.id(),
                            rect,
                            Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                            Color32::WHITE,
                        );
                    } else {
                        draw_preview_placeholder(
                            ui,
                            rect,
                            match self.ui_state.training_state {
                                crate::training::TrainingSessionState::Training
                                | crate::training::TrainingSessionState::Starting
                                | crate::training::TrainingSessionState::Stopping => {
                                    "Waiting for training snapshot…"
                                }
                                _ => "Start training to see live preview.",
                            },
                        );
                    }
                });
            });
    }
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        crate::ui::theme::configure_theme(ctx);

        self.poll_commands();
        self.poll_training_events(ctx);

        let (has_data, scene_bounds) = self
            .scene
            .lock()
            .map(|scene| (scene.has_data(), scene.bounds.clone()))
            .unwrap_or_default();

        egui::TopBottomPanel::top("title_bar")
            .exact_height(52.0)
            .frame(egui::Frame::new().fill(WINDOW_BG))
            .show(ctx, |ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.add_space(24.0);
                    ui.label(
                        egui::RichText::new("RustViewer")
                            .size(13.0)
                            .strong()
                            .color(TEXT_PRIMARY),
                    );
                });
            });

        let mut panel_actions = Vec::new();
        egui::SidePanel::left("control_panel")
            .exact_width(300.0)
            .frame(
                egui::Frame::new()
                    .fill(PANEL_BG)
                    .inner_margin(egui::Margin::same(24)),
            )
            .show(ctx, |ui| {
                if let Ok(mut scene) = self.scene.lock() {
                    panel_actions =
                        draw_side_panel(ui, &mut self.ui_state, &mut scene, &mut self.camera);
                }
            });
        self.process_panel_actions(panel_actions);

        self.draw_preview_panel(ctx);

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(VIEWPORT_BG))
            .show(ctx, |ui| {
                if !has_data {
                    draw_empty_state(ui);
                    draw_viewport_overlay(ui, &self.camera, has_data);
                    return;
                }

                let viewport_rect = ui.max_rect();
                let viewport_size = [viewport_rect.width(), viewport_rect.height()];

                let callback = egui_wgpu::Callback::new_paint_callback(
                    viewport_rect,
                    ViewerCallback {
                        scene: Arc::clone(&self.scene),
                        camera: self.camera.clone(),
                        viewport_size,
                        surface_format: self.surface_format,
                    },
                );
                let response = ui.allocate_rect(viewport_rect, egui::Sense::drag());
                ui.painter().add(callback);

                let mut camera_moved = false;
                if response.dragged_by(egui::PointerButton::Primary) {
                    let delta = response.drag_delta();
                    self.camera.orbit(delta.x, delta.y);
                    camera_moved = true;
                }
                if response.dragged_by(egui::PointerButton::Secondary) {
                    let delta = response.drag_delta();
                    self.camera.pan(delta.x, delta.y);
                    camera_moved = true;
                }
                if response.hovered() {
                    let scroll = ui.input(|input| input.smooth_scroll_delta.y);
                    if scroll != 0.0 {
                        self.camera.zoom(scroll);
                        camera_moved = true;
                    }
                }

                draw_viewport_overlay(ui, &self.camera, has_data);

                if camera_moved {
                    ctx.request_repaint();
                }

                let _ = scene_bounds;
            });
    }
}

fn host_splats_to_scene_gaussians(splats: &HostSplats) -> Vec<GaussianSplat> {
    let mut gaussians = Vec::with_capacity(splats.len());
    for idx in 0..splats.len() {
        gaussians.push(GaussianSplat {
            position: splats.position(idx),
            scale: splats.scale(idx),
            rotation: splats.rotation(idx),
            opacity: splats.opacity(idx),
            color: splats.rgb_color(idx),
        });
    }
    gaussians
}

fn draw_preview_placeholder(ui: &egui::Ui, rect: Rect, message: &str) {
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        message,
        egui::FontId::proportional(13.0),
        TEXT_SECONDARY,
    );
}
