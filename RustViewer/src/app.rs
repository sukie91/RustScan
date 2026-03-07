//! Main application struct implementing eframe::App.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};

use eframe::egui;
use eframe::egui_wgpu;
use eframe::wgpu;

use crate::loader::checkpoint::LoadError;
use crate::renderer::camera::ArcballCamera;
use crate::renderer::ViewerCallback;
use crate::renderer::scene::Scene;
use crate::ui::panel::{UiState, draw_side_panel};
use crate::ui::viewport::{draw_empty_state, draw_viewport_overlay};

pub struct ViewerApp {
    scene: Arc<Mutex<Scene>>,
    camera: ArcballCamera,
    ui_state: UiState,
    /// Channel for receiving file-open results from background threads.
    file_rx: mpsc::Receiver<(String, PathBuf)>,
    file_tx: mpsc::Sender<(String, PathBuf)>,
    /// Actual wgpu surface format read from eframe at startup.
    surface_format: wgpu::TextureFormat,
}

impl ViewerApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (file_tx, file_rx) = mpsc::channel();
        // Read the actual surface format from eframe's wgpu render state.
        // Falls back to Bgra8Unorm (macOS Metal default) if wgpu is unavailable.
        let surface_format = cc
            .wgpu_render_state
            .as_ref()
            .map(|rs| rs.target_format)
            .unwrap_or(wgpu::TextureFormat::Bgra8Unorm);
        Self {
            scene: Arc::new(Mutex::new(Scene::default())),
            camera: ArcballCamera::default(),
            ui_state: UiState::default(),
            file_rx,
            file_tx,
            surface_format,
        }
    }

    /// Process any pending file loads from background threads.
    fn poll_file_loads(&mut self) {
        while let Ok((kind, path)) = self.file_rx.try_recv() {
            if let Ok(mut scene) = self.scene.lock() {
                let result: Result<(), LoadError> = match kind.as_str() {
                    "checkpoint" => {
                        crate::loader::checkpoint::load_checkpoint(&path, &mut scene)
                    }
                    "gaussian" => {
                        crate::loader::gaussian::load_gaussians(&path, &mut scene)
                    }
                    "mesh" => {
                        crate::loader::mesh::load_mesh(&path, &mut scene)
                    }
                    _ => Ok(()),
                };

                match result {
                    Ok(_) => {
                        self.ui_state.load_error = None;
                        // Auto-fit camera after first load
                        if scene.has_data() {
                            self.camera.fit_scene(&scene.bounds);
                        }
                    }
                    Err(e) => {
                        self.ui_state.load_error = Some(e.to_string());
                    }
                }
            }
        }
    }
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Apply Apple HIG theme
        crate::ui::theme::configure_apple_style(ctx);

        // Process pending file loads
        self.poll_file_loads();

        // Take a single-lock snapshot of scene state needed for this frame.
        // This avoids holding the lock across panel/viewport closures and prevents
        // any chance of lock inversion between the two panels.
        let (has_data, scene_bounds) = self
            .scene
            .lock()
            .map(|s| (s.has_data(), s.bounds.clone()))
            .unwrap_or_default();

        // Title bar
        egui::TopBottomPanel::top("title_bar")
            .exact_height(52.0)
            .frame(egui::Frame::new().fill(egui::Color32::from_rgb(246, 246, 246)))
            .show(ctx, |ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.add_space(24.0);
                    ui.label(
                        egui::RichText::new("RustViewer")
                            .size(13.0)
                            .strong()
                            .color(egui::Color32::BLACK),
                    );
                });
            });

        // Left control panel
        egui::SidePanel::left("control_panel")
            .exact_width(280.0)
            .frame(
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(245, 245, 247))
                    .inner_margin(egui::Margin::same(24))
            )
            .show(ctx, |ui| {
                if let Ok(mut scene) = self.scene.lock() {
                    draw_side_panel(
                        ui,
                        &mut self.ui_state,
                        &mut scene,
                        &mut self.camera,
                        &self.file_tx,
                    );
                }
            });

        // Central 3D viewport
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(egui::Color32::from_rgb(250, 250, 250)))
            .show(ctx, |ui| {
                if !has_data {
                    draw_empty_state(ui);
                    // Draw axis indicator even in empty state
                    draw_viewport_overlay(ui, &self.camera, has_data);
                    return;
                }

                let viewport_rect = ui.max_rect();
                let viewport_size = [viewport_rect.width(), viewport_rect.height()];

                // Register the wgpu paint callback
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

                // Handle mouse input for camera control
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
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll != 0.0 {
                        self.camera.zoom(scroll);
                        camera_moved = true;
                    }
                }

                // Axis indicator overlay - always visible
                draw_viewport_overlay(ui, &self.camera, has_data);

                // Only request repaint when camera moved; avoids burning CPU at idle.
                if camera_moved {
                    ctx.request_repaint();
                }

                // Suppress unused warning on snapshot used for auto-fit in poll_file_loads.
                let _ = scene_bounds;
            });
    }
}
