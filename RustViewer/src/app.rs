//! Main application struct implementing eframe::App.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};

use eframe::egui;
use eframe::egui_wgpu;

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
}

impl ViewerApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (file_tx, file_rx) = mpsc::channel();
        Self {
            scene: Arc::new(Mutex::new(Scene::default())),
            camera: ArcballCamera::default(),
            ui_state: UiState::default(),
            file_rx,
            file_tx,
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
        // Process pending file loads
        self.poll_file_loads();

        // Left control panel
        egui::SidePanel::left("control_panel")
            .min_width(220.0)
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
        egui::CentralPanel::default().show(ctx, |ui| {
            let has_data = self
                .scene
                .lock()
                .map(|s| s.has_data())
                .unwrap_or(false);

            if !has_data {
                draw_empty_state(ui);
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
                },
            );
            let response = ui.allocate_rect(viewport_rect, egui::Sense::drag());
            ui.painter().add(callback);

            // Handle mouse input for camera control
            if response.dragged_by(egui::PointerButton::Primary) {
                let delta = response.drag_delta();
                self.camera.orbit(delta.x, delta.y);
            }
            if response.dragged_by(egui::PointerButton::Secondary) {
                let delta = response.drag_delta();
                self.camera.pan(delta.x, delta.y);
            }
            if response.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    self.camera.zoom(scroll);
                }
            }

            // Debug overlay (camera info)
            draw_viewport_overlay(ui, &self.camera);

            // Request continuous repaint so the scene is always live
            ctx.request_repaint();
        });
    }
}
