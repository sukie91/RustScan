//! Left-side control panel UI.

use egui::{Color32, Vec2};

use crate::renderer::camera::ArcballCamera;
use crate::renderer::scene::Scene;
use crate::ui::theme::*;

/// State shared between the UI and app logic.
#[derive(Default)]
pub struct UiState {
    pub load_error: Option<String>,
    pub is_loading: bool,
    pub loading_message: Option<String>,
}

/// Draw the left-side control panel.
pub fn draw_side_panel(
    ui: &mut egui::Ui,
    state: &mut UiState,
    scene: &mut Scene,
    camera: &mut ArcballCamera,
    file_tx: &std::sync::mpsc::Sender<(String, std::path::PathBuf)>,
) {
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(0.0, 12.0);

        // Loading indicator
        if state.is_loading {
            draw_loading_indicator(ui, state);
        }

        // Error alert
        if state.load_error.is_some() {
            let error = state.load_error.clone().unwrap();
            draw_error_alert(ui, &error, state);
        }

        // FILE OPERATIONS section
        draw_section_header(ui, "FILE OPERATIONS");

        if draw_blue_button(ui, "📂", "Load Checkpoint") {
            let tx = file_tx.clone();
            std::thread::spawn(move || {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("JSON", &["json"])
                    .pick_file()
                {
                    let _ = tx.send(("checkpoint".to_string(), path));
                }
            });
        }

        if draw_blue_button(ui, "✨", "Load Gaussians") {
            let tx = file_tx.clone();
            std::thread::spawn(move || {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("PLY", &["ply"])
                    .pick_file()
                {
                    let _ = tx.send(("gaussian".to_string(), path));
                }
            });
        }

        if draw_blue_button(ui, "🔷", "Load Mesh") {
            let tx = file_tx.clone();
            std::thread::spawn(move || {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Mesh", &["obj", "ply"])
                    .pick_file()
                {
                    let _ = tx.send(("mesh".to_string(), path));
                }
            });
        }

        // Divider
        draw_divider(ui);

        // SCENE LAYERS section
        draw_section_header(ui, "SCENE LAYERS");

        draw_layer_toggle(
            ui,
            &mut scene.layers.trajectory,
            "Camera Trajectory",
            SYSTEM_BLUE,
        );
        draw_layer_toggle(ui, &mut scene.layers.map_points, "Map Points", SYSTEM_GREEN);
        draw_layer_toggle(ui, &mut scene.layers.gaussians, "Gaussians", SYSTEM_ORANGE);
        draw_layer_toggle(
            ui,
            &mut scene.layers.mesh_wireframe,
            "Mesh Wireframe",
            SYSTEM_GRAY,
        );
        draw_layer_toggle(ui, &mut scene.layers.mesh_solid, "Mesh Solid", SYSTEM_GRAY);

        // Divider
        draw_divider(ui);

        // SCENE STATISTICS section
        draw_section_header(ui, "SCENE STATISTICS");

        // Divider
        draw_divider(ui);

        // Auto Fit button
        if draw_auto_fit_button(ui) && scene.has_data() {
            camera.fit_scene(&scene.bounds);
        }

        // Statistics cards - always visible
        draw_stats_cards(ui, scene);
    });
}

fn draw_loading_indicator(ui: &mut egui::Ui, state: &UiState) {
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(0.0, 6.0);

        // Progress bar
        let (rect, _) =
            ui.allocate_exact_size(Vec2::new(ui.available_width(), 3.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 2.0, SYSTEM_BLUE);

        // Loading text
        let msg = state.loading_message.as_deref().unwrap_or("Loading...");
        ui.label(egui::RichText::new(msg).size(12.0).color(TEXT_PRIMARY));
    });
}

fn draw_error_alert(ui: &mut egui::Ui, error: &str, state: &mut UiState) {
    let frame = egui::Frame::new()
        .fill(Color32::from_rgba_unmultiplied(255, 59, 48, 26))
        .corner_radius(6.0)
        .inner_margin(12.0);

    frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("⚠️").size(16.0));
            ui.label(egui::RichText::new(error).size(12.0).color(SYSTEM_RED));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button(egui::RichText::new("×").size(16.0).color(SYSTEM_GRAY))
                    .clicked()
                {
                    state.load_error = None;
                }
            });
        });
    });
}

fn draw_section_header(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .size(11.0)
            .strong()
            .color(SYSTEM_GRAY),
    );
}

fn draw_blue_button(ui: &mut egui::Ui, icon: &str, label: &str) -> bool {
    let button_height = 32.0;
    let (rect, response) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), button_height),
        egui::Sense::click(),
    );

    let bg_color = if response.clicked() {
        Color32::from_rgb(0, 85, 200)
    } else if response.hovered() {
        Color32::from_rgb(0, 110, 230)
    } else {
        SYSTEM_BLUE
    };

    ui.painter().rect_filled(rect, 6.0, bg_color);

    // Calculate proper centering with 8px gap between icon and text
    let font_id = egui::FontId::proportional(13.0);
    let icon_width = ui.fonts(|f| f.glyph_width(&font_id, icon.chars().next().unwrap()));
    let text_width = ui.fonts(|f| {
        f.layout_no_wrap(label.to_string(), font_id.clone(), Color32::WHITE)
            .size()
            .x
    });
    let gap = 8.0;
    let total_width = icon_width + gap + text_width;

    let start_x = rect.center().x - total_width / 2.0;
    let center_y = rect.center().y;

    ui.painter().text(
        egui::pos2(start_x + icon_width / 2.0, center_y),
        egui::Align2::CENTER_CENTER,
        icon,
        font_id.clone(),
        Color32::WHITE,
    );
    ui.painter().text(
        egui::pos2(start_x + icon_width + gap + text_width / 2.0, center_y),
        egui::Align2::CENTER_CENTER,
        label,
        font_id,
        Color32::WHITE,
    );

    response.clicked()
}

fn draw_divider(ui: &mut egui::Ui) {
    let (rect, _) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 1.0), egui::Sense::hover());
    ui.painter().rect_filled(
        rect,
        0.0,
        Color32::from_rgba_unmultiplied(229, 229, 229, 204),
    );
}

fn draw_layer_toggle(ui: &mut egui::Ui, checked: &mut bool, label: &str, color: Color32) {
    let toggle_height = 32.0;
    let (rect, response) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), toggle_height),
        egui::Sense::click(),
    );

    if response.clicked() {
        *checked = !*checked;
    }

    // Draw checkbox
    let checkbox_size = 16.0;
    let checkbox_pos = rect.left_center() + Vec2::new(12.0, -checkbox_size / 2.0);
    let checkbox_rect =
        egui::Rect::from_min_size(checkbox_pos.into(), Vec2::new(checkbox_size, checkbox_size));

    if *checked {
        ui.painter().rect_filled(checkbox_rect, 4.0, SYSTEM_BLUE);
        // Draw checkmark
        ui.painter().text(
            checkbox_rect.center(),
            egui::Align2::CENTER_CENTER,
            "✓",
            egui::FontId::proportional(10.0),
            Color32::WHITE,
        );
    } else {
        // Draw empty checkbox with rounded border
        ui.painter().rect_stroke(
            checkbox_rect,
            4.0,
            egui::Stroke::new(1.0, SYSTEM_GRAY),
            egui::StrokeKind::Outside,
        );
    }

    // Draw color badge
    let badge_size = 12.0;
    let badge_pos =
        checkbox_pos + Vec2::new(checkbox_size + 16.0, (checkbox_size - badge_size) / 2.0);
    let badge_rect = egui::Rect::from_min_size(badge_pos.into(), Vec2::new(badge_size, badge_size));
    ui.painter()
        .circle_filled(badge_rect.center(), badge_size / 2.0, color);

    // Draw label
    let label_pos = badge_pos + Vec2::new(badge_size + 8.0, badge_size / 2.0);
    ui.painter().text(
        label_pos.into(),
        egui::Align2::LEFT_CENTER,
        label,
        egui::FontId::proportional(13.0),
        TEXT_PRIMARY,
    );
}

fn draw_auto_fit_button(ui: &mut egui::Ui) -> bool {
    let button_height = 32.0;
    let (rect, response) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), button_height),
        egui::Sense::click(),
    );

    let bg_color = if response.hovered() {
        Color32::from_rgba_unmultiplied(0, 0, 0, 10)
    } else {
        Color32::TRANSPARENT
    };

    ui.painter().rect_filled(rect, 6.0, bg_color);

    // Calculate proper centering with 8px gap between icon and text
    let icon = "🎯";
    let label = "Auto Fit Scene";
    let font_id = egui::FontId::proportional(13.0);
    let icon_width = ui.fonts(|f| f.glyph_width(&font_id, icon.chars().next().unwrap()));
    let text_width = ui.fonts(|f| {
        f.layout_no_wrap(label.to_string(), font_id.clone(), TEXT_PRIMARY)
            .size()
            .x
    });
    let gap = 8.0;
    let total_width = icon_width + gap + text_width;

    let start_x = rect.center().x - total_width / 2.0;
    let center_y = rect.center().y;

    ui.painter().text(
        egui::pos2(start_x + icon_width / 2.0, center_y),
        egui::Align2::CENTER_CENTER,
        icon,
        font_id.clone(),
        TEXT_PRIMARY,
    );
    ui.painter().text(
        egui::pos2(start_x + icon_width + gap + text_width / 2.0, center_y),
        egui::Align2::CENTER_CENTER,
        label,
        font_id,
        TEXT_PRIMARY,
    );

    response.clicked()
}

fn draw_stats_cards(ui: &mut egui::Ui, scene: &Scene) {
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(0.0, 8.0);

        // Get values - show "—" when no data loaded
        let mesh_vertices = if scene.has_data() {
            scene.mesh_vertex_count()
        } else {
            0
        };

        let keyframes = if scene.has_data() {
            scene.keyframe_count()
        } else {
            0
        };

        let map_points = if scene.has_data() {
            scene.map_point_count()
        } else {
            0
        };

        let gaussians = if scene.has_data() {
            scene.gaussian_count()
        } else {
            0
        };

        // Mesh vertices card
        let frame = egui::Frame::new()
            .fill(CARD_BG)
            .corner_radius(8.0)
            .inner_margin(12.0);

        frame.show(ui, |ui| {
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing = Vec2::new(0.0, 8.0);
                ui.label(
                    egui::RichText::new("Mesh Vertices")
                        .size(11.0)
                        .color(SYSTEM_GRAY),
                );
                ui.label(
                    egui::RichText::new(format!("{}", mesh_vertices))
                        .size(15.0)
                        .strong()
                        .color(TEXT_PRIMARY),
                );
            });
        });

        // Stats card with multiple rows
        let frame = egui::Frame::new()
            .fill(CARD_BG)
            .corner_radius(8.0)
            .inner_margin(16.0);

        frame.show(ui, |ui| {
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing = Vec2::new(0.0, 12.0);

                // Keyframes row
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Keyframes")
                            .size(11.0)
                            .color(Color32::from_rgb(153, 153, 153)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!("{}", keyframes))
                                .size(15.0)
                                .strong()
                                .color(TEXT_PRIMARY),
                        );
                    });
                });

                // Map Points row
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Map Points")
                            .size(11.0)
                            .color(Color32::from_rgb(153, 153, 153)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!("{}", map_points))
                                .size(15.0)
                                .strong()
                                .color(TEXT_PRIMARY),
                        );
                    });
                });

                // Gaussians row
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Gaussians")
                            .size(11.0)
                            .color(Color32::from_rgb(153, 153, 153)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!("{}", gaussians))
                                .size(15.0)
                                .strong()
                                .color(TEXT_PRIMARY),
                        );
                    });
                });
            });
        });
    });
}
