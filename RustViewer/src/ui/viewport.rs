//! Viewport overlay and empty-state UI.

use crate::renderer::camera::ArcballCamera;
use crate::ui::theme::*;
use egui::Vec2;

/// Draw an empty-state overlay when no scene data is loaded.
pub fn draw_empty_state(ui: &mut egui::Ui) {
    ui.centered_and_justified(|ui| {
        ui.vertical_centered(|ui| {
            ui.add_space(80.0);

            // Large icon
            ui.label(
                egui::RichText::new("📂")
                    .size(64.0)
                    .color(TEXT_DISABLED.gamma_multiply(0.85)),
            );

            ui.add_space(24.0);

            // Title
            ui.label(
                egui::RichText::new("No SLAM Data Loaded")
                    .size(20.0)
                    .strong()
                    .color(TEXT_PRIMARY),
            );

            ui.add_space(24.0);

            // Description
            ui.label(
                egui::RichText::new(
                    "Load checkpoint, Gaussian, or mesh files to visualize 3D results",
                )
                .size(13.0)
                .color(TEXT_SECONDARY),
            );

            ui.add_space(24.0);

            // Open Files button (placeholder - actual file opening is in sidebar)
            let button_width = 120.0;
            let button_height = 32.0;
            let (rect, response) = ui
                .allocate_exact_size(Vec2::new(button_width, button_height), egui::Sense::click());

            let bg_color = if response.clicked() {
                egui::Color32::from_rgb(0, 85, 200)
            } else if response.hovered() {
                egui::Color32::from_rgb(0, 110, 230)
            } else {
                SYSTEM_BLUE
            };

            ui.painter().rect_filled(rect, 6.0, bg_color);
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Open Files",
                egui::FontId::proportional(13.0),
                egui::Color32::WHITE,
            );
        });
    });
}

/// Draw axis indicator and camera info overlay at the bottom-right of the viewport.
pub fn draw_viewport_overlay(ui: &mut egui::Ui, camera: &ArcballCamera, has_data: bool) {
    let rect = ui.max_rect();
    let indicator_size = 80.0;
    let margin = 20.0;
    let pos = egui::Pos2::new(
        rect.right() - indicator_size - margin,
        rect.bottom() - indicator_size - margin,
    );

    let indicator_rect = egui::Rect::from_min_size(pos, Vec2::new(indicator_size, indicator_size));

    // Background
    ui.painter().rect_filled(indicator_rect, 8.0, overlay_bg());

    // Draw axis labels
    let center = indicator_rect.center();

    // X axis (red)
    ui.painter().text(
        center + Vec2::new(-30.0, 5.0),
        egui::Align2::CENTER_CENTER,
        "X",
        egui::FontId::proportional(12.0),
        SYSTEM_RED,
    );

    // Y axis (green)
    ui.painter().text(
        center + Vec2::new(5.0, -25.0),
        egui::Align2::CENTER_CENTER,
        "Y",
        egui::FontId::proportional(12.0),
        SYSTEM_GREEN,
    );

    // Z axis (blue)
    ui.painter().text(
        center + Vec2::new(5.0, 30.0),
        egui::Align2::CENTER_CENTER,
        "Z",
        egui::FontId::proportional(12.0),
        SYSTEM_BLUE,
    );

    // Camera info (only when data is loaded)
    if has_data {
        let info_pos = indicator_rect.min + Vec2::new(6.0, 50.0);
        ui.painter().text(
            info_pos,
            egui::Align2::LEFT_TOP,
            format!(
                "yaw: {:.1}°\npitch: {:.1}°\ndist: {:.2}",
                camera.yaw.to_degrees(),
                camera.pitch.to_degrees(),
                camera.distance
            ),
            egui::FontId::proportional(10.0),
            TEXT_PRIMARY,
        );
    }
}
