//! Viewport overlay and empty-state UI.

use crate::renderer::camera::ArcballCamera;

/// Draw an empty-state overlay when no scene data is loaded.
pub fn draw_empty_state(ui: &mut egui::Ui) {
    ui.centered_and_justified(|ui| {
        ui.vertical_centered(|ui| {
            ui.add_space(80.0);
            ui.label(
                egui::RichText::new("RustViewer")
                    .size(32.0)
                    .strong(),
            );
            ui.add_space(16.0);
            ui.label("请从左侧面板加载 SLAM 结果文件");
            ui.add_space(24.0);
            ui.label(
                egui::RichText::new("支持格式：slam_checkpoint.json / scene.ply / mesh.obj / mesh.ply")
                    .color(egui::Color32::GRAY),
            );
        });
    });
}

/// Draw a small debug overlay at the bottom-right of the viewport.
pub fn draw_viewport_overlay(ui: &mut egui::Ui, camera: &ArcballCamera) {
    let rect = ui.max_rect();
    let pos = egui::Pos2::new(rect.right() - 180.0, rect.bottom() - 60.0);
    let painter = ui.painter();
    let bg = egui::Color32::from_black_alpha(120);
    painter.rect_filled(
        egui::Rect::from_min_size(pos, [175.0, 55.0].into()),
        4.0,
        bg,
    );
    painter.text(
        pos + egui::Vec2::new(6.0, 6.0),
        egui::Align2::LEFT_TOP,
        format!(
            "yaw: {:.1}°  pitch: {:.1}°\ndist: {:.2}",
            camera.yaw.to_degrees(),
            camera.pitch.to_degrees(),
            camera.distance
        ),
        egui::FontId::monospace(11.0),
        egui::Color32::WHITE,
    );
}
