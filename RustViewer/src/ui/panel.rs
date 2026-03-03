//! Left-side control panel UI.

use egui::Color32;

use crate::renderer::camera::ArcballCamera;
use crate::renderer::scene::Scene;

/// State shared between the UI and app logic.
#[derive(Default)]
pub struct UiState {
    pub load_error: Option<String>,
}

/// Draw the left-side control panel.
pub fn draw_side_panel(
    ui: &mut egui::Ui,
    state: &mut UiState,
    scene: &mut Scene,
    camera: &mut ArcballCamera,
    file_tx: &std::sync::mpsc::Sender<(String, std::path::PathBuf)>,
) {
    ui.heading("RustViewer");
    ui.separator();

    // File loading
    ui.collapsing("📂 加载文件", |ui| {
        if ui.button("📂 加载 Checkpoint (.json)").clicked() {
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

        if ui.button("✨ 加载 Gaussians (.ply)").clicked() {
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

        if ui.button("🔷 加载 Mesh (.obj / .ply)").clicked() {
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
    });

    // Error display
    if let Some(err) = &state.load_error {
        ui.colored_label(Color32::RED, format!("⚠️ {err}"));
        if ui.small_button("清除").clicked() {
            state.load_error = None;
        }
        ui.separator();
    }

    // Layer visibility
    ui.collapsing("图层控制", |ui| {
        ui.horizontal(|ui| {
            let s = egui::Stroke::new(2.0, Color32::from_rgb(51, 102, 255));
            ui.painter().rect_filled(
                egui::Rect::from_min_size(ui.cursor().min, egui::vec2(12.0, 12.0)),
                2.0,
                s.color,
            );
            ui.add_space(16.0);
            ui.checkbox(&mut scene.layers.trajectory, "相机轨迹");
        });
        ui.horizontal(|ui| {
            let s = egui::Stroke::new(2.0, Color32::from_rgb(102, 204, 102));
            ui.painter().rect_filled(
                egui::Rect::from_min_size(ui.cursor().min, egui::vec2(12.0, 12.0)),
                2.0,
                s.color,
            );
            ui.add_space(16.0);
            ui.checkbox(&mut scene.layers.map_points, "地图点");
        });
        ui.horizontal(|ui| {
            let c = Color32::from_rgb(255, 165, 0);
            ui.painter().rect_filled(
                egui::Rect::from_min_size(ui.cursor().min, egui::vec2(12.0, 12.0)),
                2.0, c,
            );
            ui.add_space(16.0);
            ui.checkbox(&mut scene.layers.gaussians, "Gaussian 点云");
        });
        ui.horizontal(|ui| {
            let c = Color32::from_rgb(180, 180, 180);
            ui.painter().rect_filled(
                egui::Rect::from_min_size(ui.cursor().min, egui::vec2(12.0, 12.0)),
                2.0, c,
            );
            ui.add_space(16.0);
            ui.checkbox(&mut scene.layers.mesh_wireframe, "Mesh 线框");
        });
        ui.horizontal(|ui| {
            let c = Color32::from_rgb(140, 140, 200);
            ui.painter().rect_filled(
                egui::Rect::from_min_size(ui.cursor().min, egui::vec2(12.0, 12.0)),
                2.0, c,
            );
            ui.add_space(16.0);
            ui.checkbox(&mut scene.layers.mesh_solid, "Mesh 实体");
        });
    });

    // Scene statistics
    if scene.has_data() {
        ui.collapsing("场景信息", |ui| {
            ui.label(format!("关键帧: {}", scene.keyframe_count()));
            ui.label(format!("地图点: {}", scene.map_point_count()));
            ui.label(format!("Gaussian: {}", scene.gaussian_count()));
            ui.label(format!("Mesh 顶点: {}", scene.mesh_vertex_count()));
        });
    }

    // Camera controls
    ui.separator();
    if ui.button("🎯 自动对焦").clicked() && scene.has_data() {
        camera.fit_scene(&scene.bounds);
    }

    ui.separator();
    ui.small("鼠标操作：");
    ui.small("• 左键拖拽：旋转");
    ui.small("• 右键拖拽：平移");
    ui.small("• 滚轮：缩放");
}
