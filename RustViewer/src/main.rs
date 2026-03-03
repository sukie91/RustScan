//! RustViewer — Interactive 3D viewer for RustScan SLAM results.

fn main() {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("RustViewer"),
        ..Default::default()
    };

    eframe::run_native(
        "RustViewer",
        native_options,
        Box::new(|cc| Ok(Box::new(rust_viewer::app::ViewerApp::new(cc)))),
    )
    .expect("Failed to start RustViewer");
}
