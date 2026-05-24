//! RustViewer — Interactive 3D viewer for RustScan SLAM results.

fn main() {
    let startup_asset = startup_asset_from_args();
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("RustViewer"),
        ..Default::default()
    };

    eframe::run_native(
        "RustViewer",
        native_options,
        Box::new(move |cc| {
            Ok(Box::new(
                rust_viewer::app::ViewerApp::new_with_startup_asset(cc, startup_asset.clone()),
            ))
        }),
    )
    .expect("Failed to start RustViewer");
}

fn startup_asset_from_args() -> Option<std::path::PathBuf> {
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--gaussian" || arg == "--scene" || arg == "--input" {
            return args.next().map(std::path::PathBuf::from);
        }
        if arg == "--help" || arg == "-h" {
            println!(
                "Usage: rust-viewer [--gaussian <scene.splat|scene.ply>|--scene <path>|--input <path>|<path>]"
            );
            std::process::exit(0);
        }
        if !arg.to_string_lossy().starts_with('-') {
            return Some(std::path::PathBuf::from(arg));
        }
    }
    None
}
