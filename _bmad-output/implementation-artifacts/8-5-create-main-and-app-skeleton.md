---
title: "Story 8.5: Create Main and App Skeleton"
status: ready-for-dev
created: 2026-03-16
---

# Story 8.5: Create Main and App Skeleton

Status: done

## Story

As a **RustScan user**,
I want **a working RustViewer application entry point and main app structure**,
so that **I can launch the 3D viewer GUI and see an interactive window**.

## Acceptance Criteria

1. [x] AC1: `cargo run -p rust-viewer` launches a GUI window without panicking
2. [x] AC2: Window title is "RustViewer - 3D SLAM Visualization"
3. [x] AC3: Window size is 1280x800 with minimum 800x600
4. [x] AC4: Empty state UI is shown when no data is loaded
5. [x] AC5: Left side panel with "RustViewer" heading is displayed
6. [x] AC6: ViewerApp implements eframe::App trait correctly
7. [x] AC7: wgpu render state is read from eframe for surface format

## Tasks / Subtasks

- [x] Task 1: Create `RustViewer/src/main.rs` entry point
  - [x] Import `eframe::egui` and `rust_viewer::app::ViewerApp`
  - [x] Configure `NativeOptions` with viewport 1280x800, min 800x600
  - [x] Call `eframe::run_native` with app title and factory closure

- [x] Task 2: Create `RustViewer/src/app.rs` with ViewerApp struct
  - [x] Define `ViewerApp` struct with fields:
    - `scene: Arc<Mutex<Scene>>`
    - `camera: ArcballCamera`
    - `ui_state: UiState`
    - `file_rx/tx: mpsc::Receiver/Sender`
    - `surface_format: wgpu::TextureFormat`
  - [x] Implement `ViewerApp::new(cc: &eframe::CreationContext)` constructor
  - [x] Implement `eframe::App::update()` method

- [x] Task 3: Implement empty state UI
  - [x] Create `RustViewer/src/ui/viewport.rs`
  - [x] Implement `draw_empty_state(ui)` with centered Chinese text
  - [x] Show "RustViewer" title and file loading instructions

- [x] Task 4: Implement side panel skeleton
  - [x] Create `RustViewer/src/ui/panel.rs`
  - [x] Implement `draw_side_panel()` with heading and basic UI structure
  - [x] Add file loading buttons (checkpoint, gaussian, mesh)
  - [x] Add layer visibility checkboxes
  - [x] Add scene statistics display

- [x] Task 5: Implement 3D viewport with wgpu callback
  - [x] Create `RustViewer/src/renderer/mod.rs`
  - [x] Implement `ViewerCallback` for wgpu paint callback
  - [x] Register callback in `CentralPanel`
  - [x] Handle mouse input for camera control (orbit, pan, zoom)

## Dev Notes

### Implementation Already Completed in Story 8.4

Most of this story was implemented during Story 8.4's crate structure work:
- `main.rs` already has working entry point
- `app.rs` already has `ViewerApp` implementing `eframe::App`
- `viewport.rs` already has `draw_empty_state`
- `panel.rs` already has `draw_side_panel`
- `renderer/mod.rs` already has `ViewerCallback` and wgpu integration

### Key Implementation Details

**main.rs:**
```rust
fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "RustViewer - 3D SLAM Visualization",
        options,
        Box::new(|cc| Ok(Box::new(ViewerApp::new(cc)))),
    )
}
```

**app.rs fields:**
```rust
pub struct ViewerApp {
    scene: Arc<Mutex<Scene>>,
    camera: ArcballCamera,
    ui_state: UiState,
    file_rx: mpsc::Receiver<(String, PathBuf)>,
    file_tx: mpsc::Sender<(String, PathBuf)>,
    surface_format: wgpu::TextureFormat,
}
```

**wgpu integration:**
- Surface format read from `cc.wgpu_render_state.as_ref().map(|rs| rs.target_format)`
- Falls back to `Bgra8Unorm` if wgpu unavailable
- `ViewerCallback` implements `egui_wgpu::CallbackTrait`

### Testing Standards

- Manual test: `cargo run -p rust-viewer` should open window
- Verify no console errors or panics on launch
- Check window title and dimensions match spec

## References

- [Source: tech-spec-rust-viewer-3d-gui.md#Task-5] - main.rs and app.rs implementation plan
- [Source: tech-spec-rust-viewer-3d-gui.md#Task-14] - Empty state UI requirements
- [Source: tech-spec-rust-viewer-3d-gui.md#Task-13] - Side panel requirements
- [Source: tech-spec-rust-viewer-3d-gui.md#Task-12] - wgpu callback integration

## Dev Agent Record

### Agent Model Used

Qwen3.5-397B-A17B-FP8

### Completion Notes

- Story 8.5 implementation was largely completed during Story 8.4
- All skeleton code is in place and compiles
- Window launches successfully with empty state UI
- Ready for next story (8.6) to implement remaining scene data model components

### File List

Files already created in Story 8.4:
- `RustViewer/src/main.rs` - Entry point
- `RustViewer/src/app.rs` - ViewerApp implementation
- `RustViewer/src/ui/viewport.rs` - Empty state UI
- `RustViewer/src/ui/panel.rs` - Side panel
- `RustViewer/src/renderer/mod.rs` - wgpu callback
