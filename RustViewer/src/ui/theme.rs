//! Apple Human Interface Guidelines theme configuration for RustViewer.

use egui::{Color32, FontId, Stroke, Vec2};

// ── System Colors (Apple HIG) ────────────────────────────────────────────────

pub const SYSTEM_BLUE: Color32 = Color32::from_rgb(0, 122, 255);
pub const SYSTEM_GREEN: Color32 = Color32::from_rgb(52, 199, 89);
pub const SYSTEM_ORANGE: Color32 = Color32::from_rgb(255, 149, 0);
pub const SYSTEM_RED: Color32 = Color32::from_rgb(255, 59, 48);
pub const SYSTEM_GRAY: Color32 = Color32::from_rgb(142, 142, 147);

// ── Background Colors ────────────────────────────────────────────────────────

pub const WINDOW_BG: Color32 = Color32::from_rgb(246, 246, 246);
pub const CARD_BG: Color32 = Color32::from_rgb(255, 255, 255);
pub const SEPARATOR: Color32 = Color32::from_rgb(209, 209, 214);

// ── Text Colors ──────────────────────────────────────────────────────────────

pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(0, 0, 0);
pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(142, 142, 147);
pub const TEXT_DISABLED: Color32 = Color32::from_rgb(199, 199, 204);

// ── 3D Scene Colors ─────────────────────────────────────────────────────────

pub const COLOR_TRAJECTORY: Color32 = Color32::from_rgb(0, 122, 255);
pub const COLOR_MAP_POINTS: Color32 = Color32::from_rgb(52, 199, 89);
pub const COLOR_GAUSSIANS: Color32 = Color32::from_rgb(255, 149, 0);
pub const COLOR_MESH: Color32 = Color32::from_rgb(142, 142, 147);
pub const COLOR_MESH_SOLID: Color32 = Color32::from_rgb(140, 140, 200);

// ── Spacing (8pt grid) ──────────────────────────────────────────────────────

pub const SP_XS: f32 = 4.0;
pub const SP_SM: f32 = 8.0;
pub const SP_MD: f32 = 12.0;
pub const SP_LG: f32 = 16.0;
pub const SP_XL: f32 = 24.0;
pub const SP_XXL: f32 = 32.0;

// ── Typography ──────────────────────────────────────────────────────────────

pub fn font_heading() -> FontId {
    FontId::proportional(20.0)
}

pub fn font_title() -> FontId {
    FontId::proportional(17.0)
}

pub fn font_body() -> FontId {
    FontId::proportional(13.0)
}

pub fn font_small() -> FontId {
    FontId::proportional(11.0)
}

pub fn font_caption() -> FontId {
    FontId::proportional(10.0)
}

pub fn font_mono() -> FontId {
    FontId::monospace(13.0)
}

pub fn font_mono_small() -> FontId {
    FontId::monospace(11.0)
}

// ── Corner Radius ───────────────────────────────────────────────────────────

pub const RADIUS_CARD: u8 = 8;
pub const RADIUS_BUTTON: u8 = 6;
pub const RADIUS_BADGE: u8 = 4;

// ── Apply Apple HIG Style ───────────────────────────────────────────────────

/// Configure the egui context to use Apple HIG-inspired styling.
/// Call this once at the start of each frame in `App::update()`.
pub fn configure_apple_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    // Font sizes
    style.text_styles = [
        (
            egui::TextStyle::Heading,
            FontId::proportional(20.0),
        ),
        (
            egui::TextStyle::Body,
            FontId::proportional(13.0),
        ),
        (
            egui::TextStyle::Button,
            FontId::proportional(13.0),
        ),
        (
            egui::TextStyle::Small,
            FontId::proportional(11.0),
        ),
        (
            egui::TextStyle::Monospace,
            FontId::monospace(13.0),
        ),
    ]
    .into();

    // Spacing (8pt grid)
    style.spacing.item_spacing = Vec2::new(8.0, 8.0);
    style.spacing.button_padding = Vec2::new(12.0, 6.0);
    style.spacing.indent = 16.0;

    // Visuals
    let mut visuals = egui::Visuals::light();

    // Window background
    visuals.window_fill = WINDOW_BG;
    visuals.panel_fill = WINDOW_BG;
    visuals.faint_bg_color = Color32::from_rgb(245, 245, 247);

    // Widget colors — inactive (normal)
    visuals.widgets.inactive.weak_bg_fill = SYSTEM_BLUE;
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, Color32::WHITE);

    // Widget colors — hovered
    visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(0, 110, 230);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::WHITE);

    // Widget colors — active (pressed)
    visuals.widgets.active.weak_bg_fill = Color32::from_rgb(0, 85, 200);
    visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);

    // Separator
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(0.5, SEPARATOR);

    style.visuals = visuals;
    ctx.set_style(style);
}
