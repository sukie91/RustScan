use std::mem::size_of;

use super::runtime::{
    MetalProjectedGaussian, MetalProjectionRecord, MetalTileDispatchRecord, METAL_TILE_SIZE,
};
use super::TrainingProfile;

const MIB: u64 = 1024 * 1024;
#[cfg(test)]
pub(crate) const GIB: u64 = 1024 * 1024 * 1024;
#[cfg(not(test))]
const GIB: u64 = 1024 * 1024 * 1024;
#[cfg(test)]
pub(crate) const DEFAULT_METAL_MEMORY_BUDGET_BYTES: u64 = 24 * GIB;
#[cfg(not(test))]
const DEFAULT_METAL_MEMORY_BUDGET_BYTES: u64 = 24 * GIB;
#[cfg(test)]
pub(crate) const METAL_SYSTEM_MEMORY_BUDGET_NUMERATOR: u64 = 13;
#[cfg(not(test))]
const METAL_SYSTEM_MEMORY_BUDGET_NUMERATOR: u64 = 13;
#[cfg(test)]
pub(crate) const METAL_SYSTEM_MEMORY_BUDGET_DENOMINATOR: u64 = 20;
#[cfg(not(test))]
const METAL_SYSTEM_MEMORY_BUDGET_DENOMINATOR: u64 = 20;
const METAL_WARN_BUDGET_NUMERATOR: u64 = 85;
const METAL_WARN_BUDGET_DENOMINATOR: u64 = 100;
const METAL_ESTIMATE_SAFETY_NUMERATOR: u64 = 15;
const METAL_ESTIMATE_SAFETY_DENOMINATOR: u64 = 100;
const METAL_ESTIMATE_MIN_SAFETY_BYTES: u64 = 256 * MIB;
const METAL_RESIDENT_FRAME_BYTES_PER_PIXEL: u64 = 16;
const METAL_ACTIVE_FRAME_BYTES_PER_PIXEL: u64 = 32;
const METAL_RENDER_BUFFER_BYTES_PER_PIXEL: u64 = 20;
const METAL_LOSS_BUFFER_BYTES_PER_PIXEL: u64 = 28;
const METAL_LOSS_READBACK_BYTES_PER_PIXEL: u64 = 20;
const METAL_GAUSSIAN_STATE_BYTES: u64 = 168;
const METAL_VISIBLE_INDEX_BYTES_PER_GAUSSIAN: u64 = (2 * size_of::<u32>()) as u64;
const METAL_GRAD_BUFFER_BYTES_PER_GAUSSIAN: u64 = (17 * size_of::<f32>()) as u64;
const METAL_TILE_INDEX_BYTES_PER_GAUSSIAN: u64 = (4 * size_of::<u32>()) as u64;
const METAL_TILE_COUNTER_BYTES_PER_TILE: u64 =
    (2 * size_of::<u32>() + size_of::<MetalTileDispatchRecord>()) as u64;
const METAL_PROJECTION_RECORD_BYTES: u64 = size_of::<MetalProjectionRecord>() as u64;
const METAL_PROJECTED_GAUSSIAN_BYTES: u64 = size_of::<MetalProjectedGaussian>() as u64;

#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalMemoryEstimate {
    pub(crate) gaussian_state_bytes: u64,
    pub(crate) frame_bytes: u64,
    pub(crate) pixel_state_bytes: u64,
    pub(crate) projection_bytes: u64,
    pub(crate) runtime_bytes: u64,
    pub(crate) safety_margin_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalMemoryBudget {
    pub(crate) safe_bytes: u64,
    pub(crate) physical_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalMemoryDecision {
    Allow,
    Warn,
    Block,
}

impl MetalMemoryEstimate {
    fn subtotal_bytes(&self) -> u64 {
        self.gaussian_state_bytes
            .saturating_add(self.frame_bytes)
            .saturating_add(self.pixel_state_bytes)
            .saturating_add(self.projection_bytes)
            .saturating_add(self.runtime_bytes)
    }

    pub(crate) fn total_bytes(&self) -> u64 {
        self.subtotal_bytes()
            .saturating_add(self.safety_margin_bytes)
    }

    fn persistent_bytes(&self) -> u64 {
        self.gaussian_state_bytes
            .saturating_add(self.pixel_state_bytes)
            .saturating_add(self.projection_bytes)
    }

    pub(crate) fn top_components_summary(&self, count: usize) -> String {
        let mut components = vec![
            ("runtime", self.runtime_bytes),
            ("frames", self.frame_bytes),
            ("pixels", self.pixel_state_bytes),
            ("persistent", self.persistent_bytes()),
            ("safety", self.safety_margin_bytes),
        ];
        components.retain(|(_, bytes)| *bytes > 0);
        components.sort_by(|lhs, rhs| rhs.1.cmp(&lhs.1));
        components
            .into_iter()
            .take(count)
            .map(|(label, bytes)| format!("{label}≈{}", format_memory(bytes)))
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub(crate) fn recommendations(&self) -> Vec<&'static str> {
        let total = self.total_bytes().max(1);
        let mut recommendations = Vec::new();
        if self
            .runtime_bytes
            .saturating_add(self.projection_bytes)
            .saturating_mul(100)
            >= total.saturating_mul(35)
        {
            recommendations.push("lower --max-initial-gaussians or increase --sampling-step");
        }
        if self
            .frame_bytes
            .saturating_add(self.pixel_state_bytes)
            .saturating_mul(100)
            >= total.saturating_mul(20)
        {
            recommendations.push("lower --metal-render-scale");
        }
        if recommendations.is_empty() {
            recommendations.push("lower --max-initial-gaussians or --metal-render-scale");
        }
        recommendations
    }
}

impl MetalMemoryBudget {
    pub(crate) fn describe(&self) -> String {
        match self.physical_bytes {
            Some(physical_bytes) => format!(
                "{:.1} GiB safe budget on {:.1} GiB system memory",
                bytes_to_gib(self.safe_bytes),
                bytes_to_gib(physical_bytes)
            ),
            None => format!(
                "{:.1} GiB safe budget (system memory unavailable)",
                bytes_to_gib(self.safe_bytes)
            ),
        }
    }
}

pub(crate) fn training_memory_budget() -> MetalMemoryBudget {
    detect_metal_memory_budget()
}

pub(crate) fn affordable_initial_gaussian_cap(
    requested_cap: usize,
    pixel_count: usize,
    source_pixel_count: usize,
    frame_count: usize,
    batch_size: usize,
    memory_budget: &MetalMemoryBudget,
) -> usize {
    let requested_cap = requested_cap.max(1);
    if assess_memory_estimate(
        &estimate_peak_memory_with_source_pixels(
            requested_cap,
            pixel_count,
            source_pixel_count,
            frame_count,
            batch_size,
        ),
        memory_budget,
    ) != MetalMemoryDecision::Block
    {
        return requested_cap;
    }

    let mut low = 0usize;
    let mut high = requested_cap;
    while low < high {
        let mid = low + (high - low + 1) / 2;
        let decision = assess_memory_estimate(
            &estimate_peak_memory_with_source_pixels(
                mid,
                pixel_count,
                source_pixel_count,
                frame_count,
                batch_size,
            ),
            memory_budget,
        );
        if decision == MetalMemoryDecision::Block {
            high = mid - 1;
        } else {
            low = mid;
        }
    }
    low
}

pub(crate) fn preflight_initial_gaussian_cap(
    training_profile: TrainingProfile,
    affordable_cap: usize,
) -> usize {
    if training_profile != TrainingProfile::LiteGsMacV1 || affordable_cap == 0 {
        return affordable_cap;
    }

    // Keep a small densification budget without discarding a large fraction of
    // the even-sampled initialization. The previous 20% reserve collapsed the
    // TUM fallback init from 552 to 442, which preserved growth but hurt PSNR.
    let reserved_headroom = affordable_cap
        .saturating_div(20)
        .clamp(16, 64)
        .min(affordable_cap.saturating_sub(1));
    affordable_cap.saturating_sub(reserved_headroom).max(1)
}

#[cfg(test)]
pub(crate) fn estimate_peak_memory(
    num_gaussians: usize,
    pixel_count: usize,
    frame_count: usize,
    batch_size: usize,
) -> MetalMemoryEstimate {
    estimate_peak_memory_with_source_pixels(
        num_gaussians,
        pixel_count,
        pixel_count,
        frame_count,
        batch_size,
    )
}

pub(crate) fn estimate_peak_memory_with_source_pixels(
    num_gaussians: usize,
    pixel_count: usize,
    source_pixel_count: usize,
    frame_count: usize,
    _batch_size: usize,
) -> MetalMemoryEstimate {
    let padded_gaussians = num_gaussians.max(1).next_power_of_two() as u64;
    let num_gaussians = num_gaussians as u64;
    let pixel_count = pixel_count as u64;
    let source_pixel_count = source_pixel_count as u64;
    let frame_count = frame_count as u64;
    let approx_tile_count = pixel_count
        .saturating_add((METAL_TILE_SIZE * METAL_TILE_SIZE - 1) as u64)
        / (METAL_TILE_SIZE * METAL_TILE_SIZE) as u64;
    let gaussian_state_bytes = num_gaussians.saturating_mul(METAL_GAUSSIAN_STATE_BYTES);
    let resident_frame_bytes = frame_count
        .saturating_mul(source_pixel_count)
        .saturating_mul(METAL_RESIDENT_FRAME_BYTES_PER_PIXEL);
    let active_frame_bytes = pixel_count.saturating_mul(METAL_ACTIVE_FRAME_BYTES_PER_PIXEL);
    let frame_bytes = resident_frame_bytes.saturating_add(active_frame_bytes);
    let pixel_state_bytes = pixel_count.saturating_mul(
        METAL_RENDER_BUFFER_BYTES_PER_PIXEL
            + METAL_LOSS_BUFFER_BYTES_PER_PIXEL
            + METAL_LOSS_READBACK_BYTES_PER_PIXEL,
    );
    let projection_bytes = padded_gaussians
        .saturating_mul(METAL_PROJECTION_RECORD_BYTES)
        .saturating_add(num_gaussians.saturating_mul(
            METAL_PROJECTED_GAUSSIAN_BYTES + METAL_VISIBLE_INDEX_BYTES_PER_GAUSSIAN,
        ));
    let runtime_bytes = num_gaussians
        .saturating_mul(METAL_GRAD_BUFFER_BYTES_PER_GAUSSIAN + METAL_TILE_INDEX_BYTES_PER_GAUSSIAN)
        .saturating_add(approx_tile_count.saturating_mul(METAL_TILE_COUNTER_BYTES_PER_TILE));
    let subtotal = gaussian_state_bytes
        .saturating_add(frame_bytes)
        .saturating_add(pixel_state_bytes)
        .saturating_add(projection_bytes)
        .saturating_add(runtime_bytes);
    let safety_margin_bytes = apply_ratio(
        subtotal,
        METAL_ESTIMATE_SAFETY_NUMERATOR,
        METAL_ESTIMATE_SAFETY_DENOMINATOR,
    )
    .max(METAL_ESTIMATE_MIN_SAFETY_BYTES);

    MetalMemoryEstimate {
        gaussian_state_bytes,
        frame_bytes,
        pixel_state_bytes,
        projection_bytes,
        runtime_bytes,
        safety_margin_bytes,
    }
}

pub(crate) fn assess_memory_estimate(
    estimate: &MetalMemoryEstimate,
    budget: &MetalMemoryBudget,
) -> MetalMemoryDecision {
    let total = estimate.total_bytes();
    if total > budget.safe_bytes {
        return MetalMemoryDecision::Block;
    }
    if total.saturating_mul(METAL_WARN_BUDGET_DENOMINATOR)
        >= budget
            .safe_bytes
            .saturating_mul(METAL_WARN_BUDGET_NUMERATOR)
    {
        return MetalMemoryDecision::Warn;
    }
    MetalMemoryDecision::Allow
}

pub(crate) fn detect_metal_memory_budget() -> MetalMemoryBudget {
    let physical_bytes = detect_physical_memory_bytes();
    let safe_bytes = physical_bytes
        .map(|bytes| {
            apply_ratio(
                bytes,
                METAL_SYSTEM_MEMORY_BUDGET_NUMERATOR,
                METAL_SYSTEM_MEMORY_BUDGET_DENOMINATOR,
            )
        })
        .map(|bytes| bytes.min(DEFAULT_METAL_MEMORY_BUDGET_BYTES))
        .unwrap_or(DEFAULT_METAL_MEMORY_BUDGET_BYTES);

    MetalMemoryBudget {
        safe_bytes,
        physical_bytes,
    }
}

pub(crate) fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / 1024f64 / 1024f64 / 1024f64
}

fn detect_physical_memory_bytes() -> Option<u64> {
    #[allow(unsafe_code)]
    unsafe {
        let pages = libc::sysconf(libc::_SC_PHYS_PAGES);
        let page_size = libc::sysconf(libc::_SC_PAGESIZE);
        if pages <= 0 || page_size <= 0 {
            return None;
        }
        Some(((pages as u128).saturating_mul(page_size as u128)).min(u64::MAX as u128) as u64)
    }
}

pub(crate) fn apply_ratio(bytes: u64, numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        return bytes;
    }
    ((bytes as u128)
        .saturating_mul(numerator as u128)
        .checked_div(denominator as u128)
        .unwrap_or(u128::MAX))
    .min(u64::MAX as u128) as u64
}

fn format_memory(bytes: u64) -> String {
    if bytes >= GIB {
        format!("{:.1} GiB", bytes_to_gib(bytes))
    } else {
        format!("{:.0} MiB", bytes as f64 / 1024f64 / 1024f64)
    }
}
