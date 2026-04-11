pub(super) const FILL_U32_SHADER: &str = include_str!("../shaders/fill_u32.metal");
pub(super) const FORWARD_RASTER_SHADER: &str = include_str!("../shaders/forward_raster.metal");
pub(super) const BACKWARD_RASTER_SHADER: &str = include_str!("../shaders/backward_raster.metal");
pub(super) const PROJECTION_SHADER: &str = include_str!("../shaders/projection.metal");
pub(super) const TILE_BINNING_SHADER: &str = include_str!("../shaders/tile_binning.metal");
pub(super) const GRADIENTS_SHADER: &str = include_str!("../shaders/gradients.metal");
pub(super) const ADAM_SHADER: &str = include_str!("../shaders/adam.metal");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum MetalKernel {
    FillU32,
    ProjectGaussians,
    TileCount,
    TileAssign,
    FillProjectionPadding,
    CountVisible,
    BitonicSortProjections,
    ExtractSourceIndices,
    TilePrefixSum,
    AdamStep,
    TileForward,
    TileBackward,
    GradMagnitudes,
    ProjectedGradMagnitudes,
}

impl MetalKernel {
    pub(super) fn function_name(self) -> &'static str {
        match self {
            Self::FillU32 => "fill_u32",
            Self::ProjectGaussians => "project_gaussians",
            Self::TileCount => "tile_count",
            Self::TileAssign => "tile_assign",
            Self::FillProjectionPadding => "fill_projection_padding",
            Self::CountVisible => "count_visible",
            Self::BitonicSortProjections => "bitonic_sort_projections",
            Self::ExtractSourceIndices => "extract_source_indices",
            Self::TilePrefixSum => "tile_prefix_sum",
            Self::AdamStep => "adam_step",
            Self::TileForward => "tile_forward",
            Self::TileBackward => "tile_backward",
            Self::GradMagnitudes => "grad_magnitudes",
            Self::ProjectedGradMagnitudes => "projected_grad_magnitudes",
        }
    }

    pub(super) fn source(self) -> &'static str {
        match self {
            Self::FillU32 => FILL_U32_SHADER,
            Self::ProjectGaussians => PROJECTION_SHADER,
            Self::TileCount
            | Self::TileAssign
            | Self::FillProjectionPadding
            | Self::CountVisible
            | Self::BitonicSortProjections
            | Self::ExtractSourceIndices
            | Self::TilePrefixSum => TILE_BINNING_SHADER,
            Self::AdamStep => ADAM_SHADER,
            Self::TileForward => FORWARD_RASTER_SHADER,
            Self::TileBackward => BACKWARD_RASTER_SHADER,
            Self::GradMagnitudes | Self::ProjectedGradMagnitudes => GRADIENTS_SHADER,
        }
    }
}
