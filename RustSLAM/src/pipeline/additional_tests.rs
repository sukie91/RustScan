//! Tests for pipeline module (inline)

#[cfg(test)]
mod tests {
    use crate::pipeline::realtime::{
        PipelineConfig, PipelineState, RealtimePipeline, RealtimePipelineBuilder,
    };

    #[test]
    fn test_pipeline_config() {
        let config = PipelineConfig::default();
        assert_eq!(config.track_map_channel_size, 16);
    }

    #[test]
    fn test_pipeline_builder_defaults() {
        let builder = RealtimePipelineBuilder::new();
        let pipeline = builder.build();
        assert_eq!(*pipeline.state(), PipelineState::Stopped);
    }

    #[test]
    fn test_pipeline_builder_custom() {
        let pipeline = RealtimePipelineBuilder::new()
            .track_map_channel_size(32)
            .keyframe_interval(10)
            .build();
        assert_eq!(*pipeline.state(), PipelineState::Stopped);
    }

    #[test]
    fn test_pipeline_stop() {
        let mut pipeline = RealtimePipeline::new();
        pipeline.stop();
        assert_eq!(*pipeline.state(), PipelineState::Stopped);
    }
}
