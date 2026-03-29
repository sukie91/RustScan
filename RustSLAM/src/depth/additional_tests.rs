//! Tests for depth module (inline)

#[cfg(test)]
mod tests {
    use crate::depth::{
        BlockMatcher, DepthFusion, DepthFusionConfig, DepthObservation, StereoConfig,
        StereoMatcher, TemporalDepthFusion,
    };

    fn create_test_stereo_pair(width: usize, height: usize) -> (Vec<u8>, Vec<u8>) {
        let left: Vec<u8> = (0..height)
            .flat_map(|y| (0..width).map(move |x| ((x + y) % 256) as u8))
            .collect();

        let right: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let shift = 10;
                    if x >= shift {
                        ((x - shift + y) % 256) as u8
                    } else {
                        0
                    }
                })
            })
            .collect();

        (left, right)
    }

    #[test]
    fn test_stereo_config() {
        let config = StereoConfig::default();
        assert_eq!(config.block_size, 5);
    }

    #[test]
    fn test_stereo_matcher_creation() {
        let config = StereoConfig::default();
        let matcher = StereoMatcher::new(config);
        // Just verify it can be created
        assert!(true);
    }

    #[test]
    fn test_block_matcher() {
        let matcher = BlockMatcher::new(5, 32);
        let (left, right) = create_test_stereo_pair(50, 50);
        let disparity = matcher.compute(&left, &right, 50, 50);
        assert_eq!(disparity.len(), 50 * 50);
    }

    #[test]
    fn test_depth_fusion_config() {
        let config = DepthFusionConfig::default();
        assert_eq!(config.max_observations, 10);
    }

    #[test]
    fn test_depth_fusion_add_observation() {
        let mut fusion = DepthFusion::default();
        fusion.add_observation(
            0,
            DepthObservation {
                depth: 1.5,
                confidence: 0.8,
                source_id: 1,
            },
        );
        assert_eq!(fusion.num_observations(0), 1);
    }

    #[test]
    fn test_depth_fusion_fuse_no_observation() {
        let fusion = DepthFusion::default();
        let result = fusion.fuse(0);
        assert!(result.is_none());
    }

    #[test]
    fn test_depth_fusion_clear() {
        let mut fusion = DepthFusion::default();
        fusion.add_observation(
            0,
            DepthObservation {
                depth: 1.0,
                confidence: 0.8,
                source_id: 0,
            },
        );
        fusion.clear();
        assert_eq!(fusion.num_observations(0), 0);
    }

    #[test]
    fn test_temporal_depth_fusion_reset() {
        let mut fusion = TemporalDepthFusion::new(DepthFusionConfig::default(), 0.5);
        let depth = vec![1.0, 2.0];
        fusion.update(&depth);
        fusion.reset();
        let result = fusion.update(&depth);
        assert_eq!(result, depth);
    }
}
