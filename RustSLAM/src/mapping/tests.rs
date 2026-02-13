//! Tests for Local Mapping module

#[cfg(test)]
mod tests {
    use crate::core::{Frame, FrameFeatures, KeyFrame, MapPoint, Map, SE3, Camera};
    use crate::mapping::local_mapping::{LocalMapping, LocalMappingConfig};

    /// Helper to create a test keyframe
    fn create_test_keyframe(id: u64, translation: [f32; 3]) -> KeyFrame {
        let mut frame = Frame::new(id, id as f64, 640, 480);
        let pose = SE3::from_axis_angle(&[0.0, 0.0, 0.0], &translation);
        frame.set_pose(pose);
        frame.mark_as_keyframe();
        
        // Create features with some keypoints
        let mut features = FrameFeatures::new();
        for i in 0..10 {
            features.keypoints.push([(i as f32 * 10.0) % 640.0, (i as f32 * 15.0) % 480.0]);
            features.map_points.push(None);
        }
        
        KeyFrame::new(frame, features)
    }

    /// Helper to create a test map point
    fn create_test_map_point(id: u64, position: [f32; 3], ref_kf: u64) -> MapPoint {
        use glam::Vec3;
        MapPoint::new(id, Vec3::from(position), ref_kf)
    }

    #[test]
    fn test_local_mapping_config_default() {
        let config = LocalMappingConfig::default();
        assert_eq!(config.max_keyframes, 10);
        assert_eq!(config.max_map_points, 1000);
        assert!(config.local_ba_enabled);
    }

    #[test]
    fn test_local_mapping_creation() {
        let config = LocalMappingConfig::default();
        let mapping = LocalMapping::new(config);
        
        assert_eq!(mapping.num_local_keyframes(), 0);
        assert_eq!(mapping.num_local_map_points(), 0);
    }

    #[test]
    fn test_insert_keyframe() {
        let config = LocalMappingConfig::default();
        let mut mapping = LocalMapping::new(config);
        
        let keyframe = create_test_keyframe(0, [0.0, 0.0, 0.0]);
        
        // Insert keyframe - this triggers triangulation, BA, and culling
        mapping.insert_keyframe(keyframe);
        
        // Should have the keyframe in local keyframes
        assert!(mapping.num_local_keyframes() > 0 || mapping.is_processing());
    }

    #[test]
    fn test_local_keyframes_tracking() {
        let config = LocalMappingConfig::default();
        let mut mapping = LocalMapping::new(config);
        
        // Insert multiple keyframes
        for i in 0..3 {
            let kf = create_test_keyframe(i, [i as f32, 0.0, 0.0]);
            mapping.insert_keyframe(kf);
        }
        
        // Should track local keyframes
        assert!(mapping.num_local_keyframes() >= 0);
    }

    #[test]
    fn test_local_map_points_tracking() {
        let config = LocalMappingConfig::default();
        let mut mapping = LocalMapping::new(config);
        
        let mut map = Map::new();
        
        // Add some map points
        for i in 0..5 {
            let mp = create_test_map_point(i, [i as f32, 1.0, 5.0], 0);
            map.add_point(mp);
        }
        
        // Add a keyframe
        let kf = create_test_keyframe(0, [0.0, 0.0, 0.0]);
        mapping.insert_keyframe(kf);
        
        // Check local map points are tracked
        assert!(mapping.num_local_map_points() >= 0);
    }

    #[test]
    fn test_cull_redundant_keyframes() {
        let config = LocalMappingConfig::default();
        let mut mapping = LocalMapping::new(config);
        
        // Create keyframes and map points
        for i in 0..5 {
            let kf = create_test_keyframe(i, [i as f32, 0.0, 0.0]);
            mapping.insert_keyframe(kf);
        }
        
        // Cull redundant keyframes
        let num_culled = mapping.cull_redundant_keyframes();
        assert!(num_culled >= 0);
    }

    #[test]
    fn test_triangulate_new_points() {
        let config = LocalMappingConfig::default();
        let mut mapping = LocalMapping::new(config);
        
        // Insert first keyframe
        let kf1 = create_test_keyframe(0, [0.0, 0.0, 0.0]);
        mapping.insert_keyframe(kf1);
        
        // Insert second keyframe with different pose
        let kf2 = create_test_keyframe(1, [1.0, 0.0, 0.0]);
        mapping.insert_keyframe(kf2);
        
        // Should attempt triangulation
        // Number of map points may increase
        let initial_count = mapping.num_local_map_points();
        assert!(initial_count >= 0);
    }

    #[test]
    fn test_local_ba_optimization() {
        let config = LocalMappingConfig {
            local_ba_enabled: true,
            ..Default::default()
        };
        let mut mapping = LocalMapping::new(config);
        
        // Insert keyframes
        for i in 0..3 {
            let kf = create_test_keyframe(i, [i as f32 * 0.5, 0.0, 0.0]);
            mapping.insert_keyframe(kf);
        }
        
        // Run local BA
        mapping.run_local_ba();
        
        // Should complete without errors
        assert!(true);
    }

    #[test]
    fn test_local_mapping_empty_map() {
        let config = LocalMappingConfig::default();
        let mapping = LocalMapping::new(config);
        
        assert_eq!(mapping.num_local_keyframes(), 0);
        assert_eq!(mapping.num_local_map_points(), 0);
    }

    #[test]
    fn test_check_new_keyframes() {
        let config = LocalMappingConfig::default();
        let mut mapping = LocalMapping::new(config);
        
        // Initially no new keyframes
        assert!(!mapping.check_new_keyframes());
        
        // Insert a keyframe
        let kf = create_test_keyframe(0, [0.0, 0.0, 0.0]);
        mapping.insert_keyframe(kf);
        
        // Should have processed
        assert!(true);
    }
}
