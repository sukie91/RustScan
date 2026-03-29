#!/usr/bin/env python3
"""
将 COLMAP 输出转换为 RustGS TrainingDataset JSON 格式

用法：
  python3 scripts/colmap_to_rustgs.py \
    --colmap output/colmap_sofa/colmap_output.json \
    --images output/colmap_sofa/images \
    --output output/colmap_sofa/training_dataset.json
"""

import json
import argparse
import numpy as np
from pathlib import Path

def quat_to_rotation_matrix(qx, qy, qz, qw):
    """四元数到旋转矩阵"""
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])

def colmap_pose_to_se3(qvec, tvec):
    """
    COLMAP pose (world-to-cam) 转换为 RustGS SE3 (camera-to-world)

    COLMAP: P_cam = R * P_world + t
    RustGS: pose = SE3(rotation, translation) where P_world = R^T * P_cam + (-R^T * t)

    所以 RustGS 的 rotation = R^T = R^{-1}, translation = -R^{-1} * t
    """
    qx, qy, qz, qw = qvec
    R = quat_to_rotation_matrix(qx, qy, qz, qw)

    # COLMAP world-to-cam: P_cam = R * P_world + t
    # Invert to get camera-to-world
    R_inv = R.T
    t_inv = -R_inv @ np.array(tvec)

    # Convert back to quaternion (R_inv -> quat)
    # Using scipy or manual method
    def rotation_matrix_to_quat(R):
        """Convert rotation matrix to quaternion [x, y, z, w]"""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return [x, y, z, w]

    q_rustgs = rotation_matrix_to_quat(R_inv)
    t_rustgs = t_inv.tolist()

    return {
        'rotation': q_rustgs,  # [x, y, z, w]
        'translation': t_rustgs  # [x, y, z]
    }

def main():
    parser = argparse.ArgumentParser(description='Convert COLMAP output to RustGS TrainingDataset JSON')
    parser.add_argument('--colmap', required=True, help='COLMAP output JSON')
    parser.add_argument('--images', required=True, help='Path to COLMAP images directory')
    parser.add_argument('--output', required=True, help='Output TrainingDataset JSON')
    parser.add_argument('--max-frames', type=int, default=0, help='Maximum frames to include (0 = all)')
    parser.add_argument('--frame-stride', type=int, default=1, help='Keep every Nth frame')
    args = parser.parse_args()

    print(f"Loading COLMAP output from {args.colmap}")
    with open(args.colmap) as f:
        colmap_data = json.load(f)

    # Extract camera intrinsics
    cameras = colmap_data['cameras']
    # Assume single camera
    cam_id = list(cameras.keys())[0]
    cam = cameras[cam_id]
    fx, fy, cx, cy = cam['params']
    width = cam['width']
    height = cam['height']

    intrinsics = {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'width': width,
        'height': height
    }

    print(f"Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}, {width}x{height}")

    # Convert poses
    images_dir = Path(args.images)
    poses = []

    colmap_poses = colmap_data['poses']
    if args.max_frames > 0:
        colmap_poses = colmap_poses[:args.max_frames]

    for i, pose_data in enumerate(colmap_poses[::args.frame_stride]):
        frame_id = pose_data['frame_id']
        image_name = pose_data['image_name']
        image_path = images_dir / image_name

        if not image_path.exists():
            print(f"Warning: image {image_path} does not exist, skipping")
            continue

        # Convert COLMAP pose to RustGS SE3
        se3 = colmap_pose_to_se3(pose_data['qvec'], pose_data['tvec'])

        poses.append({
            'frame_id': frame_id,
            'image_path': str(image_path),
            'pose': se3,
            'timestamp': frame_id / 30.0  # Assuming 30fps original video
        })

    print(f"Converted {len(poses)} poses")

    # Convert sparse points
    initial_points = []
    for point in colmap_data['map_points']:
        pos = point['position']
        color = point.get('color', [128, 128, 128])
        # Normalize color to [0, 1]
        color_normalized = [c / 255.0 for c in color]
        # Format as tuple: ([x, y, z], Some([r, g, b]))
        initial_points.append([pos, color_normalized])

    print(f"Initial point cloud: {len(initial_points)} points")

    # Build TrainingDataset
    dataset = {
        'intrinsics': intrinsics,
        'depth_scale': 1000.0,
        'poses': poses,
        'initial_points': initial_points
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\nTrainingDataset saved to {output_path}")
    print(f"  Poses: {len(poses)}")
    print(f"  Initial points: {len(initial_points)}")

if __name__ == '__main__':
    main()