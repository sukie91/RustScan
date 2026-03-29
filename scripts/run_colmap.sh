#!/bin/bash
# COLMAP pipeline for sofa.MOV video reconstruction

set -e

WORK_DIR="output/colmap_sofa"
IMAGE_DIR="$WORK_DIR/images"
SPARSE_DIR="$WORK_DIR/sparse"

echo "=== COLMAP Pipeline for sofa.MOV ==="
echo "Images: $(ls $IMAGE_DIR | wc -l)"

# 1. Feature extraction
echo "[1/5] Extracting features..."
colmap feature_extractor \
    --database_path $WORK_DIR/database.db \
    --image_path $IMAGE_DIR \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1

# 2. Sequential matching (best for video sequences)
echo "[2/5] Matching features (sequential)..."
colmap sequential_matcher \
    --database_path $WORK_DIR/database.db \
    --SequentialMatching.overlap 10 \
    --SiftMatching.use_gpu 1

# 3. Mapper (sparse reconstruction)
echo "[3/5] Running mapper..."
mkdir -p $SPARSE_DIR
colmap mapper \
    --database_path $WORK_DIR/database.db \
    --image_path $IMAGE_DIR \
    --output_path $SPARSE_DIR \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params 0

# 4. Export to text format for easy inspection
echo "[4/5] Exporting to text..."
colmap model_converter \
    --input_path $SPARSE_DIR/0 \
    --output_path $SPARSE_DIR/text \
    --output_type TXT

# 5. Export as JSON for comparison with RustSLAM
echo "[5/5] Generating summary..."
python3 << 'PYTHON'
import sqlite3
import json
import numpy as np
from pathlib import Path

work_dir = Path("output/colmap_sofa")

# Read cameras.txt and images.txt from COLMAP output
cameras_file = work_dir / "sparse/text/cameras.txt"
images_file = work_dir / "sparse/text/images.txt"
points_file = work_dir / "sparse/text/points3D.txt")

def read_cameras(filepath):
    cameras = {}
    with open(filepath) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                cam_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(p) for p in parts[4:]]
                cameras[cam_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'params': params  # [fx, fy, cx, cy] for PINHOLE
                }
    return cameras

def read_images(filepath):
    images = {}
    with open(filepath) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 10:
                img_id = int(parts[0])
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                cam_id = int(parts[8])
                name = parts[9]
                images[img_id] = {
                    'name': name,
                    'camera_id': cam_id,
                    'qvec': [qx, qy, qz, qw],  # COLMAP: qvec = [qw, qx, qy, qz] in file but stored as qx,qy,qz,qw
                    'tvec': [tx, ty, tz],
                    # COLMAP convention: world-to-cam, T = [R|t], point_cam = R * point_world + t
                }
    return images

def read_points(filepath):
    points = []
    with open(filepath) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 7:
                pt_id = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                points.append({
                    'id': pt_id,
                    'xyz': [x, y, z],
                    'rgb': [r, g, b]
                })
    return points

cameras = read_cameras(cameras_file)
images = read_images(images_file)
points = read_points(points_file)

# Compute camera centers: center = -R^T * t
def quat_to_rotation(qx, qy, qz, qw):
    """Convert quaternion to rotation matrix"""
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    return R

poses = []
for img_id, img in sorted(images.items()):
    qx, qy, qz, qw = img['qvec']
    tx, ty, tz = img['tvec']
    R = quat_to_rotation(qx, qy, qz, qw)
    t = np.array([tx, ty, tz])
    # Camera center: C = -R^T @ t
    center = -R.T @ t

    # Extract frame number from image name (frame_000001.jpg -> 1)
    frame_num = int(img['name'].split('_')[1].split('.')[0])
    # Map back to original video frame (6fps = every 5 frames)
    original_frame = frame_num * 5

    poses.append({
        'frame_id': original_frame,
        'image_name': img['name'],
        'center': center.tolist(),
        'qvec': img['qvec'],
        'tvec': img['tvec'],
        'rotation_matrix': R.tolist()
    })

output = {
    'poses': poses,
    'map_points': [{'position': p['xyz'], 'color': p['rgb']} for p in points],
    'cameras': cameras,
    'statistics': {
        'num_images': len(images),
        'num_points': len(points),
        'num_registered': len(poses)
    }
}

with open(work_dir / 'colmap_output.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"COLMAP output written: {work_dir / 'colmap_output.json'}")
print(f"  Images: {len(images)}, Points: {len(points)}, Registered: {len(poses)}")
PYTHON

echo "=== Done ==="
echo "Output files:"
echo "  - $SPARSE_DIR/text/cameras.txt"
echo "  - $SPARSE_DIR/text/images.txt"
echo "  - $SPARSE_DIR/text/points3D.txt"
echo "  - $WORK_DIR/colmap_output.json"