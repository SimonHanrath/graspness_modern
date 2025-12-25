"""
python GraspDetectionClient.py \
    --checkpoint_path logs/cluster_100scenes_13epochs/gsnet_dev_epoch10.tar \
    --port 5555 \
    --collision_thresh 0.01 \
    --top_down_grasp \
    --debug
"""

# TODO: This code is a mess, I need to clean it up and also the scripts in the franka pipeline. Make it less convoluted and also think about sending pointcloud vs depth etc.

import zmq
import numpy as np
import io
import json
import os
import sys
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

import argparse
from graspnetAPI import GraspGroup
from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import spconv_collate_fn
from utils.collision_detector import ModelFreeCollisionDetector
import open3d as o3d

import pickle

def grasp_point_to_dict(gg, offset=None):
    translation = gg.translation
    # Subtract offset to return grasps in original coordinate frame
    if offset is not None:
        translation = translation - offset
    
    return {
        'score': gg.score,
        'width': gg.width,
        'height': gg.height,
        'depth': gg.depth,
        'translation': translation.tolist(),
        'rotation_matrix': gg.rotation_matrix.tolist()
    }

def grasp_points_to_json(grasp_points, offset=None):
    """
    Convert grasp points to JSON.
    Grasps are transformed back to original camera coordinates by subtracting the offset.
    """
    grasp_points_dicts = [grasp_point_to_dict(gg, offset) for gg in grasp_points]
    return json.dumps(grasp_points_dicts)


# Configuration
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, required=True, help='Model checkpoint path')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--top_down_grasp', action='store_true', help='Filter for top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--port', type=int, default=5555, help='ZMQ server port')

cfgs = parser.parse_args()

cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

# Set workspace to filter output grasps (can be adjusted)
xmin, xmax = -1.0, 1.0
ymin, ymax = -1.0, 1.0
zmin, zmax = 0.0, 1.0
WORKSPACE_LIMITS = np.array([xmin, xmax, ymin, ymax, zmin, zmax])

def preprocess_point_cloud(points):
    """Preprocess point cloud for model input."""
    # Sample or pad to target number of points
    if len(points) >= cfgs.num_point:
        idxs = np.random.choice(len(points), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(points))
        idxs2 = np.random.choice(len(points), cfgs.num_point - len(points), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = points[idxs]

    # Shift so all coords are >= 0 (CRITICAL: must match training preprocessing!)
    offset = -cloud_sampled.min(axis=0)  # [3,]
    cloud_sampled = cloud_sampled + offset

    ret_dict = {
        'point_clouds': cloud_sampled.astype(np.float32),
        'coors': cloud_sampled.astype(np.float32) / cfgs.voxel_size,
        'feats': np.ones_like(cloud_sampled).astype(np.float32),
        'cloud_offset': offset.astype(np.float32),
    }
    return ret_dict, offset


def get_grasps(points, colors=None):
    """Run grasp detection on point cloud."""
    # Preprocess point cloud
    data_input, offset = preprocess_point_cloud(points)
    
    # Prepare batch
    batch_data = spconv_collate_fn([data_input])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)
    
    # Forward pass
    with torch.inference_mode():
        end_points = net(batch_data)
        grasp_preds = pred_decode(end_points)
    
    preds = grasp_preds[0].detach().cpu().numpy()

    
    # Apply filters if requested
    if cfgs.top_down_grasp:
        # Filter for near-vertical grasps (approach vector close to [0, 0, -1])
        mask = preds[:, 10] > 0.9  # z-component of approach vector
        preds = preds[mask]
    
    # Filter by gripper width
    if cfgs.max_gripper_width < 0.1:
        mask = preds[:, 1] <= cfgs.max_gripper_width
        preds = preds[mask]
    
    # Apply workspace limits (in shifted coordinates)
    if WORKSPACE_LIMITS is not None:
        grasp_centers = preds[:, 13:16]  # Extract centers
        # Transform to original coordinates for workspace check
        grasp_centers_orig = grasp_centers - offset
        
        mask = (grasp_centers_orig[:, 0] >= WORKSPACE_LIMITS[0]) & \
               (grasp_centers_orig[:, 0] <= WORKSPACE_LIMITS[1]) & \
               (grasp_centers_orig[:, 1] >= WORKSPACE_LIMITS[2]) & \
               (grasp_centers_orig[:, 1] <= WORKSPACE_LIMITS[3]) & \
               (grasp_centers_orig[:, 2] >= WORKSPACE_LIMITS[4]) & \
               (grasp_centers_orig[:, 2] <= WORKSPACE_LIMITS[5])
        preds = preds[mask]
    
    if len(preds) == 0:
        print('No grasp detected after filtering')
        return GraspGroup(np.zeros((0, 17))), None, offset
    
    gg = GraspGroup(preds)
    
    # Collision detection (performed in shifted coordinate space)
    if cfgs.collision_thresh > 0:
        cloud = data_input['point_clouds']  # This is in shifted coordinates
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
        gg = gg[~collision_mask]
    
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(data_input['point_clouds'].astype(np.float32))
    if colors is not None:
        cloud_o3d.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    
    return gg, cloud_o3d, offset


# Initialize GraspNet model
net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

checkpoint = torch.load(cfgs.checkpoint_path, map_location=device)
net.load_state_dict(checkpoint['model_state_dict'], strict=False)

print(f"-> Loaded checkpoint from {cfgs.checkpoint_path} (epoch: {checkpoint['epoch']})")

net.eval()


# Initialize ZeroMQ Server
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f"tcp://*:{cfgs.port}")

print(f"Grasp Detection Server is running on port {cfgs.port}...")

while True:
    try:
        # Receive compressed point cloud data
        compressed_data = socket.recv()
        
        # Decompress using numpy
        with io.BytesIO(compressed_data) as f:
            npzfile = np.load(f)
            xyzrgb_points = npzfile['xyzrgb']
        
        print(f"Received point cloud with shape: {xyzrgb_points.shape}")
        
        points = xyzrgb_points[:, :3]
        colors = xyzrgb_points[:, 3:] / 255.0 if xyzrgb_points.shape[1] > 3 else None

        # Run grasp detection
        gg, cloud, offset = get_grasps(points, colors)
        
        if len(gg) == 0:
            print("No grasps detected!")
            grasp_points = grasp_points_to_json([], offset)
        else:
            # Apply NMS and sort by score
            gg = gg.nms().sort_by_score()
            print(f"Detected {len(gg)} grasps after NMS")
            print(f"Best grasp score: {gg[0].score:.4f}")
            
            # Convert to JSON (includes offset for client-side transformation)
            grasp_points = grasp_points_to_json(gg, offset)
            
        # Send response as JSON
        socket.send_json(grasp_points)
        print("Sent grasp points back.")
        
        # Visualization (optional, for debugging)
        if cfgs.debug and len(gg) > 0 and cloud is not None:
            print("Visualizing grasps... (close window to continue)")
            # Show top 10 grasps for clarity
            gg_viz = gg[:min(10, len(gg))]
    
    except Exception as e:
        print(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        # Send empty response on error
        socket.send_json(grasp_points_to_json([], None))

