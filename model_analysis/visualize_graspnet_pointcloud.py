"""
Largely vibe coded !!!!
Visualize GraspNet-1B point clouds with downsampling and voxelization.

This script loads a scene from GraspNet-1B dataset and visualizes:
1. The downsampled point cloud (after sampling to num_points)
2. The voxelized representation (as used in sparse convolution)
3. Optionally, detected grasp poses from a trained model

The output is an interactive HTML file that can be opened in a browser,
matching the visualization format used in zmq_server.py.

Usage:
    python model_analysis/visualize_graspnet_pointcloud.py --scene_id 100 --frame_id 0 --camera realsense
    python model_analysis/visualize_graspnet_pointcloud.py --scene_id 0 --frame_id 128 --camera kinect --num_points 20000
    
    # With grasp detection:
    python model_analysis/visualize_graspnet_pointcloud.py --scene_id 100 --frame_id 0 --checkpoint_path path/to/model.tar --top_n_grasps 10
    
    # With floor/table points included (matches training pipeline with include_floor=True):
    python model_analysis/visualize_graspnet_pointcloud.py --scene_id 100 --frame_id 0 --include_floor
    
    # With workspace bounding box visualization:
    python model_analysis/visualize_graspnet_pointcloud.py --scene_id 100 --frame_id 0 --show_bbox
"""

import os
import sys
import argparse
import numpy as np
import scipy.io as scio
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, transform_point_cloud, get_workspace_mask
from dataset.graspnet_dataset import sample_floor_aware


# ============================================================================
# Grasp Detection Functions (optional - requires model checkpoint)
# ============================================================================

def load_grasp_model(checkpoint_path, device, seed_feat_dim=512, backbone='resunet'):
    """
    Load the GraspNet model from checkpoint.
    
    Args:
        checkpoint_path: path to model checkpoint (.tar file)
        device: torch device
        seed_feat_dim: point-wise feature dimension
        backbone: backbone architecture
        
    Returns:
        net: loaded GraspNet model in eval mode
    """
    import torch
    from models.graspnet import GraspNet
    
    print(f"Loading model from {checkpoint_path}...")
    net = GraspNet(seed_feat_dim=seed_feat_dim, is_training=False, backbone=backbone)
    net.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"  -> Loaded checkpoint (epoch: {checkpoint['epoch']})")
    
    net.eval()
    return net


def run_grasp_detection(net, cloud_sampled_shifted, device, voxel_size=0.005, 
                        collision_thresh=0.01, voxel_size_cd=0.01, return_graspness=False):
    """
    Run grasp detection on preprocessed point cloud.
    
    Args:
        net: GraspNet model
        cloud_sampled_shifted: (N, 3) preprocessed point cloud (after offset shift)
        device: torch device
        voxel_size: voxel size for sparse convolution
        collision_thresh: collision detection threshold (<=0 to disable)
        voxel_size_cd: voxel size for collision detection
        return_graspness: if True, also return graspness scores for all input points
        
    Returns:
        gg: GraspGroup with detected grasps (in shifted coordinates)
        graspness_scores: (N,) numpy array of graspness scores (only if return_graspness=True)
    """
    import torch
    from graspnetAPI import GraspGroup
    from models.graspnet import pred_decode
    from dataset.graspnet_dataset import spconv_collate_fn
    from utils.collision_detector import ModelFreeCollisionDetector
    
    # Prepare input dict
    ret_dict = {
        'point_clouds': cloud_sampled_shifted.astype(np.float32),
        'coors': cloud_sampled_shifted.astype(np.float32) / voxel_size,
        'feats': np.ones_like(cloud_sampled_shifted).astype(np.float32),
    }
    
    # Prepare batch using spconv collate function
    batch_data = spconv_collate_fn([ret_dict])
    
    # Transfer to GPU
    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device, non_blocking=True)
        else:
            batch_data[key] = batch_data[key].to(device, non_blocking=True)
    
    # Forward pass
    with torch.inference_mode():
        end_points = net(batch_data)
        grasp_preds = pred_decode(end_points)
    
    preds = grasp_preds[0].detach().cpu().numpy()
    
    # Extract graspness scores if requested
    graspness_scores = None
    if return_graspness and 'graspness_score' in end_points:
        graspness_scores = end_points['graspness_score'].squeeze(1)[0].detach().cpu().numpy()  # (N,)
    
    if len(preds) == 0:
        print('  No grasps detected')
        if return_graspness:
            return GraspGroup(np.zeros((0, 17))), graspness_scores
        return GraspGroup(np.zeros((0, 17)))
    
    gg = GraspGroup(preds)
    
    # Collision detection
    if collision_thresh > 0:
        mfcdetector = ModelFreeCollisionDetector(cloud_sampled_shifted, voxel_size=voxel_size_cd)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        gg = gg[~collision_mask]
        if len(gg) == 0:
            print('  No grasps after collision detection')
            if return_graspness:
                return GraspGroup(np.zeros((0, 17))), graspness_scores
            return GraspGroup(np.zeros((0, 17)))
    
    if return_graspness:
        return gg, graspness_scores
    return gg


def create_gripper_mesh_vertices(translation, rotation_matrix, width, depth, height=0.02):
    """
    Create vertices for a simplified gripper mesh visualization.
    
    The gripper is represented as:
    - A base cylinder/box (palm)
    - Two finger plates
    
    Args:
        translation: (3,) grasp center position
        rotation_matrix: (3, 3) grasp rotation matrix
        width: gripper opening width
        depth: grasp depth (finger length)
        height: finger height/thickness
        
    Returns:
        vertices: list of (x, y, z) tuples for gripper lines
    """
    # Gripper dimensions
    finger_length = depth
    finger_thickness = 0.01
    palm_width = 0.04
    
    # Local gripper frame:
    # X: approach direction (into the object)
    # Y: closing direction (between fingers)
    # Z: up direction (along finger height)
    
    # Finger positions in local frame
    half_width = width / 2
    
    # Define gripper as line segments in local frame
    # Palm back plate
    lines_local = [
        # Palm back (connects the two fingers at the back)
        [[-finger_thickness, -half_width, -height/2], [-finger_thickness, half_width, -height/2]],
        [[-finger_thickness, -half_width, height/2], [-finger_thickness, half_width, height/2]],
        [[-finger_thickness, -half_width, -height/2], [-finger_thickness, -half_width, height/2]],
        [[-finger_thickness, half_width, -height/2], [-finger_thickness, half_width, height/2]],
        
        # Left finger
        [[-finger_thickness, -half_width, -height/2], [finger_length, -half_width, -height/2]],
        [[-finger_thickness, -half_width, height/2], [finger_length, -half_width, height/2]],
        [[finger_length, -half_width, -height/2], [finger_length, -half_width, height/2]],
        
        # Right finger
        [[-finger_thickness, half_width, -height/2], [finger_length, half_width, -height/2]],
        [[-finger_thickness, half_width, height/2], [finger_length, half_width, height/2]],
        [[finger_length, half_width, -height/2], [finger_length, half_width, height/2]],
        
        # Connect palm vertices
        [[-finger_thickness, -half_width, -height/2], [-finger_thickness, -half_width, height/2]],
        [[-finger_thickness, half_width, -height/2], [-finger_thickness, half_width, height/2]],
    ]
    
    # Transform to world frame
    lines_world = []
    for line in lines_local:
        p1_local = np.array(line[0])
        p2_local = np.array(line[1])
        
        p1_world = rotation_matrix @ p1_local + translation
        p2_world = rotation_matrix @ p2_local + translation
        
        lines_world.append([p1_world, p2_world])
    
    return lines_world


def create_grasp_traces(grasps, top_n=10, offset=None, name_prefix="Grasp", color_scheme="score"):
    """
    Create Plotly traces for grasp visualization.
    
    Args:
        grasps: GraspGroup object
        top_n: number of top grasps to visualize
        offset: optional offset to subtract from grasp positions
        name_prefix: prefix for grasp names in legend (e.g., "Pred" or "GT")
        color_scheme: "score" for green-red based on score, "blue" for solid blue (GT)
        
    Returns:
        traces: list of Plotly trace objects
    """
    if len(grasps) == 0:
        return []
    
    # Sort by score and take top N
    grasps = grasps.nms().sort_by_score()
    if len(grasps) > top_n:
        grasps = grasps[:top_n]
    
    traces = []
    
    # Color scale for scores (green = high, red = low)
    scores = [g.score for g in grasps]
    max_score = max(scores) if scores else 1.0
    min_score = min(scores) if scores else 0.0
    
    for i, grasp in enumerate(grasps):
        translation = grasp.translation.copy()
        if offset is not None:
            translation = translation - offset
            
        rotation_matrix = grasp.rotation_matrix
        width = grasp.width
        depth = grasp.depth
        score = grasp.score
        
        # Determine color based on scheme
        if color_scheme == "score":
            # Green (high score) to red (low score)
            if max_score > min_score:
                norm_score = (score - min_score) / (max_score - min_score)
            else:
                norm_score = 1.0
            r = int(255 * (1 - norm_score))
            g = int(255 * norm_score)
            b = 50
            color = f'rgb({r},{g},{b})'
        elif color_scheme == "blue":
            # Blue for ground truth
            color = 'rgb(30,144,255)'  # DodgerBlue
        elif color_scheme == "purple":
            # Purple alternative
            color = 'rgb(148,0,211)'  # DarkViolet
        else:
            color = 'rgb(128,128,128)'  # Gray fallback
        
        # Get gripper line segments
        lines = create_gripper_mesh_vertices(translation, rotation_matrix, width, depth)
        
        # Create line traces for this gripper
        x_coords = []
        y_coords = []
        z_coords = []
        
        for line in lines:
            x_coords.extend([line[0][0], line[1][0], None])
            y_coords.extend([line[0][1], line[1][1], None])
            z_coords.extend([line[0][2], line[1][2], None])
        
        trace = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines',
            name=f'{name_prefix} {i+1} (score={score:.3f})',
            line=dict(color=color, width=4),
            hovertemplate=f'{name_prefix} {i+1}<br>Score: {score:.4f}<br>Width: {width:.4f}m<extra></extra>',
        )
        traces.append(trace)
    
    return traces


def create_bbox_trace(bbox_min, bbox_max, name="Workspace BBox", color='rgb(255,165,0)', width=3):
    """
    Create a Plotly trace for a 3D bounding box wireframe.
    
    Args:
        bbox_min: (3,) array with [xmin, ymin, zmin]
        bbox_max: (3,) array with [xmax, ymax, zmax]
        name: legend name for the trace
        color: line color
        width: line width
        
    Returns:
        trace: Plotly Scatter3d trace
    """
    xmin, ymin, zmin = bbox_min
    xmax, ymax, zmax = bbox_max
    
    # 12 edges of a box
    # Bottom face (z=zmin)
    # Top face (z=zmax)
    # Vertical edges connecting them
    edges = [
        # Bottom face
        [[xmin, ymin, zmin], [xmax, ymin, zmin]],
        [[xmax, ymin, zmin], [xmax, ymax, zmin]],
        [[xmax, ymax, zmin], [xmin, ymax, zmin]],
        [[xmin, ymax, zmin], [xmin, ymin, zmin]],
        # Top face
        [[xmin, ymin, zmax], [xmax, ymin, zmax]],
        [[xmax, ymin, zmax], [xmax, ymax, zmax]],
        [[xmax, ymax, zmax], [xmin, ymax, zmax]],
        [[xmin, ymax, zmax], [xmin, ymin, zmax]],
        # Vertical edges
        [[xmin, ymin, zmin], [xmin, ymin, zmax]],
        [[xmax, ymin, zmin], [xmax, ymin, zmax]],
        [[xmax, ymax, zmin], [xmax, ymax, zmax]],
        [[xmin, ymax, zmin], [xmin, ymax, zmax]],
    ]
    
    x_coords = []
    y_coords = []
    z_coords = []
    
    for edge in edges:
        x_coords.extend([edge[0][0], edge[1][0], None])
        y_coords.extend([edge[0][1], edge[1][1], None])
        z_coords.extend([edge[0][2], edge[1][2], None])
    
    trace = go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='lines',
        name=name,
        line=dict(color=color, width=width),
        hovertemplate=f'{name}<extra></extra>',
    )
    
    return trace


def create_rotated_bbox_trace(corners, name="Workspace BBox (rotated)", color='rgb(255,0,255)', width=3):
    """
    Create a Plotly trace for a rotated 3D bounding box wireframe using 8 corner points.
    
    The corners should be ordered as:
    [0-3]: bottom face (zmin in table coords)
    [4-7]: top face (zmax in table coords)
    With same XY ordering within each face.
    
    Args:
        corners: (8, 3) array of corner points in camera coordinates
        name: legend name for the trace
        color: line color
        width: line width
        
    Returns:
        trace: Plotly Scatter3d trace
    """
    # Define edges by corner indices
    # Bottom face: 0-1-2-3, Top face: 4-5-6-7
    # Vertical edges: 0-4, 1-5, 2-6, 3-7
    edge_indices = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    
    x_coords = []
    y_coords = []
    z_coords = []
    
    for i, j in edge_indices:
        x_coords.extend([corners[i, 0], corners[j, 0], None])
        y_coords.extend([corners[i, 1], corners[j, 1], None])
        z_coords.extend([corners[i, 2], corners[j, 2], None])
    
    trace = go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='lines',
        name=name,
        line=dict(color=color, width=width),
        hovertemplate=f'{name}<extra></extra>',
    )
    
    return trace


def load_gt_grasps(data_root, scene_id, frame_id, camera='realsense', fric_coef_thresh=0.4, offset=None):
    """
    Load ground truth grasps from GraspNet-1B annotations.
    
    Args:
        data_root: path to GraspNet-1B dataset root
        scene_id: scene index (0-189)
        frame_id: frame/annotation index (0-255)
        camera: 'realsense' or 'kinect'
        fric_coef_thresh: friction coefficient threshold (lower = better grasp)
        offset: offset to add to grasp positions (to match shifted point cloud)
        
    Returns:
        gg: GraspGroup with ground truth grasps (in shifted coordinates if offset provided)
    """
    from graspnetAPI import GraspNet as GraspNetAPI
    from graspnetAPI import GraspGroup
    
    print(f"  Loading ground truth grasps for scene {scene_id}, frame {frame_id}...")
    
    # Initialize GraspNet API
    gn = GraspNetAPI(root=data_root, camera=camera, split='custom', sceneIds=[scene_id])
    
    # Load grasp labels and collision labels
    print(f"  Loading grasp labels...")
    grasp_labels = gn.loadGraspLabels(objIds=gn.getObjIds(sceneIds=[scene_id]))
    
    print(f"  Loading collision labels...")
    collision_labels = gn.loadCollisionLabels(sceneIds=scene_id)
    
    # Load grasps for this frame
    print(f"  Computing valid grasps (fric_coef_thresh={fric_coef_thresh})...")
    gg = gn.loadGrasp(
        sceneId=scene_id, 
        annId=frame_id, 
        format='6d', 
        camera=camera,
        grasp_labels=grasp_labels,
        collision_labels=collision_labels,
        fric_coef_thresh=fric_coef_thresh
    )
    
    print(f"  Loaded {len(gg)} ground truth grasps")
    
    # Apply offset if provided (to match shifted coordinates)
    if offset is not None and len(gg) > 0:
        # GraspGroup stores grasps in camera coordinates
        # We need to shift them to match the offset-shifted point cloud
        grasp_array = gg.grasp_group_array.copy()
        # Translation is at indices 13:16
        grasp_array[:, 13:16] += offset
        gg = GraspGroup(grasp_array)
    
    return gg


def plot_downsampled_pointcloud(points, save_path="downsampled_pointcloud.png"):
    """
    Plot and save the downsampled point cloud to a PNG file.
    
    Args:
        points: (N, 3) numpy array of XYZ coordinates (after downsampling/padding)
        save_path: path to save the plot image
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample for plotting if too many points (for performance)
    max_plot_points = 5000
    if len(points) > max_plot_points:
        plot_idxs = np.random.choice(len(points), max_plot_points, replace=False)
        plot_points = points[plot_idxs]
    else:
        plot_points = points
    
    # Color by Z height for better visualization
    z_vals = plot_points[:, 2]
    
    scatter = ax.scatter(
        plot_points[:, 0], 
        plot_points[:, 1], 
        plot_points[:, 2],
        c=z_vals,
        cmap='viridis',
        s=1,
        alpha=0.6
    )
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Downsampled Point Cloud ({len(points)} points)')
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Z height (m)')
    
    # Set equal aspect ratio
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved downsampled point cloud visualization to: {save_path}")


def plot_downsampled_pointcloud_interactive(points, voxel_size, save_path="downsampled_pointcloud.html", 
                                            title_prefix="", grasp_traces=None, graspness_scores=None,
                                            bbox_traces=None):
    """
    Create an interactive 3D plot of the downsampled point cloud using Plotly.
    Shows both the raw sampled points and the voxelized points as proper cubes.
    Saves as HTML file that can be opened in a browser.
    
    Args:
        points: (N, 3) numpy array of XYZ coordinates (after downsampling/padding)
        voxel_size: voxel size used for sparse convolution
        save_path: path to save the interactive HTML plot
        title_prefix: optional prefix for the plot title (e.g., scene/frame info)
        grasp_traces: optional list of Plotly traces for grasp visualization
        graspness_scores: optional (N,) array of graspness scores per point (for voxel coloring)
        bbox_traces: optional list of Plotly traces for bounding box visualization
    """
    # Subsample for plotting if too many points (for performance)
    max_plot_points = 10000
    if len(points) > max_plot_points:
        plot_idxs = np.random.choice(len(points), max_plot_points, replace=False)
        plot_points = points[plot_idxs]
    else:
        plot_points = points
   
    # Voxelize: exactly as in inference (divide by voxel_size, cast to int32 = truncation, take unique)
    # This matches spconv_collate_fn: c = d["coors"]; c = c.int()
    voxel_coords = (points / voxel_size).astype(np.int32)
    unique_voxels = np.unique(voxel_coords, axis=0)
    # Voxel spans from (voxel_coord * voxel_size) to ((voxel_coord + 1) * voxel_size)
    voxel_origins = unique_voxels.astype(np.float32) * voxel_size
    voxel_centers = voxel_origins + 0.5 * voxel_size
    
    n_voxels = len(unique_voxels)
    print(f"  Voxelization: {len(points)} points -> {n_voxels} voxels (voxel_size={voxel_size})")
    
    # Compute per-voxel graspness scores if provided
    voxel_graspness = None
    if graspness_scores is not None:
        print("  Computing per-voxel graspness scores...")
        # Map each point to its voxel index
        # Create a dictionary from voxel coordinate tuple to list of point indices
        from collections import defaultdict
        voxel_to_points = defaultdict(list)
        for i, vc in enumerate(voxel_coords):
            voxel_to_points[tuple(vc)].append(i)
        
        # Build array of mean graspness for each unique voxel
        voxel_graspness = np.zeros(n_voxels, dtype=np.float32)
        for v_idx, uv in enumerate(unique_voxels):
            point_indices = voxel_to_points[tuple(uv)]
            voxel_graspness[v_idx] = np.mean(graspness_scores[point_indices])
        
        print(f"  Graspness range: [{voxel_graspness.min():.4f}, {voxel_graspness.max():.4f}]")
    
    # Build cube mesh for all voxels
    # Each cube: 8 vertices, 12 triangles (2 per face)
    # Limit to max_voxels for performance
    max_voxels = 15000
    if n_voxels > max_voxels:
        viz_idxs = np.random.choice(n_voxels, max_voxels, replace=False)
        viz_origins = voxel_origins[viz_idxs]
        viz_centers = voxel_centers[viz_idxs]
        viz_graspness = voxel_graspness[viz_idxs] if voxel_graspness is not None else None
        n_viz = max_voxels
        print(f"  Subsampled {n_voxels} -> {max_voxels} voxels for visualization")
    else:
        viz_origins = voxel_origins
        viz_centers = voxel_centers
        viz_graspness = voxel_graspness
        n_viz = n_voxels
        
    # 8 corner offsets for a unit cube
    s = voxel_size
    corner_offsets = np.array([
        [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],  # bottom face
        [0, 0, s], [s, 0, s], [s, s, s], [0, s, s],  # top face
    ], dtype=np.float32)
    
    # 12 triangles (2 per face), vertex indices within each cube
    cube_faces_i = [0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 3, 3]
    cube_faces_j = [1, 2, 5, 6, 1, 4, 2, 5, 3, 4, 2, 6]
    cube_faces_k = [3, 1, 7, 5, 4, 7, 5, 6, 7, 3, 6, 7]
    
    # Build combined mesh arrays
    all_verts = np.zeros((n_viz * 8, 3), dtype=np.float32)
    all_i = np.zeros(n_viz * 12, dtype=np.int32)
    all_j = np.zeros(n_viz * 12, dtype=np.int32)
    all_k = np.zeros(n_viz * 12, dtype=np.int32)
    all_intensity = np.zeros(n_viz * 8, dtype=np.float32)  # for coloring
    
    for idx in range(n_viz):
        v_offset = idx * 8
        f_offset = idx * 12
        origin = viz_origins[idx]
        
        all_verts[v_offset:v_offset + 8] = origin + corner_offsets
        all_i[f_offset:f_offset + 12] = np.array(cube_faces_i) + v_offset
        all_j[f_offset:f_offset + 12] = np.array(cube_faces_j) + v_offset
        all_k[f_offset:f_offset + 12] = np.array(cube_faces_k) + v_offset
        # Color all 8 vertices by graspness score or voxel center Z
        if viz_graspness is not None:
            all_intensity[v_offset:v_offset + 8] = viz_graspness[idx]
        else:
            all_intensity[v_offset:v_offset + 8] = viz_centers[idx, 2]
    
    # Trace 1: voxel cubes - use different colorscale/title based on graspness availability
    if viz_graspness is not None:
        colorscale = 'RdYlGn'  # Red-Yellow-Green: low graspness (red) to high graspness (green)
        colorbar_title = 'Graspness'
    else:
        colorscale = 'Plasma'
        colorbar_title = 'Z (m)'
    
    voxel_trace = go.Mesh3d(
        x=all_verts[:, 0],
        y=all_verts[:, 1],
        z=all_verts[:, 2],
        i=all_i, j=all_j, k=all_k,
        intensity=all_intensity,
        colorscale=colorscale,
        colorbar=dict(title=colorbar_title),
        opacity=1.0,
        flatshading=True,
        name=f'Voxels ({n_voxels}, size={voxel_size}m)',
    )
    
    # Trace 2: raw sampled points
    z_vals = plot_points[:, 2]
    raw_trace = go.Scatter3d(
        x=plot_points[:, 0],
        y=plot_points[:, 1],
        z=plot_points[:, 2],
        mode='markers',
        name=f'Sampled points ({len(points)})',
        marker=dict(
            size=1.5,
            color=z_vals,
            colorscale='Viridis',
            opacity=0.5,
        ),
        hovertemplate='X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<extra>sampled</extra>',
        visible='legendonly',  # Hidden by default, toggle via legend
    )
    
    # Build figure with all traces
    all_traces = [voxel_trace, raw_trace]
    if grasp_traces:
        all_traces.extend(grasp_traces)
    if bbox_traces:
        all_traces.extend(bbox_traces)
    
    fig = go.Figure(data=all_traces)
    
    n_grasps = len(grasp_traces) if grasp_traces else 0
    title = f'{title_prefix}Point Cloud: {len(points)} sampled → {n_voxels} voxels'
    if n_grasps > 0:
        title += f' | {n_grasps} grasps'
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            zaxis=dict(autorange='reversed'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    fig.write_html(save_path)
    print(f"Saved interactive point cloud visualization to: {save_path}")
    print(f"  -> Open in browser: file://{os.path.abspath(save_path)}")


def load_graspnet_scene(data_root, scene_id, frame_id, camera='realsense', remove_outlier=True):
    """
    Load a point cloud from a GraspNet-1B scene.
    
    Args:
        data_root: path to GraspNet-1B dataset root
        scene_id: scene index (0-189)
        frame_id: frame index within scene (0-255)
        camera: 'realsense' or 'kinect'
        remove_outlier: whether to apply workspace masking
        
    Returns:
        cloud_masked: (N, 3) numpy array of valid point cloud
        rgb_masked: (N, 3) RGB values or None
        meta_info: dict with scene metadata (includes 'trans' matrix and 'workspace_bbox' for visualization)
    """
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    scene_path = os.path.join(data_root, 'scenes', scene_name, camera)
    
    # Load depth and segmentation
    depth_path = os.path.join(scene_path, 'depth', f'{str(frame_id).zfill(4)}.png')
    seg_path = os.path.join(scene_path, 'label', f'{str(frame_id).zfill(4)}.png')
    meta_path = os.path.join(scene_path, 'meta', f'{str(frame_id).zfill(4)}.mat')
    rgb_path = os.path.join(scene_path, 'rgb', f'{str(frame_id).zfill(4)}.png')
    
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth image not found: {depth_path}")
    
    depth = np.array(Image.open(depth_path))
    seg = np.array(Image.open(seg_path))
    meta = scio.loadmat(meta_path)
    
    # Load RGB if available
    rgb = None
    if os.path.exists(rgb_path):
        rgb = np.array(Image.open(rgb_path))
    
    # Get camera intrinsics
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']
    camera_info = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], 
                             intrinsic[0][2], intrinsic[1][2], factor_depth)
    
    # Generate point cloud
    cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)
    
    # Get valid points
    depth_mask = (depth > 0)
    trans = None
    workspace_bbox = None  # Will store bbox in camera coordinates for visualization
    
    # Always load trans for potential bbox computation
    camera_poses = np.load(os.path.join(scene_path, 'camera_poses.npy'))
    align_mat = np.load(os.path.join(scene_path, 'cam0_wrt_table.npy'))
    trans = np.dot(align_mat, camera_poses[frame_id])
    
    # Compute workspace bounding box (for visualization, even if not filtering)
    h, w, _ = cloud.shape
    cloud_flat = cloud.reshape([h * w, 3])
    seg_flat = seg.reshape(h * w)
    depth_flat = depth.reshape(h * w)
    
    # Transform to table coordinates
    cloud_table = transform_point_cloud(cloud_flat, trans)
    
    # Get foreground (object) points
    valid_depth = depth_flat > 0
    foreground_mask = (seg_flat > 0) & valid_depth
    foreground_table = cloud_table[foreground_mask]
    
    if len(foreground_table) > 0:
        # Compute bbox in table coordinates
        outlier = 0.02
        xmin, ymin, zmin = foreground_table.min(axis=0) - outlier
        xmax, ymax, zmax = foreground_table.max(axis=0) + outlier
        
        # Transform bbox corners back to camera coordinates
        # Inverse of trans
        trans_inv = np.linalg.inv(trans)
        
        # 8 corners of the bbox in table coords
        corners_table = np.array([
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ])
        
        # Transform to camera coordinates
        corners_camera = transform_point_cloud(corners_table, trans_inv)
        
        # Get axis-aligned bbox in camera coordinates
        bbox_min_cam = corners_camera.min(axis=0)
        bbox_max_cam = corners_camera.max(axis=0)
        
        workspace_bbox = {
            'min_camera': bbox_min_cam,
            'max_camera': bbox_max_cam,
            'min_table': np.array([xmin, ymin, zmin]),
            'max_table': np.array([xmax, ymax, zmax]),
            'corners_camera': corners_camera,  # The actual rotated box corners
        }
    
    if remove_outlier:
        workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        mask = (depth_mask & workspace_mask)
    else:
        mask = depth_mask
    
    cloud_masked = cloud[mask]
    rgb_masked = rgb[mask] if rgb is not None else None
    
    meta_info = {
        'scene_name': scene_name,
        'scene_id': scene_id,
        'frame_id': frame_id,
        'camera': camera,
        'num_raw_points': cloud_masked.shape[0],
        'trans': trans,
        'workspace_bbox': workspace_bbox,
    }
    
    return cloud_masked, rgb_masked, meta_info


def sample_points(cloud, num_points, rgb=None, floor_sampling=False, trans=None):
    """
    Sample points from point cloud.
    
    Args:
        cloud: (N, 3) numpy array
        num_points: target number of points
        rgb: optional (N, 3) RGB values
        floor_sampling: if True, use floor-aware sampling (reduces floor point probability)
        trans: transformation matrix to table coords (required if floor_sampling=True)
        
    Returns:
        cloud_sampled: (num_points, 3) sampled points
        rgb_sampled: optional (num_points, 3) sampled RGB
        sampling_info: dict with sampling statistics
    """
    n = len(cloud)
    sampling_info = {'method': 'random', 'floor_points_ratio': None}
    
    if floor_sampling and trans is not None:
        # Use floor-aware sampling (same as training pipeline)
        idxs = sample_floor_aware(cloud, trans, num_points)
        sampling_info['method'] = 'floor_aware'
        
        # Compute floor point statistics for reporting
        ones = np.ones((n, 1))
        cloud_homo = np.hstack([cloud, ones])
        cloud_table = (trans @ cloud_homo.T).T[:, :3]
        heights = cloud_table[:, 2]
        floor_mask = heights < 0.01  # Same threshold as sample_floor_aware
        sampling_info['floor_points_ratio'] = floor_mask.sum() / n
        sampling_info['floor_points_in_sample'] = floor_mask[idxs].sum() / num_points
    else:
        # Random sampling
        if n >= num_points:
            idxs = np.random.choice(n, num_points, replace=False)
        else:
            # Need to pad with duplicates
            idxs1 = np.arange(n)
            idxs2 = np.random.choice(n, num_points - n, replace=True)
            idxs = np.concatenate([idxs1, idxs2])
    
    cloud_sampled = cloud[idxs]
    rgb_sampled = rgb[idxs] if rgb is not None else None
    
    return cloud_sampled, rgb_sampled, sampling_info


def main():
    parser = argparse.ArgumentParser(description="Visualize GraspNet-1B Point Cloud")
    # Default data_root uses the symlink in dataset/data/graspnet
    default_data_root = '/datasets/graspnet'
    parser.add_argument("--data_root", type=str, default=default_data_root,
                        help="Path to GraspNet-1B dataset root")
    parser.add_argument("--scene_id", type=int, default=100,
                        help="Scene ID (0-189)")
    parser.add_argument("--frame_id", type=int, default=0,
                        help="Frame ID within scene (0-255)")
    parser.add_argument("--camera", type=str, default="realsense",
                        choices=["realsense", "kinect"],
                        help="Camera type")
    parser.add_argument("--num_points", type=int, default=15000,
                        help="Number of points to sample")
    parser.add_argument("--voxel_size", type=float, default=0.005,
                        help="Voxel size for visualization")
    parser.add_argument("--no_outlier_removal", action="store_true",
                        help="Disable workspace outlier removal")
    parser.add_argument("--include_floor", action="store_true", default=False,
                        help="Include floor/table points in visualization (same as --no_outlier_removal, "
                             "matches training pipeline when include_floor=True)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as script location)")
    
    # Grasp detection arguments
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to model checkpoint for grasp detection (optional)")
    parser.add_argument("--top_n_grasps", type=int, default=10,
                        help="Number of top grasps to visualize")
    parser.add_argument("--collision_thresh", type=float, default=0.01,
                        help="Collision threshold (<=0 to disable)")
    parser.add_argument("--seed_feat_dim", type=int, default=512,
                        help="Point-wise feature dimension for model")
    parser.add_argument("--backbone", type=str, default='resunet',
                        choices=['transformer', 'transformer_pretrained', 'sonata', 'pointnet2', 'resunet', 'resunet18', 'resunet_rgb', 'resunet18_rgb'],
                        help="Backbone architecture")
    
    # Ground truth grasp arguments
    parser.add_argument("--show_gt_grasps", action="store_true",
                        help="Show top N ground truth grasps from GraspNet-1B annotations")
    parser.add_argument("--gt_fric_thresh", type=float, default=0.4,
                        help="Friction coefficient threshold for GT grasps (lower=better)")
    
    # Model input visualization
    parser.add_argument("--model_input", action="store_true",
                        help="Visualize exact model input: apply floor-aware sampling and show " 
                             "the point cloud exactly as fed to the network (with offset shift, "
                             "voxelization, and coordinate scaling)")
    
    # Bounding box visualization
    parser.add_argument("--show_bbox", action="store_true",
                        help="Show workspace bounding box (the region kept after outlier removal)")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dumps')
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"GraspNet-1B Point Cloud Visualization")
    print(f"{'='*60}")
    print(f"Data root:     {args.data_root}")
    print(f"Scene:         scene_{str(args.scene_id).zfill(4)}")
    print(f"Frame:         {args.frame_id}")
    print(f"Camera:        {args.camera}")
    print(f"Num points:    {args.num_points}")
    print(f"Voxel size:    {args.voxel_size}")
    print(f"Output dir:    {output_dir}")
    if args.checkpoint_path:
        print(f"Checkpoint:    {args.checkpoint_path}")
        print(f"Top N grasps:  {args.top_n_grasps}")
    if args.show_gt_grasps:
        print(f"GT grasps:     enabled (fric_thresh={args.gt_fric_thresh})")
    if args.model_input:
        print(f"Model input:   enabled (floor-aware sampling + exact model preprocessing)")
    if args.include_floor:
        print(f"Include floor: enabled (keeping floor/table points)")
    if args.show_bbox:
        print(f"Show bbox:     enabled")
    print(f"{'='*60}\n")
    
    # Load scene
    # include_floor overrides no_outlier_removal (same as in graspnet_dataset.py)
    remove_outlier = not (args.no_outlier_removal or args.include_floor)
    print("Loading scene...")
    cloud_masked, rgb_masked, meta_info = load_graspnet_scene(
        args.data_root, 
        args.scene_id, 
        args.frame_id, 
        args.camera,
        remove_outlier=remove_outlier
    )
    
    print(f"  Raw points after masking: {meta_info['num_raw_points']}")
    
    # Sample points
    use_floor_sampling = args.model_input and meta_info.get('trans') is not None
    if use_floor_sampling:
        print(f"\nSampling to {args.num_points} points (floor-aware sampling)...")
    else:
        print(f"\nSampling to {args.num_points} points (random sampling)...")
    
    cloud_sampled, rgb_sampled, sampling_info = sample_points(
        cloud_masked, args.num_points, rgb_masked,
        floor_sampling=use_floor_sampling,
        trans=meta_info.get('trans')
    )
    
    if sampling_info['method'] == 'floor_aware':
        print(f"  Sampling method: floor-aware")
        print(f"  Floor points in original cloud: {sampling_info['floor_points_ratio']*100:.1f}%")
        print(f"  Floor points in sample: {sampling_info['floor_points_in_sample']*100:.1f}%")
    
    # Apply offset shift (same as in preprocessing) - CRITICAL for voxelization to work correctly
    offset = -cloud_sampled.min(axis=0)
    cloud_sampled_shifted = cloud_sampled + offset
    
    print(f"  Sampled cloud shape: {cloud_sampled_shifted.shape}")
    print(f"  Point cloud bounds (after offset):")
    print(f"    X: [{cloud_sampled_shifted[:, 0].min():.4f}, {cloud_sampled_shifted[:, 0].max():.4f}]")
    print(f"    Y: [{cloud_sampled_shifted[:, 1].min():.4f}, {cloud_sampled_shifted[:, 1].max():.4f}]")
    print(f"    Z: [{cloud_sampled_shifted[:, 2].min():.4f}, {cloud_sampled_shifted[:, 2].max():.4f}]")
    
    # If model_input mode, show the exact tensors that go to the network
    if args.model_input:
        coors = cloud_sampled_shifted / args.voxel_size
        feats = np.ones_like(cloud_sampled_shifted).astype(np.float32)  # or RGB if available
        if rgb_sampled is not None:
            feats = rgb_sampled.astype(np.float32) / 255.0
        
        # After collate_fn, coors become int32 (truncation, not rounding)
        coors_int = coors.astype(np.int32)
        unique_voxels = np.unique(coors_int, axis=0)
        
        print(f"\n  === MODEL INPUT TENSORS ===")
        print(f"  point_clouds: shape={cloud_sampled_shifted.shape}, dtype=float32")
        print(f"  coors: shape={coors.shape}, dtype=float32 (before int conversion)")
        print(f"  coors (int32): unique voxels = {len(unique_voxels)} (from {len(coors)} points)")
        print(f"  coors range:")
        print(f"    X: [{coors_int[:, 0].min()}, {coors_int[:, 0].max()}]")
        print(f"    Y: [{coors_int[:, 1].min()}, {coors_int[:, 1].max()}]")
        print(f"    Z: [{coors_int[:, 2].min()}, {coors_int[:, 2].max()}]")
        print(f"  feats: shape={feats.shape}, dtype=float32 ({'RGB' if rgb_sampled is not None else 'ones'})")
    
    # Generate output filenames
    base_name = f"scene{args.scene_id:04d}_frame{args.frame_id:04d}_{args.camera}"
    if args.model_input:
        base_name += "_model_input"
    
    png_path = os.path.join(output_dir, f"{base_name}_pointcloud.png")
    html_path = os.path.join(output_dir, f"{base_name}_pointcloud.html")
    
    # Grasp detection (optional)
    grasp_traces = None
    graspness_scores = None  # Will be used for voxel coloring
    if args.checkpoint_path:
        import torch
        
        print("\nRunning grasp detection...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {device}")
        
        # Enable backend optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Load model
        net = load_grasp_model(args.checkpoint_path, device, args.seed_feat_dim, args.backbone)
        
        # Run detection (also get graspness scores for voxel coloring)
        gg, graspness_scores = run_grasp_detection(
            net, cloud_sampled_shifted, device,
            voxel_size=args.voxel_size,
            collision_thresh=args.collision_thresh,
            return_graspness=True
        )
        
        print(f"  Detected {len(gg)} grasps")
        if graspness_scores is not None:
            print(f"  Graspness scores: min={graspness_scores.min():.4f}, max={graspness_scores.max():.4f}, mean={graspness_scores.mean():.4f}")
        
        if len(gg) > 0:
            # Apply NMS and sort by score
            gg = gg.nms().sort_by_score()
            print(f"  After NMS: {len(gg)} grasps")
            print(f"  Best grasp score: {gg[0].score:.4f}")
            
            # Create visualization traces (grasps are already in shifted coordinates)
            grasp_traces = create_grasp_traces(gg, top_n=args.top_n_grasps, offset=None, 
                                               name_prefix="Pred", color_scheme="score")
            
            # Update output filename to indicate grasps included
            html_path = os.path.join(output_dir, f"{base_name}_pointcloud_grasps.html")
    
    # Ground truth grasps (optional)
    gt_grasp_traces = None
    if args.show_gt_grasps:
        print("\nLoading ground truth grasps...")
        try:
            gg_gt = load_gt_grasps(
                args.data_root,
                args.scene_id,
                args.frame_id,
                camera=args.camera,
                fric_coef_thresh=args.gt_fric_thresh,
                offset=offset  # Add offset to match shifted point cloud
            )
            
            if len(gg_gt) > 0:
                # Sort and take top N
                gg_gt = gg_gt.sort_by_score()
                print(f"  Best GT grasp score: {gg_gt[0].score:.4f}")
                
                # Create visualization traces (blue for GT)
                gt_grasp_traces = create_grasp_traces(
                    gg_gt, top_n=args.top_n_grasps, offset=None,
                    name_prefix="GT", color_scheme="blue"
                )
                
                # Update output filename
                if args.checkpoint_path:
                    html_path = os.path.join(output_dir, f"{base_name}_pointcloud_pred_gt_grasps.html")
                else:
                    html_path = os.path.join(output_dir, f"{base_name}_pointcloud_gt_grasps.html")
        except Exception as e:
            print(f"  Warning: Failed to load GT grasps: {e}")
            import traceback
            traceback.print_exc()
    
    # Combine all grasp traces
    all_grasp_traces = []
    if grasp_traces:
        all_grasp_traces.extend(grasp_traces)
    if gt_grasp_traces:
        all_grasp_traces.extend(gt_grasp_traces)
    
    # Create bounding box traces if requested
    bbox_traces = None
    print(f"\nDebug: show_bbox={args.show_bbox}, workspace_bbox is None: {meta_info.get('workspace_bbox') is None}")
    if args.show_bbox and meta_info.get('workspace_bbox') is not None:
        print("Creating workspace bounding box visualization...")
        workspace_bbox = meta_info['workspace_bbox']
        print(f"  corners_camera shape: {workspace_bbox['corners_camera'].shape}")
        print(f"  corners_camera:\n{workspace_bbox['corners_camera']}")
        bbox_traces = []
        
        # The bbox corners are in original camera coordinates, need to apply offset shift
        corners_shifted = workspace_bbox['corners_camera'] + offset
        
        # Create trace for the rotated bbox (actual workspace boundary)
        rotated_bbox_trace = create_rotated_bbox_trace(
            corners_shifted,
            name="Workspace BBox (objects + 2cm)",
            color='rgb(255,165,0)',  # Orange
            width=4
        )
        bbox_traces.append(rotated_bbox_trace)
        
        print(f"  offset applied: {offset}")
        print(f"  corners_shifted (first 2):\n{corners_shifted[:2]}")
        print(f"  Bounding box (table coords):")
        print(f"    X: [{workspace_bbox['min_table'][0]:.4f}, {workspace_bbox['max_table'][0]:.4f}]")
        print(f"    Y: [{workspace_bbox['min_table'][1]:.4f}, {workspace_bbox['max_table'][1]:.4f}]")
        print(f"    Z: [{workspace_bbox['min_table'][2]:.4f}, {workspace_bbox['max_table'][2]:.4f}]")
        print(f"  Number of bbox traces: {len(bbox_traces)}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Static PNG plot
    plot_downsampled_pointcloud(cloud_sampled_shifted, save_path=png_path)
    
    # Interactive HTML plot with voxels
    if args.model_input:
        title_prefix = f"[MODEL INPUT] Scene {args.scene_id}, Frame {args.frame_id}, {args.camera} - "
    else:
        title_prefix = f"Scene {args.scene_id}, Frame {args.frame_id}, {args.camera} - "
    plot_downsampled_pointcloud_interactive(
        cloud_sampled_shifted, 
        args.voxel_size, 
        save_path=html_path,
        title_prefix=title_prefix,
        grasp_traces=all_grasp_traces if all_grasp_traces else None,
        graspness_scores=graspness_scores,
        bbox_traces=bbox_traces
    )
    
    print(f"\n{'='*60}")
    print("Done! Output files:")
    print(f"  - PNG: {png_path}")
    print(f"  - HTML: {html_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
