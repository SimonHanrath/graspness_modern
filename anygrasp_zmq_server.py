"""
AnyGrasp ZMQ Server - receives point cloud and returns grasp candidates.

Run inside your Docker container:
    python anygrasp_zmq_server.py --port 5555 --checkpoint_path gsnet_dev_epoch13.tar

The Docker container should expose the port:
    docker run ... -p 5562:5557 ...

Then update ANYGRASP_ZMQ_ADDR in MolMoAnyGraspAgent.py to:
    ANYGRASP_ZMQ_ADDR = "tcp://127.0.0.1:5562"

Expected Response Format:
    [
        {
            "translation": [x, y, z],           # 3D position in CAMERA frame (meters)
            "rotation_matrix": [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]],
            "score": 0.85
        },
        ...
    ]
"""

import zmq
import numpy as np
import io
import json
import argparse
import os
import sys
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for container
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from graspnetAPI import GraspGroup
from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import spconv_collate_fn
from utils.collision_detector import ModelFreeCollisionDetector


def grasp_to_dict(grasp, offset=None):
    """Convert a single grasp to dictionary format."""
    translation = grasp.translation
    # Subtract offset to return grasps in original camera coordinate frame
    if offset is not None:
        translation = translation - offset
    
    GRASP_MAX_WIDTH = 0.08  # Maximum gripper width in meters
    return {
        'translation': translation.tolist(),
        'rotation_matrix': grasp.rotation_matrix.tolist(),
        'score': float(grasp.score),
        'width': GRASP_MAX_WIDTH,
        'height': float(grasp.height),
        'depth': float(grasp.depth),
    }


def grasps_to_json(grasps, offset=None):
    """Convert grasp group to JSON string for ZMQ response."""
    if len(grasps) == 0:
        return json.dumps([])
    grasp_dicts = [grasp_to_dict(g, offset) for g in grasps]
    return json.dumps(grasp_dicts)


def plot_downsampled_pointcloud(points, save_path="downsampled_pointcloud.png"):
    """
    Plot and save the downsampled point cloud to a file.
    
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
                                            grasp_traces=None, graspness_scores=None):
    """
    Create an interactive 3D plot of the downsampled point cloud using Plotly.
    Shows both the raw sampled points and the voxelized points as proper cubes.
    Optionally includes grasp visualizations and graspness-based voxel coloring.
    Saves as HTML file that can be opened in a browser.
    
    Args:
        points: (N, 3) numpy array of XYZ coordinates (after downsampling/padding)
        voxel_size: voxel size used for sparse convolution
        save_path: path to save the interactive HTML plot
        grasp_traces: optional list of Plotly traces for grasp visualization
        graspness_scores: optional (N,) array of graspness scores per point (for voxel coloring)
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
        from collections import defaultdict
        voxel_to_points = defaultdict(list)
        for i, vc in enumerate(voxel_coords):
            voxel_to_points[tuple(vc)].append(i)
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
        [0, 0, s], [s, 0, s], [s, s, s], [0, s, s],    # top face
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
        # Color by graspness score if available, otherwise by Z height
        if viz_graspness is not None:
            all_intensity[v_offset:v_offset + 8] = viz_graspness[idx]
        else:
            all_intensity[v_offset:v_offset + 8] = viz_centers[idx, 2]
    
    # Choose colorscale based on graspness availability
    if viz_graspness is not None:
        colorscale = 'RdYlGn'  # Red (low graspness) to Green (high graspness)
        colorbar_title = 'Graspness'
    else:
        colorscale = 'Plasma'
        colorbar_title = 'Z (m)'
    
    # Trace 1: voxel cubes
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
    
    fig = go.Figure(data=all_traces)
    
    n_grasps = len(grasp_traces) if grasp_traces else 0
    title = f'Point Cloud: {len(points)} sampled → {n_voxels} voxels (voxel_size={voxel_size}m)'
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
        lines_world: list of [p1, p2] line segments in world coordinates
    """
    finger_length = depth
    finger_thickness = 0.01
    half_width = width / 2

    # Define gripper as line segments in local frame
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
        p1_world = rotation_matrix @ np.array(line[0]) + translation
        p2_world = rotation_matrix @ np.array(line[1]) + translation
        lines_world.append([p1_world, p2_world])

    return lines_world


def create_grasp_traces(grasps, top_n=10, offset=None, name_prefix="Grasp", color_scheme="score"):
    """
    Create Plotly traces for grasp visualization.
    
    Args:
        grasps: GraspGroup object
        top_n: number of top grasps to visualize
        offset: optional offset to subtract from grasp positions
        name_prefix: prefix for grasp names in legend
        color_scheme: "score" for green-red based on score
    
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

        # Color: green (high score) to red (low score)
        if color_scheme == "score":
            if max_score > min_score:
                norm_score = (score - min_score) / (max_score - min_score)
            else:
                norm_score = 1.0
            r = int(255 * (1 - norm_score))
            g = int(255 * norm_score)
            b = 50
            color = f'rgb({r},{g},{b})'
        else:
            color = 'rgb(128,128,128)'

        lines = create_gripper_mesh_vertices(translation, rotation_matrix, width, depth)

        x_coords, y_coords, z_coords = [], [], []
        for line in lines:
            x_coords.extend([line[0][0], line[1][0], None])
            y_coords.extend([line[0][1], line[1][1], None])
            z_coords.extend([line[0][2], line[1][2], None])

        trace = go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='lines',
            name=f'{name_prefix} {i+1} (score={score:.3f})',
            line=dict(color=color, width=4),
            hovertemplate=f'{name_prefix} {i+1}<br>Score: {score:.4f}<br>Width: {width:.4f}m<extra></extra>',
        )
        traces.append(trace)

    return traces


def fit_floor_plane_ransac(points, distance_threshold=0.02, num_iterations=100):
    """
    Fit a floor plane using RANSAC (assumes floor is the dominant horizontal plane).
    
    Args:
        points: (N, 3) numpy array
        distance_threshold: max distance from plane to be considered inlier
        num_iterations: number of RANSAC iterations
    
    Returns:
        plane_model: (a, b, c, d) where ax + by + cz + d = 0
        floor_mask: boolean mask of floor points
    """
    best_inliers = 0
    best_plane = None
    best_mask = None
    
    n_points = len(points)
    
    for _ in range(num_iterations):
        # Sample 3 random points
        sample_idxs = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[sample_idxs]
        
        # Compute plane normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        
        # Check if plane is roughly horizontal (normal close to vertical)
        # In camera frame, Y is typically down, so floor normal should be ~[0, -1, 0] or [0, 1, 0]
        # But let's be flexible and just check it's not too tilted
        if abs(normal[1]) < 0.7:  # Not horizontal enough
            continue
        
        # Plane equation: normal . (p - p1) = 0 => a*x + b*y + c*z + d = 0
        d = -np.dot(normal, p1)
        
        # Compute distances to plane
        distances = np.abs(np.dot(points, normal) + d)
        inlier_mask = distances < distance_threshold
        n_inliers = inlier_mask.sum()
        
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_plane = (*normal, d)
            best_mask = inlier_mask
    
    return best_plane, best_mask


def compute_height_above_floor(points, floor_plane):
    """
    Compute signed height above the floor plane for each point.
    
    Args:
        points: (N, 3) numpy array
        floor_plane: (a, b, c, d) plane coefficients
    
    Returns:
        heights: (N,) signed distance above floor (positive = above)
    """
    a, b, c, d = floor_plane
    normal = np.array([a, b, c])
    # Signed distance (positive if on same side as normal)
    heights = np.dot(points, normal) + d
    # Make sure "above floor" is positive (floor normal might point up or down)
    if heights.mean() < 0:
        heights = -heights
    return heights


def intelligent_sample_points(points, num_point, strategy='height_biased', 
                               floor_percentile=10, height_power=3.0):
    """
    Intelligently sample points with bias towards object regions.
    
    Args:
        points: (N, 3) numpy array of XYZ coordinates
        num_point: target number of points
        strategy: sampling strategy
            - 'uniform': standard random sampling (baseline)
            - 'height_biased': sample with probability proportional to Z height
            - 'floor_removal': detect and remove floor, then uniform sample rest
            - 'height_above_floor': RANSAC floor detection + height-biased sampling
        floor_percentile: percentile of Z values to consider as floor level
        height_power: exponent for height-based weighting (higher = more aggressive)
    
    Returns:
        sampled_points: (num_point, 3) sampled point cloud
        sampling_weights: (N,) weights used for sampling (for visualization)
    """
    N = len(points)
    
    if N <= num_point:
        # Need to pad - just return with uniform weights
        idxs1 = np.arange(N)
        idxs2 = np.random.choice(N, num_point - N, replace=True)
        idxs = np.concatenate([idxs1, idxs2])
        return points[idxs], np.ones(N)
    
    if strategy == 'uniform':
        # Standard uniform random sampling
        weights = np.ones(N)
        idxs = np.random.choice(N, num_point, replace=False)
        
    elif strategy == 'height_biased':
        # Simple heuristic: higher points get higher sampling probability
        z_vals = points[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        
        # Normalize heights to [0, 1]
        if z_max - z_min > 1e-6:
            heights_norm = (z_vals - z_min) / (z_max - z_min)
        else:
            heights_norm = np.ones(N)
        
        # Apply power to increase bias (higher power = more aggressive)
        # Add small epsilon to ensure floor points still have some probability
        weights = heights_norm ** height_power + 0.1
        
        # Normalize to probability distribution
        probs = weights / weights.sum()
        idxs = np.random.choice(N, num_point, replace=False, p=probs)
        
    elif strategy == 'floor_removal':
        # Split into floor and non-floor, sample 25% from floor and 75% from non-floor
        # In camera frame: Z = depth, so floor/table is FURTHER (higher Z), objects are CLOSER (lower Z)
        z_vals = points[:, 2]
        
        print(f"  Point cloud Z range: [{z_vals.min():.4f}, {z_vals.max():.4f}]")
        
        # Floor is at the furthest depth (highest Z). Points within 2cm of max Z are floor.
        z_max = z_vals.max()
        floor_threshold = z_max - 0.02  # 2cm from the furthest surface
        
        floor_mask = z_vals >= floor_threshold
        non_floor_mask = ~floor_mask
        
        floor_idxs = np.where(floor_mask)[0]
        non_floor_idxs = np.where(non_floor_mask)[0]
        
        print(f"  Floor threshold: Z >= {floor_threshold:.4f}")
        print(f"  Floor points: {len(floor_idxs)} ({100*len(floor_idxs)/N:.1f}%)")
        print(f"  Non-floor points: {len(non_floor_idxs)} ({100*len(non_floor_idxs)/N:.1f}%)")
        
        # Sample 25% from floor, 75% from non-floor
        n_floor_sample = int(num_point * 0)
        n_non_floor_sample = num_point - n_floor_sample
        
        print(f"  Sampling: {n_floor_sample} floor + {n_non_floor_sample} non-floor = {num_point}")
        
        # Sample from floor (with replacement if needed)
        if len(floor_idxs) >= n_floor_sample:
            floor_sample = np.random.choice(floor_idxs, n_floor_sample, replace=False)
        else:
            floor_sample = np.random.choice(floor_idxs, n_floor_sample, replace=True) if len(floor_idxs) > 0 else np.array([], dtype=int)
        
        # Sample from non-floor (with replacement if needed)
        if len(non_floor_idxs) >= n_non_floor_sample:
            non_floor_sample = np.random.choice(non_floor_idxs, n_non_floor_sample, replace=False)
        else:
            non_floor_sample = np.random.choice(non_floor_idxs, n_non_floor_sample, replace=True) if len(non_floor_idxs) > 0 else np.array([], dtype=int)
        
        idxs = np.concatenate([floor_sample, non_floor_sample])
        weights = non_floor_mask.astype(float)
        
    elif strategy == 'height_above_floor':
        # Most robust: detect floor plane with RANSAC, then remove floor entirely
        floor_plane, floor_mask = fit_floor_plane_ransac(points)
        
        if floor_plane is not None:
            heights = compute_height_above_floor(points, floor_plane)
            
            # Remove floor points entirely (within 2cm of floor)
            floor_threshold = 0.02  # 2cm above floor
            non_floor_mask = heights > floor_threshold
            non_floor_idxs = np.where(non_floor_mask)[0]
            
            n_floor = (~non_floor_mask).sum()
            print(f"  RANSAC: removing {n_floor}/{N} floor points ({100*n_floor/N:.1f}%)")
            print(f"  Remaining non-floor points: {len(non_floor_idxs)}")
            
            if len(non_floor_idxs) >= num_point:
                # Sample uniformly from non-floor points
                idxs = np.random.choice(non_floor_idxs, num_point, replace=False)
            else:
                # Not enough non-floor points - take all and pad
                idxs = np.concatenate([
                    non_floor_idxs,
                    np.random.choice(non_floor_idxs, num_point - len(non_floor_idxs), replace=True)
                ])
            weights = non_floor_mask.astype(float)
        else:
            # Fallback to height-biased if RANSAC fails
            print("  RANSAC floor detection failed, falling back to height_biased")
            z_vals = points[:, 2]
            z_min, z_max = z_vals.min(), z_vals.max()
            heights_norm = (z_vals - z_min) / (z_max - z_min + 1e-6)
            weights = heights_norm ** height_power + 0.01
            probs = weights / weights.sum()
            idxs = np.random.choice(N, num_point, replace=False, p=probs)
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    return points[idxs], weights


def crop_to_workspace(points, workspace_bounds):
    """
    Crop point cloud to a 3D workspace bounding box.
    
    This mimics get_workspace_mask() from the GraspNet-1B training pipeline,
    which uses segmentation + camera-to-table transforms to tightly crop
    around objects. Since we don't have segmentation labels at inference,
    we use a user-specified 3D bounding box instead.
    
    In a RealSense camera frame:
        X: left(-) / right(+)
        Y: up(-) / down(+)  
        Z: depth (away from camera)
    
    Args:
        points: (N, 3) numpy array of XYZ coordinates in camera frame
        workspace_bounds: dict with keys 'x', 'y', 'z', each a (min, max) tuple
                         e.g. {'x': (-0.3, 0.3), 'y': (-0.1, 0.2), 'z': (0.3, 0.8)}
    
    Returns:
        points_cropped: (M, 3) numpy array of points within bounds
    """
    xmin, xmax = workspace_bounds['x']
    ymin, ymax = workspace_bounds['y']
    zmin, zmax = workspace_bounds['z']
    
    mask = (
        (points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
        (points[:, 1] >= ymin) & (points[:, 1] <= ymax) &
        (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
    )
    
    points_cropped = points[mask]
    return points_cropped


def remove_statistical_outliers(points, nb_neighbors=20, std_ratio=2.0):
    """
    Remove statistical outliers from a point cloud.
    
    For each point, computes the mean distance to its k nearest neighbors.
    Points whose mean distance is beyond (global_mean + std_ratio * global_std)
    are considered outliers and removed.
    
    This helps clean up noisy points from real-world depth sensors that
    the model never saw during training on clean GraspNet-1B data.
    
    Args:
        points: (N, 3) numpy array
        nb_neighbors: number of nearest neighbors to consider
        std_ratio: standard deviation multiplier for outlier threshold
    
    Returns:
        points_clean: (M, 3) numpy array with outliers removed
    """
    from scipy.spatial import cKDTree
    
    N = len(points)
    if N <= nb_neighbors:
        return points
    
    tree = cKDTree(points)
    # Query k+1 neighbors (first neighbor is the point itself)
    dists, _ = tree.query(points, k=nb_neighbors + 1)
    # Mean distance to neighbors (exclude self at index 0)
    mean_dists = dists[:, 1:].mean(axis=1)
    
    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    threshold = global_mean + std_ratio * global_std
    
    inlier_mask = mean_dists < threshold
    points_clean = points[inlier_mask]
    
    return points_clean


def remove_table_plane(points, distance_threshold=0.005, min_plane_ratio=0.10,
                       num_iterations=500, keep_above_only=True):
    """
    Detect and remove the dominant plane (table surface) from the point cloud
    using RANSAC, keeping only the objects sitting on top.
    
    For speed, RANSAC runs on a small random subsample (5000 points) to find
    the plane, then applies it to the full cloud with a single dot product.
    This makes the total cost O(N) — matching the training pipeline speed.
    
    Args:
        points: (N, 3) numpy array in camera frame
        distance_threshold: max distance (meters) from plane to be considered
                           an inlier/table point. Default 0.005 = 5mm.
        min_plane_ratio: minimum fraction of points that must be in the plane
                        for it to be considered a valid table. Default 0.10 = 10%.
        num_iterations: number of RANSAC iterations
        keep_above_only: if True, only keep points on the object side of the plane
    
    Returns:
        points_objects: (M, 3) numpy array of non-table points
        plane_info: dict with plane parameters and stats, or None if no plane found
    """
    import time
    t0 = time.time()
    
    N = len(points)
    if N < 10:
        return points, None
    
    # --- Step 1: Find the plane using a small subsample (fast) ---
    ransac_sample_size = min(5000, N)
    if N > ransac_sample_size:
        subsample_idxs = np.random.choice(N, ransac_sample_size, replace=False)
        pts_sub = points[subsample_idxs]
    else:
        pts_sub = points
    
    M = len(pts_sub)
    
    # Vectorized RANSAC on the subsample
    sample_idxs = np.random.randint(0, M, size=(num_iterations, 3))
    p1 = pts_sub[sample_idxs[:, 0]]
    p2 = pts_sub[sample_idxs[:, 1]]
    p3 = pts_sub[sample_idxs[:, 2]]
    
    normals = np.cross(p2 - p1, p3 - p1)  # (K, 3)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norms.squeeze() > 1e-8
    valid_idxs = np.where(valid)[0]
    
    if len(valid_idxs) == 0:
        print("  Table removal: RANSAC failed to find any plane")
        return points, None
    
    # Evaluate all valid candidates at once against the subsample
    # pts_sub: (M, 3), valid_normals: (V, 3) => dists: (M, V)
    valid_normals = normals[valid_idxs] / norms[valid_idxs]
    valid_d = -np.einsum('ij,ij->i', valid_normals, p1[valid_idxs])
    dists_sub = np.abs(pts_sub @ valid_normals.T + valid_d[np.newaxis, :])
    inlier_counts = (dists_sub < distance_threshold).sum(axis=0)
    
    best_idx = inlier_counts.argmax()
    best_normal = valid_normals[best_idx]
    best_d = valid_d[best_idx]
    
    # --- Step 2: Apply the found plane to the FULL cloud (single dot product, O(N)) ---
    distances_full = np.abs(points @ best_normal + best_d)
    best_mask = distances_full < distance_threshold
    best_inliers = best_mask.sum()
    
    plane_ratio = best_inliers / N
    normal, d = best_normal, best_d
    
    # Check if the plane is large enough to be the table
    if plane_ratio < min_plane_ratio:
        elapsed = time.time() - t0
        print(f"  Table removal: largest plane only has {100 * plane_ratio:.1f}% of points "
              f"(need {100 * min_plane_ratio:.1f}%), skipping ({elapsed:.3f}s)")
        return points, None
    
    if keep_above_only:
        # Compute signed distances: positive = one side, negative = other
        signed_distances = np.dot(points, normal) + d
        
        # Determine which side of the plane is "toward the camera" (objects)
        # vs "away from camera" (below the table / background).
        #
        # In a camera frame, Z = depth (distance from camera). Objects sit
        # ON the table, so they have SMALLER Z than the table underside.
        # We compute the mean Z of each side and keep the side with smaller
        # mean Z (= closer to camera = objects).
        above_mask = signed_distances > distance_threshold
        below_mask = signed_distances < -distance_threshold
        
        n_above = above_mask.sum()
        n_below = below_mask.sum()
        
        if n_above == 0 and n_below == 0:
            # All points are on the plane
            print("  Table removal: all points lie on the plane!")
            return points, None
        
        # Use mean depth (Z) to determine which side has the objects
        if n_above > 0 and n_below > 0:
            mean_z_above = points[above_mask, 2].mean()
            mean_z_below = points[below_mask, 2].mean()
            # Keep the side closer to the camera (lower Z)
            if mean_z_above < mean_z_below:
                object_mask = above_mask
                side_label = f"above (mean_z={mean_z_above:.3f} < {mean_z_below:.3f})"
            else:
                object_mask = below_mask
                side_label = f"below (mean_z={mean_z_below:.3f} < {mean_z_above:.3f})"
        elif n_above > 0:
            object_mask = above_mask
            side_label = "above (only side with points)"
        else:
            object_mask = below_mask
            side_label = "below (only side with points)"
        
        points_objects = points[object_mask]
    else:
        # Keep all non-plane points (both sides)
        points_objects = points[~best_mask]
        side_label = "both sides"
    
    plane_info = {
        'normal': normal,
        'd': d,
        'n_plane_points': best_inliers,
        'plane_ratio': plane_ratio,
        'n_remaining': len(points_objects),
    }
    
    elapsed = time.time() - t0
    print(f"  Table removal: detected plane with {best_inliers} points "
          f"({100 * plane_ratio:.1f}% of cloud) in {elapsed:.2f}s")
    print(f"    Plane normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
    print(f"    Keeping {len(points_objects)} points ({side_label}), "
          f"removed {N - len(points_objects)} ({100 * (N - len(points_objects)) / N:.1f}%)")
    
    return points_objects, plane_info


def preprocess_point_cloud(points, num_point, voxel_size, sampling_strategy='uniform',
                           workspace_bounds=None, remove_outliers=True,
                           outlier_neighbors=20, outlier_std_ratio=2.0,
                           remove_table=False, table_distance_threshold=0.005,
                           table_min_ratio=0.10, table_ransac_iters=1000):
    """
    Preprocess point cloud for model input.
    
    Applies the following pipeline to match training preprocessing:
    1. Workspace cropping (coarse removal of background/far surfaces)
    2. Table plane removal via RANSAC (removes the dominant flat surface)
    3. Statistical outlier removal (cleans real sensor noise)
    4. Point sampling to target count
    5. Coordinate offset shift (all coords >= 0)
    
    Steps 1+2 together mimic get_workspace_mask() from GraspNet-1B training,
    which uses segmentation + camera transforms to isolate objects. Since we
    don't have segmentation labels at inference, we use geometric methods.
    
    Args:
        points: (N, 3) numpy array of XYZ coordinates
        num_point: target number of points to sample
        voxel_size: voxel size for sparse convolution
        sampling_strategy: strategy for downsampling
        workspace_bounds: dict with 'x', 'y', 'z' bounds, or None to skip
        remove_outliers: whether to apply statistical outlier removal
        outlier_neighbors: number of neighbors for outlier detection
        outlier_std_ratio: std multiplier for outlier threshold
        remove_table: whether to detect and remove the table plane
        table_distance_threshold: RANSAC inlier distance for table detection (meters)
        table_min_ratio: minimum fraction of points in plane to count as table
        table_ransac_iters: number of RANSAC iterations
    
    Returns:
        ret_dict: dictionary with preprocessed data
        offset: (3,) offset used to shift coordinates (for inverse transform)
    """
    n_raw = len(points)
    
    # Step 1: Workspace cropping (coarse, removes background/ceiling/far walls)
    if workspace_bounds is not None:
        points = crop_to_workspace(points, workspace_bounds)
        n_after_crop = len(points)
        print(f"  Workspace crop: {n_raw} -> {n_after_crop} points "
              f"({100 * n_after_crop / n_raw:.1f}% kept)")
        if n_after_crop == 0:
            raise ValueError(
                f"Workspace crop removed ALL points! Check your bounds: {workspace_bounds}. "
                f"Use the printed 'Point cloud bounds' above to choose appropriate values."
            )
    
    # Save the full cloud BEFORE table removal and sampling for collision detection.
    # This ensures the floor/table surface is present for collision checks even when
    # --remove_table is used (table removal helps the model but we still want to
    # detect grasps that collide with the table surface).
    cloud_full_for_collision = points.copy()
    
    # Step 2: Table plane removal (removes the dominant flat surface and its edges)
    if remove_table:
        points, plane_info = remove_table_plane(
            points,
            distance_threshold=table_distance_threshold,
            min_plane_ratio=table_min_ratio,
            num_iterations=table_ransac_iters,
        )
        if len(points) == 0:
            raise ValueError(
                "Table removal removed ALL points! The scene may be empty "
                "or the distance threshold may be too large."
            )
    
    # Step 3: Sample points (BEFORE outlier removal — much cheaper to run KNN on num_point
    # than on the full cloud which can be 200k+ points)
    print(f"  Sampling {len(points)} -> {num_point} points (strategy: {sampling_strategy})")
    cloud_sampled, sampling_weights = intelligent_sample_points(
        points, num_point, strategy=sampling_strategy
    )
    
    # Step 4: Statistical outlier removal (on the sampled subset)
    if remove_outliers and len(cloud_sampled) > outlier_neighbors:
        n_before = len(cloud_sampled)
        cloud_sampled = remove_statistical_outliers(
            cloud_sampled, nb_neighbors=outlier_neighbors, std_ratio=outlier_std_ratio
        )
        n_removed = n_before - len(cloud_sampled)
        print(f"  Outlier removal: {n_before} -> {len(cloud_sampled)} points "
              f"({n_removed} outliers removed, {100 * n_removed / n_before:.1f}%)")
        # Re-pad if outlier removal dropped us below num_point
        if len(cloud_sampled) < num_point:
            deficit = num_point - len(cloud_sampled)
            pad_idxs = np.random.choice(len(cloud_sampled), deficit, replace=True)
            cloud_sampled = np.concatenate([cloud_sampled, cloud_sampled[pad_idxs]], axis=0)

    # Step 5: Shift so all coords are >= 0 (CRITICAL: must match training preprocessing!)
    offset = -cloud_sampled.min(axis=0)  # [3,]
    cloud_sampled = cloud_sampled + offset

    ret_dict = {
        'point_clouds': cloud_sampled.astype(np.float32),
        'coors': cloud_sampled.astype(np.float32) / voxel_size,
        'feats': np.ones_like(cloud_sampled).astype(np.float32),
        'cloud_offset': offset.astype(np.float32),
    }
    return ret_dict, offset, cloud_full_for_collision


def run_grasp_detection(net, points, device, cfgs):
    """
    Run grasp detection on point cloud.
    
    Args:
        net: GraspNet model
        points: (N, 3) numpy array of XYZ coordinates
        device: torch device
        cfgs: configuration namespace
    
    Returns:
        gg: GraspGroup with detected grasps
        offset: coordinate offset for inverse transform
        cloud_sampled_shifted: (N, 3) preprocessed point cloud (for visualization)
        graspness_scores: (N,) graspness scores per point, or None
    """
    # Build workspace bounds if cropping is enabled
    workspace_bounds = None
    if getattr(cfgs, 'crop_workspace', False):
        workspace_bounds = {
            'x': (cfgs.ws_x_min, cfgs.ws_x_max),
            'y': (cfgs.ws_y_min, cfgs.ws_y_max),
            'z': (cfgs.ws_z_min, cfgs.ws_z_max),
        }
    
    # Preprocess: crop -> table removal -> outlier removal -> sample -> offset shift
    data_input, offset, cloud_full_for_collision = preprocess_point_cloud(
        points, cfgs.num_point, cfgs.voxel_size, 
        sampling_strategy=cfgs.sampling_strategy,
        workspace_bounds=workspace_bounds,
        remove_outliers=getattr(cfgs, 'remove_outliers', True),
        outlier_neighbors=getattr(cfgs, 'outlier_neighbors', 20),
        outlier_std_ratio=getattr(cfgs, 'outlier_std_ratio', 2.0),
        remove_table=getattr(cfgs, 'remove_table', False),
        table_distance_threshold=getattr(cfgs, 'table_distance_threshold', 0.005),
        table_min_ratio=getattr(cfgs, 'table_min_ratio', 0.10),
        table_ransac_iters=getattr(cfgs, 'table_ransac_iters', 1000),
    )
    
    cloud_sampled_shifted = data_input['point_clouds'].copy()
    
    # Prepare batch using spconv collate function
    batch_data = spconv_collate_fn([data_input])
    
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
        grasp_preds = pred_decode(end_points, use_stable_score=True)
    
    preds = grasp_preds[0].detach().cpu().numpy()
    
    # Extract graspness scores for visualization
    graspness_scores = None
    if 'graspness_score' in end_points:
        graspness_scores = end_points['graspness_score'].squeeze(1)[0].detach().cpu().numpy()
        print(f"  Graspness scores: min={graspness_scores.min():.4f}, max={graspness_scores.max():.4f}, mean={graspness_scores.mean():.4f}")
    
    if len(preds) == 0:
        print('No grasps detected')
        return GraspGroup(np.zeros((0, 17))), offset, cloud_sampled_shifted, graspness_scores
    
    # Apply filters if requested
    if cfgs.top_down_grasp:
        # Filter for near-vertical grasps (approach vector close to [0, 0, -1])
        # preds[:, 10] is the z-component of the approach vector
        mask = preds[:, 10] > 0.9
        preds = preds[mask]
        if len(preds) == 0:
            print('No grasps after top-down filter')
            return GraspGroup(np.zeros((0, 17))), offset, cloud_sampled_shifted, graspness_scores
    
    # Filter by gripper width
    if cfgs.max_gripper_width < 0.1:
        mask = preds[:, 1] <= cfgs.max_gripper_width
        preds = preds[mask]
        if len(preds) == 0:
            print('No grasps after width filter')
            return GraspGroup(np.zeros((0, 17))), offset, cloud_sampled_shifted, graspness_scores
    
    gg = GraspGroup(preds)
    
    # Collision detection — use the FULL cloud (before table removal / sampling)
    # in camera frame, matching test.py which uses `return_raw_cloud=True`.
    # The sampled 15k-point cloud is too sparse to reliably detect floor/table
    # collisions (gripper volumes slip between the gaps), and if --remove_table
    # was used, the floor/table points are missing entirely from the sampled cloud.
    #
    # Like test.py, we transform grasps to camera frame for collision detection,
    # then apply the collision mask to the original shifted-space grasps.
    if cfgs.collision_thresh > 0:
        n_before_cd = len(gg)
        # Transform grasp centers to camera frame for collision detection
        preds_cam = preds.copy()
        preds_cam[:, 13:16] = preds_cam[:, 13:16] - offset
        gg_cam = GraspGroup(preds_cam)
        mfcdetector = ModelFreeCollisionDetector(cloud_full_for_collision, voxel_size=cfgs.voxel_size_cd)
        collision_mask = mfcdetector.detect(gg_cam, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
        gg = gg[~collision_mask]
        print(f"  Collision detection: {n_before_cd} -> {len(gg)} grasps "
              f"(removed {(n_before_cd - len(gg))} colliding, "
              f"full cloud: {len(cloud_full_for_collision)} pts)")
        if len(gg) == 0:
            print('No grasps after collision detection')
            return GraspGroup(np.zeros((0, 17))), offset, cloud_sampled_shifted, graspness_scores
    
    return gg, offset, cloud_sampled_shifted, graspness_scores


def main():
    parser = argparse.ArgumentParser(description="AnyGrasp ZMQ Server")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ server port")
    parser.add_argument("--checkpoint_path", type=str, default="gsnet_resunet_epoch05.tar", 
                        help="Path to model checkpoint")
    parser.add_argument("--seed_feat_dim", type=int, default=512, help="Point-wise feature dimension")
    parser.add_argument("--num_point", type=int, default=15000, help="Number of points to sample")
    parser.add_argument("--voxel_size", type=float, default=0.005, help="Voxel size for sparse convolution")
    parser.add_argument("--collision_thresh", type=float, default=0.01, 
                        help="Collision threshold (set <= 0 to disable)")
    parser.add_argument("--voxel_size_cd", type=float, default=0.01, 
                        help="Voxel size for collision detection")
    parser.add_argument("--max_gripper_width", type=float, default=0.1, 
                        help="Maximum gripper width (<=0.1m)")
    parser.add_argument("--top_down_grasp", action="store_true", 
                        help="Filter for top-down grasps only")
    parser.add_argument("--max_grasps", type=int, default=50, 
                        help="Maximum number of grasps to return")
    parser.add_argument("--sampling_strategy", type=str, default="uniform",
                        choices=["uniform", "height_biased", "floor_removal", "height_above_floor"],
                        help="Point cloud sampling strategy: "
                             "'uniform' (random), "
                             "'height_biased' (prefer higher Z points), "
                             "'floor_removal' (remove lowest percentile), "
                             "'height_above_floor' (RANSAC floor detection + height bias)")
    parser.add_argument("--top_n_grasps_viz", type=int, default=20,
                        help="Number of top grasps to show in the visualization")
    parser.add_argument("--boundary_margin", type=float, default=0.05,
                        help="Reject grasps within this distance (meters) of the point cloud's "
                             "X/Y boundary. Prevents edge-of-scene grasps where context is "
                             "incomplete. Set <= 0 to disable. Default: 0.01 (1 cm).")
    
    # Workspace cropping arguments (mimics get_workspace_mask from GraspNet-1B training)
    parser.add_argument("--crop_workspace", action="store_true",
                        help="Enable 3D workspace bounding box cropping. "
                             "This is STRONGLY recommended for real-world cameras to match "
                             "the training data distribution. Without it, table edges and "
                             "background surfaces produce false grasp detections.")
    parser.add_argument("--ws_x_min", type=float, default=0.0,
                        help="Workspace X min in camera frame (meters). Negative=left of center.")
    parser.add_argument("--ws_x_max", type=float, default=0.5,
                        help="Workspace X max in camera frame (meters). Positive=right of center.")
    parser.add_argument("--ws_y_min", type=float, default=-0.1,
                        help="Workspace Y min in camera frame (meters). Negative=above center (camera Y points down).")
    parser.add_argument("--ws_y_max", type=float, default=0.4,
                        help="Workspace Y max in camera frame (meters). Positive=below center.")
    parser.add_argument("--ws_z_min", type=float, default=0.0,
                        help="Workspace Z min / near depth in camera frame (meters).")
    parser.add_argument("--ws_z_max", type=float, default=0.40,
                        help="Workspace Z max / far depth in camera frame (meters).")
    
    # Outlier removal arguments (disabled by default — training pipeline doesn't use it)
    parser.add_argument("--outlier_removal", action="store_true",
                        help="Enable statistical outlier removal (not used in training, adds latency)")
    parser.add_argument("--outlier_neighbors", type=int, default=20,
                        help="Number of neighbors for statistical outlier detection")
    parser.add_argument("--outlier_std_ratio", type=float, default=2.0,
                        help="Std deviation multiplier for outlier threshold (higher=less aggressive)")
    
    # Table plane removal arguments
    parser.add_argument("--remove_table", action="store_true",
                        help="Detect and remove the table/dominant plane via RANSAC. "
                             "This removes the flat surface AND its edges, leaving only objects. "
                             "Strongly recommended to avoid false grasps on table edges.")
    parser.add_argument("--table_distance_threshold", type=float, default=0.005,
                        help="RANSAC inlier distance for table plane detection (meters). "
                             "Points within this distance of the plane are considered table. "
                             "Increase if table is rough/textured. Default: 0.005 (5mm).")
    parser.add_argument("--table_min_ratio", type=float, default=0.10,
                        help="Minimum fraction of points that must belong to the plane "
                             "for it to be considered the table. Default: 0.10 (10%%).")
    parser.add_argument("--table_ransac_iters", type=int, default=1000,
                        help="Number of RANSAC iterations for table detection. "
                             "More iterations = more robust but slower. Default: 1000.")
    
    args = parser.parse_args()
    
    # Derived flags
    args.remove_outliers = args.outlier_removal

    # Clamp gripper width
    args.max_gripper_width = max(0, min(0.1, args.max_gripper_width))

    # Initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enable backend optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Initialize GraspNet model (resunet backbone, xyz only, with stable score)
    print(f"Loading model from {args.checkpoint_path}...")
    net = GraspNet(seed_feat_dim=args.seed_feat_dim, is_training=False, backbone='resunet', enable_stable_score=True)
    net.to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"-> Loaded checkpoint (epoch: {checkpoint['epoch']})")
    
    net.eval()

    # Initialize ZeroMQ Server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, -1)
    socket.bind(f"tcp://*:{args.port}")

    print(f"\nAnyGrasp ZMQ Server running on port {args.port}...")
    print(f"Configuration:")
    print(f"  - num_point: {args.num_point}")
    print(f"  - voxel_size: {args.voxel_size}")
    print(f"  - sampling_strategy: {args.sampling_strategy}")
    print(f"  - collision_thresh: {args.collision_thresh}")
    print(f"  - max_gripper_width: {args.max_gripper_width}")
    print(f"  - top_down_grasp: {args.top_down_grasp}")
    print(f"  - max_grasps: {args.max_grasps}")
    print(f"  - boundary_margin: {args.boundary_margin}m")
    if args.crop_workspace:
        print(f"  - crop_workspace: ENABLED")
        print(f"    X: [{args.ws_x_min:.3f}, {args.ws_x_max:.3f}] m")
        print(f"    Y: [{args.ws_y_min:.3f}, {args.ws_y_max:.3f}] m")
        print(f"    Z: [{args.ws_z_min:.3f}, {args.ws_z_max:.3f}] m")
    else:
        print(f"  - crop_workspace: DISABLED (WARNING: table edges may cause false grasps)")
    print(f"  - outlier_removal: {args.remove_outliers}")
    if args.remove_outliers:
        print(f"    neighbors={args.outlier_neighbors}, std_ratio={args.outlier_std_ratio}")
    if args.remove_table:
        print(f"  - remove_table: ENABLED")
        print(f"    distance_threshold={args.table_distance_threshold}m, "
              f"min_ratio={args.table_min_ratio}, ransac_iters={args.table_ransac_iters}")
    else:
        print(f"  - remove_table: DISABLED")
    print("Waiting for point cloud data...")

    while True:
        try:
            # Receive compressed point cloud data
            compressed_data = socket.recv()
            print(f"\n{'='*50}")
            print(f"Received {len(compressed_data)} bytes")

            # Decompress using numpy
            with io.BytesIO(compressed_data) as f:
                npzfile = np.load(f)
                xyzrgb = npzfile["xyzrgb"]

            print(f"Point cloud shape: {xyzrgb.shape}")
            print(f"  - Number of points: {xyzrgb.shape[0]}")
            print(f"  - Dimensions: {xyzrgb.shape[1]} (x, y, z, r, g, b)")

            # Extract XYZ coordinates only (resunet backbone doesn't use RGB)
            points = xyzrgb[:, :3].astype(np.float32)

            print(f"\nPoint cloud bounds:")
            print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
            print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
            print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

            # Run grasp detection
            print("\nRunning grasp detection...")
            gg, offset, cloud_sampled_shifted, graspness_scores = run_grasp_detection(net, points, device, args)

            if len(gg) == 0:
                print("No grasps detected!")
                response = json.dumps([])
            else:
                # Apply NMS and sort by score
                gg = gg.nms().sort_by_score()
                
                # Filter out grasps at the X/Y boundary of the scene
                if args.boundary_margin > 0:
                    x_min, x_max = points[:, 0].min(), points[:, 0].max()
                    y_min, y_max = points[:, 1].min(), points[:, 1].max()
                    m = args.boundary_margin
                    n_before = len(gg)
                    # Grasp centers are in shifted space; subtract offset to get camera frame
                    centers_cam = gg.translations - offset  # (N, 3)
                    keep = (
                        (centers_cam[:, 0] > x_min + m) &
                        (centers_cam[:, 0] < x_max - m) &
                        (centers_cam[:, 1] > y_min + m) &
                        (centers_cam[:, 1] < y_max - m)
                    )
                    gg = gg[keep]
                    print(f"Boundary filter (margin={m}m): {n_before} -> {len(gg)} grasps "
                          f"(removed {n_before - len(gg)} at X/Y edges)")
                
                # Limit number of grasps
                if len(gg) > args.max_grasps:
                    gg = gg[:args.max_grasps]
                
                if len(gg) > 0:
                    print(f"Detected {len(gg)} grasps after NMS")
                    print(f"Best grasp score: {gg[0].score:.4f}")
                else:
                    print("No grasps remaining after boundary filter")
                
                # Convert to JSON (includes offset subtraction for camera coordinates)
                response = grasps_to_json(gg, offset)

            # Generate visualization with grasps and graspness coloring
            grasp_traces = None
            if len(gg) > 0:
                grasp_traces = create_grasp_traces(
                    gg, top_n=args.top_n_grasps_viz, offset=None,
                    name_prefix="Grasp", color_scheme="score"
                )
            
            plot_downsampled_pointcloud(cloud_sampled_shifted, save_path="downsampled_pointcloud.png")
            plot_downsampled_pointcloud_interactive(
                cloud_sampled_shifted, args.voxel_size,
                save_path="downsampled_pointcloud.html",
                grasp_traces=grasp_traces,
                graspness_scores=graspness_scores
            )

            # Send response
            socket.send_json(response)
            print(f"Sent {len(gg) if len(gg) > 0 else 0} grasps.")
            print(f"{'='*50}")

        except Exception as e:
            print(f"Error processing request: {e}")
            import traceback
            traceback.print_exc()
            # Send empty response on error
            socket.send_json(json.dumps([]))


if __name__ == "__main__":
    main()
