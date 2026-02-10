#!/usr/bin/env python3
"""
Compare 15k vs 300k point cloud sampling from GraspNet dataset.
Generates an interactive HTML visualization for Docker-friendly viewing.

Usage:
    python compare_sampling.py --scene 0 --view 0 --camera realsense
    python compare_sampling.py --scene 42 --view 128 --camera kinect
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image
import scipy.io as scio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask


def load_point_cloud(root, scene_id, view_id, camera='realsense', remove_outlier=True):
    """
    Load full point cloud from a scene/view.
    
    Returns:
        cloud_masked: (N, 3) point cloud after workspace masking
        rgb_masked: (N, 3) RGB colors (0-255)
        num_raw: original number of points before masking
    """
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    view_name = str(view_id).zfill(4)
    
    depth_path = os.path.join(root, 'scenes', scene_name, camera, 'depth', f'{view_name}.png')
    rgb_path = os.path.join(root, 'scenes', scene_name, camera, 'rgb', f'{view_name}.png')
    label_path = os.path.join(root, 'scenes', scene_name, camera, 'label', f'{view_name}.png')
    meta_path = os.path.join(root, 'scenes', scene_name, camera, 'meta', f'{view_name}.mat')
    
    # Check if files exist
    for path, name in [(depth_path, 'depth'), (rgb_path, 'rgb'), (meta_path, 'meta')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found: {path}")
    
    # Load data
    depth = np.array(Image.open(depth_path))
    rgb = np.array(Image.open(rgb_path))
    seg = np.array(Image.open(label_path)) if os.path.exists(label_path) else np.ones_like(depth)
    meta = scio.loadmat(meta_path)
    
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']
    
    camera_info = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], 
                              intrinsic[0][2], intrinsic[1][2], factor_depth)
    
    # Generate point cloud
    cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)
    
    # Create mask
    depth_mask = (depth > 0)
    
    if remove_outlier:
        camera_poses = np.load(os.path.join(root, 'scenes', scene_name, camera, 'camera_poses.npy'))
        align_mat = np.load(os.path.join(root, 'scenes', scene_name, camera, 'cam0_wrt_table.npy'))
        trans = np.dot(align_mat, camera_poses[view_id])
        workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        mask = depth_mask & workspace_mask
    else:
        mask = depth_mask
    
    cloud_masked = cloud[mask]
    rgb_masked = rgb[mask]
    
    num_raw = depth_mask.sum()
    
    return cloud_masked, rgb_masked, num_raw


def sample_points(cloud, rgb, num_points):
    """Random sample points from point cloud."""
    n = len(cloud)
    if n >= num_points:
        idxs = np.random.choice(n, num_points, replace=False)
    else:
        idxs1 = np.arange(n)
        idxs2 = np.random.choice(n, num_points - n, replace=True)
        idxs = np.concatenate([idxs1, idxs2])
    
    return cloud[idxs], rgb[idxs]


def compute_voxel_stats(cloud, voxel_size=0.005):
    """Compute voxelization statistics."""
    voxel_coords = np.floor(cloud / voxel_size).astype(np.int32)
    unique_voxels = np.unique(voxel_coords, axis=0)
    
    # Count points per voxel
    voxel_keys = [tuple(v) for v in voxel_coords]
    from collections import Counter
    counts = Counter(voxel_keys)
    count_values = list(counts.values())
    
    return {
        'num_voxels': len(unique_voxels),
        'min_per_voxel': min(count_values),
        'max_per_voxel': max(count_values),
        'mean_per_voxel': np.mean(count_values),
        'median_per_voxel': np.median(count_values),
        'empty_ratio': 0,  # Can't compute without knowing full grid
    }


def create_comparison_html(cloud_15k, rgb_15k, cloud_300k, rgb_300k, 
                           stats_15k, stats_300k, scene_id, view_id, output_path,
                           max_vis_points=50000, voxel_size=0.005):
    """Create interactive HTML comparison using Plotly with points/voxels toggle."""
    
    # Subsample for visualization (Plotly gets slow with too many points)
    # Left panel: show all 15k (no subsampling needed)
    # Right panel: subsample 300k to max_vis_points for browser performance
    
    def subsample_for_vis(cloud, rgb, max_points):
        if len(cloud) > max_points:
            idxs = np.random.choice(len(cloud), max_points, replace=False)
            return cloud[idxs], rgb[idxs]
        return cloud, rgb
    
    # Show all 15k points on left (no subsampling)
    cloud_15k_vis, rgb_15k_vis = cloud_15k, rgb_15k
    # Subsample 300k to max_vis_points on right
    cloud_300k_vis, rgb_300k_vis = subsample_for_vis(cloud_300k, rgb_300k, max_vis_points)
    
    # Convert RGB to numeric array for faster rendering (avoid string colors)
    def rgb_to_numeric(rgb):
        return rgb.astype(np.uint8)
    
    colors_15k = rgb_to_numeric(rgb_15k_vis)
    colors_300k = rgb_to_numeric(rgb_300k_vis)
    
    # Compute voxel data for both point clouds
    def compute_voxels(cloud, rgb, voxel_size):
        """Compute voxel centers and colors (same as model inference)."""
        voxel_coords = np.floor(cloud / voxel_size).astype(np.int32)
        unique_coords, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
        
        # Average color per voxel
        num_voxels = len(unique_coords)
        voxel_colors = np.zeros((num_voxels, 3), dtype=np.float32)
        voxel_counts = np.zeros(num_voxels, dtype=np.float32)
        
        np.add.at(voxel_colors, inverse_indices, rgb.astype(np.float32))
        np.add.at(voxel_counts, inverse_indices, 1)
        voxel_colors = (voxel_colors / voxel_counts[:, None]).astype(np.uint8)
        
        # Voxel centers
        voxel_centers = (unique_coords + 0.5) * voxel_size
        
        return unique_coords, voxel_centers, voxel_colors
    
    # Get voxel data
    voxel_coords_15k, voxel_centers_15k, voxel_colors_15k = compute_voxels(cloud_15k, rgb_15k, voxel_size)
    voxel_coords_300k, voxel_centers_300k, voxel_colors_300k = compute_voxels(cloud_300k, rgb_300k, voxel_size)
    
    # Find missing voxels: voxels in 300k but NOT in 15k
    # Convert to set of tuples for efficient lookup
    voxel_set_15k = set(map(tuple, voxel_coords_15k))
    voxel_set_300k = set(map(tuple, voxel_coords_300k))
    
    missing_voxel_coords = voxel_set_300k - voxel_set_15k
    shared_voxel_coords = voxel_set_300k & voxel_set_15k
    
    # Get centers for missing and shared voxels
    missing_mask = np.array([tuple(c) in missing_voxel_coords for c in voxel_coords_300k])
    shared_mask = np.array([tuple(c) in shared_voxel_coords for c in voxel_coords_300k])
    
    missing_centers = voxel_centers_300k[missing_mask]
    missing_colors = voxel_colors_300k[missing_mask]
    shared_centers = voxel_centers_300k[shared_mask]
    shared_colors = voxel_colors_300k[shared_mask]
    
    print(f"  Voxels in 15k: {len(voxel_set_15k):,}")
    print(f"  Voxels in 300k: {len(voxel_set_300k):,}")
    print(f"  Shared voxels: {len(shared_voxel_coords):,}")
    print(f"  Missing in 15k: {len(missing_voxel_coords):,} ({100*len(missing_voxel_coords)/len(voxel_set_300k):.1f}%)")
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=(
            f'15k Sampling ({len(voxel_centers_15k):,} voxels)',
            f'300k Sampling ({len(voxel_centers_300k):,} voxels)'
        ),
        horizontal_spacing=0.02
    )
    
    # Trace 0: 15k point cloud (visible by default)
    fig.add_trace(
        go.Scatter3d(
            x=cloud_15k_vis[:, 0],
            y=cloud_15k_vis[:, 1],
            z=cloud_15k_vis[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=colors_15k,
            ),
            name='15k points',
            hoverinfo='skip',
            visible=True
        ),
        row=1, col=1
    )
    
    # Trace 1: 15k voxels as markers (hidden by default)
    fig.add_trace(
        go.Scatter3d(
            x=voxel_centers_15k[:, 0],
            y=voxel_centers_15k[:, 1],
            z=voxel_centers_15k[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=voxel_colors_15k,
            ),
            name='15k voxels',
            hoverinfo='skip',
            visible=False
        ),
        row=1, col=1
    )
    
    # Trace 2: 300k point cloud (visible by default)
    fig.add_trace(
        go.Scatter3d(
            x=cloud_300k_vis[:, 0],
            y=cloud_300k_vis[:, 1],
            z=cloud_300k_vis[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=colors_300k,
            ),
            name='300k points',
            hoverinfo='skip',
            visible=True
        ),
        row=1, col=2
    )
    
    # Trace 3: 300k voxels as markers (hidden by default)
    fig.add_trace(
        go.Scatter3d(
            x=voxel_centers_300k[:, 0],
            y=voxel_centers_300k[:, 1],
            z=voxel_centers_300k[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=voxel_colors_300k,
            ),
            name='300k voxels',
            hoverinfo='skip',
            visible=False
        ),
        row=1, col=2
    )
    
    # Trace 4: Shared voxels (for "Show Missing" view) - left panel
    fig.add_trace(
        go.Scatter3d(
            x=shared_centers[:, 0],
            y=shared_centers[:, 1],
            z=shared_centers[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color='green',
                opacity=0.6,
            ),
            name='Shared voxels',
            hoverinfo='skip',
            visible=False
        ),
        row=1, col=1
    )
    
    # Trace 5: Shared voxels - right panel (green = covered by both)
    fig.add_trace(
        go.Scatter3d(
            x=shared_centers[:, 0],
            y=shared_centers[:, 1],
            z=shared_centers[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color='green',
                opacity=0.6,
            ),
            name='Shared voxels',
            hoverinfo='skip',
            visible=False
        ),
        row=1, col=2
    )
    
    # Trace 6: Missing voxels - right panel (RED = missing from 15k!)
    fig.add_trace(
        go.Scatter3d(
            x=missing_centers[:, 0],
            y=missing_centers[:, 1],
            z=missing_centers[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color='red',
                opacity=0.9,
            ),
            name=f'Missing ({len(missing_centers):,})',
            hoverinfo='skip',
            visible=False
        ),
        row=1, col=2
    )
    
    # Compute shared axis ranges for fair comparison
    all_points = np.vstack([cloud_15k, cloud_300k])
    x_range = [all_points[:, 0].min() - 0.05, all_points[:, 0].max() + 0.05]
    y_range = [all_points[:, 1].min() - 0.05, all_points[:, 1].max() + 0.05]
    z_range = [all_points[:, 2].min() - 0.05, all_points[:, 2].max() + 0.05]
    
    # Shared camera settings
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=1.5, z=1.0)
    )
    
    # Create toggle buttons for points vs voxels
    # Trace order: 0=15k_points, 1=15k_voxels, 2=300k_points, 3=300k_voxels, 4=shared_left, 5=shared_right, 6=missing_right
    updatemenus = [
        dict(
            type="buttons",
            direction="left",
            x=0.5,
            y=1.12,
            xanchor="center",
            buttons=[
                dict(
                    label="Show Points",
                    method="update",
                    args=[{"visible": [True, False, True, False, False, False, False]}]
                ),
                dict(
                    label="Show Voxels",
                    method="update", 
                    args=[{"visible": [False, True, False, True, False, False, False]}]
                ),
                dict(
                    label="Show Missing",
                    method="update",
                    args=[{"visible": [False, False, False, False, True, True, True]}]
                ),
            ],
            showactive=True,
            bgcolor="lightgray",
            font=dict(size=12),
        )
    ]
    
    # Update layout with toggle buttons
    fig.update_layout(
        title=dict(
            text=f'Point Cloud Sampling Comparison - Scene {scene_id}, View {view_id}<br><sup>Voxel size: {voxel_size}m | Green=shared | Red=missing from 15k ({len(missing_centers):,} voxels, {100*len(missing_centers)/len(voxel_set_300k):.1f}%)</sup>',
            x=0.5,
            font=dict(size=20)
        ),
        height=850,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        updatemenus=updatemenus,
        scene=dict(
            xaxis=dict(range=x_range, title='X'),
            yaxis=dict(range=y_range, title='Y'),
            zaxis=dict(range=z_range, title='Z'),
            camera=camera,
            aspectmode='data'
        ),
        scene2=dict(
            xaxis=dict(range=x_range, title='X'),
            yaxis=dict(range=y_range, title='Y'),
            zaxis=dict(range=z_range, title='Z'),
            camera=camera,
            aspectmode='data'
        ),
    )
    
    # Add statistics as annotations
    stats_text_15k = (
        f"<b>15k Sampling Stats:</b><br>"
        f"Unique voxels: {stats_15k['num_voxels']:,}<br>"
        f"Points/voxel: {stats_15k['mean_per_voxel']:.1f} avg, "
        f"{stats_15k['median_per_voxel']:.0f} median<br>"
        f"Range: {stats_15k['min_per_voxel']}-{stats_15k['max_per_voxel']}"
    )
    
    stats_text_300k = (
        f"<b>300k Sampling Stats:</b><br>"
        f"Unique voxels: {stats_300k['num_voxels']:,}<br>"
        f"Points/voxel: {stats_300k['mean_per_voxel']:.1f} avg, "
        f"{stats_300k['median_per_voxel']:.0f} median<br>"
        f"Range: {stats_300k['min_per_voxel']}-{stats_300k['max_per_voxel']}"
    )
    
    fig.add_annotation(
        x=0.25, y=-0.05, xref='paper', yref='paper',
        text=stats_text_15k, showarrow=False,
        font=dict(size=12), align='center',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='gray', borderwidth=1
    )
    
    fig.add_annotation(
        x=0.75, y=-0.05, xref='paper', yref='paper',
        text=stats_text_300k, showarrow=False,
        font=dict(size=12), align='center',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='gray', borderwidth=1
    )
    
    # Save HTML
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    print(f"\nVisualization saved to: {output_path}")
    

def main():
    parser = argparse.ArgumentParser(description='Compare 15k vs 300k point cloud sampling')
    parser.add_argument('--dataset_root', type=str, default='/datasets/graspnet',
                        help='Path to GraspNet dataset root')
    parser.add_argument('--scene', type=int, default=0, help='Scene ID (0-189)')
    parser.add_argument('--view', type=int, default=0, help='View/frame ID (0-255)')
    parser.add_argument('--camera', type=str, default='realsense', 
                        choices=['realsense', 'kinect'], help='Camera type')
    parser.add_argument('--voxel_size', type=float, default=0.005, 
                        help='Voxel size for statistics (default: 0.005m = 5mm)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML path (default: comparison_scene{scene}_view{view}.html)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no_outlier_removal', action='store_true',
                        help='Disable workspace outlier removal')
    parser.add_argument('--vis_points', type=int, default=15000,
                        help='Max points to visualize per panel (default: 15000 for performance)')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Set output path
    if args.output is None:
        args.output = f'comparison_scene{args.scene}_view{args.view}.html'
    
    print(f"Loading point cloud from Scene {args.scene}, View {args.view} ({args.camera})...")
    
    # Load full point cloud
    cloud_full, rgb_full, num_raw = load_point_cloud(
        args.dataset_root, args.scene, args.view, 
        args.camera, remove_outlier=not args.no_outlier_removal
    )
    
    print(f"  Raw depth points: {num_raw:,}")
    print(f"  After workspace masking: {len(cloud_full):,}")
    
    # Sample 15k and 300k (or max available)
    num_15k = 15000
    num_300k = min(300000, len(cloud_full))
    
    if len(cloud_full) < num_300k:
        print(f"  Warning: Only {len(cloud_full):,} points available, using all for '300k' comparison")
    
    print(f"\nSampling {num_15k:,} points...")
    cloud_15k, rgb_15k = sample_points(cloud_full, rgb_full, num_15k)
    
    print(f"Sampling {num_300k:,} points...")
    cloud_300k, rgb_300k = sample_points(cloud_full, rgb_full, num_300k)
    
    # Compute voxel statistics
    print(f"\nComputing voxel statistics (voxel_size={args.voxel_size}m)...")
    stats_15k = compute_voxel_stats(cloud_15k, args.voxel_size)
    stats_300k = compute_voxel_stats(cloud_300k, args.voxel_size)
    
    print(f"\n{'='*60}")
    print(f"{'Metric':<25} {'15k':<15} {'300k':<15}")
    print(f"{'='*60}")
    print(f"{'Input points':<25} {num_15k:<15,} {num_300k:<15,}")
    print(f"{'Unique voxels':<25} {stats_15k['num_voxels']:<15,} {stats_300k['num_voxels']:<15,}")
    print(f"{'Voxel coverage ratio':<25} {stats_15k['num_voxels']/stats_300k['num_voxels']*100:<14.1f}% {'100.0%':<15}")
    print(f"{'Mean pts/voxel':<25} {stats_15k['mean_per_voxel']:<15.2f} {stats_300k['mean_per_voxel']:<15.2f}")
    print(f"{'Median pts/voxel':<25} {stats_15k['median_per_voxel']:<15.0f} {stats_300k['median_per_voxel']:<15.0f}")
    print(f"{'Min pts/voxel':<25} {stats_15k['min_per_voxel']:<15} {stats_300k['min_per_voxel']:<15}")
    print(f"{'Max pts/voxel':<25} {stats_15k['max_per_voxel']:<15} {stats_300k['max_per_voxel']:<15}")
    print(f"{'='*60}")
    
    # Create visualization
    print(f"\nGenerating HTML visualization (max {args.vis_points:,} points per panel)...")
    create_comparison_html(
        cloud_15k, rgb_15k, cloud_300k, rgb_300k,
        stats_15k, stats_300k, args.scene, args.view, args.output,
        max_vis_points=args.vis_points,
        voxel_size=args.voxel_size
    )
    
    print(f"\nOpen {args.output} in your browser to view the comparison.")
    print("Tip: Use mouse to rotate, scroll to zoom, shift+drag to pan.")
    print("     Use buttons to toggle between Points / Voxels view.")


if __name__ == '__main__':
    main()
