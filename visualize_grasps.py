"""
Grasp visualization script without Open3D dependency.
Supports transformer, transformer_pretrained, and pointnet2 backbones.

Author: Updated for modern backbone support
"""

import os
import sys
import numpy as np
import argparse
from PIL import Image
import time
import scipy.io as scio
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from graspnetAPI.graspnet_eval import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import spconv_collate_fn
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask


def parse_args():
    parser = argparse.ArgumentParser(description='Grasp visualization with modern backbone support')
    parser.add_argument('--dataset_root', default='/data/datasets/graspnet', 
                        help='Path to GraspNet dataset root')
    parser.add_argument('--checkpoint_path', required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dump_dir', default='./dumps/visualization',
                        help='Directory to save outputs')
    parser.add_argument('--seed_feat_dim', default=512, type=int, 
                        help='Point-wise feature dimension')
    parser.add_argument('--camera', default='kinect', choices=['realsense', 'kinect'],
                        help='Camera type')
    parser.add_argument('--num_point', type=int, default=15000, 
                        help='Number of points to sample')
    parser.add_argument('--voxel_size', type=float, default=0.005, 
                        help='Voxel size for sparse convolution')
    parser.add_argument('--collision_thresh', type=float, default=0.01,
                        help='Collision threshold (set < 0 to disable)')
    parser.add_argument('--voxel_size_cd', type=float, default=0.01, 
                        help='Voxel size for collision detection')
    parser.add_argument('--scene', type=str, default='0100',
                        help='Scene number (e.g., 0100)')
    parser.add_argument('--index', type=str, default='0000',
                        help='Frame index (e.g., 0000)')
    
    # Backbone options
    parser.add_argument('--backbone', type=str, default='transformer', 
                        choices=['transformer', 'transformer_pretrained', 'pointnet2', 'resunet'],
                        help='Backbone architecture')
    parser.add_argument('--ptv3_pretrained_path', type=str, default=None,
                        help='Path to PTv3 pretrained weights (.pth file)')
    parser.add_argument('--enable_flash', action='store_true', default=False,
                        help='Enable flash attention in PTv3 backbone')
    parser.add_argument('--enable_stable_score', action='store_true', default=False,
                        help='Enable stable score prediction (use with models trained with --enable_stable_score)')
    
    # Visualization options
    parser.add_argument('--num_grasps', type=int, default=10,
                        help='Maximum number of grasps to visualize')
    parser.add_argument('--score_thresh', type=float, default=0.0,
                        help='Minimum grasp score threshold')
    parser.add_argument('--save_grasps', action='store_true', default=False,
                        help='Save grasp predictions to npy file')
    parser.add_argument('--show_all_grasps', action='store_true', default=False,
                        help='Show all grasps (not just top N)')
    
    # Alternative input: npz file
    parser.add_argument('--npz_path', type=str, default=None,
                        help='Path to npz file with xyzrgb data (alternative to dataset)')
    
    # Interactive visualization
    parser.add_argument('--interactive', action='store_true', default=False,
                        help='Create interactive HTML visualization using plotly (if installed)')
    
    return parser.parse_args()


def create_gripper_mesh(center, R, width, depth=0.02, height=0.004):
    """
    Create gripper mesh vertices and faces for matplotlib visualization.
    
    Args:
        center: (3,) grasp center position
        R: (3,3) rotation matrix
        width: gripper opening width
        depth: gripper finger depth
        height: gripper finger height
        
    Returns:
        vertices: (N, 3) mesh vertices
        faces: list of face vertex indices
    """
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02
    
    # Create box vertices for each gripper component
    def create_box(dx, dy, dz):
        """Create box vertices with dimensions dx, dy, dz starting at origin"""
        verts = np.array([
            [0, 0, 0], [dx, 0, 0], [dx, dy, 0], [0, dy, 0],
            [0, 0, dz], [dx, 0, dz], [dx, dy, dz], [0, dy, dz]
        ])
        return verts
    
    # Left finger
    left = create_box(depth + depth_base + finger_width, finger_width, height)
    left[:, 0] -= depth_base + finger_width
    left[:, 1] -= width/2 + finger_width
    left[:, 2] -= height/2
    
    # Right finger
    right = create_box(depth + depth_base + finger_width, finger_width, height)
    right[:, 0] -= depth_base + finger_width
    right[:, 1] += width/2
    right[:, 2] -= height/2
    
    # Bottom connector
    bottom = create_box(finger_width, width, height)
    bottom[:, 0] -= finger_width + depth_base
    bottom[:, 1] -= width/2
    bottom[:, 2] -= height/2
    
    # Tail
    tail = create_box(tail_length, finger_width, height)
    tail[:, 0] -= tail_length + finger_width + depth_base
    tail[:, 1] -= finger_width / 2
    tail[:, 2] -= height/2
    
    # Combine all vertices
    vertices = np.vstack([left, right, bottom, tail])
    
    # Transform to world coordinates
    vertices = np.dot(R, vertices.T).T + center
    
    # Define faces for each box (6 faces per box, 4 boxes)
    faces = []
    for box_idx in range(4):
        offset = box_idx * 8
        # Each box has 6 faces
        box_faces = [
            [0, 1, 2, 3],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # front
            [2, 3, 7, 6],  # back
            [0, 3, 7, 4],  # left
            [1, 2, 6, 5],  # right
        ]
        for face in box_faces:
            faces.append([f + offset for f in face])
    
    return vertices, faces


def load_scene_data(args):
    """Load and preprocess scene data."""
    root = args.dataset_root
    camera_type = args.camera
    scene_id = 'scene_' + args.scene
    index = args.index
    
    # Auto-enable RGB for transformer_pretrained backbone
    use_rgb = (args.backbone == 'transformer_pretrained')
    
    # Load images
    depth_path = os.path.join(root, 'scenes', scene_id, camera_type, 'depth', index + '.png')
    rgb_path = os.path.join(root, 'scenes', scene_id, camera_type, 'rgb', index + '.png')
    seg_path = os.path.join(root, 'scenes', scene_id, camera_type, 'label', index + '.png')
    meta_path = os.path.join(root, 'scenes', scene_id, camera_type, 'meta', index + '.mat')
    
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth image not found: {depth_path}")
    
    depth = np.array(Image.open(depth_path))
    rgb = np.array(Image.open(rgb_path))
    seg = np.array(Image.open(seg_path))
    meta = scio.loadmat(meta_path)
    
    try:
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
    except Exception as e:
        raise RuntimeError(f"Error loading meta data: {repr(e)}")
    
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], 
                        intrinsic[0][2], intrinsic[1][2], factor_depth)
    
    # Generate point cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    
    # Get valid points
    depth_mask = (depth > 0)
    camera_poses = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'camera_poses.npy'))
    align_mat = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'cam0_wrt_table.npy'))
    trans = np.dot(align_mat, camera_poses[int(index)])
    workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
    mask = (depth_mask & workspace_mask)
    
    cloud_masked = cloud[mask]
    colors_masked = rgb.reshape(-1, 3)[mask.flatten()]
    
    # Sample points
    if len(cloud_masked) >= args.num_point:
        idxs = np.random.choice(len(cloud_masked), args.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), args.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    
    cloud_sampled = cloud_masked[idxs]
    colors_sampled = colors_masked[idxs]
    
    # Shift so all coords are >= 0 (CRITICAL: must match training preprocessing!)
    offset = -cloud_sampled.min(axis=0)
    cloud_sampled = cloud_sampled + offset
    
    # Features: RGB for transformer_pretrained, ones otherwise
    if use_rgb:
        feats = colors_sampled.astype(np.float32) / 255.0
    else:
        feats = np.ones_like(cloud_sampled).astype(np.float32)
    
    ret_dict = {
        'point_clouds': cloud_sampled.astype(np.float32),
        'coors': cloud_sampled.astype(np.float32) / args.voxel_size,
        'feats': feats,
        'cloud_offset': offset.astype(np.float32),
        'cloud_colors': colors_sampled.astype(np.float32) / 255.0,
    }
    
    return ret_dict


def load_npz_data(args):
    """
    Load and preprocess data from npz file.
    Expects 'xyzrgb' key with shape (N, 6) containing XYZ coordinates and RGB values.
    RGB should be in range [0, 255].
    """
    if not os.path.exists(args.npz_path):
        raise FileNotFoundError(f"NPZ file not found: {args.npz_path}")
    
    # Auto-enable RGB for transformer_pretrained backbone
    use_rgb = (args.backbone == 'transformer_pretrained')
    
    data = np.load(args.npz_path)
    
    if 'xyzrgb' in data:
        xyzrgb = data['xyzrgb'].astype(np.float32)
        cloud = xyzrgb[:, :3]
        colors = xyzrgb[:, 3:6]
        # Normalize RGB if in [0, 255] range
        if colors.max() > 1.0:
            colors = colors / 255.0
    elif 'point_clouds' in data:
        cloud = data['point_clouds'].astype(np.float32)
        if 'cloud_colors' in data:
            colors = data['cloud_colors'].astype(np.float32)
            if colors.max() > 1.0:
                colors = colors / 255.0
        else:
            colors = np.ones_like(cloud) * 0.5  # Gray default
    else:
        raise ValueError("NPZ file must contain 'xyzrgb' or 'point_clouds' key")
    
    print(f"   Loaded point cloud with {len(cloud)} points from {args.npz_path}")
    
    # Sample points if needed
    if len(cloud) > args.num_point:
        idxs = np.random.choice(len(cloud), args.num_point, replace=False)
        cloud = cloud[idxs]
        colors = colors[idxs]
        print(f"   Sampled to {args.num_point} points")
    elif len(cloud) < args.num_point:
        idxs1 = np.arange(len(cloud))
        idxs2 = np.random.choice(len(cloud), args.num_point - len(cloud), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud = cloud[idxs]
        colors = colors[idxs]
        print(f"   Upsampled to {args.num_point} points")
    
    # Shift so all coords are >= 0 (CRITICAL: must match training preprocessing!)
    offset = -cloud.min(axis=0)
    cloud_shifted = cloud + offset
    
    # Features: RGB for transformer_pretrained, ones otherwise
    if use_rgb:
        feats = colors.astype(np.float32)
    else:
        feats = np.ones_like(cloud_shifted).astype(np.float32)
    
    ret_dict = {
        'point_clouds': cloud_shifted.astype(np.float32),
        'coors': cloud_shifted.astype(np.float32) / args.voxel_size,
        'feats': feats,
        'cloud_offset': offset.astype(np.float32),
        'cloud_colors': colors.astype(np.float32),
    }
    
    return ret_dict


def run_inference(data_input, args):
    """Run grasp detection inference."""
    # Enable backend optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Collate data for batch processing
    batch_data = spconv_collate_fn([data_input])
    
    # Initialize model
    net = GraspNet(
        seed_feat_dim=args.seed_feat_dim,
        is_training=False,
        backbone=args.backbone,
        ptv3_pretrained_path=args.ptv3_pretrained_path,
        enable_flash=args.enable_flash,
        enable_stable_score=args.enable_stable_score,
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    start_epoch = checkpoint['epoch']
    print(f"-> Loaded checkpoint {args.checkpoint_path} (epoch: {start_epoch})")
    print(f"   Backbone: {args.backbone}")
    if args.enable_stable_score:
        print("   Stable score enabled: grasps will be reweighted by (1 - stable_score) during ranking")
    
    net.eval()
    
    # Move data to device
    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)
    
    # Forward pass
    tic = time.time()
    with torch.inference_mode():
        end_points = net(batch_data)
        grasp_preds = pred_decode(end_points, use_stable_score=args.enable_stable_score)
    
    preds = grasp_preds[0].detach().cpu().numpy()
    
    # Transform grasp centers back to camera coordinates
    if 'cloud_offset' in batch_data:
        offset = batch_data['cloud_offset'][0].cpu().numpy()
        preds[:, 13:16] = preds[:, 13:16] - offset
    
    # Create GraspGroup
    gg = GraspGroup(preds)
    
    # Collision detection
    if args.collision_thresh > 0:
        cloud = data_input['point_clouds'] - data_input['cloud_offset']  # Back to camera coords
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=args.voxel_size_cd)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=args.collision_thresh)
        gg = gg[~collision_mask]
        print(f"   Collision filtering: {collision_mask.sum()} / {len(collision_mask)} grasps removed")
    
    toc = time.time()
    print(f'   Inference time: {toc - tic:.2f}s')
    print(f'   Total grasps detected: {len(gg)}')
    
    return gg


def visualize_grasps(point_cloud, colors, grasp_group, args, scene_id, index):
    """Create matplotlib visualization of point cloud with grasps."""
    # Apply NMS and sort by score
    gg = grasp_group.nms()
    gg = gg.sort_by_score()
    
    # Filter by score threshold
    if args.score_thresh > 0:
        mask = gg.scores >= args.score_thresh
        gg = gg[mask]
        print(f"   After score filtering (>{args.score_thresh}): {len(gg)} grasps")
    
    # Limit number of grasps
    num_grasps = min(args.num_grasps, len(gg)) if not args.show_all_grasps else len(gg)
    if num_grasps > 0:
        gg = gg[:num_grasps]
    
    print(f"   Visualizing {len(gg)} grasps")
    
    # Transform point cloud back to camera coordinates for visualization
    offset = args._data_offset  # Stored during data loading
    pc_camera = point_cloud - offset
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot point cloud with colors
    ax.scatter(pc_camera[:, 0], pc_camera[:, 1], pc_camera[:, 2],
               c=colors, marker='.', s=1, alpha=0.6)
    
    # Color scheme for grasps
    cmap = plt.cm.RdYlGn  # Red to Green colormap (bad to good)
    
    # Plot gripper meshes
    for idx in range(len(gg)):
        grasp = gg[idx]
        score = grasp.score
        width = grasp.width
        depth = grasp.depth
        center = grasp.translation
        R = grasp.rotation_matrix
        
        # Get gripper mesh
        vertices, faces = create_gripper_mesh(center, R, width, depth)
        
        # Color based on score (or rank for top grasps)
        if idx < 3:
            # Top 3 grasps: red, green, blue
            colors_top = [[1.0, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 1.0]]
            color = colors_top[idx]
            alpha = 0.9
        else:
            # Others: color by score
            color = cmap(score)[:3]
            alpha = 0.6
        
        # Create polygon collection for the gripper
        for face in faces:
            verts = [vertices[i] for i in face]
            poly = Poly3DCollection([verts], alpha=alpha)
            poly.set_facecolor(color)
            poly.set_edgecolor('black')
            poly.set_linewidth(0.1)
            ax.add_collection3d(poly)
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'Grasp Visualization - Scene {args.scene}, Frame {args.index}\n'
                 f'Backbone: {args.backbone} | Showing {len(gg)} grasps after NMS\n'
                 f'Top 3: Red, Green, Blue', fontsize=12)
    
    # Set equal aspect ratio
    max_range = np.array([pc_camera[:, 0].max() - pc_camera[:, 0].min(),
                          pc_camera[:, 1].max() - pc_camera[:, 1].min(),
                          pc_camera[:, 2].max() - pc_camera[:, 2].min()]).max() / 2.0
    mid_x = (pc_camera[:, 0].max() + pc_camera[:, 0].min()) * 0.5
    mid_y = (pc_camera[:, 1].max() + pc_camera[:, 1].min()) * 0.5
    mid_z = (pc_camera[:, 2].max() + pc_camera[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # IMPORTANT: Invert Z-axis so camera view looks correct (Z points away from camera)
    ax.set_zlim(mid_z + max_range, mid_z - max_range)
    
    # Set viewing angle - looking from above at an angle
    ax.view_init(elev=60, azim=-60)
    
    # Create output directory
    vis_dir = os.path.join(args.dump_dir, scene_id, args.camera)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Save figure
    fig_path = os.path.join(vis_dir, f'{index}_grasps_{args.backbone}.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Visualization saved to: {fig_path}")
    
    # Save grasps if requested
    if args.save_grasps:
        grasp_path = os.path.join(vis_dir, f'{index}_grasps_{args.backbone}.npy')
        gg.save_npy(grasp_path)
        print(f"   Grasps saved to: {grasp_path}")
    
    # Create interactive HTML if requested
    if args.interactive:
        html_path = create_interactive_visualization(
            pc_camera, colors, gg, args, vis_dir, index
        )
        if html_path:
            print(f"   Interactive HTML saved to: {html_path}")
    
    return fig_path


def create_interactive_visualization(pc, colors, gg, args, vis_dir, index):
    """
    Create an interactive HTML visualization using plotly.
    Returns None if plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("   Warning: plotly not installed, skipping interactive visualization")
        print("   Install with: pip install plotly")
        return None
    
    # Subsample point cloud for performance (plotly can be slow with many points)
    max_points = 50000
    if len(pc) > max_points:
        idxs = np.random.choice(len(pc), max_points, replace=False)
        pc_vis = pc[idxs]
        colors_vis = colors[idxs]
    else:
        pc_vis = pc
        colors_vis = colors
    
    # Convert colors to plotly format
    colors_rgb = ['rgb({},{},{})'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255)) 
                  for c in colors_vis]
    
    # Create figure
    fig = go.Figure()
    
    # Add point cloud
    fig.add_trace(go.Scatter3d(
        x=pc_vis[:, 0], y=pc_vis[:, 1], z=pc_vis[:, 2],
        mode='markers',
        marker=dict(size=2, color=colors_rgb, opacity=0.9),
        name='Point Cloud',
        hoverinfo='skip'
    ))
    
    # Add gripper wireframes
    for idx in range(len(gg)):
        grasp = gg[idx]
        vertices, faces = create_gripper_mesh(
            grasp.translation, grasp.rotation_matrix, grasp.width, grasp.depth
        )
        
        # Color based on rank
        if idx == 0:
            color = 'red'
            name = f'Grasp {idx+1} (Best, score={grasp.score:.3f})'
        elif idx == 1:
            color = 'green'
            name = f'Grasp {idx+1} (score={grasp.score:.3f})'
        elif idx == 2:
            color = 'blue'
            name = f'Grasp {idx+1} (score={grasp.score:.3f})'
        else:
            # Color by score: interpolate between red (0) and green (1)
            r = int((1 - grasp.score) * 255)
            g = int(grasp.score * 255)
            color = f'rgb({r},{g},0)'
            name = f'Grasp {idx+1} (score={grasp.score:.3f})'
        
        # Create mesh for each face
        for face in faces:
            verts = vertices[face]
            # Close the polygon
            verts = np.vstack([verts, verts[0:1]])
            fig.add_trace(go.Scatter3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                mode='lines',
                line=dict(color=color, width=2),
                name=name if face == faces[0] else None,
                showlegend=(face == faces[0]),
                hoverinfo='name' if face == faces[0] else 'skip'
            ))
    
    # Update layout with inverted Z-axis to match camera coordinate convention
    fig.update_layout(
        title=f'Interactive Grasp Visualization<br>Backbone: {args.backbone} | {len(gg)} grasps',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            # Invert Z-axis so camera view looks correct (Z points away from camera)
            zaxis=dict(autorange='reversed'),
            # Set a good initial camera angle
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2),
                up=dict(x=0, y=0, z=-1)  # Z-up but inverted
            )
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Save HTML
    html_path = os.path.join(vis_dir, f'{index}_grasps_{args.backbone}_interactive.html')
    fig.write_html(html_path)
    
    return html_path


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Grasp Visualization")
    print("=" * 60)
    
    # Determine input source
    if args.npz_path:
        print(f"Input: {args.npz_path}")
        input_name = os.path.splitext(os.path.basename(args.npz_path))[0]
    else:
        print(f"Scene: scene_{args.scene}, Frame: {args.index}")
        input_name = f"scene_{args.scene}_{args.index}"
    
    print(f"Backbone: {args.backbone}")
    print(f"Checkpoint: {args.checkpoint_path}")
    if args.enable_stable_score:
        print(f"Stable Score: ENABLED")
    print("=" * 60)
    
    # Create dump directory
    os.makedirs(args.dump_dir, exist_ok=True)
    
    # Load scene data
    print("\n[1/3] Loading scene data...")
    if args.npz_path:
        data_dict = load_npz_data(args)
    else:
        data_dict = load_scene_data(args)
    args._data_offset = data_dict['cloud_offset']  # Store for visualization
    print(f"   Loaded {len(data_dict['point_clouds'])} points")
    
    # Run inference
    print("\n[2/3] Running inference...")
    grasp_group = run_inference(data_dict, args)
    
    # Visualize
    print("\n[3/3] Creating visualization...")
    if args.npz_path:
        scene_id = 'npz_input'
        index = input_name
    else:
        scene_id = 'scene_' + args.scene
        index = args.index
    
    fig_path = visualize_grasps(
        data_dict['point_clouds'],
        data_dict['cloud_colors'],
        grasp_group,
        args,
        scene_id,
        index
    )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def print_usage_examples():
    """Print usage examples for the script."""
    examples = """
Usage Examples:
===============

1. Visualize grasps from npz file with PointNet2 backbone:
   python visualize_grasps.py \\
       --npz_path test_scene_0_0.npz \\
       --checkpoint_path logs/gsnet_pointnet2_input_fix/gsnet_pointnet2_epoch25.tar \\
       --backbone pointnet2 \\
       --num_grasps 20

2. Visualize grasps from npz file with transformer (PTv3) backbone:
   python visualize_grasps.py \\
       --npz_path test_scene_0_0.npz \\
       --checkpoint_path logs/gsnet_dev/gsnet_dev_best.tar \\
       --backbone transformer \\
       --num_grasps 30

3. Visualize grasps from npz file with transformer_pretrained (PTv3 with Pointcept weights):
   python visualize_grasps.py \\
       --npz_path test_scene_0_0.npz \\
       --checkpoint_path logs/gsnet_ptv3_rgb/gsnet_ptv3_rgb_epoch03.tar \\
       --backbone transformer_pretrained \\
       --num_grasps 30

4. Visualize grasps from GraspNet dataset scene:
   python visualize_grasps.py \\
       --dataset_root /data/datasets/graspnet \\
       --scene 0100 --index 0000 \\
       --camera kinect \\
       --checkpoint_path logs/gsnet_dev/gsnet_dev_best.tar \\
       --backbone transformer

5. Save grasps and disable collision filtering:
   python visualize_grasps.py \\
       --npz_path test_scene_0_0.npz \\
       --checkpoint_path <checkpoint_path> \\
       --backbone pointnet2 \\
       --save_grasps \\
       --collision_thresh -1

6. Filter grasps by score threshold:
   python visualize_grasps.py \\
       --npz_path test_scene_0_0.npz \\
       --checkpoint_path <checkpoint_path> \\
       --backbone pointnet2 \\
       --score_thresh 0.5 \\
       --num_grasps 50

Supported backbones:
  - transformer: PTv3 encoder trained from scratch
  - transformer_pretrained: PTv3 encoder with Pointcept pretrained weights (uses RGB)
  - pointnet2: PointNet++ backbone
  - resunet: ResUNet-14D sparse convolution backbone
"""
    print(examples)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1 or '--help-examples' in sys.argv:
        if '--help-examples' in sys.argv:
            print_usage_examples()
            sys.exit(0)
    main()
