"""
AnyGrasp ZMQ Server - receives point cloud and returns grasp candidates.

Run inside your Docker container:
    python anygrasp_zmq_server.py --port 5588 --checkpoint_path gsnet_dev_epoch10.tar --backbone resunet --graspness_threshold 0.01
    python anygrasp_zmq_server.py --port 5588 --checkpoint_path gsnet_sonata_epoch15_llrd.tar --backbone sonata --graspness_threshold 0.1 --max_angle_to_vertical_deg 35 --vertical_axis z
    python anygrasp_zmq_server.py --port 5588 --gsnet_resunet_epoch05.tar --backbone resunet --graspness_threshold 0.01 --enable_stable_score


The Docker container should expose the port:
    docker run ... -p 5577:5588 ...

Then update ANYGRASP_ZMQ_ADDR in MolMoAnyGraspAgent.py to:
    ANYGRASP_ZMQ_ADDR = "tcp://127.0.0.1:5577"

Expected Response Format:
    [
        {
            "translation": [x, y, z], # 3D position in CAMERA frame (meters)
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


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from graspnetAPI import GraspGroup
from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import spconv_collate_fn
from utils.collision_detector import ModelFreeCollisionDetector


def grasp_to_dict(grasp, offset=None, approach_offset=0.0):
    """Convert a single grasp to dictionary format."""
    translation = grasp.translation.copy()
    # Subtract offset to return grasps in original camera coordinate frame
    if offset is not None:
        translation = translation - offset

    # Apply offset along the grasp's approach direction (rotation_matrix[:, 0])
    # Positive values push the grasp deeper into the object along the approach axis
    if approach_offset != 0.0:
        approach_dir = grasp.rotation_matrix[:, 0]
        translation = translation + approach_offset * approach_dir

    GRASP_MAX_WIDTH = 0.08  # Maximum gripper width in meters
    return {
        'translation': translation.tolist(),
        'rotation_matrix': grasp.rotation_matrix.tolist(),
        'score': float(grasp.score),
        'width': GRASP_MAX_WIDTH,
        'height': float(grasp.height),
        'depth': float(grasp.depth),
    }


def grasps_to_json(grasps, offset=None, approach_offset=0.0):
    """Convert grasp group to JSON string for ZMQ response."""
    if len(grasps) == 0:
        return json.dumps([])
    grasp_dicts = [grasp_to_dict(g, offset, approach_offset=approach_offset) for g in grasps]
    return json.dumps(grasp_dicts)


def filter_grasps_by_vertical_angle(preds, max_angle_deg=25.0, vertical_axis='z'):
    """
    Keep grasps whose approach direction is within max_angle_deg of the vertical direction.

    Args:
        preds: (N, 17) grasp predictions from pred_decode
        max_angle_deg: maximum allowed angle to vertical in degrees
        vertical_axis: which camera-frame axis is treated as vertical ('x', 'y', or 'z')

    Returns:
        filtered_preds: subset of preds satisfying the angle constraint
    """
    if len(preds) == 0:
        return preds

    axis_to_idx = {'x': 0, 'y': 1, 'z': 2}
    if vertical_axis not in axis_to_idx:
        raise ValueError(f"vertical_axis must be one of {list(axis_to_idx.keys())}")

    # Approach vector is the first column of the 3x3 rotation matrix flattened in preds:
    # preds[:, 3:12] = rotation matrix, so approach = preds[:, [3, 6, 9]]
    approach = preds[:, [3, 6, 9]]
    vertical_idx = axis_to_idx[vertical_axis]

    # Angle to vertical up/down, so use absolute cosine
    cos_theta = np.abs(approach[:, vertical_idx])
    cos_threshold = np.cos(np.deg2rad(max_angle_deg))

    return preds[cos_theta >= cos_threshold]


def preprocess_point_cloud(points, num_point, voxel_size):
    """
    Preprocess point cloud for model input (matches training pipeline).

    Pipeline: sample -> offset shift (all coords >= 0)

    Args:
        points: (N, 3) numpy array of XYZ coordinates
        num_point: target number of points to sample
        voxel_size: voxel size for sparse convolution

    Returns:
        ret_dict: dictionary with preprocessed data
        offset: (3,) offset used to shift coordinates (for inverse transform)
        cloud_full_for_collision: (N, 3) full cloud before sampling (for collision detection)
    """
    # Save the full cloud BEFORE sampling for collision detection
    cloud_full_for_collision = points.copy()

    # Sample points (same as training: random uniform sampling)
    print(f"    Sampling {len(points)} -> {num_point} points")
    if len(points) >= num_point:
        idxs = np.random.choice(len(points), num_point, replace=False)
    else:
        idxs1 = np.arange(len(points))
        idxs2 = np.random.choice(len(points), num_point - len(points), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = points[idxs]

    # Shift so all coords are >= 0 (CRITICAL: must match training preprocessing!)
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
    # Preprocess: sample -> offset shift
    data_input, offset, cloud_full_for_collision = preprocess_point_cloud(
        points, cfgs.num_point, cfgs.voxel_size,
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
    use_stable_score = getattr(cfgs, 'enable_stable_score', False)
    with torch.inference_mode():
        end_points = net(batch_data)
        grasp_preds = pred_decode(end_points, use_stable_score=use_stable_score)

    preds = grasp_preds[0].detach().cpu().numpy()

    # Extract graspness scores for visualization
    graspness_scores = None
    if 'graspness_score' in end_points:
        graspness_scores = end_points['graspness_score'].squeeze(1)[0].detach().cpu().numpy()
        print(f"    Graspness scores: min={graspness_scores.min():.4f}, max={graspness_scores.max():.4f}, mean={graspness_scores.mean():.4f}")

    if len(preds) == 0:
        print('No grasps detected')
        return GraspGroup(np.zeros((0, 17))), offset, cloud_sampled_shifted, graspness_scores

    # Optional angle-to-vertical filter
    if cfgs.max_angle_to_vertical_deg is not None:
        n_before = len(preds)
        preds = filter_grasps_by_vertical_angle(
            preds,
            max_angle_deg=cfgs.max_angle_to_vertical_deg,
            vertical_axis=cfgs.vertical_axis,
        )
        print(f"    Angle filter ({cfgs.max_angle_to_vertical_deg}° to {cfgs.vertical_axis}-axis): "
              f"{n_before} -> {len(preds)} grasps")
        if len(preds) == 0:
            print('No grasps after angle filter')
            return GraspGroup(np.zeros((0, 17))), offset, cloud_sampled_shifted, graspness_scores

    # Filter by gripper width
    if cfgs.max_gripper_width < 0.1:
        mask = preds[:, 1] <= cfgs.max_gripper_width
        preds = preds[mask]
        if len(preds) == 0:
            print('No grasps after width filter')
            return GraspGroup(np.zeros((0, 17))), offset, cloud_sampled_shifted, graspness_scores

    gg = GraspGroup(preds)

    if cfgs.collision_thresh > 0:
        n_before_cd = len(gg)
        # Transform grasp centers to camera frame for collision detection
        preds_cam = preds.copy()
        preds_cam[:, 13:16] = preds_cam[:, 13:16] - offset
        gg_cam = GraspGroup(preds_cam)
        mfcdetector = ModelFreeCollisionDetector(cloud_full_for_collision, voxel_size=cfgs.voxel_size_cd)
        collision_mask = mfcdetector.detect(gg_cam, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
        gg = gg[~collision_mask]
        print(f"    Collision detection: {n_before_cd} -> {len(gg)} grasps "
              f"(removed {(n_before_cd - len(gg))} colliding, "
              f"full cloud: {len(cloud_full_for_collision)} pts)")
        if len(gg) == 0:
            print('No grasps after collision detection')
            return GraspGroup(np.zeros((0, 17))), offset, cloud_sampled_shifted, graspness_scores

    return gg, offset, cloud_sampled_shifted, graspness_scores


def main():
    parser = argparse.ArgumentParser(description="AnyGrasp ZMQ Server")
    parser.add_argument("--port", type=int, default=5588, help="ZMQ server port")
    parser.add_argument("--checkpoint_path", type=str, default="gsnet_sonata_epoch15_llrd.tar",
                        help="Path to model checkpoint")
    parser.add_argument("--seed_feat_dim", type=int, default=512, help="Point-wise feature dimension")
    parser.add_argument("--num_point", type=int, default=15000, help="Number of points to sample")
    parser.add_argument("--voxel_size", type=float, default=0.005, help="Voxel size for sparse convolution")
    parser.add_argument("--collision_thresh", type=float, default=0.1,
                        help="Collision threshold (set <= 0 to disable)")
    parser.add_argument("--voxel_size_cd", type=float, default=0.01,
                        help="Voxel size for collision detection")
    parser.add_argument("--max_gripper_width", type=float, default=0.08,
                        help="Maximum gripper width (<=0.1m)")
    parser.add_argument("--max_angle_to_vertical_deg", type=float, default=None,
                        help="If set, keep only grasps whose approach direction is within this "
                             "angle (degrees) of the vertical axis, e.g. 25.")
    parser.add_argument("--vertical_axis", type=str, default="z", choices=["x", "y", "z"],
                        help="Camera-frame axis treated as vertical for angle filtering.")
    parser.add_argument("--backbone", type=str, default="resunet",
                        choices=["transformer", "transformer_pretrained", "sonata",
                                 "pointnet2", "resunet", "resunet18", "resunet_rgb", "resunet18_rgb"],
                        help="Backbone architecture [default: resunet]. "
                             "Must match the backbone used during training.")
    parser.add_argument("--enable_stable_score", action="store_true", default=False,
                        help="Enable stable score prediction (model architecture) [default: False]")
    parser.add_argument("--graspness_threshold", type=float, default=0.1,
                        help="Threshold for graspness score filtering during forward pass [default: 0.1]")
    parser.add_argument("--nsample", type=int, default=16,
                        help="Number of samples for cloud crop in GraspNet [default: 16]")
    parser.add_argument("--max_grasps", type=int, default=1024,
                        help="Maximum number of grasps to return")
    parser.add_argument("--grasp_approach_offset", type=float, default=0.01,
                        help="Offset (meters) to apply along each grasp's approach direction. "
                             "Positive pushes the grasp deeper into the object. Default: 0.01 (1cm). "
                             "Set to 0 to disable.")
    parser.add_argument("--boundary_margin", type=float, default=0.05,
                        help="Reject grasps within this distance (meters) of the point cloud's "
                             "X/Y boundary. Prevents edge-of-scene grasps where context is "
                             "incomplete. Set <= 0 to disable. Default: 0.01 (1 cm).")

    args = parser.parse_args()

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

    # Initialize GraspNet model with the specified backbone
    print(f"Loading model from {args.checkpoint_path}...")
    print(f"    Backbone: {args.backbone}")
    print(f"    Stable score: {args.enable_stable_score}")
    net = GraspNet(
        seed_feat_dim=args.seed_feat_dim,
        is_training=False,
        backbone=args.backbone,
        enable_stable_score=args.enable_stable_score,
        graspness_threshold=args.graspness_threshold,
        nsample=args.nsample,
    )
    net.to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Handle DDP checkpoint format: remove 'module.' prefix if present
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        print("    Stripped 'module.' prefix from DDP checkpoint")

    # Handle old checkpoint format: bundled stable scores in conv_swad (108 outputs)
    # vs new format: separate conv_stable layer (conv_swad=96, conv_stable=12)
    if 'swad.conv_swad.weight' in state_dict:
        old_weight = state_dict['swad.conv_swad.weight']
        old_bias = state_dict['swad.conv_swad.bias']
        if old_weight.shape[0] == 108 and args.enable_stable_score:
            print("    Converting old checkpoint (bundled 108 outputs) to new format (96 + 12 separate)...")
            state_dict['swad.conv_swad.weight'] = old_weight[:96]
            state_dict['swad.conv_swad.bias'] = old_bias[:96]
            state_dict['swad.conv_stable.weight'] = old_weight[96:]
            state_dict['swad.conv_stable.bias'] = old_bias[96:]

    net.load_state_dict(state_dict, strict=False)
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
    print(f"    - backbone: {args.backbone}")
    print(f"    - enable_stable_score: {args.enable_stable_score}")
    print(f"    - num_point: {args.num_point}")
    print(f"    - voxel_size: {args.voxel_size}")
    print(f"    - collision_thresh: {args.collision_thresh}")
    print(f"    - max_gripper_width: {args.max_gripper_width}")
    print(f"    - max_angle_to_vertical_deg: {args.max_angle_to_vertical_deg}")
    print(f"    - vertical_axis: {args.vertical_axis}")
    print(f"    - max_grasps: {args.max_grasps}")
    print(f"    - grasp_approach_offset: {args.grasp_approach_offset}m ({args.grasp_approach_offset*100:.1f}cm)")
    print(f"    - boundary_margin: {args.boundary_margin}m")
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
            print(f"    - Number of points: {xyzrgb.shape[0]}")
            print(f"    - Dimensions: {xyzrgb.shape[1]} (x, y, z, r, g, b)")

            # Extract XYZ coordinates only (resunet backbone doesn't use RGB)
            points = xyzrgb[:, :3].astype(np.float32)

            print(f"\nPoint cloud bounds:")
            print(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
            print(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
            print(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

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

                # Convert to JSON (includes offset subtraction for camera coordinates
                # and optional approach-direction offset)
                response = grasps_to_json(gg, offset, approach_offset=args.grasp_approach_offset)

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

