#!/usr/bin/env python3
"""
Heavyly vibe coded
Analyze whether stable score training biases grasps towards the center of gravity (COG).

This script compares two models:
1. Trained with stable score enabled
2. Trained without stable score (vanilla)

For each model, we:
1. Run inference on test scenes
2. For each predicted grasp, compute its distance to the object's COG
3. Compare the distance distributions between the two models

The hypothesis is that stable score training should bias grasps towards lower COG distances,
as grasps closer to the COG are more stable (less tipping moment).

Usage:
    python analyze_cog_bias.py --dataset_root /path/to/graspnet

Author: Analysis script for stable score COG bias investigation
"""

import os
import sys
import numpy as np
import argparse
import time
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import json

# Get project root (two levels up from this script)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import GraspNetDataset, spconv_collate_fn, load_grasp_labels
from utils.stable_score_utils import compute_mesh_cog, HAS_TRIMESH
import scipy.io as scio
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default=None, required=True, help='Path to GraspNet dataset')
parser.add_argument('--stable_checkpoint', 
                    default='logs/gsnet_resunet_strong_stable_score_augmented_n20k/gsnet_resunet_epoch09.tar',
                    help='Path to stable score model checkpoint')
parser.add_argument('--vanilla_checkpoint', 
                    default='logs/gsnet_resunet_vanilla_bigger_fields/gsnet_resunet_epoch09.tar',
                    help='Path to vanilla model checkpoint')
parser.add_argument('--output_dir', default='experiments/cog_analysis', help='Output directory for results')
parser.add_argument('--camera', default='realsense', help='Camera type [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel size')
parser.add_argument('--num_scenes', type=int, default=50, help='Number of scenes to analyze (for quick test)')
parser.add_argument('--top_k_grasps', type=int, default=100, help='Number of top grasps to analyze per scene')
parser.add_argument('--backbone', type=str, default='resunet', help='Backbone architecture')
parser.add_argument('--seed_feat_dim', type=int, default=512, help='Seed feature dimension')
cfgs = parser.parse_args()


def load_object_cogs(dataset_root, num_objects=88):
    """
    Load or compute center of gravity for all objects.
    
    Returns:
        dict: {obj_idx: cog_array} where obj_idx is 1-indexed (1-88)
    """
    cogs = {}
    stable_labels_dir = os.path.join(dataset_root, 'stable_labels')
    
    for obj_idx in range(1, num_objects + 1):
        obj_id_str = str(obj_idx - 1).zfill(3)
        
        # Try to load from stable labels first (already computed)
        stable_file = os.path.join(stable_labels_dir, f'{obj_id_str}_stable.npz')
        if os.path.exists(stable_file):
            data = np.load(stable_file)
            if 'cog' in data:
                cogs[obj_idx] = data['cog']
                continue
        
        # Otherwise compute from mesh
        mesh_path = os.path.join(dataset_root, 'models', obj_id_str, 'nontextured.ply')
        if os.path.exists(mesh_path) and HAS_TRIMESH:
            try:
                cogs[obj_idx] = compute_mesh_cog(mesh_path)
            except Exception as e:
                print(f"Warning: Could not compute COG for object {obj_idx}: {e}")
    
    return cogs


def load_model(checkpoint_path, enable_stable_score=False):
    """Load a model from checkpoint."""
    net = GraspNet(
        seed_feat_dim=cfgs.seed_feat_dim,
        is_training=False,
        backbone=cfgs.backbone,
        enable_stable_score=enable_stable_score,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint {checkpoint_path} (epoch: {epoch})")
    
    net.eval()
    return net, device


def get_object_at_grasp_point(grasp_center, seg_labeled, cloud_points, obj_idxs, threshold=0.02):
    """
    Determine which object a grasp is on based on proximity to segmented points.
    
    Args:
        grasp_center: (3,) grasp center position
        seg_labeled: (N,) segmentation labels for each point
        cloud_points: (N, 3) point cloud
        obj_idxs: list of object indices in the scene
        threshold: distance threshold for matching
    
    Returns:
        obj_idx: matched object index (1-indexed) or None
    """
    # Find closest point in cloud
    distances = np.linalg.norm(cloud_points - grasp_center, axis=1)
    closest_idx = np.argmin(distances)
    
    if distances[closest_idx] > threshold:
        return None
    
    seg_label = seg_labeled[closest_idx]
    if seg_label in obj_idxs:
        return seg_label
    return None


def compute_grasp_to_cog_distance(grasp_centers, grasp_scores, object_poses, obj_idxs, cogs, 
                                   seg_labeled, cloud_points, top_k=100):
    """
    Compute distance from grasp centers to object COGs.
    
    Args:
        grasp_centers: (M, 3) grasp center positions in camera frame
        grasp_scores: (M,) grasp scores
        object_poses: list of (4, 4) object pose matrices
        obj_idxs: list of object indices in scene
        cogs: dict of {obj_idx: cog_in_object_frame}
        seg_labeled: (N,) segmentation labels
        cloud_points: (N, 3) point cloud
        top_k: number of top grasps to analyze
    
    Returns:
        distances: list of (distance, score, obj_idx) for valid grasps
    """
    # Sort grasps by score and take top_k
    sorted_indices = np.argsort(grasp_scores)[::-1][:top_k]
    
    distances = []
    for idx in sorted_indices:
        grasp_center = grasp_centers[idx]
        grasp_score = grasp_scores[idx]
        
        # Find which object this grasp is on
        obj_idx = get_object_at_grasp_point(grasp_center, seg_labeled, cloud_points, obj_idxs)
        if obj_idx is None:
            continue
        
        if obj_idx not in cogs:
            continue
        
        # Get object pose (4x4 transformation from object frame to camera frame)
        try:
            obj_list_idx = list(obj_idxs).index(obj_idx)
            pose = object_poses[obj_list_idx]  # (3, 4) or (4, 4)
            
            # Ensure it's 4x4
            if pose.shape == (3, 4):
                pose_4x4 = np.eye(4)
                pose_4x4[:3, :] = pose
            else:
                pose_4x4 = pose
            
            # Transform COG from object frame to camera frame
            cog_obj = cogs[obj_idx]
            cog_cam = pose_4x4[:3, :3] @ cog_obj + pose_4x4[:3, 3]
            
            # Compute distance from grasp center to COG
            dist = np.linalg.norm(grasp_center - cog_cam)
            distances.append((dist, grasp_score, obj_idx))
            
        except (IndexError, ValueError) as e:
            continue
    
    return distances


def run_inference_and_analyze(net, device, dataset, cogs, model_name, use_stable_score=False):
    """
    Run inference on dataset and compute COG distances for all grasps.
    
    Returns:
        all_distances: list of (distance, score, obj_idx) tuples
        scene_stats: dict with per-scene statistics
    """
    dataloader = DataLoader(
        dataset, 
        batch_size=cfgs.batch_size, 
        shuffle=False,
        num_workers=0, 
        collate_fn=spconv_collate_fn
    )
    
    all_distances = []
    scene_stats = {}
    
    print(f"\nRunning inference for {model_name}...")
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Analyzing {model_name}")):
        if batch_idx >= cfgs.num_scenes:
            break
            
        # Transfer to GPU
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device, non_blocking=True)
            else:
                batch_data[key] = batch_data[key].to(device, non_blocking=True)
        
        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points, use_stable_score=use_stable_score)
        
        # Process each item in batch
        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()
            
            # Get grasp information
            # Format: [score, width, height, depth, rotation(9), translation(3), obj_id]
            grasp_scores = preds[:, 0]
            grasp_centers = preds[:, 13:16]  # Translation/center
            
            # Transform back to camera coordinates if offset was applied
            if 'cloud_offset' in batch_data:
                offset = batch_data['cloud_offset'][i].cpu().numpy()
                grasp_centers = grasp_centers - offset
            
            # Get scene data for object matching
            cloud_points = batch_data['point_clouds'][i].cpu().numpy()
            if 'cloud_offset' in batch_data:
                cloud_points = cloud_points - offset
            
            # Load metadata from files (since we use load_label=False)
            meta = scio.loadmat(dataset.metapath[data_idx])
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']  # (3, 4, num_objects)
            
            # Build object_poses list
            object_poses = []
            for obj_i in range(poses.shape[2]):
                object_poses.append(poses[:, :, obj_i])
            
            # Load segmentation and map to sampled points
            seg = np.array(Image.open(dataset.labelpath[data_idx]))
            depth = np.array(Image.open(dataset.depthpath[data_idx]))
            
            # Get mask same as in dataset
            mask = depth > 0
            seg_masked = seg[mask].flatten()
            
            # Compute distances for top grasps
            distances = compute_grasp_to_cog_distance_simple(
                grasp_centers, grasp_scores, object_poses, obj_idxs, cogs, 
                cloud_points, top_k=cfgs.top_k_grasps
            )
            
            all_distances.extend(distances)
            scene_stats[data_idx] = {
                'num_grasps': len(distances),
                'mean_distance': np.mean([d[0] for d in distances]) if distances else 0,
                'mean_score': np.mean([d[1] for d in distances]) if distances else 0,
            }
    
    return all_distances, scene_stats


def compute_grasp_to_cog_distance_simple(grasp_centers, grasp_scores, object_poses, obj_idxs, cogs, 
                                          cloud_points, top_k=100):
    """
    Simplified version: For each grasp, find the nearest object and compute distance to its COG.
    
    Args:
        grasp_centers: (M, 3) grasp center positions in camera frame
        grasp_scores: (M,) grasp scores  
        object_poses: list of (3, 4) object pose matrices
        obj_idxs: array of object indices in scene (1-indexed)
        cogs: dict of {obj_idx: cog_in_object_frame}
        cloud_points: (N, 3) point cloud
        top_k: number of top grasps to analyze
    
    Returns:
        distances: list of (distance, score, obj_idx) for valid grasps
    """
    # Sort grasps by score and take top_k
    sorted_indices = np.argsort(grasp_scores)[::-1][:top_k]
    
    # Compute object COGs in camera frame
    obj_cogs_cam = {}
    for obj_i, obj_idx in enumerate(obj_idxs):
        if obj_idx not in cogs:
            continue
        if obj_i >= len(object_poses):
            continue
            
        pose = object_poses[obj_i]  # (3, 4)
        cog_obj = cogs[obj_idx]
        # Transform: R @ cog + t
        cog_cam = pose[:3, :3] @ cog_obj + pose[:3, 3]
        obj_cogs_cam[obj_idx] = cog_cam
    
    if not obj_cogs_cam:
        return []
    
    # Convert to arrays for efficient distance computation
    obj_list = list(obj_cogs_cam.keys())
    cog_array = np.array([obj_cogs_cam[o] for o in obj_list])  # (num_objs, 3)
    
    distances = []
    for idx in sorted_indices:
        grasp_center = grasp_centers[idx]
        grasp_score = grasp_scores[idx]
        
        if grasp_score <= 0:
            continue
        
        # Find closest object COG
        dists_to_cogs = np.linalg.norm(cog_array - grasp_center, axis=1)
        closest_obj_idx = np.argmin(dists_to_cogs)
        closest_dist = dists_to_cogs[closest_obj_idx]
        closest_obj = obj_list[closest_obj_idx]
        
        # Only include if grasp is reasonably close to an object (within 15cm of COG)
        if closest_dist < 0.15:
            distances.append((closest_dist, grasp_score, closest_obj))
    
    return distances


def analyze_and_plot(stable_distances, vanilla_distances, output_dir):
    """
    Analyze and compare the COG distance distributions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract just the distances
    stable_dists = np.array([d[0] for d in stable_distances])
    vanilla_dists = np.array([d[0] for d in vanilla_distances])
    
    # Extract scores
    stable_scores = np.array([d[1] for d in stable_distances])
    vanilla_scores = np.array([d[1] for d in vanilla_distances])
    
    # Compute statistics
    results = {
        'stable_score_model': {
            'num_grasps': len(stable_dists),
            'mean_cog_distance': float(np.mean(stable_dists)) if len(stable_dists) > 0 else 0,
            'std_cog_distance': float(np.std(stable_dists)) if len(stable_dists) > 0 else 0,
            'median_cog_distance': float(np.median(stable_dists)) if len(stable_dists) > 0 else 0,
            'mean_grasp_score': float(np.mean(stable_scores)) if len(stable_scores) > 0 else 0,
        },
        'vanilla_model': {
            'num_grasps': len(vanilla_dists),
            'mean_cog_distance': float(np.mean(vanilla_dists)) if len(vanilla_dists) > 0 else 0,
            'std_cog_distance': float(np.std(vanilla_dists)) if len(vanilla_dists) > 0 else 0,
            'median_cog_distance': float(np.median(vanilla_dists)) if len(vanilla_dists) > 0 else 0,
            'mean_grasp_score': float(np.mean(vanilla_scores)) if len(vanilla_scores) > 0 else 0,
        }
    }
    
    # Statistical test
    if len(stable_dists) > 0 and len(vanilla_dists) > 0:
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(stable_dists, vanilla_dists, alternative='less')
        results['statistical_test'] = {
            'test': 'Mann-Whitney U (stable < vanilla)',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant_at_0.05': bool(p_value < 0.05),
            'significant_at_0.01': bool(p_value < 0.01),
        }
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(stable_dists) + np.var(vanilla_dists)) / 2)
        cohens_d = (np.mean(vanilla_dists) - np.mean(stable_dists)) / pooled_std if pooled_std > 0 else 0
        results['effect_size'] = {
            'cohens_d': float(cohens_d),
            'interpretation': 'small' if abs(cohens_d) < 0.5 else ('medium' if abs(cohens_d) < 0.8 else 'large')
        }
    
    # Save results
    with open(os.path.join(output_dir, 'cog_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("COG DISTANCE ANALYSIS RESULTS")
    print("="*80)
    print(f"\nStable Score Model:")
    print(f"  Number of analyzed grasps: {results['stable_score_model']['num_grasps']}")
    print(f"  Mean COG distance: {results['stable_score_model']['mean_cog_distance']:.4f} m")
    print(f"  Std COG distance: {results['stable_score_model']['std_cog_distance']:.4f} m")
    print(f"  Median COG distance: {results['stable_score_model']['median_cog_distance']:.4f} m")
    print(f"  Mean grasp score: {results['stable_score_model']['mean_grasp_score']:.4f}")
    
    print(f"\nVanilla Model:")
    print(f"  Number of analyzed grasps: {results['vanilla_model']['num_grasps']}")
    print(f"  Mean COG distance: {results['vanilla_model']['mean_cog_distance']:.4f} m")
    print(f"  Std COG distance: {results['vanilla_model']['std_cog_distance']:.4f} m")
    print(f"  Median COG distance: {results['vanilla_model']['median_cog_distance']:.4f} m")
    print(f"  Mean grasp score: {results['vanilla_model']['mean_grasp_score']:.4f}")
    
    if 'statistical_test' in results:
        print(f"\nStatistical Test (stable < vanilla):")
        print(f"  Mann-Whitney U statistic: {results['statistical_test']['statistic']:.2f}")
        print(f"  p-value: {results['statistical_test']['p_value']:.6f}")
        print(f"  Significant at α=0.05: {results['statistical_test']['significant_at_0.05']}")
        print(f"  Significant at α=0.01: {results['statistical_test']['significant_at_0.01']}")
        print(f"\nEffect Size:")
        print(f"  Cohen's d: {results['effect_size']['cohens_d']:.4f} ({results['effect_size']['interpretation']})")
    
    # Calculate percentage difference
    if results['vanilla_model']['mean_cog_distance'] > 0:
        pct_diff = ((results['vanilla_model']['mean_cog_distance'] - results['stable_score_model']['mean_cog_distance']) 
                   / results['vanilla_model']['mean_cog_distance'] * 100)
        print(f"\nStable score model has {abs(pct_diff):.1f}% {'lower' if pct_diff > 0 else 'higher'} mean COG distance")
    
    print("="*80)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Histogram comparison
    ax1 = axes[0, 0]
    if len(stable_dists) > 0 and len(vanilla_dists) > 0:
        bins = np.linspace(0, max(stable_dists.max(), vanilla_dists.max()), 50)
        ax1.hist(stable_dists, bins=bins, alpha=0.6, label=f'Stable Score (μ={np.mean(stable_dists):.3f}m)', density=True)
        ax1.hist(vanilla_dists, bins=bins, alpha=0.6, label=f'Vanilla (μ={np.mean(vanilla_dists):.3f}m)', density=True)
    ax1.set_xlabel('Distance to COG (m)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Grasp-to-COG Distances')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot comparison
    ax2 = axes[0, 1]
    box_data = []
    box_labels = []
    if len(stable_dists) > 0:
        box_data.append(stable_dists)
        box_labels.append('Stable Score')
    if len(vanilla_dists) > 0:
        box_data.append(vanilla_dists)
        box_labels.append('Vanilla')
    if box_data:
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors = ['#2ecc71', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax2.set_ylabel('Distance to COG (m)')
    ax2.set_title('Comparison of COG Distances')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: CDF comparison
    ax3 = axes[1, 0]
    if len(stable_dists) > 0:
        sorted_stable = np.sort(stable_dists)
        ax3.plot(sorted_stable, np.arange(1, len(sorted_stable)+1) / len(sorted_stable), 
                label='Stable Score', linewidth=2)
    if len(vanilla_dists) > 0:
        sorted_vanilla = np.sort(vanilla_dists)
        ax3.plot(sorted_vanilla, np.arange(1, len(sorted_vanilla)+1) / len(sorted_vanilla), 
                label='Vanilla', linewidth=2)
    ax3.set_xlabel('Distance to COG (m)')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution of COG Distances')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Score vs Distance scatter
    ax4 = axes[1, 1]
    if len(stable_dists) > 0:
        ax4.scatter(stable_dists, stable_scores, alpha=0.3, s=10, label='Stable Score', c='#2ecc71')
    if len(vanilla_dists) > 0:
        ax4.scatter(vanilla_dists, vanilla_scores, alpha=0.3, s=10, label='Vanilla', c='#e74c3c')
    ax4.set_xlabel('Distance to COG (m)')
    ax4.set_ylabel('Grasp Score')
    ax4.set_title('Grasp Score vs Distance to COG')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cog_analysis_plots.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'cog_analysis_plots.pdf'), bbox_inches='tight')
    print(f"\nPlots saved to {output_dir}/cog_analysis_plots.png")
    
    # Save raw data for further analysis
    np.savez(os.path.join(output_dir, 'cog_analysis_data.npz'),
             stable_distances=stable_dists,
             stable_scores=stable_scores,
             vanilla_distances=vanilla_dists,
             vanilla_scores=vanilla_scores)
    
    return results


def main():
    print("="*80)
    print("COG BIAS ANALYSIS: Comparing Stable Score vs Vanilla Models")
    print("="*80)
    
    # Check if checkpoints exist
    if not os.path.exists(cfgs.stable_checkpoint):
        print(f"ERROR: Stable checkpoint not found: {cfgs.stable_checkpoint}")
        return
    if not os.path.exists(cfgs.vanilla_checkpoint):
        print(f"ERROR: Vanilla checkpoint not found: {cfgs.vanilla_checkpoint}")
        return
    
    print(f"\nStable Score Checkpoint: {cfgs.stable_checkpoint}")
    print(f"Vanilla Checkpoint: {cfgs.vanilla_checkpoint}")
    print(f"Dataset Root: {cfgs.dataset_root}")
    print(f"Camera: {cfgs.camera}")
    print(f"Number of scenes to analyze: {cfgs.num_scenes}")
    print(f"Top-K grasps per scene: {cfgs.top_k_grasps}")
    
    # Load object COGs
    print("\nLoading object centers of gravity...")
    cogs = load_object_cogs(cfgs.dataset_root)
    print(f"Loaded COGs for {len(cogs)} objects")
    
    # Create dataset
    print("\nLoading dataset...")
    test_dataset = GraspNetDataset(
        cfgs.dataset_root, 
        split='test_seen', 
        camera=cfgs.camera, 
        num_points=cfgs.num_point,
        voxel_size=cfgs.voxel_size, 
        remove_outlier=True, 
        augment=False, 
        load_label=False,  # Inference mode - we load metadata separately
        use_rgb=False,
        enable_stable_score=False
    )
    print(f"Dataset length: {len(test_dataset)}")
    
    # Load and run stable score model
    print("\n" + "-"*40)
    print("Loading Stable Score Model...")
    stable_net, device = load_model(cfgs.stable_checkpoint, enable_stable_score=True)
    stable_distances, stable_stats = run_inference_and_analyze(
        stable_net, device, test_dataset, cogs, "Stable Score", use_stable_score=True
    )
    del stable_net
    torch.cuda.empty_cache()
    
    # Load and run vanilla model
    print("\n" + "-"*40)
    print("Loading Vanilla Model...")
    vanilla_net, device = load_model(cfgs.vanilla_checkpoint, enable_stable_score=False)
    vanilla_distances, vanilla_stats = run_inference_and_analyze(
        vanilla_net, device, test_dataset, cogs, "Vanilla", use_stable_score=False
    )
    del vanilla_net
    torch.cuda.empty_cache()
    
    # Analyze and plot results
    print("\n" + "-"*40)
    print("Analyzing results...")
    results = analyze_and_plot(stable_distances, vanilla_distances, cfgs.output_dir)
    
    print(f"\nResults saved to: {cfgs.output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
