"""
Compute the "ranking error" metric from the GSNet paper.

Given ground-truth and predicted graspness scores in [0,1]:
1. Choose K bins (K = 20)
2. Convert each score to a discrete rank: r = floor(score * K), clipped to [0, K-1]
3. Compute ranking error: e_rank = mean(|r_pred - r_true|) / K

Computes:
- Point-wise graspness ranking error (e_p_rank)
- View-wise graspness ranking error (e_v_rank)
"""

import os
import sys
import numpy as np
import argparse
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from models.graspnet import GraspNet
from dataset.graspnet_dataset import GraspNetDataset, spconv_collate_fn, load_grasp_labels, load_grasp_labels_lazy
from utils.label_generation import process_grasp_labels

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default=None, required=True)
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default=None, required=True)
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--backbone', type=str, default='transformer', 
                    choices=['transformer', 'transformer_pretrained', 'pointnet2', 'resunet', 'resunet_rgb'],
                    help='Backbone architecture [default: transformer]')
parser.add_argument('--ptv3_pretrained_path', type=str, default=None,
                    help='Path to PTv3 pretrained weights (.pth file)')
parser.add_argument('--enable_flash', action='store_true', default=False,
                    help='Enable flash attention in PTv3 backbone')
parser.add_argument('--enable_stable_score', action='store_true', default=False,
                    help='Enable stable score prediction')
parser.add_argument('--split', type=str, default='test_seen',
                    choices=['train', 'val', 'test', 'test_seen', 'test_seen_single', 'test_seen_mini', 
                             'test_similar', 'test_similar_mini', 'test_novel', 'test_novel_single', 'test_novel_mini'],
                    help='Dataset split to evaluate on [default: test_seen]')
parser.add_argument('--num_bins', type=int, default=20, help='Number of bins K for ranking [default: 20]')
parser.add_argument('--lazy_grasp_labels', action='store_true', default=False,
                    help='Use lazy loading for grasp labels to reduce memory')
parser.add_argument('--max_samples', type=int, default=None,
                    help='Maximum number of samples to evaluate (for quick testing)')
parser.add_argument('--view_start', type=int, default=0,
                    help='Starting view index (inclusive) for each scene [default: 0]')
parser.add_argument('--view_end', type=int, default=256,
                    help='Ending view index (exclusive) for each scene [default: 256]')
cfgs = parser.parse_args()


def compute_ranking_error(pred_scores, gt_scores, K=20, mask=None):
    """
    Compute the ranking error metric.
    
    Args:
        pred_scores: Predicted scores in [0, 1], shape (*)
        gt_scores: Ground truth scores in [0, 1], shape (*)
        K: Number of bins
        mask: Optional boolean mask, only compute error where mask is True
        
    Returns:
        ranking_error: Mean absolute rank difference normalized by K
        num_valid: Number of valid comparisons
    """
    # Clamp scores to [0, 1) to ensure rank is in [0, K-1]
    pred_clamped = torch.clamp(pred_scores, 0.0, 0.9999)
    gt_clamped = torch.clamp(gt_scores, 0.0, 0.9999)
    
    # Convert to discrete rank: r = floor(score * K), clipped to [0, K-1]
    pred_rank = torch.floor(pred_clamped * K).long()
    gt_rank = torch.floor(gt_clamped * K).long()
    
    # Clip to ensure within bounds
    pred_rank = torch.clamp(pred_rank, 0, K - 1)
    gt_rank = torch.clamp(gt_rank, 0, K - 1)
    
    # Compute absolute rank difference
    rank_diff = torch.abs(pred_rank - gt_rank).float()
    
    # Apply mask if provided
    if mask is not None:
        rank_diff = rank_diff[mask]
    
    if rank_diff.numel() == 0:
        return 0.0, 0
    
    # Normalize by K
    ranking_error = (rank_diff / K).mean().item()
    
    return ranking_error, rank_diff.numel()


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def evaluate_ranking_error():
    """Evaluate ranking error for point-wise and view-wise graspness."""
    
    # Auto-enable RGB for backbones that require 6-channel input
    use_rgb = (cfgs.backbone in ['transformer_pretrained', 'resunet_rgb'])
    if use_rgb:
        print("Using RGB features for 6-channel input (XYZ + RGB)")
    
    # Load grasp labels (needed for training labels)
    print("Loading grasp labels...")
    if cfgs.lazy_grasp_labels:
        grasp_labels = load_grasp_labels_lazy(cfgs.dataset_root)
    else:
        grasp_labels = load_grasp_labels(cfgs.dataset_root)
    
    # Create dataset with labels
    test_dataset = GraspNetDataset(
        cfgs.dataset_root, 
        grasp_labels=grasp_labels,
        split=cfgs.split, 
        camera=cfgs.camera, 
        num_points=cfgs.num_point,
        voxel_size=cfgs.voxel_size, 
        remove_outlier=True, 
        augment=False, 
        load_label=True,  # Load labels for computing ranking error
        use_rgb=use_rgb,
        enable_stable_score=cfgs.enable_stable_score,
        view_start=cfgs.view_start,
        view_end=cfgs.view_end
    )
    print(f'Test dataset length: {len(test_dataset)} (split: {cfgs.split}, views: {cfgs.view_start}-{cfgs.view_end-1})')
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=cfgs.batch_size, 
        shuffle=False,
        num_workers=0, 
        worker_init_fn=my_worker_init_fn, 
        collate_fn=spconv_collate_fn
    )
    print(f'Test dataloader length: {len(test_dataloader)}')
    
    # Initialize model
    net = GraspNet(
        seed_feat_dim=cfgs.seed_feat_dim, 
        is_training=True,  # Need training mode to get intermediate outputs
        backbone=cfgs.backbone,
        ptv3_pretrained_path=cfgs.ptv3_pretrained_path,
        enable_flash=cfgs.enable_flash,
        enable_stable_score=cfgs.enable_stable_score,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Handle old checkpoint format: bundled stable scores in conv_swad (108 outputs)
    if 'swad.conv_swad.weight' in state_dict:
        old_weight = state_dict['swad.conv_swad.weight']
        old_bias = state_dict['swad.conv_swad.bias']
        if old_weight.shape[0] == 108 and cfgs.enable_stable_score:
            print("Converting old checkpoint (bundled 108 outputs) to new format (96 + 12 separate)...")
            state_dict['swad.conv_swad.weight'] = old_weight[:96]
            state_dict['swad.conv_swad.bias'] = old_bias[:96]
            state_dict['swad.conv_stable.weight'] = old_weight[96:]
            state_dict['swad.conv_stable.bias'] = old_bias[96:]
    
    net.load_state_dict(state_dict, strict=False)
    start_epoch = checkpoint['epoch']
    print(f"-> loaded checkpoint {cfgs.checkpoint_path} (epoch: {start_epoch})")
    
    net.eval()
    
    # Accumulators for ranking errors
    total_point_rank_error = 0.0
    total_point_count = 0
    total_view_rank_error = 0.0
    total_view_count = 0
    
    K = cfgs.num_bins
    print(f"\nComputing ranking error with K={K} bins...")
    
    num_samples = len(test_dataloader)
    if cfgs.max_samples is not None:
        num_samples = min(num_samples, cfgs.max_samples)
    
    tic = time.time()
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_dataloader, total=num_samples, desc="Evaluating")):
            if cfgs.max_samples is not None and batch_idx >= cfgs.max_samples:
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
            end_points = net(batch_data)
            
            # Process grasp labels to get view graspness
            end_points = process_grasp_labels(end_points)
            
            # ============================================
            # 1. Point-wise graspness ranking error
            # ============================================
            graspness_pred = end_points['graspness_score'].squeeze(1)  # (B, N)
            graspness_label = end_points['graspness_label'].squeeze(-1)  # (B, N)
            objectness_mask = end_points['objectness_label'].bool()  # (B, N)
            
            # Compute error only on object points
            for b in range(graspness_pred.shape[0]):
                mask_b = objectness_mask[b]
                if mask_b.sum() > 0:
                    err, count = compute_ranking_error(
                        graspness_pred[b], 
                        graspness_label[b], 
                        K=K, 
                        mask=mask_b
                    )
                    total_point_rank_error += err * count
                    total_point_count += count
            
            # ============================================
            # 2. View-wise graspness ranking error
            # ============================================
            if 'view_score' in end_points and 'batch_grasp_view_graspness' in end_points:
                view_pred = end_points['view_score']  # (B, M, V)
                view_label = end_points['batch_grasp_view_graspness']  # (B, M, V)
                
                # View predictions need to be normalized to [0, 1] for fair comparison
                # The model outputs unnormalized scores, so we normalize per-sample like the label
                for b in range(view_pred.shape[0]):
                    vp = view_pred[b]  # (M, V)
                    vl = view_label[b]  # (M, V)
                    
                    # Normalize view predictions to [0, 1] (same as label normalization)
                    vp_min, _ = vp.min(dim=-1, keepdim=True)
                    vp_max, _ = vp.max(dim=-1, keepdim=True)
                    vp_normalized = (vp - vp_min) / (vp_max - vp_min + 1e-8)
                    
                    # Compute error (all seed points are valid for view graspness)
                    err, count = compute_ranking_error(vp_normalized, vl, K=K)
                    total_view_rank_error += err * count
                    total_view_count += count
    
    toc = time.time()
    
    # Compute final metrics
    e_p_rank = total_point_rank_error / total_point_count if total_point_count > 0 else 0.0
    e_v_rank = total_view_rank_error / total_view_count if total_view_count > 0 else 0.0
    
    print("\n" + "=" * 60)
    print("RANKING ERROR RESULTS")
    print("=" * 60)
    print(f"Dataset split: {cfgs.split}")
    print(f"Camera: {cfgs.camera}")
    print(f"View range: [{cfgs.view_start}, {cfgs.view_end})")
    print(f"Number of bins (K): {K}")
    print(f"Number of samples evaluated: {min(num_samples, len(test_dataloader))}")
    print(f"Time elapsed: {toc - tic:.2f}s")
    print("-" * 60)
    print(f"Point-wise graspness ranking error (e_p_rank): {e_p_rank:.6f}")
    print(f"  - Total comparisons: {total_point_count:,}")
    print(f"View-wise graspness ranking error (e_v_rank): {e_v_rank:.6f}")
    print(f"  - Total comparisons: {total_view_count:,}")
    print("=" * 60)
    
    # Also print in a compact format
    print(f"\nSummary: e_p_rank={e_p_rank:.4f}, e_v_rank={e_v_rank:.4f}")
    
    return {
        'e_p_rank': e_p_rank,
        'e_v_rank': e_v_rank,
        'total_point_count': total_point_count,
        'total_view_count': total_view_count,
    }


if __name__ == '__main__':
    results = evaluate_ranking_error()
