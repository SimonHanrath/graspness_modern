#!/usr/bin/env python3
"""
Evaluate stable score prediction accuracy across different test sets.

Computes various metrics comparing predicted stable scores vs ground truth:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)  
- RMSE (Root Mean Squared Error)
- Correlation (Pearson and Spearman)
- Binary accuracy at threshold 0.5

Usage:
    python model_analysis/stable_score/evaluate_stable_score_accuracy.py --model_name <model>
    
    # Example with specific model:
    python model_analysis/stable_score/evaluate_stable_score_accuracy.py \
        --model_name lambda_0 \
        --checkpoint_path logs/gsnet_resunet_vanilla_cosine_anealing_stable_score0new_n15k/gsnet_resunet_epoch10.tar
    
    # List available models:
    python model_analysis/stable_score/evaluate_stable_score_accuracy.py --list_models
"""

import os
import sys
import json
import argparse
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, ROOT_DIR)

from models.graspnet import GraspNet
from dataset.graspnet_dataset import GraspNetDataset, spconv_collate_fn, load_grasp_labels


# ============ Available stable score models ============
MODELS = {
    "lambda_0": {
        "lambda": 0,
        "description": "No stable score loss (lambda=0)",
        "checkpoint": "logs/gsnet_resunet_vanilla_cosine_anealing_stable_score0new_n15k/gsnet_resunet_epoch10.tar"
    },
    "lambda_0_new": {
        "lambda": 0,
        "description": "Lambda 0 with t=0.01 threshold",
        "checkpoint": "logs/gsnet_resunet_vanilla_stable_score_lambda0_t001_n15k1/gsnet_resunet_epoch10.tar"
    },
    "lambda_0.1": {
        "lambda": 0.1,
        "description": "Stable score with lambda=0.1",
        "checkpoint": "logs/gsnet_resunet_vanilla_stable_score01_n15k/gsnet_resunet_epoch10.tar"
    },
    "lambda_1": {
        "lambda": 1,
        "description": "Stable score with lambda=1",
        "checkpoint": "logs/gsnet_resunet_vanilla_cosine_anealing_stable_score1new_n15k/gsnet_resunet_epoch10.tar"
    },
    "lambda_10": {
        "lambda": 10,
        "description": "Stable score with lambda=10",
        "checkpoint": "logs/gsnet_resunet_vanilla_cosine_anealing_stable_score10new_n15k/gsnet_resunet_epoch10.tar"
    },
    "fine_tuned_1": {
        "lambda": 1,
        "description": "Fine-tuned model (lambda=1, 5 epochs)",
        "checkpoint": "logs/gsnet_resunet_vanilla_stable_score_fine_tuned1/gsnet_resunet_epoch05.tar"
    },
}

# Test sets with their scene IDs
TEST_SETS = {
    "test_seen_mini": {"scenes": [100, 101, 102], "description": "Seen objects (3 scenes)"},
    "test_similar_mini": {"scenes": [130, 131, 132], "description": "Similar objects (3 scenes)"},
    "test_novel_mini": {"scenes": [160, 161, 162], "description": "Novel objects (3 scenes)"},
}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate stable score prediction accuracy')
    parser.add_argument('--dataset_root', default='/datasets/graspnet',
                        help='Path to GraspNet dataset root')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Name of model to evaluate (see --list_models)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Override checkpoint path (optional)')
    parser.add_argument('--list_models', action='store_true',
                        help='List available models and exit')
    parser.add_argument('--test_sets', nargs='+', default=None,
                        choices=['test_seen_mini', 'test_similar_mini', 'test_novel_mini', 'all'],
                        help='Test sets to evaluate (default: all)')
    parser.add_argument('--camera', default='realsense', choices=['realsense', 'kinect'],
                        help='Camera type')
    parser.add_argument('--backbone', type=str, default='resunet',
                        choices=['transformer', 'pointnet2', 'resunet'],
                        help='Backbone architecture')
    parser.add_argument('--num_point', type=int, default=15000,
                        help='Number of points to sample')
    parser.add_argument('--voxel_size', type=float, default=0.005,
                        help='Voxel size for sparse convolution')
    parser.add_argument('--seed_feat_dim', default=512, type=int,
                        help='Point-wise feature dimension')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--output_dir', default=None,
                        help='Directory to save results (default: script directory)')
    parser.add_argument('--save_detailed', action='store_true',
                        help='Save per-sample predictions for detailed analysis')
    return parser.parse_args()


def list_available_models():
    """Print available models and their info."""
    print("\n" + "=" * 60)
    print("Available Stable Score Models")
    print("=" * 60)
    for name, info in MODELS.items():
        checkpoint_exists = os.path.exists(os.path.join(ROOT_DIR, info['checkpoint']))
        status = "✓" if checkpoint_exists else "✗ (not found)"
        print(f"\n  {name}:")
        print(f"    Lambda: {info['lambda']}")
        print(f"    Description: {info['description']}")
        print(f"    Checkpoint: {info['checkpoint']}")
        print(f"    Status: {status}")
    print("\n" + "=" * 60)


def compute_metrics(predictions, labels, mask=None):
    """
    Compute various metrics comparing predictions to labels.
    
    Args:
        predictions: np.array of predicted stable scores
        labels: np.array of ground truth stable scores
        mask: optional boolean mask for valid entries
        
    Returns:
        dict with computed metrics
    """
    if mask is not None:
        predictions = predictions[mask]
        labels = labels[mask]
    
    if len(predictions) == 0:
        return {
            'mae': float('nan'),
            'mse': float('nan'),
            'rmse': float('nan'),
            'pearson_r': float('nan'),
            'spearman_r': float('nan'),
            'binary_acc': float('nan'),
            'n_samples': 0
        }
    
    # Flatten if needed
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Basic error metrics
    mae = np.mean(np.abs(predictions - labels))
    mse = np.mean((predictions - labels) ** 2)
    rmse = np.sqrt(mse)
    
    # Correlation metrics
    if len(predictions) > 2 and np.std(predictions) > 1e-8 and np.std(labels) > 1e-8:
        pearson_r, _ = pearsonr(predictions, labels)
        spearman_r, _ = spearmanr(predictions, labels)
    else:
        pearson_r = float('nan')
        spearman_r = float('nan')
    
    # Binary accuracy at threshold 0.5
    pred_binary = (predictions >= 0.5).astype(int)
    label_binary = (labels >= 0.5).astype(int)
    binary_acc = np.mean(pred_binary == label_binary)
    
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'binary_acc': float(binary_acc),
        'n_samples': int(len(predictions))
    }


def load_model(checkpoint_path, args, device):
    """Load a stable score model from checkpoint.
    
    Note: We set is_training=True to enable label processing (process_grasp_labels),
    but call net.eval() to disable dropout/batchnorm training behavior.
    """
    net = GraspNet(
        seed_feat_dim=args.seed_feat_dim,
        is_training=True,  # Enable label processing for evaluation
        backbone=args.backbone,
        enable_stable_score=True,  # Always enable for evaluation
    )
    net.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle old checkpoint format (bundled stable scores in conv_swad)
    if 'swad.conv_swad.weight' in state_dict:
        conv_swad_weight = state_dict['swad.conv_swad.weight']
        if conv_swad_weight.shape[0] == 108:  # 96 (SWAD) + 12 (stable)
            print("Converting old checkpoint format (bundled stable scores)...")
            # Split into separate layers  
            conv_swad_bias = state_dict.get('swad.conv_swad.bias')
            state_dict['swad.conv_swad.weight'] = conv_swad_weight[:96]
            state_dict['swad.conv_stable.weight'] = conv_swad_weight[96:]
            if conv_swad_bias is not None:
                state_dict['swad.conv_swad.bias'] = conv_swad_bias[:96]
                state_dict['swad.conv_stable.bias'] = conv_swad_bias[96:]
    
    # Load with strict=False to handle missing keys gracefully
    missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys}")
    
    net.eval()
    return net


def evaluate_on_dataset(net, dataset, device, args, pbar_desc="Evaluating"):
    """
    Run evaluation on a dataset and compute stable score metrics.
    
    Returns:
        dict with aggregated metrics and optionally per-sample data
    """
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0, 
        collate_fn=spconv_collate_fn
    )
    
    all_predictions = []
    all_labels = []
    all_masks = []  # Valid grasp mask
    
    def move_to_device(data, device):
        """Recursively move all tensors to device, converting numpy arrays."""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device)
        elif isinstance(data, dict):
            return {k: move_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, list):
            return [move_to_device(item, device) for item in data]
        elif isinstance(data, tuple):
            return tuple(move_to_device(item, device) for item in data)
        else:
            return data
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=pbar_desc):
            # Move all data to device (including nested lists)
            batch_data = move_to_device(batch_data, device)
            
            # Forward pass
            end_points = net(batch_data)
            
            # Extract predictions and labels
            if 'grasp_stable_pred' not in end_points:
                print("Warning: Model does not output grasp_stable_pred")
                continue
            
            stable_pred = end_points['grasp_stable_pred']  # (B, M, A)
            stable_label = end_points.get('batch_grasp_stable')  # (B, M, A)
            
            if stable_label is None:
                print("Warning: No stable labels in batch")
                continue
            
            # Get mask for valid grasps (where grasp_score > 0)
            grasp_score = end_points.get('batch_grasp_score')  # (B, M, A, D)
            if grasp_score is not None:
                valid_mask = (grasp_score > 0).any(dim=-1)  # (B, M, A)
            else:
                valid_mask = torch.ones_like(stable_pred, dtype=torch.bool)
            
            all_predictions.append(stable_pred.cpu().numpy())
            all_labels.append(stable_label.cpu().numpy())
            all_masks.append(valid_mask.cpu().numpy())
    
    if len(all_predictions) == 0:
        return {'error': 'No valid predictions collected'}
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # Compute metrics (only on valid grasps)
    metrics = compute_metrics(all_predictions, all_labels, mask=all_masks)
    
    # Add distribution statistics
    valid_preds = all_predictions[all_masks]
    valid_labels = all_labels[all_masks]
    
    metrics['pred_mean'] = float(np.mean(valid_preds))
    metrics['pred_std'] = float(np.std(valid_preds))
    metrics['pred_min'] = float(np.min(valid_preds))
    metrics['pred_max'] = float(np.max(valid_preds))
    metrics['label_mean'] = float(np.mean(valid_labels))
    metrics['label_std'] = float(np.std(valid_labels))
    
    result = {'metrics': metrics}
    
    if args.save_detailed:
        result['detailed'] = {
            'predictions': all_predictions.tolist(),
            'labels': all_labels.tolist(),
            'masks': all_masks.tolist()
        }
    
    return result


def main():
    args = parse_args()
    
    # List models and exit if requested
    if args.list_models:
        list_available_models()
        return
    
    # Validate model selection
    if args.model_name is None and args.checkpoint_path is None:
        print("Error: Must specify either --model_name or --checkpoint_path")
        print("Use --list_models to see available models")
        return
    
    # Resolve checkpoint path
    if args.model_name is not None:
        if args.model_name not in MODELS:
            print(f"Error: Unknown model '{args.model_name}'")
            list_available_models()
            return
        model_info = MODELS[args.model_name]
        checkpoint_path = os.path.join(ROOT_DIR, model_info['checkpoint'])
        if args.checkpoint_path:
            checkpoint_path = args.checkpoint_path
            print(f"Overriding checkpoint with: {checkpoint_path}")
    else:
        checkpoint_path = args.checkpoint_path
        args.model_name = "custom"
        model_info = {"lambda": "unknown", "description": "Custom checkpoint"}
    
    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    # Determine test sets
    if args.test_sets is None or 'all' in args.test_sets:
        test_sets = list(TEST_SETS.keys())
    else:
        test_sets = args.test_sets
    
    # Setup output directory
    output_dir = args.output_dir or SCRIPT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model: {args.model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    net = load_model(checkpoint_path, args, device)
    
    # Load grasp labels (needed for stable score evaluation)
    print("\nLoading grasp labels...")
    grasp_labels = load_grasp_labels(args.dataset_root)
    
    # Results container
    results = {
        'metadata': {
            'model_name': args.model_name,
            'model_info': model_info,
            'checkpoint_path': checkpoint_path,
            'timestamp': datetime.now().isoformat(),
            'camera': args.camera,
            'backbone': args.backbone,
            'num_point': args.num_point,
        },
        'results': {}
    }
    
    # Evaluate on each test set
    for test_set in test_sets:
        print(f"\n{'='*60}")
        print(f"Evaluating on: {test_set}")
        print(f"Description: {TEST_SETS[test_set]['description']}")
        print(f"{'='*60}")
        
        # Create dataset with labels enabled
        dataset = GraspNetDataset(
            args.dataset_root,
            split=test_set,
            camera=args.camera,
            num_points=args.num_point,
            voxel_size=args.voxel_size,
            remove_outlier=True,
            augment=False,
            load_label=True,  # Need labels for evaluation
            grasp_labels=grasp_labels,
            enable_stable_score=True,  # Enable stable score loading
        )
        
        print(f"Dataset size: {len(dataset)} samples")
        
        # Run evaluation
        result = evaluate_on_dataset(net, dataset, device, args, pbar_desc=f"  {test_set}")
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            results['results'][test_set] = result
            continue
        
        # Store results
        results['results'][test_set] = result
        
        # Print metrics
        metrics = result['metrics']
        print(f"\n  Results for {test_set}:")
        print(f"    MAE:        {metrics['mae']:.4f}")
        print(f"    MSE:        {metrics['mse']:.4f}")
        print(f"    RMSE:       {metrics['rmse']:.4f}")
        print(f"    Pearson r:  {metrics['pearson_r']:.4f}")
        print(f"    Spearman r: {metrics['spearman_r']:.4f}")
        print(f"    Binary Acc: {metrics['binary_acc']:.4f}")
        print(f"    N samples:  {metrics['n_samples']}")
        print(f"    Pred distribution: mean={metrics['pred_mean']:.3f}, std={metrics['pred_std']:.3f}")
        print(f"    Label distribution: mean={metrics['label_mean']:.3f}, std={metrics['label_std']:.3f}")
    
    # Save results
    output_file = os.path.join(output_dir, f"stable_score_accuracy_{args.model_name}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print(f"SUMMARY: {args.model_name}")
    print("=" * 80)
    print(f"{'Test Set':<20} {'MAE':<10} {'RMSE':<10} {'Pearson':<10} {'Spearman':<10} {'Bin.Acc':<10}")
    print("-" * 80)
    for test_set in test_sets:
        if test_set in results['results'] and 'metrics' in results['results'][test_set]:
            m = results['results'][test_set]['metrics']
            if m['n_samples'] > 0:
                print(f"{test_set:<20} {m['mae']:<10.4f} {m['rmse']:<10.4f} {m['pearson_r']:<10.4f} {m['spearman_r']:<10.4f} {m['binary_acc']:<10.4f}")
            else:
                print(f"{test_set:<20} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    print("=" * 80)


if __name__ == '__main__':
    main()
