#!/usr/bin/env python3
"""
AI generated !!!!
Multiple epoch loss evaluation script.
Evaluates loss for different epochs of the same model efficiently by loading
labels and datasets once, then only swapping model weights.

Similar to mini_model_test.py but evaluates validation loss instead of AP metrics.
Useful for finding optimal training epochs by tracking the validation loss curve.

Usage:
    # Evaluate all epochs in a log directory:
    python model_analysis/mini_loss_test.py \
        --log_dir logs/cluster_100scenes_13epochs_realsense \
        --backbone resunet \
        --split test_similar_mini \
        --graspness_threshold 0.01 \
        --epochs 1 5 9 10 11 12 13 \

    # Evaluate specific epochs:
    python model_analysis/mini_loss_test.py \
        --log_dir logs/gsnet_sonata_t01_n15_lrscale01_correct \
        --epochs 5 10 15 \
        --backbone sonata
        
    # Compare across multiple test splits:
    python model_analysis/mini_loss_test.py \
        --log_dir logs/gsnet_resunet_realsense \
        --backbone resunet \
        --splits test_seen_mini test_similar_mini test_novel_mini
"""

import json
import os
import sys
import argparse
import glob
import re
import time
import gc
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import DataLoader

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from models.graspnet import GraspNet
from models.loss import get_loss
from dataset.graspnet_dataset import GraspNetDataset, spconv_collate_fn, load_grasp_labels

# Configuration
DATASET_ROOT = "/datasets/graspnet"
CAMERA = "realsense"
BACKBONE = "resunet"
NUM_POINT = 15000
BATCH_SIZE = 1
DEFAULT_SPLIT = "test_seen_mini"


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate loss across multiple epochs')
    parser.add_argument('--log_dir', required=True,
                        help='Path to log directory containing checkpoint files (e.g., logs/gsnet_resunet_realsense)')
    parser.add_argument('--epochs', type=int, nargs='+', default=None,
                        help='Specific epochs to evaluate (default: auto-detect all available epochs)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name for results (default: derived from log_dir)')
    parser.add_argument('--dataset_root', type=str, default=DATASET_ROOT,
                        help='Path to GraspNet dataset root')
    parser.add_argument('--camera', type=str, default=CAMERA,
                        help='Camera type')
    parser.add_argument('--backbone', type=str, default=BACKBONE,
                        choices=['transformer', 'transformer_pretrained', 'sonata', 'pointnet2', 
                                 'resunet', 'resunet18', 'resunet_rgb', 'resunet18_rgb'],
                        help='Backbone architecture [default: resunet]')
    parser.add_argument('--num_point', type=int, default=NUM_POINT,
                        help='Number of points to sample')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--voxel_size', type=float, default=0.005,
                        help='Voxel size for sparse convolution')
    parser.add_argument('--split', type=str, default=DEFAULT_SPLIT,
                        help='Dataset split to evaluate on (default: test_seen_mini)')
    parser.add_argument('--splits', type=str, nargs='+', default=None,
                        help='Multiple splits to evaluate (overrides --split)')
    parser.add_argument('--graspness_threshold', type=float, default=0.01,
                        help='Threshold for graspness score filtering [default: 0.01]')
    parser.add_argument('--nsample', type=int, default=16,
                        help='Number of samples for cloud crop in GraspNet [default: 16]')
    parser.add_argument('--include_floor', action='store_true', default=False,
                        help='Include floor/table points in inference')
    parser.add_argument('--enable_stable_score', action='store_true', default=False,
                        help='Enable stable score prediction')
    parser.add_argument('--lambda_stable', type=float, default=10.0,
                        help='Weight for stable score loss term [default: 10.0]')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of DataLoader workers')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output JSON file (default: epoch_loss_<model_name>.json)')
    return parser.parse_args()


def find_checkpoints(log_dir):
    """Find all checkpoint files and extract epoch numbers."""
    patterns = [
        os.path.join(log_dir, '*epoch*.tar'),
        os.path.join(log_dir, 'epoch_*.tar'),
    ]
    
    checkpoints = {}
    for pattern in patterns:
        for path in glob.glob(pattern):
            basename = os.path.basename(path)
            match = re.search(r'epoch[_]?(\d+)', basename, re.IGNORECASE)
            if match:
                epoch = int(match.group(1))
                checkpoints[epoch] = path
    
    return checkpoints


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def load_checkpoint_weights(net, checkpoint_path, device, enable_stable_score=False):
    """Load checkpoint weights into existing model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Handle old checkpoint format with bundled stable scores
    if 'swad.conv_swad.weight' in state_dict:
        old_weight = state_dict['swad.conv_swad.weight']
        old_bias = state_dict['swad.conv_swad.bias']
        if old_weight.shape[0] == 108 and enable_stable_score:
            print("  Converting old checkpoint (bundled 108 outputs) to new format...")
            state_dict['swad.conv_swad.weight'] = old_weight[:96]
            state_dict['swad.conv_swad.bias'] = old_bias[:96]
            state_dict['swad.conv_stable.weight'] = old_weight[96:]
            state_dict['swad.conv_stable.bias'] = old_bias[96:]
    
    net.load_state_dict(state_dict, strict=False)
    epoch = checkpoint.get('epoch', 'unknown')
    return epoch


def evaluate_single_epoch(net, dataloader, device, args):
    """Run evaluation for a single epoch and return loss results."""
    net.eval()
    
    loss_accum = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in dataloader:
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
            loss, end_points = get_loss(
                end_points, 
                enable_stable_score=args.enable_stable_score,
                lambda_stable=args.lambda_stable
            )
            
            # Accumulate losses
            for key in end_points:
                if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key:
                    val = end_points[key]
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    if key not in loss_accum:
                        loss_accum[key] = 0.0
                    loss_accum[key] += val
            
            num_batches += 1
    
    # Compute averages
    results = {
        'losses': {},
        'metrics': {}
    }
    for key in sorted(loss_accum.keys()):
        avg_val = loss_accum[key] / num_batches
        if 'loss' in key:
            results['losses'][key] = avg_val
        else:
            results['metrics'][key] = avg_val
    
    return results


def main():
    args = parse_args()
    
    # Determine model name
    model_name = args.model_name or os.path.basename(args.log_dir)
    
    # Find checkpoints
    checkpoints = find_checkpoints(args.log_dir)
    if not checkpoints:
        print(f"ERROR: No checkpoint files found in {args.log_dir}")
        sys.exit(1)
    
    # Filter to specific epochs if requested
    if args.epochs:
        checkpoints = {e: p for e, p in checkpoints.items() if e in args.epochs}
        if not checkpoints:
            print(f"ERROR: None of the requested epochs {args.epochs} found")
            sys.exit(1)
    
    # Determine splits to evaluate
    splits = args.splits if args.splits else [args.split]
    sorted_epochs = sorted(checkpoints.keys())
    total_tests = len(sorted_epochs) * len(splits)
    
    print(f"\n{'='*80}")
    print(f"Multi-Epoch Loss Evaluation (Efficient Mode)")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Log dir: {args.log_dir}")
    print(f"Epochs to evaluate: {sorted_epochs}")
    print(f"Splits: {splits}")
    print(f"Backbone: {args.backbone}")
    print(f"Graspness threshold: {args.graspness_threshold}")
    if args.include_floor:
        print("Include floor: enabled")
    if args.enable_stable_score:
        print(f"Stable score: enabled (lambda={args.lambda_stable})")
    print(f"Total evaluations: {total_tests}")
    print(f"{'='*80}\n")
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable backend optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Auto-enable RGB for backbones that require 6-channel input
    use_rgb = args.backbone in ['transformer_pretrained', 'resunet_rgb', 'resunet18_rgb']
    if use_rgb:
        print("Using RGB features for 6-channel input (XYZ + RGB)")
    
    # Load grasp labels ONCE (this is the expensive part)
    print("\nLoading grasp labels (one-time operation)...")
    label_start = time.time()
    grasp_labels = load_grasp_labels(args.dataset_root)
    print(f"  Labels loaded in {time.time() - label_start:.1f}s")
    
    # Create model ONCE (will swap weights for each checkpoint)
    print("\nInitializing model...")
    net = GraspNet(
        seed_feat_dim=512, 
        is_training=True,  # Need training mode for loss computation
        backbone=args.backbone,
        enable_stable_score=args.enable_stable_score,
        graspness_threshold=args.graspness_threshold,
        nsample=args.nsample,
    )
    net.to(device)
    
    # Create dataloaders ONCE per split (reuse across epochs)
    print("\nCreating dataloaders...")
    dataloaders = {}
    for split in splits:
        dataset = GraspNetDataset(
            args.dataset_root, 
            grasp_labels=grasp_labels,
            split=split, 
            camera=args.camera, 
            num_points=args.num_point,
            voxel_size=args.voxel_size, 
            remove_outlier=True, 
            augment=False,
            load_label=True,
            use_rgb=use_rgb,
            enable_stable_score=args.enable_stable_score,
            include_floor=args.include_floor
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers, 
            worker_init_fn=my_worker_init_fn, 
            collate_fn=spconv_collate_fn
        )
        dataloaders[split] = dataloader
        print(f"  {split}: {len(dataset)} samples")
    
    # Run evaluations
    print(f"\n{'='*80}")
    print("Starting evaluations...")
    print(f"{'='*80}")
    
    results = {}
    start_time = time.time()
    test_idx = 0
    
    for epoch in sorted_epochs:
        checkpoint_path = checkpoints[epoch]
        results[epoch] = {}
        
        # Load checkpoint weights
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading epoch {epoch}: {os.path.basename(checkpoint_path)}")
        loaded_epoch = load_checkpoint_weights(net, checkpoint_path, device, args.enable_stable_score)
        
        for split in splits:
            test_idx += 1
            print(f"  Evaluating {split}... ({test_idx}/{total_tests})", end='', flush=True)
            
            eval_start = time.time()
            try:
                result = evaluate_single_epoch(net, dataloaders[split], device, args)
                result['status'] = 'completed'
                result['epoch'] = loaded_epoch
                eval_time = time.time() - eval_start
                overall_loss = result['losses'].get('loss/overall_loss', 'N/A')
                print(f" loss={overall_loss:.4f} ({eval_time:.1f}s)")
            except Exception as e:
                result = {'status': 'error', 'error': str(e)}
                print(f" ERROR: {e}")
            
            results[epoch][split] = result
        
        # Clear CUDA cache between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: Multi-Epoch Loss Evaluation")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Total time: {total_time:.1f}s")
    print()
    
    # Print table header
    header = f"{'Epoch':>6} |"
    for split in splits:
        split_short = split.replace('test_', '').replace('_mini', '')
        header += f" {split_short:>12} |"
    print(header)
    print("-" * len(header))
    
    # Print results per epoch
    for epoch in sorted_epochs:
        row = f"{epoch:>6} |"
        for split in splits:
            if results[epoch][split]['status'] == 'completed':
                overall_loss = results[epoch][split].get('losses', {}).get('loss/overall_loss', 'N/A')
                if isinstance(overall_loss, (int, float)):
                    row += f" {overall_loss:>12.4f} |"
                else:
                    row += f" {'N/A':>12} |"
            else:
                row += f" {'ERROR':>12} |"
        print(row)
    
    print("-" * len(header))
    
    # Find best epoch per split
    print("\nBest epochs (lowest loss):")
    for split in splits:
        best_epoch = None
        best_loss = float('inf')
        for epoch in sorted_epochs:
            if results[epoch][split]['status'] == 'completed':
                loss = results[epoch][split].get('losses', {}).get('loss/overall_loss', float('inf'))
                if isinstance(loss, (int, float)) and loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
        if best_epoch is not None:
            split_short = split.replace('test_', '').replace('_mini', '')
            print(f"  {split_short}: epoch {best_epoch} (loss={best_loss:.4f})")
    
    # Save results
    model_name_safe = model_name.replace(' ', '_').replace('/', '_')
    output_file = args.output_file or os.path.join(SCRIPT_DIR, 'dumps', f'epoch_loss_{model_name_safe}.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    output_data = {
        'model_name': model_name,
        'log_dir': args.log_dir,
        'backbone': args.backbone,
        'camera': args.camera,
        'graspness_threshold': args.graspness_threshold,
        'splits': splits,
        'epochs': sorted_epochs,
        'total_time_seconds': total_time,
        'timestamp': datetime.now().isoformat(),
        'results': {str(e): results[e] for e in results}
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*80}")
    
    return results


if __name__ == '__main__':
    main()
