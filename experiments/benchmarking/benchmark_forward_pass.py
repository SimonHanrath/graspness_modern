"""Benchmarking script for GraspNet model forward and backward pass.

Measures inference and training time on actual data with warmup phase
to handle JIT compilation overhead.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.graspnet import GraspNet, pred_decode
from models.loss import get_loss
from dataset.graspnet_dataset import GraspNetDataset, spconv_collate_fn, load_grasp_labels


DATASET_ROOT = "/datasets/graspnet"
CAMERA = "realsense"
CHECKPOINT = "/workspace/logs/gsnet_resunet_stable_score/gsnet_resunet_epoch08.tar"
NUM_POINTS = 15000
VOXEL_SIZE = 0.005
N_WARMUP = 5
N_RUNS = 20


def transfer_batch_to_device(batch_data, device):
    """Transfer batch data to device."""
    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)
    return batch_data


def benchmark_forward_pass(net, dataloader, device, n_warmup=5, n_runs=20):
    """Benchmark the forward pass only with warmup for JIT compilation."""
    
    batch_data = transfer_batch_to_device(next(iter(dataloader)), device)
    
    # Warmup
    print(f"Warming up ({n_warmup} iterations)...", end=" ", flush=True)
    warmup_times = []
    for _ in range(n_warmup):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = net(batch_data.copy())
        torch.cuda.synchronize()
        warmup_times.append(time.perf_counter() - start)
        print(f"{warmup_times[-1]*1000:.0f}ms", end=" ", flush=True)
    
    # Timed runs
    print(f"Benchmarking ({n_runs} iterations)...", end=" ", flush=True)
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = net(batch_data.copy())
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return np.array(warmup_times), np.array(times)


def benchmark_backward_pass(net, dataloader, device, n_warmup=5, n_runs=20):
    """Benchmark forward + backward pass (training) with warmup."""
    
    batch_data = transfer_batch_to_device(next(iter(dataloader)), device)
    
    # Warmup
    print(f"Warming up ({n_warmup} iterations)...", end=" ", flush=True)
    warmup_times = []
    for _ in range(n_warmup):
        net.zero_grad()
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        end_points = net(batch_data.copy())
        loss, _ = get_loss(end_points)
        loss.backward()
        
        torch.cuda.synchronize()
        warmup_times.append(time.perf_counter() - start)
        print(f"{warmup_times[-1]*1000:.0f}ms", end=" ", flush=True)
    
    # Timed runs
    print(f"Benchmarking ({n_runs} iterations)...", end=" ", flush=True)
    times = []
    for _ in range(n_runs):
        net.zero_grad()
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        end_points = net(batch_data.copy())
        loss, _ = get_loss(end_points)
        loss.backward()
        
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return np.array(warmup_times), np.array(times)


def plot_results(fwd_warmup, fwd_times, bwd_warmup, bwd_times, output_path):
    """Create visualization of benchmark results."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Warmup progression (JIT effect)
    ax = axes[0]
    ax.plot(range(1, len(fwd_warmup) + 1), fwd_warmup * 1000, 
            marker='o', linewidth=2, markersize=8, color='#2ecc71', label='Forward')
    ax.plot(range(1, len(bwd_warmup) + 1), bwd_warmup * 1000, 
            marker='s', linewidth=2, markersize=8, color='#e74c3c', label='Forward+Backward')
    ax.set_xlabel("Warmup Iteration", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("JIT Warmup Effect", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, max(len(fwd_warmup), len(bwd_warmup)) + 1))
    
    # Plot 2: Time comparison bar chart
    ax = axes[1]
    labels = ['Forward\n(Inference)', 'Forward+Backward\n(Training)']
    means = [fwd_times.mean() * 1000, bwd_times.mean() * 1000]
    stds = [fwd_times.std() * 1000, bwd_times.std() * 1000]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor='black')
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("Forward vs Training Time", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{mean:.1f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 3: Time distribution
    ax = axes[2]
    ax.hist(fwd_times * 1000, bins=12, color='#2ecc71', edgecolor='black', 
            alpha=0.7, label=f'Forward: {fwd_times.mean()*1000:.1f}ms')
    ax.hist(bwd_times * 1000, bins=12, color='#e74c3c', edgecolor='black', 
            alpha=0.7, label=f'Fwd+Bwd: {bwd_times.mean()*1000:.1f}ms')
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Time Distribution", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("GraspNet Forward & Backward Pass Benchmark (ResUNet)")
    print("=" * 60)
    
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load grasp labels for loss computation
    print("Loading grasp labels...")
    grasp_labels = load_grasp_labels(DATASET_ROOT)
    
    # Load dataset with labels for backward pass
    print("Loading dataset...")
    dataset = GraspNetDataset(
        DATASET_ROOT, grasp_labels=grasp_labels, split='train', camera=CAMERA,
        num_points=NUM_POINTS, voxel_size=VOXEL_SIZE,
        remove_outlier=True, augment=False, load_label=True,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=spconv_collate_fn)
    print(f"Dataset: {len(dataset)} samples")
    
    # Load model in training mode
    print("Loading model...")
    net = GraspNet(seed_feat_dim=512, is_training=True, backbone='resunet')
    net.to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Checkpoint: epoch {checkpoint['epoch']}")
    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # Benchmark forward pass (inference)
    print("\n" + "-" * 60)
    print("FORWARD PASS (Inference)")
    print("-" * 60)
    net.eval()
    fwd_warmup, fwd_times = benchmark_forward_pass(net, dataloader, device, N_WARMUP, N_RUNS)
    
    # Benchmark backward pass (training)
    print("\n" + "-" * 60)
    print("FORWARD + BACKWARD PASS (Training)")
    print("-" * 60)
    net.train()
    bwd_warmup, bwd_times = benchmark_backward_pass(net, dataloader, device, N_WARMUP, N_RUNS)
    
    # Results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nForward (Inference):")
    print(f"  Mean: {fwd_times.mean()*1000:.2f} ± {fwd_times.std()*1000:.2f} ms")
    print(f"  FPS:  {1/fwd_times.mean():.1f}")
    print(f"\nForward + Backward (Training):")
    print(f"  Mean: {bwd_times.mean()*1000:.2f} ± {bwd_times.std()*1000:.2f} ms")
    print(f"  FPS:  {1/bwd_times.mean():.1f}")
    print(f"\nBackward overhead: {(bwd_times.mean() - fwd_times.mean())*1000:.2f} ms ({bwd_times.mean()/fwd_times.mean():.2f}x total)")
    
    # Plot
    output_dir = project_root / "experiments" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_results(fwd_warmup, fwd_times, bwd_warmup, bwd_times, 
                 output_dir / "forward_backward_benchmark.png")


if __name__ == "__main__":
    main()
