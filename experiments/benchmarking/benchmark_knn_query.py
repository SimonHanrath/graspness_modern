"""
Benchmarking script for knn_query function in utils/knn_utils.py

Measures performance across:
- Different point cloud sizes (N)
- Different k values
- Single batch vs batched processing
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.knn_utils import knn_query


def benchmark_single_config(
    n_points: int,
    k: int,
    n_batches: int = 1,
    n_warmup: int = 3,
    n_runs: int = 10,
    device: str = "cuda"
) -> dict:
    """
    Benchmark knn_query for a single configuration.
    
    Args:
        n_points: Total number of points
        k: Number of neighbors
        n_batches: Number of batches (1 = single batch mode)
        n_warmup: Number of warmup runs
        n_runs: Number of timed runs
        device: Device to run on
        
    Returns:
        dict with timing statistics
    """
    # Generate random point cloud
    pos = torch.randn(n_points, 3, device=device, dtype=torch.float32)
    
    if n_batches > 1:
        # Create batch indices (roughly equal batch sizes)
        points_per_batch = n_points // n_batches
        batch = torch.repeat_interleave(
            torch.arange(n_batches, device=device),
            points_per_batch
        )
        # Handle remainder
        if len(batch) < n_points:
            batch = torch.cat([batch, torch.full((n_points - len(batch),), n_batches - 1, device=device)])
        batch = batch[:n_points]
    else:
        batch = None
    
    # Warmup
    for _ in range(n_warmup):
        _ = knn_query(pos, k, batch)
        if device == "cuda":
            torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        _ = knn_query(pos, k, batch)
        
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    times = np.array(times)
    return {
        "mean": times.mean(),
        "std": times.std(),
        "min": times.min(),
        "max": times.max(),
        "median": np.median(times),
    }


def run_benchmark_suite(device: str = "cuda"):
    """Run full benchmark suite and return results."""
    
    results = {
        "varying_n": {},
        "varying_k": {},
        "varying_batches": {},
    }
    
    # ==========================================================================
    # Benchmark 1: Varying number of points (fixed k=16)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Benchmark 1: Varying number of points (k=16)")
    print("=" * 60)
    
    n_values = [1000, 5000, 10000, 25000, 50000, 100000]
    k_fixed = 16
    
    for n in n_values:
        print(f"  N={n:,}...", end=" ", flush=True)
        try:
            result = benchmark_single_config(n, k_fixed, n_batches=1, device=device)
            results["varying_n"][n] = result
            print(f"mean={result['mean']*1000:.2f}ms, std={result['std']*1000:.2f}ms")
        except Exception as e:
            print(f"FAILED: {e}")
            results["varying_n"][n] = None
    
    # ==========================================================================
    # Benchmark 2: Varying k (fixed N=50000)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Benchmark 2: Varying k (N=50,000)")
    print("=" * 60)
    
    n_fixed = 50000
    k_values = [1, 4, 8, 16, 32, 64, 128]
    
    for k in k_values:
        print(f"  k={k}...", end=" ", flush=True)
        try:
            result = benchmark_single_config(n_fixed, k, n_batches=1, device=device)
            results["varying_k"][k] = result
            print(f"mean={result['mean']*1000:.2f}ms, std={result['std']*1000:.2f}ms")
        except Exception as e:
            print(f"FAILED: {e}")
            results["varying_k"][k] = None
    
    # ==========================================================================
    # Benchmark 3: Varying number of batches (fixed N=50000, k=16)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Benchmark 3: Varying batches (N=50,000, k=16)")
    print("=" * 60)
    
    batch_values = [1, 2, 4, 8, 16]
    
    for n_batches in batch_values:
        print(f"  batches={n_batches}...", end=" ", flush=True)
        try:
            result = benchmark_single_config(n_fixed, k_fixed, n_batches=n_batches, device=device)
            results["varying_batches"][n_batches] = result
            print(f"mean={result['mean']*1000:.2f}ms, std={result['std']*1000:.2f}ms")
        except Exception as e:
            print(f"FAILED: {e}")
            results["varying_batches"][n_batches] = None
    
    return results


def plot_results(results: dict, output_path: Path):
    """Create visualization of benchmark results."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # ==========================================================================
    # Plot 1: Varying N
    # ==========================================================================
    ax = axes[0]
    data = results["varying_n"]
    n_vals = [n for n in sorted(data.keys()) if data[n] is not None]
    means = [data[n]["mean"] * 1000 for n in n_vals]
    stds = [data[n]["std"] * 1000 for n in n_vals]
    
    ax.errorbar(n_vals, means, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel("Number of Points (N)", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("knn_query: Varying N (k=16)", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_vals)
    ax.set_xticklabels([f"{n//1000}k" if n >= 1000 else str(n) for n in n_vals])
    
    # ==========================================================================
    # Plot 2: Varying k
    # ==========================================================================
    ax = axes[1]
    data = results["varying_k"]
    k_vals = [k for k in sorted(data.keys()) if data[k] is not None]
    means = [data[k]["mean"] * 1000 for k in k_vals]
    stds = [data[k]["std"] * 1000 for k in k_vals]
    
    ax.errorbar(k_vals, means, yerr=stds, marker='s', capsize=5, linewidth=2, markersize=8, color='green')
    ax.set_xlabel("Number of Neighbors (k)", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("knn_query: Varying k (N=50k)", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_vals)
    ax.set_xticklabels([str(k) for k in k_vals])
    
    # ==========================================================================
    # Plot 3: Varying batches
    # ==========================================================================
    ax = axes[2]
    data = results["varying_batches"]
    batch_vals = [b for b in sorted(data.keys()) if data[b] is not None]
    means = [data[b]["mean"] * 1000 for b in batch_vals]
    stds = [data[b]["std"] * 1000 for b in batch_vals]
    
    ax.errorbar(batch_vals, means, yerr=stds, marker='^', capsize=5, linewidth=2, markersize=8, color='orange')
    ax.set_xlabel("Number of Batches", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("knn_query: Varying Batches (N=50k, k=16)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(batch_vals)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("knn_query Benchmark Suite")
    print("=" * 60)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Run benchmarks
    results = run_benchmark_suite(device=device)
    
    # Create output directory
    output_dir = project_root / "experiments" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot results
    output_path = output_dir / "knn_query_benchmark.png"
    plot_results(results, output_path)
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
