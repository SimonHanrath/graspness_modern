"""
This is vibe coded and needs checking
Profile kNN function usage during training iterations.

Measures:
- How often knn_points_torch is called per training iteration
- Distribution of the K parameter values
- Call context (which function called kNN)

Usage:
    python experiments/benchmarking/profile_knn_usage.py --dataset_root /path/to/graspnet --num_iters 10
"""

import sys
import argparse
import traceback
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# KNN Call Tracker
# =============================================================================
@dataclass
class KNNCall:
    """Single kNN call info."""
    k: int
    n_query: int      # Number of query points (N in (B, N, D))
    n_ref: int        # Number of reference points (M in (B, M, D))
    batch_size: int   # B
    caller: str       # Which function called kNN


@dataclass  
class KNNProfiler:
    """Tracks kNN calls during training."""
    
    calls: List[KNNCall] = field(default_factory=list)
    iteration_boundaries: List[int] = field(default_factory=list)  # indices where iterations start
    enabled: bool = True
    
    def record(self, k: int, n_query: int, n_ref: int, batch_size: int, caller: str):
        if self.enabled:
            self.calls.append(KNNCall(k, n_query, n_ref, batch_size, caller))
    
    def mark_iteration_start(self):
        self.iteration_boundaries.append(len(self.calls))
    
    def reset(self):
        self.calls.clear()
        self.iteration_boundaries.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Compute statistics from collected calls."""
        if not self.calls:
            return {"total_calls": 0}
        
        num_iters = len(self.iteration_boundaries)
        calls_per_iter = len(self.calls) / max(num_iters, 1)
        
        # K value distribution
        k_values = [c.k for c in self.calls]
        k_counts = defaultdict(int)
        for k in k_values:
            k_counts[k] += 1
        
        # Caller distribution
        caller_counts = defaultdict(int)
        for c in self.calls:
            caller_counts[c.caller] += 1
        
        # Size statistics
        query_sizes = [c.n_query for c in self.calls]
        ref_sizes = [c.n_ref for c in self.calls]
        
        return {
            "total_calls": len(self.calls),
            "num_iterations": num_iters,
            "calls_per_iteration": calls_per_iter,
            "k_distribution": dict(sorted(k_counts.items())),
            "caller_distribution": dict(sorted(caller_counts.items(), key=lambda x: -x[1])),
            "query_size_stats": {
                "min": min(query_sizes),
                "max": max(query_sizes),
                "mean": sum(query_sizes) / len(query_sizes),
            },
            "ref_size_stats": {
                "min": min(ref_sizes),
                "max": max(ref_sizes),
                "mean": sum(ref_sizes) / len(ref_sizes),
            },
        }


# Global profiler instance
knn_profiler = KNNProfiler()


def get_caller_context() -> str:
    """Get the calling function name from the stack trace."""
    stack = traceback.extract_stack()
    for frame_info in reversed(stack):
        # Skip profiling/patching frames
        if "profile_knn" in frame_info.filename:
            continue
        if frame_info.name in ("profiled_knn", "knn_points_torch", "_wrapped_knn"):
            continue
        # Return first meaningful caller
        if frame_info.name in ("ball_query", "cylinder_query", "three_nn", "knn_query", "forward"):
            return frame_info.name
    return "unknown"


# =============================================================================
# CRITICAL: Patch BEFORE importing any model code
# =============================================================================
def install_knn_patch():
    """
    Install the kNN profiling patch by modifying the source module.
    
    This must be called BEFORE any other module imports pointnet2_utils.
    We monkey-patch the module's globals dict so that all internal calls
    (ball_query, cylinder_query, three_nn) use our profiled version.
    """
    # Import the module first (it hasn't been imported elsewhere yet)
    import utils.pointnet.pointnet2_utils as pn2
    
    # Save the original
    _original_knn = pn2.knn_points_torch
    
    def _wrapped_knn(p1: torch.Tensor, p2: torch.Tensor, K: int):
        B, N, D = p1.shape
        M = p2.shape[1]
        caller = get_caller_context()
        knn_profiler.record(k=K, n_query=N, n_ref=M, batch_size=B, caller=caller)
        return _original_knn(p1, p2, K)
    
    # Replace in the module's namespace - this affects all internal calls too!
    pn2.knn_points_torch = _wrapped_knn
    
    # Also import knn_utils and patch its local reference
    import utils.knn_utils as knn_utils
    knn_utils.knn_points_torch = _wrapped_knn
    
    # ALSO patch knn_query itself to catch the k=1 fast path
    _original_knn_query = knn_utils.knn_query
    
    @torch.no_grad()
    def _wrapped_knn_query(pos, k, batch=None, query_pos=None, query_batch=None):
        # Record this call (knn_query may use fast path that bypasses knn_points_torch)
        if query_pos is None:
            query_pos_actual = pos
        else:
            query_pos_actual = query_pos
        N = pos.shape[0]
        Q = query_pos_actual.shape[0]
        knn_profiler.record(k=k, n_query=Q, n_ref=N, batch_size=1, caller="knn_query")
        return _original_knn_query(pos, k, batch, query_pos, query_batch)
    
    knn_utils.knn_query = _wrapped_knn_query
    
    # Also patch label_generation's local reference (it uses 'from ... import knn_query')
    import utils.label_generation as label_gen
    label_gen.knn_query = _wrapped_knn_query
    
    print(f"✓ Patched knn_points_torch in utils.pointnet.pointnet2_utils")
    print(f"✓ Patched knn_points_torch in utils.knn_utils")
    print(f"✓ Patched knn_query in utils.knn_utils (catches k=1 fast path)")
    print(f"✓ Patched knn_query in utils.label_generation")
    return _original_knn


# INSTALL PATCH IMMEDIATELY - before any model imports!
_original_knn_func = install_knn_patch()


# =============================================================================
# NOW we can import model code (after patching)
# =============================================================================
from models.graspnet import GraspNet
from models.loss import get_loss
from dataset.graspnet_dataset import (
    GraspNetDataset, spconv_collate_fn, load_grasp_labels
)


# =============================================================================
# Training Simulation
# =============================================================================
def run_profiling(args):
    """Run profiling on training iterations."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load grasp labels
    print("Loading grasp labels...")
    grasp_labels = load_grasp_labels(args.dataset_root)
    
    # Create dataset
    print("Creating dataset...")
    train_dataset = GraspNetDataset(
        args.dataset_root,
        grasp_labels=grasp_labels,
        camera=args.camera,
        split="train",
        num_points=args.num_points,
        voxel_size=args.voxel_size,
        augment=False,  # No augmentation for profiling
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=spconv_collate_fn,
        drop_last=True,
    )
    
    # Create model
    print(f"Creating model with backbone={args.backbone}...")
    net = GraspNet(
        seed_feat_dim=args.seed_feat_dim,
        is_training=True,
        backbone=args.backbone,
        enable_stable_score=False,
    )
    net = net.to(device)
    net.train()
    
    # Create optimizer (needed for backward pass)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    
    # Warmup
    print(f"\nRunning {args.warmup_iters} warmup iterations...")
    knn_profiler.enabled = False
    data_iter = iter(train_loader)
    
    for i in range(args.warmup_iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        # Transfer to GPU (handle nested lists like train.py does)
        for key in batch:
            if 'list' in key:
                for ii in range(len(batch[key])):
                    for jj in range(len(batch[key][ii])):
                        batch[key][ii][jj] = batch[key][ii][jj].to(device)
            elif isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        with torch.amp.autocast(device_type="cuda", enabled=False):
            end_points = net(batch)
            loss, _ = get_loss(end_points)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Warmup {i+1}/{args.warmup_iters} - Loss: {loss.item():.4f}")
    
    # Profile iterations
    print(f"\nProfiling {args.num_iters} iterations...")
    knn_profiler.enabled = True
    knn_profiler.reset()
    
    for i in range(args.num_iters):
        knn_profiler.mark_iteration_start()
        
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        # Transfer to GPU (handle nested lists like train.py does)
        for key in batch:
            if 'list' in key:
                for ii in range(len(batch[key])):
                    for jj in range(len(batch[key][ii])):
                        batch[key][ii][jj] = batch[key][ii][jj].to(device)
            elif isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        with torch.amp.autocast(device_type="cuda", enabled=False):
            end_points = net(batch)
            loss, _ = get_loss(end_points)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Show progress
        current_calls = len(knn_profiler.calls)
        iter_start = knn_profiler.iteration_boundaries[-1] if knn_profiler.iteration_boundaries else 0
        calls_this_iter = current_calls - iter_start
        print(f"  Iter {i+1}/{args.num_iters} - Loss: {loss.item():.4f}, kNN calls: {calls_this_iter}")
    
    # Print results
    print("\n" + "="*70)
    print("KNN USAGE PROFILE RESULTS")
    print("="*70)
    
    stats = knn_profiler.get_stats()
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total kNN calls: {stats['total_calls']}")
    print(f"   Training iterations: {stats['num_iterations']}")
    print(f"   Calls per iteration: {stats['calls_per_iteration']:.1f}")
    
    print(f"\n📈 K Parameter Distribution:")
    for k, count in stats['k_distribution'].items():
        pct = 100 * count / stats['total_calls']
        bar = "█" * int(pct / 2)
        print(f"   K={k:5d}: {count:5d} calls ({pct:5.1f}%) {bar}")
    
    print(f"\n🔍 Caller Distribution:")
    for caller, count in stats['caller_distribution'].items():
        pct = 100 * count / stats['total_calls']
        print(f"   {caller:20s}: {count:5d} calls ({pct:5.1f}%)")
    
    print(f"\n📐 Query Size Stats (N in (B, N, D)):")
    qs = stats['query_size_stats']
    print(f"   Min: {qs['min']:,d}, Max: {qs['max']:,d}, Mean: {qs['mean']:,.0f}")
    
    print(f"\n📐 Reference Size Stats (M in (B, M, D)):")
    rs = stats['ref_size_stats']
    print(f"   Min: {rs['min']:,d}, Max: {rs['max']:,d}, Mean: {rs['mean']:,.0f}")
    
    # Per-iteration breakdown
    if len(knn_profiler.iteration_boundaries) > 1:
        print(f"\n📋 Per-Iteration Breakdown:")
        for i in range(len(knn_profiler.iteration_boundaries)):
            start_idx = knn_profiler.iteration_boundaries[i]
            end_idx = knn_profiler.iteration_boundaries[i+1] if i+1 < len(knn_profiler.iteration_boundaries) else len(knn_profiler.calls)
            iter_calls = knn_profiler.calls[start_idx:end_idx]
            k_vals = [c.k for c in iter_calls]
            k_summary = ", ".join([f"K={k}" for k in sorted(set(k_vals))])
            print(f"   Iter {i+1}: {len(iter_calls)} calls - {k_summary}")
    
    print("\n" + "="*70)
    
    return stats


# =============================================================================
# Main
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Profile kNN usage during training")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to GraspNet dataset")
    parser.add_argument("--camera", type=str, default="realsense",
                        choices=["realsense", "kinect"])
    parser.add_argument("--num_points", type=int, default=15000)
    parser.add_argument("--voxel_size", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--backbone", type=str, default="resunet",
                        choices=["resunet", "pointnet2", "transformer", "transformer_pretrained"])
    parser.add_argument("--seed_feat_dim", type=int, default=512)
    parser.add_argument("--num_iters", type=int, default=5,
                        help="Number of iterations to profile")
    parser.add_argument("--warmup_iters", type=int, default=2,
                        help="Warmup iterations before profiling")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_profiling(args)
