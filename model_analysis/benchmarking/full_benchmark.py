"""
This is vibe coded and needs checking
Detailed profiling of GraspNet forward pass stages and operations.

Tracks time for:
- Model stages: backbone, graspable, rotation, crop, swad
- Key operations: FPS, kNN, grouping, interpolation
"""

import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# =============================================================================
# Thesis Plot Style Configuration
# =============================================================================
# Text width: 14cm = 5.51 inches (A4, 12pt base font, Computer Modern)
THESIS_TEXTWIDTH_INCHES = 14 / 2.54  # ~5.51 inches

plt.rcParams.update({
    # Figure size based on thesis text width
    'figure.figsize': (THESIS_TEXTWIDTH_INCHES, 3.5),
    'figure.constrained_layout.use': True,
    
    # Font settings - Computer Modern (LaTeX default)
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',  # Computer Modern math
    
    # Font sizes - scaled for 12pt thesis body text
    'font.size': 10,           # Base font
    'axes.labelsize': 10,      # Axis labels
    'axes.titlesize': 10,      # Subplot titles
    'xtick.labelsize': 8,      # Tick labels
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    
    # Clean styling
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
    
    # Disable LaTeX rendering (not available in container)
    'text.usetex': False,
})

# Thesis-friendly color palettes
ITERATION_COLORS = ['#E07A5F', '#3D405B', '#81B29A', '#F2CC8F']  # Terra cotta, charcoal, sage, sand
STAGE_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#95190C', '#610345', '#044B7F']

from dataset.graspnet_dataset import GraspNetDataset, spconv_collate_fn, load_grasp_labels, SceneAwareSampler


# =============================================================================
# Configuration
# =============================================================================
DATASET_ROOT = "/datasets/graspnet"
CAMERA = "realsense"
CHECKPOINT = project_root / "logs/backbone_experiments/gsnet_pointnet2_rawpoints_realsense_t01_n15/gsnet_pointnet2_rawpoints_epoch15.tar"
NUM_POINTS = 15000
VOXEL_SIZE = 0.005
N_WARMUP = 10
N_RUNS = 10


# =============================================================================
# Profiling Infrastructure
# =============================================================================
class Profiler:
    """Simple profiler to track execution times."""
    
    def __init__(self):
        self.times = defaultdict(list)
        self.enabled = True
    
    @contextmanager
    def track(self, name: str):
        """Context manager to track time for a named operation."""
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        self.times[name].append(time.perf_counter() - start)
    
    def reset(self):
        self.times.clear()
    
    def summary(self) -> dict:
        """Return mean times in milliseconds."""
        return {k: np.mean(v) * 1000 for k, v in self.times.items()}


# Global profiler instance
profiler = Profiler()


# =============================================================================
# Monkey-patch key functions to add profiling
# =============================================================================
def patch_functions():
    """Patch key functions with profiling wrappers."""
    
    import utils.pointnet.pointnet2_utils as pn2
    import utils.knn_utils as knn_utils
    
    # Patch furthest_point_sample
    _original_fps = pn2.furthest_point_sample
    def profiled_fps(*args, **kwargs):
        with profiler.track("fps"):
            return _original_fps(*args, **kwargs)
    pn2.furthest_point_sample = profiled_fps
    
    # Patch gather_operation
    _original_gather = pn2.gather_operation
    def profiled_gather(*args, **kwargs):
        with profiler.track("gather"):
            return _original_gather(*args, **kwargs)
    pn2.gather_operation = profiled_gather
    
    # Patch knn_points_torch
    _original_knn = pn2.knn_points_torch
    def profiled_knn(*args, **kwargs):
        with profiler.track("knn_points"):
            return _original_knn(*args, **kwargs)
    pn2.knn_points_torch = profiled_knn
    
    # Patch three_nn (uses knn internally but track separately)
    _original_three_nn = pn2.three_nn
    def profiled_three_nn(*args, **kwargs):
        with profiler.track("three_nn"):
            return _original_three_nn(*args, **kwargs)
    pn2.three_nn = profiled_three_nn
    
    # Patch three_interpolate
    _original_interp = pn2.three_interpolate
    def profiled_interp(*args, **kwargs):
        with profiler.track("interpolate"):
            return _original_interp(*args, **kwargs)
    pn2.three_interpolate = profiled_interp
    
    # Patch ball_query
    _original_ball = pn2.ball_query
    def profiled_ball(*args, **kwargs):
        with profiler.track("ball_query"):
            return _original_ball(*args, **kwargs)
    pn2.ball_query = profiled_ball
    
    # Patch knn_query from knn_utils
    _original_knn_query = knn_utils.knn_query
    def profiled_knn_query(*args, **kwargs):
        with profiler.track("knn_query"):
            return _original_knn_query(*args, **kwargs)
    knn_utils.knn_query = profiled_knn_query


# =============================================================================
# Profiled GraspNet Forward Pass
# =============================================================================
def profiled_forward(net, end_points):
    """Forward pass with stage-level profiling."""
    import spconv.pytorch as spconv
    from utils.pointnet.pointnet2_utils import furthest_point_sample, gather_operation
    from utils.label_generation import process_grasp_labels, match_grasp_view_and_label
    from utils.loss_utils import GRASPNESS_THRESHOLD, M_POINT
    
    seed_xyz = end_points['point_clouds']
    B, point_num, _ = seed_xyz.shape
    
    # Prepare sparse input
    with profiler.track("prepare_sparse"):
        coords = end_points['coors'].to(dtype=torch.int32)
        feats = end_points['feats']
        quantize2original = end_points['quantize2original']
        
        mins = coords[:, 1:].amin(dim=0)
        maxs = coords[:, 1:].amax(dim=0)
        coords[:, 1:] = coords[:, 1:] - mins.unsqueeze(0)
        extent = (maxs - mins + 1)
        
        MIN_SPATIAL_DIM = 16
        spatial_shape_xyz = (
            max(int(extent[0].item()), MIN_SPATIAL_DIM),
            max(int(extent[1].item()), MIN_SPATIAL_DIM),
            max(int(extent[2].item()), MIN_SPATIAL_DIM),
        )
        
        if net.backbone_type == "transformer_pretrained" and feats.shape[1] == 3:
            coord_float = coords[:, 1:].float()
            coord_min = coord_float.min(dim=0, keepdim=True).values
            coord_max = coord_float.max(dim=0, keepdim=True).values
            coord_normalized = (coord_float - coord_min) / (coord_max - coord_min + 1e-6)
            feats_6ch = torch.cat([coord_normalized, feats], dim=1)
        else:
            feats_6ch = feats
        
        sparse_input = spconv.SparseConvTensor(feats_6ch, coords.contiguous(), spatial_shape_xyz, B)
    
    # Backbone
    with profiler.track("backbone"):
        sparse_output = net.backbone(sparse_input)
        voxel_features = sparse_output.features
        seed_features = voxel_features[quantize2original].view(B, point_num, -1).transpose(1, 2)
    
    # Graspable
    with profiler.track("graspable"):
        end_points = net.graspable(seed_features, end_points)
    
    # FPS and filtering
    with profiler.track("fps_filtering"):
        seed_features_flipped = seed_features.transpose(1, 2)
        objectness_score = end_points['objectness_score']
        graspness_score = end_points['graspness_score'].squeeze(1)
        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask = graspness_score > GRASPNESS_THRESHOLD
        graspable_mask = objectness_mask & graspness_mask
        
        seed_features_graspable = []
        seed_xyz_graspable = []
        graspable_num_batch = 0.
        
        for i in range(B):
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()
            cur_feat = seed_features_flipped[i][cur_mask]
            cur_seed_xyz = seed_xyz[i][cur_mask]
            Ns = cur_seed_xyz.shape[0]
            
            cur_seed_xyz = cur_seed_xyz.unsqueeze(0)
            num_to_sample = min(Ns, M_POINT)
            
            if num_to_sample > 0:
                fps_idxs = furthest_point_sample(cur_seed_xyz, num_to_sample)
                cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()
                cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous()
                cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()
                cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous()
            else:
                cur_seed_xyz = torch.zeros((0, 3), device=seed_xyz.device, dtype=seed_xyz.dtype)
                cur_feat = torch.zeros((net.seed_feature_dim, 0), device=seed_features.device, dtype=seed_features.dtype)
            
            if num_to_sample < M_POINT:
                pad_num = M_POINT - num_to_sample
                xyz_pad = torch.zeros((pad_num, 3), device=seed_xyz.device, dtype=seed_xyz.dtype)
                cur_seed_xyz = torch.cat([cur_seed_xyz, xyz_pad], dim=0)
                feat_dim = cur_feat.shape[0] if cur_feat.shape[0] > 0 else net.seed_feature_dim
                feat_pad = torch.zeros((feat_dim, pad_num), device=seed_features.device, dtype=seed_features.dtype)
                cur_feat = torch.cat([cur_feat, feat_pad], dim=1)
            
            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)
        
        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)
        seed_features_graspable = torch.stack(seed_features_graspable)
        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['graspable_count_stage1'] = graspable_num_batch / B
    
    # Rotation
    with profiler.track("rotation"):
        end_points, res_feat = net.rotation(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat
    
    # Label processing (training mode)
    with profiler.track("label_processing"):
        if net.is_training:
            end_points = process_grasp_labels(end_points)
            grasp_top_views_rot, end_points = match_grasp_view_and_label(end_points)
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']
    
    # Crop (cylinder grouping)
    with profiler.track("crop"):
        group_features = net.crop(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
    
    # SWAD
    with profiler.track("swad"):
        end_points = net.swad(group_features, end_points)
    
    return end_points


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


def benchmark_data_loading(dataloader, device, n_samples=20, n_warmup=5):
    """Benchmark per-batch data loading time (what happens before each forward pass).
    
    Args:
        n_warmup: Number of warmup batches to populate OS page cache and dataset caches.
                  These are excluded from timing to measure steady-state performance.
    """
    
    print(f"Benchmarking per-batch data loading ({n_warmup} warmup + {n_samples} measured batches)...")
    print("  This measures: disk I/O, preprocessing, voxelization, augmentation, collation")
    print("  Note: Warmup batches populate OS page cache and dataset caches (excluded from timing)")
    
    load_times = []
    transfer_times = []
    
    data_iter = iter(dataloader)
    
    # Warmup: populate caches (excluded from timing)
    for i in range(n_warmup):
        batch_data = next(data_iter)
        batch_data = transfer_batch_to_device(batch_data, device)
        torch.cuda.synchronize()
    print(f"  Warmup complete ({n_warmup} batches)")
    
    for i in range(n_samples):
        # Time fetching batch from DataLoader (disk + preprocessing + collation)
        start = time.perf_counter()
        batch_data = next(data_iter)
        load_time = time.perf_counter() - start
        load_times.append(load_time)
        
        # Time GPU transfer
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch_data = transfer_batch_to_device(batch_data, device)
        torch.cuda.synchronize()
        transfer_time = time.perf_counter() - start
        transfer_times.append(transfer_time)
        
        if (i + 1) % 5 == 0:
            print(f"  Batch {i+1}: load={load_time*1000:.1f}ms, transfer={transfer_time*1000:.1f}ms")
    
    return {
        'load_mean': np.mean(load_times) * 1000,
        'load_std': np.std(load_times) * 1000,
        'transfer_mean': np.mean(transfer_times) * 1000,
        'transfer_std': np.std(transfer_times) * 1000,
        'total_mean': (np.mean(load_times) + np.mean(transfer_times)) * 1000,
    }


def benchmark_full_iteration(net, dataloader, device, n_iterations=10, n_warmup=10):
    """Benchmark complete training iteration (data loading + forward + backward).
    
    n_warmup iterations are used to JIT compile spconv kernels, warm up OS page cache,
    and populate dataset caches (collision labels). These are excluded from timing.
    """
    
    print(f"\nBenchmarking full training iteration ({n_iterations} iters, {n_warmup} warmup)...")
    print("  Each iteration = DataLoader fetch + GPU transfer + forward + backward")
    
    from models.loss import get_loss
    
    data_iter = iter(dataloader)
    
    iteration_times = []
    load_times = []
    transfer_times = []
    forward_times = []
    backward_times = []
    
    for i in range(n_warmup + n_iterations):
        # 1. Data loading from DataLoader
        start_load = time.perf_counter()
        try:
            batch_data = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch_data = next(data_iter)
        load_time = time.perf_counter() - start_load
        
        # 2. GPU transfer
        torch.cuda.synchronize()
        start_transfer = time.perf_counter()
        batch_data = transfer_batch_to_device(batch_data, device)
        torch.cuda.synchronize()
        transfer_time = time.perf_counter() - start_transfer
        
        # 3. Forward pass
        net.zero_grad()
        torch.cuda.synchronize()
        start_forward = time.perf_counter()
        end_points = net(batch_data)
        torch.cuda.synchronize()
        forward_time = time.perf_counter() - start_forward
        
        # 4. Loss + Backward pass
        torch.cuda.synchronize()
        start_backward = time.perf_counter()
        loss, _ = get_loss(end_points)
        loss.backward()
        torch.cuda.synchronize()
        backward_time = time.perf_counter() - start_backward
        
        total = load_time + transfer_time + forward_time + backward_time
        
        if i < n_warmup:
            print(f"  Warmup {i+1}: load={load_time*1000:.0f}ms, fwd={forward_time*1000:.0f}ms, "
                  f"bwd={backward_time*1000:.0f}ms (JIT compile)")
        else:
            load_times.append(load_time)
            transfer_times.append(transfer_time)
            forward_times.append(forward_time)
            backward_times.append(backward_time)
            iteration_times.append(total)
            print(f"  Iter {i+1-n_warmup}: load={load_time*1000:.0f}ms, transfer={transfer_time*1000:.0f}ms, "
                  f"fwd={forward_time*1000:.0f}ms, bwd={backward_time*1000:.0f}ms, total={total*1000:.0f}ms")
    
    return {
        'iteration_mean': np.mean(iteration_times) * 1000,
        'iteration_std': np.std(iteration_times) * 1000,
        'load_mean': np.mean(load_times) * 1000,
        'transfer_mean': np.mean(transfer_times) * 1000,
        'forward_mean': np.mean(forward_times) * 1000,
        'backward_mean': np.mean(backward_times) * 1000,
    }


def run_profiled_benchmark(net, dataloader, device, n_warmup=10, n_runs=10):
    """Run profiled benchmark and return timing breakdown."""
    
    batch_data = transfer_batch_to_device(next(iter(dataloader)), device)
    
    # Warmup (no profiling)
    print(f"Warming up ({n_warmup} iterations)...", end=" ", flush=True)
    profiler.enabled = False
    for _ in range(n_warmup):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = profiled_forward(net, batch_data.copy())
        torch.cuda.synchronize()
        print(f"{(time.perf_counter()-start)*1000:.0f}ms", end=" ", flush=True)
    print()
    
    # Profiled runs
    print(f"Profiling ({n_runs} iterations)...", end=" ", flush=True)
    profiler.enabled = True
    profiler.reset()
    
    total_times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = profiled_forward(net, batch_data.copy())
        torch.cuda.synchronize()
        total_times.append(time.perf_counter() - start)
    print("done")
    
    return profiler.summary(), np.mean(total_times) * 1000


def plot_breakdown(stage_times, op_times, total_time, output_path, iter_times=None):
    """Create thesis-quality visualization of timing breakdown."""
    
    def group_small_items(names, values, total, threshold_pct=1.0):
        """Group items under threshold_pct into 'other' category."""
        filtered_names = []
        filtered_values = []
        other_total = 0
        
        for name, val in zip(names, values):
            pct = (val / total) * 100 if total > 0 else 0
            if pct >= threshold_pct:
                filtered_names.append(name)
                filtered_values.append(val)
            else:
                other_total += val
        
        if other_total > 0:
            filtered_names.append('other')
            filtered_values.append(other_total)
        
        return filtered_names, filtered_values
    
    def make_autopct_ms(values, threshold_pct=0):
        """Create autopct function that shows milliseconds."""
        def autopct(pct):
            if pct < threshold_pct:
                return ''
            total = sum(values)
            val = pct * total / 100.0
            return f'{val:.0f}ms'
        return autopct
    
    if iter_times:
        # 2 plots: full iteration, stage breakdown
        fig, axes = plt.subplots(1, 2, figsize=(THESIS_TEXTWIDTH_INCHES * 1.3, 3.0))
        
        # Plot 1: Full iteration breakdown (what really matters - show ms)
        ax = axes[0]
        components = ['Data Loading', 'GPU Transfer', 'Forward', 'Backward']
        times_iter = [
            iter_times['load_mean'],
            iter_times['transfer_mean'],
            iter_times['forward_mean'],
            iter_times['backward_mean']
        ]
        
        wedges, texts, autotexts = ax.pie(
            times_iter, 
            autopct=make_autopct_ms(times_iter, threshold_pct=0),
            colors=ITERATION_COLORS, 
            startangle=90,
            pctdistance=0.75,
            wedgeprops={'linewidth': 1.0, 'edgecolor': 'white'}
        )
        for autotext in autotexts:
            autotext.set_fontsize(8)
        ax.legend(
            wedges, 
            components,
            loc='center left', 
            bbox_to_anchor=(-0.55, 0.5), 
            frameon=False
        )
        ax.set_title(f'Training Iteration ({iter_times["iteration_mean"]:.0f}ms)')
        
        # Plot 2: Forward pass stage breakdown (percentages only, group <1%)
        ax = axes[1]
        stages = ['backbone', 'fps_filtering', 'label_processing', 'crop', 'graspable', 'rotation', 'swad', 'prepare_sparse']
        times_stage = [stage_times.get(s, 0) for s in stages]
        
        # Add unaccounted time
        accounted = sum(times_stage)
        unaccounted = total_time - accounted
        if unaccounted > 0:
            stages.append('unaccounted')
            times_stage.append(unaccounted)
        
        # Group items under 1% into 'other'
        stages, times_stage = group_small_items(stages, times_stage, total_time, threshold_pct=1.0)
        
        # Use thesis colors (cycle if needed)
        n_stages = len(stages)
        colors = STAGE_COLORS[:n_stages] if n_stages <= len(STAGE_COLORS) else \
                 STAGE_COLORS + list(plt.cm.Set3(np.linspace(0, 1, n_stages - len(STAGE_COLORS))))
        
        wedges, texts, autotexts = ax.pie(
            times_stage, 
            autopct=make_autopct_ms(times_stage, threshold_pct=5),
            colors=colors, 
            startangle=90,
            pctdistance=0.75,
            wedgeprops={'linewidth': 1.0, 'edgecolor': 'white'}
        )
        for autotext in autotexts:
            autotext.set_fontsize(8)
        ax.legend(
            wedges, 
            stages,
            loc='center left', 
            bbox_to_anchor=(-0.55, 0.5), 
            frameon=False
        )
        ax.set_title('Forward Pass Breakdown')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(THESIS_TEXTWIDTH_INCHES * 0.7, 3.0))
        
        # Stage breakdown (pie chart - percentages only, group <1%)
        stages = ['backbone', 'graspable', 'fps_filtering', 'rotation', 'label_processing', 'crop', 'swad', 'prepare_sparse']
        times_stage = [stage_times.get(s, 0) for s in stages]
        
        accounted = sum(times_stage)
        unaccounted = total_time - accounted
        if unaccounted > 0:
            stages.append('unaccounted')
            times_stage.append(unaccounted)
        
        # Group items under 1% into 'other'
        stages, times_stage = group_small_items(stages, times_stage, total_time, threshold_pct=1.0)
        
        n_stages = len(stages)
        colors = STAGE_COLORS[:n_stages] if n_stages <= len(STAGE_COLORS) else \
                 STAGE_COLORS + list(plt.cm.Set3(np.linspace(0, 1, n_stages - len(STAGE_COLORS))))
        
        wedges, texts, autotexts = ax.pie(
            times_stage, 
            autopct=make_autopct_ms(times_stage, threshold_pct=5),
            colors=colors, 
            startangle=90,
            pctdistance=0.75,
            wedgeprops={'linewidth': 1.0, 'edgecolor': 'white'}
        )
        for autotext in autotexts:
            autotext.set_fontsize(8)
        ax.legend(
            wedges, 
            stages,
            loc='center left', 
            bbox_to_anchor=(-0.55, 0.5), 
            frameon=False
        )
        ax.set_title('Forward Pass Breakdown')
    
    # Save in multiple formats for thesis
    output_base = str(output_path).rsplit('.', 1)[0]
    fig.savefig(f'{output_base}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'{output_base}.png', bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {output_base}.pdf and .png")
    plt.close()


def print_breakdown(times, total_time, data_times=None):
    """Print detailed timing breakdown."""
    
    print(f"\n{'='*60}")
    print("TIMING BREAKDOWN")
    print(f"{'='*60}")
    
    if data_times:
        print(f"\nFull Iteration Timing (training-realistic):")
        print(f"  Data loading:    {data_times['load_mean']:.2f} ms  ({data_times['load_mean']/data_times['iteration_mean']*100:.1f}%)")
        print(f"  GPU transfer:    {data_times['transfer_mean']:.2f} ms  ({data_times['transfer_mean']/data_times['iteration_mean']*100:.1f}%)")
        print(f"  Forward pass:    {data_times['forward_mean']:.2f} ms  ({data_times['forward_mean']/data_times['iteration_mean']*100:.1f}%)")
        print(f"  Backward pass:   {data_times['backward_mean']:.2f} ms  ({data_times['backward_mean']/data_times['iteration_mean']*100:.1f}%)")
        print(f"  ────────────────────────")
        print(f"  Total/iter:      {data_times['iteration_mean']:.2f} ± {data_times['iteration_std']:.2f} ms")
    
    print(f"\nForward Pass Breakdown:")
    
    # Stages
    stages = ['prepare_sparse', 'backbone', 'graspable', 'fps_filtering', 'rotation', 'label_processing', 'crop', 'swad']
    print(f"\n{'Stage':<20} {'%':<12}")
    print("-" * 32)
    stage_total = 0
    for stage in stages:
        t = times.get(stage, 0)
        stage_total += t
        pct = (t / total_time) * 100 if total_time > 0 else 0
        print(f"{stage:<20} {pct:>10.1f}%")
    print("-" * 32)
    print(f"{'Accounted':<20} {(stage_total/total_time)*100:>10.1f}%")
    
    # Operations
    ops = ['fps', 'gather', 'knn_points', 'knn_query', 'ball_query', 'three_nn', 'interpolate']
    print(f"\n{'Operation':<20} {'%':<12}")
    print("-" * 32)
    for op in ops:
        t = times.get(op, 0)
        if t > 0:
            pct = (t / total_time) * 100 if total_time > 0 else 0
            print(f"{op:<20} {pct:>10.1f}%")


def main():
    print("=" * 60)
    print("GraspNet Detailed Profiling (ResUNet)")
    print("=" * 60)
    
    # Patch functions before importing GraspNet
    patch_functions()
    
    from models.graspnet import GraspNet
    
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("Loading grasp labels...")
    grasp_labels = load_grasp_labels(DATASET_ROOT)
    
    print("Loading dataset...")
    dataset = GraspNetDataset(
        DATASET_ROOT, grasp_labels=grasp_labels, split='train', camera=CAMERA,
        num_points=NUM_POINTS, voxel_size=VOXEL_SIZE,
        remove_outlier=True, augment=True, load_label=True,  # augment=True matches training
    )
    # Use SceneAwareSampler to match actual training behavior
    # This groups samples by scene for cache-friendly collision label loading
    sampler = SceneAwareSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,  # sampler handles shuffling
                            sampler=sampler,
                            num_workers=0, pin_memory=True,  # pin_memory=True matches training
                            collate_fn=spconv_collate_fn)
    print(f"Dataset: {len(dataset)} samples (using SceneAwareSampler)")
    
    # Load model
    print("Loading model...")
    net = GraspNet(seed_feat_dim=512, is_training=True, backbone="pointnet2")
    net.to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    net.train()  # Training mode for realistic benchmark
    print(f"Checkpoint: epoch {checkpoint['epoch']}")
    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # Benchmark full training iteration (realistic end-to-end)
    print("\n" + "-" * 60)
    print("FULL TRAINING ITERATION (realistic)")
    print("-" * 60)
    iter_times = benchmark_full_iteration(
        net,
        dataloader,
        device,
        n_iterations=N_RUNS,
        n_warmup=N_WARMUP,
    )
    
    print(f"\n  Summary:")
    print(f"    Data loading:   {iter_times['load_mean']:.0f} ms")
    print(f"    GPU transfer:   {iter_times['transfer_mean']:.0f} ms")
    print(f"    Forward pass:   {iter_times['forward_mean']:.0f} ms")
    print(f"    Backward pass:  {iter_times['backward_mean']:.0f} ms")
    print(f"    ─────────────────────────")
    print(f"    Total/iter:     {iter_times['iteration_mean']:.0f} ± {iter_times['iteration_std']:.0f} ms")
    print(f"    Throughput:     {1000/iter_times['iteration_mean']:.2f} samples/sec")
    
    # Run profiling (forward pass breakdown)
    print("\n" + "-" * 60)
    print("PROFILED FORWARD PASS (detailed breakdown)")
    print("-" * 60)
    net.eval()  # Eval mode for profiling (no dropout variance)
    times, total_time = run_profiled_benchmark(net, dataloader, device, N_WARMUP, N_RUNS)
    
    # Print results
    print_breakdown(times, total_time, iter_times)
    
    # Plot results
    output_dir = project_root / "model_analysis" / "benchmarking" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stage_times = {k: v for k, v in times.items() 
                   if k in ['prepare_sparse', 'backbone', 'graspable', 'fps_filtering', 
                           'rotation', 'label_processing', 'crop', 'swad']}
    op_times = {k: v for k, v in times.items() 
                if k in ['fps', 'gather', 'knn_points', 'knn_query', 'ball_query', 
                        'three_nn', 'interpolate']}
    
    plot_breakdown(stage_times, op_times, total_time, output_dir / "profiling_breakdown.png", iter_times)
    
    print("\nProfiling complete!")


if __name__ == "__main__":
    main()
