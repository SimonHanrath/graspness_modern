#!/usr/bin/env python3
"""
Analyze the distribution of stable scores across objects to understand
if the normalization is working correctly.
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stable_score_utils import (
    compute_mesh_cog, 
    generate_grasp_views, 
    compute_grasp_plane_normal
)


def analyze_object(graspnet_root, obj_id, num_views=300):
    """Analyze stable score distribution for one object."""
    obj_id_str = str(obj_id).zfill(3)
    
    mesh_path = os.path.join(graspnet_root, 'models', obj_id_str, 'nontextured.ply')
    label_path = os.path.join(graspnet_root, 'grasp_label', f'{obj_id_str}_labels.npz')
    
    if not os.path.exists(mesh_path) or not os.path.exists(label_path):
        return None
    
    labels = np.load(label_path)
    grasp_points = labels['points']
    
    cog = compute_mesh_cog(mesh_path)
    views = generate_grasp_views(num_views)
    
    Np = grasp_points.shape[0]
    V = num_views
    
    # Compute raw distances (before normalization)
    raw_distances = []
    
    for p_idx in range(Np):
        grasp_point = grasp_points[p_idx]
        cog_to_point = cog - grasp_point
        
        for v_idx in range(V):
            view = views[v_idx]
            plane_normal = compute_grasp_plane_normal(view)
            distance = np.abs(np.dot(plane_normal, cog_to_point))
            raw_distances.append(distance)
    
    raw_distances = np.array(raw_distances)
    
    # Current normalization: divide by max
    max_dist = raw_distances.max()
    normalized = raw_distances / max_dist if max_dist > 1e-8 else raw_distances
    
    return {
        'obj_id': obj_id,
        'num_points': Np,
        'cog': cog,
        'raw_min': raw_distances.min(),
        'raw_max': raw_distances.max(),
        'raw_mean': raw_distances.mean(),
        'raw_std': raw_distances.std(),
        'norm_min': normalized.min(),
        'norm_max': normalized.max(),
        'norm_mean': normalized.mean(),
        'norm_std': normalized.std(),
        'norm_25pct': np.percentile(normalized, 25),
        'norm_50pct': np.percentile(normalized, 50),
        'norm_75pct': np.percentile(normalized, 75),
        'norm_95pct': np.percentile(normalized, 95),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graspnet_root', type=str, default='/datasets/graspnet')
    parser.add_argument('--obj_ids', type=str, default='0,1,2,10,50,87',
                        help='Comma-separated object IDs to analyze')
    args = parser.parse_args()
    
    obj_ids = [int(x) for x in args.obj_ids.split(',')]
    
    print("=" * 80)
    print("STABLE SCORE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    for obj_id in obj_ids:
        print(f"\n--- Object {obj_id:03d} ---")
        result = analyze_object(args.graspnet_root, obj_id)
        
        if result is None:
            print("  [Not found]")
            continue
        
        print(f"  Grasp points: {result['num_points']}")
        print(f"  COG: ({result['cog'][0]:.4f}, {result['cog'][1]:.4f}, {result['cog'][2]:.4f})")
        print()
        print(f"  Raw distances (meters):")
        print(f"    Min:  {result['raw_min']:.6f}")
        print(f"    Max:  {result['raw_max']:.6f}")
        print(f"    Mean: {result['raw_mean']:.6f}")
        print(f"    Std:  {result['raw_std']:.6f}")
        print()
        print(f"  Normalized scores (0-1):")
        print(f"    Min:  {result['norm_min']:.4f}")
        print(f"    Max:  {result['norm_max']:.4f}  <-- Should be 1.0")
        print(f"    Mean: {result['norm_mean']:.4f}")
        print(f"    Std:  {result['norm_std']:.4f}")
        print()
        print(f"  Percentiles:")
        print(f"    25%: {result['norm_25pct']:.4f}")
        print(f"    50%: {result['norm_50pct']:.4f}")
        print(f"    75%: {result['norm_75pct']:.4f}")
        print(f"    95%: {result['norm_95pct']:.4f}")
        
        # Check if distribution is problematic
        if result['norm_75pct'] < 0.3:
            print()
            print("  ⚠️  WARNING: 75% of scores are below 0.3!")
            print("      This means stable score has minimal effect on ranking.")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    print("""
    With formula: grasp_score * (1 - stable_score)
    
    If most stable scores are in [0, 0.3]:
      - Best grasp: score * 1.0
      - Worst grasp: score * 0.7
      - Only 30% variation! Stable score barely matters.
    
    If stable scores span [0, 1]:
      - Best grasp: score * 1.0  
      - Worst grasp: score * 0.0
      - 100% variation! Stable score strongly affects ranking.
    
    POTENTIAL FIXES:
    1. Per-point normalization (each point has 0-1 range across views)
    2. Stronger weighting: score * (1 - alpha * stable_score) with alpha > 1
    3. Non-linear scaling: sqrt(stable_score) to spread low values
    4. Threshold-based: if stable_score > threshold, heavily penalize
    """)


if __name__ == '__main__':
    main()
