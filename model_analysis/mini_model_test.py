#!/usr/bin/env python3
"""
AI generated !!!!
Single model test script.
Runs one model on the 3 mini test sets (seen, similar, novel) without stable score.

Basically a script that executes test.py sequentially on the 3 mini test sets, with some added bells and whistles:

Usage:
    python model_analysis/mini_model_test.py \
        --checkpoint_path logs/cluster_100scenes_13epochs_realsense/gsnet_dev_epoch04.tar \
        --model_name "resunet realsense 4 epochs lr scale 0.01 test 0.8" \
        --backbone resunet \
        --friction 0.8 \
        --graspness_threshold 0.01
"""

import subprocess
import json
import os
import sys
import argparse
import time
import gc
import numpy as np
from datetime import datetime, timedelta

# Configuration
DATASET_ROOT = "/datasets/graspnet"
CAMERA = "realsense"
BACKBONE = "resunet"
NUM_POINT = 15000
BATCH_SIZE = 1

# Test sets
TEST_SETS =["test_seen_mini", "test_similar_mini", "test_novel_mini"]


def parse_args():
    parser = argparse.ArgumentParser(description='Run single model on 3 mini test sets')
    parser.add_argument('--checkpoint_path', required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name for results (default: derived from checkpoint path)')
    parser.add_argument('--dataset_root', type=str, default=DATASET_ROOT,
                        help='Path to GraspNet dataset root')
    parser.add_argument('--camera', type=str, default=CAMERA,
                        help='Camera type')
    parser.add_argument('--backbone', type=str, default=BACKBONE,
                        choices=['transformer', 'transformer_pretrained', 'pointnet2', 'resunet', 'resunet18', 'resunet_rgb', 'resunet18_rgb', 'sonata'],
                        help='Backbone architecture [default: resunet]. resunet=14D, resunet18=18D (more layers). Use _rgb suffix for 6-channel RGB input.')
    parser.add_argument('--num_point', type=int, default=NUM_POINT,
                        help='Number of points to sample')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output JSON file (default: single_model_results_<model_name>.json)')
    parser.add_argument('--no_collision', action='store_true', default=False,
                        help='Skip collision detection (faster but lower AP scores)')
    parser.add_argument('--infer_only', action='store_true', default=False,
                        help='Run inference only without evaluation (fastest)')
    parser.add_argument('--friction', type=float, nargs='+', default=None,
                        help='Friction coefficient(s) for AP evaluation. '
                             'Default: [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]. '
                             'Example: --friction 0.8 for single value, or --friction 0.2 0.4 0.8 for multiple.')
    parser.add_argument('--graspness_threshold', type=float, default=0.01,
                        help='Threshold for graspness score filtering during forward pass [default: -0.1]')
    parser.add_argument('--nsample', type=int, default=16,
                        help='Number of samples for cloud crop in GraspNet [default: 16]')
    parser.add_argument('--include_floor', action='store_true', default=False,
                        help='Include floor/table points in inference (for models trained with --include_floor)')
    parser.add_argument('--seed_feat_dim', type=int, default=512,
                        help='Point wise feature dim [default: 512]')
    return parser.parse_args()


def read_ap_from_npy(dump_dir, camera):
    """Read AP value directly from the numpy file saved by test.py.

    test.py saves the per-scene accuracy array to <dump_dir>/ap_<camera>.npy.
    AP is simply np.mean(res).
    """
    ap_path = os.path.join(dump_dir, f'ap_{camera}.npy')
    if not os.path.exists(ap_path):
        return None
    res = np.load(ap_path)
    return float(np.mean(res))


def run_test(args, test_set, test_idx=1, total_tests=1):
    """Run a single test and return the result."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    model_name = args.model_name or os.path.basename(os.path.dirname(args.checkpoint_path))
    model_name_safe = model_name.replace(' ', '_').replace('/', '_')
    dump_dir = f"dumps/single_test_{model_name_safe}_{test_set}"
    dump_dir_abs = os.path.join(project_root, dump_dir)
    
    os.makedirs(dump_dir_abs, exist_ok=True)
    
    cmd = [
        "python", "model_analysis/test.py",
        "--dataset_root", args.dataset_root,
        "--camera", args.camera,
        "--checkpoint_path", args.checkpoint_path,
        "--dump_dir", dump_dir,
        "--batch_size", str(args.batch_size),
        "--infer",
        "--backbone", args.backbone,
        "--num_point", str(args.num_point),
        "--split", test_set,
    ]
    
    # Add eval flag unless infer_only mode
    if not args.infer_only:
        cmd.append("--eval")
    
    # Skip collision detection if requested
    if args.no_collision:
        cmd.extend(["--collision_thresh", "0"])
    
    # Forward friction coefficients
    if args.friction is not None:
        cmd.extend(["--friction"] + [str(f) for f in args.friction])
    
    # Forward graspness threshold
    cmd.extend(["--graspness_threshold", str(args.graspness_threshold)])
    
    # Forward nsample
    cmd.extend(["--nsample", str(args.nsample)])
    
    # Forward include_floor
    if args.include_floor:
        cmd.append("--include_floor")
    
    # Forward seed_feat_dim
    cmd.extend(["--seed_feat_dim", str(args.seed_feat_dim)])
    
    print(f"\n{'='*80}", flush=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting test {test_idx}/{total_tests}", flush=True)
    print(f"  Model: {model_name}", flush=True)
    print(f"  Test set: {test_set}", flush=True)
    print(f"  Command: {' '.join(cmd)}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    try:
        # Use Popen for real-time streaming output (so you can see progress via `less` on cluster)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=project_root
        )
        
        for line in process.stdout:
            print(line, end='', flush=True)
        
        return_code = process.wait()
        
        # Read AP directly from the numpy file that test.py saves,
        # instead of fragile regex parsing of stdout.
        ap = read_ap_from_npy(dump_dir_abs, args.camera)
        
        if ap is not None:
            return {
                "status": "completed",
                "ap": ap,
                "return_code": return_code
            }
        else:
            return {
                "status": "error",
                "error": f"AP file not found at {os.path.join(dump_dir, f'ap_{args.camera}.npy')} (return_code={return_code})",
                "return_code": return_code,
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def main():
    args = parse_args()
    
    model_name = args.model_name or os.path.basename(os.path.dirname(args.checkpoint_path))
    
    print("="*80, flush=True)
    print(f"Single Model Test: {model_name}", flush=True)
    print("="*80, flush=True)
    print(f"Checkpoint: {args.checkpoint_path}", flush=True)
    print(f"Backbone: {args.backbone}", flush=True)
    print(f"Camera: {args.camera}", flush=True)
    print(f"Num points: {args.num_point}", flush=True)
    print(f"Collision detection: {'OFF' if args.no_collision else 'ON'}", flush=True)
    print(f"Eval mode: {'infer only' if args.infer_only else 'infer + eval'}", flush=True)
    print(f"Friction coefficients: {args.friction if args.friction else '[0.2, 0.4, 0.6, 0.8, 1.0, 1.2] (default)'}", flush=True)
    print(f"Test sets: {TEST_SETS}", flush=True)
    print(f"Total tests: {len(TEST_SETS)}", flush=True)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(flush=True)
    
    # Initialize results
    data = {
        "metadata": {
            "started": datetime.now().isoformat(),
            "model_name": model_name,
            "checkpoint": args.checkpoint_path,
            "camera": args.camera,
            "backbone": args.backbone,
            "num_point": args.num_point,
            "friction": args.friction if args.friction else [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        },
        "results": []
    }
    
    completed = 0
    failed = 0
    total_tests = len(TEST_SETS)
    test_times = []
    overall_start = time.time()
    
    for idx, test_set in enumerate(TEST_SETS, 1):
        test_start = time.time()
        result = run_test(args, test_set, test_idx=idx, total_tests=total_tests)
        test_elapsed = time.time() - test_start
        test_times.append(test_elapsed)
        
        test_record = {
            "test_set": test_set,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(test_elapsed, 1),
            **result
        }
        data["results"].append(test_record)
        
        if result["status"] == "completed":
            print(f"\n*** [{datetime.now().strftime('%H:%M:%S')}] Test {idx}/{total_tests} completed: {test_set} -> AP = {result['ap']:.6f} (took {timedelta(seconds=int(test_elapsed))}) ***", flush=True)
            completed += 1
        else:
            print(f"\n*** [{datetime.now().strftime('%H:%M:%S')}] Test {idx}/{total_tests} FAILED: {test_set} -> {result.get('error', 'Unknown error')} (took {timedelta(seconds=int(test_elapsed))}) ***", flush=True)
            failed += 1
        
        # Print progress & ETA
        remaining = total_tests - idx
        if remaining > 0:
            avg_time = sum(test_times) / len(test_times)
            eta = timedelta(seconds=int(avg_time * remaining))
            print(f"    Progress: {idx}/{total_tests} done | ETA for remaining {remaining} test(s): ~{eta}", flush=True)
        
        # Print intermediate results so far
        print(f"\n    --- Results so far ---", flush=True)
        for r in data["results"]:
            if r["status"] == "completed":
                print(f"    {r['test_set']:<20} AP={r['ap']:.6f}  ({r['duration_seconds']}s)", flush=True)
            else:
                print(f"    {r['test_set']:<20} FAILED", flush=True)
        print(flush=True)
        
        # Force garbage collection between tests to prevent memory accumulation
        gc.collect()
    
    # Save results
    model_name_safe = model_name.replace(' ', '_').replace('/', '_')
    output_file = args.output_file or f"single_model_results_{model_name_safe}.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    total_elapsed = time.time() - overall_start
    print(f"\nResults saved to: {output_file}", flush=True)
    
    # Print summary
    print("\n" + "="*80, flush=True)
    print("TEST COMPLETE", flush=True)
    print("="*80, flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"Completed: {completed}", flush=True)
    print(f"Failed: {failed}", flush=True)
    print(f"Total time: {timedelta(seconds=int(total_elapsed))}", flush=True)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    print("\n" + "="*80, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("="*80, flush=True)
    print(f"{'Test Set':<25} {'AP':<12} {'Time':<12}", flush=True)
    print("-"*50, flush=True)
    
    for r in data["results"]:
        dur = f"{r.get('duration_seconds', '?')}s"
        if r["status"] == "completed":
            print(f"{r['test_set']:<25} {r['ap']:<12.6f} {dur:<12}", flush=True)
        else:
            print(f"{r['test_set']:<25} {'FAILED':<12} {dur:<12}", flush=True)
    
    # Calculate average AP
    aps = [r['ap'] for r in data["results"] if r["status"] == "completed"]
    if aps:
        print("-"*50, flush=True)
        print(f"{'Average':<25} {sum(aps)/len(aps):<12.6f}", flush=True)


if __name__ == "__main__":
    main()
