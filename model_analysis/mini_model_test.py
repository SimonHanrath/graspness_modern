#!/usr/bin/env python3
"""
AI slop !!!!
Single model test script.
Runs one model on the 3 mini test sets (seen, similar, novel) without stable score.

Usage:
    python model_analysis/mini_model_test.py \
        --checkpoint_path logs/cluster_100scenes_13epochs_realsense/gsnet_dev_epoch10.tar \
        --model_name "Vanilla ResUNet Realsese 10 epochs full test results" \
        --backbone resunet
"""

import subprocess
import json
import os
import re
import sys
import argparse
from datetime import datetime

# Configuration
DATASET_ROOT = "/datasets/graspnet"
CAMERA = "realsense"
BACKBONE = "resunet"
NUM_POINT = 15000
BATCH_SIZE = 1

# Test sets
TEST_SETS = ["test_seen", "test_similar", "test_novel"]


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
                        choices=['transformer', 'transformer_pretrained', 'pointnet2', 'resunet', 'resunet_rgb'],
                        help='Backbone architecture [default: resunet]. Use resunet_rgb for ResUNet with RGB features.')
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
    return parser.parse_args()


def parse_ap_from_output(output):
    """Parse AP value from test.py output."""
    patterns = [
        r'AP Seen Mini[^=]+=([0-9.]+)',
        r'AP Similar Mini[^=]+=([0-9.]+)',
        r'AP Novel Mini[^=]+=([0-9.]+)',
        r'AP Seen[^=]+=([0-9.]+)',
        r'AP Novel[^=]+=([0-9.]+)',
        r'AP Similar[^=]+=([0-9.]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))
    return None


def run_test(args, test_set):
    """Run a single test and return the result."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    model_name = args.model_name or os.path.basename(os.path.dirname(args.checkpoint_path))
    model_name_safe = model_name.replace(' ', '_').replace('/', '_')
    dump_dir = f"dumps/single_test_{model_name_safe}_{test_set}"
    
    os.makedirs(os.path.join(project_root, dump_dir), exist_ok=True)
    
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
    
    print(f"\n{'='*80}")
    print(f"Running test:")
    print(f"  Model: {model_name}")
    print(f"  Test set: {test_set}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        full_output = result.stdout + result.stderr
        print(full_output)
        
        ap = parse_ap_from_output(full_output)
        
        if ap is not None:
            return {
                "status": "completed",
                "ap": ap,
                "return_code": result.returncode
            }
        else:
            return {
                "status": "error",
                "error": "Could not parse AP value from output",
                "return_code": result.returncode,
                "output": full_output[-2000:]
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def main():
    args = parse_args()
    
    model_name = args.model_name or os.path.basename(os.path.dirname(args.checkpoint_path))
    
    print("="*80)
    print(f"Single Model Test: {model_name}")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Test sets: {TEST_SETS}")
    print(f"Total tests: {len(TEST_SETS)}")
    print()
    
    # Initialize results
    data = {
        "metadata": {
            "started": datetime.now().isoformat(),
            "model_name": model_name,
            "checkpoint": args.checkpoint_path,
            "camera": args.camera,
            "backbone": args.backbone,
            "num_point": args.num_point
        },
        "results": []
    }
    
    completed = 0
    failed = 0
    
    for test_set in TEST_SETS:
        result = run_test(args, test_set)
        
        test_record = {
            "test_set": test_set,
            "timestamp": datetime.now().isoformat(),
            **result
        }
        data["results"].append(test_record)
        
        if result["status"] == "completed":
            print(f"\n*** Test completed: AP = {result['ap']:.6f} ***\n")
            completed += 1
        else:
            print(f"\n*** Test failed: {result.get('error', 'Unknown error')} ***\n")
            failed += 1
    
    # Save results
    model_name_safe = model_name.replace(' ', '_').replace('/', '_')
    output_file = args.output_file or f"single_model_results_{model_name_safe}.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Test Set':<25} {'AP':<12}")
    print("-"*40)
    
    for r in data["results"]:
        if r["status"] == "completed":
            print(f"{r['test_set']:<25} {r['ap']:<12.6f}")
        else:
            print(f"{r['test_set']:<25} {'FAILED':<12}")
    
    # Calculate average AP
    aps = [r['ap'] for r in data["results"] if r["status"] == "completed"]
    if aps:
        print("-"*40)
        print(f"{'Average':<25} {sum(aps)/len(aps):<12.6f}")


if __name__ == "__main__":
    main()
