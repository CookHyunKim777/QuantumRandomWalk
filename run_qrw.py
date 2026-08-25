#!/usr/bin/env python3
"""
Main QRW simulation script.
This script can also be called from a SLURM job.

Usage:
    python run_qrw.py --param_file params/qrw_m/params_001.json
    python run_qrw.py --graph_type cycle --graph_size 400 --walk_type QRW_M --time_length 160000 --save_interval 100
"""

import argparse
import json
import os
import sys
import time
import numpy as np

from qrw_utils import create_graph, run_walk


def parse_args():
    parser = argparse.ArgumentParser(description='QRW Simulation')
    
    parser.add_argument('--param_file', type=str, default=None,
                        help='Path to a JSON parameter file')
    
    # Graph settings
    parser.add_argument('--graph_type', type=str, default='cycle',
                        choices=['line', 'cycle', 'grid', 'square', 'triangular',
                                 'honeycomb', 'grid3d', 'sc', 'bcc', 'fcc',
                                 'bethe', 'uv_flower'])
    parser.add_argument('--graph_size', type=int, default=101)
    parser.add_argument('--coin_type', type=str, default='H')
    
    # Bethe
    parser.add_argument('--coordination', type=int, default=3)
    parser.add_argument('--depth', type=int, default=5)
    
    # (u,v)-flower
    parser.add_argument('--u', type=int, default=2)
    parser.add_argument('--v', type=int, default=3)
    parser.add_argument('--generation', type=int, default=5)
    
    # Walk settings
    parser.add_argument('--walk_type', type=str, default='QW',
                        choices=['RW', 'QW', 'QRW_M', 'QRW_A'])
    parser.add_argument('--time_length', type=int, default=1000)
    
    # QRW_M
    parser.add_argument('--n_qw', type=int, default=1)
    parser.add_argument('--n_rw', type=int, default=1)
    
    # QRW_A
    parser.add_argument('--alpha', type=float, default=0.5)
    
    # Other settings
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--job_id', type=str, default=None)
    
    # Output settings
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Probability snapshot interval (MSD/TVD are saved at every step)')
    
    return parser.parse_args()


def load_params_from_file(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def get_output_filename(args):
    name_parts = [args.graph_type]
    
    if args.graph_type == 'bethe':
        name_parts.append(f"z{args.coordination}_d{args.depth}")
    elif args.graph_type == 'uv_flower':
        name_parts.append(f"u{args.u}_v{args.v}_m{args.generation}")
    else:
        name_parts.append(f"N{args.graph_size}")
    
    name_parts.extend([args.walk_type, f"T{args.time_length}"])
    
    if args.walk_type == 'QRW_M':
        name_parts.append(f"qw{args.n_qw}_rw{args.n_rw}")
    elif args.walk_type == 'QRW_A':
        name_parts.append(f"a{args.alpha:.2f}")
    
    name_parts.append(f"s{args.seed}")
    
    if args.job_id:
        name_parts.append(f"j{args.job_id}")
    
    return "_".join(name_parts) + ".npz"


def get_graph_info_str(args):
    if args.graph_type == 'bethe':
        return f"bethe (z={args.coordination}, depth={args.depth})"
    elif args.graph_type == 'uv_flower':
        return f"uv_flower (u={args.u}, v={args.v}, gen={args.generation})"
    elif args.graph_type in ['grid3d', 'sc']:
        return f"{args.graph_type} (N={args.graph_size}, nodes={args.graph_size**3})"
    elif args.graph_type in ['grid', 'square', 'triangular', 'honeycomb']:
        return f"{args.graph_type} (N={args.graph_size}, nodes~{args.graph_size**2})"
    elif args.graph_type == 'bcc':
        return f"bcc (N={args.graph_size}, nodes={2*args.graph_size**3})"
    else:
        return f"{args.graph_type} (N={args.graph_size})"


def main():
    args = parse_args()
    
    if args.param_file and os.path.exists(args.param_file):
        params = load_params_from_file(args.param_file)
        for key, value in params.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("QRW Simulation")
    print("=" * 60)
    print(f"Graph: {get_graph_info_str(args)}")
    print(f"Coin: {args.coin_type}")
    print(f"Walk: {args.walk_type}")
    print(f"Time: {args.time_length} steps")
    if args.walk_type == 'QRW_M':
        print(f"QRW_M params: n_qw={args.n_qw}, n_rw={args.n_rw}")
    elif args.walk_type == 'QRW_A':
        print(f"QRW_A alpha: {args.alpha}")
    print(f"Seed: {args.seed}")
    print(f"Save interval: {args.save_interval}")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. Create the graph
    print("\n[1/3] Creating graph...")
    graph_kwargs = {
        'graph_size': args.graph_size,
        'coordination': args.coordination,
        'depth': args.depth,
        'u': args.u,
        'v': args.v,
        'generation': args.generation,
    }
    graph_data = create_graph(args.graph_type, args.coin_type, **graph_kwargs)
    print(f"      Vertices: {graph_data['num_vertices']}")
    
    # 2. Run the walk
    print(f"\n[2/3] Running {args.walk_type} simulation...")
    walk_kwargs = {
        'alpha': args.alpha,
        'n_qw': args.n_qw,
        'n_rw': args.n_rw,
    }
    result = run_walk(graph_data, args.walk_type, args.time_length,
                      save_interval=args.save_interval, **walk_kwargs)
    
    elapsed_time = time.time() - start_time
    print(f"      Completed in {elapsed_time:.2f}s")
    
    # 3. Save the results
    print("\n[3/3] Saving results...")
    output_filename = get_output_filename(args)
    output_path = os.path.join(args.output_dir, output_filename)
    
    save_data = {
        # Parameters
        'graph_type': args.graph_type,
        'graph_size': args.graph_size,
        'coin_type': args.coin_type,
        'walk_type': args.walk_type,
        'time_length': args.time_length,
        'n_qw': args.n_qw,
        'n_rw': args.n_rw,
        'alpha': args.alpha,
        'seed': args.seed,
        'coordination': args.coordination,
        'depth': args.depth,
        'u': args.u,
        'v': args.v,
        'generation': args.generation,
        
        # Values recorded at every step
        'msd_list': result['msd_list'],
        'tvd_list': result['tvd_list'],
        
        # Final values
        'p_final': result['p_final'],
        'p_avg_final': result['p_avg_final'],
        
        # Snapshots
        'p_snapshots': result['p_snapshots'],
        'snapshot_times': result['snapshot_times'],
        'save_interval': args.save_interval,
        
        # Squared distances for post-processing
        'dist_sq': graph_data['dist_sq'],
        
        # Metadata
        'elapsed_time': elapsed_time,
        'num_vertices': graph_data['num_vertices'],
    }
    
    np.savez_compressed(output_path, **save_data)
    
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"      Saved to: {output_path}")
    print(f"      File size: {file_size_mb:.2f} MB")
    print(f"      Snapshots: {len(result['snapshot_times'])}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Final TVD: {result['tvd_list'][-1]:.6f}")
    print(f"Min TVD:   {result['tvd_list'].min():.6f}")
    print(f"Final MSD: {result['msd_list'][-1]:.4f}")
    print(f"Output:    {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
