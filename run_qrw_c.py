#!/usr/bin/env python3
"""
QRW simulation with a C-accelerated backend.
The interface matches run_qrw.py; only the simulation loop uses the C library.

Usage:
    python run_qrw_c.py --graph_type cycle --graph_size 400 --walk_type QRW_M --time_length 160000 --save_interval 100
    python run_qrw_c.py --param_file params/qrw_m/params_001.json
    
    # Python fallback when the C library is unavailable
    python run_qrw_c.py --graph_type cycle --graph_size 400 --walk_type QW --time_length 1000 --use_python
"""

import argparse
import json
import os
import sys
import time
import numpy as np

from qrw_utils import create_graph


def parse_args():
    parser = argparse.ArgumentParser(description='QRW Simulation (C-accelerated)')
    
    parser.add_argument('--param_file', type=str, default=None)
    
    # Graph settings
    parser.add_argument('--graph_type', type=str, default='cycle',
                        choices=['line', 'cycle', 'grid', 'square', 'triangular',
                                 'honeycomb', 'grid3d', 'sc', 'bcc', 'fcc',
                                 'bethe', 'uv_flower'])
    parser.add_argument('--graph_size', type=int, default=101)
    parser.add_argument('--coin_type', type=str, default='H')
    parser.add_argument('--coordination', type=int, default=3)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--u', type=int, default=2)
    parser.add_argument('--v', type=int, default=3)
    parser.add_argument('--generation', type=int, default=5)
    
    # Walk settings
    parser.add_argument('--walk_type', type=str, default='QW',
                        choices=['RW', 'QW', 'QRW_M', 'QRW_A'])
    parser.add_argument('--time_length', type=int, default=1000)
    parser.add_argument('--n_qw', type=int, default=1)
    parser.add_argument('--n_rw', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.5)
    
    # Other settings
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--job_id', type=str, default=None)
    parser.add_argument('--save_interval', type=int, default=100)
    
    # Backend selection
    parser.add_argument('--use_python', action='store_true',
                        help='Force Python backend (no C)')
    parser.add_argument('--lib_path', type=str, default=None,
                        help='Path to libqrw.so')
    
    return parser.parse_args()


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


def main():
    args = parse_args()
    
    if args.param_file and os.path.exists(args.param_file):
        with open(args.param_file) as f:
            params = json.load(f)
        for key, value in params.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select the backend
    use_c = not args.use_python
    if use_c:
        try:
            from qrw_c_bridge import CSimulator
            sim = CSimulator(args.lib_path)
            backend = "C"
        except (FileNotFoundError, OSError) as e:
            print(f"[WARN] C library not available: {e}")
            print("[WARN] Falling back to Python backend")
            use_c = False
            backend = "Python (fallback)"
    else:
        backend = "Python"
    
    if not use_c:
        from qrw_utils import run_walk
    
    print("=" * 60)
    print(f"QRW Simulation [{backend}]")
    print("=" * 60)
    print(f"Graph: {args.graph_type} (size={args.graph_size})")
    print(f"Walk:  {args.walk_type}, T={args.time_length}")
    if args.walk_type == 'QRW_M':
        print(f"  n_qw={args.n_qw}, n_rw={args.n_rw}")
    elif args.walk_type == 'QRW_A':
        print(f"  alpha={args.alpha}")
    print(f"Seed: {args.seed}, save_interval: {args.save_interval}")
    print("=" * 60)
    
    # 1. Create the graph with hiperwalk in Python
    t0 = time.time()
    print("\n[1/3] Creating graph...")
    graph_kwargs = {
        'graph_size': args.graph_size,
        'coordination': args.coordination,
        'depth': args.depth,
        'u': args.u, 'v': args.v,
        'generation': args.generation,
    }
    graph_data = create_graph(args.graph_type, args.coin_type, **graph_kwargs)
    t_graph = time.time() - t0
    print(f"      Vertices: {graph_data['num_vertices']} ({t_graph:.2f}s)")
    
    # 2. Run the simulation
    print(f"\n[2/3] Running {args.walk_type} ({backend})...")
    t0 = time.time()
    
    walk_kwargs = {
        'alpha': args.alpha,
        'n_qw': args.n_qw,
        'n_rw': args.n_rw,
    }
    
    if use_c:
        result = sim.run_walk(graph_data, args.walk_type, args.time_length,
                              save_interval=args.save_interval, **walk_kwargs)
    else:
        result = run_walk(graph_data, args.walk_type, args.time_length,
                          save_interval=args.save_interval, **walk_kwargs)
    
    t_walk = time.time() - t0
    print(f"      Completed in {t_walk:.2f}s")
    
    # 3. Save the results
    print("\n[3/3] Saving results...")
    output_filename = get_output_filename(args)
    output_path = os.path.join(args.output_dir, output_filename)
    
    save_data = {
        'graph_type': args.graph_type,
        'graph_size': args.graph_size,
        'coin_type': args.coin_type,
        'walk_type': args.walk_type,
        'time_length': args.time_length,
        'n_qw': args.n_qw, 'n_rw': args.n_rw,
        'alpha': args.alpha,
        'seed': args.seed,
        'coordination': args.coordination,
        'depth': args.depth,
        'u': args.u, 'v': args.v,
        'generation': args.generation,
        
        'msd_list': result['msd_list'],
        'tvd_list': result['tvd_list'],
        'p_final': result['p_final'],
        'p_avg_final': result['p_avg_final'],
        'p_snapshots': result['p_snapshots'],
        'snapshot_times': result['snapshot_times'],
        'save_interval': args.save_interval,
        'dist_sq': graph_data['dist_sq'],
        
        'elapsed_time': t_walk,
        'num_vertices': graph_data['num_vertices'],
        'backend': backend,
    }
    
    np.savez_compressed(output_path, **save_data)
    
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"      Saved to: {output_path}")
    print(f"      File size: {file_size_mb:.2f} MB")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Backend:   {backend}")
    print(f"Walk time: {t_walk:.2f}s")
    print(f"Final TVD: {result['tvd_list'][-1]:.6f}")
    print(f"Final MSD: {result['msd_list'][-1]:.4f}")
    print(f"Output:    {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
