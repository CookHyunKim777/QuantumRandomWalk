"""
qrw_c_bridge.py — Python ↔ C Bridge for QRW Simulation

Extract graph data from hiperwalk and pass it to the C library.
The interface is compatible with run_qrw.py and qrw_utils.py.

Usage:
    from qrw_c_bridge import CSimulator
    
    # Create a graph with qrw_utils
    graph_data = create_graph('cycle', 'H', graph_size=400)
    
    # Run the walk with the C simulator
    sim = CSimulator('libqrw.so')
    result = sim.run_walk(graph_data, 'QW', time_length=160000, save_interval=100)
"""

import ctypes
import numpy as np
import os
import sys
from ctypes import (
    c_int, c_double, c_void_p, POINTER, Structure, CFUNCTYPE,
    cast, byref
)

# ============================================================
# Mirror the C structures
# ============================================================

class CQRWGraph(Structure):
    _fields_ = [
        ("n_arcs",      c_int),
        ("n_verts",     c_int),
        ("shift_perm",  POINTER(c_int)),
        ("shift_inv",   POINTER(c_int)),
        ("vert_start",  POINTER(c_int)),
        ("vert_degree", POINTER(c_int)),
        ("coin_data",   c_void_p),      # double complex*
        ("coin_offset", POINTER(c_int)),
        ("dist_sq",     POINTER(c_double)),
        ("arc_to_vert", POINTER(c_int)),
    ]

class CQRWResult(Structure):
    _fields_ = [
        ("msd_list",      POINTER(c_double)),
        ("tvd_list",      POINTER(c_double)),
        ("p_final",       POINTER(c_double)),
        ("p_avg_final",   POINTER(c_double)),
        ("p_snapshots",   POINTER(c_double)),
        ("snapshot_times", POINTER(c_int)),
        ("n_snapshots",   c_int),
    ]


PROGRESS_CB = CFUNCTYPE(None, c_int, c_int)


# ============================================================
# Extract arrays from hiperwalk for the C backend
# ============================================================

def extract_shift_permutation(shift_matrix):
    """
    Extract the permutation array from the shift matrix S.
    S[perm[a], a] = 1  →  perm[a] = argmax(S[:, a])
    """
    S = shift_matrix
    if hasattr(S, 'toarray'):
        S = S.toarray()
    n = S.shape[0]
    perm = np.zeros(n, dtype=np.int32)
    for a in range(n):
        col = S[:, a]
        perm[a] = np.argmax(np.abs(col))
    return perm


def extract_coin_blocks(coin_matrix, qw_obj):
    """
    Extract the block-diagonal structure from the coin matrix C.
    Returns:
        vert_start: arc index offset per vertex
        vert_degree: degree per vertex
        coin_data: concatenated coin blocks (flattened, complex128)
        coin_offset: offset into coin_data for each vertex
        arc_to_vert: vertex index for each arc
    """
    C = coin_matrix
    if hasattr(C, 'toarray'):
        C = C.toarray()
    
    graph = qw_obj._graph
    n_verts = graph.number_of_vertices()
    
    vert_start = np.zeros(n_verts + 1, dtype=np.int32)
    vert_degree = np.zeros(n_verts, dtype=np.int32)
    
    offset = 0
    for v in range(n_verts):
        d = graph.degree(v)
        vert_start[v] = offset
        vert_degree[v] = d
        offset += d
    vert_start[n_verts] = offset
    n_arcs = offset
    
    # arc → vertex mapping
    arc_to_vert = np.zeros(n_arcs, dtype=np.int32)
    for v in range(n_verts):
        s = vert_start[v]
        d = vert_degree[v]
        arc_to_vert[s:s+d] = v
    
    # Extract the coin blocks
    coin_blocks = []
    coin_offset = np.zeros(n_verts, dtype=np.int32)
    total_size = 0
    
    for v in range(n_verts):
        s = vert_start[v]
        d = vert_degree[v]
        block = C[s:s+d, s:s+d].copy()
        coin_offset[v] = total_size
        coin_blocks.append(block.flatten())
        total_size += d * d
    
    coin_data = np.concatenate(coin_blocks).astype(np.complex128)
    
    return vert_start, vert_degree, coin_data, coin_offset, arc_to_vert


# ============================================================
# CSimulator class
# ============================================================

class CSimulator:
    """QRW simulator backed by the C library."""
    
    def __init__(self, lib_path=None):
        if lib_path is None:
            # Search in the directory containing this module
            here = os.path.dirname(os.path.abspath(__file__))
            for name in ['libqrw.so', 'libqrw.dylib']:
                p = os.path.join(here, name)
                if os.path.exists(p):
                    lib_path = p
                    break
        
        if lib_path is None or not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"C library not found. Compile first:\n"
                f"  gcc -O3 -march=native -shared -fPIC -o libqrw.so qrw_core.c -lm"
            )
        
        self.lib = ctypes.CDLL(lib_path)
        self._setup_signatures()
        self._np_refs = []  # prevent GC
    
    def _setup_signatures(self):
        lib = self.lib
        
        # qrw_set_initial_state(double complex *rho, int n)
        lib.qrw_set_initial_state.argtypes = [c_void_p, c_int]
        lib.qrw_set_initial_state.restype = None
        
        # qrw_set_progress_callback
        lib.qrw_set_progress_callback.argtypes = [PROGRESS_CB]
        lib.qrw_set_progress_callback.restype = None
        
        # quantum_walk
        lib.qrw_quantum_walk.argtypes = [POINTER(CQRWGraph), c_int, c_int]
        lib.qrw_quantum_walk.restype = POINTER(CQRWResult)
        
        # random_walk
        lib.qrw_random_walk.argtypes = [POINTER(CQRWGraph), c_int, c_int]
        lib.qrw_random_walk.restype = POINTER(CQRWResult)
        
        # walk_m
        lib.qrw_walk_m.argtypes = [POINTER(CQRWGraph), c_int, c_int, c_int, c_int]
        lib.qrw_walk_m.restype = POINTER(CQRWResult)
        
        # walk_a
        lib.qrw_walk_a.argtypes = [POINTER(CQRWGraph), c_int, c_double, c_int]
        lib.qrw_walk_a.restype = POINTER(CQRWResult)
        
        # free
        lib.qrw_free_result.argtypes = [POINTER(CQRWResult)]
        lib.qrw_free_result.restype = None
    
    def _build_c_graph(self, graph_data):
        """Convert a Python graph_data dictionary to a C QRWGraph structure."""
        qw = graph_data['qw']
        
        # Extract matrices
        shift_matrix = graph_data['shift']
        coin_matrix = graph_data['coin']
        
        perm = extract_shift_permutation(shift_matrix)
        n_arcs = len(perm)
        
        # inverse permutation
        inv = np.zeros(n_arcs, dtype=np.int32)
        for a in range(n_arcs):
            inv[perm[a]] = a
        
        vert_start, vert_degree, coin_data, coin_offset, arc_to_vert = \
            extract_coin_blocks(coin_matrix, qw)
        n_verts = graph_data['num_vertices']
        
        dist_sq = graph_data['dist_sq'].astype(np.float64)
        
        # Convert NumPy arrays to ctypes pointers and retain references to prevent GC
        refs = []
        
        def to_ptr(arr, ctype):
            arr = np.ascontiguousarray(arr)
            refs.append(arr)
            return arr.ctypes.data_as(POINTER(ctype))
        
        cg = CQRWGraph()
        cg.n_arcs = n_arcs
        cg.n_verts = n_verts
        cg.shift_perm  = to_ptr(perm, c_int)
        cg.shift_inv   = to_ptr(inv, c_int)
        cg.vert_start  = to_ptr(vert_start[:n_verts], c_int)  # The C core needs n_verts entries
        cg.vert_degree = to_ptr(vert_degree, c_int)
        
        # complex128 → void*
        coin_data_c = np.ascontiguousarray(coin_data)
        refs.append(coin_data_c)
        cg.coin_data = coin_data_c.ctypes.data
        
        cg.coin_offset = to_ptr(coin_offset, c_int)
        cg.dist_sq     = to_ptr(dist_sq, c_double)
        cg.arc_to_vert = to_ptr(arc_to_vert, c_int)
        
        self._np_refs = refs
        return cg
    
    def _set_initial_state(self, graph_data):
        """Pass the initial density matrix to the C library."""
        rho = graph_data['initial_state_matrix']
        rho = np.ascontiguousarray(rho, dtype=np.complex128)
        n = rho.shape[0]
        self._np_refs.append(rho)
        self.lib.qrw_set_initial_state(rho.ctypes.data, n)
    
    def _extract_result(self, c_result_ptr, time_length, n_verts):
        """Convert a C QRWResult to a dictionary of NumPy arrays."""
        cr = c_result_ptr.contents
        
        msd = np.ctypeslib.as_array(cr.msd_list, shape=(time_length + 1,)).copy()
        tvd = np.ctypeslib.as_array(cr.tvd_list, shape=(time_length,)).copy()
        p_final = np.ctypeslib.as_array(cr.p_final, shape=(n_verts,)).copy()
        p_avg_final = np.ctypeslib.as_array(cr.p_avg_final, shape=(n_verts,)).copy()
        
        ns = cr.n_snapshots
        snapshot_times = np.ctypeslib.as_array(cr.snapshot_times, shape=(ns,)).copy()
        p_snapshots = np.ctypeslib.as_array(
            cr.p_snapshots, shape=(ns * n_verts,)
        ).reshape(ns, n_verts).copy()
        
        # Release memory allocated by C
        self.lib.qrw_free_result(c_result_ptr)
        
        return {
            'msd_list': msd,
            'tvd_list': tvd,
            'p_final': p_final,
            'p_avg_final': p_avg_final,
            'p_snapshots': p_snapshots,
            'snapshot_times': snapshot_times,
        }
    
    def run_walk(self, graph_data, walk_type, time_length,
                 save_interval=100, **kwargs):
        """
        Use the same interface as qrw_utils.run_walk().
        
        Parameters:
            graph_data: Return value of create_graph()
            walk_type: 'RW', 'QW', 'QRW_M', 'QRW_A'
            time_length: Total number of steps
            save_interval: Snapshot interval
            kwargs: alpha, n_qw, n_rw
        
        Returns:
            dict: msd_list, tvd_list, p_final, p_avg_final, p_snapshots, snapshot_times
        """
        cg = self._build_c_graph(graph_data)
        self._set_initial_state(graph_data)
        
        n_verts = graph_data['num_vertices']
        
        if walk_type == 'QW':
            cr = self.lib.qrw_quantum_walk(byref(cg), time_length, save_interval)
        elif walk_type == 'RW':
            cr = self.lib.qrw_random_walk(byref(cg), time_length, save_interval)
        elif walk_type == 'QRW_M':
            n_qw = kwargs.get('n_qw', 1)
            n_rw = kwargs.get('n_rw', 1)
            cr = self.lib.qrw_walk_m(byref(cg), time_length, n_qw, n_rw, save_interval)
            # actual time = n_cycles * cycle_len
            cycle_len = n_qw + n_rw
            actual_time = (time_length // cycle_len) * cycle_len
            return self._extract_result(cr, actual_time, n_verts)
        elif walk_type == 'QRW_A':
            alpha = kwargs.get('alpha', 0.5)
            cr = self.lib.qrw_walk_a(byref(cg), time_length, alpha, save_interval)
        else:
            raise ValueError(f"Unknown walk_type: {walk_type}")
        
        return self._extract_result(cr, time_length, n_verts)


# ============================================================
# Convenience function: drop-in replacement for run_qrw.py
# ============================================================

def run_walk_c(graph_data, walk_type, time_length,
               save_interval=100, lib_path=None, **kwargs):
    """
    Drop-in replacement for qrw_utils.run_walk()
    Use the C library.
    """
    sim = CSimulator(lib_path)
    return sim.run_walk(graph_data, walk_type, time_length,
                        save_interval=save_interval, **kwargs)


# ============================================================
# Tests and verification
# ============================================================

def verify_against_python(graph_type='cycle', graph_size=21, walk_type='QW',
                          time_length=50, seed=777):
    """
    Compare the C and Python results on a small graph.
    """
    from qrw_utils import create_graph, run_walk
    import time
    
    np.random.seed(seed)
    graph_data = create_graph(graph_type, 'H', graph_size=graph_size)
    
    # Python
    t0 = time.time()
    result_py = run_walk(graph_data, walk_type, time_length, save_interval=10)
    t_py = time.time() - t0
    
    # C
    t0 = time.time()
    result_c = run_walk_c(graph_data, walk_type, time_length, save_interval=10)
    t_c = time.time() - t0
    
    # Compare
    msd_diff = np.max(np.abs(result_py['msd_list'] - result_c['msd_list']))
    tvd_diff = np.max(np.abs(result_py['tvd_list'] - result_c['tvd_list']))
    p_diff   = np.max(np.abs(result_py['p_final'] - result_c['p_final']))
    
    print(f"=== Verification: {graph_type} N={graph_size} {walk_type} T={time_length} ===")
    print(f"Python: {t_py:.3f}s  |  C: {t_c:.3f}s  |  Speedup: {t_py/t_c:.1f}x")
    print(f"Max MSD diff: {msd_diff:.2e}")
    print(f"Max TVD diff: {tvd_diff:.2e}")
    print(f"Max p_final diff: {p_diff:.2e}")
    
    ok = msd_diff < 1e-10 and tvd_diff < 1e-10 and p_diff < 1e-10
    print(f"Result: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    # Verification mode
    if '--verify' in sys.argv:
        for wt in ['QW', 'RW', 'QRW_M', 'QRW_A']:
            verify_against_python(walk_type=wt)
            print()
    else:
        print("Usage: python qrw_c_bridge.py --verify")
        print("  (requires qrw_utils.py, hiperwalk, and compiled libqrw.so)")
