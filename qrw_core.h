#ifndef QRW_CORE_H
#define QRW_CORE_H

#include <complex.h>

/*
 * QRW Simulation Core (C Library)
 * 
 * Core optimizations:
 *   - Shift S is a permutation -> O(n^2) index remapping
 *   - Coin C is block diagonal -> O(n^2 d) instead of O(n^3)
 *   - After an RW step, M is diagonal and is handled as a vector
 *
 * Called from Python through ctypes:
 *   1. Create the graph with hiperwalk and extract the permutation and coin blocks
 *   2. Run the simulation loop with the C library
 *   3. Return MSD, TVD, and snapshots to Python
 */

/* ============================================================
 * Graph data structure
 * ============================================================ */
typedef struct {
    int n_arcs;          /* Hilbert space dimension = total directed arcs */
    int n_verts;         /* number of vertices */

    int *shift_perm;     /* S permutation: perm[a] = b means S|a> = |b> */
    int *shift_inv;      /* inverse: inv[b] = a */

    int *vert_start;     /* arc index offset for vertex v (size n_verts+1) */
    int *vert_degree;    /* degree(v) (size n_verts) */

    /* coin blocks: concatenated d_v × d_v matrices in row-major
     * block for vertex v starts at coin_data + coin_offset[v]
     * block size = vert_degree[v]² */
    double complex *coin_data;
    int *coin_offset;    /* size n_verts */

    double *dist_sq;     /* distance² from center for each vertex (size n_verts) */
    int *arc_to_vert;    /* which vertex each arc belongs to (size n_arcs) */
} QRWGraph;

/* ============================================================
 * Simulation result structure
 * ============================================================ */
typedef struct {
    double *msd_list;       /* size time_length+1 */
    double *tvd_list;       /* size time_length */
    double *p_final;        /* size n_verts */
    double *p_avg_final;    /* size n_verts */

    /* snapshots */
    double *p_snapshots;    /* flattened: n_snapshots × n_verts */
    int *snapshot_times;    /* size n_snapshots */
    int n_snapshots;
} QRWResult;

/* ============================================================
 * Exported simulation functions
 * ============================================================ */

/* Quantum walk */
QRWResult* qrw_quantum_walk(
    const QRWGraph *g, int time_length, int save_interval);

/* Classical random walk */
QRWResult* qrw_random_walk(
    const QRWGraph *g, int time_length, int save_interval);

/* QRW_M: alternate n_qw QW steps and n_rw RW steps */
QRWResult* qrw_walk_m(
    const QRWGraph *g, int time_length, int n_qw, int n_rw, int save_interval);

/* QRW_A: alpha * QW + (1-alpha) * RW */
QRWResult* qrw_walk_a(
    const QRWGraph *g, int time_length, double alpha, int save_interval);

/* Release result memory */
void qrw_free_result(QRWResult *r);

/* ============================================================
 * Initial-state setup
 * ============================================================ */

/* Pass an external initial density matrix (row-major, n_arcs x n_arcs) */
void qrw_set_initial_state(double complex *rho, int n_arcs);

/* Optional progress callback */
typedef void (*progress_callback_t)(int current_step, int total_steps);
void qrw_set_progress_callback(progress_callback_t cb);

#endif /* QRW_CORE_H */
