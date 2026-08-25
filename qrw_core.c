/*
 * qrw_core.c — QRW Simulation Core
 *
 * Compile:
 *   gcc -O3 -march=native -shared -fPIC -o libqrw.so qrw_core.c -lm
 *   (macOS: -dynamiclib -o libqrw.dylib)
 *
 * Optimization summary:
 *   S (shift): permutation array → row/col remap O(n²)
 *   C (coin):  block diagonal    → small matmul per block O(n²·d)
 *   A (adj):   block diagonal uniform → vector averaging per block O(n)
 *   After an RW step, M is diagonal, so only a vector is tracked
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include "qrw_core.h"

/* ============================================================
 * Global state
 * ============================================================ */
static double complex *g_initial_rho = NULL;
static int g_initial_n = 0;
static progress_callback_t g_progress_cb = NULL;

void qrw_set_initial_state(double complex *rho, int n) {
    g_initial_rho = rho;
    g_initial_n = n;
}

void qrw_set_progress_callback(progress_callback_t cb) {
    g_progress_cb = cb;
}

/* ============================================================
 * Memory helpers
 * ============================================================ */
static double complex* alloc_matrix(int n) {
    double complex *m = (double complex*)calloc((size_t)n * n, sizeof(double complex));
    if (!m) { fprintf(stderr, "alloc_matrix: OOM (n=%d)\n", n); exit(1); }
    return m;
}

static double* alloc_dvec(int n) {
    double *v = (double*)calloc(n, sizeof(double));
    if (!v) { fprintf(stderr, "alloc_dvec: OOM\n"); exit(1); }
    return v;
}

static int* alloc_ivec(int n) {
    int *v = (int*)calloc(n, sizeof(int));
    if (!v) { fprintf(stderr, "alloc_ivec: OOM\n"); exit(1); }
    return v;
}

/* M[i][j] in row-major */
#define M_IDX(i, j, n) ((size_t)(i) * (n) + (j))

/* ============================================================
 * Probability calculation: diag(M) -> vertex probability
 * p(v) = sum_{arcs a with tail v} Re(M[a,a])
 * ============================================================ */
static void compute_prob(const double complex *M, int n,
                         const QRWGraph *g, double *p_out)
{
    memset(p_out, 0, g->n_verts * sizeof(double));
    for (int a = 0; a < n; a++) {
        int v = g->arc_to_vert[a];
        p_out[v] += creal(M[M_IDX(a, a, n)]);
    }
}

/* diagonal vector → vertex probability */
static void compute_prob_diag(const double complex *diag_vec, int n,
                              const QRWGraph *g, double *p_out)
{
    memset(p_out, 0, g->n_verts * sizeof(double));
    for (int a = 0; a < n; a++) {
        int v = g->arc_to_vert[a];
        p_out[v] += creal(diag_vec[a]);
    }
}

/* MSD = sum_v dist_sq[v] * p[v] */
static double compute_msd(const double *p, const double *dist_sq, int nv) {
    double msd = 0.0;
    for (int v = 0; v < nv; v++) msd += dist_sq[v] * p[v];
    return msd;
}

/* TVD = 0.5 * sum |p_avg[v] - 1/N| */
static double compute_tvd(const double *p_avg, int nv) {
    double pi = 1.0 / nv;
    double tvd = 0.0;
    for (int v = 0; v < nv; v++) tvd += fabs(p_avg[v] - pi);
    return 0.5 * tvd;
}

/* ============================================================
 * Block-diagonal x dense (coin operation)
 *
 * T = C @ M   (left multiply by block-diagonal coin)
 * T[s+i, j] = sum_k C_v[i,k] * M[s+k, j]
 *   where s = vert_start[v], d = vert_degree[v]
 * ============================================================ */
static void coin_left_multiply(double complex *T,
                               const double complex *M,
                               const QRWGraph *g)
{
    const int n = g->n_arcs;
    for (int v = 0; v < g->n_verts; v++) {
        int s = g->vert_start[v];
        int d = g->vert_degree[v];
        const double complex *C_v = g->coin_data + g->coin_offset[v];

        for (int i = 0; i < d; i++) {
            int row = s + i;
            for (int j = 0; j < n; j++) {
                double complex val = 0.0;
                for (int k = 0; k < d; k++) {
                    val += C_v[i * d + k] * M[M_IDX(s + k, j, n)];
                }
                T[M_IDX(row, j, n)] = val;
            }
        }
    }
}

/* T2 = T @ C†   (right multiply by conjugate-transpose of block-diagonal)
 * T2[i, s+j] = sum_k T[i, s+k] * conj(C_v[j, k])
 */
static void coin_right_multiply_dagger(double complex *T2,
                                       const double complex *T,
                                       const QRWGraph *g)
{
    const int n = g->n_arcs;
    for (int v = 0; v < g->n_verts; v++) {
        int s = g->vert_start[v];
        int d = g->vert_degree[v];
        const double complex *C_v = g->coin_data + g->coin_offset[v];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                double complex val = 0.0;
                for (int k = 0; k < d; k++) {
                    val += T[M_IDX(i, s + k, n)] * conj(C_v[j * d + k]);
                }
                T2[M_IDX(i, s + j, n)] = val;
            }
        }
    }
}

/* ============================================================
 * Shift permutation:  Result[i,j] = Src[inv[i], inv[j]]
 * (= S @ Src @ S†)
 * ============================================================ */
static void shift_permute(double complex *dst,
                          const double complex *src,
                          const int *inv, int n)
{
    for (int i = 0; i < n; i++) {
        int si = inv[i];
        for (int j = 0; j < n; j++) {
            dst[M_IDX(i, j, n)] = src[M_IDX(si, inv[j], n)];
        }
    }
}

/* ============================================================
 * QW step:  M_new = S @ C @ M @ C† @ S†
 * Two work buffers are required (T1 and T2)
 * ============================================================ */
static void qw_step(double complex *M,
                    double complex *T1, double complex *T2,
                    const QRWGraph *g)
{
    coin_left_multiply(T1, M, g);        /* T1 = C @ M        */
    coin_right_multiply_dagger(T2, T1, g); /* T2 = T1 @ C†      */
    shift_permute(M, T2, g->shift_inv, g->n_arcs); /* M = S @ T2 @ S† */
}

/* ============================================================
 * RW step:  diag → A·diag → S·(A·diag)
 * A is block-diagonal and uniform: output in block v = (1/d) * sum(input block v)
 * S is a permutation
 *
 * Input: diagonal entries of M (or an existing diagonal vector)
 * Output: update M as a diagonal matrix
 * ============================================================ */
static void rw_step_from_diag(double complex *diag_out,
                              const double complex *diag_in,
                              const QRWGraph *g)
{
    const int n = g->n_arcs;
    /* Temporary buffer: A @ diag_in */
    double complex *tmp = (double complex*)malloc(n * sizeof(double complex));

    /* A @ diag_in: average within each block v */
    for (int v = 0; v < g->n_verts; v++) {
        int s = g->vert_start[v];
        int d = g->vert_degree[v];
        double complex avg = 0.0;
        for (int k = 0; k < d; k++) avg += diag_in[s + k];
        avg /= d;
        for (int k = 0; k < d; k++) tmp[s + k] = avg;
    }

    /* S @ tmp: permute */
    for (int a = 0; a < n; a++) {
        diag_out[g->shift_perm[a]] = tmp[a];
    }

    free(tmp);
}

/* RW step on full density matrix: extract diag, apply, make diagonal */
static void rw_step_full(double complex *M, const QRWGraph *g)
{
    const int n = g->n_arcs;
    double complex *d_in  = (double complex*)malloc(n * sizeof(double complex));
    double complex *d_out = (double complex*)malloc(n * sizeof(double complex));

    /* extract diagonal */
    for (int a = 0; a < n; a++) d_in[a] = M[M_IDX(a, a, n)];

    rw_step_from_diag(d_out, d_in, g);

    /* M = diag(d_out) */
    memset(M, 0, (size_t)n * n * sizeof(double complex));
    for (int a = 0; a < n; a++) M[M_IDX(a, a, n)] = d_out[a];

    free(d_in);
    free(d_out);
}

/* ============================================================
 * Result allocation and deallocation
 * ============================================================ */
static QRWResult* alloc_result(int time_length, int n_verts,
                               int max_snapshots)
{
    QRWResult *r = (QRWResult*)calloc(1, sizeof(QRWResult));
    r->msd_list     = alloc_dvec(time_length + 1);
    r->tvd_list     = alloc_dvec(time_length);
    r->p_final      = alloc_dvec(n_verts);
    r->p_avg_final  = alloc_dvec(n_verts);
    r->p_snapshots  = alloc_dvec(max_snapshots * n_verts);
    r->snapshot_times = alloc_ivec(max_snapshots);
    r->n_snapshots  = 0;
    return r;
}

void qrw_free_result(QRWResult *r) {
    if (!r) return;
    free(r->msd_list);
    free(r->tvd_list);
    free(r->p_final);
    free(r->p_avg_final);
    free(r->p_snapshots);
    free(r->snapshot_times);
    free(r);
}

static void save_snapshot(QRWResult *r, const double *p, int t, int nv) {
    int idx = r->n_snapshots;
    memcpy(r->p_snapshots + (size_t)idx * nv, p, nv * sizeof(double));
    r->snapshot_times[idx] = t;
    r->n_snapshots++;
}

/* ============================================================
 * Load the initial density matrix
 * ============================================================ */
static void load_initial_rho(double complex *M, int n) {
    if (g_initial_rho && g_initial_n == n) {
        memcpy(M, g_initial_rho, (size_t)n * n * sizeof(double complex));
    } else {
        fprintf(stderr, "Warning: no initial state set, using |0><0|\n");
        memset(M, 0, (size_t)n * n * sizeof(double complex));
        M[0] = 1.0;
    }
}

/* ============================================================
 * Quantum Walk
 * ============================================================ */
QRWResult* qrw_quantum_walk(const QRWGraph *g, int time_length,
                            int save_interval)
{
    const int n  = g->n_arcs;
    const int nv = g->n_verts;
    int max_snap = time_length / save_interval + 3;

    double complex *M  = alloc_matrix(n);
    double complex *T1 = alloc_matrix(n);
    double complex *T2 = alloc_matrix(n);
    double *p   = alloc_dvec(nv);
    double *pavg = alloc_dvec(nv);

    QRWResult *r = alloc_result(time_length, nv, max_snap);

    load_initial_rho(M, n);

    /* t=0 */
    compute_prob(M, n, g, p);
    r->msd_list[0] = compute_msd(p, g->dist_sq, nv);
    save_snapshot(r, p, 0, nv);

    for (int t = 1; t <= time_length; t++) {
        qw_step(M, T1, T2, g);

        compute_prob(M, n, g, p);
        r->msd_list[t] = compute_msd(p, g->dist_sq, nv);

        /* running average */
        for (int v = 0; v < nv; v++)
            pavg[v] = (pavg[v] * (t - 1) + p[v]) / t;
        r->tvd_list[t - 1] = compute_tvd(pavg, nv);

        if (t % save_interval == 0)
            save_snapshot(r, p, t, nv);

        if (g_progress_cb && t % 1000 == 0)
            g_progress_cb(t, time_length);
    }

    /* Ensure that the final snapshot is saved */
    if (r->snapshot_times[r->n_snapshots - 1] != time_length)
        save_snapshot(r, p, time_length, nv);

    memcpy(r->p_final, p, nv * sizeof(double));
    memcpy(r->p_avg_final, pavg, nv * sizeof(double));

    free(M); free(T1); free(T2); free(p); free(pavg);
    return r;
}

/* ============================================================
 * Random Walk
 * ============================================================ */
QRWResult* qrw_random_walk(const QRWGraph *g, int time_length,
                           int save_interval)
{
    const int n  = g->n_arcs;
    const int nv = g->n_verts;
    int max_snap = time_length / save_interval + 3;

    /* RW: track only the diagonal */
    double complex *dvec     = (double complex*)calloc(n, sizeof(double complex));
    double complex *dvec_new = (double complex*)calloc(n, sizeof(double complex));
    double *p    = alloc_dvec(nv);
    double *pavg = alloc_dvec(nv);

    QRWResult *r = alloc_result(time_length, nv, max_snap);

    /* Extract the diagonal of the initial state */
    if (g_initial_rho && g_initial_n == n) {
        for (int a = 0; a < n; a++)
            dvec[a] = g_initial_rho[M_IDX(a, a, n)];
    } else {
        dvec[0] = 1.0;
    }

    /* t=0 */
    compute_prob_diag(dvec, n, g, p);
    r->msd_list[0] = compute_msd(p, g->dist_sq, nv);
    save_snapshot(r, p, 0, nv);

    for (int t = 1; t <= time_length; t++) {
        rw_step_from_diag(dvec_new, dvec, g);

        /* swap */
        double complex *tmp = dvec; dvec = dvec_new; dvec_new = tmp;

        compute_prob_diag(dvec, n, g, p);
        r->msd_list[t] = compute_msd(p, g->dist_sq, nv);

        for (int v = 0; v < nv; v++)
            pavg[v] = (pavg[v] * (t - 1) + p[v]) / t;
        r->tvd_list[t - 1] = compute_tvd(pavg, nv);

        if (t % save_interval == 0)
            save_snapshot(r, p, t, nv);

        if (g_progress_cb && t % 1000 == 0)
            g_progress_cb(t, time_length);
    }

    if (r->snapshot_times[r->n_snapshots - 1] != time_length)
        save_snapshot(r, p, time_length, nv);

    memcpy(r->p_final, p, nv * sizeof(double));
    memcpy(r->p_avg_final, pavg, nv * sizeof(double));

    free(dvec); free(dvec_new); free(p); free(pavg);
    return r;
}

/* ============================================================
 * QRW_M: alternate n_qw QW steps and n_rw RW steps
 * ============================================================ */
QRWResult* qrw_walk_m(const QRWGraph *g, int time_length,
                      int n_qw, int n_rw, int save_interval)
{
    const int n  = g->n_arcs;
    const int nv = g->n_verts;
    int max_snap = time_length / save_interval + 3;
    int cycle_len = n_qw + n_rw;
    int n_cycles  = time_length / cycle_len;

    double complex *M  = alloc_matrix(n);
    double complex *T1 = alloc_matrix(n);
    double complex *T2 = alloc_matrix(n);
    double *p    = alloc_dvec(nv);
    double *pavg = alloc_dvec(nv);

    QRWResult *r = alloc_result(time_length, nv, max_snap);

    load_initial_rho(M, n);

    /* t=0 */
    compute_prob(M, n, g, p);
    r->msd_list[0] = compute_msd(p, g->dist_sq, nv);
    save_snapshot(r, p, 0, nv);

    int t = 0;
    for (int cyc = 0; cyc < n_cycles; cyc++) {
        /* QW steps */
        for (int q = 0; q < n_qw; q++) {
            t++;
            qw_step(M, T1, T2, g);

            compute_prob(M, n, g, p);
            r->msd_list[t] = compute_msd(p, g->dist_sq, nv);
            for (int v = 0; v < nv; v++)
                pavg[v] = (pavg[v] * (t - 1) + p[v]) / t;
            r->tvd_list[t - 1] = compute_tvd(pavg, nv);

            if (t % save_interval == 0)
                save_snapshot(r, p, t, nv);
        }

        /* RW steps */
        for (int rr = 0; rr < n_rw; rr++) {
            t++;
            rw_step_full(M, g);

            compute_prob(M, n, g, p);
            r->msd_list[t] = compute_msd(p, g->dist_sq, nv);
            for (int v = 0; v < nv; v++)
                pavg[v] = (pavg[v] * (t - 1) + p[v]) / t;
            r->tvd_list[t - 1] = compute_tvd(pavg, nv);

            if (t % save_interval == 0)
                save_snapshot(r, p, t, nv);
        }

        if (g_progress_cb && t % 1000 < cycle_len)
            g_progress_cb(t, time_length);
    }

    if (t > 0 && r->snapshot_times[r->n_snapshots - 1] != t)
        save_snapshot(r, p, t, nv);

    memcpy(r->p_final, p, nv * sizeof(double));
    memcpy(r->p_avg_final, pavg, nv * sizeof(double));

    free(M); free(T1); free(T2); free(p); free(pavg);
    return r;
}

/* ============================================================
 * QRW_A:  M_new = alpha * QW(M) + (1-alpha) * RW(M)
 * ============================================================ */
QRWResult* qrw_walk_a(const QRWGraph *g, int time_length,
                      double alpha, int save_interval)
{
    const int n  = g->n_arcs;
    const int nv = g->n_verts;
    int max_snap = time_length / save_interval + 3;

    double complex *M     = alloc_matrix(n);
    double complex *M_qw  = alloc_matrix(n);
    double complex *M_rw  = alloc_matrix(n);
    double complex *T1    = alloc_matrix(n);
    double complex *T2    = alloc_matrix(n);
    double *p    = alloc_dvec(nv);
    double *pavg = alloc_dvec(nv);

    QRWResult *r = alloc_result(time_length, nv, max_snap);

    load_initial_rho(M, n);

    /* t=0 */
    compute_prob(M, n, g, p);
    r->msd_list[0] = compute_msd(p, g->dist_sq, nv);
    save_snapshot(r, p, 0, nv);

    size_t mat_bytes = (size_t)n * n * sizeof(double complex);

    for (int t = 1; t <= time_length; t++) {
        /* QW step → M_qw */
        memcpy(M_qw, M, mat_bytes);
        qw_step(M_qw, T1, T2, g);

        /* RW step -> M_rw (copy first so that M remains unchanged) */
        memcpy(M_rw, M, mat_bytes);
        rw_step_full(M_rw, g);

        /* M = alpha * M_qw + (1-alpha) * M_rw */
        double a1 = alpha, a2 = 1.0 - alpha;
        for (size_t idx = 0; idx < (size_t)n * n; idx++)
            M[idx] = a1 * M_qw[idx] + a2 * M_rw[idx];

        compute_prob(M, n, g, p);
        r->msd_list[t] = compute_msd(p, g->dist_sq, nv);

        for (int v = 0; v < nv; v++)
            pavg[v] = (pavg[v] * (t - 1) + p[v]) / t;
        r->tvd_list[t - 1] = compute_tvd(pavg, nv);

        if (t % save_interval == 0)
            save_snapshot(r, p, t, nv);

        if (g_progress_cb && t % 1000 == 0)
            g_progress_cb(t, time_length);
    }

    if (r->snapshot_times[r->n_snapshots - 1] != time_length)
        save_snapshot(r, p, time_length, nv);

    memcpy(r->p_final, p, nv * sizeof(double));
    memcpy(r->p_avg_final, pavg, nv * sizeof(double));

    free(M); free(M_qw); free(M_rw); free(T1); free(T2);
    free(p); free(pavg);
    return r;
}
