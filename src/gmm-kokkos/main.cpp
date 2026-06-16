/*
 * Gaussian Mixture Model EM clustering - Kokkos port
 *
 * Ported from gmm-omp (CUDA/OpenMP EM clustering by Andrew Pangborn, RIT).
 * All source files (main.cpp, gaussian_kernel.cpp, cluster.cpp, readData.cpp)
 * consolidated here.
 */

#include <Kokkos_Core.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// ============================================================
// gaussian.h definitions
// ============================================================
#define PI  3.1415926535897931f
#define COVARIANCE_DYNAMIC_RANGE 1E6f
#define UNIFORM_SEED 0
#define TRUNCATE 1
#define NUM_BLOCKS 24
#define NUM_THREADS_ESTEP 256
#define NUM_THREADS_MSTEP 256
#define NUM_DIMENSIONS 24
#define NUM_CLUSTERS_PER_BLOCK 6
#define DEVICE 0
#define DIAG_ONLY 0
#define MAX_ITERS 200
#define MIN_ITERS 1
#define ENABLE_DEBUG 0
#define ENABLE_PRINT 1
#define ENABLE_OUTPUT 1

#if ENABLE_PRINT
#define PRINT(fmt, ...) printf(fmt, ## __VA_ARGS__)
#else
#define PRINT(fmt, ...)
#endif

#if ENABLE_DEBUG
#define DEBUG(fmt, ...) printf(fmt, ## __VA_ARGS__)
#else
#define DEBUG(fmt, ...)
#endif

typedef struct {
    float *N;
    float *pi;
    float *constant;
    float *avgvar;
    float *means;
    float *R;
    float *Rinv;
    float *memberships;
} clusters_t;

// ============================================================
// Kokkos view aliases
// ============================================================
using FView = Kokkos::View<float*>;
using ExecSpace = Kokkos::DefaultExecutionSpace;
using ScratchSpace = ExecSpace::scratch_memory_space;
using ScratchFView = Kokkos::View<float*, ScratchSpace, Kokkos::MemoryUnmanaged>;

// ============================================================
// invert (from gaussian_kernel.cpp) — runs on device
// ============================================================
KOKKOS_INLINE_FUNCTION
void invert_device(float *data, int actualsize, float *log_det)
{
    const int maxsize = actualsize;
    const int n       = actualsize;
    *log_det = 0.0f;

    if (actualsize == 1) {
        *log_det = logf(data[0]);
        data[0]  = 1.f / data[0];
        return;
    }

    for (int i = 1; i < actualsize; i++) data[i] /= data[0];
    for (int i = 1; i < actualsize; i++) {
        for (int j = i; j < actualsize; j++) {
            float sum = 0.0f;
            for (int k = 0; k < i; k++)
                sum += data[j * maxsize + k] * data[k * maxsize + i];
            data[j * maxsize + i] -= sum;
        }
        if (i == actualsize - 1) continue;
        for (int j = i + 1; j < actualsize; j++) {
            float sum = 0.0f;
            for (int k = 0; k < i; k++)
                sum += data[i * maxsize + k] * data[k * maxsize + j];
            data[i * maxsize + j] = (data[i * maxsize + j] - sum) / data[i * maxsize + i];
        }
    }

    for (int i = 0; i < actualsize; i++)
        *log_det += logf(fabsf(data[i * n + i]));

    for (int i = 0; i < actualsize; i++)
        for (int j = i; j < actualsize; j++) {
            float x = 1.f;
            if (i != j) {
                x = 0.0f;
                for (int k = i; k < j; k++)
                    x -= data[j * maxsize + k] * data[k * maxsize + i];
            }
            data[j * maxsize + i] = x / data[j * maxsize + j];
        }

    for (int i = 0; i < actualsize; i++)
        for (int j = i; j < actualsize; j++) {
            if (i == j) continue;
            float sum = 0.0f;
            for (int k = i; k < j; k++)
                sum += data[k * maxsize + j] * ((i == k) ? 1.f : data[i * maxsize + k]);
            data[i * maxsize + j] = -sum;
        }

    for (int i = 0; i < actualsize; i++)
        for (int j = 0; j < actualsize; j++) {
            float sum = 0.0f;
            for (int k = ((i > j) ? i : j); k < actualsize; k++)
                sum += ((j == k) ? 1.f : data[j * maxsize + k]) * data[k * maxsize + i];
            data[j * maxsize + i] = sum;
        }
}

// ============================================================
// readData.cpp
// ============================================================
using namespace std;

static float *readBIN(const char *f, int *ndims, int *nevents)
{
    FILE *fin = fopen(f, "rb");
    fread(nevents, 4, 1, fin);
    fread(ndims,   4, 1, fin);
    printf("Number of elements removed for memory alignment: %d\n", *nevents % (16 * 2));
    *nevents -= *nevents % (16 * 2);
    int num_elements = (*ndims) * (*nevents);
    printf("Number of rows: %d\nNumber of cols: %d\n", *nevents, *ndims);
    float *data = (float *)malloc(sizeof(float) * num_elements);
    fread(data, sizeof(float), num_elements, fin);
    fclose(fin);
    return data;
}

static float *readCSV(const char *f, int *ndims, int *nevents)
{
    string line1;
    ifstream file(f);
    vector<string> lines;
    int num_dims = 0;
    char *temp;
    float *data;

    if (file.is_open()) {
        while (!file.eof()) {
            getline(file, line1);
            if (!line1.empty()) lines.push_back(line1);
        }
        file.close();
    } else {
        cout << "Unable to read the file " << f << endl;
        return nullptr;
    }

    if (lines.empty()) return nullptr;
    line1 = lines[0];
    temp  = strtok((char *)line1.c_str(), ",");
    while (temp) { num_dims++; temp = strtok(nullptr, ","); }

    lines.erase(lines.begin());
    int num_events = (int)lines.size();
#if TRUNCATE
    printf("Number of events removed for alignment: %d\n", num_events % (16 * 2));
    num_events -= num_events % (16 * 2);
#endif
    data = (float *)malloc(sizeof(float) * num_dims * num_events);
    if (!data) { printf("Cannot allocate FCS data.\n"); return nullptr; }

    for (int i = 0; i < num_events; i++) {
        temp = strtok((char *)lines[i].c_str(), ",");
        for (int j = 0; j < num_dims; j++) {
            if (!temp) { free(data); return nullptr; }
            data[i * num_dims + j] = atof(temp);
            temp = strtok(nullptr, ",");
        }
    }
    *ndims   = num_dims;
    *nevents = num_events;
    return data;
}

static float *readData(const char *f, int *ndims, int *nevents)
{
    int length = strlen(f);
    printf("File Extension: %s\n", f + length - 3);
    if (strcmp(f + length - 3, "bin") == 0)
        return readBIN(f, ndims, nevents);
    return readCSV(f, ndims, nevents);
}

// ============================================================
// Cluster helpers (from cluster.cpp)
// ============================================================
static void invert_cpu(float *data, int actualsize, float *log_determinant)
{
    int maxsize = actualsize, n = actualsize;
    *log_determinant = 0.0;
    if (actualsize == 1) { *log_determinant = logf(data[0]); data[0] = 1.0f / data[0]; return; }
    for (int i = 1; i < actualsize; i++) data[i] /= data[0];
    for (int i = 1; i < actualsize; i++) {
        for (int j = i; j < actualsize; j++) {
            float sum = 0.0;
            for (int k = 0; k < i; k++) sum += data[j * maxsize + k] * data[k * maxsize + i];
            data[j * maxsize + i] -= sum;
        }
        if (i == actualsize - 1) continue;
        for (int j = i + 1; j < actualsize; j++) {
            float sum = 0.0;
            for (int k = 0; k < i; k++) sum += data[i * maxsize + k] * data[k * maxsize + j];
            data[i * maxsize + j] = (data[i * maxsize + j] - sum) / data[i * maxsize + i];
        }
    }
    for (int i = 0; i < actualsize; i++)
        *log_determinant += ::log10(fabsf(data[i * n + i]));
    for (int i = 0; i < actualsize; i++)
        for (int j = i; j < actualsize; j++) {
            float x = 1.0;
            if (i != j) { x = 0.0; for (int k = i; k < j; k++) x -= data[j*maxsize+k]*data[k*maxsize+i]; }
            data[j * maxsize + i] = x / data[j * maxsize + j];
        }
    for (int i = 0; i < actualsize; i++)
        for (int j = i; j < actualsize; j++) {
            if (i == j) continue;
            float sum = 0.0;
            for (int k = i; k < j; k++) sum += data[k*maxsize+j]*((i==k)?1.0f:data[i*maxsize+k]);
            data[i * maxsize + j] = -sum;
        }
    for (int i = 0; i < actualsize; i++)
        for (int j = 0; j < actualsize; j++) {
            float sum = 0.0;
            for (int k = ((i > j) ? i : j); k < actualsize; k++)
                sum += ((j==k)?1.0f:data[j*maxsize+k])*data[k*maxsize+i];
            data[j * maxsize + i] = sum;
        }
}

static int validateArguments(int argc, char **argv, int *num_clusters, int *target_num_clusters)
{
    if (argc < 4 || argc > 5) { printf("Usage: %s num_clusters infile outfile [target_num_clusters]\n",argv[0]); return 1; }
    if (!sscanf(argv[1], "%d", num_clusters) || *num_clusters < 1) { printf("Invalid num_clusters\n"); return 1; }
    FILE *infile = fopen(argv[2], "r");
    if (!infile) { printf("Invalid infile.\n"); return 2; }
    fclose(infile);
    if (argc == 5) {
        if (!sscanf(argv[4], "%d", target_num_clusters) || *target_num_clusters > *num_clusters) {
            printf("Invalid target_num_clusters\n"); return 4;
        }
    } else { *target_num_clusters = 0; }
    return 0;
}

static void printUsage(char **argv) {
    printf("Usage: %s num_clusters infile outfile [target_num_clusters]\n", argv[0]);
}

static void writeCluster(FILE *f, clusters_t &c, int cl, int D) {
    fprintf(f,"Probability: %f\nN: %f\nMeans:",c.pi[cl],c.N[cl]);
    for (int i=0;i<D;i++) fprintf(f," %f",c.means[cl*D+i]);
    fprintf(f,"\n\nR Matrix:\n");
    for (int i=0;i<D;i++){for(int j=0;j<D;j++) fprintf(f,"%f ",c.R[cl*D*D+i*D+j]); fprintf(f,"\n");}
    fflush(f);
}
static void printCluster(clusters_t &c, int cl, int D) { writeCluster(stdout,c,cl,D); }

static void freeCluster(clusters_t *c) {
    free(c->N); free(c->pi); free(c->constant); free(c->avgvar);
    free(c->means); free(c->R); free(c->Rinv); free(c->memberships);
}

static void setupCluster(clusters_t *c, int num_clusters, int num_events, int D) {
    c->N           = (float *)malloc(sizeof(float)*num_clusters);
    c->pi          = (float *)malloc(sizeof(float)*num_clusters);
    c->constant    = (float *)malloc(sizeof(float)*num_clusters);
    c->avgvar      = (float *)malloc(sizeof(float)*num_clusters);
    c->means       = (float *)malloc(sizeof(float)*D*num_clusters);
    c->R           = (float *)malloc(sizeof(float)*D*D*num_clusters);
    c->Rinv        = (float *)malloc(sizeof(float)*D*D*num_clusters);
    c->memberships = (float *)malloc(sizeof(float)*num_events*
        (num_clusters + NUM_CLUSTERS_PER_BLOCK - num_clusters % NUM_CLUSTERS_PER_BLOCK));
}

static void add_clusters(clusters_t &cl, int c1, int c2, clusters_t &tmp, int D) {
    float wt1 = cl.N[c1] / (cl.N[c1] + cl.N[c2]);
    float wt2 = 1.0f - wt1;
    for (int i = 0; i < D; i++)
        tmp.means[i] = wt1*cl.means[c1*D+i] + wt2*cl.means[c2*D+i];
    for (int i = 0; i < D; i++)
        for (int j = i; j < D; j++) {
            tmp.R[i*D+j] = ((tmp.means[i]-cl.means[c1*D+i])*(tmp.means[j]-cl.means[c1*D+j])
                            +cl.R[c1*D*D+i*D+j])*wt1;
            tmp.R[i*D+j] += ((tmp.means[i]-cl.means[c2*D+i])*(tmp.means[j]-cl.means[c2*D+j])
                             +cl.R[c2*D*D+i*D+j])*wt2;
            tmp.R[j*D+i] = tmp.R[i*D+j];
        }
    tmp.pi[0] = cl.pi[c1] + cl.pi[c2];
    tmp.N[0]  = cl.N[c1]  + cl.N[c2];
    float log_det;
    memcpy(tmp.Rinv, tmp.R, sizeof(float)*D*D);
    invert_cpu(tmp.Rinv, D, &log_det);
    tmp.constant[0] = (-D)*0.5f*logf(2.0f*PI) - 0.5f*log_det;
    tmp.avgvar[0]   = cl.avgvar[0];
}

static void copy_cluster(clusters_t &dst, int c_dst, clusters_t &src, int c_src, int D) {
    dst.N[c_dst] = src.N[c_src]; dst.pi[c_dst] = src.pi[c_src];
    dst.constant[c_dst] = src.constant[c_src]; dst.avgvar[c_dst] = src.avgvar[c_src];
    memcpy(&dst.means[c_dst*D], &src.means[c_src*D], sizeof(float)*D);
    memcpy(&dst.R[c_dst*D*D],   &src.R[c_src*D*D],   sizeof(float)*D*D);
    memcpy(&dst.Rinv[c_dst*D*D],&src.Rinv[c_src*D*D],sizeof(float)*D*D);
}

static float cluster_distance(clusters_t &cl, int c1, int c2, clusters_t &tmp, int D) {
    add_clusters(cl, c1, c2, tmp, D);
    return cl.N[c1]*cl.constant[c1] + cl.N[c2]*cl.constant[c2] - tmp.N[0]*tmp.constant[0];
}

// ============================================================
// GPU kernels using Kokkos
// ============================================================

// constants_kernel: invert R, compute constant, normalize pi
static void kokkos_constants(
    FView d_R, FView d_Rinv,
    FView d_N, FView d_pi, FView d_constant, FView d_avgvar,
    int num_clusters, int D)
{
    // Per-cluster inversion (sequential per cluster, parallelized across clusters)
    int scratch_sz = (int)(sizeof(float) * (D * D + 2));  // matrix + log_det + sum
    auto policy = Kokkos::TeamPolicy<>(num_clusters, 1)
                      .set_scratch_size(0, Kokkos::PerTeam(scratch_sz));

    Kokkos::parallel_for("constants_invert", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
            const int bid = team.league_rank();
            ScratchFView scratch(team.team_scratch(0), D * D + 2);
            float *matrix  = scratch.data();
            float *log_det = scratch.data() + D * D;
            // log_det[0] = log_determinant, log_det[1] = sum for pi

            // Copy R to scratch
            Kokkos::single(Kokkos::PerTeam(team), [=]() {
                for (int i = 0; i < D * D; i++)
                    matrix[i] = d_R(bid * D * D + i);
                invert_device(matrix, D, log_det);
                for (int i = 0; i < D * D; i++)
                    d_Rinv(bid * D * D + i) = matrix[i];
                d_constant(bid) = -D * 0.5f * logf(2.0f * PI) - 0.5f * log_det[0];
            });
        });
    Kokkos::fence();

    // Normalize pi
    float N_sum = 0.0f;
    Kokkos::parallel_reduce("sum_N",
        Kokkos::RangePolicy<>(0, num_clusters),
        KOKKOS_LAMBDA(int c, float &s) { s += d_N(c); },
        N_sum);
    Kokkos::parallel_for("norm_pi",
        Kokkos::RangePolicy<>(0, num_clusters),
        KOKKOS_LAMBDA(int c) {
            d_pi(c) = (d_N(c) < 0.5f) ? 1e-10f : d_N(c) / N_sum;
        });
    Kokkos::fence();
}

// estep1: compute log-likelihood for each (cluster, event)
static void kokkos_estep1(
    FView d_data_by_dim,   // [D * num_events]
    FView d_Rinv,          // [num_clusters * D * D]
    FView d_memberships,   // [num_events * num_clusters_padded]
    FView d_pi, FView d_constant, FView d_means,
    int D, int num_events, int num_clusters)
{
    Kokkos::parallel_for("estep1",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_clusters, num_events}),
        KOKKOS_LAMBDA(int c, int event) {
            float like = 0.0f;
#if DIAG_ONLY
            for (int j = 0; j < D; j++) {
                float diff = d_data_by_dim(j * num_events + event) - d_means(c * D + j);
                like += diff * diff * d_Rinv(c * D * D + j * D + j);
            }
#else
            for (int i = 0; i < D; i++) {
                float di = d_data_by_dim(i * num_events + event) - d_means(c * D + i);
                for (int j = 0; j < D; j++) {
                    float dj = d_data_by_dim(j * num_events + event) - d_means(c * D + j);
                    like += di * dj * d_Rinv(c * D * D + i * D + j);
                }
            }
#endif
            d_memberships(c * num_events + event) =
                -0.5f * like + d_constant(c) + logf(d_pi(c));
        });
    Kokkos::fence();
}

// estep2: normalize probabilities and compute log-likelihood
static float kokkos_estep2(
    FView d_memberships,
    int num_clusters, int num_events)
{
    // Normalize per event
    FView d_event_ll("event_ll", num_events);
    Kokkos::parallel_for("estep2_norm",
        Kokkos::RangePolicy<>(0, num_events),
        KOKKOS_LAMBDA(int event) {
            float max_l = d_memberships(event);
            for (int c = 1; c < num_clusters; c++)
                max_l = fmaxf(max_l, d_memberships(c * num_events + event));

            float denom = 0.0f;
            for (int c = 0; c < num_clusters; c++)
                denom += expf(d_memberships(c * num_events + event) - max_l);
            denom = max_l + logf(denom);
            d_event_ll(event) = denom;

            for (int c = 0; c < num_clusters; c++)
                d_memberships(c * num_events + event) =
                    expf(d_memberships(c * num_events + event) - denom);
        });
    Kokkos::fence();

    float total = 0.0f;
    Kokkos::parallel_reduce("estep2_sum",
        Kokkos::RangePolicy<>(0, num_events),
        KOKKOS_LAMBDA(int e, float &s) { s += d_event_ll(e); },
        total);
    return total;
}

// mstep_N: recompute expected counts
static void kokkos_mstep_N(
    FView d_memberships, FView d_N, FView d_pi,
    int num_clusters, int num_events)
{
    Kokkos::parallel_for("mstep_N",
        Kokkos::TeamPolicy<>(num_clusters, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
            const int c = team.league_rank();
            float sum   = 0.0f;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, num_events),
                [=](int event, float &s) { s += d_memberships(c * num_events + event); },
                sum);
            Kokkos::single(Kokkos::PerTeam(team), [=]() {
                d_N(c)  = sum;
                d_pi(c) = sum;
            });
        });
    Kokkos::fence();
}

// mstep_means: recompute means
static void kokkos_mstep_means(
    FView d_data_by_dim, FView d_memberships, FView d_means,
    int num_clusters, int D, int num_events)
{
    Kokkos::parallel_for("mstep_means",
        Kokkos::TeamPolicy<>(num_clusters * D, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
            const int c = team.league_rank() / D;
            const int d = team.league_rank() % D;
            float sum   = 0.0f;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, num_events),
                [=](int event, float &s) {
                    s += d_data_by_dim(d * num_events + event) *
                         d_memberships(c * num_events + event);
                },
                sum);
            Kokkos::single(Kokkos::PerTeam(team), [=]() {
                d_means(c * D + d) = sum;
            });
        });
    Kokkos::fence();
}

// mstep_covariance: recompute R matrix (lower triangle, symmetric)
static void kokkos_mstep_covariance(
    FView d_data_by_dim, FView d_memberships, FView d_means,
    FView d_R, FView d_avgvar,
    int num_clusters, int D, int num_events)
{
    // Iterate over all (c, row, col) with row >= col
    Kokkos::parallel_for("mstep_cov",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {num_clusters, D, D}),
        KOKKOS_LAMBDA(int c, int row, int col) {
            if (row < col) return;
            float cov_sum = 0.0f;
            const float means_row = d_means(c * D + row);
            const float means_col = d_means(c * D + col);
            for (int event = 0; event < num_events; event++) {
                float val_row = d_data_by_dim(row * num_events + event);
                float val_col = d_data_by_dim(col * num_events + event);
                cov_sum += (val_row - means_row) * (val_col - means_col) *
                           d_memberships(c * num_events + event);
            }
            const int off = c * D * D;
            d_R(off + row * D + col) = cov_sum;
            d_R(off + col * D + row) = cov_sum;
            if (row == col) d_R(off + row * D + col) += d_avgvar(c);
        });
    Kokkos::fence();
}

// ============================================================
// Cluster EM algorithm
// ============================================================
static clusters_t *kokkos_cluster(
    int original_num_clusters, int desired_num_clusters,
    int *final_num_clusters,
    int D, int num_events,
    float *fcs_data_by_event)
{
    int stop_number = (desired_num_clusters == 0) ? 1 : desired_num_clusters;

    // Transpose data
    float *fcs_data_by_dim = (float *)malloc(sizeof(float) * num_events * D);
    for (int e = 0; e < num_events; e++)
        for (int d = 0; d < D; d++) {
            if (isnan(fcs_data_by_event[e*D+d])) { printf("NaN in input\n"); return nullptr; }
            fcs_data_by_dim[d*num_events+e] = fcs_data_by_event[e*D+d];
        }

    PRINT("Number of events: %d\nNumber of dimensions: %d\n", num_events, D);
    PRINT("Starting with %d clusters, will stop at %d.\n", original_num_clusters, stop_number);

    clusters_t clusters;
    setupCluster(&clusters, original_num_clusters, num_events, D);
    clusters_t *saved_clusters = (clusters_t *)malloc(sizeof(clusters_t));
    setupCluster(saved_clusters, original_num_clusters, num_events, D);

    float min_rissanen = FLT_MAX;
    int   ideal_num_clusters = original_num_clusters;

    clusters_t scratch_cluster;
    setupCluster(&scratch_cluster, 1, num_events, D);

    // Device views
    int memberships_padded = original_num_clusters +
        NUM_CLUSTERS_PER_BLOCK - original_num_clusters % NUM_CLUSTERS_PER_BLOCK;

    FView d_data_by_dim  ("data_by_dim",  D * num_events);
    FView d_data_by_event("data_by_event",D * num_events);
    FView d_N            ("N",            original_num_clusters);
    FView d_pi           ("pi",           original_num_clusters);
    FView d_constant     ("constant",     original_num_clusters);
    FView d_avgvar       ("avgvar",       original_num_clusters);
    FView d_means        ("means",        D * original_num_clusters);
    FView d_R            ("R",            D * D * original_num_clusters);
    FView d_Rinv         ("Rinv",         D * D * original_num_clusters);
    FView d_memberships  ("memberships",  num_events * memberships_padded);
    FView d_likelihoods  ("likelihoods",  NUM_BLOCKS);

    // Upload static data
    {
        auto h = Kokkos::create_mirror_view(d_data_by_dim);
        for (int i = 0; i < D * num_events; i++) h(i) = fcs_data_by_dim[i];
        Kokkos::deep_copy(d_data_by_dim, h);
    }
    {
        auto h = Kokkos::create_mirror_view(d_data_by_event);
        for (int i = 0; i < D * num_events; i++) h(i) = fcs_data_by_event[i];
        Kokkos::deep_copy(d_data_by_event, h);
    }

    // ---- Seed clusters ----
    {
        // Compute means and variances on host for seed
        std::vector<float> means_h(D, 0.0f), variances_h(D, 0.0f);
        for (int e = 0; e < num_events; e++)
            for (int d = 0; d < D; d++)
                means_h[d] += fcs_data_by_event[e * D + d];
        for (int d = 0; d < D; d++) means_h[d] /= num_events;
        for (int e = 0; e < num_events; e++)
            for (int d = 0; d < D; d++) {
                float v = fcs_data_by_event[e*D+d];
                variances_h[d] += v * v;
            }
        float avgvar = 0.0f;
        for (int d = 0; d < D; d++) {
            variances_h[d] = variances_h[d] / num_events - means_h[d] * means_h[d];
            avgvar += variances_h[d];
        }
        avgvar /= D;

        float seed = (original_num_clusters > 1)
                     ? (float)(num_events - 1) / (original_num_clusters - 1)
                     : 0.0f;

        auto h_means   = Kokkos::create_mirror_view(d_means);
        auto h_R       = Kokkos::create_mirror_view(d_R);
        auto h_N       = Kokkos::create_mirror_view(d_N);
        auto h_pi      = Kokkos::create_mirror_view(d_pi);
        auto h_avgvar  = Kokkos::create_mirror_view(d_avgvar);

        for (int c = 0; c < original_num_clusters; c++) {
            for (int d = 0; d < D; d++)
                h_means(c * D + d) = fcs_data_by_event[(int)(c * seed) * D + d];
            for (int i = 0; i < D * D; i++) {
                int row = i / D, col = i % D;
                h_R(c * D * D + i) = (row == col) ? 1.0f : 0.0f;
            }
            h_pi(c)     = 1.0f / original_num_clusters;
            h_N(c)      = (float)num_events / original_num_clusters;
            h_avgvar(c) = avgvar / COVARIANCE_DYNAMIC_RANGE;
        }
        Kokkos::deep_copy(d_means,  h_means);
        Kokkos::deep_copy(d_R,      h_R);
        Kokkos::deep_copy(d_N,      h_N);
        Kokkos::deep_copy(d_pi,     h_pi);
        Kokkos::deep_copy(d_avgvar, h_avgvar);
    }

    kokkos_constants(d_R, d_Rinv, d_N, d_pi, d_constant, d_avgvar,
                     original_num_clusters, D);

    // Copy back for host-side use
    {
        auto hN = Kokkos::create_mirror_view(d_N); Kokkos::deep_copy(hN, d_N);
        auto hpi = Kokkos::create_mirror_view(d_pi); Kokkos::deep_copy(hpi, d_pi);
        auto hconst = Kokkos::create_mirror_view(d_constant); Kokkos::deep_copy(hconst, d_constant);
        auto havgvar = Kokkos::create_mirror_view(d_avgvar); Kokkos::deep_copy(havgvar, d_avgvar);
        auto hmeans = Kokkos::create_mirror_view(d_means); Kokkos::deep_copy(hmeans, d_means);
        auto hR  = Kokkos::create_mirror_view(d_R);  Kokkos::deep_copy(hR,  d_R);
        auto hRinv = Kokkos::create_mirror_view(d_Rinv); Kokkos::deep_copy(hRinv, d_Rinv);
        for (int c = 0; c < original_num_clusters; c++) {
            clusters.N[c] = hN(c); clusters.pi[c] = hpi(c);
            clusters.constant[c] = hconst(c); clusters.avgvar[c] = havgvar(c);
            for (int i = 0; i < D; i++) clusters.means[c*D+i] = hmeans(c*D+i);
            for (int i = 0; i < D*D; i++) { clusters.R[c*D*D+i] = hR(c*D*D+i); clusters.Rinv[c*D*D+i] = hRinv(c*D*D+i); }
        }
    }

    float epsilon = (1 + D + 0.5f*(D+1)*D) * logf((float)num_events*D) * 0.001f;
    PRINT("epsilon = %f\n", epsilon);

    float distance, min_distance = 0.0f;
    float rissanen;
    int min_c1 = 0, min_c2 = 1;

    for (int num_clusters = original_num_clusters; num_clusters >= stop_number; num_clusters--) {

        // Initial E-step
        kokkos_estep1(d_data_by_dim, d_Rinv, d_memberships,
                      d_pi, d_constant, d_means, D, num_events, num_clusters);
        float likelihood = kokkos_estep2(d_memberships, num_clusters, num_events);
        float old_likelihood;
        float change = epsilon * 2;
        int iters = 0;

        PRINT("Performing EM on %d clusters.\n", num_clusters);

        while (iters < MIN_ITERS || (fabsf(change) > epsilon && iters < MAX_ITERS)) {
            old_likelihood = likelihood;

            // M-step
            kokkos_mstep_N(d_memberships, d_N, d_pi, num_clusters, num_events);

            {
                auto hN = Kokkos::create_mirror_view(d_N);
                Kokkos::deep_copy(hN, d_N);
                for (int c = 0; c < num_clusters; c++) clusters.N[c] = hN(c);
            }

            kokkos_mstep_means(d_data_by_dim, d_memberships, d_means,
                               num_clusters, D, num_events);

            {
                auto hm = Kokkos::create_mirror_view(d_means);
                Kokkos::deep_copy(hm, d_means);
                for (int c = 0; c < num_clusters; c++)
                    for (int d = 0; d < D; d++) {
                        clusters.means[c*D+d] = (clusters.N[c] > 0.5f)
                            ? hm(c*D+d) / clusters.N[c] : 0.0f;
                    }
                for (int c = 0; c < num_clusters; c++)
                    for (int d = 0; d < D; d++)
                        hm(c*D+d) = clusters.means[c*D+d];
                Kokkos::deep_copy(d_means, hm);
            }

            kokkos_mstep_covariance(d_data_by_dim, d_memberships, d_means,
                                    d_R, d_avgvar, num_clusters, D, num_events);

            {
                auto hR = Kokkos::create_mirror_view(d_R);
                Kokkos::deep_copy(hR, d_R);
                for (int c = 0; c < num_clusters; c++)
                    for (int i = 0; i < D*D; i++) clusters.R[c*D*D+i] = hR(c*D*D+i);
                for (int c = 0; c < num_clusters; c++) {
                    if (clusters.N[c] > 0.5f) {
                        for (int i = 0; i < D*D; i++) clusters.R[c*D*D+i] /= clusters.N[c];
                    } else {
                        for (int i = 0; i < D; i++)
                            for (int j = 0; j < D; j++)
                                clusters.R[c*D*D+i*D+j] = (i==j) ? 1.0f : 0.0f;
                    }
                }
                for (int c = 0; c < num_clusters; c++)
                    for (int i = 0; i < D*D; i++) hR(c*D*D+i) = clusters.R[c*D*D+i];
                Kokkos::deep_copy(d_R, hR);
            }

            kokkos_constants(d_R, d_Rinv, d_N, d_pi, d_constant, d_avgvar,
                             num_clusters, D);

            kokkos_estep1(d_data_by_dim, d_Rinv, d_memberships,
                          d_pi, d_constant, d_means, D, num_events, num_clusters);
            likelihood = kokkos_estep2(d_memberships, num_clusters, num_events);

            change = likelihood - old_likelihood;
            DEBUG("Change in likelihood: %e\n", change);
            iters++;
        }

        // Copy cluster data back
        {
            auto hN = Kokkos::create_mirror_view(d_N); Kokkos::deep_copy(hN, d_N);
            auto hpi = Kokkos::create_mirror_view(d_pi); Kokkos::deep_copy(hpi, d_pi);
            auto hc = Kokkos::create_mirror_view(d_constant); Kokkos::deep_copy(hc, d_constant);
            auto hav = Kokkos::create_mirror_view(d_avgvar); Kokkos::deep_copy(hav, d_avgvar);
            auto hm = Kokkos::create_mirror_view(d_means); Kokkos::deep_copy(hm, d_means);
            auto hR = Kokkos::create_mirror_view(d_R); Kokkos::deep_copy(hR, d_R);
            auto hRi = Kokkos::create_mirror_view(d_Rinv); Kokkos::deep_copy(hRi, d_Rinv);
            auto hmem = Kokkos::create_mirror_view(d_memberships); Kokkos::deep_copy(hmem, d_memberships);
            for (int c = 0; c < num_clusters; c++) {
                clusters.N[c] = hN(c); clusters.pi[c] = hpi(c);
                clusters.constant[c] = hc(c); clusters.avgvar[c] = hav(c);
                for (int i = 0; i < D; i++) clusters.means[c*D+i] = hm(c*D+i);
                for (int i = 0; i < D*D; i++) { clusters.R[c*D*D+i] = hR(c*D*D+i); clusters.Rinv[c*D*D+i] = hRi(c*D*D+i); }
            }
            for (int i = 0; i < num_events * num_clusters; i++)
                clusters.memberships[i] = hmem(i);
        }

        rissanen = -likelihood + 0.5f*(num_clusters*(1.0f+D+0.5f*(D+1.0f)*D)-1.0f)*logf((float)num_events*D);
        PRINT("\nLikelihood: %e\nRissanen: %e\n", likelihood, rissanen);

        if (num_clusters == original_num_clusters ||
            (rissanen < min_rissanen && desired_num_clusters == 0) ||
            num_clusters == desired_num_clusters) {
            min_rissanen = rissanen;
            ideal_num_clusters = num_clusters;
            memcpy(saved_clusters->N,           clusters.N,           sizeof(float)*num_clusters);
            memcpy(saved_clusters->pi,          clusters.pi,          sizeof(float)*num_clusters);
            memcpy(saved_clusters->constant,    clusters.constant,    sizeof(float)*num_clusters);
            memcpy(saved_clusters->avgvar,      clusters.avgvar,      sizeof(float)*num_clusters);
            memcpy(saved_clusters->means,       clusters.means,       sizeof(float)*D*num_clusters);
            memcpy(saved_clusters->R,           clusters.R,           sizeof(float)*D*D*num_clusters);
            memcpy(saved_clusters->Rinv,        clusters.Rinv,        sizeof(float)*D*D*num_clusters);
            memcpy(saved_clusters->memberships, clusters.memberships, sizeof(float)*num_events*num_clusters);
        }

        if (num_clusters <= stop_number) break;

        // Reduce order: eliminate empty clusters
        for (int i = num_clusters - 1; i >= 0; i--) {
            if (clusters.N[i] < 0.5f) {
                for (int j = i; j < num_clusters - 1; j++)
                    copy_cluster(clusters, j, clusters, j + 1, D);
                num_clusters--;
            }
        }
        if (num_clusters <= stop_number) { num_clusters++; break; }  // will decrement

        // Find closest pair
        min_c1 = 0; min_c2 = 1;
        for (int c1 = 0; c1 < num_clusters; c1++)
            for (int c2 = c1 + 1; c2 < num_clusters; c2++) {
                distance = cluster_distance(clusters, c1, c2, scratch_cluster, D);
                if ((c1 == 0 && c2 == 1) || distance < min_distance) {
                    min_distance = distance; min_c1 = c1; min_c2 = c2;
                }
            }
        // Merge min_c1 and min_c2
        add_clusters(clusters, min_c1, min_c2, scratch_cluster, D);
        copy_cluster(clusters, min_c1, scratch_cluster, 0, D);
        for (int i = min_c2; i < num_clusters - 1; i++)
            copy_cluster(clusters, i, clusters, i + 1, D);

        // Re-upload merged clusters
        {
            auto hN = Kokkos::create_mirror_view(d_N);
            auto hpi = Kokkos::create_mirror_view(d_pi);
            auto hc = Kokkos::create_mirror_view(d_constant);
            auto hav = Kokkos::create_mirror_view(d_avgvar);
            auto hm = Kokkos::create_mirror_view(d_means);
            auto hR = Kokkos::create_mirror_view(d_R);
            auto hRi = Kokkos::create_mirror_view(d_Rinv);
            for (int c = 0; c < num_clusters - 1; c++) {
                hN(c) = clusters.N[c]; hpi(c) = clusters.pi[c];
                hc(c) = clusters.constant[c]; hav(c) = clusters.avgvar[c];
                for (int i = 0; i < D; i++) hm(c*D+i) = clusters.means[c*D+i];
                for (int i = 0; i < D*D; i++) { hR(c*D*D+i) = clusters.R[c*D*D+i]; hRi(c*D*D+i) = clusters.Rinv[c*D*D+i]; }
            }
            Kokkos::deep_copy(d_N, hN); Kokkos::deep_copy(d_pi, hpi);
            Kokkos::deep_copy(d_constant, hc); Kokkos::deep_copy(d_avgvar, hav);
            Kokkos::deep_copy(d_means, hm); Kokkos::deep_copy(d_R, hR);
            Kokkos::deep_copy(d_Rinv, hRi);
        }
    }

    *final_num_clusters = ideal_num_clusters;
    free(fcs_data_by_dim);
    freeCluster(&clusters);
    freeCluster(&scratch_cluster);
    return saved_clusters;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char **argv)
{
    int original_num_clusters, desired_num_clusters, ideal_num_clusters;

    if (validateArguments(argc, argv, &original_num_clusters, &desired_num_clusters))
        return 1;

    int num_dimensions, num_events;
    PRINT("Parsing input file...");
    float *fcs_data_by_event = readData(argv[2], &num_dimensions, &num_events);
    if (!fcs_data_by_event) {
        printf("Error parsing input file.\n");
        return 1;
    }

    Kokkos::initialize(argc, argv);
    clusters_t *clusters = nullptr;
    {
        auto start = std::chrono::steady_clock::now();
        clusters = kokkos_cluster(original_num_clusters, desired_num_clusters,
                                  &ideal_num_clusters,
                                  num_dimensions, num_events, fcs_data_by_event);
        auto end = std::chrono::steady_clock::now();
        auto t   = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Execution time of cluster function %f (s)\n", t * 1e-9f);
    }
    Kokkos::finalize();

    if (!clusters) { free(fcs_data_by_event); return 1; }

    clusters_t saved_clusters;
    memcpy(&saved_clusters, clusters, sizeof(clusters_t));

    const char *result_suffix  = ".results";
    const char *summary_suffix = ".summary";
    char *result_filename  = (char *)malloc(strlen(argv[3]) + strlen(result_suffix)  + 1);
    char *summary_filename = (char *)malloc(strlen(argv[3]) + strlen(summary_suffix) + 1);
    strcpy(result_filename,  argv[3]); strcat(result_filename,  result_suffix);
    strcpy(summary_filename, argv[3]); strcat(summary_filename, summary_suffix);

    PRINT("Summary filename: %s\nResults filename: %s\n", summary_filename, result_filename);

    FILE *outf = fopen(summary_filename, "w");
    if (!outf) { printf("ERROR: Unable to open '%s'.\n", argv[3]); return -1; }

    for (int c = 0; c < ideal_num_clusters; c++) {
        if (ENABLE_PRINT) { PRINT("Cluster #%d\n", c); printCluster(saved_clusters, c, num_dimensions); PRINT("\n\n"); }
        if (ENABLE_OUTPUT) { fprintf(outf, "Cluster #%d\n", c); writeCluster(outf, saved_clusters, c, num_dimensions); fprintf(outf, "\n\n"); }
    }
    fclose(outf);

    if (ENABLE_OUTPUT) {
        FILE *fresults = fopen(result_filename, "w");
        char header[1000]; FILE *input_file = fopen(argv[2], "r");
        fgets(header, 1000, input_file); fclose(input_file);
        fprintf(fresults, "%s", header);
        for (int i = 0; i < num_events; i++) {
            for (int d = 0; d < num_dimensions - 1; d++)
                fprintf(fresults, "%f,", fcs_data_by_event[i*num_dimensions+d]);
            fprintf(fresults, "%f\t", fcs_data_by_event[i*num_dimensions+num_dimensions-1]);
            for (int c = 0; c < ideal_num_clusters - 1; c++)
                fprintf(fresults, "%f,", saved_clusters.memberships[c*num_events+i]);
            fprintf(fresults, "%f\n", saved_clusters.memberships[(ideal_num_clusters-1)*num_events+i]);
        }
        fclose(fresults);
    }

    free(fcs_data_by_event);
    freeCluster(&saved_clusters);
    free(result_filename); free(summary_filename);
    return 0;
}
