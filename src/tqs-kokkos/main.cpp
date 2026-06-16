/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Ported to Kokkos.
 */

#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// Support headers live in ../tqs-cuda/support/ (via -I../tqs-cuda)
#include "support/setup.h"
#include "support/task.h"
#include "support/verify.h"

using exec_space = Kokkos::DefaultExecutionSpace;
using mem_space  = Kokkos::DefaultExecutionSpace::memory_space;

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------
struct Params {
    int         n_gpu_threads;
    int         n_gpu_blocks;
    int         n_threads;
    int         n_warmup;
    int         n_reps;
    const char *file_name;
    int         pattern;
    int         pool_size;
    int         queue_size;
    int         iterations;

    Params(int argc, char **argv) {
        n_gpu_threads = 64;
        n_gpu_blocks  = 320;
        n_threads     = 1;
        n_warmup      = 5;
        n_reps        = 1000;
        file_name     = "input/patternsNP100NB512FB25.txt";
        pattern       = 1;
        pool_size     = 3200;
        queue_size    = 320;
        iterations    = 1000;
        int opt;
        while ((opt = getopt(argc, argv, "hi:g:t:w:r:f:k:s:q:n:")) >= 0) {
            switch (opt) {
            case 'h': usage(); exit(0); break;
            case 'i': n_gpu_threads = atoi(optarg); break;
            case 'g': n_gpu_blocks  = atoi(optarg); break;
            case 't': n_threads     = atoi(optarg); break;
            case 'w': n_warmup      = atoi(optarg); break;
            case 'r': n_reps        = atoi(optarg); break;
            case 'f': file_name     = optarg;        break;
            case 'k': pattern       = atoi(optarg); break;
            case 's': pool_size     = atoi(optarg); break;
            case 'q': queue_size    = atoi(optarg); break;
            case 'n': iterations    = atoi(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
        assert(n_gpu_threads > 0 && "Invalid # of device threads!");
        assert(n_gpu_blocks  > 0 && "Invalid # of device blocks!");
        assert(n_threads     > 0 && "Invalid # of host threads!");
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./tqs [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -i <I>    # of device threads per block (default=64)"
                "\n    -g <G>    # of device blocks (default=320)"
                "\n    -t <T>    # of host threads (default=1)"
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=1000)"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    patterns file name"
                "\n              (default=input/patternsNP100NB512FB25.txt)"
                "\n    -k <K>    pattern in file (default=1)"
                "\n    -s <S>    task pool size (default=3200)"
                "\n    -q <Q>    task queue size (default=320)"
                "\n    -n <N>    # of iterations in heavy task (default=1000)"
                "\n");
    }
};

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------
void read_input(int *pattern, task_t *task_pool, const Params &p) {
    char filePatterns[100];
    sprintf(filePatterns, "%s", p.file_name);

    FILE *File;
    int r;
    if ((File = fopen(filePatterns, "rt")) != NULL) {
        for (int y = 0; y <= p.pattern; y++)
            for (int x = 0; x < 512; x++)
                fscanf(File, "%d ", &r), pattern[x] = r;
        fclose(File);
    } else {
        printf("Unable to open file %s\n", filePatterns);
        exit(-1);
    }

    for (int i = 0; i < p.pool_size; i++) {
        task_pool[i].id = i;
        task_pool[i].op = SIGNAL_NOTWORK_KERNEL;
    }
    for (int i = 0; i < p.pool_size; i++) {
        pattern[i] = pattern[i % 512];
        if (pattern[i] == 1)
            task_pool[i].op = SIGNAL_WORK_KERNEL;
    }
}

// ---------------------------------------------------------------------------
// Host task insertion (from host_task.cpp)
// ---------------------------------------------------------------------------
static void host_insert_tasks(task_t *queue, int *data_queue,
                               task_t *task_pool, int *data,
                               int *num_written_tasks,
                               int gpuQueueSize, int offset,
                               int n_work_items) {
    memcpy(&queue[0], &task_pool[offset], gpuQueueSize * sizeof(task_t));
    memcpy(&data_queue[0], &data[offset * n_work_items],
           gpuQueueSize * n_work_items * sizeof(int));
    *num_written_tasks += gpuQueueSize;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    Kokkos::initialize(argc, argv);
    {
        const Params p(argc, argv);

        assert(p.n_gpu_threads <= 256 &&
               "Thread block size exceeds device maximum");

        int    *h_pattern     = (int *)   malloc(p.pool_size * sizeof(int));
        task_t *h_task_pool   = (task_t *)malloc(p.pool_size * sizeof(task_t));
        task_t *h_task_queues = (task_t *)malloc(p.queue_size * sizeof(task_t));
        int    *h_data_pool   = (int *)   malloc(p.pool_size * p.n_gpu_threads * sizeof(int));
        int    *h_data_queues = (int *)   malloc(p.queue_size * p.n_gpu_threads * sizeof(int));
        int    *h_consumed    = (int *)   malloc(sizeof(int));

        ALLOC_ERR(h_pattern, h_task_pool, h_task_queues,
                  h_data_pool, h_data_queues, h_consumed);

        read_input(h_pattern, h_task_pool, p);
        memset(h_data_pool, 0, p.pool_size * p.n_gpu_threads * sizeof(int));
        memset(h_consumed,  0, sizeof(int));

        // Allocate device Views.  Allocate extra task slots so threads that
        // atomically claim an index >= gpuQueueSize can still safely read
        // (they get zero-initialised dummy values; the while loop exits before
        // those values are used).
        const int extra = p.n_gpu_blocks;
        Kokkos::View<task_t*, mem_space> d_task_queues(
            "d_task_queues", p.queue_size + extra);
        Kokkos::View<int*,    mem_space> d_data_queues(
            "d_data_queues", p.queue_size * p.n_gpu_threads);
        Kokkos::View<int[1],  mem_space> d_consumed("d_consumed");

        using ScratchSpace = exec_space::scratch_memory_space;
        using IntScratch   = Kokkos::View<int*, ScratchSpace,
                                          Kokkos::MemoryUnmanaged>;
        // 3 shared ints per team: next_val, task_id, task_op
        int scratch_size = IntScratch::shmem_size(3);

        const int blocks  = p.n_gpu_blocks;
        const int threads = p.n_gpu_threads;

        auto policy =
            Kokkos::TeamPolicy<exec_space>(blocks, threads)
                .set_scratch_size(0, Kokkos::PerTeam(scratch_size));

        auto start = std::chrono::steady_clock::now();

        for (int rep = 0; rep < p.n_reps + p.n_warmup; rep++) {

            memset(h_data_pool, 0, p.pool_size * p.n_gpu_threads * sizeof(int));
            int n_written_tasks = 0;

            for (int n_consumed_tasks = 0;
                 n_consumed_tasks < p.pool_size;
                 n_consumed_tasks += p.queue_size) {

                host_insert_tasks(h_task_queues, h_data_queues,
                                  h_task_pool,   h_data_pool,
                                  &n_written_tasks,
                                  p.queue_size, n_consumed_tasks,
                                  p.n_gpu_threads);

                // Copy task queue h2d (only the first queue_size slots)
                {
                    auto h_tq = Kokkos::View<task_t*, Kokkos::HostSpace,
                                             Kokkos::MemoryUnmanaged>(
                                    h_task_queues, p.queue_size);
                    auto d_tq_sub = Kokkos::subview(
                        d_task_queues,
                        Kokkos::make_pair(0, p.queue_size));
                    Kokkos::deep_copy(d_tq_sub, h_tq);
                }
                // Copy data queue h2d
                {
                    auto h_dq = Kokkos::View<int*, Kokkos::HostSpace,
                                             Kokkos::MemoryUnmanaged>(
                                    h_data_queues,
                                    p.queue_size * p.n_gpu_threads);
                    Kokkos::deep_copy(d_data_queues, h_dq);
                }
                // Reset consumed counter on device
                {
                    auto h_cons = Kokkos::View<int[1], Kokkos::HostSpace,
                                               Kokkos::MemoryUnmanaged>(
                                      h_consumed);
                    Kokkos::deep_copy(d_consumed, h_cons);
                }

                const int gpuQueueSize = p.queue_size;
                const int offset       = n_consumed_tasks;
                const int iters        = p.iterations;

                Kokkos::parallel_for(
                    "tqs", policy,
                    KOKKOS_LAMBDA(
                        const Kokkos::TeamPolicy<exec_space>::member_type
                            &team) {
                        IntScratch scratch(team.team_scratch(0), 3);
                        // scratch(0) = next index
                        // scratch(1) = current task id
                        // scratch(2) = current task op
                        int &next_ref    = scratch(0);
                        int &task_id_ref = scratch(1);
                        int &task_op_ref = scratch(2);

                        const int tid       = team.team_rank();
                        const int tile_size = team.team_size();

                        // Thread 0 fetches the first task atomically
                        Kokkos::single(Kokkos::PerTeam(team), [&]() {
                            next_ref    = Kokkos::atomic_fetch_add(
                                              &d_consumed(0), 1);
                            task_id_ref = d_task_queues(next_ref).id;
                            task_op_ref = d_task_queues(next_ref).op;
                        });
                        team.team_barrier();

                        while (next_ref < gpuQueueSize) {
                            if (task_op_ref == SIGNAL_WORK_KERNEL) {
                                for (int i = 0; i < iters; i++)
                                    d_data_queues(
                                        (task_id_ref - offset) * tile_size
                                        + tid) += tile_size;
                                d_data_queues(
                                    (task_id_ref - offset) * tile_size
                                    + tid) += task_id_ref;
                            }
                            if (task_op_ref == SIGNAL_NOTWORK_KERNEL) {
                                d_data_queues(
                                    (task_id_ref - offset) * tile_size
                                    + tid) += tile_size;
                                d_data_queues(
                                    (task_id_ref - offset) * tile_size
                                    + tid) += task_id_ref;
                            }
                            // Thread 0 fetches the next task
                            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                                next_ref    = Kokkos::atomic_fetch_add(
                                                  &d_consumed(0), 1);
                                task_id_ref = d_task_queues(next_ref).id;
                                task_op_ref = d_task_queues(next_ref).op;
                            });
                            team.team_barrier();
                        }
                    });
                Kokkos::fence();

                // Copy result d2h
                auto h_mirror = Kokkos::create_mirror_view(d_data_queues);
                Kokkos::deep_copy(h_mirror, d_data_queues);
                memcpy(&h_data_pool[n_consumed_tasks * p.n_gpu_threads],
                       h_mirror.data(),
                       p.queue_size * p.n_gpu_threads * sizeof(int));
            }
        }

        auto end  = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - start).count();
        printf("Total task execution time for %d iterations: %f (ms)\n",
               p.n_reps + p.n_warmup, time * 1e-6f);

        verify(h_data_pool, h_pattern, p.pool_size, p.iterations,
               p.n_gpu_threads);

        free(h_pattern);
        free(h_consumed);
        free(h_task_queues);
        free(h_data_queues);
        free(h_task_pool);
        free(h_data_pool);

        printf("Test Passed\n");
    }
    Kokkos::finalize();
    return 0;
}
