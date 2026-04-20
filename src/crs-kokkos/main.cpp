//  Kokkos port of crs-omp/main.cpp
//  Galois/Reed-Solomon encoding benchmark.
//
//  The OMP target teams shared-memory kernels from kernels.cpp are replaced
//  with a generic Kokkos TeamPolicy kernel that handles all (m, w) combina-
//  tions supported by the original benchmark.
//
//  Supporting math (GCRSMatrix, galois, jerasure) is compiled from the SYCL
//  variant which has no offloading annotations.

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include "GCRSMatrix.h"
#include "utils.h"

// ---- Generic GRS coding kernel (flat parallel_for, no shared memory) ------
//
// Each work item is one output element (thread `tid` in team `gid`).
// Instead of using shared memory for input staging, each work item directly
// reads the required elements from the `in` array.  This trades some memory
// bandwidth redundancy for portability (no Kokkos scratch memory needed).
//
void crs_coding_kernel(
    int                      k,
    int                      index,      // starting position in bm
    int                      m,          // output rows (1-4)
    int                      w,          // word width (4-8)
    Kokkos::View<const long*> in,
    Kokkos::View<long*>       out,
    Kokkos::View<const unsigned int*> bm,
    int                      threadDimX, // logical threads per team
    int                      blockDimX,  // number of teams
    int                      size)       // work items per output row
{
    const int total = blockDimX * threadDimX;

    Kokkos::parallel_for("crs_coding", total,
        KOKKOS_LAMBDA(int global_tid) {
            const int gid = global_tid / threadDimX;
            const int tid = global_tid % threadDimX;

            const int worksize_perblock = (threadDimX / w) * w;
            const unsigned int idx = (unsigned int)(worksize_perblock * gid + tid);

            if (tid >= worksize_perblock || (int)idx >= size) return;

            const int group_offset       = (tid / w) * w;
            const int group_inner_offset = tid % w;
            const unsigned long fullOneBit = 0xFFFFFFFFFFFFFFFFUL;

            long result[4] = {0, 0, 0, 0};

            int local_index = index;

            for (int i = 0; i < k; i++) {
                // Read the w input values for this group directly from global mem
                // (no shared memory staging — each work item reads its own slice)
                int base = (int)(worksize_perblock * gid + group_offset);

                for (int j = 0; j < w; j++) {
                    long data_val = in[i * size + base + j];
                    unsigned int matrixInt = bm[local_index];
                    for (int mi = 0; mi < m; mi++) {
                        int shift = group_inner_offset + mi * w;
                        result[mi] ^=
                            (((long)((matrixInt >> shift) & 0x1)) * (long)fullOneBit)
                            & data_val;
                    }
                    local_index++;
                }
            }

            for (int mi = 0; mi < m; mi++)
                out[(int)idx + mi * size] = result[mi];
        });
    Kokkos::fence();
}

// ---- Function pointer type (same as OMP version) --------------------------
typedef void (*coding_func)(int k, int index,
    const long *dataPtr, long *codeDevPtr,
    const unsigned int *bitMatrixPtr,
    int threadDimX, int blockDimX,
    int workSizePerGridInLong);

// ---- Kokkos-backed coding dispatch -----------------------------------------
//
// The OMP version used function pointers into the device.  For Kokkos we
// dispatch through the generic kernel above so no device function pointers
// are needed.
//
struct KokkosCodingFunc {
    int m, w;
    Kokkos::View<long*>              d_in;
    Kokkos::View<long*>              d_out;
    Kokkos::View<unsigned int*>      d_bm;

    void operator()(int k, int index,
                    int threadDimX, int blockDimX, int size) const {
        crs_coding_kernel(k, index, m, w,
                          Kokkos::View<const long*>(d_in),
                          d_out,
                          Kokkos::View<const unsigned int*>(d_bm),
                          threadDimX, blockDimX, size);
    }
};

// ---------------------------------------------------------------------------
int main(int argc, const char *argv[]) {
    if (argc != 3) {
        printf("Usage: ./%s workSizePerDataParityBlockInMB numberOfTasks\n", argv[0]);
        return 1;
    }

    int bufSize  = atoi(argv[1]) * 1024 * 1024;
    int taskNum  = atoi(argv[2]);
    double encode_time = 0.0;

    Kokkos::initialize(argc, const_cast<char**>(argv));
    {
#ifdef DUMP
        for (int m_val = 4; m_val <= 4; ++m_val)
        for (int n     = 8; n     <= 8; ++n)
        for (int k     = MAX_K; k <= MAX_K; ++k) {
#else
        for (int m_val = 1; m_val <= 4; ++m_val)
        for (int n     = 4; n     <= 8; ++n)
        for (int k     = m_val; k <= MAX_K; ++k) {
#endif
            int w = gcrs_check_k_m_w(k, m_val, n);
            if (w < 0) continue;

            int *bitmatrix = gcrs_create_bitmatrix(k, m_val, w);

            int bufSizePerTask    = (int)align_value(bufSize / taskNum, sizeof(long) * w);
            bufSize               = bufSizePerTask * taskNum;
            int bufSizeForLastTask = bufSize - (bufSizePerTask * (taskNum - 1));

            // ---- Allocate host data ----------------------------------------
            char *data = (char*)malloc((size_t)bufSize * k);
            char *code = (char*)malloc((size_t)bufSize * m_val);
            generateRandomValue(data, (size_t)bufSize * k);

            int dataSizePerAssign = bufSizePerTask * k;
            int codeSizePerAssign = bufSizePerTask * m_val;

            int taskSize = 1;
            int mRemain  = m_val;
            if (m_val >= MAX_M) {
                taskSize = m_val / MAX_M;
                if (m_val % MAX_M != 0) ++taskSize;
            }

            int *mValue  = (int*)malloc(sizeof(int) * taskSize);
            int *idx_arr = (int*)malloc(sizeof(int) * taskSize);

            for (int i = 0; i < taskSize; ++i) {
                mValue[i]  = (mRemain < MAX_M) ? mRemain : MAX_M;
                mRemain   -= mValue[i];
                idx_arr[i] = (i == 0) ? 0 : idx_arr[i - 1] + k * w;
            }

            unsigned int *all_bm = (unsigned int*)malloc(
                sizeof(unsigned int) * k * w * taskSize);
            int mValueSum = 0;
            for (int i = 0; i < taskSize; ++i) {
                unsigned int *col_bm = gcrs_create_column_coding_bitmatrix(
                    k, mValue[i], w, bitmatrix + k * w * mValueSum * w);
                memcpy(all_bm + i * k * w, col_bm, k * w * sizeof(unsigned int));
                free(col_bm);
                mValueSum += mValue[i];
            }

            // ---- Copy to device ------------------------------------------
            size_t data_sz = (size_t)bufSize * k / sizeof(long);
            size_t code_sz = (size_t)bufSize * m_val / sizeof(long);

            Kokkos::View<long*> d_in ("d_in",  data_sz);
            Kokkos::View<long*> d_out("d_out", code_sz);
            Kokkos::View<unsigned int*> d_bm("d_bm", k * w * taskSize);

            {
                auto hm_in = Kokkos::create_mirror_view(d_in);
                auto hm_bm = Kokkos::create_mirror_view(d_bm);
                memcpy(hm_in.data(), data, (size_t)bufSize * k);
                memcpy(hm_bm.data(), all_bm, sizeof(unsigned int) * k * w * taskSize);
                Kokkos::deep_copy(d_in, hm_in);
                Kokkos::deep_copy(d_bm, hm_bm);
            }

            // ---- Compute launch parameters --------------------------------
            int warpThreadNum    = 32;
            int threadNum        = MAX_THREAD_NUM;
            size_t workSizePerWarp  = (warpThreadNum / w) * w;
            size_t workSizePerBlock = (threadNum / warpThreadNum) * workSizePerWarp * sizeof(size_t);
            size_t blockNum = bufSizePerTask / workSizePerBlock;
            if (bufSizePerTask % workSizePerBlock != 0) blockNum++;

            struct timeval t0, t1;
            gettimeofday(&t0, NULL);

            for (int i = 0; i < taskNum; ++i) {
                int count = (i == taskNum - 1) ? bufSizeForLastTask : bufSizePerTask;
                int workSizePerGrid = count / (int)sizeof(long);
                int real_size = workSizePerGrid * (int)sizeof(long);

                mValueSum = 0;
                // compute offset into d_in and d_out for this task
                size_t in_offset  = (size_t)dataSizePerAssign * i / sizeof(long);
                size_t out_offset = (size_t)codeSizePerAssign * i / sizeof(long);

                for (int j = 0; j < taskSize; ++j) {
                    // subview into d_in and d_out for this task
                    auto sub_in  = Kokkos::subview(d_in,
                        Kokkos::make_pair(in_offset,  d_in.extent(0)));
                    auto sub_out = Kokkos::subview(d_out,
                        Kokkos::make_pair(out_offset + (size_t)mValueSum * real_size,
                                          d_out.extent(0)));
                    auto sub_bm  = Kokkos::subview(d_bm,
                        Kokkos::make_pair((size_t)j * k * w, d_bm.extent(0)));

                    crs_coding_kernel(k, idx_arr[j],
                                      mValue[j], w,
                                      Kokkos::View<const long*>(sub_in),
                                      sub_out,
                                      Kokkos::View<const unsigned int*>(sub_bm),
                                      threadNum, (int)blockNum, workSizePerGrid);

                    mValueSum += mValue[j];
                }
            }
            Kokkos::fence();

            gettimeofday(&t1, NULL);
            double elapsed = elapsed_time_in_ms(t0, t1);
            printf("k=%d m=%d w=%d time=%f ms\n", k, m_val, w, elapsed);
            encode_time += elapsed;

            // ---- Copy code back to host ----------------------------------
            {
                auto hm_out = Kokkos::create_mirror_view(d_out);
                Kokkos::deep_copy(hm_out, d_out);
                memcpy(code, hm_out.data(), (size_t)bufSize * m_val);
            }

            free(data); free(code);
            free(mValue); free(idx_arr); free(all_bm); free(bitmatrix);
        }
    }
    Kokkos::finalize();

    printf("Total encoding time: %f ms\n", encode_time);
    return 0;
}
