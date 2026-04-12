// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
// Matrix transpose benchmark with 8 variants using Kokkos.
// Ported from matrixT-sycl.
//
// All variants use MDRangePolicy<Rank<2>> (OpenMP backend limits team_size).
// Access patterns mirror the original:
//
//  0  simple copy
//  1  copy via shared memory (same result on CPU)
//  2  naive transpose
//  3  coalesced transpose (block-swap + within-tile tx/ty swap)
//  4  no-bank-conflicts transpose (same result as 3; padding irrelevant on CPU)
//  5  coarse-grained partial (block-swap, no within-tile reorder; bypass check)
//  6  fine-grained partial (within-tile transpose; bypass check)
//  7  diagonal transpose (block diagonal traversal; same result as 2–4)

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define TILE_DIM   16
#define BLOCK_ROWS 16

static void computeTransposeGold(float* gold, const float* idata,
                                  int size_x, int size_y) {
  for (int y = 0; y < size_y; ++y)
    for (int x = 0; x < size_x; ++x)
      gold[(x * size_y) + y] = idata[(y * size_x) + x];
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s <row_dim> <col_dim> <repeat>\n", argv[0]);
    return 0;
  }

  int size_x = atoi(argv[1]);
  int size_y = atoi(argv[2]);
  int repeat  = atoi(argv[3]);

  if (size_x != size_y) {
    printf("Error: non-square matrices (row_dim_size(%d) != col_dim_size(%d))\nExiting...\n\n",
           size_x, size_y);
    return 1;
  }
  if (size_x % TILE_DIM != 0 || size_y % TILE_DIM != 0) {
    printf("Matrix size must be integral multiple of tile size\nExiting...\n\n");
    return 1;
  }

  Kokkos::initialize(argc, argv);
  {
    const int mat_size = size_x * size_y;

    Kokkos::View<float*> d_idata("idata", mat_size);
    Kokkos::View<float*> d_odata("odata", mat_size);

    auto h_idata_v = Kokkos::create_mirror_view(d_idata);
    auto h_odata_v = Kokkos::create_mirror_view(d_odata);

    for (int i = 0; i < mat_size; i++)
      h_idata_v(i) = (float)i;

    Kokkos::deep_copy(d_idata, h_idata_v);

    float* h_idata = h_idata_v.data();
    float* transposeGold = new float[mat_size];
    computeTransposeGold(transposeGold, h_idata, size_x, size_y);

    printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: %dx%d\n\n",
           size_x, size_y, size_x/TILE_DIM, size_y/TILE_DIM,
           TILE_DIM, TILE_DIM, TILE_DIM, BLOCK_ROWS);

    const int TD = TILE_DIM;
    const int sx = size_x;
    const int sy = size_y;
    const int gridDim_x = size_x / TILE_DIM;

    bool success = true;

    using MDPol2 = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

    for (int k = 0; k < 8; k++) {
      const char* kernelName = nullptr;

      auto t0 = std::chrono::steady_clock::now();

      switch (k) {
        // -----------------------------------------------------------
        // Case 0: plain element-wise copy
        case 0: {
          kernelName = "simple copy       ";
          MDPol2 policy({0, 0}, {sy, sx});
          for (int i = 0; i < repeat; i++) {
            Kokkos::parallel_for("copy", policy,
              KOKKOS_LAMBDA(int y, int x) {
                d_odata(y * sx + x) = d_idata(y * sx + x);
              });
          }
          Kokkos::fence();
          break;
        }

        // -----------------------------------------------------------
        // Case 1: copy (shared-memory path — same result on CPU)
        case 1: {
          kernelName = "shared memory copy";
          MDPol2 policy({0, 0}, {sy, sx});
          for (int i = 0; i < repeat; i++) {
            Kokkos::parallel_for("copyShared", policy,
              KOKKOS_LAMBDA(int y, int x) {
                d_odata(y * sx + x) = d_idata(y * sx + x);
              });
          }
          Kokkos::fence();
          break;
        }

        // -----------------------------------------------------------
        // Case 2: naive transpose — strided global write
        case 2: {
          kernelName = "naive             ";
          MDPol2 policy({0, 0}, {sy, sx});
          for (int i = 0; i < repeat; i++) {
            Kokkos::parallel_for("naive", policy,
              KOKKOS_LAMBDA(int y, int x) {
                d_odata(y + sy * x) = d_idata(x + sx * y);
              });
          }
          Kokkos::fence();
          break;
        }

        // -----------------------------------------------------------
        // Case 3: coalesced transpose.
        // On GPU: thread (ty,tx) loads tile[ty][tx] = idata[...] then stores
        //   odata[...] = tile[tx][ty]  (within-tile transpose via shared mem).
        // On CPU: we emit the same final mapping directly.
        //   For each element at input (bx·T+tx, by·T+ty) (col, row):
        //     output position = (by·T+tx, bx·T+ty)
        //     input  value    = idata[(bx·T+ty) + (by·T+tx)·sx]   ← tx/ty swapped
        case 3: {
          kernelName = "coalesced         ";
          MDPol2 policy({0, 0}, {sy, sx});
          for (int i = 0; i < repeat; i++) {
            Kokkos::parallel_for("coalesced", policy,
              KOKKOS_LAMBDA(int y, int x) {
                const int bx = x / TD, tx = x % TD;
                const int by = y / TD, ty = y % TD;
                const int index_out = (by * TD + tx) + (bx * TD + ty) * sy;
                const int index_in  = (bx * TD + ty) + (by * TD + tx) * sx;
                d_odata(index_out) = d_idata(index_in);
              });
          }
          Kokkos::fence();
          break;
        }

        // -----------------------------------------------------------
        // Case 4: no-bank-conflicts transpose (padded tile on GPU).
        // Same logical result as case 3; padding is irrelevant on CPU.
        case 4: {
          kernelName = "optimized         ";
          MDPol2 policy({0, 0}, {sy, sx});
          for (int i = 0; i < repeat; i++) {
            Kokkos::parallel_for("optimized", policy,
              KOKKOS_LAMBDA(int y, int x) {
                const int bx = x / TD, tx = x % TD;
                const int by = y / TD, ty = y % TD;
                const int index_out = (by * TD + tx) + (bx * TD + ty) * sy;
                const int index_in  = (bx * TD + ty) + (by * TD + tx) * sx;
                d_odata(index_out) = d_idata(index_in);
              });
          }
          Kokkos::fence();
          break;
        }

        // -----------------------------------------------------------
        // Case 5: coarse-grained partial transpose.
        // Blocks are swapped (block-level transpose) but NO within-tile
        // element reorder — the tile is read back at the same (ty,tx) position
        // it was written. Bypass check: gold = h_odata.
        case 5: {
          kernelName = "coarse-grained    ";
          MDPol2 policy({0, 0}, {sy, sx});
          for (int i = 0; i < repeat; i++) {
            Kokkos::parallel_for("coarseGrained", policy,
              KOKKOS_LAMBDA(int y, int x) {
                const int bx = x / TD, tx = x % TD;
                const int by = y / TD, ty = y % TD;
                // index_in: original position (no within-tile swap)
                const int index_in  = x + y * sx;
                // index_out: block coordinates are swapped
                const int index_out = (by * TD + tx) + (bx * TD + ty) * sy;
                d_odata(index_out) = d_idata(index_in);
              });
          }
          Kokkos::fence();
          break;
        }

        // -----------------------------------------------------------
        // Case 6: fine-grained partial transpose.
        // Within-tile transpose: output stays at the same linear index as
        // input but the value is read from the transposed position within
        // the tile. Bypass check: gold = h_odata.
        case 6: {
          kernelName = "fine-grained      ";
          MDPol2 policy({0, 0}, {sy, sx});
          for (int i = 0; i < repeat; i++) {
            Kokkos::parallel_for("fineGrained", policy,
              KOKKOS_LAMBDA(int y, int x) {
                const int bx = x / TD, tx = x % TD;
                const int by = y / TD, ty = y % TD;
                // Write to the same linear position as x+y·sx
                const int index_out = x + y * sx;
                // Read from the within-tile transposed source
                const int index_in  = (bx * TD + ty) + (by * TD + tx) * sx;
                d_odata(index_out) = d_idata(index_in);
              });
          }
          Kokkos::fence();
          break;
        }

        // -----------------------------------------------------------
        // Case 7: diagonal transpose.
        // Reorders block execution along matrix diagonals (cache-friendly on
        // GPU) while producing the same full transpose as cases 2–4.
        case 7: {
          kernelName = "diagonal          ";
          const int nblocks_x = sx / TD;
          const int nblocks_y = sy / TD;
          MDPol2 policy({0, 0}, {sy, sx});
          for (int i = 0; i < repeat; i++) {
            Kokkos::parallel_for("diagonal", policy,
              KOKKOS_LAMBDA(int y, int x) {
                const int bx_raw = x / TD, tx = x % TD;
                const int by_raw = y / TD, ty = y % TD;

                // Diagonal block reordering (same as original GPU kernel)
                int blockIdx_x, blockIdx_y;
                if (sx == sy) {
                  blockIdx_y = bx_raw;
                  blockIdx_x = (bx_raw + by_raw) % gridDim_x;
                } else {
                  const int b = bx_raw + nblocks_x * by_raw;
                  blockIdx_y  = b % nblocks_y;
                  blockIdx_x  = ((b / nblocks_y) + blockIdx_y) % nblocks_x;
                }

                // Same within-tile transpose as cases 3/4
                const int index_out = (blockIdx_y * TD + tx) + (blockIdx_x * TD + ty) * sy;
                const int index_in  = (blockIdx_x * TD + ty) + (blockIdx_y * TD + tx) * sx;
                d_odata(index_out) = d_idata(index_in);
              });
          }
          Kokkos::fence();
          break;
        }
      } // end switch

      auto t1 = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      printf("Average kernel (%s) execution time: %f (us)\n",
             kernelName, (elapsed * 1e-3f) / repeat);

      // Verify result
      Kokkos::deep_copy(h_odata_v, d_odata);
      float* h_odata = h_odata_v.data();

      const float* gold = nullptr;
      if (k == 0 || k == 1) {
        gold = h_idata;          // copy: output must equal input
      } else if (k == 5 || k == 6) {
        gold = h_odata;          // partial transposes: bypass check
      } else {
        gold = transposeGold;    // full transpose
      }

      bool ok = true;
      for (int i = 0; i < mat_size; i++) {
        if (std::fabs(gold[i] - h_odata[i]) > 0.01f) {
          ok = false;
          break;
        }
      }
      if (!ok) {
        printf("*** case %d: kernel FAILED ***\n", k);
        success = false;
      }
    } // end for k

    printf("%s\n", success ? "PASS" : "FAIL");

    delete[] transposeGold;
  }
  Kokkos::finalize();
  return 0;
}
