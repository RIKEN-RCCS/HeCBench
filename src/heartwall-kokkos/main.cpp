// Heart wall tracking – Kokkos port
// Original OMP-target version: heartwall-omp/{main.cpp, kernel/kernel.cpp}

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <chrono>

#include <Kokkos_Core.hpp>

// ─── Pull in shared structures and utilities ─────────────────────────────────
// Reuse utility source from the OMP benchmark directory.
// Define fp and NUMBER_THREADS before including main.h.
#define fp float

// For the Kokkos serial-per-point kernel we run with 1 logical "thread"
// per team, so thread loops process all elements in one pass.
#define NUMBER_THREADS 1

// Sentinel that main.h needs
#define CHECK 37

// Hardcoded point counts (from main.h)
#define ENDO_POINTS 20
#define EPI_POINTS  31
#define ALL_POINTS  51

#include "../heartwall-omp/main.h"
#include "../heartwall-omp/util/timer/timer.h"
#include "../heartwall-omp/util/file/file.h"
#include "../heartwall-omp/util/avi/avilib.h"
#include "../heartwall-omp/util/avi/avimod.h"

// ─── Kokkos kernel wrapper ────────────────────────────────────────────────────

void kernel_gpu_wrapper(
    params_common common,
    int* endoRow,   int* endoCol,
    int* tEndoRowLoc, int* tEndoColLoc,
    int* epiRow,    int* epiCol,
    int* tEpiRowLoc,  int* tEpiColLoc,
    avi_t* frames)
{
  // Compute derived sizes (mirrors kernel.cpp setup)
  common.in_rows  = common.tSize + 1 + common.tSize;
  common.in_cols  = common.in_rows;
  common.in_elem  = common.in_rows * common.in_cols;
  common.in_mem   = sizeof(fp) * common.in_elem;

  common.in2_rows  = common.sSize + 1 + common.sSize;
  common.in2_cols  = common.in2_rows;
  common.in2_elem  = common.in2_rows * common.in2_cols;
  common.in2_mem   = sizeof(fp) * common.in2_elem;

  common.conv_rows = common.in_rows + common.in2_rows - 1;
  common.conv_cols = common.in_cols + common.in2_cols - 1;
  common.conv_elem = common.conv_rows * common.conv_cols;
  common.conv_mem  = sizeof(fp) * common.conv_elem;
  common.ioffset   = 0;
  common.joffset   = 0;

  common.in2_pad_add_rows     = common.in_rows;
  common.in2_pad_add_cols     = common.in_cols;
  common.in2_pad_cumv_rows    = common.in2_rows + 2*common.in2_pad_add_rows;
  common.in2_pad_cumv_cols    = common.in2_cols + 2*common.in2_pad_add_cols;
  common.in2_pad_cumv_elem    = common.in2_pad_cumv_rows * common.in2_pad_cumv_cols;
  common.in2_pad_cumv_mem     = sizeof(fp) * common.in2_pad_cumv_elem;

  common.in2_pad_cumv_sel_rowlow = 1 + common.in_rows;
  common.in2_pad_cumv_sel_rowhig = common.in2_pad_cumv_rows - 1;
  common.in2_pad_cumv_sel_collow = 1;
  common.in2_pad_cumv_sel_colhig = common.in2_pad_cumv_cols;
  common.in2_pad_cumv_sel_rows   = common.in2_pad_cumv_sel_rowhig - common.in2_pad_cumv_sel_rowlow + 1;
  common.in2_pad_cumv_sel_cols   = common.in2_pad_cumv_sel_colhig - common.in2_pad_cumv_sel_collow + 1;
  common.in2_pad_cumv_sel_elem   = common.in2_pad_cumv_sel_rows * common.in2_pad_cumv_sel_cols;
  common.in2_pad_cumv_sel_mem    = sizeof(fp) * common.in2_pad_cumv_sel_elem;

  common.in2_pad_cumv_sel2_rowlow = 1;
  common.in2_pad_cumv_sel2_rowhig = common.in2_pad_cumv_rows - common.in_rows - 1;
  common.in2_pad_cumv_sel2_collow = 1;
  common.in2_pad_cumv_sel2_colhig = common.in2_pad_cumv_cols;
  common.in2_sub_cumh_rows = common.in2_pad_cumv_sel2_rowhig - common.in2_pad_cumv_sel2_rowlow + 1;
  common.in2_sub_cumh_cols = common.in2_pad_cumv_sel2_colhig - common.in2_pad_cumv_sel2_collow + 1;
  common.in2_sub_cumh_elem = common.in2_sub_cumh_rows * common.in2_sub_cumh_cols;
  common.in2_sub_cumh_mem  = sizeof(fp) * common.in2_sub_cumh_elem;

  common.in2_sub_cumh_sel_rowlow = 1;
  common.in2_sub_cumh_sel_rowhig = common.in2_sub_cumh_rows;
  common.in2_sub_cumh_sel_collow = 1 + common.in_cols;
  common.in2_sub_cumh_sel_colhig = common.in2_sub_cumh_cols - 1;
  common.in2_sub_cumh_sel_rows   = common.in2_sub_cumh_sel_rowhig - common.in2_sub_cumh_sel_rowlow + 1;
  common.in2_sub_cumh_sel_cols   = common.in2_sub_cumh_sel_colhig - common.in2_sub_cumh_sel_collow + 1;
  common.in2_sub_cumh_sel_elem   = common.in2_sub_cumh_sel_rows * common.in2_sub_cumh_sel_cols;
  common.in2_sub_cumh_sel_mem    = sizeof(fp) * common.in2_sub_cumh_sel_elem;

  common.in2_sub_cumh_sel2_rowlow = 1;
  common.in2_sub_cumh_sel2_rowhig = common.in2_sub_cumh_rows;
  common.in2_sub_cumh_sel2_collow = 1;
  common.in2_sub_cumh_sel2_colhig = common.in2_sub_cumh_cols - common.in_cols - 1;
  common.in2_sub2_rows = common.in2_sub_cumh_sel2_rowhig - common.in2_sub_cumh_sel2_rowlow + 1;
  common.in2_sub2_cols = common.in2_sub_cumh_sel2_colhig - common.in2_sub_cumh_sel2_collow + 1;
  common.in2_sub2_elem = common.in2_sub2_rows * common.in2_sub2_cols;
  common.in2_sub2_mem  = sizeof(fp) * common.in2_sub2_elem;

  common.in2_sqr_rows  = common.in2_rows;
  common.in2_sqr_cols  = common.in2_cols;
  common.in2_sqr_elem  = common.in2_elem;
  common.in2_sqr_mem   = common.in2_mem;

  common.in2_sqr_sub2_rows = common.in2_sub2_rows;
  common.in2_sqr_sub2_cols = common.in2_sub2_cols;
  common.in2_sqr_sub2_elem = common.in2_sub2_elem;
  common.in2_sqr_sub2_mem  = common.in2_sub2_mem;

  common.in_sqr_rows = common.in_rows;
  common.in_sqr_cols = common.in_cols;
  common.in_sqr_elem = common.in_elem;
  common.in_sqr_mem  = common.in_mem;

  common.tMask_rows = common.in_rows + (common.sSize+1+common.sSize) - 1;
  common.tMask_cols = common.tMask_rows;
  common.tMask_elem = common.tMask_rows * common.tMask_cols;
  common.tMask_mem  = sizeof(fp) * common.tMask_elem;

  common.mask_rows = common.maxMove;
  common.mask_cols = common.mask_rows;
  common.mask_elem = common.mask_rows * common.mask_cols;
  common.mask_mem  = sizeof(fp) * common.mask_elem;

  common.mask_conv_rows = common.tMask_rows;
  common.mask_conv_cols = common.tMask_cols;
  common.mask_conv_elem = common.mask_conv_rows * common.mask_conv_cols;
  common.mask_conv_mem  = sizeof(fp) * common.mask_conv_elem;
  common.mask_conv_ioffset = (common.mask_rows-1)/2;
  if ((common.mask_rows-1) % 2 > 0.5) common.mask_conv_ioffset++;
  common.mask_conv_joffset = (common.mask_cols-1)/2;
  if ((common.mask_cols-1) % 2 > 0.5) common.mask_conv_joffset++;

  int allPoints = common.allPoints;

  // ─── Allocate device Views ──────────────────────────────────────────────────
  Kokkos::View<fp*> d_endoT    ("endoT",    common.in_elem  * common.endoPoints);
  Kokkos::View<fp*> d_epiT     ("epiT",     common.in_elem  * common.epiPoints);
  Kokkos::View<fp*> d_in2      ("in2",      common.in2_elem * allPoints);
  Kokkos::View<fp*> d_conv     ("conv",     common.conv_elem* allPoints);
  Kokkos::View<fp*> d_in2_pad_cumv    ("in2_pad_cumv",     common.in2_pad_cumv_elem    * allPoints);
  Kokkos::View<fp*> d_in2_pad_cumv_sel("in2_pad_cumv_sel", common.in2_pad_cumv_sel_elem* allPoints);
  Kokkos::View<fp*> d_in2_sub_cumh    ("in2_sub_cumh",     common.in2_sub_cumh_elem    * allPoints);
  Kokkos::View<fp*> d_in2_sub_cumh_sel("in2_sub_cumh_sel", common.in2_sub_cumh_sel_elem* allPoints);
  Kokkos::View<fp*> d_in2_sub2        ("in2_sub2",         common.in2_sub2_elem        * allPoints);
  Kokkos::View<fp*> d_in2_sqr        ("in2_sqr",           common.in2_sqr_elem         * allPoints);
  Kokkos::View<fp*> d_in2_sqr_sub2   ("in2_sqr_sub2",      common.in2_sqr_sub2_elem    * allPoints);
  Kokkos::View<fp*> d_in_sqr         ("in_sqr",            common.in_sqr_elem          * allPoints);
  Kokkos::View<fp*> d_tMask          ("tMask",             common.tMask_elem           * allPoints);
  Kokkos::View<fp*> d_mask_conv      ("mask_conv",         common.mask_conv_elem       * allPoints);
  Kokkos::View<fp*> d_in_mod_temp    ("in_mod_temp",       common.in_elem              * allPoints);
  Kokkos::View<fp*> d_in_partial_sum  ("in_partial_sum",   common.in_cols              * allPoints);
  Kokkos::View<fp*> d_in_sqr_partial_sum("in_sqr_partial_sum", common.in_sqr_rows      * allPoints);
  Kokkos::View<fp*> d_par_max_val    ("par_max_val",       common.mask_conv_rows       * allPoints);
  Kokkos::View<fp*> d_par_max_coo    ("par_max_coo",       common.mask_conv_rows       * allPoints);
  Kokkos::View<fp*> d_in_final_sum   ("in_final_sum",      allPoints);
  Kokkos::View<fp*> d_in_sqr_final_sum("in_sqr_final_sum", allPoints);
  Kokkos::View<fp*> d_denomT         ("denomT",            allPoints);

  // Input point coordinate arrays
  Kokkos::View<int*> d_endoRow    ("endoRow",     common.endoPoints);
  Kokkos::View<int*> d_endoCol    ("endoCol",     common.endoPoints);
  Kokkos::View<int*> d_epiRow     ("epiRow",      common.epiPoints);
  Kokkos::View<int*> d_epiCol     ("epiCol",      common.epiPoints);
  Kokkos::View<int*> d_tEndoRowLoc("tEndoRowLoc", common.endoPoints * common.no_frames);
  Kokkos::View<int*> d_tEndoColLoc("tEndoColLoc", common.endoPoints * common.no_frames);
  Kokkos::View<int*> d_tEpiRowLoc ("tEpiRowLoc",  common.epiPoints  * common.no_frames);
  Kokkos::View<int*> d_tEpiColLoc ("tEpiColLoc",  common.epiPoints  * common.no_frames);
  Kokkos::View<fp*>  d_frame      ("frame",       common.frame_elem);

  // Copy initial point coordinates to device
  {
    auto h_endoRow = Kokkos::create_mirror_view(d_endoRow);
    auto h_endoCol = Kokkos::create_mirror_view(d_endoCol);
    auto h_epiRow  = Kokkos::create_mirror_view(d_epiRow);
    auto h_epiCol  = Kokkos::create_mirror_view(d_epiCol);
    for (int i = 0; i < common.endoPoints; i++) {
      h_endoRow(i) = endoRow[i];
      h_endoCol(i) = endoCol[i];
    }
    for (int i = 0; i < common.epiPoints; i++) {
      h_epiRow(i) = epiRow[i];
      h_epiCol(i) = epiCol[i];
    }
    Kokkos::deep_copy(d_endoRow, h_endoRow);
    Kokkos::deep_copy(d_endoCol, h_endoCol);
    Kokkos::deep_copy(d_epiRow,  h_epiRow);
    Kokkos::deep_copy(d_epiCol,  h_epiCol);
  }

  printf("frame progress: ");
  fflush(NULL);

  for (int frame_no = 0; frame_no < common.frames_processed; frame_no++) {
    // Fetch frame from video and copy to device
    fp* h_frame_data = get_frame(frames, frame_no, 0, 0, 1);
    {
      auto h_frame_v = Kokkos::create_mirror_view(d_frame);
      for (int i = 0; i < common.frame_elem; i++)
        h_frame_v(i) = h_frame_data[i];
      Kokkos::deep_copy(d_frame, h_frame_v);
    }
    free(h_frame_data);

    // ─── Launch kernel: one thread per point (serial-per-point) ──────────────
    // Each point index maps to either an endo or epi point.
    // The kernel body is from heartwall-omp/kernel/kernel.h, transformed so
    // that omp_get_team_num()→bx, omp_get_thread_num()→0, barriers removed,
    // and NUMBER_THREADS=1 for serial element loops.
    Kokkos::parallel_for(
      "heartwall_kernel",
      Kokkos::RangePolicy<>(0, allPoints),
      KOKKOS_LAMBDA(int bx)
      {
        // Get raw pointers from Views for kernel body compatibility
        fp* endoT          = d_endoT.data();
        fp* epiT           = d_epiT.data();
        fp* in2            = d_in2.data();
        fp* conv           = d_conv.data();
        fp* in2_pad_cumv   = d_in2_pad_cumv.data();
        fp* in2_pad_cumv_sel = d_in2_pad_cumv_sel.data();
        fp* in2_sub_cumh   = d_in2_sub_cumh.data();
        fp* in2_sub_cumh_sel= d_in2_sub_cumh_sel.data();
        fp* in2_sub2       = d_in2_sub2.data();
        fp* in2_sqr        = d_in2_sqr.data();
        fp* in2_sqr_sub2   = d_in2_sqr_sub2.data();
        fp* in_sqr         = d_in_sqr.data();
        fp* tMask          = d_tMask.data();
        fp* mask_conv      = d_mask_conv.data();
        fp* in_mod_temp    = d_in_mod_temp.data();
        fp* in_partial_sum  = d_in_partial_sum.data();
        fp* in_sqr_partial_sum = d_in_sqr_partial_sum.data();
        fp* par_max_val    = d_par_max_val.data();
        fp* par_max_coo    = d_par_max_coo.data();
        fp* in_final_sum   = d_in_final_sum.data();
        fp* in_sqr_final_sum= d_in_sqr_final_sum.data();
        fp* denomT         = d_denomT.data();
        fp* frame          = d_frame.data();
        int* endoRow       = d_endoRow.data();
        int* endoCol       = d_endoCol.data();
        int* epiRow        = d_epiRow.data();
        int* epiCol        = d_epiCol.data();
        int* tEndoRowLoc   = d_tEndoRowLoc.data();
        int* tEndoColLoc   = d_tEndoColLoc.data();
        int* tEpiRowLoc    = d_tEpiRowLoc.data();
        int* tEpiColLoc    = d_tEpiColLoc.data();

        // Include the transformed kernel body.
        // kokkos_kernel.h is kernel.h with:
        //   omp_get_team_num() -> bx  (already in scope above)
        //   omp_get_thread_num() -> 0
        //   #pragma omp barrier -> (removed)
        //   ei_new += NUMBER_THREADS -> ei_new += 1  (NUMBER_THREADS is 1 here)
#include "kokkos_kernel.h"
      });

    Kokkos::fence();

    printf("%d ", frame_no);
    fflush(NULL);
  }
  printf("\n");
  fflush(NULL);

  // Copy output location arrays back to host
  {
    auto h_tEndoRowLoc = Kokkos::create_mirror_view(d_tEndoRowLoc);
    auto h_tEndoColLoc = Kokkos::create_mirror_view(d_tEndoColLoc);
    auto h_tEpiRowLoc  = Kokkos::create_mirror_view(d_tEpiRowLoc);
    auto h_tEpiColLoc  = Kokkos::create_mirror_view(d_tEpiColLoc);
    Kokkos::deep_copy(h_tEndoRowLoc, d_tEndoRowLoc);
    Kokkos::deep_copy(h_tEndoColLoc, d_tEndoColLoc);
    Kokkos::deep_copy(h_tEpiRowLoc,  d_tEpiRowLoc);
    Kokkos::deep_copy(h_tEpiColLoc,  d_tEpiColLoc);
    for (int i = 0; i < common.endoPoints * common.no_frames; i++) {
      tEndoRowLoc[i] = h_tEndoRowLoc(i);
      tEndoColLoc[i] = h_tEndoColLoc(i);
    }
    for (int i = 0; i < common.epiPoints * common.no_frames; i++) {
      tEpiRowLoc[i]  = h_tEpiRowLoc(i);
      tEpiColLoc[i]  = h_tEpiColLoc(i);
    }
  }
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    printf("Workgroup size of kernel = %d\n", NUMBER_THREADS);

    if (argc < 3) {
      printf("Usage: %s <video.avi> <num_frames>\n", argv[0]);
      Kokkos::finalize();
      return 0;
    }

    long long time0 = get_time();

    params_common common;
    common.common_mem = sizeof(params_common);

    // Open video
    const char* video_file_name = argv[1];
    avi_t* frames = (avi_t*)AVI_open_input_file(video_file_name, 1);
    if (!frames) {
      AVI_print_error((char*)"Error with AVI_open_input_file");
      Kokkos::finalize();
      return -1;
    }

    common.no_frames   = AVI_video_frames(frames);
    common.frame_rows  = AVI_video_height(frames);
    common.frame_cols  = AVI_video_width(frames);
    common.frame_elem  = common.frame_rows * common.frame_cols;
    common.frame_mem   = sizeof(fp) * common.frame_elem;

    long long time1 = get_time();

    common.frames_processed = atoi(argv[2]);
    if (common.frames_processed < 0 ||
        common.frames_processed > common.no_frames) {
      printf("ERROR: invalid frame count %d (range 0-%d)\n",
             common.frames_processed, common.no_frames);
      Kokkos::finalize();
      return 0;
    }

    long long time2 = get_time();

    // Read parameters from input file (same path as OMP version)
    const char* param_file = "../heartwall-omp/input.txt";
    read_parameters(param_file, &common.tSize, &common.sSize,
                    &common.maxMove, &common.alpha);
    read_header(param_file, &common.endoPoints, &common.epiPoints);
    common.allPoints = common.endoPoints + common.epiPoints;

    common.endo_mem = sizeof(int) * common.endoPoints;
    common.epi_mem  = sizeof(int) * common.epiPoints;

    int* endoRow     = (int*)malloc(common.endo_mem);
    int* endoCol     = (int*)malloc(common.endo_mem);
    int* tEndoRowLoc = (int*)malloc(common.endo_mem * common.no_frames);
    int* tEndoColLoc = (int*)malloc(common.endo_mem * common.no_frames);
    int* epiRow      = (int*)malloc(common.epi_mem);
    int* epiCol      = (int*)malloc(common.epi_mem);
    int* tEpiRowLoc  = (int*)malloc(common.epi_mem  * common.no_frames);
    int* tEpiColLoc  = (int*)malloc(common.epi_mem  * common.no_frames);

    read_data(param_file,
              common.endoPoints, endoRow, endoCol,
              common.epiPoints,  epiRow,  epiCol);

    long long time3 = get_time();

    kernel_gpu_wrapper(common,
                       endoRow, endoCol, tEndoRowLoc, tEndoColLoc,
                       epiRow,  epiCol,  tEpiRowLoc,  tEpiColLoc,
                       frames);

    long long time4 = get_time();

#ifdef OUTPUT
    write_data("result.txt",
               common.no_frames, common.frames_processed,
               common.endoPoints, tEndoRowLoc, tEndoColLoc,
               common.epiPoints,  tEpiRowLoc,  tEpiColLoc);
#endif

    free(endoRow);   free(endoCol);
    free(tEndoRowLoc); free(tEndoColLoc);
    free(epiRow);    free(epiCol);
    free(tEpiRowLoc);  free(tEpiColLoc);

    long long time5 = get_time();

    printf("Time spent in different stages of the application:\n");
    printf("%15.12f s : READ INITIAL VIDEO FRAME\n",   (fp)(time1-time0)/1000000);
    printf("%15.12f s : READ COMMAND LINE PARAMETERS\n",(fp)(time2-time1)/1000000);
    printf("%15.12f s : READ INPUTS FROM FILE\n",       (fp)(time3-time2)/1000000);
    printf("%15.12f s : KOKKOS ALLOCATION, COPYING, COMPUTATION\n",(fp)(time4-time3)/1000000);
    printf("%15.12f s : FREE MEMORY\n",                 (fp)(time5-time4)/1000000);
    printf("Total time: %15.12f s\n",                   (fp)(time5-time0)/1000000);
  }
  Kokkos::finalize();
  return 0;
}
