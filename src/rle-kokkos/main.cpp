// Run-Length Encoding benchmark (Kokkos port of CUB DeviceRunLengthEncode)
#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

// Parse command line: --n=val, --i=val
static int parseArg(int argc, char** argv, const char* name, int def)
{
  int len = (int)strlen(name);
  for (int a = 1; a < argc; a++) {
    if (strncmp(argv[a], name, len) == 0 && argv[a][len] == '=')
      return atoi(argv[a] + len + 1);
  }
  return def;
}

// CPU reference RLE: returns number of runs
static int rle_cpu(const int* in, int n,
                   int* uniq, int* offsets, int* lengths)
{
  if (n == 0) return 0;
  int runs = 0;
  int i = 0;
  while (i < n) {
    int j = i + 1;
    while (j < n && in[j] == in[i]) j++;
    uniq[runs]    = in[i];
    offsets[runs] = i;
    lengths[runs] = j - i;
    runs++;
    i = j;
  }
  return runs;
}

// Fill input with runs of repeated values (entropy_reduction controls run length)
static void initInput(int* h_in, int num_items, int max_segment)
{
  unsigned int max_int = (unsigned int)-1;
  int key = 0, i = 0;
  srand(12345);
  while (i < num_items) {
    int repeat;
    if (max_segment < 2) {
      repeat = 1;
    } else {
      unsigned int r;
      r = (unsigned int)rand();
      repeat = (int)((double)r * (double)max_segment / (double)max_int);
      if (repeat < 1) repeat = 1;
    }
    int end = i + repeat;
    if (end > num_items) end = num_items;
    for (int k = i; k < end; k++) h_in[k] = key;
    i = end;
    key++;
  }
}

int main(int argc, char** argv)
{
  int num_items       = parseArg(argc, argv, "--n", 1000000);
  int timing_iters    = parseArg(argc, argv, "--i", 1);

  if (num_items <= 0) num_items = 1000000;
  if (timing_iters <= 0) timing_iters = 1;

  Kokkos::initialize(argc, argv);
  {
    // Host input
    int* h_in      = new int[num_items];
    int* h_uniq    = new int[num_items];
    int* h_offsets = new int[num_items];
    int* h_lengths = new int[num_items];

    // Reference
    int* ref_uniq    = new int[num_items];
    int* ref_offsets = new int[num_items];
    int* ref_lengths = new int[num_items];

    // Test a few segment sizes
    const int max_seg_limit = (num_items < (1 << 16)) ? num_items : (1 << 16);
    for (int max_segment = 1; max_segment <= max_seg_limit; max_segment <<= 4) {
      if (max_segment < 1) max_segment = 1;
      initInput(h_in, num_items, max_segment);

      int ref_runs = rle_cpu(h_in, num_items, ref_uniq, ref_offsets, ref_lengths);
      float avg_run = (ref_runs > 0) ? (float)num_items / ref_runs : (float)num_items;

      printf("\nTest pointer: %d items, %d segments (avg run length %.3f), "
             "{int key, int offset, int length}, "
             "max_segment %d\n",
             num_items, ref_runs, avg_run, max_segment);
      fflush(stdout);

      // Kokkos views
      Kokkos::View<int*> d_in("d_in", num_items);
      Kokkos::View<int*> d_flags("d_flags", num_items);
      Kokkos::View<int*> d_scan("d_scan", num_items + 1);
      Kokkos::View<int*> d_uniq("d_uniq", num_items);
      Kokkos::View<int*> d_lengths("d_lengths", num_items);
      Kokkos::View<int*> d_num_runs("d_num_runs", 1);

      auto h_d_in = Kokkos::create_mirror_view(d_in);
      for (int i = 0; i < num_items; i++) h_d_in(i) = h_in[i];
      Kokkos::deep_copy(d_in, h_d_in);

      auto start = std::chrono::steady_clock::now();

      for (int iter = 0; iter < timing_iters; iter++) {
        // Step 1: mark run starts
        Kokkos::parallel_for("MarkFlags", num_items, KOKKOS_LAMBDA(int i) {
          d_flags(i) = (i == 0 || d_in(i) != d_in(i - 1)) ? 1 : 0;
        });

        // Step 2: exclusive scan of flags to get positions
        Kokkos::parallel_scan("ScanFlags", num_items + 1,
          KOKKOS_LAMBDA(int i, int& update, bool final) {
            int val = (i < num_items) ? d_flags(i) : 0;
            if (final) d_scan(i) = update;
            update += val;
          });

        // Step 3: scatter unique values and compute lengths
        Kokkos::parallel_for("Scatter", num_items, KOKKOS_LAMBDA(int i) {
          if (d_flags(i)) {
            int pos = d_scan(i);
            d_uniq(pos) = d_in(i);
          }
        });

        Kokkos::parallel_for("Lengths", num_items, KOKKOS_LAMBDA(int i) {
          if (d_flags(i)) {
            int pos  = d_scan(i);
            int next = d_scan(i + 1);
            // Find end of this run
            int run_end = (next > pos + 1) ? i : num_items - 1;
            // Use the scan to get run length: next_flag_pos - current_pos
            // But we need the next flag position, not next scan value
            // Use a simpler: length = scan[i+1..] - scan[i] by looking at next segment
            (void)run_end;
            // We'll compute lengths in a separate pass using d_scan
          }
        });

        // Compute lengths from flag positions using a second pass
        Kokkos::parallel_for("ComputeLengths", num_items, KOKKOS_LAMBDA(int i) {
          if (d_flags(i)) {
            int pos = d_scan(i);
            // find the start of the next run
            int next_start = num_items;
            for (int j = i + 1; j <= num_items; j++) {
              if (j == num_items || d_flags(j)) { next_start = j; break; }
            }
            d_lengths(pos) = next_start - i;
          }
        });

        // Store num_runs
        int n = num_items;
        Kokkos::parallel_for("NumRuns", 1, KOKKOS_LAMBDA(int) {
          d_num_runs(0) = d_scan(n);
        });

        Kokkos::fence();
      }

      auto end  = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      // Copy results back
      auto h_uniq_v    = Kokkos::create_mirror_view(d_uniq);
      auto h_lengths_v = Kokkos::create_mirror_view(d_lengths);
      auto h_num_runs_v = Kokkos::create_mirror_view(d_num_runs);
      Kokkos::deep_copy(h_uniq_v, d_uniq);
      Kokkos::deep_copy(h_lengths_v, d_lengths);
      Kokkos::deep_copy(h_num_runs_v, d_num_runs);

      int got_runs = h_num_runs_v(0);

      // Verify
      bool ok_keys = (got_runs == ref_runs);
      if (ok_keys) {
        for (int i = 0; i < ref_runs; i++) {
          if (h_uniq_v(i) != ref_uniq[i]) { ok_keys = false; break; }
        }
      }
      bool ok_lengths = (got_runs == ref_runs);
      if (ok_lengths) {
        for (int i = 0; i < ref_runs; i++) {
          if (h_lengths_v(i) != ref_lengths[i]) { ok_lengths = false; break; }
        }
      }
      bool ok_count = (got_runs == ref_runs);

      printf("\t Keys %s\n",    ok_keys    ? "PASS" : "FAIL");
      printf("\t Lengths %s\n", ok_lengths ? "PASS" : "FAIL");
      printf("\t Count %s\n",   ok_count   ? "PASS" : "FAIL");
      fflush(stdout);

      if (timing_iters > 0) {
        float avg_ms   = (float)time * 1e-6f / timing_iters;
        float giga_rate = (float)num_items / avg_ms / 1e9f;
        int bytes_moved = num_items * (int)sizeof(int)
                        + ref_runs * (int)(sizeof(int) + sizeof(int));
        float giga_bw   = (float)bytes_moved / avg_ms / 1e9f;
        printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s\n\n",
               avg_ms, giga_rate, giga_bw);
      } else {
        printf("\n\n");
      }
      fflush(stdout);
    }

    delete[] h_in;
    delete[] h_uniq;
    delete[] h_offsets;
    delete[] h_lengths;
    delete[] ref_uniq;
    delete[] ref_offsets;
    delete[] ref_lengths;
  }
  Kokkos::finalize();
  return 0;
}
