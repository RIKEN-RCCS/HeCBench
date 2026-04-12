#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

// Inlined from ga-cuda/reference.h
void reference(const char *__restrict__ target,
               const char *__restrict__ query,
                     char *__restrict__ batch_result,
                     uint32_t length,
                     int query_sequence_length,
                     int coarse_match_length,
                     int coarse_match_threshold,
                     int current_position)
{
  for (uint32_t tid = 0; tid < length; tid++) {
    bool match = false;
    int max_length = query_sequence_length - coarse_match_length;
    for (int i = 0; i <= max_length; i++) {
      int distance = 0;
      for (int j = 0; j < coarse_match_length; j++) {
        if (target[current_position + tid + j] != query[i + j]) distance++;
      }
      if (distance < coarse_match_threshold) { match = true; break; }
    }
    if (match) batch_result[tid] = 1;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <target sequence length> <query sequence length> "
           "<coarse match length> <coarse match threshold>\n", argv[0]);
    return 1;
  }

  const int kBatchSize = 1024;
  char seq[] = {'A', 'C', 'T', 'G'};
  const int tseq_size = atoi(argv[1]);
  const int qseq_size = atoi(argv[2]);
  const int coarse_match_length    = atoi(argv[3]);
  const int coarse_match_threshold = atoi(argv[4]);

  std::vector<char> target_sequence(tseq_size);
  std::vector<char> query_sequence(qseq_size);

  srand(123);
  for (int i = 0; i < tseq_size; i++) target_sequence[i] = seq[rand() % 4];
  for (int i = 0; i < qseq_size;  i++) query_sequence[i]  = seq[rand() % 4];

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<char*> d_target("d_target", tseq_size);
    Kokkos::View<char*> d_query("d_query", qseq_size);
    Kokkos::View<char*> d_batch_result("d_batch_result", kBatchSize);

    {
      auto ht = Kokkos::create_mirror_view(d_target);
      auto hq = Kokkos::create_mirror_view(d_query);
      for (int i = 0; i < tseq_size; i++) ht(i) = target_sequence[i];
      for (int i = 0; i < qseq_size;  i++) hq(i) = query_sequence[i];
      Kokkos::deep_copy(d_target, ht);
      Kokkos::deep_copy(d_query, hq);
    }

    uint32_t max_searchable_length = tseq_size - coarse_match_length;
    uint32_t current_position = 0;

    char batch_result_ref[kBatchSize];
    float total_time = 0.f;
    int error = 0;

    while (current_position < max_searchable_length) {
      // Zero batch result on device
      Kokkos::parallel_for("ga_zero", kBatchSize,
        KOKKOS_LAMBDA(int i) { d_batch_result(i) = 0; });
      Kokkos::fence();

      memset(batch_result_ref, 0, kBatchSize);

      uint32_t end_position = current_position + kBatchSize;
      if (end_position >= max_searchable_length) end_position = max_searchable_length;
      uint32_t length = end_position - current_position;

      auto start = std::chrono::steady_clock::now();

      Kokkos::parallel_for("ga",
        Kokkos::RangePolicy<>(0, (int)length),
        KOKKOS_LAMBDA(int tid) {
          bool match = false;
          int max_length = qseq_size - coarse_match_length;
          for (int i = 0; i <= max_length; i++) {
            int distance = 0;
            for (int j = 0; j < coarse_match_length; j++) {
              if (d_target((int)(current_position + tid + j)) != d_query(i + j))
                distance++;
            }
            if (distance < coarse_match_threshold) { match = true; break; }
          }
          if (match) d_batch_result(tid) = 1;
        });
      Kokkos::fence();

      auto end  = std::chrono::steady_clock::now();
      total_time += (float)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      reference(target_sequence.data(), query_sequence.data(), batch_result_ref,
                length, qseq_size, coarse_match_length, coarse_match_threshold,
                current_position);

      auto h_res = Kokkos::create_mirror_view(d_batch_result);
      Kokkos::deep_copy(h_res, d_batch_result);
      error = memcmp(batch_result_ref, h_res.data(), kBatchSize * sizeof(char));
      if (error) break;

      current_position = end_position;
    }

    printf("Total kernel execution time %f (s)\n", total_time * 1e-9f);
    printf("%s\n", error ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  return 0;
}
