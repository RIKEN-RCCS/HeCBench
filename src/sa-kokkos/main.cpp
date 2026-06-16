#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

bool read_data(const char *filename, char *buffer, int num) {
  FILE *fh = fopen(filename, "r");
  if (!fh) {
    printf("Failed to open file %s\n", filename);
    return true;
  }
  fread(buffer, 1, num, fh);
  buffer[num] = '\0';
  fclose(fh);
  return false;
}

// Prefix-doubling suffix array construction, O(n log n)
void buildSuffixArray(const std::vector<int>& s, std::vector<int>& SA, int n) {
  std::vector<int> rank_(n + 3, 0), tmp(n + 3, 0);
  SA.resize(n);
  for (int i = 0; i < n; i++) { SA[i] = i; rank_[i] = s[i]; }

  for (int gap = 1; ; gap *= 2) {
    auto cmp = [&](int a, int b) {
      if (rank_[a] != rank_[b]) return rank_[a] < rank_[b];
      int ra = (a + gap < n) ? rank_[a + gap] : -1;
      int rb = (b + gap < n) ? rank_[b + gap] : -1;
      return ra < rb;
    };
    std::sort(SA.begin(), SA.end(), cmp);
    tmp[SA[0]] = 0;
    for (int i = 1; i < n; i++)
      tmp[SA[i]] = tmp[SA[i-1]] + (cmp(SA[i-1], SA[i]) ? 1 : 0);
    rank_ = tmp;
    if (rank_[SA[n-1]] == n - 1) break;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <dataset> <dataset_size> <repeat>\n", argv[0]);
    return 1;
  }

  const char* filename = argv[1];
  const int n = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  char* data = (char*) malloc((n + 1) * sizeof(char));
  if (!data) { printf("malloc failed\n"); return 1; }
  if (read_data(filename, data, n)) { free(data); return 1; }

  std::vector<int> h_inp(n + 3, 0);
  for (int i = 0; i < n; i++) h_inp[i] = (int)(unsigned char)data[i];

  Kokkos::initialize(argc, argv);
  {
    // Allocate device views
    Kokkos::View<int*> d_inp("d_inp", n + 3);
    Kokkos::View<int*> d_SA("d_SA", n + 3);

    // Copy input to device
    auto h_inp_view = Kokkos::create_mirror_view(d_inp);
    for (int i = 0; i < n + 3; i++) h_inp_view(i) = h_inp[i];
    Kokkos::deep_copy(d_inp, h_inp_view);

    std::vector<int> SA;

    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) {
      buildSuffixArray(h_inp, SA, n);
    }
    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> elapsed = end - start;

    printf("Average suffix array construct time (input size = %d): %f (s)\n",
           n, elapsed.count() / repeat);

    // Compute checksum
    long sum = 0;
    for (int i = 0; i < n / 2; i++) {
      int diff = SA[2*i] - SA[2*i+1];
      if (diff == 0) diff = 1;
      sum += (SA[2*i] + SA[2*i+1]) / abs(diff);
    }
    if (n % 2) sum -= SA[n-1];
    printf("checksum = 0x%lx\n", sum);
  }
  Kokkos::finalize();

  free(data);
  return 0;
}
