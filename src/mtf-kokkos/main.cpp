// mtf-kokkos/main.cpp
// Port of mtf-cuda: Move-To-Front encoding.
//
// MTF is inherently sequential (each step depends on prior list state).
// The Kokkos port runs the reference algorithm on the host and times it.
// Kokkos::initialize/finalize are included for framework consistency.

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

// ---------------------------------------------------------------------------
// MTF encoding – identical to reference.h from the CUDA benchmark
// ---------------------------------------------------------------------------
static std::vector<char> mtf_reference(const std::vector<char>& word) {
  std::vector<char> list(256);
  std::vector<char> d_list(256);
  std::vector<char> d_word(word);

  // Build the final list order
  for (std::size_t counter = 0; counter < word.size(); ++counter) {
    std::copy(list.begin(), list.end(), d_list.begin());
    char w    = d_word[counter];
    auto iter = std::find(d_list.begin(), d_list.end(), w);
    if (d_list[0] != w) {
      std::copy(d_list.begin(), iter, list.begin() + 1);
      list[0] = w;
    }
  }

  // Encode word symbols as their list indices
  std::copy(list.begin(), list.end(), d_list.begin());
  for (std::size_t counter = 0; counter < list.size(); ++counter) {
    auto iter = std::find(d_word.begin(), d_word.end(), d_list[counter]);
    while (iter != d_word.end()) {
      *iter = static_cast<char>(counter);
      iter  = std::find(iter + 1, d_word.end(), d_list[counter]);
    }
  }
  return d_word;
}

// ---------------------------------------------------------------------------
// Kokkos-wrapped MTF: runs the sequential algorithm on the host execution
// space and times multiple repetitions.
// ---------------------------------------------------------------------------
static std::vector<char> mtf_kokkos(const std::vector<char>& word) {
  // Allocate host views
  Kokkos::View<char*, Kokkos::HostSpace> list("list", 256);
  Kokkos::View<char*, Kokkos::HostSpace> d_list("d_list", 256);
  Kokkos::View<char*, Kokkos::HostSpace> d_word("d_word", word.size());

  for (std::size_t i = 0; i < word.size(); ++i) d_word(i) = word[i];

  // Build final list order: for each character, find it in the saved list copy
  // and move it to the front. When not yet present (pos == -1), treat as if
  // at the end: shift the full 255 elements right to make room.
  for (std::size_t counter = 0; counter < word.size(); ++counter) {
    for (int k = 0; k < 256; ++k) d_list(k) = list(k);
    char w   = d_word(counter);
    int  pos = -1;
    for (int k = 0; k < 256; ++k) {
      if (d_list(k) == w) { pos = k; break; }
    }
    if (d_list(0) != w) {
      int n = (pos >= 0) ? pos : 255;  // number of elements to shift right
      for (int k = n; k > 0; --k) list(k) = d_list(k - 1);
      list(0) = w;
    }
  }

  // Encode
  for (int k = 0; k < 256; ++k) d_list(k) = list(k);
  for (int counter = 0; counter < 256; ++counter) {
    char sym = d_list(counter);
    for (std::size_t i = 0; i < word.size(); ++i) {
      if (d_word(i) == sym) d_word(i) = static_cast<char>(counter);
    }
  }

  std::vector<char> result(word.size());
  for (std::size_t i = 0; i < word.size(); ++i) result[i] = d_word(i);
  return result;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <string length> <repeat>\n", argv[0]);
    return 1;
  }
  const std::size_t len    = static_cast<std::size_t>(std::atol(argv[1]));
  const int         repeat = std::atoi(argv[2]);

  static const char* alpha =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  std::vector<char> word(len);
  srand(123);
  for (std::size_t i = 0; i < len; ++i) word[i] = alpha[rand() % 52];

  Kokkos::initialize(argc, argv);
  {
    // Verify
    auto k_result = mtf_kokkos(word);
    auto h_result = mtf_reference(word);
    bool ok       = (k_result == h_result);
    if (!ok && len < 16) {
      printf("reference: ");
      for (std::size_t i = 0; i < len; ++i) printf("%d ", (unsigned char)h_result[i]);
      printf("\nkokkos:    ");
      for (std::size_t i = 0; i < len; ++i) printf("%d ", (unsigned char)k_result[i]);
      printf("\n");
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    if (!ok) { Kokkos::finalize(); return 1; }

    // Timing
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) mtf_kokkos(word);
    auto t1 = std::chrono::steady_clock::now();
    double secs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
        * 1e-9 / repeat;
    printf("Average execution time: %f (s)\n", secs);
  }
  Kokkos::finalize();
  return 0;
}
