#include <chrono>
#include <cstring>
#include <cstdio>
#include <functional>
#include <numeric>
#include <vector>
#include <iostream>
#include <Kokkos_Core.hpp>

// Note: The range 'A'-'z' (0x41-0x7A) is preserved from the original OpenMP reference
// (wordcount-omp/wc.cpp) for a faithful port. This range includes non-alphabetic characters
// between 'Z' (0x5A) and 'a' (0x61): '[', '\', ']', '^', '_', '`'.
// Standard alphabetic checking would use (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'),
// but this implementation intentionally matches the reference for correctness comparison.
KOKKOS_INLINE_FUNCTION
bool is_alpha_dev(const char c) {
  return (c >= 'A' && c <= 'z');
}

inline bool is_alpha(const char c) {
  return (c >= 'A' && c <= 'z');
}

struct is_word_start {
  bool operator()(const char& left, const char& right) const {
    return is_alpha(right) && !is_alpha(left);
  }
};

int word_count_reference(const std::vector<char>& input) {
  if (input.empty()) return 0;
  int wc = std::inner_product(
      input.cbegin(), input.cend() - 1,
      input.cbegin() + 1,
      0,
      std::plus<int>(),
      is_word_start());
  if (is_alpha(input.front())) wc++;
  return wc;
}

int word_count(const std::vector<char>& input) {
  if (input.empty()) return 0;

  const size_t size = input.size();

  Kokkos::View<char*> d_in("d_in", size);
  auto h_in = Kokkos::create_mirror_view(d_in);
  for (size_t i = 0; i < size; i++) h_in(i) = input[i];
  Kokkos::deep_copy(d_in, h_in);

  int wc = 0;
  Kokkos::parallel_reduce("wordcount", (int)(size - 1),
    KOKKOS_LAMBDA(const int i, int& lwc) {
      if (!is_alpha_dev(d_in(i)) && is_alpha_dev(d_in(i + 1))) {
        lwc++;
      }
    }, wc);
  Kokkos::fence();

  if (is_alpha(input[0])) wc++;
  return wc;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    const char raw_input[] =
      "  But the raven, sitting lonely on the placid bust, spoke only,\n"
      "  That one word, as if his soul in that one word he did outpour.\n"
      "  Nothing further then he uttered - not a feather then he fluttered -\n"
      "  Till I scarcely more than muttered `Other friends have flown before -\n"
      "  On the morrow he will leave me, as my hopes have flown before.'\n"
      "  Then the bird said, `Nevermore.'\n";

    std::cout << "Text sample:" << std::endl;
    std::cout << raw_input << std::endl;

    std::vector<char> input(raw_input, raw_input + sizeof(raw_input));

    int wc = word_count_reference(input);
    std::cout << "Host: Text sample contains " << wc << " words" << std::endl;

    wc = word_count(input);
    std::cout << "Device: Text sample contains " << wc << " words" << std::endl;

    std::cout << "Test word count with random inputs\n";
    srand(123);
    bool ok = true;
    const char tab[] = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const int tabsize = (int)strlen(tab);

    for (size_t i = 1; i <= (size_t)1e8; i = i * 10) {
      std::vector<char> random_input(i);
      for (size_t c = 0; c < i; c++) random_input[c] = tab[rand() % tabsize];
      if (word_count_reference(random_input) != word_count(random_input)) {
        ok = false;
        break;
      }
    }
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;

    const size_t len = 1024ULL * 1024 * 1024;
    std::vector<char> random_input(len);
    for (size_t c = 0; c < len; c++) random_input[c] = tab[rand() % tabsize];

    std::cout << "Performance evaluation for random texts of character length "
              << len << std::endl;

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) word_count(random_input);
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average time of word count: "
              << time * 1e-9f / repeat << " (s)\n";
  }
  Kokkos::finalize();
  return 0;
}
