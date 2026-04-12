#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <Kokkos_Core.hpp>

const int nContractions = 18;

template <typename T>
void contract(const int max_N, const int max_C, const int repeat) {
  const size_t tensor_size = (size_t)max_N * max_N * max_N * max_C;
  const size_t adj_size    = (size_t)max_N * max_N;
  const size_t output_size = (size_t)max_N * max_N * max_C * nContractions;

  Kokkos::View<T*> d_tensor("tensor", tensor_size);
  Kokkos::View<T*> d_adj("adj", adj_size);
  Kokkos::View<T*> d_value("value", output_size);

  {
    auto h_tensor = Kokkos::create_mirror_view(d_tensor);
    auto h_adj    = Kokkos::create_mirror_view(d_adj);
    for (size_t i = 0; i < tensor_size; i++) h_tensor(i) = T(1);
    for (size_t i = 0; i < adj_size; i++)    h_adj(i)    = T(1);
    Kokkos::deep_copy(d_tensor, h_tensor);
    Kokkos::deep_copy(d_adj, h_adj);
    Kokkos::fence();
  }

  auto start = std::chrono::steady_clock::now();

  for (int iter = 0; iter < repeat; iter++) {
    Kokkos::parallel_for("contraction", (int)output_size,
      KOKKOS_LAMBDA(int tid) {
        int C = max_C;
        int B = max_N * C;
        int A = max_N * B;
        int Y = max_C * nContractions;

        int f    = (tid % Y) % C;
        int Case = (tid % Y) / C + 1;
        int y    = (tid / Y) % max_N;
        int x    = (tid / Y) / max_N;

        int a, b, c, d, e;
        T adj_value;
        T sum = T(0);

        // Case 1: Fix a, b. Contract c, d, e.
        if (Case == 1) {
          a = x;
          b = y;
          for (d = 0; d < max_N; ++d) {
            for (e = 0; e < max_N; ++e) {
              adj_value = d_adj[d * max_N + e];
              if (adj_value > 0) {
                for (c = 0; c < max_N; ++c) {
                  sum += d_tensor[a * A + b * B + c * C + f] * adj_value;
                }
              }
            }
          }
        }

        // Case 2: Fix a, d. Contract b, c, e.
        if (Case == 2) {
          a = x;
          d = y;
          for (e = 0; e < max_N; ++e) {
            adj_value = d_adj[d * max_N + e];
            if (adj_value > 0) {
              for (b = 0; b < max_N; ++b) {
                for (c = 0; c < max_N; ++c) {
                  sum += d_tensor[a * A + b * B + c * C + f] * adj_value;
                }
              }
            }
          }
        }

        // Case 3: Fix b, c. Contract a, d, e.
        if (Case == 3) {
          b = x;
          c = y;
          for (d = 0; d < max_N; ++d) {
            for (e = 0; e < max_N; ++e) {
              adj_value = d_adj[d * max_N + e];
              if (adj_value > 0) {
                for (a = 0; a < max_N; ++a) {
                  sum += d_tensor[a * A + b * B + c * C + f] * adj_value;
                }
              }
            }
          }
        }

        // Case 4: Fix b, d. Contract a, c, e.
        if (Case == 4) {
          b = x;
          d = y;
          for (e = 0; e < max_N; ++e) {
            adj_value = d_adj[d * max_N + e];
            if (adj_value > 0) {
              for (a = 0; a < max_N; ++a) {
                for (c = 0; c < max_N; ++c) {
                  sum += d_tensor[a * A + b * B + c * C + f] * adj_value;
                }
              }
            }
          }
        }

        // Case 5: Fix d, e. Contract a, b, c.
        if (Case == 5) {
          d = x;
          e = y;
          adj_value = d_adj[d * max_N + e];
          if (adj_value > 0) {
            for (a = 0; a < max_N; ++a) {
              for (b = 0; b < max_N; ++b) {
                for (c = 0; c < max_N; ++c) {
                  sum += d_tensor[a * A + b * B + c * C + f] * adj_value;
                }
              }
            }
          }
        }

        // Case 6: (a, b). Contract (c, d). Singleton (e).
        if (Case == 6) {
          a = x;
          b = y;
          for (d = 0; d < max_N; ++d) {
            for (e = 0; e < max_N; ++e) {
              adj_value = d_adj[d * max_N + e];
              c = d;
              sum += d_tensor[a * A + b * B + c * C + f] * adj_value;
            }
          }
        }

        // Case 7: (a, b). Contract (d, e). Singleton (c).
        if (Case == 7) {
          a = x;
          b = y;
          for (d = 0; d < max_N; ++d) {
            e = d;
            adj_value = d_adj[d * max_N + e];
            if (adj_value > 0) {
              for (c = 0; c < max_N; ++c) {
                sum += d_tensor[a * A + b * B + c * C + f] * adj_value;
              }
            }
          }
        }

        // Case 8: (a, d). Contract (b, c). Singleton (e).
        if (Case == 8) {
          a = x;
          d = y;
          for (e = 0; e < max_N; ++e) {
            adj_value = d_adj[d * max_N + e];
            if (adj_value > 0) {
              for (b = 0; b < max_N; ++b) {
                c = b;
                sum += d_tensor[a * A + b * B + c * C + f] * adj_value;
              }
            }
          }
        }

        // Case 9: (a, d). Contract (b, e). Singleton (c).
        if (Case == 9) {
          a = x;
          d = y;
          for (e = 0; e < max_N; ++e) {
            adj_value = d_adj[d * max_N + e];
            if (adj_value > 0) {
              b = e;
              for (c = 0; c < max_N; ++c) {
                sum += d_tensor[a * A + b * B + c * C + f] * adj_value;
              }
            }
          }
        }

        // Case 10: (b, c). Contract (a, d). Singleton (e).
        if (Case == 10) {
          b = x;
          c = y;
          for (d = 0; d < max_N; ++d) {
            for (e = 0; e < max_N; ++e) {
              adj_value = d_adj[d * max_N + e];
              if (adj_value > 0) {
                a = d;
                sum += d_tensor[a * A + b * B + c * C + f] * adj_value;
              }
            }
          }
        }

        // Case 11: (b, d). Contract (a, c). Singleton (e).
        if (Case == 11) {
          b = x;
          d = y;
          for (e = 0; e < max_N; ++e) {
            adj_value = d_adj[d * max_N + e];
            if (adj_value > 0) {
              for (a = 0; a < max_N; ++a) {
                c = a;
                sum += d_tensor[a * A + b * B + c * C + f] * adj_value;
              }
            }
          }
        }

        // Case 12: (b, d). Contract (a, e). Singleton (c).
        if (Case == 12) {
          b = x;
          d = y;
          for (e = 0; e < max_N; ++e) {
            adj_value = d_adj[d * max_N + e];
            if (adj_value > 0) {
              a = e;
              for (int lc = 0; lc < max_N; ++lc) {
                sum += d_tensor[a * A + b * B + lc * C + f] * adj_value;
              }
            }
          }
        }

        // Case 13: (b, d). Contract (c, e). Singleton (a).
        if (Case == 13) {
          b = x;
          d = y;
          for (e = 0; e < max_N; ++e) {
            adj_value = d_adj[d * max_N + e];
            if (adj_value > 0) {
              c = e;
              for (int la = 0; la < max_N; ++la) {
                sum += d_tensor[la * A + b * B + c * C + f] * adj_value;
              }
            }
          }
        }

        // Case 14: (d, e). Contract (a, b). Singleton (c).
        if (Case == 14) {
          d = x;
          e = y;
          adj_value = d_adj[d * max_N + e];
          if (adj_value > 0) {
            for (int la = 0; la < max_N; ++la) {
              b = la;
              for (int lc = 0; lc < max_N; ++lc) {
                sum += d_tensor[la * A + b * B + lc * C + f] * adj_value;
              }
            }
          }
        }

        // Case 15: (d, e). Contract (b, c). Singleton (a).
        if (Case == 15) {
          d = x;
          e = y;
          adj_value = d_adj[d * max_N + e];
          if (adj_value > 0) {
            for (int lb = 0; lb < max_N; ++lb) {
              c = lb;
              for (int la = 0; la < max_N; ++la) {
                sum += d_tensor[la * A + lb * B + c * C + f] * adj_value;
              }
            }
          }
        }

        // Case 16: (a, d). Contract (b, c, e).
        if (Case == 16) {
          a = x;
          d = y;
          for (int le = 0; le < max_N; ++le) {
            adj_value = d_adj[d * max_N + le];
            if (adj_value > 0) {
              b = le;
              c = le;
              sum += d_tensor[a * A + b * B + c * C + f] * adj_value;
            }
          }
        }

        // Case 17: (b, d). Contract (a, c, e).
        if (Case == 17) {
          b = x;
          d = y;
          for (int le = 0; le < max_N; ++le) {
            adj_value = d_adj[d * max_N + le];
            if (adj_value > 0) {
              a = le;
              c = le;
              sum += d_tensor[a * A + b * B + c * C + f] * adj_value;
            }
          }
        }

        // Case 18: (d, e). Contract (a, b, c).
        if (Case == 18) {
          d = x;
          e = y;
          adj_value = d_adj[d * max_N + e];
          if (adj_value > 0) {
            for (int la = 0; la < max_N; ++la) {
              b = la;
              c = la;
              sum += d_tensor[la * A + b * B + c * C + f] * adj_value;
            }
          }
        }

        d_value[tid] = sum;
      });
    Kokkos::fence();
  }

  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  auto h_value = Kokkos::create_mirror_view(d_value);
  Kokkos::deep_copy(h_value, d_value);

  double checksum = 0;
  T min_val = h_value(0);
  T max_val = h_value(0);
  for (size_t i = 0; i < output_size; i++) {
    checksum += (double)h_value(i);
    if (h_value(i) < min_val) min_val = h_value(i);
    if (h_value(i) > max_val) max_val = h_value(i);
  }
  printf("Checksum: %lf min:%lf max:%lf\n", checksum,
         (double)min_val, (double)max_val);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <dimension> <repeat>\n", argv[0]);
    return 1;
  }

  int max_N  = atoi(argv[1]);
  int max_C  = nContractions;
  int repeat = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    contract<float> (max_N, max_C, repeat);
    contract<double>(max_N, max_C, repeat);
  }
  Kokkos::finalize();
  return 0;
}
