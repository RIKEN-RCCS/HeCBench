// OpenACC (no acc barrier) version of the uploaded OpenMP sosfilt code
// Strategy:
//   - 1 gang  <=> 1 OpenMP team (signal)
//   - 1 vector lane <=> 1 OpenMP thread (section)
//   - Remove barriers by double-buffering s_out (prev/next) and swapping each step
//
// This keeps parallelism (gang/vector) and avoids acc barrier (unsupported by nvc++).

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#include <openacc.h>

/*------------------------*/
//Additional code for check results
/*------------------------*/
#include <cstdint>
#include <cstring>
#include <limits>

template <typename T>
struct ResultDigest {
  double sum;
  double sumsq;
  T minv;
  T maxv;
  long long nan_count;
  long long inf_count;
  uint64_t hash64;
};

static inline uint64_t fnv1a64_update(uint64_t h, const void* data, size_t nbytes) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < nbytes; i++) {
    h ^= (uint64_t)p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

template <typename T>
ResultDigest<T> make_digest(const T* a, int n) {
  ResultDigest<T> d{};
  d.sum = 0.0;
  d.sumsq = 0.0;
  d.minv = std::numeric_limits<T>::infinity();
  d.maxv = -std::numeric_limits<T>::infinity();
  d.nan_count = 0;
  d.inf_count = 0;
  d.hash64 = 14695981039346656037ULL; // FNV-1a offset basis

  for (int i = 0; i < n; i++) {
    T v = a[i];
    if (std::isnan((double)v)) d.nan_count++;
    if (std::isinf((double)v)) d.inf_count++;

    if (v < d.minv) d.minv = v;
    if (v > d.maxv) d.maxv = v;

    d.sum   += (double)v;
    d.sumsq += (double)v * (double)v;

    d.hash64 = fnv1a64_update(d.hash64, &v, sizeof(T));
  }
  return d;
}

template <typename T>
void print_digest(const char* tag, const ResultDigest<T>& d) {
  printf("[%s]\n min=% .9e\n max=% .9e\n sum=% .17e\n sumsq=% .17e\n nan=%lld\n inf=%lld\n hash64=0x%016llx\n",
         tag,
         (double)d.minv, (double)d.maxv, d.sum, d.sumsq,
         d.nan_count, d.inf_count,
         (unsigned long long)d.hash64);
}
/*------------------------*/

///////////////////////////////////////////////////////////////////////////////
//                                SOSFILT                                    //
///////////////////////////////////////////////////////////////////////////////
#define MAX_THREADS 256
#define THREADS 32
#define sos_width  6   // https://www.mathworks.com/help/signal/ref/sosfilt.html

template <typename T>
void filtering (const int repeat, const int n_signals, const int n_samples,
                const int n_sections, const int zi_width)
{
  assert(MAX_THREADS >= n_sections);
  assert(n_samples >= n_sections);

  srand(2);
  const int blocks = n_signals;

  const int sos_size = n_sections * sos_width;

  T* sos = (T*) malloc(sizeof(T) * sos_size);
  for (int i = 0; i < n_sections; i++)
    for (int j = 0; j < sos_width; j++)
      sos[i*sos_width + j] = (T)1;

  const int z_size = (n_sections + 1) * blocks * zi_width;
  T* zi = (T*) malloc(sizeof(T) * z_size);
  for (int i = 0; i < z_size; i++) zi[i] = (T)1;

  const int x_size = n_signals * n_samples;
  T* x_in = (T*) malloc(sizeof(T) * x_size);
  for (int i = 0; i < n_signals; i++)
    for (int j = 0; j < n_samples; j++)
      x_in[i*n_samples + j] = (T)sin(2*3.14*(i+1+j));

#ifdef DEBUG
  // OpenMP shared_mem_size = 32 + (32+1)*2*2 + 32*6 = 356
  static constexpr int shared_mem_size_omp = 32 + (32+1)*2*2 + 32*6;
#else
  // OpenMP shared_mem_size = 32 + (32+1)*8*2 + 32*6 = 752
  static constexpr int shared_mem_size_omp = 32 + (32+1)*8*2 + 32*6;
#endif

  // We need one more s_out buffer (n_sections elements).
  static constexpr int shared_mem_size_acc = shared_mem_size_omp + THREADS; // +n_sections (==THREADS here)

#pragma acc data copyin(sos[0:sos_size], zi[0:z_size]) copy(x_in[0:x_size])
  {
    auto start = std::chrono::steady_clock::now();

    for (int rep = 0; rep < repeat; rep++) {

      // teams/threads <=> gangs/vectors
#pragma acc parallel num_gangs(blocks) vector_length(THREADS) default(present)
      {
#pragma acc loop gang
        for (int ty = 0; ty < blocks; ty++) {

          // Guard (same as OpenMP)
          if (ty >= n_signals) continue;

          // gang-local scratch (acts like team shared memory)
          T smem[shared_mem_size_acc];

          // Double-buffer s_out
          T *s_out0 = smem;                 // [0 .. n_sections-1]
          T *s_out1 = &s_out0[n_sections];  // [n_sections .. 2*n_sections-1]
          T *s_zi   = &s_out1[n_sections];  // matches OpenMP layout after expanding
          T *s_sos  = &s_zi[n_sections * zi_width];

          // Reset both s_out buffers, load zi, load sos
#pragma acc loop vector
          for (int tx = 0; tx < THREADS; tx++) {
            if (tx < n_sections) {
              s_out0[tx] = (T)0;
              s_out1[tx] = (T)0;

              for (int i = 0; i < zi_width; i++) {
                s_zi[tx * zi_width + i] =
                  zi[ty * n_sections * zi_width + tx * zi_width + i];
              }

#pragma acc loop seq
              for (int i = 0; i < sos_width; i++) {
                s_sos[tx * sos_width + i] = sos[tx * sos_width + i];
              }
            }
          }
          // No barrier needed here: end of the vector loop is a convergence point.

          const int load_size   = n_sections - 1;
          const int unload_size = n_samples - load_size;

          // step counter for selecting prev/next buffers consistently across phases
          int step = 0;

          // -----------------------------
          // Loading phase
          // -----------------------------
          for (int n = 0; n < load_size; n++, step++) {
            T *prev = (step & 1) ? s_out1 : s_out0;
            T *next = (step & 1) ? s_out0 : s_out1;

#pragma acc loop vector
            for (int tx = 0; tx < THREADS; tx++) {
              if (tx < n_sections) {

                T x_n = (tx == 0) ? x_in[ty * n_samples + n] : prev[tx - 1];

                T temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

                s_zi[tx * zi_width + 0] =
                  s_sos[tx * sos_width + 1] * x_n
                  - s_sos[tx * sos_width + 4] * temp
                  + s_zi[tx * zi_width + 1];

                s_zi[tx * zi_width + 1] =
                  s_sos[tx * sos_width + 2] * x_n
                  - s_sos[tx * sos_width + 5] * temp;

                // Loading updates s_out for all tx (OpenMP: s_out[tx] = temp)
                next[tx] = temp;
              }
            }
            // No barrier: all reads are from prev, all writes go to next.
          }

          // -----------------------------
          // Processing phase
          // -----------------------------
          for (int n = load_size; n < n_samples; n++, step++) {
            T *prev = (step & 1) ? s_out1 : s_out0;
            T *next = (step & 1) ? s_out0 : s_out1;

#pragma acc loop vector
            for (int tx = 0; tx < THREADS; tx++) {
              if (tx < n_sections) {

                T x_n = (tx == 0) ? x_in[ty * n_samples + n] : prev[tx - 1];

                T temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

                s_zi[tx * zi_width + 0] =
                  s_sos[tx * sos_width + 1] * x_n
                  - s_sos[tx * sos_width + 4] * temp
                  + s_zi[tx * zi_width + 1];

                s_zi[tx * zi_width + 1] =
                  s_sos[tx * sos_width + 2] * x_n
                  - s_sos[tx * sos_width + 5] * temp;

                if (tx < load_size) {
                  // OpenMP: s_out[tx] = temp
                  next[tx] = temp;
                } else {
                  // OpenMP: s_out[tx] NOT written -> keep previous value
                  next[tx] = prev[tx];
                  x_in[ty * n_samples + (n - load_size)] = temp;
                }
              }
            }
          }

          // -----------------------------
          // Unloading phase
          // -----------------------------
          for (int n = 0; n < n_sections; n++, step++) {
            T *prev = (step & 1) ? s_out1 : s_out0;
            T *next = (step & 1) ? s_out0 : s_out1;

#pragma acc loop vector
            for (int tx = 0; tx < THREADS; tx++) {
              if (tx < n_sections) {

                if (tx > n) {
                  T x_n = prev[tx - 1];

                  T temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

                  s_zi[tx * zi_width + 0] =
                    s_sos[tx * sos_width + 1] * x_n
                    - s_sos[tx * sos_width + 4] * temp
                    + s_zi[tx * zi_width + 1];

                  s_zi[tx * zi_width + 1] =
                    s_sos[tx * sos_width + 2] * x_n
                    - s_sos[tx * sos_width + 5] * temp;

                  if (tx < load_size) {
                    next[tx] = temp;
                  } else {
                    next[tx] = prev[tx];
                    x_in[ty * n_samples + (n + unload_size)] = temp;
                  }
                } else {
                  // OpenMP: if(tx>n) else does nothing -> keep value
                  next[tx] = prev[tx];
                }
              }
            }
          }

        } // ty
      }   // acc parallel

    } // repeat

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %lf (s)\n", time * 1e-9 / repeat);
  } // acc data

  /*-----------------------------*/
  //Additional code for check results
  auto dig = make_digest<T>(x_in, x_size);
  print_digest("ACC", dig);
  /*-----------------------------*/

#ifdef DEBUG
  for (int i = 0; i < n_signals; i++) {
    for (int j = 0; j < n_samples; j++)
      printf("%.2f ", (double)x_in[i*n_samples + j]);
    printf("\n");
  }
#endif

  free(x_in);
  free(sos);
  free(zi);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int numSections = THREADS;

#ifdef DEBUG
  const int numSignals = 2;
  const int numSamples = THREADS + 1;
#else
  const int numSignals = 8;
  const int numSamples = 100000;
#endif

  const int zi_width = 2;
  filtering<float> (repeat, numSignals, numSamples, numSections, zi_width);
  filtering<double>(repeat, numSignals, numSamples, numSections, zi_width);
  return 0;
}
