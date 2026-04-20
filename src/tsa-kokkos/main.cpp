#include <complex>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <Kokkos_Core.hpp>

using exec_space   = Kokkos::DefaultExecutionSpace;
using mem_space    = exec_space::memory_space;
using ScratchSpace = exec_space::scratch_memory_space;

// ---------------------------------------------------------------------------
// CPU reference kernels (sequential, from reference.h of tsa-cuda)
// ---------------------------------------------------------------------------
template <typename T>
static void ref_kernel1(T *p_real, T *p_imag, float a, float b, int width, int height)
{
  for (int x = 0, peer = width; x < width; x += 2, peer += 2) {
    T tr = p_real[x], ti = p_imag[x];
    p_real[x]    = a * tr    - b * p_imag[peer];
    p_imag[x]    = a * ti    + b * p_real[peer];
    p_real[peer] = a * p_real[peer] - b * ti;
    p_imag[peer] = a * p_imag[peer] + b * tr;
  }
  for (int y = 1; y < height - 1; y++) {
    for (int idx = y*width + y%2, peer = idx + width; idx < (y+1)*width; idx += 2, peer += 2) {
      T tr = p_real[idx], ti = p_imag[idx];
      p_real[idx]  = a * p_real[idx]  - b * p_imag[peer];
      p_imag[idx]  = a * p_imag[idx]  + b * p_real[peer];
      p_real[peer] = a * p_real[peer] - b * ti;
      p_imag[peer] = a * p_imag[peer] + b * tr;
    }
  }
}

template <typename T>
static void ref_kernel2(T *p_real, T *p_imag, float a, float b, int width, int height)
{
  for (int y = 0; y < height; y++) {
    for (int idx = y*width + y%2, peer = idx + 1; idx < (y+1)*width - 1; idx += 2, peer += 2) {
      T tr = p_real[idx], ti = p_imag[idx];
      p_real[idx]  = a * tr   - b * p_imag[peer];
      p_imag[idx]  = a * ti   + b * p_real[peer];
      p_real[peer] = a * p_real[peer] - b * ti;
      p_imag[peer] = a * p_imag[peer] + b * tr;
    }
  }
}

template <typename T>
static void ref_kernel3(T *p_real, T *p_imag, float a, float b, int width, int height)
{
  for (int x = 1, peer = width + 1; x < width; x += 2, peer += 2) {
    float tr = p_real[x], ti = p_imag[x];
    p_real[x]    = a * tr          - b * p_imag[peer];
    p_imag[x]    = a * ti          + b * p_real[peer];
    p_real[peer] = a * p_real[peer] - b * ti;
    p_imag[peer] = a * p_imag[peer] + b * tr;
  }
  for (int y = 1; y < height - 1; y++) {
    for (int idx = y*width + 1 - y%2, peer = idx + width; idx < (y+1)*width; idx += 2, peer += 2) {
      float tr = p_real[idx], ti = p_imag[idx];
      p_real[idx]  = a * tr          - b * p_imag[peer];
      p_imag[idx]  = a * ti          + b * p_real[peer];
      p_real[peer] = a * p_real[peer] - b * ti;
      p_imag[peer] = a * p_imag[peer] + b * tr;
    }
  }
}

template <typename T>
static void ref_kernel4(T *p_real, T *p_imag, float a, float b, int width, int height)
{
  for (int y = 0; y < height; y++) {
    for (int idx = y*width + 1 - y%2, peer = idx + 1; idx < (y+1)*width - 1; idx += 2, peer += 2) {
      T tr = p_real[idx], ti = p_imag[idx];
      p_real[idx]  = a * tr   - b * p_imag[peer];
      p_imag[idx]  = a * ti   + b * p_real[peer];
      p_real[peer] = a * p_real[peer] - b * ti;
      p_imag[peer] = a * p_imag[peer] + b * tr;
    }
  }
}

template <typename T>
static void reference(T *p_real, T *p_imag, float a, float b, int width, int height, int repeat)
{
  for (int i = 0; i < repeat; i++) {
    ref_kernel1(p_real, p_imag, a, b, width, height);
    ref_kernel2(p_real, p_imag, a, b, width, height);
    ref_kernel3(p_real, p_imag, a, b, width, height);
    ref_kernel4(p_real, p_imag, a, b, width, height);
    ref_kernel4(p_real, p_imag, a, b, width, height);
    ref_kernel3(p_real, p_imag, a, b, width, height);
    ref_kernel2(p_real, p_imag, a, b, width, height);
    ref_kernel1(p_real, p_imag, a, b, width, height);
  }
}

// ---------------------------------------------------------------------------
// Trotter pair functions (operate on flat shared memory arrays)
// rl and im are flat arrays of size BLOCK_HEIGHT * BLOCK_WIDTH
// ---------------------------------------------------------------------------
template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int MARGIN_X, int MARGIN_Y, int BACKWARDS>
KOKKOS_INLINE_FUNCTION
void trotter_vert_pair(
    T a, T b,
    int width, int height,
    T &cell_r, T &cell_i,
    int kx, int ky, int py,
    T* rl, T* im)
{
  const int ky_peer = ky + 1 - 2 * BACKWARDS;
  if (py >= BACKWARDS && py < height - 1 + BACKWARDS &&
      ky >= BACKWARDS && ky < BLOCK_HEIGHT - 1 + BACKWARDS)
  {
    T peer_r = rl[ky_peer * BLOCK_WIDTH + kx];
    T peer_i = im[ky_peer * BLOCK_WIDTH + kx];
    rl[ky_peer * BLOCK_WIDTH + kx] = a * peer_r - b * cell_i;
    im[ky_peer * BLOCK_WIDTH + kx] = a * peer_i + b * cell_r;
    cell_r = a * cell_r - b * peer_i;
    cell_i = a * cell_i + b * peer_r;
  }
}

template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int MARGIN_X, int MARGIN_Y, int BACKWARDS>
KOKKOS_INLINE_FUNCTION
void trotter_horz_pair(
    T a, T b,
    int width, int height,
    T &cell_r, T &cell_i,
    int kx, int ky, int px,
    T* rl, T* im)
{
  const int kx_peer = kx + 1 - 2 * BACKWARDS;
  if (px >= BACKWARDS && px < width - 1 + BACKWARDS &&
      kx >= BACKWARDS && kx < BLOCK_WIDTH - 1 + BACKWARDS)
  {
    T peer_r = rl[ky * BLOCK_WIDTH + kx_peer];
    T peer_i = im[ky * BLOCK_WIDTH + kx_peer];
    rl[ky * BLOCK_WIDTH + kx_peer] = a * peer_r - b * cell_i;
    im[ky * BLOCK_WIDTH + kx_peer] = a * peer_i + b * cell_r;
    cell_r = a * cell_r - b * peer_i;
    cell_i = a * cell_i + b * peer_r;
  }
}

// ---------------------------------------------------------------------------
// Kokkos kernel wrapper (templated on all compile-time constants)
// ---------------------------------------------------------------------------
template<typename T, int STEPS, int BLOCK_X, int BLOCK_Y, int MARGIN_X, int MARGIN_Y, int STRIDE_Y>
void kernel(
    int teamX, int teamY,
    T a, T b, int width, int height,
    Kokkos::View<const T*, mem_space> p_real,
    Kokkos::View<const T*, mem_space> p_imag,
    Kokkos::View<T*, mem_space>       p2_real,
    Kokkos::View<T*, mem_space>       p2_imag)
{
  using ScratchView = Kokkos::View<T*, ScratchSpace, Kokkos::MemoryUnmanaged>;

  const int teams   = teamX * teamY;
  const int threads = BLOCK_X * STRIDE_Y;

  // Two flat shared arrays: rl[BLOCK_Y * BLOCK_X] and im[BLOCK_Y * BLOCK_X]
  const int scratch_size = ScratchView::shmem_size(2 * BLOCK_Y * BLOCK_X);

  Kokkos::parallel_for("tsa_kernel",
    Kokkos::TeamPolicy<exec_space>(teams, threads)
      .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<exec_space>::member_type& team) {

      ScratchView sm(team.team_scratch(0), 2 * BLOCK_Y * BLOCK_X);
      T* rl = sm.data();
      T* im = sm.data() + BLOCK_Y * BLOCK_X;

      const int blockIdx_x  = team.league_rank() % teamX;
      const int blockIdx_y  = team.league_rank() / teamX;
      const int threadIdx_x = team.team_rank() % BLOCK_X;
      const int threadIdx_y = team.team_rank() / BLOCK_X;

      int px = blockIdx_x * (BLOCK_X - 2*STEPS*MARGIN_X) + threadIdx_x - STEPS*MARGIN_X;
      int py = blockIdx_y * (BLOCK_Y - 2*STEPS*MARGIN_Y) + threadIdx_y - STEPS*MARGIN_Y;

      // Load from global to shared memory
      if (px >= 0 && px < width) {
        for (int i = 0, pidx = py * width + px;
             i < BLOCK_Y / STRIDE_Y;
             ++i, pidx += STRIDE_Y * width)
        {
          if (py + i * STRIDE_Y >= 0 && py + i * STRIDE_Y < height) {
            rl[(threadIdx_y + i * STRIDE_Y) * BLOCK_X + threadIdx_x] = p_real[pidx];
            im[(threadIdx_y + i * STRIDE_Y) * BLOCK_X + threadIdx_x] = p_imag[pidx];
          }
        }
      }
      team.team_barrier();

      // Place threads on black cells of a checkerboard pattern
      const int sx = threadIdx_x;
      int sy;
      if ((STEPS * MARGIN_X) % 2 == (STEPS * MARGIN_Y) % 2)
        sy = 2 * threadIdx_y + threadIdx_x % 2;
      else
        sy = 2 * threadIdx_y + 1 - threadIdx_x % 2;

      // Global y for range checks in trotter functions
      const int checkerboard_py =
          blockIdx_y * (BLOCK_Y - 2*STEPS*MARGIN_Y) + sy - STEPS*MARGIN_Y;

      // Read black cells into registers
      T cell_r[BLOCK_Y / (STRIDE_Y * 2)];
      T cell_i[BLOCK_Y / (STRIDE_Y * 2)];
      for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        cell_r[part] = rl[(sy + part * 2 * STRIDE_Y) * BLOCK_X + sx];
        cell_i[part] = im[(sy + part * 2 * STRIDE_Y) * BLOCK_X + sx];
      }

      // Apply STEPS full Trotter steps (pattern: 12344321)
      for (int step = 0; step < STEPS; step++) {
        // 1) forward vertical
        for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
          trotter_vert_pair<T, BLOCK_X, BLOCK_Y, STEPS*MARGIN_X, STEPS*MARGIN_Y, 0>(
              a, b, width, height, cell_r[part], cell_i[part],
              sx, sy + part*2*STRIDE_Y, checkerboard_py + part*2*STRIDE_Y, rl, im);
        }
        team.team_barrier();

        // 2) forward horizontal
        for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
          trotter_horz_pair<T, BLOCK_X, BLOCK_Y, STEPS*MARGIN_X, STEPS*MARGIN_Y, 0>(
              a, b, width, height, cell_r[part], cell_i[part],
              sx, sy + part*2*STRIDE_Y, px, rl, im);
        }
        team.team_barrier();

        // 3) backward vertical
        for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
          trotter_vert_pair<T, BLOCK_X, BLOCK_Y, STEPS*MARGIN_X, STEPS*MARGIN_Y, 1>(
              a, b, width, height, cell_r[part], cell_i[part],
              sx, sy + part*2*STRIDE_Y, checkerboard_py + part*2*STRIDE_Y, rl, im);
        }
        team.team_barrier();

        // 4) backward horizontal
        for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
          trotter_horz_pair<T, BLOCK_X, BLOCK_Y, STEPS*MARGIN_X, STEPS*MARGIN_Y, 1>(
              a, b, width, height, cell_r[part], cell_i[part],
              sx, sy + part*2*STRIDE_Y, px, rl, im);
        }
        team.team_barrier();

        // 4) backward horizontal (mirror of step 4)
        for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
          trotter_horz_pair<T, BLOCK_X, BLOCK_Y, STEPS*MARGIN_X, STEPS*MARGIN_Y, 1>(
              a, b, width, height, cell_r[part], cell_i[part],
              sx, sy + part*2*STRIDE_Y, px, rl, im);
        }
        team.team_barrier();

        // 3) backward vertical (mirror)
        for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
          trotter_vert_pair<T, BLOCK_X, BLOCK_Y, STEPS*MARGIN_X, STEPS*MARGIN_Y, 1>(
              a, b, width, height, cell_r[part], cell_i[part],
              sx, sy + part*2*STRIDE_Y, checkerboard_py + part*2*STRIDE_Y, rl, im);
        }
        team.team_barrier();

        // 2) forward horizontal (mirror)
        for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
          trotter_horz_pair<T, BLOCK_X, BLOCK_Y, STEPS*MARGIN_X, STEPS*MARGIN_Y, 0>(
              a, b, width, height, cell_r[part], cell_i[part],
              sx, sy + part*2*STRIDE_Y, px, rl, im);
        }
        team.team_barrier();

        // 1) forward vertical (mirror)
        for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
          trotter_vert_pair<T, BLOCK_X, BLOCK_Y, STEPS*MARGIN_X, STEPS*MARGIN_Y, 0>(
              a, b, width, height, cell_r[part], cell_i[part],
              sx, sy + part*2*STRIDE_Y, checkerboard_py + part*2*STRIDE_Y, rl, im);
        }
        team.team_barrier();
      }

      // Write registers back to shared memory
      for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        rl[(sy + part*2*STRIDE_Y) * BLOCK_X + sx] = cell_r[part];
        im[(sy + part*2*STRIDE_Y) * BLOCK_X + sx] = cell_i[part];
      }
      team.team_barrier();

      // Write output to global memory (discard halo)
      const int out_sx = threadIdx_x + STEPS * MARGIN_X;
      const int out_sy = threadIdx_y + STEPS * MARGIN_Y;
      const int out_px = px + STEPS * MARGIN_X;
      const int out_py = py + STEPS * MARGIN_Y;
      if (out_sx < BLOCK_X - STEPS*MARGIN_X && out_px < width) {
        for (int i = 0, pidx = out_py * width + out_px;
             i < BLOCK_Y / STRIDE_Y;
             ++i, pidx += STRIDE_Y * width)
        {
          if (out_sy + i*STRIDE_Y < BLOCK_Y - STEPS*MARGIN_Y && out_py + i*STRIDE_Y < height) {
            p2_real[pidx] = rl[(out_sy + i*STRIDE_Y) * BLOCK_X + out_sx];
            p2_imag[pidx] = im[(out_sy + i*STRIDE_Y) * BLOCK_X + out_sx];
          }
        }
      }
    });
  Kokkos::fence();
}

// ---------------------------------------------------------------------------
// init_p: initialise the wavepacket
// ---------------------------------------------------------------------------
template <typename T>
static void init_p(T *p_real, T *p_imag, int width, int height)
{
  double s = 64.0;
  for (int j = 1; j <= height; j++) {
    for (int i = 1; i <= width; i++) {
      std::complex<T> tmp =
        std::complex<T>((T)exp(-(pow(i-180.0,2.0)+pow(j-300.0,2.0))/(2.0*pow(s,2.0))), T(0)) *
        exp(std::complex<T>(T(0), T(0.4*(i+j-480.0))));
      p_real[(j-1)*width + (i-1)] = real(tmp);
      p_imag[(j-1)*width + (i-1)] = imag(tmp);
    }
  }
}

// ---------------------------------------------------------------------------
// tsa<T>: run the benchmark for one floating-point type
// ---------------------------------------------------------------------------
template <typename T>
void tsa(int width, int height, int repeat)
{
  T *p_real = new T[width * height];
  T *p_imag = new T[width * height];
  T *h_real = new T[width * height];
  T *h_imag = new T[width * height];

  init_p(p_real, p_imag, width, height);

  const T a = (T)cos(0.02);
  const T b = (T)sin(0.02);

  // Reference (CPU sequential)
  memcpy(h_real, p_real, sizeof(T) * width * height);
  memcpy(h_imag, p_imag, sizeof(T) * width * height);
  reference(h_real, h_imag, (float)a, (float)b, width, height, repeat);

  // Compile-time constants
  static const int BLOCK_X  = 16;
  static const int BLOCK_Y  = (sizeof(T) == 8) ? 32 : 96;
  static const int STRIDE_Y = 16;
  static const int MARGIN_X = 3;
  static const int MARGIN_Y = 4;
  static const int STEPS    = 1;

  const int teamX = (width  + (BLOCK_X - 2*STEPS*MARGIN_X) - 1) / (BLOCK_X - 2*STEPS*MARGIN_X);
  const int teamY = (height + (BLOCK_Y - 2*STEPS*MARGIN_Y) - 1) / (BLOCK_Y - 2*STEPS*MARGIN_Y);

  // Device ping-pong arrays
  Kokkos::View<T*, mem_space> d_real0("d_real0", width * height);
  Kokkos::View<T*, mem_space> d_imag0("d_imag0", width * height);
  Kokkos::View<T*, mem_space> d_real1("d_real1", width * height);
  Kokkos::View<T*, mem_space> d_imag1("d_imag1", width * height);

  {
    auto hr = Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(p_real, width*height);
    auto hi = Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(p_imag, width*height);
    Kokkos::deep_copy(d_real0, hr);
    Kokkos::deep_copy(d_imag0, hi);
  }

  // sense=0: read from [sense], write to [1-sense]
  int sense = 0;
  Kokkos::View<T*, mem_space> d_real[2] = {d_real0, d_real1};
  Kokkos::View<T*, mem_space> d_imag[2] = {d_imag0, d_imag1};

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // Dispatch with correct BLOCK_Y at compile time
    if constexpr (sizeof(T) == 8) {
      kernel<T, STEPS, BLOCK_X, 32, MARGIN_X, MARGIN_Y, STRIDE_Y>(
          teamX, teamY, a, b, width, height,
          d_real[sense],  d_imag[sense],
          d_real[1-sense], d_imag[1-sense]);
    } else {
      kernel<T, STEPS, BLOCK_X, 96, MARGIN_X, MARGIN_Y, STRIDE_Y>(
          teamX, teamY, a, b, width, height,
          d_real[sense],  d_imag[sense],
          d_real[1-sense], d_imag[1-sense]);
    }
    sense = 1 - sense;
  }

  Kokkos::fence();
  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  // Copy result back
  T *t_real = new T[width * height];
  T *t_imag = new T[width * height];
  {
    auto hr = Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(t_real, width*height);
    auto hi = Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(t_imag, width*height);
    Kokkos::deep_copy(hr, d_real[sense]);
    Kokkos::deep_copy(hi, d_imag[sense]);
  }

  // Verify
  bool ok = true;
  for (int i = 0; i < width * height; i++) {
    if (fabs((double)(t_real[i] - h_real[i])) > 1e-3) { ok = false; break; }
    if (fabs((double)(t_imag[i] - h_imag[i])) > 1e-3) { ok = false; break; }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  delete[] t_real; delete[] t_imag;
  delete[] p_real; delete[] p_imag;
  delete[] h_real; delete[] h_imag;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
  if (argc != 4) {
    printf("Usage: %s <matrix width> <matrix height> <repeat>\n", argv[0]);
    return 1;
  }
  int width  = atoi(argv[1]);
  int height = atoi(argv[2]);
  int repeat = atoi(argv[3]);

  Kokkos::initialize(argc, argv);
  {
    printf("TSA in float32\n");
    tsa<float>(width, height, repeat);

    printf("\n");

    printf("TSA in float64\n");
    tsa<double>(width, height, repeat);
  }
  Kokkos::finalize();
  return 0;
}
