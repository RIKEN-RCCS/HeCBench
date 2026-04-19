// Kokkos port of Rayleigh-Benard convection simulation.
// Uses spectral finite-difference methods with Runge-Kutta time integration.
// GEMM operations replace cuBLAS; elementwise kernels use Kokkos::parallel_for.

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

// ============================================================
// Problem parameters (matching CUDA benchmark)
// ============================================================
static constexpr int   M         = 500;
static constexpr int   N         = 1000;
static constexpr float RA        = 1e8f;
static constexpr int   XF        = 2;
static constexpr int   STARTSTEP = 0;
static constexpr float PI        = 3.14159265358979323846f;
static constexpr float DX        = 1.f / (M - 1.f);
static constexpr float DX2       = DX * DX;
static constexpr float OMEGACOEFF = -((DX2) * (DX2)) * RA;
static constexpr float DT_START  = 0.000000000000005f;
static constexpr float FRAMESIZE = DX2 / 4.f;

// ============================================================
// View aliases
// ============================================================
using View1D = Kokkos::View<float*>;
using View2D = Kokkos::View<float**>;

// ============================================================
// Elementwise kernels
// ============================================================

// A[i] *= B[i]  for i in [0, size)
KOKKOS_INLINE_FUNCTION void elemMult_kernel(int i, float* A, const float* B) {
  A[i] *= B[i];
}

static void elemMultOmega(const View1D& A, const View1D& B) {
  const int size = (M-2) * (N-2);
  Kokkos::parallel_for("ElemMultOmega", Kokkos::RangePolicy<>(0, size),
    KOKKOS_LAMBDA(int i) { A(i) *= B(i); });
}

static void elemMultT(const View1D& A, const View1D& B) {
  const int size = (M-2) * N;
  Kokkos::parallel_for("ElemMultT", Kokkos::RangePolicy<>(0, size),
    KOKKOS_LAMBDA(int i) { A(i) *= B(i); });
}

static void elemMultNu(const View1D& A, const View1D& B) {
  Kokkos::parallel_for("ElemMultNu", Kokkos::RangePolicy<>(0, N),
    KOKKOS_LAMBDA(int i) { A(i) *= B(i); });
}

static void subOne(const View1D& A, int offset = 0) {
  Kokkos::parallel_for("SubOne", Kokkos::RangePolicy<>(0, N),
    KOKKOS_LAMBDA(int i) { A(offset + i) -= 1.f; });
}

static void addOne(const View1D& A, int offset = 0) {
  Kokkos::parallel_for("AddOne", Kokkos::RangePolicy<>(0, N),
    KOKKOS_LAMBDA(int i) { A(offset + i) += 1.f; });
}

static void addX(const View1D& A, float x) {
  const int size = (M-2) * N;
  Kokkos::parallel_for("AddX", Kokkos::RangePolicy<>(0, size),
    KOKKOS_LAMBDA(int i) { A(i) += x; });
}

// ============================================================
// BLAS-like helpers implemented with Kokkos
// ============================================================

// y = alpha*x + y  (saxpy)
static void saxpy(float alpha, const View1D& x, const View1D& y, int n) {
  Kokkos::parallel_for("saxpy", Kokkos::RangePolicy<>(0, n),
    KOKKOS_LAMBDA(int i) { y(i) += alpha * x(i); });
}

// y = x  (scopy) for first n elements
static void scopy(const View1D& src, const View1D& dst, int n) {
  Kokkos::parallel_for("scopy", Kokkos::RangePolicy<>(0, n),
    KOKKOS_LAMBDA(int i) { dst(i) = src(i); });
}

// y = alpha * y  (sscal)
static void sscal(float alpha, const View1D& y, int n) {
  Kokkos::parallel_for("sscal", Kokkos::RangePolicy<>(0, n),
    KOKKOS_LAMBDA(int i) { y(i) *= alpha; });
}

// Offset variants (simulating pointer arithmetic in the CUDA code)
static void saxpy_off(float alpha, const View1D& x, int xoff,
                      const View1D& y, int yoff, int n) {
  Kokkos::parallel_for("saxpy_off", Kokkos::RangePolicy<>(0, n),
    KOKKOS_LAMBDA(int i) { y(yoff + i) += alpha * x(xoff + i); });
}

static void scopy_off(const View1D& src, int soff,
                      const View1D& dst, int doff, int n) {
  Kokkos::parallel_for("scopy_off", Kokkos::RangePolicy<>(0, n),
    KOKKOS_LAMBDA(int i) { dst(doff + i) = src(soff + i); });
}

static void sscal_off(float alpha, const View1D& y, int off, int n) {
  Kokkos::parallel_for("sscal_off", Kokkos::RangePolicy<>(0, n),
    KOKKOS_LAMBDA(int i) { y(off + i) *= alpha; });
}

// ============================================================
// General matrix multiply: C = alpha*A*B + beta*C
// A is [m x k], B is [k x n], C is [m x n], all row-major
// ============================================================
static void sgemm(int m, int n, int k,
                  float alpha,
                  const View1D& A, int A_off,
                  const View1D& B, int B_off,
                  float beta,
                  const View1D& C, int C_off) {
  Kokkos::parallel_for("sgemm",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m, n}),
    KOKKOS_LAMBDA(int row, int col) {
      float sum = 0.f;
      for (int p = 0; p < k; ++p)
        sum += A(A_off + row * k + p) * B(B_off + p * n + col);
      C(C_off + row * n + col) = alpha * sum + beta * C(C_off + row * n + col);
    });
}

// ============================================================
// Finite-difference operators
// ============================================================

// Dx: first derivative in x-direction (columns) of (M-2)xN array.
// BC: for T (is_psi==0), left/right column boundary value = 1.
static void Dx(const View1D& f, int is_psi, const View1D& y) {
  const float alpha2 = 1.f / (2.f * DX);
  // Interior columns (1..N-2): centered difference
  Kokkos::parallel_for("Dx_interior",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 1}, {M-2, N-1}),
    KOKKOS_LAMBDA(int row, int col) {
      y(row * N + col) = alpha2 * (f(row * N + col + 1) - f(row * N + col - 1));
    });
  // Left boundary (col=0): one-sided, BC=1 for T, 0 for psi
  Kokkos::parallel_for("Dx_left",
    Kokkos::RangePolicy<>(0, M-2),
    KOKKOS_LAMBDA(int row) {
      float bc = is_psi ? 0.f : 1.f;
      y(row * N) = alpha2 * (f(row * N + 1) - bc);
    });
  // Right boundary (col=N-1)
  Kokkos::parallel_for("Dx_right",
    Kokkos::RangePolicy<>(0, M-2),
    KOKKOS_LAMBDA(int row) {
      float bc = is_psi ? 0.f : 1.f;
      y(row * N + N - 1) = alpha2 * (bc - f(row * N + N - 2));
    });
}

// Dz: first derivative in z-direction (rows) of (M-2)xN array.
// BC: for T (is_psi==0), bottom BC=1 (hot), top BC=0 (cold).
static void Dz(const View1D& f, int is_psi, const View1D& y) {
  const float alpha2 = 1.f / (2.f * DX);
  // Interior rows (1..M-4): centered difference
  Kokkos::parallel_for("Dz_interior",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {M-3, N}),
    KOKKOS_LAMBDA(int row, int col) {
      y(row * N + col) = alpha2 * (f((row+1)*N + col) - f((row-1)*N + col));
    });
  // Top row (row=0): one-sided, BC at z=0 is 1 for T, 0 for psi
  Kokkos::parallel_for("Dz_top",
    Kokkos::RangePolicy<>(0, N),
    KOKKOS_LAMBDA(int col) {
      float bc = is_psi ? 0.f : 1.f;
      y(col) = alpha2 * (f(N + col) - bc);
    });
  // Bottom row (row=M-3): BC at z=1 is 0 for both T and psi
  Kokkos::parallel_for("Dz_bot",
    Kokkos::RangePolicy<>(0, N),
    KOKKOS_LAMBDA(int col) {
      y((M-3)*N + col) = alpha2 * (0.f - f((M-4)*N + col));
    });
}

// Dxx: second derivative in x-direction (columns)
static void Dxx(const View1D& f, const View1D& y) {
  const float c = 1.f / DX2;
  Kokkos::parallel_for("Dxx",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M-2, N}),
    KOKKOS_LAMBDA(int row, int col) {
      float fm = (col > 0)   ? f(row * N + col - 1) : 0.f;
      float f0 = f(row * N + col);
      float fp = (col < N-1) ? f(row * N + col + 1) : 0.f;
      y(row * N + col) = c * (fm - 2.f*f0 + fp);
    });
}

// Dzz: second derivative in z-direction (rows)
static void Dzz(const View1D& f, const View1D& y) {
  const float c = 1.f / DX2;
  Kokkos::parallel_for("Dzz",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M-2, N}),
    KOKKOS_LAMBDA(int row, int col) {
      float fm = (row > 0)   ? f((row-1) * N + col) : 0.f;
      float f0 = f(row * N + col);
      float fp = (row < M-3) ? f((row+1) * N + col) : 0.f;
      y(row * N + col) = c * (fm - 2.f*f0 + fp);
    });
}

// ============================================================
// Updatedt: adaptive timestep — finds max |u| and |v|, then updates dt
// ============================================================
static void updatedt(const View1D& u, const View1D& v, float& dt) {
  const int n = (M-2) * N;
  float maxU = 0.f, maxV = 0.f;
  Kokkos::parallel_reduce("max_u", Kokkos::RangePolicy<>(0, n),
    KOKKOS_LAMBDA(int i, float& m) {
      float av = Kokkos::abs(u(i));
      if (av > m) m = av;
    }, Kokkos::Max<float>(maxU));
  Kokkos::parallel_reduce("max_v", Kokkos::RangePolicy<>(0, n),
    KOKKOS_LAMBDA(int i, float& m) {
      float av = Kokkos::abs(v(i));
      if (av > m) m = av;
    }, Kokkos::Max<float>(maxV));
  float maxuv = std::max(maxU, maxV);
  if (maxuv > 0.f)
    dt = std::min(DX / maxuv, DX2 / 4.f);
}

// ============================================================
// Nusselt number compute (simplified version)
// ============================================================
static float nusseltCompute(const View1D& T, const View1D& trnu,
                             const View1D& nutop_buf, const View1D& nubot_buf) {
  // Copy last 3 rows of T into nutop_buf rows 0-2
  scopy_off(T, (M-5)*N, nutop_buf, 0,    N);
  scopy_off(T, (M-4)*N, nutop_buf, N,    N);
  scopy_off(T, (M-3)*N, nutop_buf, 2*N,  N);
  // Set row 3 of nutop to zero
  Kokkos::parallel_for("nutop_zero", Kokkos::RangePolicy<>(0, N),
    KOKKOS_LAMBDA(int i) { nutop_buf(3*N + i) = 0.f; });

  // Apply corrections and scaling (mimics CUDA NusseltCompute)
  // -(2/3)*row0 + 3*row1 - 6*row2 + (11/3)*row3, scaled by 1/(2*DX)
  float result = 0.f;
  auto h_nutop = Kokkos::create_mirror_view(nutop_buf);
  auto h_trnu  = Kokkos::create_mirror_view(trnu);
  Kokkos::deep_copy(h_nutop, nutop_buf);
  Kokkos::deep_copy(h_trnu,  trnu);
  for (int j = 0; j < N; ++j) {
    float dT = (-(2.f/3.f) * h_nutop(j)
                + 3.f * h_nutop(N + j)
                - 6.f * h_nutop(2*N + j)
                + (11.f/3.f) * h_nutop(3*N + j)) / (2.f * DX);
    result += dT * h_trnu(j);
  }
  return result / (-XF);
}

// ============================================================
// G: one RHS evaluation of the PDE
// ============================================================
static void G(const View1D& f, const View1D& Tbuff, const View1D& DxT,
              const View1D& y, const View1D& u, const View1D& v,
              const View1D& psi, const View1D& omega,
              const View1D& dsc, const View1D& dsr, const View1D& ei,
              float& dt, const View1D& output, bool compute_velocity) {
  const int innerSize = (M-2) * (N-2);
  const int tSize     = (M-2) * N;

  // DxT = f (save a copy), then compute Dx(f, 0, DxT)
  scopy(f, DxT, tSize);
  Dx(f, 0, DxT);

  // Extract interior columns of DxT into omega
  // omega[i*(N-2) + j] = DxT[i*N + j+1]  for j in 0..N-3
  Kokkos::parallel_for("extract_omega",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M-2, N-2}),
    KOKKOS_LAMBDA(int i, int j) { omega(i*(N-2) + j) = DxT(i*N + j + 1); });

  // Spectral solve: omega = dsc * omega * dsr * ei * dsc * omega * dsr
  // Since CUBLAS assumed column-major and we use row-major, we use
  // Transpose(omega)*Transpose(dsc) = omega * dsc^T  etc.
  // The pattern: omega = dsc * omega, omega = omega * dsr, omega .*= ei, repeat

  const float alpha = 1.f, beta = 0.f;

  // Tbuff = omega * dsc  (omega: (M-2)x(N-2), dsc: (M-2)x(M-2))
  sgemm(N-2, M-2, M-2, alpha, omega, 0, dsc, 0, beta, Tbuff, 0);
  scopy_off(Tbuff, 0, omega, 0, innerSize);

  // Tbuff = dsr * omega  (dsr: (N-2)x(N-2))
  sgemm(N-2, M-2, N-2, alpha, dsr, 0, omega, 0, beta, Tbuff, 0);
  scopy_off(Tbuff, 0, omega, 0, innerSize);

  // omega .*= ei
  elemMultOmega(omega, ei);

  // same two gemms again
  sgemm(N-2, M-2, M-2, alpha, omega, 0, dsc, 0, beta, Tbuff, 0);
  scopy_off(Tbuff, 0, omega, 0, innerSize);

  sgemm(N-2, M-2, N-2, alpha, dsr, 0, omega, 0, beta, Tbuff, 0);
  scopy_off(Tbuff, 0, omega, 0, innerSize);

  // Scale omega by OMEGACOEFF
  sscal(OMEGACOEFF, omega, innerSize);

  // Copy omega into interior columns of psi
  Kokkos::parallel_for("psi_interior",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M-2, N-2}),
    KOKKOS_LAMBDA(int i, int j) { psi(i*N + j + 1) = omega(i*(N-2) + j); });

  // u = Dz(psi)
  scopy(f, u, tSize);
  Dz(psi, 1, u);

  // v = -Dx(psi)
  scopy(f, v, tSize);
  Dx(psi, 1, v);
  sscal(-1.f, v, tSize);

  if (compute_velocity) {
    updatedt(u, v, dt);
  }

  // y = Dxx(f) + Dzz(f)
  scopy(f, y, tSize);
  Dxx(f, y);

  scopy(f, Tbuff, tSize);
  Dzz(f, Tbuff);
  saxpy(1.f, Tbuff, y, tSize);

  // u = u .* DxT  then y += u
  elemMultT(u, DxT);
  saxpy(1.f, u, y, tSize);

  // u = Dz(f,0), then u = v.*u, then y += u
  scopy(f, u, tSize);
  Dz(f, 0, u);
  elemMultT(u, v);
  saxpy(1.f, u, y, tSize);

  scopy(y, output, tSize);
}

// ============================================================
int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <timesteps>\n", argv[0]);
    return 1;
  }
  const int ENDSTEP = atoi(argv[1]);

  printf("M = %d. N = %d. DX = %E. Ra = %E.\n", M, N, DX, RA);
  printf("\nInitialization\n");

  Kokkos::initialize(argc, argv);
  {
    const int tSize    = (M-2) * N;
    const int innerSz  = (M-2) * (N-2);
    const int dscSz    = (M-2) * (M-2);
    const int dsrSz    = (N-2) * (N-2);

    // Allocate all device arrays
    View1D d_T      ("T",      tSize);
    View1D d_Tbuff  ("Tbuff",  tSize);
    View1D d_DxT    ("DxT",    tSize);
    View1D d_y      ("y",      tSize);
    View1D d_u      ("u",      tSize);
    View1D d_v      ("v",      tSize);
    View1D d_psi    ("psi",    tSize);
    View1D d_omega  ("omega",  innerSz);
    View1D d_dsc    ("dsc",    dscSz);
    View1D d_dsr    ("dsr",    dsrSz);
    View1D d_ei     ("ei",     innerSz);
    View1D d_xrk3   ("xrk3",  tSize);
    View1D d_yrk3   ("yrk3",  tSize);
    View1D d_zrk3   ("zrk3",  tSize);
    View1D d_temp   ("temp",  tSize);
    View1D d_trnu   ("trnu",  N);
    View1D d_nutop  ("nutop", 4*N);
    View1D d_nubot  ("nubot", 4*N);

    // ---- Host initialization ----
    std::vector<float> h_X(N), h_Z(M-2), h_T(tSize);
    std::vector<float> h_dsc(dscSz), h_dsr(dsrSz);
    std::vector<float> h_lambda(M-2), h_mu(N-2), h_ei(innerSz);
    std::vector<float> h_trnu(N);

    for (int i = 0; i < N; ++i)   h_X[i] = (i * XF + 0.f) / (N - 1.f);
    for (int i = 1; i < M-1; ++i) h_Z[i-1] = (i + 0.f) / (M - 1.f);

    for (int i = 0; i < M-2; ++i)
      for (int j = 0; j < N; ++j)
        h_T[i*N + j] = 1 - h_Z[i] + 0.01f * std::sin(PI * h_Z[i]) * std::cos((PI/XF) * h_X[j]);

    for (int i = 0; i < M-2; ++i)
      for (int j = 0; j < M-2; ++j)
        h_dsc[(M-2)*i + j] = std::sqrt(2.f/(M-1.f)) * std::sin((i+1.f)*(j+1.f)*PI/(M-1.f));

    for (int i = 0; i < N-2; ++i)
      for (int j = 0; j < N-2; ++j)
        h_dsr[(N-2)*i + j] = std::sqrt(2.f/(N-1.f)) * std::sin((i+1.f)*(j+1.f)*PI/(N-1.f));

    for (int i = 0; i < M-2; ++i) h_lambda[i] = 2.f*std::cos((i+1.f)*PI/(M-1.f)) - 2.f;
    for (int i = 0; i < N-2; ++i) h_mu[i]     = 2.f*std::cos((i+1.f)*PI/(N-1.f)) - 2.f;

    for (int i = 0; i < M-2; ++i)
      for (int j = 0; j < N-2; ++j) {
        float v = h_lambda[i] + h_mu[j];
        h_ei[(N-2)*i + j] = 1.f / (v * v);
      }

    for (int i = 1; i < N-1; ++i) h_trnu[i] = DX;
    h_trnu[0] = DX / 2.f;
    h_trnu[N-1] = DX / 2.f;

    // Copy to device
    auto upload = [](View1D& d, const std::vector<float>& h) {
      auto hv = Kokkos::create_mirror_view(d);
      for (int i = 0; i < (int)h.size(); ++i) hv(i) = h[i];
      Kokkos::deep_copy(d, hv);
    };
    upload(d_T,   h_T);
    upload(d_dsc, h_dsc);
    upload(d_dsr, h_dsr);
    upload(d_ei,  h_ei);
    upload(d_trnu,h_trnu);

    // Initialize auxiliary arrays to zero / T
    Kokkos::deep_copy(d_psi,   0.f);
    Kokkos::deep_copy(d_u,     0.f);
    Kokkos::deep_copy(d_v,     0.f);
    Kokkos::deep_copy(d_y,     0.f);
    scopy(d_T, d_xrk3,  tSize);
    scopy(d_T, d_yrk3,  tSize);
    scopy(d_T, d_zrk3,  tSize);
    scopy(d_T, d_Tbuff, tSize);
    scopy(d_T, d_DxT,   tSize);
    scopy(d_T, d_temp,  tSize);
    Kokkos::fence();

    printf("Begin computation:\n");

    float dt     = DT_START;
    float frames = 0.f;
    int   tstep  = 0;

    auto t_start = std::chrono::steady_clock::now();

    for (int c = STARTSTEP; c < ENDSTEP; ++c) {
      // RK3 step 1: xrk3 = G(T)
      G(d_T, d_Tbuff, d_DxT, d_y, d_u, d_v,
        d_psi, d_omega, d_dsc, d_dsr, d_ei, dt, d_xrk3, true);

      // temp = T + (dt/3)*xrk3
      scopy(d_T, d_temp, tSize);
      saxpy(dt / 3.f, d_xrk3, d_temp, tSize);

      // RK3 step 2: yrk3 = G(temp)
      G(d_temp, d_Tbuff, d_DxT, d_y, d_u, d_v,
        d_psi, d_omega, d_dsc, d_dsr, d_ei, dt, d_yrk3, false);

      // temp = T + (2*dt/3)*yrk3
      scopy(d_T, d_temp, tSize);
      saxpy(2.f * dt / 3.f, d_yrk3, d_temp, tSize);

      // RK3 step 3: zrk3 = G(temp)
      G(d_temp, d_Tbuff, d_DxT, d_y, d_u, d_v,
        d_psi, d_omega, d_dsc, d_dsr, d_ei, dt, d_zrk3, false);

      // T += (dt/4)*xrk3 + (3*dt/4)*zrk3
      saxpy(dt / 4.f,       d_xrk3, d_T, tSize);
      saxpy(3.f * dt / 4.f, d_zrk3, d_T, tSize);

      Kokkos::fence();
      frames += dt;

      if (frames > FRAMESIZE) {
        ++tstep;
        float nu = nusseltCompute(d_T, d_trnu, d_nutop, d_nubot);
        printf("Nusselt number: %.1f\n", nu);
        frames = 0.f;
      }
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9;
    printf("Total compute time: %f (s)\n", elapsed);
    if (ENDSTEP > 0)
      printf("Average compute time per step: %f (s)\n", elapsed / ENDSTEP);
  }
  Kokkos::finalize();
  return 0;
}
