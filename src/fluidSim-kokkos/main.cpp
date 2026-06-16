#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ---- Custom vector types ----
struct int8_t_ {
  int s0, s1, s2, s3, s4, s5, s6, s7;
};
struct double8 {
  double s0, s1, s2, s3, s4, s5, s6, s7;
};
struct double4 {
  double x, y, z, w;
};
struct double2 {
  double x, y;
};

// Rename to avoid conflict with standard int8_t
using lbm_int8 = int8_t_;

// ---- Device helper functions ----
KOKKOS_INLINE_FUNCTION
double dot4(double4 a, double4 b) {
  return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

KOKKOS_INLINE_FUNCTION
double4 add4(double4 a, double4 b) {
  double4 r; r.x=a.x+b.x; r.y=a.y+b.y; r.z=a.z+b.z; r.w=a.w+b.w; return r;
}

KOKKOS_INLINE_FUNCTION
double4 scale4(double s, double4 a) {
  double4 r; r.x=s*a.x; r.y=s*a.y; r.z=s*a.z; r.w=s*a.w; return r;
}

KOKKOS_INLINE_FUNCTION
double ced(double rho, double weight, double2 dir, double2 u) {
  double u2 = u.x*u.x + u.y*u.y;
  double eu = dir.x*u.x + dir.y*u.y;
  return rho * weight * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2);
}

KOKKOS_INLINE_FUNCTION
lbm_int8 newPos(int p, double8 dir) {
  lbm_int8 np;
  np.s0=p+(int)dir.s0; np.s1=p+(int)dir.s1; np.s2=p+(int)dir.s2; np.s3=p+(int)dir.s3;
  np.s4=p+(int)dir.s4; np.s5=p+(int)dir.s5; np.s6=p+(int)dir.s6; np.s7=p+(int)dir.s7;
  return np;
}

KOKKOS_INLINE_FUNCTION
lbm_int8 fma8(unsigned int a, lbm_int8 b, lbm_int8 c) {
  lbm_int8 r;
  r.s0=(int)(a*b.s0+c.s0); r.s1=(int)(a*b.s1+c.s1);
  r.s2=(int)(a*b.s2+c.s2); r.s3=(int)(a*b.s3+c.s3);
  r.s4=(int)(a*b.s4+c.s4); r.s5=(int)(a*b.s5+c.s5);
  r.s6=(int)(a*b.s6+c.s6); r.s7=(int)(a*b.s7+c.s7);
  return r;
}

// ---- Host utility ----
double computefEq(double rho, double weight, const double dir[2], const double vel[2]) {
  double u2 = vel[0]*vel[0] + vel[1]*vel[1];
  double eu = dir[0]*vel[0] + dir[1]*vel[1];
  return rho * weight * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2);
}

// ---- LBM kernel ----
void lbm(unsigned int width, unsigned int height,
         const double* if0,   double* of0,
         const double4* if1234, double4* of1234,
         const double4* if5678, double4* of5678,
         const bool* type_arr,
         double8 dirX, double8 dirY,
         const double* weight_,
         double omega)
{
  Kokkos::parallel_for(
    "lbm",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{(int)height,(int)width}),
    KOKKOS_LAMBDA(int idy, int idx) {
      unsigned int pos = (unsigned int)idx + width * (unsigned int)idy;

      double f0    = if0[pos];
      double4 f1234 = if1234[pos];
      double4 f5678 = if5678[pos];

      double e0 = f0; // boundary default
      double4 e1234, e5678;
      double rho;
      double2 u;

      if (type_arr[pos]) { // Boundary: bounce-back
        e1234.x=f1234.z; e1234.y=f1234.w; e1234.z=f1234.x; e1234.w=f1234.y;
        e5678.x=f5678.z; e5678.y=f5678.w; e5678.z=f5678.x; e5678.w=f5678.y;
        rho=0.0; u.x=0.0; u.y=0.0;
      } else { // Fluid
        double4 temp = add4(f1234, f5678);
        rho = f0 + temp.x + temp.y + temp.z + temp.w;

        double4 x1234; x1234.x=dirX.s0; x1234.y=dirX.s1; x1234.z=dirX.s2; x1234.w=dirX.s3;
        double4 y1234; y1234.x=dirY.s0; y1234.y=dirY.s1; y1234.z=dirY.s2; y1234.w=dirY.s3;
        double4 x5678; x5678.x=dirX.s4; x5678.y=dirX.s5; x5678.z=dirX.s6; x5678.w=dirX.s7;
        double4 y5678; y5678.x=dirY.s4; y5678.y=dirY.s5; y5678.z=dirY.s6; y5678.w=dirY.s7;
        u.x = (dot4(f1234,x1234) + dot4(f5678,x5678)) / rho;
        u.y = (dot4(f1234,y1234) + dot4(f5678,y5678)) / rho;

        double2 d0; d0.x=0.0; d0.y=0.0;
        e0 = ced(rho, weight_[0], d0, u);
        double2 d1; d1.x=dirX.s0; d1.y=dirY.s0; e1234.x=ced(rho,weight_[1],d1,u);
        double2 d2; d2.x=dirX.s1; d2.y=dirY.s1; e1234.y=ced(rho,weight_[2],d2,u);
        double2 d3; d3.x=dirX.s2; d3.y=dirY.s2; e1234.z=ced(rho,weight_[3],d3,u);
        double2 d4; d4.x=dirX.s3; d4.y=dirY.s3; e1234.w=ced(rho,weight_[4],d4,u);
        double2 d5; d5.x=dirX.s4; d5.y=dirY.s4; e5678.x=ced(rho,weight_[5],d5,u);
        double2 d6; d6.x=dirX.s5; d6.y=dirY.s5; e5678.y=ced(rho,weight_[6],d6,u);
        double2 d7; d7.x=dirX.s6; d7.y=dirY.s6; e5678.z=ced(rho,weight_[7],d7,u);
        double2 d8; d8.x=dirX.s7; d8.y=dirY.s7; e5678.w=ced(rho,weight_[8],d8,u);

        double omf = 1.0 - omega;
        e0     = omf*f0     + omega*e0;
        e1234  = add4(scale4(omf,f1234), scale4(omega,e1234));
        e5678  = add4(scale4(omf,f5678), scale4(omega,e5678));
      }

      bool t1 = (unsigned int)idx < width  - 1;
      bool t2 = (unsigned int)idy < height - 1;
      bool t3 = idx > 0;
      bool t4 = idy > 0;

      if (t1 && t2 && t3 && t4) {
        lbm_int8 nX = newPos((int)idx, dirX);
        lbm_int8 nY = newPos((int)idy, dirY);
        lbm_int8 nPos = fma8(width, nY, nX);

        of0[pos] = e0;
        of1234[nPos.s0].x = e1234.x;
        of1234[nPos.s1].y = e1234.y;
        of1234[nPos.s2].z = e1234.z;
        of1234[nPos.s3].w = e1234.w;
        of5678[nPos.s4].x = e5678.x;
        of5678[nPos.s5].y = e5678.y;
        of5678[nPos.s6].z = e5678.z;
        of5678[nPos.s7].w = e5678.w;
      }
    }
  );
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <iterations> <omega> <width> <height>\n", argv[0]);
    return 1;
  }
  const int    iterations = atoi(argv[1]);
  const double omega      = atof(argv[2]);
  const int    lbm_width  = atoi(argv[3]);
  const int    lbm_height = atoi(argv[4]);

  Kokkos::initialize(argc, argv);
  {
    const int dims[2] = {lbm_width, lbm_height};
    const size_t n = (size_t)dims[0] * dims[1];

    // LBM weights and directions
    double e[9][2] = {{0,0},{1,0},{0,1},{-1,0},{0,-1},{1,1},{-1,1},{-1,-1},{1,-1}};
    double w[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                   1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};

    double8 dirX, dirY;
    dirX.s0=e[1][0]; dirY.s0=e[1][1];
    dirX.s1=e[2][0]; dirY.s1=e[2][1];
    dirX.s2=e[3][0]; dirY.s2=e[3][1];
    dirX.s3=e[4][0]; dirY.s3=e[4][1];
    dirX.s4=e[5][0]; dirY.s4=e[5][1];
    dirX.s5=e[6][0]; dirY.s5=e[6][1];
    dirX.s6=e[7][0]; dirY.s6=e[7][1];
    dirX.s7=e[8][0]; dirY.s7=e[8][1];

    // Host arrays
    auto h_if0    = Kokkos::View<double*,  Kokkos::HostSpace>("h_if0",   n);
    auto h_if1234 = Kokkos::View<double4*, Kokkos::HostSpace>("h_if1234", n);
    auto h_if5678 = Kokkos::View<double4*, Kokkos::HostSpace>("h_if5678", n);
    auto h_type   = Kokkos::View<bool*,    Kokkos::HostSpace>("h_type",  n);

    double u0[2] = {0.01, 0.01};
    srand(123);
    for (int y = 0; y < dims[1]; y++) {
      for (int x = 0; x < dims[0]; x++) {
        int pos = x + y * dims[0];
        double den = (double)(rand() % 10 + 1);
        h_if0(pos)       = computefEq(den, w[0], e[0], u0);
        h_if1234(pos).x  = computefEq(den, w[1], e[1], u0);
        h_if1234(pos).y  = computefEq(den, w[2], e[2], u0);
        h_if1234(pos).z  = computefEq(den, w[3], e[3], u0);
        h_if1234(pos).w  = computefEq(den, w[4], e[4], u0);
        h_if5678(pos).x  = computefEq(den, w[5], e[5], u0);
        h_if5678(pos).y  = computefEq(den, w[6], e[6], u0);
        h_if5678(pos).z  = computefEq(den, w[7], e[7], u0);
        h_if5678(pos).w  = computefEq(den, w[8], e[8], u0);
        h_type(pos) = (x==0 || x==dims[0]-1 || y==0 || y==dims[1]-1);
      }
    }

    // Device views (double-buffered)
    Kokkos::View<double*>  d_f0_a("d_f0_a", n),   d_f0_b("d_f0_b", n);
    Kokkos::View<double4*> d_f1234_a("d_f1234_a", n), d_f1234_b("d_f1234_b", n);
    Kokkos::View<double4*> d_f5678_a("d_f5678_a", n), d_f5678_b("d_f5678_b", n);
    Kokkos::View<bool*>    d_type("d_type", n);
    Kokkos::View<double*>  d_weight("d_weight", 9);

    Kokkos::deep_copy(d_f0_a,    h_if0);
    Kokkos::deep_copy(d_f1234_a, h_if1234);
    Kokkos::deep_copy(d_f5678_a, h_if5678);
    Kokkos::deep_copy(d_f0_b,    d_f0_a);
    Kokkos::deep_copy(d_f1234_b, d_f1234_a);
    Kokkos::deep_copy(d_f5678_b, d_f5678_a);
    Kokkos::deep_copy(d_type,    h_type);
    {
      auto h_w = Kokkos::create_mirror_view(d_weight);
      for (int i = 0; i < 9; i++) h_w(i) = w[i];
      Kokkos::deep_copy(d_weight, h_w);
    }

    // Current input/output views (pointers to views for swap)
    Kokkos::View<double*>  if0   = d_f0_a,    of0   = d_f0_b;
    Kokkos::View<double4*> if1234 = d_f1234_a, of1234 = d_f1234_b;
    Kokkos::View<double4*> if5678 = d_f5678_a, of5678 = d_f5678_b;

    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
      lbm((unsigned int)dims[0], (unsigned int)dims[1],
          if0.data(), of0.data(),
          if1234.data(), of1234.data(),
          if5678.data(), of5678.data(),
          d_type.data(), dirX, dirY, d_weight.data(), omega);
      Kokkos::fence();
      // Double-buffer swap
      std::swap(if0,    of0);
      std::swap(if1234, of1234);
      std::swap(if5678, of5678);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (s)\n", (time * 1e-9) / iterations);

    // Copy results back (final output is in 'if0' after last swap)
    auto h_out0    = Kokkos::create_mirror_view(if0);
    auto h_out1234 = Kokkos::create_mirror_view(if1234);
    auto h_out5678 = Kokkos::create_mirror_view(if5678);
    Kokkos::deep_copy(h_out0,    if0);
    Kokkos::deep_copy(h_out1234, if1234);
    Kokkos::deep_copy(h_out5678, if5678);

    printf("f0[0]=%f f1234[0].x=%f\n", h_out0(0), h_out1234(0).x);
  }
  Kokkos::finalize();
  return 0;
}
