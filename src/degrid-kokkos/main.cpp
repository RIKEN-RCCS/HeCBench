#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <Kokkos_Core.hpp>

// ------ Config (mirrors degrid.h) ------
#ifndef NPOINTS
#define NPOINTS   40000
#endif
#ifndef GCF_DIM
#define GCF_DIM   256
#endif
#ifndef IMG_SIZE
#define IMG_SIZE  8192
#endif
#ifndef GCF_GRID
#define GCF_GRID  8
#endif
#ifndef REPEAT
#define REPEAT    100
#endif

#ifndef PRECISION
#define PRECISION float
#endif

struct float2  { float  x, y; };
struct double2 { double x, y; };

#if defined(DOUBLE_PRECISION)
  typedef double  PREC;
  typedef double2 PREC2;
#else
  typedef float  PREC;
  typedef float2 PREC2;
#endif

// ------ CPU reference ------
void degridCPU(PREC2* out, const PREC2* in, const PREC2* img, const PREC2* gcf_mid)
{
  for (size_t n = 0; n < NPOINTS; n++) {
    int sub_x  = (int)floorf(GCF_GRID * (in[n].x - floorf(in[n].x)));
    int sub_y  = (int)floorf(GCF_GRID * (in[n].y - floorf(in[n].y)));
    int main_x = (int)floorf(in[n].x);
    int main_y = (int)floorf(in[n].y);
    PREC sum_r = 0, sum_i = 0;
    for (int a = -GCF_DIM/2; a < GCF_DIM/2; a++)
      for (int b = -GCF_DIM/2; b < GCF_DIM/2; b++) {
        if (main_x+a < 0 || main_y+b < 0 ||
            main_x+a >= IMG_SIZE || main_y+b >= IMG_SIZE) continue;
        PREC r1 = img[main_x+a + IMG_SIZE*(main_y+b)].x;
        PREC i1 = img[main_x+a + IMG_SIZE*(main_y+b)].y;
        PREC r2 = gcf_mid[(int64_t)GCF_DIM*GCF_DIM*(GCF_GRID*sub_y+sub_x) + GCF_DIM*b+a].x;
        PREC i2 = gcf_mid[(int64_t)GCF_DIM*GCF_DIM*(GCF_GRID*sub_y+sub_x) + GCF_DIM*b+a].y;
        sum_r += r1*r2 - i1*i2;
        sum_i += r1*i2 + r2*i1;
      }
    out[n].x = sum_r;
    out[n].y = sum_i;
  }
}

void init_gcf(PREC2* gcf, size_t sz) {
  for (size_t sx = 0; sx < GCF_GRID; sx++)
    for (size_t sy = 0; sy < GCF_GRID; sy++)
      for (size_t x = 0; x < sz; x++)
        for (size_t y = 0; y < sz; y++) {
          PREC tmp = (PREC)(sin(6.28*x/sz/GCF_GRID)*exp(-(1.0*x*x+1.0*y*y*sy)/sz/sz/2));
          gcf[sz*sz*(sx+sy*GCF_GRID)+x+y*sz].x = (PREC)(tmp*sin(1.0*x*sx/(y+1)));
          gcf[sz*sz*(sx+sy*GCF_GRID)+x+y*sz].y = (PREC)(tmp*cos(1.0*x*sx/(y+1)));
        }
}

template <int compare>
int w_comp_sub(const void* A, const void* B) {
  float quota, rema, quotb, remb;
  rema = modff(((PREC2*)A)->x, &quota);
  remb = modff(((PREC2*)B)->x, &quotb);
  int sub_xa = (int)(GCF_GRID*rema);
  int sub_xb = (int)(GCF_GRID*remb);
  rema = modff(((PREC2*)A)->y, &quota);
  remb = modff(((PREC2*)B)->y, &quotb);
  int suba = (int)(GCF_GRID*rema) + GCF_GRID*sub_xa;
  int subb = (int)(GCF_GRID*remb) + GCF_GRID*sub_xb;
  if (suba > subb) return  1;
  if (suba < subb) return -1;
  return 0;
}

int main()
{
  const size_t img_padded = (size_t)IMG_SIZE*IMG_SIZE + 2*IMG_SIZE*GCF_DIM + 2*GCF_DIM;
  const size_t gcf_size   = 64LL * GCF_DIM * GCF_DIM;

  PREC2* out     = (PREC2*)malloc(sizeof(PREC2) * NPOINTS);
  PREC2* in      = (PREC2*)malloc(sizeof(PREC2) * NPOINTS);
  PREC2* img_raw = (PREC2*)malloc(sizeof(PREC2) * img_padded);
  PREC2* gcf     = (PREC2*)malloc(sizeof(PREC2) * gcf_size);

  std::cout << "img size in bytes: " << img_padded * sizeof(PREC2) << std::endl;
  std::cout << "out size in bytes: " << NPOINTS   * sizeof(PREC2) << std::endl;

  // Apply padding offset so img_raw[offset] is the real (0,0)
  const size_t IMG_OFFSET = (size_t)IMG_SIZE * GCF_DIM + GCF_DIM;
  PREC2* img = img_raw + IMG_OFFSET;

  init_gcf(gcf, GCF_DIM);
  srand(2541617);
  for (size_t n = 0; n < NPOINTS; n++) {
    in[n].x = ((float)rand()/(float)RAND_MAX) * 8000;
    in[n].y = ((float)rand()/(float)RAND_MAX) * 8000;
  }
  for (size_t x = 0; x < IMG_SIZE; x++)
    for (size_t y = 0; y < IMG_SIZE; y++) {
      img[x + IMG_SIZE*y].x = (PREC)(exp(-((x-1400.0)*(x-1400.0)+(y-3800.0)*(y-3800.0))/8000000.0)+1.0);
      img[x + IMG_SIZE*y].y = (PREC)0.4;
    }
  // Zero padding regions
  for (size_t x = 0; x < IMG_OFFSET; x++)       { img_raw[x].x = 0; img_raw[x].y = 0; }
  for (size_t x = 0; x < IMG_OFFSET; x++)       {
    img_raw[IMG_OFFSET + (size_t)IMG_SIZE*IMG_SIZE + x].x = 0;
    img_raw[IMG_OFFSET + (size_t)IMG_SIZE*IMG_SIZE + x].y = 0;
  }

  std::qsort(in, NPOINTS, sizeof(PREC2), w_comp_sub<0>);

  // ---- GPU degrid via Kokkos ----
  std::cout << "Computing on GPU..." << std::endl;

  Kokkos::initialize();
  {
    // Transfer img (full padded), gcf, in to device
    Kokkos::View<PREC2*> img_d("img_d", img_padded);
    Kokkos::View<PREC2*> gcf_d("gcf_d", gcf_size);
    Kokkos::View<PREC2*> in_d ("in_d",  NPOINTS);
    Kokkos::View<PREC2*> out_d("out_d", NPOINTS);

    {
      auto m = Kokkos::create_mirror_view(img_d);
      memcpy(m.data(), img_raw, sizeof(PREC2) * img_padded);
      Kokkos::deep_copy(img_d, m);
    }
    {
      auto m = Kokkos::create_mirror_view(gcf_d);
      memcpy(m.data(), gcf, sizeof(PREC2) * gcf_size);
      Kokkos::deep_copy(gcf_d, m);
    }
    {
      auto m = Kokkos::create_mirror_view(in_d);
      memcpy(m.data(), in, sizeof(PREC2) * NPOINTS);
      Kokkos::deep_copy(in_d, m);
    }

    auto start = std::chrono::steady_clock::now();

    for (int rep = 0; rep < REPEAT; rep++) {
      // img pointer with offset, gcf pointer with half-dim offset
      const int64_t img_off = IMG_OFFSET;
      const int64_t gcf_off = (int64_t)GCF_DIM * (GCF_DIM + 1) / 2;

      Kokkos::parallel_for("degrid", NPOINTS,
        KOKKOS_LAMBDA(const int n) {
          const PREC2* in_ptr  = in_d.data();
          const PREC2* img_ptr = img_d.data() + img_off;
          const PREC2* gcf_ptr = gcf_d.data() + gcf_off;
          PREC2*       out_ptr = out_d.data();

          int sub_x  = (int)floorf((PREC)GCF_GRID * (in_ptr[n].x - floorf(in_ptr[n].x)));
          int sub_y  = (int)floorf((PREC)GCF_GRID * (in_ptr[n].y - floorf(in_ptr[n].y)));
          int main_x = (int)floorf(in_ptr[n].x);
          int main_y = (int)floorf(in_ptr[n].y);
          PREC sum_r = 0, sum_i = 0;

          for (int a = -GCF_DIM/2; a < GCF_DIM/2; a++) {
            for (int b = -GCF_DIM/2; b < GCF_DIM/2; b++) {
              PREC r1 = img_ptr[main_x+a + IMG_SIZE*(main_y+b)].x;
              PREC i1 = img_ptr[main_x+a + IMG_SIZE*(main_y+b)].y;
              PREC r2 = gcf_ptr[(int64_t)GCF_DIM*GCF_DIM*(GCF_GRID*sub_y+sub_x) + GCF_DIM*b+a].x;
              PREC i2 = gcf_ptr[(int64_t)GCF_DIM*GCF_DIM*(GCF_GRID*sub_y+sub_x) + GCF_DIM*b+a].y;
              if (main_x+a >= 0 && main_y+b >= 0 &&
                  main_x+a < IMG_SIZE && main_y+b < IMG_SIZE) {
                sum_r += r1*r2 - i1*i2;
                sum_i += r1*i2 + r2*i1;
              }
            }
          }
          out_ptr[n].x = sum_r;
          out_ptr[n].y = sum_i;
        });
      Kokkos::fence();
    }

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time " << (time * 1e-9f) / REPEAT << " (s)\n";

    auto out_m = Kokkos::create_mirror_view(out_d);
    Kokkos::deep_copy(out_m, out_d);
    memcpy(out, out_m.data(), sizeof(PREC2) * NPOINTS);
  }
  Kokkos::finalize();

  // ---- CPU reference ----
  std::cout << "Computing on CPU..." << std::endl;
  PREC2* out_cpu = (PREC2*)malloc(sizeof(PREC2) * NPOINTS);
  const PREC2* gcf_mid = gcf + (int64_t)GCF_DIM*(GCF_DIM+1)/2;
  degridCPU(out_cpu, in, img, gcf_mid);

  std::cout << "Checking results against CPU:" << std::endl;
  PREC EPS = (sizeof(PREC) == sizeof(double)) ? 1e-7 : 1e-1f;
  std::cout << "Error bound: " << EPS << std::endl;

  bool ok = true;
  for (size_t n = 0; n < NPOINTS; n++) {
    if (fabsf((float)(out[n].x - out_cpu[n].x)) > (float)EPS ||
        fabsf((float)(out[n].y - out_cpu[n].y)) > (float)EPS) {
      ok = false;
      std::cout << n << ": F(" << in[n].x << "," << in[n].y << ") = "
                << out[n].x << "," << out[n].y
                << " vs. " << out_cpu[n].x << "," << out_cpu[n].y << std::endl;
      break;
    }
  }
  std::cout << (ok ? "PASS" : "FAIL") << std::endl;

  free(out); free(in); free(img_raw); free(gcf); free(out_cpu);
  return 0;
}
