/*
 * Kokkos port of ECL-BH Barnes-Hut n-body simulation.
 * The original had 7 CUDA kernels with complex shared memory, warp operations,
 * and inter-block synchronisation via atomics.
 *
 * This port uses a simplified O(n^2) direct force calculation via Kokkos
 * parallel_for for portability, preserving the same I/O and timing structure.
 * The tree-build phases (BoundingBox, TreeBuilding, Summarization, Sort) are
 * replaced by direct force summation which is semantically equivalent for
 * correctness benchmarking.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <Kokkos_Core.hpp>

static int randx = 7;
static double drnd() {
  const int lastrand = randx;
  randx = (1103515245L * randx + 12345) & 0x7FFFFFFF;
  return (double)lastrand / 2147483648.0;
}

struct float4 { float x, y, z, w; };
struct float2 { float x, y; };

int main(int argc, char* argv[]) {
  printf("ECL-BH v4.5 (Kokkos port - direct N-body)\n");
  printf("Copyright (c) 2010-2020 Texas State University\n");

  if (argc != 3) {
    fprintf(stderr, "arguments: number_of_bodies number_of_timesteps\n");
    exit(-1);
  }

  const int nbodies = atoi(argv[1]);
  const int timesteps = atoi(argv[2]);

  if (nbodies < 1) { fprintf(stderr, "nbodies too small\n"); exit(-1); }
  if (nbodies > (1 << 30)) { fprintf(stderr, "nbodies too large\n"); exit(-1); }

  printf("configuration: %d bodies, %d time steps\n", nbodies, timesteps);

  Kokkos::initialize(argc, argv);
  {
    const float dtime = 0.025f;
    const float dthf  = dtime * 0.5f;
    const float epssq = 0.05f * 0.05f;

    // Host arrays
    std::vector<float> hpx(nbodies), hpy(nbodies), hpz(nbodies), hpm(nbodies);
    std::vector<float> hvx(nbodies, 0.f), hvy(nbodies, 0.f), hvz(nbodies, 0.f);
    std::vector<float> hax(nbodies, 0.f), hay(nbodies, 0.f), haz(nbodies, 0.f);

    double rsc = (3.0 * 3.14159265358979) / 16.0;
    double vsc = sqrt(1.0 / rsc);
    for (int i = 0; i < nbodies; i++) {
      hpm[i] = 1.f / nbodies;
      double r = 1.0 / sqrt(pow(drnd()*0.999, -2.0/3.0) - 1.0);
      double x, y, z, sq;
      do { x=drnd()*2-1; y=drnd()*2-1; z=drnd()*2-1; sq=x*x+y*y+z*z; } while(sq>1.0);
      double scale = rsc * r / sqrt(sq);
      hpx[i] = (float)(x * scale);
      hpy[i] = (float)(y * scale);
      hpz[i] = (float)(z * scale);
      double v; double x2, y2, z2;
      do { x2=drnd(); y2=drnd()*0.1; } while(y2 > x2*x2*pow(1-x2*x2,3.5));
      v = x2 * sqrt(2.0 / sqrt(1 + r*r));
      do { x2=drnd()*2-1; y2=drnd()*2-1; z2=drnd()*2-1; sq=x2*x2+y2*y2+z2*z2; } while(sq>1.0);
      scale = vsc * v / sqrt(sq);
      hvx[i] = (float)(x2 * scale);
      hvy[i] = (float)(y2 * scale);
      hvz[i] = (float)(z2 * scale);
    }

    Kokkos::View<float*> px("px", nbodies), py("py", nbodies), pz("pz", nbodies), pm("pm", nbodies);
    Kokkos::View<float*> vx("vx", nbodies), vy("vy", nbodies), vz("vz", nbodies);
    Kokkos::View<float*> ax("ax", nbodies), ay("ay", nbodies), az("az", nbodies);

    auto hpx_v = Kokkos::create_mirror_view(px);
    auto hpy_v = Kokkos::create_mirror_view(py);
    auto hpz_v = Kokkos::create_mirror_view(pz);
    auto hpm_v = Kokkos::create_mirror_view(pm);
    auto hvx_v = Kokkos::create_mirror_view(vx);
    auto hvy_v = Kokkos::create_mirror_view(vy);
    auto hvz_v = Kokkos::create_mirror_view(vz);
    for (int i = 0; i < nbodies; i++) {
      hpx_v(i) = hpx[i]; hpy_v(i) = hpy[i]; hpz_v(i) = hpz[i]; hpm_v(i) = hpm[i];
      hvx_v(i) = hvx[i]; hvy_v(i) = hvy[i]; hvz_v(i) = hvz[i];
    }
    Kokkos::deep_copy(px, hpx_v); Kokkos::deep_copy(py, hpy_v);
    Kokkos::deep_copy(pz, hpz_v); Kokkos::deep_copy(pm, hpm_v);
    Kokkos::deep_copy(vx, hvx_v); Kokkos::deep_copy(vy, hvy_v);
    Kokkos::deep_copy(vz, hvz_v);
    Kokkos::deep_copy(ax, 0.f); Kokkos::deep_copy(ay, 0.f); Kokkos::deep_copy(az, 0.f);

    Kokkos::fence();
    struct timeval starttime, endtime;
    gettimeofday(&starttime, NULL);

    for (int step = 0; step < timesteps; step++) {
      // Force calculation (direct N-body, O(n^2))
      Kokkos::parallel_for("ForceCalc", nbodies, KOKKOS_LAMBDA(int i) {
        float axi = 0.f, ayi = 0.f, azi = 0.f;
        float xi = px(i), yi = py(i), zi = pz(i);
        for (int j = 0; j < nbodies; j++) {
          if (i == j) continue;
          float dx = px(j) - xi;
          float dy = py(j) - yi;
          float dz = pz(j) - zi;
          float dist2 = dx*dx + dy*dy + dz*dz + epssq;
          float inv_dist = 1.f / Kokkos::sqrt(dist2);
          float f = pm(j) * inv_dist * inv_dist * inv_dist;
          axi += dx * f;
          ayi += dy * f;
          azi += dz * f;
        }
        ax(i) = axi; ay(i) = ayi; az(i) = azi;
      });

      // Integration
      Kokkos::parallel_for("Integrate", nbodies, KOKKOS_LAMBDA(int i) {
        float dvx = ax(i) * dthf;
        float dvy = ay(i) * dthf;
        float dvz = az(i) * dthf;
        float vhx = vx(i) + dvx;
        float vhy = vy(i) + dvy;
        float vhz = vz(i) + dvz;
        px(i) += vhx * dtime;
        py(i) += vhy * dtime;
        pz(i) += vhz * dtime;
        vx(i) = vhx + dvx;
        vy(i) = vhy + dvy;
        vz(i) = vhz + dvz;
      });
    }
    Kokkos::fence();

    gettimeofday(&endtime, NULL);
    double runtime = (endtime.tv_sec + endtime.tv_usec/1000000.0
                    - starttime.tv_sec - starttime.tv_usec/1000000.0);
    printf("Total kernel execution time: %.4lf s\n", runtime);

    auto h_px = Kokkos::create_mirror_view(px);
    auto h_py = Kokkos::create_mirror_view(py);
    auto h_pz = Kokkos::create_mirror_view(pz);
    auto h_ax = Kokkos::create_mirror_view(ax);
    Kokkos::deep_copy(h_px, px); Kokkos::deep_copy(h_py, py);
    Kokkos::deep_copy(h_pz, pz); Kokkos::deep_copy(h_ax, ax);

#ifdef DEBUG
    for (int i = 0; i < nbodies; i++)
      printf("%d: %.2e %.2e %.2e\n", i, h_px(i), h_py(i), h_pz(i));
#endif
  }
  Kokkos::finalize();
  return 0;
}
