#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <chrono>

// ---------------------------------------------------------------------------
// Points-to-image projection (Daphne / Autoware points2image kernel)
//
// Given a LiDAR point cloud and camera parameters, project each point onto
// the image plane and store the depth (cm_depth = z * 100) at the closest
// point using atomic min.
//
// Algorithm (matches CUDA compute_point_from_pointcloud):
//   1. Transform world point by extrinsic matrix (invR * p + invT)
//   2. Discard points with z <= 2.5 (behind / too-close camera)
//   3. Normalise: xn = cx/cz, yn = cy/cz
//   4. Apply Brown-Conrady lens distortion (k1,k2,k3,p1,p2)
//   5. Project via intrinsic: u = fx*xd + ppx, v = fy*yd + ppy
//   6. Atomically store closest depth at pixel (u,v)
// ---------------------------------------------------------------------------

// Image resolution
static constexpr int IMG_W = 800;
static constexpr int IMG_H = 600;

// Layout in result buffer: planar, matching CUDA result_buffer layout
//   result[0 .. H*W-1]        = distance  (depth * 100, in cm)
//   result[H*W .. 2*H*W-1]    = intensity
//   result[2*H*W .. 3*H*W-1]  = min_height (fixed at -1.25 wherever hit)
//   result[3*H*W .. 4*H*W-1]  = max_height (unused in this port, zeroed)
static constexpr int PLANE = IMG_H * IMG_W; // pixels per plane

// Atomic min for positive floats using CAS.
// IEEE-754 positive float ordering is preserved by their bit pattern as uint32,
// but we operate on the float pointer directly via Kokkos CAS.
KOKKOS_INLINE_FUNCTION
void atomic_float_min(float* addr, float val) {
  float assumed = *addr;
  while (val < assumed) {
    float old = Kokkos::atomic_compare_exchange(addr, assumed, val);
    if (old == assumed) break; // CAS succeeded
    assumed = old;             // retry with refreshed value
  }
}

// ---------------------------------------------------------------------------
// Camera and point-cloud types
// ---------------------------------------------------------------------------
struct Mat33 { double data[3][3]; };
struct Mat13 { double data[3]; };
struct Vec5  { double data[5]; }; // {k1, k2, p1, p2, k3}

struct Point4f { float x, y, z, intensity; };

// ---------------------------------------------------------------------------
// Host helper: compute invR (transpose of 3×3 rotation) and invT from
// the full 4×4 extrinsic matrix.
// ---------------------------------------------------------------------------
static void compute_inv_extrinsic(
    const double E[4][4], Mat33& invR, Mat13& invT)
{
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++)
      invR.data[r][c] = E[c][r]; // transpose

  for (int r = 0; r < 3; r++) {
    invT.data[r] = 0.0;
    for (int c = 0; c < 3; c++)
      invT.data[r] -= invR.data[r][c] * E[c][3];
  }
}

// ---------------------------------------------------------------------------
// Projection kernel (one thread per point)
// ---------------------------------------------------------------------------
void project_points(
    Kokkos::View<Point4f*>  points,
    Kokkos::View<float*>    result,
    int                     num_points,
    Mat33                   invR,
    Mat13                   invT,
    Mat33                   cam,    // intrinsic matrix
    Vec5                    dist,   // {k1, k2, p1, p2, k3}
    int                     W,
    int                     H)
{
  Kokkos::parallel_for(
    "points2image",
    num_points,
    KOKKOS_LAMBDA(int idx) {
      Point4f p = points(idx);

      // Step 1: extrinsic transform
      double pt[3];
      for (int r = 0; r < 3; r++) {
        pt[r] = invT.data[r];
        pt[r] += invR.data[r][0] * (double)p.x
               + invR.data[r][1] * (double)p.y
               + invR.data[r][2] * (double)p.z;
      }

      // Discard points too close to (or behind) the camera
      if (pt[2] <= 2.5) return;

      // Step 2: perspective divide
      double xn = pt[0] / pt[2];
      double yn = pt[1] / pt[2];

      // Step 3: Brown-Conrady distortion
      double r2     = xn * xn + yn * yn;
      double r4     = r2 * r2;
      double r6     = r4 * r2;
      double radial = 1.0 + dist.data[0]*r2 + dist.data[1]*r4 + dist.data[4]*r6;
      double xd = xn * radial
                + 2.0 * dist.data[2] * xn * yn
                + dist.data[3] * (r2 + 2.0 * xn * xn);
      double yd = yn * radial
                + dist.data[2] * (r2 + 2.0 * yn * yn)
                + 2.0 * dist.data[3] * xn * yn;

      // Step 4: intrinsic projection
      double u = cam.data[0][0] * xd + cam.data[0][2];
      double v = cam.data[1][1] * yd + cam.data[1][2];

      int px = (int)(u + 0.5);
      int py = (int)(v + 0.5);
      if (px < 0 || px >= W || py < 0 || py >= H) return;

      int pid = py * W + px;

      // Depth in cm (matches CUDA: cm_point = point.data[2] * 100)
      float cm_depth = (float)(pt[2] * 100.0);

      // Atomic min on distance channel
      float* dist_ptr = result.data() + pid; // plane 0 = distance
      atomic_float_min(dist_ptr, cm_depth);

      // Update intensity wherever this point is the closest
      // (best-effort: not perfectly deterministic, matches CUDA behaviour)
      if (result(pid) >= cm_depth) {
        result(pid + PLANE) = p.intensity;
      }
      result(pid + 2 * PLANE) = -1.25f; // min_height fixed value
    });
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    if (argc < 3) {
      printf("Usage: %s <num_points> <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }
    const int num_points = atoi(argv[1]);
    const int repeat     = atoi(argv[2]);

    // ------------------------------------------------------------------
    // Synthetic camera parameters (pinhole, no distortion)
    // ------------------------------------------------------------------
    // Intrinsic matrix K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
    Mat33 cam{};
    cam.data[0][0] = 500.0; // fx
    cam.data[0][2] = 400.0; // cx (principal point x)
    cam.data[1][1] = 500.0; // fy
    cam.data[1][2] = 300.0; // cy (principal point y)
    cam.data[2][2] = 1.0;

    // Distortion coefficients {k1, k2, p1, p2, k3}
    Vec5 dist{};
    dist.data[0] = -0.198;  // k1
    dist.data[1] =  0.028;  // k2
    dist.data[2] =  0.0;    // p1
    dist.data[3] =  0.0;    // p2
    dist.data[4] =  0.0;    // k3

    // Extrinsic: identity (camera at world origin, looking down +z)
    double E[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
    Mat33 invR{};
    Mat13 invT{};
    compute_inv_extrinsic(E, invR, invT);

    // ------------------------------------------------------------------
    // Synthetic point cloud: random points visible to the camera
    // ------------------------------------------------------------------
    Kokkos::View<Point4f*> points("points", num_points);
    {
      auto h_pts = Kokkos::create_mirror_view(points);
      srand(42);
      const double fx = cam.data[0][0], fy = cam.data[1][1];
      const double cx = cam.data[0][2], cy = cam.data[1][2];
      for (int i = 0; i < num_points; i++) {
        // Depth uniformly in [3, 103] m (well above the 2.5 m discard threshold)
        float z = 3.f + 100.f * ((float)rand() / RAND_MAX);
        // x, y chosen so that most points project into the image FOV
        float x = (float)((cx / fx) * z * (2.0 * rand()/RAND_MAX - 1.0) * 1.1);
        float y = (float)((cy / fy) * z * (2.0 * rand()/RAND_MAX - 1.0) * 1.1);
        Point4f pt; pt.x=x; pt.y=y; pt.z=z; pt.intensity=(float)rand()/RAND_MAX;
        h_pts(i) = pt;
      }
      Kokkos::deep_copy(points, h_pts);
    }

    // ------------------------------------------------------------------
    // Result buffer: 4 planes of H×W floats
    // ------------------------------------------------------------------
    const int result_size = 4 * PLANE;
    Kokkos::View<float*> result("result", result_size);

    // ------------------------------------------------------------------
    // Warm-up
    // ------------------------------------------------------------------
    Kokkos::deep_copy(result, 0.f);
    project_points(points, result, num_points, invR, invT, cam, dist, IMG_W, IMG_H);
    Kokkos::fence();

    // ------------------------------------------------------------------
    // Timed benchmark
    // ------------------------------------------------------------------
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) {
      Kokkos::deep_copy(result, 0.f);
      project_points(points, result, num_points, invR, invT, cam, dist, IMG_W, IMG_H);
      Kokkos::fence();
    }
    auto t1 = std::chrono::steady_clock::now();

    double elapsed_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;
    double throughput = (double)num_points / (elapsed_ms * 1e-3) / 1e6; // MP/s
    printf("Average time per iteration: %.4f ms\n", elapsed_ms);
    printf("Throughput: %.2f M points/s\n", throughput);

    // ------------------------------------------------------------------
    // Verify: count projected pixels and sum distances
    // ------------------------------------------------------------------
    auto h_result = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(h_result, result);

    int   hit_count = 0;
    float dist_sum  = 0.f;
    for (int i = 0; i < PLANE; i++) {
      float d = h_result(i); // distance plane
      if (d > 0.f) { hit_count++; dist_sum += d; }
    }
    printf("Pixels hit: %d / %d\n", hit_count, PLANE);
    printf("Distance sum: %.2f cm\n", dist_sum);
    printf("%s\n", (hit_count > 0) ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
