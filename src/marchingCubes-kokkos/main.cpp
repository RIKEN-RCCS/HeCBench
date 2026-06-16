#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cmath>
#include <chrono>
#include "tables.h"

// Problem size: N x N x N grid, N-1 voxels per dimension
constexpr unsigned int N    = 1024;
constexpr unsigned int Nd2  = N / 2;

// Implicit surface function (faithfully ported from CUDA source, including
// the intentional use of z for the yf component).
KOKKOS_INLINE_FUNCTION
float f_val(unsigned int x, unsigned int y, unsigned int z)
{
    constexpr float d = 2.0f / N;
    float xf = (int(x) - int(Nd2)) * d;
    float yf = (int(z) - int(Nd2)) * d;   // uses z (matches CUDA source)
    float zf = (int(z) - int(Nd2)) * d;
    return 1.f - 16.f * xf * yf * zf - 4.f * (xf*xf + yf*yf + zf*zf);
}

// Portable popcount
KOKKOS_INLINE_FUNCTION
int popcount32(unsigned int v)
{
    int c = 0;
    while (v) { c += v & 1; v >>= 1; }
    return c;
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        printf("Usage: %s <isoValue> <repeat>\n", argv[0]);
        return 1;
    }
    float    isoValue = atof(argv[1]);
    unsigned repeat   = (unsigned)atoi(argv[2]);

    Kokkos::initialize(argc, argv);
    {
        // Copy lookup tables to device
        Kokkos::View<unsigned int*> d_edgeTable("edgeTable", 256);
        Kokkos::View<int*>          d_triTable ("triTable",  256 * 16);
        {
            auto h_et = Kokkos::create_mirror_view(d_edgeTable);
            auto h_tt = Kokkos::create_mirror_view(d_triTable);
            for (int i = 0; i < 256;      i++) h_et(i) = edgeTable[i];
            for (int i = 0; i < 256 * 16; i++) h_tt(i) = triTable[i];
            Kokkos::deep_copy(d_edgeTable, h_et);
            Kokkos::deep_copy(d_triTable,  h_tt);
        }

        // Total voxels: (N-1)^3
        const long long Nv = (long long)(N - 1);
        const long long totalVoxels = Nv * Nv * Nv;

        unsigned int activeVoxels = 0, totalVertices = 0, totalTriangles = 0;
        float elapsed = 0.f;

        for (unsigned r = 0; r < repeat; r++) {
            unsigned int av = 0, verts = 0, tris = 0;

            Kokkos::fence();
            auto t0 = std::chrono::steady_clock::now();

            // Kokkos reducer for three simultaneous sums
            // Use a struct-based reducer via parallel_reduce with array
            Kokkos::parallel_reduce(
                "marchingCubes",
                Kokkos::RangePolicy<>(0, (int)totalVoxels),
                KOKKOS_LAMBDA(long long idx, unsigned int& lav,
                              unsigned int& lverts, unsigned int& ltris) {
                    // Decode voxel coordinates
                    long long tmp = idx;
                    unsigned int xi = (unsigned int)(tmp % Nv); tmp /= Nv;
                    unsigned int yi = (unsigned int)(tmp % Nv); tmp /= Nv;
                    unsigned int zi = (unsigned int)(tmp);

                    // Evaluate f at 8 corners of this voxel
                    float v[8];
                    v[0] = f_val(xi,   yi,   zi  );
                    v[1] = f_val(xi+1, yi,   zi  );
                    v[2] = f_val(xi+1, yi+1, zi  );
                    v[3] = f_val(xi,   yi+1, zi  );
                    v[4] = f_val(xi,   yi,   zi+1);
                    v[5] = f_val(xi+1, yi,   zi+1);
                    v[6] = f_val(xi+1, yi+1, zi+1);
                    v[7] = f_val(xi,   yi+1, zi+1);

                    // Quick active-voxel test
                    float vmin = v[0], vmax = v[0];
                    for (int k = 1; k < 8; k++) {
                        if (v[k] < vmin) vmin = v[k];
                        if (v[k] > vmax) vmax = v[k];
                    }
                    if (isoValue < vmin || isoValue > vmax) return;

                    lav++;

                    // Compute cube case
                    unsigned int cubeCase = 0;
                    if (v[0] < isoValue) cubeCase |= 1;
                    if (v[1] < isoValue) cubeCase |= 2;
                    if (v[2] < isoValue) cubeCase |= 4;
                    if (v[3] < isoValue) cubeCase |= 8;
                    if (v[4] < isoValue) cubeCase |= 16;
                    if (v[5] < isoValue) cubeCase |= 32;
                    if (v[6] < isoValue) cubeCase |= 64;
                    if (v[7] < isoValue) cubeCase |= 128;

                    if (d_edgeTable[cubeCase] == 0) return;

                    // Count vertices = number of set bits in edgeTable entry
                    lverts += (unsigned int)popcount32(d_edgeTable[cubeCase]);

                    // Count triangles from triTable
                    int base = cubeCase * 16;
                    int numTris = 0;
                    for (int t = 0; t < 15; t += 3) {
                        if (d_triTable[base + t] < 0) break;
                        numTris++;
                    }
                    ltris += (unsigned int)numTris;
                },
                av, verts, tris
            );
            Kokkos::fence();

            auto t1 = std::chrono::steady_clock::now();
            elapsed += std::chrono::duration<float>(t1 - t0).count();

            activeVoxels  = av;
            totalVertices = verts;
            totalTriangles = tris;
        }

        printf("Active voxels: %u\n", activeVoxels);
        printf("Total vertices: %u\n", totalVertices);
        printf("Total triangles: %u\n", totalTriangles);
        printf("Average kernel time: %f s\n", elapsed / repeat);
    }
    Kokkos::finalize();
    return 0;
}
