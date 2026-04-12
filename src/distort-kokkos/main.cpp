#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <Kokkos_Core.hpp>

// -----------------------------------------------------------------------
// Types (no OMP dependency)
// -----------------------------------------------------------------------
typedef struct __attribute__((__aligned__(4))) {
    unsigned char x, y, z;
} uchar3;

struct Properties {
    float K;
    float centerX, centerY;
    int   width, height;
    float thresh;
    float xscale, yscale;
    float xshift, yshift;
};

// -----------------------------------------------------------------------
// Host-only recursive helper (computes barrel-distort shift boundary)
// -----------------------------------------------------------------------
static float calc_shift(float x1, float x2, float cx, float k, float thresh) {
    float x3      = x1 + (x2 - x1) * 0.5f;
    float result1 = x1 + ((x1 - cx) * k * ((x1 - cx) * (x1 - cx)));
    float result3 = x3 + ((x3 - cx) * k * ((x3 - cx) * (x3 - cx)));

    if (result1 > -thresh && result1 < thresh) return x1;
    if (result3 < 0) return calc_shift(x3, x2, cx, k, thresh);
    else             return calc_shift(x1, x3, cx, k, thresh);
}

// -----------------------------------------------------------------------
// Device-callable helpers (accept Properties by value for lambda capture)
// -----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
float getRadialX(float x, float y, Properties prop) {
    x = x * prop.xscale + prop.xshift;
    y = y * prop.yscale + prop.yshift;
    return x + (x - prop.centerX) * prop.K *
               ((x - prop.centerX) * (x - prop.centerX) +
                (y - prop.centerY) * (y - prop.centerY));
}

KOKKOS_INLINE_FUNCTION
float getRadialY(float x, float y, Properties prop) {
    x = x * prop.xscale + prop.xshift;
    y = y * prop.yscale + prop.yshift;
    return y + (y - prop.centerY) * prop.K *
               ((x - prop.centerX) * (x - prop.centerX) +
                (y - prop.centerY) * (y - prop.centerY));
}

KOKKOS_INLINE_FUNCTION
void sampleImageTest(const uchar3* src, float idx0, float idx1,
                     uchar3& result, Properties prop)
{
    if (idx0 < 0 || idx1 < 0 ||
        idx0 > prop.height - 1 || idx1 > prop.width - 1) {
        result.x = result.y = result.z = 0;
        return;
    }
    int idx0_floor = (int)floorf(idx0);
    int idx0_ceil  = (int)ceilf(idx0);
    int idx1_floor = (int)floorf(idx1);
    int idx1_ceil  = (int)ceilf(idx1);

    uchar3 s1 = src[idx0_floor * prop.width + idx1_floor];
    uchar3 s2 = src[idx0_floor * prop.width + idx1_ceil];
    uchar3 s3 = src[idx0_ceil  * prop.width + idx1_ceil];
    uchar3 s4 = src[idx0_ceil  * prop.width + idx1_floor];

    float fx = idx0 - idx0_floor;
    float fy = idx1 - idx1_floor;

    result.x = (unsigned char)(s1.x * (1.f - fx) * (1.f - fy) + s2.x * (1.f - fx) * fy +
                                s3.x * fx * fy + s4.x * fx * (1.f - fy));
    result.y = (unsigned char)(s1.y * (1.f - fx) * (1.f - fy) + s2.y * (1.f - fx) * fy +
                                s3.y * fx * fy + s4.y * fx * (1.f - fy));
    result.z = (unsigned char)(s1.z * (1.f - fx) * (1.f - fy) + s2.z * (1.f - fx) * fy +
                                s3.z * fx * fy + s4.z * fx * (1.f - fy));
}

// -----------------------------------------------------------------------
// CPU reference
// -----------------------------------------------------------------------
static void reference(const uchar3* src, uchar3* dst, const Properties* prop) {
    for (int h = 0; h < prop->height; h++) {
        for (int w = 0; w < prop->width; w++) {
            float rx = getRadialX((float)w, (float)h, *prop);
            float ry = getRadialY((float)w, (float)h, *prop);
            uchar3 temp;
            sampleImageTest(src, ry, rx, temp, *prop);
            dst[h * prop->width + w] = temp;
        }
    }
}

// -----------------------------------------------------------------------
// Kokkos kernel wrapper
// -----------------------------------------------------------------------
static void barrel_distort(Kokkos::View<uchar3*> d_src,
                           Kokkos::View<uchar3*> d_dst,
                           Properties prop)
{
    int height = prop.height;
    int width  = prop.width;
    Kokkos::parallel_for("barrel_distort", height * width,
        KOKKOS_LAMBDA(int idx) {
            int h = idx / width;
            int w = idx % width;
            float rx = getRadialX((float)w, (float)h, prop);
            float ry = getRadialY((float)w, (float)h, prop);
            uchar3 temp;
            sampleImageTest(d_src.data(), ry, rx, temp, prop);
            d_dst[idx] = temp;
        });
    Kokkos::fence();
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc != 5) {
        std::cout << "Usage: " << argv[0]
                  << " <input image width> <input image height>"
                  << " <coefficient of distortion> <repeat>\n";
        return 1;
    }

    const int   width  = atoi(argv[1]);
    const int   height = atoi(argv[2]);
    const float K      = atof(argv[3]);
    const int   repeat = atoi(argv[4]);

    Properties prop;
    prop.K       = K;
    prop.centerX = (float)(width  / 2);
    prop.centerY = (float)(height / 2);
    prop.width   = width;
    prop.height  = height;
    prop.thresh  = 1.f;

    prop.xshift = calc_shift(0, prop.centerX - 1, prop.centerX, K, prop.thresh);
    float newcenterX = (float)(width  - width  / 2);
    float xshift_2   = calc_shift(0, newcenterX - 1, newcenterX, K, prop.thresh);

    prop.yshift = calc_shift(0, prop.centerY - 1, prop.centerY, K, prop.thresh);
    float newcenterY = (float)(height - height / 2);
    float yshift_2   = calc_shift(0, newcenterY - 1, newcenterY, K, prop.thresh);

    prop.xscale = (float)(width  - prop.xshift - xshift_2) / (float)width;
    prop.yscale = (float)(height - prop.yshift - yshift_2) / (float)height;

    const int imageSize = height * width;

    uchar3* h_src = (uchar3*) malloc(imageSize * sizeof(uchar3));
    uchar3* h_dst = (uchar3*) malloc(imageSize * sizeof(uchar3));
    uchar3* r_dst = (uchar3*) malloc(imageSize * sizeof(uchar3));

    srand(123);
    for (int i = 0; i < imageSize; i++) {
        h_src[i] = { (unsigned char)(rand() % 256),
                     (unsigned char)(rand() % 256),
                     (unsigned char)(rand() % 256) };
    }

    Kokkos::initialize(argc, argv);
    {
        Kokkos::View<uchar3*> d_src("src", imageSize);
        Kokkos::View<uchar3*> d_dst("dst", imageSize);

        {
            auto hs = Kokkos::create_mirror_view(d_src);
            for (int i = 0; i < imageSize; i++) hs(i) = h_src[i];
            Kokkos::deep_copy(d_src, hs);
        }

        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat; i++)
            barrel_distort(d_src, d_dst, prop);
        auto end  = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

        // Copy result back
        {
            auto hd = Kokkos::create_mirror_view(d_dst);
            Kokkos::deep_copy(hd, d_dst);
            for (int i = 0; i < imageSize; i++) h_dst[i] = hd(i);
        }
    }
    Kokkos::finalize();

    // Verify
    reference(h_src, r_dst, &prop);
    int ex = 0, ey = 0, ez = 0;
    for (int i = 0; i < imageSize; i++) {
        int dx = abs((int)h_dst[i].x - (int)r_dst[i].x);
        int dy = abs((int)h_dst[i].y - (int)r_dst[i].y);
        int dz = abs((int)h_dst[i].z - (int)r_dst[i].z);
        if (dx > ex) ex = dx;
        if (dy > ey) ey = dy;
        if (dz > ez) ez = dz;
    }
    std::cout << "Max error of each channel: " << ex << " " << ey << " " << ez << std::endl;

    free(h_src);
    free(h_dst);
    free(r_dst);
    return 0;
}
