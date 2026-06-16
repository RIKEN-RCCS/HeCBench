#include <cmath>
#include <cstdio>

template <typename T>
void calcError(T* src, T* dst, int size) {
    double error = 0;
    for (int i = 0; i < size; ++i) {
        double diff = std::fabs(std::fabs(src[i]) - std::fabs(dst[i]));
        if (error < diff) error = diff;
    }
    printf("Maximum error: %e\n", error);
}
