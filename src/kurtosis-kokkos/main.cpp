#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>

// ── Data types ───────────────────────────────────────────────────────────────

struct storeElement {
    int32_t           tag;
    int               metric;
    unsigned long long time;
    float             value;

    KOKKOS_INLINE_FUNCTION storeElement() : tag(0), metric(0), time(0), value(0) {}
    KOKKOS_INLINE_FUNCTION storeElement(int t, int m, unsigned long long ts, float v)
        : tag(t), metric(m), time(ts), value(v) {}
};

struct kurtosisResult {
    int   count;
    float mean, m2, m3, m4;

    KOKKOS_INLINE_FUNCTION kurtosisResult() : count(0), mean(0), m2(0), m3(0), m4(0) {}
    KOKKOS_INLINE_FUNCTION kurtosisResult(int c, float mn, float M2, float M3, float M4)
        : count(c), mean(mn), m2(M2), m3(M3), m4(M4) {}
};

// ── Parallel-safe binary combine (matches Thrust implementation) ──────────────

KOKKOS_INLINE_FUNCTION
kurtosisResult kurtosis_combine(const kurtosisResult& x, const kurtosisResult& y) {
    float xc = (float)x.count;
    float yc = (float)y.count;
    float countf  = xc + yc;
    if (countf == 0.0f) return kurtosisResult();
    float count2  = countf * countf;
    float count3  = count2 * countf;

    float delta   = y.mean - x.mean;
    float delta2  = delta  * delta;
    float delta3  = delta2 * delta;
    float delta4  = delta3 * delta;

    kurtosisResult r;
    r.count = x.count + y.count;
    r.mean  = x.mean + delta * yc / countf;

    r.m2 = x.m2 + y.m2
         + delta2 * xc * yc / countf;

    r.m3 = x.m3 + y.m3
         + delta3 * xc * yc * (xc - yc) / count2
         + 3.0f  * delta * (xc * y.m2 - yc * x.m2) / countf;

    r.m4 = x.m4 + y.m4
         + delta4 * xc * yc * (xc * xc - xc * yc + yc * yc) / count3
         + 6.0f  * delta2 * (xc * xc * y.m2 + yc * yc * x.m2) / count2
         + 4.0f  * delta  * (xc * y.m3 - yc * x.m3) / countf;
    return r;
}

// ── Custom Kokkos reducer ─────────────────────────────────────────────────────

struct KurtosisReducer {
    using reducer          = KurtosisReducer;
    using value_type       = kurtosisResult;
    using result_view_type =
        Kokkos::View<value_type, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

    value_type* const value_ptr;

    KOKKOS_INLINE_FUNCTION
    explicit KurtosisReducer(value_type& val) : value_ptr(&val) {}

    KOKKOS_INLINE_FUNCTION
    void join(value_type& dst, const value_type& src) const {
        dst = kurtosis_combine(dst, src);
    }

    KOKKOS_INLINE_FUNCTION
    void init(value_type& val) const { val = value_type(); }

    KOKKOS_INLINE_FUNCTION
    value_type& reference() const { return *value_ptr; }

    KOKKOS_INLINE_FUNCTION
    result_view_type view() const { return result_view_type(value_ptr); }

    KOKKOS_INLINE_FUNCTION
    bool references_scalar() const { return true; }
};

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <elemCount> <repeat>\n";
        return 1;
    }

    const int elemCount = atoi(argv[1]);
    const int repeat    = atoi(argv[2]);

    // Generate random input on host
    storeElement* h_elem = new storeElement[elemCount];
    std::mt19937 gen(19937);
    std::uniform_real_distribution<float> dis(1.f, 2.f);
    for (int i = 0; i < elemCount; i++)
        h_elem[i] = storeElement(i, 0, (unsigned long long)i, dis(gen));

    Kokkos::initialize(argc, argv);
    {
        Kokkos::View<storeElement*> d_elem("elem", elemCount);
        {
            auto hv = Kokkos::create_mirror_view(d_elem);
            for (int i = 0; i < elemCount; i++) hv(i) = h_elem[i];
            Kokkos::deep_copy(d_elem, hv);
        }

        kurtosisResult result;

        auto t_start = std::chrono::steady_clock::now();

        for (int r = 0; r < repeat; r++) {
            result = kurtosisResult(); // reset
            Kokkos::parallel_reduce("kurtosis",
                Kokkos::RangePolicy<>(0, elemCount),
                KOKKOS_LAMBDA(const int i, kurtosisResult& lres) {
                    kurtosisResult elem_res(1, d_elem(i).value, 0.0f, 0.0f, 0.0f);
                    lres = kurtosis_combine(lres, elem_res);
                },
                KurtosisReducer(result));
            Kokkos::fence();
        }

        auto t_end = std::chrono::steady_clock::now();
        double elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9;

        std::cout << "Total device compute time: " << elapsed << " (s)\n";
        std::cout << "Results:\n";
        std::cout << sizeof(kurtosisResult) << " "
                  << result.count << " "
                  << result.m2    << " "
                  << result.m3    << " "
                  << result.m4    << "\n";
    }
    Kokkos::finalize();

    delete[] h_elem;
    return 0;
}
