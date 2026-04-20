/*
 * GW perturbation theory (GPP) self-energy - Kokkos port
 *
 * Ported from gpp-omp.
 * CustomComplex.h and utils.h inlined from gpp-cuda/.
 */

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

// ============================================================
// CustomComplex  (from gpp-cuda/CustomComplex.h)
// ============================================================
#define KOKKOS_ESS KOKKOS_INLINE_FUNCTION

#define nstart 0
#define nend   3

template <class T>
class CustomComplex
{
public:
    T x, y;

    explicit CustomComplex() { x = 0; y = 0; }

    KOKKOS_ESS explicit CustomComplex(const T &a, const T &b) : x(a), y(b) {}

    KOKKOS_ESS CustomComplex(const CustomComplex &src) : x(src.x), y(src.y) {}

    KOKKOS_ESS CustomComplex &operator=(const CustomComplex &src) {
        x = src.x; y = src.y; return *this;
    }
    KOKKOS_ESS CustomComplex &operator+=(const CustomComplex &src) {
        x += src.x; y += src.y; return *this;
    }
    KOKKOS_ESS CustomComplex conj() const { return CustomComplex(x, -y); }
    KOKKOS_ESS T real() const { return x; }
    KOKKOS_ESS T imag() const { return y; }
    KOKKOS_ESS T get_real() const { return x; }
    KOKKOS_ESS T get_imag() const { return y; }
    void set_real(T v) { x = v; }
    void set_imag(T v) { y = v; }

    void print() const { printf("( %f, %f)\n", (double)x, (double)y); }

    friend std::ostream &operator<<(std::ostream &os, const CustomComplex<T> &obj) {
        os << "( " << obj.x << ", " << obj.y << ") ";
        return os;
    }

    KOKKOS_ESS friend CustomComplex<T> operator*(const CustomComplex<T> &a,
                                                  const CustomComplex<T> &b) {
        return CustomComplex<T>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }
    KOKKOS_ESS friend CustomComplex<T> operator*(const CustomComplex<T> &a, const T &b) {
        return CustomComplex<T>(a.x * b, a.y * b);
    }
    KOKKOS_ESS friend CustomComplex<T> operator*(const T &b, const CustomComplex<T> &a) {
        return CustomComplex<T>(a.x * b, a.y * b);
    }
    KOKKOS_ESS friend CustomComplex<T> operator-(const CustomComplex<T> &a,
                                                  const CustomComplex<T> &b) {
        return CustomComplex<T>(a.x - b.x, a.y - b.y);
    }
    KOKKOS_ESS friend CustomComplex<T> operator-(const T &a, const CustomComplex<T> &src) {
        return CustomComplex<T>(a - src.x, -src.y);
    }
    KOKKOS_ESS friend CustomComplex<T> operator+(const T &a, CustomComplex<T> &src) {
        return CustomComplex<T>(a + src.x, src.y);
    }
    KOKKOS_ESS friend CustomComplex<T> operator+(CustomComplex<T> &a, CustomComplex<T> &b) {
        return CustomComplex<T>(a.x + b.x, a.y + b.y);
    }

    KOKKOS_ESS friend CustomComplex<T> CustomComplex_conj(const CustomComplex<T> &src) {
        return CustomComplex<T>(src.x, -src.y);
    }
    KOKKOS_ESS friend T CustomComplex_real(const CustomComplex<T> &src) { return src.x; }
    KOKKOS_ESS friend T CustomComplex_imag(const CustomComplex<T> &src) { return src.y; }
    KOKKOS_ESS friend T CustomComplex_abs(const CustomComplex<T> &src) {
        return Kokkos::sqrt(src.x * src.x + src.y * src.y);
    }
};

// ============================================================
// utils.h (from gpp-cuda/utils.h)
// ============================================================
#define dataType double

#define aqsmtemp_size      (number_bands * ncouls)
#define aqsntemp_size      (number_bands * ncouls)
#define I_eps_array_size   (ngpown * ncouls)
#define achtemp_size       (nend - nstart)
#define achtemp_re_size    (nend - nstart)
#define achtemp_im_size    (nend - nstart)
#define vcoul_size         ncouls
#define inv_igp_index_size ngpown
#define indinv_size        (ncouls + 1)
#define wx_array_size      (nend - nstart)
#define wtilde_array_size  (ngpown * ncouls)

inline void *safe_malloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "Fatal: failed to allocate %zu bytes.\n", n); abort(); }
    return p;
}

inline void correctness(int problem_size, CustomComplex<dataType> result) {
    if (problem_size == 0) {
        dataType re_diff = result.get_real() - -24852.551547;
        dataType im_diff = result.get_imag() -  2957453.638101;
        printf(re_diff < 0.00001 && im_diff < 0.00001
               ? "\nBenchmark result: SUCCESS\n"
               : "\nBenchmark result: FAILURE\n");
    } else {
        dataType re_diff = result.get_real() - -0.096066;
        dataType im_diff = result.get_imag() -  11.431852;
        printf(re_diff < 0.00001 && im_diff < 0.00001
               ? "\nTest result: SUCCESS\n"
               : "\nTest result: FAILURE\n");
    }
}

// ============================================================
// Kokkos view types
// ============================================================
using CxView  = Kokkos::View<CustomComplex<dataType>*>;
using DblView = Kokkos::View<dataType*>;
using IntView = Kokkos::View<int*>;

// ============================================================
// Main
// ============================================================
int main(int argc, char **argv)
{
    int number_bands = 0, nvband = 0, ncouls = 0, nodes_per_group = 0;

    if (argc == 1) {
        number_bands = 512; nvband = 2; ncouls = 512; nodes_per_group = 20;
    } else if (argc == 2) {
        if (strcmp(argv[1], "benchmark") == 0) {
            number_bands = 512; nvband = 2; ncouls = 32768; nodes_per_group = 20;
        } else if (strcmp(argv[1], "test") == 0) {
            number_bands = 512; nvband = 2; ncouls = 512;   nodes_per_group = 20;
        } else {
            std::cout << "Usage: ./main <test|benchmark>\n"; exit(0);
        }
    } else if (argc == 5) {
        number_bands  = atoi(argv[1]);
        nvband        = atoi(argv[2]);
        ncouls        = atoi(argv[3]);
        nodes_per_group = atoi(argv[4]);
    } else {
        std::cout << "Usage: ./main <number_bands> <nvband> <ncouls> <nodes_per_group>\n";
        exit(0);
    }

    const int ngpown = ncouls / nodes_per_group;

    const dataType e_lk   = 10;
    const dataType dw     = 1;
    const dataType to1    = 1e-6;
    const dataType e_n1kq = 6.0;

    timeval startTimer, endTimer;
    gettimeofday(&startTimer, NULL);

    std::cout << "Sizeof(CustomComplex<dataType>) = "
              << sizeof(CustomComplex<dataType>) << " bytes\n";
    std::cout << "number_bands=" << number_bands
              << " nvband=" << nvband
              << " ncouls=" << ncouls
              << " nodes_per_group=" << nodes_per_group
              << " ngpown=" << ngpown
              << " nend=" << nend
              << " nstart=" << nstart << "\n";

    // Host allocations (plain malloc, macro-sized)
    auto *aqsmtemp   = (CustomComplex<dataType> *)safe_malloc(aqsmtemp_size   * sizeof(CustomComplex<dataType>));
    auto *aqsntemp   = (CustomComplex<dataType> *)safe_malloc(aqsntemp_size   * sizeof(CustomComplex<dataType>));
    auto *I_eps_array= (CustomComplex<dataType> *)safe_malloc(I_eps_array_size* sizeof(CustomComplex<dataType>));
    auto *wtilde_array=(CustomComplex<dataType> *)safe_malloc(I_eps_array_size* sizeof(CustomComplex<dataType>));
    auto *vcoul      = (dataType *)safe_malloc(vcoul_size         * sizeof(dataType));
    auto *inv_igp_index=(int *)   safe_malloc(inv_igp_index_size * sizeof(int));
    auto *indinv     = (int *)    safe_malloc(indinv_size         * sizeof(int));
    auto *achtemp    = (CustomComplex<dataType> *)safe_malloc(achtemp_size    * sizeof(CustomComplex<dataType>));
    auto *achtemp_re = (dataType *)safe_malloc(achtemp_re_size    * sizeof(dataType));
    auto *achtemp_im = (dataType *)safe_malloc(achtemp_im_size    * sizeof(dataType));
    auto *wx_array   = (dataType *)safe_malloc(wx_array_size      * sizeof(dataType));

    // Initialise
    CustomComplex<dataType> expr(0.025, 0.025);
    for (int n1 = 0; n1 < number_bands; n1++)
        for (int ig = 0; ig < ncouls; ig++) {
            aqsmtemp[n1 * ncouls + ig] = expr;
            aqsntemp[n1 * ncouls + ig] = expr;
        }
    for (int my_igp = 0; my_igp < ngpown; my_igp++)
        for (int ig = 0; ig < ncouls; ig++) {
            I_eps_array [my_igp * ncouls + ig] = expr;
            wtilde_array[my_igp * ncouls + ig] = expr;
        }
    for (int i = 0; i < ncouls; i++) vcoul[i] = i * 0.025;
    for (int ig = 0; ig < ngpown; ig++)
        inv_igp_index[ig] = (ig + 1) * ncouls / ngpown;
    for (int ig = 0; ig <= ncouls; ig++) indinv[ig] = (ig < ncouls) ? ig : ncouls - 1;

    for (int iw = nstart; iw < nend; iw++) {
        achtemp_re[iw] = 0.0;
        achtemp_im[iw] = 0.0;
        wx_array[iw]   = e_lk - e_n1kq + dw * ((iw + 1) - 2);
        if (wx_array[iw] < to1) wx_array[iw] = to1;
    }

    Kokkos::initialize(argc, argv);
    {
        // Device views
        CxView  d_aqsmtemp   ("aqsmtemp",    aqsmtemp_size);
        CxView  d_aqsntemp   ("aqsntemp",    aqsntemp_size);
        CxView  d_I_eps_array("I_eps_array", I_eps_array_size);
        CxView  d_wtilde_array("wtilde_array",I_eps_array_size);
        DblView d_vcoul      ("vcoul",        vcoul_size);
        IntView d_inv_igp_index("inv_igp_index", inv_igp_index_size);
        IntView d_indinv     ("indinv",       indinv_size);
        DblView d_wx_array   ("wx_array",     wx_array_size);

        // Mirror views for upload
        {
            auto h = Kokkos::create_mirror_view(d_aqsmtemp);
            for (int i = 0; i < aqsmtemp_size; i++) h(i) = aqsmtemp[i];
            Kokkos::deep_copy(d_aqsmtemp, h);
        }
        {
            auto h = Kokkos::create_mirror_view(d_aqsntemp);
            for (int i = 0; i < aqsntemp_size; i++) h(i) = aqsntemp[i];
            Kokkos::deep_copy(d_aqsntemp, h);
        }
        {
            auto h = Kokkos::create_mirror_view(d_I_eps_array);
            for (int i = 0; i < I_eps_array_size; i++) h(i) = I_eps_array[i];
            Kokkos::deep_copy(d_I_eps_array, h);
        }
        {
            auto h = Kokkos::create_mirror_view(d_wtilde_array);
            for (int i = 0; i < I_eps_array_size; i++) h(i) = wtilde_array[i];
            Kokkos::deep_copy(d_wtilde_array, h);
        }
        {
            auto h = Kokkos::create_mirror_view(d_vcoul);
            for (int i = 0; i < vcoul_size; i++) h(i) = vcoul[i];
            Kokkos::deep_copy(d_vcoul, h);
        }
        {
            auto h = Kokkos::create_mirror_view(d_inv_igp_index);
            for (int i = 0; i < inv_igp_index_size; i++) h(i) = inv_igp_index[i];
            Kokkos::deep_copy(d_inv_igp_index, h);
        }
        {
            auto h = Kokkos::create_mirror_view(d_indinv);
            for (int i = 0; i < indinv_size; i++) h(i) = indinv[i];
            Kokkos::deep_copy(d_indinv, h);
        }
        {
            auto h = Kokkos::create_mirror_view(d_wx_array);
            for (int i = 0; i < wx_array_size; i++) h(i) = wx_array[i];
            Kokkos::deep_copy(d_wx_array, h);
        }

        // Reduction accumulator (6 values: re0..2, im0..2)
        DblView d_ach("ach_sums", 6);

        float total_time = 0.f;

        for (int iter = 0; iter < 10; iter++) {
            Kokkos::deep_copy(d_ach, 0.0);

            auto t0 = std::chrono::steady_clock::now();

            const int nb   = number_bands;
            const int nc   = ncouls;
            const int ngp  = ngpown;

            Kokkos::parallel_for(
                "gpp_main",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ngp, nb}),
                KOKKOS_LAMBDA(int my_igp, int n1) {
                    const int indigp = d_inv_igp_index(my_igp);
                    const int igp    = d_indinv(indigp);

                    dataType re_loc[nend - nstart];
                    dataType im_loc[nend - nstart];
                    for (int iw = 0; iw < nend - nstart; iw++) {
                        re_loc[iw] = 0.0;
                        im_loc[iw] = 0.0;
                    }

                    CustomComplex<dataType> sch_store1 =
                        d_aqsmtemp(n1 * nc + igp).conj() *
                        d_aqsntemp(n1 * nc + igp) *
                        static_cast<dataType>(0.5) *
                        d_vcoul(igp);

                    for (int ig = 0; ig < nc; ig++) {
                        for (int iw = nstart; iw < nend; iw++) {
                            CustomComplex<dataType> wdiff =
                                d_wx_array(iw) - d_wtilde_array(my_igp * nc + ig);
                            CustomComplex<dataType> wdiff_conj = wdiff.conj();
                            CustomComplex<dataType> ww   = wdiff * wdiff_conj;
                            dataType denom = CustomComplex_real(ww);
                            CustomComplex<dataType> delw =
                                d_wtilde_array(my_igp * nc + ig) *
                                wdiff_conj *
                                (static_cast<dataType>(1.0) / denom);
                            CustomComplex<dataType> sch_array =
                                delw * d_I_eps_array(my_igp * nc + ig) * sch_store1;

                            re_loc[iw - nstart] += sch_array.real();
                            im_loc[iw - nstart] += sch_array.imag();
                        }
                    }

                    for (int iw = 0; iw < nend - nstart; iw++) {
                        Kokkos::atomic_add(&d_ach(iw),           re_loc[iw]);
                        Kokkos::atomic_add(&d_ach(iw + (nend - nstart)), im_loc[iw]);
                    }
                });
            Kokkos::fence();

            auto t1 = std::chrono::steady_clock::now();
            total_time += (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        }

        printf("Average kernel execution time %f (s)\n", (total_time * 1e-9f) / 10.f);

        // Copy results back
        auto h_ach = Kokkos::create_mirror_view(d_ach);
        Kokkos::deep_copy(h_ach, d_ach);

        achtemp_re[0] = h_ach(0);
        achtemp_re[1] = h_ach(1);
        achtemp_re[2] = h_ach(2);
        achtemp_im[0] = h_ach(3);
        achtemp_im[1] = h_ach(4);
        achtemp_im[2] = h_ach(5);

        for (int iw = nstart; iw < nend; iw++)
            achtemp[iw] = CustomComplex<dataType>(achtemp_re[iw], achtemp_im[iw]);

        if (argc == 2) {
            if (strcmp(argv[1], "benchmark") == 0) correctness(0, achtemp[0]);
            else if (strcmp(argv[1], "test")  == 0) correctness(1, achtemp[0]);
        } else {
            correctness(1, achtemp[0]);
        }

        printf("\n Final achtemp\n");
        achtemp[0].print();
    }
    Kokkos::finalize();

    gettimeofday(&endTimer, NULL);
    double elapsedTimer = (endTimer.tv_sec  - startTimer.tv_sec) +
                          1e-6 * (endTimer.tv_usec - startTimer.tv_usec);
    std::cout << "********** Total Time Taken **********= " << elapsedTimer << " secs\n";

    free(aqsmtemp); free(aqsntemp); free(I_eps_array); free(wtilde_array);
    free(vcoul);    free(inv_igp_index); free(indinv);
    free(achtemp);  free(achtemp_re);   free(achtemp_im); free(wx_array);
    return 0;
}
