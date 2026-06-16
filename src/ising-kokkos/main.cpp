#include <chrono>
#include <getopt.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <Kokkos_Core.hpp>

#define TCRIT 2.26918531421f

// Initialize lattice spins
void init_spins(Kokkos::View<signed char*> d_lattice,
                Kokkos::View<float*>       d_randvals,
                long long n)
{
    Kokkos::parallel_for("init_spins", n, KOKKOS_LAMBDA(long long tid) {
        float randval = d_randvals[tid];
        d_lattice[tid] = (randval < 0.5f) ? -1 : 1;
    });
    Kokkos::fence();
}

// Update black sublattice (is_black = true)
void update_lattice_black(
    Kokkos::View<signed char*> d_lattice,
    Kokkos::View<signed char*> d_op_lattice,
    Kokkos::View<float*>       d_randvals,
    float inv_temp,
    long long nx, long long ny)
{
    using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    Kokkos::parallel_for("update_black",
        Policy({0LL, 0LL}, {nx, ny}),
        KOKKOS_LAMBDA(long long i, long long j) {
            long long ipp = (i + 1 < nx)  ? i + 1 : 0;
            long long inn = (i - 1 >= 0)  ? i - 1 : nx - 1;
            long long jpp = (j + 1 < ny)  ? j + 1 : 0;
            long long jnn = (j - 1 >= 0)  ? j - 1 : ny - 1;

            // is_black: joff = (i % 2) ? jpp : jnn
            long long joff = (i % 2) ? jpp : jnn;

            signed char nn_sum =
                d_op_lattice[inn * ny + j] +
                d_op_lattice[i   * ny + j] +
                d_op_lattice[ipp * ny + j] +
                d_op_lattice[i   * ny + joff];

            signed char lij = d_lattice[i * ny + j];
            float acceptance_ratio = expf(-2.0f * inv_temp * (float)nn_sum * (float)lij);
            if (d_randvals[i * ny + j] < acceptance_ratio) {
                d_lattice[i * ny + j] = -lij;
            }
        });
    Kokkos::fence();
}

// Update white sublattice (is_black = false)
void update_lattice_white(
    Kokkos::View<signed char*> d_lattice,
    Kokkos::View<signed char*> d_op_lattice,
    Kokkos::View<float*>       d_randvals,
    float inv_temp,
    long long nx, long long ny)
{
    using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    Kokkos::parallel_for("update_white",
        Policy({0LL, 0LL}, {nx, ny}),
        KOKKOS_LAMBDA(long long i, long long j) {
            long long ipp = (i + 1 < nx)  ? i + 1 : 0;
            long long inn = (i - 1 >= 0)  ? i - 1 : nx - 1;
            long long jpp = (j + 1 < ny)  ? j + 1 : 0;
            long long jnn = (j - 1 >= 0)  ? j - 1 : ny - 1;

            // is_black=false: joff = (i % 2) ? jnn : jpp
            long long joff = (i % 2) ? jnn : jpp;

            signed char nn_sum =
                d_op_lattice[inn * ny + j] +
                d_op_lattice[i   * ny + j] +
                d_op_lattice[ipp * ny + j] +
                d_op_lattice[i   * ny + joff];

            signed char lij = d_lattice[i * ny + j];
            float acceptance_ratio = expf(-2.0f * inv_temp * (float)nn_sum * (float)lij);
            if (d_randvals[i * ny + j] < acceptance_ratio) {
                d_lattice[i * ny + j] = -lij;
            }
        });
    Kokkos::fence();
}

void update(Kokkos::View<signed char*> d_lattice_b,
            Kokkos::View<signed char*> d_lattice_w,
            Kokkos::View<float*>       d_randvals,
            float inv_temp, long long nx, long long ny)
{
    long long nyh = ny / 2;
    update_lattice_black(d_lattice_b, d_lattice_w, d_randvals, inv_temp, nx, nyh);
    update_lattice_white(d_lattice_w, d_lattice_b, d_randvals, inv_temp, nx, nyh);
}

static void usage(const char *pname) {
    const char *bname = rindex(pname, '/');
    if (!bname) bname = pname; else bname++;
    fprintf(stdout,
        "Usage: %s [options]\n"
        "options:\n"
        "\t-x|--lattice-n <LATTICE_N>\n"
        "\t-y|--lattice-m <LATTICE_M>\n"
        "\t-w|--nwarmup   <NWARMUP>\n"
        "\t-n|--niters    <NITERS>\n"
        "\t-a|--alpha     <ALPHA>\n"
        "\t-s|--seed      <SEED>\n",
        bname);
    exit(EXIT_SUCCESS);
}

int main(int argc, char **argv) {
    long long nx    = 5120;
    long long ny    = 5120;
    float     alpha = 0.1f;
    int       nwarmup = 100;
    int       niters  = 1000;
    unsigned long long seed = 1234ULL;

    while (1) {
        static struct option long_options[] = {
            {"lattice-n", required_argument, 0, 'x'},
            {"lattice-m", required_argument, 0, 'y'},
            {"alpha",     required_argument, 0, 'a'},
            {"seed",      required_argument, 0, 's'},
            {"nwarmup",   required_argument, 0, 'w'},
            {"niters",    required_argument, 0, 'n'},
            {"help",      no_argument,       0, 'h'},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int ch = getopt_long(argc, argv, "x:y:a:s:w:n:h", long_options, &option_index);
        if (ch == -1) break;
        switch (ch) {
            case 0: break;
            case 'x': nx       = atoll(optarg); break;
            case 'y': ny       = atoll(optarg); break;
            case 'a': alpha    = atof(optarg);  break;
            case 's': seed     = atoll(optarg); break;
            case 'w': nwarmup  = atoi(optarg);  break;
            case 'n': niters   = atoi(optarg);  break;
            case 'h': usage(argv[0]); break;
            case '?': exit(EXIT_FAILURE);
            default:  fprintf(stderr, "unknown option: %c\n", ch); exit(EXIT_FAILURE);
        }
    }

    if (nx % 2 != 0 || ny % 2 != 0) {
        fprintf(stderr, "ERROR: Lattice dimensions must be even values.\n");
        exit(EXIT_FAILURE);
    }

    float inv_temp = 1.0f / (alpha * TCRIT);

    srand((unsigned int)seed);
    long long nhalf = nx * ny / 2;
    float     *randvals  = (float*)      malloc(nhalf * sizeof(float));
    signed char *lattice_b = (signed char*) malloc(nhalf * sizeof(signed char));
    signed char *lattice_w = (signed char*) malloc(nhalf * sizeof(signed char));
    for (long long i = 0; i < nhalf; i++)
        randvals[i] = (float)rand() / (float)RAND_MAX;

    Kokkos::initialize(argc, argv);
    {
        Kokkos::View<float*>       d_randvals ("randvals",  nhalf);
        Kokkos::View<signed char*> d_lattice_b("lattice_b", nhalf);
        Kokkos::View<signed char*> d_lattice_w("lattice_w", nhalf);

        {
            auto h = Kokkos::create_mirror_view(d_randvals);
            for (long long i = 0; i < nhalf; i++) h(i) = randvals[i];
            Kokkos::deep_copy(d_randvals, h);
        }

        init_spins(d_lattice_b, d_randvals, nhalf);
        init_spins(d_lattice_w, d_randvals, nhalf);

        printf("Starting warmup...\n");
        for (int i = 0; i < nwarmup; i++)
            update(d_lattice_b, d_lattice_w, d_randvals, inv_temp, nx, ny);

        printf("Starting trial iterations...\n");
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < niters; i++)
            update(d_lattice_b, d_lattice_w, d_randvals, inv_temp, nx, ny);
        auto t1 = std::chrono::high_resolution_clock::now();
        double duration = (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        // Copy back for checksum
        {
            auto hb = Kokkos::create_mirror_view(d_lattice_b);
            auto hw = Kokkos::create_mirror_view(d_lattice_w);
            Kokkos::deep_copy(hb, d_lattice_b);
            Kokkos::deep_copy(hw, d_lattice_w);
            for (long long i = 0; i < nhalf; i++) {
                lattice_b[i] = hb(i);
                lattice_w[i] = hw(i);
            }
        }

        printf("REPORT:\n");
        printf("\tnGPUs: %d\n", 1);
        printf("\ttemperature: %f * %f\n", alpha, TCRIT);
        printf("\tseed: %llu\n", seed);
        printf("\twarmup iterations: %d\n", nwarmup);
        printf("\ttrial iterations: %d\n", niters);
        printf("\tlattice dimensions: %lld x %lld\n", nx, ny);
        printf("\telapsed time: %f sec\n", duration * 1e-6);
        printf("\tupdates per ns: %f\n", (double)(nx * ny) * niters / duration * 1e-3);
    }
    Kokkos::finalize();

    double naivesum = 0.0;
    for (long long i = 0; i < nhalf; i++) {
        naivesum += lattice_b[i];
        naivesum += lattice_w[i];
    }
    printf("checksum = %lf\n", naivesum);

    free(randvals);
    free(lattice_b);
    free(lattice_w);
    return 0;
}
