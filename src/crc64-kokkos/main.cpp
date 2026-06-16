// *****************************************************************************
//                   Copyright (C) 2014, UChicago Argonne, LLC
//                              All Rights Reserved
//        High-Performance CRC64 Library (ANL-SF-14-095)
//                    Hal Finkel, Argonne National Laboratory
// Kokkos port of the CRC64Test.cpp + CRC64.cpp benchmark.
// crc64_omp is re-implemented with Kokkos::parallel_for.
// The serial crc64 and related helpers are implemented here using
// a table built at runtime from the ECMA-182 reflected polynomial.
// *****************************************************************************

#define _XOPEN_SOURCE 600

#include <Kokkos_Core.hpp>
#include <ctime>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cstring>

using namespace std;

// ---- CRC-64 polynomial (ECMA-182, bit-reversed) ---------------------------
static const uint64_t CRC64_POLY = UINT64_C(0xc96c5795d7870f42);

// Table for byte-at-a-time CRC64.
// Populated by build_crc64_table() before Kokkos::initialize.
static uint64_t crc64_byte_table[256];

static void build_crc64_table() {
    for (int i = 0; i < 256; i++) {
        uint64_t crc = (uint64_t)i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1u)
                crc = (crc >> 1) ^ CRC64_POLY;
            else
                crc >>= 1;
        }
        crc64_byte_table[i] = crc;
    }
}

// ---- Serial CRC64 (host + device callable with a passed table) ------------
KOKKOS_INLINE_FUNCTION
uint64_t crc64_device(const unsigned char *data, size_t nbytes,
                      const uint64_t *table)
{
    uint64_t cs = UINT64_C(0xffffffffffffffff);
    for (size_t i = 0; i < nbytes; i++) {
        uint32_t idx = ((uint32_t)(cs ^ data[i])) & 0xff;
        cs = table[idx] ^ (cs >> 8);
    }
    return cs ^ UINT64_C(0xffffffffffffffff);
}

// Host-only serial wrapper
uint64_t crc64(const void *input, size_t nbytes) {
    return crc64_device((const unsigned char *)input, nbytes, crc64_byte_table);
}

uint64_t crc64_slow(const void *input, size_t nbytes) {
    return crc64(input, nbytes); // same algorithm in this implementation
}

// ---- crc64_invert ---------------------------------------------------------
void crc64_invert(uint64_t cs, void *check_bytes) {
    unsigned char *bytes = (unsigned char *)check_bytes;
    cs ^= UINT64_C(0xffffffffffffffff);
    bytes[7] = (unsigned char)((cs >> 56) & 0xff);
    bytes[6] = (unsigned char)((cs >> 48) & 0xff);
    bytes[5] = (unsigned char)((cs >> 40) & 0xff);
    bytes[4] = (unsigned char)((cs >> 32) & 0xff);
    bytes[3] = (unsigned char)((cs >> 24) & 0xff);
    bytes[2] = (unsigned char)((cs >> 16) & 0xff);
    bytes[1] = (unsigned char)((cs >>  8) & 0xff);
    bytes[0] = (unsigned char)( cs        & 0xff);
}

// ---- crc64_combine helpers ------------------------------------------------

// Precomputed x^(2^i) mod P for i=0..63  (same values as CRC64.cpp)
static const uint64_t crc64_x_pow_2n[64] = {
    UINT64_C(0x4000000000000000), UINT64_C(0x2000000000000000),
    UINT64_C(0x0800000000000000), UINT64_C(0x0080000000000000),
    UINT64_C(0x0000800000000000), UINT64_C(0x0000000080000000),
    UINT64_C(0xc96c5795d7870f42), UINT64_C(0x6d5f4ad7e3c3afa0),
    UINT64_C(0xd49f7e445077d8ea), UINT64_C(0x040fb02a53c216fa),
    UINT64_C(0x6bec35957b9ef3a0), UINT64_C(0xb0e3bb0658964afe),
    UINT64_C(0x218578c7a2dff638), UINT64_C(0x6dbb920f24dd5cf2),
    UINT64_C(0x7a140cfcdb4d5eb5), UINT64_C(0x41b3705ecbc4057b),
    UINT64_C(0xd46ab656accac1ea), UINT64_C(0x329beda6fc34fb73),
    UINT64_C(0x51a4fcd4350b9797), UINT64_C(0x314fa85637efae9d),
    UINT64_C(0xacf27e9a1518d512), UINT64_C(0xffe2a3388a4d8ce7),
    UINT64_C(0x48b9697e60cc2e4e), UINT64_C(0xada73cb78dd62460),
    UINT64_C(0x3ea5454d8ce5c1bb), UINT64_C(0x5e84e3a6c70feaf1),
    UINT64_C(0x90fd49b66cbd81d1), UINT64_C(0xe2943e0c1db254e8),
    UINT64_C(0xecfa6adeca8834a1), UINT64_C(0xf513e212593ee321),
    UINT64_C(0xf36ae57331040916), UINT64_C(0x63fbd333b87b6717),
    UINT64_C(0xbd60f8e152f50b8b), UINT64_C(0xa5ce4a8299c1567d),
    UINT64_C(0x0bd445f0cbdb55ee), UINT64_C(0xfdd6824e20134285),
    UINT64_C(0xcead8b6ebda2227a), UINT64_C(0xe44b17e4f5d4fb5c),
    UINT64_C(0x9b29c81ad01ca7c5), UINT64_C(0x1b4366e40fea4055),
    UINT64_C(0x27bca1551aae167b), UINT64_C(0xaa57bcd1b39a5690),
    UINT64_C(0xd7fce83fa1234db9), UINT64_C(0xcce4986efea3ff8e),
    UINT64_C(0x3602a4d9e65341f1), UINT64_C(0x722b1da2df516145),
    UINT64_C(0xecfc3ddd3a08da83), UINT64_C(0x0fb96dcca83507e6),
    UINT64_C(0x125f2fe78d70f080), UINT64_C(0x842f50b7651aa516),
    UINT64_C(0x09bc34188cd9836f), UINT64_C(0xf43666c84196d909),
    UINT64_C(0xb56feb30c0df6ccb), UINT64_C(0xaa66e04ce7f30958),
    UINT64_C(0xb7b1187e9af29547), UINT64_C(0x113255f8476495de),
    UINT64_C(0x8fb19f783095d77e), UINT64_C(0xaec4aacc7c82b133),
    UINT64_C(0xf64e6d09218428cf), UINT64_C(0x036a72ea5ac258a0),
    UINT64_C(0x5235ef12eb7aaa6a), UINT64_C(0x2fed7b1685657853),
    UINT64_C(0x8ef8951d46606fb5), UINT64_C(0x9d58c1090f034d14)
};

static inline uint64_t crc64_multiply_(uint64_t a, uint64_t b) {
    if ((a ^ (a - 1)) < (b ^ (b - 1))) { uint64_t t = a; a = b; b = t; }
    if (a == 0) return 0;
    uint64_t r = 0, h = UINT64_C(1) << 63;
    for (; a != 0; a <<= 1) {
        if (a & h) { r ^= b; a ^= h; }
        b = (b >> 1) ^ ((b & 1) ? CRC64_POLY : 0);
    }
    return r;
}

static inline uint64_t crc64_x_pow_n_(uint64_t n) {
    uint64_t r = UINT64_C(1) << 63;
    for (size_t i = 0; n != 0; n >>= 1, ++i)
        if (n & 1) r = crc64_multiply_(r, crc64_x_pow_2n[i]);
    return r;
}

uint64_t crc64_combine(uint64_t cs1, uint64_t cs2, size_t nbytes2) {
    return cs2 ^ crc64_multiply_(cs1, crc64_x_pow_n_(8 * nbytes2));
}

// ---- Kokkos parallel crc64_omp -------------------------------------------
static const size_t CRC64_MIN_THREAD_BYTES = 1024;

uint64_t crc64_omp(const void *input, size_t nbytes) {
    if (nbytes <= 2 * CRC64_MIN_THREAD_BYTES)
        return crc64(input, nbytes);

    int nthreads = 96 * 8 * 32;   // same as original
    if (nbytes < (size_t)nthreads * CRC64_MIN_THREAD_BYTES)
        nthreads = (int)(nbytes / CRC64_MIN_THREAD_BYTES);

    // Device Views for per-thread CRC and size
    Kokkos::View<uint64_t*> d_thread_cs("d_cs", nthreads);
    Kokkos::View<size_t*>   d_thread_sz("d_sz", nthreads);

    // Device View for the lookup table
    Kokkos::View<uint64_t*> d_table("d_table", 256);
    {
        auto hm = Kokkos::create_mirror_view(d_table);
        for (int i = 0; i < 256; i++) hm(i) = crc64_byte_table[i];
        Kokkos::deep_copy(d_table, hm);
    }

    // Device View for input data
    Kokkos::View<unsigned char*> d_data("d_data", nbytes);
    {
        auto hm = Kokkos::create_mirror_view(d_data);
        memcpy(hm.data(), input, nbytes);
        Kokkos::deep_copy(d_data, hm);
    }

    const size_t total = nbytes;
    const int    nt    = nthreads;

    Kokkos::parallel_for("crc64_parallel", nthreads,
        KOKKOS_LAMBDA(int tid) {
            size_t bpt   = total / nt;
            size_t start = bpt * (size_t)tid;
            size_t end   = (tid != nt - 1) ? start + bpt : total;
            size_t sz    = end - start;
            d_thread_sz(tid) = sz;
            d_thread_cs(tid) = crc64_device(&d_data(start), sz, d_table.data());
        });
    Kokkos::fence();

    // Combine on host
    auto hm_cs = Kokkos::create_mirror_view(d_thread_cs);
    auto hm_sz = Kokkos::create_mirror_view(d_thread_sz);
    Kokkos::deep_copy(hm_cs, d_thread_cs);
    Kokkos::deep_copy(hm_sz, d_thread_sz);

    uint64_t cs = hm_cs(0);
    for (int i = 1; i < nthreads; ++i)
        cs = crc64_combine(cs, hm_cs(i), hm_sz(i));
    return cs;
}

// ---- Test harness (from CRC64Test.cpp) ------------------------------------
int main(int argc, char *argv[]) {
    build_crc64_table();

    int ntests = 10;
    if (argc > 1) ntests = atoi(argv[1]);
    int seed = 5;
    if (argc > 2) seed = atoi(argv[2]);
    size_t max_test_length = 2097152;
    if (argc > 3) max_test_length = (size_t)atoi(argv[3]);

    cout << "Running " << ntests << " tests with seed " << seed << endl;
    srand48(seed);

    Kokkos::initialize(argc, argv);
    {
#define THE_CLOCK CLOCK_THREAD_CPUTIME_ID
        double tot_time = 0, tot_bytes = 0;
        int ntest = 0;
        while (++ntest <= ntests) {
            cout << ntest << " ";
            size_t test_length = (size_t)(max_test_length * (drand48() + 1));
            cout << test_length << " ";

            vector<unsigned char> buffer(test_length);
            for (size_t i = 0; i < test_length; ++i)
                buffer[i] = (unsigned char)(255 * drand48());

            timespec b_start, b_end;
            clock_gettime(THE_CLOCK, &b_start);
            uint64_t cs = crc64_omp(&buffer[0], test_length);
            clock_gettime(THE_CLOCK, &b_end);
            double b_time = (b_end.tv_sec - b_start.tv_sec)
                          + 1e-9 * (b_end.tv_nsec - b_start.tv_nsec);

            if (ntest > 1) { tot_time += b_time; tot_bytes += test_length; }

            // Append check bytes and verify
            size_t tlend = 8;
            buffer.resize(test_length + tlend, 0);
            crc64_invert(cs, &buffer[test_length]);

            string pass("pass"), fail("fail");
            uint64_t csc = crc64(&buffer[0], test_length + tlend);
            cout << ((csc == (uint64_t)-1) ? pass : fail) << " ";

            size_t div_pt = (size_t)(test_length * drand48());
            uint64_t cs1  = crc64(&buffer[0], div_pt);
            uint64_t cs2  = crc64(&buffer[div_pt], test_length - div_pt);
            csc = crc64_combine(cs1, cs2, test_length - div_pt);
            cout << ((csc == cs) ? pass : fail);
            cout << endl;
        }
        cout << (tot_bytes / (1024 * 1024)) / tot_time << " MB/s" << endl;
    }
    Kokkos::finalize();
    return 0;
}
