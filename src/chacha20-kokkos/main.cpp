#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "chacha20.h"

int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    if (argc != 2) {
      printf("Usage: %s <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }
    const int repeat = atoi(argv[1]);

    // Initialize lookup table mapping hex chars to nibble values
    uint8_t char_to_uint[256] = {};
    for (int i = 0; i < 10; i++) char_to_uint[i + '0'] = i;
    for (int i = 0; i < 26; i++) char_to_uint[i + 'a'] = i + 10;
    for (int i = 0; i < 26; i++) char_to_uint[i + 'A'] = i + 10;

    const char *h_key =
      "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
    const char *h_nonce = "0001020304050607";
    const char *h_keystream =
      "f798a189f195e66982105ffb640bb7757f579da31602fc93ec01ac56f85ac3c1"
      "34a4547b733b46413042c9440049176905d3be59ea1c53f15916155c2be8241a"
      "38008b9a26bc35941e2444177c8ade6689de95264986d95889fb60e84629c9bd"
      "9a5acb1cc118be563eb9b3a4a472f82e09a7e778492b562ef7130e88dfe031c7"
      "9db9d4f7c7a899151b9a475032b63fc385245fe054e3dd5a97a5f576fe064025"
      "d3ce042c566ab2c507b138db853e3d6959660996546cc9c4a6eafdc777c040d7"
      "0eaf46f76dad3979e5c5360c3317166a1c894c94a371876a94df7628fe4eaaf2"
      "ccb27d5aaae0ad7ad0f9d4b6ad3b54098746d4524d38407a6deb3ab78fab78c9";

    const int key_len       = strlen(h_key);
    const int nonce_len     = strlen(h_nonce);
    const int keystream_len = strlen(h_keystream);
    const int result_len    = keystream_len / 2;

    uint8_t *h_result       = (uint8_t*) malloc(result_len);
    uint8_t *h_raw_keystream = (uint8_t*) malloc(result_len);

    // Device Views
    Kokkos::View<uint8_t*> d_char_to_uint("char_to_uint", 256);
    Kokkos::View<char*>    d_key("d_key", key_len);
    Kokkos::View<uint8_t*> d_raw_key("d_raw_key", key_len / 2);
    Kokkos::View<char*>    d_nonce("d_nonce", nonce_len);
    Kokkos::View<uint8_t*> d_raw_nonce("d_raw_nonce", nonce_len / 2);
    Kokkos::View<char*>    d_keystream("d_keystream", keystream_len);
    Kokkos::View<uint8_t*> d_raw_keystream("d_raw_keystream", result_len);
    Kokkos::View<uint8_t*> d_result("d_result", result_len);

    // Initialize Views from host data
    {
      auto hv = Kokkos::create_mirror_view(d_char_to_uint);
      for (int i = 0; i < 256; i++) hv(i) = char_to_uint[i];
      Kokkos::deep_copy(d_char_to_uint, hv);
    }
    {
      auto hv = Kokkos::create_mirror_view(d_key);
      for (int i = 0; i < key_len; i++) hv(i) = h_key[i];
      Kokkos::deep_copy(d_key, hv);
    }
    {
      auto hv = Kokkos::create_mirror_view(d_nonce);
      for (int i = 0; i < nonce_len; i++) hv(i) = h_nonce[i];
      Kokkos::deep_copy(d_nonce, hv);
    }
    {
      auto hv = Kokkos::create_mirror_view(d_keystream);
      for (int i = 0; i < keystream_len; i++) hv(i) = h_keystream[i];
      Kokkos::deep_copy(d_keystream, hv);
    }

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < repeat; iter++) {
      // Zero out result buffer
      Kokkos::deep_copy(d_result, uint8_t(0));

      // hex_to_raw: decode hex string to raw bytes in parallel
      Kokkos::parallel_for("hex_to_raw_key", key_len / 2,
        KOKKOS_LAMBDA(int i) {
          uint8_t hi = d_char_to_uint((uint8_t)d_key(i*2 + 0));
          uint8_t lo = d_char_to_uint((uint8_t)d_key(i*2 + 1));
          d_raw_key(i) = (hi << 4) | lo;
        });

      Kokkos::parallel_for("hex_to_raw_nonce", nonce_len / 2,
        KOKKOS_LAMBDA(int i) {
          uint8_t hi = d_char_to_uint((uint8_t)d_nonce(i*2 + 0));
          uint8_t lo = d_char_to_uint((uint8_t)d_nonce(i*2 + 1));
          d_raw_nonce(i) = (hi << 4) | lo;
        });

      Kokkos::parallel_for("hex_to_raw_ks", keystream_len / 2,
        KOKKOS_LAMBDA(int i) {
          uint8_t hi = d_char_to_uint((uint8_t)d_keystream(i*2 + 0));
          uint8_t lo = d_char_to_uint((uint8_t)d_keystream(i*2 + 1));
          d_raw_keystream(i) = (hi << 4) | lo;
        });

      Kokkos::fence();

      // ChaCha20 encryption runs single-threaded
      Kokkos::parallel_for("chacha_crypt", 1,
        KOKKOS_LAMBDA(int) {
          Chacha20 chacha(d_raw_key.data(), d_raw_nonce.data());
          chacha.crypt(d_result.data(), (size_t)result_len);
        });
    }

    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels: %f (us)\n", (time * 1e-3f) / repeat);

    // Copy results back to host
    {
      auto hv = Kokkos::create_mirror_view(d_result);
      Kokkos::deep_copy(hv, d_result);
      for (int i = 0; i < result_len; i++) h_result[i] = hv(i);
    }
    {
      auto hv = Kokkos::create_mirror_view(d_raw_keystream);
      Kokkos::deep_copy(hv, d_raw_keystream);
      for (int i = 0; i < result_len; i++) h_raw_keystream[i] = hv(i);
    }

    int error = memcmp(h_result, h_raw_keystream, result_len);
    printf("%s\n", error ? "FAIL" : "PASS");

    free(h_result);
    free(h_raw_keystream);
  }
  Kokkos::finalize();
  return 0;
}
