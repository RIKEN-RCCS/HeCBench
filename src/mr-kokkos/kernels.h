#pragma once
#include <stdint.h>
#include <Kokkos_Core.hpp>
#include "sprp32.h"
#include "sprp32_sf.h"

void mr32_sf(
  const uint32_t *h_bases,
  const uint32_t *h_n32,
  int *h_val,
  int iter)
{
  Kokkos::View<uint32_t*> d_bases("bases", BASES_CNT32);
  Kokkos::View<uint32_t*> d_n32("n32", iter);
  Kokkos::View<int*>      d_val("val", 1);

  {
    auto hv = Kokkos::create_mirror_view(d_bases);
    for (int i = 0; i < BASES_CNT32; ++i) hv(i) = h_bases[i];
    Kokkos::deep_copy(d_bases, hv);
  }
  {
    auto hv = Kokkos::create_mirror_view(d_n32);
    for (int i = 0; i < iter; ++i) hv(i) = h_n32[i];
    Kokkos::deep_copy(d_n32, hv);
  }
  Kokkos::deep_copy(d_val, 0);

  Kokkos::parallel_for("mr32_sf", iter, KOKKOS_LAMBDA(int j) {
    uint32_t n = d_n32(j);
    for (int cnt = 1; cnt <= BASES_CNT32; ++cnt) {
      Kokkos::atomic_add(&d_val(0), straightforward_mr32(d_bases.data(), cnt, n));
    }
  });
  Kokkos::fence();

  auto hv = Kokkos::create_mirror_view(d_val);
  Kokkos::deep_copy(hv, d_val);
  *h_val = hv(0);
}

void mr32_eff(
  const uint32_t *h_bases,
  const uint32_t *h_n32,
  int *h_val,
  int iter)
{
  Kokkos::View<uint32_t*> d_bases("bases", BASES_CNT32);
  Kokkos::View<uint32_t*> d_n32("n32", iter);
  Kokkos::View<int*>      d_val("val", 1);

  {
    auto hv = Kokkos::create_mirror_view(d_bases);
    for (int i = 0; i < BASES_CNT32; ++i) hv(i) = h_bases[i];
    Kokkos::deep_copy(d_bases, hv);
  }
  {
    auto hv = Kokkos::create_mirror_view(d_n32);
    for (int i = 0; i < iter; ++i) hv(i) = h_n32[i];
    Kokkos::deep_copy(d_n32, hv);
  }
  Kokkos::deep_copy(d_val, 0);

  Kokkos::parallel_for("mr32_eff", iter, KOKKOS_LAMBDA(int j) {
    uint32_t n = d_n32(j);
    for (int cnt = 1; cnt <= BASES_CNT32; ++cnt) {
      Kokkos::atomic_add(&d_val(0), efficient_mr32(d_bases.data(), cnt, n));
    }
  });
  Kokkos::fence();

  auto hv = Kokkos::create_mirror_view(d_val);
  Kokkos::deep_copy(hv, d_val);
  *h_val = hv(0);
}
