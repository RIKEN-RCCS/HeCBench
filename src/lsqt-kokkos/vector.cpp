/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, and Ari Harju

    This file is part of GPUQT.

    GPUQT is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GPUQT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GPUQT.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "vector.h"
#include <string.h> // memcpy
#include <utility>  // std::swap
#define BLOCK_SIZE 256

Vector::Vector(int n)
  : n(n)
  , array_size(n * sizeof(real))
  , d_real_part("real", n)
  , d_imag_part("imag", n)
{
  h_real_part = Kokkos::create_mirror_view(d_real_part);
  h_imag_part = Kokkos::create_mirror_view(d_imag_part);
  real_part = h_real_part.data();
  imag_part = h_imag_part.data();
  Kokkos::deep_copy(d_real_part, static_cast<real>(0.0));
  Kokkos::deep_copy(d_imag_part, static_cast<real>(0.0));
}

Vector::Vector(Vector& orig)
  : n(orig.n)
  , array_size(orig.array_size)
  , d_real_part("real", orig.n)
  , d_imag_part("imag", orig.n)
{
  h_real_part = Kokkos::create_mirror_view(d_real_part);
  h_imag_part = Kokkos::create_mirror_view(d_imag_part);
  real_part = h_real_part.data();
  imag_part = h_imag_part.data();
  Kokkos::deep_copy(d_real_part, orig.d_real_part);
  Kokkos::deep_copy(d_imag_part, orig.d_imag_part);
}

Vector::~Vector()
{
  // Kokkos::View destructor handles device memory cleanup
}

void Vector::add(Vector& other)
{
  auto dr = d_real_part;
  auto di = d_imag_part;
  auto or_ = other.d_real_part;
  auto oi = other.d_imag_part;
  Kokkos::parallel_for(
    "add", n, KOKKOS_LAMBDA(int i) {
      dr(i) += or_(i);
      di(i) += oi(i);
    });
}

void Vector::copy(Vector& other)
{
  Kokkos::deep_copy(d_real_part, other.d_real_part);
  Kokkos::deep_copy(d_imag_part, other.d_imag_part);
}

void Vector::apply_sz(Vector& other)
{
  auto dr = d_real_part;
  auto di = d_imag_part;
  auto ir = other.d_real_part;
  auto ii = other.d_imag_part;
  Kokkos::parallel_for(
    "apply_sz", n, KOKKOS_LAMBDA(int i) {
      if (i % 2 == 0) {
        dr(i) = ir(i);
        di(i) = ii(i);
      } else {
        dr(i) = -ir(i);
        di(i) = -ii(i);
      }
    });
}

void Vector::copy_from_host(real* other_real, real* other_imag)
{
  memcpy(h_real_part.data(), other_real, array_size);
  memcpy(h_imag_part.data(), other_imag, array_size);
  Kokkos::deep_copy(d_real_part, h_real_part);
  Kokkos::deep_copy(d_imag_part, h_imag_part);
}

void Vector::copy_to_host(real* target_real, real* target_imag)
{
  Kokkos::deep_copy(h_real_part, d_real_part);
  Kokkos::deep_copy(h_imag_part, d_imag_part);
  memcpy(target_real, h_real_part.data(), array_size);
  memcpy(target_imag, h_imag_part.data(), array_size);
}

void Vector::swap(Vector& other)
{
  std::swap(d_real_part, other.d_real_part);
  std::swap(d_imag_part, other.d_imag_part);
  std::swap(h_real_part, other.h_real_part);
  std::swap(h_imag_part, other.h_imag_part);
  std::swap(real_part, other.real_part);
  std::swap(imag_part, other.imag_part);
}

// inner_product_1: for each block m, accumulate partial dot products.
// target.d_{real,imag}[m + offset] = sum over k in block: <final|rand>
void Vector::inner_product_1(int number_of_atoms, Vector& other, Vector& target, int offset)
{
  int grid_size = (number_of_atoms - 1) / BLOCK_SIZE + 1;

  // Zero the output slice
  auto tr = target.d_real_part;
  auto ti = target.d_imag_part;
  Kokkos::parallel_for(
    "zero_ip1", Kokkos::RangePolicy<>(offset, offset + grid_size), KOKKOS_LAMBDA(int i) {
      tr(i) = static_cast<real>(0.0);
      ti(i) = static_cast<real>(0.0);
    });

  auto final_real = d_real_part;
  auto final_imag = d_imag_part;
  auto rand_real = other.d_real_part;
  auto rand_imag = other.d_imag_part;

  Kokkos::parallel_for(
    "inner_prod_1", number_of_atoms, KOKKOS_LAMBDA(int idx) {
      int m = idx / BLOCK_SIZE;
      real a = final_real(idx);
      real b = final_imag(idx);
      real c = rand_real(idx);
      real d = rand_imag(idx);
      Kokkos::atomic_add(&tr(m + offset), a * c + b * d);
      Kokkos::atomic_add(&ti(m + offset), b * c - a * d);
    });
}

// inner_product_2: reduce the grid_size partial sums into one value per moment.
void Vector::inner_product_2(int number_of_atoms, int number_of_moments, Vector& target)
{
  int grid_size = (number_of_atoms - 1) / BLOCK_SIZE + 1;

  auto ip1_real = d_real_part;
  auto ip1_imag = d_imag_part;
  auto tgt_real = target.d_real_part;
  auto tgt_imag = target.d_imag_part;

  Kokkos::parallel_for(
    "inner_prod_2", number_of_moments, KOKKOS_LAMBDA(int m) {
      real sr = static_cast<real>(0.0);
      real si = static_cast<real>(0.0);
      for (int k = 0; k < grid_size; ++k) {
        sr += ip1_real(m * grid_size + k);
        si += ip1_imag(m * grid_size + k);
      }
      tgt_real(m) = sr;
      tgt_imag(m) = si;
    });
}
