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

#include "hamiltonian.h"
#include "model.h"
#include "vector.h"
#define BLOCK_SIZE 256

Hamiltonian::Hamiltonian(Model& model)
{
  n = model.number_of_atoms;
  max_neighbor = model.max_neighbor;
  energy_max = model.energy_max;
  int number_of_pairs = model.number_of_pairs;

  d_neighbor_number = Kokkos::View<int*>("neighbor_number", n);
  d_potential = Kokkos::View<real*>("potential", n);
  d_neighbor_list = Kokkos::View<int*>("neighbor_list", number_of_pairs);
  d_hopping_real = Kokkos::View<real*>("hopping_real", number_of_pairs);
  d_hopping_imag = Kokkos::View<real*>("hopping_imag", number_of_pairs);
  d_xx = Kokkos::View<real*>("xx", number_of_pairs);

  auto h_nn = Kokkos::create_mirror_view(d_neighbor_number);
  auto h_pot = Kokkos::create_mirror_view(d_potential);
  auto h_nl = Kokkos::create_mirror_view(d_neighbor_list);
  auto h_hr = Kokkos::create_mirror_view(d_hopping_real);
  auto h_hi = Kokkos::create_mirror_view(d_hopping_imag);
  auto h_xx = Kokkos::create_mirror_view(d_xx);

  for (int i = 0; i < n; ++i) {
    h_nn(i) = model.neighbor_number[i];
    h_pot(i) = model.potential[i];
  }

  // Transpose from row-major [atom][neighbor] to column-major [neighbor][atom]
  // so that GPU threads (one per atom) have coalesced access.
  for (int m = 0; m < max_neighbor; ++m) {
    for (int i = 0; i < n; ++i) {
      int src = i * max_neighbor + m;
      int dst = m * n + i;
      h_nl(dst) = model.neighbor_list[src];
      h_hr(dst) = model.hopping_real[src];
      h_hi(dst) = model.hopping_imag[src];
      h_xx(dst) = model.xx[src];
    }
  }

  Kokkos::deep_copy(d_neighbor_number, h_nn);
  Kokkos::deep_copy(d_potential, h_pot);
  Kokkos::deep_copy(d_neighbor_list, h_nl);
  Kokkos::deep_copy(d_hopping_real, h_hr);
  Kokkos::deep_copy(d_hopping_imag, h_hi);
  Kokkos::deep_copy(d_xx, h_xx);
}

// |output> = H |input>
void Hamiltonian::apply(Vector& input, Vector& output)
{
  auto nn = d_neighbor_number;
  auto nl = d_neighbor_list;
  auto pot = d_potential;
  auto hr = d_hopping_real;
  auto hi = d_hopping_imag;
  auto in_r = input.d_real_part;
  auto in_i = input.d_imag_part;
  auto out_r = output.d_real_part;
  auto out_i = output.d_imag_part;
  real emax = energy_max;
  int num_atoms = n;

  Kokkos::parallel_for(
    "apply_H", num_atoms, KOKKOS_LAMBDA(int idx) {
      real tr = pot(idx) * in_r(idx);
      real ti = pot(idx) * in_i(idx);
      for (int m = 0; m < nn(idx); ++m) {
        int index_1 = m * num_atoms + idx;
        int index_2 = nl(index_1);
        real a = hr(index_1);
        real b = hi(index_1);
        real c = in_r(index_2);
        real d = in_i(index_2);
        tr += a * c - b * d;
        ti += a * d + b * c;
      }
      out_r(idx) = tr / emax;
      out_i(idx) = ti / emax;
    });
}

// |output> = [X, H] |input>
void Hamiltonian::apply_commutator(Vector& input, Vector& output)
{
  auto nn = d_neighbor_number;
  auto nl = d_neighbor_list;
  auto hr = d_hopping_real;
  auto hi = d_hopping_imag;
  auto xx = d_xx;
  auto in_r = input.d_real_part;
  auto in_i = input.d_imag_part;
  auto out_r = output.d_real_part;
  auto out_i = output.d_imag_part;
  real emax = energy_max;
  int num_atoms = n;

  Kokkos::parallel_for(
    "apply_commutator", num_atoms, KOKKOS_LAMBDA(int idx) {
      real tr = static_cast<real>(0.0);
      real ti = static_cast<real>(0.0);
      for (int m = 0; m < nn(idx); ++m) {
        int index_1 = m * num_atoms + idx;
        int index_2 = nl(index_1);
        real a = hr(index_1);
        real b = hi(index_1);
        real c = in_r(index_2);
        real d = in_i(index_2);
        real x = xx(index_1);
        tr -= (a * c - b * d) * x;
        ti -= (a * d + b * c) * x;
      }
      out_r(idx) = tr / emax;
      out_i(idx) = ti / emax;
    });
}

// |output> = V |input>  (velocity operator = i[H, X])
void Hamiltonian::apply_current(Vector& input, Vector& output)
{
  auto nn = d_neighbor_number;
  auto nl = d_neighbor_list;
  auto hr = d_hopping_real;
  auto hi = d_hopping_imag;
  auto xx = d_xx;
  auto in_r = input.d_real_part;
  auto in_i = input.d_imag_part;
  auto out_r = output.d_real_part;
  auto out_i = output.d_imag_part;
  int num_atoms = n;

  Kokkos::parallel_for(
    "apply_current", num_atoms, KOKKOS_LAMBDA(int idx) {
      real tr = static_cast<real>(0.0);
      real ti = static_cast<real>(0.0);
      for (int m = 0; m < nn(idx); ++m) {
        int index_1 = m * num_atoms + idx;
        int index_2 = nl(index_1);
        real a = hr(index_1);
        real b = hi(index_1);
        real c = in_r(index_2);
        real d = in_i(index_2);
        tr += (a * c - b * d) * xx(index_1);
        ti += (a * d + b * c) * xx(index_1);
      }
      out_r(idx) = +ti;
      out_i(idx) = -tr;
    });
}

// Chebyshev iteration: state_2 = 2*H*state_1 - state_0
void Hamiltonian::kernel_polynomial(Vector& state_0, Vector& state_1, Vector& state_2)
{
  auto nn = d_neighbor_number;
  auto nl = d_neighbor_list;
  auto pot = d_potential;
  auto hr = d_hopping_real;
  auto hi = d_hopping_imag;
  auto s0r = state_0.d_real_part;
  auto s0i = state_0.d_imag_part;
  auto s1r = state_1.d_real_part;
  auto s1i = state_1.d_imag_part;
  auto s2r = state_2.d_real_part;
  auto s2i = state_2.d_imag_part;
  real emax = energy_max;
  int num_atoms = n;

  Kokkos::parallel_for(
    "kernel_poly", num_atoms, KOKKOS_LAMBDA(int idx) {
      real tr = pot(idx) * s1r(idx);
      real ti = pot(idx) * s1i(idx);
      for (int m = 0; m < nn(idx); ++m) {
        int index_1 = m * num_atoms + idx;
        int index_2 = nl(index_1);
        real a = hr(index_1);
        real b = hi(index_1);
        real c = s1r(index_2);
        real d = s1i(index_2);
        tr += a * c - b * d;
        ti += a * d + b * c;
      }
      tr /= emax;
      ti /= emax;
      s2r(idx) = static_cast<real>(2.0) * tr - s0r(idx);
      s2i(idx) = static_cast<real>(2.0) * ti - s0i(idx);
    });
}

// First two terms of the time evolution: state = b0*state_0 +/- i*b1*state_1
void Hamiltonian::chebyshev_01(
  Vector& state_0, Vector& state_1, Vector& state, real bessel_0, real bessel_1, int direction)
{
  auto s0r = state_0.d_real_part;
  auto s0i = state_0.d_imag_part;
  auto s1r = state_1.d_real_part;
  auto s1i = state_1.d_imag_part;
  auto sr = state.d_real_part;
  auto si = state.d_imag_part;
  real b0 = bessel_0;
  real b1 = bessel_1 * direction;
  int num_atoms = n;

  Kokkos::parallel_for(
    "chebyshev_01", num_atoms, KOKKOS_LAMBDA(int idx) {
      sr(idx) = b0 * s0r(idx) + b1 * s1i(idx);
      si(idx) = b0 * s0i(idx) - b1 * s1r(idx);
    });
}

// Further terms of time evolution: update state, compute state_2 = 2*H*state_1 - state_0
void Hamiltonian::chebyshev_2(
  Vector& state_0, Vector& state_1, Vector& state_2, Vector& state, real bessel_m, int label)
{
  auto nn = d_neighbor_number;
  auto nl = d_neighbor_list;
  auto pot = d_potential;
  auto hr = d_hopping_real;
  auto hi = d_hopping_imag;
  auto s0r = state_0.d_real_part;
  auto s0i = state_0.d_imag_part;
  auto s1r = state_1.d_real_part;
  auto s1i = state_1.d_imag_part;
  auto s2r = state_2.d_real_part;
  auto s2i = state_2.d_imag_part;
  auto sr = state.d_real_part;
  auto si = state.d_imag_part;
  real emax = energy_max;
  real bm = bessel_m;
  int num_atoms = n;

  Kokkos::parallel_for(
    "chebyshev_2", num_atoms, KOKKOS_LAMBDA(int idx) {
      real tr = pot(idx) * s1r(idx);
      real ti = pot(idx) * s1i(idx);
      for (int m = 0; m < nn(idx); ++m) {
        int index_1 = m * num_atoms + idx;
        int index_2 = nl(index_1);
        real a = hr(index_1);
        real b = hi(index_1);
        real c = s1r(index_2);
        real d = s1i(index_2);
        tr += a * c - b * d;
        ti += a * d + b * c;
      }
      tr /= emax;
      ti /= emax;
      tr = static_cast<real>(2.0) * tr - s0r(idx);
      ti = static_cast<real>(2.0) * ti - s0i(idx);
      switch (label) {
        case 1:
          sr(idx) += bm * tr;
          si(idx) += bm * ti;
          break;
        case 2:
          sr(idx) -= bm * tr;
          si(idx) -= bm * ti;
          break;
        case 3:
          sr(idx) += bm * ti;
          si(idx) -= bm * tr;
          break;
        case 4:
          sr(idx) -= bm * ti;
          si(idx) += bm * tr;
          break;
      }
      s2r(idx) = tr;
      s2i(idx) = ti;
    });
}

// First commutator term: state = i*b1*state_1x
void Hamiltonian::chebyshev_1x(Vector& input, Vector& output, real bessel_1)
{
  auto ir = input.d_real_part;
  auto ii = input.d_imag_part;
  auto or_ = output.d_real_part;
  auto oi = output.d_imag_part;
  real b1 = bessel_1;
  int num_atoms = n;

  Kokkos::parallel_for(
    "chebyshev_1x", num_atoms, KOKKOS_LAMBDA(int idx) {
      or_(idx) = +b1 * ii(idx);
      oi(idx) = -b1 * ir(idx);
    });
}

// Further commutator terms: update state, compute state_2 and state_2x
void Hamiltonian::chebyshev_2x(
  Vector& state_0,
  Vector& state_0x,
  Vector& state_1,
  Vector& state_1x,
  Vector& state_2,
  Vector& state_2x,
  Vector& state,
  real bessel_m,
  int label)
{
  auto nn = d_neighbor_number;
  auto nl = d_neighbor_list;
  auto pot = d_potential;
  auto hr = d_hopping_real;
  auto hi = d_hopping_imag;
  auto xx = d_xx;
  auto s0r = state_0.d_real_part;
  auto s0i = state_0.d_imag_part;
  auto s0xr = state_0x.d_real_part;
  auto s0xi = state_0x.d_imag_part;
  auto s1r = state_1.d_real_part;
  auto s1i = state_1.d_imag_part;
  auto s1xr = state_1x.d_real_part;
  auto s1xi = state_1x.d_imag_part;
  auto s2r = state_2.d_real_part;
  auto s2i = state_2.d_imag_part;
  auto s2xr = state_2x.d_real_part;
  auto s2xi = state_2x.d_imag_part;
  auto sr = state.d_real_part;
  auto si = state.d_imag_part;
  real emax = energy_max;
  real bm = bessel_m;
  int num_atoms = n;

  Kokkos::parallel_for(
    "chebyshev_2x", num_atoms, KOKKOS_LAMBDA(int idx) {
      real tr = pot(idx) * s1r(idx);
      real ti = pot(idx) * s1i(idx);
      real tx_r = pot(idx) * s1xr(idx);
      real tx_i = pot(idx) * s1xi(idx);

      for (int m = 0; m < nn(idx); ++m) {
        int index_1 = m * num_atoms + idx;
        int index_2 = nl(index_1);
        real a = hr(index_1);
        real b = hi(index_1);
        real c = s1r(index_2);
        real d = s1i(index_2);
        tr += a * c - b * d;
        ti += a * d + b * c;

        real cx = s1xr(index_2);
        real dx = s1xi(index_2);
        tx_r += a * cx - b * dx;
        tx_i += a * dx + b * cx;

        real x = xx(index_1);
        tx_r -= (a * c - b * d) * x;
        tx_i -= (a * d + b * c) * x;
      }

      tr /= emax;
      ti /= emax;
      tr = static_cast<real>(2.0) * tr - s0r(idx);
      ti = static_cast<real>(2.0) * ti - s0i(idx);
      s2r(idx) = tr;
      s2i(idx) = ti;

      tx_r /= emax;
      tx_i /= emax;
      tx_r = static_cast<real>(2.0) * tx_r - s0xr(idx);
      tx_i = static_cast<real>(2.0) * tx_i - s0xi(idx);
      s2xr(idx) = tx_r;
      s2xi(idx) = tx_i;

      switch (label) {
        case 1:
          sr(idx) += bm * tx_r;
          si(idx) += bm * tx_i;
          break;
        case 2:
          sr(idx) -= bm * tx_r;
          si(idx) -= bm * tx_i;
          break;
        case 3:
          sr(idx) += bm * tx_i;
          si(idx) -= bm * tx_r;
          break;
        case 4:
          sr(idx) -= bm * tx_i;
          si(idx) += bm * tx_r;
          break;
      }
    });
}
