//////////////////////////////////////////////////////////////////////////////
//// Copyright (c) 2021, Lawrence Livermore National Security, LLC and SW4CK
//// project contributors. See the COPYRIGHT file for details.
////
//// SPDX-License-Identifier: GPL-2.0-only
////
//// Kokkos port: replaces OpenMP target offloading with Kokkos parallel constructs.
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstddef>
#include <sstream>
#include <string>
#include <tuple>
#include <sys/types.h>

#define float_sw4 double

class Sarray {
  public:
    Sarray() {}
    ~Sarray();
    Sarray(int nc, int ibeg, int iend, int jbeg, int jend, int kbeg, int kend);
    std::string fill(std::istringstream& iss);
    void init();
    float_sw4 norm();
    std::tuple<float_sw4,float_sw4> minmax();
    int m_nc, m_ni, m_nj, m_nk;
    int m_ib, m_ie, m_jb, m_je, m_kb, m_ke;
    ssize_t m_base;
    size_t m_offi, m_offj, m_offk, m_offc, m_npts;
    float_sw4* m_data;
    size_t size;
    int g;
};

void curvilinear4sg_ci(
    int ifirst, int ilast,
    int jfirst, int jlast,
    int kfirst, int klast,
    float_sw4* d_u,
    float_sw4* d_mu,
    float_sw4* d_lambda,
    float_sw4* d_met,
    float_sw4* d_jac,
    float_sw4* d_lu,
    int* onesided,
    float_sw4* d_cof,
    float_sw4* d_str,
    int nk, char op);
