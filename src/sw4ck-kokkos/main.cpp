//////////////////////////////////////////////////////////////////////////////
//// Copyright (c) 2021, Lawrence Livermore National Security, LLC and SW4CK
//// project contributors. See the COPYRIGHT file for details.
////
//// SPDX-License-Identifier: GPL-2.0-only
////
//// Kokkos port: replaces OpenMP target offloading with Kokkos parallel constructs.
////////////////////////////////////////////////////////////////////////////////

#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <vector>
#include <tuple>
#include <chrono>
#include <cstdlib>

#include "sw4ck.h"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <path to file> <repeat>\n";
    return 1;
  }

  std::ifstream iff;
  iff.open(argv[1]);

  const int repeat = atoi(argv[2]);

  std::map<std::string, Sarray*> arrays[10];
  std::vector<int*> onesided;
  std::string line;
  int lc = 0;
  std::cout << "Reading from file " << argv[1] << "\n";
  while (std::getline(iff, line)) {
    std::istringstream iss(line);
    int* optr = new int[14];
    const int N = 16;
    if ((lc % N) == 0) {
      if (!(iss >> optr[0] >> optr[1] >> optr[2] >> optr[3] >> optr[4] >>
            optr[5] >> optr[6] >> optr[7] >> optr[8] >> optr[9] >> optr[10] >>
            optr[11] >> optr[12] >> optr[13])) {
        std::cerr << "Error reading data on line " << lc + 1 << "\n";
        break;
      }
      onesided.push_back(optr);
    } else {
      Sarray* s = new Sarray();
      auto name = s->fill(iss);
      if (name == "Break") {
        std::cerr << "Error reading Sarray data on line " << lc + 1 << "\n";
        break;
      } else {
        arrays[lc / N][name] = s;
      }
    }
    lc++;
  }

  for (int i = 0; i < 2; i++)
    for (auto const& x : arrays[i])
      x.second->init();

  int cof_size = (6 + 384 + 24 + 48 + 6 + 384 + 6 + 6);
  float_sw4 *cof_h = (float_sw4*) malloc(sizeof(float_sw4) * cof_size);
  for (int i = 0; i < cof_size; i++) cof_h[i] = i / 1000.0;

  float_sw4 exact_norm[2] = {2.2502232733796421194, 202.0512747393526638};

  Kokkos::initialize(argc, argv);
  {
    // Allocate device View for cof (shared across iterations)
    Kokkos::View<float_sw4*> cof_d("cof_d", cof_size);
    {
      auto cof_host = Kokkos::View<float_sw4*, Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(cof_h, cof_size);
      Kokkos::deep_copy(cof_d, cof_host);
    }

    for (int i = 0; i < 2; i++) {
      int* optr = onesided[i];

      auto& arr_alpha  = *arrays[i]["a_AlphaVE_0"];
      auto& arr_mua    = *arrays[i]["mMuVE_0"];
      auto& arr_lambda = *arrays[i]["mLambdaVE_0"];
      auto& arr_met    = *arrays[i]["mMetric"];
      auto& arr_jac    = *arrays[i]["mJ"];
      auto& arr_uacc   = *arrays[i]["a_Uacc"];

      int alpha_size  = arr_alpha.m_nc  * arr_alpha.m_ni  * arr_alpha.m_nj  * arr_alpha.m_nk;
      int mua_size    = arr_mua.m_nc    * arr_mua.m_ni    * arr_mua.m_nj    * arr_mua.m_nk;
      int lambda_size = arr_lambda.m_nc * arr_lambda.m_ni * arr_lambda.m_nj * arr_lambda.m_nk;
      int met_size    = arr_met.m_nc    * arr_met.m_ni    * arr_met.m_ni    * arr_met.m_nk;
      int jac_size    = arr_jac.m_nc    * arr_jac.m_ni    * arr_jac.m_nj    * arr_jac.m_nk;
      int uacc_size   = arr_uacc.m_nc   * arr_uacc.m_ni   * arr_uacc.m_nj   * arr_uacc.m_nk;

      // Use actual size from m_npts
      int alpha_sz  = (int)arr_alpha.m_npts;
      int mua_sz    = (int)arr_mua.m_npts;
      int lambda_sz = (int)arr_lambda.m_npts;
      int met_sz    = (int)arr_met.m_npts;
      int jac_sz    = (int)arr_jac.m_npts;
      int uacc_sz   = (int)arr_uacc.m_npts;

      // Allocate device views
      Kokkos::View<float_sw4*> alpha_d ("alpha_d",  alpha_sz);
      Kokkos::View<float_sw4*> mua_d   ("mua_d",    mua_sz);
      Kokkos::View<float_sw4*> lambda_d("lambda_d", lambda_sz);
      Kokkos::View<float_sw4*> met_d   ("met_d",    met_sz);
      Kokkos::View<float_sw4*> jac_d   ("jac_d",    jac_sz);
      Kokkos::View<float_sw4*> uacc_d  ("uacc_d",   uacc_sz);

      // H2D copies
      Kokkos::deep_copy(alpha_d,
        Kokkos::View<float_sw4*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(arr_alpha.m_data, alpha_sz));
      Kokkos::deep_copy(mua_d,
        Kokkos::View<float_sw4*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(arr_mua.m_data, mua_sz));
      Kokkos::deep_copy(lambda_d,
        Kokkos::View<float_sw4*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(arr_lambda.m_data, lambda_sz));
      Kokkos::deep_copy(met_d,
        Kokkos::View<float_sw4*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(arr_met.m_data, met_sz));
      Kokkos::deep_copy(jac_d,
        Kokkos::View<float_sw4*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(arr_jac.m_data, jac_sz));

      int nkg = optr[12];
      char op = '-';

      int sg_str_size = (optr[7] - optr[6] + optr[9] - optr[8] + 2);
      float_sw4* sg_str = (float_sw4*) malloc(sg_str_size * sizeof(float_sw4));
      for (int n = 0; n < sg_str_size; n++) sg_str[n] = n / 1000.0;

      Kokkos::View<float_sw4*> sg_str_d("sg_str_d", sg_str_size);
      Kokkos::deep_copy(sg_str_d,
        Kokkos::View<float_sw4*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(sg_str, sg_str_size));

      double time = 0.0;

      for (int p = 0; p < repeat; p++) {
        // Reset uacc on device each iteration
        Kokkos::deep_copy(uacc_d,
          Kokkos::View<float_sw4*, Kokkos::HostSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>(arr_uacc.m_data, uacc_sz));

        auto start = std::chrono::high_resolution_clock::now();

        curvilinear4sg_ci(
          optr[6], optr[7], optr[8], optr[9], optr[10], optr[11],
          alpha_d.data(),  // d_u (the "alpha" acceleration array in this benchmark)
          mua_d.data(),
          lambda_d.data(),
          met_d.data(),
          jac_d.data(),
          uacc_d.data(),
          optr,
          cof_d.data(),
          sg_str_d.data(),
          nkg, op);

        auto end = std::chrono::high_resolution_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      }

      std::cout << "\nAverage execution time of sw4ck kernels: "
                << (time * 1e-6f) / repeat << " milliseconds\n\n";

      // D2H copy of uacc result
      Kokkos::deep_copy(
        Kokkos::View<float_sw4*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(arr_uacc.m_data, uacc_sz),
        uacc_d);

      float_sw4 norm = arr_uacc.norm();
      float_sw4 err = (norm - exact_norm[i]) / exact_norm[i] * 100;
      std::cout << "Error = " << err << " %\n";

      free(sg_str);
      delete optr;
    }

    free(cof_h);
  }
  Kokkos::finalize();

  for (int i = 0; i < 2; i++)
    for (auto const& x : arrays[i])
      delete x.second;

  return 0;
}
