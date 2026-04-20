#include <Kokkos_Core.hpp>
#include <iostream>
#include <fstream>
#include <chrono>

constexpr const int dim = 2;
constexpr const int NP = 4;

using real = double;

template <int np>
struct element {
  real metdet[np*np];
  real Dinv[np*np*2*2];
  real rmetdet[np*np];
};

template <int np>
struct derivative {
  real Dvv[np*np];
};

void readVelocity(real *v, const int np, std::istream *input) {
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < np; j++)
      for(int k = 0; k < np; k++)
        (*input) >> v[k*np*2+2*j+i];
}

template <int np>
void readElement(element<np> &elem, std::istream *input) {
  for(int i = 0; i < np; i++)
    for(int j = 0; j < np; j++) {
      (*input) >> elem.metdet[i*np+j];
      elem.rmetdet[i*np+j] = 1 / elem.metdet[i*np+j];
    }
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 2; j++)
      for(int k = 0; k < np; k++)
        for(int l = 0; l < np; l++)
          (*input) >> elem.Dinv[4*np*l+4*k+2*i+j];
}

template <int np>
void readDerivative(derivative<np> &deriv, std::istream *input) {
  for(int i = 0; i < np; i++)
    for(int j = 0; j < np; j++)
      (*input) >> deriv.Dvv[j*np+i];
}

template <typename real>
void readDivergence(real *divergence, const int np, std::istream *input) {
  for(int i = 0; i < np; i++)
    for(int j = 0; j < np; j++)
      (*input) >> divergence[i*np+j];
}

template <int np>
void divergence_sphere_cpu(
    const real *v,
    const derivative<np> &deriv,
    const element<np> &elem,
    real *div)
{
  real gv[np*np*dim];
  for(int j = 0; j < np; j++)
    for(int i = 0; i < np; i++)
      for(int k = 0; k < dim; k++)
        gv[j*np*dim+i*dim+k] = elem.metdet[j*np+i] *
          (elem.Dinv[j*np*dim*dim+i*dim*dim+k*dim]   * v[j*dim*np+dim*i] +
           elem.Dinv[j*np*dim*dim+i*dim*dim+k*dim+1] * v[j*dim*np+dim*i+1]);

  real vvtemp[np*np];
  for(int l = 0; l < np; l++)
    for(int j = 0; j < np; j++) {
      real dudx00 = 0, dvdy00 = 0;
      for(int i = 0; i < np; i++) {
        dudx00 += deriv.Dvv[l*np+i] * gv[j*np*dim+i*dim];
        dvdy00 += deriv.Dvv[l*np+i] * gv[i*np*dim+j*dim+1];
      }
      div[j*np+l] = dudx00;
      vvtemp[l*np+j] = dvdy00;
    }

  constexpr const real rrearth = 1.5683814303638645E-7;
  for(int i = 0; i < np; i++)
    for(int j = 0; j < np; j++)
      div[i*np+j] = (div[i*np+j] + vvtemp[i*np+j]) * (elem.rmetdet[i*np+j] * rrearth);
}

// Kokkos GPU version — pre-allocates views outside the timed loop
struct DivergenceKokkos {
  Kokkos::View<real*> d_gv;
  Kokkos::View<real*> d_Dvv;
  Kokkos::View<real*> d_rmetdet;
  Kokkos::View<real*> d_div;
  Kokkos::View<real*> d_vvtemp;

  typename Kokkos::View<real*>::HostMirror h_gv;
  typename Kokkos::View<real*>::HostMirror h_Dvv;
  typename Kokkos::View<real*>::HostMirror h_rmetdet;
  typename Kokkos::View<real*>::HostMirror h_div;

  static constexpr int np = NP;

  DivergenceKokkos()
    : d_gv("gv", np*np*dim),
      d_Dvv("Dvv", np*np),
      d_rmetdet("rmetdet", np*np),
      d_div("div", np*np),
      d_vvtemp("vvtemp", np*np)
  {
    h_gv      = Kokkos::create_mirror_view(d_gv);
    h_Dvv     = Kokkos::create_mirror_view(d_Dvv);
    h_rmetdet = Kokkos::create_mirror_view(d_rmetdet);
    h_div     = Kokkos::create_mirror_view(d_div);
  }

  void upload(const real *v,
              const derivative<NP> &deriv,
              const element<NP> &elem)
  {
    // Compute gv on host (tiny: 4*4*2=32 elements)
    for(int j = 0; j < np; j++)
      for(int i = 0; i < np; i++)
        for(int k = 0; k < dim; k++)
          h_gv(j*np*dim+i*dim+k) = elem.metdet[j*np+i] *
            (elem.Dinv[j*np*dim*dim+i*dim*dim+k*dim]   * v[j*dim*np+dim*i] +
             elem.Dinv[j*np*dim*dim+i*dim*dim+k*dim+1] * v[j*dim*np+dim*i+1]);

    for(int i = 0; i < np*np; i++) {
      h_Dvv(i)     = deriv.Dvv[i];
      h_rmetdet(i) = elem.rmetdet[i];
    }
    Kokkos::deep_copy(d_gv,      h_gv);
    Kokkos::deep_copy(d_Dvv,     h_Dvv);
    Kokkos::deep_copy(d_rmetdet, h_rmetdet);
  }

  void run()
  {
    auto l_gv      = d_gv;
    auto l_Dvv     = d_Dvv;
    auto l_rmetdet = d_rmetdet;
    auto l_div     = d_div;
    auto l_vvtemp  = d_vvtemp;

    Kokkos::parallel_for("div_k1",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{np,np}),
      KOKKOS_LAMBDA(int j, int l) {
        real dudx00 = 0.0, dvdy00 = 0.0;
        for(int i = 0; i < np; i++) {
          dudx00 += l_Dvv(l*np+i) * l_gv(j*np*dim+i*dim);
          dvdy00 += l_Dvv(l*np+i) * l_gv(i*np*dim+j*dim+1);
        }
        l_div(j*np+l)    = dudx00;
        l_vvtemp(l*np+j) = dvdy00;
      });
    Kokkos::fence();

    constexpr real rrearth = 1.5683814303638645E-7;
    Kokkos::parallel_for("div_k2",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{np,np}),
      KOKKOS_LAMBDA(int l, int j) {
        l_div(l*np+j) = (l_div(l*np+j) + l_vvtemp(l*np+j)) *
                        l_rmetdet(l*np+j) * rrearth;
      });
    Kokkos::fence();
  }

  void download(real *div_out)
  {
    Kokkos::deep_copy(h_div, d_div);
    for(int i = 0; i < np*np; i++) div_out[i] = h_div(i);
  }
};

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    constexpr const int np = NP;
    real v[np*np*dim];
    element<np> elem;
    derivative<np> deriv;
    real divergence_e[np*np];

    {
      std::istream *input;
      if(argc > 1)
        input = new std::ifstream(argv[1]);
      else
        input = &std::cin;
      readVelocity(v, np, input);
      readElement(elem, input);
      readDerivative(deriv, input);
      readDivergence(divergence_e, np, input);
      if(argc > 1)
        delete input;
    }

    const int numtests = (argc > 2) ? std::stoi(argv[2]) : 100000;

    // Warmup + timed CPU run
    std::cout << "Divergence on the CPU\n";
    real divergence_c[np*np];
    for(int i = 0; i < numtests; i++)
      divergence_sphere_cpu<np>(v, deriv, elem, divergence_c);
    auto cpu_t0 = std::chrono::steady_clock::now();
    for(int i = 0; i < numtests; i++)
      divergence_sphere_cpu<np>(v, deriv, elem, divergence_c);
    auto cpu_t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(cpu_t1-cpu_t0).count()*1e-3;

    // Warmup + timed GPU run
    std::cout << "Divergence on the GPU\n";
    DivergenceKokkos dk;
    dk.upload(v, deriv, elem);

    real divergence_f[np*np];
    for(int i = 0; i < numtests; i++) {
      dk.upload(v, deriv, elem);
      dk.run();
    }
    dk.download(divergence_f);

    auto gpu_t0 = std::chrono::steady_clock::now();
    for(int i = 0; i < numtests; i++) {
      dk.upload(v, deriv, elem);
      dk.run();
    }
    dk.download(divergence_f);
    auto gpu_t1 = std::chrono::steady_clock::now();
    double gpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(gpu_t1-gpu_t0).count()*1e-3;

    std::cout << "Divergence Errors\n";
    std::cout << "CPU             GPU\n";
    for(int i = 0; i < np; i++) {
      for(int j = 0; j < np; j++) {
        std::cout << divergence_c[i*np+j] - divergence_e[i*np+j]
                  << "    "
                  << divergence_f[i*np+j] - divergence_e[i*np+j]
                  << "\n";
      }
      std::cout << "\n";
    }
    printf("CPU time: %.3f ms  GPU time: %.3f ms\n", cpu_ms, gpu_ms);
  }
  Kokkos::finalize();
  return 0;
}
