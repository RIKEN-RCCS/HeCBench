#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <sys/resource.h>
#include <math.h>
#include <chrono>
#include <random>
#include <cassert>

//=============================================================================
// Type definitions (mirrors dslash.h but without OMP pragmas)
//=============================================================================

#ifndef ITERATIONS
#  define ITERATIONS 100
#endif
#ifndef PRECISION
#  define PRECISION 2
#endif
#ifndef LDIM
#  define LDIM 32
#endif

typedef struct { float  real; float  imag; } fcomplex;
typedef struct { double real; double imag; } dcomplex;
typedef struct { fcomplex e[3][3]; } fsu3_matrix;
typedef struct { fcomplex c[3];    } fsu3_vector;
typedef struct { dcomplex e[3][3]; } dsu3_matrix;
typedef struct { dcomplex c[3];    } dsu3_vector;

#if (PRECISION==1)
  typedef fsu3_matrix su3_matrix;
  typedef fsu3_vector su3_vector;
  typedef float       Real;
  typedef fcomplex    Complx;
  #define EPISON 2E-5
#else
  typedef dsu3_matrix su3_matrix;
  typedef dsu3_vector su3_vector;
  typedef double      Real;
  typedef dcomplex    Complx;
  #define EPISON 2E-6
#endif

#define CADD(a,b,c)    { (c).real = (a).real + (b).real; (c).imag = (a).imag + (b).imag; }
#define CSUB(a,b,c)    { (c).real = (a).real - (b).real; (c).imag = (a).imag - (b).imag; }
#define CMULSUM(a,b,c) { (c).real += (a).real*(b).real - (a).imag*(b).imag; \
                         (c).imag += (a).real*(b).imag + (a).imag*(b).real; }
#define CMUL(a,b,c)    { (c).real = (a).real*(b).real - (a).imag*(b).imag; \
                         (c).imag = (a).real*(b).imag + (a).imag*(b).real; }
#define CSUM(a,b)      { (a).real += (b).real; (a).imag += (b).imag; }
#define CONJG(a,b)     { (b).real = (a).real; (b).imag = -(a).imag; }

KOKKOS_INLINE_FUNCTION
void su3_adjoint(const su3_matrix *a, su3_matrix *b) {
  for(int i=0; i<3; i++)
    for(int j=0; j<3; j++) { CONJG(a->e[j][i], b->e[i][j]); }
}

KOKKOS_INLINE_FUNCTION
void mult_su3_mat_vec(const su3_matrix *a, const su3_vector *b, su3_vector *c) {
  for(int i=0; i<3; i++) {
    Complx x = {0.0, 0.0};
    for(int j=0; j<3; j++) { CMULSUM(a->e[i][j], b->c[j], x); }
    c->c[i] = x;
  }
}

KOKKOS_INLINE_FUNCTION
void mult_su3_mat_vec_sum(const su3_matrix *a, const su3_vector *b, su3_vector *c) {
  for(int i=0; i<3; i++) {
    Complx x = {0.0, 0.0};
    for(int j=0; j<3; j++) { CMULSUM(a->e[i][j], b->c[j], x); }
    c->c[i].real += x.real;
    c->c[i].imag += x.imag;
  }
}

KOKKOS_INLINE_FUNCTION
void add_su3_vector(const su3_vector *a, const su3_vector *b, su3_vector *c) {
  for(int i=0; i<3; i++) { CADD(a->c[i], b->c[i], c->c[i]); }
}

KOKKOS_INLINE_FUNCTION
void sub_su3_vector(const su3_vector *a, const su3_vector *b, su3_vector *c) {
  for(int i=0; i<3; i++) { CSUB(a->c[i], b->c[i], c->c[i]); }
}

//=============================================================================
// Lattice setup
//=============================================================================

static const int nx = LDIM, ny = LDIM, nz = LDIM, nt = LDIM;
static size_t sites_on_node      = (size_t)LDIM*LDIM*LDIM*LDIM;
static size_t even_sites_on_node = sites_on_node / 2;

inline size_t node_index(int x, int y, int z, int t) {
  int xr = ((x%nx)+nx)%nx, yr = ((y%ny)+ny)%ny;
  int zr = ((z%nz)+nz)%nz, tr = ((t%nt)+nt)%nt;
  size_t i = xr + nx*(yr + ny*(zr + nz*tr));
  if((x+y+z+t)%2==0) return i/2;
  else                return (i + sites_on_node)/2;
}

void set_neighbors(size_t *fwd, size_t *bck, size_t *fwd3, size_t *bck3) {
  for(int x=0; x<nx; x++)
    for(int y=0; y<ny; y++)
      for(int z=0; z<nz; z++)
        for(int t=0; t<nt; t++) {
          size_t i = node_index(x,y,z,t);
          fwd [4*i+0] = node_index(x+1,y,z,t);
          bck [4*i+0] = node_index(x-1,y,z,t);
          fwd [4*i+1] = node_index(x,y+1,z,t);
          bck [4*i+1] = node_index(x,y-1,z,t);
          fwd [4*i+2] = node_index(x,y,z+1,t);
          bck [4*i+2] = node_index(x,y,z-1,t);
          fwd [4*i+3] = node_index(x,y,z,t+1);
          bck [4*i+3] = node_index(x,y,z,t-1);
          fwd3[4*i+0] = node_index(x+3,y,z,t);
          bck3[4*i+0] = node_index(x-3,y,z,t);
          fwd3[4*i+1] = node_index(x,y+3,z,t);
          bck3[4*i+1] = node_index(x,y-3,z,t);
          fwd3[4*i+2] = node_index(x,y,z+3,t);
          bck3[4*i+2] = node_index(x,y,z-3,t);
          fwd3[4*i+3] = node_index(x,y,z,t+3);
          bck3[4*i+3] = node_index(x,y,z,t-3);
        }
}

//=============================================================================
// Initialisation helpers
//=============================================================================

std::random_device rd_g;
std::mt19937 gen_g(rd_g());
std::uniform_real_distribution<> dis_g(-1.f, 1.f);

void init_mat(su3_matrix *s) {
  Real r=dis_g(gen_g), im=dis_g(gen_g);
  for(int k=0; k<3; k++)
    for(int l=0; l<3; l++) { s->e[k][l].real=r; s->e[k][l].imag=im; }
}
void init_vec(su3_vector *s) {
  Real r=dis_g(gen_g), im=dis_g(gen_g);
  for(int k=0; k<3; k++) { s->c[k].real=r; s->c[k].imag=im; }
}
void make_data(su3_vector *src, su3_matrix *fat, su3_matrix *lng, size_t n) {
  for(size_t i=0; i<n; i++) {
    init_vec(src+i);
    for(int dir=0; dir<4; dir++) { init_mat(fat+4*i+dir); init_mat(lng+4*i+dir); }
  }
}

template<class T>
bool almost_equal(T x, T y, double tol) { return std::abs(x-y)<tol; }

//=============================================================================
// CPU reference dslash (for validation)
//=============================================================================
void dslash_fn_field_cpu(
    su3_vector *src, su3_vector *dst,
    su3_matrix *fat, su3_matrix *lng,
    su3_matrix *fatbck, su3_matrix *lngbck,
    size_t *fwd, size_t *bck, size_t *fwd3, size_t *bck3)
{
  for(size_t i=0; i<even_sites_on_node; i++) {
    su3_vector tvec;
    mult_su3_mat_vec(fat+4*i+0, src+fwd[4*i+0], dst+i);
    for(int k=1; k<4; k++) mult_su3_mat_vec_sum(fat+4*i+k, src+fwd[4*i+k], dst+i);
    mult_su3_mat_vec(lng+4*i+0, src+fwd3[4*i+0], &tvec);
    for(int k=1; k<4; k++) mult_su3_mat_vec_sum(lng+4*i+k, src+fwd3[4*i+k], &tvec);
    add_su3_vector(dst+i, &tvec, dst+i);
    mult_su3_mat_vec(fatbck+4*i+0, src+bck[4*i+0], &tvec);
    for(int k=1; k<4; k++) mult_su3_mat_vec_sum(fatbck+4*i+k, src+bck[4*i+k], &tvec);
    sub_su3_vector(dst+i, &tvec, dst+i);
    mult_su3_mat_vec(lngbck+4*i+0, src+bck3[4*i+0], &tvec);
    for(int k=1; k<4; k++) mult_su3_mat_vec_sum(lngbck+4*i+k, src+bck3[4*i+k], &tvec);
    sub_su3_vector(dst+i, &tvec, dst+i);
  }
}

//=============================================================================
// main
//=============================================================================
int main(int argc, char **argv) {
  if(argc < 2) { std::cerr << "Usage: " << argv[0] << " <workgroup_size>" << std::endl; return 1; }
  size_t workgroup_size = atoi(argv[1]);
  size_t iterations     = ITERATIONS;

  size_t total_sites = sites_on_node;

  std::vector<su3_vector> src(total_sites), dst(total_sites), chkdst(total_sites);
  std::vector<su3_matrix> fat(total_sites*4), lng(total_sites*4);
  std::vector<su3_matrix> fatbck(total_sites*4), lngbck(total_sites*4);

  std::vector<size_t> fwd(total_sites*4), bck(total_sites*4);
  std::vector<size_t> fwd3(total_sites*4), bck3(total_sites*4);

  set_neighbors(fwd.data(), bck.data(), fwd3.data(), bck3.data());
  make_data(src.data(), fat.data(), lng.data(), total_sites);

  std::cout << "Number of sites = " << LDIM << "^4" << std::endl;
  std::cout << "Executing " << iterations << " iterations" << std::endl;

  Kokkos::initialize(argc, argv);
  {
    // Allocate device views
    Kokkos::View<su3_vector*> d_src("src", total_sites);
    Kokkos::View<su3_vector*> d_dst("dst", total_sites);
    Kokkos::View<su3_matrix*> d_fat("fat", total_sites*4);
    Kokkos::View<su3_matrix*> d_lng("lng", total_sites*4);
    Kokkos::View<su3_matrix*> d_fatbck("fatbck", total_sites*4);
    Kokkos::View<su3_matrix*> d_lngbck("lngbck", total_sites*4);
    Kokkos::View<size_t*>     d_fwd("fwd",  total_sites*4);
    Kokkos::View<size_t*>     d_bck("bck",  total_sites*4);
    Kokkos::View<size_t*>     d_fwd3("fwd3", total_sites*4);
    Kokkos::View<size_t*>     d_bck3("bck3", total_sites*4);

    // Upload src, fat, lng, neighbor arrays
    {
      auto h_src  = Kokkos::create_mirror_view(d_src);
      auto h_fat  = Kokkos::create_mirror_view(d_fat);
      auto h_lng  = Kokkos::create_mirror_view(d_lng);
      auto h_fwd  = Kokkos::create_mirror_view(d_fwd);
      auto h_bck  = Kokkos::create_mirror_view(d_bck);
      auto h_fwd3 = Kokkos::create_mirror_view(d_fwd3);
      auto h_bck3 = Kokkos::create_mirror_view(d_bck3);
      for(size_t i=0; i<total_sites;   i++) h_src(i)  = src[i];
      for(size_t i=0; i<total_sites*4; i++) { h_fat(i)=fat[i]; h_lng(i)=lng[i]; }
      for(size_t i=0; i<total_sites*4; i++) {
        h_fwd(i)=fwd[i]; h_bck(i)=bck[i];
        h_fwd3(i)=fwd3[i]; h_bck3(i)=bck3[i];
      }
      Kokkos::deep_copy(d_src, h_src);
      Kokkos::deep_copy(d_fat, h_fat);
      Kokkos::deep_copy(d_lng, h_lng);
      Kokkos::deep_copy(d_fwd,  h_fwd);
      Kokkos::deep_copy(d_bck,  h_bck);
      Kokkos::deep_copy(d_fwd3, h_fwd3);
      Kokkos::deep_copy(d_bck3, h_bck3);
    }

    // Build backward links on device
    {
      auto l_fat    = d_fat;
      auto l_lng    = d_lng;
      auto l_fatbck = d_fatbck;
      auto l_lngbck = d_lngbck;
      auto l_bck    = d_bck;
      auto l_bck3   = d_bck3;
      size_t n_even = even_sites_on_node;
      Kokkos::parallel_for("build_back_links", n_even,
        KOKKOS_LAMBDA(size_t mySite) {
          for(int dir=0; dir<4; dir++) {
            su3_adjoint(l_fat.data() + 4*l_bck(4*mySite+dir) + dir,
                        l_fatbck.data() + 4*mySite + dir);
            su3_adjoint(l_lng.data() + 4*l_bck3(4*mySite+dir) + dir,
                        l_lngbck.data() + 4*mySite + dir);
          }
        });
      Kokkos::fence();
    }

    // Timed dslash loop
    size_t n_even = even_sites_on_node;
    auto tstart = std::chrono::steady_clock::now();

    for(size_t iters=0; iters<iterations; iters++) {
      auto l_src    = d_src;
      auto l_dst    = d_dst;
      auto l_fat    = d_fat;
      auto l_lng    = d_lng;
      auto l_fatbck = d_fatbck;
      auto l_lngbck = d_lngbck;
      auto l_fwd    = d_fwd;
      auto l_bck    = d_bck;
      auto l_fwd3   = d_fwd3;
      auto l_bck3   = d_bck3;

      Kokkos::parallel_for("dslash", n_even,
        KOKKOS_LAMBDA(size_t mySite) {
          su3_vector v;
          // Forward fat links
          mult_su3_mat_vec(l_fat.data() + mySite*4 + 0,
                           l_src.data() + l_fwd(4*mySite+0), l_dst.data()+mySite);
          for(int k=1; k<4; k++)
            mult_su3_mat_vec_sum(l_fat.data() + mySite*4 + k,
                                 l_src.data() + l_fwd(4*mySite+k), l_dst.data()+mySite);
          // Forward long links
          mult_su3_mat_vec(l_lng.data() + mySite*4 + 0,
                           l_src.data() + l_fwd3(4*mySite+0), &v);
          for(int k=1; k<4; k++)
            mult_su3_mat_vec_sum(l_lng.data() + mySite*4 + k,
                                 l_src.data() + l_fwd3(4*mySite+k), &v);
          add_su3_vector(l_dst.data()+mySite, &v, l_dst.data()+mySite);
          // Backward fat links
          mult_su3_mat_vec(l_fatbck.data() + mySite*4 + 0,
                           l_src.data() + l_bck(4*mySite+0), &v);
          for(int k=1; k<4; k++)
            mult_su3_mat_vec_sum(l_fatbck.data() + mySite*4 + k,
                                 l_src.data() + l_bck(4*mySite+k), &v);
          sub_su3_vector(l_dst.data()+mySite, &v, l_dst.data()+mySite);
          // Backward long links
          mult_su3_mat_vec(l_lngbck.data() + mySite*4 + 0,
                           l_src.data() + l_bck3(4*mySite+0), &v);
          for(int k=1; k<4; k++)
            mult_su3_mat_vec_sum(l_lngbck.data() + mySite*4 + k,
                                 l_src.data() + l_bck3(4*mySite+k), &v);
          sub_su3_vector(l_dst.data()+mySite, &v, l_dst.data()+mySite);
        });
      Kokkos::fence();
    }

    auto tend = std::chrono::steady_clock::now();
    double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(tend-tstart).count() / 1.0e6;

    // Copy dst back
    {
      auto h_dst    = Kokkos::create_mirror_view(d_dst);
      auto h_fatbck = Kokkos::create_mirror_view(d_fatbck);
      auto h_lngbck = Kokkos::create_mirror_view(d_lngbck);
      Kokkos::deep_copy(h_dst,    d_dst);
      Kokkos::deep_copy(h_fatbck, d_fatbck);
      Kokkos::deep_copy(h_lngbck, d_lngbck);
      for(size_t i=0; i<total_sites;   i++) dst[i]    = h_dst(i);
      for(size_t i=0; i<total_sites*4; i++) { fatbck[i]=h_fatbck(i); lngbck[i]=h_lngbck(i); }
    }

    std::cout << "Total execution time = " << ttotal << " secs" << std::endl;

    // Validate
    std::cout << "Validating..." << std::endl;
    dslash_fn_field_cpu(src.data(), chkdst.data(),
        fat.data(), lng.data(), fatbck.data(), lngbck.data(),
        fwd.data(), bck.data(), fwd3.data(), bck3.data());

    bool pass = true;
    for(size_t i=0; i<even_sites_on_node && pass; i++)
      for(int k=0; k<3; k++) {
        if(!almost_equal<Real>(dst[i].c[k].real, chkdst[i].c[k].real, EPISON) ||
           !almost_equal<Real>(dst[i].c[k].imag, chkdst[i].c[k].imag, EPISON)) {
          pass = false; break;
        }
      }
    if(!pass) { std::cout << "VALIDATION FAILED" << std::endl; }

    // Performance metrics
    const double tflop = (double)iterations * even_sites_on_node * 1182;
    std::cout << "Total GFLOP/s = " << tflop / ttotal / 1.0e9 << std::endl;
    const double memory_usage = (double)even_sites_on_node *
      (sizeof(su3_matrix)*4*4 + sizeof(su3_vector)*16 + sizeof(size_t)*16 + sizeof(su3_vector));
    std::cout << "Total GByte/s (GPU memory) = " << iterations * memory_usage / ttotal / 1.0e9 << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
