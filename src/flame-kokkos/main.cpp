// Kokkos port of flame-cuda (fractal flame IFS renderer)
// Constant memory replaced with Kokkos::View<ConstMemParams*>.
// short2/float2 device storage replaced with Kokkos::View<float*>.
// Variation functions ported from iteration.cu with KOKKOS_INLINE_FUNCTION.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <sys/time.h>
#include <Kokkos_Core.hpp>

#define WIDTH   800
#define HEIGHT  600
#define NUM_FUNCTIONS       20
#define VARIATIONS_PER_FUNCTION 5
#define NUM_THREADS         (1 << 14)
#define THREADS_PER_BLOCK   32
#define NUM_POINTS_PER_THREAD 64
#define NUM_ITERATIONS      15
#define NUM_RANDOMS         (1 << 22)
#define NUM_PERMUTATIONS    2048

#ifndef M_PI
#define M_PI 3.1415926535f
#endif

struct VariationParameter { int idx; float factor; };

struct ConstMemParams {
  VariationParameter variation_parameters[NUM_FUNCTIONS][VARIATIONS_PER_FUNCTION];
  float function_colors[NUM_FUNCTIONS];
  float pre_transform_params[NUM_FUNCTIONS][6];
  float post_transform_params[NUM_FUNCTIONS][6];
  int   frame_counter;
  int   enable_sierpinski;
  int   thread_function_mapping[NUM_FUNCTIONS];
};

struct PermSortElement {
  int value; unsigned short idx;
  bool operator<(const PermSortElement& o) const { return value < o.value; }
};

// ─── host utilities ───────────────────────────────────────────────────────────

unsigned mersenne_twister(unsigned* s)
{
  const int N=624, M=397;
  static int idx=N+1;
  const unsigned A[2]={0,0x9908b0df}, HI=0x80000000, LO=0x7fffffff;
  if(idx>=N){
    for(int k=0;k<N-M;++k){ unsigned h=(s[k]&HI)|(s[k+1]&LO); s[k]=s[k+M]^(h>>1)^A[h&1]; }
    for(int k=N-M;k<N-1;++k){ unsigned h=(s[k]&HI)|(s[k+1]&LO); s[k]=s[k+(M-N)]^(h>>1)^A[h&1]; }
    unsigned h=(s[N-1]&HI)|(s[0]&LO); s[N-1]=s[M-1]^(h>>1)^A[h&1]; idx=0;
  }
  unsigned e=s[idx++]; e^=e>>11; e^=(e<<7)&0x9d2c5680; e^=(e<<15)&0xefc60000; e^=e>>18;
  return e;
}

float radical_inverse(unsigned n, unsigned base)
{
  float res=0, div=1.f/base;
  while(n){ float d=n%base; res+=d*div; n=(n-d)/base; div/=base; }
  return res;
}

// ─── device iteration ────────────────────────────────────────────────────────

KOKKOS_INLINE_FUNCTION void affine_transform(float& px, float& py, const float* p)
{
  float tx = p[0]*px + p[1]*py + p[2];
  float ty = p[3]*px + p[4]*py + p[5];
  px=tx; py=ty;
}

KOKKOS_INLINE_FUNCTION void sierpinski(float& px, float& py, int i)
{
  px*=.5f; py*=.5f;
  switch(i%3){case 1:px+=.5f;break;case 2:py+=.5f;break;}
}

KOKKOS_INLINE_FUNCTION
int get_function_idx(int thread_idx, const ConstMemParams& params)
{
  int start=0, end=NUM_FUNCTIONS, func=0;
  thread_idx &= ~31;
  for(int i=0;i<6;i++){
    func=(start+end)/2;
    if(params.thread_function_mapping[func]<=thread_idx) start=func+1;
    else end=func;
  }
  return func;
}

KOKKOS_INLINE_FUNCTION
void iteration_fractal_flame(float& px, float& py, float& color,
    int func_idx, int thread_idx,
    const float* random_numbers,
    const ConstMemParams& params)
{
  if(params.enable_sierpinski==1) sierpinski(px,py,thread_idx);
  else if(params.enable_sierpinski==2) sierpinski(px,py,func_idx);

  affine_transform(px,py,&params.pre_transform_params[func_idx][0]);

  float rad2=px*px+py*py, rad=Kokkos::sqrt(rad2), inv_r=(rad>1e-8f)?1.f/rad:1.f;
  float theta=Kokkos::atan2(px,py), phi=Kokkos::atan2(py,px);

  float ox=0,oy=0;
  const VariationParameter* vp=&params.variation_parameters[func_idx][0];
  for(int i=0;i<VARIATIONS_PER_FUNCTION;i++){
    if(Kokkos::fabs(vp[i].factor)<0.01f) continue;
    float ex=0,ey=0;
    switch(vp[i].idx){
      case 0: ex=px;ey=py;break;
      case 1: ex=Kokkos::sin(px);ey=Kokkos::sin(py);break;
      case 2: ex=px/rad2;ey=py/rad2;break;
      case 5: ex=theta/M_PI;ey=rad-1.f;break;
      case 6: ex=rad*Kokkos::sin(theta+rad);ey=rad*Kokkos::cos(theta-rad);break;
      case 7: ex=rad*Kokkos::sin(theta*rad);ey=-rad*Kokkos::cos(theta*rad);break;
      case 9: ex=rad*(Kokkos::cos(theta)+Kokkos::sin(rad));ey=rad*(Kokkos::sin(theta)-Kokkos::cos(rad));break;
      case 13:{float sr=Kokkos::sqrt(rad),om=(random_numbers[thread_idx&(NUM_RANDOMS-1)]>0.5f)?0.f:M_PI;
               ex=sr*Kokkos::cos(theta/2+om);ey=sr*Kokkos::sin(theta/2+om);break;}
      default: ex=px;ey=py;break;
    }
    ox+=vp[i].factor*ex; oy+=vp[i].factor*ey;
  }
  px=ox; py=oy;
  affine_transform(px,py,&params.post_transform_params[func_idx][0]);
  color=.5f*(color+params.function_colors[func_idx]);
}

// ─── kernels ─────────────────────────────────────────────────────────────────

struct KernelInitialize {
  Kokkos::View<float*>    short_px, short_py;
  Kokkos::View<float*>    colors;
  Kokkos::View<unsigned short*> perms;
  int perm_num;
  Kokkos::View<float*>    start_px, start_py;
  Kokkos::View<float*>    rn;
  Kokkos::View<ConstMemParams*> p_params;

  KOKKOS_INLINE_FUNCTION void operator()(int idx) const {
    const ConstMemParams& params=p_params(0);
    int func=get_function_idx(idx,params);
    int perm_idx=perms(NUM_THREADS*perm_num+idx);
    float px=start_px(perm_idx), py=start_py(perm_idx);
    float color=0.5f;
    iteration_fractal_flame(px,py,color,func,idx,rn.data(),params);
    short_px(idx)=px; short_py(idx)=py; colors(idx)=color;
  }
};

struct KernelIterate {
  Kokkos::View<float*>    short_px, short_py, colors;
  Kokkos::View<unsigned short*> perms;
  int perm_num;
  Kokkos::View<float*>    rn;
  Kokkos::View<ConstMemParams*> p_params;

  KOKKOS_INLINE_FUNCTION void operator()(int idx) const {
    const ConstMemParams& params=p_params(0);
    int func=get_function_idx(idx,params);
    int perm_idx=perms(NUM_THREADS*perm_num+idx);
    float px=short_px(perm_idx), py=short_py(perm_idx);
    float color=colors(perm_idx);
    iteration_fractal_flame(px,py,color,func,idx,rn.data(),params);
    short_px(idx)=px; short_py(idx)=py; colors(idx)=color;
  }
};

struct KernelGenerate {
  Kokkos::View<float*>    vertices_x, vertices_y, vertices_c;
  Kokkos::View<float*>    short_px, short_py, colors;
  Kokkos::View<unsigned short*> perms;
  int perm_num;
  Kokkos::View<float*>    rn;
  Kokkos::View<ConstMemParams*> p_params;

  KOKKOS_INLINE_FUNCTION void operator()(int idx) const {
    const ConstMemParams& params=p_params(0);
    int func=get_function_idx(idx,params);
    int perm_idx=perms(NUM_THREADS*perm_num+idx);
    float px=short_px(perm_idx), py=short_py(perm_idx);
    float color=colors(perm_idx);
    for(int i=0;i<NUM_POINTS_PER_THREAD;i++){
      perm_idx=perms(NUM_THREADS*((perm_num+i)%NUM_PERMUTATIONS)+idx);
      iteration_fractal_flame(px,py,color,func,idx,rn.data(),params);
      vertices_x(idx*NUM_POINTS_PER_THREAD+i)=px;
      vertices_y(idx*NUM_POINTS_PER_THREAD+i)=py;
      vertices_c(idx*NUM_POINTS_PER_THREAD+i)=color;
    }
  }
};

int main(int argc, char** argv)
{
  if(argc!=2){ printf("Usage: %s <repeat>\n",argv[0]); return 1; }
  const int repeat=atoi(argv[1]);

  ConstMemParams cmp; float function_weights[NUM_FUNCTIONS];
  memset(&cmp,0,sizeof cmp);
  for(int i=0;i<NUM_FUNCTIONS;i++){
    function_weights[i]=0.f;
    cmp.variation_parameters[i][0].factor=1.f;
    cmp.pre_transform_params[i][0]=1.f; cmp.pre_transform_params[i][4]=1.f;
    cmp.post_transform_params[i][0]=1.f; cmp.post_transform_params[i][4]=1.f;
  }
  function_weights[0]=1.f;
  cmp.pre_transform_params[0][0]=1.4f; cmp.pre_transform_params[0][1]=0.6f;
  cmp.pre_transform_params[1][2]=0.4f;
  cmp.pre_transform_params[2][0]=0.3f; cmp.pre_transform_params[2][2]=1.f;
  cmp.function_colors[0]=1.f; cmp.function_colors[2]=0.4f;
  function_weights[1]=0.5f; function_weights[2]=0.6f;
  cmp.enable_sierpinski=1;
  cmp.variation_parameters[2][0].idx=9; cmp.variation_parameters[0][0].idx=13;
  cmp.variation_parameters[1][0].idx=5; cmp.variation_parameters[1][1].idx=7;
  cmp.variation_parameters[2][0].factor=1.f; cmp.variation_parameters[0][0].factor=1.f;
  cmp.variation_parameters[1][0].factor=1.f; cmp.variation_parameters[1][1].factor=-.3f;
  cmp.pre_transform_params[2][2]=-1.f; cmp.pre_transform_params[2][5]=-1.f;

  srand(2);
  unsigned ms[624]; for(int i=0;i<624;i++) ms[i]=rand();
  for(int i=0;i<10000;i++) mersenne_twister(ms);

  float* rn_tmp=new float[NUM_RANDOMS];
  for(int i=0;i<NUM_RANDOMS;i++) rn_tmp[i]=mersenne_twister(ms)*(1.f/4294967296.f);

  PermSortElement* to_sort=new PermSortElement[NUM_THREADS];
  unsigned short* perm_data=new unsigned short[NUM_THREADS*NUM_PERMUTATIONS];
  for(int i=0;i<NUM_PERMUTATIONS;i++){
    for(int j=0;j<NUM_THREADS;j++){ to_sort[j].value=mersenne_twister(ms); to_sort[j].idx=j; }
    std::sort(to_sort,to_sort+NUM_THREADS);
    for(int j=0;j<NUM_THREADS;j++) perm_data[i*NUM_THREADS+j]=to_sort[j].idx;
  }
  delete[] to_sort;

  float2* pts_tmp=new float2[NUM_THREADS];
  for(int i=0;i<NUM_THREADS;i++){
    pts_tmp[i].x=(float(i)/NUM_THREADS-.5f)*2.f;
    pts_tmp[i].y=(radical_inverse(i,2)-.5f)*2.f;
  }

  Kokkos::initialize(argc,argv);
  {
    Kokkos::View<float*>  d_rn    ("rn",    NUM_RANDOMS);
    Kokkos::View<unsigned short*> d_perms("perms", NUM_THREADS*NUM_PERMUTATIONS);
    Kokkos::View<float*>  d_spx   ("spx",   NUM_THREADS);
    Kokkos::View<float*>  d_spy   ("spy",   NUM_THREADS);
    Kokkos::View<float*>  d_col   ("col",   NUM_THREADS);
    Kokkos::View<float*>  d_stpx  ("stpx",  NUM_THREADS);
    Kokkos::View<float*>  d_stpy  ("stpy",  NUM_THREADS);
    Kokkos::View<float*>  d_vx    ("vx",    NUM_THREADS*NUM_POINTS_PER_THREAD);
    Kokkos::View<float*>  d_vy    ("vy",    NUM_THREADS*NUM_POINTS_PER_THREAD);
    Kokkos::View<float*>  d_vc    ("vc",    NUM_THREADS*NUM_POINTS_PER_THREAD);
    Kokkos::View<ConstMemParams*> d_par("params",1);

    {
      auto hrn=Kokkos::create_mirror_view(d_rn);
      for(int i=0;i<NUM_RANDOMS;i++) hrn(i)=rn_tmp[i];
      Kokkos::deep_copy(d_rn,hrn);
    }
    {
      auto hp=Kokkos::create_mirror_view(d_perms);
      for(int i=0;i<NUM_THREADS*NUM_PERMUTATIONS;i++) hp(i)=perm_data[i];
      Kokkos::deep_copy(d_perms,hp);
    }
    {
      auto hsp=Kokkos::create_mirror_view(d_stpx);
      auto hspy=Kokkos::create_mirror_view(d_stpy);
      for(int i=0;i<NUM_THREADS;i++){ hsp(i)=pts_tmp[i].x; hspy(i)=pts_tmp[i].y; }
      Kokkos::deep_copy(d_stpx,hsp); Kokkos::deep_copy(d_stpy,hspy);
    }
    {
      auto hpar=Kokkos::create_mirror_view(d_par);
      hpar(0)=cmp;
      Kokkos::deep_copy(d_par,hpar);
    }

    delete[] rn_tmp; delete[] perm_data; delete[] pts_tmp;

    struct timeval tv,tv2;
    gettimeofday(&tv,NULL);

    int perm_pos=0;
    for(int n=0;n<repeat;n++){
      // Recompute thread_function_mapping
      float sum=0; for(int i=0;i<NUM_FUNCTIONS;i++) sum+=function_weights[i];
      int ntsum=0;
      for(int i=0;i<NUM_FUNCTIONS;i++){
        int nt=(int)((function_weights[i]/sum)*NUM_THREADS);
        cmp.thread_function_mapping[i]=nt; ntsum+=nt;
      }
      cmp.thread_function_mapping[0]+=NUM_THREADS-ntsum;
      for(int i=1;i<NUM_FUNCTIONS;i++) cmp.thread_function_mapping[i]+=cmp.thread_function_mapping[i-1];
      {auto hpar=Kokkos::create_mirror_view(d_par); hpar(0)=cmp; Kokkos::deep_copy(d_par,hpar);}

      Kokkos::parallel_for(NUM_THREADS, KernelInitialize{d_spx,d_spy,d_col,d_perms,perm_pos++,d_stpx,d_stpy,d_rn,d_par});
      Kokkos::fence();

      for(int i=0;i<NUM_ITERATIONS;i++){
        Kokkos::parallel_for(NUM_THREADS, KernelIterate{d_spx,d_spy,d_col,d_perms,perm_pos++,d_rn,d_par});
        perm_pos%=NUM_PERMUTATIONS;
        Kokkos::fence();
      }

      Kokkos::parallel_for(NUM_THREADS, KernelGenerate{d_vx,d_vy,d_vc,d_spx,d_spy,d_col,d_perms,perm_pos++,d_rn,d_par});
      Kokkos::fence();

      perm_pos+=NUM_POINTS_PER_THREAD-1;
      perm_pos%=NUM_PERMUTATIONS;
    }

    gettimeofday(&tv2,NULL);
    float frametime=(tv2.tv_sec-tv.tv_sec)*1000000.f+tv2.tv_usec-tv.tv_usec;
    printf("Total frame time is %.1f us\n",frametime);

    auto hvx=Kokkos::create_mirror_view(d_vx);
    Kokkos::deep_copy(hvx,d_vx);
    float sum=0; for(int i=0;i<NUM_THREADS*NUM_POINTS_PER_THREAD;i++) sum+=hvx(i);
    printf("Vertex x checksum: %f\n",sum/(NUM_THREADS*NUM_POINTS_PER_THREAD));
  }
  Kokkos::finalize();
  return 0;
}
