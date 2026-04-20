#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <Kokkos_Core.hpp>

// Include data from OMP source directory
#include "../bn-omp/data45.h"

const int HIGHEST = 3;
const int ITER = 100;
const int WORKLOAD = 1;

// --- Device helper functions ---
KOKKOS_INLINE_FUNCTION void Dincr(int *bit, int n) {
  while(n<=NODE_N){ bit[n]++; if(bit[n]>=2){ bit[n]=0; n++; } else break; }
}
KOKKOS_INLINE_FUNCTION void DincrS(int *bit, int n) {
  bit[n]++; if(bit[n]>=STATE_N){ bit[n]=0; Dincr(bit,n+1); }
}
KOKKOS_INLINE_FUNCTION bool D_getState(int parN, int *sta, int time) {
  int i,j=1;
  for(i=0;i<parN;i++) j*=STATE_N;
  j--;
  if(time>j) return false;
  if(time>=1) DincrS(sta,0);
  return true;
}
KOKKOS_INLINE_FUNCTION int D_C(int n, int a) {
  int i,res=1,atmp=a;
  for(i=0;i<atmp;i++){ res*=n; n--; }
  for(i=0;i<atmp;i++){ res/=a; a--; }
  return res;
}
KOKKOS_INLINE_FUNCTION void D_findComb(int *comb, int l, int n) {
  const int len=4;
  if(l==0){ for(int i=0;i<len;i++) comb[i]=-1; return; }
  int sum=0, k=1;
  while(sum<l) sum+=D_C(n,k++);
  l-=sum-D_C(n,--k);
  int low=0, pos=0;
  while(k>1) {
    sum=0; int s=1;
    while(sum<l) sum+=D_C(n-s++,k-1);
    l-=sum-D_C(n-(--s),--k);
    low+=s; comb[pos++]=low; n-=s;
  }
  comb[pos]=low+l;
  for(int i=pos+1;i<4;i++) comb[i]=-1;
}
KOKKOS_INLINE_FUNCTION int D_findindex(int *arr, int size) {
  int i,j,index=0;
  for(i=1;i<size;i++) index+=D_C(NODE_N-1,i);
  for(i=1;i<=size-1;i++)
    for(j=arr[i-1]+1;j<=arr[i]-1;j++)
      index+=D_C(NODE_N-1-j,size-i);
  index+=arr[size]-arr[size-1];
  return index;
}

// Host helper functions
static float logGamma(int n) {
  float x=0; for(int i=2;i<=n-1;i++) x+=logf((float)i); return x;
}
static float *LG=nullptr;
static void Pre_logGamma() {
  LG=(float*)malloc((DATA_N+2)*sizeof(float));
  for(int i=0;i<DATA_N+2;i++) LG[i]=logGamma(i);
}
static int sizepernode;
static void initial() {
  int n=1; for(int i=0;i<4;i++) n+=D_C(NODE_N-1,i+1);
  sizepernode=n;
}
static float *localscore=nullptr;

int main(int argc, char **argv) {
  if(argc!=3){ printf("Usage: %s <output_file> <repeat>\n", argv[0]); return 1; }
  const int repeat=atoi(argv[2]);

  srand(2);
  initial();
  Pre_logGamma();

  localscore=(float*)malloc(NODE_N*sizepernode*sizeof(float));

  printf("NODE_N=%d, sizepernode=%d\n", NODE_N, sizepernode);

  Kokkos::initialize(argc, argv);
  {
    using ViewF = Kokkos::View<float*>;
    using ViewI = Kokkos::View<int*>;
    using ScratchF = Kokkos::View<float*, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                                   Kokkos::MemoryUnmanaged>;

    ViewF d_localscore("localscore", NODE_N*sizepernode);
    ViewI d_data("data", NODE_N*DATA_N);
    ViewF d_LG("LG", DATA_N+2);

    {
      auto hd=Kokkos::create_mirror_view(d_data);
      auto hl=Kokkos::create_mirror_view(d_LG);
      for(int i=0;i<NODE_N*DATA_N;i++) hd(i)=data[i];
      for(int i=0;i<DATA_N+2;i++) hl(i)=LG[i];
      Kokkos::deep_copy(d_data,hd);
      Kokkos::deep_copy(d_LG,hl);
    }

    const int spn = sizepernode;

    auto t0=std::chrono::steady_clock::now();
    for(int rep=0;rep<repeat;rep++) {
      // genScoreKernel: initialize to zero
      Kokkos::parallel_for("genScore_init", NODE_N*spn, KOKKOS_LAMBDA(int id) {
        d_localscore[id]=0.f;
      });

      // genScoreKernel: compute scores
      Kokkos::parallel_for("genScore", spn, KOKKOS_LAMBDA(int id) {
        int node, index;
        int parent[5]={0};
        int pre[NODE_N]={0};
        int state[5]={0};
        int parN=0, tmp;
        int Nij[STATE_N]={0};
        float ls=0;

        D_findComb(parent, id, NODE_N-1);
        for(int i=0;i<4;i++) if(parent[i]>0) parN++;

        for(node=0;node<NODE_N;node++) {
          int j=1;
          for(int i=0;i<NODE_N;i++) if(i!=node) pre[j++]=i;
          for(int t=0;t<parN;t++) state[t]=0;
          index=spn*node+id;
          int t=0;
          while(D_getState(parN,state,t++)) {
            ls=0;
            for(int t1=0;t1<STATE_N;t1++) Nij[t1]=0;
            for(int t1=0;t1<DATA_N;t1++) {
              bool flag=true;
              for(int t2=0;t2<parN;t2++) {
                int p=parent[t2]<(NODE_N-1)?pre[parent[t2]+1]:pre[NODE_N-1];
                if(d_data[p*DATA_N+t1]!=state[t2]) { flag=false; break; }
              }
              if(flag) Nij[d_data[node*DATA_N+t1]]++;
            }
            ls+=d_LG[STATE_N-1];
            tmp=0; for(int t1=0;t1<STATE_N;t1++) tmp+=Nij[t1];
            ls-=d_LG[tmp];
            for(int t1=0;t1<STATE_N;t1++) ls+=d_LG[Nij[t1]];
            d_localscore[index]+=ls;
          }
        }
      });
    }
    Kokkos::fence();
    auto t1=std::chrono::steady_clock::now();
    printf("Average execution time of genScoreKernel: %f (s)\n",
      std::chrono::duration<double>(t1-t0).count()/repeat);

    auto hl=Kokkos::create_mirror_view(d_localscore);
    Kokkos::deep_copy(hl,d_localscore);
    for(int i=0;i<NODE_N*sizepernode;i++) localscore[i]=hl(i);
  }
  Kokkos::finalize();

  free(localscore); free(LG);
  return 0;
}
