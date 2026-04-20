#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <Kokkos_Core.hpp>

#define MIN(x,y) ((x)<(y)?(x):(y))
#define MAX(x,y) ((x)>(y)?(x):(y))
#ifndef MAX_MEM
#define MAX_MEM 4
#endif

void ccsd_tengy_kokkos(
    const double *f1n, const double *f1t,
    const double *f2n, const double *f2t,
    const double *f3n, const double *f3t,
    const double *f4n, const double *f4t,
    const double *dintc1, const double *dintx1, const double *t1v1,
    const double *dintc2, const double *dintx2, const double *t1v2,
    const double *eorb,  double eaijk,
    double *emp4i_, double *emp5i_,
    double *emp4k_, double *emp5k_,
    const int ncor, const int nocc, const int nvir)
{
  const int nv2 = nvir*nvir;
  Kokkos::View<double*> df1n("f1n",nv2), df1t("f1t",nv2),
                        df2n("f2n",nv2), df2t("f2t",nv2),
                        df3n("f3n",nv2), df3t("f3t",nv2),
                        df4n("f4n",nv2), df4t("f4t",nv2),
                        ddintc1("dc1",nvir), ddintx1("dx1",nvir), dt1v1("t1v1",nvir),
                        ddintc2("dc2",nvir), ddintx2("dx2",nvir), dt1v2("t1v2",nvir),
                        deorb("eorb", ncor+nocc+nvir);

  auto copy_to = [&](Kokkos::View<double*> &d, const double *h, int n) {
    auto m = Kokkos::create_mirror_view(d);
    for (int i=0;i<n;i++) m(i)=h[i];
    Kokkos::deep_copy(d,m);
  };
  copy_to(df1n,f1n,nv2); copy_to(df1t,f1t,nv2);
  copy_to(df2n,f2n,nv2); copy_to(df2t,f2t,nv2);
  copy_to(df3n,f3n,nv2); copy_to(df3t,f3t,nv2);
  copy_to(df4n,f4n,nv2); copy_to(df4t,f4t,nv2);
  copy_to(ddintc1,dintc1,nvir); copy_to(ddintx1,dintx1,nvir); copy_to(dt1v1,t1v1,nvir);
  copy_to(ddintc2,dintc2,nvir); copy_to(ddintx2,dintx2,nvir); copy_to(dt1v2,t1v2,nvir);
  copy_to(deorb,eorb,ncor+nocc+nvir);

  Kokkos::View<double[1]> d_emp4i("emp4i"),d_emp5i("emp5i"),
                          d_emp4k("emp4k"),d_emp5k("emp5k");
  Kokkos::deep_copy(d_emp4i,0.0); Kokkos::deep_copy(d_emp5i,0.0);
  Kokkos::deep_copy(d_emp4k,0.0); Kokkos::deep_copy(d_emp5k,0.0);

  using MDpolicy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
  Kokkos::parallel_for("ccsd_tengy", MDpolicy({0,0},{nvir,nvir}),
    KOKKOS_LAMBDA(int b, int c) {
      const double denom = -1.0 / (deorb(ncor+nocc+b) + deorb(ncor+nocc+c) + eaijk);
      const int bc = b+c*nvir, cb = c+b*nvir;
      const double f1nbc=df1n(bc),f1tbc=df1t(bc),f1ncb=df1n(cb),f1tcb=df1t(cb);
      const double f2nbc=df2n(bc),f2tbc=df2t(bc),f2ncb=df2n(cb),f2tcb=df2t(cb);
      const double f3nbc=df3n(bc),f3tbc=df3t(bc),f3ncb=df3n(cb),f3tcb=df3t(cb);
      const double f4nbc=df4n(bc),f4tbc=df4t(bc),f4ncb=df4n(cb),f4tcb=df4t(cb);

      double de4i = denom*(f1tbc+f1ncb+f2tcb+f3nbc+f4ncb)*(f1tbc-f2tbc*2-f3tbc*2+f4tbc)
        -denom*(f1nbc+f1tcb+f2ncb+f3ncb)*(f1tbc*2-f2tbc-f3tbc+f4tbc*2)
        +denom*3*(f1nbc*(f1nbc+f3ncb+f4tcb*2)+f2nbc*f2tcb+f3nbc*f4tbc);
      Kokkos::atomic_add(&d_emp4i(0), de4i);

      double de4k = denom*(f1nbc+f1tcb+f2ncb+f3tbc+f4tcb)*(f1nbc-f2nbc*2-f3nbc*2+f4nbc)
        -denom*(f1tbc+f1ncb+f2tcb+f3tcb)*(f1nbc*2-f2nbc-f3nbc+f4nbc*2)
        +denom*3*(f1tbc*(f1tbc+f3tcb+f4ncb*2)+f2tbc*f2ncb+f3tbc*f4nbc);
      Kokkos::atomic_add(&d_emp4k(0), de4k);

      double t1v1b=dt1v1(b), t1v2b=dt1v2(b);
      double dintx1c=ddintx1(c),dintx2c=ddintx2(c),dintc1c=ddintc1(c),dintc2c=ddintc2(c);

      double de5i = denom*t1v1b*dintx1c*(f1tbc+f2nbc+f4ncb-(f3tbc+f4nbc+f2ncb+f1nbc+f2tbc+f3ncb)*2
          +(f3nbc+f4tbc+f1ncb)*4)+denom*t1v1b*dintc1c*(f1nbc+f4nbc+f1tcb-(f2nbc+f3nbc+f2tcb)*2);
      Kokkos::atomic_add(&d_emp5i(0), de5i);

      double de5k = denom*t1v2b*dintx2c*(f1nbc+f2tbc+f4tcb-(f3nbc+f4tbc+f2tcb+f1tbc+f2nbc+f3tcb)*2
          +(f3tbc+f4nbc+f1tcb)*4)+denom*t1v2b*dintc2c*(f1tbc+f4tbc+f1ncb-(f2tbc+f3tbc+f2ncb)*2);
      Kokkos::atomic_add(&d_emp5k(0), de5k);
    });

  Kokkos::fence();
  auto he4i=Kokkos::create_mirror_view(d_emp4i); Kokkos::deep_copy(he4i,d_emp4i);
  auto he5i=Kokkos::create_mirror_view(d_emp5i); Kokkos::deep_copy(he5i,d_emp5i);
  auto he4k=Kokkos::create_mirror_view(d_emp4k); Kokkos::deep_copy(he4k,d_emp4k);
  auto he5k=Kokkos::create_mirror_view(d_emp5k); Kokkos::deep_copy(he5k,d_emp5k);
  *emp4i_=he4i(0); *emp5i_=he5i(0); *emp4k_=he4k(0); *emp5k_=he5k(0);
}

static double *make_array(int n) {
  double *a = (double*) malloc(n*sizeof(double));
  for (int i=0;i<n;i++) a[i]=drand48();
  return a;
}

int main(int argc, char *argv[])
{
  if (argc<3) { printf("Usage: ./main nocc nvir [maxiter] [nkpass]\n"); return argc; }
  int ncor=0, nocc=atoi(argv[1]), nvir=atoi(argv[2]);
  int maxiter=100, nkpass=1;
  if (argc>3) maxiter=atoi(argv[3]);
  if (argc>4) nkpass=atoi(argv[4]);
  if (nocc<1||nvir<1) { printf("Arguments must be non-negative!\n"); return 1; }

  const int nbf=ncor+nocc+nvir;
  const int lnvv=nvir*nvir, lnov=nocc*nvir;
  const int kchunk=(nocc-1)/nkpass+1;

  printf("Test driver for ccsd with nocc=%d, nvir=%d\n",nocc,nvir);

  srand48(2);
  double *eorb=make_array(nbf);
  double *f1n=make_array(lnvv),*f2n=make_array(lnvv),*f3n=make_array(lnvv),*f4n=make_array(lnvv);
  double *f1t=make_array(lnvv),*f2t=make_array(lnvv),*f3t=make_array(lnvv),*f4t=make_array(lnvv);
  double *dintc1=make_array(lnvv),*dintc2=make_array(lnvv);
  double *dintx1=make_array(lnvv),*dintx2=make_array(lnvv);
  double *t1v1=make_array(lnvv),*t1v2=make_array(lnvv);
  int ntimers=MIN(maxiter,nocc*nocc*nocc*nocc);
  double *timers=(double*)calloc(ntimers,sizeof(double));
  double emp4=0.0,emp5=0.0;

  Kokkos::initialize(argc,argv);
  {
    int iter=0;
    for (int klo=1; klo<=nocc; klo+=kchunk) {
      int khi=MIN(nocc,klo+kchunk-1);
      int a=1;
      for (int j=1;j<=nocc;j++) {
        for (int i=1;i<=nocc;i++) {
          for (int k=klo;k<=MIN(khi,i);k++) {
            double emp4i=0,emp5i=0,emp4k=0,emp5k=0;
            const double eaijk=eorb[a-1]-(eorb[ncor+i-1]+eorb[ncor+j-1]+eorb[ncor+k-1]);

            clock_t t0=clock();
            ccsd_tengy_kokkos(f1n,f1t,f2n,f2t,f3n,f3t,f4n,f4t,
                              dintc1,dintx1,t1v1,dintc2,dintx2,t1v2,
                              eorb,eaijk,&emp4i,&emp5i,&emp4k,&emp5k,
                              ncor,nocc,nvir);
            timers[iter]=(double)(clock()-t0)/CLOCKS_PER_SEC;

            emp4+=emp4i; emp5+=emp5i;
            if (i!=k) { emp4+=emp4k; emp5+=emp5k; }

            iter++;
            if (iter==maxiter) { printf("Stopping after %d iterations\n",iter); goto maxed_out; }
            if (emp4>1000.0) emp4-=1000.0; if (emp4<-1000.0) emp4+=1000.0;
            if (emp5>1000.0) emp5-=1000.0; if (emp5<-1000.0) emp5+=1000.0;
          }
        }
      }
    }
  }
maxed_out:
  Kokkos::finalize();
  {
    double tsum=0,tmax=-1e10,tmin=1e10;
    int iter=MIN(maxiter,ntimers);
    for (int i=0;i<iter;i++) {
      tsum+=timers[i]; tmax=MAX(tmax,timers[i]); tmin=MIN(tmin,timers[i]);
    }
    printf("TIMING: min=%lf, max=%lf, avg=%lf\n",tmin,tmax,tsum/iter);
    printf("emp4=%f emp5=%f\n",emp4,emp5);
    printf("Finished\n");
  }
  free(eorb);free(f1n);free(f2n);free(f3n);free(f4n);
  free(f1t);free(f2t);free(f3t);free(f4t);
  free(dintc1);free(dintc2);free(dintx1);free(dintx2);
  free(t1v1);free(t1v2);free(timers);
  return 0;
}
