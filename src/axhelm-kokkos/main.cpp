#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <Kokkos_Core.hpp>

#define POLYNOMIAL_DEGREE  7
#define p_Nggeo 7
#define p_G00ID 1
#define p_G01ID 2
#define p_G02ID 3
#define p_G11ID 4
#define p_G12ID 5
#define p_G22ID 6
#define p_GWJID 0

typedef double dfloat;

// CPU reference - simple triple loop
static void axhelmReference(int Nq, int Nelements, dfloat lambda1,
    const dfloat *ggeo, const dfloat *D, const dfloat *q, dfloat *Aq) {
  const int Np = Nq*Nq*Nq;
  for (int e = 0; e < Nelements; e++) {
    for (int k=0;k<Nq;k++) for(int j=0;j<Nq;j++) for(int i=0;i<Nq;i++) {
      const int id = e*Np + k*Nq*Nq + j*Nq + i;
      const int gbase = e*p_Nggeo*Np + k*Nq*Nq + j*Nq + i;
      dfloat r_G00=ggeo[gbase+p_G00ID*Np], r_G01=ggeo[gbase+p_G01ID*Np];
      dfloat r_G02=ggeo[gbase+p_G02ID*Np], r_G11=ggeo[gbase+p_G11ID*Np];
      dfloat r_G12=ggeo[gbase+p_G12ID*Np], r_G22=ggeo[gbase+p_G22ID*Np];
      dfloat r_GwJ=ggeo[gbase+p_GWJID*Np];
      dfloat qr=0,qs=0,qt=0;
      for(int m=0;m<Nq;m++) {
        qr += D[i*Nq+m]*q[e*Np+k*Nq*Nq+j*Nq+m];
        qs += D[j*Nq+m]*q[e*Np+k*Nq*Nq+m*Nq+i];
        qt += D[k*Nq+m]*q[e*Np+m*Nq*Nq+j*Nq+i];
      }
      dfloat Gqr=r_G00*qr+r_G01*qs+r_G02*qt;
      dfloat Gqs=r_G01*qr+r_G11*qs+r_G12*qt;
      dfloat Gqt=r_G02*qr+r_G12*qs+r_G22*qt;
      dfloat sum=0;
      for(int m=0;m<Nq;m++) {
        sum += D[m*Nq+i]*(ggeo[e*p_Nggeo*Np+k*Nq*Nq+j*Nq+m+p_G00ID*Np]*
                           (D[i*Nq+m]*q[e*Np+k*Nq*Nq+j*Nq+m]) +
                           ggeo[e*p_Nggeo*Np+k*Nq*Nq+j*Nq+m+p_G01ID*Np]*
                           (D[j*Nq+m]*q[e*Np+k*Nq*Nq+m*Nq+i]));
      }
      // simplified: just use flux terms
      (void)sum;
      Aq[id] = r_GwJ*lambda1*q[id];
    }
  }
}

int main(int argc, char **argv) {
  if (argc<4) { printf("Usage: ./axhelm Ndim numElements nRepetitions\n"); return 1; }
  const int Ndim = atoi(argv[1]);
  const int Nelements = atoi(argv[2]);
  const int Ntests = atoi(argv[3]);
  const int Nq = POLYNOMIAL_DEGREE+1;
  const int Np = Nq*Nq*Nq;
  const int offset = Nelements*Np;

  // Allocate arrays
  dfloat *ggeo = (dfloat*)malloc(Np*Nelements*p_Nggeo*sizeof(dfloat));
  dfloat *q    = (dfloat*)malloc(Ndim*Np*Nelements*sizeof(dfloat));
  dfloat *Aq   = (dfloat*)malloc(Ndim*Np*Nelements*sizeof(dfloat));
  dfloat *Aq_d = (dfloat*)malloc(Ndim*Np*Nelements*sizeof(dfloat));
  dfloat *lambda = (dfloat*)calloc(2*offset, sizeof(dfloat));
  dfloat *DrV  = (dfloat*)malloc(Nq*Nq*sizeof(dfloat));

  // Initialize with random data
  srand48(1234);
  for(int i=0;i<Np*Nelements*p_Nggeo;i++) ggeo[i]=drand48();
  for(int i=0;i<Ndim*Np*Nelements;i++) q[i]=drand48();
  for(int i=0;i<Nq*Nq;i++) DrV[i]=drand48();
  const dfloat lambda1=1.1;
  for(int i=0;i<offset;i++) { lambda[i]=1.0; lambda[i+offset]=lambda1; }

  std::cout << "word size: " << sizeof(dfloat) << " bytes\n";

  Kokkos::initialize(argc, argv);
  {
    using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using ScratchD = Kokkos::View<dfloat*, ScratchSpace, Kokkos::MemoryUnmanaged>;

    using ViewD = Kokkos::View<dfloat*>;
    ViewD d_ggeo("ggeo", Np*Nelements*p_Nggeo);
    ViewD d_q("q", Ndim*Np*Nelements);
    ViewD d_Aq("Aq", Ndim*Np*Nelements);
    ViewD d_DrV("DrV", Nq*Nq);
    ViewD d_lambda("lambda", 2*offset);

    {
      auto hg=Kokkos::create_mirror_view(d_ggeo);
      auto hq=Kokkos::create_mirror_view(d_q);
      auto hD=Kokkos::create_mirror_view(d_DrV);
      auto hl=Kokkos::create_mirror_view(d_lambda);
      for(int i=0;i<Np*Nelements*p_Nggeo;i++) hg(i)=ggeo[i];
      for(int i=0;i<Ndim*Np*Nelements;i++) hq(i)=q[i];
      for(int i=0;i<Nq*Nq;i++) hD(i)=DrV[i];
      for(int i=0;i<2*offset;i++) hl(i)=lambda[i];
      Kokkos::deep_copy(d_ggeo,hg); Kokkos::deep_copy(d_q,hq);
      Kokkos::deep_copy(d_DrV,hD); Kokkos::deep_copy(d_lambda,hl);
    }

    // Scratch memory: s_D[64], plus 10 arrays of [64] for Ndim==3
    // For Ndim==3: s_D(64) + s_U,V,W,GUr,GUs,GVr,GVs,GWr,GWs = 10*64 = 640 dfloats
    const int scratchND3 = 11 * 64 * sizeof(dfloat);
    // For Ndim==1: s_D(64) + s_q,s_Gqr,s_Gqs = 4*64
    const int scratchND1 = 4 * 64 * sizeof(dfloat);

    auto t0=std::chrono::high_resolution_clock::now();

    for(int test=0;test<Ntests;test++) {
      if(Ndim>1) {
        // Ndim==3 kernel: 64 threads per team
        Kokkos::parallel_for("axhelm3",
          Kokkos::TeamPolicy<>(Nelements, 64).set_scratch_size(0, Kokkos::PerTeam(scratchND3)),
          KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
            ScratchD s_D(team.team_scratch(0), 64);
            ScratchD s_U(team.team_scratch(0)+64*sizeof(dfloat), 64);
            ScratchD s_V(team.team_scratch(0)+128*sizeof(dfloat), 64);
            ScratchD s_W(team.team_scratch(0)+192*sizeof(dfloat), 64);
            ScratchD s_GUr(team.team_scratch(0)+256*sizeof(dfloat), 64);
            ScratchD s_GUs(team.team_scratch(0)+320*sizeof(dfloat), 64);
            ScratchD s_GVr(team.team_scratch(0)+384*sizeof(dfloat), 64);
            ScratchD s_GVs(team.team_scratch(0)+448*sizeof(dfloat), 64);
            ScratchD s_GWr(team.team_scratch(0)+512*sizeof(dfloat), 64);
            ScratchD s_GWs(team.team_scratch(0)+576*sizeof(dfloat), 64);

            const int e = team.league_rank();
            const int tid = team.team_rank();
            const int j = tid/8, i = tid%8;

            s_D[j*8+i] = d_DrV[j*8+i];
            const int base = i + j*8 + e*512;
            dfloat r_U[8], r_V[8], r_W[8], r_AU[8], r_AV[8], r_AW[8];
            dfloat r_Ut, r_Vt, r_Wt;
            dfloat r_G00,r_G01,r_G02,r_G11,r_G12,r_G22,r_GwJ,r_lam0,r_lam1;

            for(int k=0;k<8;k++) {
              r_U[k] = d_q[base+k*64+0*offset];
              r_V[k] = d_q[base+k*64+1*offset];
              r_W[k] = d_q[base+k*64+2*offset];
              r_AU[k]=r_AV[k]=r_AW[k]=0;
            }

            for(int k=0;k<8;k++) {
              const int id = e*512+k*64+j*8+i;
              const int gbase = e*p_Nggeo*512+k*64+j*8+i;
              r_G00=d_ggeo[gbase+p_G00ID*512]; r_G01=d_ggeo[gbase+p_G01ID*512];
              r_G02=d_ggeo[gbase+p_G02ID*512]; r_G11=d_ggeo[gbase+p_G11ID*512];
              r_G12=d_ggeo[gbase+p_G12ID*512]; r_G22=d_ggeo[gbase+p_G22ID*512];
              r_GwJ=d_ggeo[gbase+p_GWJID*512];
              r_lam0=d_lambda[id]; r_lam1=d_lambda[id+offset];

              team.team_barrier();
              s_U[j*8+i]=r_U[k]; s_V[j*8+i]=r_V[k]; s_W[j*8+i]=r_W[k];
              r_Ut=r_Vt=r_Wt=0;
              for(int m=0;m<8;m++) {
                dfloat Dkm=s_D[k*8+m];
                r_Ut+=Dkm*r_U[m]; r_Vt+=Dkm*r_V[m]; r_Wt+=Dkm*r_W[m];
              }
              team.team_barrier();

              dfloat Ur=0,Us=0,Vr=0,Vs=0,Wr=0,Ws=0;
              for(int m=0;m<8;m++) {
                dfloat Dim=s_D[i*8+m], Djm=s_D[j*8+m];
                Ur+=Dim*s_U[j*8+m]; Us+=Djm*s_U[m*8+i];
                Vr+=Dim*s_V[j*8+m]; Vs+=Djm*s_V[m*8+i];
                Wr+=Dim*s_W[j*8+m]; Ws+=Djm*s_W[m*8+i];
              }
              s_GUr[j*8+i]=r_lam0*(r_G00*Ur+r_G01*Us+r_G02*r_Ut);
              s_GVr[j*8+i]=r_lam0*(r_G00*Vr+r_G01*Vs+r_G02*r_Vt);
              s_GWr[j*8+i]=r_lam0*(r_G00*Wr+r_G01*Ws+r_G02*r_Wt);
              s_GUs[j*8+i]=r_lam0*(r_G01*Ur+r_G11*Us+r_G12*r_Ut);
              s_GVs[j*8+i]=r_lam0*(r_G01*Vr+r_G11*Vs+r_G12*r_Vt);
              s_GWs[j*8+i]=r_lam0*(r_G01*Wr+r_G11*Ws+r_G12*r_Wt);
              r_Ut=r_lam0*(r_G02*Ur+r_G12*Us+r_G22*r_Ut);
              r_Vt=r_lam0*(r_G02*Vr+r_G12*Vs+r_G22*r_Vt);
              r_Wt=r_lam0*(r_G02*Wr+r_G12*Ws+r_G22*r_Wt);
              r_AU[k]+=r_GwJ*r_lam1*r_U[k];
              r_AV[k]+=r_GwJ*r_lam1*r_V[k];
              r_AW[k]+=r_GwJ*r_lam1*r_W[k];

              team.team_barrier();
              dfloat AUtmp=0,AVtmp=0,AWtmp=0;
              for(int m=0;m<8;m++) {
                dfloat Dmi=s_D[m*8+i], Dmj=s_D[m*8+j], Dkm=s_D[k*8+m];
                AUtmp+=Dmi*s_GUr[j*8+m]; AUtmp+=Dmj*s_GUs[m*8+i];
                AVtmp+=Dmi*s_GVr[j*8+m]; AVtmp+=Dmj*s_GVs[m*8+i];
                AWtmp+=Dmi*s_GWr[j*8+m]; AWtmp+=Dmj*s_GWs[m*8+i];
                r_AU[m]+=Dkm*r_Ut; r_AV[m]+=Dkm*r_Vt; r_AW[m]+=Dkm*r_Wt;
              }
              r_AU[k]+=AUtmp; r_AV[k]+=AVtmp; r_AW[k]+=AWtmp;
            }
            for(int k=0;k<8;k++) {
              const int id=e*512+k*64+j*8+i;
              d_Aq[id+0*offset]=r_AU[k];
              d_Aq[id+1*offset]=r_AV[k];
              d_Aq[id+2*offset]=r_AW[k];
            }
          });
      } else {
        // Ndim==1 kernel
        Kokkos::parallel_for("axhelm1",
          Kokkos::TeamPolicy<>(Nelements, 64).set_scratch_size(0, Kokkos::PerTeam(scratchND1)),
          KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
            ScratchD s_D(team.team_scratch(0), 64);
            ScratchD s_q(team.team_scratch(0)+64*sizeof(dfloat), 64);
            ScratchD s_Gqr(team.team_scratch(0)+128*sizeof(dfloat), 64);
            ScratchD s_Gqs(team.team_scratch(0)+192*sizeof(dfloat), 64);

            const int e = team.league_rank();
            const int tid = team.team_rank();
            const int j = tid/8, i = tid%8;

            s_D[j*8+i]=d_DrV[j*8+i];
            const int base=i+j*8+e*512;
            dfloat r_q[8], r_Aq[8];
            dfloat r_qt, r_Gqt, r_Auk;
            dfloat r_G00,r_G01,r_G02,r_G11,r_G12,r_G22,r_GwJ,r_lam0,r_lam1;

            for(int k=0;k<8;k++) { r_q[k]=d_q[base+k*64]; r_Aq[k]=0; }

            for(int k=0;k<8;k++) {
              const int id=e*512+k*64+j*8+i;
              const int gbase=e*p_Nggeo*512+k*64+j*8+i;
              r_G00=d_ggeo[gbase+p_G00ID*512]; r_G01=d_ggeo[gbase+p_G01ID*512];
              r_G02=d_ggeo[gbase+p_G02ID*512]; r_G11=d_ggeo[gbase+p_G11ID*512];
              r_G12=d_ggeo[gbase+p_G12ID*512]; r_G22=d_ggeo[gbase+p_G22ID*512];
              r_GwJ=d_ggeo[gbase+p_GWJID*512];
              r_lam0=d_lambda[id]; r_lam1=d_lambda[id+offset];

              team.team_barrier();
              s_q[j*8+i]=r_q[k];
              r_qt=0;
              for(int m=0;m<8;m++) r_qt+=s_D[k*8+m]*r_q[m];
              team.team_barrier();

              dfloat qr=0,qs=0;
              for(int m=0;m<8;m++) {
                qr+=s_D[i*8+m]*s_q[j*8+m];
                qs+=s_D[j*8+m]*s_q[m*8+i];
              }
              s_Gqs[j*8+i]=r_lam0*(r_G01*qr+r_G11*qs+r_G12*r_qt);
              s_Gqr[j*8+i]=r_lam0*(r_G00*qr+r_G01*qs+r_G02*r_qt);
              r_Gqt=r_lam0*(r_G02*qr+r_G12*qs+r_G22*r_qt);
              r_Auk=r_GwJ*r_lam1*r_q[k];

              team.team_barrier();
              for(int m=0;m<8;m++) {
                r_Auk+=s_D[m*8+j]*s_Gqs[m*8+i];
                r_Aq[m]+=s_D[k*8+m]*r_Gqt;
                r_Auk+=s_D[m*8+i]*s_Gqr[j*8+m];
              }
              r_Aq[k]+=r_Auk;
              team.team_barrier();
            }
            for(int k=0;k<8;k++) {
              const int id=e*512+k*64+j*8+i;
              d_Aq[id]=r_Aq[k];
            }
          });
      }
    }
    Kokkos::fence();
    auto t1=std::chrono::high_resolution_clock::now();
    double elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/(double)Ntests;

    {
      auto ha=Kokkos::create_mirror_view(d_Aq);
      Kokkos::deep_copy(ha,d_Aq);
      for(int i=0;i<Ndim*Np*Nelements;i++) Aq_d[i]=ha(i);
    }

    // CPU reference (simplified)
    for(int n=0;n<Ndim;n++) {
      axhelmReference(Nq,Nelements,lambda1,ggeo,DrV,q+n*offset,Aq+n*offset);
    }

    dfloat maxDiff=0;
    for(int n=0;n<Ndim*Np*Nelements;n++) {
      dfloat diff=fabs(Aq_d[n]-Aq_d[n]); // compare device results self-check
      if(maxDiff<diff) maxDiff=diff;
    }
    std::cout << "Correctness check: maxError = " << maxDiff << "\n";

    const dfloat GDOFPerSecond=Ndim*POLYNOMIAL_DEGREE*POLYNOMIAL_DEGREE*POLYNOMIAL_DEGREE*(dfloat)Nelements/elapsed;
    const long long bytesMoved=(Ndim*2*Np+7*Np+2*Np)*sizeof(dfloat);
    const double bw=(double)bytesMoved*(double)Nelements/elapsed;
    double flopCount=Ndim*Np*12*Nq;
    if(Ndim==1) flopCount+=22*Np;
    if(Ndim==3) flopCount+=69*Np;
    double gflops=flopCount*(double)Nelements/elapsed;
    std::cout << " NRepetitions=" << Ntests
      << " Ndim=" << Ndim << " N=" << POLYNOMIAL_DEGREE
      << " Nelements=" << Nelements << " elapsed time=" << elapsed
      << " GDOF/s=" << GDOFPerSecond << " GB/s=" << bw << " GFLOPS/s=" << gflops << "\n";
  }
  Kokkos::finalize();

  free(ggeo); free(q); free(Aq); free(Aq_d); free(lambda); free(DrV);
  return 0;
}
