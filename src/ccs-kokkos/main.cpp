#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <Kokkos_Core.hpp>

#define mingene   10
#define minsample 10
#define MAXSAMPLE 200

struct pair_r { float r, n_r; };

KOKKOS_INLINE_FUNCTION pair_r compute(
    float *genekj, float *geneij,
    const char *sample,
    int wid, int k, int i, int D,
    const float *gene)
{
  float sx=0,sxx=0,sy=0,sxy=0,syy=0;
  float sx_n=0,sxx_n=0,sy_n=0,sxy_n=0,syy_n=0;
  pair_r rval={0.f,0.f};
  for(int j=0;j<D;j++) {
    genekj[j]=gene[k*(D+1)+j];
    if(sample[j]=='1') sx+=genekj[j]; else sx_n+=genekj[j];
  }
  sx/=wid; sx_n/=(D-wid);
  for(int j=0;j<D;j++) {
    if(sample[j]=='1') sxx+=(sx-genekj[j])*(sx-genekj[j]);
    else sxx_n+=(sx_n-genekj[j])*(sx_n-genekj[j]);
  }
  sxx=Kokkos::sqrt(sxx); sxx_n=Kokkos::sqrt(sxx_n);
  for(int j=0;j<D;j++) {
    geneij[j]=gene[i*(D+1)+j];
    if(sample[j]=='1') sy+=geneij[j]; else sy_n+=geneij[j];
  }
  sy/=wid; sy_n/=(D-wid);
  for(int j=0;j<D;j++) {
    if(sample[j]=='1') {
      sxy+=(sx-genekj[j])*(sy-geneij[j]);
      syy+=(sy-geneij[j])*(sy-geneij[j]);
    } else {
      sxy_n+=(sx_n-genekj[j])*(sy_n-geneij[j]);
      syy_n+=(sy_n-geneij[j])*(sy_n-geneij[j]);
    }
  }
  syy=Kokkos::sqrt(syy); syy_n=Kokkos::sqrt(syy_n);
  float den=sxx*syy, den_n=sxx_n*syy_n;
  rval.r   = (den  >1e-9f) ? Kokkos::fabs(sxy/den)   : 0.f;
  rval.n_r = (den_n>1e-9f) ? Kokkos::fabs(sxy_n/den_n) : 0.f;
  return rval;
}

int main(int argc, char **argv) {
  // Parameters
  int n      = 100;   // number of genes
  int D      = 50;    // number of samples
  int maxbcn = 50;    // max number of biclusters
  float thr  = 0.8f;
  int repeat = (argc>1) ? atoi(argv[1]) : 10;

  if(D>MAXSAMPLE) D=MAXSAMPLE;
  if(maxbcn>n) maxbcn=n;

  // Generate synthetic gene expression data
  float *h_gene = (float*)malloc(n*(D+1)*sizeof(float));
  srand(42);
  for(int i=0;i<n;i++) {
    float mean=0;
    for(int j=0;j<D;j++) {
      h_gene[i*(D+1)+j]=(float)(rand()%100)/10.0f;
      mean+=h_gene[i*(D+1)+j];
    }
    h_gene[i*(D+1)+D]=mean/D;
  }

  float *h_bc_score      = (float*)malloc(maxbcn*sizeof(float));
  int   *h_bc_datacount  = (int*)malloc(maxbcn*sizeof(int));
  int   *h_bc_samplecount= (int*)malloc(maxbcn*sizeof(int));
  char  *h_bc_sample     = (char*)malloc(D*maxbcn*sizeof(char));
  char  *h_bc_data       = (char*)malloc(n*maxbcn*sizeof(char));
  char  *h_bc_sample_tmp = (char*)malloc(D*maxbcn*sizeof(char));
  char  *h_bc_data_tmp   = (char*)malloc(n*maxbcn*sizeof(char));

  Kokkos::initialize(argc, argv);
  {
    using ViewF  = Kokkos::View<float*>;
    using ViewI  = Kokkos::View<int*>;
    using ViewC  = Kokkos::View<char*>;

    const int gN=n, gD=D, gMaxbcn=maxbcn;
    ViewF d_gene("gene", n*(D+1));
    ViewF d_bc_score("bc_score", maxbcn);
    ViewI d_bc_datacount("bc_dc", maxbcn);
    ViewI d_bc_samplecount("bc_sc", maxbcn);
    ViewC d_bc_sample("bc_sample", D*maxbcn);
    ViewC d_bc_data("bc_data", n*maxbcn);
    ViewC d_bc_sample_tmp("bc_sample_tmp", D*maxbcn);
    ViewC d_bc_data_tmp("bc_data_tmp", n*maxbcn);

    {
      auto hg=Kokkos::create_mirror_view(d_gene);
      for(int i=0;i<n*(D+1);i++) hg(i)=h_gene[i];
      Kokkos::deep_copy(d_gene,hg);
    }

    auto t0=std::chrono::steady_clock::now();
    for(int rep=0;rep<repeat;rep++) {
      // Single-thread TeamPolicy: one team per bicluster candidate (k)
      // Since OMP uses thread_limit(1), a simple parallel_for is equivalent
      Kokkos::parallel_for("compute_bicluster", maxbcn, KOKKOS_LAMBDA(int k) {
        // Local scratch arrays (were team shared but single-thread in OMP)
        float s_genekj[MAXSAMPLE], s_geneij[MAXSAMPLE];
        char  s_vect[3*MAXSAMPLE];

        float jcc, mean_k, mean_i;
        int wid, wid_0, wid_1, wid_2, vl, l, l_i;
        int dif, tot, t_tot, t_dif;
        int tmpbc_datacount, tmpbc_samplecount;
        pair_r rval;

        d_bc_score[k]     = 1.f;
        d_bc_datacount[k] = 0;
        mean_k = d_gene[k*(gD+1)+gD];

        for(int i=k+1;i<gN;i++) {
          mean_i=d_gene[i*(gD+1)+gD];
          wid_0=wid_1=wid_2=0;
          for(int j=0;j<gD;j++) {
            float gkj=d_gene[k*(gD+1)+j], gij=d_gene[i*(gD+1)+j];
            if((gkj-mean_k)>=0&&(gij-mean_i)>=0) {
              s_vect[0*MAXSAMPLE+j]='1'; s_vect[1*MAXSAMPLE+j]='0'; s_vect[2*MAXSAMPLE+j]='0'; wid_0++;
            } else if((gkj-mean_k)<0&&(gij-mean_i)<0) {
              s_vect[0*MAXSAMPLE+j]='0'; s_vect[1*MAXSAMPLE+j]='1'; s_vect[2*MAXSAMPLE+j]='0'; wid_1++;
            } else if((gkj-mean_k)*(gij-mean_i)<0) {
              s_vect[0*MAXSAMPLE+j]='0'; s_vect[1*MAXSAMPLE+j]='0'; s_vect[2*MAXSAMPLE+j]='1'; wid_2++;
            }
          }

          for(vl=0;vl<3;vl++) {
            dif=tot=0;
            if(vl==0) wid=wid_0;
            else if(vl==1) wid=wid_1;
            else wid=wid_2;
            if(wid<=minsample) continue;

            rval=compute(s_genekj,s_geneij,s_vect+vl*MAXSAMPLE,wid,k,i,gD,d_gene.data());
            if(rval.r>thr) {
              tot++;
              if(rval.n_r>thr) dif++;
              for(int j=0;j<gD;j++) d_bc_sample_tmp[k*gD+j]=s_vect[vl*MAXSAMPLE+j];
              for(int j=0;j<gN;j++) d_bc_data_tmp[k*gN+j]='0';
              d_bc_data_tmp[k*gN+k]='1'; d_bc_data_tmp[k*gN+i]='1';
              tmpbc_datacount=2; tmpbc_samplecount=wid;

              for(l=0;l<gN;l++) {
                if(l!=i&&l!=k) {
                  t_tot=t_dif=0;
                  for(l_i=0;l_i<gN;l_i++) {
                    if(d_bc_data_tmp[k*gN+l_i]=='1') {
                      rval=compute(s_genekj,s_geneij,s_vect+vl*MAXSAMPLE,wid,l,l_i,gD,d_gene.data());
                      if(rval.r>thr) t_tot+=1; else { t_tot=0; break; }
                      if(rval.n_r>thr) t_dif+=1;
                    }
                  }
                  if(t_tot>0) { d_bc_data_tmp[k*gN+l]='1'; tmpbc_datacount++; tot+=t_tot; dif+=t_dif; }
                }
              }

              jcc=(tot>0)?(float)dif/tot:1.f;
              if(jcc<0.01f&&d_bc_datacount[k]<tmpbc_datacount&&tmpbc_datacount>mingene) {
                d_bc_score[k]=jcc;
                for(int j=0;j<gN;j++) d_bc_data[k*gN+j]=d_bc_data_tmp[k*gN+j];
                for(int j=0;j<gD;j++) d_bc_sample[k*gD+j]=d_bc_sample_tmp[k*gD+j];
                d_bc_datacount[k]=tmpbc_datacount;
                d_bc_samplecount[k]=tmpbc_samplecount;
              }
            }
          }
        }
      });
    }
    Kokkos::fence();
    auto t1=std::chrono::steady_clock::now();
    printf("Average kernel execution time %f (s)\n",
      std::chrono::duration<double>(t1-t0).count()/repeat);

    auto hs=Kokkos::create_mirror_view(d_bc_score);
    auto hd=Kokkos::create_mirror_view(d_bc_datacount);
    Kokkos::deep_copy(hs,d_bc_score); Kokkos::deep_copy(hd,d_bc_datacount);
    int found=0;
    for(int i=0;i<maxbcn;i++) if(hd(i)>=mingene) found++;
    printf("Biclusters found with >= %d genes: %d\n", mingene, found);
  }
  Kokkos::finalize();

  free(h_gene); free(h_bc_score); free(h_bc_datacount); free(h_bc_samplecount);
  free(h_bc_sample); free(h_bc_data); free(h_bc_sample_tmp); free(h_bc_data_tmp);
  return 0;
}
