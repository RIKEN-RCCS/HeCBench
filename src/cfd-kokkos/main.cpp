#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <Kokkos_Core.hpp>

#define GAMMA 1.4f
#define iterations 2000
#define block_length 192
#define NDIM 3
#define NNB 4
#define RK 3
#define ff_mach 1.2f
#define deg_angle_of_attack 0.0f
#define VAR_DENSITY 0
#define VAR_MOMENTUM 1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)

struct Float3 { float x, y, z; };

KOKKOS_INLINE_FUNCTION
void compute_velocity(float density, Float3 momentum, Float3 &velocity) {
  velocity.x = momentum.x / density;
  velocity.y = momentum.y / density;
  velocity.z = momentum.z / density;
}
KOKKOS_INLINE_FUNCTION float compute_speed_sqd(Float3 v){return v.x*v.x+v.y*v.y+v.z*v.z;}
KOKKOS_INLINE_FUNCTION float compute_pressure(float d,float de,float ss){return (GAMMA-1.0f)*(de-0.5f*d*ss);}
KOKKOS_INLINE_FUNCTION float compute_speed_of_sound(float d,float p){return Kokkos::sqrt(GAMMA*p/d);}

KOKKOS_INLINE_FUNCTION
void compute_flux_contribution(float d, Float3 m, float de, float p, Float3 v,
    Float3 &fmx, Float3 &fmy, Float3 &fmz, Float3 &fde)
{
  fmx.x=v.x*m.x+p; fmx.y=v.x*m.y; fmx.z=v.x*m.z;
  fmy.x=fmx.y; fmy.y=v.y*m.y+p; fmy.z=v.y*m.z;
  fmz.x=fmx.z; fmz.y=fmy.z; fmz.z=v.z*m.z+p;
  float dep=de+p;
  fde.x=v.x*dep; fde.y=v.y*dep; fde.z=v.z*dep;
}

int main(int argc, char **argv) {
  if (argc < 2) { std::cout << "Please specify data file name" << std::endl; return 0; }

  const float angle_of_attack = float(3.14159265358979323846f/180.0f)*deg_angle_of_attack;
  float h_ff_variable[NVAR];
  h_ff_variable[VAR_DENSITY] = 1.4f;
  float ff_pressure = 1.0f;
  float ff_sos = Kokkos::sqrt((float)GAMMA * ff_pressure / h_ff_variable[VAR_DENSITY]);
  float ff_speed = ff_mach * ff_sos;
  Float3 ff_vel; ff_vel.x=ff_speed*cosf(angle_of_attack); ff_vel.y=ff_speed*sinf(angle_of_attack); ff_vel.z=0;
  h_ff_variable[VAR_MOMENTUM+0]=h_ff_variable[VAR_DENSITY]*ff_vel.x;
  h_ff_variable[VAR_MOMENTUM+1]=h_ff_variable[VAR_DENSITY]*ff_vel.y;
  h_ff_variable[VAR_MOMENTUM+2]=h_ff_variable[VAR_DENSITY]*ff_vel.z;
  h_ff_variable[VAR_DENSITY_ENERGY]=h_ff_variable[VAR_DENSITY]*(0.5f*(ff_speed*ff_speed))+(ff_pressure/(GAMMA-1.0f));

  Float3 h_ff_mom;
  h_ff_mom.x=h_ff_variable[VAR_MOMENTUM+0];
  h_ff_mom.y=h_ff_variable[VAR_MOMENTUM+1];
  h_ff_mom.z=h_ff_variable[VAR_MOMENTUM+2];
  Float3 h_ffc_mx, h_ffc_my, h_ffc_mz, h_ffc_de;
  compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_mom, h_ff_variable[VAR_DENSITY_ENERGY],
    ff_pressure, ff_vel, h_ffc_mx, h_ffc_my, h_ffc_mz, h_ffc_de);

  int nel;
  std::ifstream file(argv[1], std::ifstream::in);
  if (!file.good()) { std::cout << "Cannot open file: " << argv[1] << std::endl; return 1; }
  file >> nel;
  int nelr = block_length * ((nel/block_length) + std::min(1, nel%block_length));
  std::cout << "--cambine: nel=" << nel << ", nelr=" << nelr << std::endl;

  float *h_areas = new float[nelr];
  int *h_ese = new int[nelr*NNB];
  float *h_normals = new float[nelr*NDIM*NNB];
  float *h_variables = new float[nelr*NVAR];

  for (int i = 0; i < nel; i++) {
    file >> h_areas[i];
    for (int j = 0; j < NNB; j++) {
      file >> h_ese[i+j*nelr];
      if (h_ese[i+j*nelr] < 0) h_ese[i+j*nelr] = -1;
      h_ese[i+j*nelr]--;
      for (int k = 0; k < NDIM; k++) {
        file >> h_normals[i+(j+k*NNB)*nelr];
        h_normals[i+(j+k*NNB)*nelr] = -h_normals[i+(j+k*NNB)*nelr];
      }
    }
  }
  int last = nel-1;
  for (int i = nel; i < nelr; i++) {
    h_areas[i] = h_areas[last];
    for (int j = 0; j < NNB; j++) {
      h_ese[i+j*nelr] = h_ese[last+j*nelr];
      for (int k = 0; k < NDIM; k++)
        h_normals[i+(j+k*NNB)*nelr] = h_normals[last+(j+k*NNB)*nelr];
    }
  }

  Kokkos::initialize(argc, argv);
  {
    using ViewF = Kokkos::View<float*>;
    using ViewI = Kokkos::View<int*>;

    ViewF d_areas("areas", nelr);
    ViewI d_ese("ese", nelr*NNB);
    ViewF d_normals("normals", nelr*NDIM*NNB);
    ViewF d_variables("variables", nelr*NVAR);
    ViewF d_old_variables("old_vars", nelr*NVAR);
    ViewF d_fluxes("fluxes", nelr*NVAR);
    ViewF d_step_factors("sf", nelr);
    ViewF d_ff_variable("ff_var", NVAR);

    {
      auto ha=Kokkos::create_mirror_view(d_areas);
      auto hi=Kokkos::create_mirror_view(d_ese);
      auto hn=Kokkos::create_mirror_view(d_normals);
      auto hff=Kokkos::create_mirror_view(d_ff_variable);
      for(int i=0;i<nelr;i++) ha(i)=h_areas[i];
      for(int i=0;i<nelr*NNB;i++) hi(i)=h_ese[i];
      for(int i=0;i<nelr*NDIM*NNB;i++) hn(i)=h_normals[i];
      for(int i=0;i<NVAR;i++) hff(i)=h_ff_variable[i];
      Kokkos::deep_copy(d_areas,ha); Kokkos::deep_copy(d_ese,hi);
      Kokkos::deep_copy(d_normals,hn); Kokkos::deep_copy(d_ff_variable,hff);
    }

    Float3 ffc_mx=h_ffc_mx, ffc_my=h_ffc_my, ffc_mz=h_ffc_mz, ffc_de=h_ffc_de;
    int NR=nelr;

    // initialize_variables
    auto init_vars = [&](ViewF &vars) {
      Kokkos::parallel_for("init_vars", nelr, KOKKOS_LAMBDA(int i) {
        for (int j=0;j<NVAR;j++) vars(i+j*NR)=d_ff_variable(j);
      });
    };
    init_vars(d_variables);
    init_vars(d_old_variables);
    init_vars(d_fluxes);
    Kokkos::parallel_for("init_sf", nelr, KOKKOS_LAMBDA(int i){d_step_factors(i)=0;});

    auto t0 = std::chrono::steady_clock::now();

    for (int n = 0; n < iterations; n++) {
      // copy variables -> old_variables
      Kokkos::parallel_for("copy", nelr*NVAR, KOKKOS_LAMBDA(int i){d_old_variables(i)=d_variables(i);});

      // compute_step_factor
      Kokkos::parallel_for("step_factor", nelr, KOKKOS_LAMBDA(int i) {
        float density = d_variables(i+VAR_DENSITY*NR);
        Float3 momentum; momentum.x=d_variables(i+(VAR_MOMENTUM+0)*NR);
        momentum.y=d_variables(i+(VAR_MOMENTUM+1)*NR); momentum.z=d_variables(i+(VAR_MOMENTUM+2)*NR);
        float density_energy = d_variables(i+VAR_DENSITY_ENERGY*NR);
        Float3 velocity; compute_velocity(density, momentum, velocity);
        float ss = compute_speed_sqd(velocity);
        float pressure = compute_pressure(density, density_energy, ss);
        float sos = compute_speed_of_sound(density, pressure);
        d_step_factors(i) = 0.5f / (Kokkos::sqrt(d_areas(i))*(Kokkos::sqrt(ss)+sos));
      });

      for (int j = 0; j < RK; j++) {
        // compute_flux
        Kokkos::parallel_for("flux", nelr, KOKKOS_LAMBDA(int i) {
          float sc = 0.2f;
          float density_i = d_variables(i+VAR_DENSITY*NR);
          Float3 momentum_i; momentum_i.x=d_variables(i+(VAR_MOMENTUM+0)*NR);
          momentum_i.y=d_variables(i+(VAR_MOMENTUM+1)*NR); momentum_i.z=d_variables(i+(VAR_MOMENTUM+2)*NR);
          float de_i = d_variables(i+VAR_DENSITY_ENERGY*NR);
          Float3 velocity_i; compute_velocity(density_i,momentum_i,velocity_i);
          float ss_i=compute_speed_sqd(velocity_i), speed_i=Kokkos::sqrt(ss_i);
          float pressure_i=compute_pressure(density_i,de_i,ss_i);
          float sos_i=compute_speed_of_sound(density_i,pressure_i);
          Float3 fmx_i,fmy_i,fmz_i,fde_i;
          compute_flux_contribution(density_i,momentum_i,de_i,pressure_i,velocity_i,fmx_i,fmy_i,fmz_i,fde_i);

          float fd=0, fmx=0, fmy=0, fmz=0, fde=0;

          for (int jj=0;jj<NNB;jj++) {
            int nb = d_ese(i+jj*NR);
            Float3 normal; normal.x=d_normals(i+(jj+0*NNB)*NR);
            normal.y=d_normals(i+(jj+1*NNB)*NR); normal.z=d_normals(i+(jj+2*NNB)*NR);
            float nl = Kokkos::sqrt(normal.x*normal.x+normal.y*normal.y+normal.z*normal.z);
            float factor;

            if (nb>=0) {
              float dn=d_variables(nb+VAR_DENSITY*NR);
              Float3 mn; mn.x=d_variables(nb+(VAR_MOMENTUM+0)*NR);
              mn.y=d_variables(nb+(VAR_MOMENTUM+1)*NR); mn.z=d_variables(nb+(VAR_MOMENTUM+2)*NR);
              float den=d_variables(nb+VAR_DENSITY_ENERGY*NR);
              Float3 vn; compute_velocity(dn,mn,vn);
              float ssn=compute_speed_sqd(vn), pn=compute_pressure(dn,den,ssn);
              float sosn=compute_speed_of_sound(dn,pn);
              Float3 fmxn,fmyn,fmzn,fden;
              compute_flux_contribution(dn,mn,den,pn,vn,fmxn,fmyn,fmzn,fden);
              factor=-nl*sc*0.5f*(speed_i+Kokkos::sqrt(ssn)+sos_i+sosn);
              fd+=factor*(density_i-dn); fde+=factor*(de_i-den);
              fmx+=factor*(momentum_i.x-mn.x); fmy+=factor*(momentum_i.y-mn.y); fmz+=factor*(momentum_i.z-mn.z);
              factor=0.5f*normal.x; fd+=factor*(mn.x+momentum_i.x);
              fde+=factor*(fden.x+fde_i.x); fmx+=factor*(fmxn.x+fmx_i.x);
              fmy+=factor*(fmyn.x+fmy_i.x); fmz+=factor*(fmzn.x+fmz_i.x);
              factor=0.5f*normal.y; fd+=factor*(mn.y+momentum_i.y);
              fde+=factor*(fden.y+fde_i.y); fmx+=factor*(fmxn.y+fmx_i.y);
              fmy+=factor*(fmyn.y+fmy_i.y); fmz+=factor*(fmzn.y+fmz_i.y);
              factor=0.5f*normal.z; fd+=factor*(mn.z+momentum_i.z);
              fde+=factor*(fden.z+fde_i.z); fmx+=factor*(fmxn.z+fmx_i.z);
              fmy+=factor*(fmyn.z+fmy_i.z); fmz+=factor*(fmzn.z+fmz_i.z);
            } else if (nb==-1) {
              fmx+=normal.x*pressure_i; fmy+=normal.y*pressure_i; fmz+=normal.z*pressure_i;
            } else if (nb==-2) {
              factor=0.5f*normal.x;
              fd+=factor*(d_ff_variable(VAR_MOMENTUM+0)+momentum_i.x);
              fde+=factor*(ffc_de.x+fde_i.x); fmx+=factor*(ffc_mx.x+fmx_i.x);
              fmy+=factor*(ffc_my.x+fmy_i.x); fmz+=factor*(ffc_mz.x+fmz_i.x);
              factor=0.5f*normal.y;
              fd+=factor*(d_ff_variable(VAR_MOMENTUM+1)+momentum_i.y);
              fde+=factor*(ffc_de.y+fde_i.y); fmx+=factor*(ffc_mx.y+fmx_i.y);
              fmy+=factor*(ffc_my.y+fmy_i.y); fmz+=factor*(ffc_mz.y+fmz_i.y);
              factor=0.5f*normal.z;
              fd+=factor*(d_ff_variable(VAR_MOMENTUM+2)+momentum_i.z);
              fde+=factor*(ffc_de.z+fde_i.z); fmx+=factor*(ffc_mx.z+fmx_i.z);
              fmy+=factor*(ffc_my.z+fmy_i.z); fmz+=factor*(ffc_mz.z+fmz_i.z);
            }
          }
          d_fluxes(i+VAR_DENSITY*NR)=fd;
          d_fluxes(i+(VAR_MOMENTUM+0)*NR)=fmx; d_fluxes(i+(VAR_MOMENTUM+1)*NR)=fmy; d_fluxes(i+(VAR_MOMENTUM+2)*NR)=fmz;
          d_fluxes(i+VAR_DENSITY_ENERGY*NR)=fde;
        });

        // time_step
        Kokkos::parallel_for("time_step", nelr, KOKKOS_LAMBDA(int i) {
          float factor = d_step_factors(i) / (float)(RK+1-j);
          d_variables(i+VAR_DENSITY*NR)=d_old_variables(i+VAR_DENSITY*NR)+factor*d_fluxes(i+VAR_DENSITY*NR);
          d_variables(i+VAR_DENSITY_ENERGY*NR)=d_old_variables(i+VAR_DENSITY_ENERGY*NR)+factor*d_fluxes(i+VAR_DENSITY_ENERGY*NR);
          for (int k=0;k<NDIM;k++)
            d_variables(i+(VAR_MOMENTUM+k)*NR)=d_old_variables(i+(VAR_MOMENTUM+k)*NR)+factor*d_fluxes(i+(VAR_MOMENTUM+k)*NR);
        });
      }
    }
    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();
    printf("Total execution time of kernels = %lf(s)\n",
      std::chrono::duration<double>(t1-t0).count());

    auto hv = Kokkos::create_mirror_view(d_variables);
    Kokkos::deep_copy(hv, d_variables);
    for (int i=0;i<nelr*NVAR;i++) h_variables[i]=hv(i);
  }
  Kokkos::finalize();

  delete[] h_areas; delete[] h_ese; delete[] h_normals; delete[] h_variables;
  std::cout << "Done..." << std::endl;
  return 0;
}
