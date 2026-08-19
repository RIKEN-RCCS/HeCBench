module rushlarsen_utils
  use iso_fortran_env, only: real64
  implicit none
  integer, parameter :: state_xr1=0, state_xr2=1, state_xs=2, state_m=3, state_h=4, state_j=5
  integer, parameter :: state_d=6, state_f=7, state_f2=8, state_fcass=9, state_s=10, state_r=11
  integer, parameter :: state_ca_sr=12, state_ca_i=13, state_ca_ss=14, state_r_prime=15
  integer, parameter :: state_na_i=16, state_v=17, state_k_i=18, num_states=19
  integer, parameter :: param_p_kna=0, param_g_k1=1, param_g_kr=2, param_g_ks=3, param_g_na=4
  integer, parameter :: param_g_bna=5, param_g_cal=6, param_g_bca=7, param_g_to=8, param_k_mna=9
  integer, parameter :: param_k_mk=10, param_p_nak=11, param_k_naca=12, param_k_sat=13
  integer, parameter :: param_km_ca=14, param_km_nai=15, param_alpha=16, param_gamma=17
  integer, parameter :: param_k_pca=18, param_g_pca=19, param_g_pk=20, param_buf_c=21
  integer, parameter :: param_buf_sr=22, param_buf_ss=23, param_ca_o=24, param_ec=25
  integer, parameter :: param_k_buf_c=26, param_k_buf_sr=27, param_k_buf_ss=28, param_k_up=29
  integer, parameter :: param_v_leak=30, param_v_rel=31, param_v_sr=32, param_v_ss=33
  integer, parameter :: param_v_xfer=34, param_vmax_up=35, param_k1_prime=36, param_k2_prime=37
  integer, parameter :: param_k3=38, param_k4=39, param_max_sr=40, param_min_sr=41
  integer, parameter :: param_na_o=42, param_cm=43, param_fconst=44, param_rconst=45
  integer, parameter :: param_t=46, param_v_c=47, param_stim_amplitude=48
  integer, parameter :: param_stim_duration=49, param_stim_period=50, param_stim_start=51
  integer, parameter :: param_k_o=52, num_params=53
contains
  subroutine init_state_values(states, n)
    real(real64), intent(out) :: states(0:)
    integer, intent(in) :: n
    integer :: i
    do i = 0, n - 1
      states(n*state_xr1+i)=0.0165_real64; states(n*state_xr2+i)=0.473_real64
      states(n*state_xs+i)=0.0174_real64; states(n*state_m+i)=0.00165_real64
      states(n*state_h+i)=0.749_real64; states(n*state_j+i)=0.6788_real64
      states(n*state_d+i)=3.288e-05_real64; states(n*state_f+i)=0.7026_real64
      states(n*state_f2+i)=0.9526_real64; states(n*state_fcass+i)=0.9942_real64
      states(n*state_s+i)=0.999998_real64; states(n*state_r+i)=2.347e-08_real64
      states(n*state_ca_i+i)=0.000153_real64; states(n*state_r_prime+i)=0.8978_real64
      states(n*state_ca_sr+i)=4.272_real64; states(n*state_ca_ss+i)=0.00042_real64
      states(n*state_na_i+i)=10.132_real64; states(n*state_v+i)=-85.423_real64
      states(n*state_k_i+i)=138.52_real64
    end do
  end subroutine init_state_values
  subroutine init_parameters_values(parameters, n)
    real(real64), intent(out) :: parameters(0:)
    integer, intent(in) :: n
    integer :: i
    do i = 0, n - 1
      parameters(n*param_p_kna+i)=0.03_real64; parameters(n*param_g_k1+i)=5.405_real64
      parameters(n*param_g_kr+i)=0.153_real64; parameters(n*param_g_ks+i)=0.098_real64
      parameters(n*param_g_na+i)=14.838_real64; parameters(n*param_g_bna+i)=0.00029_real64
      parameters(n*param_g_cal+i)=3.98e-05_real64; parameters(n*param_g_bca+i)=0.000592_real64
      parameters(n*param_g_to+i)=0.294_real64; parameters(n*param_k_mna+i)=40.0_real64
      parameters(n*param_k_mk+i)=1.0_real64; parameters(n*param_p_nak+i)=2.724_real64
      parameters(n*param_k_naca+i)=1000.0_real64; parameters(n*param_k_sat+i)=0.1_real64
      parameters(n*param_km_ca+i)=1.38_real64; parameters(n*param_km_nai+i)=87.5_real64
      parameters(n*param_alpha+i)=2.5_real64; parameters(n*param_gamma+i)=0.35_real64
      parameters(n*param_k_pca+i)=0.0005_real64; parameters(n*param_g_pca+i)=0.1238_real64
      parameters(n*param_g_pk+i)=0.0146_real64; parameters(n*param_buf_c+i)=0.2_real64
      parameters(n*param_buf_sr+i)=10.0_real64; parameters(n*param_buf_ss+i)=0.4_real64
      parameters(n*param_ca_o+i)=2.0_real64; parameters(n*param_ec+i)=1.5_real64
      parameters(n*param_k_buf_c+i)=0.001_real64; parameters(n*param_k_buf_sr+i)=0.3_real64
      parameters(n*param_k_buf_ss+i)=0.00025_real64; parameters(n*param_k_up+i)=0.00025_real64
      parameters(n*param_v_leak+i)=0.00036_real64; parameters(n*param_v_rel+i)=0.102_real64
      parameters(n*param_v_sr+i)=0.001094_real64; parameters(n*param_v_ss+i)=5.468e-05_real64
      parameters(n*param_v_xfer+i)=0.0038_real64; parameters(n*param_vmax_up+i)=0.006375_real64
      parameters(n*param_k1_prime+i)=0.15_real64; parameters(n*param_k2_prime+i)=0.045_real64
      parameters(n*param_k3+i)=0.06_real64; parameters(n*param_k4+i)=0.005_real64
      parameters(n*param_max_sr+i)=2.5_real64; parameters(n*param_min_sr+i)=1.0_real64
      parameters(n*param_na_o+i)=140.0_real64; parameters(n*param_cm+i)=0.185_real64
      parameters(n*param_fconst+i)=96485.3415_real64; parameters(n*param_rconst+i)=8314.472_real64
      parameters(n*param_t+i)=310.0_real64; parameters(n*param_v_c+i)=0.016404_real64
      parameters(n*param_stim_amplitude+i)=52.0_real64; parameters(n*param_stim_duration+i)=1.0_real64
      parameters(n*param_stim_period+i)=1000.0_real64; parameters(n*param_stim_start+i)=10.0_real64
      parameters(n*param_k_o+i)=5.4_real64
    end do
  end subroutine init_parameters_values
end module rushlarsen_utils
