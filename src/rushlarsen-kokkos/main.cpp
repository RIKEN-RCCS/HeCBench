#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// ---- State and parameter enums ----
enum state {
    STATE_Xr1, STATE_Xr2, STATE_Xs, STATE_m, STATE_h, STATE_j,
    STATE_d, STATE_f, STATE_f2, STATE_fCass, STATE_s, STATE_r,
    STATE_Ca_SR, STATE_Ca_i, STATE_Ca_ss, STATE_R_prime,
    STATE_Na_i, STATE_V, STATE_K_i,
    NUM_STATES,
};

enum parameter {
    PARAM_P_kna, PARAM_g_K1, PARAM_g_Kr, PARAM_g_Ks, PARAM_g_Na,
    PARAM_g_bna, PARAM_g_CaL, PARAM_g_bca, PARAM_g_to,
    PARAM_K_mNa, PARAM_K_mk, PARAM_P_NaK,
    PARAM_K_NaCa, PARAM_K_sat, PARAM_Km_Ca, PARAM_Km_Nai,
    PARAM_alpha, PARAM_gamma,
    PARAM_K_pCa, PARAM_g_pCa, PARAM_g_pK,
    PARAM_Buf_c, PARAM_Buf_sr, PARAM_Buf_ss,
    PARAM_Ca_o, PARAM_EC,
    PARAM_K_buf_c, PARAM_K_buf_sr, PARAM_K_buf_ss,
    PARAM_K_up, PARAM_V_leak, PARAM_V_rel, PARAM_V_sr, PARAM_V_ss, PARAM_V_xfer,
    PARAM_Vmax_up, PARAM_k1_prime, PARAM_k2_prime, PARAM_k3, PARAM_k4,
    PARAM_max_sr, PARAM_min_sr,
    PARAM_Na_o, PARAM_Cm, PARAM_F, PARAM_R, PARAM_T, PARAM_V_c,
    PARAM_stim_amplitude, PARAM_stim_duration, PARAM_stim_period, PARAM_stim_start,
    PARAM_K_o,
    NUM_PARAMS,
};

// ---- Initialisation helpers ----
static void init_state_values(double* states, int n)
{
    for (int i = 0; i < n; i++) {
        states[n * STATE_Xr1   + i] = 0.0165;
        states[n * STATE_Xr2   + i] = 0.473;
        states[n * STATE_Xs    + i] = 0.0174;
        states[n * STATE_m     + i] = 0.00165;
        states[n * STATE_h     + i] = 0.749;
        states[n * STATE_j     + i] = 0.6788;
        states[n * STATE_d     + i] = 3.288e-05;
        states[n * STATE_f     + i] = 0.7026;
        states[n * STATE_f2    + i] = 0.9526;
        states[n * STATE_fCass + i] = 0.9942;
        states[n * STATE_s     + i] = 0.999998;
        states[n * STATE_r     + i] = 2.347e-08;
        states[n * STATE_Ca_i  + i] = 0.000153;
        states[n * STATE_R_prime + i] = 0.8978;
        states[n * STATE_Ca_SR + i] = 4.272;
        states[n * STATE_Ca_ss + i] = 0.00042;
        states[n * STATE_Na_i  + i] = 10.132;
        states[n * STATE_V     + i] = -85.423;
        states[n * STATE_K_i   + i] = 138.52;
    }
}

static void init_parameters_values(double* parameters, int n)
{
    for (int i = 0; i < n; i++) {
        parameters[n * PARAM_P_kna + i] = 0.03;
        parameters[n * PARAM_g_K1  + i] = 5.405;
        parameters[n * PARAM_g_Kr  + i] = 0.153;
        parameters[n * PARAM_g_Ks  + i] = 0.098;
        parameters[n * PARAM_g_Na  + i] = 14.838;
        parameters[n * PARAM_g_bna + i] = 0.00029;
        parameters[n * PARAM_g_CaL + i] = 3.98e-05;
        parameters[n * PARAM_g_bca + i] = 0.000592;
        parameters[n * PARAM_g_to  + i] = 0.294;
        parameters[n * PARAM_K_mNa + i] = 40;
        parameters[n * PARAM_K_mk  + i] = 1;
        parameters[n * PARAM_P_NaK + i] = 2.724;
        parameters[n * PARAM_K_NaCa  + i] = 1000;
        parameters[n * PARAM_K_sat   + i] = 0.1;
        parameters[n * PARAM_Km_Ca   + i] = 1.38;
        parameters[n * PARAM_Km_Nai  + i] = 87.5;
        parameters[n * PARAM_alpha   + i] = 2.5;
        parameters[n * PARAM_gamma   + i] = 0.35;
        parameters[n * PARAM_K_pCa   + i] = 0.0005;
        parameters[n * PARAM_g_pCa   + i] = 0.1238;
        parameters[n * PARAM_g_pK    + i] = 0.0146;
        parameters[n * PARAM_Buf_c   + i] = 0.2;
        parameters[n * PARAM_Buf_sr  + i] = 10;
        parameters[n * PARAM_Buf_ss  + i] = 0.4;
        parameters[n * PARAM_Ca_o    + i] = 2;
        parameters[n * PARAM_EC      + i] = 1.5;
        parameters[n * PARAM_K_buf_c + i] = 0.001;
        parameters[n * PARAM_K_buf_sr + i] = 0.3;
        parameters[n * PARAM_K_buf_ss + i] = 0.00025;
        parameters[n * PARAM_K_up    + i] = 0.00025;
        parameters[n * PARAM_V_leak  + i] = 0.00036;
        parameters[n * PARAM_V_rel   + i] = 0.102;
        parameters[n * PARAM_V_sr    + i] = 0.001094;
        parameters[n * PARAM_V_ss    + i] = 5.468e-05;
        parameters[n * PARAM_V_xfer  + i] = 0.0038;
        parameters[n * PARAM_Vmax_up + i] = 0.006375;
        parameters[n * PARAM_k1_prime + i] = 0.15;
        parameters[n * PARAM_k2_prime + i] = 0.045;
        parameters[n * PARAM_k3      + i] = 0.06;
        parameters[n * PARAM_k4      + i] = 0.005;
        parameters[n * PARAM_max_sr  + i] = 2.5;
        parameters[n * PARAM_min_sr  + i] = 1.0;
        parameters[n * PARAM_Na_o    + i] = 140;
        parameters[n * PARAM_Cm      + i] = 0.185;
        parameters[n * PARAM_F       + i] = 96485.3415;
        parameters[n * PARAM_R       + i] = 8314.472;
        parameters[n * PARAM_T       + i] = 310;
        parameters[n * PARAM_V_c     + i] = 0.016404;
        parameters[n * PARAM_stim_amplitude + i] = 52;
        parameters[n * PARAM_stim_duration  + i] = 1;
        parameters[n * PARAM_stim_period    + i] = 1000;
        parameters[n * PARAM_stim_start     + i] = 10;
        parameters[n * PARAM_K_o    + i] = 5.4;
    }
}

// ---- The Rush-Larsen per-node kernel body (device and host) ----
// This macro expands the entire ODE update for one node i.
// It reads from states_ptr and parameters_ptr (both n-strided),
// and writes updated states back.
#define RUSH_LARSEN_BODY(states_ptr, parameters_ptr, n_val, i_val, t_val, dt_val) \
do { \
    const double Xr1   = (states_ptr)[(n_val) * STATE_Xr1   + (i_val)]; \
    const double Xr2   = (states_ptr)[(n_val) * STATE_Xr2   + (i_val)]; \
    const double Xs    = (states_ptr)[(n_val) * STATE_Xs    + (i_val)]; \
    const double m     = (states_ptr)[(n_val) * STATE_m     + (i_val)]; \
    const double h     = (states_ptr)[(n_val) * STATE_h     + (i_val)]; \
    const double j     = (states_ptr)[(n_val) * STATE_j     + (i_val)]; \
    const double d     = (states_ptr)[(n_val) * STATE_d     + (i_val)]; \
    const double f     = (states_ptr)[(n_val) * STATE_f     + (i_val)]; \
    const double f2    = (states_ptr)[(n_val) * STATE_f2    + (i_val)]; \
    const double fCass = (states_ptr)[(n_val) * STATE_fCass + (i_val)]; \
    const double s     = (states_ptr)[(n_val) * STATE_s     + (i_val)]; \
    const double r     = (states_ptr)[(n_val) * STATE_r     + (i_val)]; \
    const double Ca_i  = (states_ptr)[(n_val) * STATE_Ca_i  + (i_val)]; \
    const double R_prime = (states_ptr)[(n_val) * STATE_R_prime + (i_val)]; \
    const double Ca_SR = (states_ptr)[(n_val) * STATE_Ca_SR + (i_val)]; \
    const double Ca_ss = (states_ptr)[(n_val) * STATE_Ca_ss + (i_val)]; \
    const double Na_i  = (states_ptr)[(n_val) * STATE_Na_i  + (i_val)]; \
    const double V     = (states_ptr)[(n_val) * STATE_V     + (i_val)]; \
    const double K_i   = (states_ptr)[(n_val) * STATE_K_i   + (i_val)]; \
    const double P_kna = (parameters_ptr)[(n_val) * PARAM_P_kna + (i_val)]; \
    const double g_K1  = (parameters_ptr)[(n_val) * PARAM_g_K1  + (i_val)]; \
    const double g_Kr  = (parameters_ptr)[(n_val) * PARAM_g_Kr  + (i_val)]; \
    const double g_Ks  = (parameters_ptr)[(n_val) * PARAM_g_Ks  + (i_val)]; \
    const double g_Na  = (parameters_ptr)[(n_val) * PARAM_g_Na  + (i_val)]; \
    const double g_bna = (parameters_ptr)[(n_val) * PARAM_g_bna + (i_val)]; \
    const double g_CaL = (parameters_ptr)[(n_val) * PARAM_g_CaL + (i_val)]; \
    const double g_bca = (parameters_ptr)[(n_val) * PARAM_g_bca + (i_val)]; \
    const double g_to  = (parameters_ptr)[(n_val) * PARAM_g_to  + (i_val)]; \
    const double K_mNa = (parameters_ptr)[(n_val) * PARAM_K_mNa + (i_val)]; \
    const double K_mk  = (parameters_ptr)[(n_val) * PARAM_K_mk  + (i_val)]; \
    const double P_NaK = (parameters_ptr)[(n_val) * PARAM_P_NaK + (i_val)]; \
    const double K_NaCa  = (parameters_ptr)[(n_val) * PARAM_K_NaCa  + (i_val)]; \
    const double K_sat   = (parameters_ptr)[(n_val) * PARAM_K_sat   + (i_val)]; \
    const double Km_Ca   = (parameters_ptr)[(n_val) * PARAM_Km_Ca   + (i_val)]; \
    const double Km_Nai  = (parameters_ptr)[(n_val) * PARAM_Km_Nai  + (i_val)]; \
    const double alpha   = (parameters_ptr)[(n_val) * PARAM_alpha   + (i_val)]; \
    const double gamma   = (parameters_ptr)[(n_val) * PARAM_gamma   + (i_val)]; \
    const double K_pCa   = (parameters_ptr)[(n_val) * PARAM_K_pCa   + (i_val)]; \
    const double g_pCa   = (parameters_ptr)[(n_val) * PARAM_g_pCa   + (i_val)]; \
    const double g_pK    = (parameters_ptr)[(n_val) * PARAM_g_pK    + (i_val)]; \
    const double Buf_c   = (parameters_ptr)[(n_val) * PARAM_Buf_c   + (i_val)]; \
    const double Buf_sr  = (parameters_ptr)[(n_val) * PARAM_Buf_sr  + (i_val)]; \
    const double Buf_ss  = (parameters_ptr)[(n_val) * PARAM_Buf_ss  + (i_val)]; \
    const double Ca_o    = (parameters_ptr)[(n_val) * PARAM_Ca_o    + (i_val)]; \
    const double EC      = (parameters_ptr)[(n_val) * PARAM_EC      + (i_val)]; \
    const double K_buf_c = (parameters_ptr)[(n_val) * PARAM_K_buf_c + (i_val)]; \
    const double K_buf_sr = (parameters_ptr)[(n_val) * PARAM_K_buf_sr + (i_val)]; \
    const double K_buf_ss = (parameters_ptr)[(n_val) * PARAM_K_buf_ss + (i_val)]; \
    const double K_up    = (parameters_ptr)[(n_val) * PARAM_K_up    + (i_val)]; \
    const double V_leak  = (parameters_ptr)[(n_val) * PARAM_V_leak  + (i_val)]; \
    const double V_rel   = (parameters_ptr)[(n_val) * PARAM_V_rel   + (i_val)]; \
    const double V_sr    = (parameters_ptr)[(n_val) * PARAM_V_sr    + (i_val)]; \
    const double V_ss    = (parameters_ptr)[(n_val) * PARAM_V_ss    + (i_val)]; \
    const double V_xfer  = (parameters_ptr)[(n_val) * PARAM_V_xfer  + (i_val)]; \
    const double Vmax_up = (parameters_ptr)[(n_val) * PARAM_Vmax_up + (i_val)]; \
    const double k1_prime = (parameters_ptr)[(n_val) * PARAM_k1_prime + (i_val)]; \
    const double k2_prime = (parameters_ptr)[(n_val) * PARAM_k2_prime + (i_val)]; \
    const double k3      = (parameters_ptr)[(n_val) * PARAM_k3      + (i_val)]; \
    const double k4      = (parameters_ptr)[(n_val) * PARAM_k4      + (i_val)]; \
    const double max_sr  = (parameters_ptr)[(n_val) * PARAM_max_sr  + (i_val)]; \
    const double min_sr  = (parameters_ptr)[(n_val) * PARAM_min_sr  + (i_val)]; \
    const double Na_o    = (parameters_ptr)[(n_val) * PARAM_Na_o    + (i_val)]; \
    const double Cm      = (parameters_ptr)[(n_val) * PARAM_Cm      + (i_val)]; \
    const double F       = (parameters_ptr)[(n_val) * PARAM_F       + (i_val)]; \
    const double R_gas   = (parameters_ptr)[(n_val) * PARAM_R       + (i_val)]; \
    const double T       = (parameters_ptr)[(n_val) * PARAM_T       + (i_val)]; \
    const double V_c     = (parameters_ptr)[(n_val) * PARAM_V_c     + (i_val)]; \
    const double stim_amplitude = (parameters_ptr)[(n_val) * PARAM_stim_amplitude + (i_val)]; \
    const double stim_duration  = (parameters_ptr)[(n_val) * PARAM_stim_duration  + (i_val)]; \
    const double stim_period    = (parameters_ptr)[(n_val) * PARAM_stim_period    + (i_val)]; \
    const double stim_start     = (parameters_ptr)[(n_val) * PARAM_stim_start     + (i_val)]; \
    const double K_o    = (parameters_ptr)[(n_val) * PARAM_K_o      + (i_val)]; \
    const double E_Na = R_gas*T*log(Na_o/Na_i)/F; \
    const double E_K  = R_gas*T*log(K_o/K_i)/F; \
    const double E_Ks = R_gas*T*log((K_o + Na_o*P_kna)/(P_kna*Na_i + K_i))/F; \
    const double E_Ca = 0.5*R_gas*T*log(Ca_o/Ca_i)/F; \
    const double alpha_K1 = 0.1/(1. + 6.14421235332821e-6*exp(0.06*V - 0.06*E_K)); \
    const double beta_K1 = (0.367879441171442*exp(0.1*V - 0.1*E_K) + \
        3.06060402008027*exp(0.0002*V - 0.0002*E_K))/(1. + exp(0.5*E_K - 0.5*V)); \
    const double xK1_inf = alpha_K1/(alpha_K1 + beta_K1); \
    const double i_K1 = 0.430331482911935*g_K1*sqrt(K_o)*(-E_K + V)*xK1_inf; \
    const double i_Kr = 0.430331482911935*g_Kr*sqrt(K_o)*(-E_K + V)*Xr1*Xr2; \
    const double xr1_inf = 1.0/(1. + exp(-26./7. - V/7.)); \
    const double alpha_xr1 = 450./(1. + exp(-9./2. - V/10.)); \
    const double beta_xr1  = 6./(1. + 13.5813245225782*exp(0.0869565217391304*V)); \
    const double tau_xr1   = alpha_xr1*beta_xr1; \
    const double dXr1_dt   = (-Xr1 + xr1_inf)/tau_xr1; \
    const double dXr1_dt_linearized = -1./tau_xr1; \
    (states_ptr)[(n_val)*STATE_Xr1+(i_val)] = (fabs(dXr1_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*dXr1_dt_linearized))*dXr1_dt/dXr1_dt_linearized : (dt_val)*dXr1_dt) + Xr1; \
    const double xr2_inf = 1.0/(1. + exp(11./3. + V/24.)); \
    const double alpha_xr2 = 3./(1. + exp(-3. - V/20.)); \
    const double beta_xr2  = 1.12/(1. + exp(-3. + V/20.)); \
    const double tau_xr2   = alpha_xr2*beta_xr2; \
    const double dXr2_dt   = (-Xr2 + xr2_inf)/tau_xr2; \
    const double dXr2_dt_linearized = -1./tau_xr2; \
    (states_ptr)[(n_val)*STATE_Xr2+(i_val)] = (fabs(dXr2_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*dXr2_dt_linearized))*dXr2_dt/dXr2_dt_linearized : (dt_val)*dXr2_dt) + Xr2; \
    const double i_Ks = g_Ks*(Xs*Xs)*(-E_Ks + V); \
    const double xs_inf = 1.0/(1. + exp(-5./14. - V/14.)); \
    const double alpha_xs = 1400./sqrt(1. + exp(5./6. - V/6.)); \
    const double beta_xs  = 1.0/(1. + exp(-7./3. + V/15.)); \
    const double tau_xs   = 80. + alpha_xs*beta_xs; \
    const double dXs_dt   = (-Xs + xs_inf)/tau_xs; \
    const double dXs_dt_linearized = -1./tau_xs; \
    (states_ptr)[(n_val)*STATE_Xs+(i_val)] = (fabs(dXs_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*dXs_dt_linearized))*dXs_dt/dXs_dt_linearized : (dt_val)*dXs_dt) + Xs; \
    const double i_Na = g_Na*(m*m*m)*(-E_Na + V)*h*j; \
    const double m_inf = 1.0/((1. + 0.00184221158116513*exp(-0.110741971207087*V))*(1. + 0.00184221158116513*exp(-0.110741971207087*V))); \
    const double alpha_m = 1.0/(1. + exp(-12. - V/5.)); \
    const double beta_m  = 0.1/(1. + exp(7. + V/5.)) + 0.1/(1. + exp(-1./4. + V/200.)); \
    const double tau_m   = alpha_m*beta_m; \
    const double dm_dt   = (-m + m_inf)/tau_m; \
    const double dm_dt_linearized = -1./tau_m; \
    (states_ptr)[(n_val)*STATE_m+(i_val)] = (fabs(dm_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*dm_dt_linearized))*dm_dt/dm_dt_linearized : (dt_val)*dm_dt) + m; \
    const double h_inf = 1.0/((1. + 15212.5932856544*exp(0.134589502018843*V))*(1. + 15212.5932856544*exp(0.134589502018843*V))); \
    const double alpha_h = (V < -40. ? 4.43126792958051e-7*exp(-0.147058823529412*V) : 0.); \
    const double beta_h  = (V < -40. ? 310000.*exp(0.3485*V) + 2.7*exp(0.079*V) : \
        0.77/(0.13 + 0.0497581410839387*exp(-0.0900900900900901*V))); \
    const double tau_h   = 1.0/(alpha_h + beta_h); \
    const double dh_dt   = (-h + h_inf)/tau_h; \
    const double dh_dt_linearized = -1./tau_h; \
    (states_ptr)[(n_val)*STATE_h+(i_val)] = (fabs(dh_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*dh_dt_linearized))*dh_dt/dh_dt_linearized : (dt_val)*dh_dt) + h; \
    const double j_inf = 1.0/((1. + 15212.5932856544*exp(0.134589502018843*V))*(1. + 15212.5932856544*exp(0.134589502018843*V))); \
    const double alpha_j = (V < -40. ? (37.78 + V)*(-25428.*exp(0.2444*V) - 6.948e-6*exp(-0.04391*V))/(1. + 50262745825.954*exp(0.311*V)) : 0.); \
    const double beta_j  = (V < -40. ? 0.02424*exp(-0.01052*V)/(1. + 0.00396086833990426*exp(-0.1378*V)) : \
        0.6*exp(0.057*V)/(1. + 0.0407622039783662*exp(-0.1*V))); \
    const double tau_j   = 1.0/(alpha_j + beta_j); \
    const double dj_dt   = (-j + j_inf)/tau_j; \
    const double dj_dt_linearized = -1./tau_j; \
    (states_ptr)[(n_val)*STATE_j+(i_val)] = (fabs(dj_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*dj_dt_linearized))*dj_dt/dj_dt_linearized : (dt_val)*dj_dt) + j; \
    const double i_b_Na = g_bna*(-E_Na + V); \
    const double V_eff = (fabs(-15. + V) < 0.01 ? 0.01 : -15. + V); \
    const double i_CaL = 4.*g_CaL*(F*F)*(-Ca_o + 0.25*Ca_ss*exp(2.*F*V_eff/(R_gas*T)))*V_eff*d*f*f2*fCass/(R_gas*T*(-1. + exp(2.*F*V_eff/(R_gas*T)))); \
    const double d_inf = 1.0/(1. + 0.344153786865412*exp(-0.133333333333333*V)); \
    const double alpha_d = 0.25 + 1.4/(1. + exp(-35./13. - V/13.)); \
    const double beta_d  = 1.4/(1. + exp(1. + V/5.)); \
    const double gamma_d = 1.0/(1. + exp(5./2. - V/20.)); \
    const double tau_d   = alpha_d*beta_d + gamma_d; \
    const double dd_dt   = (-d + d_inf)/tau_d; \
    const double dd_dt_linearized = -1./tau_d; \
    (states_ptr)[(n_val)*STATE_d+(i_val)] = (fabs(dd_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*dd_dt_linearized))*dd_dt/dd_dt_linearized : (dt_val)*dd_dt) + d; \
    const double f_inf = 1.0/(1. + exp(20./7. + V/7.)); \
    const double tau_f = 20. + 180./(1. + exp(3. + V/10.)) + 200./(1. + exp(13./10. - V/10.)) + 1102.5*exp(-((27. + V)*(27. + V))/225.); \
    const double df_dt = (-f + f_inf)/tau_f; \
    const double df_dt_linearized = -1./tau_f; \
    (states_ptr)[(n_val)*STATE_f+(i_val)] = (fabs(df_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*df_dt_linearized))*df_dt/df_dt_linearized : (dt_val)*df_dt) + f; \
    const double f2_inf = 0.33 + 0.67/(1. + exp(5. + V/7.)); \
    const double tau_f2 = 31./(1. + exp(5./2. - V/10.)) + 80./(1. + exp(3. + V/10.)) + 562.*exp(-((27. + V)*(27. + V))/240.); \
    const double df2_dt = (-f2 + f2_inf)/tau_f2; \
    const double df2_dt_linearized = -1./tau_f2; \
    (states_ptr)[(n_val)*STATE_f2+(i_val)] = (fabs(df2_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*df2_dt_linearized))*df2_dt/df2_dt_linearized : (dt_val)*df2_dt) + f2; \
    const double fCass_inf = 0.4 + 0.6/(1. + 400.0*(Ca_ss*Ca_ss)); \
    const double tau_fCass = 2. + 80./(1. + 400.0*(Ca_ss*Ca_ss)); \
    const double dfCass_dt = (-fCass + fCass_inf)/tau_fCass; \
    const double dfCass_dt_linearized = -1./tau_fCass; \
    (states_ptr)[(n_val)*STATE_fCass+(i_val)] = (fabs(dfCass_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*dfCass_dt_linearized))*dfCass_dt/dfCass_dt_linearized : (dt_val)*dfCass_dt) + fCass; \
    const double i_b_Ca = g_bca*(-E_Ca + V); \
    const double i_to   = g_to*(-E_K + V)*r*s; \
    const double s_inf  = 1.0/(1. + exp(4. + V/5.)); \
    const double tau_s  = 3. + 5./(1. + exp(-4. + V/5.)) + 85.*exp(-((45. + V)*(45. + V))/320.); \
    const double ds_dt  = (-s + s_inf)/tau_s; \
    const double ds_dt_linearized = -1./tau_s; \
    (states_ptr)[(n_val)*STATE_s+(i_val)] = (fabs(ds_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*ds_dt_linearized))*ds_dt/ds_dt_linearized : (dt_val)*ds_dt) + s; \
    const double r_inf  = 1.0/(1. + exp(10./3. - V/6.)); \
    const double tau_r  = 0.8 + 9.5*exp(-((40. + V)*(40. + V))/1800.); \
    const double dr_dt  = (-r + r_inf)/tau_r; \
    const double dr_dt_linearized = -1./tau_r; \
    (states_ptr)[(n_val)*STATE_r+(i_val)] = (fabs(dr_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*dr_dt_linearized))*dr_dt/dr_dt_linearized : (dt_val)*dr_dt) + r; \
    const double i_NaK = K_o*P_NaK*Na_i/((K_mNa + Na_i)*(K_mk + K_o)*(1. + \
        0.0353*exp(-F*V/(R_gas*T)) + 0.1245*exp(-0.1*F*V/(R_gas*T)))); \
    const double i_NaCa = K_NaCa*(Ca_o*(Na_i*Na_i*Na_i)*exp(F*gamma*V/(R_gas*T)) - \
        alpha*(Na_o*Na_o*Na_o)*Ca_i*exp(F*(-1. + gamma)*V/(R_gas*T)))/((1. + \
        K_sat*exp(F*(-1. + gamma)*V/(R_gas*T)))*(Ca_o + Km_Ca)*((Km_Nai*Km_Nai*Km_Nai) + (Na_o*Na_o*Na_o))); \
    const double i_p_Ca = g_pCa*Ca_i/(K_pCa + Ca_i); \
    const double i_p_K  = g_pK*(-E_K + V)/(1. + 65.4052157419383*exp(-0.167224080267559*V)); \
    const double i_up   = Vmax_up/(1. + (K_up*K_up)/(Ca_i*Ca_i)); \
    const double i_leak = V_leak*(-Ca_i + Ca_SR); \
    const double i_xfer = V_xfer*(-Ca_i + Ca_ss); \
    const double kcasr  = max_sr - (max_sr - min_sr)/(1. + (EC*EC)/(Ca_SR*Ca_SR)); \
    const double Ca_i_bufc    = 1.0/(1. + Buf_c*K_buf_c/((K_buf_c + Ca_i)*(K_buf_c + Ca_i))); \
    const double Ca_sr_bufsr  = 1.0/(1. + Buf_sr*K_buf_sr/((K_buf_sr + Ca_SR)*(K_buf_sr + Ca_SR))); \
    const double Ca_ss_bufss  = 1.0/(1. + Buf_ss*K_buf_ss/((K_buf_ss + Ca_ss)*(K_buf_ss + Ca_ss))); \
    const double dCa_i_dt = (V_sr*(-i_up + i_leak)/V_c - Cm*(-2.*i_NaCa + i_b_Ca + i_p_Ca)/(2.*F*V_c) + i_xfer)*Ca_i_bufc; \
    const double dCa_i_bufc_dCa_i = 2.*Buf_c*K_buf_c/(((1. + Buf_c*K_buf_c/((K_buf_c + Ca_i)*(K_buf_c + Ca_i)))*(1. + Buf_c*K_buf_c/((K_buf_c + Ca_i)*(K_buf_c + Ca_i))))*((K_buf_c + Ca_i)*(K_buf_c + Ca_i)*(K_buf_c + Ca_i))); \
    const double di_NaCa_dCa_i = -K_NaCa*alpha*(Na_o*Na_o*Na_o)*exp(F*(-1. + gamma)*V/(R_gas*T))/((1. + K_sat*exp(F*(-1. + gamma)*V/(R_gas*T)))*(Ca_o + Km_Ca)*((Km_Nai*Km_Nai*Km_Nai) + (Na_o*Na_o*Na_o))); \
    const double di_up_dCa_i = 2.*Vmax_up*(K_up*K_up)/(((1. + (K_up*K_up)/(Ca_i*Ca_i))*(1. + (K_up*K_up)/(Ca_i*Ca_i)))*(Ca_i*Ca_i*Ca_i)); \
    const double di_p_Ca_dCa_i = g_pCa/(K_pCa + Ca_i) - g_pCa*Ca_i/((K_pCa + Ca_i)*(K_pCa + Ca_i)); \
    const double dE_Ca_dCa_i = -0.5*R_gas*T/(F*Ca_i); \
    const double dCa_i_dt_linearized = (-V_xfer + V_sr*(-V_leak - di_up_dCa_i)/V_c - \
        Cm*(-2.*di_NaCa_dCa_i - g_bca*dE_Ca_dCa_i + di_p_Ca_dCa_i)/(2.*F*V_c))*Ca_i_bufc + \
        (V_sr*(-i_up + i_leak)/V_c - Cm*(-2.*i_NaCa + i_b_Ca + i_p_Ca)/(2.*F*V_c) + i_xfer)*dCa_i_bufc_dCa_i; \
    (states_ptr)[(n_val)*STATE_Ca_i+(i_val)] = Ca_i + (fabs(dCa_i_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*dCa_i_dt_linearized))*dCa_i_dt/dCa_i_dt_linearized : (dt_val)*dCa_i_dt); \
    const double k1 = k1_prime/kcasr; \
    const double k2 = k2_prime*kcasr; \
    const double O  = (Ca_ss*Ca_ss)*R_prime*k1/(k3 + (Ca_ss*Ca_ss)*k1); \
    const double dR_prime_dt = k4*(1. - R_prime) - Ca_ss*R_prime*k2; \
    const double dR_prime_dt_linearized = -k4 - Ca_ss*k2; \
    (states_ptr)[(n_val)*STATE_R_prime+(i_val)] = (fabs(dR_prime_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*dR_prime_dt_linearized))*dR_prime_dt/dR_prime_dt_linearized : (dt_val)*dR_prime_dt) + R_prime; \
    const double i_rel = V_rel*(-Ca_ss + Ca_SR)*O; \
    const double dCa_SR_dt = (-i_leak - i_rel + i_up)*Ca_sr_bufsr; \
    const double dkcasr_dCa_SR = -2.*(EC*EC)*(max_sr - min_sr)/(((1. + (EC*EC)/(Ca_SR*Ca_SR))*(1. + (EC*EC)/(Ca_SR*Ca_SR)))*(Ca_SR*Ca_SR*Ca_SR)); \
    const double dCa_sr_bufsr_dCa_SR = 2.*Buf_sr*K_buf_sr/(((1. + Buf_sr*K_buf_sr/((K_buf_sr + Ca_SR)*(K_buf_sr + Ca_SR)))*(1. + Buf_sr*K_buf_sr/((K_buf_sr + Ca_SR)*(K_buf_sr + Ca_SR))))*((K_buf_sr + Ca_SR)*(K_buf_sr + Ca_SR)*(K_buf_sr + Ca_SR))); \
    const double di_rel_dO = V_rel*(-Ca_ss + Ca_SR); \
    const double dk1_dkcasr = -k1_prime/(kcasr*kcasr); \
    const double dO_dk1 = (Ca_ss*Ca_ss)*R_prime/(k3 + (Ca_ss*Ca_ss)*k1) - pow(Ca_ss, 4.)*R_prime*k1/((k3 + (Ca_ss*Ca_ss)*k1)*(k3 + (Ca_ss*Ca_ss)*k1)); \
    const double di_rel_dCa_SR = V_rel*O + V_rel*(-Ca_ss + Ca_SR)*dO_dk1*dk1_dkcasr*dkcasr_dCa_SR; \
    const double dCa_SR_dt_linearized = (-V_leak - di_rel_dCa_SR - dO_dk1*di_rel_dO*dk1_dkcasr*dkcasr_dCa_SR)*Ca_sr_bufsr + (-i_leak - i_rel + i_up)*dCa_sr_bufsr_dCa_SR; \
    (states_ptr)[(n_val)*STATE_Ca_SR+(i_val)] = Ca_SR + (fabs(dCa_SR_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*dCa_SR_dt_linearized))*dCa_SR_dt/dCa_SR_dt_linearized : (dt_val)*dCa_SR_dt); \
    const double dCa_ss_dt = (V_sr*i_rel/V_ss - V_c*i_xfer/V_ss - Cm*i_CaL/(2.*F*V_ss))*Ca_ss_bufss; \
    const double dO_dCa_ss = -2.*(Ca_ss*Ca_ss*Ca_ss)*(k1*k1)*R_prime/((k3 + (Ca_ss*Ca_ss)*k1)*(k3 + (Ca_ss*Ca_ss)*k1)) + 2.*Ca_ss*R_prime*k1/(k3 + (Ca_ss*Ca_ss)*k1); \
    const double di_rel_dCa_ss = -V_rel*O + V_rel*(-Ca_ss + Ca_SR)*dO_dCa_ss; \
    const double dCa_ss_bufss_dCa_ss = 2.*Buf_ss*K_buf_ss/(((1. + Buf_ss*K_buf_ss/((K_buf_ss + Ca_ss)*(K_buf_ss + Ca_ss)))*(1. + Buf_ss*K_buf_ss/((K_buf_ss + Ca_ss)*(K_buf_ss + Ca_ss))))*((K_buf_ss + Ca_ss)*(K_buf_ss + Ca_ss)*(K_buf_ss + Ca_ss))); \
    const double di_CaL_dCa_ss = 1.0*g_CaL*(F*F)*V_eff*d*exp(2.*F*V_eff/(R_gas*T))*f*f2*fCass/(R_gas*T*(-1. + exp(2.*F*V_eff/(R_gas*T)))); \
    const double dCa_ss_dt_linearized = (V_sr*(dO_dCa_ss*di_rel_dO + di_rel_dCa_ss)/V_ss - V_c*V_xfer/V_ss - Cm*di_CaL_dCa_ss/(2.*F*V_ss))*Ca_ss_bufss + (V_sr*i_rel/V_ss - V_c*i_xfer/V_ss - Cm*i_CaL/(2.*F*V_ss))*dCa_ss_bufss_dCa_ss; \
    (states_ptr)[(n_val)*STATE_Ca_ss+(i_val)] = Ca_ss + (fabs(dCa_ss_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*dCa_ss_dt_linearized))*dCa_ss_dt/dCa_ss_dt_linearized : (dt_val)*dCa_ss_dt); \
    const double dNa_i_dt = Cm*(-i_Na - i_b_Na - 3.*i_NaCa - 3.*i_NaK)/(F*V_c); \
    const double dE_Na_dNa_i = -R_gas*T/(F*Na_i); \
    const double di_NaCa_dNa_i = 3.*Ca_o*K_NaCa*(Na_i*Na_i)*exp(F*gamma*V/(R_gas*T))/((1. + K_sat*exp(F*(-1. + gamma)*V/(R_gas*T)))*(Ca_o + Km_Ca)*((Km_Nai*Km_Nai*Km_Nai) + (Na_o*Na_o*Na_o))); \
    const double di_Na_dE_Na = -g_Na*(m*m*m)*h*j; \
    const double di_NaK_dNa_i = K_o*P_NaK/((K_mNa + Na_i)*(K_mk + K_o)*(1. + 0.0353*exp(-F*V/(R_gas*T)) + 0.1245*exp(-0.1*F*V/(R_gas*T)))) - K_o*P_NaK*Na_i/(((K_mNa + Na_i)*(K_mNa + Na_i))*(K_mk + K_o)*(1. + 0.0353*exp(-F*V/(R_gas*T)) + 0.1245*exp(-0.1*F*V/(R_gas*T)))); \
    const double dNa_i_dt_linearized = Cm*(-3.*di_NaCa_dNa_i - 3.*di_NaK_dNa_i + g_bna*dE_Na_dNa_i - dE_Na_dNa_i*di_Na_dE_Na)/(F*V_c); \
    (states_ptr)[(n_val)*STATE_Na_i+(i_val)] = Na_i + (fabs(dNa_i_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*dNa_i_dt_linearized))*dNa_i_dt/dNa_i_dt_linearized : (dt_val)*dNa_i_dt); \
    const double i_Stim = ((t_val) - stim_period*floor((t_val)/stim_period) <= stim_duration + stim_start && \
        (t_val) - stim_period*floor((t_val)/stim_period) >= stim_start ? -stim_amplitude : 0.); \
    const double dV_dt = -i_CaL - i_K1 - i_Kr - i_Ks - i_Na - i_NaCa - i_NaK - i_Stim - i_b_Ca - i_b_Na - i_p_Ca - i_p_K - i_to; \
    const double dalpha_K1_dV = -3.68652741199693e-8*exp(0.06*V - 0.06*E_K)/((1. + 6.14421235332821e-6*exp(0.06*V - 0.06*E_K))*(1. + 6.14421235332821e-6*exp(0.06*V - 0.06*E_K))); \
    const double di_CaL_dV_eff = 4.*g_CaL*(F*F)*(-Ca_o + 0.25*Ca_ss*exp(2.*F*V_eff/(R_gas*T)))*d*f*f2*fCass/(R_gas*T*(-1. + exp(2.*F*V_eff/(R_gas*T)))) - 8.*g_CaL*(F*F*F)*(-Ca_o + 0.25*Ca_ss*exp(2.*F*V_eff/(R_gas*T)))*V_eff*d*exp(2.*F*V_eff/(R_gas*T))*f*f2*fCass/((R_gas*R_gas)*(T*T)*((-1. + exp(2.*F*V_eff/(R_gas*T)))*(-1. + exp(2.*F*V_eff/(R_gas*T))))) + 2.0*g_CaL*(F*F*F)*Ca_ss*V_eff*d*exp(2.*F*V_eff/(R_gas*T))*f*f2*fCass/((R_gas*R_gas)*(T*T)*(-1. + exp(2.*F*V_eff/(R_gas*T)))); \
    const double di_Ks_dV = g_Ks*(Xs*Xs); \
    const double di_p_K_dV = g_pK/(1. + 65.4052157419383*exp(-0.167224080267559*V)) + 10.9373270471469*g_pK*(-E_K + V)*exp(-0.167224080267559*V)/((1. + 65.4052157419383*exp(-0.167224080267559*V))*(1. + 65.4052157419383*exp(-0.167224080267559*V))); \
    const double di_to_dV = g_to*r*s; \
    const double dxK1_inf_dbeta_K1 = -alpha_K1/((alpha_K1 + beta_K1)*(alpha_K1 + beta_K1)); \
    const double dxK1_inf_dalpha_K1 = 1.0/(alpha_K1 + beta_K1) - alpha_K1/((alpha_K1 + beta_K1)*(alpha_K1 + beta_K1)); \
    const double dbeta_K1_dV = (0.000612120804016053*exp(0.0002*V - 0.0002*E_K) + 0.0367879441171442*exp(0.1*V - 0.1*E_K))/(1. + exp(0.5*E_K - 0.5*V)) + 0.5*(0.367879441171442*exp(0.1*V - 0.1*E_K) + 3.06060402008027*exp(0.0002*V - 0.0002*E_K))*exp(0.5*E_K - 0.5*V)/((1. + exp(0.5*E_K - 0.5*V))*(1. + exp(0.5*E_K - 0.5*V))); \
    const double di_K1_dV = 0.430331482911935*g_K1*sqrt(K_o)*xK1_inf + 0.430331482911935*g_K1*sqrt(K_o)*(-E_K + V)*(dalpha_K1_dV*dxK1_inf_dalpha_K1 + dbeta_K1_dV*dxK1_inf_dbeta_K1); \
    const double dV_eff_dV = (fabs(-15. + V) < 0.01 ? 0. : 1.); \
    const double di_Na_dV = g_Na*(m*m*m)*h*j; \
    const double di_Kr_dV = 0.430331482911935*g_Kr*sqrt(K_o)*Xr1*Xr2; \
    const double di_NaK_dV = K_o*P_NaK*(0.0353*F*exp(-F*V/(R_gas*T))/(R_gas*T) + 0.01245*F*exp(-0.1*F*V/(R_gas*T))/(R_gas*T))*Na_i/((K_mNa + Na_i)*(K_mk + K_o)*((1. + 0.0353*exp(-F*V/(R_gas*T)) + 0.1245*exp(-0.1*F*V/(R_gas*T)))*(1. + 0.0353*exp(-F*V/(R_gas*T)) + 0.1245*exp(-0.1*F*V/(R_gas*T))))); \
    const double di_K1_dxK1_inf = 0.430331482911935*g_K1*sqrt(K_o)*(-E_K + V); \
    const double di_NaCa_dV = K_NaCa*(Ca_o*F*gamma*(Na_i*Na_i*Na_i)*exp(F*gamma*V/(R_gas*T))/(R_gas*T) - F*alpha*(Na_o*Na_o*Na_o)*(-1. + gamma)*Ca_i*exp(F*(-1. + gamma)*V/(R_gas*T))/(R_gas*T))/((1. + K_sat*exp(F*(-1. + gamma)*V/(R_gas*T)))*(Ca_o + Km_Ca)*((Km_Nai*Km_Nai*Km_Nai) + (Na_o*Na_o*Na_o))) - F*K_NaCa*K_sat*(-1. + gamma)*(Ca_o*(Na_i*Na_i*Na_i)*exp(F*gamma*V/(R_gas*T)) - alpha*(Na_o*Na_o*Na_o)*Ca_i*exp(F*(-1. + gamma)*V/(R_gas*T)))*exp(F*(-1. + gamma)*V/(R_gas*T))/(R_gas*T*((1. + K_sat*exp(F*(-1. + gamma)*V/(R_gas*T)))*(1. + K_sat*exp(F*(-1. + gamma)*V/(R_gas*T))))*(Ca_o + Km_Ca)*((Km_Nai*Km_Nai*Km_Nai) + (Na_o*Na_o*Na_o))); \
    const double dV_dt_linearized = -g_bca - g_bna - di_K1_dV - di_Kr_dV - di_Ks_dV - di_NaCa_dV - di_NaK_dV - di_Na_dV - di_p_K_dV - di_to_dV - (dalpha_K1_dV*dxK1_inf_dalpha_K1 + dbeta_K1_dV*dxK1_inf_dbeta_K1)*di_K1_dxK1_inf - dV_eff_dV*di_CaL_dV_eff; \
    (states_ptr)[(n_val)*STATE_V+(i_val)] = (fabs(dV_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*dV_dt_linearized))*dV_dt/dV_dt_linearized : (dt_val)*dV_dt) + V; \
    const double dK_i_dt = Cm*(-i_K1 - i_Kr - i_Ks - i_Stim - i_p_K - i_to + 2.*i_NaK)/(F*V_c); \
    const double dE_Ks_dK_i = -R_gas*T/(F*(P_kna*Na_i + K_i)); \
    const double dbeta_K1_dE_K = (-0.000612120804016053*exp(0.0002*V - 0.0002*E_K) - 0.0367879441171442*exp(0.1*V - 0.1*E_K))/(1. + exp(0.5*E_K - 0.5*V)) - 0.5*(0.367879441171442*exp(0.1*V - 0.1*E_K) + 3.06060402008027*exp(0.0002*V - 0.0002*E_K))*exp(0.5*E_K - 0.5*V)/((1. + exp(0.5*E_K - 0.5*V))*(1. + exp(0.5*E_K - 0.5*V))); \
    const double di_Kr_dE_K = -0.430331482911935*g_Kr*sqrt(K_o)*Xr1*Xr2; \
    const double dE_K_dK_i = -R_gas*T/(F*K_i); \
    const double di_Ks_dE_Ks = -g_Ks*(Xs*Xs); \
    const double di_to_dE_K = -g_to*r*s; \
    const double dalpha_K1_dE_K = 3.68652741199693e-8*exp(0.06*V - 0.06*E_K)/((1. + 6.14421235332821e-6*exp(0.06*V - 0.06*E_K))*(1. + 6.14421235332821e-6*exp(0.06*V - 0.06*E_K))); \
    const double di_K1_dE_K = -0.430331482911935*g_K1*sqrt(K_o)*xK1_inf + 0.430331482911935*g_K1*sqrt(K_o)*(-E_K + V)*(dalpha_K1_dE_K*dxK1_inf_dalpha_K1 + dbeta_K1_dE_K*dxK1_inf_dbeta_K1); \
    const double di_p_K_dE_K = -g_pK/(1. + 65.4052157419383*exp(-0.167224080267559*V)); \
    const double dK_i_dt_linearized = Cm*(-(dE_K_dK_i*dalpha_K1_dE_K*dxK1_inf_dalpha_K1 + dE_K_dK_i*dbeta_K1_dE_K*dxK1_inf_dbeta_K1)*di_K1_dxK1_inf - dE_K_dK_i*di_K1_dE_K - dE_K_dK_i*di_Kr_dE_K - dE_K_dK_i*di_p_K_dE_K - dE_K_dK_i*di_to_dE_K - dE_Ks_dK_i*di_Ks_dE_Ks)/(F*V_c); \
    (states_ptr)[(n_val)*STATE_K_i+(i_val)] = K_i + (fabs(dK_i_dt_linearized) > 1.0e-8 ? \
        (-1.0 + exp((dt_val)*dK_i_dt_linearized))*dK_i_dt/dK_i_dt_linearized : (dt_val)*dK_i_dt); \
} while(0)

// ---- CPU reference ----
static void forward_rush_larsen(double* states, const double t, const double dt,
                                const double* parameters, const int n)
{
    for (int i = 0; i < n; i++) {
        RUSH_LARSEN_BODY(states, parameters, n, i, t, dt);
    }
}

int main(int argc, char* argv[])
{
    double t_start = 0;
    double dt = 0.02e-3;
    int num_timesteps = 1000000;
    int num_nodes = 1;

    if (argc > 1) {
        num_timesteps = atoi(argv[1]);
        printf("num_timesteps set to %d\n", num_timesteps);
        num_nodes = atoi(argv[2]);
        printf("num_nodes set to %d\n", num_nodes);
        if (num_timesteps <= 0 || num_nodes <= 0) return EXIT_FAILURE;
    }

    const size_t total_num_states = (size_t)num_nodes * NUM_STATES;
    double* states  = (double*)malloc(total_num_states * sizeof(double));
    double* states2 = (double*)malloc(total_num_states * sizeof(double));
    init_state_values(states, num_nodes);
    memcpy(states2, states, total_num_states * sizeof(double));

    const size_t total_num_params = (size_t)num_nodes * NUM_PARAMS;
    double* parameters = (double*)malloc(total_num_params * sizeof(double));
    init_parameters_values(parameters, num_nodes);

    // CPU reference
    printf("Host: Rush Larsen (exp integrator on all gates)\n");
    double t = t_start;
    for (int it = 0; it < num_timesteps; it++) {
        forward_rush_larsen(states, t, dt, parameters, num_nodes);
        t += dt;
    }

    // Device run with Kokkos
    printf("Device: Rush Larsen (exp integrator on all gates)\n");
    Kokkos::initialize(argc, argv);
    {
        using ExecSpace = Kokkos::DefaultExecutionSpace;
        using MemSpace  = ExecSpace::memory_space;

        Kokkos::View<double*, MemSpace> d_states    ("states",     total_num_states);
        Kokkos::View<double*, MemSpace> d_parameters("parameters", total_num_params);

        {
            auto h_states = Kokkos::create_mirror_view(d_states);
            auto h_params = Kokkos::create_mirror_view(d_parameters);
            for (size_t i = 0; i < total_num_states; i++) h_states(i) = states2[i];
            for (size_t i = 0; i < total_num_params; i++) h_params(i) = parameters[i];
            Kokkos::deep_copy(d_states,     h_states);
            Kokkos::deep_copy(d_parameters, h_params);
        }

        t = t_start;
        auto ts0 = std::chrono::steady_clock::now();

        for (int it = 0; it < num_timesteps; it++) {
            const double t_cur = t;
            const double dt_cur = dt;
            const int n = num_nodes;
            Kokkos::parallel_for("rushlarsen",
                Kokkos::RangePolicy<ExecSpace>(0, num_nodes),
                KOKKOS_LAMBDA(int i) {
                    RUSH_LARSEN_BODY(d_states, d_parameters, n, i, t_cur, dt_cur);
                });
            Kokkos::fence();
            t += dt;
        }

        auto ts1 = std::chrono::steady_clock::now();
        double time_elapsed = std::chrono::duration<double>(ts1 - ts0).count();
        printf("Device: computed %d time steps in %g s. Time steps per second: %g\n\n",
               num_timesteps, time_elapsed, num_timesteps / time_elapsed);

        // Copy result back
        auto h_states = Kokkos::create_mirror_view(d_states);
        Kokkos::deep_copy(h_states, d_states);
        for (size_t i = 0; i < total_num_states; i++) states2[i] = h_states(i);
    }
    Kokkos::finalize();

    double rmse = 0.0;
    for (size_t i = 0; i < total_num_states; i++) {
        double diff = states2[i] - states[i];
        rmse += diff * diff;
    }
    printf("RMSE = %lf\n", sqrt(rmse / total_num_states));

    free(states);
    free(states2);
    free(parameters);
    return 0;
}
