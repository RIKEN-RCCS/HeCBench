module rushlarsen_kernels
  use iso_fortran_env, only: real64
  use rushlarsen_utils
  implicit none
  !$omp declare target(rl_update,rush_larsen_cell)
contains
  pure real(real64) function rl_update(old, deriv, lin, dt) result(vnew)
    real(real64), intent(in) :: old, deriv, lin, dt

    if (abs(lin) > 1.0e-8_real64) then
      vnew = old + (-1.0_real64 + exp(dt * lin)) * deriv / lin
    else
      vnew = old + dt * deriv
    end if
  end function rl_update

  subroutine rush_larsen_cell(states, t, dt, parameters, n, i)
    real(real64), intent(inout) :: states(0:*)
    real(real64), intent(in) :: t, dt, parameters(0:*)
    integer, intent(in) :: n, i
    real(real64) :: xr1, xr2, xs, m_gate, h_gate, j_gate, d_gate, f_gate
    real(real64) :: f2_gate, fcass, s_gate, r_gate, ca_i, r_prime, ca_sr
    real(real64) :: ca_ss, na_i, v, k_i
    real(real64) :: p_kna, g_k1, g_kr, g_ks, g_na, g_bna, g_cal, g_bca
    real(real64) :: g_to, k_mna, k_mk, p_nak, k_naca, k_sat, km_ca, km_nai
    real(real64) :: alpha_p, gamma_p, k_pca, g_pca, g_pk, buf_c, buf_sr
    real(real64) :: buf_ss, ca_o, ec, k_buf_c, k_buf_sr, k_buf_ss, k_up
    real(real64) :: v_leak, v_rel, v_sr, v_ss, v_xfer, vmax_up, k1_prime
    real(real64) :: k2_prime, k3, k4, max_sr, min_sr, na_o, cm, ff, rr, tt
    real(real64) :: v_c, stim_amplitude, stim_duration, stim_period
    real(real64) :: stim_start, k_o
    real(real64) :: e_na, e_k, e_ks, e_ca, alpha_k1, beta_k1, xk1_inf
    real(real64) :: i_k1, i_kr, xr1_inf, alpha_xr1, beta_xr1, tau_xr1
    real(real64) :: dxr1_dt, dxr1_dt_linearized, xr2_inf, alpha_xr2
    real(real64) :: beta_xr2, tau_xr2, dxr2_dt, dxr2_dt_linearized, i_ks
    real(real64) :: xs_inf, alpha_xs, beta_xs, tau_xs, dxs_dt
    real(real64) :: dxs_dt_linearized, i_na, m_inf, alpha_m, beta_m, tau_m
    real(real64) :: dm_dt, dm_dt_linearized, h_inf, alpha_h, beta_h, tau_h
    real(real64) :: dh_dt, dh_dt_linearized, j_inf, alpha_j, beta_j, tau_j
    real(real64) :: dj_dt, dj_dt_linearized, i_b_na, v_eff, i_cal, d_inf
    real(real64) :: alpha_d, beta_d, gamma_d, tau_d, dd_dt, dd_dt_linearized
    real(real64) :: f_inf, tau_f, df_dt, df_dt_linearized, f2_inf, tau_f2
    real(real64) :: df2_dt, df2_dt_linearized, fcass_inf, tau_fcass
    real(real64) :: dfcass_dt, dfcass_dt_linearized, i_b_ca, i_to, s_inf
    real(real64) :: tau_s, ds_dt, ds_dt_linearized, r_inf, tau_r, dr_dt
    real(real64) :: dr_dt_linearized, i_nak, i_naca, i_p_ca, i_p_k, i_up
    real(real64) :: i_leak, i_xfer, kcasr, ca_i_bufc, ca_sr_bufsr, ca_ss_bufss
    real(real64) :: dca_i_dt, dca_i_bufc_dca_i, di_naca_dca_i, di_up_dca_i
    real(real64) :: di_p_ca_dca_i, de_ca_dca_i, dca_i_dt_linearized, k1, k2
    real(real64) :: o_open, d_r_prime_dt, d_r_prime_dt_linearized, i_rel
    real(real64) :: dca_sr_dt, dkcasr_dca_sr, dca_sr_bufsr_dca_sr, di_rel_do
    real(real64) :: dk1_dkcasr, do_dk1, di_rel_dca_sr, dca_sr_dt_linearized
    real(real64) :: dca_ss_dt, do_dca_ss, di_rel_dca_ss, dca_ss_bufss_dca_ss
    real(real64) :: di_cal_dca_ss, dca_ss_dt_linearized, dna_i_dt
    real(real64) :: de_na_dna_i, di_naca_dna_i, di_na_de_na, di_nak_dna_i
    real(real64) :: dna_i_dt_linearized, stim_phase, i_stim, dv_dt
    real(real64) :: dalpha_k1_dv, di_cal_dv_eff, di_ks_dv, di_p_k_dv
    real(real64) :: di_to_dv, dxk1_inf_dbeta_k1, dxk1_inf_dalpha_k1
    real(real64) :: dbeta_k1_dv, di_k1_dv, dv_eff_dv, di_na_dv, di_kr_dv
    real(real64) :: di_nak_dv, di_k1_dxk1_inf, di_naca_dv, dv_dt_linearized
    real(real64) :: dki_dt, de_ks_dki, dbeta_k1_de_k, di_kr_de_k, de_k_dki
    real(real64) :: di_ks_de_ks, di_to_de_k, dalpha_k1_de_k, di_k1_de_k
    real(real64) :: di_p_k_de_k, dki_dt_linearized

    xr1 = states(n * state_xr1 + i)
    xr2 = states(n * state_xr2 + i)
    xs = states(n * state_xs + i)
    m_gate = states(n * state_m + i)
    h_gate = states(n * state_h + i)
    j_gate = states(n * state_j + i)
    d_gate = states(n * state_d + i)
    f_gate = states(n * state_f + i)
    f2_gate = states(n * state_f2 + i)
    fcass = states(n * state_fcass + i)
    s_gate = states(n * state_s + i)
    r_gate = states(n * state_r + i)
    ca_i = states(n * state_ca_i + i)
    r_prime = states(n * state_r_prime + i)
    ca_sr = states(n * state_ca_sr + i)
    ca_ss = states(n * state_ca_ss + i)
    na_i = states(n * state_na_i + i)
    v = states(n * state_v + i)
    k_i = states(n * state_k_i + i)

    p_kna = parameters(n * param_p_kna + i)
    g_k1 = parameters(n * param_g_k1 + i)
    g_kr = parameters(n * param_g_kr + i)
    g_ks = parameters(n * param_g_ks + i)
    g_na = parameters(n * param_g_na + i)
    g_bna = parameters(n * param_g_bna + i)
    g_cal = parameters(n * param_g_cal + i)
    g_bca = parameters(n * param_g_bca + i)
    g_to = parameters(n * param_g_to + i)
    k_mna = parameters(n * param_k_mna + i)
    k_mk = parameters(n * param_k_mk + i)
    p_nak = parameters(n * param_p_nak + i)
    k_naca = parameters(n * param_k_naca + i)
    k_sat = parameters(n * param_k_sat + i)
    km_ca = parameters(n * param_km_ca + i)
    km_nai = parameters(n * param_km_nai + i)
    alpha_p = parameters(n * param_alpha + i)
    gamma_p = parameters(n * param_gamma + i)
    k_pca = parameters(n * param_k_pca + i)
    g_pca = parameters(n * param_g_pca + i)
    g_pk = parameters(n * param_g_pk + i)
    buf_c = parameters(n * param_buf_c + i)
    buf_sr = parameters(n * param_buf_sr + i)
    buf_ss = parameters(n * param_buf_ss + i)
    ca_o = parameters(n * param_ca_o + i)
    ec = parameters(n * param_ec + i)
    k_buf_c = parameters(n * param_k_buf_c + i)
    k_buf_sr = parameters(n * param_k_buf_sr + i)
    k_buf_ss = parameters(n * param_k_buf_ss + i)
    k_up = parameters(n * param_k_up + i)
    v_leak = parameters(n * param_v_leak + i)
    v_rel = parameters(n * param_v_rel + i)
    v_sr = parameters(n * param_v_sr + i)
    v_ss = parameters(n * param_v_ss + i)
    v_xfer = parameters(n * param_v_xfer + i)
    vmax_up = parameters(n * param_vmax_up + i)
    k1_prime = parameters(n * param_k1_prime + i)
    k2_prime = parameters(n * param_k2_prime + i)
    k3 = parameters(n * param_k3 + i)
    k4 = parameters(n * param_k4 + i)
    max_sr = parameters(n * param_max_sr + i)
    min_sr = parameters(n * param_min_sr + i)
    na_o = parameters(n * param_na_o + i)
    cm = parameters(n * param_cm + i)
    ff = parameters(n * param_fconst + i)
    rr = parameters(n * param_rconst + i)
    tt = parameters(n * param_t + i)
    v_c = parameters(n * param_v_c + i)
    stim_amplitude = parameters(n * param_stim_amplitude + i)
    stim_duration = parameters(n * param_stim_duration + i)
    stim_period = parameters(n * param_stim_period + i)
    stim_start = parameters(n * param_stim_start + i)
    k_o = parameters(n * param_k_o + i)

    e_na = rr * tt * log(na_o / na_i) / ff
    e_k = rr * tt * log(k_o / k_i) / ff
    e_ks = rr * tt * log((k_o + na_o * p_kna) / (p_kna * na_i + k_i)) / ff
    e_ca = 0.5_real64 * rr * tt * log(ca_o / ca_i) / ff

    alpha_k1 = 0.1_real64 / (1.0_real64 + 6.14421235332821e-6_real64 * &
      exp(0.06_real64 * v - 0.06_real64 * e_k))
    beta_k1 = (0.367879441171442_real64 * exp(0.1_real64 * v - 0.1_real64 * e_k) + &
      3.06060402008027_real64 * exp(0.0002_real64 * v - 0.0002_real64 * e_k)) / &
      (1.0_real64 + exp(0.5_real64 * e_k - 0.5_real64 * v))
    xk1_inf = alpha_k1 / (alpha_k1 + beta_k1)
    i_k1 = 0.430331482911935_real64 * g_k1 * sqrt(k_o) * (-e_k + v) * xk1_inf

    i_kr = 0.430331482911935_real64 * g_kr * sqrt(k_o) * (-e_k + v) * xr1 * xr2

    xr1_inf = 1.0_real64 / (1.0_real64 + exp(-26.0_real64 / 7.0_real64 - v / 7.0_real64))
    alpha_xr1 = 450.0_real64 / (1.0_real64 + exp(-9.0_real64 / 2.0_real64 - v / 10.0_real64))
    beta_xr1 = 6.0_real64 / (1.0_real64 + &
      13.5813245225782_real64 * exp(0.0869565217391304_real64 * v))
    tau_xr1 = alpha_xr1 * beta_xr1
    dxr1_dt = (-xr1 + xr1_inf) / tau_xr1
    dxr1_dt_linearized = -1.0_real64 / tau_xr1
    states(n * state_xr1 + i) = rl_update(xr1, dxr1_dt, dxr1_dt_linearized, dt)

    xr2_inf = 1.0_real64 / (1.0_real64 + exp(11.0_real64 / 3.0_real64 + v / 24.0_real64))
    alpha_xr2 = 3.0_real64 / (1.0_real64 + exp(-3.0_real64 - v / 20.0_real64))
    beta_xr2 = 1.12_real64 / (1.0_real64 + exp(-3.0_real64 + v / 20.0_real64))
    tau_xr2 = alpha_xr2 * beta_xr2
    dxr2_dt = (-xr2 + xr2_inf) / tau_xr2
    dxr2_dt_linearized = -1.0_real64 / tau_xr2
    states(n * state_xr2 + i) = rl_update(xr2, dxr2_dt, dxr2_dt_linearized, dt)

    i_ks = g_ks * (xs * xs) * (-e_ks + v)

    xs_inf = 1.0_real64 / (1.0_real64 + exp(-5.0_real64 / 14.0_real64 - v / 14.0_real64))
    alpha_xs = 1400.0_real64 / sqrt(1.0_real64 + exp(5.0_real64 / 6.0_real64 - v / 6.0_real64))
    beta_xs = 1.0_real64 / (1.0_real64 + exp(-7.0_real64 / 3.0_real64 + v / 15.0_real64))
    tau_xs = 80.0_real64 + alpha_xs * beta_xs
    dxs_dt = (-xs + xs_inf) / tau_xs
    dxs_dt_linearized = -1.0_real64 / tau_xs
    states(n * state_xs + i) = rl_update(xs, dxs_dt, dxs_dt_linearized, dt)

    i_na = g_na * (m_gate * m_gate * m_gate) * (-e_na + v) * h_gate * j_gate

    m_inf = 1.0_real64 / ((1.0_real64 + &
      0.00184221158116513_real64 * exp(-0.110741971207087_real64 * v)) * &
      (1.0_real64 + &
      0.00184221158116513_real64 * exp(-0.110741971207087_real64 * v)))
    alpha_m = 1.0_real64 / (1.0_real64 + exp(-12.0_real64 - v / 5.0_real64))
    beta_m = 0.1_real64 / (1.0_real64 + exp(7.0_real64 + v / 5.0_real64)) + &
      0.1_real64 / (1.0_real64 + exp(-1.0_real64 / 4.0_real64 + v / 200.0_real64))
    tau_m = alpha_m * beta_m
    dm_dt = (-m_gate + m_inf) / tau_m
    dm_dt_linearized = -1.0_real64 / tau_m
    states(n * state_m + i) = rl_update(m_gate, dm_dt, dm_dt_linearized, dt)

    h_inf = 1.0_real64 / ((1.0_real64 + &
      15212.5932856544_real64 * exp(0.134589502018843_real64 * v)) * &
      (1.0_real64 + &
      15212.5932856544_real64 * exp(0.134589502018843_real64 * v)))
    if (v < -40.0_real64) then
      alpha_h = 4.43126792958051e-7_real64 * exp(-0.147058823529412_real64 * v)
      beta_h = 310000.0_real64 * exp(0.3485_real64 * v) + &
        2.7_real64 * exp(0.079_real64 * v)
    else
      alpha_h = 0.0_real64
      beta_h = 0.77_real64 / (0.13_real64 + &
        0.0497581410839387_real64 * exp(-0.0900900900900901_real64 * v))
    end if
    tau_h = 1.0_real64 / (alpha_h + beta_h)
    dh_dt = (-h_gate + h_inf) / tau_h
    dh_dt_linearized = -1.0_real64 / tau_h
    states(n * state_h + i) = rl_update(h_gate, dh_dt, dh_dt_linearized, dt)

    j_inf = 1.0_real64 / ((1.0_real64 + &
      15212.5932856544_real64 * exp(0.134589502018843_real64 * v)) * &
      (1.0_real64 + &
      15212.5932856544_real64 * exp(0.134589502018843_real64 * v)))
    if (v < -40.0_real64) then
      alpha_j = (37.78_real64 + v) * (-25428.0_real64 * exp(0.2444_real64 * v) - &
        6.948e-6_real64 * exp(-0.04391_real64 * v)) / &
        (1.0_real64 + 50262745825.954_real64 * exp(0.311_real64 * v))
      beta_j = 0.02424_real64 * exp(-0.01052_real64 * v) / &
        (1.0_real64 + 0.00396086833990426_real64 * exp(-0.1378_real64 * v))
    else
      alpha_j = 0.0_real64
      beta_j = 0.6_real64 * exp(0.057_real64 * v) / &
        (1.0_real64 + 0.0407622039783662_real64 * exp(-0.1_real64 * v))
    end if
    tau_j = 1.0_real64 / (alpha_j + beta_j)
    dj_dt = (-j_gate + j_inf) / tau_j
    dj_dt_linearized = -1.0_real64 / tau_j
    states(n * state_j + i) = rl_update(j_gate, dj_dt, dj_dt_linearized, dt)

    i_b_na = g_bna * (-e_na + v)

    if (abs(-15.0_real64 + v) < 0.01_real64) then
      v_eff = 0.01_real64
    else
      v_eff = -15.0_real64 + v
    end if
    i_cal = 4.0_real64 * g_cal * (ff * ff) * (-ca_o + &
      0.25_real64 * ca_ss * exp(2.0_real64 * ff * v_eff / (rr * tt))) * &
      v_eff * d_gate * f_gate * f2_gate * fcass / &
      (rr * tt * (-1.0_real64 + exp(2.0_real64 * ff * v_eff / (rr * tt))))

    d_inf = 1.0_real64 / (1.0_real64 + &
      0.344153786865412_real64 * exp(-0.133333333333333_real64 * v))
    alpha_d = 0.25_real64 + 1.4_real64 / &
      (1.0_real64 + exp(-35.0_real64 / 13.0_real64 - v / 13.0_real64))
    beta_d = 1.4_real64 / (1.0_real64 + exp(1.0_real64 + v / 5.0_real64))
    gamma_d = 1.0_real64 / (1.0_real64 + exp(5.0_real64 / 2.0_real64 - v / 20.0_real64))
    tau_d = alpha_d * beta_d + gamma_d
    dd_dt = (-d_gate + d_inf) / tau_d
    dd_dt_linearized = -1.0_real64 / tau_d
    states(n * state_d + i) = rl_update(d_gate, dd_dt, dd_dt_linearized, dt)

    f_inf = 1.0_real64 / (1.0_real64 + exp(20.0_real64 / 7.0_real64 + v / 7.0_real64))
    tau_f = 20.0_real64 + 180.0_real64 / (1.0_real64 + exp(3.0_real64 + v / 10.0_real64)) + &
      200.0_real64 / (1.0_real64 + exp(13.0_real64 / 10.0_real64 - v / 10.0_real64)) + &
      1102.5_real64 * exp(-((27.0_real64 + v) * (27.0_real64 + v)) / 225.0_real64)
    df_dt = (-f_gate + f_inf) / tau_f
    df_dt_linearized = -1.0_real64 / tau_f
    states(n * state_f + i) = rl_update(f_gate, df_dt, df_dt_linearized, dt)

    f2_inf = 0.33_real64 + 0.67_real64 / (1.0_real64 + exp(5.0_real64 + v / 7.0_real64))
    tau_f2 = 31.0_real64 / (1.0_real64 + exp(5.0_real64 / 2.0_real64 - v / 10.0_real64)) + &
      80.0_real64 / (1.0_real64 + exp(3.0_real64 + v / 10.0_real64)) + &
      562.0_real64 * exp(-((27.0_real64 + v) * (27.0_real64 + v)) / 240.0_real64)
    df2_dt = (-f2_gate + f2_inf) / tau_f2
    df2_dt_linearized = -1.0_real64 / tau_f2
    states(n * state_f2 + i) = rl_update(f2_gate, df2_dt, df2_dt_linearized, dt)

    fcass_inf = 0.4_real64 + 0.6_real64 / (1.0_real64 + 400.0_real64 * (ca_ss * ca_ss))
    tau_fcass = 2.0_real64 + 80.0_real64 / (1.0_real64 + 400.0_real64 * (ca_ss * ca_ss))
    dfcass_dt = (-fcass + fcass_inf) / tau_fcass
    dfcass_dt_linearized = -1.0_real64 / tau_fcass
    states(n * state_fcass + i) = rl_update(fcass, dfcass_dt, dfcass_dt_linearized, dt)

    i_b_ca = g_bca * (-e_ca + v)

    i_to = g_to * (-e_k + v) * r_gate * s_gate

    s_inf = 1.0_real64 / (1.0_real64 + exp(4.0_real64 + v / 5.0_real64))
    tau_s = 3.0_real64 + 5.0_real64 / (1.0_real64 + exp(-4.0_real64 + v / 5.0_real64)) + &
      85.0_real64 * exp(-((45.0_real64 + v) * (45.0_real64 + v)) / 320.0_real64)
    ds_dt = (-s_gate + s_inf) / tau_s
    ds_dt_linearized = -1.0_real64 / tau_s
    states(n * state_s + i) = rl_update(s_gate, ds_dt, ds_dt_linearized, dt)

    r_inf = 1.0_real64 / (1.0_real64 + exp(10.0_real64 / 3.0_real64 - v / 6.0_real64))
    tau_r = 0.8_real64 + 9.5_real64 * &
      exp(-((40.0_real64 + v) * (40.0_real64 + v)) / 1800.0_real64)
    dr_dt = (-r_gate + r_inf) / tau_r
    dr_dt_linearized = -1.0_real64 / tau_r
    states(n * state_r + i) = rl_update(r_gate, dr_dt, dr_dt_linearized, dt)

    i_nak = k_o * p_nak * na_i / ((k_mna + na_i) * (k_mk + k_o) * &
      (1.0_real64 + 0.0353_real64 * exp(-ff * v / (rr * tt)) + &
      0.1245_real64 * exp(-0.1_real64 * ff * v / (rr * tt))))

    i_naca = k_naca * (ca_o * (na_i * na_i * na_i) * exp(ff * gamma_p * v / (rr * tt)) - &
      alpha_p * (na_o * na_o * na_o) * ca_i * exp(ff * (-1.0_real64 + gamma_p) * v / (rr * tt))) / &
      ((1.0_real64 + k_sat * exp(ff * (-1.0_real64 + gamma_p) * v / (rr * tt))) * &
      (ca_o + km_ca) * ((km_nai * km_nai * km_nai) + (na_o * na_o * na_o)))

    i_p_ca = g_pca * ca_i / (k_pca + ca_i)

    i_p_k = g_pk * (-e_k + v) / &
      (1.0_real64 + 65.4052157419383_real64 * exp(-0.167224080267559_real64 * v))

    i_up = vmax_up / (1.0_real64 + (k_up * k_up) / (ca_i * ca_i))
    i_leak = v_leak * (-ca_i + ca_sr)
    i_xfer = v_xfer * (-ca_i + ca_ss)
    kcasr = max_sr - (max_sr - min_sr) / (1.0_real64 + (ec * ec) / (ca_sr * ca_sr))
    ca_i_bufc = 1.0_real64 / (1.0_real64 + &
      buf_c * k_buf_c / ((k_buf_c + ca_i) * (k_buf_c + ca_i)))
    ca_sr_bufsr = 1.0_real64 / (1.0_real64 + &
      buf_sr * k_buf_sr / ((k_buf_sr + ca_sr) * (k_buf_sr + ca_sr)))
    ca_ss_bufss = 1.0_real64 / (1.0_real64 + &
      buf_ss * k_buf_ss / ((k_buf_ss + ca_ss) * (k_buf_ss + ca_ss)))
    dca_i_dt = (v_sr * (-i_up + i_leak) / v_c - cm * (-2.0_real64 * i_naca + &
      i_b_ca + i_p_ca) / (2.0_real64 * ff * v_c) + i_xfer) * ca_i_bufc
    dca_i_bufc_dca_i = 2.0_real64 * buf_c * k_buf_c / (((1.0_real64 + &
      buf_c * k_buf_c / ((k_buf_c + ca_i) * (k_buf_c + ca_i))) * (1.0_real64 + &
      buf_c * k_buf_c / ((k_buf_c + ca_i) * (k_buf_c + ca_i)))) * &
      ((k_buf_c + ca_i) * (k_buf_c + ca_i) * (k_buf_c + ca_i)))
    di_naca_dca_i = -k_naca * alpha_p * (na_o * na_o * na_o) * &
      exp(ff * (-1.0_real64 + gamma_p) * v / (rr * tt)) / &
      ((1.0_real64 + k_sat * exp(ff * (-1.0_real64 + gamma_p) * v / (rr * tt))) * &
      (ca_o + km_ca) * ((km_nai * km_nai * km_nai) + (na_o * na_o * na_o)))
    di_up_dca_i = 2.0_real64 * vmax_up * (k_up * k_up) / (((1.0_real64 + &
      (k_up * k_up) / (ca_i * ca_i)) * (1.0_real64 + &
      (k_up * k_up) / (ca_i * ca_i))) * (ca_i * ca_i * ca_i))
    di_p_ca_dca_i = g_pca / (k_pca + ca_i) - g_pca * ca_i / ((k_pca + ca_i) * (k_pca + ca_i))
    de_ca_dca_i = -0.5_real64 * rr * tt / (ff * ca_i)
    dca_i_dt_linearized = (-v_xfer + v_sr * (-v_leak - di_up_dca_i) / v_c - &
      cm * (-2.0_real64 * di_naca_dca_i - g_bca * de_ca_dca_i + di_p_ca_dca_i) / &
      (2.0_real64 * ff * v_c)) * ca_i_bufc + (v_sr * (-i_up + i_leak) / v_c - &
      cm * (-2.0_real64 * i_naca + i_b_ca + i_p_ca) / (2.0_real64 * ff * v_c) + &
      i_xfer) * dca_i_bufc_dca_i
    states(n * state_ca_i + i) = rl_update(ca_i, dca_i_dt, dca_i_dt_linearized, dt)

    k1 = k1_prime / kcasr
    k2 = k2_prime * kcasr
    o_open = (ca_ss * ca_ss) * r_prime * k1 / (k3 + (ca_ss * ca_ss) * k1)
    d_r_prime_dt = k4 * (1.0_real64 - r_prime) - ca_ss * r_prime * k2
    d_r_prime_dt_linearized = -k4 - ca_ss * k2
    states(n * state_r_prime + i) = rl_update(r_prime, d_r_prime_dt, d_r_prime_dt_linearized, dt)

    i_rel = v_rel * (-ca_ss + ca_sr) * o_open
    dca_sr_dt = (-i_leak - i_rel + i_up) * ca_sr_bufsr
    dkcasr_dca_sr = -2.0_real64 * (ec * ec) * (max_sr - min_sr) / (((1.0_real64 + &
      (ec * ec) / (ca_sr * ca_sr)) * (1.0_real64 + (ec * ec) / (ca_sr * ca_sr))) * &
      (ca_sr * ca_sr * ca_sr))
    dca_sr_bufsr_dca_sr = 2.0_real64 * buf_sr * k_buf_sr / (((1.0_real64 + &
      buf_sr * k_buf_sr / ((k_buf_sr + ca_sr) * (k_buf_sr + ca_sr))) * (1.0_real64 + &
      buf_sr * k_buf_sr / ((k_buf_sr + ca_sr) * (k_buf_sr + ca_sr)))) * &
      ((k_buf_sr + ca_sr) * (k_buf_sr + ca_sr) * (k_buf_sr + ca_sr)))
    di_rel_do = v_rel * (-ca_ss + ca_sr)
    dk1_dkcasr = -k1_prime / (kcasr * kcasr)
    do_dk1 = (ca_ss * ca_ss) * r_prime / (k3 + (ca_ss * ca_ss) * k1) - &
      (ca_ss**4) * r_prime * k1 / ((k3 + (ca_ss * ca_ss) * k1) * &
      (k3 + (ca_ss * ca_ss) * k1))
    di_rel_dca_sr = v_rel * o_open + v_rel * (-ca_ss + ca_sr) * do_dk1 * &
      dk1_dkcasr * dkcasr_dca_sr
    dca_sr_dt_linearized = (-v_leak - di_rel_dca_sr - do_dk1 * di_rel_do * &
      dk1_dkcasr * dkcasr_dca_sr) * ca_sr_bufsr + (-i_leak - i_rel + i_up) * &
      dca_sr_bufsr_dca_sr
    states(n * state_ca_sr + i) = rl_update(ca_sr, dca_sr_dt, dca_sr_dt_linearized, dt)

    dca_ss_dt = (v_sr * i_rel / v_ss - v_c * i_xfer / v_ss - &
      cm * i_cal / (2.0_real64 * ff * v_ss)) * ca_ss_bufss
    do_dca_ss = -2.0_real64 * (ca_ss * ca_ss * ca_ss) * (k1 * k1) * r_prime / &
      ((k3 + (ca_ss * ca_ss) * k1) * (k3 + (ca_ss * ca_ss) * k1)) + &
      2.0_real64 * ca_ss * r_prime * k1 / (k3 + (ca_ss * ca_ss) * k1)
    di_rel_dca_ss = -v_rel * o_open + v_rel * (-ca_ss + ca_sr) * do_dca_ss
    dca_ss_bufss_dca_ss = 2.0_real64 * buf_ss * k_buf_ss / (((1.0_real64 + &
      buf_ss * k_buf_ss / ((k_buf_ss + ca_ss) * (k_buf_ss + ca_ss))) * (1.0_real64 + &
      buf_ss * k_buf_ss / ((k_buf_ss + ca_ss) * (k_buf_ss + ca_ss)))) * &
      ((k_buf_ss + ca_ss) * (k_buf_ss + ca_ss) * (k_buf_ss + ca_ss)))
    di_cal_dca_ss = 1.0_real64 * g_cal * (ff * ff) * v_eff * d_gate * &
      exp(2.0_real64 * ff * v_eff / (rr * tt)) * f_gate * f2_gate * fcass / &
      (rr * tt * (-1.0_real64 + exp(2.0_real64 * ff * v_eff / (rr * tt))))
    dca_ss_dt_linearized = (v_sr * (do_dca_ss * di_rel_do + di_rel_dca_ss) / v_ss - &
      v_c * v_xfer / v_ss - cm * di_cal_dca_ss / (2.0_real64 * ff * v_ss)) * &
      ca_ss_bufss + (v_sr * i_rel / v_ss - v_c * i_xfer / v_ss - &
      cm * i_cal / (2.0_real64 * ff * v_ss)) * dca_ss_bufss_dca_ss
    states(n * state_ca_ss + i) = rl_update(ca_ss, dca_ss_dt, dca_ss_dt_linearized, dt)

    dna_i_dt = cm * (-i_na - i_b_na - 3.0_real64 * i_naca - 3.0_real64 * i_nak) / (ff * v_c)
    de_na_dna_i = -rr * tt / (ff * na_i)
    di_naca_dna_i = 3.0_real64 * ca_o * k_naca * (na_i * na_i) * &
      exp(ff * gamma_p * v / (rr * tt)) / ((1.0_real64 + &
      k_sat * exp(ff * (-1.0_real64 + gamma_p) * v / (rr * tt))) * &
      (ca_o + km_ca) * ((km_nai * km_nai * km_nai) + (na_o * na_o * na_o)))
    di_na_de_na = -g_na * (m_gate * m_gate * m_gate) * h_gate * j_gate
    di_nak_dna_i = k_o * p_nak / ((k_mna + na_i) * (k_mk + k_o) * &
      (1.0_real64 + 0.0353_real64 * exp(-ff * v / (rr * tt)) + &
      0.1245_real64 * exp(-0.1_real64 * ff * v / (rr * tt)))) - &
      k_o * p_nak * na_i / (((k_mna + na_i) * (k_mna + na_i)) * (k_mk + k_o) * &
      (1.0_real64 + 0.0353_real64 * exp(-ff * v / (rr * tt)) + &
      0.1245_real64 * exp(-0.1_real64 * ff * v / (rr * tt))))
    dna_i_dt_linearized = cm * (-3.0_real64 * di_naca_dna_i - 3.0_real64 * di_nak_dna_i + &
      g_bna * de_na_dna_i - de_na_dna_i * di_na_de_na) / (ff * v_c)
    states(n * state_na_i + i) = rl_update(na_i, dna_i_dt, dna_i_dt_linearized, dt)

    stim_phase = t - stim_period * real(floor(t / stim_period), real64)
    if (stim_phase <= stim_duration + stim_start .and. stim_phase >= stim_start) then
      i_stim = -stim_amplitude
    else
      i_stim = 0.0_real64
    end if
    dv_dt = -i_cal - i_k1 - i_kr - i_ks - i_na - i_naca - i_nak - &
      i_stim - i_b_ca - i_b_na - i_p_ca - i_p_k - i_to
    dalpha_k1_dv = -3.68652741199693e-8_real64 * exp(0.06_real64 * v - &
      0.06_real64 * e_k) / ((1.0_real64 + 6.14421235332821e-6_real64 * &
      exp(0.06_real64 * v - 0.06_real64 * e_k)) * (1.0_real64 + &
      6.14421235332821e-6_real64 * exp(0.06_real64 * v - 0.06_real64 * e_k)))
    di_cal_dv_eff = 4.0_real64 * g_cal * (ff * ff) * (-ca_o + &
      0.25_real64 * ca_ss * exp(2.0_real64 * ff * v_eff / (rr * tt))) * &
      d_gate * f_gate * f2_gate * fcass / &
      (rr * tt * (-1.0_real64 + exp(2.0_real64 * ff * v_eff / (rr * tt)))) - &
      8.0_real64 * g_cal * (ff * ff * ff) * (-ca_o + &
      0.25_real64 * ca_ss * exp(2.0_real64 * ff * v_eff / (rr * tt))) * &
      v_eff * d_gate * exp(2.0_real64 * ff * v_eff / (rr * tt)) * &
      f_gate * f2_gate * fcass / ((rr * rr) * (tt * tt) * &
      ((-1.0_real64 + exp(2.0_real64 * ff * v_eff / (rr * tt))) * &
      (-1.0_real64 + exp(2.0_real64 * ff * v_eff / (rr * tt))))) + &
      2.0_real64 * g_cal * (ff * ff * ff) * ca_ss * v_eff * d_gate * &
      exp(2.0_real64 * ff * v_eff / (rr * tt)) * f_gate * f2_gate * fcass / &
      ((rr * rr) * (tt * tt) * (-1.0_real64 + exp(2.0_real64 * ff * v_eff / (rr * tt))))
    di_ks_dv = g_ks * (xs * xs)
    di_p_k_dv = g_pk / (1.0_real64 + &
      65.4052157419383_real64 * exp(-0.167224080267559_real64 * v)) + &
      10.9373270471469_real64 * g_pk * (-e_k + v) * &
      exp(-0.167224080267559_real64 * v) / ((1.0_real64 + &
      65.4052157419383_real64 * exp(-0.167224080267559_real64 * v)) * &
      (1.0_real64 + 65.4052157419383_real64 * exp(-0.167224080267559_real64 * v)))
    di_to_dv = g_to * r_gate * s_gate
    dxk1_inf_dbeta_k1 = -alpha_k1 / ((alpha_k1 + beta_k1) * (alpha_k1 + beta_k1))
    dxk1_inf_dalpha_k1 = 1.0_real64 / (alpha_k1 + beta_k1) - &
      alpha_k1 / ((alpha_k1 + beta_k1) * (alpha_k1 + beta_k1))
    dbeta_k1_dv = (0.000612120804016053_real64 * exp(0.0002_real64 * v - &
      0.0002_real64 * e_k) + 0.0367879441171442_real64 * exp(0.1_real64 * v - &
      0.1_real64 * e_k)) / (1.0_real64 + exp(0.5_real64 * e_k - 0.5_real64 * v)) + &
      0.5_real64 * (0.367879441171442_real64 * exp(0.1_real64 * v - &
      0.1_real64 * e_k) + 3.06060402008027_real64 * exp(0.0002_real64 * v - &
      0.0002_real64 * e_k)) * exp(0.5_real64 * e_k - 0.5_real64 * v) / &
      ((1.0_real64 + exp(0.5_real64 * e_k - 0.5_real64 * v)) * &
      (1.0_real64 + exp(0.5_real64 * e_k - 0.5_real64 * v)))
    di_k1_dv = 0.430331482911935_real64 * g_k1 * sqrt(k_o) * xk1_inf + &
      0.430331482911935_real64 * g_k1 * sqrt(k_o) * (-e_k + v) * &
      (dalpha_k1_dv * dxk1_inf_dalpha_k1 + dbeta_k1_dv * dxk1_inf_dbeta_k1)
    if (abs(-15.0_real64 + v) < 0.01_real64) then
      dv_eff_dv = 0.0_real64
    else
      dv_eff_dv = 1.0_real64
    end if
    di_na_dv = g_na * (m_gate * m_gate * m_gate) * h_gate * j_gate
    di_kr_dv = 0.430331482911935_real64 * g_kr * sqrt(k_o) * xr1 * xr2
    di_nak_dv = k_o * p_nak * (0.0353_real64 * ff * exp(-ff * v / (rr * tt)) / (rr * tt) + &
      0.01245_real64 * ff * exp(-0.1_real64 * ff * v / (rr * tt)) / (rr * tt)) * &
      na_i / ((k_mna + na_i) * (k_mk + k_o) * &
      ((1.0_real64 + 0.0353_real64 * exp(-ff * v / (rr * tt)) + &
      0.1245_real64 * exp(-0.1_real64 * ff * v / (rr * tt))) * &
      (1.0_real64 + 0.0353_real64 * exp(-ff * v / (rr * tt)) + &
      0.1245_real64 * exp(-0.1_real64 * ff * v / (rr * tt)))))
    di_k1_dxk1_inf = 0.430331482911935_real64 * g_k1 * sqrt(k_o) * (-e_k + v)
    di_naca_dv = k_naca * (ca_o * ff * gamma_p * (na_i * na_i * na_i) * &
      exp(ff * gamma_p * v / (rr * tt)) / (rr * tt) - ff * alpha_p * &
      (na_o * na_o * na_o) * (-1.0_real64 + gamma_p) * ca_i * &
      exp(ff * (-1.0_real64 + gamma_p) * v / (rr * tt)) / (rr * tt)) / &
      ((1.0_real64 + k_sat * exp(ff * (-1.0_real64 + gamma_p) * v / (rr * tt))) * &
      (ca_o + km_ca) * ((km_nai * km_nai * km_nai) + (na_o * na_o * na_o))) - &
      ff * k_naca * k_sat * (-1.0_real64 + gamma_p) * &
      (ca_o * (na_i * na_i * na_i) * exp(ff * gamma_p * v / (rr * tt)) - &
      alpha_p * (na_o * na_o * na_o) * ca_i * &
      exp(ff * (-1.0_real64 + gamma_p) * v / (rr * tt))) * &
      exp(ff * (-1.0_real64 + gamma_p) * v / (rr * tt)) / &
      (rr * tt * ((1.0_real64 + k_sat * exp(ff * (-1.0_real64 + gamma_p) * v / (rr * tt))) * &
      (1.0_real64 + k_sat * exp(ff * (-1.0_real64 + gamma_p) * v / (rr * tt)))) * &
      (ca_o + km_ca) * ((km_nai * km_nai * km_nai) + (na_o * na_o * na_o)))
    dv_dt_linearized = -g_bca - g_bna - di_k1_dv - di_kr_dv - di_ks_dv - &
      di_naca_dv - di_nak_dv - di_na_dv - di_p_k_dv - di_to_dv - &
      (dalpha_k1_dv * dxk1_inf_dalpha_k1 + dbeta_k1_dv * dxk1_inf_dbeta_k1) * &
      di_k1_dxk1_inf - dv_eff_dv * di_cal_dv_eff
    states(n * state_v + i) = rl_update(v, dv_dt, dv_dt_linearized, dt)

    dki_dt = cm * (-i_k1 - i_kr - i_ks - i_stim - i_p_k - i_to + &
      2.0_real64 * i_nak) / (ff * v_c)
    de_ks_dki = -rr * tt / (ff * (p_kna * na_i + k_i))
    dbeta_k1_de_k = (-0.000612120804016053_real64 * exp(0.0002_real64 * v - &
      0.0002_real64 * e_k) - 0.0367879441171442_real64 * exp(0.1_real64 * v - &
      0.1_real64 * e_k)) / (1.0_real64 + exp(0.5_real64 * e_k - 0.5_real64 * v)) - &
      0.5_real64 * (0.367879441171442_real64 * exp(0.1_real64 * v - &
      0.1_real64 * e_k) + 3.06060402008027_real64 * exp(0.0002_real64 * v - &
      0.0002_real64 * e_k)) * exp(0.5_real64 * e_k - 0.5_real64 * v) / &
      ((1.0_real64 + exp(0.5_real64 * e_k - 0.5_real64 * v)) * &
      (1.0_real64 + exp(0.5_real64 * e_k - 0.5_real64 * v)))
    di_kr_de_k = -0.430331482911935_real64 * g_kr * sqrt(k_o) * xr1 * xr2
    de_k_dki = -rr * tt / (ff * k_i)
    di_ks_de_ks = -g_ks * (xs * xs)
    di_to_de_k = -g_to * r_gate * s_gate
    dalpha_k1_de_k = 3.68652741199693e-8_real64 * exp(0.06_real64 * v - &
      0.06_real64 * e_k) / ((1.0_real64 + 6.14421235332821e-6_real64 * &
      exp(0.06_real64 * v - 0.06_real64 * e_k)) * (1.0_real64 + &
      6.14421235332821e-6_real64 * exp(0.06_real64 * v - 0.06_real64 * e_k)))
    di_k1_de_k = -0.430331482911935_real64 * g_k1 * sqrt(k_o) * xk1_inf + &
      0.430331482911935_real64 * g_k1 * sqrt(k_o) * (-e_k + v) * &
      (dalpha_k1_de_k * dxk1_inf_dalpha_k1 + dbeta_k1_de_k * dxk1_inf_dbeta_k1)
    di_p_k_de_k = -g_pk / &
      (1.0_real64 + 65.4052157419383_real64 * exp(-0.167224080267559_real64 * v))
    dki_dt_linearized = cm * (-(de_k_dki * dalpha_k1_de_k * dxk1_inf_dalpha_k1 + &
      de_k_dki * dbeta_k1_de_k * dxk1_inf_dbeta_k1) * di_k1_dxk1_inf - &
      de_k_dki * di_k1_de_k - de_k_dki * di_kr_de_k - de_k_dki * di_p_k_de_k - &
      de_k_dki * di_to_de_k - de_ks_dki * di_ks_de_ks) / (ff * v_c)
    states(n * state_k_i + i) = rl_update(k_i, dki_dt, dki_dt_linearized, dt)
  end subroutine rush_larsen_cell
  subroutine rush_step(states, t, dt, parameters, n)
    real(real64), intent(inout) :: states(0:*)
    real(real64), intent(in) :: t, dt, parameters(0:*)
    integer, intent(in) :: n
    integer :: i

    do i = 0, n - 1
      call rush_larsen_cell(states, t, dt, parameters, n, i)
    end do
  end subroutine rush_step

  subroutine k_forward_rush_larsen(states, t, dt, parameters, n)
    real(real64), intent(inout) :: states(0:*)
    real(real64), intent(in) :: t, dt, parameters(0:*)
    integer, intent(in) :: n
    integer :: i

    !$omp target teams distribute parallel do thread_limit(256) &
    !$omp& map(tofrom:states(0:num_states*n-1)) map(to:parameters(0:num_params*n-1))
    do i = 0, n - 1
      call rush_larsen_cell(states, t, dt, parameters, n, i)
    end do
    !$omp end target teams distribute parallel do
  end subroutine k_forward_rush_larsen
end module rushlarsen_kernels
