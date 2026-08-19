program sheath
  use iso_c_binding, only: c_int
  use iso_fortran_env, only: real64
  use omp_lib
  use sheath_mod
  implicit none

  type(species) :: ions, electrons
  integer :: p, ts, i
  real(real64) :: sp_time, delta_ions, delta_electrons, v_thi, v_the, start_time, end_time, total_time, max_phi
  logical :: ignored

  sp_time = 0.0_real64
  allocate(domain%phi(0:domain%ni-1), domain%rho(0:domain%ni-1), domain%ef(0:domain%ni-1), &
           domain%nde(0:domain%ni-1), domain%ndi(0:domain%ni-1))
  domain%phi = 0.0_real64

  ions%mass = 16.0_real64 * amu
  ions%charge = qe
  ions%spwt = plasma_den * domain%xl / num_ions
  ions%np = 0
  ions%np_alloc = num_ions
  allocate(ions%part(0:num_ions-1))

  electrons%mass = me
  electrons%charge = -qe
  electrons%spwt = plasma_den * domain%xl / num_electrons
  electrons%np = 0
  electrons%np_alloc = num_electrons
  allocate(electrons%part(0:num_electrons-1))

  call c_srand(123_c_int)
  delta_ions = domain%xl / num_ions
  v_thi = sqrt(2.0_real64 * k_b * ion_temp * ev_to_k / ions%mass)
  do p = 0, num_ions - 1
    call add_particle(ions, domain%x0 + p * delta_ions, sample_vel(v_thi))
  end do

  delta_electrons = domain%xl / num_electrons
  v_the = sqrt(2.0_real64 * k_b * electron_temp * ev_to_k / electrons%mass)
  do p = 0, num_electrons - 1
    call add_particle(electrons, domain%x0 + p * delta_electrons, sample_vel(v_the))
  end do

  !$omp target data map(to: ions%part(0:num_ions-1), electrons%part(0:num_electrons-1)) &
  !$omp& map(alloc: domain%nde(0:domain%ni-1), domain%ndi(0:domain%ni-1), domain%ef(0:domain%ni-1))
    call scatter_species(ions, ions%part, domain%ndi, sp_time)
    call scatter_species(electrons, electrons%part, domain%nde, sp_time)
    call compute_rho(ions, electrons)
    ignored = solve_potential(domain%phi, domain%rho)
    call compute_ef(domain%phi, domain%ef)
    call rewind_species(ions, ions%part, domain%ef)
    call rewind_species(electrons, electrons%part, domain%ef)

    open(newunit=file_res, file='result.dat', status='replace', action='write')
    write(file_res, '(a)') 'VARIABLES = x nde ndi rho phi ef'
    call write_results(0)

    start_time = omp_get_wtime()
    do ts = 1, num_ts
      call scatter_species(ions, ions%part, domain%ndi, sp_time)
      call scatter_species(electrons, electrons%part, domain%nde, sp_time)
      call compute_rho(ions, electrons)
      ignored = solve_potential(domain%phi, domain%rho)
      call compute_ef(domain%phi, domain%ef)
      call push_species(electrons, electrons%part, domain%ef)
      call push_species(ions, ions%part, domain%ef)
      if (mod(ts, 25) == 0) then
        max_phi = abs(domain%phi(0))
        do i = 0, domain%ni - 1
          if (abs(domain%phi(i)) > max_phi) max_phi = abs(domain%phi(i))
        end do
        write(*,'(a,i0,a,i0,a,i0,a,es10.3)') 'TS:', ts, char(9)//'np_i:', ions%np, char(9)//'np_e:', electrons%np, char(9)//'dphi:', max_phi - domain%phi(0)
      end if
      if (mod(ts, 1000) == 0) call write_results(ts)
    end do
    end_time = omp_get_wtime()
    total_time = end_time - start_time
    close(file_res)

    print '(a,es10.3,a)', 'Total kernel execution time (scatter particles) : ', sp_time * 1.0e-9_real64, ' (s)'
    print '(a,i0,a,es10.3,a)', 'Total time for ', num_ts, ' time steps: ', total_time, ' (s)'
    print '(a,es10.3,a)', 'Time per time step: ', (total_time * 1.0e3_real64) / num_ts, ' (ms)'
  !$omp end target data
end program sheath
