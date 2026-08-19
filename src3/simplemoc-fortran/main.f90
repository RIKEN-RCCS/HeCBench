program simplemoc
  use iso_c_binding, only: c_int
  use iso_fortran_env, only: real32, real64
  use omp_lib
  use simplemoc_mod
  implicit none

  type(input_type) :: input
  integer(c_int) :: seed
  integer, allocatable :: qsr_id(:), fai_id(:)
  real(real32), allocatable :: fine_source(:), fine_flux(:), sigt(:), state_flux(:), state_flux_device(:), v_acc(:)
  real(real64) :: start_time, stop_time, kstart, kstop, tpi
  integer(int64) :: source_size, sigt_size

  seed = 2_c_int
  call c_srand(seed)
  call set_default_input(input)
  call read_cli(input)
  input%source_3d_regions = ceiling(real(input%source_2d_regions * input%coarse_axial_intervals, real64) / real(input%decomp_assemblies_ax, real64))

  call logo(4)
  call initialize_sources(input, fine_source, fine_flux, sigt)
  call print_input_summary(input)
  call center_print('SIMULATION')
  call border_print()
  print '(a)', 'Attentuating fluxes across segments...'

  call build_segment_ids(input, qsr_id, fai_id, state_flux, state_flux_device, seed)
  allocate(v_acc(0:input%egroups*14-1))
  source_size = int(input%source_3d_regions, int64) * input%fine_axial_intervals * input%egroups
  sigt_size = int(input%source_3d_regions, int64) * input%egroups

  start_time = omp_get_wtime()
  !$omp target data map(to: qsr_id(0:input%segments-1), fai_id(0:input%segments-1), sigt(0:sigt_size-1), fine_source(0:source_size-1)) &
  !$omp& map(tofrom: fine_flux(0:source_size-1), state_flux_device(0:input%egroups-1)) map(alloc: v_acc(0:input%egroups*14-1))
    kstart = omp_get_wtime()
    call attenuate_device(input%repeat, input%segments, input%egroups, input%fine_axial_intervals, input%source_3d_regions, &
                          qsr_id, fai_id, fine_flux, fine_source, sigt, state_flux_device, v_acc)
    kstop = omp_get_wtime()
  !$omp end target data
  print '(a)', 'Simulation Complete.'
  stop_time = omp_get_wtime()

  call border_print()
  call center_print('RESULTS SUMMARY')
  call border_print()
  print '(a25,f8.3,a)', 'Total kernel time:', kstop - kstart, ' seconds'
  print '(a25,f8.3,a)', 'Device offload time:', stop_time - start_time, ' seconds'
  tpi = ((kstop - kstart) / real(input%repeat, real64) / real(input%segments, real64) / real(input%egroups, real64)) * 1.0e9_real64
  print '(a25,f8.3,a)', 'Time per Intersection:', tpi, ' ns'
  call border_print()
end program simplemoc
