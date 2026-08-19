module simplemoc_mod
  use iso_c_binding, only: c_int
  use iso_fortran_env, only: int64, real32, real64
  use omp_lib
  implicit none

  type :: input_type
    integer :: source_2d_regions
    integer :: source_3d_regions
    integer :: coarse_axial_intervals
    integer :: fine_axial_intervals
    integer :: decomp_assemblies_ax
    integer(int64) :: segments
    integer :: egroups
    integer :: nthreads
    integer :: repeat
    integer(int64) :: nbytes
  end type input_type

  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand
    function c_rand() bind(C, name="rand") result(r)
      import :: c_int
      integer(c_int) :: r
    end function c_rand
    function c_rand_r(seed) bind(C, name="rand_r") result(r)
      import :: c_int
      integer(c_int) :: seed
      integer(c_int) :: r
    end function c_rand_r
  end interface

contains

  subroutine set_default_input(i)
    type(input_type), intent(out) :: i
    i%source_2d_regions = 5000
    i%coarse_axial_intervals = 27
    i%fine_axial_intervals = 5
    i%decomp_assemblies_ax = 20
    i%segments = 50000000_int64
    i%egroups = 128
    i%repeat = 1
    i%nthreads = 1
    i%nbytes = 0_int64
  end subroutine set_default_input

  subroutine read_cli(input)
    type(input_type), intent(inout) :: input
    integer :: argc, i
    character(len=128) :: arg, val
    argc = command_argument_count()
    i = 1
    do while (i <= argc)
      call get_command_argument(i, arg)
      if (trim(arg) == '-t') then
        i = i + 1; call get_command_argument(i, val); read(val, *) input%nthreads
      else if (trim(arg) == '-s') then
        i = i + 1; call get_command_argument(i, val); read(val, *) input%segments
      else if (trim(arg) == '-e') then
        i = i + 1; call get_command_argument(i, val); read(val, *) input%egroups
      else if (trim(arg) == '-n') then
        i = i + 1; call get_command_argument(i, val); read(val, *) input%repeat
      else
        call print_cli_error()
      end if
      i = i + 1
    end do
    if (input%nthreads < 1) call print_cli_error()
  end subroutine read_cli

  subroutine print_cli_error()
    print '(a)', 'Usage: ./SimpleMOC <options>'
    print '(a)', 'Options include:'
    print '(a)', '  -t <threads>        Number of OpenMP threads to run'
    print '(a)', '  -s <segments>       Number of segments to process'
    print '(a)', '  -e <energy groups>  Number of energy groups'
    print '(a)', '  -n <kernel runs>    Number of kernel execution on a device (GPU)'
    stop 1
  end subroutine print_cli_error

  subroutine border_print()
    print '(a)', '==============================================================================='
  end subroutine border_print

  subroutine center_print(s)
    character(len=*), intent(in) :: s
    integer :: pad
    pad = max((79 - len_trim(s)) / 2, 0)
    print '(a,a)', repeat(' ', pad), trim(s)
  end subroutine center_print

  subroutine logo(version)
    integer, intent(in) :: version
    character(len=64) :: v
    call border_print()
    print '(a)', '   __           __        ___        __   __           ___  __        ___     '
    print '(a)', '  /__` |  |\/| |__) |    |__   |\/| /  \ /  ` __ |__/ |__  |__) |\ | |__  |   '
    print '(a)', '  .__/ |  |  | |    |___ |___ |  | \__/ \__,    |  \ |___ |  \ | \| |___ |___'
    print '(a)', ''
    call border_print()
    print '(a)', ''
    call center_print('Developed at')
    call center_print('The Massachusetts Institute of Technology')
    call center_print('and')
    call center_print('Argonne National Laboratory')
    print '(a)', ''
    write(v, '(a,i0)') 'Version: ', version
    call center_print(trim(v))
    print '(a)', ''
    call border_print()
  end subroutine logo

  subroutine print_input_summary(input)
    type(input_type), intent(in) :: input
    call center_print('INPUT SUMMARY')
    call border_print()
    print '(a25,i0)', 'Kernel execution times:', input%repeat
    print '(a25,i0)', 'Energy Groups:', input%egroups
    print '(a25,i0)', '2D Source Regions:', input%source_2d_regions
    print '(a25,i0)', 'Coarse Axial Intervals:', input%coarse_axial_intervals
    print '(a25,i0)', 'Fine Axial Intervals:', input%fine_axial_intervals
    print '(a25,i0)', 'Axial Decomposition:', input%decomp_assemblies_ax
    print '(a25,i0)', '3D Source Regions:', input%source_3d_regions
    print '(a25,i0)', 'Segments:', input%segments
    print '(a25,f10.2)', 'Memory Estimate (MB):', real(input%nbytes, real64) / 1024.0_real64 / 1024.0_real64
    call border_print()
  end subroutine print_input_summary

  subroutine initialize_sources(input, fine_source, fine_flux, sigt)
    type(input_type), intent(inout) :: input
    real(real32), allocatable, intent(out) :: fine_source(:), fine_flux(:), sigt(:)
    integer :: i, j, k, idx
    integer(int64) :: source_size, sigt_size
    input%nbytes = 0_int64
    source_size = int(input%source_3d_regions, int64) * input%fine_axial_intervals * input%egroups
    sigt_size = int(input%source_3d_regions, int64) * input%egroups
    input%nbytes = input%nbytes + source_size * 4_int64 * 2_int64 + sigt_size * 4_int64
    allocate(fine_source(0:source_size-1), fine_flux(0:source_size-1), sigt(0:sigt_size-1))
    do i = 0, input%source_3d_regions - 1
      do j = 0, input%fine_axial_intervals - 1
        do k = 0, input%egroups - 1
          idx = i * input%fine_axial_intervals * input%egroups + j * input%egroups + k
          fine_source(idx) = real(c_rand(), real32) / 2147483647.0_real32
          fine_flux(idx) = real(c_rand(), real32) / 2147483647.0_real32
        end do
      end do
    end do
    do i = 0, input%source_3d_regions - 1
      do k = 0, input%egroups - 1
        sigt(i * input%egroups + k) = real(c_rand(), real32) / 2147483647.0_real32
      end do
    end do
  end subroutine initialize_sources

  subroutine build_segment_ids(input, qsr_id, fai_id, state_flux, state_flux_device, seed)
    type(input_type), intent(in) :: input
    integer, allocatable, intent(out) :: qsr_id(:), fai_id(:)
    real(real32), allocatable, intent(out) :: state_flux(:), state_flux_device(:)
    integer(c_int), intent(inout) :: seed
    integer(int64) :: i
    integer :: g
    allocate(qsr_id(0:input%segments-1), fai_id(0:input%segments-1), state_flux(0:input%egroups-1), state_flux_device(0:input%egroups-1))
    do g = 0, input%egroups - 1
      state_flux_device(g) = real(c_rand_r(seed), real32) / 2147483647.0_real32
      state_flux(g) = state_flux_device(g)
    end do
    do i = 0, input%segments - 1
      qsr_id(i) = mod(c_rand_r(seed), input%source_3d_regions)
      fai_id(i) = mod(c_rand_r(seed), input%fine_axial_intervals)
    end do
  end subroutine build_segment_ids

  subroutine attenuate_device(repeat, segments, egroups, fine_axial_intervals, source_3d_regions, qsr_id, fai_id, fine_flux, fine_source, sigt, state_flux, v_acc)
    integer, intent(in) :: repeat, egroups, fine_axial_intervals, source_3d_regions
    integer(int64), intent(in) :: segments
    integer, intent(in) :: qsr_id(0:), fai_id(0:)
    real(real32), intent(inout) :: fine_flux(0:), state_flux(0:)
    real(real32), intent(in) :: fine_source(0:), sigt(0:)
    real(real32), intent(inout) :: v_acc(0:)
    integer :: n, gid, qsr, fai, offset, sig_offset, g
    real(real32) :: dz, zin, weight, mu, mu2, ds, y1, y2, y3, c0, c1, c2
    real(real32) :: tau, sigt2, expval, reuse, flux_integral, tally, t1, t2, t3, t4

    do n = 0, repeat - 1
      !$omp target teams distribute parallel do thread_limit(128) private(qsr,fai,offset,sig_offset,g,dz,zin,weight,mu,mu2,ds,y1,y2,y3,c0,c1,c2,tau,sigt2,expval,reuse,flux_integral,tally,t1,t2,t3,t4)
      do gid = 0, segments - 1
        dz = 0.1_real32
        zin = 0.3_real32
        weight = 0.5_real32
        mu = 0.9_real32
        mu2 = 0.3_real32
        ds = 0.7_real32
        qsr = qsr_id(gid)
        fai = fai_id(gid)
        offset = qsr * fine_axial_intervals * egroups
        if (fai == 0) then
          do g = 0, egroups - 1
            y2 = fine_source(offset + fai * egroups + g)
            y3 = fine_source(offset + (fai + 1) * egroups + g)
            c0 = y2
            c1 = (y3 - y2) / dz
            v_acc(g) = c0 + c1 * zin
            v_acc(egroups + g) = c1
            v_acc(2 * egroups + g) = 0.0_real32
          end do
        else if (fai == fine_axial_intervals - 1) then
          do g = 0, egroups - 1
            y1 = fine_source(offset + (fai - 1) * egroups + g)
            y2 = fine_source(offset + fai * egroups + g)
            c0 = y2
            c1 = (y2 - y1) / dz
            v_acc(g) = c0 + c1 * zin
            v_acc(egroups + g) = c1
            v_acc(2 * egroups + g) = 0.0_real32
          end do
        else
          do g = 0, egroups - 1
            y1 = fine_source(offset + (fai - 1) * egroups + g)
            y2 = fine_source(offset + fai * egroups + g)
            y3 = fine_source(offset + (fai + 1) * egroups + g)
            c0 = y2
            c1 = (y1 - y3) / (2.0_real32 * dz)
            c2 = (y1 - 2.0_real32 * y2 + y3) / (2.0_real32 * dz * dz)
            v_acc(g) = c0 + c1 * zin + c2 * zin * zin
            v_acc(egroups + g) = c1 + 2.0_real32 * c2 * zin
            v_acc(2 * egroups + g) = c2
          end do
        end if
        sig_offset = qsr * egroups
        do g = 0, egroups - 1
          v_acc(3 * egroups + g) = sigt(sig_offset + g)
          tau = v_acc(3 * egroups + g) * ds
          sigt2 = v_acc(3 * egroups + g) * v_acc(3 * egroups + g)
          expval = 1.0_real32 - exp(-tau)
          reuse = tau * (tau - 2.0_real32) + 2.0_real32 * expval / (v_acc(3 * egroups + g) * sigt2)
          flux_integral = (v_acc(g) * tau + (v_acc(3 * egroups + g) * state_flux(g) - v_acc(g)) * expval) / sigt2 + &
            v_acc(egroups + g) * mu * reuse + v_acc(2 * egroups + g) * mu2 * &
            (tau * (tau * (tau - 3.0_real32) + 6.0_real32) - 6.0_real32 * expval) / (3.0_real32 * sigt2 * sigt2)
          tally = weight * flux_integral
          fine_flux(offset + fai * egroups + g) = fine_flux(offset + fai * egroups + g) + tally
          t1 = v_acc(g) * expval / v_acc(3 * egroups + g)
          t2 = v_acc(egroups + g) * mu * (tau - expval) / sigt2
          t3 = v_acc(2 * egroups + g) * mu2 * reuse
          t4 = state_flux(g) * (1.0_real32 - expval)
          state_flux(g) = t1 + t2 + t3 + t4
          v_acc(4*egroups+g) = tau; v_acc(5*egroups+g) = sigt2; v_acc(6*egroups+g) = expval
          v_acc(7*egroups+g) = reuse; v_acc(8*egroups+g) = flux_integral; v_acc(9*egroups+g) = tally
          v_acc(10*egroups+g) = t1; v_acc(11*egroups+g) = t2; v_acc(12*egroups+g) = t3; v_acc(13*egroups+g) = t4
        end do
      end do
      !$omp end target teams distribute parallel do
    end do
  end subroutine attenuate_device

end module simplemoc_mod
