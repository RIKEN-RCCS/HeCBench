module sheath_mod
  use iso_c_binding, only: c_int, c_double, c_bool
  use iso_fortran_env, only: real32, real64
  use omp_lib
  implicit none

  real(real64), parameter :: eps_0 = 8.85418782e-12_real64
  real(real64), parameter :: k_b = 1.38065e-23_real64
  real(real64), parameter :: me = 9.10938215e-31_real64
  real(real64), parameter :: qe = 1.602176565e-19_real64
  real(real64), parameter :: amu = 1.660538921e-27_real64
  real(real64), parameter :: ev_to_k = 11604.52_real64
  real(real64), parameter :: plasma_den = 1.0e16_real64
  integer, parameter :: num_ions = 500000
  integer, parameter :: num_electrons = 500000
  real(real64), parameter :: dx_const = 1.0e-4_real64
  integer, parameter :: nc = 100
  integer, parameter :: num_ts = 1000
  real(real64), parameter :: dt = 1.0e-11_real64
  real(real64), parameter :: electron_temp = 3.0_real64
  real(real64), parameter :: ion_temp = 1.0_real64
  real(real64), parameter :: x0_const = 0.0_real64
  real(real64), parameter :: xl_const = nc * dx_const
  real(real64), parameter :: xmax_const = x0_const + xl_const
  integer, parameter :: threads_per_block = 256

  type :: domain_type
    integer :: ni = nc + 1
    real(real64) :: x0 = x0_const
    real(real64) :: dx = dx_const
    real(real64) :: xl = xl_const
    real(real64) :: xmax = xmax_const
    real(real64), allocatable :: phi(:), ef(:), rho(:)
    real(real32), allocatable :: ndi(:), nde(:)
  end type domain_type

  type, bind(C) :: particle
    real(c_double) :: x
    real(c_double) :: v
    logical(c_bool) :: alive
  end type particle

  type :: species
    real(real64) :: mass
    real(real64) :: charge
    real(real64) :: spwt
    integer :: np
    integer :: np_alloc
    type(particle), allocatable :: part(:)
  end type species

  type(domain_type) :: domain
  integer :: file_res = -1
!$omp declare target(xtol,scatter,gather)

  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand
    function c_rand() bind(C, name="rand") result(r)
      import :: c_int
      integer(c_int) :: r
    end function c_rand
  end interface

contains

  real(real64) function rnd() result(v)
    v = real(c_rand(), real64) / 2147483647.0_real64
  end function rnd

  real(real64) function sample_vel(v_th) result(v)
    real(real64), intent(in) :: v_th
    integer, parameter :: m = 12
    integer :: i
    real(real64) :: sumv
    sumv = 0.0_real64
    do i = 1, m
      sumv = sumv + rnd()
    end do
    v = sqrt(0.5_real64) * v_th * (sumv - real(m, real64) / 2.0_real64) / sqrt(real(m, real64) / 12.0_real64)
  end function sample_vel

  subroutine add_particle(sp, x, v)
    type(species), intent(inout) :: sp
    real(real64), intent(in) :: x, v
    if (sp%np > sp%np_alloc - 1) then
      print '(a)', 'Too many particles!'
      stop -1
    end if
    sp%part(sp%np)%x = x
    sp%part(sp%np)%v = v
    sp%part(sp%np)%alive = .true._c_bool
    sp%np = sp%np + 1
  end subroutine add_particle

  real(real64) function xtol(pos) result(li)
    real(real64), intent(in) :: pos
    li = (pos - 0.0_real64) / dx_const
  end function xtol

  subroutine scatter(lc, value, field)
    real(real64), intent(in) :: lc
    real(real32), intent(in) :: value
    real(real32), intent(inout) :: field(0:*)
    integer :: i
    real(real32) :: di
    i = int(lc)
    di = real(lc - real(i, real64), real32)
    !$omp atomic update
    field(i) = field(i) + value * (1.0_real32 - di)
    !$omp atomic update
    field(i + 1) = field(i + 1) + value * di
  end subroutine scatter

  real(real64) function gather(lc, field) result(val)
    real(real64), intent(in) :: lc
    real(real64), intent(in) :: field(0:*)
    integer :: i
    real(real64) :: di
    i = int(lc)
    di = lc - real(i, real64)
    val = field(i) * (1.0_real64 - di) + field(i + 1) * di
  end function gather

  subroutine scatter_species(sp, particles, den, sp_time)
    type(species), intent(in) :: sp
    type(particle), intent(in) :: particles(0:*)
    real(real32), intent(inout) :: den(0:*)
    real(real64), intent(inout) :: sp_time
    integer :: p, nodes, size
    real(real64) :: lc, start_time, end_time

    nodes = domain%ni
    !$omp target teams distribute parallel do thread_limit(threads_per_block) &
    !$omp& map(tofrom:den(0:nodes-1))
    do p = 0, nodes - 1
      den(p) = 0.0_real32
    end do
    !$omp end target teams distribute parallel do

    size = sp%np_alloc
    start_time = omp_get_wtime()
    !$omp target teams distribute parallel do thread_limit(threads_per_block) &
    !$omp& map(to:particles(0:size-1)) map(tofrom:den(0:nodes-1)) private(lc)
    do p = 0, size - 1
      if (particles(p)%alive) then
        lc = xtol(particles(p)%x)
        call scatter(lc, 1.0_real32, den)
      end if
    end do
    !$omp end target teams distribute parallel do
    end_time = omp_get_wtime()
    sp_time = sp_time + (end_time - start_time) * 1.0e9_real64

    !$omp target update from(den(0:nodes-1))
    do p = 0, domain%ni - 1
      den(p) = den(p) * real(sp%spwt / domain%dx, real32)
    end do
    den(0) = den(0) * 2.0_real32
    den(domain%ni - 1) = den(domain%ni - 1) * 2.0_real32
  end subroutine scatter_species

  subroutine compute_rho(ions, electrons)
    type(species), intent(in) :: ions, electrons
    integer :: i
    do i = 0, domain%ni - 1
      domain%rho(i) = ions%charge * domain%ndi(i) + electrons%charge * domain%nde(i)
    end do
  end subroutine compute_rho

  logical function solve_potential(phi, rho) result(converged)
    real(real64), intent(inout) :: phi(0:)
    real(real64), intent(in) :: rho(0:)
    integer :: solver_it, i
    real(real64) :: l2, dx2, sumv, r, g
    dx2 = domain%dx * domain%dx
    phi(0) = 0.0_real64
    phi(domain%ni - 1) = 0.0_real64
    l2 = 0.0_real64
    do solver_it = 0, 39999
      do i = 1, domain%ni - 2
        g = 0.5_real64 * (phi(i - 1) + phi(i + 1) + dx2 * rho(i) / eps_0)
        phi(i) = phi(i) + 1.4_real64 * (g - phi(i))
      end do
      if (mod(solver_it, 25) == 0) then
        sumv = 0.0_real64
        do i = 1, domain%ni - 2
          r = -rho(i) / eps_0 - (phi(i - 1) - 2.0_real64 * phi(i) + phi(i + 1)) / dx2
          sumv = sumv + r * r
        end do
        l2 = sqrt(sumv) / domain%ni
        if (l2 < 1.0e-4_real64) then
          converged = .true.
          return
        end if
      end if
    end do
    print '(a,es10.3,a)', 'Gauss-Seidel solver failed to converge, L2=', l2, '!'
    converged = .false.
  end function solve_potential

  subroutine compute_ef(phi, ef)
    real(real64), intent(in) :: phi(0:)
    real(real64), intent(inout) :: ef(0:)
    integer :: i
    do i = 1, domain%ni - 2
      ef(i) = -(phi(i + 1) - phi(i - 1)) / (2.0_real64 * domain%dx)
    end do
    ef(0) = -(phi(1) - phi(0)) / domain%dx
    ef(domain%ni - 1) = -(phi(domain%ni - 1) - phi(domain%ni - 2)) / domain%dx
    !$omp target update to(ef(0:domain%ni-1))
  end subroutine compute_ef

  subroutine push_species(sp, particles, ef)
    type(species), intent(in) :: sp
    type(particle), intent(inout) :: particles(0:*)
    real(real64), intent(in) :: ef(0:*)
    integer :: p, size
    real(real64) :: qm, lc, part_ef
    qm = sp%charge / sp%mass
    size = sp%np_alloc
    !$omp target teams distribute parallel do thread_limit(threads_per_block) &
    !$omp& map(tofrom:particles(0:size-1)) map(to:ef(0:domain%ni-1)) private(lc,part_ef)
    do p = 0, size - 1
      if (particles(p)%alive) then
        lc = xtol(particles(p)%x)
        part_ef = gather(lc, ef)
        particles(p)%v = particles(p)%v + dt * qm * part_ef
        particles(p)%x = particles(p)%x + dt * particles(p)%v
        if (particles(p)%x < x0_const .or. particles(p)%x >= xmax_const) particles(p)%alive = .false._c_bool
      end if
    end do
    !$omp end target teams distribute parallel do
  end subroutine push_species

  subroutine rewind_species(sp, particles, ef)
    type(species), intent(in) :: sp
    type(particle), intent(inout) :: particles(0:*)
    real(real64), intent(in) :: ef(0:*)
    integer :: p, size
    real(real64) :: qm, lc, part_ef
    qm = sp%charge / sp%mass
    size = sp%np_alloc
    !$omp target teams distribute parallel do thread_limit(threads_per_block) &
    !$omp& map(tofrom:particles(0:size-1)) map(to:ef(0:domain%ni-1)) private(lc,part_ef)
    do p = 0, size - 1
      if (particles(p)%alive) then
        lc = xtol(particles(p)%x)
        part_ef = gather(lc, ef)
        particles(p)%v = particles(p)%v - 0.5_real64 * dt * qm * part_ef
      end if
    end do
    !$omp end target teams distribute parallel do
  end subroutine rewind_species

  subroutine write_results(ts)
    integer, intent(in) :: ts
    integer :: i
    write(file_res, '(a,i0,a,i6.6)') 'ZONE I=', domain%ni, ' T=ZONE_', ts
    do i = 0, domain%ni - 1
      write(file_res, '(6(es14.6,1x))') real(i, real64) * domain%dx, real(domain%nde(i), real64), real(domain%ndi(i), real64), &
        domain%rho(i), domain%phi(i), domain%ef(i)
    end do
    flush(file_res)
  end subroutine write_results

end module sheath_mod
