module c_rng
  use iso_c_binding, only: c_int
  implicit none
  interface
    subroutine srand(seed) bind(C, name="srand")
      import c_int
      integer(c_int), value :: seed
    end subroutine srand
    function rand() result(r) bind(C, name="rand")
      import c_int
      integer(c_int) :: r
    end function rand
  end interface
end module c_rng

module md_types
  use iso_fortran_env, only: real32
  implicit none
  integer, parameter :: fp = real32
  real(fp), parameter :: cutsq = 13.5_fp
  integer, parameter :: max_neighbors = 128
  integer, parameter :: domain_edge = 20
  real(fp), parameter :: lj1 = 1.5_fp
  real(fp), parameter :: lj2 = 2.0_fp
  type :: vec4
    real(fp) :: x = 0.0_fp
    real(fp) :: y = 0.0_fp
    real(fp) :: z = 0.0_fp
    real(fp) :: w = 0.0_fp
  end type vec4
contains
  pure real(fp) function distance(position, i, j)
    type(vec4), intent(in) :: position(0:)
    integer, intent(in) :: i, j
    real(fp) :: delx, dely, delz
    delx = position(i)%x - position(j)%x
    dely = position(i)%y - position(j)%y
    delz = position(i)%z - position(j)%z
    distance = delx*delx + dely*dely + delz*delz
  end function distance
end module md_types

module md_kernels
  use md_types
  implicit none
contains
  subroutine md(position, force, neighbor_list, natom, maxneighbors, lj1_t, lj2_t, cutsq_t)
    type(vec4), intent(in) :: position(0:)
    type(vec4), intent(inout) :: force(0:)
    integer, intent(in) :: neighbor_list(0:)
    integer, intent(in) :: natom, maxneighbors
    real(fp), intent(in) :: lj1_t, lj2_t, cutsq_t
    integer :: idx, j, jidx
    type(vec4) :: ipos, jpos, f
    real(fp) :: delx, dely, delz, r2inv, r6inv, forcec
!$omp target teams distribute parallel do thread_limit(256) private(ipos,jpos,f,j,jidx,delx,dely,delz,r2inv,r6inv,forcec)
    do idx = 0, natom-1
      ipos = position(idx)
      f%x = 0.0_fp; f%y = 0.0_fp; f%z = 0.0_fp; f%w = 0.0_fp
      j = 0
      do while (j < maxneighbors)
        jidx = neighbor_list(j*natom + idx)
        jpos = position(jidx)
        delx = ipos%x - jpos%x
        dely = ipos%y - jpos%y
        delz = ipos%z - jpos%z
        r2inv = delx*delx + dely*dely + delz*delz
        if (r2inv > 0.0_fp .and. r2inv < cutsq_t) then
          r2inv = 1.0_fp / r2inv
          r6inv = r2inv * r2inv * r2inv
          forcec = r2inv * r6inv * (lj1_t*r6inv - lj2_t)
          f%x = f%x + delx * forcec
          f%y = f%y + dely * forcec
          f%z = f%z + delz * forcec
        end if
        j = j + 1
      end do
      force(idx) = f
    end do
!$omp end target teams distribute parallel do
  end subroutine md

  integer function build_neighbor_list(natom, position, neighbor_list)
    integer, intent(in) :: natom
    type(vec4), intent(in) :: position(0:)
    integer, intent(out) :: neighbor_list(0:)
    real(fp) :: curr_dist(0:max_neighbors-1), distij
    integer :: curr_list(0:max_neighbors-1)
    integer :: i, j, k, insert_at, valid_pairs
    build_neighbor_list = 0
    do i = 0, natom-1
      curr_dist = huge(1.0_fp)
      curr_list = -1
      do j = 0, natom-1
        if (i == j) cycle
        distij = distance(position, i, j)
        if (distij > curr_dist(max_neighbors-1)) cycle
        insert_at = max_neighbors
        do k = 0, max_neighbors-1
          if (distij < curr_dist(k)) then
            insert_at = k
            exit
          end if
        end do
        if (insert_at < max_neighbors) then
          do k = max_neighbors-1, insert_at+1, -1
            curr_dist(k) = curr_dist(k-1)
            curr_list(k) = curr_list(k-1)
          end do
          curr_dist(insert_at) = distij
          curr_list(insert_at) = j
        end if
      end do
      valid_pairs = 0
      do k = 0, max_neighbors-1
        neighbor_list(k*natom + i) = curr_list(k)
        if (curr_dist(k) < cutsq) valid_pairs = valid_pairs + 1
      end do
      build_neighbor_list = build_neighbor_list + valid_pairs
    end do
  end function build_neighbor_list

  subroutine check_results(device_force, position, neighbor_list, natom)
    type(vec4), intent(in) :: device_force(0:), position(0:)
    integer, intent(in) :: neighbor_list(0:), natom
    type(vec4) :: ipos, jpos, f
    integer :: i, j, jidx
    real(fp) :: delx, dely, delz, r2inv, r6inv, forcec, max_error
    max_error = 0.0_fp
    do i = 0, natom-1
      ipos = position(i)
      f%x = 0.0_fp; f%y = 0.0_fp; f%z = 0.0_fp; f%w = 0.0_fp
      j = 0
      do while (j < max_neighbors)
        jidx = neighbor_list(j*natom + i)
        jpos = position(jidx)
        delx = ipos%x - jpos%x
        dely = ipos%y - jpos%y
        delz = ipos%z - jpos%z
        r2inv = delx*delx + dely*dely + delz*delz
        if (r2inv > 0.0_fp .and. r2inv < cutsq) then
          r2inv = 1.0_fp / r2inv
          r6inv = r2inv * r2inv * r2inv
          forcec = r2inv * r6inv * (lj1*r6inv - lj2)
          f%x = f%x + delx * forcec
          f%y = f%y + dely * forcec
          f%z = f%z + delz * forcec
        end if
        j = j + 1
      end do
      max_error = max(max_error, abs(f%x - device_force(i)%x), abs(f%y - device_force(i)%y), abs(f%z - device_force(i)%z))
    end do
    print '(a,es12.5)', 'Max error between host and device: ', max_error
  end subroutine check_results
end module md_kernels

program main
  use iso_fortran_env, only: real32
  use omp_lib
  use c_rng
  use md_types
  use md_kernels
  implicit none
  integer, parameter :: prob_sizes(0:3) = [12288, 24576, 36864, 73728]
  integer :: size_class, iteration, natom, total_pairs, i, argc
  character(len=64) :: arg
  type(vec4), allocatable :: position(:), force(:)
  integer, allocatable :: neighbor_list(:)
  real(8) :: start_time, elapsed

  argc = command_argument_count()
  if (argc /= 2) then
    print '(a)', 'usage: ./main <class size> <iteration>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*) size_class
  call get_command_argument(2,arg); read(arg,*) iteration
  if (size_class < 0 .or. size_class >= 4 .or. iteration < 0) stop 1
  natom = prob_sizes(size_class)
  allocate(position(0:natom-1), force(0:natom-1), neighbor_list(0:max_neighbors*natom-1))

  print '(a)', 'Initializing test problem (this can take several minutes for large problems).'
  call srand(123)
  do i = 0, natom-1
    position(i)%x = real(mod(rand(), domain_edge), real32)
    position(i)%y = real(mod(rand(), domain_edge), real32)
    position(i)%z = real(mod(rand(), domain_edge), real32)
    position(i)%w = 0.0_real32
  end do
  print '(a)', 'Finished.'
  total_pairs = build_neighbor_list(natom, position, neighbor_list)
  print '(i0,a,i0,a,f0.6,a)', total_pairs, ' of ', natom*max_neighbors, &
    ' pairs within cutoff distance = ', 100.0*real(total_pairs)/real(natom*max_neighbors), ' %'

!$omp target data map(to:position(0:natom-1),neighbor_list(0:natom*max_neighbors-1)) map(alloc:force(0:natom-1))
  call md(position, force, neighbor_list, natom, max_neighbors, lj1, lj2, cutsq)
!$omp target update from(force(0:natom-1))
  print '(a)', 'Performing Correctness Check (may take several minutes)'
  call check_results(force, position, neighbor_list, natom)

  start_time = omp_get_wtime()
  do i = 1, iteration
    call md(position, force, neighbor_list, natom, max_neighbors, lj1, lj2, cutsq)
  end do
  elapsed = omp_get_wtime() - start_time
  print '(a,f0.6,a)', 'Average kernel execution time ', elapsed / iteration, ' (s)'
!$omp end target data
end program main
