module xsbench_simulation
  use iso_fortran_env, only: real64, int64
  implicit none
  integer, parameter :: unionized = 0, nuclide = 1, hash = 2
  integer(int64), parameter :: starting_seed = 1070_int64
  type :: inputs_t
    integer :: nthreads = 1
    integer(int64) :: n_isotopes = 355_int64
    integer(int64) :: n_gridpoints = 11303_int64
    integer :: lookups = 150000
    integer :: grid_type = unionized
    integer :: hash_bins = 10000
    integer :: kernel_repeat = 10
  end type inputs_t
  type :: nuclide_grid_point
    real(real64) :: energy = 0.0_real64
    real(real64) :: total_xs = 0.0_real64
    real(real64) :: elastic_xs = 0.0_real64
    real(real64) :: absorbtion_xs = 0.0_real64
    real(real64) :: fission_xs = 0.0_real64
    real(real64) :: nu_fission_xs = 0.0_real64
  end type nuclide_grid_point
  type :: simulation_data_t
    integer, allocatable :: num_nucs(:), mats(:), index_grid(:)
    real(real64), allocatable :: concs(:), unionized_energy_array(:)
    type(nuclide_grid_point), allocatable :: nuclide_grid(:)
    integer :: max_num_nucs = 0
  end type simulation_data_t
contains
  function fast_forward_lcg(seed, n) result(out)
    integer(int64), intent(in) :: seed, n
    integer(int64) :: out, i
    out = seed
    do i = 1, n
      out = lcg_next(out)
    end do
  end function fast_forward_lcg

  pure integer(int64) function lcg_next(seed)
    integer(int64), intent(in) :: seed
    lcg_next = iand(seed * 2862933555777941757_int64 + 3037000493_int64, int(z'7FFFFFFFFFFFFFFF', int64))
  end function lcg_next

  real(real64) function lcg_random_double(seed)
    integer(int64), intent(inout) :: seed
    seed = lcg_next(seed)
    lcg_random_double = real(iand(seed, int(z'00000000FFFFFFFF', int64)), real64) / real(huge(1), real64)
  end function lcg_random_double

  integer function pick_mat(seed)
    integer(int64), intent(inout) :: seed
    real(real64) :: roll
    roll = lcg_random_double(seed)
    if (roll < 0.140_real64) then
      pick_mat = 0
    else if (roll < 0.200_real64) then
      pick_mat = 1
    else if (roll < 0.520_real64) then
      pick_mat = 2
    else if (roll < 0.820_real64) then
      pick_mat = 3
    else
      pick_mat = 4
    end if
  end function pick_mat

  integer(int64) function grid_search(n, quarry, grid, offset)
    integer(int64), intent(in) :: n, offset
    real(real64), intent(in) :: quarry
    type(nuclide_grid_point), intent(in) :: grid(0:)
    integer(int64) :: lower, upper, mid, length
    lower = offset
    upper = offset + n - 1
    length = upper - lower
    do while (length > 1)
      mid = lower + length / 2
      if (grid(mid)%energy > quarry) then
        upper = mid
      else
        lower = mid
      end if
      length = upper - lower
    end do
    grid_search = lower
  end function grid_search

  subroutine calculate_micro_xs(p_energy, nuc, n_isotopes, n_gridpoints, nuclide_grid_data, xs)
    real(real64), intent(in) :: p_energy
    integer, intent(in) :: nuc
    integer(int64), intent(in) :: n_isotopes, n_gridpoints
    type(nuclide_grid_point), intent(in) :: nuclide_grid_data(0:)
    real(real64), intent(out) :: xs(0:4)
    integer(int64) :: low, high
    real(real64) :: f
    low = grid_search(n_gridpoints, p_energy, nuclide_grid_data, int(nuc, int64) * n_gridpoints)
    high = min(low + 1_int64, int(nuc, int64) * n_gridpoints + n_gridpoints - 1_int64)
    f = (p_energy - nuclide_grid_data(low)%energy) / max(nuclide_grid_data(high)%energy - nuclide_grid_data(low)%energy, 1.0e-12_real64)
    xs(0) = nuclide_grid_data(low)%total_xs + f * (nuclide_grid_data(high)%total_xs - nuclide_grid_data(low)%total_xs)
    xs(1) = nuclide_grid_data(low)%elastic_xs + f * (nuclide_grid_data(high)%elastic_xs - nuclide_grid_data(low)%elastic_xs)
    xs(2) = nuclide_grid_data(low)%absorbtion_xs + f * (nuclide_grid_data(high)%absorbtion_xs - nuclide_grid_data(low)%absorbtion_xs)
    xs(3) = nuclide_grid_data(low)%fission_xs + f * (nuclide_grid_data(high)%fission_xs - nuclide_grid_data(low)%fission_xs)
    xs(4) = nuclide_grid_data(low)%nu_fission_xs + f * (nuclide_grid_data(high)%nu_fission_xs - nuclide_grid_data(low)%nu_fission_xs)
  end subroutine calculate_micro_xs

  subroutine calculate_macro_xs(p_energy, mat, in, sd, macro)
    real(real64), intent(in) :: p_energy
    integer, intent(in) :: mat
    type(inputs_t), intent(in) :: in
    type(simulation_data_t), intent(in) :: sd
    real(real64), intent(out) :: macro(0:4)
    real(real64) :: micro(0:4)
    integer :: i, nuc, n_nucs, channel
    macro = 0.0_real64
    n_nucs = sd%num_nucs(mat)
    do i = 0, n_nucs - 1
      nuc = sd%mats(mat * sd%max_num_nucs + i)
      call calculate_micro_xs(p_energy, nuc, in%n_isotopes, in%n_gridpoints, sd%nuclide_grid, micro)
      do channel = 0, 4
        macro(channel) = macro(channel) + micro(channel) * sd%concs(mat * sd%max_num_nucs + i)
      end do
    end do
  end subroutine calculate_macro_xs

  function run_event_based_simulation(in, sd, kernel_time) result(verification_scalar)
    type(inputs_t), intent(in) :: in
    type(simulation_data_t), intent(in) :: sd
    real(real64), intent(out) :: kernel_time
    integer(int64) :: verification_scalar
    integer, allocatable :: verification(:)
    integer :: i, n, mat, j, max_idx
    integer(int64) :: seed
    real(real64) :: p_energy, macro(0:4), maxv, t0, t1
    print '(a)', 'Beginning event based simulation...'
    print '(a,f8.1,a)', 'Allocating an additional ', real(in%lookups * 4, real64) / 1024.0_real64 / 1024.0_real64, ' MB of memory for verification arrays...'
    allocate(verification(0:in%lookups-1))
!$omp target data map(to:sd%num_nucs,sd%concs,sd%mats,sd%nuclide_grid) map(from:verification)
    t0 = wall_seconds()
    do n = 1, in%kernel_repeat
!$omp target teams distribute parallel do firstprivate(in) thread_limit(256) private(i,seed,p_energy,mat,macro,maxv,max_idx,j)
      do i = 0, in%lookups - 1
        seed = fast_forward_lcg(starting_seed, int(2*i, int64))
        p_energy = lcg_random_double(seed)
        mat = pick_mat(seed)
        call calculate_macro_xs(p_energy, mat, in, sd, macro)
        maxv = -1.0_real64
        max_idx = 0
        do j = 0, 4
          if (macro(j) > maxv) then
            maxv = macro(j)
            max_idx = j
          end if
        end do
        verification(i) = max_idx + 1
      end do
!$omp end target teams distribute parallel do
    end do
    t1 = wall_seconds()
    kernel_time = (t1 - t0) / real(in%kernel_repeat, real64)
!$omp end target data
    verification_scalar = sum(int(verification, int64))
  end function run_event_based_simulation

  function wall_seconds() result(t)
    real(real64) :: t
    integer(int64) :: count, rate
    call system_clock(count, rate)
    t = real(count, real64) / real(rate, real64)
  end function wall_seconds
end module xsbench_simulation
