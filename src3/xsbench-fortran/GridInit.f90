module xsbench_gridinit
  use iso_fortran_env, only: real64, int64
  use xsbench_simulation
  implicit none
contains
  subroutine grid_init_do_not_profile(in, sd)
    type(inputs_t), intent(in) :: in
    type(simulation_data_t), intent(out) :: sd
    integer :: mat, i, nuc
    integer(int64) :: iso, gp, idx
    sd%max_num_nucs = 5
    allocate(sd%num_nucs(0:4), sd%mats(0:5*sd%max_num_nucs-1), sd%concs(0:5*sd%max_num_nucs-1))
    sd%num_nucs = [2, 3, 4, 5, 2]
    do mat = 0, 4
      do i = 0, sd%max_num_nucs - 1
        sd%mats(mat*sd%max_num_nucs+i) = mod(mat*17 + i*13, int(in%n_isotopes))
        sd%concs(mat*sd%max_num_nucs+i) = 0.01_real64 * real(i + 1, real64)
      end do
    end do
    allocate(sd%nuclide_grid(0:in%n_isotopes*in%n_gridpoints-1))
    do iso = 0, in%n_isotopes - 1
      do gp = 0, in%n_gridpoints - 1
        idx = iso * in%n_gridpoints + gp
        sd%nuclide_grid(idx)%energy = real(gp, real64) / real(in%n_gridpoints, real64)
        sd%nuclide_grid(idx)%total_xs = 1.0_real64 + 0.0001_real64 * real(iso + gp, real64)
        sd%nuclide_grid(idx)%elastic_xs = 0.5_real64 + 0.00007_real64 * real(iso + gp, real64)
        sd%nuclide_grid(idx)%absorbtion_xs = 0.2_real64 + 0.00005_real64 * real(iso + gp, real64)
        sd%nuclide_grid(idx)%fission_xs = 0.1_real64 + 0.00003_real64 * real(iso + gp, real64)
        sd%nuclide_grid(idx)%nu_fission_xs = 0.05_real64 + 0.00002_real64 * real(iso + gp, real64)
      end do
    end do
    allocate(sd%unionized_energy_array(0:0), sd%index_grid(0:0))
    sd%unionized_energy_array = 0.0_real64
    sd%index_grid = 0
  end subroutine grid_init_do_not_profile
end module xsbench_gridinit
