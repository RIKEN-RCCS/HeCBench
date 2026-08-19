module fileio_mod
  use iso_fortran_env, only: real64
  use sph_types
  implicit none
  integer, save :: file_num = 0
contains

  subroutine write_file(particles, params)
    type(fluid_particle), intent(in) :: particles(0:)
    type(param), intent(in) :: params
    character(len=64) :: name
    integer :: i, unit

    write(name, '("sim-",i0,".csv")') file_num
    open(newunit=unit, file=trim(name), status='replace', action='write')
    do i = 0, params%number_fluid_particles - 1
      write(unit, '(f0.6,",",f0.6,",",f0.6)') particles(i)%pos%x, particles(i)%pos%y, particles(i)%pos%z
    end do
    close(unit)
    file_num = file_num + 1
    print '(a,a)', 'wrote file: ', trim(name)
  end subroutine write_file

  subroutine write_boundary_file(boundary, params)
    type(boundary_particle), intent(in) :: boundary(0:)
    type(param), intent(in) :: params
    character(len=64) :: name
    integer :: i, unit

    write(name, '("boundary-",i0,".csv")') file_num
    open(newunit=unit, file=trim(name), status='replace', action='write')
    do i = 0, params%number_boundary_particles - 1
      write(unit, '(f0.6,",",f0.6,",",f0.6)') boundary(i)%pos%x, boundary(i)%pos%y, boundary(i)%pos%z
    end do
    close(unit)
    print '(a,a)', 'wrote file: ', trim(name)
  end subroutine write_boundary_file
end module fileio_mod
