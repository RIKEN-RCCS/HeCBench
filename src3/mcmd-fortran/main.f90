program main
  use iso_fortran_env, only: real64
  use omp_lib
  use force_kernel_mod
  implicit none
  integer :: n_atoms, i
  character(len=256) :: input_file
  type(d_atom), allocatable :: atoms(:)
  real(real64) :: basis(0:8), rbasis(0:8), start_time, elapsed

  if (command_argument_count() /= 1) then
    print '(a)', 'Usage: ./main mcmd.inp'
    stop 1
  end if
  call get_command_argument(1,input_file)
  n_atoms = 1024
  allocate(atoms(0:n_atoms-1))
  basis = [20.0_real64,0.0_real64,0.0_real64, 0.0_real64,20.0_real64,0.0_real64, 0.0_real64,0.0_real64,20.0_real64]
  rbasis = [0.05_real64,0.0_real64,0.0_real64, 0.0_real64,0.05_real64,0.0_real64, 0.0_real64,0.0_real64,0.05_real64]
  do i=0,n_atoms-1
    atoms(i)%pos = [mod(real(i,real64)*0.173_real64,20.0_real64), mod(real(i,real64)*0.271_real64,20.0_real64), mod(real(i,real64)*0.319_real64,20.0_real64)]
    atoms(i)%eps = 0.2_real64 + 0.001_real64*mod(i,7)
    atoms(i)%sig = 3.0_real64 + 0.01_real64*mod(i,5)
    atoms(i)%charge = merge(0.1_real64,-0.1_real64,mod(i,2)==0)
    atoms(i)%molid = i/3
    atoms(i)%frozen = 0
    atoms(i)%u = [0.01_real64,0.02_real64,0.03_real64]
    atoms(i)%polar = 0.5_real64
  end do
  start_time = omp_get_wtime()
  call force_kernel(n_atoms,256,2,12.0_real64,0.25_real64,2,1,0.39_real64,basis,rbasis,atoms)
  elapsed = omp_get_wtime()-start_time
  print '(a,a)', 'Input file: ', trim(input_file)
  print '(a,f0.6,a)', 'Average kernel execution time: ', elapsed, ' (s)'
  print '(a,es16.8)', 'Force checksum: ', sum(atoms%f)
end program main
