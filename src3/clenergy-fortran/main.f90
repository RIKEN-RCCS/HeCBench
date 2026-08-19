module clenergy_support
  use iso_fortran_env, only: real32, real64
  use iso_c_binding, only: c_int
  implicit none
  integer, parameter :: maxatoms = 4000, unrollx = 8, blocksizex = 8
  type :: float4
    sequence
    real(real32) :: x, y, z, w
  end type float4
  interface
    subroutine c_srand(seed) bind(C, name='srand')
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand
    function c_rand() bind(C, name='rand') result(value)
      import :: c_int
      integer(c_int) :: value
    end function c_rand
  end interface
contains
  subroutine initatoms(atoms, count, vx, vy, vz, spacing)
    real(real32), allocatable, intent(out) :: atoms(:)
    integer, intent(in) :: count, vx, vy, vz
    real(real32), intent(in) :: spacing
    integer :: i, addr
    real(real32) :: sx, sy, sz
    call c_srand(2_c_int)
    allocate(atoms(0:count*4-1))
    sx = spacing*real(vx,real32); sy = spacing*real(vy,real32); sz = spacing*real(vz,real32)
    do i = 0, count-1
      addr = 4*i
      atoms(addr)   = real(c_rand(),real32)/2147483647.0_real32*sx
      atoms(addr+1) = real(c_rand(),real32)/2147483647.0_real32*sy
      atoms(addr+2) = real(c_rand(),real32)/2147483647.0_real32*sz
      atoms(addr+3) = real(c_rand(),real32)/2147483647.0_real32*2.0_real32 - 1.0_real32
    end do
  end subroutine initatoms

  subroutine copyatoms(atoms, count, zplane, atominfo)
    real(real32), intent(in) :: atoms(0:), zplane
    integer, intent(in) :: count
    type(float4), intent(inout) :: atominfo(0:)
    integer :: i
    real(real32) :: dz
    do i = 0, count-1
      atominfo(i)%x = atoms(4*i); atominfo(i)%y = atoms(4*i+1)
      dz = zplane-atoms(4*i+2); atominfo(i)%z = dz*dz; atominfo(i)%w = atoms(4*i+3)
    end do
    !$omp target update to(atominfo(0:count))
  end subroutine copyatoms
end module clenergy_support

program main
  use iso_fortran_env, only: real32, real64
  use omp_lib, only: omp_get_wtime
  use clenergy_support
  implicit none
  integer, parameter :: atomcount=1000000, vx=768, vy=768, vz=1, volmem=vx*vy*vz
  real(real32), parameter :: gridspacing=0.1_real32
  real(real32), allocatable :: atoms(:), energy(:)
  type(float4), allocatable :: atominfo(:)
  integer :: atomstart, runatoms, remaining, iterations, state, i, j, yindex, xindex, atomid, outaddr
  real(real64) :: copytotal, runtotal, mastertotal, hostcopytotal, atomevalssec
  real(real64) :: master_start, master_stop, copy_start, copy_stop, run_start, run_stop, host_start, host_stop
  real(real32) :: coory, coorx, spacing_u, dy, dyz2, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8
  real(real32) :: e1,e2,e3,e4,e5,e6,e7,e8
  character(len=4), parameter :: statestr='|/-\\'

  print '(a)', 'GPU accelerated coulombic potential microbenchmark'
  print '(a)', '--------------------------------------------------------'
  print '(a)', '  Single-threaded single-device test run.'
  print '(a,3(i0,a))', 'Grid size: ', vx, ' x ', vy, ' x ', vz
  print '(a,i0,a,f0.1,a,i0,a)', 'Running kernel(atoms:',atomcount,', gridspacing ',gridspacing,', z ',0,')'
  call initatoms(atoms, atomcount, vx, vy, vz, gridspacing)
  print '(a,f0.2,a)', 'Allocating ', real(4*volmem,real64)/(1024.0_real64*1024.0_real64), 'MB of memory for output buffer...'
  allocate(energy(0:volmem-1), atominfo(0:maxatoms-1))
  print '(a)', 'starting run...'
  copytotal=0.0_real64; runtotal=0.0_real64; hostcopytotal=0.0_real64; iterations=0; state=0
  master_start = omp_get_wtime()
  !$omp target enter data map(alloc:atominfo,energy)
  !$omp target teams distribute parallel do simd
  do i = 0, volmem-1
    energy(i)=0.0_real32
  end do
  !$omp end target teams distribute parallel do simd
  do atomstart = 0, atomcount-1, maxatoms
    iterations=iterations+1; remaining=atomcount-atomstart; runatoms=min(maxatoms,remaining)
    write(*,'(a)',advance='no') statestr(state+1:state+1)//achar(13); flush(6); state=iand(state+1,3)
    copy_start = omp_get_wtime()
    call copyatoms(atoms(4*atomstart:), runatoms, 0.0_real32*gridspacing, atominfo)
    copy_stop = omp_get_wtime(); copytotal=copytotal + copy_stop - copy_start
    run_start = omp_get_wtime()
    !$omp target teams distribute parallel do collapse(2)
    do yindex=0,vy-1
      do xindex=0,vx/unrollx-1
        outaddr=yindex*vx+xindex; coory=gridspacing*real(yindex,real32); coorx=gridspacing*real(xindex,real32)
        e1=0.0_real32; e2=0.0_real32; e3=0.0_real32; e4=0.0_real32; e5=0.0_real32; e6=0.0_real32; e7=0.0_real32; e8=0.0_real32
        spacing_u=gridspacing*real(blocksizex,real32)
        do atomid=0,runatoms-1
          dy=coory-atominfo(atomid)%y; dyz2=dy*dy+atominfo(atomid)%z
          dx1=coorx-atominfo(atomid)%x; dx2=dx1+spacing_u; dx3=dx2+spacing_u; dx4=dx3+spacing_u
          dx5=dx4+spacing_u; dx6=dx5+spacing_u; dx7=dx6+spacing_u; dx8=dx7+spacing_u
          e1=e1+atominfo(atomid)%w/sqrt(dx1*dx1+dyz2); e2=e2+atominfo(atomid)%w/sqrt(dx2*dx2+dyz2)
          e3=e3+atominfo(atomid)%w/sqrt(dx3*dx3+dyz2); e4=e4+atominfo(atomid)%w/sqrt(dx4*dx4+dyz2)
          e5=e5+atominfo(atomid)%w/sqrt(dx5*dx5+dyz2); e6=e6+atominfo(atomid)%w/sqrt(dx6*dx6+dyz2)
          e7=e7+atominfo(atomid)%w/sqrt(dx7*dx7+dyz2); e8=e8+atominfo(atomid)%w/sqrt(dx8*dx8+dyz2)
        end do
        energy(outaddr)=energy(outaddr)+e1; energy(outaddr+blocksizex)=energy(outaddr+blocksizex)+e2
        energy(outaddr+2*blocksizex)=energy(outaddr+2*blocksizex)+e3; energy(outaddr+3*blocksizex)=energy(outaddr+3*blocksizex)+e4
        energy(outaddr+4*blocksizex)=energy(outaddr+4*blocksizex)+e5; energy(outaddr+5*blocksizex)=energy(outaddr+5*blocksizex)+e6
        energy(outaddr+6*blocksizex)=energy(outaddr+6*blocksizex)+e7; energy(outaddr+7*blocksizex)=energy(outaddr+7*blocksizex)+e8
      end do
    end do
    !$omp end target teams distribute parallel do
    run_stop = omp_get_wtime(); runtotal=runtotal + run_stop - run_start
  end do
  print '(a)', 'Done'
  master_stop = omp_get_wtime(); mastertotal = master_stop - master_start
  host_start = omp_get_wtime()
  !$omp target exit data map(from:energy) map(delete:atominfo)
  host_stop = omp_get_wtime(); hostcopytotal = host_stop - host_start
  do j=0,7
    do i=0,7
      write(*,'(a,i0,a,f0.1,1x)',advance='no') '[',j*vx+i,'] ',energy(j*vx+i)
    end do
    print *
  end do
  print '(a,i0,a,i0,a)', 'Final calculation required ',iterations,' iterations of ',maxatoms,' atoms'
  print '(a,f0.6,a,f0.6)', 'Copy time: ',copytotal,' seconds, ',copytotal/real(iterations,real64),' per iteration'
  print '(a,f0.6,a,f0.6)', 'Kernel time: ',runtotal,' seconds, ',runtotal/real(iterations,real64),' per iteration'
  print '(a,f0.6,a)', 'Total time: ',mastertotal,' seconds'
  print '(a,f0.6,a)', 'Kernel invocation rate: ',real(iterations,real64)/mastertotal,' iterations per second'
  print '(a,f0.6,a,f0.6,a)', 'GPU to host copy bandwidth: ', &
      (real(4*volmem,real64)/(1024.0_real64*1024.0_real64))/hostcopytotal, &
      'MB/sec, ',hostcopytotal,' seconds total'
  atomevalssec=real(vx*vy*vz,real64)*real(atomcount,real64)/(mastertotal*1.0e9_real64)
  print '(a,g0,a)', 'Efficiency metric, ',atomevalssec,' billion atom evals per second'
  print '(a,g0,a)', 'FP performance: ',atomevalssec*(59.0_real64/8.0_real64),' GFLOPS'
end program main
