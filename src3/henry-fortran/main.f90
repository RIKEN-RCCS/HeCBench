module henry_kernel
  use iso_fortran_env,only:int64,real64
  implicit none
  real(real64),parameter::temperature=298.0_real64,gas_constant=8.314_real64
!$omp declare target (lcg,boltzmann)
contains
  function lcg(seed) result(value)
    integer(int64),intent(inout)::seed
    real(real64)::value
    integer(int64),parameter::a=2806196910506780709_int64
    ! The source uses uint64 arithmetic modulo 2^63.  IAND preserves the
    ! low 63 bits without emitting the unsupported NVFORTRAN modulo helper.
    seed=iand(a*seed+1_int64,int(z'7FFFFFFFFFFFFFFF',int64))
    value=real(seed,real64)/9223372036854775808.0_real64
  end function
  function boltzmann(x,y,z,atoms,natoms,box_length) result(value)
    integer,intent(in)::natoms
    real(real64),intent(in)::x,y,z,atoms(0:,0:),box_length
    real(real64)::value,e,dx,dy,dz,rinv,sigr,s6,s12,upper,lower
    integer::i
    e=0.0_real64;upper=0.5_real64*box_length;lower=-upper
    do i=0,natoms-1
      dx=x-atoms(0,i);dy=y-atoms(1,i);dz=z-atoms(2,i)
      if(dx>upper)dx=dx-box_length;if(dx>upper)dx=dx-box_length
      if(dy>upper)dy=dy-box_length;if(dy<=lower)dy=dy-box_length
      if(dz<=lower)dz=dz-box_length;if(dz<=lower)dz=dz-box_length
      rinv=1.0_real64/sqrt(dx*dx+dy*dy+dz*dz);sigr=rinv*atoms(4,i);s6=sigr**6;s12=s6*s6;e=e+4.0_real64*atoms(3,i)*(s12-s6)
    end do
    value=exp(-e/(gas_constant*temperature))
  end function
  subroutine henry_gpu(atoms,natoms,box_length,factors)
    integer,intent(in)::natoms
    real(real64),intent(in)::atoms(0:4,0:natoms-1),box_length
    real(real64),intent(out)::factors(0:262143)
    integer::id
    integer(int64)::seed
    real(real64)::x,y,z
!$omp target teams distribute parallel do thread_limit(256) private(seed,x,y,z)
    do id=0,262143
      seed=int(id,int64);x=box_length*lcg(seed);y=box_length*lcg(seed);z=box_length*lcg(seed);factors(id)=boltzmann(x,y,z,atoms,natoms,box_length)
    end do
!$omp end target teams distribute parallel do
  end subroutine
end module
program henry
  use iso_fortran_env,only:int64,real64
  use henry_kernel
  implicit none
  integer::argc,unit,ios,ncycles,natoms,i,cycle,atomno
  integer(int64)::t0,t1,rate
  real(real64)::box_length,xf,yf,zf,total_time,kh
  real(real64),allocatable::atoms(:,:),factors(:)
  character(len=1024)::file,line,element;character(len=64)::arg
  argc=command_argument_count();if(argc/=2)then;print '(a)','Usage: ./main <material file> <ninsertions>';stop 1;end if
  call get_command_argument(1,file);call get_command_argument(2,arg);read(arg,*)ncycles
  open(newunit=unit,file=trim(file),status='old',action='read',iostat=ios);if(ios/=0)then;print '(a,a)','Failed to import file ',trim(file);stop 1;end if
  read(unit,'(A)')line;read(line,*)box_length;print '(a,f0.6)','L = ',box_length
  read(unit,'(A)')line;read(unit,'(A)')line;read(line,*)natoms;print '(i0,a)',natoms,' atoms';read(unit,'(A)')line
  allocate(atoms(0:4,0:natoms-1),factors(0:262143))
  do i=0,natoms-1
    read(unit,*)atomno,element,xf,yf,zf;atoms(0,i)=box_length*xf;atoms(1,i)=box_length*yf;atoms(2,i)=box_length*zf
    select case(trim(element));case('Zn');atoms(3,i)=96.152688_real64;atoms(4,i)=3.095775_real64;case('O');atoms(3,i)=66.884614_real64;atoms(4,i)=3.424075_real64;case('C');atoms(3,i)=88.480032_real64;atoms(4,i)=3.580425_real64;case default;atoms(3,i)=57.276566_real64;atoms(4,i)=3.150565_real64;end select
  end do;close(unit)
!$omp target data map(to:atoms(0:4,0:natoms-1)) map(alloc:factors(0:262143))
  total_time=0.0_real64;kh=0.0_real64
  do cycle=1,ncycles
    call system_clock(t0,rate);call henry_gpu(atoms,natoms,box_length,factors);call system_clock(t1);total_time=total_time+real(t1-t0,real64)/rate
!$omp target update from(factors(0:262143))
    kh=kh+sum(factors)
  end do
!$omp end target data
  kh=kh/real(ncycles*262144,real64)/(gas_constant*temperature)
  print '(a)','Used 1024 blocks with 256 thread each';print '(a,es15.6,a)','Henry constant = ',kh,' mol/(m3 - Pa)';print '(a,i0)','Number of actual insertions: ',ncycles*262144;print '(a,i0)','Number of times we called the device kernel: ',ncycles;print '(a,f0.6,a)','Average kernel execution time ',total_time/ncycles,' (s)'
end program
