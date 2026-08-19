program log2_main
  use iso_fortran_env, only: real32, real64
  use omp_lib; use log2_kernel; implicit none
  integer::repeat,precision_count,ceilingVal,ninputs,i,j,k,unit,ios
  character(len=256)::cfg,placeholder
  integer,allocatable::precision(:)
  real(real32),allocatable::inputs(:),outputs(:),ref(:)
  real(real32)::s
  real(real64)::start,elapsed
  if(command_argument_count()/=1)then; print '(a)','Usage: ./main <config filename>'; stop 1; end if
  call get_command_argument(1,cfg); open(newunit=unit,file=trim(cfg),status='old',iostat=ios)
  if(ios/=0)then; print '(a)','Usage: ./main <config filename>'; stop 1; else; read(unit,*)placeholder,ceilingVal; read(unit,*)placeholder,repeat; read(unit,*)placeholder,precision_count; allocate(precision(0:precision_count-1)); read(unit,*)placeholder,precision; close(unit); end if
  ninputs=ceilingVal; allocate(inputs(0:ninputs-1),outputs(0:ninputs*precision_count-1),ref(0:ninputs-1)); do i=0,ninputs-1; inputs(i)=real(i+1,real32); ref(i)=log(inputs(i))/log(2.0_real32); end do
  print '(a,i0)','Number of precision counts : ',precision_count; print '(a,i0)',' Number of inputs to evaluate for each precision: ',ninputs; print '(a,i0)',' Number of runs for each precision : ',repeat
!$omp target data map(to:inputs,precision) map(from:outputs)
  do i=0,precision_count-1; start=omp_get_wtime(); do k=1,repeat
!$omp target teams distribute parallel do thread_limit(256)
    do j=0,ninputs-1; outputs(i*ninputs+j)=binary_log(inputs(j),precision(i)); end do
!$omp end target teams distribute parallel do
  end do; elapsed=omp_get_wtime()-start; print '(a,i0,a)','Iterative approximation with ',precision(i),' bits of precision'; print '(a,f12.6,a)','Average kernel execution time ',elapsed*1.0e6_real64/repeat,' (us)'; end do
!$omp end target data
  print '(a)','-------------- SUMMARY (Device results): --------------'; do i=0,precision_count-1; s=sum((outputs(i*ninputs:i*ninputs+ninputs-1)-ref)**2)/ninputs; print '(a,i0,a)','----- Iterative approximation with ',precision(i),' bits of precision -----'; print '(a,f12.6)','RMSE : ',sqrt(s); end do
end program
