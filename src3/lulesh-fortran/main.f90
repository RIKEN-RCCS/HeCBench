program lulesh
  use omp_lib
  use lulesh_types
  use lulesh_kernels
  implicit none
  type(cmd_options)::opts
  type(domain_t)::dom
  integer::argc,i
  character(len=128)::arg
  real(8)::start_time,elapsed,grind
  call parse_options(opts)
  if (opts%quiet == 0) then
    print '(a,i0,a)', 'Running problem size ', opts%nx, '^3 per domain until completion'
    print '(a,i0)', 'Num processors: ', 1
    print '(a,i0)', 'Num threads (hardcoded): ', 256
    print '(a,i0)', 'Total number of elements: ', opts%nx*opts%nx*opts%nx
    print *
  end if
  call initialize_domain(dom,opts%nx)
  print '(a,i0,a,i0)', 'numNode=', dom%num_node, ' numElem=', dom%num_elem
!$omp target data map(tofrom:dom%x(0:dom%num_node-1),dom%y(0:dom%num_node-1),dom%z(0:dom%num_node-1),dom%xd(0:dom%num_node-1),dom%yd(0:dom%num_node-1),dom%zd(0:dom%num_node-1),dom%xdd(0:dom%num_node-1),dom%ydd(0:dom%num_node-1),dom%zdd(0:dom%num_node-1),dom%fx(0:dom%num_node-1),dom%fy(0:dom%num_node-1),dom%fz(0:dom%num_node-1),dom%nodal_mass(0:dom%num_node-1),dom%e(0:dom%num_elem-1),dom%p(0:dom%num_elem-1),dom%q(0:dom%num_elem-1),dom%ql(0:dom%num_elem-1),dom%qq(0:dom%num_elem-1),dom%v(0:dom%num_elem-1),dom%volo(0:dom%num_elem-1),dom%delv(0:dom%num_elem-1),dom%vdov(0:dom%num_elem-1),dom%arealg(0:dom%num_elem-1),dom%ss(0:dom%num_elem-1),dom%elem_mass(0:dom%num_elem-1),dom%nodelist(0:8*dom%num_elem-1))
  start_time=omp_get_wtime()
  do while (dom%time < dom%stoptime .and. dom%cycle < opts%its)
    call lagrange_leapfrog(dom)
    if (opts%show_progress /= 0) print '(a,i0,a,es12.5,a,es12.5)', 'cycle = ', dom%cycle, ', time = ', dom%time, ', dt=', dom%deltatime
  end do
  elapsed=omp_get_wtime()-start_time
!$omp target update from(dom%e(0:dom%num_elem-1))
!$omp end target data
  call verify_and_write(dom,opts,elapsed)
contains
  subroutine parse_options(opts)
    type(cmd_options),intent(inout)::opts
    integer::j,argc_local
    character(len=128)::a,v
    argc_local=command_argument_count(); j=1
    do while (j <= argc_local)
      call get_command_argument(j,a)
      select case(trim(a))
      case('-i'); j=j+1; call get_command_argument(j,v); read(v,*)opts%its
      case('-s'); j=j+1; call get_command_argument(j,v); read(v,*)opts%nx
      case('-r'); j=j+1; call get_command_argument(j,v); read(v,*)opts%num_reg
      case('-b'); j=j+1; call get_command_argument(j,v); read(v,*)opts%balance
      case('-c'); j=j+1; call get_command_argument(j,v); read(v,*)opts%cost
      case('-p'); opts%show_progress=1
      case('-q'); opts%quiet=1
      end select
      j=j+1
    end do
  end subroutine parse_options
  subroutine verify_and_write(dom,opts,elapsed)
    type(domain_t),intent(in)::dom
    type(cmd_options),intent(in)::opts
    real(8),intent(in)::elapsed
    real(RealK)::max_abs_diff,total_abs_diff,max_rel_diff,ref
    integer::j,k,elem
    ref=dom%e(0); max_abs_diff=0.0_RealK; total_abs_diff=0.0_RealK; max_rel_diff=0.0_RealK
    do k=0,opts%nx-1; do j=0,opts%nx-1
      elem=elem_index(j,k,0,opts%nx)
      max_abs_diff=max(max_abs_diff,abs(dom%e(elem)-ref))
      total_abs_diff=total_abs_diff+abs(dom%e(elem)-ref)
      if (ref /= 0.0_RealK) max_rel_diff=max(max_rel_diff,abs((dom%e(elem)-ref)/ref))
    end do; end do
    print '(a)', 'Run completed:  '
    print '(a,i0)', '   Problem size        =  ', opts%nx
    print '(a,i0)', '   MPI tasks           =  ', 1
    print '(a,i0)', '   Iteration count     =  ', dom%cycle
    print '(a,es12.6)', '   Final Origin Energy = ', dom%e(0)
    print '(a)', '   Testing Plane 0 of Energy Array on rank 0:'
    print '(a,es12.6)', '        MaxAbsDiff   = ', max_abs_diff
    print '(a,es12.6)', '        TotalAbsDiff = ', total_abs_diff
    print '(a,es12.6)', '        MaxRelDiff   = ', max_rel_diff
    grind=elapsed*1.0e6/(real(dom%cycle,8)*real(dom%num_elem,8))
    print '(a,f10.2,a)', 'Elapsed time         = ', elapsed, ' (s)'
    print '(a,es12.5)', 'Grind time (us/z/c)  = ', grind
    print '(a,es12.5)', 'FOM                  = ', 1000.0_RealK/grind
  end subroutine verify_and_write
end program lulesh
