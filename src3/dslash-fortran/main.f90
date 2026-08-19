module dslash_module
  use, intrinsic :: iso_fortran_env, only : real64
  implicit none
  integer, parameter :: ldim = 32, directions = 4, colors = 3, iterations_default = 100, warmups = 1
  integer, parameter :: total_sites = ldim**4, even_sites = total_sites/2
  real(real64), parameter :: epsilon = 2.0e-6_real64
contains
  integer function node_index(x, y, z, t) result(index)
    integer, intent(in) :: x, y, z, t
    integer :: xr, yr, zr, tr, linear
    xr = modulo(x+ldim, ldim); yr = modulo(y+ldim, ldim)
    zr = modulo(z+ldim, ldim); tr = modulo(t+ldim, ldim)
    linear = xr + ldim*(yr + ldim*(zr + ldim*tr))
    if (modulo(x+y+z+t, 2) == 0) then
      index = linear/2
    else
      index = (linear+total_sites)/2
    end if
  end function node_index

  subroutine set_neighbors(fwd, bck, fwd3, bck3)
    integer, intent(out) :: fwd(0:), bck(0:), fwd3(0:), bck3(0:)
    integer :: x, y, z, t, site
    do x = 0, ldim-1
      do y = 0, ldim-1
        do z = 0, ldim-1
          do t = 0, ldim-1
            site = node_index(x,y,z,t)
            fwd(4*site) = node_index(x+1,y,z,t); bck(4*site) = node_index(x-1,y,z,t)
            fwd(4*site+1) = node_index(x,y+1,z,t); bck(4*site+1) = node_index(x,y-1,z,t)
            fwd(4*site+2) = node_index(x,y,z+1,t); bck(4*site+2) = node_index(x,y,z-1,t)
            fwd(4*site+3) = node_index(x,y,z,t+1); bck(4*site+3) = node_index(x,y,z,t-1)
            fwd3(4*site) = node_index(x+3,y,z,t); bck3(4*site) = node_index(x-3,y,z,t)
            fwd3(4*site+1) = node_index(x,y+3,z,t); bck3(4*site+1) = node_index(x,y-3,z,t)
            fwd3(4*site+2) = node_index(x,y,z+3,t); bck3(4*site+2) = node_index(x,y,z-3,t)
            fwd3(4*site+3) = node_index(x,y,z,t+3); bck3(4*site+3) = node_index(x,y,z,t-3)
          end do
        end do
      end do
    end do
  end subroutine set_neighbors

  subroutine make_data(src, fat, lng)
    complex(real64), intent(out) :: src(0:2,0:total_sites-1), fat(0:2,0:2,0:3,0:total_sites-1), &
      lng(0:2,0:2,0:3,0:total_sites-1)
    real(real64) :: r, im
    integer :: site, direction
    do site = 0, total_sites-1
      call random_number(r); call random_number(im)
      src(:,site) = cmplx(-1.0_real64+2.0_real64*r, -1.0_real64+2.0_real64*im, real64)
      do direction = 0, directions-1
        call random_number(r); call random_number(im)
        fat(:,:,direction,site) = cmplx(-1.0_real64+2.0_real64*r, -1.0_real64+2.0_real64*im, real64)
        call random_number(r); call random_number(im)
        lng(:,:,direction,site) = cmplx(-1.0_real64+2.0_real64*r, -1.0_real64+2.0_real64*im, real64)
      end do
    end do
  end subroutine make_data

  subroutine dslash_field(src, dst, fat, lng, fatbck, lngbck, fwd, bck, fwd3, bck3)
    complex(real64), intent(in) :: src(0:2,0:total_sites-1), fat(0:2,0:2,0:3,0:total_sites-1), &
      lng(0:2,0:2,0:3,0:total_sites-1)
    complex(real64), intent(in) :: fatbck(0:2,0:2,0:3,0:total_sites-1), lngbck(0:2,0:2,0:3,0:total_sites-1)
    complex(real64), intent(out) :: dst(0:2,0:total_sites-1)
    integer, intent(in) :: fwd(0:), bck(0:), fwd3(0:), bck3(0:)
    complex(real64) :: temp(0:2)
    integer :: site, direction, row, col, neighbor
    do site = 0, even_sites-1
      do row = 0, 2
        dst(row,site) = cmplx(0.0_real64,0.0_real64,real64)
      end do
      do direction = 0, 3
        neighbor = fwd(4*site+direction)
        do row = 0, 2
          do col = 0, 2
            dst(row,site) = dst(row,site) + fat(row,col,direction,site)*src(col,neighbor)
          end do
        end do
      end do
      do row = 0, 2
        temp(row) = cmplx(0.0_real64,0.0_real64,real64)
      end do
      do direction = 0, 3
        neighbor = fwd3(4*site+direction)
        do row = 0, 2
          do col = 0, 2
            temp(row) = temp(row) + lng(row,col,direction,site)*src(col,neighbor)
          end do
        end do
      end do
      dst(:,site) = dst(:,site) + temp
      do row = 0, 2
        temp(row) = cmplx(0.0_real64,0.0_real64,real64)
      end do
      do direction = 0, 3
        neighbor = bck(4*site+direction)
        do row = 0, 2
          do col = 0, 2
            temp(row) = temp(row) + fatbck(row,col,direction,site)*src(col,neighbor)
          end do
        end do
      end do
      dst(:,site) = dst(:,site) - temp
      do row = 0, 2
        temp(row) = cmplx(0.0_real64,0.0_real64,real64)
      end do
      do direction = 0, 3
        neighbor = bck3(4*site+direction)
        do row = 0, 2
          do col = 0, 2
            temp(row) = temp(row) + lngbck(row,col,direction,site)*src(col,neighbor)
          end do
        end do
      end do
      dst(:,site) = dst(:,site) - temp
    end do
  end subroutine dslash_field
end module dslash_module

program dslash
  use, intrinsic :: iso_fortran_env, only : real64
  use omp_lib
  use dslash_module
  implicit none
  complex(real64), allocatable :: src(:,:), dst(:,:), chkdst(:,:)
  complex(real64), allocatable :: fat(:,:,:,:), lng(:,:,:,:), fatbck(:,:,:,:), lngbck(:,:,:,:)
  integer, allocatable :: fwd(:), bck(:), fwd3(:), bck3(:)
  complex(real64) :: temp(0:2)
  integer :: argc, workgroup_size, iteration, site, direction, row, col, neighbor, i
  real(real64) :: start_time, total_time, tflop, memory_usage, memory_allocated
  character(len=64) :: argument
  argc = command_argument_count()
  if (argc < 1) then
    print '(a)', 'Usage <workgroup size>'; stop 1
  end if
  call get_command_argument(1, argument); read(argument, *) workgroup_size
  allocate(src(0:2,0:total_sites-1), dst(0:2,0:total_sites-1), chkdst(0:2,0:total_sites-1))
  allocate(fat(0:2,0:2,0:3,0:total_sites-1), lng(0:2,0:2,0:3,0:total_sites-1))
  allocate(fatbck(0:2,0:2,0:3,0:total_sites-1), lngbck(0:2,0:2,0:3,0:total_sites-1))
  allocate(fwd(0:4*total_sites-1), bck(0:4*total_sites-1), fwd3(0:4*total_sites-1), bck3(0:4*total_sites-1))
  call set_neighbors(fwd,bck,fwd3,bck3); call make_data(src,fat,lng)
  print '(a,i0,a)', 'Number of sites = ', ldim, '^4'
  print '(a,i0,a,i0,a)', 'Executing ', iterations_default, ' iterations with ', warmups, ' warmups'
  if (workgroup_size /= 0) print '(a,i0)', 'Threads per group = ', workgroup_size
!$omp target data map(to:src,fat,lng,fwd,bck,fwd3,bck3) map(from:dst,fatbck,lngbck)
!$omp target teams distribute parallel do thread_limit(1) private(direction,row,col,neighbor)
  do site = 0, even_sites-1
    do direction = 0, 3
      neighbor = bck(4*site+direction)
      do row = 0, 2
        do col = 0, 2
          fatbck(row,col,direction,site) = conjg(fat(col,row,direction,neighbor))
        end do
      end do
      neighbor = bck3(4*site+direction)
      do row = 0, 2
        do col = 0, 2
          lngbck(row,col,direction,site) = conjg(lng(col,row,direction,neighbor))
        end do
      end do
    end do
  end do
!$omp end target teams distribute parallel do
  print '(a)', 'Running dslash loop'
  print '(a,i0)', 'Setting number of work items to ', even_sites
  print '(a,i0)', 'Setting workgroup size to ', workgroup_size
  start_time = omp_get_wtime()
  do iteration = 0, iterations_default+warmups-1
    if (iteration == warmups) start_time = omp_get_wtime()
!$omp target teams distribute parallel do thread_limit(workgroup_size) private(temp,direction,row,col,neighbor)
    do site = 0, even_sites-1
      do row = 0, 2
        dst(row,site) = cmplx(0.0_real64,0.0_real64,real64)
      end do
      do direction = 0, 3
        neighbor = fwd(4*site+direction)
        do row = 0, 2
          do col = 0, 2
            dst(row,site) = dst(row,site) + fat(row,col,direction,site)*src(col,neighbor)
          end do
        end do
      end do
      do row = 0, 2
        temp(row) = cmplx(0.0_real64,0.0_real64,real64)
      end do
      do direction = 0, 3
        neighbor = fwd3(4*site+direction)
        do row = 0, 2
          do col = 0, 2
            temp(row) = temp(row) + lng(row,col,direction,site)*src(col,neighbor)
          end do
        end do
      end do
      dst(:,site) = dst(:,site) + temp
      do row = 0, 2
        temp(row) = cmplx(0.0_real64,0.0_real64,real64)
      end do
      do direction = 0, 3
        neighbor = bck(4*site+direction)
        do row = 0, 2
          do col = 0, 2
            temp(row) = temp(row) + fatbck(row,col,direction,site)*src(col,neighbor)
          end do
        end do
      end do
      dst(:,site) = dst(:,site) - temp
      do row = 0, 2
        temp(row) = cmplx(0.0_real64,0.0_real64,real64)
      end do
      do direction = 0, 3
        neighbor = bck3(4*site+direction)
        do row = 0, 2
          do col = 0, 2
            temp(row) = temp(row) + lngbck(row,col,direction,site)*src(col,neighbor)
          end do
        end do
      end do
      dst(:,site) = dst(:,site) - temp
    end do
!$omp end target teams distribute parallel do
  end do
  total_time = omp_get_wtime() - start_time
!$omp end target data
  print '(a,f0.6,a)', 'Total execution time = ', total_time, ' secs'
  print '(a)', 'Validating the result'
  call dslash_field(src,chkdst,fat,lng,fatbck,lngbck,fwd,bck,fwd3,bck3)
  do site = 0, even_sites-1
    do i = 0, 2
      if (abs(real(dst(i,site)-chkdst(i,site),real64)) >= epsilon .or. &
          abs(aimag(dst(i,site)-chkdst(i,site))) >= epsilon) error stop 'dslash validation failed'
    end do
  end do
  tflop = real(iterations_default,real64)*real(even_sites,real64)*1182.0_real64
  print '(a,f0.6)', 'Total GFLOP/s = ', tflop/total_time/1.0e9_real64
  memory_usage = real(even_sites,real64)*(144.0_real64*4.0_real64*4.0_real64 + 48.0_real64*16.0_real64 + 8.0_real64*16.0_real64 + 48.0_real64)
  print '(a,f0.6)', 'Total GByte/s (GPU memory) = ', real(iterations_default,real64)*memory_usage/total_time/1.0e9_real64
  memory_allocated = real(total_sites,real64)*(144.0_real64*4.0_real64*4.0_real64 + 48.0_real64*2.0_real64 + 8.0_real64*4.0_real64*4.0_real64)
  print '(a,f0.6)', 'Total allocation for matrices = ', memory_allocated/1048576.0_real64
  deallocate(src,dst,chkdst,fat,lng,fatbck,lngbck,fwd,bck,fwd3,bck3)
end program dslash
