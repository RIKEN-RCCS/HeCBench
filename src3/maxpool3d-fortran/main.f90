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

program maxpool3d
  use iso_fortran_env, only: real32
  use omp_lib
  use c_rng
  implicit none
  integer :: i_img_width, i_img_height, i_img_count, repeat
  integer :: hstride, vstride, o_img_width, o_img_height, size_image, size_output
  integer :: z, y, x, r, c, n, xidx, yidx, idxintmp, idxin, status, i
  character(len=64) :: arg
  real(real32), allocatable :: h_image(:), h_output(:), d_output(:)
  real(real32) :: maxval
  real(8) :: start_time, elapsed

  if (command_argument_count() /= 4) then
    print '(a)', 'Usage: ./main <image width> <image height> <image count> <repeat>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*) i_img_width
  call get_command_argument(2,arg); read(arg,*) i_img_height
  call get_command_argument(3,arg); read(arg,*) i_img_count
  call get_command_argument(4,arg); read(arg,*) repeat

  hstride = 2; vstride = 2
  o_img_width = i_img_width / hstride
  o_img_height = i_img_height / vstride
  print '(a,i0,a,i0)', 'input image width ', i_img_width, ' Hstride ', hstride
  print '(a,i0,a,i0)', 'input image height ', i_img_height, ' Vstride ', vstride
  print '(a,i0)', 'output image width ', o_img_width
  print '(a,i0)', 'output image height ', o_img_height

  size_image = i_img_width * i_img_height
  size_output = o_img_width * o_img_height
  allocate(h_image(0:size_image*i_img_count-1), h_output(0:size_output*i_img_count-1), d_output(0:size_output*i_img_count-1))

  call srand(2)
  do z = 0, i_img_count-1
    do i = 0, size_image-1
      h_image(z*size_image+i) = real(mod(rand(), 256), real32) / 255.0_real32
    end do
  end do

!$omp target data map(to:h_image(0:size_image*i_img_count-1)) map(from:d_output(0:size_output*i_img_count-1))
  start_time = omp_get_wtime()
  do n = 1, repeat
!$omp target teams distribute parallel do collapse(3) thread_limit(256) private(xidx,yidx,idxintmp,idxin,r,c,maxval)
    do z = 0, i_img_count-1
      do y = 0, o_img_height-1
        do x = 0, o_img_width-1
          xidx = hstride*x
          yidx = vstride*y
          maxval = 0.0_real32
          do r = 0, vstride-1
            idxintmp = ((z*i_img_height + yidx + r) * i_img_width) + xidx
            do c = 0, hstride-1
              idxin = idxintmp + c
              maxval = max(maxval, h_image(idxin))
            end do
          end do
          d_output(((z*o_img_height + y)*o_img_width)+x) = maxval
        end do
      end do
    end do
!$omp end target teams distribute parallel do
  end do
  elapsed = omp_get_wtime() - start_time
  print '(a,f0.6,a)', 'Average kernel execution time: ', elapsed / repeat, ' (s)'
!$omp end target data

  do z = 0, i_img_count-1
    do y = 0, o_img_height-1
      do x = 0, o_img_width-1
        xidx = hstride*x
        yidx = vstride*y
        maxval = 0.0_real32
        do r = 0, vstride-1
          idxintmp = ((z*i_img_height + yidx + r) * i_img_width) + xidx
          do c = 0, hstride-1
            idxin = idxintmp + c
            maxval = max(maxval, h_image(idxin))
          end do
        end do
        h_output(((z*o_img_height + y)*o_img_width)+x) = maxval
      end do
    end do
  end do
  status = merge(0, 1, all(h_output == d_output))
  print '(a)', merge('PASS', 'FAIL', status == 0)
  if (status /= 0) stop status
end program maxpool3d
