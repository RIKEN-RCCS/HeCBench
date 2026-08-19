module background_subtract_support
  use iso_fortran_env, only: int8, int32, int64, real32
  implicit none

  integer(int64), parameter :: mt_mask = 4294967295_int64
  integer(int64), parameter :: mt_upper = 2147483648_int64
  integer(int64), parameter :: mt_lower = 2147483647_int64
  integer(int64), parameter :: mt_matrix_a = 2567483615_int64
  integer(int64) :: mt(0:623)
  integer :: mt_index = 624
!$omp declare target (u8, as_u8)

contains

  subroutine mt_seed(seed)
    integer(int32), intent(in) :: seed
    integer :: i

    mt(0) = iand(int(seed, int64), mt_mask)
    do i = 1, 623
      mt(i) = iand(1812433253_int64 * ieor(mt(i - 1), ishft(mt(i - 1), -30)) + &
                       int(i, int64), mt_mask)
    end do
    mt_index = 624
  end subroutine mt_seed

  subroutine mt_twist()
    integer :: i
    integer(int64) :: x

    do i = 0, 623
      x = ior(iand(mt(i), mt_upper), iand(mt(modulo(i + 1, 624)), mt_lower))
      mt(i) = ieor(mt(modulo(i + 397, 624)), ishft(x, -1))
      if (btest(x, 0)) mt(i) = ieor(mt(i), mt_matrix_a)
      mt(i) = iand(mt(i), mt_mask)
    end do
    mt_index = 0
  end subroutine mt_twist

  function mt_uniform_0_255() result(value)
    integer(int32) :: value
    integer(int64) :: x

    if (mt_index >= 624) call mt_twist()
    x = mt(mt_index)
    mt_index = mt_index + 1
    x = ieor(x, ishft(x, -11))
    x = ieor(x, iand(ishft(x, 7), 2636928640_int64))
    x = ieor(x, iand(ishft(x, 15), 4022730752_int64))
    x = ieor(x, ishft(x, -18))
    x = iand(x, mt_mask)
    value = int(x / 16777216_int64, int32)
  end function mt_uniform_0_255

  pure integer(int32) function u8(value)
    integer(int8), intent(in) :: value

    u8 = int(value, int32)
    if (u8 < 0_int32) u8 = u8 + 256_int32
  end function u8

  pure integer(int8) function as_u8(value)
    integer(int32), intent(in) :: value
    integer(int32) :: wrapped

    wrapped = modulo(value, 256_int32)
    as_u8 = transfer(wrapped, as_u8)
  end function as_u8

  subroutine merge_reference(img_size, img, img1, img2, tn, bn)
    integer(int32), intent(in) :: img_size
    integer(int8), intent(in) :: img(0:), img1(0:), img2(0:)
    integer(int8), intent(inout) :: tn(0:), bn(0:)
    integer(int32) :: i
    real(real32) :: threshold

    do i = 0, img_size - 1
      if (abs(u8(img(i)) - u8(img1(i))) <= u8(tn(i)) .and. &
          abs(u8(img(i)) - u8(img2(i))) <= u8(tn(i))) then
        bn(i) = as_u8(int(0.92_real32 * real(u8(bn(i)), real32) + &
                           0.08_real32 * real(u8(img(i)), real32), int32))
        threshold = 0.92_real32 * real(u8(tn(i)), real32) + &
                    0.24_real32 * real(u8(img(i)) - u8(bn(i)), real32)
        tn(i) = as_u8(max(int(threshold, int32), 20_int32))
      end if
    end do
  end subroutine merge_reference
end module background_subtract_support

program background_subtract
  use iso_fortran_env, only: int8, int32, real32, real64
  use omp_lib, only: omp_get_wtime
  use background_subtract_support
  implicit none

  integer(int32) :: width, height, merged, repeat, img_size
  integer(int32) :: i, j, image_value, max_error
  integer(int32) :: current_image, image1, image2, temporary_image
  integer(int8), allocatable :: images(:, :), bn(:), bn_ref(:), mp(:), tn(:), tn_ref(:)
  real(real64) :: start_time, end_time, elapsed_time
  real(real32) :: kernel_time, threshold
  character(len=128) :: argument

  if (command_argument_count() /= 4) then
    print '(a)', 'Usage: ./main <image width> <image height> <merge> <repeat>'
    stop 1
  end if
  call get_command_argument(1, argument); read(argument, *) width
  call get_command_argument(2, argument); read(argument, *) height
  call get_command_argument(3, argument); read(argument, *) merged
  call get_command_argument(4, argument); read(argument, *) repeat

  img_size = width * height
  allocate(images(0:img_size - 1, 0:2), bn(0:img_size - 1), bn_ref(0:img_size - 1))
  allocate(mp(0:img_size - 1), tn(0:img_size - 1), tn_ref(0:img_size - 1))

  call mt_seed(123_int32)
  do j = 0, img_size - 1
    image_value = mt_uniform_0_255()
    bn(j) = as_u8(image_value)
    bn_ref(j) = bn(j)
    tn(j) = as_u8(128_int32)
    tn_ref(j) = tn(j)
  end do

  current_image = 0_int32
  image1 = 1_int32
  image2 = 2_int32
  elapsed_time = 0.0_real64

!$omp target data map(tofrom: bn(0:img_size-1), tn(0:img_size-1)) &
!$omp& map(alloc: mp(0:img_size-1), images(0:img_size-1,0:2))
  do i = 0, repeat - 1
    do j = 0, img_size - 1
      images(j, current_image) = as_u8(mt_uniform_0_255())
    end do
!$omp target update to(images(0:img_size-1,current_image))

    temporary_image = image2
    image2 = image1
    image1 = current_image
    current_image = temporary_image

    if (i >= 2) then
      start_time = omp_get_wtime()
      if (merged /= 0_int32) then
!$omp target teams distribute parallel do thread_limit(256) private(threshold)
        do j = 0, img_size - 1
          if (abs(u8(images(j,current_image)) - u8(images(j,image1))) <= u8(tn(j)) .and. &
              abs(u8(images(j,current_image)) - u8(images(j,image2))) <= u8(tn(j))) then
            bn(j) = as_u8(int(0.92_real32 * real(u8(bn(j)), real32) + &
                               0.08_real32 * real(u8(images(j,current_image)), real32), int32))
            threshold = 0.92_real32 * real(u8(tn(j)), real32) + &
                        0.24_real32 * real(u8(images(j,current_image)) - u8(bn(j)), real32)
            tn(j) = as_u8(max(int(threshold, int32), 20_int32))
          end if
        end do
!$omp end target teams distribute parallel do
      else
!$omp target teams distribute parallel do thread_limit(256)
        do j = 0, img_size - 1
          if (abs(u8(images(j,current_image)) - u8(images(j,image1))) > u8(tn(j)) .or. &
              abs(u8(images(j,current_image)) - u8(images(j,image2))) > u8(tn(j))) then
            mp(j) = as_u8(255_int32)
          else
            mp(j) = as_u8(0_int32)
          end if
        end do
!$omp end target teams distribute parallel do
!$omp target teams distribute parallel do thread_limit(256)
        do j = 0, img_size - 1
          if (u8(mp(j)) == 0_int32) then
            bn(j) = as_u8(int(0.92_real32 * real(u8(bn(j)), real32) + &
                               0.08_real32 * real(u8(images(j,current_image)), real32), int32))
          end if
        end do
!$omp end target teams distribute parallel do
!$omp target teams distribute parallel do thread_limit(256) private(threshold)
        do j = 0, img_size - 1
          if (u8(mp(j)) == 0_int32) then
            threshold = 0.92_real32 * real(u8(tn(j)), real32) + &
                        0.24_real32 * real(u8(images(j,current_image)) - u8(bn(j)), real32)
            tn(j) = as_u8(max(int(threshold, int32), 20_int32))
          end if
        end do
!$omp end target teams distribute parallel do
      end if
      end_time = omp_get_wtime()
      elapsed_time = elapsed_time + end_time - start_time
      call merge_reference(img_size, images(:,current_image), images(:,image1), images(:,image2), tn_ref, bn_ref)
    end if
  end do
!$omp end target data

  if (repeat <= 2_int32) then
    kernel_time = 0.0_real32
  else
    kernel_time = real(elapsed_time * 1.0e6_real64 / real(repeat - 2, real64), real32)
  end if
  print '(a,f0.6,a)', 'Average kernel execution time: ', kernel_time, ' (us)'

  max_error = 0_int32
  do i = 0, img_size - 1
    max_error = max(max_error, abs(u8(tn(i)) - u8(tn_ref(i))))
    max_error = max(max_error, abs(u8(bn(i)) - u8(bn_ref(i))))
  end do
  print '(a,i0)', 'Max error is ', max_error
  if (max_error == 0_int32) then
    print '(a)', 'PASS'
  else
    print '(a)', 'FAIL'
  end if

  deallocate(images, bn, bn_ref, mp, tn, tn_ref)
end program background_subtract
