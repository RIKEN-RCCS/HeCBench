program asmooth
  use, intrinsic :: iso_c_binding, only : c_int
  use, intrinsic :: iso_fortran_env, only : int64, real32, real64
  implicit none

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

  integer :: lx, ly, size, threshold, max_rad, repeat
  integer :: iteration, x, y, i, j, s, q, ksum
  integer(int64) :: start_count, end_count, count_rate
  integer(c_int) :: random_value
  real(real32) :: sum_value
  real(real64) :: elapsed, time
  real(real32), allocatable :: img(:), h_img(:), norm(:), h_norm(:), out(:), h_out(:)
  integer, allocatable :: box(:), h_box(:)
  character(len=128) :: argument, executable

  if (command_argument_count() /= 4) then
    call get_command_argument(0, executable)
    write(*, '(A)') './' // trim(executable) // &
      ' <image dimension> <threshold> <max box size> <iterations>'
    error stop 1
  end if

  call get_command_argument(1, argument)
  read(argument, *) lx
  ly = lx
  size = lx * ly
  call get_command_argument(2, argument)
  read(argument, *) threshold
  call get_command_argument(3, argument)
  read(argument, *) max_rad
  call get_command_argument(4, argument)
  read(argument, *) repeat

  allocate(img(0:size - 1), h_img(0:size - 1), norm(0:size - 1), h_norm(0:size - 1), out(0:size - 1), h_out(0:size - 1))
  allocate(box(0:size - 1), h_box(0:size - 1))

  call c_srand(123_c_int)
  do i = 0, size - 1
    random_value = c_rand()
    img(i) = real(modulo(random_value, 256_c_int), real32)
    norm(i) = 0.0_real32
    box(i) = 0
    out(i) = 0.0_real32
  end do

  h_img = img
  ! reference.h accumulates into h_norm but its C++ allocation is not explicitly
  ! initialized.  The test assumes the intended all-zero initial accumulator.
  h_norm = 0.0_real32
  time = 0.0_real64

  !$omp target data map(alloc: img(0:size-1), norm(0:size-1), box(0:size-1)) &
  !$omp& map(to: out(0:size-1))
  do iteration = 0, repeat - 1
    ! With a GPU map(alloc:), these are host-only assignments: the following
    ! target updates transfer the same initial state as the C++ reference.
    ! They also make a host-fallback execution emulate the separate device copy.
    img = h_img
    norm = 0.0_real32
    !$omp target update to(img(0:size-1))
    !$omp target update to(norm(0:size-1))

    call system_clock(start_count, count_rate)

    !$omp target teams distribute parallel do collapse(2) thread_limit(256) &
    !$omp& private(sum_value, s, q, ksum, i, j)
    do x = 0, lx - 1
      do y = 0, ly - 1
        sum_value = 0.0_real32
        s = 1
        q = 1
        ksum = 0
        do while (sum_value < real(threshold, real32) .and. q < max_rad)
          s = q
          sum_value = 0.0_real32
          ksum = 0
          do i = -s, s
            do j = -s, s
              if (x - s >= 0 .and. x + s < lx .and. y - s >= 0 .and. y + s < ly) then
                sum_value = sum_value + img((x + i) * ly + y + j)
                ksum = ksum + 1
              end if
            end do
          end do
          q = q + 1
        end do

        box(x * ly + y) = s
        do i = -s, s
          do j = -s, s
            if (x - s >= 0 .and. x + s < lx .and. y - s >= 0 .and. y + s < ly) then
              if (ksum /= 0) then
                !$omp atomic update
                norm((x + i) * ly + y + j) = norm((x + i) * ly + y + j) + 1.0_real32 / real(ksum, real32)
              end if
            end if
          end do
        end do
      end do
    end do
    !$omp end target teams distribute parallel do

    !$omp target teams distribute parallel do collapse(2) thread_limit(256)
    do x = 0, lx - 1
      do y = 0, ly - 1
        if (norm(x * ly + y) /= 0.0_real32) img(x * ly + y) = img(x * ly + y) / norm(x * ly + y)
      end do
    end do
    !$omp end target teams distribute parallel do

    !$omp target teams distribute parallel do collapse(2) thread_limit(256) &
    !$omp& private(s, sum_value, ksum, i, j)
    do x = 0, lx - 1
      do y = 0, ly - 1
        s = box(x * ly + y)
        sum_value = 0.0_real32
        ksum = 0
        do i = -s, s
          do j = -s, s
            if (x - s >= 0 .and. x + s < lx .and. y - s >= 0 .and. y + s < ly) then
              sum_value = sum_value + img((x + i) * ly + y + j)
              ksum = ksum + 1
            end if
          end do
        end do
        if (ksum /= 0) out(x * ly + y) = sum_value / real(ksum, real32)
      end do
    end do
    !$omp end target teams distribute parallel do

    call system_clock(end_count)
    elapsed = real(end_count - start_count, real64) / real(count_rate, real64)
    time = time + elapsed
  end do

  write(*, '(A,F0.6,A)') 'Average filtering time ', time / real(repeat, real64), ' (s)'
  !$omp target update from(out(0:size-1))
  !$omp target update from(box(0:size-1))
  !$omp target update from(norm(0:size-1))
  !$omp end target data

  ! The C++ map(alloc:) leaves host img at its original values.  Use h_img
  ! here so host-fallback testing has that same reference input.
  call reference(lx, ly, threshold, max_rad, h_img, h_box, h_norm, h_out)
  call verify(size, max_rad, norm, h_norm, out, h_out, box, h_box)

contains

  subroutine reference(lx, ly, threshold, max_rad, img, box, norm, out)
    integer, intent(in) :: lx, ly, threshold, max_rad
    real(real32), intent(inout) :: img(0:), norm(0:), out(0:)
    integer, intent(out) :: box(0:)
    integer :: x, y, i, j, s, q, ksum
    real(real32) :: sum_value

    do x = 0, lx - 1
      do y = 0, ly - 1
        sum_value = 0.0_real32
        s = 1; q = 1; ksum = 0
        do while (sum_value < real(threshold, real32) .and. q < max_rad)
          s = q; sum_value = 0.0_real32; ksum = 0
          do i = -s, s
            do j = -s, s
              if (x - s >= 0 .and. x + s < lx .and. y - s >= 0 .and. y + s < ly) then
                sum_value = sum_value + img((x + i) * ly + y + j)
                ksum = ksum + 1
              end if
            end do
          end do
          q = q + 1
        end do
        box(x * ly + y) = s
        do i = -s, s
          do j = -s, s
            if (x - s >= 0 .and. x + s < lx .and. y - s >= 0 .and. y + s < ly) then
              if (ksum /= 0) norm((x + i) * ly + y + j) = &
                norm((x + i) * ly + y + j) + 1.0_real32 / real(ksum, real32)
            end if
          end do
        end do
      end do
    end do
    do x = 0, lx - 1
      do y = 0, ly - 1
        if (norm(x * ly + y) /= 0.0_real32) img(x * ly + y) = img(x * ly + y) / norm(x * ly + y)
      end do
    end do
    do x = 0, lx - 1
      do y = 0, ly - 1
        s = box(x * ly + y); sum_value = 0.0_real32; ksum = 0
        do i = -s, s
          do j = -s, s
            if (x - s >= 0 .and. x + s < lx .and. y - s >= 0 .and. y + s < ly) then
              sum_value = sum_value + img((x + i) * ly + y + j)
              ksum = ksum + 1
            end if
          end do
        end do
        if (ksum /= 0) out(x * ly + y) = sum_value / real(ksum, real32)
      end do
    end do
  end subroutine reference

  subroutine verify(size, max_rad, norm, h_norm, out, h_out, box, h_box)
    integer, intent(in) :: size, max_rad
    real(real32), intent(in) :: norm(0:), h_norm(0:), out(0:), h_out(0:)
    integer, intent(in) :: box(0:), h_box(0:)
    integer :: i, j
    integer, allocatable :: count(:)
    logical :: ok

    ok = .true.
    allocate(count(0:max_rad - 1)); count = 0
    do i = 0, size - 1
      if (abs(norm(i) - h_norm(i)) > 1.0e-3_real32) then
        write(*, '(A,I0,1X,F0.6,1X,F0.6)') 'norm: ', i, norm(i), h_norm(i); ok = .false.; exit
      end if
      if (abs(out(i) - h_out(i)) > 1.0e-3_real32) then
        write(*, '(A,I0,1X,F0.6,1X,F0.6)') 'out: ', i, out(i), h_out(i); ok = .false.; exit
      end if
      if (box(i) /= h_box(i)) then
        write(*, '(A,I0,1X,I0,1X,I0)') 'box: ', i, box(i), h_box(i); ok = .false.; exit
      else
        do j = 0, max_rad - 1
          if (box(i) == j) then
            count(j) = count(j) + 1
            exit
          end if
        end do
      end if
    end do
    if (ok) then
      write(*, '(A)') 'PASS'
      write(*, '(A)') 'Distribution of box sizes:'
      do j = 1, max_rad - 1
        write(*, '(A,I0,A,F0.6)') 'size=', j, ': ', real(count(j), real32) / real(size, real32)
      end do
    else
      write(*, '(A)') 'FAIL'
    end if
  end subroutine verify

end program asmooth
