module aes_precision
  use, intrinsic :: iso_fortran_env, only : int8, int32, int64, real32, real64
  implicit none
end module aes_precision

module aes_common
  use aes_precision
  implicit none
  !$omp declare target (byte_value, galois_multiplication)
contains
  pure integer(int32) function byte_value(value)
    integer(int32), intent(in) :: value
    byte_value = iand(value, int(z'000000FF', int32))
  end function byte_value

  pure integer(int32) function galois_multiplication(a_in, b_in)
    integer(int32), intent(in) :: a_in, b_in
    integer(int32) :: a, b, p, high_bit
    integer :: i

    a = byte_value(a_in)
    b = byte_value(b_in)
    p = 0_int32
    do i = 0, 7
      if (iand(b, 1_int32) == 1_int32) p = ieor(p, a)
      high_bit = iand(a, int(z'80', int32))
      a = byte_value(shiftl(a, 1))
      if (high_bit == int(z'80', int32)) a = ieor(a, int(z'1B', int32))
      b = shiftr(b, 1)
    end do
    galois_multiplication = byte_value(p)
  end function galois_multiplication

  pure integer(int32) function galois_power(base, exponent)
    integer(int32), intent(in) :: base
    integer, intent(in) :: exponent
    integer(int32) :: result, factor
    integer :: power

    result = 1_int32
    factor = byte_value(base)
    power = exponent
    do while (power > 0)
      if (iand(power, 1) /= 0) result = galois_multiplication(result, factor)
      factor = galois_multiplication(factor, factor)
      power = shiftr(power, 1)
    end do
    galois_power = result
  end function galois_power

  pure integer(int32) function aes_sbox_value(value)
    integer(int32), intent(in) :: value
    integer(int32) :: inverse, transformed

    if (byte_value(value) == 0_int32) then
      inverse = 0_int32
    else
      inverse = galois_power(value, 254)
    end if
    transformed = inverse
    transformed = ieor(transformed, ishftc(inverse, 1, 8))
    transformed = ieor(transformed, ishftc(inverse, 2, 8))
    transformed = ieor(transformed, ishftc(inverse, 3, 8))
    transformed = ieor(transformed, ishftc(inverse, 4, 8))
    aes_sbox_value = byte_value(ieor(transformed, int(z'63', int32)))
  end function aes_sbox_value

  subroutine initialize_sboxes(sbox, rsbox)
    integer(int32), intent(out) :: sbox(0:255), rsbox(0:255)
    integer :: i
    do i = 0, 255
      sbox(i) = aes_sbox_value(int(i, int32))
    end do
    do i = 0, 255
      rsbox(sbox(i)) = int(i, int32)
    end do
  end subroutine initialize_sboxes

  subroutine rotate_word(word)
    integer(int32), intent(inout) :: word(0:3)
    integer(int32) :: temporary
    temporary = word(0)
    word(0:2) = word(1:3)
    word(3) = temporary
  end subroutine rotate_word

  subroutine key_core(word, iteration, sbox)
    integer(int32), intent(inout) :: word(0:3)
    integer, intent(in) :: iteration
    integer(int32), intent(in) :: sbox(0:255)
    integer(int32) :: rcon
    integer :: i

    call rotate_word(word)
    do i = 0, 3
      word(i) = sbox(word(i))
    end do
    rcon = 1_int32
    do i = 1, iteration - 1
      rcon = galois_multiplication(rcon, 2_int32)
    end do
    word(0) = ieor(word(0), rcon)
  end subroutine key_core

  subroutine key_expansion(key, expanded_key, sbox)
    integer(int32), intent(in) :: key(0:15), sbox(0:255)
    integer(int32), intent(out) :: expanded_key(0:175)
    integer(int32) :: temporary(0:3)
    integer :: current_size, rcon_iteration, i

    expanded_key(0:15) = key(0:15)
    current_size = 16
    rcon_iteration = 1
    do while (current_size < 176)
      temporary = expanded_key(current_size - 4:current_size - 1)
      if (mod(current_size, 16) == 0) then
        call key_core(temporary, rcon_iteration, sbox)
        rcon_iteration = rcon_iteration + 1
      end if
      do i = 0, 3
        expanded_key(current_size) = ieor(expanded_key(current_size - 16), temporary(i))
        current_size = current_size + 1
      end do
    end do
  end subroutine key_expansion

  subroutine create_round_key(expanded_key, round_key)
    integer(int32), intent(in) :: expanded_key(0:175)
    integer(int32), intent(out) :: round_key(0:175)
    integer :: round, i, j
    do round = 0, 10
      do i = 0, 3
        do j = 0, 3
          round_key(round * 16 + i + j * 4) = expanded_key(round * 16 + i * 4 + j)
        end do
      end do
    end do
  end subroutine create_round_key
end module aes_common

module aes_bitmap
  use aes_precision
  implicit none
contains
  pure integer(int32) function unsigned_byte(value)
    integer(int8), intent(in) :: value
    unsigned_byte = iand(int(value, int32), int(z'000000FF', int32))
  end function unsigned_byte

  pure integer(int32) function little_endian_16(raw, offset)
    integer(int8), intent(in) :: raw(0:)
    integer, intent(in) :: offset
    little_endian_16 = unsigned_byte(raw(offset)) + shiftl(unsigned_byte(raw(offset + 1)), 8)
  end function little_endian_16

  pure integer(int32) function little_endian_32(raw, offset)
    integer(int8), intent(in) :: raw(0:)
    integer, intent(in) :: offset
    little_endian_32 = unsigned_byte(raw(offset)) + shiftl(unsigned_byte(raw(offset + 1)), 8) + &
      shiftl(unsigned_byte(raw(offset + 2)), 16) + shiftl(unsigned_byte(raw(offset + 3)), 24)
  end function little_endian_32

  subroutine load_bmp_gray(path, decrypt, input, width, height, status)
    character(len=*), intent(in) :: path
    logical, intent(in) :: decrypt
    integer(int32), allocatable, intent(out) :: input(:)
    integer, intent(out) :: width, height, status
    integer :: file_unit, io_status, file_size, bits_per_pixel, data_offset, bitmap_size
    integer :: x, y, index, pixel_offset, palette_offset, palette_index, padding
    integer(int8), allocatable :: raw(:)
    integer(int32) :: red, green, blue

    status = 1
    open(newunit=file_unit, file=trim(path), access='stream', form='unformatted', status='old', &
         action='read', iostat=io_status)
    if (io_status /= 0) then
      write(*, '(A)') 'Failed to load file ' // trim(path)
      return
    end if
    inquire(unit=file_unit, size=file_size)
    if (file_size < 54) then
      close(file_unit)
      return
    end if
    allocate(raw(0:file_size - 1))
    read(file_unit, iostat=io_status) raw
    close(file_unit)
    if (io_status /= 0) return
    if (little_endian_16(raw, 0) /= int(z'4D42', int32)) return

    bitmap_size = little_endian_32(raw, 2)
    data_offset = little_endian_32(raw, 10)
    width = little_endian_32(raw, 18)
    height = little_endian_32(raw, 22)
    bits_per_pixel = little_endian_16(raw, 28)
    if (little_endian_32(raw, 30) /= 0 .or. (bits_per_pixel /= 8 .and. bits_per_pixel /= 24)) return
    if (width <= 0 .or. height <= 0 .or. data_offset < 54 .or. data_offset >= file_size) return
    if (width * height * (bits_per_pixel / 8) /= bitmap_size - data_offset) then
      write(*, '(A)') 'This is not a valid bitmap file.'
      return
    end if
    allocate(input(0:width * height - 1))
    index = data_offset
    padding = mod(4 - mod(3 * width, 4), 4)
    do y = 0, height - 1
      do x = 0, width - 1
        pixel_offset = y * width + x
        if (bits_per_pixel == 8) then
          palette_index = unsigned_byte(raw(index))
          index = index + 1
          palette_offset = 54 + 4 * palette_index
          if (palette_offset + 3 >= data_offset) return
          red = unsigned_byte(raw(palette_offset))
          green = unsigned_byte(raw(palette_offset + 1))
          blue = unsigned_byte(raw(palette_offset + 2))
        else
          blue = unsigned_byte(raw(index))
          green = unsigned_byte(raw(index + 1))
          red = unsigned_byte(raw(index + 2))
          index = index + 3
        end if
        if (decrypt) then
          input(pixel_offset) = red
        else
          input(pixel_offset) = int(real(red, real32) * 0.3_real32 + real(green, real32) * 0.59_real32 + &
                                   real(blue, real32) * 0.11_real32, int32)
        end if
      end do
      if (bits_per_pixel == 24) index = index + padding
    end do
    status = 0
  end subroutine load_bmp_gray
end module aes_bitmap

module aes_reference_mod
  use aes_precision
  use aes_common
  implicit none
  !$omp declare target (add_round_key, sub_bytes, shift_rows_reference, mix_columns_reference)
contains
  subroutine add_round_key(state, key)
    integer(int32), intent(inout) :: state(0:15)
    integer(int32), intent(in) :: key(0:15)
    integer :: i
    do i = 0, 15
      state(i) = ieor(state(i), key(i))
    end do
  end subroutine add_round_key

  subroutine sub_bytes(state, inverse, sbox, rsbox)
    integer(int32), intent(inout) :: state(0:15)
    logical, intent(in) :: inverse
    integer(int32), intent(in) :: sbox(0:255), rsbox(0:255)
    integer :: i
    do i = 0, 15
      if (inverse) then
        state(i) = rsbox(state(i))
      else
        state(i) = sbox(state(i))
      end if
    end do
  end subroutine sub_bytes

  subroutine shift_rows_reference(state, inverse)
    integer(int32), intent(inout) :: state(0:15)
    logical, intent(in) :: inverse
    integer(int32) :: row(0:3), temporary
    integer :: i, j
    do i = 0, 3
      row = state(i * 4:i * 4 + 3)
      do j = 1, i
        if (inverse) then
          temporary = row(3); row(1:3) = row(0:2); row(0) = temporary
        else
          temporary = row(0); row(0:2) = row(1:3); row(3) = temporary
        end if
      end do
      state(i * 4:i * 4 + 3) = row
    end do
  end subroutine shift_rows_reference

  subroutine mix_columns_reference(state, inverse)
    integer(int32), intent(inout) :: state(0:15)
    logical, intent(in) :: inverse
    integer(int32) :: column(0:3), copy_column(0:3)
    integer :: i, j
    do i = 0, 3
      do j = 0, 3
        column(j) = state(j * 4 + i)
      end do
      copy_column = column
      if (inverse) then
        column(0) = ieor(ieor(galois_multiplication(copy_column(0), 14_int32), &
                              galois_multiplication(copy_column(3), 9_int32)), &
                         ieor(galois_multiplication(copy_column(2), 13_int32), &
                              galois_multiplication(copy_column(1), 11_int32)))
        column(1) = ieor(ieor(galois_multiplication(copy_column(1), 14_int32), &
                              galois_multiplication(copy_column(0), 9_int32)), &
                         ieor(galois_multiplication(copy_column(3), 13_int32), &
                              galois_multiplication(copy_column(2), 11_int32)))
        column(2) = ieor(ieor(galois_multiplication(copy_column(2), 14_int32), &
                              galois_multiplication(copy_column(1), 9_int32)), &
                         ieor(galois_multiplication(copy_column(0), 13_int32), &
                              galois_multiplication(copy_column(3), 11_int32)))
        column(3) = ieor(ieor(galois_multiplication(copy_column(3), 14_int32), &
                              galois_multiplication(copy_column(2), 9_int32)), &
                         ieor(galois_multiplication(copy_column(1), 13_int32), &
                              galois_multiplication(copy_column(0), 11_int32)))
      else
        column(0) = ieor(ieor(galois_multiplication(copy_column(0), 2_int32), &
                              galois_multiplication(copy_column(3), 1_int32)), &
                         ieor(galois_multiplication(copy_column(2), 1_int32), &
                              galois_multiplication(copy_column(1), 3_int32)))
        column(1) = ieor(ieor(galois_multiplication(copy_column(1), 2_int32), &
                              galois_multiplication(copy_column(0), 1_int32)), &
                         ieor(galois_multiplication(copy_column(3), 1_int32), &
                              galois_multiplication(copy_column(2), 3_int32)))
        column(2) = ieor(ieor(galois_multiplication(copy_column(2), 2_int32), &
                              galois_multiplication(copy_column(1), 1_int32)), &
                         ieor(galois_multiplication(copy_column(0), 1_int32), &
                              galois_multiplication(copy_column(3), 3_int32)))
        column(3) = ieor(ieor(galois_multiplication(copy_column(3), 2_int32), &
                              galois_multiplication(copy_column(2), 1_int32)), &
                         ieor(galois_multiplication(copy_column(1), 1_int32), &
                              galois_multiplication(copy_column(0), 3_int32)))
      end if
      do j = 0, 3
        state(j * 4 + i) = byte_value(column(j))
      end do
    end do
  end subroutine mix_columns_reference

  subroutine aes_reference(output, input, round_key, width, height, inverse, rounds, sbox, rsbox)
    integer(int32), intent(out) :: output(0:)
    integer(int32), intent(in) :: input(0:), round_key(0:175), sbox(0:255), rsbox(0:255)
    integer, intent(in) :: width, height, rounds
    logical, intent(in) :: inverse
    integer(int32) :: state(0:15)
    integer :: block_x, block_y, i, j, index, round

    do block_y = 0, height / 4 - 1
      do block_x = 0, width / 4 - 1
        do i = 0, 3
          do j = 0, 3
            index = ((block_y * (width / 4) + block_x) * 16) + i * 4 + j
            state(i * 4 + j) = input(index)
          end do
        end do
        if (inverse) then
          call add_round_key(state, round_key(rounds * 16:rounds * 16 + 15))
          do round = rounds - 1, 1, -1
            call shift_rows_reference(state, .true.)
            call sub_bytes(state, .true., sbox, rsbox)
            call add_round_key(state, round_key(round * 16:round * 16 + 15))
            call mix_columns_reference(state, .true.)
          end do
          call shift_rows_reference(state, .true.)
          call sub_bytes(state, .true., sbox, rsbox)
          call add_round_key(state, round_key(0:15))
        else
          call add_round_key(state, round_key(0:15))
          do round = 1, rounds - 1
            call sub_bytes(state, .false., sbox, rsbox)
            call shift_rows_reference(state, .false.)
            call mix_columns_reference(state, .false.)
            call add_round_key(state, round_key(round * 16:round * 16 + 15))
          end do
          call sub_bytes(state, .false., sbox, rsbox)
          call shift_rows_reference(state, .false.)
          call add_round_key(state, round_key(rounds * 16:rounds * 16 + 15))
        end if
        do i = 0, 3
          do j = 0, 3
            index = ((block_y * (width / 4) + block_x) * 16) + i * 4 + j
            output(index) = state(i * 4 + j)
          end do
        end do
      end do
    end do
  end subroutine aes_reference
end module aes_reference_mod

module aes_device
  use aes_precision
  use aes_common
  use aes_reference_mod
  use omp_lib
  implicit none
  !$omp declare target (rotate_vector, device_mix_columns)
contains
  subroutine aes_transform_gpu(output, input, round_key, sbox, rsbox, width, height, inverse, rounds)
    integer(int32), intent(inout) :: output(0:)
    integer(int32), intent(in) :: input(0:), round_key(0:175), sbox(0:255), rsbox(0:255)
    integer, intent(in) :: width, height, rounds
    logical, intent(in) :: inverse
    integer :: block, i, round
    integer(int32) :: state(0:15)
    !$omp target teams distribute parallel do thread_limit(256) private(i,round,state)
    do block = 0, width * height / 16 - 1
      state = input(block * 16:block * 16 + 15)
      if (inverse) then
        call add_round_key(state, round_key(rounds * 16:rounds * 16 + 15))
        do round = rounds - 1, 1, -1
          call shift_rows_reference(state, .true.)
          call sub_bytes(state, .true., sbox, rsbox)
          call add_round_key(state, round_key(round * 16:round * 16 + 15))
          call mix_columns_reference(state, .true.)
        end do
        call shift_rows_reference(state, .true.)
        call sub_bytes(state, .true., sbox, rsbox)
        call add_round_key(state, round_key(0:15))
      else
        call add_round_key(state, round_key(0:15))
        do round = 1, rounds - 1
          call sub_bytes(state, .false., sbox, rsbox)
          call shift_rows_reference(state, .false.)
          call mix_columns_reference(state, .false.)
          call add_round_key(state, round_key(round * 16:round * 16 + 15))
        end do
        call sub_bytes(state, .false., sbox, rsbox)
        call shift_rows_reference(state, .false.)
        call add_round_key(state, round_key(rounds * 16:rounds * 16 + 15))
      end if
      output(block * 16:block * 16 + 15) = state
    end do
    !$omp end target teams distribute parallel do
  end subroutine aes_transform_gpu
  subroutine rotate_vector(vector, count, inverse)
    integer(int32), intent(inout) :: vector(0:3)
    integer, intent(in) :: count
    logical, intent(in) :: inverse
    integer(int32) :: temporary
    integer :: i
    do i = 1, count
      if (inverse) then
        temporary = vector(3); vector(1:3) = vector(0:2); vector(0) = temporary
      else
        temporary = vector(0); vector(0:2) = vector(1:3); vector(3) = temporary
      end if
    end do
  end subroutine rotate_vector

  subroutine device_mix_columns(block, coefficients, local_index, result)
    integer(int32), intent(in) :: block(0:3,0:3), coefficients(0:3)
    integer, intent(in) :: local_index
    integer(int32), intent(out) :: result(0:3)
    integer :: component, k, coefficient_index
    integer(int32) :: accumulator
    do component = 0, 3
      accumulator = 0_int32
      do k = 0, 3
        coefficient_index = mod(k + 4 - local_index, 4)
        accumulator = ieor(accumulator, galois_multiplication(block(component, k), coefficients(coefficient_index)))
      end do
      result(component) = byte_value(accumulator)
    end do
  end subroutine device_mix_columns

  subroutine aes_encrypt_gpu(output, input, round_key, sbox, width, height, rounds)
    integer(int32), intent(inout) :: output(0:)
    integer(int32), intent(in) :: input(0:), round_key(0:175), sbox(0:255)
    integer, intent(in) :: width, height, rounds
    integer :: teams, team, tid, global_index, round
    integer(int32) :: block0(0:3,0:3), block1(0:3,0:3), coefficients(0:3), vector(0:3)

    teams = width * height / 16
    !$omp target teams distribute num_teams(teams) thread_limit(4) private(block0, block1, tid, global_index, round, coefficients, vector)
    do team = 0, teams - 1
      !$omp parallel num_threads(4) private(tid, global_index, round, coefficients, vector) shared(block0, block1)
      tid = omp_get_thread_num()
      global_index = team * 4 + tid
      coefficients = [2_int32, 3_int32, 1_int32, 1_int32]
      block0(:, tid) = input(global_index * 4:global_index * 4 + 3)
      block0(:, tid) = ieor(block0(:, tid), round_key(tid * 4:tid * 4 + 3))
      do round = 1, rounds - 1
        block0(:, tid) = [sbox(block0(0, tid)), sbox(block0(1, tid)), sbox(block0(2, tid)), sbox(block0(3, tid))]
        vector = block0(:, tid)
        call rotate_vector(vector, tid, .false.)
        block0(:, tid) = vector
        !$omp barrier
        call device_mix_columns(block0, coefficients, tid, block1(:, tid))
        !$omp barrier
        block0(:, tid) = ieor(block1(:, tid), round_key(round * 16 + tid * 4:round * 16 + tid * 4 + 3))
      end do
      block0(:, tid) = [sbox(block0(0, tid)), sbox(block0(1, tid)), sbox(block0(2, tid)), sbox(block0(3, tid))]
      vector = block0(:, tid)
      call rotate_vector(vector, tid, .false.)
      output(global_index * 4:global_index * 4 + 3) = ieor(vector, round_key(rounds * 16 + tid * 4:rounds * 16 + tid * 4 + 3))
      !$omp end parallel
    end do
    !$omp end target teams distribute
  end subroutine aes_encrypt_gpu

  subroutine aes_decrypt_gpu(output, input, round_key, rsbox, width, height, rounds)
    integer(int32), intent(inout) :: output(0:)
    integer(int32), intent(in) :: input(0:), round_key(0:175), rsbox(0:255)
    integer, intent(in) :: width, height, rounds
    integer :: teams, team, tid, global_index, round
    integer(int32) :: block0(0:3,0:3), block1(0:3,0:3), coefficients(0:3), vector(0:3)

    teams = width * height / 16
    !$omp target teams distribute num_teams(teams) thread_limit(4) private(block0, block1, tid, global_index, round, coefficients, vector)
    do team = 0, teams - 1
      !$omp parallel num_threads(4) private(tid, global_index, round, coefficients, vector) shared(block0, block1)
      tid = omp_get_thread_num()
      global_index = team * 4 + tid
      coefficients = [14_int32, 11_int32, 13_int32, 9_int32]
      block0(:, tid) = input(global_index * 4:global_index * 4 + 3)
      block0(:, tid) = ieor(block0(:, tid), round_key(rounds * 16 + tid * 4:rounds * 16 + tid * 4 + 3))
      do round = rounds - 1, 1, -1
        vector = block0(:, tid)
        call rotate_vector(vector, tid, .true.)
        block0(:, tid) = [rsbox(vector(0)), rsbox(vector(1)), rsbox(vector(2)), rsbox(vector(3))]
        !$omp barrier
        block1(:, tid) = ieor(block0(:, tid), round_key(round * 16 + tid * 4:round * 16 + tid * 4 + 3))
        !$omp barrier
        call device_mix_columns(block1, coefficients, tid, block0(:, tid))
      end do
      vector = block0(:, tid)
      call rotate_vector(vector, tid, .true.)
      vector = [rsbox(vector(0)), rsbox(vector(1)), rsbox(vector(2)), rsbox(vector(3))]
      output(global_index * 4:global_index * 4 + 3) = ieor(vector, round_key(tid * 4:tid * 4 + 3))
      !$omp end parallel
    end do
    !$omp end target teams distribute
  end subroutine aes_decrypt_gpu
end module aes_device

program aes
  use, intrinsic :: iso_c_binding, only : c_int, c_double
  use aes_precision
  use aes_common
  use aes_bitmap
  use aes_reference_mod
  use aes_device
  implicit none
  interface
    subroutine c_srand(seed) bind(C, name='srand')
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand
    integer(c_int) function c_rand() bind(C, name='rand')
      import :: c_int
    end function c_rand
  end interface
  integer, parameter :: rounds = 10, key_size = 16, expanded_key_size = 176
  integer :: argc, iterations, decrypt_integer, width, height, image_status, i
  integer(int64) :: start_count, end_count, clock_rate
  logical :: decrypt
  character(len=1024) :: argument, image_path
  integer(int32), allocatable :: input(:), output(:), verification_output(:)
  integer(int32) :: key(0:15), expanded_key(0:175), round_key(0:175), sbox(0:255), rsbox(0:255)
  real(real64) :: elapsed_seconds

  argc = command_argument_count()
  if (argc /= 3) then
    write(*, '(A)') 'Usage: ./main <iterations> <0 or 1> <path to bitmap image file>'
    write(*, '(A)') '0=encrypt, 1=decrypt'
    stop 1
  end if
  call get_command_argument(1, argument); read(argument, *) iterations
  call get_command_argument(2, argument); read(argument, *) decrypt_integer
  call get_command_argument(3, image_path)
  decrypt = decrypt_integer /= 0

  call load_bmp_gray(trim(image_path), decrypt, input, width, height, image_status)
  if (image_status /= 0) stop 1
  if (mod(width, 4) /= 0 .or. mod(height, 4) /= 0) then
    write(*, '(A)') 'Bitmap dimensions must be divisible by four.'
    stop 1
  end if
  write(*, '(A,1X,I0,1X,I0)') 'Image width and height:', width, height

  call initialize_sboxes(sbox, rsbox)
  call c_srand(123_c_int)
  do i = 0, key_size - 1
    key(i) = int(floor(256.0_c_double * real(c_rand(), c_double) / (real(huge(0_c_int), c_double) + 1.0_c_double)), int32)
  end do
  call key_expansion(key, expanded_key, sbox)
  call create_round_key(expanded_key, round_key)
  allocate(output(0:size(input) - 1), verification_output(0:size(input) - 1))

  write(*, '(A,I0,A)') 'Executing kernel for ', iterations, ' iterations'
  write(*, '(A)') '-------------------------------------------'
  !$omp target data map(to: input, round_key, sbox, rsbox) map(alloc: output)
  call system_clock(start_count, clock_rate)
  do i = 1, iterations
    call aes_transform_gpu(output, input, round_key, sbox, rsbox, width, height, decrypt, rounds)
  end do
  call system_clock(end_count)
  elapsed_seconds = real(end_count - start_count, real64) / real(clock_rate, real64) / real(iterations, real64)
  write(*, '(A,ES16.8,A)') 'Average kernel execution time ', elapsed_seconds, ' (s)'
  !$omp target update from(output)
  !$omp end target data

  call aes_reference(verification_output, input, round_key, width, height, decrypt, rounds, sbox, rsbox)
  if (all(output == verification_output)) then
    write(*, '(A)') 'Pass'
  else
    write(*, '(A)') 'Fail'
    stop 2
  end if
end program aes
