module snake_mod
  use iso_fortran_env, only: int32
  implicit none
  integer, parameter :: warp_size = 32
  integer, parameter :: nbytes = 8

contains

  integer(int32) function lsr(x, sa) result(r)
    integer(int32), intent(in) :: x
    integer, intent(in) :: sa
    if (sa > 0 .and. sa < 32) then
      r = shiftr(x, sa)
    else
      r = x
    end if
  end function lsr

  integer(int32) function lsl(x, sa) result(r)
    integer(int32), intent(in) :: x
    integer, intent(in) :: sa
    if (sa > 0 .and. sa < 32) then
      r = shiftl(x, sa)
    else
      r = x
    end if
  end function lsl

  integer(int32) function set_bit(data, y) result(r)
    integer(int32), intent(in) :: data
    integer, intent(in) :: y
    r = ior(data, lsl(1_int32, y))
  end function set_bit

  integer function clz32(x) result(r)
    integer(int32), intent(in) :: x
    r = leadz(x)
  end function clz32

  subroutine sneaky_snake(nteams, nthreads, readseq, refseq, results, num_reads, error_threshold)
    integer, intent(in) :: nteams, nthreads, num_reads, error_threshold
    integer(int32), intent(in) :: readseq(0:), refseq(0:)
    integer, intent(out) :: results(0:)
    call sneaky_snake_body(.true., nteams, nthreads, readseq, refseq, results, num_reads, error_threshold)
  end subroutine sneaky_snake

  subroutine sneaky_snake_ref(readseq, refseq, results, num_reads, error_threshold)
    integer, intent(in) :: num_reads, error_threshold
    integer(int32), intent(in) :: readseq(0:), refseq(0:)
    integer, intent(out) :: results(0:)
    call sneaky_snake_body(.false., 1, 1, readseq, refseq, results, num_reads, error_threshold)
  end subroutine sneaky_snake_ref

  subroutine sneaky_snake_body(device, nteams, nthreads, readseq, refseq, results, num_reads, error_threshold)
    logical, intent(in) :: device
    integer, intent(in) :: nteams, nthreads, num_reads, error_threshold
    integer(int32), intent(in) :: readseq(0:), refseq(0:)
    integer, intent(out) :: results(0:)
    integer :: tid

    if (device) then
      !$omp target teams distribute parallel do num_teams(nteams) thread_limit(nthreads)
      do tid = 0, num_reads - 1
        call one_read(tid, readseq, refseq, results, error_threshold)
      end do
      !$omp end target teams distribute parallel do
    else
      do tid = 0, num_reads - 1
        call one_read(tid, readseq, refseq, results, error_threshold)
      end do
    end if
  end subroutine sneaky_snake_body

  subroutine one_read(tid, readseq, refseq, results, error_threshold)
    integer, intent(in) :: tid, error_threshold
    integer(int32), intent(in) :: readseq(0:), refseq(0:)
    integer, intent(inout) :: results(0:)
    integer(int32) :: reads_per_thread(0:nbytes-1), refs_per_thread(0:nbytes-1)
    integer(int32) :: read_comp_tmp, ref_comp_tmp, diagonal_result, read_tmp1, read_tmp2, ref_tmp1, ref_tmp2, corner_case
    integer :: i, e, ci, j, diagonal, shift_value, local_counter, local_counter_max, global_counter
    integer :: max_leading_zeros, accumulated_errs

    do i = 0, nbytes - 1
      reads_per_thread(i) = readseq(tid * 8 + i)
      refs_per_thread(i) = refseq(tid * 8 + i)
    end do
    results(tid) = 1
    read_comp_tmp = 0; ref_comp_tmp = 0; diagonal_result = 0
    read_tmp1 = 0; read_tmp2 = 0; ref_tmp1 = 0; ref_tmp2 = 0; corner_case = 0
    local_counter = 0; local_counter_max = 0; global_counter = 0; max_leading_zeros = 0; accumulated_errs = 0
    diagonal = 0; shift_value = 0; j = 0

    do while (j < 7 .and. global_counter < 200)
      diagonal = 0
      ref_tmp1 = lsl(refs_per_thread(j), shift_value)
      ref_tmp2 = lsr(refs_per_thread(j + 1), 32 - shift_value)
      read_tmp1 = lsl(reads_per_thread(j), shift_value)
      read_tmp2 = lsr(reads_per_thread(j + 1), 32 - shift_value)
      read_comp_tmp = ior(read_tmp1, read_tmp2)
      ref_comp_tmp = ior(ref_tmp1, ref_tmp2)
      diagonal_result = ieor(read_comp_tmp, ref_comp_tmp)
      local_counter_max = clz32(diagonal_result)

      do e = 1, error_threshold
        diagonal = diagonal + 1
        corner_case = 0
        if (j == 0 .and. shift_value - 2 * e < 0) then
          read_tmp1 = lsr(reads_per_thread(j), 2 * e - shift_value)
          read_tmp2 = 0
          read_comp_tmp = ior(read_tmp1, read_tmp2)
          ref_comp_tmp = ior(ref_tmp1, ref_tmp2)
          diagonal_result = ieor(read_comp_tmp, ref_comp_tmp)
          do ci = 0, (2 * e) - shift_value - 1
            corner_case = set_bit(corner_case, 31 - ci)
          end do
          diagonal_result = ior(diagonal_result, corner_case)
          local_counter = clz32(diagonal_result)
        else if (shift_value - 2 * e < 0) then
          read_tmp1 = lsl(reads_per_thread(j - 1), 32 - (2 * e - shift_value))
          read_tmp2 = lsr(reads_per_thread(j), 2 * e - shift_value)
          read_comp_tmp = ior(read_tmp1, read_tmp2)
          ref_comp_tmp = ior(ref_tmp1, ref_tmp2)
          diagonal_result = ieor(read_comp_tmp, ref_comp_tmp)
          local_counter = clz32(diagonal_result)
        else
          read_tmp1 = lsl(reads_per_thread(j), shift_value - 2 * e)
          read_tmp2 = lsr(reads_per_thread(j + 1), 32 - (shift_value - 2 * e))
          read_comp_tmp = ior(read_tmp1, read_tmp2)
          ref_comp_tmp = ior(ref_tmp1, ref_tmp2)
          diagonal_result = ieor(read_comp_tmp, ref_comp_tmp)
          local_counter = clz32(diagonal_result)
        end if
        if (local_counter > local_counter_max) local_counter_max = local_counter
      end do

      do e = 1, error_threshold
        diagonal = diagonal + 1
        corner_case = 0
        if (j < 5) then
          if (shift_value + 2 * e < 32) then
            read_tmp1 = lsl(reads_per_thread(j), shift_value + 2 * e)
            read_tmp2 = lsr(reads_per_thread(j + 1), 32 - (shift_value + 2 * e))
          else
            read_tmp1 = lsl(reads_per_thread(j + 1), mod(shift_value + 2 * e, 32))
            read_tmp2 = lsr(reads_per_thread(j + 2), 32 - mod(shift_value + 2 * e, 32))
          end if
          read_comp_tmp = ior(read_tmp1, read_tmp2)
          ref_comp_tmp = ior(ref_tmp1, ref_tmp2)
          diagonal_result = ieor(read_comp_tmp, ref_comp_tmp)
          local_counter = clz32(diagonal_result)
        else
          read_tmp1 = lsl(reads_per_thread(j), shift_value + 2 * e)
          read_tmp2 = lsr(reads_per_thread(j + 1), 32 - (shift_value + 2 * e))
          read_comp_tmp = ior(read_tmp1, read_tmp2)
          ref_comp_tmp = ior(ref_tmp1, ref_tmp2)
          diagonal_result = ieor(read_comp_tmp, ref_comp_tmp)
          if (global_counter + 32 > 200) then
            do ci = global_counter + 32 - 200, global_counter + 32 - 200 + 2 * e - 1
              corner_case = set_bit(corner_case, ci)
            end do
          else if (global_counter + 32 >= 200 - 2 * e) then
            do ci = 0, 2 * e - 1
              corner_case = set_bit(corner_case, ci)
            end do
          end if
          diagonal_result = ior(diagonal_result, corner_case)
          local_counter = clz32(diagonal_result)
        end if
        if (local_counter > local_counter_max) local_counter_max = local_counter
      end do

      max_leading_zeros = 0
      if (j == 6 .and. (local_counter_max / 2) * 2 >= 8) then
        max_leading_zeros = 8
        exit
      else if ((local_counter_max / 2) * 2 > max_leading_zeros) then
        max_leading_zeros = (local_counter_max / 2) * 2
      end if
      if (max_leading_zeros / 2 < 16 .and. j < 5) then
        accumulated_errs = accumulated_errs + 1
      else if (j == 6 .and. max_leading_zeros / 2 < 4) then
        accumulated_errs = accumulated_errs + 1
      end if
      if (accumulated_errs > error_threshold) then
        results(tid) = 0
        exit
      end if
      if (shift_value + max_leading_zeros + 2 >= 32) j = j + 1
      if (max_leading_zeros == 32) then
        global_counter = global_counter + max_leading_zeros
      else
        shift_value = mod(shift_value + max_leading_zeros + 2, 32)
        global_counter = global_counter + max_leading_zeros + 2
      end if
    end do
  end subroutine one_read

end module snake_mod
