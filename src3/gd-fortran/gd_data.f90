module gd_data
  use iso_fortran_env, only: real32, int32
  implicit none

  type :: classification_data_crs
    integer(int32) :: m = 0, n = 0, nnz = 0
    integer(int32), allocatable :: row_ptr(:), col_index(:), y_label(:)
    real(real32), allocatable :: values(:)
  end type classification_data_crs

contains

  subroutine load_libsvm(filename, a)
    character(len=*), intent(in) :: filename
    type(classification_data_crs), intent(out) :: a
    character(len=65536) :: line
    integer :: unit, ios, i, p, q, feature, label, rows, entries, max_feature
    real(real32) :: value, norm2

    open(newunit=unit, file=trim(filename), status='old', action='read', iostat=ios)
    if (ios /= 0) error stop 'Could not open the LIBSVM input file'
    rows = 0; entries = 0; max_feature = 0
    do
      read(unit, '(A)', iostat=ios) line
      if (ios /= 0) exit
      if (len_trim(line) == 0) cycle
      rows = rows + 1
      call count_features(line, entries, max_feature)
    end do
    if (rows == 0) error stop 'The LIBSVM input file is empty'
    allocate(a%row_ptr(0:rows), a%y_label(rows), a%col_index(entries), a%values(entries))
    rewind(unit)
    a%row_ptr(0) = 0
    i = 0
    entries = 0
    do
      read(unit, '(A)', iostat=ios) line
      if (ios /= 0) exit
      if (len_trim(line) == 0) cycle
      i = i + 1
      call parse_label(line, label, p)
      a%y_label(i) = label
      do
        call next_pair(line, p, feature, value, q)
        if (q == 0) exit
        entries = entries + 1
        a%col_index(entries) = feature
        a%values(entries) = value
        p = q
      end do
      a%row_ptr(i) = entries
    end do
    close(unit)
    a%m = rows; a%n = max_feature; a%nnz = entries
    do i = 1, a%m
      norm2 = sum(a%values(a%row_ptr(i-1)+1:a%row_ptr(i))**2)
      if (norm2 > 0.0_real32) a%values(a%row_ptr(i-1)+1:a%row_ptr(i)) = &
        a%values(a%row_ptr(i-1)+1:a%row_ptr(i)) / sqrt(norm2)
    end do
    write(*,'(a,i0,a,i0,a,i0)') 'Finished processing the LIBSVM file. ', a%m, &
      ' observations and ', a%n, ' features; non-zero entries: ', a%nnz
  end subroutine load_libsvm

  subroutine count_features(line, entries, max_feature)
    character(len=*), intent(in) :: line
    integer, intent(inout) :: entries, max_feature
    integer :: p, feature, q
    real(real32) :: value
    call parse_label(line, feature, p)
    do
      call next_pair(line, p, feature, value, q)
      if (q == 0) exit
      entries = entries + 1; max_feature = max(max_feature, feature); p = q
    end do
  end subroutine count_features

  subroutine parse_label(line, label, pos)
    character(len=*), intent(in) :: line
    integer, intent(out) :: label, pos
    integer :: first, last, ios
    first = 1
    do while (first <= len_trim(line) .and. line(first:first) == ' ')
      first = first + 1
    end do
    last = first
    do while (last <= len_trim(line) .and. line(last:last) /= ' ')
      last = last + 1
    end do
    read(line(first:last-1), *, iostat=ios) label
    if (ios /= 0) error stop 'Invalid LIBSVM label'
    pos = last
  end subroutine parse_label

  subroutine next_pair(line, pos, feature, value, next_pos)
    character(len=*), intent(in) :: line
    integer, intent(in) :: pos
    integer, intent(out) :: feature, next_pos
    real(real32), intent(out) :: value
    integer :: begin, colon, finish, ios, n
    n = len_trim(line); begin = pos
    do while (begin <= n .and. line(begin:begin) == ' ')
      begin = begin + 1
    end do
    if (begin > n) then; next_pos = 0; return; end if
    colon = begin
    do while (colon <= n .and. line(colon:colon) /= ':')
      colon = colon + 1
    end do
    if (colon > n) error stop 'Invalid LIBSVM feature'
    finish = colon + 1
    do while (finish <= n .and. line(finish:finish) /= ' ')
      finish = finish + 1
    end do
    read(line(begin:colon-1), *, iostat=ios) feature
    if (ios /= 0) error stop 'Invalid LIBSVM index'
    read(line(colon+1:finish-1), *, iostat=ios) value
    if (ios /= 0) error stop 'Invalid LIBSVM value'
    next_pos = finish
  end subroutine next_pair
end module gd_data
