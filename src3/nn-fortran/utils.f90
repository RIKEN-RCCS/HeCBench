module nn_utils
  use iso_fortran_env, only: int64, real32, real64
  implicit none
  integer, parameter :: rec_length = 49
  type :: latlong
    real(real32) :: lat, lng
  end type
  type :: record
    character(len=rec_length) :: rec_string
    real(real32) :: distance
  end type
contains
  real(real64) function seconds() result(t)
    integer(int64) :: c, r
    call system_clock(c, r)
    t = real(c, real64) / real(r, real64)
  end function

  subroutine print_usage()
    print '(a)', 'Nearest Neighbor Usage'
    print *
    print '(a)', 'nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt]'
    print *
    print '(a)', 'example:'
    print '(a)', '$ ./nearestNeighbor filelist.txt -r 5 -lat 30 -lng 90 -i 100'
  end subroutine

  integer function parse_commandline(filename, results_count, lat, lng, repeat, quiet, timing) result(err)
    character(len=*), intent(out) :: filename
    integer, intent(inout) :: results_count, repeat, quiet, timing
    real(real32), intent(inout) :: lat, lng
    integer :: argc, i, ios
    character(len=128) :: arg, val
    argc = command_argument_count()
    if (argc < 1) then
      err = 1; return
    end if
    call get_command_argument(1, filename)
    i = 1; err = 0
    do while (i <= argc)
      call get_command_argument(i, arg)
      if (arg(1:1) == '-') then
        select case (arg(2:2))
        case ('r')
          i = i + 1; call get_command_argument(i, val); read(val,*,iostat=ios) results_count
        case ('l')
          i = i + 1; call get_command_argument(i, val)
          if (len_trim(arg) >= 3 .and. arg(3:3) == 'a') then
            read(val,*,iostat=ios) lat
          else
            read(val,*,iostat=ios) lng
          end if
        case ('i')
          i = i + 1; call get_command_argument(i, val); read(val,*,iostat=ios) repeat
        case ('h')
          err = 1; return
        case ('q')
          quiet = 1
        case ('t')
          timing = 1
        end select
      end if
      i = i + 1
    end do
  end function

  subroutine append_record(records, locations, n, rec, ll)
    type(record), allocatable, intent(inout) :: records(:)
    type(latlong), allocatable, intent(inout) :: locations(:)
    integer, intent(inout) :: n
    type(record), intent(in) :: rec
    type(latlong), intent(in) :: ll
    type(record), allocatable :: nr(:)
    type(latlong), allocatable :: nl(:)
    integer :: cap
    if (.not. allocated(records)) then
      allocate(records(0:1023), locations(0:1023))
    else if (n > ubound(records,1)) then
      cap = size(records) * 2
      allocate(nr(0:cap-1), nl(0:cap-1))
      nr(0:n-1) = records(0:n-1); nl(0:n-1) = locations(0:n-1)
      call move_alloc(nr, records); call move_alloc(nl, locations)
    end if
    records(n) = rec; locations(n) = ll; n = n + 1
  end subroutine

  integer function load_data(filename, records, locations) result(rec_num)
    character(len=*), intent(in) :: filename
    type(record), allocatable, intent(out) :: records(:)
    type(latlong), allocatable, intent(out) :: locations(:)
    integer :: flist, fp, ios
    character(len=256) :: dbname, line, substr
    type(record) :: rec
    type(latlong) :: ll
    rec_num = 0
    open(newunit=flist, file=trim(filename), status='old', action='read', iostat=ios)
    if (ios /= 0) stop 'error reading filelist'
    do
      read(flist,'(a)',iostat=ios) dbname
      if (ios /= 0) exit
      if (len_trim(dbname) == 0) cycle
      open(newunit=fp, file=trim(dbname), status='old', action='read', iostat=ios)
      if (ios /= 0) stop 'error opening a db'
      do
        read(fp,'(a)',iostat=ios) line
        if (ios /= 0) exit
        if (len_trim(line) < 38) cycle
        rec%rec_string = line(1:min(len(line),rec_length))
        substr = line(29:33); read(substr,*) ll%lat
        substr = line(34:38); read(substr,*) ll%lng
        call append_record(records, locations, rec_num, rec, ll)
      end do
      close(fp)
    end do
    close(flist)
  end function

  subroutine find_lowest(records, distances, num_records, topn)
    type(record), intent(inout) :: records(0:num_records-1)
    real(real32), intent(inout) :: distances(0:num_records-1)
    integer, intent(in) :: num_records, topn
    integer :: i, j, minloc
    real(real32) :: val, tmpd
    type(record) :: tmpr
    do i = 0, topn-1
      minloc = i
      do j = i, num_records-1
        val = distances(j)
        if (val < distances(minloc)) minloc = j
      end do
      tmpr = records(i); records(i) = records(minloc); records(minloc) = tmpr
      tmpd = distances(i); distances(i) = distances(minloc); distances(minloc) = tmpd
      records(i)%distance = distances(i)
    end do
  end subroutine
end module
