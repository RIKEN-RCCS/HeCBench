module option_parser_mod
  implicit none
  type :: option_t
    logical :: verbose = .false.
    integer :: passes = 10
    logical :: help = .false.
  end type option_t
contains
  logical function parse_options(opt)
    type(option_t), intent(inout) :: opt
    integer :: i, n, stat
    character(len=128) :: arg, nxt
    parse_options = .true.
    n = command_argument_count()
    i = 1
    do while (i <= n)
      call get_command_argument(i, arg)
      select case (trim(arg))
      case ('--help', '-h')
        opt%help = .true.; parse_options = .false.; return
      case ('--verbose', '-v')
        opt%verbose = .true.
      case ('--passes', '-n')
        if (i == n) then
          parse_options = .false.; return
        end if
        call get_command_argument(i + 1, nxt)
        read(nxt, *, iostat=stat) opt%passes
        if (stat /= 0) parse_options = .false.
        i = i + 1
      case default
        print '(a,a)', 'Option not recognized: ', trim(arg)
        parse_options = .false.; return
      end select
      i = i + 1
    end do
  end function parse_options

  subroutine usage()
    print '(a)', 'Options:'
    print '(a)', '  -v, --verbose'
    print '(a)', '  -n, --passes <int>'
  end subroutine usage
end module option_parser_mod
