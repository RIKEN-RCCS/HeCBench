program triad
  use option_parser_mod
  use triad_kernel_mod
  implicit none
  type(option_t) :: opt
  logical :: ok
  ok = parse_options(opt)
  if (.not. ok) then
    call usage()
    if (opt%help) stop 0
    stop 1
  end if
  call run_benchmark(opt%verbose, opt%passes)
end program triad
