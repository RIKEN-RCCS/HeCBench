program main
  use iso_fortran_env, only: real64, int64
  use wc_mod
  implicit none
  character(len=*), parameter :: raw_input = &
    '  But the raven, sitting lonely on the placid bust, spoke only,' // new_line('a') // &
    '  That one word, as if his soul in that one word he did outpour.' // new_line('a') // &
    '  Nothing further then he uttered - not a feather then he fluttered -' // new_line('a') // &
    '  Till I scarcely more than muttered `Other friends have flown before -' // new_line('a') // &
    '  On the morrow he will leave me, as my hopes have flown before.''' // new_line('a') // &
    '  Then the bird said, `Nevermore.''' // new_line('a') // char(0)
  character(len=1), allocatable :: input(:), random_input(:)
  character(len=*), parameter :: tab = 'abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  integer :: argc, repeat, stat, wc_host, wc_device, i, c, len_input, tab_size
  integer(int64) :: test_len, perf_len
  integer :: c0, c1, rate
  character(len=128) :: arg
  real(real64) :: elapsed
  logical :: ok

  argc = command_argument_count()
  if (argc /= 1) then
    print '(a)', 'Usage: ./main <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *, iostat=stat) repeat
  print '(a)', 'Text sample:'
  print '(a)', raw_input
  len_input = len(raw_input)
  allocate(input(0:len_input-1))
  do i = 0, len_input - 1
    input(i) = raw_input(i+1:i+1)
  end do
  wc_host = word_count_reference(input)
  print '(a,i0,a)', 'Host: Text sample contains ', wc_host, ' words'
  wc_device = word_count(input)
  print '(a,i0,a)', 'Device: Text sample contains ', wc_device, ' words'
  print '(a)', 'Test word count with random inputs'
  ok = .true.
  tab_size = len(tab)
  test_len = 1_int64
  do while (test_len <= 100000000_int64)
    allocate(random_input(0:test_len-1))
    do c = 0, int(test_len) - 1
      random_input(c) = tab(mod(c * 1103515245 + 123, tab_size) + 1:mod(c * 1103515245 + 123, tab_size) + 1)
    end do
    if (word_count_reference(random_input) /= word_count(random_input)) ok = .false.
    deallocate(random_input)
    if (.not. ok) exit
    test_len = test_len * 10_int64
  end do
  print '(a)', merge('PASS', 'FAIL', ok)
  perf_len = 1024_int64 * 1024_int64 * 256_int64
  allocate(random_input(0:perf_len-1))
  do c = 0, int(perf_len) - 1
    random_input(c) = tab(mod(c * 1103515245 + 123, tab_size) + 1:mod(c * 1103515245 + 123, tab_size) + 1)
  end do
  print '(a,i0)', 'Performance evaluation for random texts of character length ', perf_len
  call system_clock(c0, rate)
  do i = 1, repeat
    wc_device = word_count(random_input)
  end do
  call system_clock(c1)
  elapsed = real(c1 - c0, real64) / real(rate, real64)
  print '(a,f12.6,a)', 'Average time of word count: ', elapsed / real(repeat, real64), ' (s)'
end program main
