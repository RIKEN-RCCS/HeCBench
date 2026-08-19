module present_mod
  use iso_fortran_env, only: int32, int64, real32, real64
  implicit none
  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: int32
      integer(int32), value :: seed
    end subroutine
    function c_rand() bind(C, name="rand") result(v)
      import :: int32
      integer(int32) :: v
    end function
  end interface
contains
  real(real64) function seconds() result(t)
    integer(int64) :: c, r
    call system_clock(c, r)
    t = real(c, real64) / real(r, real64)
  end function

  integer(int32) function u8(x) result(v)
    integer(int32), intent(in) :: x
    v = iand(x, 255_int32)
  end function

  subroutine parse_table(name, table, n)
    character(len=*), intent(in) :: name
    integer(int32), intent(out) :: table(0:n-1)
    integer, intent(in) :: n
    character(len=:), allocatable :: text
    character(len=512) :: line
    character(len=16) :: tok
    integer :: u, ios, p, start, count, pos
    text = ''
    open(newunit=u, file='../present-omp/main.cpp', status='old', action='read')
    do
      read(u,'(a)',iostat=ios) line
      if (ios /= 0) exit
      text = text // trim(line) // ' '
    end do
    close(u)
    start = index(text, trim(name)//'[')
    if (start == 0) stop 'missing PRESENT table'
    start = index(text(start:), '{') + start
    count = 0; p = start
    do while (count < n)
      do while (p <= len(text) .and. text(p:p) /= '0'); p = p + 1; end do
      tok = ''; pos = 0
      do while (p <= len(text) .and. ((text(p:p)>='0'.and.text(p:p)<='9') .or. text(p:p)=='x' .or. &
        (text(p:p)>='A'.and.text(p:p)<='F') .or. (text(p:p)>='a'.and.text(p:p)<='f')))
        pos = pos + 1; tok(pos:pos) = text(p:p); p = p + 1
      end do
      if (index(tok,'x') > 0) then
        read(tok(index(tok,'x')+1:pos),'(z16)') table(count)
      else
        read(tok(1:pos),*) table(count)
      end if
      count = count + 1
    end do
  end subroutine

  subroutine load_tables(sbox, p3, p2, p1, p0)
    integer(int32), intent(out) :: sbox(0:15), p3(0:255), p2(0:255), p1(0:255), p0(0:255)
    call parse_table('sbox', sbox, 16)
    call parse_table('sbox_pmt_3', p3, 256)
    call parse_table('sbox_pmt_2', p2, 256)
    call parse_table('sbox_pmt_1', p1, 256)
    call parse_table('sbox_pmt_0', p0, 256)
  end subroutine

  subroutine present_rounds(plain, key, rounds, cipher, sbox, p3, p2, p1, p0)
    integer(int32), intent(in) :: plain(0:7), key(0:9), rounds, sbox(0:15), p3(0:255), p2(0:255), p1(0:255), p0(0:255)
    integer(int32), intent(out) :: cipher(0:7)
    integer(int32) :: state(0:7), round_key(0:9), round_counter
    integer :: i
    do i = 0, 7
      state(i) = ieor(plain(i), key(i))
    end do
    round_key(9)=u8(ior(ishft(key(6),5), shiftr(key(7),3)))
    round_key(8)=u8(ior(ishft(key(5),5), shiftr(key(6),3)))
    round_key(7)=u8(ior(ishft(key(4),5), shiftr(key(5),3)))
    round_key(6)=u8(ior(ishft(key(3),5), shiftr(key(4),3)))
    round_key(5)=u8(ior(ishft(key(2),5), shiftr(key(3),3)))
    round_key(4)=u8(ior(ishft(key(1),5), shiftr(key(2),3)))
    round_key(3)=u8(ior(ishft(key(0),5), shiftr(key(1),3)))
    round_key(2)=u8(ior(ishft(key(9),5), shiftr(key(0),3)))
    round_key(1)=u8(ior(ishft(key(8),5), shiftr(key(9),3)))
    round_key(0)=u8(ior(ishft(key(7),5), shiftr(key(8),3)))
    round_key(0)=ior(iand(round_key(0),15), sbox(shiftr(round_key(0),4)))
    round_counter = 1
    round_key(7)=ieor(round_key(7), shiftr(round_counter,1))
    round_key(8)=ieor(round_key(8), u8(ishft(round_counter,7)))
    call sp_layer(state, cipher, p3, p2, p1, p0)
    do round_counter = 2, rounds
      do i = 0, 7
        state(i) = ieor(cipher(i), round_key(i))
      end do
      call sp_layer(state, cipher, p3, p2, p1, p0)
      round_key(5) = ieor(round_key(5), u8(ishft(round_counter,2)))
      state(2) = round_key(9); state(1) = round_key(8); state(0) = round_key(7)
      round_key(9)=u8(ior(ishft(round_key(6),5), shiftr(round_key(7),3)))
      round_key(8)=u8(ior(ishft(round_key(5),5), shiftr(round_key(6),3)))
      round_key(7)=u8(ior(ishft(round_key(4),5), shiftr(round_key(5),3)))
      round_key(6)=u8(ior(ishft(round_key(3),5), shiftr(round_key(4),3)))
      round_key(5)=u8(ior(ishft(round_key(2),5), shiftr(round_key(3),3)))
      round_key(4)=u8(ior(ishft(round_key(1),5), shiftr(round_key(2),3)))
      round_key(3)=u8(ior(ishft(round_key(0),5), shiftr(round_key(1),3)))
      round_key(2)=u8(ior(ishft(state(2),5), shiftr(round_key(0),3)))
      round_key(1)=u8(ior(ishft(state(1),5), shiftr(state(2),3)))
      round_key(0)=u8(ior(ishft(state(0),5), shiftr(state(1),3)))
      round_key(0)=ior(iand(round_key(0),15), sbox(shiftr(round_key(0),4)))
    end do
    if (rounds == 31) then
      do i = 0, 7
        cipher(i) = ieor(cipher(i), round_key(i))
      end do
    end if
  end subroutine

  subroutine sp_layer(state, cipher, p3, p2, p1, p0)
    integer(int32), intent(in) :: state(0:7), p3(0:255), p2(0:255), p1(0:255), p0(0:255)
    integer(int32), intent(out) :: cipher(0:7)
    cipher(0)=ior(ior(ior(iand(p3(state(0)),int(z'C0')), iand(p2(state(1)),int(z'30'))), iand(p1(state(2)),int(z'0C'))), iand(p0(state(3)),int(z'03')))
    cipher(1)=ior(ior(ior(iand(p3(state(4)),int(z'C0')), iand(p2(state(5)),int(z'30'))), iand(p1(state(6)),int(z'0C'))), iand(p0(state(7)),int(z'03')))
    cipher(2)=ior(ior(ior(iand(p0(state(0)),int(z'C0')), iand(p3(state(1)),int(z'30'))), iand(p2(state(2)),int(z'0C'))), iand(p1(state(3)),int(z'03')))
    cipher(3)=ior(ior(ior(iand(p0(state(4)),int(z'C0')), iand(p3(state(5)),int(z'30'))), iand(p2(state(6)),int(z'0C'))), iand(p1(state(7)),int(z'03')))
    cipher(4)=ior(ior(ior(iand(p1(state(0)),int(z'C0')), iand(p0(state(1)),int(z'30'))), iand(p3(state(2)),int(z'0C'))), iand(p2(state(3)),int(z'03')))
    cipher(5)=ior(ior(ior(iand(p1(state(4)),int(z'C0')), iand(p0(state(5)),int(z'30'))), iand(p3(state(6)),int(z'0C'))), iand(p2(state(7)),int(z'03')))
    cipher(6)=ior(ior(ior(iand(p2(state(0)),int(z'C0')), iand(p1(state(1)),int(z'30'))), iand(p0(state(2)),int(z'0C'))), iand(p3(state(3)),int(z'03')))
    cipher(7)=ior(ior(ior(iand(p2(state(4)),int(z'C0')), iand(p1(state(5)),int(z'30'))), iand(p0(state(6)),int(z'0C'))), iand(p3(state(7)),int(z'03')))
  end subroutine
end module

program main
  use present_mod
  implicit none
  integer :: num, repeat, rounds, i, k, r, ios
  character(len=64) :: arg
  integer(int32), allocatable :: h_plain(:), h_key(:), h_cipher(:), ciphers(:)
  integer(int32), allocatable :: sbox(:), p3(:), p2(:), p1(:), p0(:)
  integer(int64) :: h_checksum, d_checksum
  integer(int32) :: plain(0:7), key(0:9), cipher(0:7), tmp
  real(real64) :: time, t0, t1
  if (command_argument_count() /= 2) then
    print '(a)', 'Usage: main <number of plain texts> <repeat>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*,iostat=ios) num
  call get_command_argument(2,arg); read(arg,*,iostat=ios) repeat
  allocate(h_plain(0:8*num-1), h_key(0:10*num-1), h_cipher(0:8*num-1), ciphers(0:8*num-1), &
           sbox(0:15), p3(0:255), p2(0:255), p1(0:255), p0(0:255))
  call load_tables(sbox, p3, p2, p1, p0)
  call c_srand(8_int32)
  plain = [iachar('P'), iachar('R'), iachar('E'), iachar('S'), iachar('E'), iachar('N'), iachar('T'), 0]
  do i = 0, num-1
    do k = 0, 9
      h_key(i*10+k) = mod(c_rand(), 256)
    end do
    h_plain(i*8:i*8+7) = plain
    tmp = plain(0); plain(0:6) = plain(1:7); plain(7) = tmp
  end do
  rounds = 31
  h_checksum = 0_int64
  do r = 0, repeat
    do i = 0, num-1
      call present_rounds(h_plain(i*8:), h_key(i*10:), rounds, h_cipher(i*8:), sbox, p3, p2, p1, p0)
      do k = 0, 7
        h_checksum = h_checksum + h_cipher(i*8+k)
      end do
    end do
  end do
  d_checksum = 0_int64
  !$omp target data map(to:h_plain(0:8*num-1),h_key(0:10*num-1),sbox(0:15),p3(0:255),p2(0:255),p1(0:255),p0(0:255)) map(alloc:ciphers(0:8*num-1))
  time = 0.0_real64
  do r = 0, repeat
    t0 = seconds()
    !$omp target teams distribute parallel do private(cipher) thread_limit(256)
    do i = 0, num-1
      call present_rounds(h_plain(i*8:), h_key(i*10:), rounds, ciphers(i*8:), sbox, p3, p2, p1, p0)
    end do
    !$omp end target teams distribute parallel do
    t1 = seconds()
    if (r > 0) time = time + (t1-t0)*1.0e9_real64
    !$omp target update from(ciphers(0:8*num-1))
    do i = 0, 8*num-1
      d_checksum = d_checksum + ciphers(i)
    end do
  end do
  print '(a,f10.6,a)', 'Average kernel execution time: ', (time*1.0e-3_real64)/real(repeat,real64), ' (us)'
  !$omp end target data
  print '(a)', merge('PASS', 'FAIL', h_checksum == d_checksum)
end program
