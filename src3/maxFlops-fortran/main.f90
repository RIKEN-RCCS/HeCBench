module c_drand48
  use iso_c_binding, only: c_long, c_double
  implicit none
  interface
    subroutine srand48(seed) bind(C, name="srand48")
      import c_long
      integer(c_long), value :: seed
    end subroutine srand48
    function drand48() result(r) bind(C, name="drand48")
      import c_double
      real(c_double) :: r
    end function drand48
  end interface
end module c_drand48

module maxflops_kernels
  use iso_fortran_env, only: real32, real64
  implicit none
  integer, parameter :: BLOCK_SIZE = 256
!$omp declare target (op_repeats, init_lanes_sp, init_lanes_dp, apply_ops_sp, apply_ops_dp)
contains
  subroutine kernel_sp(data, nfloats, niters, family, lanes, v1, v2)
    real(real32), intent(inout) :: data(0:)
    integer, intent(in) :: nfloats, niters, family, lanes
    real(real32), intent(in) :: v1, v2
    select case (family)
    case (1)
      select case (lanes)
      case (1); call add1_sp(data, nfloats, niters, v1)
      case (2); call add2_sp(data, nfloats, niters, v1)
      case (4); call add4_sp(data, nfloats, niters, v1)
      case (8); call add8_sp(data, nfloats, niters, v1)
      end select
    case (2)
      select case (lanes)
      case (1); call mul1_sp(data, nfloats, niters, v1)
      case (2); call mul2_sp(data, nfloats, niters, v1)
      case (4); call mul4_sp(data, nfloats, niters, v1)
      case (8); call mul8_sp(data, nfloats, niters, v1)
      end select
    case (3)
      select case (lanes)
      case (1); call madd1_sp(data, nfloats, niters, v1, v2)
      case (2); call madd2_sp(data, nfloats, niters, v1, v2)
      case (4); call madd4_sp(data, nfloats, niters, v1, v2)
      case (8); call madd8_sp(data, nfloats, niters, v1, v2)
      end select
    case (4)
      select case (lanes)
      case (1); call mulmadd1_sp(data, nfloats, niters, v1, v2)
      case (2); call mulmadd2_sp(data, nfloats, niters, v1, v2)
      case (4); call mulmadd4_sp(data, nfloats, niters, v1, v2)
      case (8); call mulmadd8_sp(data, nfloats, niters, v1, v2)
      end select
    end select
  end subroutine kernel_sp

  subroutine kernel_dp(data, nfloats, niters, family, lanes, v1, v2)
    real(real64), intent(inout) :: data(0:)
    integer, intent(in) :: nfloats, niters, family, lanes
    real(real64), intent(in) :: v1, v2
    select case (family)
    case (1)
      select case (lanes)
      case (1); call add1_dp(data, nfloats, niters, v1)
      case (2); call add2_dp(data, nfloats, niters, v1)
      case (4); call add4_dp(data, nfloats, niters, v1)
      case (8); call add8_dp(data, nfloats, niters, v1)
      end select
    case (2)
      select case (lanes)
      case (1); call mul1_dp(data, nfloats, niters, v1)
      case (2); call mul2_dp(data, nfloats, niters, v1)
      case (4); call mul4_dp(data, nfloats, niters, v1)
      case (8); call mul8_dp(data, nfloats, niters, v1)
      end select
    case (3)
      select case (lanes)
      case (1); call madd1_dp(data, nfloats, niters, v1, v2)
      case (2); call madd2_dp(data, nfloats, niters, v1, v2)
      case (4); call madd4_dp(data, nfloats, niters, v1, v2)
      case (8); call madd8_dp(data, nfloats, niters, v1, v2)
      end select
    case (4)
      select case (lanes)
      case (1); call mulmadd1_dp(data, nfloats, niters, v1, v2)
      case (2); call mulmadd2_dp(data, nfloats, niters, v1, v2)
      case (4); call mulmadd4_dp(data, nfloats, niters, v1, v2)
      case (8); call mulmadd8_dp(data, nfloats, niters, v1, v2)
      end select
    end select
  end subroutine kernel_dp

  pure integer function op_repeats(family, lanes)
    integer, intent(in) :: family, lanes
    select case (family)
    case (1,3)
      select case (lanes)
      case (1); op_repeats = 240
      case (2); op_repeats = 120
      case (4); op_repeats = 60
      case default; op_repeats = 30
      end select
    case (2)
      select case (lanes)
      case (1); op_repeats = 200
      case (2); op_repeats = 100
      case (4); op_repeats = 50
      case default; op_repeats = 25
      end select
    case default
      select case (lanes)
      case (1); op_repeats = 160
      case (2); op_repeats = 80
      case (4); op_repeats = 40
      case default; op_repeats = 20
      end select
    end select
  end function op_repeats

  pure subroutine init_lanes_sp(x, family, lanes, s)
    real(real32), intent(in) :: x
    integer, intent(in) :: family, lanes
    real(real32), intent(out) :: s(8)
    s = 0.0_real32
    select case (family)
    case (2)
      s(1) = x - x + 0.999_real32
      s(2) = s(1) - 0.0001_real32
      s(3) = s(1) - 0.0002_real32
      s(4) = s(1) - 0.0003_real32
      s(5) = s(1) - 0.0004_real32
      s(6) = s(1) - 0.0005_real32
      s(7) = s(1) - 0.0006_real32
      s(8) = s(1) - 0.0007_real32
    case default
      s(1) = x
      s(2) = 10.0_real32 - x
      s(3) = 9.0_real32 - x
      s(4) = 9.0_real32 - s(2)
      s(5) = 8.0_real32 - x
      s(6) = 8.0_real32 - s(2)
      s(7) = 7.0_real32 - x
      s(8) = 7.0_real32 - s(2)
    end select
  end subroutine init_lanes_sp

  pure subroutine init_lanes_dp(x, family, lanes, s)
    real(real64), intent(in) :: x
    integer, intent(in) :: family, lanes
    real(real64), intent(out) :: s(8)
    s = 0.0_real64
    select case (family)
    case (2)
      s(1) = x - x + 0.999_real64
      s(2) = s(1) - 0.0001_real64
      s(3) = s(1) - 0.0002_real64
      s(4) = s(1) - 0.0003_real64
      s(5) = s(1) - 0.0004_real64
      s(6) = s(1) - 0.0005_real64
      s(7) = s(1) - 0.0006_real64
      s(8) = s(1) - 0.0007_real64
    case default
      s(1) = x
      s(2) = 10.0_real64 - x
      s(3) = 9.0_real64 - x
      s(4) = 9.0_real64 - s(2)
      s(5) = 8.0_real64 - x
      s(6) = 8.0_real64 - s(2)
      s(7) = 7.0_real64 - x
      s(8) = 7.0_real64 - s(2)
    end select
  end subroutine init_lanes_dp

  pure subroutine apply_ops_sp(family, lanes, s, v1, v2)
    integer, intent(in) :: family, lanes
    real(real32), intent(inout) :: s(8)
    real(real32), intent(in) :: v1, v2
    integer :: i
    do i = 1, lanes
      select case (family)
      case (1)
        s(i) = v1 - s(i)
      case (2)
        s(i) = s(i) * s(i) * v1
      case (3)
        s(i) = v1 - s(i) * v2
      case default
        s(i) = (v1 - v2*s(i)) * s(i)
      end select
    end do
  end subroutine apply_ops_sp

  pure subroutine apply_ops_dp(family, lanes, s, v1, v2)
    integer, intent(in) :: family, lanes
    real(real64), intent(inout) :: s(8)
    real(real64), intent(in) :: v1, v2
    integer :: i
    do i = 1, lanes
      select case (family)
      case (1)
        s(i) = v1 - s(i)
      case (2)
        s(i) = s(i) * s(i) * v1
      case (3)
        s(i) = v1 - s(i) * v2
      case default
        s(i) = (v1 - v2*s(i)) * s(i)
      end select
    end do
  end subroutine apply_ops_dp

#define RK real32
#define ADD1 add1_sp
#define ADD2 add2_sp
#define ADD4 add4_sp
#define ADD8 add8_sp
#define MUL1 mul1_sp
#define MUL2 mul2_sp
#define MUL4 mul4_sp
#define MUL8 mul8_sp
#define MADD1 madd1_sp
#define MADD2 madd2_sp
#define MADD4 madd4_sp
#define MADD8 madd8_sp
#define MULMADD1 mulmadd1_sp
#define MULMADD2 mulmadd2_sp
#define MULMADD4 mulmadd4_sp
#define MULMADD8 mulmadd8_sp
#include "maxflops_kernel_template.inc"
#undef MULMADD8
#undef MULMADD4
#undef MULMADD2
#undef MULMADD1
#undef MADD8
#undef MADD4
#undef MADD2
#undef MADD1
#undef MUL8
#undef MUL4
#undef MUL2
#undef MUL1
#undef ADD8
#undef ADD4
#undef ADD2
#undef ADD1
#undef RK
#define RK real64
#define ADD1 add1_dp
#define ADD2 add2_dp
#define ADD4 add4_dp
#define ADD8 add8_dp
#define MUL1 mul1_dp
#define MUL2 mul2_dp
#define MUL4 mul4_dp
#define MUL8 mul8_dp
#define MADD1 madd1_dp
#define MADD2 madd2_dp
#define MADD4 madd4_dp
#define MADD8 madd8_dp
#define MULMADD1 mulmadd1_dp
#define MULMADD2 mulmadd2_dp
#define MULMADD4 mulmadd4_dp
#define MULMADD8 mulmadd8_dp
#include "maxflops_kernel_template.inc"
#undef MULMADD8
#undef MULMADD4
#undef MULMADD2
#undef MULMADD1
#undef MADD8
#undef MADD4
#undef MADD2
#undef MADD1
#undef MUL8
#undef MUL4
#undef MUL2
#undef MUL1
#undef ADD8
#undef ADD4
#undef ADD2
#undef ADD1
#undef RK
end module maxflops_kernels

program main
  use iso_fortran_env, only: real32, real64
  use omp_lib
  use c_drand48
  use maxflops_kernels
  implicit none
  integer, parameter :: num_floats = 2*1024*1024
  integer :: repeat
  character(len=64) :: arg

  if (command_argument_count() /= 1) then
    print '(a)', 'Usage: ./main <repeat>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*) repeat
  print '(a)', '=== Single-precision floating-point kernels ==='
  call test_sp(repeat, num_floats)
  print '(a)', '=== Double-precision floating-point kernels ==='
  call test_dp(repeat, num_floats)

contains
  subroutine test_sp(repeat, nfloats)
    integer, intent(in) :: repeat, nfloats
    real(real32), allocatable :: host_mem(:)
    integer :: j
    allocate(host_mem(0:nfloats-1))
    call srand48(int(123, c_long))
    do j = 0, nfloats/2-1
      host_mem(j) = real(drand48()*10.0d0, real32)
      host_mem(nfloats-j-1) = host_mem(j)
    end do
!$omp target data map(alloc:host_mem(0:nfloats-1))
    call warm_sp(host_mem, nfloats, repeat)
    call timed_sp('Add1', 1, 1, 10.0_real32, 0.0_real32, host_mem, nfloats, repeat, .true.)
    call timed_sp('Add2', 1, 2, 10.0_real32, 0.0_real32, host_mem, nfloats, repeat, .true.)
    call timed_sp('Add4', 1, 4, 10.0_real32, 0.0_real32, host_mem, nfloats, repeat, .true.)
    call timed_sp('Add8', 1, 8, 10.0_real32, 0.0_real32, host_mem, nfloats, repeat, .true.)
    call warm_mul_sp(host_mem, nfloats, repeat)
    call timed_sp('Mul1', 2, 1, 1.01_real32, 0.0_real32, host_mem, nfloats, repeat, .false.)
    call timed_sp('Mul2', 2, 2, 1.01_real32, 0.0_real32, host_mem, nfloats, repeat, .true.)
    call timed_sp('Mul4', 2, 4, 1.01_real32, 0.0_real32, host_mem, nfloats, repeat, .true.)
    call timed_sp('Mul8', 2, 8, 1.01_real32, 0.0_real32, host_mem, nfloats, repeat, .true.)
    call warm_madd_sp(host_mem, nfloats, repeat)
    call timed_sp('MAdd1', 3, 1, 10.0_real32, 0.9899_real32, host_mem, nfloats, repeat, .true.)
    call timed_sp('MAdd2', 3, 2, 10.0_real32, 0.9899_real32, host_mem, nfloats, repeat, .true.)
    call timed_sp('MAdd4', 3, 4, 10.0_real32, 0.9899_real32, host_mem, nfloats, repeat, .true.)
    call timed_sp('MAdd8', 3, 8, 10.0_real32, 0.9899_real32, host_mem, nfloats, repeat, .true.)
    call warm_mulmadd_sp(host_mem, nfloats, repeat)
    call timed_sp('MulMAdd1', 4, 1, 3.75_real32, 0.355_real32, host_mem, nfloats, repeat, .true.)
    call timed_sp('MulMAdd2', 4, 2, 3.75_real32, 0.355_real32, host_mem, nfloats, repeat, .true.)
    call timed_sp('MulMAdd4', 4, 4, 3.75_real32, 0.355_real32, host_mem, nfloats, repeat, .true.)
    call timed_sp('MulMAdd8', 4, 8, 3.75_real32, 0.355_real32, host_mem, nfloats, repeat, .true.)
!$omp end target data
  end subroutine test_sp

  subroutine timed_sp(label, family, lanes, v1, v2, data, nfloats, repeat, update_first)
    character(len=*), intent(in) :: label
    integer, intent(in) :: family, lanes, nfloats, repeat
    real(real32), intent(in) :: v1, v2
    real(real32), intent(inout) :: data(0:)
    logical, intent(in) :: update_first
    real(8) :: t0, t1
    if (update_first) then
!$omp target update to(data(0:nfloats-1))
    end if
    t0 = omp_get_wtime()
    call kernel_sp(data, nfloats, repeat, family, lanes, v1, v2)
    t1 = omp_get_wtime()
    print '(a,a,a,f0.6,a)', 'kernel execution time (', trim(label), '): ', t1-t0, ' (s)'
  end subroutine timed_sp

  subroutine warm_sp(data, nfloats, repeat)
    real(real32), intent(inout) :: data(0:)
    integer, intent(in) :: nfloats, repeat
    integer :: i
    do i = 1, 4
      call kernel_sp(data,nfloats,repeat,1,1,10.0_real32,0.0_real32); call kernel_sp(data,nfloats,repeat,1,2,10.0_real32,0.0_real32)
      call kernel_sp(data,nfloats,repeat,1,4,10.0_real32,0.0_real32); call kernel_sp(data,nfloats,repeat,1,8,10.0_real32,0.0_real32)
    end do
  end subroutine warm_sp
  subroutine warm_mul_sp(data,nfloats,repeat)
    real(real32), intent(inout) :: data(0:); integer,intent(in)::nfloats,repeat; integer::i
    do i=1,4; call kernel_sp(data,nfloats,repeat,2,1,1.01_real32,0.0_real32); call kernel_sp(data,nfloats,repeat,2,2,1.01_real32,0.0_real32); call kernel_sp(data,nfloats,repeat,2,4,1.01_real32,0.0_real32); call kernel_sp(data,nfloats,repeat,2,8,1.01_real32,0.0_real32); end do
  end subroutine warm_mul_sp
  subroutine warm_madd_sp(data,nfloats,repeat)
    real(real32), intent(inout) :: data(0:); integer,intent(in)::nfloats,repeat; integer::i
    do i=1,4; call kernel_sp(data,nfloats,repeat,3,1,10.0_real32,0.9899_real32); call kernel_sp(data,nfloats,repeat,3,2,10.0_real32,0.9899_real32); call kernel_sp(data,nfloats,repeat,3,4,10.0_real32,0.9899_real32); call kernel_sp(data,nfloats,repeat,3,8,10.0_real32,0.9899_real32); end do
  end subroutine warm_madd_sp
  subroutine warm_mulmadd_sp(data,nfloats,repeat)
    real(real32), intent(inout) :: data(0:); integer,intent(in)::nfloats,repeat; integer::i
    do i=1,4; call kernel_sp(data,nfloats,repeat,4,1,3.75_real32,0.355_real32); call kernel_sp(data,nfloats,repeat,4,2,3.75_real32,0.355_real32); call kernel_sp(data,nfloats,repeat,4,4,3.75_real32,0.355_real32); call kernel_sp(data,nfloats,repeat,4,8,3.75_real32,0.355_real32); end do
  end subroutine warm_mulmadd_sp

  subroutine test_dp(repeat, nfloats)
    integer, intent(in) :: repeat, nfloats
    real(real64), allocatable :: host_mem(:)
    integer :: j
    allocate(host_mem(0:nfloats-1))
    call srand48(int(123, c_long))
    do j = 0, nfloats/2-1
      host_mem(j) = drand48()*10.0d0
      host_mem(nfloats-j-1) = host_mem(j)
    end do
!$omp target data map(alloc:host_mem(0:nfloats-1))
    call warm_dp(host_mem, nfloats, repeat)
    call timed_dp('Add1', 1, 1, 10.0_real64, 0.0_real64, host_mem, nfloats, repeat, .true.)
    call timed_dp('Add2', 1, 2, 10.0_real64, 0.0_real64, host_mem, nfloats, repeat, .true.)
    call timed_dp('Add4', 1, 4, 10.0_real64, 0.0_real64, host_mem, nfloats, repeat, .true.)
    call timed_dp('Add8', 1, 8, 10.0_real64, 0.0_real64, host_mem, nfloats, repeat, .true.)
    call warm_mul_dp(host_mem, nfloats, repeat)
    call timed_dp('Mul1', 2, 1, 1.01_real64, 0.0_real64, host_mem, nfloats, repeat, .false.)
    call timed_dp('Mul2', 2, 2, 1.01_real64, 0.0_real64, host_mem, nfloats, repeat, .true.)
    call timed_dp('Mul4', 2, 4, 1.01_real64, 0.0_real64, host_mem, nfloats, repeat, .true.)
    call timed_dp('Mul8', 2, 8, 1.01_real64, 0.0_real64, host_mem, nfloats, repeat, .true.)
    call warm_madd_dp(host_mem, nfloats, repeat)
    call timed_dp('MAdd1', 3, 1, 10.0_real64, 0.9899_real64, host_mem, nfloats, repeat, .true.)
    call timed_dp('MAdd2', 3, 2, 10.0_real64, 0.9899_real64, host_mem, nfloats, repeat, .true.)
    call timed_dp('MAdd4', 3, 4, 10.0_real64, 0.9899_real64, host_mem, nfloats, repeat, .true.)
    call timed_dp('MAdd8', 3, 8, 10.0_real64, 0.9899_real64, host_mem, nfloats, repeat, .true.)
    call warm_mulmadd_dp(host_mem, nfloats, repeat)
    call timed_dp('MulMAdd1', 4, 1, 3.75_real64, 0.355_real64, host_mem, nfloats, repeat, .true.)
    call timed_dp('MulMAdd2', 4, 2, 3.75_real64, 0.355_real64, host_mem, nfloats, repeat, .true.)
    call timed_dp('MulMAdd4', 4, 4, 3.75_real64, 0.355_real64, host_mem, nfloats, repeat, .true.)
    call timed_dp('MulMAdd8', 4, 8, 3.75_real64, 0.355_real64, host_mem, nfloats, repeat, .true.)
!$omp end target data
  end subroutine test_dp

  subroutine timed_dp(label, family, lanes, v1, v2, data, nfloats, repeat, update_first)
    character(len=*), intent(in) :: label
    integer, intent(in) :: family, lanes, nfloats, repeat
    real(real64), intent(in) :: v1, v2
    real(real64), intent(inout) :: data(0:)
    logical, intent(in) :: update_first
    real(8) :: t0, t1
    if (update_first) then
!$omp target update to(data(0:nfloats-1))
    end if
    t0 = omp_get_wtime()
    call kernel_dp(data, nfloats, repeat, family, lanes, v1, v2)
    t1 = omp_get_wtime()
    print '(a,a,a,f0.6,a)', 'kernel execution time (', trim(label), '): ', t1-t0, ' (s)'
  end subroutine timed_dp
  subroutine warm_dp(data,nfloats,repeat)
    real(real64), intent(inout) :: data(0:); integer,intent(in)::nfloats,repeat; integer::i
    do i=1,4; call kernel_dp(data,nfloats,repeat,1,1,10.0_real64,0.0_real64); call kernel_dp(data,nfloats,repeat,1,2,10.0_real64,0.0_real64); call kernel_dp(data,nfloats,repeat,1,4,10.0_real64,0.0_real64); call kernel_dp(data,nfloats,repeat,1,8,10.0_real64,0.0_real64); end do
  end subroutine warm_dp
  subroutine warm_mul_dp(data,nfloats,repeat)
    real(real64), intent(inout) :: data(0:); integer,intent(in)::nfloats,repeat; integer::i
    do i=1,4; call kernel_dp(data,nfloats,repeat,2,1,1.01_real64,0.0_real64); call kernel_dp(data,nfloats,repeat,2,2,1.01_real64,0.0_real64); call kernel_dp(data,nfloats,repeat,2,4,1.01_real64,0.0_real64); call kernel_dp(data,nfloats,repeat,2,8,1.01_real64,0.0_real64); end do
  end subroutine warm_mul_dp
  subroutine warm_madd_dp(data,nfloats,repeat)
    real(real64), intent(inout) :: data(0:); integer,intent(in)::nfloats,repeat; integer::i
    do i=1,4; call kernel_dp(data,nfloats,repeat,3,1,10.0_real64,0.9899_real64); call kernel_dp(data,nfloats,repeat,3,2,10.0_real64,0.9899_real64); call kernel_dp(data,nfloats,repeat,3,4,10.0_real64,0.9899_real64); call kernel_dp(data,nfloats,repeat,3,8,10.0_real64,0.9899_real64); end do
  end subroutine warm_madd_dp
  subroutine warm_mulmadd_dp(data,nfloats,repeat)
    real(real64), intent(inout) :: data(0:); integer,intent(in)::nfloats,repeat; integer::i
    do i=1,4; call kernel_dp(data,nfloats,repeat,4,1,3.75_real64,0.355_real64); call kernel_dp(data,nfloats,repeat,4,2,3.75_real64,0.355_real64); call kernel_dp(data,nfloats,repeat,4,4,3.75_real64,0.355_real64); call kernel_dp(data,nfloats,repeat,4,8,3.75_real64,0.355_real64); end do
  end subroutine warm_mulmadd_dp
end program main
