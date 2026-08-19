module nqueen_mod
  use iso_fortran_env, only: int8, int32, int64, real64
  use iso_c_binding, only: c_int8_t, c_int32_t
  implicit none
  integer, parameter :: queens_block_size = 128, empty = -1
  type, bind(C) :: queen_root
    integer(c_int32_t) :: control
    integer(c_int8_t) :: board(0:11)
  end type
!$omp declare target(queens_still_legal)
contains
  real(real64) function seconds() result(t)
    integer(int64) :: c, r
    call system_clock(c, r)
    t = real(c, real64) / real(r, real64)
  end function

  subroutine prefixes_handle_sol(root_prefixes, flag, board, initial_depth, num_sol)
    type(queen_root), intent(inout) :: root_prefixes(0:)
    integer(int32), intent(in) :: flag
    integer(int8), intent(in) :: board(0:*)
    integer, intent(in) :: initial_depth
    integer(int64), intent(in) :: num_sol
    integer :: i
    root_prefixes(num_sol)%control = flag
    do i = 0, initial_depth-1
      root_prefixes(num_sol)%board(i) = board(i)
    end do
  end subroutine

  logical function mc_still_legal(board, r) result(safe)
    integer(int8), intent(in) :: board(0:*)
    integer, intent(in) :: r
    integer :: i, ld, rd
    safe = .true.
    do i = 0, r-1
      if (board(i) == board(r)) safe = .false.
    end do
    ld = board(r); rd = board(r)
    do i = r-1, 0, -1
      ld = ld - 1; rd = rd + 1
      if (board(i) == ld .or. board(i) == rd) safe = .false.
    end do
  end function

  logical function queens_still_legal(board, r) result(safe)
    integer(int8), intent(in) :: board(0:)
    integer, intent(in) :: r
    integer :: i, ld, rd
    safe = .true.
    do i = 0, r-1
      if (board(i) == board(r)) safe = .false.
    end do
    ld = board(r); rd = board(r)
    do i = r-1, 0, -1
      ld = ld - 1; rd = rd + 1
      if (board(i) == ld .or. board(i) == rd) safe = .false.
    end do
  end function

  subroutine bp_queens_root_dfs(n, n_prefixes, depth_prefixes, root_prefixes, vector_tree_size, sols)
    integer, intent(in) :: n, depth_prefixes
    integer(int64), intent(in) :: n_prefixes
    type(queen_root), intent(in) :: root_prefixes(0:*)
    integer(int64), intent(out) :: vector_tree_size(0:*), sols(0:*)
    integer(int64) :: idx, qtd_solutions_thread, tree_size
    integer(int32) :: flag, bit_test
    integer(int8) :: vertice(0:19)
    integer :: i, depth, depth_global, n_l
    !$omp target teams distribute parallel do &
    !$omp& map(to:root_prefixes(0:n_prefixes-1)) &
    !$omp& map(from:vector_tree_size(0:n_prefixes-1),sols(0:n_prefixes-1)) &
    !$omp& private(flag,bit_test,vertice,n_l,i,depth,depth_global,qtd_solutions_thread,tree_size) &
    !$omp& thread_limit(queens_block_size)
    do idx = 0, n_prefixes-1
      n_l = n
      do i = 0, n_l-1
        vertice(i) = empty
      end do
      flag = root_prefixes(idx)%control
      depth_global = depth_prefixes
      do i = 0, depth_global-1
        vertice(i) = root_prefixes(idx)%board(i)
      end do
      depth = depth_global
      qtd_solutions_thread = 0_int64
      tree_size = 0_int64
      do
        vertice(depth) = vertice(depth) + 1
        bit_test = ishft(1_int32, int(vertice(depth)))
        if (vertice(depth) == n_l) then
          vertice(depth) = empty
        else if (iand(flag, bit_test) == 0 .and. queens_still_legal(vertice, depth)) then
          tree_size = tree_size + 1_int64
          flag = ior(flag, ishft(1_int32, int(vertice(depth))))
          depth = depth + 1
          if (depth == n_l) then
            qtd_solutions_thread = qtd_solutions_thread + 1_int64
          else
            cycle
          end if
        else
          cycle
        end if
        depth = depth - 1
        if (depth < depth_global) exit
        flag = iand(flag, not(ishft(1_int32, int(vertice(depth)))))
      end do
      sols(idx) = qtd_solutions_thread
      vector_tree_size(idx) = tree_size
    end do
    !$omp end target teams distribute parallel do
  end subroutine

  integer(int64) function bp_queens_prefixes(size, initial_depth, tree_size, root_prefixes) result(num_sol)
    integer, intent(in) :: size, initial_depth
    integer(int64), intent(out) :: tree_size
    type(queen_root), intent(inout) :: root_prefixes(0:)
    integer(int32) :: flag, bit_test
    integer(int8) :: vertice(0:19)
    integer :: i, nivel
    integer(int64) :: local_tree
    do i = 0, size-1
      vertice(i) = empty
    end do
    flag = 0; nivel = 0; local_tree = 0_int64; num_sol = 0_int64
    do
      vertice(nivel) = vertice(nivel) + 1
      bit_test = ishft(1_int32, int(vertice(nivel)))
      if (vertice(nivel) == size) then
        vertice(nivel) = empty
      else if (mc_still_legal(vertice, nivel) .and. iand(flag, bit_test) == 0) then
        flag = ior(flag, ishft(1_int32, int(vertice(nivel))))
        nivel = nivel + 1
        local_tree = local_tree + 1_int64
        if (nivel == initial_depth) then
          call prefixes_handle_sol(root_prefixes, flag, vertice, initial_depth, num_sol)
          num_sol = num_sol + 1_int64
        else
          cycle
        end if
      else
        cycle
      end if
      nivel = nivel - 1
      if (nivel < 0) exit
      flag = iand(flag, not(ishft(1_int32, int(vertice(nivel)))))
    end do
    tree_size = local_tree
  end function

  subroutine nqueens(size, initial_depth, n_explorers, root_prefixes, vector_tree_size, sols, repeat)
    integer, intent(in) :: size, initial_depth, repeat
    integer(int64), intent(in) :: n_explorers
    type(queen_root), intent(in) :: root_prefixes(0:n_explorers-1)
    integer(int64), intent(out) :: vector_tree_size(0:n_explorers-1), sols(0:n_explorers-1)
    integer :: i
    real(real64) :: t0, t1
    print '(a)', ''
    print '(a)', '### Regular BP-DFS search. ###'
    !$omp target data map(to:root_prefixes(0:n_explorers-1)) map(from:vector_tree_size(0:n_explorers-1),sols(0:n_explorers-1))
    t0 = seconds()
    do i = 1, repeat
      call bp_queens_root_dfs(size, n_explorers, initial_depth, root_prefixes, vector_tree_size, sols)
    end do
    t1 = seconds()
    print '(a,f10.6,a)', 'Average kernel execution time: ', (t1-t0)/real(repeat,real64), ' (s)'
    !$omp end target data
  end subroutine
end module

program main
  use nqueen_mod
  implicit none
  integer :: size, initial_depth, repeat, ios
  integer(int64) :: tree_size, qtd_sols_global, n_explorers, i
  integer, parameter :: n_max_prefixes = 75580635
  character(len=64) :: arg
  type(queen_root), allocatable :: root_prefixes(:)
  integer(int64), allocatable :: vector_tree_size(:), solutions(:)
  if (command_argument_count() /= 3) then
    print '(a)', 'Usage: main <size> <initial depth> <repeat>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*,iostat=ios) size
  call get_command_argument(2,arg); read(arg,*,iostat=ios) initial_depth
  call get_command_argument(3,arg); read(arg,*,iostat=ios) repeat
  print '(a,i0,a,i0,a)', '### Initial depth: ', initial_depth, ' - Size: ', size, ':'
  allocate(root_prefixes(0:n_max_prefixes-1), vector_tree_size(0:n_max_prefixes-1), solutions(0:n_max_prefixes-1))
  n_explorers = bp_queens_prefixes(size, initial_depth, tree_size, root_prefixes)
  call nqueens(size, initial_depth, n_explorers, root_prefixes, vector_tree_size, solutions, repeat)
  print '(a,i0)', 'Tree size: ', tree_size
  qtd_sols_global = 0_int64
  do i = 0, n_explorers-1
    if (solutions(i) > 0) qtd_sols_global = qtd_sols_global + solutions(i)
    if (vector_tree_size(i) > 0) tree_size = tree_size + vector_tree_size(i)
  end do
  print '(a,i0,a,i0)', 'Number of solutions found: ', qtd_sols_global, ' Tree size: ', tree_size
  if (size == 15 .and. initial_depth == 7) then
    if (qtd_sols_global == 2279184_int64 .and. tree_size == 171129071_int64) then
      print '(a)', 'PASS'
    else
      print '(a)', 'FAIL'
    end if
  end if
end program
