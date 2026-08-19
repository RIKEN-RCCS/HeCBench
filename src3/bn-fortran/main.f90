module bn_support
  use, intrinsic :: iso_c_binding, only : c_int
  use, intrinsic :: iso_fortran_env, only : real32, real64
  use omp_lib
  implicit none

  integer, parameter :: node_n = 45, state_n = 2, data_n = 600
  integer, parameter :: highest = 3, iter = 100, workload = 1
  integer :: sizepernode
  real(real32) :: prescore = -99999999999.0_real32, score = 0.0_real32
  real(real32) :: maxscore(0:highest-1) = -999999999.0_real32
  logical :: orders(0:node_n-1,0:node_n-1), preorders(0:node_n-1,0:node_n-1)
  logical :: pregraph(0:node_n-1,0:node_n-1), graph(0:node_n-1,0:node_n-1)
  logical :: bestgraph(0:highest-1,0:node_n-1,0:node_n-1)
  integer :: data(0:data_n*node_n-1)
  real(real32), allocatable :: localscore(:), scores(:), lg(:)
  integer, allocatable :: parents(:)

  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand
    function c_rand() bind(C, name="rand") result(value)
      import :: c_int
      integer(c_int) :: value
    end function c_rand
  end interface

  !$omp declare target (d_incr, d_incrs, d_getstate, d_findcomb, d_findindex, d_comb)

contains

  integer function rand_mod(modulus) result(value)
    integer, intent(in) :: modulus
    value = modulo(int(c_rand()), modulus)
  end function rand_mod

  integer function comb(n, a) result(res)
    integer, intent(in) :: n, a
    integer :: i, nn
    res = 1
    nn = n
    do i = 0, a - 1
      res = res * nn
      nn = nn - 1
    end do
    do i = 0, a - 1
      res = res / (a - i)
    end do
  end function comb

  subroutine load_data()
    integer :: unit, ios, i
    open(newunit=unit, file="data45.dat", status="old", action="read", iostat=ios)
    if (ios /= 0) error stop "Error: failed to open data45.dat"
    do i = 0, data_n * node_n - 1
      read(unit, *, iostat=ios) data(i)
      if (ios /= 0) error stop "Error: invalid data45.dat"
    end do
    close(unit)
  end subroutine load_data

  subroutine initial()
    integer :: i, j, tmp, a, b, r, total
    logical :: tmpd
    tmp = 1
    do i = 1, 4
      tmp = tmp + comb(node_n - 1, i)
    end do
    sizepernode = tmp
    total = tmp * node_n
    allocate(localscore(0:total-1))
    localscore = 0.0_real32
    orders = .false.
    do i = 0, node_n - 1
      do j = 0, i - 1
        orders(i,j) = .true.
      end do
    end do
    r = rand_mod(10000)
    do i = 0, r - 1
      a = rand_mod(node_n)
      b = rand_mod(node_n)
      do j = 0, node_n - 1
        tmpd = orders(j,a); orders(j,a) = orders(j,b); orders(j,b) = tmpd
      end do
      do j = 0, node_n - 1
        tmpd = orders(a,j); orders(a,j) = orders(b,j); orders(b,j) = tmpd
      end do
    end do
    preorders = orders
  end subroutine initial

  integer function gen_orders() result(result_value)
    integer :: a, b, j
    logical :: tmp
    a = rand_mod(node_n)
    b = rand_mod(node_n)
    do j = 0, node_n - 1
      tmp = orders(a,j); orders(a,j) = orders(b,j); orders(b,j) = tmp
    end do
    do j = 0, node_n - 1
      tmp = orders(j,a); orders(j,a) = orders(j,b); orders(j,b) = tmp
    end do
    result_value = 1
  end function gen_orders

  integer function con_core() result(result_value)
    real(real32) :: tmp
    tmp = log(real(rand_mod(100000), real32) / 100000.0_real32)
    if (tmp < score - prescore) then
      preorders = orders
      pregraph = graph
      prescore = score
      result_value = 1
    else
      result_value = 0
    end if
  end function con_core

  subroutine pre_log_gamma()
    integer :: i
    allocate(lg(0:data_n+1))
    lg = 0.0_real32
    lg(1) = log(1.0_real32)
    do i = 2, data_n + 1
      lg(i) = lg(i-1) + log(real(i, real32))
    end do
  end subroutine pre_log_gamma

  integer function d_comb(n, a) result(res)
    integer, intent(in) :: n, a
    integer :: i, nn
    res = 1; nn = n
    do i = 0, a - 1
      res = res * nn; nn = nn - 1
    end do
    do i = 0, a - 1
      res = res / (a - i)
    end do
  end function d_comb

  subroutine d_incr(bit, n)
    integer, intent(inout) :: bit(0:4)
    integer, intent(in) :: n
    integer :: position
    position = n
    do while (position <= 4)
      bit(position) = bit(position) + 1
      if (bit(position) >= 2) then
        bit(position) = 0
        position = position + 1
      else
        exit
      end if
    end do
  end subroutine d_incr

  subroutine d_incrs(bit, n)
    integer, intent(inout) :: bit(0:4)
    integer, intent(in) :: n
    bit(n) = bit(n) + 1
    if (bit(n) >= state_n) then
      bit(n) = 0
      call d_incr(bit, n + 1)
    end if
  end subroutine d_incrs

  logical function d_getstate(parn, state, time) result(value)
    integer, intent(in) :: parn, time
    integer, intent(inout) :: state(0:4)
    integer :: i, maximum
    maximum = 1
    do i = 0, parn - 1
      maximum = maximum * state_n
    end do
    maximum = maximum - 1
    if (time > maximum) then
      value = .false.
      return
    end if
    if (time >= 1) call d_incrs(state, 0)
    value = .true.
  end function d_getstate

  subroutine d_findcomb(comb_array, l_input, n_input)
    integer, intent(out) :: comb_array(0:4)
    integer, intent(in) :: l_input, n_input
    integer :: l, n, sum, k, low, pos, s, i
    if (l_input == 0) then
      comb_array(0:3) = -1
      return
    end if
    l = l_input; n = n_input; sum = 0; k = 1
    do while (sum < l)
      sum = sum + d_comb(n, k)
      k = k + 1
    end do
    k = k - 1
    l = l - (sum - d_comb(n, k))
    low = 0; pos = 0
    do while (k > 1)
      sum = 0; s = 1
      do while (sum < l)
        sum = sum + d_comb(n-s, k-1)
        s = s + 1
      end do
      s = s - 1
      k = k - 1
      l = l - (sum - d_comb(n-s, k))
      low = low + s
      comb_array(pos) = low
      pos = pos + 1
      n = n - s
    end do
    comb_array(pos) = low + l
    do i = pos + 1, 3
      comb_array(i) = -1
    end do
  end subroutine d_findcomb

  integer function d_findindex(arr, size) result(index)
    integer, intent(in) :: arr(0:4), size
    integer :: i, j
    index = 0
    do i = 1, size - 1
      index = index + d_comb(node_n-1, i)
    end do
    do i = 1, size - 1
      do j = arr(i-1) + 1, arr(i) - 1
        index = index + d_comb(node_n-1-j, size-i)
      end do
    end do
    index = index + arr(size) - arr(size-1)
  end function d_findindex
  subroutine gen_score_kernel(size_per_node, d_localscore, d_data, d_lg)
    integer, intent(in) :: size_per_node
    real(real32), intent(inout) :: d_localscore(0:)
    integer, intent(in) :: d_data(0:)
    real(real32), intent(in) :: d_lg(0:)
    integer :: id, node, index, parent(0:4), pre(0:node_n-1), state(0:4), nij(0:state_n-1)
    integer :: i, j, parn, tmp, t, t1, t2
    logical :: flag
    real(real32) :: ls
!$omp target teams distribute parallel do thread_limit(256) private(id)
    do id = 0, size_per_node * node_n - 1
      d_localscore(id) = 0.0_real32
    end do
!$omp end target teams distribute parallel do
!$omp target teams distribute parallel do thread_limit(256) private(node,index,parent,pre,state,nij,i,j,parn,tmp,t,t1,t2,flag,ls)
    do id = 0, size_per_node - 1
      parent = 0; pre = 0; state = 0; nij = 0; parn = 0
      call d_findcomb(parent, id, node_n-1)
      do i = 0, 3
        if (parent(i) > 0) parn = parn + 1
      end do
      do node = 0, node_n - 1
        j = 1
        do i = 0, node_n - 1
          if (i /= node) then
            pre(j) = i
            j = j + 1
          end if
        end do
        do tmp = 0, parn - 1
          state(tmp) = 0
        end do
        index = size_per_node * node + id
        t = 0
        do while (d_getstate(parn, state, t))
          t = t + 1
          ls = 0.0_real32; nij = 0
          do t1 = 0, data_n - 1
            flag = .true.
            do t2 = 0, parn - 1
              if (d_data(t1*node_n+pre(parent(t2))) /= state(t2)) then
                flag = .false.; exit
              end if
            end do
            if (.not. flag) cycle
            nij(d_data(t1*node_n+node)) = nij(d_data(t1*node_n+node)) + 1
          end do
          tmp = state_n - 1
          do t1 = 0, state_n - 1
            ls = ls + d_lg(nij(t1)); tmp = tmp + nij(t1)
          end do
          ls = ls - d_lg(tmp) + d_lg(state_n-1)
          d_localscore(index) = d_localscore(index) + ls
        end do
      end do
    end do
!$omp end target teams distribute parallel do
!$omp target update from(d_localscore(0:node_n*size_per_node-1))
  end subroutine gen_score_kernel

  subroutine compute_kernel(taskperthr, size_per_node, d_localscore, d_parent, node, total, d_score, d_resp, blocknum)
    integer, intent(in) :: taskperthr, size_per_node, node, total, blocknum
    real(real32), intent(in) :: d_localscore(0:)
    logical, intent(in) :: d_parent(0:)
    real(real32), intent(inout) :: d_score(0:)
    integer, intent(inout) :: d_resp(0:)
    real(real32) :: lsinblock(0:255)
    integer :: tid, bid, id, posn, i, index, t, tmp, parn, stride
    integer :: pre(0:node_n-1), bestparent(0:3), parent(0:4)
    real(real32) :: bestls, ls
!$omp target teams num_teams(blocknum) thread_limit(256) private(lsinblock)
!$omp parallel private(tid,bid,id,posn,i,index,t,tmp,parn,stride,pre,bestparent,parent,bestls,ls) shared(lsinblock)
    tid = omp_get_thread_num(); bid = omp_get_team_num(); id = bid * 256 + tid
    posn = 1; pre = 0; bestparent = 0; parent = -1; parn = 0; bestls = -999999999999999.0_real32
    do i = 0, node_n - 1
      if (d_parent(i)) then
        pre(posn) = i; posn = posn + 1
      end if
    end do
    do i = 0, taskperthr - 1
      if (id*taskperthr+i >= total) exit
      call d_findcomb(parent, id*taskperthr+i, posn)
      do parn = 0, 3
        if (parent(parn) < 0) exit
        if (pre(parent(parn)) > node) then
          parent(parn) = pre(parent(parn))
        else
          parent(parn) = pre(parent(parn)) + 1
        end if
      end do
      do tmp = parn, 1, -1
        parent(tmp) = parent(tmp-1)
      end do
      parent(0) = 0
      index = d_findindex(parent, parn) + size_per_node * node
      ls = d_localscore(index)
      if (ls > bestls) then
        bestls = ls
        do tmp = 0, 3
          bestparent(tmp) = parent(tmp+1)
        end do
      end if
    end do
    lsinblock(tid) = bestls
!$omp barrier
    stride = 128
    do while (stride >= 1)
      if (tid < stride) then
        if (lsinblock(tid+stride) > lsinblock(tid) .and. lsinblock(tid+stride) < 0.0_real32) then
          lsinblock(tid) = lsinblock(tid+stride); lsinblock(tid+stride) = real(tid+stride,real32)
        else if (lsinblock(tid+stride) < lsinblock(tid) .and. lsinblock(tid) < 0.0_real32) then
          lsinblock(tid+stride) = real(tid,real32)
        else if (lsinblock(tid) > 0.0_real32 .and. lsinblock(tid+stride) < 0.0_real32) then
          lsinblock(tid) = lsinblock(tid+stride); lsinblock(tid+stride) = real(tid+stride,real32)
        else if (lsinblock(tid) < 0.0_real32 .and. lsinblock(tid+stride) > 0.0_real32) then
          lsinblock(tid+stride) = real(tid,real32)
        end if
      end if
!$omp barrier
      stride = stride / 2
    end do
!$omp barrier
    if (tid == 0) then
      d_score(bid) = lsinblock(0); t = 0
      do i = 0, 6
        if (t >= 128 .or. t < 0) exit
        t = int(lsinblock(2**i+t))
      end do
      lsinblock(0) = real(t,real32)
    end if
!$omp barrier
    if (tid == int(lsinblock(0))) then
      do i = 0, 3
        d_resp(bid*4+i) = bestparent(i)
      end do
    end if
!$omp end parallel
!$omp end target teams
  end subroutine compute_kernel

  real(real32) function find_best_graph(d_localscore, d_resp, d_score, d_parent) result(total_score)
    real(real32), intent(in) :: d_localscore(0:)
    integer, intent(inout) :: d_resp(0:)
    real(real32), intent(inout) :: d_score(0:)
    logical, intent(inout) :: d_parent(0:)
    real(real32) :: bestls
    integer :: bestparent(0:4), bestpn, total, node, pre(0:node_n-1), parent(0:node_n-1)
    integer :: posn, i, parn, tmp, blocknum, size_per_node
    graph = .false.; total_score = 0.0_real32
    do node = 0, node_n - 1
      bestls = -99999999.0_real32; posn = 0; pre = 0; parent = 0; bestparent = 0
      do i = 0, node_n - 1
        if (orders(node,i)) then; pre(posn) = i; posn = posn + 1; end if
      end do
      total = comb(posn,4)+comb(posn,3)+comb(posn,2)+posn+1
      blocknum = total/(256*workload)+1; size_per_node = sizepernode
!$omp target teams distribute parallel do thread_limit(256)
      do i = 0, blocknum*4-1
        d_resp(i) = 0
      end do
!$omp end target teams distribute parallel do
!$omp target teams distribute parallel do thread_limit(256)
      do i = 0, blocknum-1
        d_score(i) = -999999.0_real32
      end do
!$omp end target teams distribute parallel do
      d_parent(0:node_n-1) = orders(node,0:node_n-1)
!$omp target update to(d_parent(0:node_n-1))
      call compute_kernel(workload, size_per_node, d_localscore, d_parent, node, total, d_score, d_resp, blocknum)
!$omp target update from(d_resp(0:blocknum*4-1))
      parents(0:blocknum*4-1) = d_resp(0:blocknum*4-1)
!$omp target update from(d_score(0:blocknum-1))
      do i = 0, blocknum-1
        if (d_score(i) > bestls) then
          bestls = d_score(i); parn = 0
          do tmp = 0, 3
            if (parents(i*4+tmp) < 0) exit
            bestparent(tmp) = parents(i*4+tmp); parn = parn + 1
          end do
          bestpn = parn
        end if
      end do
      if (bestls > -99999999.0_real32) then
        do i = 0, bestpn-1
          if (bestparent(i) < node) then
            graph(node,bestparent(i)-1) = .true.
          else
            graph(node,bestparent(i)) = .true.
          end if
        end do
        total_score = total_score + bestls
      end if
    end do
  end function find_best_graph

  subroutine sort_graph()
    real(real32) :: maximum, tmp
    integer :: maxi, i, j
    do j = 0, highest - 2
      maximum = maxscore(j); maxi = j
      do i = j+1, highest-1
        if (maxscore(i) > maximum) then; maximum = maxscore(i); maxi = i; end if
      end do
      call swap_graph(j, maxi)
      tmp = maxscore(j); maxscore(j) = maximum; maxscore(maxi) = tmp
    end do
  end subroutine sort_graph

  subroutine swap_graph(a, b)
    integer, intent(in) :: a, b
    logical :: tmp
    integer :: i, j
    do i = 0, node_n-1; do j = 0, node_n-1
      tmp = bestgraph(a,i,j); bestgraph(a,i,j) = bestgraph(b,i,j); bestgraph(b,i,j) = tmp
    end do; end do
  end subroutine swap_graph

end module bn_support

program bn
  use, intrinsic :: iso_fortran_env, only : real32, real64, int64
  use bn_support
  implicit none
  character(len=1024) :: output_path, argument
  integer :: argc, repeat, i, j, a, b, c, tmp
  integer(int64) :: clock_rate, start_count, end_count, find_start, find_end, find_rate
  real(real32) :: tmpd
  real(real64) :: elapsed, find_time
  logical, allocatable :: d_parent(:)
  integer, allocatable :: d_resp(:)

  argc = command_argument_count()
  if (argc /= 2) then
    print '(A)', 'Usage: ./main <path to output file> <repeat>'
    stop 1
  end if
  call get_command_argument(1, output_path)
  call get_command_argument(2, argument); read(argument, *) repeat
  open(unit=10, file=trim(output_path), status='replace', action='write', iostat=tmp)
  if (tmp /= 0) error stop 'Error: failed to open output file. Exit..'
  print '(A,I0,A)', 'NODE_N=', node_n, new_line('a')//'Initialization...'
  call c_srand(2_c_int); call load_data(); call initial(); call pre_log_gamma()
  allocate(scores(0:sizepernode/(256*workload)), parents(0:(sizepernode/(256*workload)+1)*4-1))
  allocate(d_parent(0:node_n-1), d_resp(0:(sizepernode/(256*workload)+1)*4-1))
!$omp target data map(to:data(0:node_n*data_n-1),lg(0:data_n+1)) &
!$omp& map(alloc:localscore(0:node_n*sizepernode-1),scores(0:sizepernode/(256*workload)), &
!$omp& d_parent(0:node_n-1),d_resp(0:(sizepernode/(256*workload)+1)*4-1))
  call system_clock(start_count, clock_rate)
  do i = 0, repeat - 1
    call gen_score_kernel(sizepernode, localscore, data, lg)
  end do
  call system_clock(end_count)
  elapsed = real(end_count-start_count,real64)/real(clock_rate,real64)
  print '(A,F0.6,A)', 'Average execution time of genScoreKernel: ', elapsed/real(repeat,real64), ' (s)'
  find_time = 0.0_real64; c = 0; i = 0
  do while (i /= iter)
    i = i + 1; score = 0.0_real32; orders = preorders
    tmp = rand_mod(6)
    do j = 0, tmp-1
      tmp = gen_orders()
    end do
    call system_clock(find_start, find_rate)
    score = find_best_graph(localscore, d_resp, scores, d_parent)
    call system_clock(find_end)
    find_time = find_time + real(find_end-find_start,real64)/real(find_rate,real64)
    tmp = con_core()
    if (c < highest) then
      tmp = 1
      do j = 0, c-1
        if (maxscore(j) == prescore) tmp = 0
      end do
      if (tmp /= 0) then
        maxscore(c) = prescore; bestgraph(c,:,:) = pregraph; c = c + 1
      end if
    else if (c == highest) then
      call sort_graph(); c = c + 1
    else
      tmp = 1
      do j = 0, highest-1
        if (maxscore(j) == prescore) then; tmp = 0; exit; end if
      end do
      if (tmp /= 0 .and. prescore > maxscore(highest-1)) then
        maxscore(highest-1) = prescore; bestgraph(highest-1,:,:) = pregraph; b = highest-1
        do a = highest-2, 0, -1
          if (maxscore(b) > maxscore(a)) then
            call swap_graph(a,b); tmpd = maxscore(a); maxscore(a) = maxscore(b); maxscore(b) = tmpd; b = a
          end if
        end do
      end if
    end if
  end do
  print '(A,F0.6,A)', 'Find best graph time ', find_time, ' (s)'
!$omp end target data
  do j = 0, highest-1
    write(10,'(A,F0.6)') 'score:', maxscore(j)
    write(10,'(A)') 'Best Graph:'
    do a = 0, node_n-1
      do b = 0, node_n-1
        write(10,'(I0,1X)',advance='no') merge(1,0,bestgraph(j,a,b))
      end do
      write(10,*)
    end do
    write(10,'(A)') '--------------------------------------------------------------------'
  end do
  close(10)
end program bn
