module hmm_module
  use, intrinsic :: iso_fortran_env, only : int32, int64, real32, real64
  use omp_lib
  implicit none
  integer(int64), parameter :: modulus32 = 4294967296_int64, rand_max = 2147483647_int64
  integer(int64) :: random_state(0:30), random_index
contains
  subroutine c_srand_default()
    integer :: i
    random_state(0) = 1_int64
    do i = 1, 30
      random_state(i) = modulo(16807_int64 * random_state(i-1), rand_max)
    end do
    random_index = 0_int64
    ! glibc's TYPE_3 generator warms its degree-31 state for ten full cycles.
    do i = 1, 310
      call advance_random()
    end do
  end subroutine c_srand_default

  subroutine advance_random()
    integer :: current, previous3
    current = int(modulo(random_index, 31_int64))
    previous3 = int(modulo(random_index + 3_int64, 31_int64))
    random_state(current) = modulo(random_state(current) + random_state(previous3), modulus32)
    random_index = random_index + 1_int64
  end subroutine advance_random

  function c_rand() result(value)
    integer(int64) :: value
    call advance_random()
    value = random_state(int(modulo(random_index - 1_int64, 31_int64))) / 2_int64
  end function c_rand

  subroutine init_hmm(init_prob, mt_state, mt_emit, n_state, n_emit)
    integer, intent(in) :: n_state, n_emit
    real(real32), intent(out) :: init_prob(0:), mt_state(0:), mt_emit(0:)
    integer :: i, j
    real(real32) :: total
    call c_srand_default()
    do i = 0, n_state - 1
      init_prob(i) = real(c_rand(), real32)
    end do
    total = 0.0_real32
    do i = 0, n_state - 1; total = total + init_prob(i); end do
    do i = 0, n_state - 1; init_prob(i) = init_prob(i) / total; end do
    do i = 0, n_state - 1
      do j = 0, n_state - 1
        mt_state(i*n_state+j) = real(c_rand(), real32) / real(rand_max, real32)
      end do
    end do
    do i = 0, n_emit - 1
      do j = 0, n_state - 1
        mt_emit(i*n_state+j) = real(c_rand(), real32)
      end do
    end do
    do j = 0, n_state - 1
      total = 0.0_real32
      do i = 0, n_emit - 1; total = total + mt_emit(i*n_state+j); end do
      do i = 0, n_emit - 1; mt_emit(i*n_state+j) = mt_emit(i*n_state+j) / total; end do
    end do
  end subroutine init_hmm

  subroutine viterbi_cpu(viterbi_prob, viterbi_path, obs, n_obs, init_prob, mt_state, n_state, mt_emit)
    integer, intent(in) :: n_obs, n_state, obs(0:)
    integer, intent(out) :: viterbi_path(0:)
    real(real32), intent(out) :: viterbi_prob
    real(real32), intent(in) :: init_prob(0:), mt_state(0:), mt_emit(0:)
    real(real32), allocatable :: max_new(:), max_old(:)
    integer, allocatable :: path(:)
    integer :: t, i_state, pre_state, max_state
    real(real32) :: max_prob, p
    allocate(max_new(0:n_state-1), max_old(0:n_state-1), path(0:(n_obs-1)*n_state-1))
!$omp parallel do
    do i_state = 0, n_state - 1; max_old(i_state) = init_prob(i_state); end do
!$omp end parallel do
    do t = 1, n_obs - 1
!$omp parallel do private(max_prob,max_state,pre_state,p)
      do i_state = 0, n_state - 1
        max_prob = 0.0_real32; max_state = -1
        do pre_state = 0, n_state - 1
          p = max_old(pre_state) + mt_state(i_state*n_state+pre_state)
          if (p > max_prob) then; max_prob = p; max_state = pre_state; end if
        end do
        max_new(i_state) = max_prob + mt_emit(obs(t)*n_state+i_state)
        path((t-1)*n_state+i_state) = max_state
      end do
!$omp end parallel do
!$omp parallel do
      do i_state = 0, n_state - 1; max_old(i_state) = max_new(i_state); end do
!$omp end parallel do
    end do
    max_prob = 0.0_real32; max_state = -1
    do i_state = 0, n_state - 1
      if (max_new(i_state) > max_prob) then; max_prob = max_new(i_state); max_state = i_state; end if
    end do
    viterbi_prob = max_prob; viterbi_path(n_obs-1) = max_state
    do t = n_obs - 2, 0, -1; viterbi_path(t) = path(t*n_state+viterbi_path(t+1)); end do
  end subroutine viterbi_cpu

  subroutine viterbi_gpu(viterbi_prob, viterbi_path, obs, n_obs, init_prob, mt_state, n_state, n_emit, mt_emit)
    integer, intent(in) :: n_obs, n_state, n_emit, obs(0:)
    integer, intent(out) :: viterbi_path(0:)
    real(real32), intent(out) :: viterbi_prob
    real(real32), intent(inout) :: init_prob(0:)
    real(real32), intent(in) :: mt_state(0:), mt_emit(0:)
    real(real32), allocatable :: max_new(:)
    integer, allocatable :: path(:)
    integer :: t, i_state, pre_state, max_state
    real(real32) :: max_prob, p
    real(real64) :: start_time, end_time
    allocate(max_new(0:n_state-1), path(0:(n_obs-1)*n_state-1))
!$omp target data map(to:init_prob(0:n_state),mt_state(0:n_state*n_state),mt_emit(0:n_emit*n_state),obs(0:n_obs)) map(from:max_new(0:n_state),path(0:(n_obs-1)*n_state))
    start_time = omp_get_wtime()
    do t = 1, n_obs - 1
!$omp target teams distribute parallel do thread_limit(256) private(max_prob,max_state,pre_state,p)
      do i_state = 0, n_state - 1
        max_prob = 0.0_real32; max_state = -1
        do pre_state = 0, n_state - 1
          p = init_prob(pre_state) + mt_state(i_state*n_state+pre_state)
          if (p > max_prob) then; max_prob = p; max_state = pre_state; end if
        end do
        max_new(i_state) = max_prob + mt_emit(obs(t)*n_state+i_state)
        path((t-1)*n_state+i_state) = max_state
      end do
!$omp end target teams distribute parallel do
!$omp target teams distribute parallel do thread_limit(256)
      do i_state = 0, n_state - 1; init_prob(i_state) = max_new(i_state); end do
!$omp end target teams distribute parallel do
    end do
    end_time = omp_get_wtime()
    print '(a,f0.6,a)', 'Device execution time of Viterbi iterations ', real((end_time-start_time),real32), ' (s)'
!$omp end target data
    max_prob = 0.0_real32; max_state = -1
    do i_state = 0, n_state - 1
      if (max_new(i_state) > max_prob) then; max_prob = max_new(i_state); max_state = i_state; end if
    end do
    viterbi_prob = max_prob; viterbi_path(n_obs-1) = max_state
    do t = n_obs - 2, 0, -1; viterbi_path(t) = path(t*n_state+viterbi_path(t+1)); end do
  end subroutine viterbi_gpu
end module hmm_module

program hidden_markov_model
  use, intrinsic :: iso_fortran_env, only : real32
  use hmm_module
  implicit none
  integer, parameter :: n_state=4096, n_emit=4096, n_obs=500
  integer :: i, error
  integer, allocatable :: obs(:), path_cpu(:), path_gpu(:)
  real(real32), allocatable :: init_prob(:), mt_state(:), mt_emit(:)
  real(real32) :: prob_cpu, prob_gpu
  allocate(init_prob(0:n_state-1),mt_state(0:n_state*n_state-1),mt_emit(0:n_emit*n_state-1),obs(0:n_obs-1),path_cpu(0:n_obs-1),path_gpu(0:n_obs-1))
  call init_hmm(init_prob, mt_state, mt_emit, n_state, n_emit)
  do i=0,n_obs-1; obs(i)=modulo(i,15); end do
  print '(a,i0)', '# of states = ', n_state
  print '(a,i0)', '# of possible observations = ', n_emit
  print '(a,i0,a)', 'Size of observational sequence = ', n_obs, achar(10)
  print '(a)', achar(10)//'Compute Viterbi path on GPU'
  call viterbi_gpu(prob_gpu,path_gpu,obs,n_obs,init_prob,mt_state,n_state,n_emit,mt_emit)
  print '(a)', achar(10)//'Compute Viterbi path on CPU'
  call init_hmm(init_prob, mt_state, mt_emit, n_state, n_emit)
  ! The C++ GPU function mutates the mapped initProb device copy only; restore the host input before CPU execution.
  call viterbi_cpu(prob_cpu,path_cpu,obs,n_obs,init_prob,mt_state,n_state,mt_emit)
  error=0
  do i=0,n_obs-1
    if(path_cpu(i)/=path_gpu(i)) then; error=1;exit;end if
  end do
  if(error==0) then; print '(a)','Success'; else; print '(a)','Fail'; end if
end program hidden_markov_model
