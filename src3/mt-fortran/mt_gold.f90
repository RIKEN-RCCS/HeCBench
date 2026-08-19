module mt_gold_mod
  use mt_types
  use mt_kernel_mod, only: box_muller_trans
  implicit none
contains
  subroutine random_ref(h_mt, h_rand, n_per_rng, seed)
    type(mt_struct_stripped), intent(in) :: h_mt(0:mt_rng_count-1)
    real(real32), intent(out) :: h_rand(0:mt_rng_count*n_per_rng-1)
    integer, intent(in) :: n_per_rng
    integer(int32), intent(in) :: seed
    type(mt_struct_stripped) :: mtcopy(0:mt_rng_count-1)
    real(real32), allocatable :: transposed(:)
    integer :: i, j
    mtcopy = h_mt
    mtcopy%seed = seed
    allocate(transposed(0:mt_rng_count*n_per_rng-1))
    call mersenne_host(mtcopy, transposed, n_per_rng)
    do i = 0, mt_rng_count-1
      do j = 0, n_per_rng-1
        h_rand(i*n_per_rng+j) = transposed(i + j*mt_rng_count)
      end do
    end do
    do i = 0, mt_rng_count*n_per_rng-1, 2
      call box_muller_trans(h_rand(i), h_rand(i+1))
    end do
  end subroutine

  subroutine mersenne_host(h_mt, h_rand_gpu, n_per_rng)
    type(mt_struct_stripped), intent(in) :: h_mt(0:mt_rng_count-1)
    real(real32), intent(out) :: h_rand_gpu(0:mt_rng_count*n_per_rng-1)
    integer, intent(in) :: n_per_rng
    integer :: global_id, istate, istate1, istatem, iout
    integer(int32) :: mti, mti1, mtim, x, mt(0:mt_nn-1), matrix_a, mask_b, mask_c
    do global_id = 0, mt_rng_count-1
      matrix_a = h_mt(global_id)%matrix_a; mask_b = h_mt(global_id)%mask_b; mask_c = h_mt(global_id)%mask_c
      mt(0) = h_mt(global_id)%seed
      do istate = 1, mt_nn-1
        mt(istate) = iand(int(1812433253_int64 * int(ieor(mt(istate-1), shiftr(mt(istate-1),30)),int64) + istate, int32), mt_wmask)
      end do
      istate = 0; mti1 = mt(0)
      do iout = 0, n_per_rng-1
        istate1 = istate + 1; istatem = istate + mt_mm
        if (istate1 >= mt_nn) istate1 = istate1 - mt_nn
        if (istatem >= mt_nn) istatem = istatem - mt_nn
        mti = mti1; mti1 = mt(istate1); mtim = mt(istatem)
        x = ior(iand(mti,mt_umask), iand(mti1,mt_lmask))
        x = ieor(ieor(mtim, shiftr(x,1)), merge(matrix_a,0_int32,iand(x,1_int32)/=0))
        mt(istate) = x; istate = istate1
        x = ieor(x, shiftr(x,mt_shift0)); x = ieor(x, iand(ishft(x,mt_shiftb),mask_b))
        x = ieor(x, iand(ishft(x,mt_shiftc),mask_c)); x = ieor(x, shiftr(x,mt_shift1))
        h_rand_gpu(global_id + iout*mt_rng_count) = (real(iand(int(x,int64), int(z'FFFFFFFF',int64)),real32)+1.0_real32)/4294967296.0_real32
      end do
    end do
  end subroutine
end module
