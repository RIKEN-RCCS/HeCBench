module mt_kernel_mod
  use mt_types
  implicit none
contains
  subroutine box_muller_trans(u1, u2)
    real(real32), intent(inout) :: u1, u2
    real(real32) :: r, phi
    r = sqrt(-2.0_real32 * log(u1))
    phi = 2.0_real32 * pi * u2
    u1 = r * cos(phi)
    u2 = r * sin(phi)
  end subroutine

  subroutine mersenne_kernel(h_mt, h_rand_gpu, n_per_rng, global_work_size, local_work_size)
    type(mt_struct_stripped), intent(in) :: h_mt(0:mt_rng_count-1)
    real(real32), intent(out) :: h_rand_gpu(0:mt_rng_count*n_per_rng-1)
    integer, intent(in) :: n_per_rng, global_work_size, local_work_size
    integer :: global_id, istate, istate1, istatem, iout
    integer(int32) :: mti, mti1, mtim, x, mt(0:mt_nn-1), matrix_a, mask_b, mask_c
    !$omp target teams distribute parallel do private(istate,istate1,istatem,iout,mti,mti1,mtim,x,mt,matrix_a,mask_b,mask_c) thread_limit(local_work_size)
    do global_id = 0, global_work_size-1
      matrix_a = h_mt(global_id)%matrix_a
      mask_b = h_mt(global_id)%mask_b
      mask_c = h_mt(global_id)%mask_c
      mt(0) = h_mt(global_id)%seed
      do istate = 1, mt_nn-1
        mt(istate) = iand(int(1812433253_int64 * int(ieor(mt(istate-1), shiftr(mt(istate-1),30)),int64) + istate, int32), mt_wmask)
      end do
      istate = 0
      mti1 = mt(0)
      do iout = 0, n_per_rng-1
        istate1 = istate + 1
        istatem = istate + mt_mm
        if (istate1 >= mt_nn) istate1 = istate1 - mt_nn
        if (istatem >= mt_nn) istatem = istatem - mt_nn
        mti = mti1
        mti1 = mt(istate1)
        mtim = mt(istatem)
        x = ior(iand(mti, mt_umask), iand(mti1, mt_lmask))
        x = ieor(ieor(mtim, shiftr(x, 1)), merge(matrix_a, 0_int32, iand(x,1_int32) /= 0))
        mt(istate) = x
        istate = istate1
        x = ieor(x, shiftr(x, mt_shift0))
        x = ieor(x, iand(ishft(x, mt_shiftb), mask_b))
        x = ieor(x, iand(ishft(x, mt_shiftc), mask_c))
        x = ieor(x, shiftr(x, mt_shift1))
        h_rand_gpu(global_id + iout * mt_rng_count) = (real(iand(int(x,int64), int(z'FFFFFFFF',int64)), real32) + 1.0_real32) / 4294967296.0_real32
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine

  subroutine box_muller_kernel(h_rand_gpu, n_per_rng, global_work_size, local_work_size)
    real(real32), intent(inout) :: h_rand_gpu(0:mt_rng_count*n_per_rng-1)
    integer, intent(in) :: n_per_rng, global_work_size, local_work_size
    integer :: global_id, iout
    !$omp target teams distribute parallel do private(iout) thread_limit(local_work_size)
    do global_id = 0, global_work_size-1
      do iout = 0, n_per_rng-1, 2
        call box_muller_trans(h_rand_gpu(global_id + iout*mt_rng_count), h_rand_gpu(global_id + (iout+1)*mt_rng_count))
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine
end module
