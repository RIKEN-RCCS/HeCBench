module genmtrand_mod
  use mt_types
  implicit none
contains
  subroutine load_mt_gpu(fname, seed, h_mt)
    character(len=*), intent(in) :: fname
    integer(int32), intent(in) :: seed
    type(mt_struct_stripped), intent(out) :: h_mt(0:mt_rng_count-1)
    integer :: u, i
    open(newunit=u, file=trim(fname), access='stream', form='unformatted', status='old', action='read')
    do i = 0, mt_rng_count-1
      read(u) h_mt(i)%matrix_a
      read(u) h_mt(i)%mask_b
      read(u) h_mt(i)%mask_c
      read(u) h_mt(i)%seed
    end do
    close(u)
    do i = 0, mt_rng_count-1
      h_mt(i)%seed = seed
    end do
  end subroutine
end module
