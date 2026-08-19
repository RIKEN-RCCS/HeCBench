module convolution_kernels
  use, intrinsic :: iso_fortran_env, only : real32, real64, int16
  use omp_lib
  implicit none
  integer, parameter :: max_mask_width = 10, block_size = 256, tile_size = block_size

  interface conv_basic
    module procedure conv_basic_r64, conv_basic_r32, conv_basic_i16
  end interface
  interface conv_tiled
    module procedure conv_tiled_r64, conv_tiled_r32, conv_tiled_i16
  end interface
  interface conv_cached
    module procedure conv_cached_r64, conv_cached_r32, conv_cached_i16
  end interface
  interface check_result
    module procedure check_r64, check_r32, check_i16
  end interface

contains

  subroutine conv_basic_r64(mask, input, output, width, mask_width)
    integer, intent(in) :: width, mask_width
    real(real64), intent(in) :: mask(0:max_mask_width-1), input(0:width-1)
    real(real64), intent(out) :: output(0:width-1)
    integer :: i, j, start
    real(real64) :: sum
!$omp target teams distribute parallel do num_threads(block_size) private(sum,start,j)
    do i = 0, width - 1
      sum = 0.0_real64; start = i - mask_width / 2
      do j = 0, mask_width - 1
        if (start+j >= 0 .and. start+j < width) sum = sum + input(start+j)*mask(j)
      end do
      output(i) = sum
    end do
!$omp end target teams distribute parallel do
  end subroutine

  subroutine conv_tiled_r64(mask, input, output, width, mask_width)
    integer, intent(in) :: width, mask_width
    real(real64), intent(in) :: mask(0:max_mask_width-1), input(0:width-1)
    real(real64), intent(out) :: output(0:width-1)
    real(real64) :: tile(0:tile_size+max_mask_width-2), sum
    integer :: bid, lid, dim, i, n, halo_left, halo_right, j
!$omp target teams num_teams(width/block_size) thread_limit(block_size) private(tile)
!$omp parallel shared(mask,input,output,tile) private(bid,lid,dim,i,n,halo_left,halo_right,j,sum)
    bid = omp_get_team_num(); lid = omp_get_thread_num(); dim = omp_get_num_threads()
    i = bid*dim + lid; n = mask_width/2
    halo_left = (bid-1)*dim + lid
    if (lid >= dim-n) then
      if (halo_left < 0) then; tile(lid-(dim-n)) = 0.0_real64
      else; tile(lid-(dim-n)) = input(halo_left); end if
    end if
    tile(n+lid) = input(bid*dim+lid)
    halo_right = (bid+1)*dim+lid
    if (lid < n) then
      if (halo_right >= width) then; tile(lid+dim+n) = 0.0_real64
      else; tile(lid+dim+n) = input(halo_right); end if
    end if
!$omp barrier
    sum = 0.0_real64
    do j=0,mask_width-1; sum=sum+tile(lid+j)*mask(j); end do
    output(i)=sum
!$omp end parallel
!$omp end target teams
  end subroutine

  subroutine conv_cached_r64(mask, input, output, width, mask_width)
    integer, intent(in) :: width, mask_width
    real(real64), intent(in) :: mask(0:max_mask_width-1), input(0:width-1)
    real(real64), intent(out) :: output(0:width-1)
    real(real64) :: tile(0:tile_size-1), sum
    integer :: bid,lid,dim,i,this_tile_start,next_tile_start,start,j,in_index
!$omp target teams num_teams(width/block_size) thread_limit(block_size) private(tile)
!$omp parallel shared(mask,input,output,tile) private(bid,lid,dim,i,this_tile_start,next_tile_start,start,j,in_index,sum)
    bid=omp_get_team_num(); lid=omp_get_thread_num(); dim=omp_get_num_threads(); i=bid*dim+lid
    tile(lid)=input(i)
!$omp barrier
    this_tile_start=bid*dim; next_tile_start=(bid+1)*dim; start=i-mask_width/2; sum=0.0_real64
    do j=0,mask_width-1
      in_index=start+j
      if (in_index >= 0 .and. in_index < width) then
        if (in_index >= this_tile_start .and. in_index < next_tile_start) then
          sum=sum+tile(lid+j-mask_width/2)*mask(j)
        else
          sum=sum+input(in_index)*mask(j)
        end if
      end if
    end do
    output(i)=sum
!$omp end parallel
!$omp end target teams
  end subroutine

  subroutine check_r64(input, output, mask, width, mask_width)
    integer,intent(in)::width,mask_width
    real(real64),intent(in)::input(0:width-1),output(0:width-1),mask(0:max_mask_width-1)
    integer::i,j,start; real(real64)::sum
    do i=0,width-1
      sum=0.0_real64; start=i-mask_width/2
      do j=0,mask_width-1; if(start+j>=0 .and. start+j<width) sum=sum+input(start+j)*mask(j); end do
      if(abs(sum-output(i))>1.0e-3_real64) then; print '(a)','FAIL'; return; end if
    end do
    print '(a)','PASS'
  end subroutine

  subroutine conv_basic_r32(mask, input, output, width, mask_width)
    integer,intent(in)::width,mask_width
    real(real32),intent(in)::mask(0:max_mask_width-1),input(0:width-1)
    real(real32),intent(out)::output(0:width-1)
    integer::i,j,start; real(real32)::sum
!$omp target teams distribute parallel do num_threads(block_size) private(sum,start,j)
    do i=0,width-1
      sum=0.0_real32; start=i-mask_width/2
      do j=0,mask_width-1; if(start+j>=0 .and. start+j<width) sum=sum+input(start+j)*mask(j); end do
      output(i)=sum
    end do
!$omp end target teams distribute parallel do
  end subroutine

  subroutine conv_tiled_r32(mask,input,output,width,mask_width)
    integer,intent(in)::width,mask_width
    real(real32),intent(in)::mask(0:max_mask_width-1),input(0:width-1)
    real(real32),intent(out)::output(0:width-1)
    real(real32)::tile(0:tile_size+max_mask_width-2),sum
    integer::bid,lid,dim,i,n,halo_left,halo_right,j
!$omp target teams num_teams(width/block_size) thread_limit(block_size) private(tile)
!$omp parallel shared(mask,input,output,tile) private(bid,lid,dim,i,n,halo_left,halo_right,j,sum)
    bid=omp_get_team_num(); lid=omp_get_thread_num(); dim=omp_get_num_threads(); i=bid*dim+lid; n=mask_width/2
    halo_left=(bid-1)*dim+lid
    if(lid>=dim-n) then; if(halo_left<0) then; tile(lid-(dim-n))=0.0_real32; else; tile(lid-(dim-n))=input(halo_left); end if; end if
    tile(n+lid)=input(bid*dim+lid); halo_right=(bid+1)*dim+lid
    if(lid<n) then; if(halo_right>=width) then; tile(lid+dim+n)=0.0_real32; else; tile(lid+dim+n)=input(halo_right); end if; end if
!$omp barrier
    sum=0.0_real32; do j=0,mask_width-1; sum=sum+tile(lid+j)*mask(j); end do; output(i)=sum
!$omp end parallel
!$omp end target teams
  end subroutine

  subroutine conv_cached_r32(mask,input,output,width,mask_width)
    integer,intent(in)::width,mask_width
    real(real32),intent(in)::mask(0:max_mask_width-1),input(0:width-1)
    real(real32),intent(out)::output(0:width-1)
    real(real32)::tile(0:tile_size-1),sum
    integer::bid,lid,dim,i,this_tile_start,next_tile_start,start,j,in_index
!$omp target teams num_teams(width/block_size) thread_limit(block_size) private(tile)
!$omp parallel shared(mask,input,output,tile) private(bid,lid,dim,i,this_tile_start,next_tile_start,start,j,in_index,sum)
    bid=omp_get_team_num(); lid=omp_get_thread_num(); dim=omp_get_num_threads(); i=bid*dim+lid; tile(lid)=input(i)
!$omp barrier
    this_tile_start=bid*dim; next_tile_start=(bid+1)*dim; start=i-mask_width/2; sum=0.0_real32
    do j=0,mask_width-1
      in_index=start+j
      if(in_index>=0 .and. in_index<width) then
        if(in_index>=this_tile_start .and. in_index<next_tile_start) then; sum=sum+tile(lid+j-mask_width/2)*mask(j)
        else; sum=sum+input(in_index)*mask(j); end if
      end if
    end do
    output(i)=sum
!$omp end parallel
!$omp end target teams
  end subroutine

  subroutine check_r32(input,output,mask,width,mask_width)
    integer,intent(in)::width,mask_width
    real(real32),intent(in)::input(0:width-1),output(0:width-1),mask(0:max_mask_width-1)
    integer::i,j,start; real(real32)::sum
    do i=0,width-1
      sum=0.0_real32; start=i-mask_width/2
      do j=0,mask_width-1; if(start+j>=0 .and. start+j<width) sum=sum+input(start+j)*mask(j); end do
      if(abs(sum-output(i))>1.0e-3_real32) then; print '(a)','FAIL'; return; end if
    end do
    print '(a)','PASS'
  end subroutine

  subroutine conv_basic_i16(mask,input,output,width,mask_width)
    integer,intent(in)::width,mask_width
    integer(int16),intent(in)::mask(0:max_mask_width-1),input(0:width-1)
    integer(int16),intent(out)::output(0:width-1)
    integer::i,j,start; integer(int16)::sum
!$omp target teams distribute parallel do num_threads(block_size) private(sum,start,j)
    do i=0,width-1
      sum=0_int16; start=i-mask_width/2
      do j=0,mask_width-1; if(start+j>=0 .and. start+j<width) sum=sum+input(start+j)*mask(j); end do
      output(i)=sum
    end do
!$omp end target teams distribute parallel do
  end subroutine

  subroutine conv_tiled_i16(mask,input,output,width,mask_width)
    integer,intent(in)::width,mask_width
    integer(int16),intent(in)::mask(0:max_mask_width-1),input(0:width-1)
    integer(int16),intent(out)::output(0:width-1)
    integer(int16)::tile(0:tile_size+max_mask_width-2),sum
    integer::bid,lid,dim,i,n,halo_left,halo_right,j
!$omp target teams num_teams(width/block_size) thread_limit(block_size) private(tile)
!$omp parallel shared(mask,input,output,tile) private(bid,lid,dim,i,n,halo_left,halo_right,j,sum)
    bid=omp_get_team_num(); lid=omp_get_thread_num(); dim=omp_get_num_threads(); i=bid*dim+lid; n=mask_width/2; halo_left=(bid-1)*dim+lid
    if(lid>=dim-n) then; if(halo_left<0) then; tile(lid-(dim-n))=0_int16; else; tile(lid-(dim-n))=input(halo_left); end if; end if
    tile(n+lid)=input(bid*dim+lid); halo_right=(bid+1)*dim+lid
    if(lid<n) then; if(halo_right>=width) then; tile(lid+dim+n)=0_int16; else; tile(lid+dim+n)=input(halo_right); end if; end if
!$omp barrier
    sum=0_int16; do j=0,mask_width-1; sum=sum+tile(lid+j)*mask(j); end do; output(i)=sum
!$omp end parallel
!$omp end target teams
  end subroutine

  subroutine conv_cached_i16(mask,input,output,width,mask_width)
    integer,intent(in)::width,mask_width
    integer(int16),intent(in)::mask(0:max_mask_width-1),input(0:width-1)
    integer(int16),intent(out)::output(0:width-1)
    integer(int16)::tile(0:tile_size-1),sum
    integer::bid,lid,dim,i,this_tile_start,next_tile_start,start,j,in_index
!$omp target teams num_teams(width/block_size) thread_limit(block_size) private(tile)
!$omp parallel shared(mask,input,output,tile) private(bid,lid,dim,i,this_tile_start,next_tile_start,start,j,in_index,sum)
    bid=omp_get_team_num(); lid=omp_get_thread_num(); dim=omp_get_num_threads(); i=bid*dim+lid; tile(lid)=input(i)
!$omp barrier
    this_tile_start=bid*dim; next_tile_start=(bid+1)*dim; start=i-mask_width/2; sum=0_int16
    do j=0,mask_width-1
      in_index=start+j
      if(in_index>=0 .and. in_index<width) then
        if(in_index>=this_tile_start .and. in_index<next_tile_start) then; sum=sum+tile(lid+j-mask_width/2)*mask(j)
        else; sum=sum+input(in_index)*mask(j); end if
      end if
    end do
    output(i)=sum
!$omp end parallel
!$omp end target teams
  end subroutine

  subroutine check_i16(input,output,mask,width,mask_width)
    integer,intent(in)::width,mask_width
    integer(int16),intent(in)::input(0:width-1),output(0:width-1),mask(0:max_mask_width-1)
    integer::i,j,start; integer(int16)::sum
    do i=0,width-1
      sum=0_int16; start=i-mask_width/2
      do j=0,mask_width-1; if(start+j>=0 .and. start+j<width) sum=sum+input(start+j)*mask(j); end do
      if(sum/=output(i)) then; print '(a)','FAIL'; return; end if
    end do
    print '(a)','PASS'
  end subroutine
end module convolution_kernels

program convolution1d
  use, intrinsic :: iso_fortran_env, only : real32, real64, int16
  use, intrinsic :: iso_c_binding, only : c_int
  use omp_lib
  use convolution_kernels
  implicit none
  interface
    subroutine c_srand(seed) bind(C,name='srand')
      import c_int
      integer(c_int), value :: seed
    end subroutine
    function c_rand() bind(C,name='rand') result(value)
      import c_int
      integer(c_int) :: value
    end function
  end interface
  integer :: argc, width, repeat, mask_width
  character(len=64) :: argument
  argc=command_argument_count()
  if(argc/=2) then
    print '(a)','Usage: ./main <input_width> <repeat>'; stop 1
  end if
  call get_command_argument(1,argument); read(argument,*) width
  call get_command_argument(2,argument); read(argument,*) repeat
  width=((width+block_size-1)/block_size)*block_size
  do mask_width=3,max_mask_width-1,2
    print '(a)',''
    print '(a)','---------------------'
    print '(a,i0)','Mask width: ',mask_width
    print '(a)','1D convolution (FP64)'; call run_r64(width,mask_width,repeat)
    print '(a)','1D convolution (FP32)'; call run_r32(width,mask_width,repeat)
    print '(a)','1D convolution (INT16)'; call run_i16(width,mask_width,repeat)
  end do
contains
  subroutine run_r64(n,mw,reps)
    integer,intent(in)::n,mw,reps; real(real64),allocatable::a(:),b(:); real(real64)::mask(0:max_mask_width-1),t0,t1; integer::i,r
    allocate(a(0:n-1),b(0:n-1)); mask=1.0_real64; call c_srand(123_c_int)
    do i=0,n-1; a(i)=real(modulo(c_rand(),256_c_int),real64); end do
!$omp target data map(to:a,mask) map(alloc:b)
    t0=omp_get_wtime(); do r=1,reps; call conv_basic(mask,a,b,n,mw); end do; t1=omp_get_wtime()
    print '(a,f0.6,a)','Average kernel execution time of conv1d kernel: ',(t1-t0)*1.0e6_real64/reps,' (us)'
!$omp target update from(b)
    call check_result(a,b,mask,n,mw)
    t0=omp_get_wtime(); do r=1,reps; call conv_tiled(mask,a,b,n,mw); end do; t1=omp_get_wtime()
    print '(a,f0.6,a)','Average kernel execution time of conv1d-tiled kernel: ',(t1-t0)*1.0e6_real64/reps,' (us)'
!$omp target update from(b)
    call check_result(a,b,mask,n,mw)
    t0=omp_get_wtime(); do r=1,reps; call conv_cached(mask,a,b,n,mw); end do; t1=omp_get_wtime()
    print '(a,f0.6,a)','Average kernel execution time of conv1d-tiled-caching kernel: ',(t1-t0)*1.0e6_real64/reps,' (us)'
!$omp target update from(b)
    call check_result(a,b,mask,n,mw)
!$omp end target data
    deallocate(a,b)
  end subroutine
  subroutine run_r32(n,mw,reps)
    integer,intent(in)::n,mw,reps; real(real32),allocatable::a(:),b(:); real(real32)::mask(0:max_mask_width-1); real(real64)::t0,t1; integer::i,r
    allocate(a(0:n-1),b(0:n-1)); mask=1.0_real32; call c_srand(123_c_int)
    do i=0,n-1; a(i)=real(modulo(c_rand(),256_c_int),real32); end do
!$omp target data map(to:a,mask) map(alloc:b)
    t0=omp_get_wtime(); do r=1,reps; call conv_basic(mask,a,b,n,mw); end do; t1=omp_get_wtime()
    print '(a,f0.6,a)','Average kernel execution time of conv1d kernel: ',(t1-t0)*1.0e6_real64/reps,' (us)'
!$omp target update from(b)
    call check_result(a,b,mask,n,mw)
    t0=omp_get_wtime(); do r=1,reps; call conv_tiled(mask,a,b,n,mw); end do; t1=omp_get_wtime()
    print '(a,f0.6,a)','Average kernel execution time of conv1d-tiled kernel: ',(t1-t0)*1.0e6_real64/reps,' (us)'
!$omp target update from(b)
    call check_result(a,b,mask,n,mw)
    t0=omp_get_wtime(); do r=1,reps; call conv_cached(mask,a,b,n,mw); end do; t1=omp_get_wtime()
    print '(a,f0.6,a)','Average kernel execution time of conv1d-tiled-caching kernel: ',(t1-t0)*1.0e6_real64/reps,' (us)'
!$omp target update from(b)
    call check_result(a,b,mask,n,mw)
!$omp end target data
    deallocate(a,b)
  end subroutine
  subroutine run_i16(n,mw,reps)
    integer,intent(in)::n,mw,reps; integer(int16),allocatable::a(:),b(:); integer(int16)::mask(0:max_mask_width-1); real(real64)::t0,t1; integer::i,r
    allocate(a(0:n-1),b(0:n-1)); mask=1_int16; call c_srand(123_c_int)
    do i=0,n-1; a(i)=int(modulo(c_rand(),256_c_int),int16); end do
!$omp target data map(to:a,mask) map(alloc:b)
    t0=omp_get_wtime(); do r=1,reps; call conv_basic(mask,a,b,n,mw); end do; t1=omp_get_wtime()
    print '(a,f0.6,a)','Average kernel execution time of conv1d kernel: ',(t1-t0)*1.0e6_real64/reps,' (us)'
!$omp target update from(b)
    call check_result(a,b,mask,n,mw)
    t0=omp_get_wtime(); do r=1,reps; call conv_tiled(mask,a,b,n,mw); end do; t1=omp_get_wtime()
    print '(a,f0.6,a)','Average kernel execution time of conv1d-tiled kernel: ',(t1-t0)*1.0e6_real64/reps,' (us)'
!$omp target update from(b)
    call check_result(a,b,mask,n,mw)
    t0=omp_get_wtime(); do r=1,reps; call conv_cached(mask,a,b,n,mw); end do; t1=omp_get_wtime()
    print '(a,f0.6,a)','Average kernel execution time of conv1d-tiled-caching kernel: ',(t1-t0)*1.0e6_real64/reps,' (us)'
!$omp target update from(b)
    call check_result(a,b,mask,n,mw)
!$omp end target data
    deallocate(a,b)
  end subroutine
end program convolution1d
