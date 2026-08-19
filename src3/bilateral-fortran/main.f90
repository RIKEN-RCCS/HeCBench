program bilateral
  use iso_c_binding, only: c_int
  use omp_lib
  implicit none
  interface
    subroutine c_srand(seed) bind(C,name='srand'); import c_int; integer(c_int),value :: seed; end subroutine
    function c_rand() bind(C,name='rand') result(value); import c_int; integer(c_int)::value; end function
  end interface
  integer :: w,h,img_size,repeat,i,ios
  real :: variance_i,variance_spatial,a_square,t0,t1
  character(len=64)::arg
  real,allocatable::src(:),dst(:),ref(:)
  logical::ok
  if(command_argument_count()/=5) then; print *,'Usage: ./main <image width> <image height> <intensity> <spatial> <repeat>'; stop 1; end if
  call get_command_argument(1,arg); read(arg,*,iostat=ios)w
  call get_command_argument(2,arg); read(arg,*,iostat=ios)h
  call get_command_argument(3,arg); read(arg,*,iostat=ios)variance_i
  call get_command_argument(4,arg); read(arg,*,iostat=ios)variance_spatial
  call get_command_argument(5,arg); read(arg,*,iostat=ios)repeat
  if(ios/=0 .or. w<=0 .or. h<=0 .or. repeat<=0) stop 1
  img_size=w*h; a_square=0.5/(variance_i*acos(-1.0))
  allocate(src(0:img_size-1),dst(0:img_size-1),ref(0:img_size-1))
  call c_srand(123_c_int); do i=0,img_size-1; src(i)=real(modulo(c_rand(),256_c_int)); end do
  !$omp target data map(to:src(0:img_size-1)) map(alloc:dst(0:img_size-1))
  t0=omp_get_wtime(); do i=1,repeat; call filter(3,src,dst,w,h,a_square,variance_i,variance_spatial); end do; t1=omp_get_wtime()
  print '(a,f12.6,a)','Average kernel execution time (3x3) ',(t1-t0)*1.e3/repeat,' (ms)'
  !$omp target update from(dst(0:img_size-1))
  call filter_cpu(3,src,ref,w,h,a_square,variance_i,variance_spatial); ok=all(abs(ref-dst)<=1.e-3)
  t0=omp_get_wtime(); do i=1,repeat; call filter(6,src,dst,w,h,a_square,variance_i,variance_spatial); end do; t1=omp_get_wtime()
  print '(a,f12.6,a)','Average kernel execution time (6x6) ',(t1-t0)*1.e3/repeat,' (ms)'
  !$omp target update from(dst(0:img_size-1))
  call filter_cpu(6,src,ref,w,h,a_square,variance_i,variance_spatial); ok=ok .and. all(abs(ref-dst)<=1.e-3)
  t0=omp_get_wtime(); do i=1,repeat; call filter(9,src,dst,w,h,a_square,variance_i,variance_spatial); end do; t1=omp_get_wtime()
  print '(a,f12.6,a)','Average kernel execution time (9x9) ',(t1-t0)*1.e3/repeat,' (ms)'
  !$omp target update from(dst(0:img_size-1))
  call filter_cpu(9,src,ref,w,h,a_square,variance_i,variance_spatial); ok=ok .and. all(abs(ref-dst)<=1.e-3)
  if(ok) then; print *,'PASS'; else; print *,'FAIL'; end if
  !$omp end target data
contains
  subroutine filter(r,input,output,w,h,a,vi,vs)
    integer,intent(in)::r,w,h; real,intent(in)::input(0:),a,vi,vs; real,intent(out)::output(0:)
    integer::idx,idy,id,idk,idl,idw,ii,jj; real::intensity,res,norm,iw,range,spatial,weight
    !$omp target teams distribute parallel do collapse(2) thread_limit(256) private(id,idk,idl,idw,ii,jj,intensity,res,norm,iw,range,spatial,weight)
    do idy=0,h-1
      do idx=0,w-1
        id=idy*w+idx; intensity=input(id); res=0.; norm=0.
        do ii=-r,r
          do jj=-r,r
            idk=idx+ii; idl=idy+jj
            if(idk<0)idk=-idk; if(idl<0)idl=-idl
            if(idk>w-1)idk=w-1-ii; if(idl>h-1)idl=h-1-jj
            idw=idl*w+idk; iw=input(idw)
            range=-(intensity-iw)*(intensity-iw)/(2.*vi); spatial=-real((idk-idx)*(idk-idx)+(idl-idy)*(idl-idy))/(2.*vs)
            weight=a*exp(spatial+range); norm=norm+weight; res=res+iw*weight
          end do
        end do
        output(id)=res/norm
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine
  subroutine filter_cpu(r,input,output,w,h,a,vi,vs)
    integer,intent(in)::r,w,h; real,intent(in)::input(0:),a,vi,vs; real,intent(out)::output(0:)
    integer::idx,idy,id,idk,idl,idw,ii,jj; real::intensity,res,norm,iw,range,spatial,weight
    do idy=0,h-1; do idx=0,w-1
      id=idy*w+idx; intensity=input(id); res=0.; norm=0.
      do ii=-r,r; do jj=-r,r
        idk=idx+ii; idl=idy+jj; if(idk<0)idk=-idk; if(idl<0)idl=-idl; if(idk>w-1)idk=w-1-ii; if(idl>h-1)idl=h-1-jj
        idw=idl*w+idk; iw=input(idw); range=-(intensity-iw)*(intensity-iw)/(2.*vi); spatial=-real((idk-idx)*(idk-idx)+(idl-idy)*(idl-idy))/(2.*vs)
        weight=a*exp(spatial+range); norm=norm+weight; res=res+iw*weight
      end do; end do; output(id)=res/norm
    end do; end do
  end subroutine
end program
