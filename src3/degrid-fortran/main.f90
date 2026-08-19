program degrid
  use, intrinsic :: iso_fortran_env, only : int32, real32, real64
  use, intrinsic :: iso_c_binding, only : c_int
  use omp_lib, only : omp_get_wtime
  implicit none

  integer(int32), parameter :: npoints=40000_int32, gcf_dim=256_int32, img_size=8192_int32, gcf_grid=8_int32, repeat=100_int32
  integer(int32), parameter :: image_offset=img_size*gcf_dim+gcf_dim
  integer(int32), parameter :: image_count=img_size*img_size+2_int32*image_offset
  integer(int32), parameter :: gcf_count=64_int32*gcf_dim*gcf_dim
  type :: precision2
    real(real64) :: x, y
  end type precision2
  type(precision2), allocatable :: out(:), input(:), image(:), gcf(:), out_cpu(:)
  integer(int32) :: x, y, n
  integer :: i
  real(real64) :: begin_time, elapsed
  logical :: ok

  interface
    subroutine c_srand(seed) bind(C,name='srand')
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand
    function c_rand() bind(C,name='rand') result(value)
      import :: c_int
      integer(c_int) :: value
    end function c_rand
  end interface
!$omp declare target (degrid_value)

  allocate(out(0:npoints-1), input(0:npoints-1), image(0:image_count-1), &
           gcf(0:gcf_count-1), out_cpu(0:npoints-1))
  write(*,'(a,i0)') 'img size in bytes: ', int(image_count,kind=8)*16_8
  write(*,'(a,i0)') 'out size in bytes: ', int(npoints,kind=8)*16_8
  call init_gcf(gcf, gcf_dim)
  call c_srand(2541617_c_int)
  do n=0,npoints-1
    input(n)%x=real(c_rand(),real64)/2147483647.0_real64*8000.0_real64
    input(n)%y=real(c_rand(),real64)/2147483647.0_real64*8000.0_real64
  end do
  image%x=0.0_real64; image%y=0.0_real64
  do y=0,img_size-1
    do x=0,img_size-1
      image(image_offset+x+img_size*y)%x=exp(-((real(x,real64)-1400.0_real64)**2 &
        +(real(y,real64)-3800.0_real64)**2)/8000000.0_real64)+1.0_real64
      image(image_offset+x+img_size*y)%y=0.4_real64
    end do
  end do
  call sort_by_subgrid(input)
  write(*,'(a)') 'Computing on GPU...'
  call degrid_gpu(out, input, image, gcf)
  write(*,'(a)') 'Computing on CPU...'
  call degrid_cpu(out_cpu, input, image, gcf)
  write(*,'(a)') 'Checking results against CPU:'
  write(*,'(a,es12.5)') 'Error bound: ', 1.0e-7_real64
  ok=.true.
  do n=0,npoints-1
    if (abs(out(n)%x-out_cpu(n)%x)>1.0e-7_real64 .or. abs(out(n)%y-out_cpu(n)%y)>1.0e-7_real64) then
      ok=.false.; write(*,'(i0,a,es20.12,a,es20.12,a,es20.12,a,es20.12)') &
        n, ': F(', input(n)%x, ', ', input(n)%y, ') = ', out(n)%x, ', ', out(n)%y
      exit
    end if
  end do
  if(ok) then; write(*,'(a)') 'PASS'; else; write(*,'(a)') 'FAIL'; end if
  deallocate(out,input,image,gcf,out_cpu)

contains

  subroutine init_gcf(table, size)
    type(precision2), intent(out) :: table(0:)
    integer(int32), intent(in) :: size
    integer(int32) :: sub_x,sub_y,x,y,index
    real(real64) :: tmp
    do sub_x=0,gcf_grid-1
      do sub_y=0,gcf_grid-1
        do x=0,size-1
          do y=0,size-1
            tmp=sin(6.28_real64*real(x,real64)/real(size*gcf_grid,real64)) &
              *exp(-(real(x*x+y*y*sub_y,real64))/real(size*size,real64)/2.0_real64)
            index=size*size*(sub_x+sub_y*gcf_grid)+x+y*size
            table(index)%x=tmp*sin(real(x*sub_x,real64)/real(y+1,real64))
            table(index)%y=tmp*cos(real(x*sub_x,real64)/real(y+1,real64))
          end do
        end do
      end do
    end do
  end subroutine init_gcf

  subroutine sort_by_subgrid(values)
    type(precision2), intent(inout) :: values(0:)
    type(precision2), allocatable :: temporary(:)
    integer(int32) :: counts(0:63), starts(0:63), next(0:63), key, n, sx, sy
    counts=0
    do n=0,npoints-1
      sx=int(real(gcf_grid,real32)*real(values(n)%x-floor(values(n)%x),real32),int32)
      sy=int(real(gcf_grid,real32)*real(values(n)%y-floor(values(n)%y),real32),int32)
      counts(sx+gcf_grid*sy)=counts(sx+gcf_grid*sy)+1
    end do
    starts(0)=0
    do key=1,63; starts(key)=starts(key-1)+counts(key-1); end do
    next=starts; allocate(temporary(0:npoints-1))
    do n=0,npoints-1
      sx=int(real(gcf_grid,real32)*real(values(n)%x-floor(values(n)%x),real32),int32)
      sy=int(real(gcf_grid,real32)*real(values(n)%y-floor(values(n)%y),real32),int32); key=sx+gcf_grid*sy
      temporary(next(key))=values(n); next(key)=next(key)+1
    end do
    values=temporary; deallocate(temporary)
  end subroutine sort_by_subgrid

  subroutine degrid_value(point, image, table, value)
    type(precision2), intent(in) :: point, image(0:), table(0:)
    type(precision2), intent(out) :: value
    integer(int32) :: sub_x,sub_y,main_x,main_y,a,b,table_index,image_index
    real(real64) :: sum_r,sum_i,r1,i1,r2,i2
    sub_x=int(real(gcf_grid,real32)*real(point%x-floor(point%x),real32),int32)
    sub_y=int(real(gcf_grid,real32)*real(point%y-floor(point%y),real32),int32)
    main_x=int(floor(point%x),int32); main_y=int(floor(point%y),int32)
    sum_r=0.0_real64;sum_i=0.0_real64
    do a=-gcf_dim/2,gcf_dim/2-1
      do b=-gcf_dim/2,gcf_dim/2-1
        image_index=image_offset+main_x+a+img_size*(main_y+b)
        table_index=gcf_dim*(gcf_dim+1)/2+gcf_dim*gcf_dim*(gcf_grid*sub_y+sub_x)+gcf_dim*b+a
        r1=image(image_index)%x;i1=image(image_index)%y;r2=table(table_index)%x;i2=table(table_index)%y
        if(main_x+a>=0 .and. main_y+b>=0 .and. main_x+a<img_size .and. main_y+b<img_size) then
          sum_r=sum_r+r1*r2-i1*i2;sum_i=sum_i+r1*i2+r2*i1
        end if
      end do
    end do
    value%x=sum_r;value%y=sum_i
  end subroutine degrid_value

  subroutine degrid_gpu(output, values, image, table)
    type(precision2), intent(out) :: output(0:)
    type(precision2), intent(in) :: values(0:), image(0:), table(0:)
    integer(int32) :: iteration,n,a,b,sub_x,sub_y,main_x,main_y,table_index,image_index
    real(real64) :: begin_time,elapsed
    real(real64) :: sum_r,sum_i,r1,i1,r2,i2
!$omp target data map(to:image(0:image_count-1),table(0:gcf_count-1),values(0:npoints-1)) map(from:output(0:npoints-1))
    begin_time=omp_get_wtime()
    do iteration=1,repeat
!$omp target teams distribute parallel do num_teams(npoints/32) thread_limit(256) &
!$omp& private(a,b,sub_x,sub_y,main_x,main_y,table_index,image_index,sum_r,sum_i,r1,i1,r2,i2)
      do n=0,npoints-1
        sub_x=int(real(gcf_grid,real32)*real(values(n)%x-floor(values(n)%x),real32),int32)
        sub_y=int(real(gcf_grid,real32)*real(values(n)%y-floor(values(n)%y),real32),int32)
        main_x=int(floor(values(n)%x),int32); main_y=int(floor(values(n)%y),int32)
        sum_r=0.0_real64; sum_i=0.0_real64
        do a=-gcf_dim/2,gcf_dim/2-1
          do b=-gcf_dim/2,gcf_dim/2-1
            image_index=image_offset+main_x+a+img_size*(main_y+b)
            table_index=gcf_dim*(gcf_dim+1)/2+gcf_dim*gcf_dim*(gcf_grid*sub_y+sub_x)+gcf_dim*b+a
            r1=image(image_index)%x; i1=image(image_index)%y
            r2=table(table_index)%x; i2=table(table_index)%y
            if(main_x+a>=0 .and. main_y+b>=0 .and. main_x+a<img_size .and. main_y+b<img_size) then
              sum_r=sum_r+r1*r2-i1*i2; sum_i=sum_i+r1*i2+r2*i1
            end if
          end do
        end do
        output(n)%x=sum_r; output(n)%y=sum_i
      end do
!$omp end target teams distribute parallel do
    end do
    elapsed=(omp_get_wtime()-begin_time)/real(repeat,real64)
!$omp end target data
    write(*,'(a,f0.9,a)') 'Average kernel execution time ',elapsed,' (s)'
  end subroutine degrid_gpu

  subroutine degrid_cpu(output,values,image,table)
    type(precision2),intent(out)::output(0:)
    type(precision2),intent(in)::values(0:),image(0:),table(0:)
    integer(int32)::n,a,b,sx,sy,mx,my,table_index,image_index
    real(real64)::sum_r,sum_i,r1,i1,r2,i2
    do n=0,npoints-1
      sx=int(real(gcf_grid,real32)*real(values(n)%x-floor(values(n)%x),real32),int32)
      sy=int(real(gcf_grid,real32)*real(values(n)%y-floor(values(n)%y),real32),int32)
      mx=int(floor(values(n)%x),int32);my=int(floor(values(n)%y),int32);sum_r=0.0_real64;sum_i=0.0_real64
      do a=-gcf_dim/2,gcf_dim/2-1
        do b=-gcf_dim/2,gcf_dim/2-1
          image_index=image_offset+mx+a+img_size*(my+b);table_index=gcf_dim*(gcf_dim+1)/2+gcf_dim*gcf_dim*(gcf_grid*sy+sx)+gcf_dim*b+a
          r1=image(image_index)%x;i1=image(image_index)%y;r2=table(table_index)%x;i2=table(table_index)%y
          if(mx+a>=0 .and. my+b>=0 .and. mx+a<img_size .and. my+b<img_size) then
            sum_r=sum_r+r1*r2-i1*i2;sum_i=sum_i+r1*i2+r2*i1
          end if
        end do
      end do
      output(n)%x=sum_r;output(n)%y=sum_i
    end do
  end subroutine degrid_cpu
end program degrid
