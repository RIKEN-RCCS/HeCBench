program face
  use iso_fortran_env, only: int32, int64, real32
  use face_types
  use face_image
  use face_haar
  implicit none
  character(len=1024)::input_file,info_file,class_file,output_file
  type(image_t)::image,scaled_image
  type(rect_t),allocatable::candidates(:),result(:)
  integer(int32),allocatable::stage_count(:),rectangles(:),weights(:),alpha1(:),alpha2(:),tree_threshold(:),stage_threshold(:)
  integer(int32),allocatable::sum(:),sqsum(:),scaled(:),hit(:)
  integer(int32)::argc,stages,nodes,sw,sh,x,y,idx,ncand,nresult,iter,x2,y2
  integer(int64)::start_count,end_count,count_rate
  real(real32)::factor

  argc=command_argument_count()
  if(argc/=4)then
    write(*,'(a)') 'Usage: ./main <input image file> <classifier information> <class information> <output image file>'
    error stop 1
  end if
  call get_command_argument(1,input_file);call get_command_argument(2,info_file);call get_command_argument(3,class_file);call get_command_argument(4,output_file)
  write(*,'(a)') '-- entering main function --';write(*,'(a)') '-- loading image --'
  call read_pgm(trim(input_file),image)
  call load_classifier(trim(info_file),trim(class_file),stages,stage_count,rectangles,weights,alpha1,alpha2,tree_threshold,stage_threshold,nodes)
  write(*,'(a)') '-- loading cascade classifier --';write(*,'(a)') '-- detecting faces --'
  allocate(candidates(0));ncand=0;factor=1.0_real32;iter=0
  call system_clock(start_count,count_rate)
  do
    iter=iter+1;sw=int(real(image%width,real32)/factor,int32);sh=int(real(image%height,real32)/factor)
    if(sw<24.or.sh<24)exit
    if(sw==image%width.and.sh==image%height)then
      scaled_image=image
    else
      call nearest_neighbor(image,scaled_image,sw,sh)
    end if
    call integral_images(scaled_image,sum,sqsum);call scaled_corners(rectangles,nodes,sw,scaled)
    x2=sw-24;y2=sh-24;allocate(hit((x2+1)*(y2+1)));hit=0
    !$omp target data map(to: sum, sqsum, stage_count, scaled, weights, alpha1, alpha2, tree_threshold, stage_threshold) map(from: hit)
    !$omp target teams distribute parallel do thread_limit(256)
    do idx=0,(x2+1)*(y2+1)-1
      x=mod(idx,x2+1);y=idx/(x2+1)
      if(cascade_at(x,y,sw,sum,sqsum,stages,stage_count,scaled,weights,alpha1,alpha2,tree_threshold,stage_threshold)>0)hit(idx+1)=1
    end do
    !$omp end target teams distribute parallel do
    !$omp end target data
    call append_hits(hit,x2,y2,factor,candidates,ncand)
    write(*,'(a,i0)') 'detecting faces, iter := ',iter
    deallocate(sum,sqsum,scaled,hit)
    if(allocated(scaled_image%data).and.sw/=image%width)deallocate(scaled_image%data)
    factor=factor*1.2_real32
  end do
  call system_clock(end_count)
  write(*,'(a,f12.6,a)') 'Object detection time ',real(end_count-start_count,real32)/real(count_rate,real32),' (s)'
  call group_rectangles(candidates,ncand,result,nresult)
  do idx=1,nresult;call draw_rectangle(image,result(idx));end do
  write(*,'(a)') '-- saving output --';call write_pgm(trim(output_file),image);write(*,'(a)') '-- image saved --'

contains
  subroutine append_hits(hit,x2,y2,factor,rects,n)
    integer(int32),intent(in)::hit(:),x2,y2
    real(real32),intent(in)::factor
    type(rect_t),allocatable,intent(inout)::rects(:)
    integer,intent(inout)::n
    type(rect_t),allocatable::tmp(:)
    integer::i,x,y,add,at
    add=count(hit/=0);if(add==0)return
    allocate(tmp(n+add));if(n>0)tmp(1:n)=rects(1:n);at=n
    do i=0,size(hit)-1
      if(hit(i+1)==0)cycle
      x=mod(i,x2+1);y=i/(x2+1);at=at+1
      tmp(at)%x=iround(real(x,real32)*factor);tmp(at)%y=iround(real(y,real32)*factor)
      tmp(at)%width=iround(24.0_real32*factor);tmp(at)%height=iround(24.0_real32*factor)
    end do
    call move_alloc(tmp,rects);n=at
  end subroutine append_hits

  subroutine group_rectangles(input,n,out,m)
    type(rect_t),intent(in)::input(:)
    integer,intent(in)::n
    type(rect_t),allocatable,intent(out)::out(:)
    integer,intent(out)::m
    integer,allocatable::parent(:),count_class(:)
    type(rect_t),allocatable::average(:)
    integer::i,j,r,ri,rj,classes,dx,dy
    if(n==0)then;allocate(out(0));m=0;return;end if
    allocate(parent(n));parent=[(i,i=1,n)]
    do i=1,n-1
      do j=i+1,n
        dx=int(.4_real32*real(min(input(i)%width,input(j)%width)+min(input(i)%height,input(j)%height),real32)*.5_real32)
        if(abs(input(i)%x-input(j)%x)<=dx .and. abs(input(i)%y-input(j)%y)<=dx)then
          ri=root(parent,i);rj=root(parent,j);if(ri/=rj)parent(rj)=ri
        end if
      end do
    end do
    do i=1,n;parent(i)=root(parent,i);end do
    classes=0;do i=1,n;if(parent(i)==i)classes=classes+1;end do
    allocate(average(n),count_class(n));average=rect_t();count_class=0
    do i=1,n;r=parent(i);average(r)%x=average(r)%x+input(i)%x;average(r)%y=average(r)%y+input(i)%y;average(r)%width=average(r)%width+input(i)%width;average(r)%height=average(r)%height+input(i)%height;count_class(r)=count_class(r)+1;end do
    allocate(out(classes));m=0
    do i=1,n
      if(parent(i)/=i.or.count_class(i)<=1)cycle
      m=m+1;out(m)%x=iround(real(average(i)%x,real32)/count_class(i));out(m)%y=iround(real(average(i)%y,real32)/count_class(i));out(m)%width=iround(real(average(i)%width,real32)/count_class(i));out(m)%height=iround(real(average(i)%height,real32)/count_class(i))
    end do
  end subroutine group_rectangles
  recursive integer function root(parent,i) result(r)
    integer,intent(inout)::parent(:);integer,intent(in)::i
    if(parent(i)==i)then;r=i;else;r=root(parent,parent(i));parent(i)=r;end if
  end function root
end program face
