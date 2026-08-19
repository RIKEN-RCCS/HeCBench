module face_haar
  use iso_fortran_env, only: int32, real32
  use face_types
  implicit none
  !$omp declare target (cascade_at)
contains
  subroutine load_classifier(info_file,class_file,stages,stage_count,rectangles,weights,alpha1,alpha2,tree_threshold,stage_threshold,nodes)
    character(len=*),intent(in)::info_file,class_file
    integer(int32),intent(out)::stages,nodes
    integer(int32),allocatable,intent(out)::stage_count(:),rectangles(:),weights(:),alpha1(:),alpha2(:),tree_threshold(:),stage_threshold(:)
    integer::u,ios,i,j,k,l,ri,wi,ti
    open(newunit=u,file=trim(info_file),status='old',action='read',iostat=ios); if(ios/=0) error stop 'Unable to open face classifier info'
    read(u,*,iostat=ios) stages; if(ios/=0.or.stages<=0) error stop 'Invalid classifier info'
    allocate(stage_count(stages)); nodes=0
    do i=1,stages; read(u,*)stage_count(i);nodes=nodes+stage_count(i);end do;close(u)
    allocate(rectangles(12*nodes),weights(3*nodes),alpha1(nodes),alpha2(nodes),tree_threshold(nodes),stage_threshold(stages))
    open(newunit=u,file=trim(class_file),status='old',action='read',iostat=ios);if(ios/=0) error stop 'Unable to open face classifier data'
    ri=1;wi=1;ti=1
    do i=1,stages
      do j=1,stage_count(i)
        do k=1,3
          do l=1,4;read(u,*)rectangles(ri);ri=ri+1;end do
          read(u,*)weights(wi);wi=wi+1
        end do
        read(u,*)tree_threshold(ti);read(u,*)alpha1(ti);read(u,*)alpha2(ti);ti=ti+1
      end do
      read(u,*)stage_threshold(i)
    end do;close(u)
  end subroutine load_classifier

  integer(int32) function isqrt32(value)
    integer(int32),intent(in)::value
    integer(int32)::i,a,b,c,v
    a=0;b=0;c=0;v=value
    do i=1,16
      c=shiftl(c,2)+shiftr(v,30);v=shiftl(v,2);a=shiftl(a,1);b=ior(shiftl(a,1),1)
      if(c>=b)then;c=c-b;a=a+1;end if
    end do
    isqrt32=a
  end function isqrt32

  integer(int32) function cascade_at(x,y,width,sum,sqsum,stages,stage_count,scaled,weights,alpha1,alpha2,tree_threshold,stage_threshold)
    integer(int32),intent(in)::x,y,width,stages,stage_count(:),sum(:),sqsum(:),scaled(:),weights(:),alpha1(:),alpha2(:),tree_threshold(:),stage_threshold(:)
    integer(int32)::p,pq,variance,mean,i,j,h,w,r,stage_sum,ss
    p=y*width+x; pq=p; variance=sqsum(pq+1)-sqsum(pq+25)+sqsum(pq+24*width+1)-sqsum(pq+24*width+25)
    mean=sum(p+1)-sum(p+25)-sum(p+24*width+1)+sum(p+24*width+25)
    variance=variance/576-mean*mean;if(variance>0)variance=isqrt32(variance);if(variance<=0)variance=1
    h=1;w=1;r=1
    do i=1,stages
      stage_sum=0
      do j=1,stage_count(i)
        ss=(sum(scaled(r)+p)-sum(scaled(r+1)+p)-sum(scaled(r+2)+p)+sum(scaled(r+3)+p))*weights(w)
        ss=ss+(sum(scaled(r+4)+p)-sum(scaled(r+5)+p)-sum(scaled(r+6)+p)+sum(scaled(r+7)+p))*weights(w+1)
        if(scaled(r+8)>0) ss=ss+(sum(scaled(r+8)+p)-sum(scaled(r+9)+p)-sum(scaled(r+10)+p)+sum(scaled(r+11)+p))*weights(w+2)
        if(ss>=tree_threshold(h)*variance)then;stage_sum=stage_sum+alpha2(h);else;stage_sum=stage_sum+alpha1(h);end if
        h=h+1;w=w+3;r=r+12
      end do
      if(real(stage_sum,real32)<.4_real32*real(stage_threshold(i),real32))then;cascade_at=-i;return;end if
    end do
    cascade_at=1
  end function cascade_at

  subroutine scaled_corners(rectangles,nodes,width,scaled)
    integer(int32),intent(in)::rectangles(:),nodes,width
    integer(int32),allocatable,intent(out)::scaled(:)
    integer::i,k,r,x,y,w,h
    allocate(scaled(12*nodes));scaled=0
    do i=0,nodes-1
      do k=0,2
        r=i*12+k*4+1;x=rectangles(r);y=rectangles(r+1);w=rectangles(r+2);h=rectangles(r+3)
        if(k==2.and.x==0.and.y==0.and.w==0.and.h==0)cycle
        scaled(r)=y*width+x+1;scaled(r+1)=y*width+x+w+1;scaled(r+2)=(y+h)*width+x+1;scaled(r+3)=(y+h)*width+x+w+1
      end do
    end do
  end subroutine scaled_corners
end module face_haar
