module face_image
  use iso_fortran_env, only: int8, int32
  use face_types
  implicit none
contains
  subroutine read_pgm(filename, image)
    character(len=*), intent(in) :: filename
    type(image_t), intent(out) :: image
    character(len=512) :: line
    integer :: u, ios, p
    integer(int8), allocatable :: raw(:)
    open(newunit=u,file=trim(filename),access='stream',form='unformatted',status='old',action='read',iostat=ios)
    if (ios /= 0) error stop 'Unable to open input PGM image'
    call read_header_line(u,line); if (trim(line) /= 'P5') error stop 'Only raw P5 PGM files are supported'
    call read_header_line(u,line); read(line,*) image%width,image%height
    call read_header_line(u,line); read(line,*) image%maxgrey
    allocate(raw(image%width*image%height), image%data(image%width*image%height))
    read(u,iostat=ios) raw; close(u)
    if (ios /= 0) error stop 'Unable to read PGM pixels'
    image%data = iand(int(raw,int32),255_int32)
  end subroutine read_pgm

  subroutine read_header_line(u,line)
    integer,intent(in) :: u
    character(len=*),intent(out) :: line
    integer :: i, ios
    character(len=1) :: ch
    do
      line = ''; i=0
      do
        read(u,iostat=ios) ch
        if (ios /= 0) error stop 'Unexpected PGM header EOF'
        if (ch == achar(10)) exit
        i=i+1; if(i<=len(line)) line(i:i)=ch
      end do
      if (len_trim(line)>0 .and. line(1:1)/='#') return
    end do
  end subroutine read_header_line

  subroutine write_pgm(filename,image)
    character(len=*),intent(in) :: filename
    type(image_t),intent(in) :: image
    integer :: u,ios
    integer(int8),allocatable :: raw(:)
    character(len=64) :: header, dimensions, maximum
    allocate(raw(size(image%data))); raw=int(image%data,int8)
    open(newunit=u,file=trim(filename),access='stream',form='unformatted',status='replace',action='write',iostat=ios)
    if(ios/=0) error stop 'Unable to create output PGM image'
    write(dimensions,'(i0,1x,i0)') image%width,image%height
    write(maximum,'(i0)') image%maxgrey
    header = 'P5'//achar(10)//trim(dimensions)//achar(10)//trim(maximum)//achar(10)
    write(u) trim(header); write(u) raw; close(u)
  end subroutine write_pgm

  subroutine nearest_neighbor(source,dest,w,h)
    type(image_t),intent(in) :: source
    integer(int32),intent(in) :: w,h
    type(image_t),intent(out) :: dest
    integer :: x,y,xr,yr,sx,sy
    dest%width=w;dest%height=h;dest%maxgrey=source%maxgrey
    allocate(dest%data(w*h)); xr=(source%width*65536)/w+1; yr=(source%height*65536)/h+1
    do y=0,h-1
      sy=(y*yr)/65536
      do x=0,w-1
        sx=(x*xr)/65536; dest%data(y*w+x+1)=source%data(sy*source%width+sx+1)
      end do
    end do
  end subroutine nearest_neighbor

  subroutine integral_images(image,sum,sqsum)
    type(image_t),intent(in)::image
    integer(int32),allocatable,intent(out)::sum(:),sqsum(:)
    integer :: x,y,s,sq,idx
    allocate(sum(size(image%data)),sqsum(size(image%data)))
    do y=0,image%height-1
      s=0;sq=0
      do x=0,image%width-1
        idx=y*image%width+x+1; s=s+image%data(idx); sq=sq+image%data(idx)*image%data(idx)
        sum(idx)=s; sqsum(idx)=sq
        if(y>0) then; sum(idx)=sum(idx)+sum(idx-image%width); sqsum(idx)=sqsum(idx)+sqsum(idx-image%width); end if
      end do
    end do
  end subroutine integral_images

  subroutine draw_rectangle(image,r)
    type(image_t),intent(inout)::image
    type(rect_t),intent(in)::r
    integer::i,x,y
    do i=0,r%width-1
      x=r%x+i; if(x>=0.and.x<image%width.and.r%y>=0.and.r%y<image%height) image%data(r%y*image%width+x+1)=255
    end do
    do i=0,r%height-1
      y=r%y+i; if(r%x>=0.and.r%x<image%width.and.y>=0.and.y<image%height) image%data(y*image%width+r%x+1)=255
    end do
  end subroutine draw_rectangle
end module face_image
