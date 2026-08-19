module easywave_fault
  use easywave_constants
  use easywave_okada, only: okada_vertical
  implicit none
  type :: fault_t
    real(dp) :: lon, lat, depth, strike, dip, rake, length, width, slip
    real(dp) :: sind, cosd, sins, coss, coslat, zbot, dslip, sslip, wp
    integer :: refpos
  end type fault_t
contains
  subroutine read_faults(filename, faults, nfault, ierr)
    character(*), intent(in) :: filename
    type(fault_t), allocatable, intent(out) :: faults(:)
    integer, intent(out) :: nfault, ierr
    integer :: unit, ios, n
    character(len=512) :: line
    nfault = 0; ierr = 0
    open(newunit=unit,file=filename,status='old',action='read',iostat=ios)
    if (ios /= 0) then; ierr = 1; return; end if
    do
      read(unit,'(A)',iostat=ios) line
      if (ios /= 0) exit
      if (len_trim(line) > 0 .and. line(1:1) /= ';') nfault = nfault + 1
    end do
    if (nfault == 0) then; close(unit); ierr=2; return; end if
    allocate(faults(nfault)); rewind(unit); n=0
    do
      read(unit,'(A)',iostat=ios) line
      if (ios /= 0) exit
      if (len_trim(line) == 0 .or. line(1:1) == ';') cycle
      n=n+1
      call parse_fault(line, faults(n), ios)
      if (ios /= 0) then; ierr=3; close(unit); return; end if
    end do
    close(unit)
  end subroutine read_faults

  subroutine parse_fault(line, f, ierr)
    character(*), intent(in) :: line
    type(fault_t), intent(out) :: f
    integer, intent(out) :: ierr
    character(len=32) :: ref
    ierr=0; f%refpos=1
    call scalar_after(line,'-location',f%lon,ierr); if(ierr/=0)return
    call two_after(line,'-location',f%lon,f%lat,f%depth,ierr); if(ierr/=0)return
    f%depth=f%depth*1000.0_dp
    call word_after(line,'-refpos',ref,ierr); if(ierr/=0)return
    select case(trim(adjustl(ref)))
    case('C','c'); f%refpos=0
    case('MT','mt'); f%refpos=1
    case('BT','bt'); f%refpos=2
    case('BB','bb'); f%refpos=3
    case('MB','mb'); f%refpos=4
    case default; ierr=1; return
    end select
    call scalar_after(line,'-strike',f%strike,ierr); if(ierr/=0)return
    call scalar_after(line,'-dip',f%dip,ierr); if(ierr/=0)return
    call scalar_after(line,'-rake',f%rake,ierr); if(ierr/=0)return
    call two_after(line,'-size',f%length,f%width,ierr=ierr); if(ierr/=0)return
    f%length=f%length*1000.0_dp; f%width=f%width*1000.0_dp
    call scalar_after(line,'-slip',f%slip,ierr); if(ierr/=0)return
    f%sind=sindeg(f%dip); f%cosd=cosdeg(f%dip)
    f%sins=sindeg(90.0_dp-f%strike); f%coss=cosdeg(90.0_dp-f%strike); f%coslat=cosdeg(f%lat)
    select case(f%refpos)
    case(0); f%zbot=f%depth+0.5_dp*f%width*f%sind
    case(1,2); f%zbot=f%depth+f%width*f%sind
    case(3,4); f%zbot=f%depth
    end select
    f%dslip=f%slip*sindeg(f%rake); f%sslip=f%slip*cosdeg(f%rake); f%wp=f%width*f%cosd
  end subroutine parse_fault

  subroutine scalar_after(line,key,value,ierr)
    character(*),intent(in)::line,key
    real(dp),intent(out)::value
    integer,intent(out)::ierr
    integer::p,ios
    p=index(line,key); ierr=0
    if(p==0) then; ierr=1; return; end if
    read(line(p+len_trim(key):),*,iostat=ios) value
    if(ios/=0) ierr=1
  end subroutine scalar_after

  subroutine two_after(line,key,a,b,c,ierr)
    character(*),intent(in)::line,key
    real(dp),intent(out)::a,b
    real(dp),intent(out),optional::c
    integer,intent(out)::ierr
    integer::p,ios
    p=index(line,key); ierr=0
    if(p==0) then; ierr=1; return; end if
    if(present(c)) then
      read(line(p+len_trim(key):),*,iostat=ios) a,b,c
    else
      read(line(p+len_trim(key):),*,iostat=ios) a,b
    end if
    if(ios/=0) ierr=1
  end subroutine two_after

  subroutine word_after(line,key,word,ierr)
    character(*),intent(in)::line,key
    character(*),intent(out)::word
    integer,intent(out)::ierr
    integer::p,ios
    p=index(line,key); ierr=0
    if(p==0) then; ierr=1; return; end if
    read(line(p+len_trim(key):),*,iostat=ios) word
    if(ios/=0) ierr=1
  end subroutine word_after

  real(dp) function displacement(f,lon,lat)
    type(fault_t),intent(in)::f
    real(dp),intent(in)::lon,lat
    real(dp)::x,y,lx,ly
    y=rearth*deg2rad(lat-f%lat); x=rearth*f%coslat*deg2rad(lon-f%lon)
    lx=x*f%coss+y*f%sins; ly=-x*f%sins+y*f%coss
    select case(f%refpos)
    case(0); lx=lx+f%length/2.0_dp; ly=ly+f%wp/2.0_dp
    case(1); lx=lx+f%length/2.0_dp; ly=ly+f%wp
    case(2); ly=ly+f%wp
    case(4); lx=lx+f%length/2.0_dp
    end select
    call okada_vertical(f%length,f%width,f%zbot,f%sind,f%cosd,f%sslip,f%dslip,lx,ly,displacement)
  end function displacement
end module easywave_fault
