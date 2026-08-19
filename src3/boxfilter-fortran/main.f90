module boxfilter_mod
  use, intrinsic :: iso_fortran_env, only : int8, int32, int64, real32, real64
  use omp_lib
  implicit none
  integer, parameter :: radius = 10
  real(real32), parameter :: scale = 1.0_real32 / 21.0_real32
!$omp declare target (component, pack_pixel)
contains
  pure integer(int32) function component(pixel, channel) result(value)
    integer(int32), intent(in) :: pixel
    integer, intent(in) :: channel
    value = ibits(pixel, 8 * channel, 8)
  end function component

  pure integer(int32) function pack_pixel(r, g, b, a) result(pixel)
    real(real32), intent(in) :: r, g, b, a
    integer(int32) :: ir, ig, ib, ia
    ir = int(r * scale, int32); ig = int(g * scale, int32)
    ib = int(b * scale, int32); ia = int(a * scale, int32)
    pixel = ior(iand(ir, 255_int32), shiftl(iand(ig, 255_int32), 8))
    pixel = ior(pixel, shiftl(iand(ib, 255_int32), 16))
    pixel = ior(pixel, shiftl(iand(ia, 255_int32), 24))
  end function pack_pixel

  subroutine run_gpu(input, temp, output, width, height, cycles)
    integer(int32), intent(in) :: input(0:*)
    integer(int32), intent(inout) :: temp(0:*), output(0:*)
    integer, intent(in) :: width, height, cycles
    integer :: aligned, outputs, blocks, teams, threads
    integer :: iteration, lid, gx, gy, x, offset, begin_x, end_x, y
    integer :: start_count, stop_count, rate
    integer(int32) :: tile(0:89)
    real(real32) :: sr, sg, sb, sa, tr, tg, tb, ta, br, bg, bb, ba
    aligned = ((radius + 15) / 16) * 16
    outputs = 64
    if (256 < aligned + outputs + radius) outputs = 256 - aligned - radius
    blocks = (width + outputs - 1) / outputs
    teams = height * blocks
    threads = aligned + outputs + radius
    call system_clock(start_count, rate)
    do iteration = 1, cycles
!$omp target teams num_teams(teams) thread_limit(threads) private(tile) &
!$omp& map(to:input(0:width*height-1)) map(tofrom:temp(0:width*height-1))
!$omp parallel num_threads(threads) private(lid,gx,gy,x,offset,begin_x,end_x,sr,sg,sb,sa) shared(tile)
        lid = omp_get_thread_num()
        gx = mod(omp_get_team_num(), blocks)
        gy = omp_get_team_num() / blocks
        x = gx * outputs + lid - aligned
        offset = gy * width + x
        if (x >= 0 .and. x < width) then
          tile(lid) = input(offset)
        else
          tile(lid) = 0_int32
        end if
!$omp barrier
        if (x >= 0 .and. x < width .and. lid >= aligned .and. lid < aligned + outputs) then
          sr = 0.0_real32; sg = 0.0_real32
          sb = 0.0_real32; sa = 0.0_real32
          begin_x = lid - radius; end_x = begin_x + 2 * radius
          do while (begin_x <= end_x)
            sr = sr + real(component(tile(begin_x), 0), real32)
            sg = sg + real(component(tile(begin_x), 1), real32)
            sb = sb + real(component(tile(begin_x), 2), real32)
            sa = sa + real(component(tile(begin_x), 3), real32)
            begin_x = begin_x + 1
          end do
          temp(offset) = pack_pixel(sr, sg, sb, sa)
        end if
!$omp end parallel
!$omp end target teams

!$omp target teams distribute parallel do thread_limit(64) &
!$omp& map(to:temp(0:width*height-1)) map(tofrom:output(0:width*height-1)) &
!$omp& private(y,sr,sg,sb,sa,tr,tg,tb,ta,br,bg,bb,ba)
      do x = 0, width - 1
        tr = real(component(temp(x), 0), real32)
        tg = real(component(temp(x), 1), real32)
        tb = real(component(temp(x), 2), real32)
        ta = real(component(temp(x), 3), real32)
        br = real(component(temp((height-1)*width+x), 0), real32)
        bg = real(component(temp((height-1)*width+x), 1), real32)
        bb = real(component(temp((height-1)*width+x), 2), real32)
        ba = real(component(temp((height-1)*width+x), 3), real32)
        sr = tr * real(radius,real32); sg = tg * real(radius,real32)
        sb = tb * real(radius,real32); sa = ta * real(radius,real32)
        do y = 0, radius
          sr = sr + real(component(temp(y*width+x), 0), real32)
          sg = sg + real(component(temp(y*width+x), 1), real32)
          sb = sb + real(component(temp(y*width+x), 2), real32)
          sa = sa + real(component(temp(y*width+x), 3), real32)
        end do
        output(x) = pack_pixel(sr,sg,sb,sa)
        do y = 1, radius
          sr = sr + real(component(temp((y+radius)*width+x),0),real32) - tr
          sg = sg + real(component(temp((y+radius)*width+x),1),real32) - tg
          sb = sb + real(component(temp((y+radius)*width+x),2),real32) - tb
          sa = sa + real(component(temp((y+radius)*width+x),3),real32) - ta
          output(y*width+x) = pack_pixel(sr,sg,sb,sa)
        end do
        do y = radius + 1, height - radius - 1
          sr = sr + real(component(temp((y+radius)*width+x),0),real32) - &
                    real(component(temp((y-radius-1)*width+x),0),real32)
          sg = sg + real(component(temp((y+radius)*width+x),1),real32) - &
                    real(component(temp((y-radius-1)*width+x),1),real32)
          sb = sb + real(component(temp((y+radius)*width+x),2),real32) - &
                    real(component(temp((y-radius-1)*width+x),2),real32)
          sa = sa + real(component(temp((y+radius)*width+x),3),real32) - &
                    real(component(temp((y-radius-1)*width+x),3),real32)
          output(y*width+x) = pack_pixel(sr,sg,sb,sa)
        end do
        do y = height - radius, height - 1
          sr = sr + br - real(component(temp((y-radius-1)*width+x),0),real32)
          sg = sg + bg - real(component(temp((y-radius-1)*width+x),1),real32)
          sb = sb + bb - real(component(temp((y-radius-1)*width+x),2),real32)
          sa = sa + ba - real(component(temp((y-radius-1)*width+x),3),real32)
          output(y*width+x) = pack_pixel(sr,sg,sb,sa)
        end do
      end do
!$omp end target teams distribute parallel do
    end do
    call system_clock(stop_count)
    write(*,'(a,f0.6,a)') 'Average kernel execution time ', &
      real(stop_count-start_count,real64)*1.0e6_real64/real(rate,real64)/ &
      real(cycles,real64), ' (us)'
  end subroutine run_gpu

  subroutine run_host(input, temp, output, width, height)
    integer(int32), intent(in) :: input(0:)
    integer(int32), intent(inout) :: temp(0:), output(0:)
    integer, intent(in) :: width, height
    integer :: x, y, p
    real(real32) :: sr, sg, sb, sa
    do y = 0, height - 1
      do x = 0, width - 1
        sr=0.0_real32; sg=0.0_real32; sb=0.0_real32; sa=0.0_real32
        do p = x-radius, x+radius
          if (p >= 0 .and. p < width) then
            sr=sr+real(component(input(y*width+p),0),real32)
            sg=sg+real(component(input(y*width+p),1),real32)
            sb=sb+real(component(input(y*width+p),2),real32)
            sa=sa+real(component(input(y*width+p),3),real32)
          end if
        end do
        temp(y*width+x)=pack_pixel(sr,sg,sb,sa)
      end do
    end do
    do x = 0, width - 1
      do y = 0, height - 1
        sr=0.0_real32; sg=0.0_real32; sb=0.0_real32; sa=0.0_real32
        do p = y-radius, y+radius
          if (p < 0) then
            sr=sr+real(component(temp(x),0),real32)
            sg=sg+real(component(temp(x),1),real32)
            sb=sb+real(component(temp(x),2),real32)
            sa=sa+real(component(temp(x),3),real32)
          else if (p >= height) then
            sr=sr+real(component(temp((height-1)*width+x),0),real32)
            sg=sg+real(component(temp((height-1)*width+x),1),real32)
            sb=sb+real(component(temp((height-1)*width+x),2),real32)
            sa=sa+real(component(temp((height-1)*width+x),3),real32)
          else
            sr=sr+real(component(temp(p*width+x),0),real32)
            sg=sg+real(component(temp(p*width+x),1),real32)
            sb=sb+real(component(temp(p*width+x),2),real32)
            sa=sa+real(component(temp(p*width+x),3),real32)
          end if
        end do
        output(y*width+x)=pack_pixel(sr,sg,sb,sa)
      end do
    end do
  end subroutine run_host

  subroutine read_ppm(path, image, width, height, ok)
    character(*), intent(in) :: path
    integer(int32), allocatable, intent(out) :: image(:)
    integer, intent(out) :: width, height
    logical, intent(out) :: ok
    integer :: unit, ios, i, pos, n, r, g, b
    integer(int64) :: bytes
    integer(int8), allocatable :: raw(:)
    character(64) :: header
    ok=.false.; width=0; height=0
    open(newunit=unit,file=path,access='stream',form='unformatted',status='old',iostat=ios)
    if (ios /= 0) return
    inquire(unit=unit,size=bytes); allocate(raw(int(bytes)))
    read(unit,iostat=ios) raw; close(unit)
    if (ios /= 0) return
    pos=1; call token(raw,pos,header); if(trim(header)/='P6') return
    call token(raw,pos,header); read(header,*,iostat=ios) width; if(ios/=0)return
    call token(raw,pos,header); read(header,*,iostat=ios) height; if(ios/=0)return
    call token(raw,pos,header); if(trim(header)/='255') return
    do while(pos<=size(raw) .and. (raw(pos)==10_int8 .or. raw(pos)==13_int8 .or. raw(pos)==32_int8)); pos=pos+1; end do
    if(pos+3*width*height-1>size(raw)) return
    allocate(image(0:width*height-1))
    do i=0,width*height-1
      r=iand(int(raw(pos+3*i),int32),255_int32)
      g=iand(int(raw(pos+3*i+1),int32),255_int32)
      b=iand(int(raw(pos+3*i+2),int32),255_int32)
      image(i)=ior(ior(r,shiftl(g,8)),shiftl(b,16))
    end do
    ok=.true.
  end subroutine read_ppm

  recursive subroutine token(raw,pos,text)
    integer(int8),intent(in)::raw(:); integer,intent(inout)::pos
    character(*),intent(out)::text
    integer::n,code
    text=''; n=0
    do while(pos<=size(raw) .and. (raw(pos)==10_int8 .or. raw(pos)==13_int8 .or. raw(pos)==32_int8)); pos=pos+1; end do
    if(pos<=size(raw) .and. raw(pos)==35_int8) then
      do while(pos<=size(raw) .and. raw(pos)/=10_int8); pos=pos+1; end do
      call token(raw,pos,text); return
    end if
    do while(pos<=size(raw))
      code=iand(int(raw(pos),int32),255_int32)
      if(code==10 .or. code==13 .or. code==32) exit
      n=n+1; if(n<=len(text)) text(n:n)=achar(code); pos=pos+1
    end do
  end subroutine token
end module boxfilter_mod

program main
  use, intrinsic :: iso_fortran_env, only : int32, int64
  use boxfilter_mod
  implicit none
  integer :: width,height,cycles,ios,i
  integer(int64) :: first,last,pixels
  integer(int32),allocatable :: input(:),temp(:),device_output(:),host_output(:)
  character(1024)::path,arg
  logical::ok
  if(command_argument_count()/=2) then
    write(*,'(a)') 'Usage ./main <PPM image> <repeat>'; stop 1
  end if
  call get_command_argument(1,path); call get_command_argument(2,arg)
  read(arg,*,iostat=ios) cycles; if(ios/=0 .or. cycles<=0) stop 1
  call read_ppm(trim(path),input,width,height,ok)
  write(*,'(a,i0,a,i0,a,i0,a,i0)') 'Image Width = ',width,', Height = ',height, &
    ', bpp = ',32,', Mask Radius = ',radius
  if(width>1920 .or. height>1080) ok=.false.
  if(.not.ok) stop 1
  pixels=int(width,int64)*int(height,int64)
  allocate(temp(0:pixels-1),device_output(0:pixels-1),host_output(0:pixels-1))
  temp=0; device_output=0; host_output=0
  write(*,'(a)') 'Using Local Memory for Row Processing'; write(*,*)
!$omp target data map(to:input(0:pixels-1),temp(0:pixels-1)) map(tofrom:device_output(0:pixels-1))
  write(*,'(a)') 'Warmup..'
  call run_gpu(input,temp,device_output,width,height,cycles)
  write(*,'(/,a,i0,a,/)') 'Running BoxFilterGPU for ',cycles,' cycles...'
  call run_gpu(input,temp,device_output,width,height,cycles)
!$omp end target data
  call run_host(input,temp,host_output,width,height)
  first=int(radius,int64)*int(width,int64); last=pixels-first-1
  do i=int(first),int(last)
    if(device_output(i)/=host_output(i)) then
      write(*,'(i0,1x,z8.8,1x,z8.8)') i,device_output(i),host_output(i)
      write(*,'(a)') 'FAIL'; stop
    end if
  end do
  write(*,'(a)') 'PASS'
end program main
