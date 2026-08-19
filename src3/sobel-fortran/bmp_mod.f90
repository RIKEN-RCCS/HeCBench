module bmp_mod
  use iso_fortran_env, only: int8
  use sobel_mod, only: uchar4, u8
  implicit none
contains
  integer function le16(buf,pos) result(v)
    integer(int8), intent(in) :: buf(0:)
    integer, intent(in) :: pos
    v = u8(buf(pos)) + 256*u8(buf(pos+1))
  end function le16
  integer function le32(buf,pos) result(v)
    integer(int8), intent(in) :: buf(0:)
    integer, intent(in) :: pos
    v = u8(buf(pos)) + 256*u8(buf(pos+1)) + 65536*u8(buf(pos+2)) + 16777216*u8(buf(pos+3))
  end function le32
  subroutine read_bmp24(path, pixels, width, height)
    character(len=*), intent(in) :: path
    type(uchar4), allocatable, intent(out) :: pixels(:)
    integer, intent(out) :: width, height
    integer :: unit, ios, size_bytes, offset, bits, row_stride, row, col, src, dst
    integer(int8), allocatable :: filebuf(:)
    character(len=:), allocatable :: actual_path
    actual_path = trim(path)
    open(newunit=unit,file=actual_path,access='stream',form='unformatted',status='old',action='read',iostat=ios)
    if (ios /= 0 .and. index(actual_path, '../') == 1) then
      actual_path = '../../src/' // actual_path(4:)
      open(newunit=unit,file=actual_path,access='stream',form='unformatted',status='old',action='read',iostat=ios)
    end if
    if (ios /= 0) stop 1
    inquire(unit=unit,size=size_bytes)
    allocate(filebuf(0:size_bytes-1))
    read(unit) filebuf
    close(unit)
    offset = le32(filebuf,10); width = le32(filebuf,18); height = le32(filebuf,22); bits = le16(filebuf,28)
    if (bits /= 24) stop 1
    row_stride = ((width*3+3)/4)*4
    allocate(pixels(0:width*height-1))
    do row = 0, height - 1
      do col = 0, width - 1
        ! SDKBitMap keeps the BMP file's row order rather than vertically
        ! flipping it, and initializes the fourth uchar component to 255.
        src = offset + row * row_stride + col * 3
        dst = row * width + col
        pixels(dst) = uchar4(filebuf(src+2), filebuf(src+1), filebuf(src), -1_int8)
      end do
    end do
  end subroutine read_bmp24
end module bmp_mod
