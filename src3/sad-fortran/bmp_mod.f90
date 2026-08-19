module bmp_mod
  use iso_fortran_env, only: int8
  implicit none

  type :: bmp_image
    integer :: width = 0
    integer :: height = 0
    integer(int8), allocatable :: rgb(:)
  end type bmp_image

!$omp declare target (u8)
contains

  integer function u8(v) result(r)
    integer(int8), intent(in) :: v
    r = int(v)
    if (r < 0) r = r + 256
  end function u8

  integer function le16(buf, pos) result(v)
    integer(int8), intent(in) :: buf(0:)
    integer, intent(in) :: pos
    v = u8(buf(pos)) + 256 * u8(buf(pos + 1))
  end function le16

  integer function le32(buf, pos) result(v)
    integer(int8), intent(in) :: buf(0:)
    integer, intent(in) :: pos
    v = u8(buf(pos)) + 256 * u8(buf(pos + 1)) + 65536 * u8(buf(pos + 2)) + 16777216 * u8(buf(pos + 3))
  end function le32

  subroutine read_bmp24(path, img)
    character(len=*), intent(in) :: path
    type(bmp_image), intent(out) :: img
    integer :: unit, ios, size_bytes, offset, bits, row_stride, row, col, src, dst
    integer(int8), allocatable :: filebuf(:)
    character(len=:), allocatable :: actual_path

    actual_path = trim(path)
    open(newunit=unit, file=actual_path, access='stream', form='unformatted', status='old', action='read', iostat=ios)
    if (ios /= 0 .and. index(actual_path, '../') == 1) then
      actual_path = '../../src/' // actual_path(4:)
      open(newunit=unit, file=actual_path, access='stream', form='unformatted', status='old', action='read', iostat=ios)
    end if
    if (ios /= 0) then
      print '(a,a)', 'Failed to load input image: ', trim(path)
      stop 1
    end if

    inquire(unit=unit, size=size_bytes)
    allocate(filebuf(0:size_bytes-1))
    read(unit) filebuf
    close(unit)

    offset = le32(filebuf, 10)
    img%width = le32(filebuf, 18)
    img%height = le32(filebuf, 22)
    bits = le16(filebuf, 28)
    if (bits /= 24) then
      print '(a)', 'Only 24-bit BMP input is supported by this Fortran port'
      stop 1
    end if
    row_stride = ((img%width * 3 + 3) / 4) * 4
    allocate(img%rgb(0:img%width*img%height*3-1))

    do row = 0, img%height - 1
      do col = 0, img%width - 1
        src = offset + (img%height - 1 - row) * row_stride + col * 3
        dst = (row * img%width + col) * 3
        img%rgb(dst) = filebuf(src + 2)
        img%rgb(dst + 1) = filebuf(src + 1)
        img%rgb(dst + 2) = filebuf(src)
      end do
    end do
  end subroutine read_bmp24

end module bmp_mod
