module spm_mod
  use iso_fortran_env, only: int8, real32
  implicit none
  integer, parameter :: num_threads = 128
  integer, parameter :: num_blocks = 256

  type :: int3
    integer :: x, y, z
  end type int3

  real(real32), parameter :: ran(0:96) = [ &
    0.656619_real32,0.891183_real32,0.488144_real32,0.992646_real32,0.373326_real32,0.531378_real32,0.181316_real32,0.501944_real32,0.422195_real32, &
    0.660427_real32,0.673653_real32,0.95733_real32,0.191866_real32,0.111216_real32,0.565054_real32,0.969166_real32,0.0237439_real32,0.870216_real32, &
    0.0268766_real32,0.519529_real32,0.192291_real32,0.715689_real32,0.250673_real32,0.933865_real32,0.137189_real32,0.521622_real32,0.895202_real32, &
    0.942387_real32,0.335083_real32,0.437364_real32,0.471156_real32,0.14931_real32,0.135864_real32,0.532498_real32,0.725789_real32,0.398703_real32, &
    0.358419_real32,0.285279_real32,0.868635_real32,0.626413_real32,0.241172_real32,0.978082_real32,0.640501_real32,0.229849_real32,0.681335_real32, &
    0.665823_real32,0.134718_real32,0.0224933_real32,0.262199_real32,0.116515_real32,0.0693182_real32,0.85293_real32,0.180331_real32,0.0324186_real32, &
    0.733926_real32,0.536517_real32,0.27603_real32,0.368458_real32,0.0128863_real32,0.889206_real32,0.866021_real32,0.254247_real32,0.569481_real32, &
    0.159265_real32,0.594364_real32,0.3311_real32,0.658613_real32,0.863634_real32,0.567623_real32,0.980481_real32,0.791832_real32,0.152594_real32, &
    0.833027_real32,0.191863_real32,0.638987_real32,0.669_real32,0.772088_real32,0.379818_real32,0.441585_real32,0.48306_real32,0.608106_real32, &
    0.175996_real32,0.00202556_real32,0.790224_real32,0.513609_real32,0.213229_real32,0.10345_real32,0.157337_real32,0.407515_real32,0.407757_real32, &
    0.0526927_real32,0.941815_real32,0.149972_real32,0.384374_real32,0.311059_real32,0.168534_real32,0.896648_real32 ]
  integer(int8), parameter :: byte_value(0:255) = [ &
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, &
    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31, &
    32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47, &
    48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63, &
    64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79, &
    80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95, &
    96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111, &
    112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127, &
    -128,-127,-126,-125,-124,-123,-122,-121,-120,-119,-118,-117,-116,-115,-114,-113, &
    -112,-111,-110,-109,-108,-107,-106,-105,-104,-103,-102,-101,-100,-99,-98,-97, &
    -96,-95,-94,-93,-92,-91,-90,-89,-88,-87,-86,-85,-84,-83,-82,-81, &
    -80,-79,-78,-77,-76,-75,-74,-73,-72,-71,-70,-69,-68,-67,-66,-65, &
    -64,-63,-62,-61,-60,-59,-58,-57,-56,-55,-54,-53,-52,-51,-50,-49, &
    -48,-47,-46,-45,-44,-43,-42,-41,-40,-39,-38,-37,-36,-35,-34,-33, &
    -32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17, &
    -16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1 ]
!$omp declare target (ran, byte_value, u8, interp)

contains

  integer function u8(v) result(r)
    integer(int8), intent(in) :: v
    r = int(v)
    if (r < 0) r = r + 256
  end function u8

  real(real32) function interp(d, f, x, y, z) result(vf)
    type(int3), intent(in) :: d
    integer(int8), intent(in) :: f(0:)
    real(real32), intent(in) :: x, y, z
    integer :: ix, iy, iz, base, k111, k112, k121, k122, k211, k212, k221, k222
    real(real32) :: dx1, dy1, dz1, dx2, dy2, dz2

    ix = int(floor(x)); dx1 = x - real(ix, real32); dx2 = 1.0_real32 - dx1
    iy = int(floor(y)); dy1 = y - real(iy, real32); dy2 = 1.0_real32 - dy1
    iz = int(floor(z)); dz1 = z - real(iz, real32); dz2 = 1.0_real32 - dz1
    base = ix - 1 + d%x * (iy - 1 + d%y * (iz - 1))
    k222 = u8(f(base)); k122 = u8(f(base + 1))
    k212 = u8(f(base + d%x)); k112 = u8(f(base + d%x + 1))
    base = base + d%x * d%y
    k221 = u8(f(base)); k121 = u8(f(base + 1))
    k211 = u8(f(base + d%x)); k111 = u8(f(base + d%x + 1))
    vf = (((k222*dx2+k122*dx1)*dy2 + (k212*dx2+k112*dx1)*dy1))*dz2 + &
         (((k221*dx2+k121*dx1)*dy2 + (k211*dx2+k111*dx1)*dy1))*dz1
  end function interp

  subroutine spm_kernel(M, data_size, g_d, f_d, dg, df, ivf_d, ivg_d, data_threshold_d)
    real(real32), intent(in) :: M(0:)
    integer, intent(in) :: data_size
    integer(int8), intent(in) :: g_d(0:), f_d(0:)
    type(int3), intent(in) :: dg, df
    integer(int8), intent(out) :: ivf_d(0:), ivg_d(0:)
    logical(kind=1), intent(out) :: data_threshold_d(0:)
    integer :: i, x_datasize, y_datasize
    real(real32) :: xx_temp, yy_temp, zz_temp, rx, ry, rz, xp, yp, zp

    !$omp target teams distribute parallel do num_teams(num_blocks) thread_limit(num_threads) &
    !$omp& private(x_datasize,y_datasize,xx_temp,yy_temp,zz_temp,rx,ry,rz,xp,yp,zp)
    do i = 0, data_size - 1
      x_datasize = dg%x - 2
      y_datasize = dg%y - 2
      xx_temp = real(mod(i, x_datasize), real32) + 1.0_real32
      yy_temp = real(mod(int(floor(real(i, real32) / real(x_datasize, real32))), y_datasize), real32) + 1.0_real32
      zz_temp = floor(real(i, real32) / real(x_datasize, real32)) / real(y_datasize, real32) + 1.0_real32
      rx = xx_temp + ran(mod(i, 97))
      ry = yy_temp + ran(mod(i, 97))
      rz = zz_temp + ran(mod(i, 97))
      xp = M(0)*rx + M(4)*ry + M(8)*rz + M(12)
      yp = M(1)*rx + M(5)*ry + M(9)*rz + M(13)
      zp = M(2)*rx + M(6)*ry + M(10)*rz + M(14)
      if (zp >= 1.0_real32 .and. zp < real(df%z, real32) .and. &
          yp >= 1.0_real32 .and. yp < real(df%y, real32) .and. &
          xp >= 1.0_real32 .and. xp < real(df%x, real32)) then
        ivf_d(i) = byte_value(int(floor(interp(df, f_d, xp, yp, zp) + 0.5_real32)))
        ivg_d(i) = byte_value(int(floor(interp(dg, g_d, rx, ry, rz) + 0.5_real32)))
        data_threshold_d(i) = .true.
      else
        ivf_d(i) = 0_int8
        ivg_d(i) = 0_int8
        data_threshold_d(i) = .false.
      end if
    end do
    !$omp end target teams distribute parallel do
  end subroutine spm_kernel

  subroutine spm_reference(M, data_size, g_d, f_d, dg, df, ivf_d, ivg_d, data_threshold_d)
    real(real32), intent(in) :: M(0:)
    integer, intent(in) :: data_size
    integer(int8), intent(in) :: g_d(0:), f_d(0:)
    type(int3), intent(in) :: dg, df
    integer(int8), intent(out) :: ivf_d(0:), ivg_d(0:)
    logical(kind=1), intent(out) :: data_threshold_d(0:)
    integer :: i, x_datasize, y_datasize
    real(real32) :: xx_temp, yy_temp, zz_temp, rx, ry, rz, xp, yp, zp

    x_datasize = dg%x - 2
    y_datasize = dg%y - 2
    do i = 0, data_size - 1
      xx_temp = real(mod(i, x_datasize), real32) + 1.0_real32
      yy_temp = real(mod(int(floor(real(i, real32) / real(x_datasize, real32))), y_datasize), real32) + 1.0_real32
      zz_temp = floor(real(i, real32) / real(x_datasize, real32)) / real(y_datasize, real32) + 1.0_real32
      rx = xx_temp + ran(mod(i, 97)); ry = yy_temp + ran(mod(i, 97)); rz = zz_temp + ran(mod(i, 97))
      xp = M(0)*rx + M(4)*ry + M(8)*rz + M(12)
      yp = M(1)*rx + M(5)*ry + M(9)*rz + M(13)
      zp = M(2)*rx + M(6)*ry + M(10)*rz + M(14)
      if (zp >= 1.0_real32 .and. zp < real(df%z, real32) .and. yp >= 1.0_real32 .and. yp < real(df%y, real32) .and. xp >= 1.0_real32 .and. xp < real(df%x, real32)) then
        ivf_d(i) = byte_value(int(floor(interp(df, f_d, xp, yp, zp) + 0.5_real32)))
        ivg_d(i) = byte_value(int(floor(interp(dg, g_d, rx, ry, rz) + 0.5_real32)))
        data_threshold_d(i) = .true.
      else
        ivf_d(i) = 0_int8; ivg_d(i) = 0_int8; data_threshold_d(i) = .false.
      end if
    end do
  end subroutine spm_reference

end module spm_mod
