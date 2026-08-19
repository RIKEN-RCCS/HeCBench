module haversine_distance
  use iso_fortran_env, only: int64, real64
  implicit none
  real(real64), parameter :: degree_to_radian=acos(-1.0_real64)/180.0_real64, earth_radius_km=6371.0_real64
contains
  subroutine distance_device(loc, dist, n, iteration)
    integer(int64), intent(in) :: n
    integer, intent(in) :: iteration
    real(real64), intent(in) :: loc(0:3,0:n-1)
    real(real64), intent(out) :: dist(0:n-1)
    integer(int64) :: p
    integer :: i
    real(real64) :: ay, ax, by, bx, xx, yy, sinysqrd, sinxsqrd, scale
!$omp target data map(to:loc(0:3,0:n-1)) map(from:dist(0:n-1))
    do i=1,iteration
!$omp target teams distribute parallel do thread_limit(256) private(ay,ax,by,bx,xx,yy,sinysqrd,sinxsqrd,scale)
      do p=0,n-1
        ay=loc(0,p)*degree_to_radian; ax=loc(1,p)*degree_to_radian
        by=loc(2,p)*degree_to_radian; bx=loc(3,p)*degree_to_radian
        xx=(bx-ax)/2.0_real64; yy=(by-ay)/2.0_real64
        sinysqrd=sin(yy)*sin(yy); sinxsqrd=sin(xx)*sin(xx); scale=cos(ay)*cos(by)
        dist(p)=2.0_real64*earth_radius_km*asin(sqrt(sinysqrd+sinxsqrd*scale))
      end do
!$omp end target teams distribute parallel do
    end do
!$omp end target data
  end subroutine
end module
