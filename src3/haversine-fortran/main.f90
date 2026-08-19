program haversine
  use iso_fortran_env, only: int64, real64
  use haversine_distance
  implicit none
  integer :: argc, repeat, unit, ios, city, c, j
  integer(int64), parameter :: num_cities=2097152_int64, num_ref_cities=6_int64
  integer(int64), parameter :: index_map(0:5)=[436483_int64,1952407_int64,627919_int64,377884_int64,442703_int64,1863423_int64]
  integer(int64) :: n, index, p
  character(len=1024) :: filename, arg
  real(real64) :: lat, lon, maximum_error, ay, ax, by, bx, xx, yy, sinysqrd, sinxsqrd, scale
  real(real64), allocatable :: input(:,:), output(:), expected(:)
  argc=command_argument_count()
  if(argc/=2) then; print '(a)', 'Usage: ./main <file> <repeat>'; stop 1; end if
  call get_command_argument(1,filename); call get_command_argument(2,arg); read(arg,*) repeat
  print '(a,a,a)', 'Reading city locations from file ',trim(filename),'...'
  open(newunit=unit,file=trim(filename),status='old',action='read',iostat=ios)
  if(ios/=0) then; print '(a)', 'Error opening the file'; stop 1; end if
  n=num_cities*num_ref_cities; allocate(input(0:3,0:n-1),output(0:n-1),expected(0:n-1))
  city=0
  do
    read(unit,*,iostat=ios) lat,lon
    if(ios/=0 .or. city==num_cities) exit
    input(0,city)=lat; input(1,city)=lon; city=city+1
  end do
  close(unit)
  if(city/=num_cities) then; print '(a)', 'Input does not contain enough city locations'; stop 1; end if
  do c=1,int(num_ref_cities)-1
    input(:,c*num_cities:(c+1)*num_cities-1)=input(:,0:num_cities-1)
  end do
  do c=0,int(num_ref_cities)-1
    index=index_map(c)-1_int64
    do p=int(c,int64)*num_cities,(int(c,int64)+1_int64)*num_cities-1
      input(2,p)=input(0,index); input(3,p)=input(1,index)
    end do
  end do
  do p=0,n-1
    ay=input(0,p)*degree_to_radian; ax=input(1,p)*degree_to_radian
    by=input(2,p)*degree_to_radian; bx=input(3,p)*degree_to_radian
    xx=(bx-ax)/2.0_real64; yy=(by-ay)/2.0_real64
    sinysqrd=sin(yy)*sin(yy); sinxsqrd=sin(xx)*sin(xx); scale=cos(ay)*cos(by)
    expected(p)=2.0_real64*earth_radius_km*asin(sqrt(sinysqrd+sinxsqrd*scale))
  end do
  call distance_device(input,output,n,repeat)
  maximum_error=0.0_real64
  do p=0,n-1; maximum_error=max(maximum_error,abs(output(p)-expected(p))); end do
  print '(a,f0.6)', 'The maximum error in distance is ',maximum_error
end program
