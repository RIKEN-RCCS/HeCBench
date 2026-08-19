module geo_math
 use iso_fortran_env,only:real32
 implicit none
!$omp declare target (distance)
contains
 real(real32) function distance(a,b,c,d)
  real(real32),intent(in)::a,b,c,d
  real(real32),parameter::pi=3.141592654_real32,flat=1._real32-6356752.31424518_real32/6378137._real32,ec=6356752.31424518_real32/6378137._real32,ell=1._real32/(6356752.31414_real32/6378137._real32)**2-1._real32,minor=6356752.31424518_real32,eps=.5e-5_real32
  real(real32)::u1,u2,cu1,cu2,dist,baz,faz,x,sx,cx,sy,cy,y,sa,c2a,cz,e,cc,dd
  u1=ec*sin(a*pi/180._real32)/cos(a*pi/180._real32);u2=ec*sin(c*pi/180._real32)/cos(c*pi/180._real32);cu1=1./sqrt(u1*u1+1.);u1=cu1*u1;cu2=1./sqrt(u2*u2+1.);dist=cu1*cu2;baz=dist*u2;faz=baz*u1;x=(d-b)*pi/180._real32
  do
   sx=sin(x);cx=cos(x);u1=cu2*sx;u2=baz-u1*cu2*cx;sy=sqrt(u1*u1+u2*u2);cy=dist*cx+faz;y=atan2(sy,cy);sa=dist*sx/sy;c2a=1.-sa*sa;cz=faz+faz;if(c2a>0.)cz=-cz/c2a+cy;e=cz*cz*2.-1.;cc=((-3.*c2a+4.)*flat+4.)*c2a*flat/16.;dd=x;x=((e*cy*cc+cz)*sy*cc+y)*sa;x=(1.-cc)*x*flat+(d-b)*pi/180._real32;if(abs(dd-x)<=eps)exit
  end do
  x=sqrt(ell*c2a+1.)+1.;x=(x-2.)/x;cc=1.-x;cc=(x*x/4.+1.)/cc;dd=(.375*x*x-1.)*x;x=e*cy;dist=1.-e-e;distance=((((sy*sy*4.-3.)*dist*cz*dd/6.-x)*dd/4.+cz)*sy*dd+y)*cc*minor
 end function
end module
program geodesic
 use iso_fortran_env,only:real32,real64,int64
 use geo_math
 implicit none
 integer,parameter::cities=2097152,refs=6
 integer::rep,i,j,c,ios,idx(0:5)=[436483,1952407,627919,377884,442703,1863423]
 integer(int64)::n,s,e,r
 real(real32),allocatable::inp(:,:),out(:),expect(:)
 real(real64)::us,err
 character(64)::arg
 if(command_argument_count()/=1)then;write(*,'(a)')'Usage ./main <repeat>';stop 1;end if;call get_command_argument(1,arg);read(arg,*)rep;n=int(cities,int64)*refs;allocate(inp(0:3,0:n-1),out(0:n-1),expect(0:n-1))
 write(*,'(a)')'Reading city locations from file ../geodesic-omp/locations.txt...';open(10,file='../geodesic-omp/locations.txt',status='old',iostat=ios);if(ios/=0)error stop 'missing ../geodesic-omp/locations.txt';do i=0,cities-1;read(10,*,iostat=ios)inp(0,i),inp(1,i);if(ios/=0)error stop 'invalid locations input';end do;close(10)
 do c=1,refs-1;inp(:,c*cities:(c+1)*cities-1)=inp(:,0:cities-1);end do;do c=0,refs-1;do j=c*cities,(c+1)*cities-1;inp(2,j)=inp(0,idx(c)-1);inp(3,j)=inp(1,idx(c)-1);end do;end do
 do i=0,n-1;expect(i)=distance(inp(0,i),inp(1,i),inp(2,i),inp(3,i));end do
!$omp target data map(to:inp(0:3,0:n-1)) map(from:out(0:n-1))
 call system_clock(s,r);do j=1,rep
!$omp target teams distribute parallel do thread_limit(256)
 do i=0,n-1;out(i)=distance(inp(0,i),inp(1,i),inp(2,i),inp(3,i));end do
!$omp end target teams distribute parallel do
 end do;call system_clock(e);us=real(e-s,real64)*1d6/real(r,real64)/rep;write(*,'(a,f0.6,a)')'Average kernel execution time ',us,' (us)'
!$omp end target data
 err=maxval(abs(out-expect));write(*,'(a,f0.6)')'The maximum error in distance for single precision is ',err
end program
