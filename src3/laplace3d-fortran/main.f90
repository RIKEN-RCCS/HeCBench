module laplace3d_kernel
 use iso_fortran_env,only:real32
 implicit none
contains
 subroutine stencil(nx,ny,nz,u1,u2)
  integer,intent(in)::nx,ny,nz;real(real32),intent(in)::u1(:);real(real32),intent(out)::u2(:)
  integer::i,j,k,ind
  !$omp target teams distribute parallel do collapse(3) thread_limit(256) private(ind)
  do k=0,nz-1;do j=0,ny-1;do i=0,nx-1
   ind=i+j*nx+k*nx*ny+1
   if(i==0.or.i==nx-1.or.j==0.or.j==ny-1.or.k==0.or.k==nz-1)then;u2(ind)=u1(ind)
   else;u2(ind)=(u1(ind-1)+u1(ind+1)+u1(ind-nx)+u1(ind+nx)+u1(ind-nx*ny)+u1(ind+nx*ny))/6.0_real32;end if
  end do;end do;end do
  !$omp end target teams distribute parallel do
 end subroutine
end module
program main
 use iso_fortran_env,only:real32,real64;use laplace3d_kernel;implicit none
 integer::nx,ny,nz,reps,verify,i,j,k,ind,r,c0,c1,rate;character(len=32)::arg
 real(real32),allocatable::u1(:),u2(:),u3(:),r1(:),r2(:),r3(:);real(real32)::err
 if(command_argument_count()/=5)stop 1
 call get_command_argument(1,arg);read(arg,*)nx;call get_command_argument(2,arg);read(arg,*)ny;call get_command_argument(3,arg);read(arg,*)nz;call get_command_argument(4,arg);read(arg,*)reps;call get_command_argument(5,arg);read(arg,*)verify
 if(nx<=0.or.mod(nx,32)/=0.or.ny<=0.or.nz<=0.or.reps<=0)stop 1
 print '(a,3(i0,a))','Grid dimensions: ',nx,' x ',ny,' x ',nz;allocate(u1(nx*ny*nz),u2(nx*ny*nz),u3(nx*ny*nz),r1(nx*ny*nz),r2(nx*ny*nz),r3(nx*ny*nz))
 do k=0,nz-1;do j=0,ny-1;do i=0,nx-1;ind=i+j*nx+k*nx*ny+1;if(i==0.or.i==nx-1.or.j==0.or.j==ny-1.or.k==0.or.k==nz-1)then;u1(ind)=1;u2(ind)=1;else;u1(ind)=0;u2(ind)=0;end if;end do;end do;end do
 r1=u1;r2=u2
 !$omp target data map(tofrom:u1,u2)
 call stencil(nx,ny,nz,u1,u2);call system_clock(c0,rate);do r=1,reps;if(mod(r,2)==1)then;call stencil(nx,ny,nz,u1,u2);else;call stencil(nx,ny,nz,u2,u1);end if;end do;call system_clock(c1)
 !$omp end target data
 write(*,'(a,f10.6,a)')'Average kernel execution time: ',real(c1-c0,real64)/(rate*reps),' (s)'
 if(verify/=0)then
  call reference(nx,ny,nz,r1,r2)
  do r=1,reps;if(mod(r,2)==1)then;call reference(nx,ny,nz,r1,r2);else;call reference(nx,ny,nz,r2,r1);end if;end do
  if(mod(reps,2)==0)then;err=sqrt(sum((u1-r1)**2)/real(nx*ny*nz,real32));else;err=sqrt(sum((u2-r2)**2)/real(nx*ny*nz,real32));end if
  print '(a,f10.6)','RMS error = ',err;if(err<1.e-3)then;print '(a)','PASS';else;print '(a)','FAIL';stop 2;end if
 else;print '(a)','PASS';end if
contains
 subroutine reference(ax,ay,az,a,b);integer,intent(in)::ax,ay,az;real(real32),intent(in)::a(:);real(real32),intent(out)::b(:);integer::x,y,z,q
  do z=0,az-1;do y=0,ay-1;do x=0,ax-1;q=x+y*ax+z*ax*ay+1;if(x==0.or.x==ax-1.or.y==0.or.y==ay-1.or.z==0.or.z==az-1)then;b(q)=a(q);else;b(q)=(a(q-1)+a(q+1)+a(q-ax)+a(q+ax)+a(q-ax*ay)+a(q+ax*ay))/6;end if;end do;end do;end do
 end subroutine
end program
