module laplace_ops
 use iso_fortran_env,only:real32
 implicit none
 real(real32),parameter::omega=1.85_real32
contains
 subroutine red(ap,aw,ae,as,an,b,black,redv,residual,norm,num,rows)
 integer,intent(in)::num,rows;real(real32),intent(in)::ap(:),aw(:),ae(:),as(:),an(:),b(:),black(:);real(real32),intent(inout)::redv(:),residual(:);real(real32),intent(out)::norm
 integer::row,col,ir,ind;real(real32)::old,res,newv
 norm=0.0
 !$omp target teams distribute parallel do collapse(2) num_threads(256) private(ir,ind,old,res,newv)
 do row=1,num/2;do col=1,num
  ir=col*rows+row;ind=2*row-iand(col,1)-1+num*(col-1)+1;old=redv(ir+1)
  res=b(ind)+aw(ind)*black(row+(col-1)*rows+1)+ae(ind)*black(row+(col+1)*rows+1)+as(ind)*black(row-iand(col,1)+col*rows+1)+an(ind)*black(row+iand(col+1,1)+col*rows+1)
  newv=old*(1.0-omega)+omega*res/ap(ind);redv(ir+1)=newv;residual(ir+1)=(newv-old)**2
 end do;end do
 !$omp end target teams distribute parallel do
 !$omp target teams distribute parallel do reduction(+:norm)
 do ir=1,size(residual);norm=norm+residual(ir);end do
 !$omp end target teams distribute parallel do
 end subroutine
 subroutine black_step(ap,aw,ae,as,an,b,redv,black,residual,norm,num,rows)
 integer,intent(in)::num,rows;real(real32),intent(in)::ap(:),aw(:),ae(:),as(:),an(:),b(:),redv(:);real(real32),intent(inout)::black(:),residual(:);real(real32),intent(inout)::norm
 integer::row,col,ib,ind;real(real32)::old,res,newv
 !$omp target teams distribute parallel do collapse(2) num_threads(256) private(ib,ind,old,res,newv)
 do row=1,num/2;do col=1,num
  ib=col*rows+row;ind=2*row-iand(col+1,1)-1+num*(col-1)+1;old=black(ib+1)
  res=b(ind)+aw(ind)*redv(row+(col-1)*rows+1)+ae(ind)*redv(row+(col+1)*rows+1)+as(ind)*redv(row-iand(col+1,1)+col*rows+1)+an(ind)*redv(row+iand(col,1)+col*rows+1)
  newv=old*(1.0-omega)+omega*res/ap(ind);black(ib+1)=newv;residual(ib+1)=(newv-old)**2
 end do;end do
 !$omp end target teams distribute parallel do
 !$omp target teams distribute parallel do reduction(+:norm)
 do ib=1,size(residual);norm=norm+residual(ib);end do
 !$omp end target teams distribute parallel do
 end subroutine
end module
program laplace
 use iso_fortran_env,only:real32,real64
 use laplace_ops
 implicit none
 integer::num,rows,cols,size,size_temp,i,row,col,ind,iter,itmax,clk0,clk1,rate
 real(real32)::dx,dy,norm,nref,err;real(real32),allocatable::ap(:),aw(:),ae(:),as(:),an(:),b(:),redv(:),black(:),residual(:),redref(:),blackref(:)
 character(len=32)::arg
 num=1024;if(command_argument_count()==1)then;call get_command_argument(1,arg);read(arg,*)num;end if
 rows=num/2+2;cols=num+2;size=num*num;size_temp=rows*cols;allocate(ap(size),aw(size),ae(size),as(size),an(size),b(size),redv(size_temp),black(size_temp),residual(size_temp),redref(size_temp),blackref(size_temp));redv=0;black=0;residual=0;redref=0;blackref=0;dx=1.0_real32/num;dy=dx
 do col=0,num-1;do row=0,num-1;ind=col*num+row+1;b(ind)=0
  aw(ind)=merge(0.0_real32,.01_real32*dy/dx,col==0);ae(ind)=merge(0.0_real32,.01_real32*dy/dx,col==num-1);as(ind)=merge(0.0_real32,.01_real32*dx/dy,row==0);an(ind)=merge(0.0_real32,.01_real32*dx/dy,row==num-1)
  if(row==num-1)b(ind)=.02_real32*dx/dy;ap(ind)=aw(ind)+ae(ind)+as(ind)+an(ind);if(col==0 .or. col==num-1)ap(ind)=ap(ind)+.02_real32*dy/dx;if(row==0 .or. row==num-1)ap(ind)=ap(ind)+.02_real32*dx/dy
 end do;end do
 itmax=1000000;call system_clock(clk0,rate)
 !$omp target data map(to:ap,aw,ae,as,an,b) map(tofrom:redv,black,residual)
 do iter=1,itmax;call red(ap,aw,ae,as,an,b,black,redv,residual,norm,num,rows);call black_step(ap,aw,ae,as,an,b,redv,black,residual,norm,num,rows);norm=sqrt(norm/real(size,real32));if(norm<1.e-6)exit;end do
 !$omp end target data
 call system_clock(clk1);write(*,'(a,i0,a,f10.4,a)')'Total time for ',iter,' iterations: ',real(clk1-clk0,real64)/rate,' s'
 do i=1,iter;call host_step(ap,aw,ae,as,an,b,blackref,redref,residual,nref,num,rows,.true.);call host_step(ap,aw,ae,as,an,b,redref,blackref,residual,nref,num,rows,.false.);end do
 err=maxval(abs(redv-redref))+maxval(abs(black-blackref));if(err<1.e-3_real32)then;print '(a)','PASS';else;print '(a,f12.6)','FAIL, max error = ',err;stop 2;end if
contains
 subroutine host_step(ap,aw,ae,as,an,b,other,self,res,norm,num,rows,isred)
  integer,intent(in)::num,rows;logical,intent(in)::isred;real(real32),intent(in)::ap(:),aw(:),ae(:),as(:),an(:),b(:),other(:);real(real32),intent(inout)::self(:),res(:);real(real32),intent(out)::norm
  integer::rr,cc,ii,gg;real(real32)::old,v,newv
  norm=0;do rr=1,num/2;do cc=1,num
   ii=cc*rows+rr
   if(isred)then;gg=2*rr-iand(cc,1)-1+num*(cc-1)+1;v=b(gg)+aw(gg)*other(rr+(cc-1)*rows+1)+ae(gg)*other(rr+(cc+1)*rows+1)+as(gg)*other(rr-iand(cc,1)+cc*rows+1)+an(gg)*other(rr+iand(cc+1,1)+cc*rows+1)
   else;gg=2*rr-iand(cc+1,1)-1+num*(cc-1)+1;v=b(gg)+aw(gg)*other(rr+(cc-1)*rows+1)+ae(gg)*other(rr+(cc+1)*rows+1)+as(gg)*other(rr-iand(cc+1,1)+cc*rows+1)+an(gg)*other(rr+iand(cc,1)+cc*rows+1);end if
   old=self(ii+1);newv=old*(1-omega)+omega*v/ap(gg);self(ii+1)=newv;res(ii+1)=(newv-old)**2;norm=norm+res(ii+1)
  end do;end do
 end subroutine
end program
