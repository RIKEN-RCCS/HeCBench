program epistasis
  use, intrinsic :: iso_fortran_env, only : int32, int64, real32, real64
  use, intrinsic :: iso_c_binding, only : c_int
  use omp_lib, only : omp_get_wtime
  implicit none
!$omp declare target (gamma_value, score_pair, add_counts, count9)
  integer(int32) :: patients,snps,iterations,pp_zero,pp_one,phen_ones,mask_zero,mask_one,snps_m,i,j,k,x,ios,p1,p2
  integer(int32),allocatable :: snp(:),phenotype(:),snp_trans(:),zero_data(:),one_data(:),zero_trans(:),one_trans(:)
  real(real32),allocatable :: scores(:),scores_ref(:)
  real(real64)::begin_time,elapsed
  character(len=64)::argument
  interface
    subroutine c_srand(seed) bind(C,name='srand');import c_int;integer(c_int),value::seed;end subroutine
    function c_rand() bind(C,name='rand') result(value);import c_int;integer(c_int)::value;end function
  end interface
  if(command_argument_count()/=3)then;write(*,'(a)')'Usage: ./main <number of samples> <number of SNPs> <repeat>';stop 1;end if
  call get_command_argument(1,argument);read(argument,*,iostat=ios)patients;if(ios/=0.or.patients<=0)error stop 'invalid samples'
  call get_command_argument(2,argument);read(argument,*,iostat=ios)snps;if(ios/=0.or.snps<=1)error stop 'invalid SNPs'
  call get_command_argument(3,argument);read(argument,*,iostat=ios)iterations;if(ios/=0.or.iterations<=0)error stop 'invalid repeat'
  allocate(snp(0:patients*snps-1),phenotype(0:patients),snp_trans(0:patients*snps-1))
  call c_srand(100_c_int)
  do i=0,patients-1;do j=0,snps-1;snp(i*snps+j)=mod(c_rand(),3_c_int);end do;phenotype(i)=mod(c_rand(),2_c_int);end do
  do i=0,patients-1;do j=0,snps-1;snp_trans(j*patients+i)=snp(i*snps+j);end do;end do
  phen_ones=0;do i=0,patients-1;if(phenotype(i)==1)phen_ones=phen_ones+1;end do
  pp_zero=(patients-phen_ones+31)/32;pp_one=(phen_ones+31)/32
  allocate(zero_data(0:snps*pp_zero*2-1),one_data(0:snps*pp_one*2-1));zero_data=0;one_data=0
  do i=0,snps-1;call pack_snp(i,patients,pp_zero,pp_one,snp_trans,phenotype,zero_data,one_data);end do
  mask_zero=int(z'FFFFFFFF',int32);do x=patients-phen_ones,pp_zero*32-1;mask_zero=shiftr(mask_zero,1);end do
  mask_one=int(z'FFFFFFFF',int32);do x=phen_ones,pp_one*32-1;mask_one=shiftr(mask_one,1);end do
  allocate(zero_trans(0:snps*pp_zero*2-1),one_trans(0:snps*pp_one*2-1))
  do i=0,snps-1;do j=0,pp_zero-1;zero_trans((j*snps+i)*2)=zero_data((i*pp_zero+j)*2);zero_trans((j*snps+i)*2+1)=zero_data((i*pp_zero+j)*2+1);end do;end do
  do i=0,snps-1;do j=0,pp_one-1;one_trans((j*snps+i)*2)=one_data((i*pp_one+j)*2);one_trans((j*snps+i)*2+1)=one_data((i*pp_one+j)*2+1);end do;end do
  allocate(scores(0:snps*snps-1),scores_ref(0:snps*snps-1));scores=huge(0.0_real32);scores_ref=huge(0.0_real32)
  snps_m=snps;do while(mod(snps_m,64)/=0);snps_m=snps_m+1;end do
!$omp target data map(to:zero_trans(0:snps*pp_zero*2-1),one_trans(0:snps*pp_one*2-1)) map(tofrom:scores(0:snps*snps-1))
  begin_time=omp_get_wtime()
  do k=1,iterations
!$omp target teams distribute parallel do collapse(2) thread_limit(64)
    do i=0,snps_m-1
      do j=0,snps_m-1
        if(j>i .and. i<snps .and. j<snps) scores(i*snps+j)=score_pair(zero_trans,one_trans,i,j,snps,pp_zero,pp_one,mask_zero,mask_one)
      end do
    end do
!$omp end target teams distribute parallel do
  end do
  elapsed=(omp_get_wtime()-begin_time)/real(iterations,real64)
!$omp end target data
  write(*,'(a,f0.6,a)')'Average kernel execution time: ',elapsed,' (s)'
  do i=0,snps-1;do j=0,snps-1;if(j>i)scores_ref(i*snps+j)=score_pair(zero_trans,one_trans,i,j,snps,pp_zero,pp_one,mask_zero,mask_one);end do;end do
  p1=min_score(scores);p2=min_score(scores_ref)
  if(p1==p2 .and. abs(scores(p1)-scores_ref(p2))<1.0e-3_real32)then;write(*,'(a)')'PASS';else;write(*,'(a)')'FAIL';end if
  deallocate(snp,phenotype,snp_trans,zero_data,one_data,zero_trans,one_trans,scores,scores_ref)
contains
  subroutine pack_snp(which,np,pz,po,trans,ph,zeros,ones)
    integer(int32),intent(in)::which,np,pz,po,trans(0:),ph(0:);integer(int32),intent(inout)::zeros(0:),ones(0:)
    integer(int32)::j,temp,xz,xo,nz,no
    xz=-1;xo=-1;nz=0;no=0
    do j=0,np-1
      temp=trans(which*np+j)
      if(ph(j)==1)then
        if(mod(no,32)==0)xo=xo+1;ones(which*po*2+xo*2)=shiftl(ones(which*po*2+xo*2),1);ones(which*po*2+xo*2+1)=shiftl(ones(which*po*2+xo*2+1),1)
        if(temp==0.or.temp==1)ones(which*po*2+xo*2+temp)=ior(ones(which*po*2+xo*2+temp),1);no=no+1
      else
        if(mod(nz,32)==0)xz=xz+1;zeros(which*pz*2+xz*2)=shiftl(zeros(which*pz*2+xz*2),1);zeros(which*pz*2+xz*2+1)=shiftl(zeros(which*pz*2+xz*2+1),1)
        if(temp==0.or.temp==1)zeros(which*pz*2+xz*2+temp)=ior(zeros(which*pz*2+xz*2+temp),1);nz=nz+1
      end if
    end do
  end subroutine pack_snp
  function gamma_value(n) result(value)
    integer(int32),intent(in)::n;real(real32)::value
    if(n==0)then;value=0.0_real32;else;value=(real(n,real32)+0.5_real32)*log(real(n,real32))-(real(n,real32)-1.0_real32);end if
  end function gamma_value
  subroutine add_counts(ft,base,data,i,j,snps,block,mask)
    integer(int32),intent(inout)::ft(0:);integer(int32),intent(in)::base,data(0:),i,j,snps,block,mask
    integer(int32)::p,si,sj,di,dj,t00,t01,t02,t10,t11,t12,t20,t21,t22,last
    if(block>1)then
      do p=0,2*(block-1)*snps-1,2*snps
        si=i*2+p;sj=j*2+p;di=not(ior(data(si),data(si+1)));dj=not(ior(data(sj),data(sj+1)));call count9(ft,base,data(si),data(si+1),data(sj),data(sj+1),di,dj)
      end do
    end if
    last=2*block*snps-2*snps;si=i*2+last;sj=j*2+last;di=iand(not(ior(data(si),data(si+1))),mask);dj=iand(not(ior(data(sj),data(sj+1))),mask)
    call count9(ft,base,data(si),data(si+1),data(sj),data(sj+1),di,dj)
  end subroutine add_counts
  subroutine count9(ft,b,a0,a1,b0,b1,a2,b2)
    integer(int32),intent(inout)::ft(0:);integer(int32),intent(in)::b,a0,a1,b0,b1,a2,b2
    ft(b)=ft(b)+popcnt(iand(a0,b0));ft(b+1)=ft(b+1)+popcnt(iand(a0,b1));ft(b+2)=ft(b+2)+popcnt(iand(a0,b2));ft(b+3)=ft(b+3)+popcnt(iand(a1,b0));ft(b+4)=ft(b+4)+popcnt(iand(a1,b1));ft(b+5)=ft(b+5)+popcnt(iand(a1,b2));ft(b+6)=ft(b+6)+popcnt(iand(a2,b0));ft(b+7)=ft(b+7)+popcnt(iand(a2,b1));ft(b+8)=ft(b+8)+popcnt(iand(a2,b2))
  end subroutine count9
  function score_pair(zeros,ones,i,j,ns,pz,po,mz,mo) result(score)
    integer(int32),intent(in)::zeros(0:),ones(0:),i,j,ns,pz,po,mz,mo;integer(int32)::ft(0:17),k
    real(real32)::score
    ft=0;call add_counts(ft,0,zeros,i,j,ns,pz,mz);call add_counts(ft,9,ones,i,j,ns,po,mo);score=0.0_real32
    do k=0,8;score=score+gamma_value(ft(k)+ft(9+k)+1)-gamma_value(ft(k))-gamma_value(ft(9+k));end do
    score=abs(score);if(score==0.0_real32)score=huge(0.0_real32)
  end function score_pair
  function min_score(array) result(index)
    real(real32),intent(in)::array(0:);integer(int32)::index,i;real(real32)::best
    index=0;best=array(0);do i=1,size(array)-1;if(best>array(i))then;best=array(i);index=i;end if;end do
  end function min_score
end program epistasis
