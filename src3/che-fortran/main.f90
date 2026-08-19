module che_kernel
  use iso_fortran_env, only : real64
  implicit none
  integer, parameter :: data_x_size=256, data_y_size=256, data_z_size=256
  integer, parameter :: dp=real64
  !$omp declare target (laplacian, gradient_x, gradient_y, gradient_z, free_energy)
contains
  pure real(dp) function laplacian(c,dx,dy,dz,x,y,z)
    real(dp), intent(in) :: c(0:data_x_size-1,0:data_y_size-1,0:data_z_size-1),dx,dy,dz
    integer, intent(in) :: x,y,z
    integer :: xp,xn,yp,yn,zp,zn
    xp=x+1; xn=x-1; yp=y+1; yn=y-1; zp=z+1; zn=z-1
    if(xp>data_x_size-1) xp=0
    if(yp>data_y_size-1) yp=0
    if(zp>data_z_size-1) zp=0
    if(xn<0) xn=data_x_size-1
    if(yn<0) yn=data_y_size-1
    if(zn<0) zn=data_z_size-1
    laplacian=(c(xp,y,z)+c(xn,y,z)-2.0_dp*c(x,y,z))/(dx*dx) + &
      (c(x,yp,z)+c(x,yn,z)-2.0_dp*c(x,y,z))/(dy*dy) + &
      (c(x,y,zp)+c(x,y,zn)-2.0_dp*c(x,y,z))/(dz*dz)
  end function laplacian

  pure real(dp) function gradient_x(phi,dx,dy,dz,x,y,z)
    real(dp), intent(in) :: phi(0:data_x_size-1,0:data_y_size-1,0:data_z_size-1),dx,dy,dz
    integer, intent(in) :: x,y,z
    integer :: xp,xn
    xp=x+1; xn=x-1
    if(xp>data_x_size-1) xp=0
    if(xn<0) xn=data_x_size-1
    gradient_x=(phi(xp,y,z)-phi(xn,y,z))/(2.0_dp*dx)
  end function gradient_x

  pure real(dp) function gradient_y(phi,dx,dy,dz,x,y,z)
    real(dp), intent(in) :: phi(0:data_x_size-1,0:data_y_size-1,0:data_z_size-1),dx,dy,dz
    integer, intent(in) :: x,y,z
    integer :: yp,yn
    yp=y+1; yn=y-1
    if(yp>data_y_size-1) yp=0
    if(yn<0) yn=data_y_size-1
    gradient_y=(phi(x,yp,z)-phi(x,yn,z))/(2.0_dp*dy)
  end function gradient_y

  pure real(dp) function gradient_z(phi,dx,dy,dz,x,y,z)
    real(dp), intent(in) :: phi(0:data_x_size-1,0:data_y_size-1,0:data_z_size-1),dx,dy,dz
    integer, intent(in) :: x,y,z
    integer :: zp,zn
    zp=z+1; zn=z-1
    if(zp>data_z_size-1) zp=0
    if(zn<0) zn=data_z_size-1
    gradient_z=(phi(x,y,zp)-phi(x,y,zn))/(2.0_dp*dz)
  end function gradient_z

  pure real(dp) function free_energy(c,e_aa,e_bb,e_ab)
    real(dp), intent(in) :: c,e_aa,e_bb,e_ab
    free_energy=(9.0_dp/4.0_dp)*((c*c+2.0_dp*c+1.0_dp)*e_aa+(c*c-2.0_dp*c+1.0_dp)*e_bb+ &
      2.0_dp*(1.0_dp-c*c)*e_ab)+(3.0_dp/2.0_dp)*c*c+(3.0_dp/12.0_dp)*c*c*c*c
  end function free_energy
  subroutine chemical_potential(c,mu,dx,dy,dz,gamma,e_aa,e_bb,e_ab)
    real(dp), intent(in) :: c(0:data_x_size-1,0:data_y_size-1,0:data_z_size-1),dx,dy,dz,gamma,e_aa,e_bb,e_ab
    real(dp), intent(out) :: mu(0:data_x_size-1,0:data_y_size-1,0:data_z_size-1)
    integer :: x,y,z
    !$omp target teams distribute parallel do collapse(3)
    do z=0,data_z_size-1
      do y=0,data_y_size-1
        do x=0,data_x_size-1
          mu(x,y,z)=4.5_dp*((c(x,y,z)+1.0_dp)*e_aa+(c(x,y,z)-1.0_dp)*e_bb-2.0_dp*c(x,y,z)*e_ab) + &
            3.0_dp*c(x,y,z)+c(x,y,z)*c(x,y,z)*c(x,y,z)-gamma*laplacian(c,dx,dy,dz,x,y,z)
        end do
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine chemical_potential

  subroutine local_free_energy_functional(c,f,dx,dy,dz,gamma,e_aa,e_bb,e_ab)
    real(dp), intent(in) :: c(0:data_x_size-1,0:data_y_size-1,0:data_z_size-1),dx,dy,dz,gamma,e_aa,e_bb,e_ab
    real(dp), intent(out) :: f(0:data_x_size-1,0:data_y_size-1,0:data_z_size-1)
    integer :: x,y,z
    !$omp target teams distribute parallel do collapse(3)
    do z=0,data_z_size-1
      do y=0,data_y_size-1
        do x=0,data_x_size-1
          f(x,y,z)=free_energy(c(x,y,z),e_aa,e_bb,e_ab)+(gamma/2.0_dp)*(gradient_x(c,dx,dy,dz,x,y,z)**2 + &
            gradient_y(c,dx,dy,dz,x,y,z)**2+gradient_z(c,dx,dy,dz,x,y,z)**2)
        end do
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine local_free_energy_functional

  subroutine cahn_hilliard(cnew,cold,mu,d,dt,dx,dy,dz)
    real(dp), intent(in) :: cold(0:data_x_size-1,0:data_y_size-1,0:data_z_size-1),mu(0:data_x_size-1,0:data_y_size-1,0:data_z_size-1),d,dt,dx,dy,dz
    real(dp), intent(out) :: cnew(0:data_x_size-1,0:data_y_size-1,0:data_z_size-1)
    integer :: x,y,z
    !$omp target teams distribute parallel do collapse(3)
    do z=0,data_z_size-1
      do y=0,data_y_size-1
        do x=0,data_x_size-1
          cnew(x,y,z)=cold(x,y,z)+dt*d*laplacian(mu,dx,dy,dz,x,y,z)
        end do
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine cahn_hilliard

  subroutine swap_fields(cnew,cold)
    real(dp), intent(inout) :: cnew(0:data_x_size-1,0:data_y_size-1,0:data_z_size-1),cold(0:data_x_size-1,0:data_y_size-1,0:data_z_size-1)
    integer :: x,y,z
    real(dp) :: temporary
    !$omp target teams distribute parallel do collapse(3) private(temporary)
    do z=0,data_z_size-1
      do y=0,data_y_size-1
        do x=0,data_x_size-1
          temporary=cnew(x,y,z); cnew(x,y,z)=cold(x,y,z); cold(x,y,z)=temporary
        end do
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine swap_fields

  subroutine initialize(c)
    use iso_c_binding, only : c_int
    real(dp), intent(out) :: c(0:data_x_size-1,0:data_y_size-1,0:data_z_size-1)
    interface
      subroutine c_srand(seed) bind(C,name='srand')
        import :: c_int
        integer(c_int), value :: seed
      end subroutine c_srand
      function c_rand() bind(C,name='rand') result(value)
        import :: c_int
        integer(c_int) :: value
      end function c_rand
    end interface
    integer :: x,y,z
    call c_srand(2_c_int)
    do z=0,data_z_size-1
      do y=0,data_y_size-1
        do x=0,data_x_size-1
          c(x,y,z)=-1.0_dp+2.0_dp*real(c_rand(),dp)/2147483647.0_dp
        end do
      end do
    end do
  end subroutine initialize

  real(dp) function integral(c,nx,ny,nz)
    real(dp), intent(in) :: c(0:data_x_size-1,0:data_y_size-1,0:data_z_size-1)
    integer, intent(in) :: nx,ny,nz
    integer :: x,y,z
    integral=0.0_dp
    do z=0,nz-1
      do y=0,ny-1
        do x=0,nx-1
          integral=integral+c(x,y,z)
        end do
      end do
    end do
  end function integral
end module che_kernel

program main
  use iso_fortran_env, only : real64, int64
  use omp_lib, only : omp_get_wtime
  use che_kernel
  implicit none
  integer :: argc,t_f,t_freq,t,ios
  integer, parameter :: nx=data_x_size,ny=data_y_size,nz=data_z_size,vol=nx*ny*nz
  real(dp), parameter :: dx=1.0_dp,dy=1.0_dp,dz=1.0_dp,dt=0.01_dp,e_aa=-(2.0_dp/9.0_dp),e_bb=-(2.0_dp/9.0_dp),e_ab=2.0_dp/9.0_dp,gamma=0.5_dp,diffusion=1.0_dp
  real(dp), allocatable :: cold(:,:,:),cnew(:,:,:),muold(:,:,:),fold(:,:,:)
  real(dp) :: start_time,end_time,integral_c,integral_mu,integral_f
  character(len=64) :: argument
  integer :: unit_c,unit_mu,unit_f
  argc=command_argument_count()
  if(argc < 1) error stop 1
  call get_command_argument(1,argument); read(argument,*,iostat=ios) t_f
  if(ios /= 0) error stop 1
  t_freq=t_f
  allocate(cold(0:nx-1,0:ny-1,0:nz-1),cnew(0:nx-1,0:ny-1,0:nz-1),muold(0:nx-1,0:ny-1,0:nz-1),fold(0:nx-1,0:ny-1,0:nz-1))
  open(newunit=unit_c,file='./out/integral_c.txt',status='replace',action='write')
  open(newunit=unit_mu,file='./out/integral_mu.txt',status='replace',action='write')
  open(newunit=unit_f,file='./out/integral_f.txt',status='replace',action='write')
  call initialize(cold)
  !$omp target data map(to:cold(0:nx-1,0:ny-1,0:nz-1)) map(alloc:cnew(0:nx-1,0:ny-1,0:nz-1),muold(0:nx-1,0:ny-1,0:nz-1),fold(0:nx-1,0:ny-1,0:nz-1))
  start_time=omp_get_wtime()
  integral_c=0.0_dp; integral_mu=0.0_dp; integral_f=0.0_dp
  do t=0,t_f-1
    call chemical_potential(cold,muold,dx,dy,dz,gamma,e_aa,e_bb,e_ab)
    call local_free_energy_functional(cold,fold,dx,dy,dz,gamma,e_aa,e_bb,e_ab)
    call cahn_hilliard(cnew,cold,muold,diffusion,dt,dx,dy,dz)
    if(t > 0) then
      if(mod(t,t_freq-1) == 0) then
        !$omp target update from(cnew(0:nx-1,0:ny-1,0:nz-1))
        !$omp target update from(muold(0:nx-1,0:ny-1,0:nz-1))
        !$omp target update from(fold(0:nx-1,0:ny-1,0:nz-1))
        integral_c=integral(cnew,nx,ny,nz); write(unit_c,'(i0,a,es24.16)') t,',',integral_c
        integral_mu=integral(muold,nx,ny,nz); write(unit_mu,'(i0,a,es24.16)') t,',',integral_mu
        integral_f=integral(fold,nx,ny,nz); write(unit_f,'(i0,a,es24.16)') t,',',integral_f
      end if
    end if
    call swap_fields(cnew,cold)
  end do
  end_time=omp_get_wtime()
  print '(a,i0,a,f0.3,a)','Kernel exeuction time on the GPU (',t_f,' iterations) = ',end_time-start_time,' (s)'
  !$omp end target data
  close(unit_c); close(unit_mu); close(unit_f)
  deallocate(cold,cnew,muold,fold)
end program main
