module gsimulation_mod
  use particle_mod
  implicit none
contains
  subroutine init_particles(p, n)
    type(particle), intent(out) :: p(0:n-1)
    integer, intent(in) :: n
    integer :: i
    real(real32) :: r(3)
    call random_seed()
    do i = 0, n-1
      call random_number(r); p(i)%pos = r
    end do
    call random_seed()
    do i = 0, n-1
      call random_number(r); p(i)%vel = (2.0_real32*r - 1.0_real32) * 1.0e-3_real32
    end do
    do i = 0, n-1
      p(i)%acc = 0.0_real32
    end do
    call random_seed()
    do i = 0, n-1
      call random_number(r(1)); p(i)%mass = real(n,real32) * r(1)
    end do
  end subroutine

  subroutine start_simulation(n, nsteps)
    integer, intent(in) :: n, nsteps
    type(particle), allocatable, target :: p(:)
    real(real32), allocatable, target :: e(:)
    integer :: s, i, j, nf
    real(real32), parameter :: dt = 0.1_real32, k_softening_squared = 1.0e-3_real32, kg = 6.67259e-11_real32
    real(real32) :: dx, dy, dz, distance_sqr, distance_inv, acc0, acc1, acc2
    real(real64) :: t_all, ts0, elapsed_seconds, total_time, gflops, av, dev, kenergy
    type(particle) :: pi, pj
    allocate(p(0:n-1), e(0:n-1))
    print '(a)', '==============================='
    print '(a)', ' Initialize Gravity Simulation'
    call init_particles(p, n)
    e = 0.0_real32
    total_time = 0.0_real64
    gflops = 1.0e-9_real64 * ((11.0_real64 + 18.0_real64) * n * n + n * 19.0_real64)
    nf = 0; av = 0.0_real64; dev = 0.0_real64; kenergy = 0.0_real64
    t_all = seconds()
    !$omp target data map(to:p(0:n-1)) map(alloc:e(0:n-1))
    do s = 1, nsteps
      ts0 = seconds()
      !$omp target teams distribute parallel do private(j,pi,pj,dx,dy,dz,distance_sqr,distance_inv,acc0,acc1,acc2) thread_limit(256)
      do i = 0, n-1
        pi = p(i)
        acc0 = pi%acc(0); acc1 = pi%acc(1); acc2 = pi%acc(2)
        do j = 0, n-1
          pj = p(j)
          dx = pj%pos(0) - pi%pos(0)
          dy = pj%pos(1) - pi%pos(1)
          dz = pj%pos(2) - pi%pos(2)
          distance_sqr = dx*dx + dy*dy + dz*dz + k_softening_squared
          distance_inv = 1.0_real32 / sqrt(distance_sqr)
          acc0 = acc0 + dx * kg * pj%mass * distance_inv * distance_inv * distance_inv
          acc1 = acc1 + dy * kg * pj%mass * distance_inv * distance_inv * distance_inv
          acc2 = acc2 + dz * kg * pj%mass * distance_inv * distance_inv * distance_inv
        end do
        pi%acc(0) = acc0; pi%acc(1) = acc1; pi%acc(2) = acc2
        p(i) = pi
      end do
      !$omp end target teams distribute parallel do

      !$omp target teams distribute parallel do private(pi) thread_limit(256)
      do i = 0, n-1
        pi = p(i)
        pi%vel(0) = pi%vel(0) + pi%acc(0) * dt
        pi%vel(1) = pi%vel(1) + pi%acc(1) * dt
        pi%vel(2) = pi%vel(2) + pi%acc(2) * dt
        pi%pos(0) = pi%pos(0) + pi%vel(0) * dt
        pi%pos(1) = pi%pos(1) + pi%vel(1) * dt
        pi%pos(2) = pi%pos(2) + pi%vel(2) * dt
        pi%acc = 0.0_real32
        e(i) = pi%mass * (pi%vel(0)*pi%vel(0) + pi%vel(1)*pi%vel(1) + pi%vel(2)*pi%vel(2))
        p(i) = pi
      end do
      !$omp end target teams distribute parallel do

      !$omp target
      do i = 1, n-1
        e(0) = e(0) + e(i)
      end do
      !$omp end target

      elapsed_seconds = seconds() - ts0
      !$omp target update from(e(0:0))
      kenergy = 0.5_real64 * e(0)
      e(0) = 0.0_real32
      nf = nf + 1
      if (nf > 2) then
        av = av + gflops / elapsed_seconds
        dev = dev + gflops * gflops / (elapsed_seconds * elapsed_seconds)
      end if
    end do
    !$omp end target data
    total_time = seconds() - t_all
    if (nf > 2) then
      av = av / real(nf-2,real64)
      dev = sqrt(dev / real(nf-2,real64) - av*av)
    end if
    print *
    print '(a,es14.6)', '# Total Energy        : ', kenergy
    print '(a,f10.6)', '# Total Time (s)      : ', total_time
    print '(a,es14.6,a,es14.6)', '# Average Performance : ', av, ' +- ', dev
    print '(a)', '==============================='
  end subroutine
end module
