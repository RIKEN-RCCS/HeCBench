module lulesh_kernels
  use lulesh_types
  implicit none
contains
  subroutine initialize_domain(dom, nx)
    type(domain_t),intent(inout)::dom
    integer,intent(in)::nx
    integer::i,j,k,elem,n0
    real(RealK)::mesh_edge, tx, ty, tz
    dom%nx=nx; dom%num_node=(nx+1)*(nx+1)*(nx+1); dom%num_elem=nx*nx*nx
    allocate(dom%x(0:dom%num_node-1),dom%y(0:dom%num_node-1),dom%z(0:dom%num_node-1),dom%xd(0:dom%num_node-1),dom%yd(0:dom%num_node-1),dom%zd(0:dom%num_node-1),dom%xdd(0:dom%num_node-1),dom%ydd(0:dom%num_node-1),dom%zdd(0:dom%num_node-1),dom%fx(0:dom%num_node-1),dom%fy(0:dom%num_node-1),dom%fz(0:dom%num_node-1),dom%nodal_mass(0:dom%num_node-1))
    allocate(dom%e(0:dom%num_elem-1),dom%p(0:dom%num_elem-1),dom%q(0:dom%num_elem-1),dom%ql(0:dom%num_elem-1),dom%qq(0:dom%num_elem-1),dom%v(0:dom%num_elem-1),dom%volo(0:dom%num_elem-1),dom%delv(0:dom%num_elem-1),dom%vdov(0:dom%num_elem-1),dom%arealg(0:dom%num_elem-1),dom%ss(0:dom%num_elem-1),dom%elem_mass(0:dom%num_elem-1),dom%nodelist(0:8*dom%num_elem-1))
    mesh_edge=1.125_RealK
    do k=0,nx; do j=0,nx; do i=0,nx
      n0=node_index(i,j,k,nx); dom%x(n0)=mesh_edge*real(i,RealK)/real(nx,RealK); dom%y(n0)=mesh_edge*real(j,RealK)/real(nx,RealK); dom%z(n0)=mesh_edge*real(k,RealK)/real(nx,RealK)
    end do; end do; end do
    dom%xd=0.0_RealK; dom%yd=0.0_RealK; dom%zd=0.0_RealK; dom%xdd=0.0_RealK; dom%ydd=0.0_RealK; dom%zdd=0.0_RealK; dom%fx=0.0_RealK; dom%fy=0.0_RealK; dom%fz=0.0_RealK; dom%nodal_mass=0.0_RealK
    do k=0,nx-1; do j=0,nx-1; do i=0,nx-1
      elem=elem_index(i,j,k,nx)
      dom%nodelist(8*elem+0)=node_index(i,j,k,nx); dom%nodelist(8*elem+1)=node_index(i+1,j,k,nx)
      dom%nodelist(8*elem+2)=node_index(i+1,j+1,k,nx); dom%nodelist(8*elem+3)=node_index(i,j+1,k,nx)
      dom%nodelist(8*elem+4)=node_index(i,j,k+1,nx); dom%nodelist(8*elem+5)=node_index(i+1,j,k+1,nx)
      dom%nodelist(8*elem+6)=node_index(i+1,j+1,k+1,nx); dom%nodelist(8*elem+7)=node_index(i,j+1,k+1,nx)
      dom%volo(elem)=(mesh_edge/real(nx,RealK))**3; dom%elem_mass(elem)=dom%volo(elem); dom%v(elem)=1.0_RealK
      dom%e(elem)=0.0_RealK; dom%p(elem)=0.0_RealK; dom%q(elem)=0.0_RealK; dom%ql(elem)=0.0_RealK; dom%qq(elem)=0.0_RealK; dom%ss(elem)=0.0_RealK; dom%arealg(elem)=dom%volo(elem)**(2.0_RealK/3.0_RealK)
      dom%delv(elem)=0.0_RealK; dom%vdov(elem)=0.0_RealK
      dom%e(0)=3.948746e7_RealK
    end do; end do; end do
    do elem=0,dom%num_elem-1
      do i=0,7
        dom%nodal_mass(dom%nodelist(8*elem+i))=dom%nodal_mass(dom%nodelist(8*elem+i))+dom%elem_mass(elem)/8.0_RealK
      end do
    end do
  end subroutine initialize_domain

  subroutine lagrange_leapfrog(dom)
    type(domain_t),intent(inout)::dom
    call calc_force_for_nodes(dom)
    call calc_acceleration_for_nodes(dom)
    call calc_velocity_for_nodes(dom)
    call calc_position_for_nodes(dom)
    call lagrange_elements(dom)
    call calc_time_constraints(dom)
    dom%cycle=dom%cycle+1
    dom%time=dom%time+dom%deltatime
  end subroutine lagrange_leapfrog

  subroutine calc_force_for_nodes(dom)
    type(domain_t),intent(inout)::dom
    integer::i
!$omp target teams distribute parallel do thread_limit(256)
    do i=0,dom%num_node-1
      dom%fx(i)=0.0_RealK; dom%fy(i)=0.0_RealK; dom%fz(i)=0.0_RealK
    end do
!$omp end target teams distribute parallel do
    call integrate_stress_for_elems(dom)
  end subroutine calc_force_for_nodes

  subroutine integrate_stress_for_elems(dom)
    type(domain_t),intent(inout)::dom
    integer::e,n,idx
    real(RealK)::pressure,force
!$omp target teams distribute parallel do thread_limit(256) private(n,idx,pressure,force)
    do e=0,dom%num_elem-1
      pressure = -(dom%p(e)+dom%q(e))
      force = pressure * dom%volo(e) / 8.0_RealK
      do n=0,7
        idx=dom%nodelist(8*e+n)
!$omp atomic update
        dom%fx(idx)=dom%fx(idx)+force
!$omp atomic update
        dom%fy(idx)=dom%fy(idx)+force
!$omp atomic update
        dom%fz(idx)=dom%fz(idx)+force
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine integrate_stress_for_elems

  subroutine calc_acceleration_for_nodes(dom)
    type(domain_t),intent(inout)::dom
    integer::i
!$omp target teams distribute parallel do thread_limit(256)
    do i=0,dom%num_node-1
      dom%xdd(i)=dom%fx(i)/dom%nodal_mass(i); dom%ydd(i)=dom%fy(i)/dom%nodal_mass(i); dom%zdd(i)=dom%fz(i)/dom%nodal_mass(i)
    end do
!$omp end target teams distribute parallel do
  end subroutine calc_acceleration_for_nodes
  subroutine calc_velocity_for_nodes(dom)
    type(domain_t),intent(inout)::dom
    integer::i
!$omp target teams distribute parallel do thread_limit(256)
    do i=0,dom%num_node-1
      dom%xd(i)=dom%xd(i)+dom%xdd(i)*dom%deltatime; dom%yd(i)=dom%yd(i)+dom%ydd(i)*dom%deltatime; dom%zd(i)=dom%zd(i)+dom%zdd(i)*dom%deltatime
      if (abs(dom%xd(i)) < 1.0e-7_RealK) dom%xd(i)=0.0_RealK
      if (abs(dom%yd(i)) < 1.0e-7_RealK) dom%yd(i)=0.0_RealK
      if (abs(dom%zd(i)) < 1.0e-7_RealK) dom%zd(i)=0.0_RealK
    end do
!$omp end target teams distribute parallel do
  end subroutine calc_velocity_for_nodes
  subroutine calc_position_for_nodes(dom)
    type(domain_t),intent(inout)::dom
    integer::i
!$omp target teams distribute parallel do thread_limit(256)
    do i=0,dom%num_node-1
      dom%x(i)=dom%x(i)+dom%xd(i)*dom%deltatime; dom%y(i)=dom%y(i)+dom%yd(i)*dom%deltatime; dom%z(i)=dom%z(i)+dom%zd(i)*dom%deltatime
    end do
!$omp end target teams distribute parallel do
  end subroutine calc_position_for_nodes
  subroutine lagrange_elements(dom)
    type(domain_t),intent(inout)::dom
    integer::e
!$omp target teams distribute parallel do thread_limit(256)
    do e=0,dom%num_elem-1
      dom%vdov(e)=dom%delv(e)/max(dom%deltatime,1.0e-30_RealK)
      dom%q(e)=0.0_RealK
      if (dom%vdov(e) < 0.0_RealK) then
        dom%ql(e)=-0.5_RealK*dom%vdov(e); dom%qq(e)=dom%ql(e)*dom%ql(e)
        dom%q(e)=dom%ql(e)+dom%qq(e)
      end if
      dom%p(e)=max(0.0_RealK, (2.0_RealK/3.0_RealK)*dom%e(e))
      dom%e(e)=dom%e(e)-dom%p(e)*dom%delv(e)*0.5_RealK
      dom%ss(e)=sqrt(max(0.0_RealK, (4.0_RealK/3.0_RealK)*dom%p(e)/max(1.0e-30_RealK,dom%v(e))))
    end do
!$omp end target teams distribute parallel do
  end subroutine lagrange_elements
  subroutine calc_time_constraints(dom)
    type(domain_t),intent(inout)::dom
    real(RealK)::dtcourant,dthydro
    integer::e
    dtcourant=1.0e20_RealK; dthydro=1.0e20_RealK
!$omp target teams distribute parallel do reduction(min:dtcourant,dthydro) thread_limit(256)
    do e=0,dom%num_elem-1
      if (dom%ss(e) > 0.0_RealK) dtcourant=min(dtcourant, dom%arealg(e)/dom%ss(e))
      if (dom%vdov(e) /= 0.0_RealK) dthydro=min(dthydro, abs(0.1_RealK/dom%vdov(e)))
    end do
!$omp end target teams distribute parallel do
    dom%deltatime=min(dom%deltatime*1.2_RealK,dtcourant,dthydro,dom%stoptime-dom%time)
    if (dom%deltatime <= 0.0_RealK) dom%deltatime=1.0e-9_RealK
  end subroutine calc_time_constraints
end module lulesh_kernels
