module lulesh_types
  use iso_fortran_env, only: real64
  implicit none
  integer, parameter :: RealK = real64
  type :: cmd_options
    integer :: its = 9999999
    integer :: nx = 30
    integer :: num_reg = 11
    integer :: balance = 1
    integer :: cost = 1
    integer :: show_progress = 0
    integer :: quiet = 0
  end type cmd_options
  type :: domain_t
    integer :: nx=0, num_node=0, num_elem=0, cycle=0
    real(RealK) :: time=0.0_RealK, deltatime=1.0e-7_RealK, stoptime=1.0e-2_RealK
    real(RealK), allocatable :: x(:), y(:), z(:), xd(:), yd(:), zd(:), xdd(:), ydd(:), zdd(:), fx(:), fy(:), fz(:), nodal_mass(:)
    real(RealK), allocatable :: e(:), p(:), q(:), ql(:), qq(:), v(:), volo(:), delv(:), vdov(:), arealg(:), ss(:), elem_mass(:)
    integer, allocatable :: nodelist(:)
  end type domain_t
contains
  pure integer function node_index(i,j,k,nx)
    integer,intent(in)::i,j,k,nx
    node_index = i + (nx+1)*(j + (nx+1)*k)
  end function node_index
  pure integer function elem_index(i,j,k,nx)
    integer,intent(in)::i,j,k,nx
    elem_index = i + nx*(j + nx*k)
  end function elem_index
end module lulesh_types
