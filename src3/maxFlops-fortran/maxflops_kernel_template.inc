#define REP5(X) X X X X X
#define REP10(X) REP5(X) REP5(X)
#define REP20(X) REP10(X) REP10(X)
#define ADD1_OP s=v-s;
#define ADD2_OP ADD1_OP s2=v-s2;
#define ADD4_OP ADD2_OP s3=v-s3; s4=v-s4;
#define ADD8_OP ADD4_OP s5=v-s5; s6=v-s6; s7=v-s7; s8=v-s8;
#define MUL1_OP s=s*s*v;
#define MUL2_OP MUL1_OP s2=s2*s2*v;
#define MUL4_OP MUL2_OP s3=s3*s3*v; s4=s4*s4*v;
#define MUL8_OP MUL4_OP s5=s5*s5*v; s6=s6*s6*v; s7=s7*s7*v; s8=s8*s8*v;
#define MADD1_OP s=v1-s*v2;
#define MADD2_OP MADD1_OP s2=v1-s2*v2;
#define MADD4_OP MADD2_OP s3=v1-s3*v2; s4=v1-s4*v2;
#define MADD8_OP MADD4_OP s5=v1-s5*v2; s6=v1-s6*v2; s7=v1-s7*v2; s8=v1-s8*v2;
#define MULMADD1_OP s=(v1-v2*s)*s;
#define MULMADD2_OP MULMADD1_OP s2=(v1-v2*s2)*s2;
#define MULMADD4_OP MULMADD2_OP s3=(v1-v2*s3)*s3; s4=(v1-v2*s4)*s4;
#define MULMADD8_OP MULMADD4_OP s5=(v1-v2*s5)*s5; s6=(v1-v2*s6)*s6; s7=(v1-v2*s7)*s7; s8=(v1-v2*s8)*s8;

  subroutine ADD1(data, nfloats, niters, v)
    real(RK), intent(inout) :: data(0:)
    integer, intent(in) :: nfloats, niters
    real(RK), intent(in) :: v
    integer :: gid, j
    real(RK) :: s
    !$omp target teams distribute parallel do thread_limit(BLOCK_SIZE) private(j,s)
    do gid = 0, nfloats - 1
      s = data(gid)
      do j = 1, niters
        REP20(ADD1_OP)
        REP20(ADD1_OP)
        REP20(ADD1_OP)
        REP20(ADD1_OP)
        REP20(ADD1_OP)
        REP20(ADD1_OP)
        REP20(ADD1_OP)
        REP20(ADD1_OP)
        REP20(ADD1_OP)
        REP20(ADD1_OP)
        REP20(ADD1_OP)
        REP20(ADD1_OP)
      end do
      data(gid) = s
    end do
    !$omp end target teams distribute parallel do
  end subroutine ADD1

  subroutine ADD2(data, nfloats, niters, v)
    real(RK), intent(inout) :: data(0:)
    integer, intent(in) :: nfloats, niters
    real(RK), intent(in) :: v
    integer :: gid, j
    real(RK) :: s, s2
    !$omp target teams distribute parallel do thread_limit(BLOCK_SIZE) private(j,s,s2)
    do gid = 0, nfloats - 1
      s = data(gid); s2 = real(10.0_real32,RK) - s
      do j = 1, niters
        REP20(ADD2_OP)
        REP20(ADD2_OP)
        REP20(ADD2_OP)
        REP20(ADD2_OP)
        REP20(ADD2_OP)
        REP20(ADD2_OP)
      end do
      data(gid) = s + s2
    end do
    !$omp end target teams distribute parallel do
  end subroutine ADD2

  subroutine ADD4(data, nfloats, niters, v)
    real(RK), intent(inout) :: data(0:)
    integer, intent(in) :: nfloats, niters
    real(RK), intent(in) :: v
    integer :: gid, j
    real(RK) :: s, s2, s3, s4
    !$omp target teams distribute parallel do thread_limit(BLOCK_SIZE) private(j,s,s2,s3,s4)
    do gid = 0, nfloats - 1
      s = data(gid); s2 = real(10.0_real32,RK)-s
      s3 = real(9.0_real32,RK)-s; s4 = real(9.0_real32,RK)-s2
      do j = 1, niters
        REP10(ADD4_OP)
        REP10(ADD4_OP)
        REP10(ADD4_OP)
        REP10(ADD4_OP)
        REP10(ADD4_OP)
        REP10(ADD4_OP)
      end do
      data(gid) = (s+s2) + (s3+s4)
    end do
    !$omp end target teams distribute parallel do
  end subroutine ADD4

  subroutine ADD8(data, nfloats, niters, v)
    real(RK), intent(inout) :: data(0:)
    integer, intent(in) :: nfloats, niters
    real(RK), intent(in) :: v
    integer :: gid, j
    real(RK) :: s, s2, s3, s4, s5, s6, s7, s8
    !$omp target teams distribute parallel do thread_limit(BLOCK_SIZE) private(j,s,s2,s3,s4,s5,s6,s7,s8)
    do gid = 0, nfloats - 1
      s = data(gid); s2 = real(10.0_real32,RK)-s
      s3 = real(9.0_real32,RK)-s; s4 = real(9.0_real32,RK)-s2
      s5 = real(8.0_real32,RK)-s; s6 = real(8.0_real32,RK)-s2
      s7 = real(7.0_real32,RK)-s; s8 = real(7.0_real32,RK)-s2
      do j = 1, niters
        REP5(ADD8_OP)
        REP5(ADD8_OP)
        REP5(ADD8_OP)
        REP5(ADD8_OP)
        REP5(ADD8_OP)
        REP5(ADD8_OP)
      end do
      data(gid) = ((s+s2)+(s3+s4)) + ((s5+s6)+(s7+s8))
    end do
    !$omp end target teams distribute parallel do
  end subroutine ADD8

  subroutine MUL1(data, nfloats, niters, v)
    real(RK), intent(inout) :: data(0:)
    integer, intent(in) :: nfloats, niters
    real(RK), intent(in) :: v
    integer :: gid, j
    real(RK) :: s
    !$omp target teams distribute parallel do thread_limit(BLOCK_SIZE) private(j,s)
    do gid = 0, nfloats - 1
      s = data(gid)-data(gid)+real(0.999_real32,RK)
      do j = 1, niters
        REP20(MUL1_OP)
        REP20(MUL1_OP)
        REP20(MUL1_OP)
        REP20(MUL1_OP)
        REP20(MUL1_OP)
        REP20(MUL1_OP)
        REP20(MUL1_OP)
        REP20(MUL1_OP)
        REP20(MUL1_OP)
        REP20(MUL1_OP)
      end do
      data(gid) = s
    end do
    !$omp end target teams distribute parallel do
  end subroutine MUL1

  subroutine MUL2(data, nfloats, niters, v)
    real(RK), intent(inout) :: data(0:)
    integer, intent(in) :: nfloats, niters
    real(RK), intent(in) :: v
    integer :: gid, j
    real(RK) :: s, s2
    !$omp target teams distribute parallel do thread_limit(BLOCK_SIZE) private(j,s,s2)
    do gid = 0, nfloats - 1
      s = data(gid)-data(gid)+real(0.999_real32,RK)
      s2 = s-real(0.0001_real32,RK)
      do j = 1, niters
        REP20(MUL2_OP)
        REP20(MUL2_OP)
        REP20(MUL2_OP)
        REP20(MUL2_OP)
        REP20(MUL2_OP)
      end do
      data(gid) = s+s2
    end do
    !$omp end target teams distribute parallel do
  end subroutine MUL2

  subroutine MUL4(data, nfloats, niters, v)
    real(RK), intent(inout) :: data(0:)
    integer, intent(in) :: nfloats, niters
    real(RK), intent(in) :: v
    integer :: gid, j
    real(RK) :: s, s2, s3, s4
    !$omp target teams distribute parallel do thread_limit(BLOCK_SIZE) private(j,s,s2,s3,s4)
    do gid = 0, nfloats - 1
      s = data(gid)-data(gid)+real(0.999_real32,RK)
      s2=s-real(0.0001_real32,RK); s3=s-real(0.0002_real32,RK); s4=s-real(0.0003_real32,RK)
      do j = 1, niters
        REP10(MUL4_OP)
        REP10(MUL4_OP)
        REP10(MUL4_OP)
        REP10(MUL4_OP)
        REP10(MUL4_OP)
      end do
      data(gid) = (s+s2)+(s3+s4)
    end do
    !$omp end target teams distribute parallel do
  end subroutine MUL4

  subroutine MUL8(data, nfloats, niters, v)
    real(RK), intent(inout) :: data(0:)
    integer, intent(in) :: nfloats, niters
    real(RK), intent(in) :: v
    integer :: gid, j
    real(RK) :: s, s2, s3, s4, s5, s6, s7, s8
    !$omp target teams distribute parallel do thread_limit(BLOCK_SIZE) private(j,s,s2,s3,s4,s5,s6,s7,s8)
    do gid = 0, nfloats - 1
      s = data(gid)-data(gid)+real(0.999_real32,RK)
      s2=s-real(0.0001_real32,RK); s3=s-real(0.0002_real32,RK); s4=s-real(0.0003_real32,RK)
      s5=s-real(0.0004_real32,RK); s6=s-real(0.0005_real32,RK)
      s7=s-real(0.0006_real32,RK); s8=s-real(0.0007_real32,RK)
      do j = 1, niters
        REP5(MUL8_OP)
        REP5(MUL8_OP)
        REP5(MUL8_OP)
        REP5(MUL8_OP)
        REP5(MUL8_OP)
      end do
      data(gid) = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8))
    end do
    !$omp end target teams distribute parallel do
  end subroutine MUL8

  subroutine MADD1(data, nfloats, niters, v1, v2)
    real(RK), intent(inout) :: data(0:)
    integer, intent(in) :: nfloats, niters
    real(RK), intent(in) :: v1, v2
    integer :: gid, j
    real(RK) :: s
    !$omp target teams distribute parallel do thread_limit(BLOCK_SIZE) private(j,s)
    do gid=0,nfloats-1
      s=data(gid)
      do j=1,niters
        REP20(MADD1_OP)
        REP20(MADD1_OP)
        REP20(MADD1_OP)
        REP20(MADD1_OP)
        REP20(MADD1_OP)
        REP20(MADD1_OP)
        REP20(MADD1_OP)
        REP20(MADD1_OP)
        REP20(MADD1_OP)
        REP20(MADD1_OP)
        REP20(MADD1_OP)
        REP20(MADD1_OP)
      end do
      data(gid)=s
    end do
    !$omp end target teams distribute parallel do
  end subroutine MADD1

  subroutine MADD2(data,nfloats,niters,v1,v2)
    real(RK),intent(inout)::data(0:); integer,intent(in)::nfloats,niters
    real(RK),intent(in)::v1,v2; integer::gid,j; real(RK)::s,s2
    !$omp target teams distribute parallel do thread_limit(BLOCK_SIZE) private(j,s,s2)
    do gid=0,nfloats-1
      s=data(gid); s2=real(10.0_real32,RK)-s
      do j=1,niters
        REP20(MADD2_OP)
        REP20(MADD2_OP)
        REP20(MADD2_OP)
        REP20(MADD2_OP)
        REP20(MADD2_OP)
        REP20(MADD2_OP)
      end do
      data(gid)=s+s2
    end do
    !$omp end target teams distribute parallel do
  end subroutine MADD2

  subroutine MADD4(data,nfloats,niters,v1,v2)
    real(RK),intent(inout)::data(0:); integer,intent(in)::nfloats,niters
    real(RK),intent(in)::v1,v2; integer::gid,j; real(RK)::s,s2,s3,s4
    !$omp target teams distribute parallel do thread_limit(BLOCK_SIZE) private(j,s,s2,s3,s4)
    do gid=0,nfloats-1
      s=data(gid); s2=real(10.0_real32,RK)-s
      s3=real(9.0_real32,RK)-s; s4=real(9.0_real32,RK)-s2
      do j=1,niters
        REP10(MADD4_OP)
        REP10(MADD4_OP)
        REP10(MADD4_OP)
        REP10(MADD4_OP)
        REP10(MADD4_OP)
        REP10(MADD4_OP)
      end do
      data(gid)=(s+s2)+(s3+s4)
    end do
    !$omp end target teams distribute parallel do
  end subroutine MADD4

  subroutine MADD8(data,nfloats,niters,v1,v2)
    real(RK),intent(inout)::data(0:); integer,intent(in)::nfloats,niters
    real(RK),intent(in)::v1,v2; integer::gid,j; real(RK)::s,s2,s3,s4,s5,s6,s7,s8
    !$omp target teams distribute parallel do thread_limit(BLOCK_SIZE) private(j,s,s2,s3,s4,s5,s6,s7,s8)
    do gid=0,nfloats-1
      s=data(gid); s2=real(10.0_real32,RK)-s
      s3=real(9.0_real32,RK)-s; s4=real(9.0_real32,RK)-s2
      s5=real(8.0_real32,RK)-s; s6=real(8.0_real32,RK)-s2
      s7=real(7.0_real32,RK)-s; s8=real(7.0_real32,RK)-s2
      do j=1,niters
        REP5(MADD8_OP)
        REP5(MADD8_OP)
        REP5(MADD8_OP)
        REP5(MADD8_OP)
        REP5(MADD8_OP)
        REP5(MADD8_OP)
      end do
      data(gid)=((s+s2)+(s3+s4))+((s5+s6)+(s7+s8))
    end do
    !$omp end target teams distribute parallel do
  end subroutine MADD8

  subroutine MULMADD1(data,nfloats,niters,v1,v2)
    real(RK),intent(inout)::data(0:); integer,intent(in)::nfloats,niters
    real(RK),intent(in)::v1,v2; integer::gid,j; real(RK)::s
    !$omp target teams distribute parallel do thread_limit(BLOCK_SIZE) private(j,s)
    do gid=0,nfloats-1
      s=data(gid)
      do j=1,niters
        REP20(MULMADD1_OP)
        REP20(MULMADD1_OP)
        REP20(MULMADD1_OP)
        REP20(MULMADD1_OP)
        REP20(MULMADD1_OP)
        REP20(MULMADD1_OP)
        REP20(MULMADD1_OP)
        REP20(MULMADD1_OP)
      end do
      data(gid)=s
    end do
    !$omp end target teams distribute parallel do
  end subroutine MULMADD1

  subroutine MULMADD2(data,nfloats,niters,v1,v2)
    real(RK),intent(inout)::data(0:); integer,intent(in)::nfloats,niters
    real(RK),intent(in)::v1,v2; integer::gid,j; real(RK)::s,s2
    !$omp target teams distribute parallel do thread_limit(BLOCK_SIZE) private(j,s,s2)
    do gid=0,nfloats-1
      s=data(gid); s2=real(10.0_real32,RK)-s
      do j=1,niters
        REP20(MULMADD2_OP)
        REP20(MULMADD2_OP)
        REP20(MULMADD2_OP)
        REP20(MULMADD2_OP)
      end do
      data(gid)=s+s2
    end do
    !$omp end target teams distribute parallel do
  end subroutine MULMADD2

  subroutine MULMADD4(data,nfloats,niters,v1,v2)
    real(RK),intent(inout)::data(0:); integer,intent(in)::nfloats,niters
    real(RK),intent(in)::v1,v2; integer::gid,j; real(RK)::s,s2,s3,s4
    !$omp target teams distribute parallel do thread_limit(BLOCK_SIZE) private(j,s,s2,s3,s4)
    do gid=0,nfloats-1
      s=data(gid); s2=real(10.0_real32,RK)-s
      s3=real(9.0_real32,RK)-s; s4=real(9.0_real32,RK)-s2
      do j=1,niters
        REP10(MULMADD4_OP)
        REP10(MULMADD4_OP)
        REP10(MULMADD4_OP)
        REP10(MULMADD4_OP)
      end do
      data(gid)=(s+s2)+(s3+s4)
    end do
    !$omp end target teams distribute parallel do
  end subroutine MULMADD4

  subroutine MULMADD8(data,nfloats,niters,v1,v2)
    real(RK),intent(inout)::data(0:); integer,intent(in)::nfloats,niters
    real(RK),intent(in)::v1,v2; integer::gid,j; real(RK)::s,s2,s3,s4,s5,s6,s7,s8
    !$omp target teams distribute parallel do thread_limit(BLOCK_SIZE) private(j,s,s2,s3,s4,s5,s6,s7,s8)
    do gid=0,nfloats-1
      s=data(gid); s2=real(10.0_real32,RK)-s
      s3=real(9.0_real32,RK)-s; s4=real(9.0_real32,RK)-s2
      s5=real(8.0_real32,RK)-s; s6=real(8.0_real32,RK)-s2
      s7=real(7.0_real32,RK)-s; s8=real(7.0_real32,RK)-s2
      do j=1,niters
        REP5(MULMADD8_OP)
        REP5(MULMADD8_OP)
        REP5(MULMADD8_OP)
        REP5(MULMADD8_OP)
      end do
      data(gid)=((s+s2)+(s3+s4))+((s5+s6)+(s7+s8))
    end do
    !$omp end target teams distribute parallel do
  end subroutine MULMADD8

#undef MULMADD8_OP
#undef MULMADD4_OP
#undef MULMADD2_OP
#undef MULMADD1_OP
#undef MADD8_OP
#undef MADD4_OP
#undef MADD2_OP
#undef MADD1_OP
#undef MUL8_OP
#undef MUL4_OP
#undef MUL2_OP
#undef MUL1_OP
#undef ADD8_OP
#undef ADD4_OP
#undef ADD2_OP
#undef ADD1_OP
#undef REP20
#undef REP10
#undef REP5
