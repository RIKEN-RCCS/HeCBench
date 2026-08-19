module wsm5_spt
  implicit none
  integer, parameter :: mkx = 4
  integer, parameter :: xxx = 8, yyy = 8
!$omp declare target (i2, i3, p2, p3)
contains
  pure integer function i2(i, j, m) result(idx)
    integer, intent(in) :: i, j, m
    idx = i + j * m
  end function i2

  pure integer function i3(i, j, m, k, n) result(idx)
    integer, intent(in) :: i, j, m, k, n
    idx = i2(i, j, m) + k * m * n
  end function i3

  pure integer function p2(ti, tj, bi, bj, bx, by, ips, ims, jps, jms, ime) result(idx)
    integer, intent(in) :: ti, tj, bi, bj, bx, by, ips, ims, jps, jms, ime
    idx = i2(ti + bi * bx + ips - ims, tj + bj * by + jps - jms, ime - ims + 1)
  end function p2

  pure integer function p3(ti, k, tj, bi, bj, bx, by, ips, ims, jps, jms, ime, kms, kme) result(idx)
    integer, intent(in) :: ti, k, tj, bi, bj, bx, by, ips, ims, jps, jms, ime, kms, kme
    idx = i3(ti + bi * bx + ips - ims, k, ime - ims + 1, tj + bj * by + jps - jms, kme - kms + 1)
  end function p3
end module wsm5_spt
