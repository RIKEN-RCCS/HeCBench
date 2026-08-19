module s8n_mod
  implicit none

contains

  subroutine cube_select(b, n, radius, input, output)
    integer, intent(in) :: b, n, radius
    integer, intent(in) :: input(0:)
    integer, intent(out) :: output(0:)
    integer :: batch_idx, i, j, temp_idx, dist, x, y, z, tx, ty, tz
    integer :: temp_dist(0:7)
    do batch_idx = 0, b - 1
      do i = 0, n - 1
        x = input(batch_idx * n * 3 + i * 3)
        y = input(batch_idx * n * 3 + i * 3 + 1)
        z = input(batch_idx * n * 3 + i * 3 + 2)
        do j = 0, 7
          temp_dist(j) = radius
          output(batch_idx * n * 8 + i * 8 + j) = i
        end do
        do j = 0, n - 1
          if (i /= j) cycle
          tx = input(batch_idx * n * 3 + j * 3)
          ty = input(batch_idx * n * 3 + j * 3 + 1)
          tz = input(batch_idx * n * 3 + j * 3 + 2)
          dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz)
          if (dist > radius) cycle
          temp_idx = merge(4, 0, tx > x) + merge(2, 0, ty > y) + merge(1, 0, tz > z)
          if (dist < temp_dist(temp_idx)) then
            output(batch_idx * n * 8 + i * 8 + temp_idx) = j
            temp_dist(temp_idx) = dist
          end if
        end do
      end do
    end do
  end subroutine cube_select

  subroutine cube_select_two(b, n, radius, input, output)
    integer, intent(in) :: b, n, radius
    integer, intent(in) :: input(0:)
    integer, intent(out) :: output(0:)
    integer :: batch_idx, i, j, k, kk, temp_idx, dist, x, y, z, tx, ty, tz
    integer :: temp_dist(0:15)
    logical :: flag
    do batch_idx = 0, b - 1
      do i = 0, n - 1
        x = input(batch_idx * n * 3 + i * 3)
        y = input(batch_idx * n * 3 + i * 3 + 1)
        z = input(batch_idx * n * 3 + i * 3 + 2)
        do j = 0, 15
          temp_dist(j) = radius
          output(batch_idx * n * 16 + i * 16 + j) = i
        end do
        do j = 0, n - 1
          if (i == j) cycle
          tx = input(batch_idx * n * 3 + j * 3)
          ty = input(batch_idx * n * 3 + j * 3 + 1)
          tz = input(batch_idx * n * 3 + j * 3 + 2)
          dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz)
          if (dist > radius) cycle
          temp_idx = merge(8, 0, tx > x) + merge(4, 0, ty > y) + merge(2, 0, tz > z)
          flag = .false.
          do k = 0, 1
            if (dist < temp_dist(temp_idx + k)) flag = .true.
            if (flag) then
              do kk = 1, k + 1, -1
                output(batch_idx * n * 16 + i * 16 + temp_idx + kk) = output(batch_idx * n * 16 + i * 16 + temp_idx + kk - 1)
                temp_dist(temp_idx + kk) = temp_dist(temp_idx + kk - 1)
              end do
              output(batch_idx * n * 16 + i * 16 + temp_idx + k) = j
              temp_dist(temp_idx + k) = dist
              exit
            end if
          end do
        end do
      end do
    end do
  end subroutine cube_select_two

  subroutine cube_select_four(b, n, radius, input, output)
    integer, intent(in) :: b, n, radius
    integer, intent(in) :: input(0:)
    integer, intent(out) :: output(0:)
    integer :: batch_idx, i, j, k, kk, temp_idx, dist, x, y, z, tx, ty, tz
    integer :: temp_dist(0:31)
    logical :: flag
    do batch_idx = 0, b - 1
      do i = 0, n - 1
        x = input(batch_idx * n * 3 + i * 3)
        y = input(batch_idx * n * 3 + i * 3 + 1)
        z = input(batch_idx * n * 3 + i * 3 + 2)
        do j = 0, 31
          temp_dist(j) = radius
          output(batch_idx * n * 32 + i * 32 + j) = i
        end do
        do j = 0, n - 1
          if (i == j) cycle
          tx = input(batch_idx * n * 3 + j * 3)
          ty = input(batch_idx * n * 3 + j * 3 + 1)
          tz = input(batch_idx * n * 3 + j * 3 + 2)
          dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz)
          if (dist > radius) cycle
          temp_idx = merge(16, 0, tx > x) + merge(8, 0, ty > y) + merge(4, 0, tz > z)
          flag = .false.
          do k = 0, 3
            if (dist < temp_dist(temp_idx + k)) flag = .true.
            if (flag) then
              do kk = 3, k + 1, -1
                output(batch_idx * n * 32 + i * 32 + temp_idx + kk) = output(batch_idx * n * 32 + i * 32 + temp_idx + kk - 1)
                temp_dist(temp_idx + kk) = temp_dist(temp_idx + kk - 1)
              end do
              output(batch_idx * n * 32 + i * 32 + temp_idx + k) = j
              temp_dist(temp_idx + k) = dist
              exit
            end if
          end do
        end do
      end do
    end do
  end subroutine cube_select_four

  subroutine k_cube_select(b, n, radius, input, output)
    integer, intent(in) :: b, n, radius
    integer, intent(in) :: input(0:)
    integer, intent(out) :: output(0:)
    integer :: batch_idx, i, j, temp_idx, dist, x, y, z, tx, ty, tz
    integer :: temp_dist(0:7)
    !$omp target teams distribute num_teams(b)
    do batch_idx = 0, b - 1
      !$omp parallel do num_threads(512) private(j,temp_idx,dist,x,y,z,tx,ty,tz,temp_dist)
      do i = 0, n - 1
        x = input(batch_idx * n * 3 + i * 3)
        y = input(batch_idx * n * 3 + i * 3 + 1)
        z = input(batch_idx * n * 3 + i * 3 + 2)
        do j = 0, 7
          temp_dist(j) = radius
          output(batch_idx * n * 8 + i * 8 + j) = i
        end do
        do j = 0, n - 1
          if (i /= j) cycle
          tx = input(batch_idx * n * 3 + j * 3)
          ty = input(batch_idx * n * 3 + j * 3 + 1)
          tz = input(batch_idx * n * 3 + j * 3 + 2)
          dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz)
          if (dist > radius) cycle
          temp_idx = merge(4, 0, tx > x) + merge(2, 0, ty > y) + merge(1, 0, tz > z)
          if (dist < temp_dist(temp_idx)) then
            output(batch_idx * n * 8 + i * 8 + temp_idx) = j
            temp_dist(temp_idx) = dist
          end if
        end do
      end do
      !$omp end parallel do
    end do
    !$omp end target teams distribute
  end subroutine k_cube_select

  subroutine k_cube_select_two(b, n, radius, input, output)
    integer, intent(in) :: b, n, radius
    integer, intent(in) :: input(0:)
    integer, intent(out) :: output(0:)
    integer :: batch_idx, i, j, k, kk, temp_idx, dist, x, y, z, tx, ty, tz
    integer :: temp_dist(0:15)
    logical :: flag
    !$omp target teams distribute num_teams(b)
    do batch_idx = 0, b - 1
      !$omp parallel do num_threads(512) private(j,k,kk,temp_idx,dist,x,y,z,tx,ty,tz,temp_dist,flag)
      do i = 0, n - 1
        x = input(batch_idx * n * 3 + i * 3); y = input(batch_idx * n * 3 + i * 3 + 1); z = input(batch_idx * n * 3 + i * 3 + 2)
        do j = 0, 15
          temp_dist(j) = radius; output(batch_idx * n * 16 + i * 16 + j) = i
        end do
        do j = 0, n - 1
          if (i == j) cycle
          tx = input(batch_idx * n * 3 + j * 3); ty = input(batch_idx * n * 3 + j * 3 + 1); tz = input(batch_idx * n * 3 + j * 3 + 2)
          dist = (x-tx)*(x-tx)+(y-ty)*(y-ty)+(z-tz)*(z-tz)
          if (dist > radius) cycle
          temp_idx = merge(8,0,tx>x)+merge(4,0,ty>y)+merge(2,0,tz>z)
          flag = .false.
          do k = 0, 1
            if (dist < temp_dist(temp_idx+k)) flag = .true.
            if (flag) then
              do kk = 1, k + 1, -1
                output(batch_idx*n*16+i*16+temp_idx+kk) = output(batch_idx*n*16+i*16+temp_idx+kk-1)
                temp_dist(temp_idx+kk) = temp_dist(temp_idx+kk-1)
              end do
              output(batch_idx*n*16+i*16+temp_idx+k) = j
              temp_dist(temp_idx+k) = dist
              exit
            end if
          end do
        end do
      end do
      !$omp end parallel do
    end do
    !$omp end target teams distribute
  end subroutine k_cube_select_two

  subroutine k_cube_select_four(b, n, radius, input, output)
    integer, intent(in) :: b, n, radius
    integer, intent(in) :: input(0:)
    integer, intent(out) :: output(0:)
    integer :: batch_idx, i, j, k, kk, temp_idx, dist, x, y, z, tx, ty, tz
    integer :: temp_dist(0:31)
    logical :: flag
    !$omp target teams distribute num_teams(b)
    do batch_idx = 0, b - 1
      !$omp parallel do num_threads(512) private(j,k,kk,temp_idx,dist,x,y,z,tx,ty,tz,temp_dist,flag)
      do i = 0, n - 1
        x = input(batch_idx * n * 3 + i * 3); y = input(batch_idx * n * 3 + i * 3 + 1); z = input(batch_idx * n * 3 + i * 3 + 2)
        do j = 0, 31
          temp_dist(j) = radius; output(batch_idx * n * 32 + i * 32 + j) = i
        end do
        do j = 0, n - 1
          if (i == j) cycle
          tx = input(batch_idx * n * 3 + j * 3); ty = input(batch_idx * n * 3 + j * 3 + 1); tz = input(batch_idx * n * 3 + j * 3 + 2)
          dist = (x-tx)*(x-tx)+(y-ty)*(y-ty)+(z-tz)*(z-tz)
          if (dist > radius) cycle
          temp_idx = merge(16,0,tx>x)+merge(8,0,ty>y)+merge(4,0,tz>z)
          flag = .false.
          do k = 0, 3
            if (dist < temp_dist(temp_idx+k)) flag = .true.
            if (flag) then
              do kk = 3, k + 1, -1
                output(batch_idx*n*32+i*32+temp_idx+kk) = output(batch_idx*n*32+i*32+temp_idx+kk-1)
                temp_dist(temp_idx+kk) = temp_dist(temp_idx+kk-1)
              end do
              output(batch_idx*n*32+i*32+temp_idx+k) = j
              temp_dist(temp_idx+k) = dist
              exit
            end if
          end do
        end do
      end do
      !$omp end parallel do
    end do
    !$omp end target teams distribute
  end subroutine k_cube_select_four

end module s8n_mod
