module wc_mod
  implicit none
contains
  pure logical function is_alpha(c)
    character(len=1), intent(in) :: c
    is_alpha = c >= 'A' .and. c <= 'z'
  end function is_alpha

  function word_count(input) result(wc)
    character(len=1), intent(in) :: input(0:)
    integer :: wc, i, n
    n = size(input)
    if (n == 0) then
      wc = 0
      return
    end if
    wc = 0
!$omp target data map(to:input) map(tofrom:wc)
!$omp target teams distribute parallel do thread_limit(256) reduction(+:wc)
    do i = 0, n - 2
      if ((.not. is_alpha(input(i))) .and. is_alpha(input(i + 1))) wc = wc + 1
    end do
!$omp end target teams distribute parallel do
!$omp end target data
    if (is_alpha(input(0))) wc = wc + 1
  end function word_count

  function word_count_reference(input) result(wc)
    character(len=1), intent(in) :: input(0:)
    integer :: wc, i, n
    n = size(input)
    if (n == 0) then
      wc = 0
      return
    end if
    wc = 0
    do i = 0, n - 2
      if ((.not. is_alpha(input(i))) .and. is_alpha(input(i + 1))) wc = wc + 1
    end do
    if (is_alpha(input(0))) wc = wc + 1
  end function word_count_reference
end module wc_mod
