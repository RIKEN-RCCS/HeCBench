program main
  use nn_utils
  implicit none
  type(record), allocatable :: records(:)
  type(latlong), allocatable :: locations(:)
  real(real32), allocatable :: record_distances(:)
  character(len=100) :: filename
  integer :: results_count, quiet, timing, repeat, num_records, i
  real(real32) :: lat, lng
  real(real64) :: t0, t1
  results_count = 10; quiet = 0; timing = 0; repeat = 1; lat = 0.0_real32; lng = 0.0_real32
  if (parse_commandline(filename, results_count, lat, lng, repeat, quiet, timing) /= 0) then
    call print_usage()
    stop
  end if
  num_records = load_data(filename, records, locations)
  if (quiet == 0) then
    print '(a,i0)', 'Number of records: ', num_records
    print '(a,i0,a)', 'Finding the ', results_count, ' closest neighbors.'
  end if
  if (results_count > num_records) results_count = num_records
  allocate(record_distances(0:num_records-1))
  t0 = seconds()
  call find_nearest_neighbors(num_records, locations, lat, lng, record_distances, repeat)
  t1 = seconds()
  if (timing /= 0) print '(a,f10.6,a)', 'Device offloading time ', t1-t0, ' (s)'
  call find_lowest(records, record_distances, num_records, results_count)
  if (quiet == 0) then
    do i = 0, results_count-1
      print '(a,a,f10.6)', trim(records(i)%rec_string)//' --> Distance=', '', records(i)%distance
    end do
  end if
contains
  subroutine find_nearest_neighbors(num_records, locations, lat, lng, distances, repeat)
    integer, intent(in) :: num_records, repeat
    type(latlong), intent(in) :: locations(0:num_records-1)
    real(real32), intent(in) :: lat, lng
    real(real32), intent(out) :: distances(0:num_records-1)
    integer :: gid, r
    type(latlong) :: ll
    real(real64) :: start, finish
    !$omp target data map(to:locations(0:num_records-1)) map(from:distances(0:num_records-1))
    start = seconds()
    do r = 1, repeat
      !$omp target teams distribute parallel do private(ll) thread_limit(64)
      do gid = 0, num_records-1
        ll = locations(gid)
        distances(gid) = sqrt((lat-ll%lat)*(lat-ll%lat) + (lng-ll%lng)*(lng-ll%lng))
      end do
      !$omp end target teams distribute parallel do
    end do
    finish = seconds()
    print '(a,f10.6,a)', 'Average kernel execution time: ', ((finish-start)*1.0e6_real64)/real(repeat,real64), ' (us)'
    !$omp end target data
  end subroutine
end program
